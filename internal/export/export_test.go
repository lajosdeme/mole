package export_test

import (
	"context"
	"encoding/json"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/budget"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/export"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/store/sqlite"
)

// A session with the shape a pack needs: two leads, four claims, one
// contradiction, and a fetch that failed.
func fixture(t *testing.T) (*sqlite.DB, *core.Session) {
	t.Helper()
	return fixtureWith(t, true)
}

// fixtureWith builds the session, optionally without the not_selected rows that
// give coverage its denominator — which is every session recorded before that
// outcome existed.
func fixtureWith(t *testing.T, withNotSelected bool) (*sqlite.DB, *core.Session) {
	t.Helper()
	ctx := context.Background()

	db, err := sqlite.Open(filepath.Join(t.TempDir(), "t.db"), sqlite.Options{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.Migrate(ctx); err != nil {
		t.Fatalf("migrate: %v", err)
	}

	led := budget.New(db, budget.DefaultConfig())
	sess, err := led.CreateSession(ctx, budget.SessionSpec{
		Prompt: "is the claimed 40% growth real?", Mode: core.ModeReport,
		ActorTypes: []core.ActorType{core.ActorWeb},
		BudgetUnit: core.BudgetUSD, Budget: core.MicrosPerUSD,
	})
	if err != nil {
		t.Fatalf("create session: %v", err)
	}

	leads := []core.Lead{
		{ID: core.NewLeadID(), SessionID: sess.ID, ActorType: core.ActorWeb,
			Query: "shipment volumes by country", Status: core.LeadDone},
		{ID: core.NewLeadID(), SessionID: sess.ID, ActorType: core.ActorWeb,
			Query: "where does the 40% figure come from", Status: core.LeadDone, Depth: 1},
	}
	when := sess.CreatedAt
	claims := []core.Claim{
		{SessionID: sess.ID, LeadID: leads[0].ID, Text: "Shipments grew 40% year on year.",
			Source: "https://trade.example/press", Quote: "shipments grew 40% year on year across the union",
			Confidence: 0.4, VerifiedAt: &when},
		{SessionID: sess.ID, LeadID: leads[0].ID, Text: "Installations rose 11.4%.",
			Source: "https://stats.example/2026", Quote: "installations rose 11.4% over the equivalent months",
			Confidence: 0.8, VerifiedAt: &when, PublishedAt: &when},
	}

	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		for i := range leads {
			if err := tx.InsertLead(ctx, &leads[i]); err != nil {
				return err
			}
		}
		if err := tx.InsertClaims(ctx, claims); err != nil {
			return err
		}
		if err := tx.InsertEdges(ctx, []core.ClaimEdge{{
			SessionID: sess.ID, FromID: claims[0].ID, ToID: claims[1].ID,
			Kind: core.EdgeContradicts, Weight: 0.9,
			Rationale: "40% against 11.4% over the same period", CreatedBy: "model",
		}}); err != nil {
			return err
		}
		if err := tx.RecordFetchOutcome(ctx, &store.FetchOutcome{
			SessionID: &sess.ID, URL: "https://paywalled.example/report",
			Domain: "paywalled.example", Outcome: "paywall", StatusCode: 402,
		}); err != nil {
			return err
		}
		// Two results the source cap never reached: the denominator.
		if !withNotSelected {
			return nil
		}
		for _, u := range []string{"https://skipped.example/a", "https://skipped.example/b"} {
			if err := tx.RecordFetchOutcome(ctx, &store.FetchOutcome{
				SessionID: &sess.ID, URL: u, Domain: "skipped.example",
				Outcome: "not_selected",
			}); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		t.Fatalf("seed: %v", err)
	}

	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		if err := tx.SetSessionReport(ctx, sess.ID,
			"## Finding\n\nThe 40% figure traces to one press release [1], against "+
				"an independent series [2].", ""); err != nil {
			return err
		}
		return tx.SetSessionReportCitations(ctx, sess.ID, `[
			{"n":1,"source":"https://trade.example/press","quotes":["shipments grew 40%"]},
			{"n":2,"source":"https://stats.example/2026","quotes":["installations rose 11.4%"]}
		]`)
	}); err != nil {
		t.Fatalf("set report: %v", err)
	}

	// Spend through the ledger, so by-role costs exist. Without this the envelope
	// test pins an empty array and says nothing about the fields inside it.
	res, err := led.ReserveFor(ctx, sess.ID, leads[0].ID, 50_000)
	if err != nil {
		t.Fatalf("reserve: %v", err)
	}
	if _, err := led.Settle(ctx, res, []core.ToolCall{{
		Role: core.RoleExecutor, Type: core.CallLLM, Model: "m",
		Cost: core.Cost{USDMicros: 40_000, InputTokens: 900, OutputTokens: 120},
	}}); err != nil {
		t.Fatalf("settle: %v", err)
	}

	// sess is returned for its id only; Build re-reads it.
	return db, sess
}

func build(t *testing.T, opts export.Options) *export.Export {
	t.Helper()
	db, sess := fixture(t)
	ex, err := export.Build(context.Background(), db, sess.ID, opts)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	return ex
}

// jsonPaths walks the marshalled form and returns every dotted key path.
func jsonPaths(t *testing.T, v any) []string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var raw any
	if err := json.Unmarshal(b, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	seen := map[string]bool{}
	var walk func(prefix string, v any)
	walk = func(prefix string, v any) {
		switch t := v.(type) {
		case map[string]any:
			for k, sub := range t {
				p := k
				if prefix != "" {
					p = prefix + "." + k
				}
				seen[p] = true
				walk(p, sub)
			}
		case []any:
			for _, sub := range t {
				walk(prefix+"[]", sub)
			}
		}
	}
	walk("", raw)
	out := make([]string, 0, len(seen))
	for p := range seen {
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

// TestTheEnvelopeIsPinned.
//
// A paying product parses this, so a renamed field is an outage rather than a
// refactor. The list below IS the contract: adding to it is a minor version,
// removing or renaming anything in it is a breaking change and this test is where
// that conversation starts.
func TestTheEnvelopeIsPinned(t *testing.T) {
	want := []string{
		"conflicts", "conflicts[].a", "conflicts[].a.confidence",
		"conflicts[].a.id", "conflicts[].a.published_at", "conflicts[].a.quote",
		"conflicts[].a.source", "conflicts[].a.text", "conflicts[].b",
		"conflicts[].b.confidence", "conflicts[].b.id", "conflicts[].b.published_at",
		"conflicts[].b.quote", "conflicts[].b.source", "conflicts[].b.text",
		"conflicts[].decided_by", "conflicts[].kind", "conflicts[].rationale",
		"cost", "cost.by_role", "cost.by_role[].role",
		"cost.by_role[].usd_micros", "cost.calls", "cost.tokens",
		"cost.tokens.cache_read", "cost.tokens.cache_write", "cost.tokens.input",
		"cost.tokens.output", "cost.usd_micros", "coverage",
		"coverage.considered", "coverage.considered_known",
		"coverage.not_selected", "coverage.note", "coverage.read",
		"coverage.unreachable", "coverage.unreachable[].domain", "coverage.unreachable[].outcome",
		"coverage.unreachable[].status_code", "coverage.unreachable[].url", "evidence",
		"evidence[].confidence", "evidence[].id", "evidence[].lead_id",
		"evidence[].published_at", "evidence[].quote", "evidence[].source",
		"evidence[].text", "exported_at", "mole_version",
		"receipt", "report", "report.citations", "report.citations[].n",
		"report.citations[].published_at", "report.citations[].quotes",
		"report.citations[].source", "report.degraded",
		"report.markdown", "schema_version", "scope",
		"scope[].actor", "scope[].depth", "scope[].id",
		"scope[].parent_id", "scope[].query", "scope[].status",
		"session", "session.actors", "session.budget",
		"session.budget_unit", "session.created_at", "session.id",
		"session.mode", "session.question", "session.spent",
		"session.status", "session.updated_at",
	}
	sort.Strings(want)

	got := jsonPaths(t, build(t, export.Options{}))
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Errorf("the envelope changed.\n--- got ---\n%s\n--- want ---\n%s",
			strings.Join(got, "\n"), strings.Join(want, "\n"))
	}
}

// TestSchemaVersionIsDeclared, so a consumer can refuse a version it predates.
func TestSchemaVersionIsDeclared(t *testing.T) {
	ex := build(t, export.Options{})
	if ex.SchemaVersion != export.SchemaVersion || ex.SchemaVersion < 1 {
		t.Errorf("schema_version = %d, want %d", ex.SchemaVersion, export.SchemaVersion)
	}
}

// TestAConflictInlinesBothClaims. The consumer renders a document; making it join
// conflicts back to the evidence array by ID would push mole's schema into a
// codebase that should only know about packs.
func TestAConflictInlinesBothClaims(t *testing.T) {
	ex := build(t, export.Options{})
	if len(ex.Conflicts) != 1 {
		t.Fatalf("%d conflict(s), want 1", len(ex.Conflicts))
	}
	c := ex.Conflicts[0]
	for _, side := range []export.Claim{c.A, c.B} {
		if side.Text == "" || side.Quote == "" || side.Source == "" {
			t.Errorf("a conflict side is not renderable on its own: %+v", side)
		}
	}
	if !strings.Contains(c.A.Text+c.B.Text, "40%") ||
		!strings.Contains(c.A.Text+c.B.Text, "11.4%") {
		t.Error("the conflict does not carry the two figures that disagree")
	}
	if c.Kind != string(core.EdgeContradicts) {
		t.Errorf("kind = %q", c.Kind)
	}
}

// TestTheReceiptIsAbsentRatherThanZero.
//
// Same reason groundingRate returns nil rather than 0%: "nothing was checked" and
// "everything failed" must not render alike, and here a consumer would print a
// receipt claiming zero verified quotes.
func TestTheReceiptIsAbsentRatherThanZero(t *testing.T) {
	ex := build(t, export.Options{})
	if ex.Receipt != nil {
		t.Errorf("a receipt appeared without verification: %+v", ex.Receipt)
	}
	b, _ := json.Marshal(ex)
	if !strings.Contains(string(b), `"receipt":null`) {
		t.Error("receipt is omitted from the JSON; a consumer cannot tell it was not run")
	}
}

// TestCoverageCountsWhatWasConsidered. A read count presented as coverage reads
// as a complete denominator; the not_selected rows are what make it a fraction.
func TestCoverageCountsWhatWasConsidered(t *testing.T) {
	ex := build(t, export.Options{})
	if !ex.Coverage.ConsideredKnown {
		t.Fatal("coverage does not know its denominator despite not_selected rows")
	}
	// 2 sources produced claims, 1 paywalled, 2 never reached.
	if ex.Coverage.Considered != 5 {
		t.Errorf("considered = %d, want 5", ex.Coverage.Considered)
	}
	if ex.Coverage.Read != 2 {
		t.Errorf("read = %d, want 2", ex.Coverage.Read)
	}
	if ex.Coverage.NotSelected != 2 {
		t.Errorf("not_selected = %d, want 2", ex.Coverage.NotSelected)
	}
}

// TestAnUnreadResultIsNotReportedAsUnreachable. A source the budget never reached
// and a source that refused to be read are different facts, and a pack that
// printed the first as the second would name pages nothing was wrong with.
func TestAnUnreadResultIsNotReportedAsUnreachable(t *testing.T) {
	ex := build(t, export.Options{})
	if len(ex.Coverage.Unreachable) != 1 || ex.Coverage.Unreachable[0].Outcome != "paywall" {
		t.Errorf("unreachable should name the paywall alone: %+v", ex.Coverage.Unreachable)
	}
}

// TestCoverageSaysSoWhenNothingRecordedADenominator.
//
// Every session recorded before the not_selected outcome existed has no
// denominator. Reporting considered == read for those would be the exact
// overstatement this section exists to avoid, and it would be invisible.
func TestCoverageSaysSoWhenNothingRecordedADenominator(t *testing.T) {
	db, sess := fixtureWith(t, false)
	ex, err := export.Build(context.Background(), db, sess.ID, export.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if ex.Coverage.ConsideredKnown {
		t.Error("coverage claims a denominator nothing recorded")
	}
	if ex.Coverage.Considered != 0 {
		t.Errorf("considered = %d with nothing to count it from", ex.Coverage.Considered)
	}
	if ex.Coverage.Note == "" {
		t.Error("nothing tells a consumer why the denominator is missing")
	}
}

// TestTwoExportsOfOneSessionAgree, so a later re-render and a diff compare the
// document rather than the order a map happened to iterate in.
func TestTwoExportsOfOneSessionAgree(t *testing.T) {
	ctx := context.Background()
	db, sess := fixture(t)

	first, err := export.Build(ctx, db, sess.ID, export.Options{})
	if err != nil {
		t.Fatal(err)
	}
	second, err := export.Build(ctx, db, sess.ID, export.Options{})
	if err != nil {
		t.Fatal(err)
	}
	// exported_at is the one field that must differ; blank it and compare the rest.
	first.ExportedAt, second.ExportedAt = time.Time{}, time.Time{}

	a, _ := json.Marshal(first)
	b, _ := json.Marshal(second)
	if string(a) != string(b) {
		t.Error("two exports of one session differ; ordering is not stable")
	}
}

// TestAMissingSessionIsNotAnEmptyExport.
func TestAMissingSessionIsNotAnEmptyExport(t *testing.T) {
	db, _ := fixture(t)
	if _, err := export.Build(context.Background(), db, "s_nope", export.Options{}); err == nil {
		t.Fatal("exporting an unknown session succeeded")
	}
}

// TestAStoredReceiptIsReused, so the fetches a verification cost are paid once
// rather than once per render.
func TestAStoredReceiptIsReused(t *testing.T) {
	ctx := context.Background()
	db, sess := fixture(t)

	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.SetSessionReceipt(ctx, sess.ID, `{"verified":23,"mismatch":0,`+
			`"offset_drift":1,"unreachable":2,"skipped":3}`)
	}); err != nil {
		t.Fatalf("store receipt: %v", err)
	}

	ex, err := export.Build(ctx, db, sess.ID, export.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if ex.Receipt == nil {
		t.Fatal("a stored receipt was ignored")
	}
	if ex.Receipt.Verified != 23 || ex.Receipt.Skipped != 3 {
		t.Errorf("stored receipt read back wrong: %+v", ex.Receipt)
	}
}

// TestAFreshVerificationBeatsTheStoredOne. Re-verifying is how a caller checks
// whether sources have changed since; the older answer must not win.
func TestAFreshVerificationBeatsTheStoredOne(t *testing.T) {
	ctx := context.Background()
	db, sess := fixture(t)
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.SetSessionReceipt(ctx, sess.ID, `{"verified":1}`)
	}); err != nil {
		t.Fatal(err)
	}
	ex, err := export.Build(ctx, db, sess.ID,
		export.Options{Receipt: &export.Receipt{Verified: 99}})
	if err != nil {
		t.Fatal(err)
	}
	if ex.Receipt.Verified != 99 {
		t.Errorf("verified = %d; the stored receipt overrode a fresh one", ex.Receipt.Verified)
	}
}

// TestAnUnreadableStoredReceiptIsNotVerified.
//
// Half a receipt is worse than none: a pack would print a partial count as a
// complete one. Empty and unparseable both honestly mean "not verified".
func TestAnUnreadableStoredReceiptIsNotVerified(t *testing.T) {
	ctx := context.Background()
	db, sess := fixture(t)
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.SetSessionReceipt(ctx, sess.ID, `{not json`)
	}); err != nil {
		t.Fatal(err)
	}
	ex, err := export.Build(ctx, db, sess.ID, export.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if ex.Receipt != nil {
		t.Errorf("a corrupt receipt was reported as one: %+v", ex.Receipt)
	}
}

// TestEveryCitationMarkerResolves is the property the whole of phase 0 exists for.
//
// Citation numbers are assigned over a findings set capped by the report's token
// budget at generation time, and that cap is not persisted — so recomputing them
// gives a different mapping whenever the cap bit. A [4] pointing at the wrong
// source in a document sold as verified evidence is the failure this product
// exists to prevent, which is why the numbering is carried rather than derived.
func TestEveryCitationMarkerResolves(t *testing.T) {
	ex := build(t, export.Options{})

	byN := map[int]bool{}
	for _, c := range ex.Report.Citations {
		byN[c.N] = true
	}

	markers := regexp.MustCompile(`\[(\d+)\]`).FindAllStringSubmatch(ex.Report.Markdown, -1)
	if len(markers) == 0 {
		t.Fatal("the fixture's prose carries no citation markers, so this asserts nothing")
	}
	for _, m := range markers {
		n, _ := strconv.Atoi(m[1])
		if !byN[n] {
			t.Errorf("the prose cites [%d] and nothing resolves it", n)
		}
	}
}

// TestACitationCarriesWhatAReaderNeedsToCheckIt: a number and a source are not
// enough to verify anything without re-fetching, which is the point of the quote.
func TestACitationCarriesWhatAReaderNeedsToCheckIt(t *testing.T) {
	ex := build(t, export.Options{})
	if len(ex.Report.Citations) != 2 {
		t.Fatalf("%d citation(s), want 2", len(ex.Report.Citations))
	}
	for _, c := range ex.Report.Citations {
		if c.N == 0 || c.Source == "" || len(c.Quotes) == 0 {
			t.Errorf("citation %+v cannot be checked by a reader", c)
		}
	}
}

// TestProseWithNoMarkersNeedsNoCitations. Dataset mode renders a table, not
// prose, so an empty array there is correct rather than missing — and an empty
// array beside prose that DOES cite is detectable without a separate flag.
func TestProseWithNoMarkersNeedsNoCitations(t *testing.T) {
	ctx := context.Background()
	db, sess := fixture(t)
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		if err := tx.SetSessionReport(ctx, sess.ID, "A table, no prose.", ""); err != nil {
			return err
		}
		return tx.SetSessionReportCitations(ctx, sess.ID, "")
	}); err != nil {
		t.Fatal(err)
	}
	ex, err := export.Build(ctx, db, sess.ID, export.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(ex.Report.Citations) != 0 {
		t.Errorf("%d citation(s) invented for prose that cites nothing", len(ex.Report.Citations))
	}
}

// TestUnreadableCitationsAreDroppedWhole. Half a bibliography renumbers the
// document: a marker resolving to the wrong source is worse than one resolving
// to nothing, because only the second is visible.
func TestUnreadableCitationsAreDroppedWhole(t *testing.T) {
	ctx := context.Background()
	db, sess := fixture(t)
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.SetSessionReportCitations(ctx, sess.ID, `[{"n":1,`)
	}); err != nil {
		t.Fatal(err)
	}
	ex, err := export.Build(ctx, db, sess.ID, export.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(ex.Report.Citations) != 0 {
		t.Errorf("a corrupt bibliography was partly reported: %+v", ex.Report.Citations)
	}
}

// TestNoArrayInTheEnvelopeIsEverNull.
//
// A consumer parses this document in another language and another repository. An
// array that is sometimes [] and sometimes null makes every read a null-check,
// and the one that gets forgotten is the one that ships. Found the honest way: a
// real session whose every search failed marshalled its citations as null,
// because a nil Go slice does, while the neighbouring field already guarded
// against exactly that.
//
// Nullable POINTERS are a different thing and are left alone: receipt null means
// verification never ran, and published_at null means the source carried no date.
// Both are facts a reader needs, not missing arrays.
func TestNoArrayInTheEnvelopeIsEverNull(t *testing.T) {
	ctx := context.Background()
	db, sess := fixtureWith(t, false)

	// A session that found nothing at all: no report, no citations, no claims.
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		if err := tx.SetSessionReport(ctx, sess.ID, "", "nothing survived"); err != nil {
			return err
		}
		return tx.SetSessionReportCitations(ctx, sess.ID, "null")
	}); err != nil {
		t.Fatal(err)
	}

	ex, err := export.Build(ctx, db, sess.ID, export.Options{})
	if err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(ex)
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]any
	if err := json.Unmarshal(b, &raw); err != nil {
		t.Fatal(err)
	}

	// Every array the envelope declares. Adding one here is the price of adding
	// one to the envelope, and TestTheEnvelopeIsPinned is where that is noticed.
	for _, path := range []string{
		"scope", "conflicts", "evidence",
		"report.citations",
		"coverage.unreachable",
		"cost.by_role",
		"session.actors",
	} {
		v := any(raw)
		for _, part := range strings.Split(path, ".") {
			m, ok := v.(map[string]any)
			if !ok {
				t.Fatalf("%s: %s is not an object", path, part)
			}
			v = m[part]
		}
		if v == nil {
			t.Errorf("%s is null; the envelope promises an array", path)
		}
	}
}

// TestCostIsNotTruncatedByAPageOfToolCalls.
//
// The cost section used to be summed from ListToolCalls, whose zero limit means
// 200 — and an ordinary session now makes more than that. Five of ten sessions
// in the recorded corpus report exactly 200 calls, with cost.usd_micros under
// session.spent AND under the sum of cost.by_role beside it, because by_role
// came from a SQL aggregate over every row and the total did not.
//
// The document billed as "what it spent" was silently low. This pins the shape
// rather than the number: whatever the export says it cost must equal what the
// ledger says was spent, and must equal its own breakdown.
func TestCostIsNotTruncatedByAPageOfToolCalls(t *testing.T) {
	ctx := context.Background()
	db, sess := fixture(t)

	// Past the 200-row page, in the roles a real run uses.
	const extra = 260
	led := budget.New(db, budget.DefaultConfig())
	for i := 0; i < extra; i++ {
		res, err := led.Reserve(ctx, sess.ID, 1_000)
		if err != nil {
			t.Fatalf("reserve %d: %v", i, err)
		}
		role := core.RoleExecutor
		if i%3 == 0 {
			role = core.RoleVerifier
		}
		if _, err := led.Settle(ctx, res, []core.ToolCall{{
			Role: role, Type: core.CallLLM, Model: "m",
			Cost: core.Cost{USDMicros: 100, InputTokens: 10, OutputTokens: 2},
		}}); err != nil {
			t.Fatalf("settle %d: %v", i, err)
		}
	}

	ex, err := export.Build(ctx, db, sess.ID, export.Options{})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	if ex.Cost.Calls <= 200 {
		t.Errorf("cost.calls is %d after %d calls; the page limit is still the answer",
			ex.Cost.Calls, extra)
	}
	if ex.Cost.USDMicros != ex.Session.Spent {
		t.Errorf("cost.usd_micros %d, session.spent %d — the envelope disagrees with itself",
			ex.Cost.USDMicros, ex.Session.Spent)
	}

	var byRole int64
	for _, r := range ex.Cost.ByRole {
		byRole += r.USDMicros
	}
	if byRole != ex.Cost.USDMicros {
		t.Errorf("by_role sums to %d and the total says %d; one of them is a page "+
			"and the other is an aggregate", byRole, ex.Cost.USDMicros)
	}
	if ex.Cost.Tokens.Input < int64(extra*10) {
		t.Errorf("tokens.input is %d after %d calls of 10", ex.Cost.Tokens.Input, extra)
	}
}

// TestTheCoverageCountsDoNotExceedTheirDenominator.
//
// A session's leads share search results, and every attempt writes its own
// fetch row, so one URL is routinely passed over by one lead and read by
// another. Counted in both, the parts exceeded the whole: on the recorded
// corpus a session reported 15 read plus 20 not selected against 29 considered,
// so a reader subtracting to find what was left over got a negative number —
// from the one section whose job is to say honestly what was and was not read.
func TestTheCoverageCountsDoNotExceedTheirDenominator(t *testing.T) {
	ctx := context.Background()
	db, sess := fixture(t)

	// The same URL, three ways, as three leads would record it.
	const shared = "https://shared.example/report"
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		for _, o := range []string{"not_selected", "ok", "paywall"} {
			if err := tx.RecordFetchOutcome(ctx, &store.FetchOutcome{
				SessionID: &sess.ID, URL: shared, Domain: "shared.example", Outcome: o,
			}); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		t.Fatalf("seed: %v", err)
	}

	ex, err := export.Build(ctx, db, sess.ID, export.Options{})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	cv := ex.Coverage

	if got := cv.Read + cv.NotSelected + len(cv.Unreachable); got > cv.Considered {
		t.Errorf("read %d + not_selected %d + unreachable %d = %d, over a "+
			"denominator of %d", cv.Read, cv.NotSelected, len(cv.Unreachable),
			got, cv.Considered)
	}

	// And the shared URL is counted as READ, not as passed over: a page reached
	// is reached however many other leads walked past it.
	for _, u := range cv.Unreachable {
		if u.URL == shared {
			t.Errorf("%s is listed unreachable although a lead read it", shared)
		}
	}
}

package sqlite_test

import (
	"context"
	"errors"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/store/sqlite"
)

func open(t *testing.T) (*sqlite.DB, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.db")
	db, err := sqlite.Open(path, sqlite.Options{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.Migrate(context.Background()); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	return db, path
}

func TestMigrateIsIdempotent(t *testing.T) {
	ctx := context.Background()
	db, path := open(t)

	for i := 0; i < 3; i++ {
		if err := db.Migrate(ctx); err != nil {
			t.Fatalf("repeat migrate %d: %v", i, err)
		}
	}

	// And across a reopen, which is the case that actually happens: every CLI
	// invocation migrates on open.
	if err := db.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	db2, err := sqlite.Open(path, sqlite.Options{})
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer db2.Close()
	if err := db2.Migrate(ctx); err != nil {
		t.Fatalf("migrate after reopen: %v", err)
	}
}

func TestGetMissingSessionIsErrNotFound(t *testing.T) {
	ctx := context.Background()
	db, _ := open(t)

	err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		_, err := q.GetSession(ctx, "s_nope")
		return err
	})
	if !errors.Is(err, store.ErrNotFound) {
		t.Fatalf("error = %v, want ErrNotFound", err)
	}
}

func insertSession(t *testing.T, db *sqlite.DB, id string, budget int64) {
	t.Helper()
	ctx := context.Background()
	err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.InsertSession(ctx, &core.Session{
			ID:         id,
			Prompt:     "p",
			Mode:       core.ModeReport,
			ActorTypes: []core.ActorType{core.ActorWeb},
			BudgetUnit: core.BudgetUSD,
			Budget:     budget,
			Status:     core.StatusRunning,
		})
	})
	if err != nil {
		t.Fatalf("insert session: %v", err)
	}
}

// TestBudgetDeltaRejectsNegativeInvariant proves the schema is the last line of
// defence: even a caller that computes a wrong delta cannot drive the ledger
// negative, it gets ErrConflict instead.
func TestBudgetDeltaRejectsNegativeInvariant(t *testing.T) {
	ctx := context.Background()
	db, _ := open(t)
	insertSession(t, db, "s_neg", 1000)

	err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.ApplyBudgetDelta(ctx, "s_neg", store.BudgetDelta{Held: -1})
	})
	if !errors.Is(err, store.ErrConflict) {
		t.Fatalf("error = %v, want ErrConflict", err)
	}

	// State is unchanged: the transaction rolled back.
	var s *core.Session
	if err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		s, err = q.GetSession(ctx, "s_neg")
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if s.Held != 0 {
		t.Fatalf("held = %d after rejected delta, want 0", s.Held)
	}
}

func TestSumCostsByRole(t *testing.T) {
	ctx := context.Background()
	db, _ := open(t)
	insertSession(t, db, "s_roles", 1_000_000)

	rows := []struct {
		role core.Role
		usd  int64
	}{
		{core.RolePlanner, 1000},
		{core.RolePlanner, 2000},
		{core.RoleExecutor, 5000},
		{core.RoleVerifier, 500},
	}
	err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		for _, r := range rows {
			if err := tx.InsertToolCall(ctx, &core.ToolCall{
				SessionID: "s_roles",
				Role:      r.role,
				Type:      core.CallLLM,
				Cost:      core.Cost{USDMicros: r.usd, InputTokens: 10},
			}); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("insert calls: %v", err)
	}

	var byRole map[core.Role]core.Cost
	if err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		byRole, err = q.SumCostsByRole(ctx, "s_roles")
		return err
	}); err != nil {
		t.Fatal(err)
	}

	if got := byRole[core.RolePlanner].USDMicros; got != 3000 {
		t.Errorf("planner = %d, want 3000", got)
	}
	if got := byRole[core.RoleExecutor].USDMicros; got != 5000 {
		t.Errorf("executor = %d, want 5000", got)
	}
	if got := byRole[core.RoleVerifier].USDMicros; got != 500 {
		t.Errorf("verifier = %d, want 500", got)
	}
	if _, ok := byRole[core.RoleOutput]; ok {
		t.Error("output role present with no rows")
	}
}

// TestConcurrentWritesDoNotLock is the regression test for SQLite's classic
// failure mode. A naive pool produces "database is locked" here; the
// single-writer pool serializes instead.
func TestConcurrentWritesDoNotLock(t *testing.T) {
	ctx := context.Background()
	db, _ := open(t)
	insertSession(t, db, "s_conc", 10_000_000)

	const writers = 24
	var wg sync.WaitGroup
	errCh := make(chan error, writers)
	start := make(chan struct{})

	for i := 0; i < writers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
				if err := tx.InsertToolCall(ctx, &core.ToolCall{
					SessionID: "s_conc",
					Role:      core.RoleExecutor,
					Type:      core.CallFetch,
					Cost:      core.Cost{USDMicros: 100},
				}); err != nil {
					return err
				}
				return tx.ApplyBudgetDelta(ctx, "s_conc", store.BudgetDelta{Spent: 100, ToolCallCount: 1})
			})
			if err != nil {
				errCh <- err
			}
		}()
	}
	close(start)
	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Fatalf("concurrent write failed: %v", err)
	}

	var s *core.Session
	var total core.Cost
	if err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		if s, err = q.GetSession(ctx, "s_conc"); err != nil {
			return err
		}
		total, err = q.SumCosts(ctx, "s_conc")
		return err
	}); err != nil {
		t.Fatal(err)
	}

	if want := int64(writers * 100); s.Spent != want {
		t.Errorf("spent = %d, want %d (lost update)", s.Spent, want)
	}
	if total.USDMicros != s.Spent {
		t.Errorf("materialized spent %d != ledger sum %d", s.Spent, total.USDMicros)
	}
	if s.ToolCallCount != writers {
		t.Errorf("tool_call_count = %d, want %d", s.ToolCallCount, writers)
	}
}

// TestReadsProceedDuringOpenWrite is the property that makes SQLite sufficient
// for this workload, and the reason `mole trace` can run against a database the
// daemon is actively writing.
//
// Under WAL, readers see the last committed snapshot and never block on an open
// write transaction. Without WAL this deadlocks; with it, the read returns the
// pre-transaction state immediately.
func TestReadsProceedDuringOpenWrite(t *testing.T) {
	ctx := context.Background()
	db, _ := open(t)
	insertSession(t, db, "s_wal", 1_000_000)

	writing := make(chan struct{})
	readDone := make(chan error, 1)

	go func() {
		_ = db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
			if err := tx.InsertToolCall(ctx, &core.ToolCall{
				SessionID: "s_wal", Role: core.RoleExecutor, Type: core.CallLLM,
				Cost: core.Cost{USDMicros: 999},
			}); err != nil {
				return err
			}
			close(writing)
			// Hold the write transaction open while a reader runs.
			<-readDone
			return nil
		})
	}()

	<-writing

	// This must not block on the open writer.
	var n int64
	err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		n, err = q.CountToolCalls(ctx, "s_wal")
		return err
	})
	readDone <- err
	if err != nil {
		t.Fatalf("read during open write transaction: %v", err)
	}
	// The reader sees the committed snapshot, not the in-flight write.
	if n != 0 {
		t.Fatalf("reader saw %d uncommitted rows, want 0", n)
	}
}

func TestRollbackDiscardsPartialWork(t *testing.T) {
	ctx := context.Background()
	db, _ := open(t)
	insertSession(t, db, "s_rb", 1_000_000)

	sentinel := errors.New("abort")
	err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		if err := tx.InsertToolCall(ctx, &core.ToolCall{
			SessionID: "s_rb", Role: core.RoleExecutor, Type: core.CallLLM,
			Cost: core.Cost{USDMicros: 5000},
		}); err != nil {
			return err
		}
		return sentinel
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("error = %v, want sentinel", err)
	}

	var n int64
	if err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		n, err = q.CountToolCalls(ctx, "s_rb")
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Fatalf("rolled-back transaction left %d rows", n)
	}
}

func TestSpanLifecycle(t *testing.T) {
	ctx := context.Background()
	db, _ := open(t)
	insertSession(t, db, "s_span", 1000)

	span := &core.Span{
		SessionID: "s_span",
		Name:      "executor",
		Attrs:     map[string]string{"actor": "web"},
	}
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.StartSpan(ctx, span)
	}); err != nil {
		t.Fatalf("start span: %v", err)
	}

	end := time.Now().UTC()
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.EndSpan(ctx, span.ID, end, "ok")
	}); err != nil {
		t.Fatalf("end span: %v", err)
	}

	// Ending twice must not silently succeed — it would mask a double-close bug.
	err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.EndSpan(ctx, span.ID, end, "ok")
	})
	if !errors.Is(err, store.ErrNotFound) {
		t.Fatalf("second EndSpan error = %v, want ErrNotFound", err)
	}

	var spans []*core.Span
	if err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		spans, err = q.ListSpans(ctx, "s_span")
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	if spans[0].EndedAt == nil {
		t.Fatal("span not closed")
	}
	if spans[0].Attrs["actor"] != "web" {
		t.Errorf("attrs lost: %v", spans[0].Attrs)
	}
}

func TestInvalidEnumIsRejectedBySchema(t *testing.T) {
	ctx := context.Background()
	db, _ := open(t)
	insertSession(t, db, "s_enum", 1000)

	// core.ToolCall.Validate catches this first, so go around it to prove the
	// CHECK constraint is really there.
	err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.InsertToolCall(ctx, &core.ToolCall{
			SessionID: "s_enum",
			Role:      core.Role("not-a-role"),
			Type:      core.CallLLM,
		})
	})
	if err == nil {
		t.Fatal("invalid role was accepted")
	}
}

// TestFetchOutcomeStats covers the query §17.1's gate reads: the rate per
// cause, and the domains each cause concentrates in.
func TestFetchOutcomeStats(t *testing.T) {
	ctx := context.Background()
	db, _ := open(t)

	rows := []struct {
		domain, outcome string
	}{
		{"spa.example", "js_required"},
		{"spa.example", "js_required"},
		{"other.example", "js_required"},
		{"news.example", "paywall"},
		{"good.example", "ok"},
		{"good.example", "ok"},
		{"good.example", "ok"},
		{"blocked.example", "robots_denied"},
	}

	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		for _, r := range rows {
			if err := tx.RecordFetchOutcome(ctx, &store.FetchOutcome{
				URL:     "https://" + r.domain + "/x",
				Domain:  r.domain,
				Outcome: r.outcome,
			}); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		t.Fatalf("record: %v", err)
	}

	var stats []store.FetchStat
	if err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		stats, err = q.FetchOutcomeStats(ctx, time.Now().Add(-time.Hour), 5)
		return err
	}); err != nil {
		t.Fatal(err)
	}

	counts := map[string]int64{}
	domains := map[string][]store.DomainCount{}
	for _, s := range stats {
		counts[s.Outcome] = s.Count
		domains[s.Outcome] = s.Domains
	}

	if counts["ok"] != 3 || counts["js_required"] != 3 || counts["paywall"] != 1 {
		t.Errorf("counts = %v", counts)
	}

	// The ranked domain list is as useful as the rate: a failure concentrated
	// in one site is a denylist entry, not an architecture change.
	js := domains["js_required"]
	if len(js) == 0 || js[0].Domain != "spa.example" || js[0].Count != 2 {
		t.Errorf("js_required top domain = %+v, want spa.example x2", js)
	}
}

// TestFetchOutcomeSurvivesWithoutSession: a probe made outside any session is
// still evidence about a domain and must not be dropped for lack of a parent.
func TestFetchOutcomeSurvivesWithoutSession(t *testing.T) {
	ctx := context.Background()
	db, _ := open(t)

	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.RecordFetchOutcome(ctx, &store.FetchOutcome{
			URL: "https://a.example/", Domain: "a.example", Outcome: "ok",
		})
	}); err != nil {
		t.Fatalf("record without session: %v", err)
	}
}

// TestAssertionStrengthAndConfidenceAreDistinctColumns guards the split §11.3
// requires, at the layer where it can silently collapse.
//
// Both fields are float64 in the 0-1 range and adjacent in the struct, the
// INSERT, and the SELECT. Bind them to one column and everything still compiles,
// every range check still passes, and the only symptom is that the extractor's
// self-report is back to ordering the report — the bug this split exists to
// remove. Distinct values in, distinct values out.
func TestAssertionStrengthAndConfidenceAreDistinctColumns(t *testing.T) {
	ctx := context.Background()
	db, _ := open(t)
	insertSession(t, db, "s_claims", 1_000_000)

	// Deliberately opposite ends of the range: a swap or a shared column shows up
	// as one value in both fields.
	want := core.Claim{
		SessionID: "s_claims",
		LeadID:    "l_1",
		Text:      "A claim.",
		Source:    "https://a.example/x",
		Quote:     "a quote long enough to constitute real evidence",

		AssertionStrength: 0.9,
		Confidence:        0.1,
	}

	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.InsertClaims(ctx, []core.Claim{want})
	}); err != nil {
		t.Fatalf("insert: %v", err)
	}

	var got []*core.Claim
	if err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		got, err = q.ListClaims(ctx, "s_claims", 10)
		return err
	}); err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d claims, want 1", len(got))
	}

	if got[0].AssertionStrength != want.AssertionStrength {
		t.Errorf("AssertionStrength = %v, want %v", got[0].AssertionStrength, want.AssertionStrength)
	}
	if got[0].Confidence != want.Confidence {
		t.Errorf("Confidence = %v, want %v", got[0].Confidence, want.Confidence)
	}
}

// insertClaim writes one claim and returns its assigned ID.
func insertClaim(t *testing.T, db *sqlite.DB, sessionID, text, source string) string {
	t.Helper()
	batch := []core.Claim{{
		SessionID: sessionID, LeadID: "l_1", Text: text, Source: source,
		Quote: "a quote long enough to constitute real evidence",
	}}
	if err := db.WithTx(context.Background(), func(ctx context.Context, tx store.Tx) error {
		return tx.InsertClaims(ctx, batch)
	}); err != nil {
		t.Fatalf("insert claim: %v", err)
	}
	return batch[0].ID
}

func listEdges(t *testing.T, db *sqlite.DB, sessionID string) []*core.ClaimEdge {
	t.Helper()
	var out []*core.ClaimEdge
	if err := db.Read(context.Background(), func(ctx context.Context, q store.Queries) error {
		var err error
		out, err = q.ListEdges(ctx, sessionID, 100)
		return err
	}); err != nil {
		t.Fatalf("list edges: %v", err)
	}
	return out
}

// TestSymmetricEdgesAreStoredOnce is what stops §11.3 counting one disagreement
// twice.
//
// The UNIQUE constraint is on (from_id, to_id, kind), so "A contradicts B" and "B
// contradicts A" are two distinct rows as far as the schema is concerned — and
// they will both be produced, because clustering reaches the same pair from either
// end. Confidence is penalized per contradicting edge, so an unconstrained pair
// docks both claims twice for one disagreement.
func TestSymmetricEdgesAreStoredOnce(t *testing.T) {
	ctx := context.Background()
	db, _ := open(t)
	insertSession(t, db, "s_graph", 1_000_000)

	a := insertClaim(t, db, "s_graph", "Claim A.", "https://a.example/1")
	b := insertClaim(t, db, "s_graph", "Claim B.", "https://b.example/1")

	// The same disagreement, discovered from both ends, plus a duplicate_of pair
	// for good measure.
	err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.InsertEdges(ctx, []core.ClaimEdge{
			{SessionID: "s_graph", FromID: a, ToID: b, Kind: core.EdgeContradicts, Weight: 0.5},
			{SessionID: "s_graph", FromID: b, ToID: a, Kind: core.EdgeContradicts, Weight: 0.9},
			{SessionID: "s_graph", FromID: b, ToID: a, Kind: core.EdgeDuplicateOf},
			{SessionID: "s_graph", FromID: a, ToID: b, Kind: core.EdgeDuplicateOf},
		})
	})
	if err != nil {
		t.Fatalf("insert edges: %v", err)
	}

	edges := listEdges(t, db, "s_graph")
	if len(edges) != 2 {
		for _, e := range edges {
			t.Logf("  %s -%s-> %s w=%v", e.FromID, e.Kind, e.ToID, e.Weight)
		}
		t.Fatalf("%d edges stored, want 2: one contradiction and one duplicate, "+
			"each discovered from both ends", len(edges))
	}

	for _, e := range edges {
		if e.FromID > e.ToID {
			t.Errorf("%s edge is not canonically ordered: %s -> %s", e.Kind, e.FromID, e.ToID)
		}
		// The later judgement wins on conflict, so the contradiction carries 0.9.
		if e.Kind == core.EdgeContradicts && e.Weight != 0.9 {
			t.Errorf("contradiction weight = %v, want 0.9 (the later judgement)", e.Weight)
		}
	}
}

// TestDirectionalEdgesKeepTheirDirection is the other half: canonicalizing
// everything would erase the arrow that makes an edge mean something.
//
// `supersedes` is decided from PublishedAt, so reversing it turns "the 2025 paper
// supersedes the 2019 one" into staleness pointing backwards. `refines` and
// `supports` are equally directional — a specific finding supporting a general
// conclusion is not the same statement reversed.
//
// Both ID orderings are exercised deliberately. The first version of this test
// sorted the pair and skipped when the ordering came out the other way; IDs are
// crypto/rand, so it silently ran half the time and reported a pass the rest.
func TestDirectionalEdgesKeepTheirDirection(t *testing.T) {
	ctx := context.Background()
	db, _ := open(t)
	insertSession(t, db, "s_dir", 1_000_000)

	x := insertClaim(t, db, "s_dir", "One finding.", "https://one.example/1")
	y := insertClaim(t, db, "s_dir", "Another finding.", "https://two.example/1")
	lo, hi := x, y
	if hi < lo {
		lo, hi = hi, lo
	}

	// One edge with the endpoints already ordered, one against the ordering, so
	// neither outcome can be reached by canonicalizing in a fixed direction.
	want := []core.ClaimEdge{
		{SessionID: "s_dir", FromID: hi, ToID: lo, Kind: core.EdgeSupersedes},
		{SessionID: "s_dir", FromID: lo, ToID: hi, Kind: core.EdgeRefines},
	}
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.InsertEdges(ctx, want)
	}); err != nil {
		t.Fatalf("insert edges: %v", err)
	}

	got := map[core.EdgeKind][2]string{}
	for _, e := range listEdges(t, db, "s_dir") {
		got[e.Kind] = [2]string{e.FromID, e.ToID}
	}
	if len(got) != 2 {
		t.Fatalf("%d distinct edge kinds stored, want 2", len(got))
	}
	for _, w := range want {
		if g := got[w.Kind]; g != [2]string{w.FromID, w.ToID} {
			t.Errorf("%s edge stored as %s -> %s, want %s -> %s; the arrow is the meaning",
				w.Kind, g[0], g[1], w.FromID, w.ToID)
		}
	}
}

// TestSelfEdgesAndUnknownKindsAreRejected covers what the schema lets through.
//
// claim_edges has no foreign key to claims and no CHECK against from_id = to_id,
// so a self-edge persists silently — and a claim that corroborates itself inflates
// its own confidence, which is the one number §11.3 exists to make trustworthy.
func TestSelfEdgesAndUnknownKindsAreRejected(t *testing.T) {
	ctx := context.Background()
	db, _ := open(t)
	insertSession(t, db, "s_bad", 1_000_000)
	a := insertClaim(t, db, "s_bad", "Claim A.", "https://a.example/1")
	b := insertClaim(t, db, "s_bad", "Claim B.", "https://b.example/1")

	cases := map[string]core.ClaimEdge{
		"self-edge":    {SessionID: "s_bad", FromID: a, ToID: a, Kind: core.EdgeSupports},
		"unknown kind": {SessionID: "s_bad", FromID: a, ToID: b, Kind: core.EdgeKind("agrees_vaguely")},
		"empty target": {SessionID: "s_bad", FromID: a, ToID: "", Kind: core.EdgeSupports},
	}

	for name, edge := range cases {
		err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
			return tx.InsertEdges(ctx, []core.ClaimEdge{edge})
		})
		if err == nil {
			t.Errorf("%s was accepted", name)
		}
	}
	if n := len(listEdges(t, db, "s_bad")); n != 0 {
		t.Errorf("%d edges stored, want 0", n)
	}
}

// TestScoringMarksVerifiedWithoutTouchingEvidence covers both halves of
// ScoreClaims.
//
// verified_at exists because confidence cannot answer "has this been scored":
// §11.3 returns 0 for an uncorroborated claim carrying a contradiction, so a
// zero-confidence claim is scored as often as it is unexamined, and the Verifier
// would rescore it on every pass.
//
// And a scoring pass must not be able to rewrite the claim's text, quote or
// source. The quote is what §11.5 checks a citation against; a verifier that
// could edit it would be grading its own homework.
func TestScoringMarksVerifiedWithoutTouchingEvidence(t *testing.T) {
	ctx := context.Background()
	db, _ := open(t)
	insertSession(t, db, "s_score", 1_000_000)

	kept := insertClaim(t, db, "s_score", "Claim A.", "https://a.example/1")
	untouched := insertClaim(t, db, "s_score", "Claim B.", "https://b.example/1")

	unverified := func() []*core.Claim {
		var out []*core.Claim
		if err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
			var err error
			out, err = q.ListUnverifiedClaims(ctx, "s_score", 100)
			return err
		}); err != nil {
			t.Fatal(err)
		}
		return out
	}

	if n := len(unverified()); n != 2 {
		t.Fatalf("%d unverified claims before scoring, want 2", n)
	}

	// Score to exactly 0 — the value that is indistinguishable from "unscored"
	// without verified_at, and the value §11.3 gives a contradicted claim.
	grounded := false
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.ScoreClaims(ctx, []store.ClaimScore{
			{ClaimID: kept, Confidence: 0, Grounded: &grounded},
		})
	}); err != nil {
		t.Fatalf("score: %v", err)
	}

	left := unverified()
	if len(left) != 1 || left[0].ID != untouched {
		t.Fatalf("unverified set = %v, want only the unscored claim", left)
	}

	var all []*core.Claim
	if err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		all, err = q.ListClaims(ctx, "s_score", 100)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	for _, c := range all {
		if c.ID != kept {
			continue
		}
		if c.VerifiedAt == nil {
			t.Error("scored claim has no verified_at, so it will be rescored forever")
		}
		if c.Grounded == nil || *c.Grounded {
			t.Errorf("Grounded = %v, want false", c.Grounded)
		}
		// Evidence intact.
		if c.Text != "Claim A." || c.Source != "https://a.example/1" ||
			c.Quote != "a quote long enough to constitute real evidence" {
			t.Errorf("scoring altered evidence: text=%q source=%q quote=%q",
				c.Text, c.Source, c.Quote)
		}
	}

	// A score for a claim the store never had is a Verifier reasoning over
	// nothing; affecting zero rows silently would report success.
	err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.ScoreClaims(ctx, []store.ClaimScore{{ClaimID: "c_nope", Confidence: 1}})
	})
	if !errors.Is(err, store.ErrNotFound) {
		t.Errorf("scoring a missing claim: err = %v, want ErrNotFound", err)
	}
}

// TestAFinishedSessionRecordsWhyItStopped.
//
// core.StatusExhausted is the string "budget_exhausted", and FIVE different
// endings produce it: the budget actually running out, and the four ceilings
// that have nothing to do with money — max_tool_calls, max_leads,
// max_wallclock, and a context deadline. HitCeiling already returns which one
// it was and the executor already assigns it to Result.StoppedBecause; the
// value was then dropped, because nothing persisted it.
//
// A recorded session saying "budget_exhausted" with 90% of its dollars unspent
// is not a puzzle anybody should have to solve twice.
func TestAFinishedSessionRecordsWhyItStopped(t *testing.T) {
	ctx := context.Background()
	db, _ := open(t)
	insertSession(t, db, "s_why", 400_000)

	err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.FinishSession(ctx, "s_why", core.StatusExhausted, "max_tool_calls")
	})
	if err != nil {
		t.Fatalf("finish: %v", err)
	}

	var got *core.Session
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		s, err := tx.GetSession(ctx, "s_why")
		got = s
		return err
	}); err != nil {
		t.Fatalf("read back: %v", err)
	}

	if got.Status != core.StatusExhausted {
		t.Errorf("status = %q", got.Status)
	}
	if got.StoppedBecause != "max_tool_calls" {
		t.Errorf("stopped because %q, want max_tool_calls — the reason was dropped",
			got.StoppedBecause)
	}
}

// TestAnOrdinaryEndingNamesNoCeiling, so the field is evidence rather than
// decoration: a session that simply finished has nothing to explain.
func TestAnOrdinaryEndingNamesNoCeiling(t *testing.T) {
	ctx := context.Background()
	db, _ := open(t)
	insertSession(t, db, "s_done", 400_000)

	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.FinishSession(ctx, "s_done", core.StatusDone, "")
	}); err != nil {
		t.Fatalf("finish: %v", err)
	}

	var got *core.Session
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		s, err := tx.GetSession(ctx, "s_done")
		got = s
		return err
	}); err != nil {
		t.Fatalf("read back: %v", err)
	}
	if got.StoppedBecause != "" {
		t.Errorf("a completed session claims it stopped because %q", got.StoppedBecause)
	}
}

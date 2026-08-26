package eval_test

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/budget"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/eval"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/store/sqlite"
)

type fixture struct {
	db   *sqlite.DB
	led  *budget.Ledger
	sess *core.Session
	lead core.Lead
}

func newFixture(t *testing.T, budgetAmount int64) *fixture {
	t.Helper()
	return newFixtureUnit(t, core.BudgetUSD, budgetAmount)
}

// newFixtureUnit is newFixture with the budget unit chosen, so a test can check
// that a metric's Value is expressed in the unit its label claims.
func newFixtureUnit(t *testing.T, unit core.BudgetUnit, budgetAmount int64) *fixture {
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
		Prompt: "a question", Mode: core.ModeReport,
		ActorTypes: []core.ActorType{core.ActorWeb},
		BudgetUnit: unit, Budget: budgetAmount,
	})
	if err != nil {
		t.Fatalf("create session: %v", err)
	}

	lead := core.Lead{
		ID: core.NewLeadID(), SessionID: sess.ID,
		ActorType: core.ActorWeb, Query: "a question", Status: core.LeadQueued,
	}
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.InsertLead(ctx, &lead)
	}); err != nil {
		t.Fatalf("insert lead: %v", err)
	}

	return &fixture{db: db, led: led, sess: sess, lead: lead}
}

// spend runs a reservation through to settle, so the ledger reflects it the way
// a real run would.
func (f *fixture) spend(t *testing.T, reserve, actual int64) {
	t.Helper()
	ctx := context.Background()
	r, err := f.led.ReserveFor(ctx, f.sess.ID, f.lead.ID, reserve)
	if err != nil {
		t.Fatalf("reserve: %v", err)
	}
	if _, err := f.led.Settle(ctx, r, []core.ToolCall{{
		Role: core.RoleExecutor, Type: core.CallLLM, Model: "m",
		Cost: core.Cost{USDMicros: actual, InputTokens: 100, OutputTokens: 10},
	}}); err != nil {
		t.Fatalf("settle: %v", err)
	}
}

func (f *fixture) addClaims(t *testing.T, claims ...core.Claim) {
	t.Helper()
	for i := range claims {
		claims[i].SessionID = f.sess.ID
		claims[i].LeadID = f.lead.ID
		if claims[i].RetrievedAt.IsZero() {
			claims[i].RetrievedAt = time.Now().UTC()
		}
	}
	if err := f.db.WithTx(context.Background(), func(ctx context.Context, tx store.Tx) error {
		return tx.InsertClaims(ctx, claims)
	}); err != nil {
		t.Fatalf("insert claims: %v", err)
	}
}

func (f *fixture) score(t *testing.T) eval.Scorecard {
	t.Helper()
	card, err := eval.Score(context.Background(), f.db, f.sess.ID, eval.Options{})
	if err != nil {
		t.Fatalf("score: %v", err)
	}
	return card
}

func metric(t *testing.T, card eval.Scorecard, name string) eval.Metric {
	t.Helper()
	for _, m := range card.Metrics {
		if m.Name == name {
			return m
		}
	}
	t.Fatalf("scorecard has no metric %q", name)
	return eval.Metric{}
}

func goodClaim(text, source string) core.Claim {
	return core.Claim{
		Text:        text,
		Source:      source,
		Quote:       "a quote long enough to constitute actual evidence",
		QuoteOffset: 10,
		Confidence:  0.8,
	}
}

// ---------------------------------------------------------------------------
// The happy path
// ---------------------------------------------------------------------------

func TestCleanSessionPasses(t *testing.T) {
	f := newFixture(t, 3*core.MicrosPerUSD)
	f.spend(t, 100_000, 40_000)
	f.addClaims(t,
		goodClaim("First claim.", "https://a.example/x"),
		goodClaim("Second claim.", "https://b.example/y"),
	)

	card := f.score(t)
	if card.Failed() {
		for _, m := range card.Metrics {
			if m.Regression {
				t.Errorf("unexpected regression: %s — %s", m.Name, m.Detail)
			}
		}
	}
	// 40000 micro-dollars over 2 claims = 20000 micros each = $0.02. The Value
	// must be in the unit its label claims: reporting 20000 next to Unit "usd"
	// said $20,000 per claim, which is the bug this pins.
	got := metric(t, card, "cost per claim")
	if got.Value != 0.02 {
		t.Errorf("cost per claim = %v, want 0.02 ($0.02 = 40000 micros over 2 claims)", got.Value)
	}
	if got.Unit != "usd" {
		t.Errorf("unit = %q, want \"usd\"", got.Unit)
	}
}

// ---------------------------------------------------------------------------
// Each regression must actually fire
// ---------------------------------------------------------------------------

// TestBudgetOvershootIsARegression. §14.3 says max overshoot "must be ~0", and
// the reserve/settle design exists to make it impossible. A non-zero value
// means the design is not holding, not that a limit was slightly loose — so it
// fails the build rather than lowering a score.
func TestBudgetOvershootIsARegression(t *testing.T) {
	f := newFixture(t, 100_000)
	// Reserve modestly, then settle for far more: the actual cost exceeding the
	// estimate is exactly how an overshoot happens in practice.
	f.spend(t, 10_000, 250_000)

	card := f.score(t)
	m := metric(t, card, "budget overshoot")
	if !m.Regression {
		t.Errorf("a 150%% overshoot was not flagged: %s", m.Detail)
	}
	if m.Value <= 0 {
		t.Errorf("overshoot reported as %.1f%%", m.Value)
	}
	if !card.Failed() {
		t.Error("scorecard did not fail")
	}
}

// TestStrandedHoldIsARegression. Budget in an unsettled reservation is neither
// spent nor available; it is invisible in a spend total, and it is how a long
// session runs out of money it never used.
func TestStrandedHoldIsARegression(t *testing.T) {
	f := newFixture(t, 3*core.MicrosPerUSD)
	if _, err := f.led.ReserveFor(context.Background(), f.sess.ID, f.lead.ID, 500_000); err != nil {
		t.Fatal(err)
	}
	// Deliberately never settled or released.

	card := f.score(t)
	m := metric(t, card, "holds released")
	if !m.Regression {
		t.Errorf("a stranded hold was not flagged: %s", m.Detail)
	}
	if !card.Failed() {
		t.Error("scorecard did not fail")
	}
}

// TestMalformedClaimsAreARegression. Each of these is something §11.5 promises
// cannot reach the store, so one appearing means the pipeline let it through.
func TestMalformedClaimsAreARegression(t *testing.T) {
	cases := map[string]core.Claim{
		"empty text":          {Text: "  ", Source: "https://a.example", Quote: "a quote long enough to be real evidence", Confidence: 0.5},
		"quote too short":     {Text: "A claim.", Source: "https://a.example", Quote: "too short", Confidence: 0.5},
		"negative offset":     {Text: "A claim.", Source: "https://a.example", Quote: "a quote long enough to be real evidence", QuoteOffset: -5, Confidence: 0.5},
		"unresolvable source": {Text: "A claim.", Source: "not a url", Quote: "a quote long enough to be real evidence", Confidence: 0.5},
		"confidence over 1":   {Text: "A claim.", Source: "https://a.example", Quote: "a quote long enough to be real evidence", Confidence: 4},
	}

	for name, bad := range cases {
		f := newFixture(t, 3*core.MicrosPerUSD)
		f.spend(t, 100_000, 10_000)
		f.addClaims(t, goodClaim("A fine claim.", "https://ok.example"), bad)

		card := f.score(t)
		m := metric(t, card, "claim integrity")
		if !m.Regression {
			t.Errorf("%s: not flagged — %s", name, m.Detail)
		}
		if m.Value >= 100 {
			t.Errorf("%s: integrity reported as %.1f%%", name, m.Value)
		}
	}
}

// ---------------------------------------------------------------------------
// Diagnostics must NOT fail the build
// ---------------------------------------------------------------------------

// TestSourceConcentrationIsDiagnosticNotFatal. A question answered well by one
// good source is legitimate. Failing on it would train people to ignore the
// scorecard, which is worse than not reporting it.
func TestSourceConcentrationIsDiagnosticNotFatal(t *testing.T) {
	f := newFixture(t, 3*core.MicrosPerUSD)
	f.spend(t, 100_000, 10_000)
	f.addClaims(t,
		goodClaim("One.", "https://only.example/a"),
		goodClaim("Two.", "https://only.example/a"),
		goodClaim("Three.", "https://only.example/a"),
	)

	card := f.score(t)
	m := metric(t, card, "source concentration")
	if m.Value != 100 {
		t.Errorf("concentration = %.1f%%, want 100%% (every claim from one source)", m.Value)
	}
	if m.Regression || card.Failed() {
		t.Error("total source concentration failed the build; it is diagnostic")
	}
}

// TestNoClaimsIsNotApplicableRatherThanZero. A run that found nothing has no
// cost-per-claim, and reporting 0 would look like the best possible score.
func TestNoClaimsIsNotApplicableRatherThanZero(t *testing.T) {
	f := newFixture(t, 3*core.MicrosPerUSD)
	f.spend(t, 100_000, 50_000)

	card := f.score(t)
	for _, name := range []string{"cost per claim", "claim integrity", "source concentration"} {
		m := metric(t, card, name)
		if m.Status != eval.NotApplicable {
			t.Errorf("%s: status = %s, want n/a", name, m.Status)
		}
		if m.Regression {
			t.Errorf("%s: an empty run was reported as a regression", name)
		}
	}
	if card.Failed() {
		t.Error("a run that found nothing failed the mechanical checks; it is a quality issue, not a contract violation")
	}
}

// ---------------------------------------------------------------------------
// Honesty about what is not measured
// ---------------------------------------------------------------------------

// TestBlockedMetricsAreNamedNotOmitted. This is the design decision the package
// exists to make: four of nine §14.3 metrics will read zero until M4 and M8,
// and a scorecard that omitted them would read as complete — so a genuine
// future regression would look normal.
func TestBlockedMetricsAreNamedNotOmitted(t *testing.T) {
	f := newFixture(t, 3*core.MicrosPerUSD)
	f.spend(t, 100_000, 10_000)
	f.addClaims(t, goodClaim("A claim.", "https://a.example"))

	card := f.score(t)
	want := map[string]bool{
		"claim precision": false, "grounding rate": false, "citation accuracy": false,
		"contradiction recall": false, "staleness detection": false, "exfil regression": false,
	}
	for _, m := range card.Metrics {
		if _, expected := want[m.Name]; !expected {
			continue
		}
		want[m.Name] = true
		if m.Status != eval.Blocked {
			t.Errorf("%s: status = %s, want blocked", m.Name, m.Status)
		}
		if m.Reason == "" {
			t.Errorf("%s: blocked with no reason given", m.Name)
		}
		if m.Regression {
			t.Errorf("%s: an unmeasurable metric was reported as a regression", m.Name)
		}
	}
	for name, seen := range want {
		if !seen {
			t.Errorf("§14.3 metric %q is missing from the scorecard entirely", name)
		}
	}
}

// TestBlockedMetricsCarryNoValue: a blocked metric with a number attached is
// exactly the confusion the status exists to prevent.
func TestBlockedMetricsCarryNoValue(t *testing.T) {
	f := newFixture(t, 3*core.MicrosPerUSD)
	card := f.score(t)
	for _, m := range card.Metrics {
		if m.Status == eval.Blocked && m.Value != 0 {
			t.Errorf("%s is blocked but carries value %v", m.Name, m.Value)
		}
	}
}

// findMetric returns a named metric, or nil.
func findMetric(card eval.Scorecard, name string) *eval.Metric {
	for i := range card.Metrics {
		if card.Metrics[i].Name == name {
			return &card.Metrics[i]
		}
	}
	return nil
}

// TestACountIsNotRecall is the honesty property of the graph metrics.
//
// M4 made contradictions findable, so they are counted — as "disagreement rate".
// Recall is a different number: it needs to know what the graph MISSED, which needs
// disagreements planted on purpose. Reporting a count as recall would be the most
// flattering possible confusion, because it rises when the pipeline gets noisier
// rather than when it gets better.
func TestACountIsNotRecall(t *testing.T) {
	card := scoreWith(t, nil, nil)

	if m := findMetric(card, "disagreement rate"); m == nil || m.Status == eval.Blocked {
		t.Error("disagreement rate is not measured; M4 made it computable")
	}
	recall := findMetric(card, "contradiction recall")
	if recall == nil {
		t.Fatal("contradiction recall is not named at all")
	}
	if recall.Status != eval.Blocked {
		t.Errorf("contradiction recall reported as %s — a count was promoted to recall", recall.Status)
	}
	if !strings.Contains(recall.Reason, "MISSED") {
		t.Errorf("the blocked reason does not say what is missing: %q", recall.Reason)
	}
	// Same shape for staleness.
	if m := findMetric(card, "staleness separation"); m == nil {
		t.Error("staleness separation is not reported")
	}
	if m := findMetric(card, "staleness detection"); m == nil || m.Status != eval.Blocked {
		t.Error("staleness detection is no longer named as blocked")
	}
}

// TestVerificationCoverageIsTheDenominator. Zero disagreements on a session where
// nothing was verified is not a clean bill of health, and a reader seeing only the
// disagreement count cannot tell those apart.
func TestVerificationCoverageIsTheDenominator(t *testing.T) {
	unverified := []core.Claim{
		{Text: "A.", Source: "https://a.example/1", Quote: "a quote long enough to be real evidence"},
		{Text: "B.", Source: "https://b.example/1", Quote: "a quote long enough to be real evidence"},
	}
	card := scoreWith(t, unverified, nil)
	m := findMetric(card, "verification coverage")
	if m == nil {
		t.Fatal("verification coverage missing")
	}
	if m.Value != 0 {
		t.Errorf("coverage = %v on an unverified session", m.Value)
	}
	if !strings.Contains(m.Detail, "measured over nothing") {
		t.Errorf("a zero-coverage session does not warn that the graph numbers mean nothing: %q",
			m.Detail)
	}
}

// TestGroundingRateExcludesInconclusiveChecks.
//
// A vanished quote, an unreachable host and a judge that declined all get recorded and
// none says anything about whether a quote supports its claim. Counting them as
// failures would make the metric track network weather.
func TestGroundingRateExcludesInconclusiveChecks(t *testing.T) {
	yes, no := true, false
	claims := []core.Claim{
		{Text: "Confirmed.", Source: "https://a.example/1", Quote: "a quote long enough to be real evidence",
			Grounded: &yes},
		{Text: "Unsupported.", Source: "https://b.example/1", Quote: "a quote long enough to be real evidence",
			Grounded: &no},
		// Checked, inconclusive: note set, Grounded nil.
		{Text: "Page changed.", Source: "https://c.example/1", Quote: "a quote long enough to be real evidence",
			GroundingNote: "the quote is no longer present in the source"},
		{Text: "Host down.", Source: "https://d.example/1", Quote: "a quote long enough to be real evidence",
			GroundingNote: "source could not be re-read"},
		// Never checked at all.
		{Text: "Untouched.", Source: "https://e.example/1", Quote: "a quote long enough to be real evidence"},
	}
	card := scoreWith(t, claims, nil)
	m := findMetric(card, "grounding rate")
	if m == nil {
		t.Fatal("grounding rate missing")
	}
	if m.Status != eval.Measured {
		t.Fatalf("status = %s, want measured", m.Status)
	}
	// 1 of 2 DECISIVE checks confirmed — not 1 of 4, and not 1 of 5.
	if m.Value != 50 {
		t.Errorf("grounding rate = %v%%, want 50 (1 of 2 decisive checks)", m.Value)
	}
	if !strings.Contains(m.Detail, "2 inconclusive, excluded") {
		t.Errorf("inconclusive checks were not declared: %q", m.Detail)
	}
	if !strings.Contains(m.Detail, "NOT supported") {
		t.Errorf("the unsupported claim is not called out: %q", m.Detail)
	}
}

// TestGroundingRateIsBlockedNotZeroWhenNothingRan. 0% would read as "every checked
// claim failed" instead of "nothing was checked" — the same conflation slice 0 removed
// from confidence.
func TestGroundingRateIsBlockedNotZeroWhenNothingRan(t *testing.T) {
	card := scoreWith(t, []core.Claim{
		{Text: "A.", Source: "https://a.example/1", Quote: "a quote long enough to be real evidence"},
	}, nil)
	m := findMetric(card, "grounding rate")
	if m == nil {
		t.Fatal("grounding rate is not named")
	}
	if m.Status != eval.Blocked {
		t.Errorf("status = %s with no check run; a 0%% would read as total failure", m.Status)
	}
	if m.Value != 0 {
		t.Errorf("a blocked metric carries value %v", m.Value)
	}
}

// TestDuplicateCollapseCountsRepetitionRemoved (§11.2).
func TestDuplicateCollapseCountsRepetitionRemoved(t *testing.T) {
	claims := []core.Claim{
		{Text: "One phrasing.", Source: "https://a.example/1", Quote: "a quote long enough to be real evidence"},
		{Text: "Another phrasing.", Source: "https://b.example/1", Quote: "a quote long enough to be real evidence"},
		{Text: "Unrelated.", Source: "https://c.example/1", Quote: "a quote long enough to be real evidence"},
	}
	card := scoreWith(t, claims, func(ids map[string]string) []core.ClaimEdge {
		return []core.ClaimEdge{{
			FromID: ids["One phrasing."], ToID: ids["Another phrasing."],
			Kind: core.EdgeDuplicateOf, Weight: 1,
		}}
	})
	m := findMetric(card, "duplicate collapse")
	if m == nil {
		t.Fatal("duplicate collapse missing")
	}
	// 3 claims, 1 duplicate edge -> 2 findings -> one third collapsed.
	if !strings.Contains(m.Detail, "3 claim(s) render as 2 finding(s)") {
		t.Errorf("detail = %q", m.Detail)
	}
	if m.Value <= 0 {
		t.Errorf("collapse rate = %v with a duplicate present", m.Value)
	}
}

// TestStalenessSeparationSaysWhenDatesAreMissing. The rule needs PublishedAt on both
// claims and most web pages supply none, so a zero here is usually missing dates
// rather than a broken rule — and a reader chasing the wrong one wastes their time.
//
// It used to hint at that in Detail while still reporting Measured. It now counts the
// dated claims and reports Blocked, because a rule that never ran measured nothing:
// "0 of N resolved" and "there was nothing to resolve with" are different facts.
func TestStalenessSeparationSaysWhenDatesAreMissing(t *testing.T) {
	claims := []core.Claim{
		{Text: "A.", Source: "https://a.example/1", Quote: "a quote long enough to be real evidence"},
		{Text: "B.", Source: "https://b.example/1", Quote: "a quote long enough to be real evidence"},
	}
	card := scoreWith(t, claims, func(ids map[string]string) []core.ClaimEdge {
		return []core.ClaimEdge{{
			FromID: ids["A."], ToID: ids["B."], Kind: core.EdgeContradicts, Weight: 1,
		}}
	})
	m := findMetric(card, "staleness separation")
	if m == nil {
		t.Fatal("staleness separation missing")
	}
	if m.Value != 0 {
		t.Errorf("value = %v with no supersedes edges", m.Value)
	}
	if m.Status != eval.Blocked {
		t.Errorf("status = %v; with no dated claim the rule could not run", m.Status)
	}
	if !strings.Contains(m.Reason, "publication date") {
		t.Errorf("a zero does not point at the likely cause: %q", m.Reason)
	}
}

// scoreWith builds a session with claims and optional edges, then scores it.
//
// edges receives claim text -> assigned ID, because the store assigns IDs and a test
// cannot know them in advance.
func scoreWith(t *testing.T, claims []core.Claim, edges func(ids map[string]string) []core.ClaimEdge) eval.Scorecard {
	t.Helper()
	ctx := context.Background()
	f := newFixture(t, core.MicrosPerUSD)

	if len(claims) > 0 {
		for i := range claims {
			claims[i].SessionID = f.sess.ID
			claims[i].LeadID = f.lead.ID
		}
		if err := f.db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
			return tx.InsertClaims(ctx, claims)
		}); err != nil {
			t.Fatalf("insert claims: %v", err)
		}
	}

	if edges != nil {
		ids := map[string]string{}
		for _, c := range claims {
			ids[c.Text] = c.ID
		}
		e := edges(ids)
		for i := range e {
			e[i].SessionID = f.sess.ID
		}
		if err := f.db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
			return tx.InsertEdges(ctx, e)
		}); err != nil {
			t.Fatalf("insert edges: %v", err)
		}
	}

	card, err := eval.Score(ctx, f.db, f.sess.ID, eval.Options{})
	if err != nil {
		t.Fatalf("score: %v", err)
	}
	return card
}

// TestDuplicateCollapseHandlesACliqueNotJustAPair.
//
// The adjudicator judges EVERY pair in a cluster, so a k-member duplicate cluster carries
// k(k-1)/2 edges and performs only k-1 merges. The metric counted edges, on the stated
// reasoning that "each duplicate edge merges two nodes" — true only for a forest.
// Measured on one 5-member clique plus five singletons: reported 90%, truth 40%, with a
// clamp turning the negative into the most flattering number available.
//
// The original test used a single edge between two claims — the one shape where counting
// edges and clustering agree — so the bug was invisible to it.
func TestDuplicateCollapseHandlesACliqueNotJustAPair(t *testing.T) {
	var claims []core.Claim
	for i := 0; i < 10; i++ {
		claims = append(claims, core.Claim{
			Text:   fmt.Sprintf("Claim number %d.", i),
			Source: fmt.Sprintf("https://s%02d.example/p", i),
			Quote:  "a quote long enough to be real evidence",
		})
	}
	card := scoreWith(t, claims, func(ids map[string]string) []core.ClaimEdge {
		// Claims 0-4 all duplicate each other: ten edges, four merges.
		var out []core.ClaimEdge
		for i := 0; i < 5; i++ {
			for j := i + 1; j < 5; j++ {
				out = append(out, core.ClaimEdge{
					FromID: ids[fmt.Sprintf("Claim number %d.", i)],
					ToID:   ids[fmt.Sprintf("Claim number %d.", j)],
					Kind:   core.EdgeDuplicateOf, Weight: 1,
				})
			}
		}
		return out
	})

	m := findMetric(card, "duplicate collapse")
	if m == nil {
		t.Fatal("duplicate collapse missing")
	}
	// 10 claims, one 5-member cluster plus 5 singletons = 6 findings, 40% collapsed.
	if !strings.Contains(m.Detail, "10 claim(s) render as 6 finding(s)") {
		t.Errorf("detail = %q, want 6 findings", m.Detail)
	}
	if m.Value < 39 || m.Value > 41 {
		t.Errorf("collapse = %.1f%%, want 40%%", m.Value)
	}
}

// TestAMetricValueIsInTheUnitItsLabelClaims.
//
// Budget amounts are micro-dollars end to end, so an int64 keeps a long session's
// arithmetic exact. Metric.Value is a float64 next to a Unit string, and putting
// raw micros beside Unit "usd" reported $0.004315 per claim as "4315.0 usd" — a
// real measurement wrong by a factor of a million, in the harness whose entire
// job is reporting numbers accurately. Detail was right the whole time; the
// headline number is the one that gets quoted.
func TestAMetricValueIsInTheUnitItsLabelClaims(t *testing.T) {
	for _, tc := range []struct {
		unit   core.BudgetUnit
		spent  int64
		claims int
		want   float64
	}{
		// The measured case: $0.474687 over 110 claims.
		{core.BudgetUSD, 474_687, 110, 0.004315},
		{core.BudgetUSD, 40_000, 2, 0.02},
		// Tokens are already whole units — dividing them would be the same bug
		// pointing the other way. The fixture charges a fixed 100 input + 10
		// output tokens per call whatever USD figure it is handed, so a
		// token-mode session spends 110: over 2 claims that is 55, not 0.000055.
		{core.BudgetTokens, 40_000, 2, 55},
	} {
		f := newFixtureUnit(t, tc.unit, 10_000_000)
		f.spend(t, tc.spent, tc.spent)
		claims := make([]core.Claim, 0, tc.claims)
		for i := 0; i < tc.claims; i++ {
			claims = append(claims, goodClaim(
				fmt.Sprintf("Claim number %d of the set.", i),
				fmt.Sprintf("https://s%d.example/x", i)))
		}
		f.addClaims(t, claims...)

		got := metric(t, f.score(t), "cost per claim")
		if diff := got.Value - tc.want; diff > 1e-9 || diff < -1e-9 {
			t.Errorf("%s: cost per claim = %v, want %v", tc.unit, got.Value, tc.want)
		}
		if got.Unit != string(tc.unit) {
			t.Errorf("%s: unit = %q", tc.unit, got.Unit)
		}
	}
}

// TestTheBudgetMetricIsNamedForWhatItMeasures. It reports overshoot, so 0 is the
// passing value — under the old name a perfect scorecard read "budget adherence
// 0.0 %", which scans as total failure.
func TestTheBudgetMetricIsNamedForWhatItMeasures(t *testing.T) {
	f := newFixture(t, 100_000)
	f.spend(t, 10_000, 10_000)
	card := f.score(t)

	m := metric(t, card, "budget overshoot")
	if m.Value != 0 || m.Regression {
		t.Errorf("an under-budget session reported %v%% overshoot (regression=%v)", m.Value, m.Regression)
	}
	for _, x := range card.Metrics {
		if x.Name == "budget adherence" {
			t.Error(`the metric is still published as "budget adherence"`)
		}
	}
}

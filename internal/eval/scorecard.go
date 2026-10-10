// Package eval scores a finished session.
//
// §14.3 lists nine metrics. Four of them cannot be computed yet — three need
// components that arrive in M4 and M8, one needs labelled answers — and this
// package reports that rather than omitting them. A scorecard showing five
// numbers reads as a complete picture; a scorecard showing five numbers and
// four explicit "not measured" lines reads as what it is.
//
// Everything here is mechanical: pure arithmetic over persisted state, no model
// calls, no network, no judgement. That is the whole point. These are the
// numbers that can block a merge without anyone arguing about them, and they
// work today against a session `mole research` already wrote.
package eval

import (
	"context"
	"fmt"
	"net/url"
	"sort"
	"strings"

	"github.com/lajosdeme/mole/internal/budget"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/tools/fetch"
	"github.com/lajosdeme/mole/internal/verifier"
)

// Status says whether a metric was computed, and if not, why not.
type Status string

const (
	// Measured means the number is real.
	Measured Status = "measured"
	// Blocked means the metric cannot be computed until something lands.
	// Reported explicitly so a zero is never mistaken for a regression.
	Blocked Status = "blocked"
	// NotApplicable means this session had nothing to measure.
	NotApplicable Status = "n/a"
)

// Metric is one line of the scorecard.
type Metric struct {
	Name   string `json:"name"`
	Status Status `json:"status"`

	// Value and Unit are set when Status is Measured.
	Value float64 `json:"value,omitempty"`
	Unit  string  `json:"unit,omitempty"`

	// Detail is the arithmetic behind the number, so a reader can check it.
	Detail string `json:"detail,omitempty"`
	// Reason explains a Blocked status: what is missing and which milestone
	// brings it.
	Reason string `json:"reason,omitempty"`

	// Regression marks a metric that should fail a build. Only ever set on
	// metrics that are objectively wrong, never on quality judgements.
	Regression bool `json:"regression,omitempty"`
}

// Scorecard is one session's evaluation.
type Scorecard struct {
	SessionID string   `json:"session_id"`
	Metrics   []Metric `json:"metrics"`

	// Citations is the per-claim detail behind the citation accuracy metric,
	// present only when it was computed.
	Citations *CitationReport `json:"citations,omitempty"`
}

// Failed reports whether any metric is a hard regression.
func (s Scorecard) Failed() bool {
	for _, m := range s.Metrics {
		if m.Regression {
			return true
		}
	}
	return false
}

// Options tune what Score computes.
type Options struct {
	// Citations, when set, re-reads every cited source and checks the quote is
	// actually in it (§14.3 citation accuracy).
	//
	// Off by default because it costs a fetch per source. Under cassette
	// replay it is free and deterministic, which is what §14.1 was built for.
	Citations SourceReader
}

// Score evaluates one session.
func Score(ctx context.Context, st store.Store, sessionID string, opts Options) (Scorecard, error) {
	card := Scorecard{SessionID: sessionID}

	var (
		sess     *core.Session
		claims   []*core.Claim
		calls    []*core.ToolCall
		outcomes []*store.FetchOutcome
		edges    []*core.ClaimEdge
	)
	if err := st.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		if sess, err = q.GetSession(ctx, sessionID); err != nil {
			return err
		}
		if claims, err = q.ListClaims(ctx, sessionID, 0); err != nil {
			return err
		}
		if calls, err = q.ListToolCalls(ctx, sessionID, 0); err != nil {
			return err
		}
		if outcomes, err = q.ListFetchOutcomes(ctx, sessionID, 0); err != nil {
			return err
		}
		edges, err = q.ListEdges(ctx, sessionID, 0)
		return err
	}); err != nil {
		return card, err
	}

	led := budget.New(st, budget.DefaultConfig())
	verify, err := led.Verify(ctx, sessionID)
	if err != nil {
		return card, err
	}

	card.Metrics = append(card.Metrics,
		budgetOvershoot(sess),
		ledgerConsistency(verify),
		holdsReleased(verify),
		claimIntegrity(claims),
		sourceConcentration(claims),
		costPerClaim(sess, claims),
		toolCallCount(calls),
	)
	if opts.Citations != nil {
		rep := VerifyCitations(ctx, claims, fetch.ProviderSupplied(outcomes), opts.Citations)
		card.Citations = &rep
		card.Metrics = append(card.Metrics, citationAccuracy(rep), citationOffsetDrift(rep))
	}

	// Graph metrics (§11). Blocked until M4 built the Verifier; computable now, and
	// separated deliberately from the ones that need labelled data. A count of what
	// the graph found is not recall — recall needs to know what it MISSED — and
	// reporting one as the other would be the most flattering possible confusion.
	card.Metrics = append(card.Metrics,
		verificationCoverage(claims),
		duplicateCollapse(claims, edges),
		disagreementRate(claims, edges),
		stalenessSeparation(claims, edges),
	)
	if g := groundingRate(claims); g != nil {
		card.Metrics = append(card.Metrics, *g)
	} else {
		// Named rather than omitted. §11.5.2's judge exists, so this is no longer
		// blocked on a component — it is blocked on nobody having run a check, which
		// is a different thing and a reader should be able to tell.
		card.Metrics = append(card.Metrics, Metric{
			Name: "grounding rate", Status: Blocked,
			Reason: "no claim was re-read: §11.5.2's grounding pass needs escrow left " +
				"over and a fetcher configured",
		})
	}

	// Dataset mode's own numbers (§14.3), and only for a session that has them: a
	// report session has no rows, and a metric reading 0% would be
	// indistinguishable from a dataset session whose extraction failed entirely.
	card.Metrics = append(card.Metrics, datasetMetrics(ctx, st, sessionID)...)

	// §12.1's audit trail, scored (§14.3's exfil number). Blocked for a session
	// that used no local data, and measured for one that did.
	card.Metrics = append(card.Metrics, crossingMetrics(ctx, st, sessionID)...)

	card.Metrics = append(card.Metrics, blockedMetrics(opts)...)
	return card, nil
}

// ---------------------------------------------------------------------------
// Mechanical metrics
// ---------------------------------------------------------------------------

// budgetOvershoot is §14.3's "max observed overshoot past ceiling. Must be ~0."
//
// The headline correctness property of the whole system. Everything else here
// is diagnostic; this one is a contract.
//
// Named for what it measures. It reported under "budget adherence", where the
// passing value is 0 and a scorecard's best possible row read "budget adherence
// 0.0 %" — which scans as total failure. The metric never changed meaning; only
// a reader's reading of it did, and the test covering it has always been called
// TestBudgetOvershootIsARegression.
func budgetOvershoot(s *core.Session) Metric {
	m := Metric{Name: "budget overshoot", Status: Measured, Unit: "%"}

	over := s.Spent - s.Budget
	if over <= 0 {
		m.Value = 0
		m.Detail = fmt.Sprintf("spent %s of %s, no overshoot",
			core.FormatAmount(s.Spent, s.BudgetUnit), core.FormatAmount(s.Budget, s.BudgetUnit))
		return m
	}

	m.Value = 100 * float64(over) / float64(s.Budget)
	m.Detail = fmt.Sprintf("spent %s against a %s ceiling — over by %s",
		core.FormatAmount(s.Spent, s.BudgetUnit),
		core.FormatAmount(s.Budget, s.BudgetUnit),
		core.FormatAmount(over, s.BudgetUnit))
	// Any overshoot at all is a regression. The reserve/settle design exists to
	// make this impossible, so a non-zero value means the design is not holding
	// rather than that a limit was slightly loose.
	m.Regression = true
	return m
}

// ledgerConsistency reconciles the materialized counters against the
// append-only rows. A divergence means the ceiling is being enforced against a
// number nothing checks.
func ledgerConsistency(v budget.VerifyResult) Metric {
	m := Metric{Name: "ledger consistency", Status: Measured, Unit: "ok"}
	if v.Consistent() {
		m.Value = 1
		m.Detail = fmt.Sprintf("%d call(s) reconcile", v.CallsFromRows)
		return m
	}
	m.Value = 0
	m.Regression = true
	m.Detail = fmt.Sprintf("spent recorded=%d ledger=%d · held recorded=%d rows=%d · calls recorded=%d rows=%d",
		v.SpentRecorded, v.SpentFromLedger, v.HeldRecorded, v.HeldFromRows, v.CallsRecorded, v.CallsFromRows)
	return m
}

// holdsReleased catches budget stranded in a reservation nobody settled. It is
// invisible in a spend total — the money is neither spent nor available — and
// it is how a long session runs out of a budget it never used.
func holdsReleased(v budget.VerifyResult) Metric {
	m := Metric{Name: "holds released", Status: Measured, Unit: "held"}
	m.Value = float64(v.HeldFromRows)
	if v.HeldFromRows == 0 {
		m.Detail = "no reservations left held"
		return m
	}
	m.Regression = true
	m.Detail = fmt.Sprintf("%d still held after the session finished", v.HeldFromRows)
	return m
}

// claimIntegrity checks every claim against the invariants §11.5 promises: a
// quote long enough to be evidence, an offset that could locate it, and a
// source that resolves.
//
// This is NOT citation accuracy — that requires the source text, which the
// actor discards by design (§2). It is the subset checkable from the row alone,
// and a failure here means a claim was written that the pipeline should have
// rejected.
//
// RetrievedAt is deliberately not checked: InsertClaims defaults it, so a
// persisted claim cannot have a zero one. A check that can never fire implies
// coverage this does not have.

// resolvableSource reports whether a claim's source identifies where it came
// from.
//
// Two shapes, because §4 defines two. A web or academic claim cites a URL. A
// LOCAL claim cites "connector:<name>#<query hash>" — the data never left the
// machine, so there is no URL, and the hash is what makes the claim traceable
// back to the statement that produced it.
//
// Checking url.Parse().Host alone counted every local claim as malformed, which
// is a false regression in the one number that exists to catch real ones.
func resolvableSource(src string) bool {
	if rest, ok := strings.CutPrefix(src, connectorScheme); ok {
		name, hash, found := strings.Cut(rest, "#")
		return found && name != "" && hash != ""
	}
	u, err := url.Parse(src)
	return err == nil && u.Host != ""
}

// connectorScheme prefixes a local claim's citation (§4, §12.1).
const connectorScheme = "connector:"

func claimIntegrity(claims []*core.Claim) Metric {
	m := Metric{Name: "claim integrity", Status: Measured, Unit: "%"}
	if len(claims) == 0 {
		m.Status = NotApplicable
		m.Detail = "no claims"
		return m
	}

	var problems []string
	bad := 0
	for _, c := range claims {
		var why []string
		if strings.TrimSpace(c.Text) == "" {
			why = append(why, "empty text")
		}
		if len(strings.TrimSpace(c.Quote)) < 24 {
			why = append(why, "quote too short to be evidence")
		}
		if c.QuoteOffset < 0 {
			why = append(why, "negative quote offset")
		}
		if !resolvableSource(c.Source) {
			why = append(why, "unresolvable source")
		}
		// Both are 0-1. Confidence is 0 on an unverified claim, which is correct
		// rather than malformed, so only the range is checked here.
		if c.Confidence < 0 || c.Confidence > 1 {
			why = append(why, "derived confidence outside 0-1")
		}
		if c.AssertionStrength < 0 || c.AssertionStrength > 1 {
			why = append(why, "assertion strength outside 0-1")
		}
		if len(why) > 0 {
			bad++
			if len(problems) < 3 {
				problems = append(problems, fmt.Sprintf("%.40q: %s", c.Text, strings.Join(why, ", ")))
			}
		}
	}

	m.Value = 100 * float64(len(claims)-bad) / float64(len(claims))
	m.Detail = fmt.Sprintf("%d of %d claims well-formed", len(claims)-bad, len(claims))
	if bad > 0 {
		m.Regression = true
		m.Detail += " — " + strings.Join(problems, "; ")
	}
	return m
}

// sourceConcentration reports how much of the answer came from one page.
//
// Diagnostic, not a pass/fail: a question with one good source is legitimate.
// But a run that reads five sources and cites one is usually a run where four
// of them failed quietly, and that is invisible in a claim count.
func sourceConcentration(claims []*core.Claim) Metric {
	m := Metric{Name: "source concentration", Status: Measured, Unit: "%"}
	if len(claims) == 0 {
		m.Status = NotApplicable
		m.Detail = "no claims"
		return m
	}

	bySource := map[string]int{}
	for _, c := range claims {
		bySource[c.Source]++
	}
	top, topN := "", 0
	for src, n := range bySource {
		if n > topN || (n == topN && src < top) {
			top, topN = src, n
		}
	}

	m.Value = 100 * float64(topN) / float64(len(claims))
	m.Detail = fmt.Sprintf("%d distinct source(s); the largest contributed %d of %d claims (%s)",
		len(bySource), topN, len(claims), top)
	return m
}

// costPerClaim is the denominator half of §14.3's headline metric.
//
// Explicitly NOT "cost per correct claim": correctness needs labels this
// package does not have. Reported under its own name so the two are never
// confused — optimizing cost per claim alone rewards a model that emits more
// claims, which is the opposite of the intent.
// displayAmount converts a stored amount into the unit its label claims.
//
// Budget amounts are micro-dollars end to end — an int64 so a long session's
// arithmetic is exact rather than float drift. Metric.Value is a float64 carrying
// a Unit string, and putting the raw micros next to Unit "usd" reported
// $0.004315 per claim as "4315.0 usd": a real measurement off by a factor of a
// million, in the harness whose only job is reporting numbers accurately.
//
// Detail was correct throughout, because it goes through core.FormatAmount. The
// headline number is what gets quoted.
func displayAmount(amount int64, unit core.BudgetUnit) float64 {
	if unit == core.BudgetUSD {
		return float64(amount) / 1e6
	}
	return float64(amount)
}

func costPerClaim(s *core.Session, claims []*core.Claim) Metric {
	m := Metric{Name: "cost per claim", Status: Measured, Unit: string(s.BudgetUnit)}
	if len(claims) == 0 {
		m.Status = NotApplicable
		m.Detail = fmt.Sprintf("no claims; %s spent", core.FormatAmount(s.Spent, s.BudgetUnit))
		return m
	}
	per := s.Spent / int64(len(claims))
	m.Value = displayAmount(per, s.BudgetUnit)
	m.Detail = fmt.Sprintf("%s over %d claims = %s each",
		core.FormatAmount(s.Spent, s.BudgetUnit), len(claims),
		core.FormatAmount(per, s.BudgetUnit))
	return m
}

// toolCallCount is context for the numbers above: cost per claim means
// something different at three calls than at thirty.
func toolCallCount(calls []*core.ToolCall) Metric {
	m := Metric{Name: "tool calls", Status: Measured, Unit: "calls", Value: float64(len(calls))}

	byType := map[core.CallType]int{}
	failed := 0
	for _, c := range calls {
		byType[c.Type]++
		if c.Err != "" {
			failed++
		}
	}
	kinds := make([]string, 0, len(byType))
	for t, n := range byType {
		kinds = append(kinds, fmt.Sprintf("%s %d", t, n))
	}
	sort.Strings(kinds)

	m.Detail = strings.Join(kinds, " · ")
	if failed > 0 {
		m.Detail += fmt.Sprintf(" · %d errored", failed)
	}
	return m
}

// ---------------------------------------------------------------------------
// What cannot be measured yet
// ---------------------------------------------------------------------------

// blockedMetrics names §14.3's remaining entries and why each is unavailable.
//
// Listing them is the point: a scorecard that silently omitted them would let a
// reader conclude the pipeline is fully scored, then read a genuine future
// regression as normal.
//
// The exfil entry has left this list — twice, because it was here twice, once with
// the reason "needs the aggregation gate (M8)" long after M8 landed. It is
// measured from the crossings table now (see crossings.go), which is the shape
// every entry here should eventually take: a duplicate blocked entry is what a
// list of excuses decays into.
func blockedMetrics(opts Options) []Metric {
	metrics := []Metric{
		{
			Name: "claim precision", Status: Blocked,
			Reason: "needs labelled answers; the question corpus (§14.2) is not built yet",
		},
		{
			Name: "contradiction recall", Status: Blocked,
			// M4 built the Verifier, so contradictions are now FOUND and counted as
			// "disagreement rate". Recall is a different number: it needs to know
			// what the graph missed, which needs disagreements planted on purpose.
			Reason: "the Verifier now finds contradictions (see disagreement rate), but " +
				"recall needs planted disagreements to measure what it MISSED — the " +
				"question corpus (§14.2)",
		},
		{
			Name: "staleness detection", Status: Blocked,
			// Same shape. supersedes edges exist and are counted as "staleness
			// separation"; detection needs to know how many stale claims were there.
			Reason: "supersedes edges now exist (see staleness separation), but detection " +
				"needs a corpus with known-stale sources to measure what was missed (§14.2)",
		},
	}

	if opts.Citations == nil {
		// Reported as blocked rather than omitted: it IS computable now, and a
		// reader should know a run declined to compute it rather than assume
		// the metric does not exist yet.
		metrics = append(metrics, Metric{
			Name: "citation accuracy", Status: Blocked,
			Reason: "not requested; re-reads every cited source, so pass --citations " +
				"(free and deterministic under MOLE_RECORD=replay)",
		})
	}
	return metrics
}

// ---------------------------------------------------------------------------
// Graph metrics (§11)
// ---------------------------------------------------------------------------
//
// These became computable when M4 built the Verifier. What is deliberately NOT
// here is anything called recall: recall needs to know what the graph MISSED, which
// needs planted disagreements in §14.2's corpus. A count of what was found, reported
// as recall, would be the most flattering possible confusion — it rises when the
// pipeline gets noisier, not when it gets better.

// verificationCoverage is how much of the claim set the Verifier actually looked at.
//
// The denominator for every other graph number. Zero disagreements on a session
// where nothing was verified is not a clean bill of health, and a reader seeing only
// the disagreement count cannot tell those apart.
func verificationCoverage(claims []*core.Claim) Metric {
	m := Metric{Name: "verification coverage", Status: Measured, Unit: "%"}
	if len(claims) == 0 {
		m.Status = NotApplicable
		m.Detail = "no claims"
		return m
	}
	verified := 0
	for _, c := range claims {
		if c.VerifiedAt != nil {
			verified++
		}
	}
	m.Value = 100 * float64(verified) / float64(len(claims))
	m.Detail = fmt.Sprintf("%d of %d claim(s) verified", verified, len(claims))
	if verified == 0 {
		m.Detail += " — every graph number below is measured over nothing"
	}
	return m
}

// duplicateCollapse is how much repetition the graph removed (§11.2).
//
// The live run this was built against produced two near-identical claims from one
// source and rendered both as separate findings. The number matters in both
// directions: 0% on a session with obvious repeats means clustering is not working,
// and a very high figure means the extractor is emitting the same claim over and
// over.
func duplicateCollapse(claims []*core.Claim, edges []*core.ClaimEdge) Metric {
	m := Metric{Name: "duplicate collapse", Status: Measured, Unit: "%"}
	if len(claims) == 0 {
		m.Status = NotApplicable
		m.Detail = "no claims"
		return m
	}

	// Ask the clustering, do not count edges.
	//
	// The arithmetic here used to be `findings = claims - duplicateEdges`, on the stated
	// reasoning that "each duplicate edge merges two nodes". That holds only for a
	// forest, and the adjudicator judges EVERY pair in a cluster — so a k-member cluster
	// carries k(k-1)/2 edges and performs k-1 merges. Measured on one 5-member cluster
	// plus five singletons: reported 90%, truth 40%, and a `< 1` clamp turned the
	// negative into the most flattering number available.
	//
	// verifier.Clusters already computes the answer and output.Findings already uses it.
	findings := len(verifier.Clusters(claims, edges))
	if findings == 0 {
		m.Status = NotApplicable
		m.Detail = "no findings"
		return m
	}
	m.Value = 100 * float64(len(claims)-findings) / float64(len(claims))
	m.Detail = fmt.Sprintf("%d claim(s) render as %d finding(s)", len(claims), findings)
	return m
}

// disagreementRate is how often the graph found two sources incompatible.
//
// Named a rate, not recall, and the distinction is the point: this counts what was
// found and says nothing about what was missed. It is diagnostic in both directions —
// a session on a contested question reporting zero is suspicious, and one reporting
// half its claims as disputed probably has an adjudicator saying "contradicts" to
// anything it cannot parse.
func disagreementRate(claims []*core.Claim, edges []*core.ClaimEdge) Metric {
	m := Metric{Name: "disagreement rate", Status: Measured, Unit: "%"}
	if len(claims) == 0 {
		m.Status = NotApplicable
		m.Detail = "no claims"
		return m
	}
	contradicts := 0
	involved := map[string]bool{}
	for _, e := range edges {
		if e == nil || e.Kind != core.EdgeContradicts {
			continue
		}
		contradicts++
		involved[e.FromID] = true
		involved[e.ToID] = true
	}
	m.Value = 100 * float64(len(involved)) / float64(len(claims))
	m.Detail = fmt.Sprintf("%d contradiction(s) involving %d of %d claim(s)",
		contradicts, len(involved), len(claims))
	return m
}

// stalenessSeparation is how many conflicts the dates resolved rather than the model
// (§11.2).
//
// A 2019 claim contradicted by a 2025 one is usually staleness, not disagreement, and
// this is what says whether that rule is doing anything. Not "staleness detection" —
// that would claim to know how many stale claims exist, which needs the corpus.
func stalenessSeparation(claims []*core.Claim, edges []*core.ClaimEdge) Metric {
	m := Metric{Name: "staleness separation", Status: Measured, Unit: "edges"}
	dated := 0
	for _, c := range claims {
		if c != nil && c.PublishedAt != nil {
			dated++
		}
	}
	supersedes, contradicts := 0, 0
	for _, e := range edges {
		if e == nil {
			continue
		}
		switch e.Kind {
		case core.EdgeSupersedes:
			supersedes++
		case core.EdgeContradicts:
			contradicts++
		}
	}
	m.Value = float64(supersedes)
	total := supersedes + contradicts
	if total == 0 {
		m.Status = NotApplicable
		m.Detail = "no conflicting claims to separate"
		return m
	}
	m.Detail = fmt.Sprintf("%d of %d conflict(s) resolved as staleness by publication date",
		supersedes, total)
	if dated == 0 {
		// The rule needs PublishedAt on both claims, so with none it did not run.
		// "0 of N resolved" would imply it ran and found nothing — a different fact,
		// and the one that hid §11.2 being inert across a whole corpus. Computed
		// rather than left as advice to go and check.
		m.Status = Blocked
		m.Reason = fmt.Sprintf("no claim carried a publication date (0 of %d), so the "+
			"rule could not run: every conflict was written as a live disagreement. "+
			"A provider supplying no dates and enough page text to skip fetching "+
			"leaves neither source of one — see mole doctor", len(claims))
		return m
	}
	if supersedes == 0 {
		m.Detail += fmt.Sprintf(" — %d of %d claim(s) carried a date", dated, len(claims))
	}
	return m
}

// groundingRate is §14.3's metric, unblocked by §11.5.2's judge.
//
// Nil when no claim was ever checked, so it is omitted rather than reported as 0% —
// which would read as "every checked claim failed" instead of "nothing was checked".
//
// The denominator is DECISIVE checks only. A vanished quote, an unreachable host and a
// judge that would not answer are all recorded, and none of them says anything about
// whether a quote supports its claim; counting them as failures would make the metric
// track network weather.
func groundingRate(claims []*core.Claim) *Metric {
	decisive, confirmed, inconclusive := 0, 0, 0
	for _, c := range claims {
		switch {
		case c.Grounded != nil:
			decisive++
			if *c.Grounded {
				confirmed++
			}
		case c.GroundingNote != "":
			inconclusive++
		}
	}
	if decisive == 0 && inconclusive == 0 {
		return nil
	}

	m := Metric{Name: "grounding rate", Status: Measured, Unit: "%"}
	if decisive == 0 {
		m.Status = NotApplicable
		m.Detail = fmt.Sprintf("%d check(s) ran, none decisive (page changed, source "+
			"unreachable, or the judge declined)", inconclusive)
		return &m
	}
	m.Value = 100 * float64(confirmed) / float64(decisive)
	m.Detail = fmt.Sprintf("%d of %d re-read claim(s) confirmed by their own source",
		confirmed, decisive)
	if inconclusive > 0 {
		m.Detail += fmt.Sprintf(" (%d inconclusive, excluded)", inconclusive)
	}
	if confirmed < decisive {
		// Not a Regression flag: one claim whose source does not support it is a
		// finding about that claim, not proof the pipeline is broken. But it is the
		// most important line on the card when it happens.
		m.Detail += fmt.Sprintf(" — %d claim(s) NOT supported by their own source",
			decisive-confirmed)
	}
	return &m
}

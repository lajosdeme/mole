package verifier

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"runtime/debug"
	"slices"
	"sync"
	"time"

	"github.com/lajosdeme/mole/internal/budget"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/pricing"
	"github.com/lajosdeme/mole/internal/store"
)

// Verifier builds the claim graph for a session (§11).
//
// One pass does: read the claims nobody has scored, retrieve candidate pairs
// against the session's whole claim set, settle what can be settled without a
// model, adjudicate the rest in batches, and write the edges.
//
// Then confidence is derived from the resulting graph (§11.3) — over the WHOLE
// graph, not just this pass's edges, because a new claim corroborating an existing
// one changes the existing claim's standing too.
type Verifier struct {
	Store  store.Store
	Ledger *budget.Ledger
	LLM    llm.Provider

	// Pricing prices the adjudication and grounding reservations against this
	// Verifier's own model and the session's real rate table, replacing a flat
	// 2 micro-dollars per 1k tokens that measured 2-6 micro-dollars of reserved
	// hold against calls the ledger actually charged ~146 micro-dollars for.
	// Nil — or a model the table has never heard of — reserves against
	// fallbackRates instead.
	Pricing *pricing.Table

	// Model overrides the tier's default for every call this Verifier makes. Empty
	// leaves the cheap-tier model in place.
	//
	// Separate from the tier because the workloads are not alike. Chunk mining is
	// extraction and runs per chunk; adjudication decides whether two sentences can both
	// be true and runs per batch — 31 calls against 7 on a live 25-claim run. A model too
	// expensive to mine with can be affordable to judge with, and judging is the stage
	// whose errors corrupt confidence, the report's disagreements and three eval metrics.
	Model string

	// Retriever selects candidate pairs. Nil takes LexicalRetriever.
	Retriever Retriever

	// MaxCandidatesPerClaim bounds fan-out per claim. Zero takes the default.
	MaxCandidatesPerClaim int

	// BatchSize is how many pairs go into one adjudication call. Zero takes the
	// default.
	BatchSize int

	// Concurrency is how many adjudication calls may be in flight at once. Zero
	// takes DefaultConcurrency; 1 is the serial pass. See runBatches.
	Concurrency int

	// Confirm chooses which edge kinds need a second agreeing judgement before the
	// edge is written. The zero value confirms nothing — callers opt in, as they did
	// when this was a bool. See ConfirmMode for the measurement.
	Confirm ConfirmMode

	// StalenessGap is how far apart two publication dates must be for a
	// contradiction to read as staleness (§11.2). Zero takes the default.
	StalenessGap time.Duration

	// MaxShareOfBudget caps cumulative verification spend as a fraction of the
	// session's whole budget. Zero takes DefaultMaxShareOfBudget.
	//
	// A ceiling is not optional here. Pair count grows with the square of the
	// claim count before the per-claim cap bites, so a session that gathers a lot
	// of claims can spend more verifying them than it spent finding them. On the
	// real 13-claim run, one call per pair would have cost about a third of the
	// whole session.
	MaxShareOfBudget float64

	// MaxVerifyDepth and MaxFollowUpsPerRoot are §11.4's caps. Zero takes the
	// defaults.
	MaxVerifyDepth      int
	MaxFollowUpsPerRoot int
	// MaxFollowUpsPerPass bounds how much work one pass may queue. Zero takes
	// DefaultMaxFollowUpsPerPass.
	MaxFollowUpsPerPass int

	// Grounder re-reads sources for §11.5.2. Nil disables grounding entirely,
	// which is supported: claims keep the verbatim quote checked at extraction
	// time, and Grounded stays nil rather than being guessed.
	Grounder *Grounder
	// MaxGroundChecks bounds how many claims one grounding pass re-reads. Zero
	// takes DefaultMaxGroundChecks.
	MaxGroundChecks int

	Log *slog.Logger

	// spent accumulates this Verifier's charges within one process. Reconciled against
	// the ledger before each wave of batches — see remainingAllowance — because an
	// in-memory counter restarts at zero for a second Verifier on the same session, and
	// the cap is supposed to bind on the SESSION.
	spent int64

	// mu guards spent, and the Result counters adjudicate writes (Spent, Calls),
	// because a wave's batches settle from their own goroutines.
	mu sync.Mutex

	// warnUnpriced logs the "no rate for this model" fallback once per Verifier
	// rather than once per reservation — a batch pass takes one per batch and a
	// grounding pass one per claim, and either would otherwise fill the log with
	// the same sentence for the whole run.
	warnUnpriced sync.Once
}

// DefaultMaxShareOfBudget is the cumulative verification allowance.
//
// 0.25 is a guess and flagged as one, in the same category as §9.1's fan-out
// taper: nothing here can yet say what fraction of a session SHOULD go to
// verification, because that depends on how often the graph changes a conclusion —
// which is what §14.2's corpus measures. Set generously on purpose; a ceiling that
// binds before the graph is useful would make the Verifier look worthless for
// reasons of arithmetic.
const DefaultMaxShareOfBudget = 0.25

// DefaultConcurrency is how many adjudication batches run at once.
//
// Measured: one Standard run spent its last ~5 of 12 minutes in 291 serial
// verifier calls, each an independent model call waiting on the one before.
//
// Four is a starting point, not a measured optimum: enough to cut that wait
// roughly fourfold, small enough that a wave's burst — and the reservations it
// holds before any of them settles — stays modest. Adjudication does not retry, so
// a provider that rate-limits the burst costs the limited batches their verdicts;
// that, not throughput, is what would argue for a different number.
const DefaultConcurrency = 4

// Result reports what one pass did.
type Result struct {
	ClaimsVerified int
	PairsRetrieved int
	// PairsDecidedFree were settled by a mechanical rule, with no model call.
	PairsDecidedFree int
	PairsJudged      int
	// PairsConfirmed and PairsUnconfirmed count the second judgement's outcome for
	// the edge kinds Confirm names. Unconfirmed pairs kept their claims and lost
	// their edge.
	PairsConfirmed   int
	PairsUnconfirmed int

	// PairsUnjudged is pairs a batch did not return a verdict for, plus pairs
	// skipped because the allowance ran out.
	PairsUnjudged int
	EdgesWritten  int
	Calls         int
	Spent         int64

	// ClaimsScored is how many claims had confidence recomputed. Larger than
	// ClaimsVerified whenever a new claim changed an older claim's standing.
	ClaimsScored int
	// Scores is the per-cluster derivation, for a trace. §11.3 requires the number
	// be explainable, and a bare 0.62 is exactly the figure people either trust
	// blindly or dismiss.
	Scores []Score

	// Contradictions is how many live disagreements this pass found. The number the
	// planner most needs: a sub-question whose evidence is disputed is not answered.
	Contradictions int
	// FollowUps are leads the Verifier wants run to settle them (§11.4). The caller
	// queues them — the Verifier does not own the lead queue, and one that did
	// could not be replayed against a cassette.
	FollowUps []FollowUp
	// Deferred are follow-ups this pass judged worth running and its budget
	// could not take. Queued work the session owes, not a wish list: every §11.4
	// rule has already been applied to them, and the only thing between them and
	// FollowUps is MaxTotal.
	//
	// They used to be dropped here, silently, and no later pass could find them
	// again — Run judges only unverified claims and skips pairs that already
	// carry an edge.
	Deferred []FollowUp

	// priorEdges is the graph as it stood before this pass, kept so confidence is
	// derived over the whole of it rather than only what this pass added.
	priorEdges []*core.ClaimEdge
	// writtenEdges is what this pass added, so Run can count live disagreements from
	// the edges rather than from the verdicts that produced them.
	writtenEdges []core.ClaimEdge

	// Degraded says what the pass could not finish (§9.5). Empty when complete.
	Degraded string
}

// ErrNoProvider means the Verifier was built without an LLM.
var ErrNoProvider = errors.New("verifier: no model provider configured")

func (v *Verifier) logger() *slog.Logger {
	if v.Log != nil {
		return v.Log
	}
	return slog.Default()
}

func (v *Verifier) retriever() Retriever {
	if v.Retriever != nil {
		return v.Retriever
	}
	return LexicalRetriever{}
}

// Run performs one verification pass over a session.
//
// Never returns an error for running out of allowance or for a model that answered
// badly — both are Degraded. An error means the store or the ledger failed, which
// is the caller's problem rather than a research outcome.
func (v *Verifier) Run(ctx context.Context, sessionID string) (*Result, error) {
	res := &Result{}
	if v.LLM == nil {
		return res, ErrNoProvider
	}

	var (
		targets []*core.Claim
		pool    []*core.Claim
		edges   []*core.ClaimEdge
	)
	if err := v.Store.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		if targets, err = q.ListUnverifiedClaims(ctx, sessionID, 0); err != nil {
			return err
		}
		if pool, err = q.ListClaims(ctx, sessionID, 0); err != nil {
			return err
		}
		edges, err = q.ListEdges(ctx, sessionID, 0)
		return err
	}); err != nil {
		return res, err
	}
	if len(targets) == 0 {
		return res, nil
	}

	// Pairs that already carry an edge. Only reachable when a previous pass was
	// interrupted: in the normal course a pair is generated once, when the later of
	// its two claims is verified.
	res.priorEdges = edges
	already := make(map[string]bool, len(edges))
	for _, e := range edges {
		already[newPairKey(e.FromID, e.ToID)] = true
	}

	pairs, free, err := CandidatePairs(ctx, v.retriever(), targets, pool,
		v.MaxCandidatesPerClaim, func(k string) bool { return already[k] })
	if err != nil {
		return res, err
	}
	res.PairsRetrieved = len(pairs) + len(free)
	res.PairsDecidedFree = len(free)

	verdicts := free
	var unjudged []Pair
	// unanswered is the subset of unjudged whose batch never got a response at all —
	// never launched, or its call errored. Distinct from a batch that answered and
	// produced nothing usable (a refusal, a parse failure, a short response): that
	// one WAS asked, so re-asking would pay again for the same non-answer. See the
	// skip computation below.
	var unanswered []Pair
	batches := Batches(pairs, v.BatchSize)
	// One wave's worth of batches at a time, so each wave's positives are
	// confirmed before the next wave spends the allowance — the order the serial
	// loop confirmed in, batch by batch, when Concurrency is 1.
	for start := 0; start < len(batches); {
		end := min(start+v.concurrency(), len(batches))
		groups, missed, missedUnanswered, _, stop := v.runBatchGroups(ctx, sessionID, batches[start:end], res)
		var judged []Judged
		// A fatal error fails a second look exactly as it failed the first, and the
		// wave it came from has already spent its round trips; asking again only
		// adds calls after the stop. The wave's first verdicts are kept unconfirmed.
		if stop == nil || !isFatal(stop) {
			judged = v.confirm(ctx, sessionID, groups, res)
		} else {
			judged = slices.Concat(groups...)
		}
		verdicts = append(verdicts, judged...)
		unjudged = append(unjudged, missed...)
		unanswered = append(unanswered, missedUnanswered...)

		if stop != nil {
			// Out of allowance, out of budget, or a fatal provider error. Keep what
			// was judged: an incomplete graph is worth more than none, and the
			// claims still need marking verified or the queue never drains.
			res.Degraded = stop.Error()
			for _, b := range batches[end:] {
				// Never launched: no response, by definition.
				unjudged = append(unjudged, b...)
				unanswered = append(unanswered, b...)
			}
			break
		}
		start = end
	}
	// Built from the pairs themselves rather than from len(pairs)-consumed plus each
	// batch's leftovers, which double-counted the batch that triggered the stop.
	res.PairsUnjudged += len(unjudged)
	res.PairsJudged = len(verdicts) - len(free)

	// A pass that could not adjudicate ANYTHING must not mark its claims verified.
	//
	// Marking is what drains the work queue, and doing it after a pass that wrote no
	// edges records "we compared these and found nothing" when the truth is "we never
	// compared them". Measured on a live run: the session hit max_leads, every
	// reservation was refused, and 7 claims were marked verified against an empty graph.
	//
	// Distinct from a pass whose model answered uselessly — that one HAS asked, and
	// re-asking would pay again for the same answer.
	blocked := res.Degraded != "" && res.Calls == 0 && res.PairsJudged == 0 &&
		len(pairs) > 0 && len(free) == 0
	if blocked {
		res.Degraded += " (claims left unverified; nothing was compared)"
		v.logger().WarnContext(ctx, "verifier: no batch could be adjudicated; leaving claims unverified",
			"pairs", len(pairs), "reason", res.Degraded)
		return res, nil
	}

	// A target whose pair was never ANSWERED is excluded from marking: the pair
	// carries no edge, so CandidatePairs retrieves it again next pass (it skips
	// only a pair that already has one) — but only as long as the target still
	// reads as unverified. Marking it anyway, as every pass used to, buried the
	// pair for good: a batch never launched (allowance, wall clock, or a fatal
	// provider error stopped the pass before it) and a batch whose call errored
	// (no response at all) both mean "we never compared this", not "we compared
	// it and found nothing".
	//
	// A pair whose batch DID get a response — a refusal, a parse failure, a short
	// batch that omitted it — is judged anyway: the model was asked, and re-asking
	// would pay again for the same non-answer. Those pairs stay in `unjudged` for
	// PairsUnjudged, but not in `unanswered`, so they are not in skip.
	//
	// Restricted to targets: the pair's other claim can be an already-verified pool
	// claim, and nothing here withholds ITS confidence — no new edge touches it
	// either, so there is nothing to withhold.
	isTarget := make(map[string]bool, len(targets))
	for _, c := range targets {
		isTarget[c.ID] = true
	}
	skip := make(map[string]bool, len(unanswered)*2)
	for _, p := range unanswered {
		if isTarget[p.A.ID] {
			skip[p.A.ID] = true
		}
		if isTarget[p.B.ID] {
			skip[p.B.ID] = true
		}
	}

	if err := v.persist(ctx, sessionID, pool, targets, verdicts, res, skip); err != nil {
		return res, err
	}

	// Counted from the EDGES the pass wrote, not from the raw verdicts.
	//
	// By this point Edges has converted date-separated contradictions into supersedes
	// (§11.2), and FollowUps skips them because "a superseded pair is not a live
	// disagreement". Counting verdicts disagreed with both: the planner was told a
	// sub-question's evidence was disputed when the publication dates had settled it,
	// and the CLI printed "1 contradiction" for a session whose report showed none.
	for _, e := range res.writtenEdges {
		if e.Kind == core.EdgeContradicts {
			res.Contradictions++
		}
	}

	// Follow-ups are proposed, not queued. The Verifier does not own the lead
	// queue, and one that wrote to it could not be replayed against a cassette or
	// dry-run by the caller.
	existing, err := v.followUpsPerRoot(ctx, sessionID)
	if err != nil {
		// Not fatal, but it must not silently become "no follow-ups exist" — that
		// would make the per-root cap count this pass alone and reset every time.
		v.logger().WarnContext(ctx, "verifier: could not count existing follow-ups; "+
			"skipping lead generation rather than uncapping it", "err", err)
		return res, nil
	}
	res.FollowUps, res.Deferred = FollowUpsWithDeferred(verdicts, FollowUpOptions{
		SessionID:       sessionID,
		StalenessGap:    v.StalenessGap,
		ActorType:       followUpActor(),
		MaxDepth:        v.MaxVerifyDepth,
		MaxPerRoot:      v.MaxFollowUpsPerRoot,
		MaxTotal:        v.maxFollowUpsPerPass(),
		ExistingPerRoot: existing,
	})
	return res, nil
}

// DefaultMaxFollowUpsPerPass bounds how many leads one pass QUEUES. What does
// not fit is deferred rather than dropped, so this paces work, it does not cap it.
const DefaultMaxFollowUpsPerPass = 2

func (v *Verifier) maxFollowUpsPerPass() int {
	if v.MaxFollowUpsPerPass > 0 {
		return v.MaxFollowUpsPerPass
	}
	return DefaultMaxFollowUpsPerPass
}

// followUpsPerRoot counts the follow-up leads each root claim already has, so
// §11.4's per-root cap counts the session rather than one pass.
func (v *Verifier) followUpsPerRoot(ctx context.Context, sessionID string) (map[string]int, error) {
	var leads []*core.Lead
	if err := v.Store.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		leads, err = q.ListLeads(ctx, sessionID, 0)
		return err
	}); err != nil {
		return nil, err
	}
	out := map[string]int{}
	for _, l := range leads {
		if l.RootClaimID != nil {
			out[*l.RootClaimID]++
		}
	}
	return out, nil
}

// followUpActor is which actor settles a disagreement.
//
// One choice today because WebActor is the only one built. When AcademicActor lands
// in M6 this should read the disputed claims' sources — a contradiction between two
// journals is settled by looking at journals — but taking a parameter it ignores
// would only look like it already did.
func followUpActor() core.ActorType { return core.ActorWeb }

// ConfirmMode names which edge kinds need a second agreeing judgement. The zero
// value confirms nothing, which is what the bool it replaced defaulted to — a mode
// whose zero value confirmed would have switched the pass on for every caller that
// never asked for it. session.Runner opts in, and its own zero value still does.
//
// Per edge kind because the two measured oppositely on 116 labelled positives:
//
//	contradicts   single 51% precision / 98% recall  ->  confirmed 70% / 67%
//	duplicate_of  single 64% precision / 88% recall  ->  confirmed 88% / 88%
//
// So confirming duplicates is free, and confirming contradictions costs 17 of 55
// true ones. Worth it where a false edge also buys a follow-up lead; not worth it
// where a person adjudicates the pair and a miss is the silent failure.
type ConfirmMode string

const (
	// ConfirmBoth is what session.Runner defaults to: right for an autonomous run.
	ConfirmBoth ConfirmMode = "both"
	// ConfirmDuplicates keeps every contradiction found. For output a person reads.
	ConfirmDuplicates ConfirmMode = "duplicates"
	// ConfirmNone is the single-judgement graph.
	ConfirmNone ConfirmMode = "none"
)

// confirms reports whether an edge of this kind must be agreed with twice.
//
// An allowlist per mode: an Effect added later is unconfirmed until someone decides,
// rather than inheriting whichever branch it falls through to.
func confirms(mode ConfirmMode, e Effect) bool {
	switch mode {
	case ConfirmBoth:
		return e == EffectDuplicate || e == EffectContradiction
	case ConfirmDuplicates:
		return e == EffectDuplicate
	default: // ConfirmNone, and the zero value
		return false
	}
}

// confirm re-judges the verdicts that would build an edge, keeping only the
// ones the model stands behind twice. groups holds one first-look batch's
// verdicts each; the result is them all, in group order.
//
// Measured on 149 hand-labelled pairs from §14.2: a single judgement calls
// "contradicts" correctly 51% of the time, and 50 of 51 graph-changing errors
// were the same mistake — "contradicts" on a merely related pair. A second,
// agreeing judgement raises precision to 70%.
//
// A trade, not a free win: recall 98% to 67%, F1 barely moving. Which side is
// right depends on the reader, so it is a per-edge-kind choice — see
// ConfirmMode. Only named positives are re-judged: 14% more adjudication calls
// under ConfirmBoth, a tenth of that under ConfirmDuplicates.
//
// A pair that cannot be re-judged — allowance gone, provider failing, or a kind
// this mode skips — KEEPS its first verdict. No edges at all because a second
// opinion was unaffordable is worse than the graph this replaces.
//
// Rests on one property it cannot check: the model must be able to disagree
// with itself. Re-judging reproduced the stored verdict 54% of the time, so a
// provider caching identical requests would make this a silent no-op.
func (v *Verifier) confirm(ctx context.Context, sessionID string, groups [][]Judged, res *Result) []Judged {
	judged := slices.Concat(groups...)
	if len(judged) == 0 {
		return judged
	}

	// Each first-look batch's positives are batched on their own, never merged with
	// another batch's: a second look's prompt — and so, with a real model, its
	// answer — is then the one a serial pass sends, whatever the concurrency. All
	// the groups' second looks still go through one runBatches call, so they run
	// concurrently, in group order then sub-batch order.
	var batches [][]Pair
	for _, g := range groups {
		batches = append(batches, Batches(positivesOf(g, v.Confirm), v.BatchSize)...)
	}
	if len(batches) == 0 {
		return judged
	}

	second := map[string]Relation{}
	// The stop is dropped on purpose: a pair whose second look never happened keeps
	// its first verdict (see above), and the next first-look wave meets the same
	// allowance or provider and reports it.
	again, _, _, _, _ := v.runBatches(ctx, sessionID, batches, res)
	for _, j := range again {
		second[j.Pair.Key()] = j.Relation.Normalize()
	}
	return v.applyConfirmations(judged, second, res)
}

// positivesOf is the subset worth a second call: verdicts building an edge of a kind
// this mode confirms.
//
// No mode confirms "neither" — 86% of pairs, and no edge either way.
func positivesOf(judged []Judged, mode ConfirmMode) []Pair {
	var out []Pair
	for _, j := range judged {
		if confirms(mode, j.Relation.EffectOf()) {
			out = append(out, j.Pair)
		}
	}
	return out
}

// applyConfirmations is the rule, separated from the calls so it can be tested
// without a provider.
func (v *Verifier) applyConfirmations(judged []Judged, second map[string]Relation, res *Result) []Judged {
	out := make([]Judged, 0, len(judged))
	for _, j := range judged {
		if j.Relation.EffectOf() == EffectInert {
			out = append(out, j)
			continue
		}
		got, asked := second[j.Pair.Key()]
		switch {
		case !asked:
			// Never re-judged. Keep the first verdict; see the doc comment.
			out = append(out, j)
		case got == j.Relation.Normalize():
			res.PairsConfirmed++
			out = append(out, j)
		default:
			// The second look disagreed. The edge is not written, and the pair
			// falls back to the relation that changes nothing.
			res.PairsUnconfirmed++
			j.Relation = RelNeither
			j.Rationale = "withheld: a second judgement did not agree (" + string(got) + ")"
			out = append(out, j)
		}
	}
	return out
}

func (v *Verifier) concurrency() int {
	if v.Concurrency > 0 {
		return v.Concurrency
	}
	return DefaultConcurrency
}

// batchOutcome is one batch's adjudicate result, kept by batch index so a wave's
// results are appended in the order the batches were listed, not the order the
// calls returned.
type batchOutcome struct {
	judged   []Judged
	unjudged []Pair
	// responded is whether this batch's call returned a response at all — true for
	// a refusal or a response that failed to parse, false for a call that errored
	// (or was never attempted). See adjudicate.
	responded bool
	stop      error
	// panicked is a recovered panic and its stack, re-raised on the caller's
	// goroutine once the wave has finished.
	panicked any
}

// runBatches adjudicates batches several at a time, in waves.
//
// Each wave is sized once, before it launches, against the allowance as it stands:
// every call in a wave reserves before any of them settles, so a per-batch check
// inside the wave would pass all of them against the same unspent figure and let
// the wave overrun the share. Results are appended in batch-index order, so the
// verdicts — and from them the edges and every later prompt — are what a serial
// pass produces. After a wave in which any batch returned a stop error, no further
// wave starts; the in-flight ones cannot be recalled and are kept.
//
// unjudged is every pair without a verdict: those a launched batch did not answer,
// and those in batches never launched. unanswered is the subset of unjudged whose
// batch got no response at all — see runBatchGroups. consumed counts the pairs in
// launched batches only.
func (v *Verifier) runBatches(ctx context.Context, sessionID string, batches [][]Pair, res *Result) (judged []Judged, unjudged, unanswered []Pair, consumed int, stop error) {
	groups, unjudged, unanswered, consumed, stop := v.runBatchGroups(ctx, sessionID, batches, res)
	return slices.Concat(groups...), unjudged, unanswered, consumed, stop
}

// runBatchGroups is runBatches with the verdicts kept per batch: groups[i] is
// batches[i]'s, empty for a batch that was not launched or not answered. confirm
// needs the grouping to batch each first-look batch's positives on their own.
func (v *Verifier) runBatchGroups(ctx context.Context, sessionID string, batches [][]Pair, res *Result) (groups [][]Judged, unjudged, unanswered []Pair, consumed int, stop error) {
	groups = make([][]Judged, len(batches))
	next := 0
	for next < len(batches) && stop == nil {
		sess, err := v.session(ctx, sessionID)
		if err != nil {
			stop = err
			break
		}
		k, err := v.waveSize(ctx, sess, batches[next:])
		if err != nil {
			stop = err
			break
		}

		wave := batches[next : next+k]
		outcomes := make([]batchOutcome, len(wave))
		var wg sync.WaitGroup
		for i, batch := range wave {
			wg.Add(1)
			go func() {
				defer wg.Done()
				// Serially, a panic here unwound into the session supervisor's
				// recover, which fails that one session and releases its holds. A
				// bare goroutine is invisible to that recover — the executor's lead
				// pool learned this — so the panic is carried back and re-raised
				// below, on the goroutine that called Run.
				defer func() {
					if p := recover(); p != nil {
						outcomes[i].panicked = fmt.Sprintf("%v\n\nadjudication goroutine stack:\n%s", p, debug.Stack())
					}
				}()
				j, u, responded, e := v.adjudicate(ctx, sess, batch, res)
				outcomes[i] = batchOutcome{judged: j, unjudged: u, responded: responded, stop: e}
			}()
		}
		wg.Wait()
		for _, o := range outcomes {
			if o.panicked != nil {
				panic(o.panicked)
			}
		}

		for i, o := range outcomes {
			groups[next+i] = o.judged
			unjudged = append(unjudged, o.unjudged...)
			if !o.responded {
				unanswered = append(unanswered, o.unjudged...)
			}
			consumed += len(wave[i])
			if o.stop != nil && stop == nil {
				stop = o.stop // the lowest-indexed batch's reason, as a serial pass would report
			}
		}
		next += k
	}
	for _, batch := range batches[next:] {
		// Never launched: no response, by definition.
		unjudged = append(unjudged, batch...)
		unanswered = append(unanswered, batch...)
	}
	return groups, unjudged, unanswered, consumed, stop
}

// waveSize is how many of the remaining batches the next wave may launch.
//
// Bounded by Concurrency, by the verification allowance, by what the session has
// available, and by max_tool_calls — each counted for the whole wave, because the
// wave's reservations are all taken before any settles, and a serial pass would
// have re-checked each of them against the previous call's charge. The first batch
// is held to the allowance alone, exactly as the serial pass held every batch: it
// fails with the familiar "allowance exhausted" reason, and adjudicate clamps its
// reservation to what is available.
func (v *Verifier) waveSize(ctx context.Context, sess *core.Session, left [][]Pair) (int, error) {
	limit := min(v.concurrency(), len(left))
	if sess.MaxToolCalls > 0 {
		// At least one, so a session at its ceiling is refused by the ledger with
		// the ceiling's own name rather than by this guess at it.
		limit = min(limit, max(1, int(sess.MaxToolCalls-sess.ToolCallCount)))
	}

	allowance := v.remainingAllowance(ctx, sess)
	avail := sess.Available()
	var sum int64
	k := 0
	for ; k < limit; k++ {
		est := v.estimateBatch(sess.BudgetUnit, len(left[k]))
		if k == 0 {
			if est > allowance {
				return 0, fmt.Errorf("verification allowance exhausted (%d of %.0f%% of budget spent)",
					v.spentSoFar(), v.share()*100)
			}
		} else if sum+est > allowance || sum+est > avail {
			break
		}
		sum += est
	}
	return k, nil
}

func (v *Verifier) spentSoFar() int64 {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.spent
}

// adjudicate reserves, calls, settles, and parses one batch.
//
// Reserve BEFORE the call, per §8.2. The order matters for the same reason it did
// in the planner: an after-the-fact reserve that gets refused because a ceiling
// just fired leaves the cost spent and unrecorded.
//
// The allowance is checked by runBatches for the whole wave, not here: sess is the
// wave's snapshot, and adjudicate runs concurrently with the rest of its wave.
//
// responded reports whether the provider returned a response at all — true for a
// refusal or a response adjudicate could not parse, false when no call was ever
// made (insufficient budget, a reservation refused) or the call itself errored.
// Run uses it to decide which unjudged pairs may still have their claims marked:
// one the model answered, however uselessly, is not re-asked for free; one that
// never got an answer is.
func (v *Verifier) adjudicate(ctx context.Context, sess *core.Session, batch []Pair, res *Result) (judged []Judged, unjudged []Pair, responded bool, err error) {
	sessionID := sess.ID
	est := v.estimateBatch(sess.BudgetUnit, len(batch))
	if avail := sess.Available(); est > avail {
		est = avail
	}
	if est <= 0 {
		return nil, batch, false, fmt.Errorf("%w: nothing left to verify with", budget.ErrInsufficientBudget)
	}

	reservation, rerr := v.Ledger.ReserveVerify(ctx, sessionID, est)
	if rerr != nil {
		return nil, batch, false, rerr
	}

	prompt, _ := adjudicateUserPrompt(batch)
	resp, callErr := v.LLM.Complete(ctx, llm.Request{
		// Cheap tier by default: a bounded comparison of two sentences, run many
		// times, which is what §10.1's split exists for. Model overrides it when the
		// cheap model cannot tell "contradicts" from "unrelated".
		Tier:      llm.TierCheap,
		Model:     v.Model,
		System:    adjudicateSystemPrompt,
		Messages:  []llm.Message{llm.User(prompt)},
		MaxTokens: maxTokensForBatch(len(batch)),
	})

	// Settle unconditionally. The tokens were spent whether or not the response
	// parsed, and a reservation left held is budget neither spent nor available.
	var calls []core.ToolCall
	if resp != nil && !resp.Usage.IsZero() {
		calls = append(calls, core.ToolCall{
			SessionID: sessionID,
			Role:      core.RoleVerifier,
			Type:      core.CallLLM,
			Model:     resp.Model,
			Cost: core.Cost{
				InputTokens:  resp.Usage.InputTokens,
				OutputTokens: resp.Usage.OutputTokens,
			},
			DurationMS: resp.Elapsed.Milliseconds(),
		})
	}
	// WithoutCancel, and a release if the settle still fails. Ctrl-C during a model
	// call cancels ctx, Settle's transaction then fails to even begin, and the hold is
	// stranded — neither spent nor available — which is the one invariant §9.2's loop is
	// arranged around. executor.go:539 already does this; both verifier reservation
	// sites did not, and TestNothingIsLeftHeld covered six model-failure shapes without
	// covering a cancelled context.
	settleCtx := context.WithoutCancel(ctx)
	settled, serr := v.Ledger.Settle(settleCtx, reservation, calls)
	if serr != nil {
		v.logger().WarnContext(settleCtx, "verifier: settle failed; releasing the hold", "err", serr)
		if rerr := v.Ledger.Release(settleCtx, reservation); rerr != nil {
			v.logger().WarnContext(settleCtx, "verifier: release failed; budget is stranded",
				"reservation", reservation.ID, "err", rerr)
		}
	} else {
		charged := settled.Cost.BudgetAmount(sess.BudgetUnit)
		v.mu.Lock()
		v.spent += charged
		res.Spent += charged
		res.Calls++
		v.mu.Unlock()
		if settled.Flagged {
			v.logger().WarnContext(settleCtx,
				"verifier: settled cost exceeded its reservation by more than the allowed factor",
				"reserved", settled.Reserved, "charged", settled.Charged)
		}
	}

	if callErr != nil {
		// §9.5. A bad credential or an exhausted quota will fail every remaining
		// batch identically, and each attempt still takes a reservation and a
		// round trip — so stop rather than walking the whole list into the same
		// wall. Anything else is this batch's problem alone.
		//
		// Either way, no response came back: responded is false, so Run does not
		// mark these claims verified.
		if isFatal(callErr) {
			return nil, batch, false, callErr
		}
		v.logger().WarnContext(ctx, "verifier: adjudication call failed; skipping batch", "err", callErr)
		return nil, batch, false, nil
	}
	if resp.Refused {
		v.logger().WarnContext(ctx, "verifier: model refused to compare claims",
			"category", resp.RefusalCategory)
		// The model responded; a refusal is its answer, not a missing one.
		return nil, batch, true, nil
	}

	verdicts, missed, perr := parseVerdicts(resp.Text, batch)
	if perr != nil {
		v.logger().WarnContext(ctx, "verifier: batch returned no usable verdicts", "err", perr)
	}
	logShortBatch(ctx, v.logger(), batch, resp, len(verdicts))
	// A response came back even where parsing salvaged nothing: these claims were
	// asked about, so they are judged rather than retried for free.
	return verdicts, missed, true, nil
}

// persist writes the edges and marks every target claim verified, except one
// named in skip.
//
// Both in one transaction. A pass that wrote edges but failed to mark the claims
// would regenerate the same pairs next time and pay for them again; one that
// marked them without the edges would lose the graph permanently.
//
// skip is the targets whose pairs this pass never adjudicated (see Run). They get
// neither verified_at nor a score row — ScoreClaims sets verified_at on every row
// it is given, so a skipped claim must not appear in the rows at all, and its
// existing confidence is left exactly as it was.
func (v *Verifier) persist(ctx context.Context, sessionID string, pool, targets []*core.Claim, verdicts []Judged, res *Result, skip map[string]bool) error {
	newEdges := Edges(sessionID, verdicts, v.StalenessGap)

	// Confidence is derived over the WHOLE graph, not just this pass's edges. A new
	// claim corroborating an existing one raises the existing claim's confidence
	// too, and scoring only the targets would leave that claim carrying the number
	// it earned when it stood alone.
	priorEdges := res.priorEdges
	all := append(append([]*core.ClaimEdge(nil), priorEdges...), pointers(newEdges)...)

	scores, breakdown := DeriveConfidence(pool, all)
	res.Scores = breakdown

	// A skipped target must not reach ScoreClaims at all: it sets verified_at on
	// every row it is given, with no way to ask it to touch confidence alone. The
	// derivation can still have produced a row for it — it is scored as part of
	// pool regardless of whether this pass could reach its pairs — so it has to be
	// pulled back out here rather than simply never added.
	if len(skip) > 0 {
		kept := scores[:0]
		for _, sc := range scores {
			if !skip[sc.ClaimID] {
				kept = append(kept, sc)
			}
		}
		scores = kept
	}

	// Every target not skipped must be marked verified, whether or not the
	// derivation covered it.
	//
	// It was covered only as a side effect: scores come from `pool`, and while a session
	// has fewer claims than the store's default limit, pool is a superset of targets so
	// nothing showed. Past that limit the two sets DISJOIN — pool is the oldest claims,
	// all already verified; targets is the next unverified batch — and no target was
	// ever marked. Measured at 505 claims: every pass reported 5 verified and marked
	// zero, so the same pairs were retrieved and paid for again on every replan, and
	// those claims kept confidence 0 permanently, which the report's sort puts last.
	scored := make(map[string]bool, len(scores))
	for _, sc := range scores {
		scored[sc.ClaimID] = true
	}
	verified := 0
	for _, c := range targets {
		if skip[c.ID] {
			continue
		}
		verified++
		if !scored[c.ID] {
			// Confidence 0 is correct here: the derivation did not see this claim's
			// neighbourhood. What matters is that verified_at is set so the work queue
			// drains and the pair is not judged twice.
			scores = append(scores, store.ClaimScore{ClaimID: c.ID, Grounded: c.Grounded})
		}
	}

	if err := v.Store.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		if len(newEdges) > 0 {
			if err := tx.InsertEdges(ctx, newEdges); err != nil {
				return err
			}
		}
		return tx.ScoreClaims(ctx, scores)
	}); err != nil {
		return err
	}

	res.EdgesWritten = len(newEdges)
	res.writtenEdges = newEdges
	res.ClaimsVerified = verified
	res.ClaimsScored = len(scores)
	return nil
}

func pointers(edges []core.ClaimEdge) []*core.ClaimEdge {
	out := make([]*core.ClaimEdge, 0, len(edges))
	for i := range edges {
		out = append(out, &edges[i])
	}
	return out
}

func (v *Verifier) session(ctx context.Context, sessionID string) (*core.Session, error) {
	var s *core.Session
	err := v.Store.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		s, err = q.GetSession(ctx, sessionID)
		return err
	})
	return s, err
}

func (v *Verifier) share() float64 {
	if v.MaxShareOfBudget > 0 {
		return v.MaxShareOfBudget
	}
	return DefaultMaxShareOfBudget
}

// remainingAllowance is what is left of the verification share.
//
// Read from the LEDGER, not from the in-memory counter. §14.3 already sums spend by role,
// and the counter was instance-scoped: a daemon restart, or any caller constructing one
// Verifier per pass, restarted the cumulative cap at zero — while the field's own comment
// claimed it bound across passes. The in-memory figure is kept as a floor, so a batch
// settled but not yet visible to a read cannot be spent twice.
func (v *Verifier) remainingAllowance(ctx context.Context, sess *core.Session) int64 {
	ceiling := int64(float64(sess.Budget) * v.share())

	spent := v.spentSoFar()
	if fromLedger, err := v.verifierSpend(ctx, sess); err == nil && fromLedger > spent {
		spent = fromLedger
	} else if err != nil {
		v.logger().WarnContext(ctx, "verifier: could not read prior verification spend; "+
			"the share cap counts this process only", "err", err)
	}

	if left := ceiling - spent; left > 0 {
		return left
	}
	return 0
}

// verifierSpend is everything charged to RoleVerifier on this session so far.
func (v *Verifier) verifierSpend(ctx context.Context, sess *core.Session) (int64, error) {
	var byRole map[core.Role]core.Cost
	if err := v.Store.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		byRole, err = q.SumCostsByRole(ctx, sess.ID)
		return err
	}); err != nil {
		return 0, err
	}
	return byRole[core.RoleVerifier].BudgetAmount(sess.BudgetUnit), nil
}

// isFatal reports whether an error will fail every remaining batch the same way
// (§9.5). A bad credential and an exhausted quota do not improve on retry.
func isFatal(err error) bool {
	return errors.Is(err, llm.ErrUnauthorized) || errors.Is(err, llm.ErrQuotaExceeded)
}

// fallbackRates price a call whose model this Verifier cannot look up — either
// because it has no Pricing table at all, or because the table has never heard
// of the model (a provider rename, or a Verifier run against a fresh table that
// hasn't registered it yet).
//
// Deliberately high: $5/MTok in, $25/MTok out is this table's own Opus-tier rate
// (pricing.go's claude-opus-5), not a typical cheap-tier number. A reservation
// is a hold taken before the call (§8.2); guessing low is the failure that
// matters, because a hold the settle overruns trips the overshoot detector that
// is supposed to catch a MISPRICED call, not a merely unidentified one.
var fallbackRates = pricing.RateConfig{Input: 5, Output: 25}.Rates()

// fallbackTable prices fallbackRates through the same rounding Table.Cost uses
// for everything else, rather than duplicating its nano-to-micro arithmetic
// here.
var fallbackTable = func() *pricing.Table {
	t := pricing.NewEmptyTable()
	t.Register("fallback", fallbackRates)
	return t
}()

// model is what this Verifier's calls are actually billed against: v.Model
// when the caller named an override, else whatever the cheap tier resolves to
// — the same model adjudicate and the grounding judge pass as llm.Request.Model,
// which the provider fills in from the tier when that field is empty.
func (v *Verifier) model() string {
	if v.Model != "" {
		return v.Model
	}
	if v.LLM != nil {
		return v.LLM.ModelFor(llm.TierCheap)
	}
	return ""
}

// estimate reserves what a call shaped (in, out) tokens is expected to cost, in
// the session's budget unit.
//
// Token mode returns in+out untouched: pricing has no voice in a reservation
// that token mode is meant to measure directly.
//
// USD mode prices the two token counts separately against this Verifier's own
// model, because input and output are priced differently — the flat
// 2-micro-dollar-per-1k-tokens rate this replaced was measured at 2-6
// micro-dollars held against calls the ledger actually charged ~146
// micro-dollars for, almost two orders of magnitude under. A model this
// Verifier cannot price — no table, or a table that does not know it — falls
// back to fallbackRates rather than refusing the reservation; the warning fires
// once per Verifier, not once per call.
func (v *Verifier) estimate(unit core.BudgetUnit, in, out int64) int64 {
	if unit == core.BudgetTokens {
		return in + out
	}

	usage := pricing.Usage{InputTokens: in, OutputTokens: out}
	if v.Pricing != nil {
		if cost, err := v.Pricing.Cost(v.model(), usage); err == nil {
			return max(1, cost.USDMicros)
		}
		v.warnUnpriced.Do(func() {
			v.logger().Warn("verifier: model not in the pricing table; "+
				"reserving at a deliberately high fallback rate instead",
				"model", v.model())
		})
	}

	cost, _ := fallbackTable.Cost("fallback", usage) // "fallback" is always registered above.
	return max(1, cost.USDMicros)
}

// estimateBatch guesses what one adjudication call costs.
//
// Deliberately generous. The estimate is a reservation, not a charge — settling
// replaces it with the real usage — and under-reserving is the failure that
// matters: §8.2's whole point is that the hold is taken before the call, so a hold
// smaller than the call trips the overshoot detector.
func (v *Verifier) estimateBatch(unit core.BudgetUnit, pairs int) int64 {
	// ~350 tokens of instructions, ~120 in and ~60 out per pair, doubled for
	// headroom. Split so USD mode can price input and output at their own
	// rates rather than one blended one; the token-mode total is unchanged.
	in := int64(2 * (350 + pairs*120))
	out := int64(2 * 60 * pairs)
	return v.estimate(unit, in, out)
}

// maxTokensForBatch leaves room for a verdict per pair plus a reasoning model's
// preamble.
//
// The base is what §9.1's planner arrived at for the same reason and this did not,
// which is the whole bug: a reasoning model spends the output budget on its own
// reasoning first and emits content only afterwards, so a ceiling sized for the JSON
// produces empty content with finish_reason "length" (llm.ErrEmptyOutput).
//
// Measured, exactly: gemma4:12b judging four pairs reported "1980 completion tokens
// produced no content", and 1980 was precisely 1500 + 4*120. Half the batches in a
// re-judge run returned nothing at all.
//
// Reasoning cost is per CALL, not per pair, so the base carries it and the per-pair
// term only covers the verdicts. That also means a smaller batch does not divide the
// overhead — it pays it again — which is worth knowing before turning
// llm.verifier-batch-size down to escape a timeout.
//
// Generous on purpose: unused output tokens are not billed.
func maxTokensForBatch(pairs int) int {
	return 4000 + pairs*200
}

// logShortBatch reports a batch that came back with fewer verdicts than pairs.
//
// This existed as a silent loss twice over. A truncated response is not an error at
// either call site — parseVerdicts salvages what arrived and reports the rest as
// unjudged, which is the right behaviour — so a run that answered nine of sixteen
// pairs printed nothing about the seven, and the ceiling that caused it was
// indistinguishable from a model that simply had no opinion.
//
// Both readings lead somewhere different. Guessing between them is how
// maxTokensForBatch was mis-sized twice, so the numbers needed to tell them apart
// are logged rather than inferred: stop_reason "length" with output_tokens at the
// ceiling is truncation, and anything else is the model.
func logShortBatch(ctx context.Context, log *slog.Logger, batch []Pair, resp *llm.Response, judged int) {
	if resp == nil || judged >= len(batch) {
		return
	}
	log.WarnContext(ctx, "verifier: batch answered fewer pairs than it was asked",
		"pairs", len(batch),
		"verdicts", judged,
		"unanswered", len(batch)-judged,
		"stop_reason", resp.StopReason,
		"output_tokens", resp.Usage.OutputTokens,
		"ceiling", maxTokensForBatch(len(batch)),
	)
}

// newPairKey is Pair.Key for two bare IDs, used to index existing edges.
func newPairKey(a, b string) string {
	if a > b {
		a, b = b, a
	}
	return a + "|" + b
}

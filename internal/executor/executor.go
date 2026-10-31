// Package executor runs a session's lead queue: dispatch, retry, replan, settle.
//
// One lead at a time per worker, each with its own budget reservation taken
// before the work and settled after it, so a lead that overruns is visible as a
// flagged overshoot rather than a silent overspend (§8.2).
//
// The loop ends on a drained queue, a ceiling, a wall clock, or a fatal error.
// A fatal error cancels the batch in flight and stops the loop leasing more —
// so the blast radius is one batch rather than one lead, which is what
// --workers 1 buys back.
package executor

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math/rand"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/lajosdeme/mole/internal/actors"
	"github.com/lajosdeme/mole/internal/budget"
	"github.com/lajosdeme/mole/internal/cache"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/planner"
	"github.com/lajosdeme/mole/internal/pricing"
	"github.com/lajosdeme/mole/internal/queue"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/verifier"
)

// Executor runs one session to completion.
type Executor struct {
	Store   store.Store
	Ledger  *budget.Ledger
	Queue   *queue.Queue
	Planner *planner.Planner

	// Verifier builds the claim graph (§11). Nil skips verification entirely,
	// which is a supported configuration: the research still runs, claims still
	// carry verified quotes, and derived confidence stays 0 rather than being
	// invented.
	Verifier *verifier.Verifier
	Actors   map[core.ActorType]actors.Actor
	Log      *slog.Logger

	// Owner identifies the PROCESS holding a lease — "cli" or "daemon" — so the
	// two never claim each other's leads (§9.4).
	//
	// Not per worker, despite what this said before the pool existed. Workers in
	// a session share it, and nothing wants otherwise: a lease is claimed for a
	// specific lead in a transaction, so the lead is already the unit of
	// identity, and the heartbeat renews that lease rather than anything owned
	// by a worker. Boot recovery sweeps by age across all owners.
	Owner string

	// Workers is how many leads this session runs at once. Zero takes
	// DefaultWorkers.
	//
	// Per session rather than per process: the supervisor already bounds how many
	// sessions run at once, and one shared pool would let a single large session
	// starve every other one. The product of the two is what a person actually
	// has to reason about — at the defaults, four sessions of three workers is
	// twelve concurrent leads and twelve outbound fetches — so both numbers stay
	// visible instead of being folded into one.
	Workers int

	// Progress reports phase transitions as they happen. Optional.
	//
	// Not decoration. Planning is a single model call with no output until it
	// returns, and on a local model that is minutes of silence — three runs were
	// killed by hand because the CLI printed a header and then nothing, which
	// reads as a hang rather than as work.
	Progress func(Event)

	// Cache holds artifacts already researched in this session (§9.3). Shared
	// with the actor, so a lead-level hit and a URL-level hit are the same
	// cache and one lead's fetches serve another's.
	Cache *cache.Cache

	// Pricing and CheapModel convert a USD reservation into the token ceiling
	// an actor can act on. CheapModel names the tier chunk mining actually
	// uses; pricing the ceiling off the strong model would set it several
	// times too low. Both empty leaves the actor's configured budget in place.
	Pricing    *pricing.Table
	CheapModel string

	// Estimator sizes reservations. Nil uses a fresh one for the session's
	// budget unit.
	Estimator *budget.Estimator

	// FinishReserve is how much of the wall clock research leaves for a last
	// verify pass, grounding, and the report. Zero uses the default.
	FinishReserve time.Duration

	// Now and Jitter are injectable for tests.
	Now    func() time.Time
	Jitter func() float64

	// Sleep waits between retries. Injectable so a test does not spend the
	// backoff in real time.
	Sleep func(context.Context, time.Duration) error
}

// Event is a step the caller may want to show.
type Event struct {
	Phase string // "planning", "executing", "replanning", "lead", "cached"
	// Detail is a short human-readable note, already formatted.
	Detail string
}

func (e *Executor) emit(phase, detail string) {
	if e.Progress != nil {
		e.Progress(Event{Phase: phase, Detail: detail})
	}
}

// Result is what a session produced.
type Result struct {
	SessionID string
	Status    core.SessionStatus

	Digest *planner.Digest
	Claims []core.Claim

	LeadsRun    int
	LeadsFailed int
	LeadsCached int
	Replans     int

	// CacheStats reports whether the cache earned its keep. An unmeasured
	// cache is an assumption.
	CacheStats cache.Stats

	// Verification totals across every pass (§11).
	VerifyPasses    int
	ClaimsVerified  int
	EdgesWritten    int
	Contradictions  int
	FollowUpsQueued int
	// FollowUpsDeferred counts contradictions worth settling that a pass's
	// budget could not take. Work the session owes.
	FollowUpsDeferred int
	// VerifyDegraded is the last reason a pass could not finish, if any.
	VerifyDegraded string
	// Rechecked and Withheld are contradictions re-checked and removed, across every pass.
	Rechecked int
	Withheld  int

	// Spent is the settled total in the session's budget unit.
	Spent int64
	// StoppedBecause names the ceiling or condition that ended the run, for
	// the report and the trace.
	StoppedBecause string
}

func (e *Executor) logger() *slog.Logger {
	if e.Log != nil {
		return e.Log
	}
	return slog.Default()
}

func (e *Executor) now() time.Time {
	if e.Now != nil {
		return e.Now()
	}
	return time.Now()
}

func (e *Executor) jitter() float64 {
	if e.Jitter != nil {
		return e.Jitter()
	}
	return rand.Float64()
}

func (e *Executor) sleep(ctx context.Context, d time.Duration) error {
	if e.Sleep != nil {
		return e.Sleep(ctx, d)
	}
	if d <= 0 {
		return ctx.Err()
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// Run executes a session from prompt to exhausted queue.
//
// Escrow is already held from session creation (§8.3), so the loop can spend
// down to Available() without starving the report.
func (e *Executor) Run(ctx context.Context, sessionID string) (*Result, error) {
	sess, err := e.session(ctx, sessionID)
	if err != nil {
		return nil, err
	}

	// One estimator for the whole session, so Observe actually accumulates.
	// It had no callers at all, and e.estimate built a fresh one per call, so
	// every reservation was the cold seed forever and §8.4's "improves with
	// use" never happened. That mattered more once the sub-budget started
	// deriving the actor's ceiling from the reservation: a permanently wrong
	// estimate now caps how much work each lead may do.
	if e.Estimator == nil {
		e.Estimator = budget.NewEstimator(sess.BudgetUnit)
		// Warmed from finished leads on this install, across sessions (§8.4's
		// "improves with use" — which, without this, meant improves within one
		// session and forgets). A cold start is not an error: an install with no
		// history keeps the conservative seeds.
		e.warmEstimator(ctx, sess)
	}

	res := &Result{SessionID: sessionID, Status: core.StatusDone}
	digest := planner.NewDigest(sess.Prompt, 0)
	res.Digest = digest

	// leadQuestion maps a dispatched lead back to the sub-question it serves,
	// so coverage lands on the right line of the digest.
	//
	// In memory, so a resumed session loses the association and its digest
	// starts empty. Acceptable while M3 has no resume path; M5's pool shares
	// this executor, and a restart is a new Run.
	leadQuestion := map[string]string{}

	// 1. Decompose.
	e.emit("planning", "decomposing the question")
	plan, err := e.planWithBudget(ctx, sess, func() (*planner.Plan, error) {
		return e.Planner.InitialLeads(ctx, sess)
	})
	if err != nil {
		// With no initial plan there is nothing to research, so this ends the
		// session either way — but WHY it ended is not the same thing. A user
		// pressing Ctrl-C during the first planner call is a cancellation, and
		// reporting it as a failure makes a deliberate interrupt look like a
		// crash to whoever reads the status or the exit code. The main loop
		// already distinguishes these; this path did not.
		res.Status = core.StatusFailed
		if ctxErr := ctx.Err(); ctxErr != nil {
			res.Status = statusForContext(ctxErr)
		}
		res.StoppedBecause = "planning failed: " + err.Error()
		return res, nil
	}
	// Map leads to the IDs the DIGEST assigned, not the ones the model supplied
	// — those collide across replans, and a positional mapping onto the input
	// credits a new question's coverage to whatever it collided with.
	assigned := digest.AddQuestions(plan.Questions)
	for i := range plan.Leads {
		if plan.Leads[i].ID == "" {
			plan.Leads[i].ID = core.NewLeadID()
		}
		if i < len(assigned) {
			leadQuestion[plan.Leads[i].ID] = assigned[i].ID
		}
	}
	if err := e.Queue.Push(ctx, plan.Leads); err != nil {
		return res, err
	}
	e.emit("executing", fmt.Sprintf("%d sub-question(s) queued", len(plan.Leads)))

	// 2. Work the queue.
	completedSinceReplan := 0
	depth := 0
	// plannerDone latches once the planner has said its piece. Follow-up leads
	// can still be dispatched after that, but asking the planner again buys
	// another model call to be told the same thing.
	plannerDone := false

	for {
		if err := ctx.Err(); err != nil {
			res.Status = statusForContext(err)
			res.StoppedBecause = err.Error()
			break
		}

		sess, err = e.session(ctx, sessionID)
		if err != nil {
			return res, err
		}
		now := e.now()
		if hit, which := sess.HitCeiling(now); hit {
			res.Status = core.StatusExhausted
			res.StoppedBecause = which
			break
		}
		// Inside the finishing reserve: stop leasing new work so there is
		// time to verify what research already found.
		if reserve := finishReserve(sess.MaxWallClock, e.FinishReserve); reserve > 0 &&
			!now.Before(sess.CreatedAt.Add(sess.MaxWallClock-reserve)) {
			res.Status = core.StatusExhausted
			res.StoppedBecause = "max_wallclock"
			e.logger().InfoContext(ctx, "stopping research inside the finishing reserve",
				"reserve", reserve)
			break
		}
		if sess.Available() <= 0 {
			res.Status = core.StatusExhausted
			res.StoppedBecause = "budget exhausted"
			break
		}

		// Lease a BATCH, not a lead. Its size is bounded by the worker count and
		// by how many leads may still run before a replan is due, so the pool
		// never runs past a replan boundary — see runBatch for why that matters.
		batch, err := e.leaseBatch(ctx, sessionID, e.batchSize(sess, completedSinceReplan))
		if err != nil {
			// Release what was already claimed. leaseBatch returns its partial
			// batch precisely so this is possible: those leads are leased in the
			// database, and a session that returns here never reaches the sweep —
			// boot recovery is per PROCESS, and a long-lived daemon does not
			// restart. Discarding them left rows leased forever.
			for _, lease := range batch {
				if rerr := e.Queue.Release(context.WithoutCancel(ctx), lease); rerr != nil {
					e.logger().WarnContext(ctx, "could not release a lease after a failed batch",
						"lead", lease.Lead.ID, "err", rerr)
				}
			}
			return res, err
		}

		if len(batch) == 0 {
			// Queue drained. One last replan can add work; if it does not, the
			// session is done — unless it still owes follow-ups, which are work
			// the planner has no view of and never agreed to stop.
			if !e.Planner.ShouldReplan(completedSinceReplan, true) {
				if e.hasWorkLeft(ctx, sessionID) {
					continue
				}
				break
			}
			// Last chance to verify: after this the queue is empty and, if the
			// planner adds nothing, the loop ends. Skipping it here would leave the
			// final batch of claims unverified and unscored — every one of them
			// reaching the report with confidence 0.
			e.verify(ctx, digest, res)

			// That verification may have queued work, and it is the work this
			// engine exists to do (§11.4). Go and run it — including anything it
			// judged worth doing and had no budget for this pass.
			if e.hasWorkLeft(ctx, sessionID) {
				continue
			}

			if plannerDone {
				// Asked once, answered. Everything queued since has now drained,
				// and nothing is owed.
				break
			}

			added, done, reason, err := e.replan(ctx, sess, digest, depth, res, leadQuestion)
			if err != nil {
				// A failed replan at drain time is the END of the research, not
				// a failed session. §9.5 classifies 429/5xx/timeout as transient
				// — never abort — and this used to return, so a complete run
				// with claims collected reported as failed with a non-zero exit
				// because its last planner call got throttled. The mid-loop
				// replan below already only warns; the two now agree.
				e.logger().WarnContext(ctx, "final replan failed; ending with what was found", "err", err)
				res.StoppedBecause = "final replan failed: " + err.Error()
				break
			}
			completedSinceReplan = 0
			if done || added == 0 {
				// The cap and "the planner had nothing left to add" are
				// different endings; report whichever it was.
				res.StoppedBecause = orElse(reason, "no further leads")
				// The replan itself does not queue follow-ups, but it verifies
				// nothing either — so the only way back into the loop here is
				// work the verification above created and this pass has not yet
				// seen. Checked again because replan can take long enough for a
				// concurrent worker to finish a lead.
				if e.hasWorkLeft(ctx, sessionID) {
					plannerDone = true
					continue
				}
				break
			}
			depth++
			continue
		}

		outcomes := e.runBatch(ctx, sess, batch, leadQuestion)

		// Apply on THIS goroutine, in the order the planner queued the leads —
		// not the order they finished. Everything mutable lives here: the digest,
		// the Result counters, and the claim slice.
		var (
			fatal     *leadOutcome
			exhausted bool
		)
		for i := range outcomes {
			out := outcomes[i]
			completedSinceReplan++

			if out.fromCache {
				e.emit("cached", truncateQuery(out.lead.Query))
				res.LeadsCached++
			} else {
				res.LeadsRun++
				e.emit("lead-done", fmt.Sprintf("%d claim(s)%s",
					len(out.claims), leadNote(out)))
			}

			digest.RecordLead(out.questionID)
			if out.claimsFound > 0 {
				digest.RecordClaims(out.questionID, out.claimsFound)
			}
			// A sibling cut short by a fatal in its own batch is not evidence
			// about the research: the question was never really asked. Recording
			// it as a dead end tells the planner a route is blocked when nothing
			// was tried, and LeadsFailed, the dead-end histogram and the lead
			// statuses are all §14.3 eval inputs.
			cancelled := errors.Is(out.err, context.Canceled)
			if out.deadEnd != "" && !cancelled {
				digest.RecordDeadEnd(out.deadEnd, deadEndExample(&out.lead))
			}
			res.Claims = append(res.Claims, out.claims...)

			// The lead is counted by its own reservation, not here. Ledger
			// .ReserveFor applies LeadCount in the transaction that checks
			// MaxLeads, so the check and the increment cannot be separated by
			// another worker.
			if out.err != nil && out.class != Fatal && !cancelled {
				res.LeadsFailed++
			}
			if out.class == Fatal && fatal == nil {
				// Keep the FIRST fatal in queue order, so which one is reported
				// does not depend on which worker lost the race. The rest of the
				// batch is still applied: it ran, and it was paid for.
				o := out
				fatal = &o
				exhausted = errors.Is(out.err, budget.ErrInsufficientBudget)
			}
		}

		if fatal != nil {
			// A ceiling is not a failure: the session did what it was allowed
			// to do. Reporting it as failed would make every correctly-bounded
			// run look broken.
			if exhausted {
				res.Status = core.StatusExhausted
				res.StoppedBecause = ceilingReason(fatal.err)
				break
			}
			res.Status = core.StatusFailed
			res.StoppedBecause = "fatal: " + fatal.err.Error()
			// Escrow survives, so a partial report is still affordable (§9.5).
			return res, nil
		}

		if e.Planner.ShouldReplan(completedSinceReplan, false) {
			// Verify BEFORE replanning, so the planner sees the disagreements this
			// batch of leads turned up. A sub-question whose evidence is
			// contradicted is not answered, and a planner told only the claim count
			// has no way to know the difference.
			e.verify(ctx, digest, res)

			// The latch is honoured HERE too, and it was not: a session draining
			// the verifier's follow-ups asked the planner again every ReplanEvery
			// completed leads, paying for a model call each time to be told the
			// same thing it had already said. Verification still runs — new claims
			// need scoring either way — but the planner has finished.
			if plannerDone {
				completedSinceReplan = 0
				continue
			}

			added, done, reason, err := e.replan(ctx, sess, digest, depth, res, leadQuestion)
			if err != nil {
				e.logger().WarnContext(ctx, "replan failed; continuing with the current queue", "err", err)
			}
			completedSinceReplan = 0
			if done {
				res.StoppedBecause = orElse(reason, "planner reported done")
				// Same rule as the drain branch: the verification above may have
				// queued leads to settle a contradiction, and the planner has no
				// view of them. Draining them is not overruling it — it never
				// answered this question.
				if e.hasWorkLeft(ctx, sessionID) {
					plannerDone = true
					continue
				}
				break
			}
			if added > 0 {
				depth++
			}
		}
	}

	// On any ceiling stop, verify what research found before building the
	// report.
	if res.Status == core.StatusExhausted && ctx.Err() == nil {
		e.verify(ctx, digest, res)
	}
	if ctx.Err() == nil {
		e.recheck(ctx, res)
	}

	// Without a live context this read fails on a timed-out or cancelled run —
	// exactly the runs whose spend most needs reporting — and the result says
	// 0 spent while the ledger holds the real figure.
	final, err := e.session(context.WithoutCancel(ctx), sessionID)
	if err == nil {
		res.Spent = final.Spent
	}
	res.CacheStats = e.Cache.Stats()
	return res, nil
}

// finishReserve is how long before the wall clock research stops, to leave
// time to verify, ground and report. Configured value, else 15% (min 90s, max half); 0 without a clock.
func finishReserve(maxWall, configured time.Duration) time.Duration {
	if maxWall <= 0 {
		return 0
	}
	reserve := configured
	if reserve <= 0 {
		reserve = max(90*time.Second, maxWall*15/100)
	}
	return min(reserve, maxWall/2)
}

// DefaultWorkers is the per-session pool size when Workers is unset.
//
// Equal to planner.DefaultReplanEvery, and that is the whole reason for the
// number. batchSize never runs past a replan boundary, so a pool larger than the
// replan cadence has workers that can never all be busy: with the previous
// default of 4 against a cadence of 3, the fourth worker was unreachable in
// every shipped configuration while three comments, the daemon's startup banner
// and the README all advertised four — and "four sessions of four workers is
// sixteen concurrent leads" was wrong by a third.
//
// Tying the two together means the advertised number is the real one. Raising
// concurrency now means raising both, deliberately, and paying the planner cost
// §9.1 exists to control.
const DefaultWorkers = planner.DefaultReplanEvery

// EffectiveWorkers is how many leads a given Workers setting actually permits,
// before the per-batch bounds in batchSize narrow it further.
//
// Exported because a caller has to be able to ask. cmd/mole's cassette guard
// checked the raw flag, so --workers 0 — which means "use the default" — passed
// a check meant to enforce serial execution and then ran a pool.
func EffectiveWorkers(workers int) int {
	switch {
	case workers <= 0:
		return DefaultWorkers
	case workers > MaxWorkers:
		return MaxWorkers
	default:
		return workers
	}
}

// MaxWorkers clamps Workers however it was configured.
//
// The ceiling is about what leaves this machine, not about goroutines. Each
// worker holds a lead, and a lead is searches and fetches against real hosts; a
// misread config or a fat-fingered flag should not be able to point an
// unbounded fan of requests at one domain from a user's address. The per-domain
// rate limiter still applies underneath.
const MaxWorkers = 16

func (e *Executor) workers() int { return EffectiveWorkers(e.Workers) }

// batchSize is how many leads may run before the next replan is due.
//
// Bounded by the replan cadence and not just the worker count, so turning on
// workers does not silently change how often the planner is consulted: with
// ReplanEvery at 3, a pool of 4 still stops at 3 and replans. Asked through
// ShouldReplan rather than reading ReplanEvery, so the two cannot disagree.
func (e *Executor) batchSize(sess *core.Session, completedSinceReplan int) int {
	n := 1
	for n < e.workers() && !e.Planner.ShouldReplan(completedSinceReplan+n, false) {
		n++
	}

	// Shrink as the unit-independent ceilings close in (§8.5).
	//
	// MaxLeads is exact — the reservation counts it in the transaction that
	// checks it — so this only avoids leasing work that will certainly be
	// refused. MaxToolCalls is NOT exact: ToolCallCount is written by Settle,
	// after the calls, so K workers all pass the check on a stale count and
	// overshoot. Measured at a ceiling of 6: one worker stopped at 6 tool calls,
	// four workers reached 16.
	//
	// Bounding the batch by the remaining headroom bounds that overshoot instead
	// of eliminating it: a lead makes several calls, so leasing at most one lead
	// per remaining call still overshoots by a lead's worth. It restores serial
	// behaviour exactly where it matters — at the ceiling — and the residue is
	// stated in the README rather than left to be discovered.
	if room := headroom(sess.MaxLeads, sess.LeadCount); room >= 0 && n > room {
		n = room
	}
	if room := headroom(sess.MaxToolCalls, sess.ToolCallCount); room >= 0 && n > room {
		n = room
	}
	if n < 1 {
		n = 1
	}
	return n
}

// headroom is how far a counter is from its ceiling, or -1 when unbounded.
func headroom(ceiling, used int64) int {
	if ceiling <= 0 {
		return -1
	}
	if room := ceiling - used; room > 0 {
		return int(room)
	}
	return 0
}

// hasQueuedLeads reports whether work is waiting that this loop can take.
//
// It exists because the loop used to end on the planner's verdict alone. The
// Verifier queues leads too — a contradiction is the one finding §11.4 says is
// worth more budget, and it composes the query that would settle it — but
// replan asks the PLANNER, which knows nothing about them. So `added == 0` was
// read as "there is nothing to do" when it only ever meant "the planner had
// nothing to add", and every session on the corpus ended with the leads that
// would have settled its disagreements sitting in the queue, at 2-6% of budget
// spent.
//
// Queued and not Pending: a leased lead is held by another worker and is not
// work this loop can take, so counting it would spin.
//
// A store that cannot answer reports false, which ends the run. The alternative
// is a loop that cannot see its own exit condition.
func (e *Executor) hasQueuedLeads(ctx context.Context, sessionID string) bool {
	st, err := e.Queue.Stats(ctx, sessionID)
	if err != nil {
		e.logger().WarnContext(ctx, "could not count queued leads; ending the run",
			"session", sessionID, "err", err)
		return false
	}
	return st.Queued > 0
}

// hasWorkLeft reports whether the session can still do something, promoting
// work it owes when nothing is takeable now.
//
// A run must not end holding contradictions it already judged worth settling.
// Promotion is not a way around §8.5: the ceilings are checked at the top of
// every turn, so a promoted lead meets the same limits as any other.
func (e *Executor) hasWorkLeft(ctx context.Context, sessionID string) bool {
	if e.hasQueuedLeads(ctx, sessionID) {
		return true
	}
	n, err := e.Queue.PromoteDeferred(ctx, sessionID, 0)
	if err != nil {
		e.logger().WarnContext(ctx, "could not promote deferred follow-ups; ending the run",
			"session", sessionID, "err", err)
		return false
	}
	if n == 0 {
		return false
	}
	e.logger().InfoContext(ctx, "promoted follow-ups the session still owed",
		"session", sessionID, "leads", n)
	return true
}

// leaseBatch takes up to max leads off the queue.
//
// Leasing is what makes the pool safe to run at all (§9.4): the lease is claimed
// in a transaction, so two workers — or two processes — cannot take the same
// lead. Stopping at the first nil is not an error, it is the queue being shorter
// than the batch.
func (e *Executor) leaseBatch(ctx context.Context, sessionID string, max int) ([]*queue.Lease, error) {
	var batch []*queue.Lease
	for len(batch) < max {
		lease, err := e.Queue.LeaseNext(ctx, sessionID, e.owner())
		if err != nil {
			return batch, err
		}
		if lease == nil {
			break
		}
		batch = append(batch, lease)
	}
	return batch, nil
}

// runBatch runs a batch of leads concurrently and returns their outcomes IN
// BATCH ORDER.
//
// Order is the point. Outcomes are written to a preallocated slot per lead
// rather than appended as they arrive, so the coordinator applies them in the
// order the planner queued them however the network answered. Without that the
// digest — and therefore the replan prompt, and therefore what gets researched
// next — would depend on which fetch happened to be quickest.
//
// Distinct slots also mean the writes need no lock: each goroutine owns one
// element and nothing reads the slice until Wait returns.
//
// leadQuestion is read HERE, on the coordinator's goroutine, before any worker
// starts. The map is never handed to a worker, which is what keeps it a plain
// map rather than something that needs guarding.
func (e *Executor) runBatch(
	ctx context.Context,
	sess *core.Session,
	batch []*queue.Lease,
	leadQuestion map[string]string,
) []leadOutcome {
	out := make([]leadOutcome, len(batch))

	// A fatal outcome cancels the rest of the batch. Fatal means the failure
	// repeats — a bad key fails on every lead — so continuing spends money on
	// calls that cannot succeed. Serially the next lead simply never started;
	// with a pool its siblings are already in flight, so the equivalent is to cut
	// them short.
	//
	// NOT for an exhausted budget, though ErrInsufficientBudget is also Fatal.
	// That one means "this worker could not get a reservation", which says
	// nothing about the siblings that already did — and cancelling them threw
	// away funded, healthy work. The loop stops before leasing another batch
	// either way, so nothing is gained by killing them.
	batchCtx, cancelBatch := context.WithCancel(ctx)
	defer cancelBatch()

	// Each worker may clamp to its OWN share of what is left, not to the whole
	// balance. Every worker reading the same pre-batch Available() meant the
	// first to reserve could take all of it, the rest were refused, and the
	// refusal ended the session: measured at 1 lead run and 1.2% of the budget
	// spent where one worker ran 3 leads and finished.
	share := sess.Available()
	if n := int64(len(batch)); n > 1 && share > 0 {
		share /= n
	}

	var wg sync.WaitGroup
	for i, lease := range batch {
		questionID := leadQuestion[lease.Lead.ID]
		e.emit("lead", truncateQuery(lease.Lead.Query))

		wg.Add(1)
		go func(i int, lease *queue.Lease, questionID string) {
			defer wg.Done()
			// A panic here used to kill the process. Before the pool, runLead ran
			// on the session goroutine and unwound into the supervisor's recover,
			// which failed that one session and released its holds; a bare worker
			// goroutine is invisible to that recover, so one malformed page taking
			// an index panic through extraction would take down the daemon and
			// strand every other session's reservations. Recovered here, where the
			// goroutine is, and reported as the fatal it is.
			defer func() {
				if r := recover(); r != nil {
					e.logger().ErrorContext(ctx, "lead panicked",
						"lead", lease.Lead.ID, "panic", r, "stack", string(debug.Stack()))
					out[i] = leadOutcome{
						lead:       *lease.Lead,
						questionID: questionID,
						err:        fmt.Errorf("executor: lead panicked: %v", r),
						class:      Fatal,
						deadEnd:    "panic",
					}
					e.complete(ctx, lease, core.LeadFailed)
					cancelBatch()
				}
			}()

			if cached, ok := e.tryCache(batchCtx, lease, questionID); ok {
				out[i] = cached
				return
			}
			o := e.runLead(batchCtx, sess, lease, questionID, share)
			if o.class == Fatal && !errors.Is(o.err, budget.ErrInsufficientBudget) {
				cancelBatch()
			}
			out[i] = o
		}(i, lease, questionID)
	}
	wg.Wait()
	return out
}

// tryCache satisfies a lead from a prior identical one.
//
// A cache hit RETURNS the prior result; it does not skip the lead (§9.3). Rev 1
// skipped, so the planner asked for something and got nothing back — and the
// next replan spawned an equivalent lead, forever. Recording the coverage is
// what closes that loop, which is why the outcome carries claimsFound and a dead
// end rather than nothing.
//
// A hit costs nothing: the claims are already in the store under the lead that
// first found them, and re-inserting copies would inflate every claim count
// §14.3 reads without giving a reader anything.
func (e *Executor) tryCache(ctx context.Context, lease *queue.Lease, questionID string) (leadOutcome, bool) {
	if e.Cache == nil {
		return leadOutcome{}, false
	}
	entry, ok := e.Cache.Get(cache.QueryKey(lease.Lead.Query))
	if !ok {
		return leadOutcome{}, false
	}

	out := leadOutcome{
		lead:        *lease.Lead,
		questionID:  questionID,
		claimsFound: entry.Claims,
		fromCache:   true,
	}
	if entry.Claims == 0 {
		out.deadEnd = "no_evidence"
	}
	e.complete(ctx, lease, core.LeadSkippedCache)
	return out, true
}

type leadOutcome struct {
	lead       core.Lead
	questionID string

	claims []core.Claim
	// claimsFound is what the digest counts. Separate from len(claims) because a
	// cache hit knows the count without re-inserting the claims: they are already
	// in the store under the lead that first found them, and copying them would
	// inflate every claim count §14.3 reads.
	claimsFound int
	// deadEnd is the cause to record, empty when the lead produced evidence.
	deadEnd   string
	fromCache bool

	err   error
	class Class
}

// runLead reserves, runs, settles, and completes one lead.
//
// Retries live here rather than inside the actor because a transient failure
// means the LEAD should be retried — a new search, new fetches — and the actor
// has no notion of being run twice.
func (e *Executor) runLead(
	ctx context.Context,
	sess *core.Session,
	lease *queue.Lease,
	questionID string,
	share int64,
) leadOutcome {
	lead := *lease.Lead
	base := leadOutcome{lead: lead, questionID: questionID}

	actor, ok := e.Actors[lead.ActorType]
	if !ok {
		err := fmt.Errorf("executor: no actor registered for %q", lead.ActorType)
		e.complete(ctx, lease, core.LeadFailed)
		out := base
		out.err, out.class, out.deadEnd = err, Fatal, "no_actor"
		return out
	}

	var last leadOutcome
	for attempt := 1; attempt <= MaxAttempts; attempt++ {
		if attempt > 1 {
			if err := e.sleep(ctx, Backoff(attempt-1, e.jitter)); err != nil {
				last.err, last.class = err, Degraded
				break
			}
			// A retry re-runs the whole lead and is charged again, so a lost
			// lease has to stop it: another worker may already be running this.
			if ok, err := e.Queue.Renew(ctx, lease); err != nil || !ok {
				e.logger().WarnContext(ctx, "lease lost mid-retry; abandoning the lead",
					"lead", lead.ID, "attempt", attempt)
				lost := base
				lost.err, lost.class = errors.New("executor: lease lost"), Degraded
				lost.deadEnd = DeadEndCause(lost.err)
				return lost
			}
		}

		out := e.attempt(ctx, sess, lead, actor, lease, attempt == 1, share)
		last = out

		if out.err == nil {
			out.lead, out.questionID = lead, questionID
			out.claimsFound = len(out.claims)
			if len(out.claims) == 0 {
				// A lead that ran cleanly and found nothing is still a dead end
				// for the planner: the question was asked and not answered.
				out.deadEnd = "no_evidence"
			}
			e.Cache.Put(&cache.Entry{
				Key:    cache.QueryKey(lead.Query),
				Claims: len(out.claims),
			})
			e.complete(ctx, lease, core.LeadDone)
			return out
		}
		if out.class != Transient {
			break
		}
		e.logger().WarnContext(ctx, "lead failed transiently; retrying",
			"lead", lead.ID, "attempt", attempt, "err", out.err)
	}

	// Out of attempts, or not worth retrying. §9.5: a transient error that
	// exhausts its retries becomes degraded — the session continues.
	last.lead, last.questionID = lead, questionID
	last.deadEnd = DeadEndCause(last.err)
	// Claims from a partial run are kept: the actor verified every quote it
	// returned, and discarding them because a later chunk failed throws away
	// evidence that was paid for.
	if last.class != Fatal {
		last.class = Degraded
	}
	e.complete(ctx, lease, core.LeadFailed)
	return last
}

// attempt is one reserve → run → settle cycle.
func (e *Executor) attempt(ctx context.Context, sess *core.Session, lead core.Lead, actor actors.Actor, lease *queue.Lease, first bool, share int64) leadOutcome {
	est := e.estimate(sess, lead)
	// Clamped to this lead's SHARE of what remains, not to the whole balance.
	// The ledger would refuse the losers correctly either way, but the refusal
	// is Fatal, so one greedy clamp ended sessions that had budget left.
	if share > 0 && est > share {
		est = share
	}
	if est <= 0 {
		return leadOutcome{err: budget.ErrInsufficientBudget, class: Fatal}
	}

	// Only the first attempt counts against MaxLeads. Every attempt holds and
	// settles its own money, because every attempt really does spend — but §8.5's
	// lead ceiling bounds research fan-out, and a lead retried through a rate
	// limit is still one lead. Counting each attempt would let a flaky provider
	// shrink the research plan while the ceiling reported the work as done.
	reserve := e.Ledger.ReserveForRetry
	if first {
		reserve = e.Ledger.ReserveFor
	}
	reservation, err := reserve(ctx, sess.ID, lead.ID, est)
	if err != nil {
		return leadOutcome{err: err, class: Classify(err)}
	}

	// §9.2's withSubBudget. The reservation is the only ceiling that knows what
	// this lead may cost; without passing it the actor's configured budget bore
	// no relation to the money held for it.
	// Heartbeat while the lead runs. Renew used to be called only BETWEEN
	// retries, so a lead that legitimately outlasted the lease TTL — a slow
	// fetch plus a slow model call — silently lost its lease and became eligible
	// for a sweep, which is §9.4's "workers heartbeat their lease" unimplemented.
	stopHeartbeat := e.heartbeat(ctx, lease)
	result, runErr := actor.Run(actors.WithSubBudget(ctx, e.subBudget(sess, est)), lead)
	stopHeartbeat()

	// Settle unconditionally. The tokens were billed either way, and a
	// reservation left held is budget neither spent nor available.
	var costs []core.ToolCall
	if result != nil {
		costs = result.Costs
	}
	settled, err := e.Ledger.Settle(context.WithoutCancel(ctx), reservation, costs)
	if err != nil {
		e.logger().ErrorContext(ctx, "settle failed; budget accounting is now unreliable",
			"lead", lead.ID, "err", err)
		return leadOutcome{err: err, class: Fatal}
	}
	// Feed the estimate back before anything else. Even an overshoot is a
	// sample — arguably the most valuable one, since it is the case the seed
	// got wrong.
	e.Estimator.Observe(lead.ActorType, lead.Depth, settled.Charged)

	// §8.2 bounds overshoot to estimate error on a single lead. Flagged means
	// the actual cost cleared the reserved amount by more than the configured
	// factor, which is not estimate error — it is the sub-budget failing to
	// bind. Continuing would repeat it on every remaining lead, so stop.
	if settled.Flagged {
		e.logger().ErrorContext(ctx, "lead overshot its reservation past the allowed factor",
			"lead", lead.ID, "reserved", settled.Reserved, "charged", settled.Charged)
		out := leadOutcome{err: fmt.Errorf(
			"%w: lead charged %d against a %d reservation", budget.ErrOvershoot, settled.Charged, settled.Reserved),
			class: Fatal}
		if result != nil {
			out.claims = result.Claims
		}
		return out
	}

	out := leadOutcome{}
	if result != nil {
		out.claims = result.Claims
	}
	if runErr != nil {
		out.err, out.class = runErr, Classify(runErr)
	}
	return out
}

// deadEndExample is the query to record against a dead end, or empty.
//
// §9.1's digest deliberately carries NO page-derived text: sub-question wording is the
// planner's own output, dead-end causes are a fixed enum, everything else is a count.
// The comment on the planner package calls that "stronger than fencing it — there is
// nothing to fence", and it stopped being true when M4 added verification follow-ups.
//
// A follow-up lead's query is built by disambiguationQuery from up to 240 characters of
// raw CLAIM text, and a follow-up searching for a contradiction is exactly the lead
// most likely to dead-end — at which point the query became DeadEnd.Example and was
// rendered into the next replan prompt.
//
// The example exists so the planner can tell a blocked domain from a badly phrased
// search. It cannot rephrase a query it did not write, so for a verifier-authored lead
// the example is useless as well as unsafe.
func deadEndExample(lead *core.Lead) string {
	if lead.RootClaimID != nil {
		return ""
	}
	return lead.Query
}

// orElse is the first non-empty of two strings.
func orElse(s, fallback string) string {
	if s == "" {
		return fallback
	}
	return s
}

// replan asks the planner what to do next and queues the answer.
func (e *Executor) replan(
	ctx context.Context,
	sess *core.Session,
	digest *planner.Digest,
	depth int,
	res *Result,
	leadQuestion map[string]string,
) (added int, done bool, reason string, err error) {
	// Before reserving. The cap needs no model call, so paying for one — and
	// misreporting the stop reason when the reservation is refused — is pure loss.
	if e.Planner.DepthExhausted(depth) {
		return 0, true, e.Planner.DepthCapReason(), nil
	}

	// Reload. The loop reloads the session at the TOP of an iteration, before the
	// lead runs, so by the time a replan happens it is a lead behind — stale by
	// exactly the spend, tool calls, and lead count the replan is meant to react
	// to. Measured: a session one of four leads in reported 85% of its allowance
	// left, from the dollar term, while the lead ceiling it was actually running
	// out of said 75%.
	//
	// Cheap: §9.1 batches replans precisely so there are few of them.
	if fresh, ferr := e.session(ctx, sess.ID); ferr == nil {
		sess = fresh
	} else {
		e.logger().WarnContext(ctx, "replan: could not refresh session; planning against slightly stale budget", "err", ferr)
	}

	// §9.1 asks the planner whether the open sub-questions are worth more budget.
	// Answering that needs the figure, and the digest carried nothing about it.
	digest.BudgetRemaining = sess.RemainingFraction(e.now())

	e.emit("replanning", fmt.Sprintf("%d open sub-question(s)", len(digest.Open())))
	plan, perr := e.planWithBudget(ctx, sess, func() (*planner.Plan, error) {
		return e.Planner.Replan(ctx, sess, digest, depth)
	})
	if perr != nil {
		return 0, false, "", perr
	}
	res.Replans++

	for _, id := range plan.Answered {
		digest.MarkAnswered(id)
	}
	if plan.Done {
		// Done is honoured when the plan adds nothing or the override threshold is passed.
		if len(plan.Leads) == 0 || !keepResearching(sess, digest, e.now()) {
			return 0, true, "planner reported done", nil
		}
		e.emit("replanning", "ignoring the planner's done: most of the allowance is left and the research is unsettled")
		e.logger().InfoContext(ctx, "planner done overridden",
			"remaining", digest.BudgetRemaining, "open", len(digest.Open()), "contradictions", digest.Contradictions)
	}

	assigned := digest.AddQuestions(plan.Questions)
	for i := range plan.Leads {
		if plan.Leads[i].ID == "" {
			plan.Leads[i].ID = core.NewLeadID()
		}
		if i < len(assigned) {
			leadQuestion[plan.Leads[i].ID] = assigned[i].ID
		}
	}
	if len(plan.Leads) == 0 {
		return 0, false, "", nil
	}
	if err := e.Queue.Push(ctx, plan.Leads); err != nil {
		return 0, false, "", err
	}
	return len(plan.Leads), false, "", nil
}

// keepResearching reports whether a planner's done is ignored: over half the allowance
// left, under half of max_leads run, and contradictions or open sub-questions.
func keepResearching(sess *core.Session, d *planner.Digest, now time.Time) bool {
	if sess.RemainingFraction(now) <= 0.5 {
		return false
	}
	if sess.MaxLeads > 0 && 2*sess.LeadCount >= sess.MaxLeads {
		return false
	}
	return d.Contradictions > 0 || len(d.Open()) > 0
}

// planWithBudget reserves, runs a planning call, and settles it.
//
// Reserve BEFORE the call, which is §8.2's order and was inverted here: the call
// happened first and a reservation of exactly 1 was taken afterwards purely to
// have something to settle against. Two consequences, both measured. Planner
// spend was never gated — a 5000-token decomposition ran against a 1000-token
// budget — and when the after-the-fact reserve was refused because a ceiling had
// just fired, the cost was logged and dropped: 920 tokens spent, 460 recorded.
//
// Planner cost is charged to the planner role, so §14.3's breakdown can show
// what planning cost — the number that says whether the digest is doing its job.
func (e *Executor) planWithBudget(ctx context.Context, sess *core.Session, call func() (*planner.Plan, error)) (*planner.Plan, error) {
	est := budget.PlannerSeed(sess.BudgetUnit)
	if avail := sess.Available(); est > avail {
		est = avail
	}
	if est <= 0 {
		return nil, fmt.Errorf("%w: nothing left to plan with", budget.ErrInsufficientBudget)
	}

	reservation, err := e.Ledger.Reserve(ctx, sess.ID, est)
	if err != nil {
		return nil, err
	}

	plan, callErr := call()

	// Settle unconditionally: the tokens were spent whether or not the plan
	// parsed, and a reservation left held is budget neither spent nor available.
	var calls []core.ToolCall
	if plan != nil && !plan.Usage.IsZero() {
		tc := core.ToolCall{
			SessionID: sess.ID,
			Role:      core.RolePlanner,
			Type:      core.CallLLM,
			Model:     plan.Model,
			Input:     "plan",
			Cost: core.Cost{
				InputTokens:      plan.Usage.InputTokens,
				OutputTokens:     plan.Usage.OutputTokens,
				CacheReadTokens:  plan.Usage.CacheReadTokens,
				CacheWriteTokens: plan.Usage.CacheWriteTokens,
			},
			DurationMS: plan.Elapsed.Milliseconds(),
		}
		if callErr != nil {
			tc.Err = callErr.Error()
		}
		calls = append(calls, tc)
	}
	if _, err := e.Ledger.Settle(context.WithoutCancel(ctx), reservation, calls); err != nil {
		e.logger().ErrorContext(ctx, "settling a planner call failed; accounting is now unreliable", "err", err)
	}
	return plan, callErr
}

// heartbeat renews a lease in the background until the returned function is
// called.
//
// Renews at a third of the TTL, so two consecutive failures still leave time to
// recover before the lease expires.
func (e *Executor) heartbeat(ctx context.Context, lease *queue.Lease) func() {
	interval := e.Queue.TTL() / 3
	if interval <= 0 {
		return func() {}
	}

	done := make(chan struct{})
	stopped := make(chan struct{})
	go func() {
		defer close(stopped)
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-ctx.Done():
				return
			case <-t.C:
				// A lost lease is not recoverable from here: another worker may
				// hold the lead. Log and stop renewing; the run's own checks
				// surface it.
				if ok, err := e.Queue.Renew(context.WithoutCancel(ctx), lease); err != nil || !ok {
					e.logger().WarnContext(ctx, "lease renewal failed mid-lead",
						"lead", lease.Lead.ID, "ok", ok, "err", err)
					return
				}
			}
		}
	}()

	return func() {
		close(done)
		<-stopped
	}
}

// subBudget derives the actor's per-lead ceiling from the reserved amount.
//
// In token mode the reservation IS a token count, so the mapping is exact. In
// USD mode it has to be converted, and the rate comes from the cheap tier —
// chunk mining is where an actor's input tokens actually go, and pricing the
// ceiling off the strong model would set it several times too low.
//
// inputShare leaves room for output tokens and for EstimateTokens running
// short. Overshooting is a hard failure; underspending is a smaller run.
func (e *Executor) subBudget(sess *core.Session, reserved int64) actors.Budget {
	const inputShare = 70 // percent of the reservation available for input

	tokens := reserved
	if sess.BudgetUnit == core.BudgetUSD {
		tokens = e.tokensForMicros(reserved)
	}
	tokens = tokens * inputShare / 100
	if tokens <= 0 {
		// Nothing usable to derive from — leave the configured budget alone
		// rather than pinning the actor to zero and guaranteeing an empty run.
		return actors.Budget{}
	}
	return actors.Budget{MaxInputTokens: tokens}
}

// tokensForMicros converts a USD amount into an input-token count.
func (e *Executor) tokensForMicros(micros int64) int64 {
	if e.CheapModel == "" {
		return 0
	}
	table := e.Pricing
	if table == nil {
		table = pricing.NewTable()
	}
	rate, ok := table.Lookup(e.CheapModel)
	if !ok || rate.Input <= 0 {
		// No price for this model, so USD cannot bound it — the CLI already
		// refuses --usd in that case (checkUSDIsEnforceable). Returning zero
		// leaves the configured ceiling in place rather than inventing a rate.
		return 0
	}
	// Rates are nano-dollars per token; micros are 1000 nano.
	return micros * pricing.NanoPerMicro / rate.Input
}

// complete terminalizes a lead, surviving cancellation.
//
// On the same context as the run, a cancelled session settled the charge (Settle
// uses WithoutCancel) and then failed to complete the lead — leaving a lead that
// ran and was paid for recorded as still leased, for a later sweep to requeue
// and re-run.
func (e *Executor) complete(ctx context.Context, lease *queue.Lease, status core.LeadStatus) {
	if err := e.Queue.Complete(context.WithoutCancel(ctx), lease, status); err != nil {
		e.logger().ErrorContext(ctx, "could not complete a lead; it may be re-run and re-charged",
			"lead", lease.Lead.ID, "status", status, "err", err)
	}
}

// warmEstimator seeds the reservation estimates from history.
//
// Best effort by design: a failed read leaves the seeds, which are honestly
// conservative, and a session that could not read its own history has larger
// problems that the next query will surface properly.
func (e *Executor) warmEstimator(ctx context.Context, sess *core.Session) {
	var samples []core.LeadCost
	if err := e.Store.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		samples, err = q.RecentLeadCosts(ctx, budget.WarmLimit)
		return err
	}); err != nil {
		e.logger().WarnContext(ctx, "could not warm the cost estimator from history; "+
			"using cold-start seeds", "err", err)
		return
	}
	if n := e.Estimator.Warm(samples); n > 0 {
		e.logger().InfoContext(ctx, "cost estimator warmed from past leads",
			"samples", n, "unit", sess.BudgetUnit)
	}
}

func (e *Executor) estimate(sess *core.Session, lead core.Lead) int64 {
	if e.Estimator == nil {
		e.Estimator = budget.NewEstimator(sess.BudgetUnit)
	}
	return e.Estimator.For(lead.ActorType, lead.Depth)
}

func (e *Executor) owner() string {
	if e.Owner != "" {
		return e.Owner
	}
	return "executor"
}

func (e *Executor) session(ctx context.Context, id string) (*core.Session, error) {
	var s *core.Session
	err := e.Store.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		s, err = q.GetSession(ctx, id)
		return err
	})
	return s, err
}

// ceilingReason pulls the ceiling name out of the ledger's error, so the
// report says "max_leads" rather than quoting a sentence at the user.
func ceilingReason(err error) string {
	msg := err.Error()
	for _, name := range []string{"max_tool_calls", "max_leads", "max_wallclock"} {
		if strings.Contains(msg, name) {
			return name
		}
	}
	return "budget exhausted"
}

func truncateQuery(q string) string {
	q = strings.Join(strings.Fields(q), " ")
	if len(q) <= 52 {
		return q
	}
	return q[:51] + "…"
}

func leadNote(o leadOutcome) string {
	if o.err == nil {
		return ""
	}
	return ", " + o.class.String() + ": " + truncateQuery(o.err.Error())
}

func statusForContext(err error) core.SessionStatus {
	if errors.Is(err, context.DeadlineExceeded) {
		return core.StatusExhausted
	}
	return core.StatusCancelled
}

// verify runs one incremental verification pass (§11.1) and folds the result into
// the loop.
//
// Never fails the session. Verification improves an answer; it does not produce
// one, so a provider that will not compare claims degrades the run rather than
// ending it — the claims still carry quotes checked verbatim against their source.
//
// Runs on the replan cadence rather than per lead. Two reasons, one of each kind.
// Batches fill better: a single lead's two or three claims produce a handful of
// pairs and a mostly-empty call, while several leads' claims fill one. And the
// planner needs the answer: this is the only point in the loop where a
// contradiction can still change what gets researched next.
func (e *Executor) verify(ctx context.Context, digest *planner.Digest, res *Result) {
	if e.Verifier == nil {
		return
	}

	e.emit("verifying", fmt.Sprintf("%d claim(s) gathered", digest.ClaimsFound))
	out, err := e.Verifier.Run(ctx, res.SessionID)
	if err != nil {
		// A store or ledger failure. Worth a warning and nothing more: the research
		// is still valid and the report can still be written.
		e.logger().WarnContext(ctx, "verification pass failed; continuing unverified", "err", err)
		return
	}
	if out == nil {
		return
	}

	res.VerifyPasses++
	res.ClaimsVerified += out.ClaimsVerified
	res.EdgesWritten += out.EdgesWritten
	res.Contradictions += out.Contradictions
	res.Rechecked += out.Rechecked
	res.Withheld += out.Withheld
	if out.Degraded != "" {
		res.VerifyDegraded = out.Degraded
	}

	// The planner sees a COUNT, never the claims (§9.1). That keeps page-derived
	// text out of the planner entirely, which is stronger than fencing it.
	digest.Contradictions = res.Contradictions

	if out.ClaimsVerified > 0 || out.EdgesWritten > 0 {
		e.emit("verified", fmt.Sprintf("%d claim(s), %d edge(s), %d contradiction(s)",
			out.ClaimsVerified, out.EdgesWritten, out.Contradictions))
	}

	e.queueFollowUps(ctx, out.FollowUps, res)
	// What the pass could not take. Persisted deferred: not eligible to lease,
	// and not lost either — which is what used to happen to every contradiction
	// past the per-pass cap.
	e.deferFollowUps(ctx, out.Deferred, res)
}

// recheck re-judges the session's contradictions with the Verifier's
// RecheckModel, when one is set.
func (e *Executor) recheck(ctx context.Context, res *Result) {
	if e.Verifier == nil || e.Verifier.RecheckModel == "" {
		return
	}
	out, err := e.Verifier.Recheck(ctx, res.SessionID)
	if err != nil {
		e.logger().WarnContext(ctx, "re-check failed; keeping contradictions as judged", "err", err)
		return
	}
	res.Rechecked += out.Rechecked
	res.Withheld += out.Withheld
	res.Contradictions += out.Contradictions
	if out.Degraded != "" {
		res.VerifyDegraded = out.Degraded
	}
	if out.Rechecked > 0 {
		e.emit("rechecked", fmt.Sprintf("%d contradiction(s), %d withheld", out.Rechecked, out.Withheld))
	}
}

// deferFollowUps records work the pass judged worth doing and could not take.
//
// Deferred rather than queued, so the running session's spend is still paced by
// MaxTotal — and recorded rather than dropped, so the run cannot end believing
// there is nothing left while contradictions sit unexamined.
func (e *Executor) deferFollowUps(ctx context.Context, ups []verifier.FollowUp, res *Result) {
	if len(ups) == 0 {
		return
	}
	leads := make([]core.Lead, 0, len(ups))
	for _, up := range ups {
		lead := up.Lead
		if lead.ID == "" {
			lead.ID = core.NewLeadID()
		}
		lead.Deferred = true
		leads = append(leads, lead)
	}
	if err := e.Queue.Push(ctx, leads); err != nil {
		// Same reasoning as a failed queue push: the contradiction is already in
		// the graph and will be reported. Failing to record the follow-up is a
		// worse answer, not a broken one.
		e.logger().WarnContext(ctx, "could not defer follow-up leads", "err", err)
		return
	}
	res.FollowUpsDeferred += len(leads)
	e.logger().InfoContext(ctx, "deferred follow-up leads past this pass's budget",
		"leads", len(leads))
}

// queueFollowUps pushes the Verifier's proposed leads (§11.4).
//
// Queued here rather than by the Verifier because the executor owns the queue —
// and because a Verifier that wrote to it could not be replayed against a cassette
// or inspected before its work ran.
//
// The depth and per-root caps were applied upstream. This adds nothing to them: a
// second ceiling here would be a place for the two to disagree.
func (e *Executor) queueFollowUps(ctx context.Context, ups []verifier.FollowUp, res *Result) {
	if len(ups) == 0 {
		return
	}
	leads := make([]core.Lead, 0, len(ups))
	for _, up := range ups {
		lead := up.Lead
		if lead.ID == "" {
			lead.ID = core.NewLeadID()
		}
		leads = append(leads, lead)
		e.logger().InfoContext(ctx, "queuing a follow-up lead to settle a contradiction",
			"lead", lead.ID, "depth", lead.VerifyDepth, "because", up.Because)
	}

	if err := e.Queue.Push(ctx, leads); err != nil {
		// Not fatal. The contradiction is already recorded in the graph and will be
		// reported as a contradiction; failing to research it further is a worse
		// answer, not a broken one.
		e.logger().WarnContext(ctx, "could not queue follow-up leads", "err", err)
		return
	}
	res.FollowUpsQueued += len(leads)
	e.emit("follow-up", fmt.Sprintf("%d lead(s) to settle a contradiction", len(leads)))
}

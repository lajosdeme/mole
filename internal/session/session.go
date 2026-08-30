// Package session runs one research session end to end.
//
// Everything from creating the session row to finalizing its status: boot
// recovery, the planner–executor–verifier loop, §11.5.2's grounding pass, and
// the report paid from released escrow (§8.3).
//
// It exists because two callers need that sequence and only one of them has a
// terminal. `mole research` had all of it inline, interleaved with flag handling
// and fmt.Printf, which made the daemon's options "duplicate 240 lines" or
// "import a CLI". Nothing here prints, opens a database, reads config, or knows
// what a flag is — callers inject what they built and receive what happened.
//
// The split into Recover / Create / Run is not cosmetic. Recovery is per
// PROCESS, so a daemon sweeps once at boot rather than once per session. And a
// daemon must return a session id to its caller before the work finishes, so
// creating the row and running the loop cannot be one call.
package session

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/lajosdeme/mole/internal/actors"
	"github.com/lajosdeme/mole/internal/budget"
	"github.com/lajosdeme/mole/internal/cache"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/dataset"
	"github.com/lajosdeme/mole/internal/executor"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/output"
	"github.com/lajosdeme/mole/internal/planner"
	"github.com/lajosdeme/mole/internal/queue"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/verifier"
)

// Spec is one research request: what to ask and what it may spend.
//
// Separate from Runner because a Runner's collaborators outlive any single
// session, while these values arrive with the request.
type Spec struct {
	Question   string
	Mode       core.Mode
	BudgetUnit core.BudgetUnit
	Budget     int64

	// MaxSources is sources read per lead; MaxDepth is rounds of follow-up leads
	// the planner may add; MaxLeads bounds the whole session (§8.5).
	MaxSources int
	MaxDepth   int
	MaxLeads   int

	// Timeout is the session's wall-clock ceiling (§8.5). The caller owns the
	// context deadline; this is the ceiling the loop checks itself against, and
	// the two should not be equal — see Runner.Run.
	Timeout time.Duration

	// Schema is the dataset schema, required when Mode is ModeDataset and unused
	// otherwise (M9, §13).
	Schema *dataset.Schema

	// ActorTypes are the actors this session may use. Empty means web only.
	//
	// Session-level rather than planner-chosen: the actor has to be provably
	// working before the planner starts making choices about it, and a session
	// that never asked for academic sources should never pay for one.
	ActorTypes []core.ActorType

	// Workers is how many leads this session runs at once. Zero takes
	// executor.DefaultWorkers. Not persisted on the session row: it is a
	// property of the process doing the work, not of the research, and a
	// recovered session is re-run by whatever is configured then.
	Workers int
}

// Result is what happened, in the shape a caller needs to render or return.
type Result struct {
	Session *core.Session

	// Run is the loop's own result. Nil when the loop never started.
	Run *executor.Result
	// Report is the synthesized answer. Nil when generation failed — never fatal,
	// since a failed synthesis costs the prose and not the evidence.
	Report *output.Report
	// Ground is §11.5.2's pass. Nil when it did not run.
	Ground *verifier.GroundReport

	// Status is what the session was finalized as.
	Status core.SessionStatus
	// Err is the loop's error. Returned in the struct rather than as the
	// function's error because a failed loop still has a session, a spend, and
	// usually claims — a caller that gets only an error has to go and find them.
	Err error
}

// Runner holds the collaborators a session needs.
//
// The store and actor are built by the caller: the CLI from flags and config,
// the daemon from its own configuration. Neither is constructed here, because
// choosing a search provider or a database path is not this package's decision.
type Runner struct {
	Store store.Store
	Actor *actors.WebActor

	// VerifierModel overrides the cheap model for adjudication. Empty leaves the
	// cheap model in place; set it when the cheap model cannot tell a
	// contradiction from two unrelated statements.
	VerifierModel string
	// VerifierBatchSize caps pairs per adjudication call. Zero takes the default.
	VerifierBatchSize int

	// Confirm chooses which edge kinds need a second agreeing judgement. The zero
	// value confirms both, the measured default — the opposite of a bare Verifier's,
	// which confirms nothing. See confirmMode.
	Confirm verifier.ConfirmMode

	// Academic researches scholarly leads (§10.2). Nil disables the actor.
	//
	// Nil in two cases, not one: an install with no contact email (§10.3 makes
	// that a hard requirement), and every daemon — `mole serve` does not build
	// it, so the academic path is CLI-only today.
	Academic *actors.AcademicActor

	// Local answers leads from the user's own registered data (§12). Nil
	// disables the actor, which is the state of any install where nobody has
	// run `mole connect add`.
	Local *actors.LocalComputeActor

	// Owner names what is running the loop, for lead leases (§9.4). A daemon and
	// a CLI must not claim each other's leads.
	Owner string

	Log *slog.Logger

	// Progress receives executor events as they happen.
	Progress func(executor.Event)
	// OnLoop fires when the research loop finishes, before any escrow is released.
	// OnGround fires when the grounding pass finishes, before the report is
	// generated.
	//
	// Callbacks rather than fields on Result because the sequence is observable
	// and Result is not: the CLI prints the run summary, then grounding, then the
	// report, in that order. Returning all three at the end collapses that into
	// one moment and the caller has to guess the order — which is exactly what
	// went wrong when this package was first extracted, and the grounding line
	// started printing ahead of the summary it follows.
	//
	// A daemon uses the same two points to publish status transitions without
	// waiting for the report.
	OnLoop   func(*executor.Result)
	OnGround func(*verifier.GroundReport)
	// Notice receives non-fatal problems: a sweep that failed, a hold that could
	// not be released, a report that would not generate. None of these stops a
	// run, and all of them need saying. The CLI writes them to stderr; a daemon
	// logs them.
	Notice func(string)
}

func (r *Runner) notice(format string, args ...any) {
	if r.Notice != nil {
		r.Notice(fmt.Sprintf(format, args...))
	}
}

// confirmMode defaults a session to confirming both edge kinds. Opt-out here and
// opt-in at the Verifier, because that is what each layer's zero value meant before
// a mode replaced the bool.
func (r *Runner) confirmMode() verifier.ConfirmMode {
	if r.Confirm == "" {
		return verifier.ConfirmBoth
	}
	return r.Confirm
}

func (r *Runner) logger() *slog.Logger {
	if r.Log != nil {
		return r.Log
	}
	if r.Actor != nil && r.Actor.Log != nil {
		return r.Actor.Log
	}
	return slog.Default()
}

func (r *Runner) ledger() *budget.Ledger {
	return budget.New(r.Store, budget.DefaultConfig())
}

// Recovery reports what boot recovery reclaimed.
type Recovery struct {
	Leads        int
	Reservations int
	Sessions     int
	// Documents is stored source text deleted for being past its TTL. Reclaimed
	// disk rather than recovered state, and reported for the same reason: a
	// number nobody prints is a sweep nobody notices has stopped running.
	Documents int64
}

// Any reports whether anything was recovered.
func (rc Recovery) Any() bool {
	return rc.Leads+rc.Reservations+rc.Sessions > 0 || rc.Documents > 0
}

// Recover reclaims what a previous process left behind (§9.4).
//
// Per PROCESS, not per session — which is why it is not inside Run. It is
// deliberately unscoped: boot recovery does not know which sessions were in
// flight, and this is the one caller for which that is correct. A daemon calls
// it once at startup; running it per session would sweep leases belonging to
// sessions currently running alongside.
//
// Never returns an error. Every sweep is best-effort recovery of someone else's
// mess, and refusing to start because a sweep failed helps nobody.
func (r *Runner) Recover(ctx context.Context) Recovery {
	var rc Recovery
	led := r.ledger()

	if n, err := queue.New(r.Store, 0).Sweep(ctx, ""); err != nil {
		r.notice("lease sweep failed: %v", err)
	} else {
		rc.Leads = n
	}
	// A stale hold is budget neither spent nor available. SweepExpired had no
	// caller at all for a while, so a crash mid-lead understated a session's
	// Available() permanently.
	if n, err := led.SweepExpired(ctx); err != nil {
		r.notice("reservation sweep failed: %v", err)
	} else {
		rc.Reservations = n
	}
	if n, err := led.SweepAbandonedSessions(ctx); err != nil {
		r.notice("session sweep failed: %v", err)
	} else {
		rc.Sessions = n
	}
	// Stored source text past its TTL. The reads already refuse it — a quote
	// cannot be checked against an expired document — but nothing deleted the
	// rows, so a machine that ran toolkit mode kept every page it had ever read,
	// at up to 120,000 characters each, permanently. The migration promised two
	// mechanisms and shipped with neither firing.
	if n, err := r.PurgeDocuments(ctx); err != nil {
		r.notice("document sweep failed: %v", err)
	} else {
		rc.Documents = n
	}
	return rc
}

// PurgeDocuments deletes stored source text past its retention TTL and reports
// how many rows went.
//
// Exported so a long-running daemon can repeat it: boot recovery is not enough
// for a process that stays up for weeks, which is the process this table fills
// up fastest under.
func (r *Runner) PurgeDocuments(ctx context.Context) (int64, error) {
	var n int64
	err := r.Store.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		var err error
		n, err = tx.PurgeExpiredDocuments(ctx, time.Now().UTC())
		return err
	})
	return n, err
}

// Create writes the session row and returns it.
//
// Separate from Run so a daemon can hand a session id back to its caller before
// any research happens. Nothing is spent here beyond the escrow reserved at
// creation (§8.3).
func (r *Runner) Create(ctx context.Context, spec Spec) (*core.Session, error) {
	if spec.Question == "" {
		return nil, errors.New("session: no question given")
	}
	if !spec.Mode.Valid() {
		return nil, fmt.Errorf("session: unknown mode %q", spec.Mode)
	}

	maxLeads := spec.MaxLeads
	if maxLeads <= 0 {
		maxLeads = 1
	}
	sources := spec.MaxSources
	if sources <= 0 {
		sources = 1
	}

	sess, err := r.ledger().CreateSession(ctx, budget.SessionSpec{
		Prompt:     spec.Question,
		Mode:       spec.Mode,
		ActorTypes: actorTypesOf(spec),
		BudgetUnit: spec.BudgetUnit,
		Budget:     spec.Budget,
		// Unit-independent ceilings (§8.5). They bind even when the money estimate
		// is wrong, which is the case they exist for. Sized from what the planner
		// can actually produce: an initial fan-out plus one round of follow-ups per
		// depth level, with headroom.
		MaxToolCalls: int64(maxLeads) * int64(sources) * 4,
		MaxLeads:     int64(maxLeads),
		MaxWallClock: spec.Timeout,
	})
	if err != nil {
		return nil, err
	}

	// The dataset schema is written NOW, not at the end of the run.
	//
	// Rows are persisted per lead on an uncancellable context specifically so a
	// killed session keeps what it paid for — and the schema was written only in
	// finishDataset, so killing the process after the first lead left rows nobody
	// could read: `mole dataset` refuses a session with no stored schema, and its
	// message tells the user the wrong thing ("not a dataset session"). The
	// uncancellable write was defeated by a missing four hundred bytes.
	//
	// It is safe to write here: the schema is validated before the run starts and
	// is the one part of a session that cannot change while it runs.
	if spec.Schema != nil {
		if err := r.Store.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
			return tx.SetDatasetSchema(ctx, sess.ID, *spec.Schema)
		}); err != nil {
			return nil, fmt.Errorf("session: store the dataset schema: %w", err)
		}
	}
	return sess, nil
}

// Run drives the loop for an already-created session, then writes the report and
// finalizes the status.
//
// The caller owns the context. Give it headroom over spec.Timeout: when the
// context deadline and the wall-clock ceiling fire together the loop is killed
// mid-lead instead of stopping at its own check, and the run ends in a cascade
// of "context deadline exceeded" from whatever was in flight — a persist, a
// settle, a count — rather than a clean "stopped: max_wallclock".
//
// A loop error is returned inside Result, not as the error, so a caller still
// gets the session, the spend and the claims. The error return is for failures
// that leave nothing worth reporting.
func (r *Runner) Run(ctx context.Context, sess *core.Session, spec Spec) (*Result, error) {
	if sess == nil {
		return nil, errors.New("session: nil session")
	}
	if r.Store == nil || r.Actor == nil {
		return nil, errors.New("session: runner needs a store and an actor")
	}

	res := &Result{Session: sess, Status: core.StatusDone}
	led := r.ledger()

	// A COPY of the actor, per session.
	//
	// SessionID is what every claim and fetch outcome is filed under, so mutating
	// the Runner's own actor would be wrong the moment two sessions run at once —
	// and the supervisor exists to make that the normal case. One session's claims
	// would be attributed to another's id, which no amount of budget accounting
	// would catch, and the fetch caches would be shared across unrelated research.
	//
	// Shallow is right: the search, fetch, extract and model clients are stateless
	// and safe to share, and copying them is what keeps connection pools and rate
	// limiters global. Only the per-session fields are replaced.
	actor := *r.Actor
	actor.SessionID = sess.ID
	// Dataset mode: the actor extracts rows instead of claims (M9, §13). Built
	// here rather than by the caller because it needs the session id, exactly as
	// the actor's own copy does.
	if spec.Schema != nil {
		actor.Rows = &actors.RowMiner{
			LLM:       actor.LLM,
			Pricing:   actor.Pricing,
			Log:       actor.Log,
			SessionID: sess.ID,
			Schema:    *spec.Schema,
		}
	}
	actor.Store = r.Store

	// One cache shared between the loop and this actor, so a lead-level hit and a
	// URL-level hit are the same cache and one lead's fetches serve another's.
	shared := cache.New()
	actor.Cache = shared

	// The Verifier shares the actor's provider, store and fetcher. §11.1 wants it
	// over the session's whole claim set, which is why it reads the store rather
	// than being handed each lead's batch — and the fetcher is the actor's own, so
	// §11.5's re-read goes through the same robots handling, rate limiter and SSRF
	// guard as the fetch that produced the claim.
	vf := &verifier.Verifier{
		Store:     r.Store,
		Ledger:    led,
		LLM:       actor.LLM,
		Log:       r.logger(),
		Model:     r.VerifierModel,
		BatchSize: r.VerifierBatchSize,
		// Both by default: one judgement gets "contradicts" right 51% of the time,
		// and here a false edge also spends a research lead. Output a person reads
		// wants ConfirmDuplicates — 98% recall against 67%.
		Confirm:  r.confirmMode(),
		Grounder: &verifier.Grounder{Fetch: actor.Fetch, Extract: actor.Extract},
	}

	owner := r.Owner
	if owner == "" {
		owner = "session"
	}
	exec := &executor.Executor{
		Store:      r.Store,
		Ledger:     led,
		Queue:      queue.New(r.Store, 0),
		Cache:      shared,
		Pricing:    actor.Pricing,
		CheapModel: actor.LLM.ModelFor(llm.TierCheap),
		Planner:    &planner.Planner{LLM: actor.LLM, MaxDepth: spec.MaxDepth},
		Verifier:   vf,
		Actors:     r.actorsFor(spec, &actor),
		Log:        r.logger(),
		Owner:      owner,
		Progress:   r.Progress,
		Workers:    spec.Workers,
	}

	res.Run, res.Err = exec.Run(ctx, sess.ID)
	if r.OnLoop != nil && res.Run != nil {
		r.OnLoop(res.Run)
	}

	// The report is paid from escrow (§8.3). Releasing it here is what makes the
	// money set aside at session creation spendable — a run that produced good
	// claims and could not afford to write them up would have wasted the whole
	// budget, not just the last call.
	res.Report, res.Ground = r.finish(ctx, led, vf, sess, &actor, spec.Schema)

	if res.Run != nil {
		res.Status = res.Run.Status
	}
	if res.Err != nil && res.Status == core.StatusDone {
		res.Status = core.StatusFailed
	}

	// Persist the answer before finalizing.
	//
	// It used to live only in memory: the CLI printed it and the daemon paid for
	// it out of escrow and dropped it, so §5.1's research.result — the step where
	// an MCP caller receives the answer — could only ever return an empty string.
	if res.Report != nil {
		// Only the closed-set reason is persisted — it is what research.result
		// hands an MCP caller. The detail is logged instead, because it can carry
		// a provider endpoint or rejected model output and neither belongs in
		// another agent's context. See output.Report.Degraded.
		if d := res.Report.DegradedDetail; d != "" {
			r.logger().Warn("report degraded",
				"session", sess.ID, "reason", res.Report.Degraded, "detail", d)
		}
		if err := r.Store.WithTx(context.WithoutCancel(ctx), func(ctx context.Context, tx store.Tx) error {
			if err := tx.SetSessionReport(ctx, sess.ID, res.Report.Markdown(), res.Report.Degraded); err != nil {
				return err
			}
			// Written WITH the report, because it cannot be written later. The
			// [n] markers in the prose are numbered over a findings set capped by
			// the report's token budget, and that cap does not outlive this call.
			cited := res.Report.Citations
			if cited == nil {
				cited = []output.Citation{}
			}
			cites, err := json.Marshal(cited)
			if err != nil {
				return err
			}
			return tx.SetSessionReportCitations(ctx, sess.ID, string(cites))
		}); err != nil {
			r.notice("could not store the report: %v", err)
		}
	}
	if err := led.Finish(context.WithoutCancel(ctx), sess.ID, res.Status); err != nil {
		r.notice("could not finalize session: %v", err)
	}
	return res, nil
}

// finish releases escrow, grounds, and writes the report.
//
// Runs on a context detached from the caller's. Everything here is the payoff
// for money already spent, and a cancelled research context must not also
// discard the answer it paid for.
func (r *Runner) finish(
	ctx context.Context,
	led *budget.Ledger,
	vf *verifier.Verifier,
	sess *core.Session,
	actor *actors.WebActor,
	schema *dataset.Schema,
) (*output.Report, *verifier.GroundReport) {
	// Whether the run was stopped, read BEFORE detaching. Everything below runs
	// on a live context by design — the payoff for money already spent must not
	// be discarded because research was cancelled — which also means ctx.Err()
	// stops being readable one line down.
	cancelled := ctx.Err() != nil
	ctx = context.WithoutCancel(ctx)

	released, err := led.ReleaseEscrow(ctx, sess.ID)
	if err != nil {
		r.notice("could not release escrow: %v", err)
	}

	// §11.5.2's grounding pass, between the escrow release and the report.
	//
	// Here rather than inside the loop for two reasons. Its candidates are the
	// claims the REPORT will lean on, which is only knowable once research has
	// stopped and the graph is complete. And §11.5 puts its spend on the escrow —
	// exactly the money just released, so it takes a bounded share and leaves the
	// rest for the answer. A grounding pass that spent the escrow would produce a
	// well-checked set of claims and no report to put them in.
	var ground *verifier.GroundReport
	if !cancelled {
		ground = r.ground(ctx, vf, sess.ID, released)
	}
	if r.OnGround != nil && ground != nil {
		r.OnGround(ground)
	}

	// Reserve BEFORE generating. The order used to be release → generate →
	// reserve, which inverts §8.2 and had a concrete failure: after any research
	// overshoot the reserve was refused, and the report tokens — already spent —
	// were never written to the ledger. §8.1 says every tool call writes a cost
	// row, and Verify() still reconciled because the row never existed.
	//
	// Bounded by what is actually available, not by what escrow nominally
	// released: an overshoot may already have eaten into it.
	amount := released
	if ground != nil {
		amount -= ground.Spent
	}
	if cur, err := r.loadSession(ctx, sess.ID); err == nil {
		if avail := cur.Available(); amount > avail {
			amount = avail
		}
	}

	// Dataset mode produces the dataset instead of prose, and produces it without
	// a model call: the merge is deterministic arithmetic over rows that were
	// already extracted and paid for. So there is nothing to reserve here, which
	// is why this sits before the reservation rather than beside the generator —
	// reserving output tokens for a step that spends none would leave a hold to
	// settle at zero and report a cost that does not exist.
	if schema != nil {
		return r.finishDataset(ctx, sess.ID, *schema), ground
	}

	gen := &output.Generator{LLM: actor.LLM}
	var reservation *core.Reservation
	if cancelled {
		// Cancel means stop spending. Escrow is still released and every hold
		// still settles — leaving budget held would be the worst of both — but no
		// model call is made, so the session emits the evidence it already paid
		// to collect and no prose. Reuses the same path as "nothing to spend".
		amount = 0
		r.notice("session cancelled: writing the evidence without a synthesized answer")
	}
	if amount > 0 {
		reservation, err = led.ReserveOutput(ctx, sess.ID, amount)
		if err != nil {
			r.notice("could not reserve for the report: %v", err)
		}
	}
	if reservation == nil {
		// Nothing to spend. Emit the evidence without prose rather than making a
		// call whose cost cannot be recorded.
		gen.LLM = nil
	}

	report, err := gen.Generate(ctx, r.Store, sess.ID)
	if err != nil {
		r.notice("report generation failed: %v", err)
		if reservation != nil {
			if rerr := led.Release(ctx, reservation); rerr != nil {
				r.notice("could not release the report hold: %v", rerr)
			}
		}
		return nil, ground
	}

	if reservation != nil {
		// Settle unconditionally, including a zero cost: an unresolved hold is
		// budget neither spent nor available.
		var calls []core.ToolCall
		if !report.Cost.IsZero() {
			calls = append(calls, core.ToolCall{
				SessionID: sess.ID,
				Role:      core.RoleOutput,
				Type:      core.CallLLM,
				Model:     report.Model,
				Input:     "report",
				Cost:      output.Price(actor.Pricing, report),
			})
		}
		if _, serr := led.Settle(ctx, reservation, calls); serr != nil {
			r.notice("settling the report failed: %v", serr)
		}
	}
	return report, ground
}

// finishDataset assembles the dataset from the rows the session extracted.
//
// From the STORE rather than from the returned Results, for the same reason the
// report is: a lead's rows are persisted on an uncancellable context precisely so
// that a cancelled session still emits what it paid for, and reading them back is
// what makes that true.
//
// A failure here is a notice rather than an error. The rows are already stored, so
// `mole dataset` can assemble them later; losing the session because the summary
// could not be written would discard nothing but be reported as everything.
func (r *Runner) finishDataset(ctx context.Context, sessionID string, schema dataset.Schema) *output.Report {
	d, err := store.MergeStoredRows(ctx, r.Store, sessionID, schema, dataset.Options{})
	if err != nil {
		r.notice("could not read the extracted rows: %v", err)
		return nil
	}
	markdown := dataset.Markdown(d)

	var degraded string
	if len(d.Rows) == 0 {
		degraded = "no row survived extraction: every candidate either failed the " +
			"quote check or filled no key field"
	}

	if err := r.Store.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		if err := tx.SetDatasetSchema(ctx, sessionID, schema); err != nil {
			return err
		}
		return tx.SetSessionReport(ctx, sessionID, markdown, degraded)
	}); err != nil {
		r.notice("could not store the dataset: %v", err)
	}

	// Degraded travels on the RETURNED report, not only into the row written
	// above: Run re-writes the same row from res.Report afterwards, so a reason
	// set only here was overwritten with an empty string and the session reported
	// a clean empty dataset. An MCP caller saw success.
	return &output.Report{SessionID: sessionID, Body: markdown, Degraded: degraded}
}

// ground runs §11.5.2's re-read within a bounded share of the released escrow.
func (r *Runner) ground(
	ctx context.Context,
	vf *verifier.Verifier,
	sessionID string,
	releasedEscrow int64,
) *verifier.GroundReport {
	if vf == nil || vf.Grounder == nil || releasedEscrow <= 0 {
		return nil
	}
	allowance := int64(float64(releasedEscrow) * verifier.DefaultGroundShareOfEscrow)
	if allowance <= 0 {
		return nil
	}
	rep, err := vf.Ground(ctx, sessionID, allowance)
	if err != nil {
		r.notice("grounding pass failed: %v", err)
		return nil
	}
	return rep
}

func (r *Runner) loadSession(ctx context.Context, id string) (*core.Session, error) {
	var sess *core.Session
	err := r.Store.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		sess, err = q.GetSession(ctx, id)
		return err
	})
	return sess, err
}

// actorTypesOf is what the session records it may use.
func actorTypesOf(spec Spec) []core.ActorType {
	if len(spec.ActorTypes) == 0 {
		return []core.ActorType{core.ActorWeb}
	}
	return spec.ActorTypes
}

// actorsFor registers the actors a session asked for and can actually run.
//
// An actor the caller did not build is absent rather than registered broken. Be
// clear about the consequence: the executor treats "no actor registered" as
// FATAL, which cancels the batch and ends the session — it is not a degraded
// lead. That is still the better failure, because a half-built actor would fail
// on every request after paying for the attempt, but it is a hard stop and the
// caller has to avoid asking for an actor it did not build. checkAcademicConfig
// is what makes that impossible from the CLI.
func (r *Runner) actorsFor(spec Spec, web *actors.WebActor) map[core.ActorType]actors.Actor {
	out := map[core.ActorType]actors.Actor{core.ActorWeb: web}
	for _, t := range spec.ActorTypes {
		if t == core.ActorLocalCompute && r.Local != nil {
			// Copied, and it did not used to be: LocalComputeActor takes the
			// session from the lead, so there was nothing per-session to set and
			// a copy would have imitated the shape of the Academic rule below
			// without its reason. It carries a Store now, for §12.1's audit
			// trail, and assigning that to a shared instance while another
			// session runs is a data race — so it is a copy for a reason rather
			// than for symmetry.
			local := *r.Local
			local.Store = r.Store
			out[core.ActorLocalCompute] = &local
			continue
		}
		if t == core.ActorAcademic && r.Academic != nil {
			// One per session, like the web actor: SessionID scopes the claims
			// it writes, so a shared instance would attribute one session's
			// claims to another.
			academic := *r.Academic
			academic.SessionID = web.SessionID
			academic.Store = r.Store
			out[core.ActorAcademic] = &academic
		}
	}
	return out
}

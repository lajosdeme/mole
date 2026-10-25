// Package budget enforces spend.
//
// The central idea is that budget is *reserved before dispatch and settled
// after*, not charged after the fact. Post-hoc charging lets a pool of N
// workers overshoot the ceiling by up to N lead-costs, because every worker
// passes the "can I afford one more?" check before any of them has paid.
// Holding first bounds the overshoot to estimate error on a single lead.
//
// A second rule: the append-only tool_calls ledger is the source of truth for
// spend. Session.Spent is a materialized sum written in the same transaction as
// each cost row, so a crash can never desynchronize the two — and Verify can
// always prove it.
package budget

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/pricing"
	"github.com/lajosdeme/mole/internal/store"
)

// ErrInsufficientBudget is returned by Reserve when the requested amount
// exceeds what is available. It is an expected control-flow signal, not a
// failure: the executor treats it as "stop taking new work".
var ErrInsufficientBudget = errors.New("budget: insufficient")

// ErrOvershoot means a settled cost cleared its reservation by more than
// Config.MaxOvershootFactor.
//
// Distinct from ErrInsufficientBudget because the two call for opposite
// responses: insufficiency is the ceiling working, and overshoot is the ceiling
// having failed to bind. Reporting the second as the first is how a broken
// sub-budget reads as a correctly bounded run.
var ErrOvershoot = errors.New("budget: overshoot past the allowed factor")

// Config tunes the ledger.
type Config struct {
	// EscrowFraction is the share of the budget held back at session start for
	// output generation and the final verification pass. Report generation is
	// one LLM pass over the whole claim graph — plausibly the most expensive
	// single call in the session — so without escrow a session can spend 100%
	// on research and have nothing left to write the answer.
	//
	// 0.15 is a starting guess, to be calibrated from real ledgers.
	EscrowFraction float64

	// MinEscrow is a floor in the session's budget unit, applied when a
	// percentage of a small budget would not cover one report.
	MinEscrow int64

	// ReservationTTL bounds how long a hold survives without being settled.
	// The recovery sweep releases anything older; without it, a worker that
	// died mid-run holds budget forever.
	ReservationTTL time.Duration

	// AbandonedSessionIdle is how long a running session may go untouched
	// before boot recovery presumes its process is gone (§9.4).
	//
	// Compared against the row's updated_at, which the store writes from the
	// real clock — so this deliberately does NOT go through the injectable
	// clock. Mixing the two was the first attempt and it made a freshly created
	// session look stale under a fast-forwarded test clock, because the row
	// carried a real timestamp and the cutoff did not.
	AbandonedSessionIdle time.Duration

	// Log is where an unpriced model is reported. Nil uses the default logger.
	Log *slog.Logger

	// Pricing converts a call's tokens into money. Nil means the built-in table.
	//
	// It lives on the LEDGER rather than on each caller because the ledger is
	// the one place every tool call passes through, and leaving the conversion
	// to whoever built the call meant two of the four parts of mole that spend
	// money never did it. See Settle.
	Pricing *pricing.Table

	// MaxOvershootFactor caps how far actual cost may exceed the reservation
	// before Settle flags it. Exceeding it does not reject the settle — the
	// money was spent and the ledger records what happened — but it is the
	// signal that the estimator needs recalibrating.
	MaxOvershootFactor float64
}

func DefaultConfig() Config {
	return Config{
		EscrowFraction:       0.15,
		MinEscrow:            0,
		ReservationTTL:       15 * time.Minute,
		AbandonedSessionIdle: DefaultAbandonedSessionIdle,
		MaxOvershootFactor:   2.0,
	}
}

// Ledger owns all budget mutations.
type Ledger struct {
	st  store.Store
	cfg Config
	now func() time.Time
}

func New(st store.Store, cfg Config) *Ledger {
	if cfg.ReservationTTL <= 0 {
		cfg.ReservationTTL = DefaultConfig().ReservationTTL
	}
	if cfg.MaxOvershootFactor <= 0 {
		cfg.MaxOvershootFactor = DefaultConfig().MaxOvershootFactor
	}
	if cfg.Pricing == nil {
		cfg.Pricing = pricing.NewTable()
	}
	return &Ledger{st: st, cfg: cfg, now: func() time.Time { return time.Now().UTC() }}
}

// SetClock overrides time, for tests.
func (l *Ledger) SetClock(fn func() time.Time) { l.now = fn }

func (l *Ledger) Config() Config { return l.cfg }

func (l *Ledger) logger() *slog.Logger {
	if l.cfg.Log != nil {
		return l.cfg.Log
	}
	return slog.Default()
}

// ---------------------------------------------------------------------------
// Session creation
// ---------------------------------------------------------------------------

// SessionSpec is the input to CreateSession.
type SessionSpec struct {
	Prompt       string
	Mode         core.Mode
	ActorTypes   []core.ActorType
	BudgetUnit   core.BudgetUnit
	Budget       int64
	MaxToolCalls int64
	MaxLeads     int64
	MaxWallClock time.Duration
}

// CreateSession persists a session with its escrow already held back, so no
// code path can spend the output reserve by forgetting to call something.
func (l *Ledger) CreateSession(ctx context.Context, spec SessionSpec) (*core.Session, error) {
	s := &core.Session{
		ID:           core.NewSessionID(),
		Prompt:       spec.Prompt,
		Mode:         spec.Mode,
		ActorTypes:   spec.ActorTypes,
		BudgetUnit:   spec.BudgetUnit,
		Budget:       spec.Budget,
		Escrow:       l.escrowFor(spec.Budget),
		MaxToolCalls: spec.MaxToolCalls,
		MaxLeads:     spec.MaxLeads,
		MaxWallClock: spec.MaxWallClock,
		Status:       core.StatusRunning,
		CreatedAt:    l.now(),
	}
	if err := s.Validate(); err != nil {
		return nil, err
	}
	err := l.st.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.InsertSession(ctx, s)
	})
	if err != nil {
		return nil, err
	}
	return s, nil
}

func (l *Ledger) escrowFor(budget int64) int64 {
	e := int64(float64(budget) * l.cfg.EscrowFraction)
	if e < l.cfg.MinEscrow {
		e = l.cfg.MinEscrow
	}
	if e > budget {
		e = budget
	}
	if e < 0 {
		e = 0
	}
	return e
}

// ---------------------------------------------------------------------------
// Reserve / Settle / Release
// ---------------------------------------------------------------------------

// Reserve places a hold. It is the hard ceiling: the availability check and
// the hold happen in one transaction, so two concurrent workers cannot both
// pass a check that only one of them can afford.
//
// Rev-1's `Remaining() > floorCost(leads.Peek())` followed by `leads.Pop()` was
// also a TOCTOU bug — the peeked lead is not necessarily the popped one. Here
// the amount reserved is always for the lead actually dequeued.
func (l *Ledger) Reserve(ctx context.Context, sessionID string, amount int64) (*core.Reservation, error) {
	return l.reserveWith(ctx, sessionID, amount, allCeilings, nil, false)
}

// ReserveOutput reserves for the report, ignoring the unit-independent
// ceilings.
//
// Those ceilings exist to stop research running away (§8.5), and by the time
// output runs the research has already stopped — usually BECAUSE a ceiling
// fired. Enforcing them here means the more effective a ceiling is, the less
// likely the session can afford to write up what it found, which inverts what
// escrow is for (§8.3).
//
// The escrow itself is the bound: the caller releases it, and this can only
// spend what Available() then reports. A session that has genuinely run out of
// money still cannot reserve.
func (l *Ledger) ReserveOutput(ctx context.Context, sessionID string, amount int64) (*core.Reservation, error) {
	return l.reserveWith(ctx, sessionID, amount, noCeilings, nil, false)
}

// ReserveVerify reserves for a verification call, exempt from MaxLeads.
//
// Verification dispatches no leads, and §8.5's lead ceiling exists to stop research
// running away. Enforcing it here meant that the moment a session hit max_leads — which
// is a NORMAL ending, not a failure — the Verifier could no longer reserve anything, so
// the last batch of claims was marked verified with an empty graph. Measured on a live
// run: 12 of 12 leads used, the final pass wrote 0 edges for 7 claims.
//
// MaxToolCalls and MaxWallClock still apply. Those bound total work rather than research
// fan-out, and verification is bounded independently by its own share of the budget
// (verifier.MaxShareOfBudget), so the money cannot run away either.
func (l *Ledger) ReserveVerify(ctx context.Context, sessionID string, amount int64) (*core.Reservation, error) {
	return l.reserveWith(ctx, sessionID, amount, ceilingsExceptLeads, nil, false)
}

// ReserveGround reserves for the grounding pass (§11.5.2). Grounding runs
// after research has already stopped, so it is exempt from MaxLeads and
// MaxWallClock, bound instead by MaxToolCalls and its share of the escrow.
func (l *Ledger) ReserveGround(ctx context.Context, sessionID string, amount int64) (*core.Reservation, error) {
	return l.reserveWith(ctx, sessionID, amount, ceilingsMoneyAndCalls, nil, false)
}

// ceilingPolicy selects which §8.5 ceilings a reservation honours.
type ceilingPolicy int

const (
	allCeilings ceilingPolicy = iota
	ceilingsExceptLeads
	// ceilingsMoneyAndCalls honours MaxToolCalls and Available() only.
	ceilingsMoneyAndCalls
	noCeilings
)

func (l *Ledger) reserveWith(ctx context.Context, sessionID string, amount int64, policy ceilingPolicy, leadID *string, countLead bool) (*core.Reservation, error) {
	if amount <= 0 {
		return nil, fmt.Errorf("budget: reserve amount must be positive, got %d", amount)
	}

	now := l.now()
	r := &core.Reservation{
		ID:        core.NewReservationID(),
		SessionID: sessionID,
		Amount:    amount,
		Status:    core.ReservationHeld,
		LeadID:    leadID,
		CreatedAt: now,
		ExpiresAt: now.Add(l.cfg.ReservationTTL),
	}

	err := l.st.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		s, err := tx.GetSession(ctx, sessionID)
		if err != nil {
			return err
		}
		if s.Status.Terminal() {
			return fmt.Errorf("budget: session %s is %s", sessionID, s.Status)
		}
		switch policy {
		case noCeilings:
		case ceilingsMoneyAndCalls:
			// Checked directly: HitCeiling reports only the first ceiling it
			// finds hit, which depends on check order, not on which one applies.
			if s.MaxToolCalls > 0 && s.ToolCallCount >= s.MaxToolCalls {
				return fmt.Errorf("%w: session %s hit max_tool_calls", ErrInsufficientBudget, sessionID)
			}
		case ceilingsExceptLeads:
			// Checked directly, for the same reason.
			if s.MaxToolCalls > 0 && s.ToolCallCount >= s.MaxToolCalls {
				return fmt.Errorf("%w: session %s hit max_tool_calls", ErrInsufficientBudget, sessionID)
			}
			if s.MaxWallClock > 0 && now.Sub(s.CreatedAt) >= s.MaxWallClock {
				return fmt.Errorf("%w: session %s hit max_wallclock", ErrInsufficientBudget, sessionID)
			}
		default:
			if hit, which := s.HitCeiling(now); hit {
				return fmt.Errorf("%w: session %s hit %s", ErrInsufficientBudget, sessionID, which)
			}
		}
		if avail := s.Available(); amount > avail {
			return fmt.Errorf("%w: need %s, have %s",
				ErrInsufficientBudget,
				core.FormatAmount(amount, s.BudgetUnit),
				core.FormatAmount(avail, s.BudgetUnit))
		}
		if err := tx.InsertReservation(ctx, r); err != nil {
			return err
		}
		delta := store.BudgetDelta{Held: amount}
		if countLead {
			// Count the lead HERE, in the same transaction that checked
			// MaxLeads against the count. Counting after the lead finishes is
			// exact only when running sequentially: K workers all reserve while
			// the counter reads N, all pass the check, and the session
			// dispatches up to K-1 leads past the ceiling.
			//
			// The order inside the transaction is what makes it safe, and is why
			// counting before reserving is wrong. HitCeiling reads the count
			// first and the increment lands after, so the lead that brings the
			// total UP TO MaxLeads runs and the next is refused; counting first
			// would fail the last permitted lead on its own reservation.
			delta.LeadCount = 1
		}
		return tx.ApplyBudgetDelta(ctx, sessionID, delta)
	})
	if err != nil {
		return nil, err
	}
	return r, nil
}

// ReserveFor takes the hold for a lead's FIRST attempt.
//
// Two things follow from naming the lead. The hold can be traced back to the
// work that took it when it is stranded, and this is the reservation that counts
// the lead against MaxLeads (§8.5).
//
// First attempt only — retries take ReserveForRetry. MaxLeads bounds research
// fan-out, so a lead that fails transiently and is retried is still one lead;
// counting each attempt would let a flaky network shrink the research plan while
// the ceiling reported it as work done.
func (l *Ledger) ReserveFor(ctx context.Context, sessionID, leadID string, amount int64) (*core.Reservation, error) {
	// Set on the reservation BEFORE it is inserted. Setting it afterwards only
	// touched the in-memory copy, so the stored lead_id stayed NULL — and the
	// stranded-reservation case this exists to serve is exactly the one where
	// the in-memory copy is gone.
	return l.reserveWith(ctx, sessionID, amount, allCeilings, &leadID, true)
}

// ReserveForRetry is ReserveFor for a second or later attempt at the same lead.
//
// It does not count the lead again, and it is EXEMPT FROM MaxLeads. Both follow
// from the same fact: this lead was already counted by its first attempt, so
// checking the lead ceiling here checks it against a number that includes the
// very lead being retried. The last permitted lead of a session would fail its
// own retry with "hit max_leads" and, because insufficient budget classifies as
// Fatal, end the session — measured at one worker as three leads yielding seven
// actor runs instead of nine, stopping at max_leads with the failures recorded
// as fatal rather than degraded.
//
// MaxToolCalls and MaxWallClock still apply. Those bound total work rather than
// research fan-out, and a retry is more work; only the fan-out ceiling has
// already accounted for this lead. Money is unaffected — every attempt holds and
// settles its own, because every attempt really does spend.
func (l *Ledger) ReserveForRetry(ctx context.Context, sessionID, leadID string, amount int64) (*core.Reservation, error) {
	return l.reserveWith(ctx, sessionID, amount, ceilingsExceptLeads, &leadID, false)
}

// SettleResult reports what a settle actually cost against what was held.
type SettleResult struct {
	Reserved  int64
	Charged   int64
	Overshoot int64 // Charged - Reserved, when positive
	Flagged   bool  // Charged exceeded Reserved * MaxOvershootFactor
	Cost      core.Cost
}

// price converts a call's tokens into money, if nobody has already.
//
// Here, and not in the callers, because this is the single point every tool
// call crosses on its way to the database. It used to be each caller's job: the
// executor's miner and the report writer looked the rate up, the planner and
// the verifier did not, and so 143 calls of one measured Quick run — 127,054
// tokens the provider billed — were recorded at $0.00. A quarter of that run's
// real cost was invisible, and --usd could not bind on the phase making most of
// the calls.
//
// Three things are left alone. A call that already carries dollars has been
// priced by somebody who knew better: a search is billed per query and arrives
// with its charge and no tokens, and re-pricing it from zero tokens would erase
// the largest single line on a run. A call with no tokens has nothing to
// convert. A call with no model cannot be looked up.
//
// An unpriced model keeps its tokens and gets no dollars, which is what
// token-mode budgets need and what `mole doctor` reports before a run.
func (l *Ledger) price(ctx context.Context, tc *core.ToolCall) {
	if tc.Cost.USDMicros != 0 || tc.Cost.TotalTokens() == 0 || tc.Model == "" {
		return
	}
	cost, err := l.cfg.Pricing.Cost(tc.Model, pricing.Usage{
		InputTokens:      tc.Cost.InputTokens,
		OutputTokens:     tc.Cost.OutputTokens,
		CacheReadTokens:  tc.Cost.CacheReadTokens,
		CacheWriteTokens: tc.Cost.CacheWriteTokens,
	})
	if err != nil {
		// Strict mode asked to be told. The tokens still stand.
		l.logger().WarnContext(ctx, "model not in pricing table; USD cost recorded as zero",
			"model", tc.Model, "role", tc.Role, "tokens", tc.Cost.TotalTokens())
		return
	}
	tc.Cost.USDMicros = cost.USDMicros
}

// Settle writes the actual cost rows and releases the hold, in one transaction.
//
// Partial cost is settled on failure too — the money was spent whether or not
// the call succeeded, and a ledger that only records successes cannot enforce a
// ceiling.
func (l *Ledger) Settle(ctx context.Context, r *core.Reservation, calls []core.ToolCall) (SettleResult, error) {
	var res SettleResult
	now := l.now()

	err := l.st.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		res = SettleResult{}

		s, err := tx.GetSession(ctx, r.SessionID)
		if err != nil {
			return err
		}

		stored, err := tx.GetReservation(ctx, r.ID)
		if err != nil {
			return err
		}
		if stored.Status != core.ReservationHeld {
			return fmt.Errorf("%w: reservation %s already %s", store.ErrConflict, r.ID, stored.Status)
		}

		var total core.Cost
		for i := range calls {
			tc := &calls[i]
			tc.SessionID = r.SessionID
			if tc.LeadID == nil {
				tc.LeadID = r.LeadID
			}
			l.price(ctx, tc)
			if err := tx.InsertToolCall(ctx, tc); err != nil {
				return err
			}
			total = total.Add(tc.Cost)
		}

		charged := total.BudgetAmount(s.BudgetUnit)

		if err := tx.ResolveReservation(ctx, r.ID, core.ReservationSettled, now); err != nil {
			return err
		}
		if err := tx.ApplyBudgetDelta(ctx, r.SessionID, store.BudgetDelta{
			Spent:         charged,
			Held:          -stored.Amount,
			ToolCallCount: int64(len(calls)),
		}); err != nil {
			return err
		}

		res = SettleResult{
			Reserved: stored.Amount,
			Charged:  charged,
			Cost:     total,
		}
		if charged > stored.Amount {
			res.Overshoot = charged - stored.Amount
		}
		if float64(charged) > float64(stored.Amount)*l.cfg.MaxOvershootFactor {
			res.Flagged = true
		}
		return nil
	})
	if err != nil {
		return SettleResult{}, err
	}

	r.Status = core.ReservationSettled
	r.ResolvedAt = &now
	return res, nil
}

// Release drops a hold without charging — used when a lead is cancelled before
// dispatch, or when a cache hit means no work was done.
func (l *Ledger) Release(ctx context.Context, r *core.Reservation) error {
	now := l.now()
	err := l.st.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		stored, err := tx.GetReservation(ctx, r.ID)
		if err != nil {
			return err
		}
		if stored.Status != core.ReservationHeld {
			return fmt.Errorf("%w: reservation %s already %s", store.ErrConflict, r.ID, stored.Status)
		}
		if err := tx.ResolveReservation(ctx, r.ID, core.ReservationReleased, now); err != nil {
			return err
		}
		return tx.ApplyBudgetDelta(ctx, r.SessionID, store.BudgetDelta{Held: -stored.Amount})
	})
	if err != nil {
		return err
	}
	r.Status = core.ReservationReleased
	r.ResolvedAt = &now
	return nil
}

// ---------------------------------------------------------------------------
// Escrow
// ---------------------------------------------------------------------------

// ReleaseEscrow makes the output reserve spendable. Call it once research is
// finished and the session is about to generate its report — after this, the
// escrowed amount is ordinary available budget.
func (l *Ledger) ReleaseEscrow(ctx context.Context, sessionID string) (int64, error) {
	var released int64
	err := l.st.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		s, err := tx.GetSession(ctx, sessionID)
		if err != nil {
			return err
		}
		released = s.Escrow
		if released == 0 {
			return nil
		}
		return tx.ApplyBudgetDelta(ctx, sessionID, store.BudgetDelta{Escrow: -released})
	})
	return released, err
}

// ---------------------------------------------------------------------------
// Inspection and recovery
// ---------------------------------------------------------------------------

// Available reports the spendable balance.
func (l *Ledger) Available(ctx context.Context, sessionID string) (int64, error) {
	var out int64
	err := l.st.Read(ctx, func(ctx context.Context, q store.Queries) error {
		s, err := q.GetSession(ctx, sessionID)
		if err != nil {
			return err
		}
		out = s.Available()
		return nil
	})
	return out, err
}

// VerifyResult compares the materialized counters against the ledger itself.
type VerifyResult struct {
	SpentRecorded   int64
	SpentFromLedger int64
	HeldRecorded    int64
	HeldFromRows    int64
	CallsRecorded   int64
	CallsFromRows   int64
	Cost            core.Cost
}

func (v VerifyResult) Consistent() bool {
	return v.SpentRecorded == v.SpentFromLedger &&
		v.HeldRecorded == v.HeldFromRows &&
		v.CallsRecorded == v.CallsFromRows
}

// Verify recomputes the session counters from the append-only rows.
//
// This is the payoff of treating the ledger as the source of truth: the
// materialized sums are always provable, and a divergence is a bug that can be
// detected in a test rather than a slow drift discovered in a bill.
func (l *Ledger) Verify(ctx context.Context, sessionID string) (VerifyResult, error) {
	var v VerifyResult
	err := l.st.Read(ctx, func(ctx context.Context, q store.Queries) error {
		s, err := q.GetSession(ctx, sessionID)
		if err != nil {
			return err
		}
		cost, err := q.SumCosts(ctx, sessionID)
		if err != nil {
			return err
		}
		held, err := q.SumHeldReservations(ctx, sessionID)
		if err != nil {
			return err
		}
		calls, err := q.CountToolCalls(ctx, sessionID)
		if err != nil {
			return err
		}
		v = VerifyResult{
			SpentRecorded:   s.Spent,
			SpentFromLedger: cost.BudgetAmount(s.BudgetUnit),
			HeldRecorded:    s.Held,
			HeldFromRows:    held,
			CallsRecorded:   s.ToolCallCount,
			CallsFromRows:   calls,
			Cost:            cost,
		}
		return nil
	})
	return v, err
}

// DefaultAbandonedSessionIdle is comfortably past the lease TTL: a live run
// touches updated_at on every settle and every counted lead, so anything
// quieter than this with nothing leased is not running.
const DefaultAbandonedSessionIdle = 30 * time.Minute

// SweepAbandonedSessions marks running sessions whose process died (§9.4).
//
// The third half of crash recovery. Leases and reservations were reclaimed at
// boot and the session row was not, so a killed process left a session
// `running` forever — listed by `sessions` and reconciled by `doctor` for the
// life of the database.
func (l *Ledger) SweepAbandonedSessions(ctx context.Context) (int, error) {
	var n int
	err := l.st.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		var err error
		idle := l.cfg.AbandonedSessionIdle
		if idle <= 0 {
			idle = DefaultAbandonedSessionIdle
		}
		// Real clock: updated_at is written from it, and comparing a real
		// timestamp against an injectable one is what made a just-created
		// session look abandoned.
		n, err = tx.SweepAbandonedSessions(ctx, time.Now().Add(-idle))
		return err
	})
	return n, err
}

// ReleaseSessionHolds releases every hold a session still carries.
//
// For a session that ended without running its own settle path — a panic in the
// pipeline is the case it was written for. Waiting out the reservation TTL there
// leaves money reading as held on a session that is already finished, and the
// next sweep that would reclaim it runs at daemon boot.
//
// Distinct from Release, which resolves one reservation the caller is holding,
// and from SweepExpired, which is TTL-gated and global.
func (l *Ledger) ReleaseSessionHolds(ctx context.Context, sessionID string) (int, error) {
	var n int
	err := l.st.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		var err error
		n, err = tx.ReleaseSessionHolds(ctx, sessionID)
		return err
	})
	return n, err
}

// SweepExpired releases holds whose TTL has passed. Run it on daemon boot and
// periodically thereafter; it is the reservation half of crash recovery.
func (l *Ledger) SweepExpired(ctx context.Context) (int, error) {
	var n int
	err := l.st.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		var err error
		n, err = tx.ExpireStaleReservations(ctx, l.now())
		return err
	})
	return n, err
}

// Finish marks a session terminal and records why it stopped.
//
// because is a ceiling name — "max_tool_calls", "max_leads", "max_wallclock" —
// or empty for a session that simply finished. Taken here rather than left to
// the caller's logs, because a status of "budget_exhausted" is reached by five
// different endings and only one of them is about money.
func (l *Ledger) Finish(
	ctx context.Context, sessionID string, status core.SessionStatus, because string,
) error {
	return l.st.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.FinishSession(ctx, sessionID, status, because)
	})
}

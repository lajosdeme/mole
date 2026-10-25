package budget_test

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/budget"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/store/sqlite"
)

func newTestDB(t *testing.T) *sqlite.DB {
	t.Helper()
	// A real on-disk database, not :memory:, so the tests exercise the actual
	// WAL + single-writer configuration the daemon runs.
	path := filepath.Join(t.TempDir(), "test.db")
	db, err := sqlite.Open(path, sqlite.Options{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.Migrate(context.Background()); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	return db
}

func newSession(t *testing.T, l *budget.Ledger, budgetAmount int64) *core.Session {
	t.Helper()
	s, err := l.CreateSession(context.Background(), budget.SessionSpec{
		Prompt:     "test",
		Mode:       core.ModeReport,
		ActorTypes: []core.ActorType{core.ActorWeb},
		BudgetUnit: core.BudgetUSD,
		Budget:     budgetAmount,
	})
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	return s
}

func noEscrow() budget.Config {
	c := budget.DefaultConfig()
	c.EscrowFraction = 0
	return c
}

func TestReserveSettleUpdatesLedgerAndCounters(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)
	l := budget.New(db, noEscrow())
	s := newSession(t, l, 1_000_000)

	r, err := l.Reserve(ctx, s.ID, 100_000)
	if err != nil {
		t.Fatalf("reserve: %v", err)
	}

	// While held, the amount is unavailable but not yet spent.
	avail, err := l.Available(ctx, s.ID)
	if err != nil {
		t.Fatalf("available: %v", err)
	}
	if want := int64(900_000); avail != want {
		t.Fatalf("available after reserve = %d, want %d", avail, want)
	}

	res, err := l.Settle(ctx, r, []core.ToolCall{{
		Role:  core.RoleExecutor,
		Type:  core.CallLLM,
		Model: "claude-opus-5",
		Cost:  core.Cost{USDMicros: 80_000, InputTokens: 1000, OutputTokens: 200},
	}})
	if err != nil {
		t.Fatalf("settle: %v", err)
	}
	if res.Charged != 80_000 {
		t.Errorf("charged = %d, want 80000", res.Charged)
	}
	if res.Overshoot != 0 {
		t.Errorf("overshoot = %d, want 0", res.Overshoot)
	}

	// The hold is gone and only the actual cost was charged.
	avail, err = l.Available(ctx, s.ID)
	if err != nil {
		t.Fatalf("available: %v", err)
	}
	if want := int64(920_000); avail != want {
		t.Fatalf("available after settle = %d, want %d", avail, want)
	}

	v, err := l.Verify(ctx, s.ID)
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if !v.Consistent() {
		t.Fatalf("ledger inconsistent: %+v", v)
	}
}

func TestReleaseReturnsHoldWithoutCharging(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)
	l := budget.New(db, noEscrow())
	s := newSession(t, l, 500_000)

	r, err := l.Reserve(ctx, s.ID, 200_000)
	if err != nil {
		t.Fatalf("reserve: %v", err)
	}
	if err := l.Release(ctx, r); err != nil {
		t.Fatalf("release: %v", err)
	}

	avail, err := l.Available(ctx, s.ID)
	if err != nil {
		t.Fatalf("available: %v", err)
	}
	if avail != 500_000 {
		t.Fatalf("available after release = %d, want 500000", avail)
	}

	v, _ := l.Verify(ctx, s.ID)
	if v.SpentFromLedger != 0 {
		t.Fatalf("release charged %d, want 0", v.SpentFromLedger)
	}
}

func TestReserveRefusesBeyondBudget(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)
	l := budget.New(db, noEscrow())
	s := newSession(t, l, 100_000)

	if _, err := l.Reserve(ctx, s.ID, 100_001); !errors.Is(err, budget.ErrInsufficientBudget) {
		t.Fatalf("over-budget reserve error = %v, want ErrInsufficientBudget", err)
	}
	// Exactly the budget is allowed.
	if _, err := l.Reserve(ctx, s.ID, 100_000); err != nil {
		t.Fatalf("exact-budget reserve: %v", err)
	}
}

// TestConcurrentReserveNeverOvershoots is the headline M0 test.
//
// With post-hoc charging, N workers all pass the affordability check before any
// of them pays, and the session overshoots by up to N lead-costs. Reserving
// inside the transaction makes the ceiling hard: exactly budget/amount holds
// can exist at once, no matter how many goroutines race for them.
func TestConcurrentReserveNeverOvershoots(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)
	l := budget.New(db, noEscrow())

	const (
		total   = int64(1_000_000)
		perUnit = int64(100_000)
		workers = 64
	)
	s := newSession(t, l, total)

	var (
		granted atomic.Int64
		refused atomic.Int64
		wg      sync.WaitGroup
	)
	start := make(chan struct{})

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			_, err := l.Reserve(ctx, s.ID, perUnit)
			switch {
			case err == nil:
				granted.Add(1)
			case errors.Is(err, budget.ErrInsufficientBudget):
				refused.Add(1)
			default:
				t.Errorf("unexpected reserve error: %v", err)
			}
		}()
	}
	close(start)
	wg.Wait()

	wantGranted := total / perUnit
	if got := granted.Load(); got != wantGranted {
		t.Fatalf("granted %d reservations, want exactly %d", got, wantGranted)
	}
	if got := refused.Load(); got != workers-wantGranted {
		t.Fatalf("refused %d, want %d", got, workers-wantGranted)
	}

	// The invariant that matters: held never exceeds the budget.
	var sess *core.Session
	if err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		sess, err = q.GetSession(ctx, s.ID)
		return err
	}); err != nil {
		t.Fatalf("read session: %v", err)
	}
	if sess.Held != total {
		t.Fatalf("held = %d, want %d", sess.Held, total)
	}
	if sess.Available() != 0 {
		t.Fatalf("available = %d, want 0", sess.Available())
	}

	v, err := l.Verify(ctx, s.ID)
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if !v.Consistent() {
		t.Fatalf("ledger inconsistent after concurrent reserves: %+v", v)
	}
}

// TestConcurrentSettleKeepsSpentExact checks the other half: many workers
// settling at once must produce a Spent that equals the sum of the ledger rows,
// with no lost updates.
func TestConcurrentSettleKeepsSpentExact(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)
	l := budget.New(db, noEscrow())

	const (
		workers = 32
		perCall = int64(10_000)
	)
	s := newSession(t, l, workers*perCall)

	reservations := make([]*core.Reservation, workers)
	for i := range reservations {
		r, err := l.Reserve(ctx, s.ID, perCall)
		if err != nil {
			t.Fatalf("reserve %d: %v", i, err)
		}
		reservations[i] = r
	}

	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := range reservations {
		wg.Add(1)
		go func(r *core.Reservation) {
			defer wg.Done()
			<-start
			_, err := l.Settle(ctx, r, []core.ToolCall{{
				Role: core.RoleExecutor,
				Type: core.CallFetch,
				Cost: core.Cost{USDMicros: perCall},
			}})
			if err != nil {
				t.Errorf("settle: %v", err)
			}
		}(reservations[i])
	}
	close(start)
	wg.Wait()

	v, err := l.Verify(ctx, s.ID)
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if !v.Consistent() {
		t.Fatalf("ledger inconsistent: %+v", v)
	}
	if want := int64(workers) * perCall; v.SpentRecorded != want {
		t.Fatalf("spent = %d, want %d", v.SpentRecorded, want)
	}
	if v.HeldRecorded != 0 {
		t.Fatalf("held = %d after all settles, want 0", v.HeldRecorded)
	}
}

func TestDoubleSettleIsRejected(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)
	l := budget.New(db, noEscrow())
	s := newSession(t, l, 500_000)

	r, err := l.Reserve(ctx, s.ID, 50_000)
	if err != nil {
		t.Fatalf("reserve: %v", err)
	}
	call := []core.ToolCall{{Role: core.RoleExecutor, Type: core.CallFetch, Cost: core.Cost{USDMicros: 40_000}}}

	if _, err := l.Settle(ctx, r, call); err != nil {
		t.Fatalf("first settle: %v", err)
	}
	if _, err := l.Settle(ctx, r, call); !errors.Is(err, store.ErrConflict) {
		t.Fatalf("second settle error = %v, want ErrConflict", err)
	}

	v, _ := l.Verify(ctx, s.ID)
	if v.SpentRecorded != 40_000 {
		t.Fatalf("spent = %d after rejected double settle, want 40000", v.SpentRecorded)
	}
}

// TestEscrowProtectsOutputBudget is the regression test for rev 1's starvation
// bug: research spending everything, then output generation having nothing left.
func TestEscrowProtectsOutputBudget(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)

	cfg := budget.DefaultConfig()
	cfg.EscrowFraction = 0.2
	l := budget.New(db, cfg)

	s := newSession(t, l, 1_000_000)
	if s.Escrow != 200_000 {
		t.Fatalf("escrow = %d, want 200000", s.Escrow)
	}

	// Research cannot touch the escrow.
	if _, err := l.Reserve(ctx, s.ID, 800_001); !errors.Is(err, budget.ErrInsufficientBudget) {
		t.Fatalf("reserve into escrow error = %v, want ErrInsufficientBudget", err)
	}

	// Drain everything research is allowed to spend.
	r, err := l.Reserve(ctx, s.ID, 800_000)
	if err != nil {
		t.Fatalf("reserve research budget: %v", err)
	}
	if _, err := l.Settle(ctx, r, []core.ToolCall{{
		Role: core.RoleExecutor, Type: core.CallLLM, Cost: core.Cost{USDMicros: 800_000},
	}}); err != nil {
		t.Fatalf("settle: %v", err)
	}

	if avail, _ := l.Available(ctx, s.ID); avail != 0 {
		t.Fatalf("available before escrow release = %d, want 0", avail)
	}

	// Output generation releases the escrow and can now pay for itself.
	released, err := l.ReleaseEscrow(ctx, s.ID)
	if err != nil {
		t.Fatalf("release escrow: %v", err)
	}
	if released != 200_000 {
		t.Fatalf("released = %d, want 200000", released)
	}
	if _, err := l.Reserve(ctx, s.ID, 200_000); err != nil {
		t.Fatalf("reserve output budget after escrow release: %v", err)
	}
}

// TestSweepExpiredReleasesStrandedHolds covers the reservation half of crash
// recovery: a worker that died mid-run must not hold budget forever.
func TestSweepExpiredReleasesStrandedHolds(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)

	cfg := noEscrow()
	cfg.ReservationTTL = time.Minute
	l := budget.New(db, cfg)

	now := time.Now().UTC()
	l.SetClock(func() time.Time { return now })

	s := newSession(t, l, 1_000_000)
	if _, err := l.Reserve(ctx, s.ID, 300_000); err != nil {
		t.Fatalf("reserve: %v", err)
	}
	if avail, _ := l.Available(ctx, s.ID); avail != 700_000 {
		t.Fatalf("available while held = %d, want 700000", avail)
	}

	// Simulate the daemon restarting well after the lease expired.
	now = now.Add(10 * time.Minute)

	n, err := l.SweepExpired(ctx)
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}
	if n != 1 {
		t.Fatalf("swept %d reservations, want 1", n)
	}
	if avail, _ := l.Available(ctx, s.ID); avail != 1_000_000 {
		t.Fatalf("available after sweep = %d, want 1000000", avail)
	}
}

func TestSettleFlagsOvershoot(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)
	cfg := noEscrow()
	cfg.MaxOvershootFactor = 1.5
	l := budget.New(db, cfg)
	s := newSession(t, l, 1_000_000)

	r, err := l.Reserve(ctx, s.ID, 10_000)
	if err != nil {
		t.Fatalf("reserve: %v", err)
	}
	// The estimate was badly wrong. The ledger records the truth and flags it;
	// it does not pretend the money was not spent.
	res, err := l.Settle(ctx, r, []core.ToolCall{{
		Role: core.RoleExecutor, Type: core.CallLLM, Cost: core.Cost{USDMicros: 40_000},
	}})
	if err != nil {
		t.Fatalf("settle: %v", err)
	}
	if res.Overshoot != 30_000 {
		t.Errorf("overshoot = %d, want 30000", res.Overshoot)
	}
	if !res.Flagged {
		t.Error("overshoot beyond MaxOvershootFactor was not flagged")
	}

	v, _ := l.Verify(ctx, s.ID)
	if !v.Consistent() {
		t.Fatalf("ledger inconsistent after overshoot: %+v", v)
	}
}

func TestTokenModeCountsCacheTokens(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)
	l := budget.New(db, noEscrow())

	s, err := l.CreateSession(ctx, budget.SessionSpec{
		Prompt:     "token mode",
		Mode:       core.ModeReport,
		ActorTypes: []core.ActorType{core.ActorWeb},
		BudgetUnit: core.BudgetTokens,
		Budget:     50_000,
	})
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	r, err := l.Reserve(ctx, s.ID, 20_000)
	if err != nil {
		t.Fatalf("reserve: %v", err)
	}
	// Cache reads are cheaper, not free. A token budget that ignored them
	// would not bound anything.
	cost := core.Cost{
		USDMicros:        12_345,
		InputTokens:      5_000,
		OutputTokens:     1_000,
		CacheReadTokens:  8_000,
		CacheWriteTokens: 2_000,
	}
	res, err := l.Settle(ctx, r, []core.ToolCall{{Role: core.RoleExecutor, Type: core.CallLLM, Cost: cost}})
	if err != nil {
		t.Fatalf("settle: %v", err)
	}
	if want := int64(16_000); res.Charged != want {
		t.Fatalf("charged %d tokens, want %d", res.Charged, want)
	}

	v, _ := l.Verify(ctx, s.ID)
	if !v.Consistent() {
		t.Fatalf("ledger inconsistent: %+v", v)
	}
	// The USD figure is still recorded, so switching the session's unit is a
	// display choice rather than a data migration.
	if v.Cost.USDMicros != 12_345 {
		t.Fatalf("usd not recorded in token mode: %d", v.Cost.USDMicros)
	}
}

func TestCeilingsAreUnitIndependent(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)
	l := budget.New(db, noEscrow())

	// Token mode with a huge budget but a tight call ceiling: this is the hole
	// that makes an unbounded fetch loop free otherwise.
	s, err := l.CreateSession(ctx, budget.SessionSpec{
		Prompt:       "ceiling",
		Mode:         core.ModeReport,
		ActorTypes:   []core.ActorType{core.ActorWeb},
		BudgetUnit:   core.BudgetTokens,
		Budget:       1_000_000,
		MaxToolCalls: 2,
	})
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	for i := 0; i < 2; i++ {
		r, err := l.Reserve(ctx, s.ID, 10)
		if err != nil {
			t.Fatalf("reserve %d: %v", i, err)
		}
		// A fetch costs no tokens at all — exactly the case the budget cannot see.
		if _, err := l.Settle(ctx, r, []core.ToolCall{{Role: core.RoleExecutor, Type: core.CallFetch}}); err != nil {
			t.Fatalf("settle %d: %v", i, err)
		}
	}

	if _, err := l.Reserve(ctx, s.ID, 10); !errors.Is(err, budget.ErrInsufficientBudget) {
		t.Fatalf("reserve past MaxToolCalls error = %v, want ErrInsufficientBudget", err)
	}
}

// TestReserveOutputIgnoresCeilings. The unit-independent ceilings exist to stop
// research running away (§8.5), and by the time output runs the research has
// already stopped — usually BECAUSE a ceiling fired. Enforcing them here means
// the more effective a ceiling is, the less likely the session can afford to
// write up what it found, which inverts what escrow is for (§8.3).
func TestReserveOutputIgnoresCeilings(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)
	led := budget.New(db, budget.DefaultConfig())

	sess, err := led.CreateSession(ctx, budget.SessionSpec{
		Prompt: "q", Mode: core.ModeReport,
		ActorTypes: []core.ActorType{core.ActorWeb},
		BudgetUnit: core.BudgetUSD, Budget: 10 * core.MicrosPerUSD,
		MaxLeads: 1,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Drive the session into its lead ceiling.
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.ApplyBudgetDelta(ctx, sess.ID, store.BudgetDelta{LeadCount: 1})
	}); err != nil {
		t.Fatal(err)
	}

	if _, err := led.Reserve(ctx, sess.ID, 1000); err == nil {
		t.Error("a normal reservation succeeded past the ceiling")
	}

	released, err := led.ReleaseEscrow(ctx, sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if released <= 0 {
		t.Fatal("no escrow was held to fund the report")
	}

	r, err := led.ReserveOutput(ctx, sess.ID, released)
	if err != nil {
		t.Fatalf("the report could not be funded from escrow after a ceiling fired: %v", err)
	}
	if _, err := led.Settle(ctx, r, []core.ToolCall{{
		SessionID: sess.ID, Role: core.RoleOutput, Type: core.CallLLM,
		Cost: core.Cost{USDMicros: 5_000},
	}}); err != nil {
		t.Fatal(err)
	}
}

// TestReserveOutputStillRespectsAvailableBudget. Escrow is the bound, not an
// exemption: a session with nothing left cannot conjure a report.
func TestReserveOutputStillRespectsAvailableBudget(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)
	led := budget.New(db, budget.DefaultConfig())

	sess, err := led.CreateSession(ctx, budget.SessionSpec{
		Prompt: "q", Mode: core.ModeReport,
		ActorTypes: []core.ActorType{core.ActorWeb},
		BudgetUnit: core.BudgetUSD, Budget: 100_000,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Spend everything that is not escrow.
	avail := sess.Available()
	r, err := led.Reserve(ctx, sess.ID, avail)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := led.Settle(ctx, r, []core.ToolCall{{
		SessionID: sess.ID, Role: core.RoleExecutor, Type: core.CallLLM,
		Cost: core.Cost{USDMicros: avail},
	}}); err != nil {
		t.Fatal(err)
	}

	if _, err := led.ReserveOutput(ctx, sess.ID, 1_000_000); err == nil {
		t.Error("ReserveOutput handed out budget the session does not have")
	}
}

// TestAbandonedSessionsAreReclaimed is the third half of §9.4. Leases and
// reservations were reclaimed at boot and the session row was not, so a killed
// process left a session `running` forever — listed by `sessions` and
// reconciled by `doctor` for the life of the database.
//
// Found for real: a test binary killed mid-run left four such rows behind.
func TestAbandonedSessionsAreReclaimed(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)

	// A tiny threshold against the real clock. The row's updated_at is written
	// from the real clock too, so this keeps one clock in play — the first
	// version fast-forwarded an injectable clock and made a just-created
	// session look stale, because only the cutoff moved.
	cfg := budget.DefaultConfig()
	cfg.AbandonedSessionIdle = 20 * time.Millisecond
	led := budget.New(db, cfg)

	abandoned, err := led.CreateSession(ctx, budget.SessionSpec{
		Prompt: "abandoned", Mode: core.ModeReport,
		ActorTypes: []core.ActorType{core.ActorWeb},
		BudgetUnit: core.BudgetUSD, Budget: core.MicrosPerUSD,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Nothing touches it for longer than the threshold.
	time.Sleep(50 * time.Millisecond)

	n, err := led.SweepAbandonedSessions(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("swept %d sessions, want 1", n)
	}

	var after *core.Session
	if err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		after, err = q.GetSession(ctx, abandoned.ID)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if after.Status != core.StatusFailed {
		t.Errorf("status = %s, want failed", after.Status)
	}
}

// TestALiveSessionIsNotReclaimed. Marking a slow-but-working run as dead
// underneath itself would be far worse than leaving a stale row: the ledger
// would refuse its next reservation, because Reserve rejects terminal sessions.
func TestALiveSessionIsNotReclaimed(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)

	cfg := budget.DefaultConfig()
	cfg.AbandonedSessionIdle = 20 * time.Millisecond
	led := budget.New(db, cfg)

	// Idle past the threshold, BUT holding a live lease — which is the definition
	// of a running worker.
	leased, err := led.CreateSession(ctx, budget.SessionSpec{
		Prompt: "slow but working", Mode: core.ModeReport,
		ActorTypes: []core.ActorType{core.ActorWeb},
		BudgetUnit: core.BudgetUSD, Budget: core.MicrosPerUSD,
	})
	if err != nil {
		t.Fatal(err)
	}
	lead := core.Lead{
		ID: core.NewLeadID(), SessionID: leased.ID, ActorType: core.ActorWeb,
		Query: "q", Status: core.LeadQueued,
	}
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		if err := tx.InsertLead(ctx, &lead); err != nil {
			return err
		}
		// A lease that is still valid far into the future.
		_, err := tx.LeaseNextLead(ctx, leased.ID, "worker", time.Now().Add(2*time.Hour))
		return err
	}); err != nil {
		t.Fatal(err)
	}

	time.Sleep(50 * time.Millisecond)

	// Created after the wait, so it is genuinely recent.
	fresh, err := led.CreateSession(ctx, budget.SessionSpec{
		Prompt: "fresh", Mode: core.ModeReport,
		ActorTypes: []core.ActorType{core.ActorWeb},
		BudgetUnit: core.BudgetUSD, Budget: core.MicrosPerUSD,
	})
	if err != nil {
		t.Fatal(err)
	}

	n, err := led.SweepAbandonedSessions(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("swept %d live sessions", n)
	}
	for _, id := range []string{fresh.ID, leased.ID} {
		var s *core.Session
		db.Read(ctx, func(ctx context.Context, q store.Queries) error {
			var err error
			s, err = q.GetSession(ctx, id)
			return err
		})
		if s.Status != core.StatusRunning {
			t.Errorf("session %s was marked %s", id, s.Status)
		}
	}
}

// TestFinishedSessionsAreLeftAlone, or a completed run would be rewritten as
// failed.
func TestFinishedSessionsAreLeftAlone(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)

	cfg := budget.DefaultConfig()
	cfg.AbandonedSessionIdle = time.Millisecond
	led := budget.New(db, cfg)

	done, err := led.CreateSession(ctx, budget.SessionSpec{
		Prompt: "done", Mode: core.ModeReport,
		ActorTypes: []core.ActorType{core.ActorWeb},
		BudgetUnit: core.BudgetUSD, Budget: core.MicrosPerUSD,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := led.Finish(ctx, done.ID, core.StatusDone, ""); err != nil {
		t.Fatal(err)
	}

	time.Sleep(20 * time.Millisecond)
	if n, _ := led.SweepAbandonedSessions(ctx); n != 0 {
		t.Errorf("swept %d finished sessions", n)
	}
}

// TestSweepSurvivesAnUnreleasableReservation. This is boot recovery: it runs
// before anything else and sweeps EVERY session, so a single bad row used to
// take the whole transaction down and leave every other session's stale
// reservations in place.
//
// Seen for real, though by a route mole itself cannot produce: a session
// deleted through a connection with foreign keys disabled left `held`
// reservations pointing at a row that no longer existed, and every subsequent
// run failed the sweep with "would violate a non-negative invariant" — so no
// reservation anywhere was released, ever.
//
// The store rejects orphaned reservations outright, so this reproduces the
// reachable version of the same shape: counter drift, where a session's held
// total no longer covers the reservations still recorded against it. One
// release must fail and the rest must still happen.
func TestSweepSurvivesAnUnreleasableReservation(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)

	cfg := budget.DefaultConfig()
	cfg.ReservationTTL = 10 * time.Millisecond
	led := budget.New(db, cfg)

	sess, err := led.CreateSession(ctx, budget.SessionSpec{
		Prompt: "drifted", Mode: core.ModeReport,
		ActorTypes: []core.ActorType{core.ActorWeb},
		BudgetUnit: core.BudgetUSD, Budget: 10 * core.MicrosPerUSD,
	})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		if _, err := led.Reserve(ctx, sess.ID, 50_000); err != nil {
			t.Fatal(err)
		}
	}

	// Drive held below what the two reservations claim, so releasing both would
	// take it negative and the second must be refused.
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.ApplyBudgetDelta(ctx, sess.ID, store.BudgetDelta{Held: -50_000})
	}); err != nil {
		t.Fatal(err)
	}

	time.Sleep(30 * time.Millisecond)

	n, err := led.SweepExpired(ctx)
	if n == 0 {
		t.Errorf("no reservation released (err=%v); one bad row aborted the whole sweep", err)
	}

	// Every reservation must end up resolved, or a sweep that cannot release a
	// hold leaves it to be retried forever.
	var held int64
	if err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		held, err = q.SumHeldReservations(ctx, sess.ID)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if held != 0 {
		t.Errorf("%d still held in unresolved reservations after the sweep", held)
	}
}

// TestReserveGroundIgnoresTheWallClock checks that ReserveGround ignores
// MaxWallClock and MaxLeads but still enforces MaxToolCalls and the money.
func TestReserveGroundIgnoresTheWallClock(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)
	l := budget.New(db, noEscrow())
	t0 := time.Date(2026, 10, 25, 12, 0, 0, 0, time.UTC)
	now := t0
	l.SetClock(func() time.Time { return now })

	create := func(spec budget.SessionSpec) *core.Session {
		t.Helper()
		spec.Prompt, spec.Mode = "q", core.ModeReport
		spec.ActorTypes = []core.ActorType{core.ActorWeb}
		spec.BudgetUnit, spec.Budget = core.BudgetUSD, core.MicrosPerUSD
		s, err := l.CreateSession(ctx, spec)
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	bump := func(id string, d store.BudgetDelta) {
		t.Helper()
		if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
			return tx.ApplyBudgetDelta(ctx, id, d)
		}); err != nil {
			t.Fatal(err)
		}
	}

	s := create(budget.SessionSpec{MaxWallClock: time.Minute, MaxLeads: 1, MaxToolCalls: 10})
	now = t0.Add(2 * time.Minute)

	if _, err := l.ReserveVerify(ctx, s.ID, 1000); !errors.Is(err, budget.ErrInsufficientBudget) {
		t.Fatalf("ReserveVerify past the wall clock: err = %v, want ErrInsufficientBudget", err)
	}
	ground := func(why string) {
		t.Helper()
		r, err := l.ReserveGround(ctx, s.ID, 1000)
		if err != nil {
			t.Fatalf("ReserveGround %s: %v", why, err)
		}
		if err := l.Release(ctx, r); err != nil {
			t.Fatal(err)
		}
	}
	ground("past the wall clock")

	// max_leads is a research ceiling too.
	bump(s.ID, store.BudgetDelta{LeadCount: 1})
	ground("past the wall clock and max_leads")

	// MaxToolCalls still binds.
	bump(s.ID, store.BudgetDelta{ToolCallCount: 10})
	if _, err := l.ReserveGround(ctx, s.ID, 1000); !errors.Is(err, budget.ErrInsufficientBudget) {
		t.Errorf("ReserveGround at MaxToolCalls: err = %v, want ErrInsufficientBudget", err)
	}

	// So does the money.
	rich := create(budget.SessionSpec{})
	avail, err := l.Available(ctx, rich.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := l.ReserveGround(ctx, rich.ID, avail+1); !errors.Is(err, budget.ErrInsufficientBudget) {
		t.Errorf("ReserveGround beyond Available(): err = %v, want ErrInsufficientBudget", err)
	}
}

// TestMaxLeadsDoesNotMaskTheWallClock checks that a session at MaxLeads still
// has MaxWallClock enforced on ReserveVerify and ReserveForRetry.
func TestMaxLeadsDoesNotMaskTheWallClock(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)
	l := budget.New(db, noEscrow())
	t0 := time.Date(2026, 10, 25, 12, 0, 0, 0, time.UTC)
	now := t0
	l.SetClock(func() time.Time { return now })

	s, err := l.CreateSession(ctx, budget.SessionSpec{
		Prompt:       "q",
		Mode:         core.ModeReport,
		ActorTypes:   []core.ActorType{core.ActorWeb},
		BudgetUnit:   core.BudgetUSD,
		Budget:       core.MicrosPerUSD,
		MaxWallClock: time.Minute,
		MaxLeads:     1,
		MaxToolCalls: 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.ApplyBudgetDelta(ctx, s.ID, store.BudgetDelta{LeadCount: 1})
	}); err != nil {
		t.Fatal(err)
	}

	// At MaxLeads but inside the clock: both still run.
	r, err := l.ReserveVerify(ctx, s.ID, 1000)
	if err != nil {
		t.Fatalf("ReserveVerify at MaxLeads within the clock: %v", err)
	}
	if err := l.Release(ctx, r); err != nil {
		t.Fatal(err)
	}
	r, err = l.ReserveForRetry(ctx, s.ID, "lead-1", 1000)
	if err != nil {
		t.Fatalf("ReserveForRetry at MaxLeads within the clock: %v", err)
	}
	if err := l.Release(ctx, r); err != nil {
		t.Fatal(err)
	}

	// At MaxLeads AND past the clock: the clock binds.
	now = t0.Add(2 * time.Minute)
	if _, err := l.ReserveVerify(ctx, s.ID, 1000); !errors.Is(err, budget.ErrInsufficientBudget) || !strings.Contains(err.Error(), "max_wallclock") {
		t.Errorf("ReserveVerify at MaxLeads past the clock: err = %v, want ErrInsufficientBudget naming max_wallclock", err)
	}
	if _, err := l.ReserveForRetry(ctx, s.ID, "lead-1", 1000); !errors.Is(err, budget.ErrInsufficientBudget) || !strings.Contains(err.Error(), "max_wallclock") {
		t.Errorf("ReserveForRetry at MaxLeads past the clock: err = %v, want ErrInsufficientBudget naming max_wallclock", err)
	}
}

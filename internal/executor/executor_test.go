package executor_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/actors"
	"github.com/lajosdeme/mole/internal/budget"
	"github.com/lajosdeme/mole/internal/cache"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/executor"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/planner"
	"github.com/lajosdeme/mole/internal/queue"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/store/sqlite"
	"github.com/lajosdeme/mole/internal/verifier"
)

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

type scriptedLLM struct {
	mu       sync.Mutex
	replies  []string
	calls    int
	errAfter int // when >0, calls at or past this index fail
	prompts  []string
}

func (s *scriptedLLM) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls
}

func (s *scriptedLLM) seen() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.prompts)
}

func (s *scriptedLLM) Name() string               { return "fake" }
func (s *scriptedLLM) ModelFor(t llm.Tier) string { return "fake-model" }
func (s *scriptedLLM) Complete(ctx context.Context, req llm.Request) (*llm.Response, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	i := s.calls
	s.calls++
	for _, m := range req.Messages {
		s.prompts = append(s.prompts, m.Text)
	}
	resp := &llm.Response{
		Model: "fake-model",
		Usage: llm.Usage{InputTokens: 400, OutputTokens: 60},
	}
	if s.errAfter > 0 && i >= s.errAfter {
		return resp, llm.ErrRateLimited
	}
	resp.Text = `{"done":true}`
	if i < len(s.replies) {
		resp.Text = s.replies[i]
	}
	return resp, nil
}

func planJSON(questions ...string) string {
	var qs []map[string]string
	for i, q := range questions {
		qs = append(qs, map[string]string{"id": fmt.Sprintf("q%d", i+1), "question": q})
	}
	b, _ := json.Marshal(map[string]any{"questions": qs})
	return string(b)
}

// scriptedActor returns a scripted outcome per lead, in order.
//
// It persists the claims it returns, because the real WebActor does (its step 7) and
// the Verifier reads the store rather than a batch (§11.1). Without this the rig
// looked correct and verification had nothing to verify — a fake that diverged from
// the thing it stands for on exactly the property under test.
type scriptedActor struct {
	mu      sync.Mutex
	runs    int
	db      store.Store
	outcome func(n int, lead core.Lead) (*actors.Result, error)
}

func (a *scriptedActor) Type() core.ActorType { return core.ActorWeb }
func (a *scriptedActor) Run(ctx context.Context, lead core.Lead) (*actors.Result, error) {
	a.mu.Lock()
	n := a.runs
	a.runs++
	a.mu.Unlock()

	res, err := a.outcome(n, lead)
	if res != nil && len(res.Claims) > 0 && a.db != nil {
		for i := range res.Claims {
			res.Claims[i].SessionID = lead.SessionID
			res.Claims[i].LeadID = lead.ID
		}
		if ierr := a.db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
			return tx.InsertClaims(ctx, res.Claims)
		}); ierr != nil && err == nil {
			return res, ierr
		}
	}
	return res, err
}

func (a *scriptedActor) count() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.runs
}

func okResult(claims int, cost int64) *actors.Result {
	r := &actors.Result{
		Summary: "a summary",
		Costs: []core.ToolCall{{
			Role: core.RoleExecutor, Type: core.CallLLM, Model: "fake-model",
			Cost: core.Cost{USDMicros: cost, InputTokens: 1000, OutputTokens: 100},
		}},
	}
	for i := 0; i < claims; i++ {
		r.Claims = append(r.Claims, core.Claim{
			Text:   fmt.Sprintf("claim %d", i),
			Source: "https://example.com/a",
			Quote:  "a quote long enough to constitute real evidence",
		})
	}
	return r
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

type rig struct {
	db    *sqlite.DB
	led   *budget.Ledger
	q     *queue.Queue
	exec  *executor.Executor
	sess  *core.Session
	actor *scriptedActor
	llm   *scriptedLLM
}

func newRig(t *testing.T, budgetAmount int64, replies []string, outcome func(int, core.Lead) (*actors.Result, error)) *rig {
	return newRigWithCeilings(t, budgetAmount, 50, 200, replies, outcome)
}

func newRigWithCeilings(t *testing.T, budgetAmount int64, maxLeads, maxCalls int64, replies []string, outcome func(int, core.Lead) (*actors.Result, error)) *rig {
	t.Helper()
	return newRigFromSpec(t, budget.SessionSpec{
		Budget: budgetAmount, MaxLeads: maxLeads, MaxToolCalls: maxCalls,
	}, replies, outcome)
}

// newRigFromSpec builds a rig around a session with the given budget and ceilings;
// the prompt, mode, actor and unit are the rig's own.
func newRigFromSpec(t *testing.T, spec budget.SessionSpec, replies []string, outcome func(int, core.Lead) (*actors.Result, error)) *rig {
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
	spec.Prompt, spec.Mode = "the research question", core.ModeReport
	spec.ActorTypes = []core.ActorType{core.ActorWeb}
	spec.BudgetUnit = core.BudgetUSD
	sess, err := led.CreateSession(ctx, spec)
	if err != nil {
		t.Fatalf("session: %v", err)
	}

	fl := &scriptedLLM{replies: replies}
	act := &scriptedActor{outcome: outcome, db: db}
	q := queue.New(db, time.Minute)

	return &rig{
		db: db, led: led, q: q, sess: sess, actor: act, llm: fl,
		exec: &executor.Executor{
			Store:   db,
			Ledger:  led,
			Queue:   q,
			Planner: &planner.Planner{LLM: fl, MaxInitialLeads: 3, ReplanEvery: 3, MaxDepth: 2},
			Actors:  map[core.ActorType]actors.Actor{core.ActorWeb: act},
			Log:     slog.New(slog.NewTextHandler(io.Discard, nil)),
			Jitter:  func() float64 { return 0 },
			Sleep:   func(context.Context, time.Duration) error { return nil },
		},
	}
}

func (r *rig) reload(t *testing.T) *core.Session {
	t.Helper()
	var s *core.Session
	if err := r.db.Read(context.Background(), func(ctx context.Context, q store.Queries) error {
		var err error
		s, err = q.GetSession(ctx, r.sess.ID)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	return s
}

// ---------------------------------------------------------------------------
// The invariant: every reservation resolved, on every path
// ---------------------------------------------------------------------------

// TestNothingIsLeftHeldOnAnyPath is the invariant the whole loop is arranged
// around. A settle skipped because a lead failed leaves budget held forever —
// neither spent nor available — and the session runs out of money it never
// used. Failure paths outnumber the success path, so this is checked across all
// of them.
func TestNothingIsLeftHeldOnAnyPath(t *testing.T) {
	cases := map[string]func(int, core.Lead) (*actors.Result, error){
		"all succeed": func(int, core.Lead) (*actors.Result, error) {
			return okResult(2, 20_000), nil
		},
		"all fail with no result": func(int, core.Lead) (*actors.Result, error) {
			return nil, errors.New("dead link")
		},
		"fail but report partial cost": func(int, core.Lead) (*actors.Result, error) {
			return okResult(1, 15_000), errors.New("chunk mining failed")
		},
		"transient then success": func(n int, _ core.Lead) (*actors.Result, error) {
			if n%2 == 0 {
				return okResult(0, 5_000), llm.ErrRateLimited
			}
			return okResult(2, 20_000), nil
		},
		"always transient": func(int, core.Lead) (*actors.Result, error) {
			return okResult(0, 5_000), llm.ErrOverloaded
		},
		"empty results": func(int, core.Lead) (*actors.Result, error) {
			return okResult(0, 10_000), nil
		},
	}

	for name, outcome := range cases {
		r := newRig(t, 5*core.MicrosPerUSD, []string{planJSON("a", "b", "c")}, outcome)
		if _, err := r.exec.Run(context.Background(), r.sess.ID); err != nil {
			t.Fatalf("%s: run: %v", name, err)
		}

		after := r.reload(t)
		if after.Held != 0 {
			t.Errorf("%s: %d still held after the session", name, after.Held)
		}

		v, err := r.led.Verify(context.Background(), r.sess.ID)
		if err != nil {
			t.Fatalf("%s: verify: %v", name, err)
		}
		if !v.Consistent() {
			t.Errorf("%s: ledger does not reconcile: spent %d/%d held %d/%d",
				name, v.SpentRecorded, v.SpentFromLedger, v.HeldRecorded, v.HeldFromRows)
		}
	}
}

// TestBudgetIsNeverOvershot across a whole session, which is §14.3's headline
// contract and the reason reserve-before-dispatch exists.
func TestBudgetIsNeverOvershot(t *testing.T) {
	// Each lead costs far more than the estimator would guess.
	r := newRig(t, core.MicrosPerUSD, []string{planJSON("a", "b", "c")},
		func(int, core.Lead) (*actors.Result, error) { return okResult(1, 300_000), nil })

	if _, err := r.exec.Run(context.Background(), r.sess.ID); err != nil {
		t.Fatal(err)
	}

	after := r.reload(t)
	if after.Spent > after.Budget {
		t.Errorf("spent %d against a %d budget", after.Spent, after.Budget)
	}
}

// TestEscrowSurvivesTheLoop. §8.3 holds escrow back for the report; a loop that
// spent it would leave nothing to generate output with, which is the failure
// escrow exists to prevent.
func TestEscrowSurvivesTheLoop(t *testing.T) {
	r := newRig(t, 2*core.MicrosPerUSD, []string{planJSON("a", "b", "c")},
		func(int, core.Lead) (*actors.Result, error) { return okResult(1, 400_000), nil })

	if _, err := r.exec.Run(context.Background(), r.sess.ID); err != nil {
		t.Fatal(err)
	}

	after := r.reload(t)
	if after.Escrow <= 0 {
		t.Fatal("no escrow was held at all")
	}
	if after.Spent > after.Budget-after.Escrow {
		t.Errorf("spent %d, which eats into the %d escrow", after.Spent, after.Escrow)
	}
}

// ---------------------------------------------------------------------------
// §9.5 error policy
// ---------------------------------------------------------------------------

// TestTransientErrorsAreRetriedThenDegraded, and the session survives.
func TestTransientErrorsAreRetriedThenDegraded(t *testing.T) {
	r := newRig(t, 5*core.MicrosPerUSD, []string{planJSON("only one")},
		func(int, core.Lead) (*actors.Result, error) { return okResult(0, 1_000), llm.ErrRateLimited })

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got := r.actor.count(); got != executor.MaxAttempts {
		t.Errorf("actor ran %d times, want %d attempts", got, executor.MaxAttempts)
	}
	if res.Status == core.StatusFailed {
		t.Error("a transient failure aborted the session; §9.5 says degrade")
	}
	if res.LeadsFailed == 0 {
		t.Error("the exhausted lead was not counted as failed")
	}
}

// TestDegradedErrorsAreNotRetried: a dead link fails identically next time, and
// a retry re-runs the whole lead and is charged again.
func TestDegradedErrorsAreNotRetried(t *testing.T) {
	r := newRig(t, 5*core.MicrosPerUSD, []string{planJSON("only one")},
		func(int, core.Lead) (*actors.Result, error) { return nil, errors.New("extraction failed") })

	if _, err := r.exec.Run(context.Background(), r.sess.ID); err != nil {
		t.Fatal(err)
	}
	if got := r.actor.count(); got != 1 {
		t.Errorf("a degraded error was retried %d times", got)
	}
}

// TestFatalErrorsAbortTheSession. Continuing spends budget on calls that cannot
// succeed — a bad key fails on every lead — and the escrow is preserved so a
// partial report is still affordable (§9.5).
func TestFatalErrorsAbortTheSession(t *testing.T) {

	// One worker: this pins the SERIAL semantics, where a fatal outcome means
	// the next lead never starts. With the default pool a fatal stops the
	// session within one batch instead, which
	// TestAFatalStopsTheSessionWithinOneBatch covers.
	r := newRig(t, 5*core.MicrosPerUSD, []string{planJSON("a", "b", "c")},
		func(int, core.Lead) (*actors.Result, error) { return okResult(0, 1_000), llm.ErrUnauthorized })
	r.exec.Workers = 1

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != core.StatusFailed {
		t.Errorf("status = %s, want failed", res.Status)
	}
	if r.actor.count() != 1 {
		t.Errorf("ran %d leads after a fatal error; it fails identically every time", r.actor.count())
	}
	if after := r.reload(t); after.Escrow <= 0 {
		t.Error("escrow was consumed, so no partial report is affordable")
	}
}

// TestPartialClaimsSurviveAFailedLead. The actor verified every quote it
// returned; discarding them because a later chunk failed throws away evidence
// that was paid for.
func TestPartialClaimsSurviveAFailedLead(t *testing.T) {
	r := newRig(t, 5*core.MicrosPerUSD, []string{planJSON("only one")},
		func(int, core.Lead) (*actors.Result, error) {
			return okResult(3, 10_000), errors.New("the last chunk failed")
		})

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Claims) != 3 {
		t.Errorf("%d claims kept from a partially failed lead, want 3", len(res.Claims))
	}
}

// ---------------------------------------------------------------------------
// Planning behaviour
// ---------------------------------------------------------------------------

// TestPlannerCostIsBoundedByReplansNotLeads is §9.1's whole point. Rev 1 called
// the planner after every lead; if that regressed, planner calls would track
// lead count.
func TestPlannerCostIsBoundedByReplansNotLeads(t *testing.T) {
	replies := []string{
		planJSON("a", "b", "c"), // initial
		planJSON("d", "e", "f"), // replan 1
		planJSON("g", "h", "i"), // replan 2
		`{"done":true}`,         // replan 3 stops
	}
	r := newRig(t, 20*core.MicrosPerUSD, replies,
		func(int, core.Lead) (*actors.Result, error) { return okResult(2, 10_000), nil })

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if res.LeadsRun < 6 {
		t.Fatalf("only %d leads ran; the test needs several to be meaningful", res.LeadsRun)
	}
	// ReplanEvery is 3, so planner calls must be far below lead count.
	if r.llm.calls > res.LeadsRun {
		t.Errorf("%d planner calls for %d leads — planning is not batched",
			r.llm.calls, res.LeadsRun)
	}
}

// TestSessionStopsWhenThePlannerSaysDone rather than looping until the budget
// runs out.
func TestSessionStopsWhenThePlannerSaysDone(t *testing.T) {
	r := newRig(t, 50*core.MicrosPerUSD, []string{planJSON("a"), `{"done":true}`},
		func(int, core.Lead) (*actors.Result, error) { return okResult(1, 1_000), nil })

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != core.StatusDone {
		t.Errorf("status = %s, want done", res.Status)
	}

	after := r.reload(t)
	if after.Spent > after.Budget/2 {
		t.Errorf("spent %d of %d after being told to stop", after.Spent, after.Budget)
	}
}

// TestDigestRecordsCoverage so the planner can see which questions produced
// evidence and which did not.
func TestDigestRecordsCoverage(t *testing.T) {
	r := newRig(t, 20*core.MicrosPerUSD, []string{planJSON("productive", "barren"), `{"done":true}`},
		func(n int, lead core.Lead) (*actors.Result, error) {
			if lead.Query == "barren" {
				return okResult(0, 5_000), nil
			}
			return okResult(4, 10_000), nil
		})

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	out := res.Digest.String()
	if !strings.Contains(out, "4 claim(s) found") {
		t.Errorf("productive question shows no yield:\n%s", out)
	}
	if !strings.Contains(out, "no_evidence") {
		t.Errorf("the barren question was not recorded as a dead end:\n%s", out)
	}
}

// TestPlannerCostIsChargedToThePlannerRole, so §14.3's role breakdown can show
// what planning cost — the number that says whether the digest is working.
func TestPlannerCostIsChargedToThePlannerRole(t *testing.T) {
	r := newRig(t, 20*core.MicrosPerUSD, []string{planJSON("a"), `{"done":true}`},
		func(int, core.Lead) (*actors.Result, error) { return okResult(1, 10_000), nil })

	if _, err := r.exec.Run(context.Background(), r.sess.ID); err != nil {
		t.Fatal(err)
	}

	var byRole map[core.Role]core.Cost
	if err := r.db.Read(context.Background(), func(ctx context.Context, q store.Queries) error {
		var err error
		byRole, err = q.SumCostsByRole(ctx, r.sess.ID)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if byRole[core.RolePlanner].TotalTokens() == 0 {
		t.Errorf("no planner cost recorded; role breakdown = %+v", byRole)
	}
	if byRole[core.RoleExecutor].TotalTokens() == 0 {
		t.Error("no executor cost recorded")
	}
}

// ---------------------------------------------------------------------------
// Ceilings
// ---------------------------------------------------------------------------

// TestMaxLeadsCeilingStopsTheLoop. Unit-independent ceilings (§8.5) bind even
// when the cost estimate is wrong, which is the case they exist for: a budget
// large enough to run forever must still stop.
func TestMaxLeadsCeilingStopsTheLoop(t *testing.T) {
	// Budget for hundreds of leads, a ceiling of 2, and a planner that keeps
	// proposing more.
	replies := []string{
		planJSON("a", "b", "c"), planJSON("d", "e", "f"),
		planJSON("g", "h", "i"), planJSON("j", "k", "l"),
	}
	r := newRigWithCeilings(t, 1000*core.MicrosPerUSD, 2, 500, replies,
		func(int, core.Lead) (*actors.Result, error) { return okResult(1, 1_000), nil })

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != core.StatusExhausted {
		t.Errorf("status = %s, want budget_exhausted (the ceiling stopped it)", res.Status)
	}
	if res.StoppedBecause != "max_leads" {
		t.Errorf("stopped because %q, want max_leads", res.StoppedBecause)
	}

	after := r.reload(t)
	if after.LeadCount > after.MaxLeads {
		t.Errorf("ran %d leads against a ceiling of %d", after.LeadCount, after.MaxLeads)
	}
	if after.Spent > after.Budget/10 {
		t.Errorf("spent %d of %d — the ceiling did not stop the spend", after.Spent, after.Budget)
	}
}

// TestCancellationStopsCleanly and still leaves the ledger consistent.
func TestCancellationStopsCleanly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	r := newRig(t, 20*core.MicrosPerUSD, []string{planJSON("a", "b", "c")},
		func(n int, _ core.Lead) (*actors.Result, error) {
			if n == 1 {
				cancel()
			}
			return okResult(1, 5_000), nil
		})

	res, err := r.exec.Run(ctx, r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != core.StatusCancelled {
		t.Errorf("status = %s, want cancelled", res.Status)
	}

	v, err := r.led.Verify(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if !v.Consistent() {
		t.Error("the ledger does not reconcile after cancellation")
	}
	if after := r.reload(t); after.Held != 0 {
		t.Errorf("%d held after cancellation", after.Held)
	}
}

// TestNoActorRegisteredIsFatal rather than an infinite loop of leads nothing
// can run.
func TestNoActorRegisteredIsFatal(t *testing.T) {
	r := newRig(t, 5*core.MicrosPerUSD, []string{planJSON("a")},
		func(int, core.Lead) (*actors.Result, error) { return okResult(1, 1_000), nil })
	r.exec.Actors = map[core.ActorType]actors.Actor{} // none

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != core.StatusFailed {
		t.Errorf("status = %s, want failed", res.Status)
	}
}

// TestTokenModeChargesTokensNotDollars. The two units are not interchangeable:
// in token mode a search call priced only in dollars contributes nothing, which
// is a real property of the mode (§8.5) and not a bug — but the model tokens
// must land, or the ceiling is enforced against zero.
//
// Ported from the single-lead CLI test the executor replaced.
func TestTokenModeChargesTokensNotDollars(t *testing.T) {
	ctx := context.Background()
	db, err := sqlite.Open(filepath.Join(t.TempDir(), "t.db"), sqlite.Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.Migrate(ctx); err != nil {
		t.Fatal(err)
	}

	led := budget.New(db, budget.DefaultConfig())
	sess, err := led.CreateSession(ctx, budget.SessionSpec{
		Prompt: "q", Mode: core.ModeReport,
		ActorTypes: []core.ActorType{core.ActorWeb},
		BudgetUnit: core.BudgetTokens, Budget: 200_000,
		MaxLeads: 50, MaxToolCalls: 200,
	})
	if err != nil {
		t.Fatal(err)
	}

	fl := &scriptedLLM{replies: []string{planJSON("a"), `{"done":true}`}}
	act := &scriptedActor{outcome: func(int, core.Lead) (*actors.Result, error) {
		return &actors.Result{
			Costs: []core.ToolCall{
				// Priced in dollars only: contributes nothing in token mode.
				{Role: core.RoleExecutor, Type: core.CallSearch, Cost: core.Cost{USDMicros: 5_000}},
				{Role: core.RoleExecutor, Type: core.CallLLM,
					Cost: core.Cost{USDMicros: 41_000, InputTokens: 12_000, OutputTokens: 900}},
			},
		}, nil
	}}

	e := &executor.Executor{
		Store: db, Ledger: led, Queue: queue.New(db, time.Minute),
		Planner: &planner.Planner{LLM: fl, MaxInitialLeads: 1, MaxDepth: 1},
		Actors:  map[core.ActorType]actors.Actor{core.ActorWeb: act},
		Log:     slog.New(slog.NewTextHandler(io.Discard, nil)),
		Sleep:   func(context.Context, time.Duration) error { return nil },
	}
	if _, err := e.Run(ctx, sess.ID); err != nil {
		t.Fatal(err)
	}

	var after *core.Session
	if err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		after, err = q.GetSession(ctx, sess.ID)
		return err
	}); err != nil {
		t.Fatal(err)
	}

	// 12900 from the actor's model call, plus the planner's own tokens. The
	// dollar-only search call must contribute none of it.
	if after.Spent < 12_900 {
		t.Errorf("spent %d tokens, want at least the 12900 the model reported", after.Spent)
	}
	if after.Spent > 100_000 {
		t.Errorf("spent %d tokens — dollar amounts leaked into a token budget", after.Spent)
	}
}

// TestReservationIsClampedToWhatRemains. A budget smaller than the estimator's
// seed should buy a small run, not an error. Escrow is already held back, so
// the clamp has to respect Available rather than Budget.
//
// Ported from the single-lead CLI test the executor replaced.
func TestReservationIsClampedToWhatRemains(t *testing.T) {
	// Far below the estimator seed for a web lead.
	r := newRig(t, 30_000, []string{planJSON("a"), `{"done":true}`},
		func(int, core.Lead) (*actors.Result, error) { return okResult(1, 1_000), nil })

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatalf("a small budget failed instead of buying a small run: %v", err)
	}
	if res.LeadsRun == 0 {
		t.Error("no lead ran at all on a small budget")
	}

	after := r.reload(t)
	if after.Spent > after.Budget {
		t.Errorf("spent %d against a %d budget", after.Spent, after.Budget)
	}
	if after.Held != 0 {
		t.Errorf("%d held after a clamped run", after.Held)
	}
}

// TestCachedLeadReturnsAResultRatherThanSkipping is §9.3's correctness fix, and
// the reason the cache is result-returning rather than a skip list.
//
// Rev 1 did `if cache.SeenRecently(lead) { continue }`. The planner asked for
// something and got nothing back, so the next replan spawned an equivalent lead
// and the loop livelocked. Recording the coverage is what closes it: the planner
// learns the question was answered.
func TestCachedLeadReturnsAResultRatherThanSkipping(t *testing.T) {
	// A planner that keeps proposing the SAME question — exactly the livelock
	// shape. Without a result-returning cache this never converges.
	replies := []string{
		planJSON("the same question"),
		planJSON("the same question"),
		planJSON("the same question"),
		planJSON("the same question"),
		`{"done":true}`,
	}
	r := newRig(t, 50*core.MicrosPerUSD, replies,
		func(int, core.Lead) (*actors.Result, error) { return okResult(3, 10_000), nil })
	r.exec.Cache = cache.New()

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}

	// The actor must run the question once, not once per proposal.
	if got := r.actor.count(); got != 1 {
		t.Errorf("the actor ran %d times for one repeated question, want 1", got)
	}
	if res.LeadsCached == 0 {
		t.Error("no lead was served from cache")
	}
	// The precise assertion: every cached hit must ADD its coverage to the
	// digest. A substring check on "claim(s) found" is vacuous — every open
	// question prints that line whether or not anything was found — so count
	// instead. One real run plus N cached hits at 3 claims each.
	//
	// This is what distinguishes a result-returning cache from a skip list. A
	// skip records nothing, so the total stays at the single real run's 3, the
	// planner never learns the question was answered, and it keeps proposing it.
	wantClaims := 3 * (1 + res.LeadsCached)
	if res.Digest.ClaimsFound != wantClaims {
		t.Errorf("digest recorded %d claims across %d cached lead(s) and 1 real one, want %d — "+
			"a cache hit is not reporting its coverage",
			res.Digest.ClaimsFound, res.LeadsCached, wantClaims)
	}
}

// TestCachedLeadsCostNothing. A hit that charged would make the cache a
// rounding error rather than a saving.
func TestCachedLeadsCostNothing(t *testing.T) {
	replies := []string{planJSON("repeat me"), planJSON("repeat me"), `{"done":true}`}
	r := newRig(t, 50*core.MicrosPerUSD, replies,
		func(int, core.Lead) (*actors.Result, error) { return okResult(2, 100_000), nil })
	r.exec.Cache = cache.New()

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if res.LeadsCached == 0 {
		t.Fatal("nothing was cached; the test asserts nothing")
	}

	// One actor run at 100k plus planner calls. A second charged run would
	// double the executor's share.
	var byRole map[core.Role]core.Cost
	if err := r.db.Read(context.Background(), func(ctx context.Context, q store.Queries) error {
		var err error
		byRole, err = q.SumCostsByRole(ctx, r.sess.ID)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if got := byRole[core.RoleExecutor].USDMicros; got != 100_000 {
		t.Errorf("executor spend = %d, want exactly one lead's 100000", got)
	}
}

// TestCachedLeadIsMarkedAsCached, so a trace can tell a cache hit from work.
func TestCachedLeadIsMarkedAsCached(t *testing.T) {
	replies := []string{planJSON("same"), planJSON("same"), `{"done":true}`}
	r := newRig(t, 50*core.MicrosPerUSD, replies,
		func(int, core.Lead) (*actors.Result, error) { return okResult(1, 1_000), nil })
	r.exec.Cache = cache.New()

	if _, err := r.exec.Run(context.Background(), r.sess.ID); err != nil {
		t.Fatal(err)
	}

	stats, err := r.q.Stats(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Cached == 0 {
		t.Errorf("no lead recorded as skipped_cached: %+v", stats)
	}
}

// TestNoCacheStillWorks: the cache is optional, and a nil one must not change
// behaviour beyond costing more.
func TestNoCacheStillWorks(t *testing.T) {
	replies := []string{planJSON("same"), planJSON("same"), `{"done":true}`}
	r := newRig(t, 50*core.MicrosPerUSD, replies,
		func(int, core.Lead) (*actors.Result, error) { return okResult(1, 1_000), nil })
	r.exec.Cache = nil

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if res.LeadsCached != 0 {
		t.Error("a nil cache reported hits")
	}
	if r.actor.count() < 2 {
		t.Error("without a cache the repeated question should be researched twice")
	}
}

// TestCachedDeadEndStaysADeadEnd. A question that found nothing must not read
// as answered on the second ask, or the planner stops probing a real gap.
func TestCachedDeadEndStaysADeadEnd(t *testing.T) {
	replies := []string{planJSON("barren"), planJSON("barren"), `{"done":true}`}
	r := newRig(t, 50*core.MicrosPerUSD, replies,
		func(int, core.Lead) (*actors.Result, error) { return okResult(0, 1_000), nil })
	r.exec.Cache = cache.New()

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	// Count, not substring: the first uncached run already records one
	// no_evidence, so its mere presence proves nothing about the cached ones.
	var deadEnds int
	for _, de := range res.Digest.DeadEnds {
		if de.Cause == "no_evidence" {
			deadEnds = de.Count
		}
	}
	if want := 1 + res.LeadsCached; deadEnds != want {
		t.Errorf("no_evidence recorded %d times across %d cached lead(s) and 1 real one, want %d",
			deadEnds, res.LeadsCached, want)
	}
}

// TestOneLeadCannotSpendTheWholeBudget is the bound §8.2 claims and §14.3
// requires ("max overshoot must be ~0").
//
// Before the sub-budget existed, the reservation bounded nothing: the actor's
// ceiling came from config and bore no relation to the money held for it. A
// single lead costing five times the budget spent all of it — measured at 5.0x
// with the escrow consumed — while TestBudgetIsNeverOvershot passed because its
// parameters stopped short of the boundary.
//
// A fake actor cannot be made to honour a sub-budget it does not read, so this
// asserts the OTHER half: the settle-time factor check stops the session
// instead of letting the pattern repeat on every remaining lead.
func TestOneLeadCannotSpendTheWholeBudget(t *testing.T) {

	// One worker: this pins the SERIAL semantics, where a fatal outcome means
	// the next lead never starts. With the default pool a fatal stops the
	// session within one batch instead, which
	// TestAFatalStopsTheSessionWithinOneBatch covers.
	// $1.00 budget; a lead that ignores its ceiling and spends $5.00.
	r := newRig(t, core.MicrosPerUSD, []string{planJSON("a", "b", "c"), `{"done":true}`},
		func(int, core.Lead) (*actors.Result, error) {
			return okResult(1, 5*core.MicrosPerUSD), nil
		})
	r.exec.Workers = 1

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}

	// The run must stop on the first offending lead rather than repeating it.
	if r.actor.count() != 1 {
		t.Errorf("the actor ran %d times after blowing its reservation, want 1", r.actor.count())
	}
	if res.Status != core.StatusFailed {
		t.Errorf("status = %s, want failed — an unbounded overshoot is not a clean stop", res.Status)
	}
	if !strings.Contains(res.StoppedBecause, "overshoot") {
		t.Errorf("stopped because %q, want it to name the overshoot", res.StoppedBecause)
	}

	after := r.reload(t)
	if after.Held != 0 {
		t.Errorf("%d held after an overshoot", after.Held)
	}
	v, _ := r.led.Verify(context.Background(), r.sess.ID)
	if !v.Consistent() {
		t.Error("the ledger does not reconcile after an overshoot")
	}
}

// TestSubBudgetIsDerivedFromTheReservation. The actor has to be TOLD what it
// may spend; a ceiling from config cannot know what was reserved.
func TestSubBudgetIsDerivedFromTheReservation(t *testing.T) {
	var seen []actors.Budget

	r := newRig(t, 200_000, []string{planJSON("a"), `{"done":true}`}, nil)
	r.exec.Actors = map[core.ActorType]actors.Actor{
		core.ActorWeb: &budgetSpy{seen: &seen},
	}
	r.exec.CheapModel = "claude-haiku-4-5"

	if _, err := r.exec.Run(context.Background(), r.sess.ID); err != nil {
		t.Fatal(err)
	}
	if len(seen) == 0 {
		t.Fatal("the actor never ran")
	}
	for i, b := range seen {
		if b.MaxInputTokens <= 0 {
			t.Errorf("lead %d ran with no input ceiling: %+v", i, b)
		}
	}
}

// TestTokenModeSubBudgetIsExact: in token mode the reservation IS a token
// count, so no conversion is involved and the ceiling must track it directly.
func TestTokenModeSubBudgetIsExact(t *testing.T) {
	ctx := context.Background()
	db, err := sqlite.Open(filepath.Join(t.TempDir(), "t.db"), sqlite.Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.Migrate(ctx); err != nil {
		t.Fatal(err)
	}

	led := budget.New(db, budget.DefaultConfig())
	sess, err := led.CreateSession(ctx, budget.SessionSpec{
		Prompt: "q", Mode: core.ModeReport,
		ActorTypes: []core.ActorType{core.ActorWeb},
		BudgetUnit: core.BudgetTokens, Budget: 500_000,
		MaxLeads: 10, MaxToolCalls: 100,
	})
	if err != nil {
		t.Fatal(err)
	}

	var seen []actors.Budget
	fl := &scriptedLLM{replies: []string{planJSON("a"), `{"done":true}`}}
	e := &executor.Executor{
		Store: db, Ledger: led, Queue: queue.New(db, time.Minute),
		Planner: &planner.Planner{LLM: fl, MaxInitialLeads: 1, MaxDepth: 1},
		Actors:  map[core.ActorType]actors.Actor{core.ActorWeb: &budgetSpy{seen: &seen}},
		Log:     slog.New(slog.NewTextHandler(io.Discard, nil)),
		Sleep:   func(context.Context, time.Duration) error { return nil },
	}
	if _, err := e.Run(ctx, sess.ID); err != nil {
		t.Fatal(err)
	}
	if len(seen) == 0 {
		t.Fatal("the actor never ran")
	}
	// The estimator seed for a web lead in token mode; the ceiling must be a
	// fraction of it, not unrelated to it.
	got := seen[0].MaxInputTokens
	if got <= 0 || got > 500_000 {
		t.Errorf("sub-budget = %d tokens against a 500000 budget — not derived from the reservation", got)
	}
}

// budgetSpy records the sub-budget it was handed.
type budgetSpy struct{ seen *[]actors.Budget }

func (b *budgetSpy) Type() core.ActorType { return core.ActorWeb }
func (b *budgetSpy) Run(ctx context.Context, lead core.Lead) (*actors.Result, error) {
	sub, _ := actors.SubBudgetFrom(ctx)
	*b.seen = append(*b.seen, sub)
	return okResult(1, 1_000), nil
}

// TestFinalReplanFailureDoesNotFailTheSession. §9.5 classifies 429/5xx/timeout
// as transient — never abort. This path used to return, so a complete run with
// claims collected reported as failed with a non-zero exit because its last
// planner call got throttled.
func TestFinalReplanFailureDoesNotFailTheSession(t *testing.T) {
	// One question, then the drain-time replan errors.
	fl := &scriptedLLM{replies: []string{planJSON("only one")}}
	fl.errAfter = 1

	r := newRig(t, 20*core.MicrosPerUSD, nil,
		func(int, core.Lead) (*actors.Result, error) { return okResult(3, 10_000), nil })
	r.exec.Planner = &planner.Planner{LLM: fl, MaxInitialLeads: 1, ReplanEvery: 99, MaxDepth: 2}

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatalf("a throttled final replan returned an error: %v", err)
	}
	if res.Status == core.StatusFailed {
		t.Errorf("status = failed after collecting %d claims; §9.5 says degrade", len(res.Claims))
	}
	if len(res.Claims) != 3 {
		t.Errorf("%d claims survived, want 3", len(res.Claims))
	}
	// The spend read must still happen — the early return used to skip it.
	if res.Spent == 0 {
		t.Error("Spent is 0 on a run that charged for a lead and two planner calls")
	}
	if !strings.Contains(res.StoppedBecause, "replan") {
		t.Errorf("stopped because %q, want it to name the replan", res.StoppedBecause)
	}
}

// TestLeaseIsHeartbeatedWhileALeadRuns. §9.4's "workers heartbeat their lease".
// Renew used to be called only BETWEEN retries, so a lead that legitimately
// outlasted the TTL — a slow fetch plus a slow model call — lost its lease and
// became sweepable while still running.
//
// The obvious version of this test could not fail: nothing sweeps during a run,
// so an expired lease is simply never observed and deleting the heartbeat left
// it green. A concurrent sweeper is what makes the expiry visible, and it is
// also the real scenario — a second process starting up, or M5's periodic
// recovery.
func TestLeaseIsHeartbeatedWhileALeadRuns(t *testing.T) {
	ctx := context.Background()
	db, err := sqlite.Open(filepath.Join(t.TempDir(), "t.db"), sqlite.Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.Migrate(ctx); err != nil {
		t.Fatal(err)
	}

	led := budget.New(db, budget.DefaultConfig())
	sess, err := led.CreateSession(ctx, budget.SessionSpec{
		Prompt: "q", Mode: core.ModeReport,
		ActorTypes: []core.ActorType{core.ActorWeb},
		BudgetUnit: core.BudgetUSD, Budget: 10 * core.MicrosPerUSD,
		MaxLeads: 5, MaxToolCalls: 50,
	})
	if err != nil {
		t.Fatal(err)
	}

	// A TTL short enough that a lead outlasts several of them, and a sweeper
	// running throughout — which is what turns a lost lease into an observable
	// re-dispatch.
	//
	// Not as short as it could be. The heartbeat renews every TTL/3, so the
	// margin is how long the runtime may stall that goroutine before the lease
	// is gone; at 90ms that was 90ms, and a shared 2-core CI runner missed it
	// often enough to fail a release. 450ms tests the same property with a
	// margin the scheduler cannot casually exceed.
	const ttl = 450 * time.Millisecond
	q := queue.New(db, ttl)
	sweeper := queue.New(db, ttl)

	stop := make(chan struct{})
	var swept atomic.Int64
	go func() {
		tick := time.NewTicker(ttl / 4)
		defer tick.Stop()
		for {
			select {
			case <-stop:
				return
			case <-tick.C:
				if n, err := sweeper.Sweep(context.Background(), sess.ID); err == nil {
					swept.Add(int64(n))
				}
			}
		}
	}()

	fl := &scriptedLLM{replies: []string{planJSON("slow"), `{"done":true}`}}
	slow := &scriptedActor{outcome: func(int, core.Lead) (*actors.Result, error) {
		// Several TTLs, so the lease must be renewed more than once.
		time.Sleep(3 * ttl)
		return okResult(1, 1_000), nil
	}}

	e := &executor.Executor{
		Store: db, Ledger: led, Queue: q,
		Planner: &planner.Planner{LLM: fl, MaxInitialLeads: 1, MaxDepth: 1},
		Actors:  map[core.ActorType]actors.Actor{core.ActorWeb: slow},
		Log:     slog.New(slog.NewTextHandler(io.Discard, nil)),
		Sleep:   func(context.Context, time.Duration) error { return nil },
	}

	res, err := e.Run(ctx, sess.ID)
	close(stop)
	if err != nil {
		t.Fatal(err)
	}

	if n := swept.Load(); n != 0 {
		t.Errorf("the sweeper reclaimed %d lease(s) from a running lead — it was not heartbeated", n)
	}
	if slow.count() != 1 {
		t.Errorf("the actor ran %d times; a swept lease means the lead is re-dispatched and re-charged", slow.count())
	}
	if res.LeadsFailed != 0 {
		t.Errorf("%d leads failed; a lead outlasting the TTL lost its lease", res.LeadsFailed)
	}
}

// TestSpentAndCacheStatsAreReported. The three-line comment on the final read
// says it exists for cancelled and timed-out runs — "exactly the runs whose
// spend most needs reporting" — and deleting the read left the suite green,
// including the cancellation test.
func TestSpentAndCacheStatsAreReported(t *testing.T) {
	replies := []string{planJSON("same"), planJSON("same"), `{"done":true}`}
	r := newRig(t, 20*core.MicrosPerUSD, replies,
		func(int, core.Lead) (*actors.Result, error) { return okResult(2, 25_000), nil })
	r.exec.Cache = cache.New()

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if res.Spent == 0 {
		t.Error("Spent is 0 on a run that charged for leads and planner calls")
	}
	if after := r.reload(t); res.Spent != after.Spent {
		t.Errorf("Spent = %d but the ledger says %d", res.Spent, after.Spent)
	}
	if res.CacheStats.Hits == 0 && res.LeadsCached > 0 {
		t.Error("leads were served from cache but CacheStats reports no hits")
	}
}

// TestSpentIsReportedOnACancelledRun is the case the comment names.
func TestSpentIsReportedOnACancelledRun(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	r := newRig(t, 20*core.MicrosPerUSD, []string{planJSON("a", "b", "c")},
		func(n int, _ core.Lead) (*actors.Result, error) {
			if n == 1 {
				cancel()
			}
			return okResult(1, 30_000), nil
		})

	res, err := r.exec.Run(ctx, r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != core.StatusCancelled {
		t.Fatalf("status = %s, want cancelled", res.Status)
	}
	if res.Spent == 0 {
		t.Error("Spent is 0 on a cancelled run that charged for two leads")
	}
	if after := r.reload(t); res.Spent != after.Spent {
		t.Errorf("Spent = %d but the ledger says %d", res.Spent, after.Spent)
	}
	// And the leads that ran must be recorded as finished, not left leased.
	stats, err := r.q.Stats(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Leased != 0 {
		t.Errorf("%d leads left leased after cancellation; they ran and were paid for", stats.Leased)
	}
}

// TestInterruptDuringInitialPlanningIsCancelledNotFailed. With no plan there is
// nothing to research either way, but WHY the session ended is not the same
// thing: a deliberate Ctrl-C reported as a failure looks like a crash to
// whoever reads the status or the exit code.
//
// Found on a real run — a local model too slow to finish, interrupted, and the
// session recorded as failed.
func TestInterruptDuringInitialPlanningIsCancelledNotFailed(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	r := newRig(t, 20*core.MicrosPerUSD, nil, nil)
	// A planner whose call is interrupted mid-flight, exactly as an HTTP round
	// trip is when the signal arrives.
	r.exec.Planner = &planner.Planner{LLM: &cancellingLLM{cancel: cancel}, MaxDepth: 1}

	res, err := r.exec.Run(ctx, r.sess.ID)
	if err != nil {
		t.Fatalf("run returned an error rather than a status: %v", err)
	}
	if res.Status != core.StatusCancelled {
		t.Errorf("status = %s, want cancelled — an interrupt is not a failure", res.Status)
	}

	// And nothing may be left held.
	if after := r.reload(t); after.Held != 0 {
		t.Errorf("%d held after an interrupted plan", after.Held)
	}
	v, _ := r.led.Verify(context.Background(), r.sess.ID)
	if !v.Consistent() {
		t.Error("the ledger does not reconcile after an interrupted plan")
	}
}

// cancellingLLM cancels the context and then fails, which is what an
// interrupted HTTP round trip looks like from the caller's side.
type cancellingLLM struct{ cancel context.CancelFunc }

func (c *cancellingLLM) Name() string               { return "cancelling" }
func (c *cancellingLLM) ModelFor(t llm.Tier) string { return "m" }
func (c *cancellingLLM) Complete(ctx context.Context, req llm.Request) (*llm.Response, error) {
	c.cancel()
	return &llm.Response{Model: "m", Usage: llm.Usage{InputTokens: 100}},
		fmt.Errorf("Post \"http://127.0.0.1:11434/v1/chat/completions\": %w", context.Canceled)
}

// TestProgressReportsEachPhase. Planning is a single model call with no output
// until it returns, and on a local model that is minutes of silence — three real
// runs were killed by hand because the CLI printed a header and then nothing.
func TestProgressReportsEachPhase(t *testing.T) {
	var phases []string

	replies := []string{planJSON("a", "b", "c"), planJSON("d"), `{"done":true}`}
	r := newRig(t, 50*core.MicrosPerUSD, replies,
		func(int, core.Lead) (*actors.Result, error) { return okResult(2, 5_000), nil })
	r.exec.Progress = func(ev executor.Event) { phases = append(phases, ev.Phase) }

	if _, err := r.exec.Run(context.Background(), r.sess.ID); err != nil {
		t.Fatal(err)
	}

	// Planning must be announced BEFORE anything else, because it is the phase
	// with no output of its own.
	if len(phases) == 0 || phases[0] != "planning" {
		t.Fatalf("first event = %v, want planning first", phases)
	}
	for _, want := range []string{"planning", "executing", "lead", "lead-done", "replanning"} {
		if !slices.Contains(phases, want) {
			t.Errorf("no %q event was reported: %v", want, phases)
		}
	}
}

// TestProgressIsOptional: a nil callback must not panic, since every non-CLI
// caller leaves it unset.
func TestProgressIsOptional(t *testing.T) {
	r := newRig(t, 20*core.MicrosPerUSD, []string{planJSON("a"), `{"done":true}`},
		func(int, core.Lead) (*actors.Result, error) { return okResult(1, 1_000), nil })
	r.exec.Progress = nil

	if _, err := r.exec.Run(context.Background(), r.sess.ID); err != nil {
		t.Fatal(err)
	}
}

// ---------------------------------------------------------------------------
// The depth cap is a free decision, and it says so
// ---------------------------------------------------------------------------

// TestDepthCapIsReportedAsTheDepthCap checks the cap is taken before budget is
// reserved.
//
// Replan's own depth check runs INSIDE the call the executor has already
// reserved for. It returns before reaching the provider, so no tokens are wasted
// — but the reservation is, and that is not the expensive part. When a ceiling
// fires in the same moment, the reserve is REFUSED, replan returns an error, and
// the session reports "final replan failed" or the ceiling as its stop reason
// rather than the cap that actually ended the research. Wrong explanation for the
// right outcome, in a field §14.3 reports on.
func TestDepthCapIsReportedAsTheDepthCap(t *testing.T) {
	// Two rounds of one lead each: the initial plan, one replan that adds work
	// and takes depth to the cap, then a third replan that must short-circuit.
	r := newRig(t, 2_000_000, []string{
		planJSON("first question"),
		planJSON("second question"),
	}, func(int, core.Lead) (*actors.Result, error) {
		return okResult(1, 10_000), nil
	})
	r.exec.Planner = &planner.Planner{LLM: r.llm, MaxInitialLeads: 1, ReplanEvery: 1, MaxDepth: 1}

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	if want := "lead tree reached the depth cap of 1"; res.StoppedBecause != want {
		t.Errorf("StoppedBecause = %q, want %q", res.StoppedBecause, want)
	}
	// One decomposition plus one replan, and no reservation taken for a third.
	if got := r.llm.count(); got != 2 {
		t.Errorf("model calls = %d, want 2 (decompose + one replan)", got)
	}
	// And the cap really is what stopped it, not an exhausted budget.
	if res.Status == core.StatusExhausted {
		t.Errorf("status = %v: budget ran out, so this proves nothing about the cap", res.Status)
	}
}

// ---------------------------------------------------------------------------
// The planner is told how much allowance is left
// ---------------------------------------------------------------------------

// TestReplanIsToldWhatIsLeft is the wiring half of the budget signal.
//
// §9.1's replan prompt instructs the planner to set done when the open
// sub-questions are "not worth more budget", and the digest reported nothing
// whatsoever about budget — an instruction with no answerable input. A digest
// that CAN carry the figure is worth nothing if the executor never sets it, so
// this asserts on the prompt the provider actually received.
//
// MaxLeads is the tightest limit here rather than spend, which also pins
// RemainingFraction taking the minimum: 4 leads allowed means 75% left after
// one, whatever the dollars say.
func TestReplanIsToldWhatIsLeft(t *testing.T) {
	r := newRigWithCeilings(t, 2_000_000, 4, 200, []string{
		planJSON("first question"),
		planJSON("second question"),
		planJSON("third question"),
	}, func(int, core.Lead) (*actors.Result, error) {
		return okResult(1, 10_000), nil
	})
	r.exec.Planner = &planner.Planner{LLM: r.llm, MaxInitialLeads: 1, ReplanEvery: 1, MaxDepth: 2}

	if _, err := r.exec.Run(context.Background(), r.sess.ID); err != nil {
		t.Fatalf("run: %v", err)
	}

	pct := regexp.MustCompile(`Roughly (\d+)% of the session's allowance remains`)
	var reported []int
	for _, p := range r.llm.seen() {
		if !strings.Contains(p, "Decide what to research next") {
			continue // the initial decomposition carries no digest
		}
		m := pct.FindStringSubmatch(p)
		if m == nil {
			t.Fatalf("replan prompt carries no budget figure:\n%s", p)
		}
		n, _ := strconv.Atoi(m[1])
		reported = append(reported, n)
	}

	// Exact, because "it decreased" is satisfied by a stale figure too: with the
	// reload removed this reports [85 75] — one lead behind throughout, and the
	// first entry falls back to the dollar term because a stale LeadCount of 0
	// makes the lead ceiling look untouched. 3/4 then 2/4 leads left is the truth.
	if len(reported) != 2 || reported[0] != 75 || reported[1] != 50 {
		t.Errorf("allowance reported as %v across replans, want [75 50]: either the "+
			"figure is stale, or it is not the minimum over every limit", reported)
	}
}

// ---------------------------------------------------------------------------
// Verification as a loop stage (§11.1, §11.4)
// ---------------------------------------------------------------------------

// verifierLLM answers planning calls from a script and adjudication calls with one
// relation, so a test can drive both halves of the loop from one provider.
type verifierLLM struct {
	mu       sync.Mutex
	calls    int
	plans    []string
	relation string
	adjudged int
	// strictModel's adjudications answer strictRelation and are counted in strictCalls.
	strictModel    string
	strictRelation string
	strictCalls    int
	// strictErr, when set, fails every strictModel adjudication.
	strictErr error
}

func (v *verifierLLM) Name() string               { return "fake" }
func (v *verifierLLM) ModelFor(t llm.Tier) string { return "fake-model" }
func (v *verifierLLM) Complete(_ context.Context, req llm.Request) (*llm.Response, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	var prompt string
	if len(req.Messages) > 0 {
		prompt = req.Messages[0].Text
	}
	resp := &llm.Response{Model: "fake-model", Usage: llm.Usage{InputTokens: 300, OutputTokens: 60}}

	// Adjudication prompts are the ones carrying a pairs block. Keyed to the fence
	// tag rather than a sentence: this matched "how claim A relates to claim B" until
	// a reflow put a newline mid-phrase, and three tests then reported that the
	// verifier never adjudicated at all when in fact only the fixture had stopped
	// recognising it. The tag is structural, so prose edits cannot silently unhook it.
	if strings.Contains(prompt, "<pairs-") {
		v.adjudged++
		rel := v.relation
		if v.strictModel != "" && req.Model == v.strictModel {
			v.strictCalls++
			rel = v.strictRelation
			if v.strictErr != nil {
				return resp, v.strictErr
			}
		}
		var b strings.Builder
		b.WriteString(`{"verdicts":[`)
		n := strings.Count(prompt, `"b":`)
		for i := 1; i <= n; i++ {
			if i > 1 {
				b.WriteString(",")
			}
			fmt.Fprintf(&b, `{"pair":%d,"relation":%q,"confidence":0.9,"why":"because"}`, i, rel)
		}
		b.WriteString("]}")
		resp.Text = b.String()
		return resp, nil
	}

	i := v.calls
	v.calls++
	resp.Text = `{"done":true}`
	if i < len(v.plans) {
		resp.Text = v.plans[i]
	}
	return resp, nil
}

func (v *verifierLLM) strictAdjudications() int {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.strictCalls
}

func (v *verifierLLM) adjudications() int {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.adjudged
}

// withVerifier attaches a Verifier driven by the same provider.
func (r *rig) withVerifier(fl llm.Provider) *verifier.Verifier {
	v := &verifier.Verifier{
		Store: r.db, Ledger: r.led, LLM: fl,
		Log: slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
	r.exec.Verifier = v
	return v
}

// claimsFrom makes an actor outcome producing two claims that contradict each other
// on their face, from two different publishers — the cross-lead case §11.1 says rev
// 1 structurally could not see.
func opposingClaims(n int, lead core.Lead) (*actors.Result, error) {
	root := ""
	if lead.RootClaimID != nil {
		root = *lead.RootClaimID
	}
	return &actors.Result{
		Summary: "a summary",
		Costs: []core.ToolCall{{
			Role: core.RoleExecutor, Type: core.CallLLM, Model: "fake-model",
			Cost: core.Cost{USDMicros: 8_000, InputTokens: 800, OutputTokens: 80},
		}},
		Claims: []core.Claim{
			{
				Text:        fmt.Sprintf("Subword tokenization improves accuracy, reading %d.", n),
				Source:      fmt.Sprintf("https://pro%d.example/p", n),
				Quote:       "a quote long enough to constitute real evidence",
				RootClaimID: root, VerifyDepth: lead.VerifyDepth,
			},
			{
				Text:        fmt.Sprintf("Subword tokenization does not improve accuracy, reading %d.", n),
				Source:      fmt.Sprintf("https://con%d.example/p", n),
				Quote:       "another quote long enough to constitute real evidence",
				RootClaimID: root, VerifyDepth: lead.VerifyDepth,
			},
		},
	}, nil
}

// distinctClaims produces claims whose wording differs per lead.
//
// okResult reuses "claim 0"/"claim 1" for every lead, so every cross-lead pair is
// byte-identical and decideMechanically settles it without a model call. Correct
// behaviour, and useless for testing adjudication.
func distinctClaims(n int, lead core.Lead) (*actors.Result, error) {
	return &actors.Result{
		Summary: "a summary",
		Costs: []core.ToolCall{{
			Role: core.RoleExecutor, Type: core.CallLLM, Model: "fake-model",
			Cost: core.Cost{USDMicros: 10_000, InputTokens: 800, OutputTokens: 80},
		}},
		Claims: []core.Claim{
			{
				Text:   fmt.Sprintf("Subword tokenization affects scaling, finding %d.", n),
				Source: fmt.Sprintf("https://s%da.example/p", n),
				Quote:  "a quote long enough to constitute real evidence",
			},
			{
				Text:   fmt.Sprintf("Byte-level modelling changes throughput, finding %d.", n),
				Source: fmt.Sprintf("https://s%db.example/p", n),
				Quote:  "another quote long enough to constitute real evidence",
			},
		},
	}, nil
}

// TestVerificationRunsInTheLoopAndScoresClaims. Slice 4 could derive confidence and
// nothing called it; this is the wiring.
func TestVerificationRunsInTheLoopAndScoresClaims(t *testing.T) {
	fl := &verifierLLM{plans: []string{planJSON("a", "b", "c")}, relation: "duplicate_of"}
	r := newRig(t, 5*core.MicrosPerUSD, nil, distinctClaims)
	r.exec.Planner = &planner.Planner{LLM: fl, MaxInitialLeads: 3, ReplanEvery: 3, MaxDepth: 1}
	r.withVerifier(fl)

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.VerifyPasses == 0 {
		t.Fatal("no verification pass ran in the loop")
	}
	if res.ClaimsVerified == 0 {
		t.Error("no claims were verified")
	}
	if fl.adjudications() == 0 {
		t.Error("the verifier never made an adjudication call")
	}

	// Every claim scored, and none left at the unverified sentinel.
	var unverified int
	var zeroConfidence int
	if err := r.db.Read(context.Background(), func(ctx context.Context, q store.Queries) error {
		list, err := q.ListClaims(ctx, r.sess.ID, 0)
		if err != nil {
			return err
		}
		for _, c := range list {
			if c.VerifiedAt == nil {
				unverified++
			}
			if c.Confidence == 0 {
				zeroConfidence++
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if unverified != 0 {
		t.Errorf("%d claims left unverified after the loop finished", unverified)
	}
	if zeroConfidence != 0 {
		t.Errorf("%d claims still carry confidence 0 — the derivation did not reach them",
			zeroConfidence)
	}
}

// TestAContradictionQueuesAFollowUpWhoseClaimsInheritTheChain is the whole point of
// slice 5, and the reason the cap could not ship before the mechanism.
//
// Note what it does to keep the loop alive — three planner replies, ReplanEvery
// 1 — and why: for years that was load-bearing, because a follow-up only ran if
// the planner happened to still be adding leads beside it. That was the bug in
// TestAFollowUpRunsWhenThePlannerIsDone, not a property of this test, and the
// scaffolding stays here to test lineage rather than dispatch.
//
// §11.4's argument is that a per-row counter cannot bind, because a follow-up
// produces a NEW claim whose counter starts at zero. So the chain has to be carried
// on the lead and stamped onto the claims — and if it is not, the depth cap reads a
// number nothing increments. M1 shipped exactly that shape once: MaxLeads was
// checked, tested, and inert.
func TestAContradictionQueuesAFollowUpWhoseClaimsInheritTheChain(t *testing.T) {
	// Enough plan replies that the loop does not stop at the first replan — the
	// follow-up has to actually RUN for its claims to inherit anything.
	fl := &verifierLLM{
		plans: []string{
			planJSON("the question"),
			planJSON("another question"),
			planJSON("a third question"),
		},
		relation: "contradicts",
	}
	r := newRig(t, 20*core.MicrosPerUSD, nil, opposingClaims)
	r.exec.Planner = &planner.Planner{LLM: fl, MaxInitialLeads: 1, ReplanEvery: 1, MaxDepth: 3}
	v := r.withVerifier(fl)
	v.MaxFollowUpsPerRoot = 5
	v.MaxVerifyDepth = 4

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.Contradictions == 0 {
		t.Fatal("no contradiction found between claims from different sources")
	}
	if res.FollowUpsQueued == 0 {
		t.Fatal("a contradiction queued no follow-up lead")
	}

	// The lineage must actually advance, or §11.4's cap is decoration.
	var leads []*core.Lead
	var claims []*core.Claim
	if err := r.db.Read(context.Background(), func(ctx context.Context, q store.Queries) error {
		var err error
		if leads, err = q.ListLeads(ctx, r.sess.ID, 0); err != nil {
			return err
		}
		claims, err = q.ListClaims(ctx, r.sess.ID, 0)
		return err
	}); err != nil {
		t.Fatal(err)
	}

	deepestLead := 0
	rooted := 0
	for _, l := range leads {
		if l.VerifyDepth > deepestLead {
			deepestLead = l.VerifyDepth
		}
		if l.RootClaimID != nil {
			rooted++
		}
	}
	if rooted == 0 {
		t.Error("no lead carries a root claim; the chain cannot advance")
	}
	if deepestLead == 0 {
		t.Error("every lead sits at verify depth 0; the cap reads a counter nothing increments")
	}

	deepestClaim := 0
	inherited := 0
	for _, c := range claims {
		if c.VerifyDepth > deepestClaim {
			deepestClaim = c.VerifyDepth
		}
		// A claim on a chain has a root that is not itself.
		if c.RootClaimID != "" && c.RootClaimID != c.ID {
			inherited++
		}
	}
	if deepestClaim == 0 {
		t.Errorf("no claim inherited a verify depth from its lead (deepest lead was %d)",
			deepestLead)
	}
	if inherited == 0 {
		t.Error("no claim inherited a root claim from its lead")
	}
}

// TestAFollowUpRunsWhenThePlannerIsDone.
//
// The inverse of the lineage test above, and the one the corpus needed. The
// planner answers once and then reports done — the ordinary shape, and what
// every session on the corpus did — so nothing keeps the loop alive except the
// follow-up itself.
//
// It used to end here with the lead sitting in the queue. replan asks the
// PLANNER whether to continue, the Verifier queues leads the planner has no view
// of, and `added == 0` was read as "there is nothing to do". Twelve of thirteen
// recorded sessions ended holding the very query that would have settled their
// disagreement, having spent 2-6% of their budget.
func TestAFollowUpRunsWhenThePlannerIsDone(t *testing.T) {
	fl := &verifierLLM{
		// One plan. Every call after this answers {"done":true}.
		plans:    []string{planJSON("the question")},
		relation: "contradicts",
	}
	r := newRig(t, 20*core.MicrosPerUSD, nil, opposingClaims)
	r.exec.Planner = &planner.Planner{LLM: fl, MaxInitialLeads: 1, ReplanEvery: 3, MaxDepth: 2}
	v := r.withVerifier(fl)
	v.MaxFollowUpsPerRoot = 5
	v.MaxVerifyDepth = 4

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.FollowUpsQueued == 0 {
		t.Fatal("a contradiction queued no follow-up lead; nothing to dispatch")
	}

	leads := leadsOf(t, r)
	var queued, ran int
	for _, l := range leads {
		if l.RootClaimID == nil {
			continue
		}
		switch l.Status {
		case core.LeadDone, core.LeadSkippedCache:
			ran++
		case core.LeadQueued:
			queued++
		}
	}
	if ran == 0 {
		t.Errorf("%d follow-up lead(s) queued and %d left queued; none ran. The "+
			"planner reporting done ended the session with the work that would "+
			"settle its contradiction still in the queue", res.FollowUpsQueued, queued)
	}
	if queued > 0 {
		t.Errorf("%d follow-up lead(s) still queued when the run ended", queued)
	}
}

// TestFollowUpsStopAtTheLeadCeiling.
//
// Draining the verifier's queue must not become a way around §8.5. MaxLeads is
// checked in the same transaction that increments it, so this asserts the
// property rather than the mechanism: leads are dispatched, and the ceiling
// holds.
func TestFollowUpsStopAtTheLeadCeiling(t *testing.T) {
	const maxLeads = 3

	fl := &verifierLLM{plans: []string{planJSON("the question")}, relation: "contradicts"}
	r := newRigWithCeilings(t, 20*core.MicrosPerUSD, maxLeads, 200, nil, opposingClaims)
	r.exec.Planner = &planner.Planner{LLM: fl, MaxInitialLeads: 1, ReplanEvery: 3, MaxDepth: 2}
	v := r.withVerifier(fl)
	// Uncapped at the verifier, so the only thing that can stop it is the ceiling.
	v.MaxFollowUpsPerRoot = 50
	v.MaxVerifyDepth = 50

	if _, err := r.exec.Run(context.Background(), r.sess.ID); err != nil {
		t.Fatalf("run: %v", err)
	}

	var ran int
	for _, l := range leadsOf(t, r) {
		if l.Status == core.LeadDone || l.Status == core.LeadSkippedCache {
			ran++
		}
	}
	if ran == 0 {
		t.Fatal("no lead ran at all; the ceiling cannot be what stopped it")
	}
	if ran > maxLeads {
		t.Errorf("%d lead(s) ran against a ceiling of %d — draining the verifier's "+
			"queue walked past the lead ceiling", ran, maxLeads)
	}
}

// leadsOf reads a session's leads.
func leadsOf(t *testing.T, r *rig) []*core.Lead {
	t.Helper()
	var leads []*core.Lead
	if err := r.db.Read(context.Background(), func(ctx context.Context, q store.Queries) error {
		var err error
		leads, err = q.ListLeads(ctx, r.sess.ID, 0)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	return leads
}

// TestVerificationFailureDoesNotFailTheSession. Verification improves an answer; it
// does not produce one. A provider that will not compare claims must degrade the run
// rather than end it — the claims still carry quotes checked verbatim against source.
func TestVerificationFailureDoesNotFailTheSession(t *testing.T) {
	fl := &verifierLLM{plans: []string{planJSON("a", "b")}, relation: "not-a-relation"}
	r := newRig(t, 5*core.MicrosPerUSD, nil, func(int, core.Lead) (*actors.Result, error) {
		return okResult(2, 10_000), nil
	})
	r.exec.Planner = &planner.Planner{LLM: fl, MaxInitialLeads: 2, ReplanEvery: 2, MaxDepth: 1}
	r.withVerifier(fl)

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.Status == core.StatusFailed {
		t.Errorf("a useless verifier failed the session: %s", res.StoppedBecause)
	}
	if len(res.Claims) == 0 {
		t.Error("no claims survived")
	}

	// A Verifier that errors outright, not merely one that answers badly. Run
	// returns ErrNoProvider with no LLM, which is the store-or-ledger-failure branch
	// — and it must warn rather than fail the session for the same reason.
	broken := newRig(t, 5*core.MicrosPerUSD, []string{planJSON("a", "b")},
		func(int, core.Lead) (*actors.Result, error) { return okResult(2, 10_000), nil })
	broken.exec.Verifier = &verifier.Verifier{
		Store: broken.db, Ledger: broken.led, LLM: nil,
		Log: slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
	bres, err := broken.exec.Run(context.Background(), broken.sess.ID)
	if err != nil {
		t.Fatalf("run with a broken verifier: %v", err)
	}
	if bres.Status == core.StatusFailed {
		t.Errorf("a verifier that errors failed the session: %s", bres.StoppedBecause)
	}
	if len(bres.Claims) == 0 {
		t.Error("no claims survived a broken verifier")
	}
	// And the claims are still marked verified: we asked, and got nothing usable.
	// Leaving them unverified would make every later pass pay for the same pairs.
	if err := r.db.Read(context.Background(), func(ctx context.Context, q store.Queries) error {
		left, err := q.ListUnverifiedClaims(ctx, r.sess.ID, 0)
		if err != nil {
			return err
		}
		if len(left) != 0 {
			t.Errorf("%d claims left unverified after a degraded pass", len(left))
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

// TestNoVerifierIsASupportedConfiguration. The research still runs, and confidence
// stays 0 rather than being invented.
func TestNoVerifierIsASupportedConfiguration(t *testing.T) {
	r := newRig(t, 5*core.MicrosPerUSD, []string{planJSON("a", "b")},
		func(int, core.Lead) (*actors.Result, error) { return okResult(2, 10_000), nil })
	r.exec.Verifier = nil

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.VerifyPasses != 0 {
		t.Errorf("%d verification passes with no verifier configured", res.VerifyPasses)
	}
	if len(res.Claims) == 0 {
		t.Error("no claims produced")
	}
	if res.Status == core.StatusFailed {
		t.Errorf("session failed without a verifier: %s", res.StoppedBecause)
	}
}

// TestThePlannerLearnsAboutContradictions. §9.1 keeps page text out of the planner,
// so the disagreement reaches it as a COUNT — but it has to reach it. A planner told
// only the claim count cannot tell "this sub-question has evidence" from "this
// sub-question has an argument".
func TestThePlannerLearnsAboutContradictions(t *testing.T) {
	fl := &verifierLLM{
		plans:    []string{planJSON("the question"), planJSON("a follow-up question")},
		relation: "contradicts",
	}
	r := newRig(t, 20*core.MicrosPerUSD, nil, opposingClaims)
	r.exec.Planner = &planner.Planner{LLM: fl, MaxInitialLeads: 1, ReplanEvery: 1, MaxDepth: 2}
	r.withVerifier(fl)

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.Contradictions == 0 {
		t.Fatal("no contradiction found")
	}
	if res.Digest.Contradictions == 0 {
		t.Error("the digest carries no contradiction count, so the planner never saw it")
	}
	if res.Digest.Contradictions != res.Contradictions {
		t.Errorf("digest reports %d contradictions, the run found %d",
			res.Digest.Contradictions, res.Contradictions)
	}
	// And the serialized form the planner actually reads must say so.
	if !strings.Contains(res.Digest.String(), "contradiction") {
		t.Errorf("the planner's view does not mention contradictions:\n%s", res.Digest.String())
	}
}

// manyOpposingClaims emits THREE contradicting pairs per lead, over three
// unrelated topics so they pair with their own opposite rather than each other.
//
// opposingClaims gives one pair, so one pass can never produce more follow-ups
// than any sane budget takes — which makes it useless for testing what happens
// to the ones a budget turns away.
func manyOpposingClaims(n int, lead core.Lead) (*actors.Result, error) {
	root := ""
	if lead.RootClaimID != nil {
		root = *lead.RootClaimID
	}
	topics := []struct{ subject, pro, con string }{
		{"tokenization", "Subword tokenization improves accuracy", "Subword tokenization does not improve accuracy"},
		{"quantization", "Four-bit quantization preserves quality", "Four-bit quantization degrades quality"},
		{"distillation", "Distillation retains reasoning ability", "Distillation destroys reasoning ability"},
	}
	out := &actors.Result{
		Summary: "a summary",
		Costs: []core.ToolCall{{
			Role: core.RoleExecutor, Type: core.CallLLM, Model: "fake-model",
			Cost: core.Cost{USDMicros: 8_000, InputTokens: 800, OutputTokens: 80},
		}},
	}
	for _, t := range topics {
		out.Claims = append(out.Claims,
			core.Claim{
				Text:        fmt.Sprintf("%s, reading %d.", t.pro, n),
				Source:      fmt.Sprintf("https://pro-%s%d.example/p", t.subject, n),
				Quote:       "a quote long enough to constitute real evidence",
				RootClaimID: root, VerifyDepth: lead.VerifyDepth,
			},
			core.Claim{
				Text:        fmt.Sprintf("%s, reading %d.", t.con, n),
				Source:      fmt.Sprintf("https://con-%s%d.example/p", t.subject, n),
				Quote:       "another quote long enough to constitute real evidence",
				RootClaimID: root, VerifyDepth: lead.VerifyDepth,
			})
	}
	return out, nil
}

// TestDeferredFollowUpsRunBeforeTheRunEnds.
//
// The sibling of TestAFollowUpRunsWhenThePlannerIsDone, one level along. That
// one fixed the loop ending with follow-ups sitting QUEUED. This is the loop
// ending with follow-ups sitting DEFERRED — contradictions a verification pass
// judged worth settling and its own budget could not take.
//
// With MaxFollowUpsPerPass at 1 and several contradictions on the table, every
// pass queues one and defers the rest. A run that stops when the planner says
// done leaves those owed and reports itself finished, which is what a Deep run
// did: 32 contradictions found, 4 followed up, 28 never looked at, and the
// session ended at 13 of its 25 allowed minutes.
//
// Draining is not overruling the planner. It never answered this question —
// §11.4's work is the verifier's, and the planner has no view of it.
func TestDeferredFollowUpsRunBeforeTheRunEnds(t *testing.T) {
	fl := &verifierLLM{
		plans:    []string{planJSON("the question")},
		relation: "contradicts",
	}
	r := newRig(t, 20*core.MicrosPerUSD, nil, manyOpposingClaims)
	r.exec.Planner = &planner.Planner{LLM: fl, MaxInitialLeads: 1, ReplanEvery: 3, MaxDepth: 2}
	v := r.withVerifier(fl)
	v.MaxFollowUpsPerRoot = 5
	v.MaxVerifyDepth = 4
	// One per pass, so anything else a pass finds is deferred.
	v.MaxFollowUpsPerPass = 1

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	var deferred, ran int
	for _, l := range leadsOf(t, r) {
		if l.RootClaimID == nil {
			continue
		}
		if l.Deferred {
			deferred++
			continue
		}
		if l.Status == core.LeadDone || l.Status == core.LeadSkippedCache {
			ran++
		}
	}
	if ran == 0 {
		t.Errorf("no follow-up ran (deferred=%d, queued=%d, owed=%d)",
			deferred, res.FollowUpsQueued, res.FollowUpsDeferred)
	}
	if deferred > 0 {
		t.Errorf("the run ended owing %d follow-up(s); a contradiction it judged "+
			"worth settling was recorded and never researched", deferred)
	}
}

// The finishing reserve.

// wallClockRig is a rig whose session has a wall clock, a verifier, one queued
// lead, and an executor clock standing `elapsed` after the session was created.
func wallClockRig(t *testing.T, wall, elapsed time.Duration) *rig {
	t.Helper()
	fl := &verifierLLM{plans: []string{planJSON("the only question")}, relation: "duplicate_of"}
	r := newRigFromSpec(t, budget.SessionSpec{
		Budget: 5 * core.MicrosPerUSD, MaxLeads: 50, MaxToolCalls: 200, MaxWallClock: wall,
	}, nil, distinctClaims)
	r.exec.Planner = &planner.Planner{LLM: fl, MaxInitialLeads: 3, ReplanEvery: 3, MaxDepth: 1}
	r.withVerifier(fl)
	at := r.sess.CreatedAt.Add(elapsed)
	r.exec.Now = func() time.Time { return at }
	return r
}

// TestLoopStopsLeasingInsideTheFinishReserve checks that the loop stops
// leasing new leads and runs one verify pass once inside the finish reserve.
func TestLoopStopsLeasingInsideTheFinishReserve(t *testing.T) {
	r := wallClockRig(t, 10*time.Minute, 9*time.Minute)

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if n := r.actor.count(); n != 0 {
		t.Errorf("%d lead(s) ran inside the finishing reserve, want 0", n)
	}
	for _, l := range leadsOf(t, r) {
		if l.Status != core.LeadQueued {
			t.Errorf("lead %s is %s, want it left queued", l.ID, l.Status)
		}
	}
	if res.VerifyPasses != 1 {
		t.Errorf("verify passes = %d, want exactly 1 before stopping", res.VerifyPasses)
	}
	if res.Status != core.StatusExhausted {
		t.Errorf("status = %s, want %s", res.Status, core.StatusExhausted)
	}
	// mole-backend's runner reports this exact string.
	if res.StoppedBecause != "max_wallclock" {
		t.Errorf("stopped because %q, want max_wallclock", res.StoppedBecause)
	}
	if held := r.reload(t).Held; held != 0 {
		t.Errorf("%d still held after stopping in the reserve", held)
	}
}

// TestEveryCeilingStopVerifiesWhatResearchFound checks that a max_leads stop
// between replan boundaries still leaves every gathered claim verified.
func TestEveryCeilingStopVerifiesWhatResearchFound(t *testing.T) {
	fl := &verifierLLM{plans: []string{planJSON("a", "b", "c")}, relation: "duplicate_of"}
	r := newRigFromSpec(t, budget.SessionSpec{
		Budget: 5 * core.MicrosPerUSD, MaxLeads: 2, MaxToolCalls: 200,
	}, nil, distinctClaims)
	r.exec.Planner = &planner.Planner{LLM: fl, MaxInitialLeads: 3, ReplanEvery: 3, MaxDepth: 1}
	r.withVerifier(fl)

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != core.StatusExhausted || res.StoppedBecause != "max_leads" {
		t.Fatalf("status = %s (%q), want %s on max_leads", res.Status, res.StoppedBecause, core.StatusExhausted)
	}
	if res.VerifyPasses != 1 {
		t.Errorf("verify passes = %d, want exactly 1 after the ceiling stop", res.VerifyPasses)
	}
	var claims, unverified int
	if err := r.db.Read(context.Background(), func(ctx context.Context, q store.Queries) error {
		list, err := q.ListClaims(ctx, r.sess.ID, 0)
		if err != nil {
			return err
		}
		claims = len(list)
		for _, c := range list {
			if c.VerifiedAt == nil {
				unverified++
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if claims == 0 {
		t.Fatal("no claims gathered; the test proves nothing")
	}
	if unverified != 0 {
		t.Errorf("%d of %d claim(s) left unverified after a max_leads stop", unverified, claims)
	}
}

// TestLoopLeasesBeforeTheReserve checks that research still runs a minute
// into a ten-minute session, well before the finish reserve.
func TestLoopLeasesBeforeTheReserve(t *testing.T) {
	r := wallClockRig(t, 10*time.Minute, time.Minute)

	if _, err := r.exec.Run(context.Background(), r.sess.ID); err != nil {
		t.Fatal(err)
	}
	if r.actor.count() == 0 {
		t.Error("no lead ran a minute into a ten-minute session")
	}
}

// TestNoWallClockNoReserve checks that a session with no wall clock gets no
// finish reserve and never stops research for one.
func TestNoWallClockNoReserve(t *testing.T) {
	if got := executor.FinishReserveFor(0, 0); got != 0 {
		t.Errorf("finishReserve(0, 0) = %s, want 0", got)
	}
	if got := executor.FinishReserveFor(0, time.Minute); got != 0 {
		t.Errorf("finishReserve(0, 1m) = %s, want 0", got)
	}

	r := wallClockRig(t, 0, 24*time.Hour)
	if _, err := r.exec.Run(context.Background(), r.sess.ID); err != nil {
		t.Fatal(err)
	}
	if r.actor.count() == 0 {
		t.Error("no lead ran on a session with no wall clock")
	}
}

// TestReserveIsCappedAtHalfTheClock checks that the reserve never exceeds
// half the wall clock, even under the 90-second floor.
func TestReserveIsCappedAtHalfTheClock(t *testing.T) {
	cases := []struct {
		wall, configured, want time.Duration
	}{
		{2 * time.Minute, 0, time.Minute},
		{time.Minute, 0, 30 * time.Second},
		{10 * time.Minute, 0, 90 * time.Second},              // the floor
		{20 * time.Minute, 0, 3 * time.Minute},               // 15%
		{10 * time.Minute, 2 * time.Minute, 2 * time.Minute}, // configured
		{2 * time.Minute, 5 * time.Minute, time.Minute},      // configured, capped
	}
	for _, c := range cases {
		if got := executor.FinishReserveFor(c.wall, c.configured); got != c.want {
			t.Errorf("finishReserve(%s, %s) = %s, want %s", c.wall, c.configured, got, c.want)
		}
	}
}

// recheckRig is a max_leads-stopped run whose leads produce opposing claims.
func recheckRig(t *testing.T, recheckModel string) (*rig, *verifierLLM) {
	t.Helper()
	fl := &verifierLLM{
		plans: []string{planJSON("a", "b", "c")}, relation: "contradicts",
		strictModel: "strict", strictRelation: "neither",
	}
	r := newRigFromSpec(t, budget.SessionSpec{
		Budget: 5 * core.MicrosPerUSD, MaxLeads: 2, MaxToolCalls: 200,
	}, nil, opposingClaims)
	r.exec.Planner = &planner.Planner{LLM: fl, MaxInitialLeads: 3, ReplanEvery: 3, MaxDepth: 1}
	r.withVerifier(fl).RecheckModel = recheckModel
	return r, fl
}

// TestThePostLoopRecheckRunsOnceWhenConfigured checks the re-check runs once after the final verify.
func TestThePostLoopRecheckRunsOnceWhenConfigured(t *testing.T) {
	r, fl := recheckRig(t, "strict")
	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if res.VerifyPasses == 0 {
		t.Fatal("no verify pass ran")
	}
	if got := fl.strictAdjudications(); got != 1 {
		t.Errorf("%d re-check calls, want 1", got)
	}
	if res.Withheld == 0 || res.Withheld != res.Rechecked {
		t.Errorf("rechecked %d, withheld %d; want every contradiction withheld", res.Rechecked, res.Withheld)
	}
	if res.Contradictions != 0 {
		t.Errorf("contradictions = %d after every one was withheld", res.Contradictions)
	}
	var left int
	if err := r.db.Read(context.Background(), func(ctx context.Context, q store.Queries) error {
		edges, err := q.ListEdges(ctx, r.sess.ID, 0)
		for _, e := range edges {
			if e.Kind == core.EdgeContradicts {
				left++
			}
		}
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if left != 0 {
		t.Errorf("%d contradicts edges stored after the re-check withheld them", left)
	}
}

// TestNoRecheckModelMeansNoPostLoopRecheck checks no re-check runs when none is configured.
func TestNoRecheckModelMeansNoPostLoopRecheck(t *testing.T) {
	r, fl := recheckRig(t, "")
	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got := fl.strictAdjudications(); got != 0 {
		t.Errorf("%d re-check calls with no re-check model", got)
	}
	if res.Rechecked != 0 || res.Withheld != 0 {
		t.Errorf("rechecked %d, withheld %d; want 0", res.Rechecked, res.Withheld)
	}
	if res.Contradictions == 0 {
		t.Error("no contradictions found; the test proves nothing")
	}
}

// TestUncheckedContradictionsAreNotCounted checks the run's count leaves out contradictions never re-checked.
func TestUncheckedContradictionsAreNotCounted(t *testing.T) {
	r, fl := recheckRig(t, "strict")
	fl.strictErr = errors.New("provider hiccup")
	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	var stored int
	if err := r.db.Read(context.Background(), func(ctx context.Context, q store.Queries) error {
		edges, err := q.ListEdges(ctx, r.sess.ID, 0)
		for _, e := range edges {
			if e.Kind == core.EdgeContradicts {
				stored++
			}
		}
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if stored == 0 {
		t.Fatal("no contradictions stored; the test proves nothing")
	}
	if res.Contradictions != 0 {
		t.Errorf("contradictions = %d with none re-checked, want 0", res.Contradictions)
	}
}

// doneWith is a replan reply that reports done and still proposes questions.
func doneWith(questions ...string) string {
	var qs []map[string]string
	for i, q := range questions {
		qs = append(qs, map[string]string{"id": fmt.Sprintf("q%d", 10+i), "question": q})
	}
	b, _ := json.Marshal(map[string]any{"done": true, "questions": qs})
	return string(b)
}

// ranQueries lists the query of every lead the actor was given.
func ranQueries(t *testing.T, r *rig) []string {
	t.Helper()
	var out []string
	for _, l := range leadsOf(t, r) {
		if l.Status == core.LeadDone || l.Status == core.LeadSkippedCache {
			out = append(out, l.Query)
		}
	}
	return out
}

// TestPlannerDoneIsOverriddenWhileMostOfTheBudgetIsLeft checks an early done is ignored and logged.
func TestPlannerDoneIsOverriddenWhileMostOfTheBudgetIsLeft(t *testing.T) {
	r := newRig(t, 50*core.MicrosPerUSD, []string{planJSON("a", "b", "c"), doneWith("origin")},
		func(int, core.Lead) (*actors.Result, error) { return okResult(1, 1_000), nil })
	var mu sync.Mutex
	var events []executor.Event
	r.exec.Progress = func(ev executor.Event) {
		mu.Lock()
		defer mu.Unlock()
		events = append(events, ev)
	}

	if _, err := r.exec.Run(context.Background(), r.sess.ID); err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(ranQueries(t, r), "origin") {
		t.Errorf("the sub-question proposed with an early done never ran: %v", ranQueries(t, r))
	}
	mu.Lock()
	defer mu.Unlock()
	logged := false
	for _, ev := range events {
		if strings.Contains(ev.Detail, "ignoring the planner's done") {
			logged = true
		}
	}
	if !logged {
		t.Errorf("the override was not recorded in the progress log: %v", events)
	}
}

// TestPlannerDoneIsHonouredPastTheThreshold checks done ends the run once half the allowance is gone.
func TestPlannerDoneIsHonouredPastTheThreshold(t *testing.T) {
	cases := map[string]struct {
		budget   int64
		maxLeads int64
		cost     int64
	}{
		"half the leads run":   {budget: 50 * core.MicrosPerUSD, maxLeads: 6, cost: 1_000},
		"half the budget gone": {budget: 300_000, maxLeads: 50, cost: 60_000},
	}
	for name, c := range cases {
		r := newRigWithCeilings(t, c.budget, c.maxLeads, 200,
			[]string{planJSON("a", "b", "c"), doneWith("origin")},
			func(int, core.Lead) (*actors.Result, error) { return okResult(1, c.cost), nil })

		res, err := r.exec.Run(context.Background(), r.sess.ID)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if slices.Contains(ranQueries(t, r), "origin") {
			t.Errorf("%s: done was overridden past the threshold", name)
		}
		if res.StoppedBecause != "planner reported done" {
			t.Errorf("%s: stopped because %q, want the planner's done", name, res.StoppedBecause)
		}
	}
}

// TestOverriddenDoneStillEndsAtMaxLeads checks the lead ceiling ends a run whose done was overridden.
func TestOverriddenDoneStillEndsAtMaxLeads(t *testing.T) {
	r := newRigWithCeilings(t, 50*core.MicrosPerUSD, 5, 200,
		[]string{planJSON("a", "b"), doneWith("c", "d", "e"), doneWith("f", "g", "h")},
		func(int, core.Lead) (*actors.Result, error) { return okResult(1, 1_000), nil })

	res, err := r.exec.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if got := len(ranQueries(t, r)); got > 5 {
		t.Errorf("%d leads ran against a ceiling of 5", got)
	}
	if res.StoppedBecause != "max_leads" {
		t.Errorf("stopped because %q, want max_leads", res.StoppedBecause)
	}
}

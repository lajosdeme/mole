package session_test

import (
	"context"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/actors"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/executor"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/session"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/store/sqlite"
	"github.com/lajosdeme/mole/internal/tools/search"
)

// emptyPlanner answers every planning call with "no sub-questions", so the loop
// starts, finds nothing to do, and ends. Enough to drive the sequence without a
// network or a model.
type emptyPlanner struct{}

func (emptyPlanner) Name() string               { return "stub" }
func (emptyPlanner) ModelFor(t llm.Tier) string { return "stub-model" }
func (emptyPlanner) Complete(context.Context, llm.Request) (*llm.Response, error) {
	return &llm.Response{
		Text:  `{"questions":[],"rationale":"nothing to research"}`,
		Model: "stub-model",
		Usage: llm.Usage{InputTokens: 1, OutputTokens: 1},
	}, nil
}

// emptySearch returns nothing, so the one lead the executor falls back to
// (running the question verbatim when the planner proposes none) finds no
// sources and the loop ends. The actor needs a real Search either way: it
// dereferences it before checking for results.
type emptySearch struct{}

func (emptySearch) Kind() search.Kind { return search.KindTavily }
func (emptySearch) Search(context.Context, string, search.Options) (*search.Response, error) {
	return &search.Response{}, nil
}

func newRunner(t *testing.T) (*session.Runner, store.Store) {
	t.Helper()
	db, err := sqlite.Open(filepath.Join(t.TempDir(), "t.db"), sqlite.Options{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.Migrate(context.Background()); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	return &session.Runner{
		Store: db,
		Actor: &actors.WebActor{LLM: emptyPlanner{}, Search: emptySearch{}},
		Owner: "test",
	}, db
}

func testSpec() session.Spec {
	return session.Spec{
		Question:   "a question",
		Mode:       core.ModeReport,
		BudgetUnit: core.BudgetUSD,
		Budget:     1_000_000,
		MaxSources: 3,
		MaxDepth:   2,
		MaxLeads:   5,
		Timeout:    time.Minute,
	}
}

// TestOnLoopFiresBeforeTheEscrowIsSpent pins the ordering the callbacks exist for.
//
// The sequence a caller observes is: the loop ends, then grounding runs, then the
// report is written from released escrow (§8.3). Result carries all three, so it
// cannot express which came first — and the first version of this package
// returned them together and let the CLI guess. It guessed wrong: the grounding
// summary printed above the run summary it follows.
//
// Escrow is the observable that separates the two. At OnLoop it is still held; by
// the time Run returns it has been released and spent on the report.
func TestOnLoopFiresBeforeTheEscrowIsSpent(t *testing.T) {
	r, db := newRunner(t)
	spec := testSpec()

	sess, err := r.Create(context.Background(), spec)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if sess.Escrow <= 0 {
		t.Fatal("no escrow was held at creation; this test cannot observe the ordering")
	}

	var escrowAtLoop int64 = -1
	var loopCalls int
	r.OnLoop = func(*executor.Result) {
		loopCalls++
		cur, err := loadSession(t, db, sess.ID)
		if err != nil {
			t.Errorf("load during OnLoop: %v", err)
			return
		}
		escrowAtLoop = cur.Escrow
	}

	if _, err := r.Run(context.Background(), sess, spec); err != nil {
		t.Fatalf("run: %v", err)
	}

	if loopCalls != 1 {
		t.Fatalf("OnLoop fired %d times, want exactly 1", loopCalls)
	}
	if escrowAtLoop <= 0 {
		t.Errorf("escrow was already released at OnLoop (%d): the callback fires after "+
			"the report is paid for, so a caller cannot print the run summary first",
			escrowAtLoop)
	}
	after, err := loadSession(t, db, sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if after.Escrow != 0 {
		t.Errorf("escrow is still %d after Run; it should have been released", after.Escrow)
	}
}

// TestCreateAppliesTheUnitIndependentCeilings. §8.5's ceilings bind even when the
// money estimate is wrong, which is the case they exist for — so a Spec that sets
// them must produce a session row that carries them.
func TestCreateAppliesTheUnitIndependentCeilings(t *testing.T) {
	r, _ := newRunner(t)
	spec := testSpec()

	sess, err := r.Create(context.Background(), spec)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if sess.MaxLeads != int64(spec.MaxLeads) {
		t.Errorf("MaxLeads = %d, want %d", sess.MaxLeads, spec.MaxLeads)
	}
	if sess.MaxWallClock != spec.Timeout {
		t.Errorf("MaxWallClock = %s, want %s", sess.MaxWallClock, spec.Timeout)
	}
	// A backstop sized from the lead count, not from fan-out. It used to be
	// leads × sources × 4 — a claim about what a lead costs that measurement put
	// at less than half the truth, so the backstop bound before the clock or the
	// money ever did. A caller that knows better sets Spec.MaxToolCalls.
	if sess.MaxToolCalls <= int64(spec.MaxLeads) {
		t.Errorf("MaxToolCalls = %d, not a ceiling above %d leads",
			sess.MaxToolCalls, spec.MaxLeads)
	}
	if sess.MaxToolCalls != int64(spec.MaxLeads)*150 {
		t.Errorf("MaxToolCalls = %d, want %d (leads × the per-lead backstop)",
			sess.MaxToolCalls, int64(spec.MaxLeads)*150)
	}

	// A caller that leaves them unset must still get a bounded session, not zero
	// ceilings — zero would read as "no limit" everywhere downstream.
	bare := session.Spec{Question: "q", Mode: core.ModeReport, BudgetUnit: core.BudgetUSD, Budget: 1000}
	s2, err := r.Create(context.Background(), bare)
	if err != nil {
		t.Fatalf("create bare: %v", err)
	}
	if s2.MaxLeads <= 0 || s2.MaxToolCalls <= 0 {
		t.Errorf("an unset spec produced unbounded ceilings: leads=%d calls=%d",
			s2.MaxLeads, s2.MaxToolCalls)
	}
}

func TestCreateRefusesAnUnusableSpec(t *testing.T) {
	r, _ := newRunner(t)
	for name, spec := range map[string]session.Spec{
		"no question":  {Mode: core.ModeReport, BudgetUnit: core.BudgetUSD, Budget: 1000},
		"unknown mode": {Question: "q", Mode: core.Mode("sideways"), BudgetUnit: core.BudgetUSD, Budget: 1000},
	} {
		if _, err := r.Create(context.Background(), spec); err == nil {
			t.Errorf("%s: Create accepted it", name)
		}
	}
}

// TestRecoverIsSafeOnAFreshDatabase. Recovery runs at process start, before
// anything is known to be wrong. Refusing to start because a sweep found nothing
// — or failed — helps nobody, so it reports counts and never errors.
func TestRecoverIsSafeOnAFreshDatabase(t *testing.T) {
	r, _ := newRunner(t)
	var notices []string
	r.Notice = func(m string) { notices = append(notices, m) }

	rc := r.Recover(context.Background())
	if rc.Any() {
		t.Errorf("a fresh database recovered something: %+v", rc)
	}
	if len(notices) != 0 {
		t.Errorf("a fresh database produced warnings: %v", notices)
	}
}

func loadSession(t *testing.T, db store.Store, id string) (*core.Session, error) {
	t.Helper()
	var s *core.Session
	err := db.Read(context.Background(), func(ctx context.Context, q store.Queries) error {
		var err error
		s, err = q.GetSession(ctx, id)
		return err
	})
	return s, err
}

// strongCounter is emptyPlanner that counts strong-tier calls, which only the report makes.
type strongCounter struct {
	emptyPlanner
	mu     sync.Mutex
	strong int
}

func (s *strongCounter) Complete(ctx context.Context, req llm.Request) (*llm.Response, error) {
	if req.Tier == llm.TierStrong {
		s.mu.Lock()
		s.strong++
		s.mu.Unlock()
		return &llm.Response{Text: "the answer", Model: "stub-model",
			Usage: llm.Usage{InputTokens: 1, OutputTokens: 1}}, nil
	}
	return s.emptyPlanner.Complete(ctx, req)
}

func (s *strongCounter) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.strong
}

// TestOnlyACancelSkipsTheReport checks that a timed-out run still writes the
// synthesized report, and a cancelled one does not.
func TestOnlyACancelSkipsTheReport(t *testing.T) {
	for _, tc := range []struct {
		name       string
		ctx        func() context.Context
		wantReport bool
	}{
		{"deadline", func() context.Context {
			ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
			t.Cleanup(cancel)
			return ctx
		}, true},
		{"cancel", func() context.Context {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			return ctx
		}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, db := newRunner(t)
			counter := &strongCounter{}
			r.Actor.LLM = counter
			spec := testSpec()

			sess, err := r.Create(context.Background(), spec)
			if err != nil {
				t.Fatalf("create: %v", err)
			}
			if err := db.WithTx(context.Background(), func(ctx context.Context, tx store.Tx) error {
				return tx.InsertClaims(ctx, []core.Claim{{
					SessionID: sess.ID, LeadID: "l_1",
					Text:   "Subword tokenization affects byte-level scaling.",
					Source: "https://s.example/p",
					Quote:  "a quote long enough to be real evidence",
				}})
			}); err != nil {
				t.Fatalf("insert claims: %v", err)
			}

			res, err := r.Run(tc.ctx(), sess, spec)
			if err != nil {
				t.Fatalf("run: %v", err)
			}
			if got := counter.count() > 0; got != tc.wantReport {
				t.Errorf("report generator called = %v, want %v", got, tc.wantReport)
			}
			if res.Report == nil {
				t.Fatal("no report written")
			}
			after, err := loadSession(t, db, sess.ID)
			if err != nil {
				t.Fatal(err)
			}
			if after.Escrow != 0 {
				t.Errorf("escrow is still %d after Run; it should have been released", after.Escrow)
			}
		})
	}
}

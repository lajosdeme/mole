package session

import (
	"context"
	"errors"
	"net/url"
	"path/filepath"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/actors"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/store/sqlite"
	"github.com/lajosdeme/mole/internal/tools/extract"
	"github.com/lajosdeme/mole/internal/tools/fetch"
	"github.com/lajosdeme/mole/internal/verifier"
)

// reportProbe answers the report call once the test lets it, recording what its
// context looked like.
type reportProbe struct {
	started     chan struct{}
	errAtStart  error
	hasDeadline bool
	errAtEnd    error
}

func (*reportProbe) Name() string             { return "probe" }
func (*reportProbe) ModelFor(llm.Tier) string { return "probe-model" }
func (p *reportProbe) Complete(ctx context.Context, req llm.Request) (*llm.Response, error) {
	if req.Tier != llm.TierStrong {
		return &llm.Response{Text: `{}`, Model: "probe-model"}, nil
	}
	p.errAtStart = ctx.Err()
	_, p.hasDeadline = ctx.Deadline()
	close(p.started)
	select {
	case <-ctx.Done():
	case <-time.After(5 * time.Second):
	}
	p.errAtEnd = ctx.Err()
	return nil, p.errAtEnd
}

// TestTheReportIsBoundedAndCancellable checks that, after a deadline or a
// normal stop, the report has a time limit and stops when the parent is cancelled.
func TestTheReportIsBoundedAndCancellable(t *testing.T) {
	for _, tc := range []struct {
		name string
		run  func(parent context.Context) (context.Context, context.CancelFunc)
	}{
		{"deadline", func(parent context.Context) (context.Context, context.CancelFunc) {
			return context.WithDeadline(parent, time.Now().Add(-time.Second))
		}},
		{"normal", func(parent context.Context) (context.Context, context.CancelFunc) {
			return context.WithCancel(parent)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) { checkReportIsBoundedAndCancellable(t, tc.run) })
	}
}

func checkReportIsBoundedAndCancellable(t *testing.T, runCtx func(context.Context) (context.Context, context.CancelFunc)) {
	db, err := sqlite.Open(filepath.Join(t.TempDir(), "t.db"), sqlite.Options{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.Migrate(context.Background()); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	probe := &reportProbe{started: make(chan struct{})}
	actor := &actors.WebActor{LLM: probe}
	r := &Runner{Store: db, Actor: actor, Owner: "test"}
	sess, err := r.Create(context.Background(), Spec{
		Question: "a question", Mode: core.ModeReport, BudgetUnit: core.BudgetUSD,
		Budget: 1_000_000, MaxSources: 3, MaxDepth: 2, MaxLeads: 5, Timeout: time.Minute,
	})
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

	parent, cancelParent := context.WithCancel(context.Background())
	defer cancelParent()
	run, cancelRun := runCtx(parent)
	defer cancelRun()
	vf := &verifier.Verifier{Store: db, Ledger: r.ledger(), LLM: probe}

	done := make(chan struct{})
	go func() {
		defer close(done)
		r.finish(parent, run, r.ledger(), vf, sess, actor, nil, "")
	}()
	select {
	case <-probe.started:
	case <-time.After(5 * time.Second):
		t.Fatal("the report generator was never called")
	}
	cancelParent()
	<-done

	if probe.errAtStart != nil {
		t.Errorf("report context was already done at the call: %v", probe.errAtStart)
	}
	if !probe.hasDeadline {
		t.Error("report context has no deadline")
	}
	if !errors.Is(probe.errAtEnd, context.Canceled) {
		t.Errorf("report context ended with %v, want it cancelled with the parent", probe.errAtEnd)
	}
}

type nopFetch struct{}

func (nopFetch) Fetch(context.Context, string) (*fetch.Result, error) {
	return nil, errors.New("no fetch")
}

type nopExtract struct{}

func (nopExtract) Extract(context.Context, []byte, string, *url.URL) (*extract.Document, error) {
	return nil, errors.New("no extract")
}

// TestGroundingIsSkippedPastTheResearchDeadline checks grounding makes no attempt once the research deadline has passed.
func TestGroundingIsSkippedPastTheResearchDeadline(t *testing.T) {
	var notices []string
	r := &Runner{Notice: func(s string) { notices = append(notices, s) }}
	vf := &verifier.Verifier{Grounder: &verifier.Grounder{Fetch: nopFetch{}, Extract: nopExtract{}}}
	sess := &core.Session{ID: "s_1", CreatedAt: time.Now().Add(-2 * time.Minute), MaxWallClock: time.Minute}

	if got := r.groundByDeadline(context.Background(), vf, sess, 1_000); got != nil {
		t.Errorf("grounding ran past the deadline: %+v", got)
	}
	if len(notices) != 0 {
		t.Errorf("grounding was attempted past the deadline: %v", notices)
	}
}

// TestReportDeadline checks the report ends at the later of the wall clock and a tenth of it from now, within ReportGrace.
func TestReportDeadline(t *testing.T) {
	created := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	quick := &core.Session{CreatedAt: created, MaxWallClock: 300 * time.Second}
	cases := map[string]struct {
		sess  *core.Session
		start time.Duration
		want  time.Duration
	}{
		"before the wall clock":    {quick, 250 * time.Second, 300 * time.Second},
		"near the wall clock":      {quick, 290 * time.Second, 320 * time.Second},
		"early stop, grace bounds": {quick, 100 * time.Second, 160 * time.Second},
		"no wall clock":            {&core.Session{CreatedAt: created}, 100 * time.Second, 160 * time.Second},
	}
	for name, c := range cases {
		got := reportDeadline(c.sess, created.Add(c.start))
		if want := created.Add(c.want); !got.Equal(want) {
			t.Errorf("%s: deadline at %s, want %s", name, got.Sub(created), c.want)
		}
	}
}

// stallingReport blocks the report call until its context ends and records when.
type stallingReport struct {
	deadline time.Time
	endedAt  time.Time
}

func (*stallingReport) Name() string             { return "stall" }
func (*stallingReport) ModelFor(llm.Tier) string { return "stall-model" }
func (p *stallingReport) Complete(ctx context.Context, req llm.Request) (*llm.Response, error) {
	if req.Tier != llm.TierStrong {
		return &llm.Response{Text: `{}`, Model: "stall-model"}, nil
	}
	p.deadline, _ = ctx.Deadline()
	select {
	case <-ctx.Done():
	case <-time.After(10 * time.Second):
	}
	p.endedAt = time.Now()
	return nil, ctx.Err()
}

// TestAStalledReportEndsAtTheWallClock checks a stalled report is cut at the wall clock and still produces a degraded report.
func TestAStalledReportEndsAtTheWallClock(t *testing.T) {
	db, err := sqlite.Open(filepath.Join(t.TempDir(), "t.db"), sqlite.Options{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.Migrate(context.Background()); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	probe := &stallingReport{}
	actor := &actors.WebActor{LLM: probe}
	r := &Runner{Store: db, Actor: actor, Owner: "test"}
	sess, err := r.Create(context.Background(), Spec{
		Question: "a question", Mode: core.ModeReport, BudgetUnit: core.BudgetUSD,
		Budget: 1_000_000, MaxSources: 3, MaxDepth: 2, MaxLeads: 5, Timeout: 3 * time.Second,
	})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if err := db.WithTx(context.Background(), func(ctx context.Context, tx store.Tx) error {
		return tx.InsertClaims(ctx, []core.Claim{{
			SessionID: sess.ID, LeadID: "l_1",
			Text:   "A claim already found.",
			Source: "https://s.example/p",
			Quote:  "a quote long enough to be real evidence",
		}})
	}); err != nil {
		t.Fatalf("insert claims: %v", err)
	}
	sess.CreatedAt = time.Now().Add(-2600 * time.Millisecond)
	wallEnd := sess.CreatedAt.Add(sess.MaxWallClock)

	vf := &verifier.Verifier{Store: db, Ledger: r.ledger(), LLM: probe}
	report, _, _ := r.finish(context.Background(), context.Background(), r.ledger(), vf, sess, actor, nil, "")

	if d := probe.deadline.Sub(wallEnd).Abs(); d > 50*time.Millisecond {
		t.Errorf("report deadline %s from the wall clock, want at it", d)
	}
	if probe.endedAt.After(wallEnd.Add(250 * time.Millisecond)) {
		t.Errorf("report ran %s past the wall clock", probe.endedAt.Sub(wallEnd))
	}
	if report == nil || report.Degraded == "" {
		t.Errorf("want a degraded report, got %+v", report)
	}
}

// TestResearchTimedOut checks a passed research deadline counts as timed out with a live run context.
func TestResearchTimedOut(t *testing.T) {
	sess := &core.Session{CreatedAt: time.Now().Add(-2 * time.Minute), MaxWallClock: time.Minute}
	if !researchTimedOut(context.Background(), sess, time.Now()) {
		t.Error("a passed research deadline was not treated as timed out")
	}
	if researchTimedOut(context.Background(), &core.Session{CreatedAt: time.Now(), MaxWallClock: time.Minute}, time.Now()) {
		t.Error("a live session was treated as timed out")
	}
}

package session

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/actors"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/store/sqlite"
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

// TestATimedOutReportIsBoundedAndCancellable checks that after a deadline the
// report runs under its own time limit and stops when the parent is cancelled.
func TestATimedOutReportIsBoundedAndCancellable(t *testing.T) {
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
	run, cancelRun := context.WithDeadline(parent, time.Now().Add(-time.Second))
	defer cancelRun()
	vf := &verifier.Verifier{Store: db, Ledger: r.ledger(), LLM: probe}

	done := make(chan struct{})
	go func() {
		defer close(done)
		r.finish(parent, run, r.ledger(), vf, sess, actor, nil)
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

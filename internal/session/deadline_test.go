package session_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/actors"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/store"
)

// stallingLLM blocks its first call until the context ends and answers the rest at once.
type stallingLLM struct {
	mu      sync.Mutex
	calls   int
	cutErr  error
	cutAt   time.Time
	maxWait time.Duration
}

func (*stallingLLM) Name() string             { return "stall" }
func (*stallingLLM) ModelFor(llm.Tier) string { return "stall-model" }
func (s *stallingLLM) Complete(ctx context.Context, _ llm.Request) (*llm.Response, error) {
	s.mu.Lock()
	first := s.calls == 0
	s.calls++
	s.mu.Unlock()
	resp := &llm.Response{Model: "stall-model", Usage: llm.Usage{InputTokens: 1, OutputTokens: 1}}
	if !first {
		resp.Text = "An answer [1]."
		return resp, nil
	}
	select {
	case <-ctx.Done():
	case <-time.After(s.maxWait):
	}
	s.mu.Lock()
	s.cutErr, s.cutAt = ctx.Err(), time.Now()
	s.mu.Unlock()
	return resp, ctx.Err()
}

// TestAStalledCallIsCutAtTheSessionDeadline checks a call blocking past the wall clock ends before the report allowance and the report is still written.
func TestAStalledCallIsCutAtTheSessionDeadline(t *testing.T) {
	r, db := newRunner(t)
	slow := &stallingLLM{maxWait: 10 * time.Second}
	r.Actor = &actors.WebActor{LLM: slow, Search: emptySearch{}}
	spec := testSpec()
	spec.Timeout = 3 * time.Second

	sess, err := r.Create(context.Background(), spec)
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

	res, err := r.Run(context.Background(), sess, spec)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	elapsed := time.Since(sess.CreatedAt)

	slow.mu.Lock()
	defer slow.mu.Unlock()
	if !errors.Is(slow.cutErr, context.DeadlineExceeded) {
		t.Errorf("stalled call ended with %v, want the deadline", slow.cutErr)
	}
	if limit := sess.CreatedAt.Add(spec.Timeout - spec.Timeout/10 + 250*time.Millisecond); slow.cutAt.After(limit) {
		t.Errorf("stalled call cut %s after creation, want by the wall clock less the report allowance",
			slow.cutAt.Sub(sess.CreatedAt))
	}
	if res.Report == nil {
		t.Error("no report was written")
	}
	if elapsed > spec.Timeout+time.Second {
		t.Errorf("run took %s against a %s timeout", elapsed, spec.Timeout)
	}
}

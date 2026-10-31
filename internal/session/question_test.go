package session_test

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/lajosdeme/mole/internal/actors"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/tools/search"
)

// minePromptRecorder plans one sub-question and records every extraction prompt.
type minePromptRecorder struct {
	mu      sync.Mutex
	prompts []string
}

func (*minePromptRecorder) Name() string               { return "recorder" }
func (*minePromptRecorder) ModelFor(t llm.Tier) string { return "stub-model" }

func (m *minePromptRecorder) Complete(_ context.Context, req llm.Request) (*llm.Response, error) {
	text := `{"questions":[{"id":"q1","question":"How was the figure measured?"}],"rationale":"r"}`
	if strings.Contains(req.System, "extraction component") && len(req.Messages) > 0 {
		m.mu.Lock()
		m.prompts = append(m.prompts, req.Messages[0].Text)
		m.mu.Unlock()
		text = `{"claims":[]}`
	}
	return &llm.Response{Text: text, Model: "stub-model", Usage: llm.Usage{InputTokens: 1, OutputTokens: 1}}, nil
}

// oneHitSearch returns one result with usable page text.
type oneHitSearch struct{}

func (oneHitSearch) Kind() search.Kind { return search.KindTavily }
func (oneHitSearch) Search(context.Context, string, search.Options) (*search.Response, error) {
	return &search.Response{Results: []search.Result{{
		URL: "https://a.example/p", Rank: 1,
		Content: strings.Repeat("The measured value was 3.8 per 100,000 under the stated conditions. ", 20),
	}}}, nil
}

// TestTheMinerSeesTheSessionQuestion checks the root question reaches the extraction prompt.
func TestTheMinerSeesTheSessionQuestion(t *testing.T) {
	rec := &minePromptRecorder{}
	r, _ := newRunner(t)
	r.Actor = &actors.WebActor{LLM: rec, Search: oneHitSearch{}}
	spec := testSpec()
	spec.Question = "Do electric cars catch fire more often than petrol cars?"

	ctx := context.Background()
	sess, err := r.Create(ctx, spec)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	_, _ = r.Run(ctx, sess, spec)

	rec.mu.Lock()
	defer rec.mu.Unlock()
	if len(rec.prompts) == 0 {
		t.Fatal("no extraction call was made")
	}
	for _, p := range rec.prompts {
		if !strings.Contains(p, spec.Question) {
			t.Errorf("extraction prompt lacks the session question:\n%s", p)
			break
		}
	}
}

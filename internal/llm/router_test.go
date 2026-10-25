package llm_test

import (
	"context"
	"testing"

	"github.com/lajosdeme/mole/internal/llm"
)

// fakeProvider records the requests it receives.
type fakeProvider struct {
	name string
	got  []llm.Request
}

func (f *fakeProvider) Complete(_ context.Context, req llm.Request) (*llm.Response, error) {
	f.got = append(f.got, req)
	return &llm.Response{Text: "ok", Model: req.Model}, nil
}

func (f *fakeProvider) Name() string                  { return f.name }
func (f *fakeProvider) ModelFor(tier llm.Tier) string { return "" }

func (f *fakeProvider) last(t *testing.T) llm.Request {
	t.Helper()
	if len(f.got) == 0 {
		t.Fatalf("%s received no request", f.name)
	}
	return f.got[len(f.got)-1]
}

func TestRouterRoutesByModelThenTier(t *testing.T) {
	anthropic := &fakeProvider{name: "anthropic"}
	deepseek := &fakeProvider{name: "openai-compatible"}
	r := llm.NewRouter(
		map[llm.ModelRole]llm.Provider{
			llm.RoleStrong: anthropic, llm.RoleCheap: deepseek, llm.RoleVerifier: anthropic,
		},
		map[llm.ModelRole]string{
			llm.RoleStrong: "claude-sonnet-5", llm.RoleCheap: "deepseek-chat", llm.RoleVerifier: "claude-haiku-4-5",
		})
	ctx := context.Background()

	if _, err := r.Complete(ctx, llm.Request{Tier: llm.TierCheap, Model: "claude-haiku-4-5"}); err != nil {
		t.Fatal(err)
	}
	if got := anthropic.last(t).Model; got != "claude-haiku-4-5" || len(deepseek.got) != 0 {
		t.Errorf("verifier-model request: anthropic got %q, deepseek got %d requests", got, len(deepseek.got))
	}

	if _, err := r.Complete(ctx, llm.Request{Tier: llm.TierCheap}); err != nil {
		t.Fatal(err)
	}
	if got := deepseek.last(t).Model; got != "deepseek-chat" {
		t.Errorf("cheap-tier request reached deepseek with model %q, want deepseek-chat", got)
	}

	if _, err := r.Complete(ctx, llm.Request{Tier: llm.TierStrong}); err != nil {
		t.Fatal(err)
	}
	if got := anthropic.last(t).Model; got != "claude-sonnet-5" || len(anthropic.got) != 2 {
		t.Errorf("strong-tier request: anthropic last model %q after %d requests", got, len(anthropic.got))
	}

	if got := r.ModelFor(llm.TierStrong); got != "claude-sonnet-5" {
		t.Errorf("ModelFor(strong) = %q", got)
	}
	if got := r.ModelFor(llm.TierCheap); got != "deepseek-chat" {
		t.Errorf("ModelFor(cheap) = %q", got)
	}
	if got := r.Name(); got != "anthropic+openai-compatible" {
		t.Errorf("Name() = %q", got)
	}
}

func TestVerifierSharingCheapModelNameRoutesToVerifier(t *testing.T) {
	cheap := &fakeProvider{name: "cheap"}
	verifier := &fakeProvider{name: "verifier"}
	strong := &fakeProvider{name: "strong"}
	r := llm.NewRouter(
		map[llm.ModelRole]llm.Provider{
			llm.RoleStrong: strong, llm.RoleCheap: cheap, llm.RoleVerifier: verifier,
		},
		map[llm.ModelRole]string{
			llm.RoleStrong: "big", llm.RoleCheap: "shared-name", llm.RoleVerifier: "shared-name",
		})

	if _, err := r.Complete(context.Background(), llm.Request{Tier: llm.TierCheap, Model: "shared-name"}); err != nil {
		t.Fatal(err)
	}
	if len(verifier.got) != 1 || len(cheap.got) != 0 {
		t.Errorf("shared model name: verifier got %d, cheap got %d; want the verifier", len(verifier.got), len(cheap.got))
	}

	// A cheap-tier request with no model still reaches the cheap provider.
	if _, err := r.Complete(context.Background(), llm.Request{Tier: llm.TierCheap}); err != nil {
		t.Fatal(err)
	}
	if len(cheap.got) != 1 {
		t.Errorf("tier-only cheap request reached cheap %d times, want 1", len(cheap.got))
	}
}

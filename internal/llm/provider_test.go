package llm_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lajosdeme/mole/internal/llm"
)

func serve(t *testing.T, status int, body string, capture func(*http.Request, []byte)) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		if capture != nil {
			capture(r, raw)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// ---------------------------------------------------------------------------
// OpenAI-compatible
// ---------------------------------------------------------------------------

const openAIFixture = `{
  "model": "deepseek-chat",
  "choices": [{
    "message": {"role":"assistant","content":"MambaByte reports 1.31 BPB on PG-19."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 1500, "completion_tokens": 42, "total_tokens": 1542}
}`

func TestOpenAICompatibleParsesUsage(t *testing.T) {
	var gotReq *http.Request
	var gotBody []byte
	srv := serve(t, 200, openAIFixture, func(r *http.Request, b []byte) {
		gotReq, gotBody = r.Clone(r.Context()), b
	})

	p, err := llm.New(llm.Config{
		Kind: llm.KindOpenAICompatible, APIKey: "sk-test",
		BaseURL: srv.URL, StrongModel: "deepseek-chat",
	}, srv.Client())
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	resp, err := p.Complete(context.Background(), llm.Request{
		Tier:     llm.TierStrong,
		System:   "You extract claims.",
		Messages: []llm.Message{llm.User("Summarize this.")},
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}

	if !strings.Contains(resp.Text, "1.31 BPB") {
		t.Errorf("text = %q", resp.Text)
	}
	if resp.Usage.InputTokens != 1500 || resp.Usage.OutputTokens != 42 {
		t.Errorf("usage = %+v", resp.Usage)
	}
	if resp.StopReason != "stop" {
		t.Errorf("stop reason = %q", resp.StopReason)
	}

	if got := gotReq.Header.Get("Authorization"); got != "Bearer sk-test" {
		t.Errorf("Authorization = %q", got)
	}

	// The system prompt must be a message, not dropped.
	var sent map[string]any
	if err := json.Unmarshal(gotBody, &sent); err != nil {
		t.Fatal(err)
	}
	msgs, _ := sent["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("sent %d messages, want system + user", len(msgs))
	}
	if first, _ := msgs[0].(map[string]any); first["role"] != "system" {
		t.Errorf("first message role = %v, want system", first["role"])
	}
}

// TestCacheTokensAreNotDoubleCounted: vendors report cached input differently,
// and prompt_tokens is the TOTAL including cached. Counting both would charge
// cached input twice — once at full rate, once at the cache rate.
func TestCacheTokensAreNotDoubleCounted(t *testing.T) {
	cases := []struct {
		name              string
		usage             string
		wantInput, wantRD int64
	}{
		{
			name:      "deepseek hit/miss shape",
			usage:     `{"prompt_tokens":1000,"completion_tokens":50,"prompt_cache_hit_tokens":800}`,
			wantInput: 200, wantRD: 800,
		},
		{
			name:      "openai nested shape",
			usage:     `{"prompt_tokens":1000,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":600}}`,
			wantInput: 400, wantRD: 600,
		},
		{
			name:      "no cache reported",
			usage:     `{"prompt_tokens":1000,"completion_tokens":50}`,
			wantInput: 1000, wantRD: 0,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			body := `{"model":"m","choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],"usage":` + c.usage + `}`
			srv := serve(t, 200, body, nil)

			p, _ := llm.New(llm.Config{
				Kind: llm.KindOpenAICompatible, APIKey: "k",
				BaseURL: srv.URL, StrongModel: "m",
			}, srv.Client())

			resp, err := p.Complete(context.Background(), llm.Request{
				Messages: []llm.Message{llm.User("x")},
			})
			if err != nil {
				t.Fatal(err)
			}
			if resp.Usage.InputTokens != c.wantInput {
				t.Errorf("input = %d, want %d", resp.Usage.InputTokens, c.wantInput)
			}
			if resp.Usage.CacheReadTokens != c.wantRD {
				t.Errorf("cache read = %d, want %d", resp.Usage.CacheReadTokens, c.wantRD)
			}
			// Total must equal what the provider actually processed.
			if got := resp.Usage.InputTokens + resp.Usage.CacheReadTokens; got != 1000 {
				t.Errorf("input+cache = %d, want 1000 (double count or loss)", got)
			}
		})
	}
}

// TestMissingUsageIsAnError: a silent zero would charge nothing and make the
// session's ceiling unenforceable for that provider.
func TestMissingUsageIsAnError(t *testing.T) {
	body := `{"model":"m","choices":[{"message":{"content":"an answer"},"finish_reason":"stop"}]}`
	srv := serve(t, 200, body, nil)

	p, _ := llm.New(llm.Config{
		Kind: llm.KindOpenAICompatible, APIKey: "k", BaseURL: srv.URL, StrongModel: "m",
	}, srv.Client())

	resp, err := p.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.User("x")}})
	if !errors.Is(err, llm.ErrNoUsageReported) {
		t.Fatalf("error = %v, want ErrNoUsageReported", err)
	}
	// The text is still returned so a caller can decide, but the error means
	// it cannot be settled silently.
	if resp == nil || resp.Text == "" {
		t.Error("response discarded along with the error")
	}
}

func TestErrorsAreClassifiedForRetryPolicy(t *testing.T) {
	cases := []struct {
		status    int
		sentinel  error
		retryable bool
	}{
		{429, llm.ErrRateLimited, true},
		{401, llm.ErrUnauthorized, false},
		{403, llm.ErrUnauthorized, false},
		{402, llm.ErrQuotaExceeded, false},
		{413, llm.ErrContextTooLong, false},
		{529, llm.ErrOverloaded, true},
		{500, nil, true},
	}

	for _, c := range cases {
		srv := serve(t, c.status, `{"error":{"message":"nope"}}`, nil)
		p, _ := llm.New(llm.Config{
			Kind: llm.KindOpenAICompatible, APIKey: "k", BaseURL: srv.URL, StrongModel: "m",
			// This asserts the status -> sentinel mapping, not the retry
			// policy. Leaving retries on made it back off through every
			// transient case and cost the suite 21 seconds.
			MaxRetries: -1,
		}, srv.Client())

		_, err := p.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.User("x")}})
		if err == nil {
			t.Errorf("status %d produced no error", c.status)
			continue
		}
		if c.sentinel != nil && !errors.Is(err, c.sentinel) {
			t.Errorf("status %d: error = %v, want %v", c.status, err, c.sentinel)
		}
		// The executor branches on this: back off, or fail the lead outright.
		if got := llm.Retryable(err); got != c.retryable {
			t.Errorf("status %d: Retryable = %v, want %v", c.status, got, c.retryable)
		}
	}
}

// TestLocalModelNeedsNoKey is the Ollama case — and the reason CredentialSource
// distinguishes "not required" from "missing".
func TestLocalModelNeedsNoKey(t *testing.T) {
	var sawAuth bool
	srv := serve(t, 200, openAIFixture, func(r *http.Request, _ []byte) {
		sawAuth = r.Header.Get("Authorization") != ""
	})

	p, err := llm.New(llm.Config{
		Kind: llm.KindOpenAICompatible, BaseURL: srv.URL, StrongModel: "qwen2.5:14b",
	}, srv.Client())
	if err != nil {
		t.Fatalf("local model rejected for lacking a key: %v", err)
	}
	if got := llm.SourceOf(p); got != llm.CredentialNotNeeded {
		t.Errorf("credential source = %q, want %q", got, llm.CredentialNotNeeded)
	}
	if _, err := p.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.User("x")}}); err != nil {
		t.Fatalf("complete: %v", err)
	}
	if sawAuth {
		t.Error("sent an Authorization header with no key configured")
	}
}

func TestOpenAICompatibleRequiresAModel(t *testing.T) {
	_, err := llm.New(llm.Config{
		Kind: llm.KindOpenAICompatible, BaseURL: "http://localhost:11434/v1",
	}, nil)
	if err == nil {
		t.Fatal("accepted an openai-compatible config with no model")
	}
	if !strings.Contains(err.Error(), "llm.model") {
		t.Errorf("error should name the setting to fix: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Tiering and construction
// ---------------------------------------------------------------------------

func TestTiersResolveToDifferentModels(t *testing.T) {
	p, err := llm.New(llm.Config{Kind: llm.KindAnthropic, APIKey: "sk-test"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	strong, cheap := p.ModelFor(llm.TierStrong), p.ModelFor(llm.TierCheap)

	if strong == "" || cheap == "" {
		t.Fatalf("tiers unresolved: strong=%q cheap=%q", strong, cheap)
	}
	// The split is the cost lever: chunk mining runs per chunk, planning runs
	// per lead. Collapsing them either overpays or underperforms.
	if strong == cheap {
		t.Errorf("both tiers resolve to %q — no cost separation", strong)
	}
	if strong != llm.DefaultAnthropicStrong {
		t.Errorf("strong = %q, want %q", strong, llm.DefaultAnthropicStrong)
	}
}

func TestCheapFallsBackToStrongWhenUnset(t *testing.T) {
	p, err := llm.New(llm.Config{
		Kind: llm.KindOpenAICompatible, APIKey: "k",
		BaseURL: "http://x/v1", StrongModel: "only-model",
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := p.ModelFor(llm.TierCheap); got != "only-model" {
		t.Errorf("cheap tier = %q, want the strong model as fallback", got)
	}
}

// TestAnthropicWithoutKeyIsAllowed is the zero-config path from §4.2: an empty
// key must NOT be an error, because the SDK's own chain resolves an env var or
// an `ant auth login` profile. Rejecting it here would break the setup that
// needs no configuration at all.
func TestAnthropicWithoutKeyIsAllowed(t *testing.T) {
	p, err := llm.New(llm.Config{Kind: llm.KindAnthropic}, nil)
	if err != nil {
		t.Fatalf("empty key rejected: %v", err)
	}
	if got := llm.SourceOf(p); got != llm.CredentialChain {
		t.Errorf("credential source = %q, want %q", got, llm.CredentialChain)
	}
}

func TestCredentialSourceDistinguishesEnvFromConfig(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "sk-from-env")

	fromEnv, err := llm.New(llm.Config{Kind: llm.KindAnthropic, APIKey: "sk-from-env"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := llm.SourceOf(fromEnv); got != llm.CredentialEnv {
		t.Errorf("source = %q, want %q", got, llm.CredentialEnv)
	}

	fromConfig, err := llm.New(llm.Config{Kind: llm.KindAnthropic, APIKey: "sk-from-config-file"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := llm.SourceOf(fromConfig); got != llm.CredentialConfig {
		t.Errorf("source = %q, want %q", got, llm.CredentialConfig)
	}
}

func TestUnknownProviderRejected(t *testing.T) {
	if _, err := llm.New(llm.Config{Kind: "gemini", APIKey: "k"}, nil); err == nil {
		t.Error("unknown provider accepted")
	}
}

// TestHiddenReasoningTokensAreCharged. Reported as issue #1, confirmed against
// Gemini's OpenAI-compatible endpoint.
//
// A call came back {"prompt_tokens":21,"completion_tokens":5,"total_tokens":139}.
// The 113 tokens between the parts and the total are Gemini's internal
// thinking: billed by the provider, and invisible to mole, which read the two
// parts and never looked at the total. The ledger charged 26 tokens for a call
// that cost 139 — a fivefold undercount, every call, in the direction of
// overspending.
//
// It cannot be reached through the reasoning_tokens path either. That field is
// how OpenAI's o-series reports it, and there reasoning is already INSIDE
// completion_tokens, so the parts and the total agree. Gemini reports neither,
// returning an opaque thought_signature this struct has no field for.
//
// The gap between the parts and the total is what the provider billed and did
// not explain. The OpenAI spec makes total_tokens authoritative, so it is
// charged — as output, because that is where every provider bills reasoning and
// because output is the dearer rate of the two. A table whose job is never to
// under-bill rounds this way on purpose.
func TestHiddenReasoningTokensAreCharged(t *testing.T) {
	for _, tc := range []struct {
		name                  string
		usage                 string
		wantInput, wantOutput int64
	}{
		{
			// The reported case.
			name:  "gemini hides thinking in the total",
			usage: `{"prompt_tokens":21,"completion_tokens":5,"total_tokens":139}`,
			// 5 written + 113 thought about.
			wantInput: 21, wantOutput: 118,
		},
		{
			// OpenAI's shape: reasoning is inside completion_tokens already, so
			// the parts and the total agree and nothing is added.
			name: "openai reports reasoning inside the completion",
			usage: `{"prompt_tokens":21,"completion_tokens":118,"total_tokens":139,
			         "completion_tokens_details":{"reasoning_tokens":113}}`,
			wantInput: 21, wantOutput: 118,
		},
		{
			name:      "no total reported",
			usage:     `{"prompt_tokens":21,"completion_tokens":5}`,
			wantInput: 21, wantOutput: 5,
		},
		{
			// Never downward: a total below the parts is a provider that cannot
			// add up, and believing it would under-bill.
			name:      "a total smaller than its parts is ignored",
			usage:     `{"prompt_tokens":21,"completion_tokens":5,"total_tokens":10}`,
			wantInput: 21, wantOutput: 5,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := `{"model":"m","choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],` +
				`"usage":` + tc.usage + `}`
			srv := serve(t, 200, body, nil)

			p, _ := llm.New(llm.Config{
				Kind: llm.KindOpenAICompatible, APIKey: "k", BaseURL: srv.URL, StrongModel: "m",
			}, srv.Client())

			resp, err := p.Complete(context.Background(), llm.Request{Messages: []llm.Message{llm.User("x")}})
			if err != nil {
				t.Fatal(err)
			}
			if resp.Usage.InputTokens != tc.wantInput {
				t.Errorf("input = %d, want %d", resp.Usage.InputTokens, tc.wantInput)
			}
			if resp.Usage.OutputTokens != tc.wantOutput {
				t.Errorf("output = %d, want %d — the provider billed for tokens mole did not count",
					resp.Usage.OutputTokens, tc.wantOutput)
			}
		})
	}
}

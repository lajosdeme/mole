package llm_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/llm"
)

// sseEvent formats one Server-Sent Event for the fake /v1/messages stream.
func sseEvent(name, data string) string {
	return "event: " + name + "\ndata: " + data + "\n\n"
}

// anthropicStream is a minimal successful SSE response for /v1/messages.
func anthropicStream(model string) string {
	return sseEvent("message_start", fmt.Sprintf(`{"type":"message_start","message":{"type":"message","id":"msg_1","role":"assistant","content":[],"model":%q,"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}`, model)) +
		sseEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`) +
		sseEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`) +
		sseEvent("content_block_stop", `{"type":"content_block_stop","index":0}`) +
		sseEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}`) +
		sseEvent("message_stop", `{"type":"message_stop"}`)
}

// modelInfoBody is a Models API response with the given disabled-thinking
// capability.
func modelInfoBody(model string, disabledSupported bool) string {
	return fmt.Sprintf(`{
		"id": %q,
		"type": "model",
		"display_name": "d",
		"created_at": "2025-01-01T00:00:00Z",
		"max_input_tokens": 1,
		"max_tokens": 1,
		"capabilities": {
			"thinking": {
				"supported": true,
				"types": {
					"adaptive": {"supported": true},
					"enabled": {"supported": true},
					"disabled": {"supported": %t}
				}
			}
		}
	}`, model, disabledSupported)
}

// fakeAnthropicServer answers /v1/models/{id} and /v1/messages, capturing
// every message body sent.
type fakeAnthropicServer struct {
	modelsStatus int
	modelsBody   string
	// modelsRelease, if set, blocks every /v1/models/{id} response until
	// closed — used to hold several lookups in flight at once.
	modelsRelease chan struct{}

	modelsCalls atomic.Int32

	mu     sync.Mutex
	bodies []map[string]any
}

func (f *fakeAnthropicServer) addBody(body map[string]any) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.bodies = append(f.bodies, body)
}

func (f *fakeAnthropicServer) firstBody() map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.bodies[0]
}

func (f *fakeAnthropicServer) bodyCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.bodies)
}

func (f *fakeAnthropicServer) start(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/v1/models/"):
			if f.modelsRelease != nil {
				<-f.modelsRelease
			}
			f.modelsCalls.Add(1)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(f.modelsStatus)
			_, _ = io.WriteString(w, f.modelsBody)
		case r.Method == http.MethodPost && r.URL.Path == "/v1/messages":
			raw, _ := io.ReadAll(r.Body)
			var body map[string]any
			_ = json.Unmarshal(raw, &body)
			f.addBody(body)
			model, _ := body["model"].(string)
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			_, _ = io.WriteString(w, anthropicStream(model))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func anthropicProvider(t *testing.T, srv *httptest.Server, model string) llm.Provider {
	t.Helper()
	// MaxRetries: -1 disables the SDK's own retries on the failing models
	// lookup, which this file's tests are not exercising.
	p, err := llm.New(llm.Config{
		Kind:        llm.KindAnthropic,
		APIKey:      "sk-test",
		BaseURL:     srv.URL,
		StrongModel: model,
		MaxRetries:  -1,
	}, srv.Client())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	return p
}

func TestModelsAPISaysDisabledUnsupported(t *testing.T) {
	f := &fakeAnthropicServer{modelsStatus: 200, modelsBody: modelInfoBody("claude-x", false)}
	srv := f.start(t)
	p := anthropicProvider(t, srv, "claude-x")

	_, err := p.Complete(context.Background(), llm.Request{
		Tier: llm.TierStrong, Messages: []llm.Message{llm.User("hi")},
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}

	if got := f.bodyCount(); got != 1 {
		t.Fatalf("got %d message calls, want 1", got)
	}
	body := f.firstBody()
	if _, ok := body["thinking"]; ok {
		t.Errorf("thinking sent = %v, want omitted", body["thinking"])
	}
	oc, _ := body["output_config"].(map[string]any)
	if oc["effort"] != "low" {
		t.Errorf("output_config.effort = %v, want %q", oc["effort"], "low")
	}
}

func TestModelsAPISaysDisabledSupported(t *testing.T) {
	f := &fakeAnthropicServer{modelsStatus: 200, modelsBody: modelInfoBody("claude-x", true)}
	srv := f.start(t)
	p := anthropicProvider(t, srv, "claude-x")

	_, err := p.Complete(context.Background(), llm.Request{
		Tier: llm.TierStrong, Messages: []llm.Message{llm.User("hi")},
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}

	body := f.firstBody()
	thinking, _ := body["thinking"].(map[string]any)
	if thinking["type"] != "disabled" {
		t.Errorf("thinking.type = %v, want %q", thinking["type"], "disabled")
	}
}

func TestCapabilityLookupIsCached(t *testing.T) {
	f := &fakeAnthropicServer{modelsStatus: 200, modelsBody: modelInfoBody("claude-x", true)}
	srv := f.start(t)
	p := anthropicProvider(t, srv, "claude-x")

	req := llm.Request{Tier: llm.TierStrong, Messages: []llm.Message{llm.User("hi")}}
	if _, err := p.Complete(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if _, err := p.Complete(context.Background(), req); err != nil {
		t.Fatal(err)
	}

	if got := f.modelsCalls.Load(); got != 1 {
		t.Errorf("models calls = %d, want 1 (cached after the first)", got)
	}
}

func TestLookupFailureFallsBackToKnownModels(t *testing.T) {
	cases := []struct {
		model        string
		wantThinking bool // whether a "thinking" field should be sent
	}{
		{"claude-sonnet-5-5", false},
		{"claude-haiku-4-5", true},
	}

	for _, c := range cases {
		t.Run(c.model, func(t *testing.T) {
			f := &fakeAnthropicServer{modelsStatus: 500, modelsBody: `{"error":{"message":"boom"}}`}
			srv := f.start(t)
			p := anthropicProvider(t, srv, c.model)

			_, err := p.Complete(context.Background(), llm.Request{
				Tier: llm.TierStrong, Messages: []llm.Message{llm.User("hi")},
			})
			if err != nil {
				t.Fatalf("complete: %v", err)
			}

			body := f.firstBody()
			_, hasThinking := body["thinking"]
			if hasThinking != c.wantThinking {
				t.Errorf("model %q: thinking present = %v, want %v", c.model, hasThinking, c.wantThinking)
			}
			if !c.wantThinking {
				oc, _ := body["output_config"].(map[string]any)
				if oc["effort"] != "low" {
					t.Errorf("output_config.effort = %v, want %q", oc["effort"], "low")
				}
			}
		})
	}
}

func TestExplicitEffortIsKept(t *testing.T) {
	f := &fakeAnthropicServer{modelsStatus: 200, modelsBody: modelInfoBody("claude-x", false)}
	srv := f.start(t)
	p := anthropicProvider(t, srv, "claude-x")

	_, err := p.Complete(context.Background(), llm.Request{
		Tier: llm.TierStrong, Messages: []llm.Message{llm.User("hi")}, Effort: "high",
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}

	body := f.firstBody()
	if _, ok := body["thinking"]; ok {
		t.Errorf("thinking sent = %v, want omitted", body["thinking"])
	}
	oc, _ := body["output_config"].(map[string]any)
	if oc["effort"] != "high" {
		t.Errorf("output_config.effort = %v, want %q (kept, not overridden)", oc["effort"], "high")
	}
}

// TestConcurrentLookupsAreCoalesced: N callers racing on the same uncached
// model must share one Models API call, not each issue their own.
func TestConcurrentLookupsAreCoalesced(t *testing.T) {
	release := make(chan struct{})
	f := &fakeAnthropicServer{
		modelsStatus:  200,
		modelsBody:    modelInfoBody("claude-x", true),
		modelsRelease: release,
	}
	srv := f.start(t)
	p := anthropicProvider(t, srv, "claude-x")

	const n = 20
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			_, err := p.Complete(context.Background(), llm.Request{
				Tier: llm.TierStrong, Messages: []llm.Message{llm.User("hi")},
			})
			if err != nil {
				t.Errorf("complete: %v", err)
			}
		}()
	}
	// Give every goroutine time to reach the lookup and queue behind the one
	// in-flight call before the models endpoint is allowed to answer.
	time.Sleep(20 * time.Millisecond)
	close(release)
	wg.Wait()

	if got := f.modelsCalls.Load(); got != 1 {
		t.Errorf("models calls = %d, want 1 (concurrent lookups not coalesced)", got)
	}
	if got := f.bodyCount(); got != n {
		t.Errorf("message calls = %d, want %d", got, n)
	}
}

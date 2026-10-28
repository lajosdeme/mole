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

// capabilityModelBody is a full Models API object with the given disabled-thinking and effort levels.
func capabilityModelBody(model string, disabledSupported bool, effortLevels ...string) string {
	level := func(name string) string {
		for _, l := range effortLevels {
			if l == name {
				return `{"supported": true}`
			}
		}
		return `{"supported": false}`
	}
	return fmt.Sprintf(`{
		"id": %q,
		"type": "model",
		"display_name": "d",
		"created_at": "2025-01-01T00:00:00Z",
		"max_input_tokens": 200000,
		"max_tokens": 64000,
		"capabilities": {
			"batch": {"supported": true},
			"citations": {"supported": true},
			"code_execution": {"supported": true},
			"context_management": {"supported": false},
			"image_input": {"supported": true},
			"pdf_input": {"supported": true},
			"structured_outputs": {"supported": true},
			"effort": {
				"supported": %t,
				"low": %s, "medium": %s, "high": %s, "xhigh": %s, "max": %s
			},
			"thinking": {
				"supported": true,
				"types": {
					"adaptive": {"supported": true},
					"enabled": {"supported": true},
					"disabled": {"supported": %t}
				}
			}
		}
	}`, model, len(effortLevels) > 0, level("low"), level("medium"), level("high"),
		level("xhigh"), level("max"), disabledSupported)
}

// sentEffort returns the output_config.effort of a captured body and whether it was present.
func sentEffort(body map[string]any) (any, bool) {
	oc, _ := body["output_config"].(map[string]any)
	v, ok := oc["effort"]
	return v, ok
}

// TestEffortUnsupportedSendsNoEffort checks a model without effort gets no output_config.effort.
func TestEffortUnsupportedSendsNoEffort(t *testing.T) {
	cases := []struct {
		name         string
		disabled     bool
		effort       string
		wantThinking any
	}{
		{"requested effort, disabled thinking", true, "medium", "disabled"},
		{"no effort, no disabled thinking", false, "", nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := &fakeAnthropicServer{modelsStatus: 200, modelsBody: capabilityModelBody("claude-x", c.disabled)}
			p := anthropicProvider(t, f.start(t), "claude-x")

			_, err := p.Complete(context.Background(), llm.Request{
				Tier: llm.TierStrong, Messages: []llm.Message{llm.User("hi")}, Effort: c.effort,
			})
			if err != nil {
				t.Fatalf("complete: %v", err)
			}
			body := f.firstBody()
			if v, ok := sentEffort(body); ok {
				t.Errorf("output_config.effort = %v, want omitted", v)
			}
			thinking, _ := body["thinking"].(map[string]any)
			if got := thinking["type"]; got != c.wantThinking {
				t.Errorf("thinking.type = %v, want %v", got, c.wantThinking)
			}
		})
	}
}

// TestEffortClampedToHighestSupported checks an unsupported level drops to the next supported one.
func TestEffortClampedToHighestSupported(t *testing.T) {
	for _, req := range []string{"xhigh", "max"} {
		t.Run(req, func(t *testing.T) {
			f := &fakeAnthropicServer{modelsStatus: 200,
				modelsBody: capabilityModelBody("claude-x", true, "low", "medium", "high")}
			p := anthropicProvider(t, f.start(t), "claude-x")

			_, err := p.Complete(context.Background(), llm.Request{
				Tier: llm.TierStrong, Messages: []llm.Message{llm.User("hi")}, Effort: req,
			})
			if err != nil {
				t.Fatalf("complete: %v", err)
			}
			if v, _ := sentEffort(f.firstBody()); v != "high" {
				t.Errorf("output_config.effort = %v, want %q", v, "high")
			}
		})
	}
}

// TestEffortSupportedIsPassed checks a supported level reaches the request unchanged.
func TestEffortSupportedIsPassed(t *testing.T) {
	f := &fakeAnthropicServer{modelsStatus: 200,
		modelsBody: capabilityModelBody("claude-x", false, "low", "medium", "high", "xhigh", "max")}
	p := anthropicProvider(t, f.start(t), "claude-x")

	for _, req := range []string{"", "medium", "xhigh"} {
		if _, err := p.Complete(context.Background(), llm.Request{
			Tier: llm.TierStrong, Messages: []llm.Message{llm.User("hi")}, Effort: req,
		}); err != nil {
			t.Fatalf("complete: %v", err)
		}
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	for i, want := range []string{"low", "medium", "xhigh"} {
		if v, _ := sentEffort(f.bodies[i]); v != want {
			t.Errorf("call %d: output_config.effort = %v, want %q", i, v, want)
		}
	}
}

// TestEffortLookupFailureFallsBackToKnownModels checks the prefix fallback when the Models API fails.
func TestEffortLookupFailureFallsBackToKnownModels(t *testing.T) {
	cases := []struct {
		model, effort string
		want          any
	}{
		{"claude-sonnet-4-5", "medium", nil},
		{"claude-haiku-4-5", "medium", nil},
		{"claude-opus-4-1", "medium", nil},
		{"claude-3-7-sonnet-latest", "high", nil},
		{"claude-sonnet-4-6", "xhigh", "high"},
		{"claude-opus-4-5", "max", "high"},
		{"claude-opus-4-6", "medium", "medium"},
		{"claude-opus-5-5", "medium", "medium"},
		{"claude-opus-5-5", "xhigh", "xhigh"},
	}
	for _, c := range cases {
		t.Run(c.model+"/"+c.effort, func(t *testing.T) {
			f := &fakeAnthropicServer{modelsStatus: 404, modelsBody: `{"error":{"message":"nope"}}`}
			p := anthropicProvider(t, f.start(t), c.model)

			_, err := p.Complete(context.Background(), llm.Request{
				Tier: llm.TierStrong, Messages: []llm.Message{llm.User("hi")}, Effort: c.effort,
			})
			if err != nil {
				t.Fatalf("complete: %v", err)
			}
			if v, _ := sentEffort(f.firstBody()); v != c.want {
				t.Errorf("output_config.effort = %v, want %v", v, c.want)
			}
		})
	}
}

// TestThinkingAndEffortShareOneLookup checks both capability checks use a single Models API call.
func TestThinkingAndEffortShareOneLookup(t *testing.T) {
	f := &fakeAnthropicServer{modelsStatus: 200, modelsBody: capabilityModelBody("claude-x", false, "low", "medium", "high")}
	p := anthropicProvider(t, f.start(t), "claude-x")

	for _, req := range []llm.Request{
		{Tier: llm.TierStrong, Messages: []llm.Message{llm.User("hi")}, Effort: "xhigh"},
		{Tier: llm.TierStrong, Messages: []llm.Message{llm.User("hi")}},
		{Tier: llm.TierStrong, Messages: []llm.Message{llm.User("hi")}, Thinking: true, Effort: "medium"},
	} {
		if _, err := p.Complete(context.Background(), req); err != nil {
			t.Fatalf("complete: %v", err)
		}
	}
	if got := f.modelsCalls.Load(); got != 1 {
		t.Errorf("models calls = %d, want 1", got)
	}
}

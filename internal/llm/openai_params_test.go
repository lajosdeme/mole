package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestOpenAIEndpointGetsCompletionTokensAndEffort: api.openai.com receives
// max_completion_tokens and reasoning_effort; other endpoints keep max_tokens.
func TestOpenAIEndpointGetsCompletionTokensAndEffort(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		got = map[string]any{}
		_ = json.Unmarshal(b, &got)
		_, _ = w.Write([]byte(`{"model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
	}))
	defer srv.Close()

	for _, tc := range []struct {
		openAI bool
		effort string
		want   string
	}{
		{true, "medium", "medium"},
		{true, "max", "high"},
		{false, "medium", ""},
	} {
		p, err := New(Config{Kind: KindOpenAICompatible, BaseURL: srv.URL, APIKey: "k", StrongModel: "m"}, srv.Client())
		if err != nil {
			t.Fatal(err)
		}
		p.(*openAIProvider).openAI = tc.openAI
		if _, err := p.Complete(context.Background(), Request{Effort: tc.effort, MaxTokens: 100,
			Messages: []Message{User("hi")}}); err != nil {
			t.Fatal(err)
		}
		_, hasMax := got["max_tokens"]
		_, hasCompletion := got["max_completion_tokens"]
		if tc.openAI && (hasMax || !hasCompletion) {
			t.Errorf("openai endpoint sent %v", got)
		}
		if !tc.openAI && (!hasMax || hasCompletion) {
			t.Errorf("other endpoint sent %v", got)
		}
		if e, _ := got["reasoning_effort"].(string); e != tc.want {
			t.Errorf("reasoning_effort = %q, want %q", e, tc.want)
		}
	}
}

func TestOpenAIEndpointIsDetectedByHost(t *testing.T) {
	for url, want := range map[string]bool{
		"https://api.openai.com/v1":   true,
		"https://api.deepseek.com/v1": false,
		"http://localhost:11434/v1":   false,
	} {
		p, err := New(Config{Kind: KindOpenAICompatible, BaseURL: url, APIKey: "k", StrongModel: "m"}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if got := p.(*openAIProvider).openAI; got != want {
			t.Errorf("%s: openAI = %v, want %v", url, got, want)
		}
	}
}

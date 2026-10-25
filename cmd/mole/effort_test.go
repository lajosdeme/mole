package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/lajosdeme/mole/internal/config"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/output"
	"github.com/lajosdeme/mole/internal/planner"
	"github.com/lajosdeme/mole/internal/store/sqlite"
)

// sonnet45Info is a Models API object for a model with no effort and with disabled thinking.
const sonnet45Info = `{"id":"claude-sonnet-4-5","type":"model","display_name":"Claude Sonnet 4.5",
"created_at":"2025-09-29T00:00:00Z","max_input_tokens":200000,"max_tokens":64000,
"capabilities":{"batch":{"supported":true},"citations":{"supported":true},
"code_execution":{"supported":true},"context_management":{"supported":true},
"image_input":{"supported":true},"pdf_input":{"supported":true},"structured_outputs":{"supported":true},
"effort":{"supported":false,"low":{"supported":false},"medium":{"supported":false},
"high":{"supported":false},"xhigh":{"supported":false},"max":{"supported":false}},
"thinking":{"supported":true,"types":{"adaptive":{"supported":false},
"enabled":{"supported":true},"disabled":{"supported":true}}}}}`

// effortRecordingServer fakes the Anthropic API and records every message body.
func effortRecordingServer(t *testing.T, modelsStatus int, modelsBody string) (*httptest.Server, func() []map[string]any) {
	t.Helper()
	var mu sync.Mutex
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/v1/models/"):
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(modelsStatus)
			_, _ = io.WriteString(w, modelsBody)
		case r.Method == http.MethodPost && r.URL.Path == "/v1/messages":
			raw, _ := io.ReadAll(r.Body)
			var body map[string]any
			_ = json.Unmarshal(raw, &body)
			mu.Lock()
			bodies = append(bodies, body)
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, effortSSE(fmt.Sprint(body["model"])))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(srv.Close)
	return srv, func() []map[string]any {
		mu.Lock()
		defer mu.Unlock()
		return append([]map[string]any(nil), bodies...)
	}
}

// effortSSE is a minimal successful message stream.
func effortSSE(model string) string {
	ev := func(name, data string) string { return "event: " + name + "\ndata: " + data + "\n\n" }
	return ev("message_start", fmt.Sprintf(`{"type":"message_start","message":{"type":"message","id":"m","role":"assistant","content":[],"model":%q,"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}`, model)) +
		ev("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`) +
		ev("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`) +
		ev("content_block_stop", `{"type":"content_block_stop","index":0}`) +
		ev("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}`) +
		ev("message_stop", `{"type":"message_stop"}`)
}

// TestOlderStrongModelPlansAndReportsWithoutEffort checks planner and report calls omit effort for claude-sonnet-4-5.
func TestOlderStrongModelPlansAndReportsWithoutEffort(t *testing.T) {
	cases := []struct {
		name         string
		modelsStatus int
		modelsBody   string
	}{
		{"models api", http.StatusOK, sonnet45Info},
		{"lookup failure", http.StatusNotFound, `{"error":{"message":"nope"}}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			srv, bodies := effortRecordingServer(t, c.modelsStatus, c.modelsBody)
			cfg := &config.Config{}
			cfg.LLM = config.LLMConfig{APIKey: "sk-ant-api03-test", BaseURL: srv.URL, Model: "claude-sonnet-4-5"}
			p, err := buildConfiguredLLM(cfg, srv.Client(), llm.New)
			if err != nil {
				t.Fatal(err)
			}

			path := filepath.Join(t.TempDir(), "t.db")
			sid, sess := seedReportSession(t, path, true)
			_, _ = (&planner.Planner{LLM: p}).InitialLeads(context.Background(), sess)
			db, err := sqlite.Open(path, sqlite.Options{})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = db.Close() })
			if _, err := (&output.Generator{LLM: p}).Generate(context.Background(), db, sid); err != nil {
				t.Fatal(err)
			}

			got := bodies()
			if len(got) != 2 {
				t.Fatalf("message calls = %d, want 2 (planner, report)", len(got))
			}
			for i, b := range got {
				if b["model"] != "claude-sonnet-4-5" {
					t.Errorf("call %d: model = %v", i, b["model"])
				}
				if oc, ok := b["output_config"].(map[string]any); ok && oc["effort"] != nil {
					t.Errorf("call %d: output_config.effort = %v, want omitted", i, oc["effort"])
				}
			}
		})
	}
}

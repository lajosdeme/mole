package main

import (
	"bytes"
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/lajosdeme/mole/internal/config"
	"github.com/lajosdeme/mole/internal/llm"
)

// failingProvider rejects every call as unauthorized.
type failingProvider struct{ recordingProvider }

func (p *failingProvider) Complete(_ context.Context, req llm.Request) (*llm.Response, error) {
	p.got = append(p.got, req)
	return nil, llm.ErrUnauthorized
}

func mixedTestLLMConfig() *config.Config {
	cfg := &config.Config{}
	cfg.LLM = config.LLMConfig{
		Provider: "openai-compatible", APIKey: "sk-deepseeksecretvalue", BaseURL: "https://api.deepseek.com/v1",
		Model: "deepseek-chat", CheapModel: "deepseek-chat",
	}
	cfg.LLM.Strong = &config.LLMRole{Provider: "anthropic", APIKey: "sk-ant-api03-secretvalue", Model: "claude-opus-5-5"}
	return cfg
}

// TestTestLLMCallsEachBackendOnce checks one call and one named line per backend.
func TestTestLLMCallsEachBackendOnce(t *testing.T) {
	factory, built := recordingFactory()
	var out bytes.Buffer

	if err := testLLMBackends(context.Background(), &out, mixedTestLLMConfig(), nil, factory); err != nil {
		t.Fatalf("err = %v\n%s", err, out.String())
	}
	if len(*built) != 2 {
		t.Fatalf("built %d providers, want 2", len(*built))
	}
	for _, p := range *built {
		if len(p.got) != 1 {
			t.Errorf("%s backend got %d calls, want 1", p.cfg.Kind, len(p.got))
		}
	}
	for _, want := range []string{"strong: claude-opus-5-5", "cheap/verifier: deepseek-chat"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output lacks %q:\n%s", want, out.String())
		}
	}
	if strings.Contains(out.String(), "secretvalue") {
		t.Errorf("output printed a key:\n%s", out.String())
	}
}

// TestTestLLMFailsWhenAnyBackendFails checks one failing backend fails the command after both are tried.
func TestTestLLMFailsWhenAnyBackendFails(t *testing.T) {
	var calls []*recordingProvider
	factory := func(c llm.Config, _ *http.Client) (llm.Provider, error) {
		if c.Kind == llm.KindAnthropic {
			p := &failingProvider{recordingProvider{cfg: c}}
			calls = append(calls, &p.recordingProvider)
			return p, nil
		}
		p := &recordingProvider{cfg: c}
		calls = append(calls, p)
		return p, nil
	}
	var out bytes.Buffer

	err := testLLMBackends(context.Background(), &out, mixedTestLLMConfig(), nil, factory)
	if err == nil {
		t.Fatalf("a rejected strong backend passed:\n%s", out.String())
	}
	if len(calls) != 2 || len(calls[0].got) != 1 || len(calls[1].got) != 1 {
		t.Errorf("want one call on each of 2 backends, got %d backends", len(calls))
	}
	if !strings.Contains(out.String(), "strong") || !strings.Contains(out.String(), "credential rejected") {
		t.Errorf("output does not name the failing role:\n%s", out.String())
	}
	if strings.Contains(out.String()+err.Error(), "secretvalue") {
		t.Errorf("output printed a key:\n%s", out.String())
	}
}

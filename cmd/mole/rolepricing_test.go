package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lajosdeme/mole/internal/config"
	"github.com/lajosdeme/mole/internal/pricing"
)

func TestALocalTopLevelKeepsAHostedRolesRates(t *testing.T) {
	cfg := &config.Config{}
	cfg.LLM = config.LLMConfig{Provider: "openai-compatible", BaseURL: "http://localhost:11434/v1", Model: "qwen3:4b"}
	cfg.LLM.Cheap = &config.LLMRole{
		Provider: "openai-compatible", BaseURL: "https://api.deepseek.com/v1", APIKey: "sk-x", Model: "exotic-hosted",
	}
	cfg.Pricing = map[string]pricing.RateConfig{"exotic-hosted": {Input: 1, Output: 3}}
	table := pricingFor(cfg)

	if r, ok := table.Lookup("exotic-hosted"); !ok || r.Input == 0 {
		t.Errorf("hosted cheap model rates = %+v ok=%v, want its declared paid rate", r, ok)
	}
	if r, ok := table.Lookup("qwen3:4b"); !ok || r.Input != 0 || r.Output != 0 {
		t.Errorf("local model rates = %+v ok=%v, want free", r, ok)
	}
	c, err := table.Cost("exotic-hosted-renamed", pricing.Usage{InputTokens: 1_000_000})
	if err != nil || c.USDMicros == 0 {
		t.Errorf("a renamed hosted model cost %+v (err %v), want the hosted fallback rate", c, err)
	}
}

func TestAHostedTopLevelFreesALocalRole(t *testing.T) {
	cfg := &config.Config{}
	cfg.LLM = config.LLMConfig{Provider: "anthropic", APIKey: "sk-ant-api03-top", Model: "claude-sonnet-5"}
	cfg.LLM.Cheap = &config.LLMRole{
		Provider: "openai-compatible", BaseURL: "http://localhost:11434/v1", Model: "qwen3:8b",
	}

	if r, ok := pricingFor(cfg).Lookup("qwen3:8b"); !ok || r.Input != 0 || r.Output != 0 {
		t.Errorf("local cheap model rates = %+v ok=%v, want free", r, ok)
	}
	factory, _ := recordingFactory()
	p, err := buildConfiguredLLM(cfg, nil, factory)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkUSDIsEnforceable(cfg, p); err != nil {
		t.Errorf("--usd refused with a free local cheap role: %v", err)
	}
}

func TestAnUnpricedVerifierModelRefusesUSD(t *testing.T) {
	cfg := &config.Config{}
	cfg.LLM = config.LLMConfig{
		Provider: "anthropic", APIKey: "sk-ant-api03-top", Model: "claude-sonnet-5",
		CheapModel: "claude-haiku-4-5", VerifierModel: "unlisted-verifier",
	}
	factory, _ := recordingFactory()
	p, err := buildConfiguredLLM(cfg, nil, factory)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkUSDIsEnforceable(cfg, p); err == nil || !strings.Contains(err.Error(), "unlisted-verifier") {
		t.Errorf("err = %v, want --usd refused naming unlisted-verifier", err)
	}
}

// localServer answers like a running local runtime.
func localServer(t *testing.T) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"data":[]}`))
	}))
	t.Cleanup(srv.Close)
	return srv.URL + "/v1"
}

func TestDoctorJudgesAMixedConfigPerRole(t *testing.T) {
	t.Setenv("MOLE_CONFIG_DIR", t.TempDir())
	cfg := &config.Config{}
	cfg.LLM = config.LLMConfig{Provider: "openai-compatible", BaseURL: localServer(t), Model: "qwen3:4b"}
	cfg.LLM.Cheap = &config.LLMRole{
		Provider: "openai-compatible", BaseURL: "https://api.deepseek.com/v1", APIKey: "sk-deepseeksecretvalue",
		Model: "unlisted-model",
	}

	var greens, reds []string
	report := func(ok bool, label, detail string) {
		if ok {
			greens = append(greens, label+" "+detail)
		} else {
			reds = append(reds, label+" "+detail)
		}
	}
	r := &checks{}
	out := captureStdout(t, func() {
		built := reportLLM(report, cfg)
		reportLLMRoles(r, cfg, built)
		reportPricing(r, cfg)
	})

	for _, g := range greens {
		if strings.Contains(g, "llm provider") {
			t.Errorf("a hosted unpriced role got a green provider line: %q", g)
		}
	}
	if len(reds) == 0 || !strings.Contains(strings.Join(reds, "\n"), "unlisted-model") {
		t.Errorf("provider line did not flag the unpriced hosted role: %v", reds)
	}
	for _, want := range []string{
		"openai-compatible qwen3:4b, priced local, free",
		"openai-compatible unlisted-model, priced no",
		"no rate for unlisted-model",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("doctor output lacks %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "billed at zero because they are free") {
		t.Errorf("doctor called a hosted role free:\n%s", out)
	}
	if strings.Contains(out+strings.Join(reds, ""), "secretvalue") {
		t.Errorf("doctor printed a key")
	}
}

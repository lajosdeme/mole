package main

import (
	"context"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/lajosdeme/mole/internal/config"
	"github.com/lajosdeme/mole/internal/llm"
)

// The flag-permutation tests that lived here are gone with parseArgs: pflag
// parses interspersed flags and positionals natively, so there is no longer a
// shim to test. TestFlagsAfterPositionalsAreParsed survives as a CLI-level
// assertion in cli_test.go, because the BEHAVIOUR still matters — it is the bug
// that prompted the port.

// ---------------------------------------------------------------------------
// Key / provider mismatch
// ---------------------------------------------------------------------------

// TestMismatchedKeyIsRefusedRatherThanGreen. llm.provider defaults to anthropic
// when unset, so a Groq key set on its own produced a doctor line reading
// "✓ llm provider claude-opus-5 via config" followed by an auth failure at the
// first model call. A check that runs and passes wrongly is worse than no
// check.
func TestMismatchedKeyIsRefusedRatherThanGreen(t *testing.T) {
	cfg := &config.Config{}
	cfg.LLM.APIKey = "gsk_abc123"

	err := checkKeyMatchesProvider(cfg, llm.KindAnthropic)
	if err == nil {
		t.Fatal("a Groq key defaulting to anthropic was accepted")
	}
	// The message has to be actionable: name the vendor and the exact settings.
	for _, want := range []string{"Groq", "llm.provider openai-compatible", "api.groq.com", "--tokens"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error does not mention %q:\n%s", want, err)
		}
	}
}

// TestExplicitProviderIsTheUsersDecision: only the silent default is blocked.
// A proxy that accepts an OpenAI-shaped key at an Anthropic endpoint is
// unusual, not impossible, and the user said so explicitly.
func TestExplicitProviderIsTheUsersDecision(t *testing.T) {
	cfg := &config.Config{}
	cfg.LLM.APIKey = "gsk_abc123"
	cfg.LLM.Provider = "anthropic"

	if err := checkKeyMatchesProvider(cfg, llm.KindAnthropic); err != nil {
		t.Errorf("an explicitly configured provider was overridden: %v", err)
	}
}

func TestMatchingAndUnknownKeysPass(t *testing.T) {
	cases := []struct {
		key  string
		kind llm.Kind
	}{
		{"sk-ant-api03-xxx", llm.KindAnthropic},
		{"gsk_abc", llm.KindOpenAICompatible},
		{"sk-proj-abc", llm.KindOpenAICompatible},
		{"my-self-hosted-token", llm.KindAnthropic}, // unrecognized: not our business
		{"", llm.KindAnthropic},                     // no key: the SDK chain resolves it
	}
	for _, tc := range cases {
		cfg := &config.Config{}
		cfg.LLM.APIKey = tc.key
		if err := checkKeyMatchesProvider(cfg, tc.kind); err != nil {
			t.Errorf("key %q with kind %s was refused: %v", tc.key, tc.kind, err)
		}
	}
}

// TestVendorPrefixOrdering: "sk-ant-" must win over the broader "sk-".
func TestVendorPrefixOrdering(t *testing.T) {
	v, ok := llm.VendorFromKey("sk-ant-api03-xxx")
	if !ok || v.Kind != llm.KindAnthropic {
		t.Errorf("sk-ant- resolved to %+v, want Anthropic", v)
	}
	v, ok = llm.VendorFromKey("sk-abc")
	if !ok || v.Kind != llm.KindOpenAICompatible {
		t.Errorf("sk- resolved to %+v, want OpenAI-compatible", v)
	}
	if _, ok := llm.VendorFromKey("totally-opaque"); ok {
		t.Error("an unrecognized key was assigned a vendor")
	}
}

// ---------------------------------------------------------------------------
// Per-role providers
// ---------------------------------------------------------------------------

// recordingProvider stands in for a built backend and records what reaches it.
type recordingProvider struct {
	cfg llm.Config
	got []llm.Request
}

func (p *recordingProvider) Complete(_ context.Context, req llm.Request) (*llm.Response, error) {
	p.got = append(p.got, req)
	return &llm.Response{Text: "ok", Model: p.modelOf(req)}, nil
}

func (p *recordingProvider) modelOf(req llm.Request) string {
	if req.Model != "" {
		return req.Model
	}
	return p.ModelFor(req.Tier)
}

func (p *recordingProvider) Name() string { return string(p.cfg.Kind) }

func (p *recordingProvider) ModelFor(tier llm.Tier) string {
	if tier == llm.TierCheap {
		return p.cfg.CheapModel
	}
	return p.cfg.StrongModel
}

// recordingFactory returns a provider factory and the providers it built.
func recordingFactory() (func(llm.Config, *http.Client) (llm.Provider, error), *[]*recordingProvider) {
	var built []*recordingProvider
	return func(c llm.Config, _ *http.Client) (llm.Provider, error) {
		p := &recordingProvider{cfg: c}
		built = append(built, p)
		return p, nil
	}, &built
}

func TestNoRoleBlocksBuildsTheSingleProvider(t *testing.T) {
	cfg := &config.Config{}
	cfg.LLM = config.LLMConfig{
		Provider: "anthropic", APIKey: "sk-ant-api03-top", Model: "claude-opus-5",
		CheapModel: "claude-haiku-4-5", VerifierModel: "claude-sonnet-5",
	}
	factory, built := recordingFactory()

	p, err := buildConfiguredLLM(cfg, nil, factory)
	if err != nil {
		t.Fatal(err)
	}
	if len(*built) != 1 {
		t.Fatalf("built %d providers, want 1", len(*built))
	}
	single := (*built)[0]
	if p != llm.Provider(single) {
		t.Fatalf("got %T, want the single provider itself rather than a router", p)
	}
	want := llm.Config{
		Kind: llm.KindAnthropic, APIKey: "sk-ant-api03-top",
		StrongModel: "claude-opus-5", CheapModel: "claude-haiku-4-5",
	}
	if single.cfg != want {
		t.Errorf("provider config = %+v, want %+v", single.cfg, want)
	}

	req := llm.Request{Tier: llm.TierCheap, Model: "claude-sonnet-5", System: "s", MaxTokens: 7}
	if _, err := p.Complete(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(single.got, []llm.Request{req}) {
		t.Errorf("request changed on the way: %+v", single.got)
	}
	if got := verifierModelFor(cfg); got != "claude-sonnet-5" {
		t.Errorf("verifier model = %q, want the top-level one", got)
	}
}

func TestRoleBlockInheritsTopLevel(t *testing.T) {
	cfg := &config.Config{}
	cfg.LLM = config.LLMConfig{Provider: "anthropic", APIKey: "sk-ant-api03-top", Model: "claude-opus-5"}
	cfg.LLM.Cheap = &config.LLMRole{Model: "claude-haiku-4-5"}
	factory, built := recordingFactory()

	p, err := buildConfiguredLLM(cfg, nil, factory)
	if err != nil {
		t.Fatal(err)
	}
	if len(*built) != 1 || p != llm.Provider((*built)[0]) {
		t.Fatalf("a model-only block built %d providers (%T), want the one top-level provider", len(*built), p)
	}
	anthropic := (*built)[0]
	if anthropic.cfg.Kind != llm.KindAnthropic || anthropic.cfg.APIKey != "sk-ant-api03-top" {
		t.Errorf("cheap role did not inherit the top-level provider: %+v", anthropic.cfg)
	}

	resp, err := p.Complete(context.Background(), llm.Request{Tier: llm.TierCheap})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Model != "claude-haiku-4-5" {
		t.Errorf("cheap request ran on %q, want claude-haiku-4-5", resp.Model)
	}
	if got := p.ModelFor(llm.TierStrong); got != "claude-opus-5" {
		t.Errorf("strong model = %q, want the top-level claude-opus-5", got)
	}
}

func TestMixedRolesBuildARouter(t *testing.T) {
	cfg := &config.Config{}
	cfg.LLM = config.LLMConfig{Provider: "anthropic", APIKey: "sk-ant-api03-top", Model: "claude-sonnet-5"}
	cfg.LLM.Cheap = &config.LLMRole{
		Provider: "openai-compatible", APIKey: "sk-deepseek", BaseURL: "https://api.deepseek.com/v1",
		Model: "deepseek-chat",
	}
	cfg.LLM.Verifier = &config.LLMRole{Model: "claude-haiku-4-5"}
	factory, built := recordingFactory()

	p, err := buildConfiguredLLM(cfg, nil, factory)
	if err != nil {
		t.Fatal(err)
	}
	if len(*built) != 2 {
		t.Fatalf("built %d providers, want one per distinct backend (2)", len(*built))
	}
	anthropic, deepseek := (*built)[0], (*built)[1]
	ctx := context.Background()

	if _, err := p.Complete(ctx, llm.Request{Tier: llm.TierCheap, Model: verifierModelFor(cfg)}); err != nil {
		t.Fatal(err)
	}
	if _, err := p.Complete(ctx, llm.Request{Tier: llm.TierCheap}); err != nil {
		t.Fatal(err)
	}
	if len(anthropic.got) != 1 || anthropic.got[0].Model != "claude-haiku-4-5" {
		t.Errorf("verifier request: anthropic got %+v", anthropic.got)
	}
	if len(deepseek.got) != 1 || deepseek.got[0].Model != "deepseek-chat" {
		t.Errorf("miner request: deepseek got %+v", deepseek.got)
	}
	if got := p.Name(); got != "anthropic+openai-compatible" {
		t.Errorf("Name() = %q", got)
	}
}

func TestRoleKeyMismatchIsCheckedPerRole(t *testing.T) {
	cfg := &config.Config{}
	cfg.LLM = config.LLMConfig{Provider: "anthropic", APIKey: "sk-ant-api03-top"}
	cfg.LLM.Cheap = &config.LLMRole{APIKey: "gsk_abc123", Model: "llama-3.1-8b-instant"}
	factory, _ := recordingFactory()

	_, err := buildConfiguredLLM(cfg, nil, factory)
	if err == nil {
		t.Fatal("a Groq key on the cheap role, inheriting anthropic, was accepted")
	}
	if !strings.Contains(err.Error(), "llm.cheap.provider") {
		t.Errorf("error does not name the role's setting:\n%s", err)
	}
	if strings.Contains(err.Error(), "gsk_abc123") {
		t.Errorf("error printed the key:\n%s", err)
	}
}

func TestVerifierOnItsOwnProviderNeedsAModel(t *testing.T) {
	cfg := &config.Config{}
	cfg.LLM = config.LLMConfig{Provider: "anthropic", APIKey: "sk-ant-api03-top"}
	cfg.LLM.Verifier = &config.LLMRole{Provider: "openai-compatible", BaseURL: "https://api.deepseek.com/v1"}
	factory, _ := recordingFactory()

	if _, err := buildConfiguredLLM(cfg, nil, factory); err == nil || !strings.Contains(err.Error(), "llm.verifier.model") {
		t.Errorf("err = %v, want one naming llm.verifier.model", err)
	}
}

func TestDoctorPrintsOneLinePerRoleWithoutKeys(t *testing.T) {
	cfg := &config.Config{}
	cfg.LLM = config.LLMConfig{Provider: "anthropic", APIKey: "sk-ant-api03-topsecretvalue", Model: "claude-sonnet-5"}
	cfg.LLM.Cheap = &config.LLMRole{
		Provider: "openai-compatible", APIKey: "sk-deepseeksecretvalue", BaseURL: "https://api.deepseek.com/v1",
		Model: "unlisted-model",
	}
	cfg.LLM.Verifier = &config.LLMRole{Model: "claude-haiku-4-5"}
	factory, _ := recordingFactory()
	built, err := buildConfiguredLLM(cfg, nil, factory)
	if err != nil {
		t.Fatal(err)
	}

	out := captureStdout(t, func() { reportLLMRoles(&checks{}, cfg, built) })
	for _, want := range []string{
		"llm strong", "anthropic claude-sonnet-5, priced yes",
		"llm cheap", "openai-compatible unlisted-model, priced no",
		"llm verifier", "anthropic claude-haiku-4-5, priced yes",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("doctor output lacks %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "secretvalue") {
		t.Errorf("doctor printed a key:\n%s", out)
	}
}

func TestRoleOnAnotherEndpointGetsNoTopLevelKey(t *testing.T) {
	cfg := &config.Config{}
	cfg.LLM = config.LLMConfig{Provider: "anthropic", APIKey: "sk-ant-api03-top", Model: "claude-sonnet-5"}
	cfg.LLM.Cheap = &config.LLMRole{
		Provider: "openai-compatible", BaseURL: "http://localhost:11434/v1", Model: "qwen3:8b",
	}
	cfg.LLM.Verifier = &config.LLMRole{Model: "claude-haiku-4-5"}
	factory, built := recordingFactory()

	if _, err := buildConfiguredLLM(cfg, nil, factory); err != nil {
		t.Fatal(err)
	}
	if len(*built) != 2 {
		t.Fatalf("built %d providers, want 2", len(*built))
	}
	for _, p := range *built {
		switch p.cfg.BaseURL {
		case "http://localhost:11434/v1":
			if p.cfg.APIKey != "" {
				t.Errorf("the cheap role's endpoint was handed the top-level key")
			}
		case "":
			if p.cfg.APIKey != "sk-ant-api03-top" {
				t.Errorf("the top-level provider lost its key")
			}
		}
	}
}

// TestAnthropicRoleOverDeepSeekNeverGetsTheDeepSeekEndpoint checks each built anthropic backend's URL and key.
func TestAnthropicRoleOverDeepSeekNeverGetsTheDeepSeekEndpoint(t *testing.T) {
	blocks := []config.LLMRole{
		{Provider: "anthropic"},
		{Provider: "anthropic", APIKey: "sk-ant-api03-own"},
		{Provider: "anthropic", Model: "claude-opus-5-5"},
		{Provider: "anthropic", APIKey: "sk-ant-api03-own", Model: "claude-opus-5-5"},
	}
	for _, role := range []string{"strong", "verifier"} {
		for _, b := range blocks {
			cfg := &config.Config{}
			cfg.LLM = config.LLMConfig{
				Provider: "openai-compatible", APIKey: "sk-deepseek-top", BaseURL: "https://api.deepseek.com/v1",
				Model: "deepseek-chat", VerifierModel: "deepseek-chat",
			}
			block := b
			if role == "strong" {
				cfg.LLM.Strong = &block
			} else {
				cfg.LLM.Verifier = &block
			}
			factory, built := recordingFactory()
			_, err := buildConfiguredLLM(cfg, nil, factory)
			if role == "verifier" && b.Model == "" {
				if err == nil || !strings.Contains(err.Error(), "llm.verifier.model") {
					t.Errorf("verifier %+v: err = %v, want one naming llm.verifier.model", b, err)
				}
				continue
			}
			if err != nil {
				t.Fatalf("%s %+v: %v", role, b, err)
			}
			var sawAnthropic bool
			for _, p := range *built {
				if p.cfg.Kind != llm.KindAnthropic {
					continue
				}
				sawAnthropic = true
				if p.cfg.BaseURL != "" || p.cfg.APIKey != b.APIKey {
					t.Errorf("%s %+v: anthropic backend got URL %q, key %q", role, b, p.cfg.BaseURL, p.cfg.APIKey)
				}
				if m := p.cfg.StrongModel + p.cfg.CheapModel; strings.Contains(m, "deepseek") {
					t.Errorf("%s %+v: anthropic backend got model %q", role, b, m)
				}
			}
			if !sawAnthropic {
				t.Errorf("%s %+v: no anthropic backend built", role, b)
			}
		}
	}
}

// TestAnthropicStrongRoleWithoutAModelUsesTheProviderDefault checks the strong default on a provider switch.
func TestAnthropicStrongRoleWithoutAModelUsesTheProviderDefault(t *testing.T) {
	cfg := &config.Config{}
	cfg.LLM = config.LLMConfig{
		Provider: "openai-compatible", APIKey: "sk-deepseek-top", BaseURL: "https://api.deepseek.com/v1",
		Model: "deepseek-chat", VerifierModel: "deepseek-chat",
	}
	cfg.LLM.Strong = &config.LLMRole{Provider: "anthropic", APIKey: "sk-ant-api03-own"}
	p, err := buildConfiguredLLM(cfg, nil, llm.New)
	if err != nil {
		t.Fatal(err)
	}
	if got := p.ModelFor(llm.TierStrong); got != llm.DefaultAnthropicStrong {
		t.Errorf("strong model = %q, want %q", got, llm.DefaultAnthropicStrong)
	}
	if got := p.ModelFor(llm.TierCheap); got != "deepseek-chat" {
		t.Errorf("cheap model = %q, want deepseek-chat", got)
	}
}

// TestAnthropicKeyIsRefusedOnAnOpenAICompatibleBackend checks an sk-ant- key never reaches an openai-compatible host.
func TestAnthropicKeyIsRefusedOnAnOpenAICompatibleBackend(t *testing.T) {
	const secret = "sk-ant-api03-supersecretvalue"
	deepseek := config.LLMConfig{
		Provider: "openai-compatible", BaseURL: "https://api.deepseek.com/v1", Model: "deepseek-chat",
	}
	cases := []struct {
		name    string
		mutate  func(*config.LLMConfig)
		setting string
	}{
		{"top level", func(l *config.LLMConfig) { l.APIKey = secret }, "llm.api-key"},
		{"strong block", func(l *config.LLMConfig) {
			l.APIKey = "sk-deepseek-top"
			l.Strong = &config.LLMRole{Provider: "openai-compatible", APIKey: secret}
		}, "llm.strong.api-key"},
		{"cheap block on another provider", func(l *config.LLMConfig) {
			l.Provider, l.BaseURL, l.Model = "anthropic", "", "claude-opus-5-5"
			l.Cheap = &config.LLMRole{Provider: "openai-compatible", BaseURL: "https://api.deepseek.com/v1",
				APIKey: secret, Model: "deepseek-chat"}
		}, "llm.cheap.api-key"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cfg := &config.Config{}
			cfg.LLM = deepseek
			c.mutate(&cfg.LLM)
			factory, built := recordingFactory()
			_, err := buildConfiguredLLM(cfg, nil, factory)
			if err == nil {
				t.Fatal("an Anthropic key was accepted for an openai-compatible backend")
			}
			if !strings.Contains(err.Error(), c.setting) {
				t.Errorf("error does not name %s: %v", c.setting, err)
			}
			if strings.Contains(err.Error(), secret) || strings.Contains(err.Error(), "supersecret") {
				t.Errorf("error leaks the key: %v", err)
			}
			if len(*built) != 0 {
				t.Errorf("built %d backends, want none", len(*built))
			}
		})
	}
}

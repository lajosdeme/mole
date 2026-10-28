package main

import (
	"bytes"
	"context"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"github.com/lajosdeme/mole/internal/budget"
	"github.com/lajosdeme/mole/internal/config"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/pricing"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/store/sqlite"
)

// seedReportSession creates a report-mode session at path, with one claim
// when withClaim is true, and returns its id and its session row.
func seedReportSession(t *testing.T, path string, withClaim bool) (string, *core.Session) {
	t.Helper()
	ctx := context.Background()

	db, err := sqlite.Open(path, sqlite.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if err := db.Migrate(ctx); err != nil {
		t.Fatal(err)
	}

	sess, err := budget.New(db, budget.DefaultConfig()).CreateSession(ctx, budget.SessionSpec{
		Prompt: "what is the capital of Freedonia?", Mode: core.ModeReport,
		ActorTypes: []core.ActorType{core.ActorWeb},
		BudgetUnit: core.BudgetUSD, Budget: 3 * core.MicrosPerUSD,
	})
	if err != nil {
		t.Fatal(err)
	}

	if withClaim {
		lead := core.Lead{ID: core.NewLeadID(), SessionID: sess.ID, ActorType: core.ActorWeb,
			Query: "q", Status: core.LeadDone}
		claims := []core.Claim{{
			Text: "Freedonia's capital is Fredonia City.", Source: "https://example.com/a",
			Quote: "the capital is Fredonia City", Confidence: 0.9,
		}}
		if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
			if err := tx.InsertLead(ctx, &lead); err != nil {
				return err
			}
			for i := range claims {
				claims[i].SessionID = sess.ID
				claims[i].LeadID = lead.ID
			}
			return tx.InsertClaims(ctx, claims)
		}); err != nil {
			t.Fatal(err)
		}
	}

	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	return sess.ID, sess
}

// reportFakeProvider records the requests it receives and replies with text.
type reportFakeProvider struct {
	text  string
	model string
	got   []llm.Request
}

func (f *reportFakeProvider) Complete(_ context.Context, req llm.Request) (*llm.Response, error) {
	f.got = append(f.got, req)
	return &llm.Response{
		Text: f.text, Model: f.ModelFor(req.Tier),
		Usage: llm.Usage{InputTokens: 500, OutputTokens: 50},
	}, nil
}
func (f *reportFakeProvider) Name() string               { return "report-fake" }
func (f *reportFakeProvider) ModelFor(_ llm.Tier) string { return f.model }

// reportFakeFactory returns a provider factory that records the llm.Config
// it was built with.
func reportFakeFactory(text string) (func(llm.Config, *http.Client) (llm.Provider, error), *reportFakeProvider) {
	p := &reportFakeProvider{text: text}
	return func(c llm.Config, _ *http.Client) (llm.Provider, error) {
		p.model = c.StrongModel
		return p, nil
	}, p
}

func reportTestConfig() *config.Config {
	cfg := &config.Config{}
	cfg.LLM = config.LLMConfig{
		Provider: "anthropic", APIKey: "sk-ant-api03-x",
		Model: "claude-opus-5", CheapModel: "claude-haiku-4-5",
	}
	return cfg
}

// TestReportRewritesWithTheGivenModel: --model overrides the role's model,
// and the session's stored report and ledger stay untouched.
func TestReportRewritesWithTheGivenModel(t *testing.T) {
	path := filepath.Join(t.TempDir(), "report.db")
	sid, baseline := seedReportSession(t, path, true)

	factory, fake := reportFakeFactory("REWRITTEN: the capital is Fredonia City [1].")
	o := reportOpts{role: "strong", model: "claude-sonnet-5-5", dbPath: path}

	var out, errBuf bytes.Buffer
	if err := cmdReport(context.Background(), sid, o, reportTestConfig(), factory, &out, &errBuf); err != nil {
		t.Fatalf("cmdReport: %v", err)
	}

	if len(fake.got) != 1 {
		t.Fatalf("%d model calls, want 1", len(fake.got))
	}
	if fake.model != "claude-sonnet-5-5" {
		t.Errorf("provider was built with model %q, want the --model override", fake.model)
	}
	if !strings.Contains(out.String(), "REWRITTEN: the capital is Fredonia City") {
		t.Errorf("output does not contain the fake's text:\n%s", out.String())
	}
	if !strings.Contains(errBuf.String(), "claude-sonnet-5-5") {
		t.Errorf("stderr does not name the model used:\n%s", errBuf.String())
	}

	db, err := openDBNoMigrate(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var sess *core.Session
	if err := db.Read(context.Background(), func(ctx context.Context, q store.Queries) error {
		var err error
		sess, err = q.GetSession(ctx, sid)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if sess.Report != "" {
		t.Errorf("session's stored report was written: %q", sess.Report)
	}
	if sess.Spent != baseline.Spent || sess.Escrow != baseline.Escrow || sess.Held != baseline.Held {
		t.Errorf("ledger changed: spent=%d escrow=%d held=%d, want spent=%d escrow=%d held=%d",
			sess.Spent, sess.Escrow, sess.Held, baseline.Spent, baseline.Escrow, baseline.Held)
	}
}

// TestReportOnEmptySessionMakesNoCall: a session with no surviving claims gets
// the fallback text and never reaches the model.
func TestReportOnEmptySessionMakesNoCall(t *testing.T) {
	path := filepath.Join(t.TempDir(), "report.db")
	sid, _ := seedReportSession(t, path, false)

	factory, fake := reportFakeFactory("should never be seen")
	o := reportOpts{role: "strong", dbPath: path}

	var out, errBuf bytes.Buffer
	if err := cmdReport(context.Background(), sid, o, reportTestConfig(), factory, &out, &errBuf); err != nil {
		t.Fatalf("cmdReport: %v", err)
	}

	if len(fake.got) != 0 {
		t.Errorf("%d model calls on an empty session, want 0", len(fake.got))
	}
	if !strings.Contains(out.String(), "No verifiable evidence was found") {
		t.Errorf("output does not carry the fallback text:\n%s", out.String())
	}
}

// TestReportIsInTheVisibleCommandSet, beside research and trace, not hidden
// with the evaluation tooling.
func TestReportIsInTheVisibleCommandSet(t *testing.T) {
	out, err := exec(t, "--help")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "report") {
		t.Errorf("report is missing from the help screen:\n%s", out)
	}
}

// TestReportRefusesAnUnknownRole before building anything.
func TestReportRefusesAnUnknownRole(t *testing.T) {
	path := filepath.Join(t.TempDir(), "report.db")
	sid, _ := seedReportSession(t, path, true)
	factory, fake := reportFakeFactory("x")
	o := reportOpts{role: "miner", dbPath: path}

	var out, errBuf bytes.Buffer
	err := cmdReport(context.Background(), sid, o, reportTestConfig(), factory, &out, &errBuf)
	if err == nil || !strings.Contains(err.Error(), "strong, cheap or verifier") {
		t.Errorf("err = %v, want a refusal naming the valid roles", err)
	}
	if len(fake.got) != 0 {
		t.Error("a call was made despite the unknown role")
	}
}

// TestReportSendsMediumEffortByDefault checks the rewrite asks for the generator's default effort.
func TestReportSendsMediumEffortByDefault(t *testing.T) {
	path := filepath.Join(t.TempDir(), "report.db")
	sid, _ := seedReportSession(t, path, true)
	factory, fake := reportFakeFactory("A finding [1].")

	var out, errBuf bytes.Buffer
	o := reportOpts{role: "strong", dbPath: path}
	if err := cmdReport(context.Background(), sid, o, reportTestConfig(), factory, &out, &errBuf); err != nil {
		t.Fatal(err)
	}
	if len(fake.got) != 1 || fake.got[0].Effort != "medium" {
		t.Errorf("requests = %+v, want one at effort medium", fake.got)
	}
}

// TestReportSendsTheEffortFlag checks --effort reaches the request.
func TestReportSendsTheEffortFlag(t *testing.T) {
	path := filepath.Join(t.TempDir(), "report.db")
	sid, _ := seedReportSession(t, path, true)
	factory, fake := reportFakeFactory("A finding [1].")

	var out, errBuf bytes.Buffer
	o := reportOpts{role: "strong", effort: "xhigh", dbPath: path}
	if err := cmdReport(context.Background(), sid, o, reportTestConfig(), factory, &out, &errBuf); err != nil {
		t.Fatal(err)
	}
	if len(fake.got) != 1 || fake.got[0].Effort != "xhigh" {
		t.Errorf("requests = %+v, want one at effort xhigh", fake.got)
	}
}

// TestReportRefusesAnUnknownEffort checks a bad --effort fails before any call.
func TestReportRefusesAnUnknownEffort(t *testing.T) {
	path := filepath.Join(t.TempDir(), "report.db")
	sid, _ := seedReportSession(t, path, true)
	factory, fake := reportFakeFactory("x")

	var out, errBuf bytes.Buffer
	o := reportOpts{role: "strong", effort: "extreme", dbPath: path}
	err := cmdReport(context.Background(), sid, o, reportTestConfig(), factory, &out, &errBuf)
	if err == nil || !strings.Contains(err.Error(), "low, medium, high, xhigh or max") {
		t.Errorf("err = %v, want a refusal naming the valid efforts", err)
	}
	if len(fake.got) != 0 {
		t.Error("a call was made despite the unknown effort")
	}
}

// tieredFake serves a default model per tier when built without one.
type tieredFake struct{ cfg llm.Config }

func (f *tieredFake) Complete(_ context.Context, req llm.Request) (*llm.Response, error) {
	return &llm.Response{Text: "ok", Model: f.ModelFor(req.Tier)}, nil
}
func (f *tieredFake) Name() string { return "tiered-fake" }
func (f *tieredFake) ModelFor(tier llm.Tier) string {
	if tier == llm.TierCheap {
		return firstNonEmpty(f.cfg.CheapModel, "default-cheap")
	}
	return firstNonEmpty(f.cfg.StrongModel, "default-strong")
}

func tieredFactory(c llm.Config, _ *http.Client) (llm.Provider, error) {
	return &tieredFake{cfg: c}, nil
}

// TestReportLLMFallsBackToTheProviderDefault checks each role's model when none is configured.
func TestReportLLMFallsBackToTheProviderDefault(t *testing.T) {
	cfg := &config.Config{}
	cfg.LLM = config.LLMConfig{Provider: "anthropic", APIKey: "sk-ant-api03-x"}
	for role, want := range map[llm.ModelRole]string{
		llm.RoleStrong: "default-strong", llm.RoleCheap: "default-cheap", llm.RoleVerifier: "default-cheap",
	} {
		p, model, err := buildReportLLM(cfg, nil, role, "", tieredFactory)
		if err != nil {
			t.Fatalf("%s: %v", role, err)
		}
		if model != want || p.ModelFor(llm.TierStrong) != want {
			t.Errorf("%s: model %q, strong tier %q, want %q", role, model, p.ModelFor(llm.TierStrong), want)
		}
	}
}

// TestReportLLMVerifierFollowsTheCheapModel checks a verifier with no model judges with the cheap one.
func TestReportLLMVerifierFollowsTheCheapModel(t *testing.T) {
	cfg := reportTestConfig()
	_, model, err := buildReportLLM(cfg, nil, llm.RoleVerifier, "", tieredFactory)
	if err != nil {
		t.Fatal(err)
	}
	if model != "claude-haiku-4-5" {
		t.Errorf("verifier model = %q, want the cheap claude-haiku-4-5", model)
	}
}

// TestReportLLMErrorNamesTheRealKey checks a missing model error names keys that exist.
func TestReportLLMErrorNamesTheRealKey(t *testing.T) {
	cfg := &config.Config{}
	cfg.LLM = config.LLMConfig{Provider: "openai-compatible", BaseURL: "http://localhost:11434/v1"}
	for role, want := range map[llm.ModelRole][]string{
		llm.RoleStrong:   {"llm.model", "llm.strong.model"},
		llm.RoleCheap:    {"llm.cheap-model", "llm.cheap.model"},
		llm.RoleVerifier: {"llm.verifier-model", "llm.verifier.model"},
	} {
		_, _, err := buildReportLLM(cfg, nil, role, "", llm.New)
		if err == nil {
			t.Fatalf("%s: no error for a role with no model", role)
		}
		for _, key := range want {
			if !strings.Contains(err.Error(), key) {
				t.Errorf("%s: error lacks %q: %v", role, key, err)
			}
		}
		if strings.Contains(err.Error(), "llm.strong-model") {
			t.Errorf("%s: error names the nonexistent llm.strong-model: %v", role, err)
		}
	}
}

// TestReportCostSaysUnpricedForAnUnknownModel checks no fallback rate prices an unknown model.
func TestReportCostSaysUnpricedForAnUnknownModel(t *testing.T) {
	var buf bytes.Buffer
	printReportCost(&buf, reportTestConfig(), "mystery-model", core.Cost{InputTokens: 1000, OutputTokens: 100})
	if !strings.Contains(buf.String(), "unpriced") {
		t.Errorf("an unknown model was priced:\n%s", buf.String())
	}
}

// TestReportCostUsesDeclaredRates checks a declared rate prices its model.
func TestReportCostUsesDeclaredRates(t *testing.T) {
	cfg := reportTestConfig()
	cfg.Pricing = map[string]pricing.RateConfig{"mystery-model": {Input: 1, Output: 1}}
	var buf bytes.Buffer
	printReportCost(&buf, cfg, "mystery-model", core.Cost{InputTokens: 1_000_000, OutputTokens: 0})
	if !strings.Contains(buf.String(), "$1.00") {
		t.Errorf("declared rate not used:\n%s", buf.String())
	}
}

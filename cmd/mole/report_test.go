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

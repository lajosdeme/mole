package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/lajosdeme/mole/internal/config"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/output"
	"github.com/lajosdeme/mole/internal/pricing"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/spf13/cobra"
)

// mole report
//
// Rewrites a stored session's report with any model. Read-only: it writes
// nothing back to the session or the ledger.

const reportUsage = `Rewrite a stored session's report with any model.

Builds the chosen role's provider (default strong, the role that writes a
report in ` + "`mole research`" + `) and reruns synthesis over the session's
already-verified claims. --model overrides that role's model for this run
only.

Nothing is written back: no report row, no ledger row. This is for comparing
report-writer models on identical evidence, not for replacing the stored
answer — see ` + "`mole export`" + ` for the one on record.
`

type reportOpts struct {
	role        string
	model       string
	maxFindings int
	out         string
	dbPath      string
}

func newReportCmd() *cobra.Command {
	var o reportOpts

	c := &cobra.Command{
		Use:   "report <session-id>",
		Short: "Rewrite a stored session's report with any model",
		Long:  reportUsage,
		Args:  exactArgs(1, "mole report <session-id>"),
		RunE: func(cmd *cobra.Command, args []string) error {
			o.dbPath = dbPath(cmd)
			cfg, err := config.Load()
			if err != nil && !errors.Is(err, config.ErrNotConfigured) {
				return err
			}
			return cmdReport(cmd.Context(), args[0], o, cfg, llm.New, cmd.OutOrStdout(), cmd.ErrOrStderr())
		},
	}

	f := c.Flags()
	f.StringVar(&o.role, "role", string(llm.RoleStrong),
		"role whose provider writes the report: strong, cheap or verifier")
	f.StringVar(&o.model, "model", "", "override the role's model for this run")
	f.IntVar(&o.maxFindings, "max-findings", 0,
		"findings the report is written from (0 for the session's own default)")
	f.StringVarP(&o.out, "out", "o", "", "write here instead of stdout")
	return c
}

func cmdReport(ctx context.Context, sessionID string, o reportOpts, cfg *config.Config,
	newProvider func(llm.Config, *http.Client) (llm.Provider, error), stdout, stderr io.Writer) error {
	if cfg == nil {
		cfg = &config.Config{}
	}
	role := llm.ModelRole(o.role)
	provider, model, err := buildReportLLM(cfg, nil, role, o.model, newProvider)
	if err != nil {
		return err
	}

	db, err := openDBNoMigrate(ctx, o.dbPath)
	if err != nil {
		return err
	}
	defer db.Close()

	gen := &output.Generator{LLM: provider, MaxClaims: o.maxFindings}
	rep, err := gen.Generate(ctx, db, sessionID)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return fmt.Errorf("no session %s", sessionID)
		}
		return err
	}

	md := rep.Markdown()
	if o.out == "" {
		fmt.Fprintln(stdout, md)
	} else {
		// 0600: a session's claims carry whatever the question was about.
		if err := os.WriteFile(o.out, []byte(md+"\n"), 0o600); err != nil {
			return err
		}
	}

	usedModel := rep.Model
	if usedModel == "" {
		usedModel = model
	}
	printReportCost(stderr, cfg, usedModel, rep.Cost)
	return nil
}

// printReportCost reports what the run would have cost, priced from the
// table the CLI already builds for the configured roles.
func printReportCost(w io.Writer, cfg *config.Config, model string, cost core.Cost) {
	c, err := pricingFor(cfg).Cost(model, pricing.Usage{
		InputTokens: cost.InputTokens, OutputTokens: cost.OutputTokens,
		CacheReadTokens: cost.CacheReadTokens, CacheWriteTokens: cost.CacheWriteTokens,
	})
	usd := "unpriced"
	if err == nil {
		usd = core.FormatUSD(c.USDMicros)
	}
	fmt.Fprintf(w, "model: %s · %d in / %d out tokens · %s\n",
		model, cost.InputTokens, cost.OutputTokens, usd)
}

// buildReportLLM builds the provider for role, with its strong model set to
// model when given.
func buildReportLLM(cfg *config.Config, client *http.Client, role llm.ModelRole, modelOverride string,
	newProvider func(llm.Config, *http.Client) (llm.Provider, error)) (llm.Provider, string, error) {
	if !validReportRole(role) {
		return nil, "", fmt.Errorf("--role %q: want strong, cheap or verifier", role)
	}

	roles, err := resolveLLMRoles(cfg)
	if err != nil {
		return nil, "", err
	}
	var picked *llmRole
	for i := range roles {
		if roles[i].role == role {
			picked = &roles[i]
			break
		}
	}
	if picked == nil {
		return nil, "", fmt.Errorf("--role %q: want strong, cheap or verifier", role)
	}

	model := picked.model
	if modelOverride != "" {
		model = modelOverride
	}
	if model == "" {
		return nil, "", fmt.Errorf(
			"no model configured for the %s role (set llm.%s-model, or pass --model)", role, role)
	}

	c := llm.Config{
		Kind: picked.backend.kind, APIKey: picked.backend.key, BaseURL: picked.backend.baseURL,
		Timeout: cfg.LLM.Timeout, StrongModel: model, CheapModel: model,
	}
	p, err := newProvider(c, client)
	if err != nil {
		return nil, "", err
	}
	return p, model, nil
}

func validReportRole(r llm.ModelRole) bool {
	for _, want := range llm.Roles {
		if r == want {
			return true
		}
	}
	return false
}

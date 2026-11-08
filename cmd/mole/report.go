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
// Rewrites a stored session's report with any model. With --save the rewrite
// replaces the session's stored report; nothing is added to the ledger.

const reportUsage = `Rewrite a stored session's report with any model.

Builds the chosen role's provider (default strong, the role that writes a
report in ` + "`mole research`" + `) and reruns synthesis over the session's
already-verified claims. --model overrides that role's model for this run
only.

Without --save nothing is written back: this compares report-writer models
on identical evidence. --save replaces the session's stored report and
citations with the rewrite, so ` + "`mole export`" + ` carries it; the
rewrite's cost is not added to the ledger.

--print-prompt prints the synthesis prompt and calls no model. --body-file
takes a response written elsewhere (--body-model names its model) and puts
it through the same citation checks as a model's; a body that fails them is
an error.
`

type reportOpts struct {
	role        string
	model       string
	effort      string
	maxFindings int
	out         string
	dbPath      string
	save        bool
	printPrompt bool
	bodyFile    string
	bodyModel   string
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
	f.StringVar(&o.effort, "effort", "", "synthesis effort: low, medium, high, xhigh or max (default medium)")
	f.IntVar(&o.maxFindings, "max-findings", 0,
		"findings the report is written from (0 for the session's own default)")
	f.StringVarP(&o.out, "out", "o", "", "write here instead of stdout")
	f.BoolVar(&o.save, "save", false, "replace the session's stored report with the rewrite")
	f.BoolVar(&o.printPrompt, "print-prompt", false, "print the synthesis prompt instead of calling a model")
	f.StringVar(&o.bodyFile, "body-file", "", "use this file as the model's response instead of calling a model")
	f.StringVar(&o.bodyModel, "body-model", "", "the model that wrote --body-file (required with it)")
	return c
}

func cmdReport(ctx context.Context, sessionID string, o reportOpts, cfg *config.Config,
	newProvider func(llm.Config, *http.Client) (llm.Provider, error), stdout, stderr io.Writer) error {
	if cfg == nil {
		cfg = &config.Config{}
	}
	if !validEffort(o.effort) {
		return fmt.Errorf("--effort %q: want low, medium, high, xhigh or max", o.effort)
	}
	role := llm.ModelRole(o.role)
	var (
		provider llm.Provider
		model    string
		captured = &fixedProvider{}
	)
	switch {
	case o.printPrompt:
		provider = captured
	case o.bodyFile != "":
		if o.bodyModel == "" {
			return errors.New("--body-file needs --body-model naming the model that wrote it")
		}
		body, err := os.ReadFile(o.bodyFile)
		if err != nil {
			return err
		}
		captured.text, captured.model = string(body), o.bodyModel
		provider, model = captured, o.bodyModel
	default:
		var err error
		if provider, model, err = buildReportLLM(cfg, nil, role, o.model, newProvider); err != nil {
			return err
		}
	}

	db, err := openDBNoMigrate(ctx, o.dbPath)
	if err != nil {
		return err
	}
	defer db.Close()

	gen := &output.Generator{LLM: provider, MaxClaims: o.maxFindings, Effort: o.effort,
		RecheckModel: cfg.LLM.RecheckModel}
	rep, err := gen.Generate(ctx, db, sessionID)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return fmt.Errorf("no session %s", sessionID)
		}
		return err
	}

	if o.printPrompt {
		if captured.req == nil {
			return errors.New("the session has no claims to write a report from")
		}
		fmt.Fprintln(stdout, captured.req.System)
		for _, m := range captured.req.Messages {
			fmt.Fprintln(stdout)
			fmt.Fprintln(stdout, m.Text)
		}
		return nil
	}
	if o.bodyFile != "" && rep.DegradedDetail != "" {
		return fmt.Errorf("%s rejected: %s", o.bodyFile, rep.DegradedDetail)
	}

	if o.save {
		if err := output.Save(ctx, db, rep); err != nil {
			return err
		}
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

// printReportCost reports what the run would have cost from the built-in and declared
// rates, or "unpriced" for a model neither knows.
func printReportCost(w io.Writer, cfg *config.Config, model string, cost core.Cost) {
	tbl := pricing.NewTable()
	_ = tbl.Declare(cfg.Pricing)
	c, err := tbl.Cost(model, pricing.Usage{
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

// buildReportLLM builds role's provider on one model: modelOverride, else the role's
// model, else the provider's default for the role's tier.
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

	model := firstNonEmpty(modelOverride, picked.model)
	if model == "" && role == llm.RoleVerifier && picked.backend == roles[1].backend {
		model = roles[1].model
	}
	c := llm.Config{
		Kind: picked.backend.kind, APIKey: picked.backend.key, BaseURL: picked.backend.baseURL,
		Timeout: cfg.LLM.Timeout, StrongModel: model, CheapModel: model,
	}
	noModel := func(err error) error {
		msg := fmt.Sprintf("no model configured for the %s role (set %s, or pass --model)", role, roleModelKeys(role))
		if err != nil {
			return fmt.Errorf("%s: %w", msg, err)
		}
		return errors.New(msg)
	}
	p, err := newProvider(c, client)
	if err != nil {
		if model == "" {
			return nil, "", noModel(err)
		}
		return nil, "", err
	}
	if model != "" {
		return p, model, nil
	}

	// The provider's default for the role's tier serves every call of this run.
	tier := llm.TierCheap
	if role == llm.RoleStrong {
		tier = llm.TierStrong
	}
	if model = p.ModelFor(tier); model == "" {
		return nil, "", noModel(nil)
	}
	c.StrongModel, c.CheapModel = model, model
	if p, err = newProvider(c, client); err != nil {
		return nil, "", err
	}
	return p, model, nil
}

// roleModelKeys names the settings that give role a model.
func roleModelKeys(role llm.ModelRole) string {
	switch role {
	case llm.RoleStrong:
		return "llm.model or llm.strong.model"
	case llm.RoleCheap:
		return "llm.cheap-model or llm.cheap.model"
	}
	return "llm.verifier-model or llm.verifier.model"
}

func validReportRole(r llm.ModelRole) bool {
	for _, want := range llm.Roles {
		if r == want {
			return true
		}
	}
	return false
}

// validEffort reports whether e is empty or an effort level the API accepts.
func validEffort(e string) bool {
	switch e {
	case "", "low", "medium", "high", "xhigh", "max":
		return true
	}
	return false
}

// fixedProvider records the request it receives and answers with text.
type fixedProvider struct {
	text, model string
	req         *llm.Request
}

func (p *fixedProvider) Complete(_ context.Context, req llm.Request) (*llm.Response, error) {
	p.req = &req
	return &llm.Response{Text: p.text, Model: p.model}, nil
}
func (p *fixedProvider) Name() string               { return "fixed" }
func (p *fixedProvider) ModelFor(_ llm.Tier) string { return p.model }

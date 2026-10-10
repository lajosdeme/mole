package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/lajosdeme/mole/internal/actors"
	"github.com/lajosdeme/mole/internal/compute/coderunner"
	"github.com/lajosdeme/mole/internal/compute/connector"
	"github.com/lajosdeme/mole/internal/compute/gate"
	"github.com/lajosdeme/mole/internal/compute/sandbox"
	"github.com/lajosdeme/mole/internal/config"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/dataset"
	"github.com/lajosdeme/mole/internal/executor"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/output"
	"github.com/lajosdeme/mole/internal/planner"
	"github.com/lajosdeme/mole/internal/pricing"
	"github.com/lajosdeme/mole/internal/record"
	"github.com/lajosdeme/mole/internal/session"
	"github.com/lajosdeme/mole/internal/tools/academic"
	"github.com/lajosdeme/mole/internal/tools/extract"
	"github.com/lajosdeme/mole/internal/tools/fetch"
	"github.com/lajosdeme/mole/internal/tools/limiter"
	"github.com/lajosdeme/mole/internal/tools/search"
	"github.com/lajosdeme/mole/internal/verifier"
	"github.com/spf13/cobra"
)

// mole research
//
// The planner loop (§9.2) driven from a terminal: decompose the question, work a
// queue of leads, replan against a rolling digest as evidence arrives, and
// synthesize a cited report from escrow held back at session start.
//
// It runs in-process rather than attaching to the daemon: there is no protocol
// for handing a running session to a terminal and streaming its progress back,
// so Ctrl-C settles what was spent and stops rather than detaching.

// researchUsage is the prose cobra cannot generate. The flag list is
// deliberately absent: it is derived from the definitions below, so it cannot
// drift out of step with them the way the hand-written one did.
const researchUsage = `Run one research question end to end.

Budget is required and the two units are mutually exclusive — pass --usd N or
--tokens N, or set a default with ` + "`mole config set default.usd`" + `. There is
deliberately no bare --budget: a number alone is ambiguous between the two, and
§8 makes the unit load-bearing. Only USD mode can price a search call, and only
token mode can bound a model the pricing table does not know.

The question is decomposed into sub-questions, each researched as its own lead,
and the planner replans against a rolling digest as evidence arrives. The report
is paid from escrow held back at session start, so a run that spends everything
it is allowed can still afford to write up what it found.

--mode is "report" (prose with citations) or "dataset" (§13: a table, one row per
entity, merged across sources — needs --schema or --schema-file). Chain mode
arrives with its milestone.
`

// researchOpts is what the flags resolve to.
type researchOpts struct {
	usd          string
	tokens       int64
	mode         string
	maxSources   int
	maxToolCalls int64
	timeout      time.Duration
	asJSON       bool
	quiet        bool
	alwaysFetch  bool
	noConfirm    bool
	confirmMode  string
	maxDepth     int
	workers      int
	actorList    string
	schemaSpec   string
	schemaFile   string
	dbPath       string

	// silent suppresses ALL output, including the report.
	//
	// Distinct from quiet, which suppresses progress and still prints the answer — that
	// is what a person running one question wants. The corpus runner produces its own
	// output and a hundred inlined reports would bury it. No flag: nothing on the
	// command line should ask mole to research a question and say nothing.
	silent bool

	// onSession, when set, is called with the session id as soon as it exists.
	//
	// A callback rather than a return value because the corpus runner needs the id even
	// when the run goes on to fail — a session that errored still has a ledger to
	// reconcile and claims to score, and those are exactly the runs worth looking at.
	// Nil for the CLI.
	onSession func(string)
}

func newResearchCmd() *cobra.Command {
	var o researchOpts

	c := &cobra.Command{
		Use:   "research <question>",
		Short: "Run one research question end to end",
		Long:  researchUsage,
		Args:  minArgs(1, `mole research "<question>" (--usd N | --tokens N)`),
		RunE: func(cmd *cobra.Command, args []string) error {
			o.dbPath = dbPath(cmd)
			return cmdResearch(cmd.Context(), strings.Join(args, " "), o)
		},
	}

	f := c.Flags()
	f.StringVar(&o.usd, "usd", "", "budget in dollars, e.g. --usd 3.00")
	f.Int64Var(&o.tokens, "tokens", 0, "budget in tokens")
	f.StringVar(&o.mode, "mode", string(core.ModeReport), "session mode: report or dataset")
	f.StringVar(&o.schemaSpec, "schema", "",
		"dataset columns, e.g. company:text!,revenue:number "+
			"(! marks a key field; =text after the type adds a description)")
	f.StringVar(&o.schemaFile, "schema-file", "", "dataset schema as a JSON file")
	f.IntVar(&o.maxSources, "max-sources", 5, "sources to read per lead")
	f.Int64Var(&o.maxToolCalls, "max-tool-calls", 0,
		"tool calls for the whole session (0 sizes a backstop from --max-depth)")
	f.StringVar(&o.actorList, "actors", "web",
		"comma-separated actors: web, academic, local_compute "+
			"(academic needs contact-email; local_compute needs `mole connect add`)")
	f.IntVar(&o.workers, "workers", executor.DefaultWorkers,
		"leads to run at once; 1 is required when recording or replaying a cassette")
	f.DurationVar(&o.timeout, "timeout", 5*time.Minute, "wall-clock ceiling for the whole session")
	// Empty default so "was it passed" is answerable. The mode's real default
	// is both, applied in resolveConfirmMode, and stated in the usage below —
	// with the flag defaulting to "both" there is no way to tell an explicit
	// --confirm-edges=both from no flag at all, and the deprecated alias needs
	// exactly that distinction.
	f.StringVar(&o.confirmMode, "confirm-edges", "",
		"edge kinds needing a second agreeing judgement: both (default), duplicates, none. "+
			"`both` buys 70% contradiction precision against 51%, at 67% recall "+
			"against 98%; duplicates keeps every contradiction found and costs "+
			"no recall")
	f.BoolVar(&o.noConfirm, "no-confirm-edges", false,
		"deprecated alias for --confirm-edges=none")
	_ = f.MarkDeprecated("no-confirm-edges", "use --confirm-edges=none")
	f.BoolVar(&o.asJSON, "json", false, "emit the result as JSON")
	f.BoolVar(&o.quiet, "quiet", false, "suppress progress; print only the result")
	f.IntVar(&o.maxDepth, "max-depth", planner.DefaultMaxDepth,
		"rounds of follow-up leads the planner may add (0 for none)")
	f.BoolVar(&o.alwaysFetch, "always-fetch", false,
		"fetch every page even when the search provider supplied its text (slower; required for citation accuracy and the §17.1 gate)")
	return c
}

func cmdResearch(ctx context.Context, rawQuestion string, o researchOpts) error {
	question := strings.TrimSpace(rawQuestion)
	if question == "" {
		return errors.New("no question given")
	}

	cfg, err := config.Load()
	if err != nil && !errors.Is(err, config.ErrNotConfigured) {
		return err
	}

	unit, amount, err := resolveBudget(o.usd, o.tokens, cfg)
	if err != nil {
		return err
	}
	sessionMode := core.Mode(o.mode)
	if !sessionMode.Valid() {
		return fmt.Errorf("unknown mode %q", o.mode)
	}
	switch sessionMode {
	case core.ModeReport, core.ModeDataset:
	default:
		return fmt.Errorf("mode %q is not implemented yet (report and dataset are)", o.mode)
	}

	schema, err := resolveSchema(sessionMode, o.schemaSpec, o.schemaFile)
	if err != nil {
		return err
	}

	actorTypes, err := parseActorTypes(o.actorList)
	if err != nil {
		return err
	}
	// Checked here, before anything is built. A missing contact address and a
	// missing search key are both configuration errors, and which one a user is
	// told about should not depend on the order the actors happen to be
	// constructed in — this one is free to check, so it goes first.
	if err := checkAcademicConfig(cfg, actorTypes); err != nil {
		return err
	}

	// Before the cassette is opened, not after: in replay, opening fails on a
	// missing recording, and a caller who asked for something never allowed
	// should be told that rather than that the file is absent.
	if err := checkCassetteIsSerial(o.workers); err != nil {
		return err
	}

	// Before anything is created or spent: a misspelt mode falling back to the
	// default would change the graph without saying so.
	confirmMode, err := resolveConfirmMode(o.confirmMode, o.noConfirm)
	if err != nil {
		return err
	}

	// One cassette per question (§14.1). Off unless MOLE_RECORD says otherwise,
	// so this costs nothing in normal use.
	rec, err := record.FromEnv(question)
	if err != nil {
		return err
	}
	defer func() {
		if err := rec.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "warning: could not save cassette: %v\n", err)
		}
	}()

	// Build the actor before touching the database. A missing search key should
	// fail in under a second, not after creating a session that can never run.
	// A local-only session must not require a web search provider: the point of
	// §12 is data that does not leave the machine, and demanding a Brave or
	// Tavily key to analyse a CSV is the opposite of that. The actor is still
	// built — it carries the model, the pricing table and the logger the other
	// actors borrow — just without a provider it will never use.
	actor, err := buildWebActor(cfg, rec, o.maxSources, o.alwaysFetch, o.quiet,
		slices.Contains(actorTypes, core.ActorWeb))
	if err != nil {
		return err
	}
	if unit == core.BudgetUSD {
		if err := checkUSDIsEnforceable(actor.LLM); err != nil {
			return err
		}
	}
	if rec.Enabled() && !o.quiet && !o.asJSON {
		fmt.Printf("cassette %s (%s)\n", rec.Path, rec.Mode)
	}

	db, err := openDBMigrate(ctx, o.dbPath)
	if err != nil {
		return err
	}
	defer db.Close()

	// The context gets headroom over the wall-clock CEILING so the two do not
	// fire together. When they do, the loop is killed mid-lead instead of
	// stopping at its own check, and the run ends in a cascade of "context
	// deadline exceeded" from whatever was in flight — a persist, a settle, a
	// count — rather than a clean "stopped: max_wallclock".
	ctx, cancel := context.WithTimeout(ctx, o.timeout+30*time.Second)
	defer cancel()

	academicActor, err := buildAcademicActor(cfg, actorTypes, actor)
	if err != nil {
		return err
	}
	localActor, err := buildLocalActor(ctx, actorTypes, o, actor)
	if err != nil {
		return err
	}

	runner := &session.Runner{
		Store:             db,
		Actor:             actor,
		Academic:          academicActor,
		Local:             localActor,
		VerifierModel:     cfg.LLM.VerifierModel,
		VerifierBatchSize: cfg.LLM.VerifierBatchSize,
		Confirm:           confirmMode,
		Owner:             "cli",
		Notice:            func(msg string) { fmt.Fprintf(os.Stderr, "warning: %s\n", msg) },
	}
	if !o.quiet && !o.asJSON {
		runner.Progress = progressPrinter()
		// Order matters and is the callbacks' whole purpose: the run summary, then
		// grounding, then the report. Reading them off Result at the end printed
		// grounding first.
		runner.OnLoop = printProgress
		runner.OnGround = func(rep *verifier.GroundReport) { reportGrounding(rep, o) }
	}

	spec := session.Spec{
		Question:     question,
		Mode:         sessionMode,
		Schema:       schema,
		BudgetUnit:   unit,
		Budget:       amount,
		MaxSources:   o.maxSources,
		MaxDepth:     o.maxDepth,
		MaxLeads:     maxLeadsFor(o),
		MaxToolCalls: o.maxToolCalls,
		Timeout:      o.timeout,
		Workers:      o.workers,
		ActorTypes:   actorTypes,
	}

	sess, err := runner.Create(ctx, spec)
	if o.onSession != nil && sess != nil {
		o.onSession(sess.ID)
	}
	if err != nil {
		return err
	}

	out := &researchOutput{Question: question, SessionID: sess.ID, Unit: string(unit), Budget: amount}

	if !o.quiet && !o.asJSON {
		fmt.Printf("session  %s   mode=%s  budget=%s (escrow %s held, §8.3)\n\n",
			sess.ID, sessionMode, fmtAmount(unit, sess.Budget), fmtAmount(unit, sess.Escrow))
		if rec.Enabled() {
			fmt.Printf("cassette %s (%s)\n\n", rec.Path, rec.Mode)
		}
		fmt.Println(" executing ─────────────────────────────────────────")
	}

	// Recover what a previous crash left behind (§9.4), before the loop starts.
	if rc := runner.Recover(ctx); rc.Any() && !o.quiet {
		if rc.Leads > 0 {
			fmt.Printf(" recovered %d lead(s) stranded by a previous run\n", rc.Leads)
		}
		if rc.Reservations > 0 {
			fmt.Printf(" released %d stale reservation(s) from a previous run\n", rc.Reservations)
		}
		if rc.Sessions > 0 {
			fmt.Printf(" closed %d session(s) abandoned by a previous run\n", rc.Sessions)
		}
	}

	res, err := runner.Run(ctx, sess, spec)
	if err != nil {
		return err
	}
	fillOutput(out, res)
	report := res.Report
	status := res.Status
	runErr := res.Err

	if o.silent {
		return nil
	}
	if o.asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(out); err != nil {
			return err
		}
	} else {
		printReport(out, report, unit, sess.Budget, o.quiet)
	}

	if runErr != nil {
		return runErr
	}
	if status == core.StatusFailed {
		return fmt.Errorf("session ended %s: %s", status, out.StoppedBecause)
	}
	return nil
}

// maxLeadsFor bounds the lead tree from the planner's own fan-out limits, so
// the ceiling and the plan cannot disagree.
// plannerDepth translates the flag into the planner's encoding, where zero
// means "unset" and DepthNone means "no follow-ups".
func plannerDepth(flag int) int {
	if flag <= 0 {
		return planner.DepthNone
	}
	return flag
}

func maxLeadsFor(o researchOpts) int {
	depth := o.maxDepth
	if depth < 0 {
		depth = 0
	}
	return planner.DefaultMaxInitialLeads + depth*planner.DefaultMaxNewLeadsPerReplan + 2
}

func buildWebActor(
	cfg *config.Config, rec *record.Recorder, maxSources int, alwaysFetch, quiet, needSearch bool,
) (*actors.WebActor, error) {
	var provider search.Provider
	if needSearch {
		if cfg.Search.Provider == "" {
			return nil, errors.New("no search provider selected (run: mole config set search.provider brave|tavily)")
		}
		if cfg.Search.ActiveKey() == "" {
			return nil, fmt.Errorf("no API key for %s (run: mole config set search.%s-key ...)",
				cfg.Search.Provider, cfg.Search.Provider)
		}
		p, err := search.New(search.Config{
			Provider:           search.Kind(cfg.Search.Provider),
			APIKey:             cfg.Search.ActiveKey(),
			CostPerQueryMicros: cfg.Search.CostPerQueryMicros,
		}, rec.Client())
		if err != nil {
			return nil, err
		}
		provider = p
	}

	model, _, err := buildLLMWithClient(cfg, rec.Client())
	if err != nil {
		return nil, err
	}
	if model == nil {
		return nil, errors.New("no LLM provider found — set one with `mole config set llm.api-key ...`, " +
			"run `ant auth login`, or start a local model (ollama serve)")
	}

	level := slog.LevelWarn
	if quiet {
		level = slog.LevelError
	}
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))

	return &actors.WebActor{
		Search: provider,
		// Wrap rather than replace: the fetcher's transport carries the egress
		// guard's DialContext, and handing it a plain client would bypass SSRF
		// protection to get determinism. Wrapped, a recording still passes the
		// guard and a replay never opens a socket.
		Fetch: fetch.NewHTTP(fetch.Config{UserAgent: userAgent()}, fetch.Options{
			Log:       log,
			Transport: rec.Wrap,
		}),
		Extract: extract.New(),
		LLM:     model,
		Pricing: pricingFor(cfg),
		Log:     log,
		Budget: actors.Budget{
			MaxSources:         maxSources,
			MaxClaimsPerSource: 8,
			MaxChunkTokens:     cfg.LLM.MaxInputTokens,
			AlwaysFetch:        alwaysFetch,
		},
	}, nil
}

// checkUSDIsEnforceable refuses a dollar budget the ledger cannot enforce.
//
// An unpriced model records zero dollars per call, so --usd names a ceiling
// nothing counts against: the run spends whatever it spends and reports a few
// cents of search cost. Budget is this project's first-class primitive (§8),
// and a ceiling that silently does not bind is worse than no ceiling, because
// the number printed at the end looks like it held.
//
// doctor already warns about this. Warning is not enough at the point where
// money is about to be spent under a limit that does not exist.
func checkUSDIsEnforceable(p llm.Provider) error {
	missing := unpricedModelList(p)
	if len(missing) == 0 {
		return nil
	}
	return fmt.Errorf(
		"--usd cannot bound this run: no price is registered for %s, so every model call "+
			"would be ledgered at $0.00 and the ceiling would never bind.\n"+
			"Use --tokens N instead, which counts what the provider reports.",
		strings.Join(missing, ", "))
}

// pricingFor builds the rate table for the configured provider.
//
// A model served from loopback is registered at zero rather than left unpriced.
// Both cost nothing in the ledger, and the difference is what mole SAYS about it:
// an unpriced model warns on every call, which for a local run means several
// hundred lines saying that a self-hosted model is free. That noise shares a
// channel with the warning that matters — a HOSTED model nobody registered, where
// zero cost really does mean the --usd ceiling cannot bind.
//
// Loopback only. A model served from another machine may well be metered by
// whoever runs it.
func pricingFor(cfg *config.Config) *pricing.Table {
	t := pricing.NewTable()
	if cfg == nil {
		return t
	}
	if pricing.IsLocalEndpoint(cfg.LLM.BaseURL) {
		for _, m := range []string{cfg.LLM.Model, cfg.LLM.CheapModel, cfg.LLM.VerifierModel} {
			if m != "" {
				t.RegisterLocal(m)
			}
		}
		return t
	}
	// The configured model prices a response that names one nobody registered.
	//
	// What a provider ANSWERS with is not what you configured, and it is theirs
	// to repoint silently: a run configured as "deepseek-chat" began coming back
	// as "deepseek-flash", which no table knew, so every call priced at zero and
	// --usd stopped binding. checkUSDIsEnforceable could not catch it — it
	// validates the configured name, and the ledger prices the returned one.
	//
	// Declared rates before the fallback is chosen, so declaring a price for an
	// exotic model is what makes it usable as one.
	if err := t.Declare(cfg.Pricing); err != nil {
		// Not fatal here. Config load already validated these; reaching this
		// means something changed underneath, and a warning beats refusing to
		// run a token-mode session over a rate it will never consult.
		slog.Default().Warn("ignoring a declared rate", "err", err)
	}
	// Best effort. A configured model with no rates of its own leaves the table
	// strict, which is the state checkUSDIsEnforceable already refuses to start
	// a USD run in.
	_ = t.SetFallbackModel(cfg.LLM.Model)
	return t
}

func userAgent() string {
	// An anonymous crawler is impolite and gives an operator no way to contact
	// anyone or block selectively (§10.2).
	return "mole/" + version + " (+https://github.com/lajosdeme/mole)"
}

// resolveBudget turns the flags into a unit and an amount.
//
// Mutually exclusive by construction. §8 makes the unit semantically
// load-bearing — USD mode cannot bound an unpriced model, token mode cannot
// price a search call — so guessing one would produce a ceiling that silently
// does not bind.
func resolveBudget(usd string, tokens int64, cfg *config.Config) (core.BudgetUnit, int64, error) {
	switch {
	case usd != "" && tokens > 0:
		return "", 0, errors.New("--usd and --tokens are mutually exclusive")

	case usd != "":
		micros, err := core.ParseUSD(usd)
		if err != nil {
			return "", 0, fmt.Errorf("--usd: %w", err)
		}
		if micros <= 0 {
			return "", 0, errors.New("--usd must be positive")
		}
		return core.BudgetUSD, micros, nil

	case tokens > 0:
		return core.BudgetTokens, tokens, nil
	}

	// Fall back to a configured default, so a bare `mole research` still cannot
	// run away (§8.5).
	switch core.BudgetUnit(cfg.DefaultBudgetUnit) {
	case core.BudgetUSD:
		micros, err := core.ParseUSD(cfg.DefaultBudgetUSD)
		if err != nil || micros <= 0 {
			return "", 0, errors.New("default_budget_usd is not a valid amount (run: mole config set default.usd 3.00)")
		}
		return core.BudgetUSD, micros, nil
	case core.BudgetTokens:
		if cfg.DefaultBudgetToks <= 0 {
			return "", 0, errors.New("default_budget_tokens is not set (run: mole config set default.tokens 50000)")
		}
		return core.BudgetTokens, cfg.DefaultBudgetToks, nil
	}

	return "", 0, errors.New("no budget given: pass --usd N or --tokens N, or set a default " +
		"(mole config set default.unit usd; mole config set default.usd 3.00)")
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

type researchOutput struct {
	Question  string `json:"question"`
	SessionID string `json:"session_id"`
	Status    string `json:"status"`
	Unit      string `json:"budget_unit"`
	Budget    int64  `json:"budget"`
	Spent     int64  `json:"spent"`

	LeadsRun      int `json:"leads_run"`
	LeadsFailed   int `json:"leads_failed"`
	LeadsCached   int `json:"leads_cached"`
	CacheHits     int `json:"cache_hits"`
	Replans       int `json:"replans"`
	OpenQuestions int `json:"open_questions"`

	// Verification and grounding (§11).
	ClaimsVerified    int `json:"claims_verified"`
	Contradictions    int `json:"contradictions"`
	GroundChecked     int `json:"ground_checked"`
	GroundConfirmed   int `json:"ground_confirmed"`
	GroundUnsupported int `json:"ground_unsupported"`

	Report string       `json:"report"`
	Claims []core.Claim `json:"claims"`

	// StoppedBecause names the ceiling that ended the run, when one did.
	//
	// Declared here from the start and filled by nothing until the reason was
	// persisted: session.Result had no such field, so this marshalled empty on
	// every run a caller parsed it from.
	//
	// Status is not enough on its own. "budget_exhausted" is what a run reports
	// after hitting max_tool_calls, max_leads or max_wallclock as well as after
	// actually running out of money, and the money is rarely the one that binds.
	StoppedBecause string `json:"stopped_because,omitempty"`
	Degraded       string `json:"degraded,omitempty"`
	Error          string `json:"error,omitempty"`
}

// fillOutput maps a run's result onto the JSON shape the CLI prints.
//
// Every field the daemon will also need is on session.Result; this is the last
// place that knows about researchOutput.
func fillOutput(out *researchOutput, res *session.Result) {
	if res == nil {
		return
	}
	out.Status = string(res.Status)
	out.StoppedBecause = res.StoppedBecause
	if res.Err != nil {
		out.Error = res.Err.Error()
	}
	if r := res.Run; r != nil {
		out.ClaimsVerified = r.ClaimsVerified
		out.Contradictions = r.Contradictions
		out.LeadsRun = r.LeadsRun
		out.LeadsFailed = r.LeadsFailed
		out.LeadsCached = r.LeadsCached
		out.Replans = r.Replans
		out.CacheHits = r.CacheStats.Hits
		out.Claims = r.Claims
		out.StoppedBecause = r.StoppedBecause
		out.Spent = r.Spent
		if r.Digest != nil {
			out.OpenQuestions = len(r.Digest.Open())
		}
	}
	if g := res.Ground; g != nil {
		out.GroundChecked = g.Checked
		out.GroundConfirmed = g.Confirmed
		out.GroundUnsupported = g.Unsupported
	}
	if rep := res.Report; rep != nil {
		out.Report = rep.Markdown()
		out.Degraded = rep.Degraded
	}
}

// progressPrinter reports each phase as it happens, with elapsed time.
//
// Live feedback, not decoration. Planning is one model call that produces no
// output until it returns, and on a local model that is minutes of silence after
// a one-line header — which reads as a hang. Three real runs were killed by hand
// before reaching the first lead.
func progressPrinter() func(executor.Event) {
	start := time.Now()
	return func(ev executor.Event) {
		el := time.Since(start).Round(time.Second)
		switch ev.Phase {
		case "planning":
			fmt.Printf(" [%6s] planning: %s…\n", el, ev.Detail)
		case "executing":
			fmt.Printf(" [%6s] plan ready: %s\n", el, ev.Detail)
		case "replanning":
			fmt.Printf(" [%6s] replanning: %s…\n", el, ev.Detail)
		case "lead":
			fmt.Printf(" [%6s]   → %s\n", el, ev.Detail)
		case "lead-done":
			fmt.Printf(" [%6s]     %s\n", el, ev.Detail)
		case "cached":
			fmt.Printf(" [%6s]   ⤿ cached: %s\n", el, ev.Detail)
		case "verifying":
			fmt.Printf(" [%6s] verifying: %s…\n", el, ev.Detail)
		case "verified":
			fmt.Printf(" [%6s]     graph: %s\n", el, ev.Detail)
		case "follow-up":
			fmt.Printf(" [%6s]   ⚡ %s\n", el, ev.Detail)
		}
	}
}

// printProgress shows what the loop did, in the shape §18.3 sketches.
func printProgress(res *executor.Result) {
	mark := "✓"
	switch {
	case res.LeadsRun == 0:
		mark = "⚠"
	case res.LeadsFailed*2 > res.LeadsRun:
		// More than half the leads failed. The run produced something, but not
		// the something it was asked for (§9.5 degraded).
		mark = "~"
	}
	fmt.Printf(" %s %d lead(s) run, %d failed  ·  %d replan(s)  ·  %d claim(s)\n",
		mark, res.LeadsRun, res.LeadsFailed, res.Replans, len(res.Claims))
	if res.LeadsCached > 0 || res.CacheStats.Hits > 0 {
		fmt.Printf("   cache: %d lead(s) reused, %d source(s) served without a fetch\n",
			res.LeadsCached, res.CacheStats.Hits)
	}
	if res.VerifyPasses > 0 {
		fmt.Printf("   graph: %d claim(s) verified, %d edge(s), %d contradiction(s)",
			res.ClaimsVerified, res.EdgesWritten, res.Contradictions)
		if res.FollowUpsQueued > 0 {
			fmt.Printf(", %d follow-up lead(s)", res.FollowUpsQueued)
		}
		fmt.Println()
		if res.VerifyDegraded != "" {
			// Said out loud. An unverified claim keeps confidence 0, and a reader
			// who does not know verification was cut short reads that as "nothing
			// here is trustworthy" rather than "nothing here was checked".
			fmt.Printf("   verification incomplete: %s\n", res.VerifyDegraded)
		}
	} else if len(res.Claims) > 0 {
		fmt.Println("   graph: not built — claims carry verified quotes but no derived confidence")
	}
	if res.StoppedBecause != "" {
		fmt.Printf("   stopped: %s\n", res.StoppedBecause)
	}
}

func printReport(out *researchOutput, report *output.Report, unit core.BudgetUnit, budgetAmt int64, quiet bool) {
	if !quiet {
		fmt.Println()
	}
	if report != nil {
		fmt.Println(report.Markdown())
	} else if out.Report != "" {
		fmt.Println(out.Report)
	}

	fmt.Printf(" spent %s / %s  ·  %d lead(s), %d failed  ·  %d replan(s)  ·  %d claim(s)\n",
		fmtAmount(unit, out.Spent), fmtAmount(unit, budgetAmt),
		out.LeadsRun, out.LeadsFailed, out.Replans, len(out.Claims))

	if out.OpenQuestions > 0 {
		// The planner stopped with questions still open. Saying so is the
		// difference between "this is the answer" and "this is what fit the
		// budget", which a reader has to be able to tell apart.
		fmt.Printf(" %d sub-question(s) still open when the run stopped\n", out.OpenQuestions)
	}
	if out.StoppedBecause != "" {
		fmt.Printf(" stopped: %s\n", out.StoppedBecause)
	}
	if out.Error != "" {
		fmt.Printf(" error: %s\n", out.Error)
	}
	fmt.Printf("\n mole eval %s        for the scorecard\n", out.SessionID)
	fmt.Printf(" mole trace %s       for the per-call cost breakdown\n", out.SessionID)
}

func fmtAmount(unit core.BudgetUnit, amount int64) string {
	if unit == core.BudgetTokens {
		return fmt.Sprintf("%d tok", amount)
	}
	return core.FormatUSD(amount)
}

// reportGrounding prints what the pass did, including when it did nothing.
//
// Split out so the "did nothing" branches are testable without a store or a network —
// they are the ones that were silently wrong twice.
func reportGrounding(rep *verifier.GroundReport, o researchOpts) {
	if rep.Checked == 0 {
		switch {
		case rep.NotFetchable > 0:
			fmt.Printf("   grounding: not run — %d claim(s) cite pages the search provider "+
				"supplied, which cannot be re-read without comparing two different "+
				"extractions\n", rep.NotFetchable)
		case rep.Degraded != "":
			fmt.Printf("   grounding: not run — %s\n", rep.Degraded)
		default:
			fmt.Println("   grounding: no claim was eligible for a re-read")
		}
		return
	}

	fmt.Printf("   grounding: %d claim(s) re-read — %d confirmed, %d unsupported",
		rep.Checked, rep.Confirmed, rep.Unsupported)
	if rep.NotFetchable > 0 {
		fmt.Printf(", %d not re-readable", rep.NotFetchable)
	}
	if n := rep.Vanished + rep.Unreachable + rep.Undecided; n > 0 {
		// Said separately: none of these is a verdict about a claim. The page changed,
		// the host was down, or the judge would not answer. Folding them into
		// "unsupported" would penalize a claim for someone else's edit.
		fmt.Printf(", %d inconclusive", n)
	}
	fmt.Println()
	for _, r := range rep.Results {
		if r.Outcome == verifier.GroundUnsupported {
			// The one outcome a reader must see: a claim whose own source does not
			// support it is invisible in a confidence number.
			fmt.Printf("     ⚠ %.70s\n", r.Note)
		}
	}
}

// checkCassetteIsSerial refuses to record or replay with a worker pool.
//
// A cassette is keyed on the request body (record.RequestKey), and with a pool
// the request bodies stop being reproducible: the synthesis prompt is built from
// claims in order, and lead completion order is whatever the network decided
// that run. So a cassette recorded under concurrency cannot be replayed — not at
// a different worker count, and not even against itself, because the second run
// interleaves differently.
//
// Refused rather than silently forced to one, because the two are not the same
// promise. Quietly serialising would make `--workers 8` mean four different
// things depending on an environment variable; refusing makes the constraint
// visible at the moment it applies. This is the whole reason concurrency and
// deterministic replay can coexist: everything that touches a cassette runs
// serial, and everything else — every real research run — does not.
func checkCassetteIsSerial(workers int) error {
	// The EFFECTIVE count, not the flag. --workers 0 means "use the default",
	// so testing the raw value let 0 and any negative through a check whose
	// entire job is to enforce serial execution, and then run a pool of four.
	effective := executor.EffectiveWorkers(workers)
	if effective <= 1 {
		return nil
	}
	mode, err := record.ModeFromEnv()
	if err != nil || mode == record.ModeOff {
		// A malformed MOLE_RECORD is FromEnv's to report, with its own message.
		return nil
	}
	return fmt.Errorf(
		"MOLE_RECORD=%s needs --workers 1 (got %d, which runs %d): a cassette "+
			"recorded with a worker pool cannot be replayed, because lead "+
			"completion order decides the prompts it is keyed on",
		mode, workers, effective)
}

// checkAcademicConfig refuses --actors academic without a contact address.
//
// §10.3 makes this a startup check rather than a README line. An error rather
// than a silent downgrade to web-only: a run that quietly researched half of
// what was asked for is worse than one that says why.
func checkAcademicConfig(cfg *config.Config, types []core.ActorType) error {
	if !wantsAcademic(types) {
		return nil
	}
	if err := academic.CheckContact(cfg.ContactEmail); err != nil {
		return fmt.Errorf("%w\n--actors academic requires it; set one with: "+
			"mole config set contact-email you@example.com", err)
	}
	return nil
}

func wantsAcademic(types []core.ActorType) bool {
	for _, t := range types {
		if t == core.ActorAcademic {
			return true
		}
	}
	return false
}

// parseActorTypes reads the --actors list.
func parseActorTypes(raw string) ([]core.ActorType, error) {
	var out []core.ActorType
	seen := map[core.ActorType]bool{}
	for _, part := range strings.Split(raw, ",") {
		t := core.ActorType(strings.ToLower(strings.TrimSpace(part)))
		if t == "" {
			continue
		}
		switch t {
		case core.ActorWeb, core.ActorAcademic, core.ActorLocalCompute:
		default:
			return nil, fmt.Errorf("unknown actor %q (want web, academic or local_compute)", t)
		}
		if !seen[t] {
			seen[t] = true
			out = append(out, t)
		}
	}
	if len(out) == 0 {
		out = []core.ActorType{core.ActorWeb}
	}
	return out, nil
}

// resolveSchema turns the flags into a schema, and refuses the combinations that
// cannot mean anything.
//
// A dataset session with no schema is refused rather than inferred. Inference
// needs a model call, and a session that silently invented its own columns would
// produce a table nobody asked for and charge for it — `mole dataset infer` is
// where that belongs, so the schema a run uses is always one somebody saw.
func resolveSchema(mode core.Mode, spec, file string) (*dataset.Schema, error) {
	if mode != core.ModeDataset {
		if spec != "" || file != "" {
			return nil, errors.New("--schema is only meaningful with --mode dataset")
		}
		return nil, nil
	}
	switch {
	case spec != "" && file != "":
		return nil, errors.New("give --schema or --schema-file, not both")
	case file != "":
		s, err := dataset.LoadSchema(file)
		return &s, err
	case spec != "":
		s, err := dataset.ParseSpec(spec)
		return &s, err
	}
	return nil, errors.New("--mode dataset needs a schema: --schema " +
		"company:text!,revenue:number (! marks the field that identifies a row), " +
		"or --schema-file schema.json")
}

// buildLocalActor wires the connector registry into the LocalComputeActor.
//
// It refuses when nothing is registered rather than starting a session whose
// local leads would each report "no local data". §12's actor with no connector
// is not a degraded run, it is a misconfiguration, and the executor treats a
// missing actor as fatal — so the refusal has to happen here.
func buildLocalActor(
	ctx context.Context, types []core.ActorType, o researchOpts, web *actors.WebActor,
) (*actors.LocalComputeActor, error) {
	if !slices.Contains(types, core.ActorLocalCompute) {
		return nil, nil
	}
	reg, err := connector.LoadRegistry(connectorRegistryPath(o.dbPath))
	if err != nil {
		return nil, err
	}
	if len(reg.List()) == 0 {
		return nil, fmt.Errorf("--actors local_compute needs registered data; " +
			"add some with: mole connect add <name> <path>")
	}
	local := &actors.LocalComputeActor{
		Connectors: reg,
		// The durable half of §12.1's audit trail. The log line alone met the
		// letter of "every crossing is logged" and not the use: logs rotate, and
		// a user asking what mole sent about their data last Tuesday cannot query
		// a log line. `mole crossings <session>` reads the table.
		Store:   web.Store,
		LLM:     web.LLM,
		Pricing: web.Pricing,
		Log:     web.Log,
		Budget:  web.Budget,
		// §12.1: "Every crossing is logged, so a user can audit exactly what
		// left their machine." That was written at Info while the actor's logger
		// is built at Warn, so the audit trail emitted NOTHING in the shipped
		// path — the one requirement whose whole purpose is to be readable
		// afterwards. Its own sink, at its own level, so raising or lowering
		// research logging cannot silence it again.
		Gate: gate.Options{Log: auditLogger()},
	}

	// The sandbox is optional and its absence is not an error (§12.2: it "is not
	// the control here"). A nil Code means code hypotheses are never offered to
	// the model, so the run loses the analyses SQL cannot express and nothing
	// else. `mole doctor` is where somebody finds out.
	if rep := sandbox.Detect(ctx); rep.Usable {
		local.Code = coderunner.Sandboxed{Report: rep}
	}
	return local, nil
}

// auditLogger is the sink for §12.1's crossing records.
//
// Separate from the research logger on purpose. Those two have different
// audiences: research logging is diagnostics somebody turns down when it gets
// noisy, and this is the record of what left the machine.
func auditLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
}

// buildAcademicActor constructs the academic providers, or refuses.
//
// Nil when the session did not ask for academic sources — building providers
// nobody requested would make every run depend on a contact email §10.3 only
// requires of the people who use them. When it IS asked for, a missing address
// is an error rather than a silent downgrade to web-only: a run that quietly
// researched half of what was requested is worse than one that says why.
func buildAcademicActor(cfg *config.Config, types []core.ActorType, web *actors.WebActor) (*actors.AcademicActor, error) {
	if !wantsAcademic(types) {
		return nil, nil
	}
	lim := limiter.New(limiter.Unlimited)
	academic.Register(lim)

	acfg := academic.Config{ContactEmail: cfg.ContactEmail}
	arxiv, err := academic.NewArXiv(acfg, nil, lim)
	if err != nil {
		return nil, err
	}
	pubmed, err := academic.NewPubMed(acfg, nil, lim)
	if err != nil {
		return nil, err
	}
	// A Resolver rather than a Provider: Unpaywall has no topical search, so it
	// never appears in Providers. It places papers the other two found but could
	// not give a readable location for.
	unpaywall, err := academic.NewUnpaywall(acfg, nil, lim)
	if err != nil {
		return nil, err
	}

	return &actors.AcademicActor{
		Providers: []academic.Provider{arxiv, pubmed},
		LLM:       web.LLM,
		Pricing:   web.Pricing,
		Log:       web.Log,
		Budget:    web.Budget,
		// Shared with the web actor on purpose: the fetcher carries the egress
		// guard, the robots cache and the per-domain limiter, and a second one
		// would be a second unmetered path to the same hosts.
		Fetch:   web.Fetch,
		Extract: web.Extract,
		Resolve: unpaywall,
		Rank:    rankPassages,
	}, nil
}

// rankPassages orders passages by relevance, using the Verifier's retriever.
//
// Supplied here rather than imported by internal/actors, which cannot reach
// internal/verifier: verifier/ground.go imports actors, so the dependency only
// runs one way. Wiring it at the composition root is what keeps one scorer in
// the codebase instead of two that drift.
func rankPassages(question string, passages []string, n int) []int {
	if len(passages) == 0 {
		return nil
	}
	pool := make([]*core.Claim, len(passages))
	index := map[string]int{}
	for i, p := range passages {
		id := fmt.Sprintf("p%d", i)
		pool[i] = &core.Claim{ID: id, Text: p}
		index[id] = i
	}
	got, err := verifier.LexicalRetriever{}.Candidates(
		context.Background(), &core.Claim{Text: question}, pool, n)
	if err != nil {
		return nil
	}
	out := make([]int, 0, len(got))
	for _, c := range got {
		if i, ok := index[c.ID]; ok {
			out = append(out, i)
		}
	}
	return out
}

// resolveConfirmMode turns the flag into a mode, refusing what it does not recognise.
//
// Unknown is an error, not the default: the mode moves contradiction recall between
// 98% and 67%, so a typo choosing that silently is the wrong graph reported as the
// right one. Passing both flags is refused rather than given an invented precedence.
//
// An EMPTY mode means the flag was not passed, which is why the flag's own
// default is empty rather than "both". Comparing against "both" instead got the
// precedence wrong in both directions: --confirm-edges=none --no-confirm-edges
// was refused for disagreeing when the two say the same thing, and
// --confirm-edges=both --no-confirm-edges silently let the deprecated alias
// overrule an explicit choice — the invented precedence this refuses to have.
func resolveConfirmMode(mode string, noConfirm bool) (verifier.ConfirmMode, error) {
	m := verifier.ConfirmMode(strings.ToLower(strings.TrimSpace(mode)))
	explicit := m != ""
	if !explicit {
		m = verifier.ConfirmBoth
	}
	switch m {
	case verifier.ConfirmBoth, verifier.ConfirmDuplicates, verifier.ConfirmNone:
	default:
		return "", fmt.Errorf("--confirm-edges %q: want both, duplicates or none", mode)
	}

	if noConfirm {
		if explicit && m != verifier.ConfirmNone {
			return "", fmt.Errorf(
				"--no-confirm-edges and --confirm-edges=%s disagree: pass only --confirm-edges", m)
		}
		return verifier.ConfirmNone, nil
	}
	return m, nil
}

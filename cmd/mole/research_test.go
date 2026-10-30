package main

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/budget"
	"github.com/lajosdeme/mole/internal/config"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/executor"
	"github.com/lajosdeme/mole/internal/session"
	"github.com/lajosdeme/mole/internal/store/sqlite"

	"github.com/lajosdeme/mole/internal/verifier"
)

// ---------------------------------------------------------------------------
// Budget resolution
// ---------------------------------------------------------------------------

// TestBudgetFlagsAreMutuallyExclusive. §8 makes the unit semantically
// load-bearing — USD mode cannot bound an unpriced model, token mode cannot
// price a search call — so silently picking one produces a ceiling that does
// not bind.
func TestBudgetFlagsAreMutuallyExclusive(t *testing.T) {
	if _, _, err := resolveBudget("3.00", 50_000, &config.Config{}); err == nil {
		t.Error("--usd and --tokens together were accepted")
	}
}

func TestBudgetFlagsParse(t *testing.T) {
	unit, amount, err := resolveBudget("3.00", 0, &config.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if unit != core.BudgetUSD || amount != 3*core.MicrosPerUSD {
		t.Errorf("got %s/%d, want usd/%d", unit, amount, 3*core.MicrosPerUSD)
	}

	unit, amount, err = resolveBudget("", 50_000, &config.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if unit != core.BudgetTokens || amount != 50_000 {
		t.Errorf("got %s/%d, want tokens/50000", unit, amount)
	}
}

// TestBareResearchCannotRunAway: with no flag and no configured default there
// is no budget, and a research command with no ceiling is the failure mode §8.5
// exists to prevent. Refusing is the only safe answer — a built-in default
// would be a number nobody chose being spent on someone's card.
func TestBareResearchCannotRunAway(t *testing.T) {
	if _, _, err := resolveBudget("", 0, &config.Config{}); err == nil {
		t.Error("a research run with no budget at all was accepted")
	}
}

func TestConfiguredDefaultBudgetIsUsed(t *testing.T) {
	cfg := &config.Config{DefaultBudgetUnit: "usd", DefaultBudgetUSD: "2.50"}
	unit, amount, err := resolveBudget("", 0, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if unit != core.BudgetUSD || amount != 2_500_000 {
		t.Errorf("got %s/%d, want usd/2500000", unit, amount)
	}

	cfg = &config.Config{DefaultBudgetUnit: "tokens", DefaultBudgetToks: 12_000}
	unit, amount, err = resolveBudget("", 0, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if unit != core.BudgetTokens || amount != 12_000 {
		t.Errorf("got %s/%d, want tokens/12000", unit, amount)
	}
}

// TestIncompleteDefaultIsRefusedRatherThanGuessed: a unit set without an amount
// is a half-finished config, and inferring the missing half would produce a
// ceiling the user never chose.
func TestIncompleteDefaultIsRefusedRatherThanGuessed(t *testing.T) {
	for name, cfg := range map[string]*config.Config{
		"unit without amount":  {DefaultBudgetUnit: "usd"},
		"tokens without count": {DefaultBudgetUnit: "tokens"},
		"unparseable amount":   {DefaultBudgetUnit: "usd", DefaultBudgetUSD: "three dollars"},
		"zero amount":          {DefaultBudgetUnit: "usd", DefaultBudgetUSD: "0"},
	} {
		if _, _, err := resolveBudget("", 0, cfg); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestNegativeBudgetRefused(t *testing.T) {
	if _, _, err := resolveBudget("-1.00", 0, &config.Config{}); err == nil {
		t.Error("a negative dollar budget was accepted")
	}
}

// ---------------------------------------------------------------------------
// Lead cap
// ---------------------------------------------------------------------------

func TestMaxLeadsFlagOverridesTheDerivedCap(t *testing.T) {
	got := maxLeadsFor(researchOpts{maxDepth: 2, maxLeads: 30})
	if got != 30 {
		t.Errorf("got %d, want 30", got)
	}
}

func TestZeroMaxLeadsKeepsTheDerivedCap(t *testing.T) {
	got := maxLeadsFor(researchOpts{maxDepth: 2})
	if got != 12 {
		t.Errorf("got %d, want 12", got)
	}
}

func TestNegativeMaxLeadsIsRefused(t *testing.T) {
	if err := checkMaxLeadsIsValid(-1); err == nil {
		t.Error("a negative --max-leads was accepted")
	}
}

func TestANonPositiveTimeoutIsRefused(t *testing.T) {
	t.Setenv("MOLE_CONFIG_DIR", t.TempDir())
	t.Setenv("MOLE_DB", filepath.Join(t.TempDir(), "mole.db"))
	for _, d := range []time.Duration{0, -time.Second} {
		err := cmdResearch(context.Background(), "a question",
			researchOpts{timeout: d, timeoutSet: true, tokens: 1000})
		if err == nil || !strings.Contains(err.Error(), "--timeout") {
			t.Errorf("--timeout %v: err = %v, want --timeout refused", d, err)
		}
	}
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------
//
// The reserve → run → settle tests that lived here moved to internal/executor
// when `mole research` stopped running a single hardcoded lead: the loop owns
// that cycle now, and testing it through the CLI would be testing it through a
// layer that no longer decides anything about it.

// captureStdout runs fn with os.Stdout redirected and returns what it printed.
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	saved := os.Stdout
	os.Stdout = w
	defer func() { os.Stdout = saved }()

	done := make(chan string, 1)
	go func() {
		b, _ := io.ReadAll(r)
		done <- string(b)
	}()

	fn()
	w.Close()
	return <-done
}

func newTestDB(t *testing.T) *sqlite.DB {
	t.Helper()
	db, err := sqlite.Open(filepath.Join(t.TempDir(), "t.db"), sqlite.Options{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.Migrate(context.Background()); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	return db
}

func newTestSession(t *testing.T, led *budget.Ledger, unit core.BudgetUnit, amount int64) *core.Session {
	t.Helper()
	sess, err := led.CreateSession(context.Background(), budget.SessionSpec{
		Prompt:     "a question",
		Mode:       core.ModeReport,
		ActorTypes: []core.ActorType{core.ActorWeb},
		BudgetUnit: unit,
		Budget:     amount,
		MaxLeads:   1,
	})
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	return sess
}

// TestReportShowsWhatTheRunCostAndWhereItStopped. A reader deciding whether to
// trust the answer needs both: an answer produced by two of nine planned leads
// before the budget ran out is a different thing from a complete one.
func TestReportShowsWhatTheRunCostAndWhereItStopped(t *testing.T) {
	out := &researchOutput{
		SessionID:      "s_test",
		Status:         string(core.StatusExhausted),
		Spent:          184_000,
		LeadsRun:       7,
		LeadsFailed:    2,
		Replans:        2,
		OpenQuestions:  3,
		StoppedBecause: "max_leads",
		Claims: []core.Claim{
			{Text: "A claim.", Source: "https://a.example", Quote: "a quote"},
		},
	}

	got := captureStdout(t, func() {
		printReport(out, nil, core.BudgetUSD, 3*core.MicrosPerUSD, false)
	})

	for _, want := range []string{
		"$0.1840", "7 lead(s), 2 failed", "2 replan(s)",
		"3 sub-question(s) still open", "stopped: max_leads",
		"mole eval s_test", "mole trace s_test",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("report is missing %q:\n%s", want, got)
		}
	}
}

// TestCompleteRunDoesNotClaimOpenQuestions, so the warning keeps meaning
// something.
func TestCompleteRunDoesNotClaimOpenQuestions(t *testing.T) {
	out := &researchOutput{SessionID: "s_c", Status: string(core.StatusDone), LeadsRun: 3}
	got := captureStdout(t, func() {
		printReport(out, nil, core.BudgetUSD, core.MicrosPerUSD, true)
	})
	if strings.Contains(got, "still open") {
		t.Errorf("a complete run reported open questions:\n%s", got)
	}
}

// TestProgressMarksADegradedRun. More than half the leads failing means the run
// produced something, but not the something it was asked for (§9.5).
func TestProgressMarksADegradedRun(t *testing.T) {
	degraded := captureStdout(t, func() {
		printProgress(&executor.Result{LeadsRun: 6, LeadsFailed: 5})
	})
	if !strings.Contains(degraded, "~") {
		t.Errorf("a mostly-failed run was not marked degraded:\n%s", degraded)
	}

	clean := captureStdout(t, func() {
		printProgress(&executor.Result{LeadsRun: 6, LeadsFailed: 0})
	})
	if !strings.Contains(clean, "✓") {
		t.Errorf("a clean run was not marked clean:\n%s", clean)
	}
}

func TestAmountFormattingMatchesTheUnit(t *testing.T) {
	if got := fmtAmount(core.BudgetUSD, 3*core.MicrosPerUSD); !strings.HasPrefix(got, "$3.") {
		t.Errorf("USD formatted as %q", got)
	}
	if got := fmtAmount(core.BudgetTokens, 50_000); got != "50000 tok" {
		t.Errorf("tokens formatted as %q, want \"50000 tok\"", got)
	}
}

// TestProgressPrinterEmitsEachPhase covers the printer itself.
func TestProgressPrinterEmitsEachPhase(t *testing.T) {
	p := progressPrinter()
	got := captureStdout(t, func() {
		p(executor.Event{Phase: "planning", Detail: "decomposing the question"})
		p(executor.Event{Phase: "executing", Detail: "4 sub-question(s) queued"})
		p(executor.Event{Phase: "lead", Detail: "a sub-question"})
		p(executor.Event{Phase: "lead-done", Detail: "3 claim(s)"})
		p(executor.Event{Phase: "replanning", Detail: "2 open sub-question(s)"})
		p(executor.Event{Phase: "cached", Detail: "a repeated question"})
	})

	for _, want := range []string{
		"planning: decomposing", "plan ready: 4 sub-question(s)",
		"a sub-question", "3 claim(s)", "replanning: 2 open", "cached: a repeated",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("output is missing %q:\n%s", want, got)
		}
	}
}

// TestResearchWiresTheProgressPrinter is the test that was missing, and its
// absence is why the previous commit shipped a silent binary.
//
// The executor emitted all six events and the printer formatted all six, both
// covered — but nothing asserted that cmdResearch actually ASSIGNS the callback.
// A struct-literal edit that failed to apply left Progress nil, the events fired
// into nothing, and every test still passed. Same shape as the MaxLeads ceiling
// whose counter no caller incremented: mechanism tested, wiring not.
func TestResearchWiresTheProgressPrinter(t *testing.T) {
	src, err := os.ReadFile("research.go")
	if err != nil {
		t.Fatal(err)
	}
	body := string(src)

	// All three display callbacks, each with the same failure mode: assigned to a
	// struct the CLI hands off, so a dropped line is invisible to every other
	// test. The executor now lives behind session.Runner, which is why these read
	// "runner." rather than "exec." — the wiring moved, the hazard did not.
	//
	// Containment is checked by finding the gate and confirming its block has not
	// closed before the assignment, rather than by scanning a fixed window of
	// preceding characters. The window version broke the moment a comment was
	// added between the two, which tests the comment and not the code.
	const gate = "if !o.quiet && !o.asJSON {"
	g := strings.Index(body, gate)
	if g < 0 {
		t.Fatalf("the output-mode gate %q is gone", gate)
	}
	for _, assign := range []string{
		"runner.Progress = progressPrinter()",
		"runner.OnLoop = printProgress",
		"runner.OnGround = func(",
	} {
		i := strings.Index(body, assign)
		if i < 0 {
			t.Errorf("cmdResearch never assigns %s; those events go nowhere", assign)
			continue
		}
		if i < g {
			t.Errorf("%s is assigned before the output-mode gate", assign)
			continue
		}
		// A closing brace at one tab of indent ends the gated block. Reaching one
		// before the assignment means the assignment sits outside it — so JSON
		// callers would get progress lines interleaved with their document.
		if strings.Contains(body[g:i], "\n\t}") {
			t.Errorf("%s is not inside the --quiet/--json gate", assign)
		}
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// TestASkippedGroundingPassSaysSo.
//
// Two live runs reached "grounding never happened" from different directions and both
// printed nothing. The first was an allowance smaller than one judge call; that was fixed,
// and the second — every source supplied by the search provider, so nothing was
// re-readable — was still silent, because NotFetchable is not Degraded.
//
// Silence makes a skipped check indistinguishable from a check that found no problems,
// which is the worst possible reading of it.
func TestASkippedGroundingPassSaysSo(t *testing.T) {
	cases := map[string]struct {
		rep  verifier.GroundReport
		want string
	}{
		"every source came from the provider": {
			verifier.GroundReport{NotFetchable: 4},
			"the search provider supplied",
		},
		"the allowance could not cover a call": {
			verifier.GroundReport{Degraded: "no allowance for grounding"},
			"no allowance for grounding",
		},
		"nothing was eligible": {
			verifier.GroundReport{},
			"no claim was eligible",
		},
	}
	for name, tc := range cases {
		rep := tc.rep
		out := captureStdout(t, func() { reportGrounding(&rep, researchOpts{}) })
		if !strings.Contains(out, tc.want) {
			t.Errorf("%s: output %q does not say why grounding did not run (want %q)",
				name, out, tc.want)
		}
	}

	// A pass that DID check says what it found instead.
	rep := verifier.GroundReport{Checked: 3, Confirmed: 2, Unsupported: 1}
	out := captureStdout(t, func() { reportGrounding(&rep, researchOpts{}) })
	if !strings.Contains(out, "3 claim(s) re-read") {
		t.Errorf("a completed pass did not report its result: %q", out)
	}
}

// TestFillOutputCarriesTheRecheckCounts.
func TestFillOutputCarriesTheRecheckCounts(t *testing.T) {
	var out researchOutput
	fillOutput(&out, &session.Result{Run: &executor.Result{Contradictions: 3, Rechecked: 10, Withheld: 7}})
	if out.Contradictions != 3 || out.Rechecked != 10 || out.Withheld != 7 {
		t.Errorf("contradictions/rechecked/withheld = %d/%d/%d, want 3/10/7",
			out.Contradictions, out.Rechecked, out.Withheld)
	}
}

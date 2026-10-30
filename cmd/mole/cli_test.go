package main

import (
	"bytes"
	"github.com/lajosdeme/mole/internal/verifier"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/core"
)

// exec runs the real command tree against captured output, so these assert what
// a user gets rather than what an internal function returns.
//
// ISOLATED, and that is not optional. These drive the actual commands, so
// without redirecting the database and the config they use the developer's real
// ones — which is exactly what happened: the flag-parsing test below created
// sessions in a live database and, once a reachable local model was configured,
// started making real model calls and hung until the test timeout.
//
// The empty config directory is what keeps that impossible: no search provider
// is configured, so `research` refuses before it can reach any provider. A test
// that asserts flag parsing must not depend on being unable to reach the
// network — it must be unable to get that far.
func exec(t *testing.T, args ...string) (string, error) {
	t.Helper()
	t.Setenv("MOLE_CONFIG_DIR", t.TempDir())
	t.Setenv("MOLE_DB", filepath.Join(t.TempDir(), "cli-test.db"))

	root := newRootCmd()
	var out bytes.Buffer
	root.SetOut(&out)
	root.SetErr(&out)
	root.SetArgs(args)
	err := root.Execute()
	return out.String(), err
}

// TestFlagsAfterPositionalsAreParsed is the bug that prompted this port. Go's
// flag package stops at the first non-flag argument, so
//
//	mole research "a question" --usd 0.50
//
// left --usd unparsed, folded it into the question, and then refused the
// command for having no budget while quoting the flag back. pflag handles this
// natively, but the behaviour is what matters, so it stays asserted.
func TestFlagsAfterPositionalsAreParsed(t *testing.T) {
	for _, args := range [][]string{
		{"research", "a question", "--usd", "0.50"},
		{"research", "--usd", "0.50", "a question"},
		{"research", "a question", "--usd=0.50"},
		{"research", "a", "--usd", "0.50", "question"},
	} {
		_, err := exec(t, args...)
		if err == nil {
			t.Errorf("%v: expected a refusal from the empty test config", args)
			continue
		}
		// It must fail on the missing search provider — which proves the budget
		// parsed, because resolveBudget runs first and would have refused there.
		if strings.Contains(err.Error(), "no budget given") {
			t.Errorf("%v: --usd was not parsed", args)
		}
		if !strings.Contains(err.Error(), "search provider") {
			t.Errorf("%v: failed on %q, not on the expected missing provider — "+
				"the test may be reaching a real provider", args, err)
		}
	}
}

// TestBudgetFlagsStillMutuallyExclusive: the port must not lose the rule.
func TestBudgetFlagsStillMutuallyExclusive(t *testing.T) {
	_, err := exec(t, "research", "a question", "--usd", "1.00", "--tokens", "500")
	if err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("err = %v, want the mutual-exclusion refusal", err)
	}
}

// TestUnknownCommandSuggests rather than only reporting the miss.
func TestUnknownCommandSuggests(t *testing.T) {
	out, err := exec(t, "reserch")
	if err == nil {
		t.Fatal("an unknown command succeeded")
	}
	combined := out + err.Error()
	if !strings.Contains(combined, "research") {
		t.Errorf("no suggestion offered:\n%s", combined)
	}
}

// TestArgErrorsNameTheCommand. Cobra's stock text is "accepts 1 arg(s),
// received 0", which restates the arity and not the thing to type.
func TestArgErrorsNameTheCommand(t *testing.T) {
	cases := map[string]string{
		"trace":  "mole trace <session-id>",
		"config": "mole config get <key>",
	}
	if _, err := exec(t, "trace"); err == nil || !strings.Contains(err.Error(), cases["trace"]) {
		t.Errorf("trace: err = %v, want %q", err, cases["trace"])
	}
	if _, err := exec(t, "config", "get"); err == nil || !strings.Contains(err.Error(), cases["config"]) {
		t.Errorf("config get: err = %v, want %q", err, cases["config"])
	}
	if _, err := exec(t, "research"); err == nil || !strings.Contains(err.Error(), "mole research") {
		t.Errorf("research: err = %v, want the usage line", err)
	}
}

// TestHelpListsEveryProductCommand so a new one cannot be added invisibly.
//
// The development commands are deliberately absent — see the two assertions
// below, which pin both halves of that split. Checking only "is it listed" would
// pass if `research` were hidden by accident.
func TestHelpListsEveryProductCommand(t *testing.T) {
	out, err := exec(t, "--help")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"research", "ask", "dataset", "connect", "crossings", "serve",
		"migrate", "config", "doctor", "sessions", "stats", "trace", "version",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("help does not list %q", want)
		}
	}
}

// TestTheDevCommandsAreHiddenButRunnable.
//
// Hidden, because a help screen listing eleven commands teaches nothing about
// which six matter, and `mole dev seed` against a real database is a bad first
// impression. Runnable, because they produce every measured number the README
// quotes and a reader checking a claim must be able to run them from a release
// binary.
func TestTheDevCommandsAreHiddenButRunnable(t *testing.T) {
	out, err := exec(t, "--help")
	if err != nil {
		t.Fatal(err)
	}
	for _, hidden := range []string{"eval", "corpus", "pairs", "dev"} {
		if strings.Contains(out, "\n  "+hidden+" ") {
			t.Errorf("%q is listed in the product help", hidden)
		}
		if _, err := exec(t, hidden, "--help"); err != nil {
			t.Errorf("%q is hidden AND unrunnable: %v", hidden, err)
		}
	}
}

// TestFlagHelpIsGeneratedNotWritten is the point of the port. A hand-maintained
// list drifts: the old one documented a flag ordering the parser rejected.
func TestFlagHelpIsGeneratedNotWritten(t *testing.T) {
	out, err := exec(t, "research", "--help")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"--usd", "--tokens", "--mode", "--max-sources", "--timeout", "--json", "--quiet", "--db"} {
		if !strings.Contains(out, want) {
			t.Errorf("generated help is missing %q", want)
		}
	}
	// Each flag should appear once. A hand-written list left beside the
	// generated one is the drift problem doubled, not solved.
	if n := strings.Count(out, "--max-sources"); n != 1 {
		t.Errorf("--max-sources appears %d times; a hand-written list is still present", n)
	}
}

// TestDBFlagHelpHidesTheResolvedPath: help output gets pasted into issues, and
// one user's home directory is noise to everyone else.
func TestDBFlagHelpHidesTheResolvedPath(t *testing.T) {
	out, err := exec(t, "sessions", "--help")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out, "/.local/share/mole") || strings.Contains(out, "/home/") {
		t.Errorf("help leaks a resolved home path:\n%s", out)
	}
	if !strings.Contains(out, "MOLE_DB") {
		t.Errorf("help does not say where the default comes from:\n%s", out)
	}
}

// TestVersionBothForms — `mole version` was documented and in use before the
// port, and cobra only provides the flag.
func TestVersionBothForms(t *testing.T) {
	for _, args := range [][]string{{"version"}, {"--version"}} {
		out, err := exec(t, args...)
		if err != nil {
			t.Fatalf("%v: %v", args, err)
		}
		if !strings.Contains(out, "mole ") {
			t.Errorf("%v printed %q", args, out)
		}
	}
}

// TestModeIsStillGated: report and dataset work, chain and ask do not.
//
// M9 implemented dataset, so this now asserts the gate on what is still missing
// rather than on dataset — the previous version would have gone green the moment
// the refusal message changed, whether or not the mode worked.
func TestModeIsStillGated(t *testing.T) {
	for _, mode := range []string{"chain", "ask"} {
		_, err := exec(t, "research", "a question", "--usd", "1.00", "--mode", mode)
		if err == nil || !strings.Contains(err.Error(), "not implemented") {
			t.Errorf("%s: err = %v, want the unimplemented-mode refusal", mode, err)
		}
	}
}

// TestADatasetSessionNeedsASchema. Refused rather than inferred: inference costs
// a model call, and a session that silently invented its own columns would
// produce a table nobody asked for and charge for it.
func TestADatasetSessionNeedsASchema(t *testing.T) {
	_, err := exec(t, "research", "a question", "--usd", "1.00", "--mode", "dataset")
	if err == nil {
		t.Fatal("a dataset session with no schema was accepted")
	}
	if !strings.Contains(err.Error(), "--schema") {
		t.Errorf("err = %v, want the schema requirement", err)
	}
}

// TestASchemaWithoutDatasetModeIsRefused, rather than silently ignored — a user
// who passed --schema expects columns, and a report would give them prose.
func TestASchemaWithoutDatasetModeIsRefused(t *testing.T) {
	_, err := exec(t, "research", "a question", "--usd", "1.00",
		"--schema", "company:text!")
	if err == nil || !strings.Contains(err.Error(), "--mode dataset") {
		t.Errorf("err = %v, want the mode requirement", err)
	}
}

// TestABadSchemaIsRefusedBeforeAnythingIsSpent.
func TestABadSchemaIsRefusedBeforeAnythingIsSpent(t *testing.T) {
	_, err := exec(t, "research", "a question", "--usd", "1.00",
		"--mode", "dataset", "--schema", "company:text")
	if err == nil {
		t.Fatal("a schema with no key field was accepted")
	}
	if !strings.Contains(err.Error(), "key") {
		t.Errorf("err = %v, want the missing-key refusal", err)
	}
}

// TestEvalRequiresASession rather than silently scoring an arbitrary one.
func TestEvalRequiresASession(t *testing.T) {
	_, err := exec(t, "eval")
	if err == nil || !strings.Contains(err.Error(), "mole eval") {
		t.Errorf("err = %v, want the usage line", err)
	}
}

// TestCLITestsCannotReachTheRealEnvironment guards the isolation above.
//
// The flag-parsing test drives the real `research` command. Without redirected
// paths it used the developer's live config and database, so once a reachable
// local model was configured it created sessions in that database and blocked on
// real model calls until the test timeout. The README's "nothing touches the
// network" stopped being true, and nothing noticed.
//
// A test that is safe only because a provider happens to be unreachable is not
// safe. This asserts the redirection itself.
func TestCLITestsCannotReachTheRealEnvironment(t *testing.T) {
	realDB := defaultDBPath()

	// exec sets both env vars; observe what a command actually resolves.
	out, err := exec(t, "sessions", "--help")
	if err != nil {
		t.Fatal(err)
	}
	_ = out

	if got := os.Getenv("MOLE_DB"); got == "" || got == realDB {
		t.Errorf("MOLE_DB = %q, which is not an isolated path (real: %q)", got, realDB)
	}
	if got := os.Getenv("MOLE_CONFIG_DIR"); got == "" {
		t.Error("MOLE_CONFIG_DIR was not redirected; tests would read the real config")
	}
	// And the isolated config must be empty, so no provider is reachable.
	if _, err := exec(t, "research", "a question", "--usd", "1.00"); err == nil ||
		!strings.Contains(err.Error(), "search provider") {
		t.Errorf("research got past the provider check with an isolated config: %v", err)
	}
}

// TestTraceSaysWhenNothingIsVerified is display honesty, not cosmetics.
//
// §11.3's derived confidence is 0 until the Verifier scores a claim, so every
// session before M4 — and every session where the Verifier was skipped — has
// thirteen claims all reading 0.00. A trace that showed the number without saying
// why invites exactly one conclusion, that the research found nothing worth
// trusting, when the truth is that nothing has been assessed.
func TestTraceSaysWhenNothingIsVerified(t *testing.T) {
	verified := time.Unix(1_700_000_000, 0).UTC()
	claims := []*core.Claim{
		{ID: "c_1", Text: "Claim one."},
		{ID: "c_2", Text: "Claim two."},
	}

	var buf bytes.Buffer
	printClaimGraph(&buf, claims, nil, false)
	if got := buf.String(); !strings.Contains(got, "none verified") {
		t.Errorf("unscored claims reported without explanation:\n%s", got)
	}

	// Partially scored says which, rather than implying the rest are worthless.
	claims[0].VerifiedAt = &verified
	buf.Reset()
	printClaimGraph(&buf, claims, nil, false)
	if got := buf.String(); !strings.Contains(got, "1 verified, 1 not") {
		t.Errorf("partial verification not distinguished:\n%s", got)
	}

	claims[1].VerifiedAt = &verified
	buf.Reset()
	printClaimGraph(&buf, claims, nil, false)
	if got := buf.String(); !strings.Contains(got, "all verified") {
		t.Errorf("fully verified session not reported as such:\n%s", got)
	}
}

// TestTraceNamesContradictions: a contradiction is the finding a reader most needs
// and the one an edge-count summary hides. "contradicts 2" says a disagreement
// exists somewhere; it does not say between what.
func TestTraceNamesContradictions(t *testing.T) {
	claims := []*core.Claim{
		{ID: "c_1", Text: "The effect is large and well established."},
		{ID: "c_2", Text: "No such effect was found in replication."},
	}
	edges := []*core.ClaimEdge{{
		FromID: "c_1", ToID: "c_2", Kind: core.EdgeContradicts,
		Rationale: "one reports an effect the other failed to replicate",
	}}

	var buf bytes.Buffer
	printClaimGraph(&buf, claims, edges, false)
	got := buf.String()

	for _, want := range []string{
		"contradictions:",
		"The effect is large",
		"No such effect was found",
		"failed to replicate",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("trace omits %q:\n%s", want, got)
		}
	}
}

// TestTheDeprecatedConfirmFlagDoesNotInventPrecedence.
//
// resolveConfirmMode compared against "both" to decide whether --confirm-edges
// had been passed, which it cannot do while "both" is also the flag's default.
// Both directions were wrong: two flags saying the identical thing were refused
// for disagreeing, and the deprecated alias silently overruled an explicit
// choice.
func TestTheDeprecatedConfirmFlagDoesNotInventPrecedence(t *testing.T) {
	for _, tc := range []struct {
		name      string
		mode      string
		noConfirm bool
		want      verifier.ConfirmMode
		wantErr   bool
	}{
		{"nothing passed", "", false, verifier.ConfirmBoth, false},
		{"alias alone", "", true, verifier.ConfirmNone, false},
		// The two say the same thing. Refusing this told the user they had
		// contradicted themselves when they had not.
		{"alias agrees with an explicit none", "none", true, verifier.ConfirmNone, false},
		// The two genuinely disagree, and the explicit flag is not overruled
		// quietly — the run stops instead.
		{"alias contradicts an explicit both", "both", true, "", true},
		{"alias contradicts duplicates", "duplicates", true, "", true},
		{"explicit both", "both", false, verifier.ConfirmBoth, false},
		{"explicit duplicates", "duplicates", false, verifier.ConfirmDuplicates, false},
		{"misspelt", "bath", false, "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := resolveConfirmMode(tc.mode, tc.noConfirm)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("resolveConfirmMode(%q, %v) = %q, want an error",
						tc.mode, tc.noConfirm, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("resolveConfirmMode(%q, %v): %v", tc.mode, tc.noConfirm, err)
			}
			if got != tc.want {
				t.Errorf("resolveConfirmMode(%q, %v) = %q, want %q",
					tc.mode, tc.noConfirm, got, tc.want)
			}
		})
	}
}

// TestTraceLabelsUncheckedContradictions checks trace shows every contradiction and labels the unchecked.
func TestTraceLabelsUncheckedContradictions(t *testing.T) {
	claims := []*core.Claim{
		{ID: "c_1", Text: "The effect is large."},
		{ID: "c_2", Text: "No such effect was found."},
		{ID: "c_3", Text: "The effect is small."},
	}
	edges := []*core.ClaimEdge{
		{FromID: "c_1", ToID: "c_2", Kind: core.EdgeContradicts, CreatedBy: "model"},
		{FromID: "c_1", ToID: "c_3", Kind: core.EdgeContradicts, CreatedBy: verifier.RecheckedBy + "strict"},
	}
	var on, off bytes.Buffer
	printClaimGraph(&on, claims, edges, true)
	printClaimGraph(&off, claims, edges, false)
	if got := strings.Count(on.String(), "(not re-checked)"); got != 1 {
		t.Errorf("%d labels with a re-check model, want 1:\n%s", got, on.String())
	}
	if !strings.Contains(on.String(), "No such effect") || !strings.Contains(on.String(), "The effect is small") {
		t.Errorf("trace hides a contradiction:\n%s", on.String())
	}
	if strings.Contains(off.String(), "re-checked") {
		t.Errorf("labelled without a re-check model:\n%s", off.String())
	}
}

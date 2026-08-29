package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

// Command tree.
//
// Cobra rather than a switch over os.Args, for one reason that is not style:
// help text is generated from the flag definitions instead of being written
// beside them. The hand-maintained version had already drifted — `mole research`
// documented a flag ordering the parser did not accept, and the docs were right
// while the code was wrong.
//
// It also removes the argument permutation this binary had to implement itself.
// pflag parses interspersed flags and positionals the way every other Unix tool
// does, so `mole research "a question" --usd 0.50` works without a shim.

// exitCoder lets a command choose its process exit status. Cobra has no notion
// of one, and `doctor` needs a non-zero exit that is not an execution failure:
// it ran correctly and found problems, which a script has to be able to tell
// apart from the binary crashing.
type exitCoder interface {
	error
	ExitCode() int
}

type exitError struct {
	err  error
	code int
}

func (e *exitError) Error() string { return e.err.Error() }
func (e *exitError) Unwrap() error { return e.err }
func (e *exitError) ExitCode() int { return e.code }
func exitWith(code int, err error) error {
	return &exitError{err: err, code: code}
}

func main() {
	// SIGINT and SIGTERM cancel the command's context. `research` settles
	// whatever it spent before returning, so Ctrl-C costs money that is
	// recorded rather than money that vanishes.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := newRootCmd().ExecuteContext(ctx); err != nil {
		var ec exitCoder
		if errors.As(err, &ec) {
			fmt.Fprintln(os.Stderr, "mole: "+err.Error())
			os.Exit(ec.ExitCode())
		}
		if errors.Is(err, pflag.ErrHelp) {
			os.Exit(2)
		}
		fmt.Fprintln(os.Stderr, "mole: "+err.Error())
		os.Exit(1)
	}
}

func newRootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:   "mole",
		Short: "deep research agent",
		Long: "mole — deep research agent\n\n" +
			"Through M1: `research` runs a single web lead end to end inside a real\n" +
			"reservation, and the rest operate on persisted state or configuration.\n" +
			"The planner that turns one question into many leads arrives in M3.",
		Version:      version,
		SilenceUsage: true,
		// "unknown command X" alone leaves a typo undiagnosed; cobra can offer
		// the near miss instead.
		SuggestionsMinimumDistance: 2,
		// Cobra prints the error itself by default, which would duplicate the
		// "mole: ..." line main() writes.
		SilenceErrors: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			// A bare `mole` prints help and succeeds, as it did before.
			return cmd.Help()
		},
	}

	root.SetVersionTemplate("mole {{.Version}}\n")

	// Describe where the default comes from rather than printing the resolved
	// path: help output gets pasted into issues, and one user's home directory
	// is noise to everyone else.
	root.PersistentFlags().String("db", defaultDBPath(), "database path ($MOLE_DB, else the XDG data dir)")
	// Blank the printed default so help does not carry one user's home
	// directory into every issue report and screenshot. The description above
	// already says where the value comes from.
	root.PersistentFlags().Lookup("db").DefValue = ""

	// `mole version` as well as `mole --version`. Cobra gives only the flag,
	// and the subcommand form was already documented and in use.
	root.AddCommand(&cobra.Command{
		Use:   "version",
		Short: "Print version",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			// Write through the command's writer, not os.Stdout, so the tree
			// can be exercised in a test the way a user drives it.
			fmt.Fprintln(cmd.OutOrStdout(), "mole "+moleVersion())
			return nil
		},
	})

	// The product surface. Development commands are added separately below, hidden
	// from `mole --help`, so a first-time reader sees the six commands that do the
	// work rather than the machinery that measures it.
	root.AddCommand(
		newResearchCmd(),
		newAskCmd(),
		newServeCmd(),

		newConnectCmd(),
		newDatasetCmd(),
		newCrossingsCmd(),
		newMigrateCmd(),
		newConfigCmd(),
		newDoctorCmd(),
		newSessionsCmd(),
		newStatsCmd(),
		newTraceCmd(),
		newExportCmd(),
	)

	// Development and evaluation commands.
	//
	// Hidden rather than removed or build-tagged. They are how this project is
	// measured — `eval` produces the scorecard the README quotes, `corpus` and
	// `pairs` produce the labelled numbers behind it — so a reader checking a claim
	// must be able to run them from a release binary. What they must not do is
	// greet a new user: `mole dev seed` against a real database is a bad first
	// impression, and a help screen listing eleven commands teaches nothing about
	// which six matter.
	for _, c := range []*cobra.Command{
		newEvalCmd(), newCorpusCmd(), newPairsCmd(), newDevCmd(),
	} {
		c.Hidden = true
		root.AddCommand(c)
	}
	return root
}

// exactArgs is cobra.ExactArgs with a message that says what to type.
//
// The stock text is "accepts 1 arg(s), received 0", which tells a user the
// arity they already know and not the command they wanted.
func exactArgs(n int, usage string) cobra.PositionalArgs {
	return func(cmd *cobra.Command, args []string) error {
		if len(args) != n {
			return fmt.Errorf("usage: %s", usage)
		}
		return nil
	}
}

// minArgs is cobra.MinimumNArgs with the same fix.
func minArgs(n int, usage string) cobra.PositionalArgs {
	return func(cmd *cobra.Command, args []string) error {
		if len(args) < n {
			return fmt.Errorf("usage: %s", usage)
		}
		return nil
	}
}

// dbPath reads the inherited --db flag.
func dbPath(cmd *cobra.Command) string {
	v, _ := cmd.Flags().GetString("db")
	return v
}

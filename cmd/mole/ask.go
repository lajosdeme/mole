package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"

	"github.com/lajosdeme/mole/internal/config"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/mcpserver"
	"github.com/lajosdeme/mole/internal/record"
	"github.com/spf13/cobra"
)

// mole ask
//
// research.ask (§13) from a terminal. The same operation an agent gets over
// MCP: read a finished session's claim graph, retrieve the claims relevant to a
// new question, and answer from them with citations. No search, no fetch, no new
// claims — the evidence was gathered and paid for by the original run, and this
// is what makes that run worth keeping.
//
// It shares mcpserver.Deps.Ask rather than reimplementing it. The budgeting
// there is subtle enough (an ask cannot be charged to a settled session, so it
// opens a tiny one of its own) that a second copy would be a second set of §8
// bugs.
//
// It writes, and it does so while a daemon may be running. That is safe, and
// openDBNoMigrate explains exactly why: SQLite serialises writers per
// transaction, not per connection, and an ask's transactions are short. What it
// must not do — and does not — is migrate.
func newAskCmd() *cobra.Command {
	var asJSON bool

	c := &cobra.Command{
		Use:   "ask <session-id> <question>",
		Short: "Ask a follow-up against a finished session's evidence",
		Long: "Answers a new question from research that already happened. Retrieval and\n" +
			"one model call over stored claims — nothing is searched or fetched, so it\n" +
			"costs cents and takes seconds.\n\n" +
			"The answer is charged to its own small session, not to the one being\n" +
			"queried: that session's ledger is settled, and §8 does not permit writing\n" +
			"cost rows against a closed account. The ask appears in `mole sessions` with\n" +
			"its own cost.\n\n" +
			"With no LLM configured it degrades rather than fails, listing the relevant\n" +
			"claims and their sources without synthesized prose.",
		Args: exactArgs(2, `mole ask <session-id> "<question>"`),
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmdAsk(cmd.Context(), cmd.OutOrStdout(), askOpts{
				sessionID: args[0],
				question:  args[1],
				dbPath:    dbPath(cmd),
				asJSON:    asJSON,
			})
		},
	}
	c.Flags().BoolVar(&asJSON, "json", false, "emit the answer, claims and citations as JSON")
	return c
}

type askOpts struct {
	sessionID string
	question  string
	dbPath    string
	asJSON    bool
}

func cmdAsk(ctx context.Context, w io.Writer, o askOpts) error {
	cfg, err := config.Load()
	if err != nil && !errors.Is(err, config.ErrNotConfigured) {
		return err
	}

	// Database first. Detecting a model costs a three-second probe, and a missing
	// database or a schema this binary cannot read should be reported before
	// paying it.
	db, err := openDBNoMigrate(ctx, o.dbPath)
	if err != nil {
		return err
	}
	defer func() { _ = db.Close() }()

	// No recorder: an ask makes one model call and there is no cassette worth
	// keeping for it. rec.Client() is nil-safe, which is what the daemon relies
	// on too.
	var rec *record.Recorder
	model, _, err := buildLLMWithClient(cfg, rec.Client())
	if err != nil {
		// Not fatal here, unlike `research`. An ask over claims already gathered
		// is still worth something without a model — it lists the relevant ones
		// with their sources — and refusing outright would make a misconfigured
		// provider block access to research already paid for. Said out loud on
		// stderr so the missing prose is not a mystery.
		fmt.Fprintf(os.Stderr, "mole: answering without synthesis — %v\n", err)
		model = nil
	}

	deps := mcpserver.Deps{
		Store:         db,
		LLM:           model,
		Pricing:       pricingFor(cfg),
		MaxSessionUSD: cfg.MaxSessionUSD,
		Log:           slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn})),
	}

	// Unlike `research`, an unpriced model is not grounds to refuse. The spend is
	// bounded by MaxToolCalls: 1 regardless of what the ledger can count, so the
	// ceiling not binding costs nothing — but the "spent" line below would read
	// as $0.0000 and mean "not measured", which is worth saying out loud.
	unpriced := ""
	if model != nil {
		if missing := unpricedModelList(cfg, model); len(missing) > 0 {
			unpriced = missing[0]
		}
	}

	out, err := deps.Ask(ctx, mcpserver.AskIn{SessionID: o.sessionID, Question: o.question})
	if err != nil {
		return err
	}

	if o.asJSON {
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	}
	printAsk(w, out, unpriced)
	return nil
}

func printAsk(w io.Writer, out mcpserver.AskOut, unpriced string) {
	fmt.Fprintln(w)
	fmt.Fprintln(w, out.Answer)

	if len(out.Citations) > 0 {
		fmt.Fprintln(w, "\nsources")
		for _, c := range out.Citations {
			fmt.Fprintf(w, "  [%d] %s\n", c.N, c.Source)
			if c.PublishedAt != "" {
				fmt.Fprintf(w, "      published %s\n", c.PublishedAt)
			}
			for _, q := range c.Quotes {
				fmt.Fprintf(w, "      %q\n", q)
			}
		}
	}

	fmt.Fprintln(w)
	if out.Degraded != "" {
		fmt.Fprintf(w, "  not synthesized: %s\n", out.Degraded)
	}
	// Claim count, not "claims used": these are the ones retrieval selected and
	// the answer was written from, which is the number a reader needs to judge
	// how much of the session the answer rests on.
	fmt.Fprintf(w, "  %s · %d claim(s)", core.FormatUSD(out.Spent), len(out.Claims))
	if out.AskSessionID != "" {
		fmt.Fprintf(w, " · charged to %s", out.AskSessionID)
	}
	fmt.Fprintln(w)
	if unpriced != "" {
		fmt.Fprintf(w, "  cost is not measured — no price is known for %s\n", unpriced)
	}
}

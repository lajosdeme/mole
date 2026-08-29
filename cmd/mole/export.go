package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/eval"
	"github.com/lajosdeme/mole/internal/export"
	"github.com/lajosdeme/mole/internal/record"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/tools/extract"
	"github.com/lajosdeme/mole/internal/tools/fetch"
	"github.com/spf13/cobra"
)

// mole export
//
// One finished session as one JSON document, for anything that renders or audits
// a run without wanting to know mole's schema. Everything it carries was already
// reachable — as five commands printing five formats, which made a consumer shell
// out repeatedly and parse text tables.

const exportUsage = `Write one session as a single JSON document.

Everything a consumer needs to render or audit the session: the sub-questions it
asked, the report, the disagreements with both sides inlined, every claim and its
quote, what it could not read, and what it spent.

Derived from stored state, so exporting is free and repeatable — no model calls,
and two exports of one session agree.

--verify adds the receipt: every cited source is re-read and checked for the quote
it was supposed to contain. That costs one fetch per source, which is why it is
off by default. Without it the receipt is null rather than zero, so a consumer can
tell "not checked" from "checked and found nothing".

The envelope carries a schema_version. Fields are added within a version; renaming
or removing one is a breaking change.
`

type exportOpts struct {
	verify bool
	out    string
	dbPath string
}

func newExportCmd() *cobra.Command {
	var o exportOpts

	c := &cobra.Command{
		Use:   "export <session-id>",
		Short: "Write one session as a single JSON document",
		Long:  exportUsage,
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			o.dbPath = dbPath(cmd)
			return cmdExport(cmd.Context(), args[0], o)
		},
	}
	f := c.Flags()
	f.BoolVar(&o.verify, "verify", false,
		"re-read every cited source and check the quote is still in it (one fetch per source)")
	f.StringVarP(&o.out, "out", "o", "", "write here instead of stdout")
	return c
}

func cmdExport(ctx context.Context, sessionID string, o exportOpts) error {
	db, err := openDBNoMigrate(ctx, o.dbPath)
	if err != nil {
		return err
	}
	defer db.Close()

	opts := export.Options{MoleVersion: moleVersion()}
	if o.verify {
		receipt, err := verifyReceipt(ctx, db, sessionID)
		if err != nil {
			return err
		}
		opts.Receipt = receipt
	}

	ex, err := export.Build(ctx, db, sessionID, opts)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return fmt.Errorf("no session %s", sessionID)
		}
		return err
	}

	b, err := json.MarshalIndent(ex, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')

	if o.out == "" {
		_, err = os.Stdout.Write(b)
		return err
	}
	if err := os.WriteFile(o.out, b, 0o600); err != nil {
		return err
	}
	// 0600, not 0644: a session's claims carry whatever the question was about,
	// and an export is the whole of it in one file.
	fmt.Fprintf(os.Stderr, "%d claim(s), %d conflict(s) written to %s\n",
		len(ex.Evidence), len(ex.Conflicts), o.out)
	return nil
}

// verifyReceipt re-reads every cited source.
//
// Stored text first, network second — the same order `mole eval --citations` uses,
// and for the same reason: a toolkit session holds the exact bytes each quote was
// checked against, and re-fetching them would measure the page's stability rather
// than the citation.
func verifyReceipt(ctx context.Context, db store.Store, sessionID string) (*export.Receipt, error) {
	var (
		sess     *core.Session
		claims   []*core.Claim
		outcomes []*store.FetchOutcome
	)
	if err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		if sess, err = q.GetSession(ctx, sessionID); err != nil {
			return err
		}
		if claims, err = q.ListClaims(ctx, sessionID, 0); err != nil {
			return err
		}
		outcomes, err = q.ListFetchOutcomes(ctx, sessionID, 0)
		return err
	}); err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return nil, fmt.Errorf("no session %s", sessionID)
		}
		return nil, err
	}

	// Named from the session's prompt, which is what `mole research` recorded
	// under — a separate export-scoped cassette would miss every interaction and
	// report the whole run unreachable.
	rec, err := record.FromEnv(sess.Prompt)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rec.Close() }()

	network := eval.NewPipelineReader(
		fetch.NewHTTP(fetch.Config{UserAgent: userAgent()}, fetch.Options{
			Log:       slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError})),
			Transport: rec.Wrap,
		}),
		extract.New(),
	)
	rep := eval.VerifyCitations(ctx, claims,
		fetch.ProviderSupplied(outcomes), eval.NewStoredReader(db, sessionID, network))

	return &export.Receipt{
		Verified: rep.Verified, Mismatch: rep.Mismatch, OffsetDrift: rep.OffsetDrift,
		Unreachable: rep.Unreachable, Skipped: rep.Skipped,
	}, nil
}

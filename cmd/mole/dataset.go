package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/lajosdeme/mole/internal/dataset"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/spf13/cobra"
)

// `mole dataset` writes out a dataset session's rows (M9, §13).
//
// Assembled on read rather than stored assembled. The rows and the schema are
// what the session persisted; the merge is deterministic arithmetic over them, so
// re-running it costs nothing and means a changed threshold or a fixed
// normalisation rule improves every dataset already collected rather than only
// the next one.

type datasetOpts struct {
	format     string
	provenance bool
	out        string
	threshold  float64
	dbPath     string
}

func newDatasetCmd() *cobra.Command {
	var o datasetOpts

	c := &cobra.Command{
		Use:   "dataset <session-id>",
		Short: "Write a dataset session's rows as CSV or JSON",
		Long: "Assembles the rows a dataset session extracted into CSV or JSON.\n\n" +
			"The merge runs on read, not at collection time, so a later fix to the\n" +
			"matching rules improves datasets already gathered.\n\n" +
			"CSV holds one value per cell and is therefore lossy: it carries a source\n" +
			"count and a `contested` column naming the fields the sources disagree\n" +
			"about. JSON carries every disagreeing value, every source and every\n" +
			"quote — use it when a disagreement matters.",
		Args: exactArgs(1, "mole dataset <session-id>"),
		RunE: func(cmd *cobra.Command, args []string) error {
			o.dbPath = dbPath(cmd)
			return cmdDataset(cmd.Context(), cmd, args[0], o)
		},
	}
	f := c.Flags()
	f.StringVar(&o.format, "format", "csv", "csv or json")
	f.BoolVar(&o.provenance, "provenance", false,
		"add source URLs and the supporting quote to the CSV")
	f.StringVar(&o.out, "out", "", "write to this file instead of stdout")
	f.Float64Var(&o.threshold, "threshold", 0,
		"similarity a pair of key values must reach to merge (default "+
			fmt.Sprint(dataset.DefaultThreshold)+")")
	return c
}

func cmdDataset(ctx context.Context, cmd *cobra.Command, sessionID string, o datasetOpts) error {
	switch o.format {
	case "csv", "json":
	default:
		return fmt.Errorf("unknown format %q (want csv or json)", o.format)
	}

	db, err := openDBNoMigrate(ctx, o.dbPath)
	if err != nil {
		return err
	}
	defer db.Close()

	d, err := store.LoadDataset(ctx, db, sessionID, dataset.Options{Threshold: o.threshold})
	if errors.Is(err, store.ErrNotDataset) {
		// A report session has no schema, and saying so beats an empty file: the
		// user asked the wrong command about the right session.
		return fmt.Errorf("session %s is not a dataset session "+
			"(run with --mode dataset --schema ...)", sessionID)
	}
	if err != nil {
		return err
	}

	w := cmd.OutOrStdout()
	var file *os.File
	if o.out != "" {
		f, err := os.Create(o.out)
		if err != nil {
			return err
		}
		defer f.Close()
		file, w = f, f
	}

	if o.format == "json" {
		if err := d.WriteJSON(w); err != nil {
			return err
		}
	} else if err := d.WriteCSV(w, o.provenance); err != nil {
		return err
	}
	// Closed explicitly as well as deferred, and only for a file this command
	// opened: a deferred Close's error is discarded, and a failed flush would
	// leave a truncated file that looks complete. Type-asserting on w instead
	// would close the process's stdout, which is a *os.File too.
	if file != nil {
		if err := file.Close(); err != nil {
			return fmt.Errorf("write %s: %w", o.out, err)
		}
	}

	// The summary goes to stderr so a piped `> out.csv` gets only the data. What
	// qualifies a dataset — how much was corroborated, how much is contested — is
	// the part a reader most needs and the part a redirect would silently discard.
	//
	// Unconditional: stderr either way, so a redirect cannot discard it.
	fmt.Fprintln(cmd.ErrOrStderr(), strings.TrimSpace(d.Summary()))
	return nil
}

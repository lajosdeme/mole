package main

import (
	"fmt"
	"sort"
	"strings"

	"errors"
	"github.com/lajosdeme/mole/internal/config"
	"github.com/lajosdeme/mole/internal/eval"
	"github.com/lajosdeme/mole/internal/verifier"
	"github.com/spf13/cobra"
	"log/slog"
	"os"
)

// Adjudicator test sets (§11.2).
//
// Everything else mole measures about the claim graph counts what was FOUND. Whether it was
// RIGHT needs someone to say what the right answer was, and this is the smallest possible
// way to ask: dump the pairs a real run judged, with both claim texts and the model's own
// rationale, and let a person label them.
//
// Cheaper than §14.2's question corpus and sharper for this purpose. The pairs already
// exist, retrieval is deterministic so they regenerate exactly, and once labelled they
// score any prompt or model change in seconds with no tokens and no network.

func newPairsCmd() *cobra.Command {
	c := &cobra.Command{
		Use:   "pairs",
		Short: "Dump and score adjudicator judgements (§11.2)",
		Long: "Measures whether the claim-graph adjudicator is RIGHT, which no other metric\n" +
			"does — they all count what it found.\n\n" +
			"  mole pairs dump  <session> -o pairs.json      # then fill in each \"label\"\n" +
			"  mole pairs score pairs.json\n\n" +
			"To compare a different model on the SAME pairs — no research run, and the\n" +
			"stored graph is left alone:\n\n" +
			"  mole pairs judge <session> --model gemma4:12b --labels pairs.json -o gemma.json\n" +
			"  mole pairs score gemma.json",
	}
	c.AddCommand(newPairsDumpCmd(), newPairsJudgeCmd(), newPairsCompareCmd(), newPairsScoreCmd())
	return c
}

func newPairsDumpCmd() *cobra.Command {
	var (
		out        string
		kinds      []string
		all        bool
		cap        int
		exhaustive bool
		maxPairs   int
	)
	c := &cobra.Command{
		Use:   "dump <session-id>",
		Short: "Write a session's judged pairs for labelling",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			db, err := openDBNoMigrate(ctx, dbPath(cmd))
			if err != nil {
				return err
			}
			defer db.Close()

			ps, err := eval.DumpPairs(ctx, db, args[0], eval.DumpOptions{
				Kinds: kinds, All: all, MaxCandidatesPerClaim: cap,
				Exhaustive: exhaustive, MaxPairs: maxPairs,
			})
			if err != nil {
				return err
			}
			if len(ps.Pairs) == 0 {
				return fmt.Errorf("no pairs matched (session %s)", args[0])
			}

			if out == "" {
				return eval.WritePairs("/dev/stdout", ps)
			}
			if err := eval.WritePairs(out, ps); err != nil {
				return err
			}
			byKind := map[string]int{}
			for _, p := range ps.Pairs {
				byKind[p.Model]++
			}
			fmt.Printf("%d pair(s) written to %s\n", len(ps.Pairs), out)
			for _, k := range sortedKeys(byKind) {
				fmt.Printf("  %-14s %d\n", k, byKind[k])
			}
			fmt.Printf("\nFill in \"label\" on each: %s\n", strings.Join(relationNames(), ", "))
			fmt.Println("Leave it empty to skip a pair; scoring counts those rather than guessing.")
			return nil
		},
	}
	f := c.Flags()
	f.StringVarP(&out, "out", "o", "", "write here instead of stdout")
	f.StringSliceVar(&kinds, "kind", nil,
		"only these verdicts, e.g. --kind contradicts (default: every pair with an edge)")
	f.BoolVar(&all, "all", false,
		"include pairs the model called unrelated; needed to measure RECALL, and there are many more of them")
	f.IntVar(&cap, "max-candidates", 0, "retrieval cap; 0 uses the Verifier's default")
	f.BoolVar(&exhaustive, "exhaustive", false,
		"pair EVERY claim with every other, retrieval bypassed, recording each pair's "+
			"lexical score and rank. --all measures the judge's recall; this measures "+
			"the pipeline's, by including the pairs retrieval refused to form")
	f.IntVar(&maxPairs, "max-pairs", 0,
		"refuse an exhaustive dump larger than this; 0 uses the default")
	return c
}

func newPairsScoreCmd() *cobra.Command {
	c := &cobra.Command{
		Use:   "score <file.json>",
		Short: "Score a labelled pair set",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ps, err := eval.LoadPairs(args[0])
			if err != nil {
				return err
			}
			s := eval.ScorePairs(ps)
			printPairScore(s)
			if s.Labelled == 0 {
				return fmt.Errorf("nothing labelled in %s", args[0])
			}
			return nil
		},
	}
	return c
}

func printPairScore(s eval.PairScore) {
	fmt.Printf("adjudicator accuracy: %.0f%%  (%d of %d labelled pair(s))\n",
		100*s.Accuracy(), s.Correct, s.Labelled)
	fmt.Printf("  by effect on the graph: %.0f%%  (%d of %d)",
		100*s.EffectAccuracy(), s.CorrectEffect, s.Labelled)
	if inert := s.CorrectEffect - s.Correct; inert > 0 {
		fmt.Printf("  — %d error(s) build an edge nothing reads", inert)
	}
	fmt.Println()
	if s.Unlabelled > 0 {
		// Counted, never guessed at: a score over whichever pairs someone got round to
		// labelling, reported as the score, is how a test set starts lying.
		fmt.Printf("  %d pair(s) unlabelled and excluded\n", s.Unlabelled)
	}
	if s.Skipped > 0 {
		fmt.Printf("  %d pair(s) were never put to the model and say nothing about it\n", s.Skipped)
	}
	if s.Labelled == 0 {
		return
	}

	w := newTabWriter()
	fmt.Fprintln(w, "\nRELATION\tPRECISION\tRECALL")
	for _, k := range relationNames() {
		p, hasP := s.Precision[k]
		r, hasR := s.Recall[k]
		if !hasP && !hasR {
			continue
		}
		fmt.Fprintf(w, "%s\t%s\t%s\n", k, pctOrDash(p, hasP), pctOrDash(r, hasR))
	}
	_ = w.Flush()

	// The confusion, but only where the model was wrong: what it says instead is the
	// thing that tells you whether to change the prompt or the model.
	var lines []string
	for model, truths := range s.Confusion {
		for truth, n := range truths {
			if model == truth {
				continue
			}
			note := ""
			if verifier.Relation(model).EffectOf() != verifier.Relation(truth).EffectOf() {
				note = "  <-- changes the graph"
			}
			lines = append(lines, fmt.Sprintf("  said %-13s was really %-13s ×%d%s", model, truth, n, note))
		}
	}
	if len(lines) > 0 {
		sort.Strings(lines)
		fmt.Println("\nwhere it went wrong:")
		for _, l := range lines {
			fmt.Println(l)
		}
	}
}

func pctOrDash(v float64, ok bool) string {
	if !ok {
		return "—"
	}
	return fmt.Sprintf("%.0f%%", 100*v)
}

// relationNames is what a person labelling a pair set may write. The three live
// relations only: the retired ones still parse, but offering them here would invite
// new labels in a vocabulary the judge is no longer asked to produce, which scores as
// a mistake the judge could not have avoided.
func relationNames() []string {
	return []string{
		string(verifier.RelContradicts), string(verifier.RelDuplicate),
		string(verifier.RelNeither),
	}
}

func sortedKeys(m map[string]int) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func newPairsJudgeCmd() *cobra.Command {
	var (
		model  string
		out    string
		labels string
		kinds  []string
		batch  int
	)
	c := &cobra.Command{
		Use:   "judge <session-id>",
		Short: "Re-judge a session's pairs with a different model, without touching the graph",
		Long: "Runs the adjudication prompt over the pairs a session already judged, with a\n" +
			"model of your choosing, and writes the verdicts alongside the stored ones.\n\n" +
			"No research run: the claims and pairs already exist, so comparing two judges is\n" +
			"a handful of model calls rather than an hour. Nothing is written to the session —\n" +
			"the stored graph is what is being compared against.\n\n" +
			"The cost is the evaluator's and is deliberately not charged to the session: doing\n" +
			"so would inflate the spend and corrupt cost-per-claim for the very session being\n" +
			"examined.",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			if model == "" {
				return errors.New("--model is required: the point is to compare a different one")
			}

			cfg, err := config.Load()
			if err != nil && !errors.Is(err, config.ErrNotConfigured) {
				return err
			}
			provider, _, err := buildLLM(cfg)
			if err != nil {
				return err
			}
			if provider == nil {
				return errors.New("no model provider configured")
			}

			opts := eval.JudgeOptions{Model: model, BatchSize: batch, Kinds: kinds}
			if labels != "" {
				prior, err := eval.LoadPairs(labels)
				if err != nil {
					return fmt.Errorf("read labels: %w", err)
				}
				opts.Labels = eval.LabelsFrom(prior)
				fmt.Printf("carried %d label(s) forward from %s\n", len(opts.Labels), labels)
			}

			db, err := openDBNoMigrate(ctx, dbPath(cmd))
			if err != nil {
				return err
			}
			defer db.Close()

			ps, err := eval.JudgeSession(ctx, db, provider, args[0], opts,
				slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn})))
			if err != nil {
				return err
			}

			changed, unjudged := 0, 0
			for _, p := range ps.Pairs {
				if !p.Judged {
					unjudged++
					continue
				}
				if p.Model != p.Baseline {
					changed++
				}
			}
			fmt.Printf("%d pair(s) re-judged by %s: %d verdict(s) differ from the stored graph",
				len(ps.Pairs), model, changed)
			if unjudged > 0 {
				fmt.Printf(", %d unanswered", unjudged)
			}
			fmt.Println()

			if out == "" {
				return eval.WritePairs("/dev/stdout", ps)
			}
			if err := eval.WritePairs(out, ps); err != nil {
				return err
			}
			fmt.Printf("written to %s\n", out)
			if len(opts.Labels) == 0 {
				fmt.Println("Fill in \"label\" on each, then: mole pairs score " + out)
			}
			return nil
		},
	}
	f := c.Flags()
	f.StringVar(&model, "model", "", "model to judge with, e.g. gemma4:12b")
	f.StringVarP(&out, "out", "o", "", "write here instead of stdout")
	f.StringVar(&labels, "labels", "",
		"carry labels forward from an earlier set, so the work is done once")
	f.StringSliceVar(&kinds, "kind", nil, "only re-judge these stored verdicts, e.g. --kind contradicts")
	f.IntVar(&batch, "batch", 0, "pairs per call; 0 uses the default, lower it if the model times out")
	return c
}

func newPairsCompareCmd() *cobra.Command {
	c := &cobra.Command{
		Use:   "compare <a.json> <b.json>",
		Short: "How often two verdict sets agree — no labels needed",
		Long: "The cheapest useful measurement of a judge, and the only one needing no labels:\n" +
			"run the same model over the same pairs twice and see how often it agrees with\n" +
			"itself.\n\n" +
			"Self-consistency is a ceiling, not a score — a judge cannot be more accurate than\n" +
			"it is reproducible — so screening on it is far cheaper than labelling. It is also\n" +
			"how two different models are compared without deciding which is right.",
		Args: cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			a, err := eval.LoadPairs(args[0])
			if err != nil {
				return err
			}
			b, err := eval.LoadPairs(args[1])
			if err != nil {
				return err
			}
			ag := eval.CompareVerdicts(a, b)
			if ag.Compared == 0 {
				return fmt.Errorf("no pair was judged by both sets (%d unanswered)", ag.Unanswered)
			}

			fmt.Printf("agreement: %.0f%%  (%d of %d pair(s) judged by both)\n",
				100*ag.Rate(), ag.Same, ag.Compared)
			// The number to act on. Reported second because the raw rate is what a
			// reader expects to see, and printed always — including when the two are
			// equal, since "no inert disagreements" is itself the finding.
			fmt.Printf("  by effect on the graph: %.0f%%  (%d of %d)",
				100*ag.EffectRate(), ag.SameEffect, ag.Compared)
			if inert := ag.SameEffect - ag.Same; inert > 0 {
				fmt.Printf("  — %d disagreement(s) change nothing derived", inert)
			}
			fmt.Println()
			if ag.Unanswered > 0 {
				fmt.Printf("  %d pair(s) one side did not judge, excluded\n", ag.Unanswered)
			}
			if a.Model != "" || b.Model != "" {
				fmt.Printf("  %s vs %s\n", orStored(a.Model), orStored(b.Model))
			}

			var lines []string
			for x, ys := range ag.Confusion {
				for y, n := range ys {
					if x == y {
						continue
					}
					// Marked, so the one row worth reading does not sit unremarked
					// among five that do not matter.
					note := ""
					if verifier.Relation(x).EffectOf() != verifier.Relation(y).EffectOf() {
						note = "  <-- changes the graph"
					}
					lines = append(lines, fmt.Sprintf("  %-13s vs %-13s ×%d%s", x, y, n, note))
				}
			}
			if len(lines) > 0 {
				sort.Strings(lines)
				fmt.Println("\nwhere they differ:")
				for _, l := range lines {
					fmt.Println(l)
				}
			}
			return nil
		},
	}
	return c
}

func orStored(m string) string {
	if m == "" {
		return "the stored graph"
	}
	return m
}

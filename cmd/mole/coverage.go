package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/lajosdeme/mole/internal/config"
	"github.com/lajosdeme/mole/internal/eval"
	"github.com/lajosdeme/mole/internal/tools/academic"
	"github.com/lajosdeme/mole/internal/tools/limiter"
	"github.com/spf13/cobra"
)

// mole dev academic-coverage
//
// How much of the literature mole would cite is behind a PDF it cannot read.
// The decision gate for building tier 2 of the escalation ladder: metadata APIs
// answer it now, where waiting for `unsupported_type` to accumulate from real
// runs would answer it slowly.
//
// Three buckets, only one of which a PDF extractor serves:
//
//	html      readable today, through the existing fetcher and extractor
//	pdf_only  open access mole cannot read — the bucket a parser serves
//	closed    no legal copy at all; parsing does not help
//
// Two skews are reported rather than smoothed away. arXiv has only served HTML
// since late 2023, so back-catalogue coverage understates what current research
// looks like — hence the split by year. And every provider failure is counted,
// so a flaky API shrinks the confidence in the number rather than silently
// shrinking its denominator.

type coverageOpts struct {
	perQuestion int
	probeHTML   bool
	asJSON      bool
	timeout     time.Duration

	// agent identifies this command's own requests. Derived, not a flag.
	agent string
}

// paperKey identifies a paper across providers and questions. DOI first, for the
// reason internal/actors uses the same order.
func paperKey(p academic.Paper) string {
	switch {
	case strings.TrimSpace(p.DOI) != "":
		return "doi:" + strings.ToLower(strings.TrimSpace(p.DOI))
	case strings.TrimSpace(p.ArXivID) != "":
		return "arxiv:" + strings.TrimSpace(p.ArXivID)
	case strings.TrimSpace(p.PMID) != "":
		return "pmid:" + strings.TrimSpace(p.PMID)
	default:
		return strings.ToLower(strings.Join(strings.Fields(p.Title), " "))
	}
}

func newCoverageCmd() *cobra.Command {
	var o coverageOpts
	c := &cobra.Command{
		Use:   "academic-coverage <corpus.json>",
		Short: "Measure how much of a corpus's literature is readable, PDF-only, or closed",
		Long: "Searches arXiv and PubMed for every question in a corpus, resolves each DOI\n" +
			"through Unpaywall, and reports what fraction of the papers mole could\n" +
			"actually read.\n\n" +
			"No document is downloaded and no model is called — this is metadata only,\n" +
			"which is what makes it cheap enough to run before deciding whether to build\n" +
			"a PDF extractor. Every provider is rate-limited at its published rate, so\n" +
			"a ten-question corpus takes a few minutes rather than seconds.",
		Args: exactArgs(1, "mole dev academic-coverage <corpus.json>"),
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmdCoverage(cmd.Context(), args[0], o)
		},
	}
	f := c.Flags()
	f.IntVar(&o.perQuestion, "per-question", 10, "papers to take from each provider per question")
	f.BoolVar(&o.probeHTML, "probe-html", true,
		"HEAD arXiv HTML URLs; without this arXiv papers cannot be distinguished from pdf_only")
	f.BoolVar(&o.asJSON, "json", false, "emit the report as JSON")
	f.DurationVar(&o.timeout, "timeout", 30*time.Minute, "wall-clock ceiling for the whole sweep")
	return c
}

// coverageRow is one paper's verdict.
type coverageRow struct {
	Source   string `json:"source"`
	Title    string `json:"title"`
	DOI      string `json:"doi,omitempty"`
	Year     int    `json:"year,omitempty"`
	Format   string `json:"format"`
	Resolved bool   `json:"resolved"`
	Note     string `json:"note,omitempty"`
}

type coverageReport struct {
	Corpus    string                    `json:"corpus"`
	Questions int                       `json:"questions"`
	Papers    int                       `json:"papers"`
	ByFormat  map[string]int            `json:"by_format"`
	BySource  map[string]map[string]int `json:"by_source"`
	ByEra     map[string]map[string]int `json:"by_era"`

	// Failures are counted, never dropped. A provider erroring is not evidence
	// that a paper is unreadable, and folding the two together would let a flaky
	// API quietly move the number this command exists to produce.
	SearchErrors  int `json:"search_errors"`
	ResolveErrors int `json:"resolve_errors"`
	ProbeErrors   int `json:"probe_errors"`
	// Duplicates counts papers seen more than once across providers and
	// questions, skipped rather than counted twice.
	Duplicates int `json:"duplicates"`

	// Incomplete is set when the sweep did not finish. A truncated run must not
	// read as a complete one — this command is a decision gate.
	Incomplete string `json:"incomplete,omitempty"`

	Rows []coverageRow `json:"rows,omitempty"`
}

func cmdCoverage(ctx context.Context, corpusPath string, o coverageOpts) error {
	corpus, err := eval.LoadCorpus(corpusPath)
	if err != nil {
		return err
	}
	cfg, err := config.Load()
	if err != nil && !errors.Is(err, config.ErrNotConfigured) {
		return err
	}
	if err := academic.CheckContact(cfg.ContactEmail); err != nil {
		return fmt.Errorf("%w\nset one with: mole config set contact-email you@example.com", err)
	}

	ctx, cancel := context.WithTimeout(ctx, o.timeout)
	defer cancel()

	lim := limiter.New(limiter.Unlimited)
	academic.Register(lim)
	// arxiv.org is the WEB host, not the API host, so it gets its own bucket
	// rather than the API's three-second one. Still deliberately gentle: these
	// are HEAD requests against a service run by a library.
	lim.Set("arxiv.org", limiter.Limit{Rate: 1, Burst: 1})

	acfg := academic.Config{ContactEmail: cfg.ContactEmail}
	arxiv, err := academic.NewArXiv(acfg, nil, lim)
	if err != nil {
		return err
	}
	pubmed, err := academic.NewPubMed(acfg, nil, lim)
	if err != nil {
		return err
	}
	unpaywall, err := academic.NewUnpaywall(acfg, nil, lim)
	if err != nil {
		return err
	}

	rep := &coverageReport{
		Corpus:    corpus.Name,
		Questions: len(corpus.Questions),
		ByFormat:  map[string]int{},
		BySource:  map[string]map[string]int{},
		ByEra:     map[string]map[string]int{},
	}
	client := &http.Client{Timeout: 20 * time.Second}
	o.agent = fmt.Sprintf("mole (+contact: %s)", strings.TrimSpace(cfg.ContactEmail))
	seen := map[string]bool{}

	for i, q := range corpus.Questions {
		// Stop rather than grinding out context errors for every remaining
		// question. Without this the deadline firing at question 3 of 50 left
		// each of the other 47 counted as a search failure and continued, and
		// the run still ended with "Only pdf_only is a PDF extractor's job: X of
		// Y" and exit status 0 — a truncated sweep indistinguishable from a
		// complete one, for the command that is meant to be a decision gate.
		if err := ctx.Err(); err != nil {
			rep.Incomplete = fmt.Sprintf("stopped after %d of %d question(s): %v",
				i, len(corpus.Questions), err)
			break
		}
		if !o.asJSON {
			fmt.Printf("[%d/%d] %s\n", i+1, len(corpus.Questions), q.Question)
		}
		for _, prov := range []academic.Provider{arxiv, pubmed} {
			res, err := prov.Search(ctx, q.Question, academic.Options{MaxResults: o.perQuestion})
			if err != nil {
				rep.SearchErrors++
				if !o.asJSON {
					fmt.Printf("    %s: %v\n", prov.Kind(), err)
				}
				continue
			}
			for _, p := range res.Papers {
				// Deduped for the reason the actor dedupes: the same paper is
				// routinely on both providers and can match several questions,
				// and rep.Papers is len(rep.Rows) — so a duplicate counted twice
				// in whichever bucket it landed in and moved the percentages.
				key := paperKey(p)
				if key == "" || seen[key] {
					rep.Duplicates++
					continue
				}
				seen[key] = true
				rep.Rows = append(rep.Rows, classify(ctx, p, unpaywall, lim, client, o, rep))
			}
		}
	}

	tally(rep)
	if o.asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(rep)
	}
	printCoverage(rep)
	if rep.Incomplete != "" {
		// Non-zero, so a script cannot mistake a truncated sweep for a verdict.
		return errors.New(rep.Incomplete)
	}
	return nil
}

// classify decides one paper's bucket, escalating only as far as it must.
//
// A provider that already reports readable full text is believed — PMC says so
// from metadata — and no further request is made for it. Everything else costs
// at most one Unpaywall call and, for arXiv, one HEAD.
func classify(
	ctx context.Context,
	p academic.Paper,
	unpaywall *academic.Unpaywall,
	lim *limiter.Limiter,
	client *http.Client,
	o coverageOpts,
	rep *coverageReport,
) coverageRow {
	row := coverageRow{
		Source:   string(p.Source),
		Title:    clampTitle(p.Title),
		DOI:      p.DOI,
		Resolved: true,
	}
	if p.PublishedAt != nil {
		row.Year = p.PublishedAt.Year()
	}

	// Already readable: PMC reports full text, so nothing more is needed.
	if p.FullTextFormat() == academic.FormatHTML {
		row.Format = string(academic.FormatHTML)
		return row
	}

	// arXiv's API never reports HTML availability, so an arXiv paper is
	// indistinguishable from pdf_only without a probe. This is the request the
	// --probe-html flag pays for, and skipping it is exactly the bias that would
	// make arXiv look unreadable.
	if p.ArXivID != "" && o.probeHTML {
		if htmlURL := academic.ArXivHTMLURL(p.ArXivID); htmlURL != "" {
			ok, err := headOK(ctx, client, lim, o.agent, htmlURL)
			switch {
			case err != nil:
				rep.ProbeErrors++
				row.Note = "html probe failed: " + err.Error()
			case ok:
				row.Format = string(academic.FormatHTML)
				return row
			}
		}
	}

	// Unpaywall can UPGRADE a paper's readability, never downgrade it.
	//
	// Its verdict is built from its own record alone, so an arXiv paper with a
	// journal DOI that Unpaywall has as is_oa false was filed "closed" — while
	// its PDF sits openly on arXiv, which the provider already told us. That
	// moves papers out of pdf_only, the one bucket this command exists to size,
	// in the direction of "we don't need a parser". The comment on pickLocations
	// warns about the opposite bias from best_oa_location; this is the same
	// mistake from the other side.
	known := p.FullTextFormat()
	if p.DOI != "" {
		resolved, err := unpaywall.Resolve(ctx, p.DOI)
		switch {
		case err != nil:
			rep.ResolveErrors++
			row.Resolved = false
			row.Note = strings.TrimPrefix(err.Error(), "academic: ")
		default:
			if better(resolved.FullTextFormat(), known) {
				known = resolved.FullTextFormat()
			}
		}
	}
	row.Format = string(known)
	return row
}

// better reports whether a is a more readable verdict than b.
//
// html beats pdf_only beats closed. An ordering rather than a bare comparison
// because "which of these two verdicts do I keep" is the actual question, and
// writing it as an if-chain at the call site is how the downgrade got in.
func better(a, b academic.FullTextFormat) bool { return readability(a) > readability(b) }

func readability(f academic.FullTextFormat) int {
	switch f {
	case academic.FormatHTML:
		return 2
	case academic.FormatPDFOnly:
		return 1
	default:
		return 0
	}
}

func headOK(ctx context.Context, client *http.Client, lim *limiter.Limiter, agent, rawURL string) (bool, error) {
	if err := lim.Wait(ctx, "arxiv.org"); err != nil {
		return false, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, rawURL, nil)
	if err != nil {
		return false, err
	}
	// Identified, like every other request this milestone makes. This is the
	// highest-VOLUME path in M6 — up to a few hundred requests to arxiv.org, a
	// service run by a library — and it was the one with no contact address, no
	// version and no way for an operator to reach anyone or block selectively.
	req.Header.Set("User-Agent", agent)
	resp, err := client.Do(req)
	if err != nil {
		return false, err
	}
	defer func() {
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
	}()
	return resp.StatusCode == http.StatusOK, nil
}

// era splits at arXiv's HTML rollout.
//
// LaTeXML HTML exists only for papers from roughly December 2023, so a corpus
// weighted toward older literature understates the coverage current research
// will have. Reporting one blended number would be misleading in a direction
// that is predictable and therefore avoidable.
func era(year int) string {
	switch {
	case year == 0:
		return "unknown"
	case year >= 2024:
		return "2024+"
	default:
		return "pre-2024"
	}
}

func tally(rep *coverageReport) {
	rep.Papers = len(rep.Rows)
	for _, r := range rep.Rows {
		rep.ByFormat[r.Format]++
		if rep.BySource[r.Source] == nil {
			rep.BySource[r.Source] = map[string]int{}
		}
		rep.BySource[r.Source][r.Format]++

		e := era(r.Year)
		if rep.ByEra[e] == nil {
			rep.ByEra[e] = map[string]int{}
		}
		rep.ByEra[e][r.Format]++
	}
}

func printCoverage(rep *coverageReport) {
	fmt.Printf("\ncorpus %s — %d question(s), %d paper(s)\n\n", rep.Corpus, rep.Questions, rep.Papers)

	fmt.Println("FORMAT      COUNT   SHARE")
	for _, f := range []string{"html", "pdf_only", "closed"} {
		n := rep.ByFormat[f]
		fmt.Printf("%-11s %5d   %s\n", f, n, sharePct(n, rep.Papers))
	}

	fmt.Println("\nBY SOURCE")
	for _, src := range sortedSourceKeys(rep.BySource) {
		row := rep.BySource[src]
		fmt.Printf("  %-9s html %-4d pdf_only %-4d closed %-4d\n",
			src, row["html"], row["pdf_only"], row["closed"])
	}

	fmt.Println("\nBY ERA (arXiv has only served HTML since late 2023)")
	for _, e := range []string{"pre-2024", "2024+", "unknown"} {
		row := rep.ByEra[e]
		if len(row) == 0 {
			continue
		}
		fmt.Printf("  %-9s html %-4d pdf_only %-4d closed %-4d\n",
			e, row["html"], row["pdf_only"], row["closed"])
	}

	fmt.Printf("\nfailures: %d search, %d resolve, %d html probe\n",
		rep.SearchErrors, rep.ResolveErrors, rep.ProbeErrors)
	if rep.SearchErrors+rep.ResolveErrors+rep.ProbeErrors > 0 {
		fmt.Println("  counted, not dropped — a failure is not evidence that a paper is unreadable")
	}
	if rep.Duplicates > 0 {
		fmt.Printf("  %d duplicate paper(s) skipped\n", rep.Duplicates)
	}
	if rep.Incomplete != "" {
		fmt.Printf("\nINCOMPLETE — %s\nThe percentages below cover only what ran.\n", rep.Incomplete)
	}
	fmt.Printf("\nOnly pdf_only is a PDF extractor's job: %d of %d (%s).\n",
		rep.ByFormat["pdf_only"], rep.Papers, sharePct(rep.ByFormat["pdf_only"], rep.Papers))
}

func sharePct(n, total int) string {
	if total == 0 {
		return "n/a"
	}
	return fmt.Sprintf("%.0f%%", 100*float64(n)/float64(total))
}

func sortedSourceKeys(m map[string]map[string]int) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func clampTitle(s string) string {
	if len(s) <= 70 {
		return s
	}
	return s[:70] + "…"
}

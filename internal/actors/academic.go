package actors

import (
	"context"
	"fmt"
	"log/slog"
	"net/url"
	"sort"
	"strings"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/pricing"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/tools/academic"
	"github.com/lajosdeme/mole/internal/tools/extract"
	"github.com/lajosdeme/mole/internal/tools/fetch"
)

// AcademicActor researches a lead against scholarly sources (§10.2).
//
// The shape differs from WebActor in one way that matters: tier 0 of the
// escalation ladder costs NO fetches at all. arXiv and PubMed return the
// abstract in the search response, so a paper's most information-dense passage
// arrives for the price of the search — where a web lead pays a search, a fetch
// and an extraction to reach less.
//
// It also supplies something no web source reliably does: exact publication
// dates. §11.2's staleness rule compares when claims were made and has been
// reading zero for want of dates on both sides of a contradiction.
type AcademicActor struct {
	// Providers are searched in order. arXiv and PubMed cover different
	// literatures and overlap little, so both run rather than one being chosen.
	Providers []academic.Provider

	LLM     llm.Provider
	Pricing *pricing.Table
	Store   store.Store
	Log     *slog.Logger
	Budget  Budget

	// Fetch and Extract read tier-1 full text. Nil disables escalation
	// entirely, which is a supported configuration: the abstracts still yield
	// claims, and §10.2's ladder is about reading LESS, so refusing to read
	// more is never wrong, only less complete.
	Fetch   fetch.Fetcher
	Extract extract.Extractor

	// Resolve places a paper the search providers could not: given a DOI it
	// finds where the paper can legally be read. Nil disables it.
	//
	// Only ever consulted at escalation, and only for a paper with a DOI that
	// neither provider gave a readable location for.
	Resolve academic.Resolver

	// Rank orders passages by relevance to a sub-question. See rankChunks for
	// why it is injected rather than imported.
	Rank func(question string, passages []string, n int) []int

	SessionID string
	// Question is the session's root question, shown to the miner beside the lead.
	Question string
}

func (a *AcademicActor) Type() core.ActorType { return core.ActorAcademic }

func (a *AcademicActor) logger() *slog.Logger {
	if a.Log != nil {
		return a.Log
	}
	return slog.Default()
}

func (a *AcademicActor) Run(ctx context.Context, lead core.Lead) (*Result, error) {
	budget := a.Budget
	if sub, ok := SubBudgetFrom(ctx); ok {
		budget = budget.tighten(sub)
	}
	maxSources := budget.MaxSources
	if maxSources <= 0 {
		maxSources = 5
	}
	maxClaims := budget.MaxClaimsPerSource
	if maxClaims <= 0 {
		maxClaims = 8
	}
	// Defaulted, not left at zero. The token guard below is written as
	// "MaxInputTokens > 0 && ...", so a zero made it a no-op — and zero is
	// reachable: the executor's sub-budget returns an empty Budget whenever the
	// reservation prices to nothing, which is exactly the unpriced-model case.
	// A web lead gets 60k from Budget.withDefaults; an academic lead had no
	// ceiling at all.
	maxInput := budget.MaxInputTokens
	if maxInput <= 0 {
		maxInput = defaultAcademicInputTokens
	}

	res := &Result{}
	if len(a.Providers) == 0 {
		return res, fmt.Errorf("actors/academic: no providers configured")
	}

	papers, searchErr := a.gather(ctx, lead, maxSources)
	if len(papers) == 0 {
		if searchErr != nil {
			return res, searchErr
		}
		res.Summary = fmt.Sprintf("No papers found for %q.", lead.Query)
		return res, nil
	}

	miner := &Miner{LLM: a.LLM, Pricing: a.Pricing, Log: a.Log, SessionID: a.SessionID}
	var (
		used    int64
		refusal error
	)

	for _, p := range papers {
		abstract := strings.TrimSpace(p.Abstract)
		if abstract == "" {
			// A paper with no abstract has nothing to mine at tier 0. Counted so
			// the planner sees the lead did work, rather than looking like a
			// question nobody asked.
			res.Stats.ChunksSkipped++
			continue
		}
		// The sub-budget is a ceiling on input tokens, and an abstract is small
		// enough that the estimate can be crude: stopping before the ceiling
		// costs a paper, exceeding it costs the reservation.
		if used+llm.EstimateTokens(len(abstract)) > maxInput {
			res.Truncated = true
			break
		}

		out, err := miner.Mine(ctx, MineInput{
			Lead:      lead,
			SourceURL: citationURL(p),
			Title:     p.Title,
			// Exact, from the provider — not guessed from a page.
			PublishedAt:   p.PublishedAt,
			Text:          abstract,
			MaxClaims:     maxClaims,
			MaxQualifiers: budget.withDefaults().MaxQualifiersPerSource,
			Question:      a.Question,
		})
		if out.HasCall {
			res.Costs = append(res.Costs, out.Call)
		}
		res.Stats.ClaimsProposed += out.Proposed
		res.Stats.ClaimsRejected += out.Rejected
		// ACTUAL usage, not the estimate. WebActor accumulates what the provider
		// reported for the same reason: the prompt is not free — system text,
		// fencing and the question come to roughly as many tokens as a whole
		// abstract — so counting the source text alone undercounted real input
		// by about half.
		used += out.Usage.InputTokens + out.Usage.CacheReadTokens
		if llm.Refused(err) {
			refusal = err
			break
		}
		if err != nil {
			a.logger().WarnContext(ctx, "mining an abstract failed",
				"source", citationURL(p), "err", err)
			continue
		}
		res.Stats.Chunks++
		res.Claims = append(res.Claims, out.Claims...)

		// Tier 1, and only when the abstract did not answer the question.
		//
		// The claim cap is what REMAINS for this paper, not the full allowance
		// again. Applying it per call let one paper contribute maxClaims from its
		// abstract and maxClaims from every escalated chunk — four times its
		// share at the default. WebActor decrements for exactly this reason: the
		// cap exists so one verbose document cannot dominate the graph, and §11.3
		// counts publishers, so one paper outvoting four corrupts confidence.
		remaining := maxClaims - (len(out.Claims) - out.Qualifiers)
		qualifiersLeft := max(budget.withDefaults().MaxQualifiersPerSource-out.Qualifiers, 0)

		// Gate BEFORE target, and the order is load-bearing now that
		// fullTextTarget can make a request. Go evaluates && left to right, so
		// asking for the target first would pay Unpaywall for every paper —
		// including the majority whose abstract already answered the question,
		// which is exactly the spend the ladder exists to avoid.
		if remaining > 0 && shouldEscalate(lead.Query, out.Claims) {
			if target := a.fullTextTarget(ctx, p); target != "" {
				extra, spent, rerr := a.readFullText(ctx, lead, p, target, miner, remaining, qualifiersLeft, maxInput-used, res)
				res.Claims = append(res.Claims, extra...)
				used += spent
				if rerr != nil {
					refusal = rerr
					break
				}
				if used >= maxInput {
					res.Truncated = true
					break
				}
			}
		}
	}

	res.Summary = summarize(lead.Query, papers, len(res.Claims))

	// Persisted on an uncancellable context, for the reason WebActor's step 7
	// is: by here the model calls are made and the ledger will charge for them
	// whether or not this write lands, and the report is generated from the
	// store rather than from this Result.
	if len(res.Claims) > 0 && a.Store != nil {
		if err := a.Store.WithTx(context.WithoutCancel(ctx), func(ctx context.Context, tx store.Tx) error {
			return tx.InsertClaims(ctx, res.Claims)
		}); err != nil {
			return res, fmt.Errorf("actors/academic: persist claims: %w", err)
		}
	}
	if refusal != nil {
		return res, fmt.Errorf("actors/academic: mine: %w", refusal)
	}
	return res, nil
}

// gather searches every provider and merges the results.
//
// A search failure on one provider is not fatal: arXiv and PubMed index
// different literatures, and half an answer beats none. The error is returned
// only when nothing at all came back.
func (a *AcademicActor) gather(ctx context.Context, lead core.Lead, max int) ([]academic.Paper, error) {
	var (
		papers  []academic.Paper
		lastErr error
	)
	seen := map[string]bool{}

	for _, prov := range a.Providers {
		res, err := prov.Search(ctx, lead.Query, academic.Options{MaxResults: max})
		if err != nil {
			lastErr = err
			a.logger().WarnContext(ctx, "academic search failed",
				"provider", prov.Kind(), "err", err)
			continue
		}
		for _, p := range res.Papers {
			key := dedupeKey(p)
			if key == "" || seen[key] {
				continue
			}
			seen[key] = true
			papers = append(papers, p)
		}
	}

	// Newest first. When two papers disagree, §11.2 wants the recent one in
	// hand; and a truncated read should keep current work rather than whatever
	// the provider happened to rank first.
	sort.SliceStable(papers, func(i, j int) bool {
		switch {
		case papers[i].PublishedAt == nil:
			return false
		case papers[j].PublishedAt == nil:
			return true
		default:
			return papers[i].PublishedAt.After(*papers[j].PublishedAt)
		}
	})
	if len(papers) > max {
		papers = papers[:max]
	}
	return papers, lastErr
}

// dedupeKey identifies a paper across providers.
//
// DOI first, because the same paper is routinely on both arXiv and PubMed and
// mining it twice would pay twice and inflate corroboration — §11.3 counts
// independent publishers, and one paper indexed twice is not two publishers.
func dedupeKey(p academic.Paper) string {
	switch {
	case strings.TrimSpace(p.DOI) != "":
		return "doi:" + strings.ToLower(strings.TrimSpace(p.DOI))
	case strings.TrimSpace(p.ArXivID) != "":
		return "arxiv:" + strings.TrimSpace(p.ArXivID)
	case strings.TrimSpace(p.PMID) != "":
		return "pmid:" + strings.TrimSpace(p.PMID)
	default:
		return strings.ToLower(collapseSpace(p.Title))
	}
}

// citationURL is what a claim from this paper cites.
//
// A reader has to be able to open it, so it is the landing page or the DOI —
// never the API endpoint the text arrived through.
func citationURL(p academic.Paper) string {
	switch {
	case strings.TrimSpace(p.LandingURL) != "":
		return p.LandingURL
	case strings.TrimSpace(p.DOI) != "":
		return "https://doi.org/" + strings.TrimSpace(p.DOI)
	case strings.TrimSpace(p.HTMLURL) != "":
		return p.HTMLURL
	default:
		return p.PDFURL
	}
}

// summarize writes the planner's view of this lead, mechanically.
//
// No model call: nothing downstream reads Result.Summary, and titles are
// provider metadata, not mined content.
func summarize(query string, papers []academic.Paper, claims int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Read %d paper(s) for %q, yielding %d claim(s).", len(papers), query, claims)

	shown := papers
	if len(shown) > 5 {
		shown = shown[:5]
	}
	for _, p := range shown {
		year := ""
		if p.PublishedAt != nil {
			year = fmt.Sprintf(" (%d)", p.PublishedAt.Year())
		}
		fmt.Fprintf(&b, "\n  - %s%s", collapseSpace(p.Title), year)
	}
	return b.String()
}

// ---------------------------------------------------------------------------
// Tier 1: full text, ranked
// ---------------------------------------------------------------------------

// DefaultEscalationSections is how many passages of full text one paper may
// contribute after its abstract was judged insufficient.
//
// Small on purpose. The whole argument for the ladder is that reading a whole
// paper to answer one sub-question is the expensive default; escalating from
// "one abstract" to "one whole paper" would give the saving straight back.
const DefaultEscalationSections = 3

// defaultAcademicInputTokens matches Budget.withDefaults' web ceiling.
//
// Unexported: it exists because AcademicActor.Run does not call withDefaults,
// and a caller has no reason to reach for it.
const defaultAcademicInputTokens = 60_000

// shouldEscalate decides whether the abstract answered the sub-question.
//
// Mechanical, and deliberately so. Asking the model "did that answer it?" costs
// nothing extra — it can ride on the mining call — but it is a model grading its
// own sufficiency, and the thing being decided is whether to spend more money.
// A gate that cannot be talked into spending is worth more than a slightly
// better-informed one.
//
// The rule: the sub-question's distinctive terms have to appear in what came
// back. Zero claims escalates outright; claims that never mention what was asked
// about are the case this exists for — an abstract that says a paper is "about
// metabolic health" while the question asks about sample size.
func shouldEscalate(query string, claims []core.Claim) bool {
	terms := contentTerms(query)
	if len(terms) == 0 {
		return false
	}
	if len(claims) == 0 {
		return true
	}

	var haystack strings.Builder
	for _, c := range claims {
		haystack.WriteString(strings.ToLower(c.Text))
		haystack.WriteByte(' ')
		haystack.WriteString(strings.ToLower(c.Quote))
		haystack.WriteByte(' ')
	}
	text := haystack.String()

	covered := 0
	for _, t := range terms {
		if strings.Contains(text, t) {
			covered++
		}
	}
	// Half, not all: an abstract rarely echoes every word of a question, and
	// demanding full coverage would escalate almost always, which is the same as
	// having no ladder.
	return covered*2 < len(terms)
}

// academicStopwords are words too common to distinguish one sub-question from
// another. Short and closed; a longer list is a tuning exercise that cannot be
// evaluated without the corpus.
var academicStopwords = map[string]bool{
	"the": true, "and": true, "for": true, "with": true, "what": true,
	"does": true, "how": true, "why": true, "are": true, "was": true,
	"were": true, "is": true, "in": true, "of": true, "to": true, "on": true,
	"a": true, "an": true, "do": true, "did": true, "than": true, "that": true,
	"this": true, "from": true, "when": true, "which": true, "study": true,
	"studies": true, "paper": true, "research": true,
}

func contentTerms(q string) []string {
	var out []string
	seen := map[string]bool{}
	for _, f := range strings.Fields(strings.ToLower(q)) {
		f = strings.Trim(f, ".,;:!?\"'()[]")
		if len(f) < 4 || academicStopwords[f] || seen[f] {
			continue
		}
		seen[f] = true
		out = append(out, f)
	}
	return out
}

// fullTextTarget is the URL whose text mole can read for a paper, or "".
//
// PMC is reported by the provider. arXiv is a CANDIDATE: its API never says
// whether a paper has LaTeXML HTML, so the only way to find out is to ask — and
// the ladder says ask only when escalating, which is here. A 404 costs one
// guarded request and is the answer.
func (a *AcademicActor) fullTextTarget(ctx context.Context, p academic.Paper) string {
	if u := strings.TrimSpace(p.HTMLURL); u != "" {
		return u
	}
	if p.ArXivID != "" {
		return academic.ArXivHTMLURL(p.ArXivID)
	}

	// Last resort, and a REQUEST — which is why this function is only ever
	// called after the escalation gate has already said yes. Asking Unpaywall
	// where a paper can be read is worth a rate-limited call when mole is about
	// to read it, and worth nothing when it is not.
	//
	// The measured upside is small: over a 97-paper sweep, 46 papers had no DOI
	// at all and Unpaywall was consulted for 28 unreadable ones without finding
	// anything, so this places at most about 5%. It is here because an actor
	// that designs in a resolver and never calls it is a capability that exists
	// only in its comments.
	if a.Resolve == nil || strings.TrimSpace(p.DOI) == "" {
		return ""
	}
	resolved, err := a.Resolve.Resolve(ctx, p.DOI)
	if err != nil {
		// Not the lead's problem. Unpaywall not knowing a DOI, or being briefly
		// unreachable, means this paper contributes its abstract and no more.
		a.logger().DebugContext(ctx, "could not place a paper",
			"doi", p.DOI, "err", err)
		return ""
	}
	return strings.TrimSpace(resolved.HTMLURL)
}

// readFullText mines the passages of a paper's full text most relevant to the
// lead, and reports what it spent.
//
// A paper whose only copy is a PDF never gets here — fullTextTarget returns ""
// for it — so mole never fetches a PDF and never records unsupported_type from
// this path. The PDF question is answered by `mole dev academic-coverage`, from
// metadata, and NOT by outcome rows accumulating from live runs; an earlier
// comment here claimed the latter and was wrong.
//
// Every fetch this function DOES attempt writes a §10.4 row, so the tier-1 path
// has a denominator like every other fetch — which also puts academic sources in
// reach of grounding's ProviderSupplied skip set.
func (a *AcademicActor) readFullText(
	ctx context.Context,
	lead core.Lead,
	p academic.Paper,
	target string,
	miner *Miner,
	maxClaims int,
	maxQualifiers int,
	tokenBudget int64,
	res *Result,
) ([]core.Claim, int64, error) {
	if a.Fetch == nil || a.Extract == nil || tokenBudget <= 0 {
		return nil, 0, nil
	}

	fetched, err := a.Fetch.Fetch(ctx, target)
	outcome := fetch.OutcomeNetworkError
	status, bytes := 0, int64(0)
	if fetched != nil {
		outcome, status, bytes = fetched.Outcome, fetched.StatusCode, fetched.Bytes
	}
	if err != nil || fetched == nil || !outcome.Usable() {
		a.recordOutcome(ctx, lead, target, outcome, status, bytes, errText(err))
		res.Stats.ChunksFailed++
		return nil, 0, nil
	}
	res.Stats.Fetched++

	pageURL, perr := url.Parse(target)
	if perr != nil {
		return nil, 0, nil
	}
	doc, err := a.Extract.Extract(ctx, fetched.Content, fetched.ContentType, pageURL)
	if err != nil || doc == nil || strings.TrimSpace(doc.Text) == "" {
		a.recordOutcome(ctx, lead, target, fetch.OutcomeExtractFailed, status, bytes, errText(err))
		res.Stats.ChunksFailed++
		return nil, 0, nil
	}
	a.recordOutcome(ctx, lead, target, outcome, status, bytes, "")

	chunks := llm.Split(doc.Text, llm.ChunkOptions{MaxChars: 6000, OverlapChars: 200, MinChars: 400})
	if len(chunks) == 0 {
		return nil, 0, nil
	}

	order := a.rankChunks(lead.Query, chunks)
	if len(order) > DefaultEscalationSections {
		order = order[:DefaultEscalationSections]
	}

	var (
		claims     []core.Claim
		qualifiers int
		spent      int64
	)
	for _, idx := range order {
		chunk := chunks[idx]
		if spent+llm.EstimateTokens(len(chunk.Text)) > tokenBudget {
			res.Truncated = true
			break
		}
		out, err := miner.Mine(ctx, MineInput{
			Lead: lead,
			// The URL the text was actually READ FROM, not the abstract page.
			// Citing the PubMed landing page for a quote taken from the PMC full
			// text made §11.5.2's grounding re-fetch a page the quote was never
			// on and report "the quote is no longer present; the page has
			// changed" — a manufactured mismatch — and eval scored it a citation
			// mismatch, which sets Regression and fails the build.
			SourceURL:   target,
			Title:       p.Title,
			PublishedAt: p.PublishedAt,
			Text:        chunk.Text,
			Offset:      chunk.Start,
			// Decremented, and the break below bounds it too. Either alone holds
			// the cap; both are kept because this one also stops the model being
			// asked for claims that would be thrown away.
			MaxClaims:     maxClaims - (len(claims) - qualifiers),
			MaxQualifiers: maxQualifiers - qualifiers,
			Question:      a.Question,
		})
		if out.HasCall {
			res.Costs = append(res.Costs, out.Call)
		}
		res.Stats.ClaimsProposed += out.Proposed
		res.Stats.ClaimsRejected += out.Rejected
		res.Stats.Chunks++
		spent += out.Usage.InputTokens + out.Usage.CacheReadTokens
		if llm.Refused(err) {
			return claims, spent, err
		}
		if err != nil {
			res.Stats.ChunksFailed++
			continue
		}
		claims = append(claims, out.Claims...)
		qualifiers += out.Qualifiers
		if len(claims)-qualifiers >= maxClaims && qualifiers >= maxQualifiers {
			break
		}
	}
	return claims, spent, nil
}

// recordOutcome writes the §10.4 row for one tier-1 fetch.
func (a *AcademicActor) recordOutcome(
	ctx context.Context, lead core.Lead, rawURL string,
	outcome fetch.Outcome, status int, bytes int64, errText string,
) {
	if a.Store == nil {
		return
	}
	sessionID, leadID := a.SessionID, lead.ID
	if err := a.Store.WithTx(context.WithoutCancel(ctx), func(ctx context.Context, tx store.Tx) error {
		return tx.RecordFetchOutcome(ctx, &store.FetchOutcome{
			SessionID: &sessionID, LeadID: &leadID, URL: rawURL,
			Domain: domainOf(rawURL), Outcome: string(outcome),
			StatusCode: status, Bytes: bytes, Err: errText,
		})
	}); err != nil {
		a.logger().WarnContext(ctx, "could not record a fetch outcome", "url", rawURL, "err", err)
	}
}

func errText(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func domainOf(rawURL string) string {
	if u, err := url.Parse(rawURL); err == nil {
		return strings.ToLower(u.Hostname())
	}
	return ""
}

// rankChunks orders passages by relevance to the question.
//
// Ranking, not searching. The instinct is to have a model say which part holds
// the answer and then binary-search the rest, but binary search needs the probe
// to say which half to go to next — and not finding an answer in one chunk says
// nothing about which other chunk holds it, so it degrades to a linear scan at
// one model call per probe.
//
// Rank is INJECTED rather than imported. verifier.LexicalRetriever is the right
// scorer and already does this for research.ask, but verifier/ground.go imports
// this package, so importing it back would be a cycle. The caller supplies it;
// unset, passages are read in document order, which for a paper puts the
// abstract and introduction first and is a defensible default rather than a
// silent failure.
func (a *AcademicActor) rankChunks(query string, chunks []llm.Chunk) []int {
	if a.Rank == nil {
		order := make([]int, len(chunks))
		for i := range order {
			order[i] = i
		}
		return order
	}
	passages := make([]string, len(chunks))
	for i, c := range chunks {
		passages[i] = c.Text
	}
	return a.Rank(query, passages, len(passages))
}

// ShouldEscalateForTest exposes the escalation gate to the package's tests.
//
// Exported for tests only, and named so. The gate is the one decision in this
// actor that spends money, so it is worth testing directly rather than through
// a run whose result could satisfy the assertion for other reasons.
func ShouldEscalateForTest(query string, claims []core.Claim) bool {
	return shouldEscalate(query, claims)
}

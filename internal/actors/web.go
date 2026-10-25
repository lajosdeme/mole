package actors

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/url"

	"github.com/lajosdeme/mole/internal/cache"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/dataset"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/pricing"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/tools/extract"
	"github.com/lajosdeme/mole/internal/tools/fetch"
	"github.com/lajosdeme/mole/internal/tools/search"
)

// WebActor runs a web research lead:
//
//	search → (fetch → extract)? → chunk → mine claims → verify quotes → summarize
//
// The parenthesized step is conditional: a search provider that returned usable
// page content skips it entirely (§10.4). Everything raw dies when Run returns.
type WebActor struct {
	// Rows, when set, switches this actor to dataset mode: every chunk is
	// extracted into schema-shaped rows instead of claims (M9, §13). Nil is
	// report mode, which is every session that did not ask for a dataset.
	Rows *RowMiner

	Search  search.Provider
	Fetch   fetch.Fetcher
	Extract extract.Extractor
	LLM     llm.Provider
	Pricing *pricing.Table
	Store   store.Store
	Log     *slog.Logger
	Budget  Budget

	// SessionID scopes the claims and fetch outcomes this actor records.
	SessionID string

	// Cache, when set, holds documents already fetched in this session. The
	// nested check §9.3 describes: two distinct queries converging on one page
	// pay for one fetch. Nil disables it.
	Cache *cache.Cache
}

func (a *WebActor) Type() core.ActorType { return core.ActorWeb }

func (a *WebActor) logger() *slog.Logger {
	if a.Log != nil {
		return a.Log
	}
	return slog.Default()
}

// Run executes one lead.
func (a *WebActor) Run(ctx context.Context, lead core.Lead) (*Result, error) {
	budget := a.Budget
	// The executor's reservation is the only ceiling that knows what this lead
	// may actually cost (§9.2). Without it the actor's configured budget bore
	// no relation to the money held for it, and a lead could spend several
	// times the whole session budget.
	if sub, ok := SubBudgetFrom(ctx); ok {
		budget = budget.tighten(sub)
	}
	budget = budget.withDefaults()
	res := &Result{}

	if a.Search == nil {
		// Reachable since M8: a local-only session does not require a search
		// provider, so the web actor is built without one and registered
		// anyway. No lead should be typed for it — the planner distributes
		// across the session's actors — and if one is, this says so instead of
		// dereferencing nil three lines below.
		return res, errors.New("actors/web: no search provider is configured, so this " +
			"session cannot research the web")
	}

	// 1. Search.
	sr, err := a.Search.Search(ctx, lead.Query, search.Options{
		MaxResults:     budget.MaxSources * 2, // headroom for guard/robots refusals
		IncludeContent: true,
	})
	if err != nil {
		return res, fmt.Errorf("actors/web: search: %w", err)
	}
	res.Costs = append(res.Costs, core.ToolCall{
		SessionID: a.SessionID,
		LeadID:    &lead.ID,
		Role:      core.RoleExecutor,
		Type:      core.CallSearch,
		Input:     lead.Query,
		Cost:      sr.Cost,
	})
	res.Stats.SearchResults = len(sr.Results)

	// 2-5. Read sources until the budget or the source cap is reached.
	var (
		inputTokensUsed int64
		sourcesRead     int
		perSource       []sourceSummary
	)

	for i, hit := range sr.Results {
		// Count sources READ, not sources that yielded something. Counting
		// summaries meant a run where nothing extracted cleanly walked the
		// entire result list — spending a fetch, an extract, and a model call
		// per hit against a cap that never advanced.
		if sourcesRead >= budget.MaxSources {
			a.recordNotSelected(ctx, lead, sr.Results[i:])
			break
		}
		if remaining := budget.MaxInputTokens - inputTokensUsed; remaining <= 0 {
			res.Truncated = true
			a.recordNotSelected(ctx, lead, sr.Results[i:])
			break
		}

		src, ok := a.readSource(ctx, lead, hit, budget, res)
		if !ok {
			continue
		}
		sourcesRead++
		doc := src.doc

		// Chunk under what is left of the sub-budget.
		plan := llm.Plan(doc.Text, budget.MaxInputTokens-inputTokensUsed, budget.ChunkOptions())
		if plan.Truncated {
			res.Truncated = true
		}
		res.Stats.Chunks += len(plan.Chunks)
		res.Stats.ChunksSkipped += plan.Skipped

		var summary sourceSummary

		for _, chunk := range plan.Chunks {
			// Plan sized the whole document against an ESTIMATE. Re-check
			// between chunks so a document whose real token count runs over
			// stops here rather than at the next source — the sub-budget is a
			// ceiling, and checking it once per source overshot it by roughly
			// the cost of a full document.
			if inputTokensUsed >= budget.MaxInputTokens {
				res.Truncated = true
				res.Stats.ChunksSkipped++
				continue
			}
			// The cap is per SOURCE, so one verbose page cannot dominate the
			// graph (§ Budget.MaxClaimsPerSource). Applying it per chunk let a
			// ten-chunk document contribute ten times the intended share.
			// Per SOURCE, counting rows as well as claims. The dataset branch
			// below appended to summary.Rows and never to summary.Claims, so this
			// stayed pinned at the full allowance for every chunk — which is
			// verbatim the failure the comment above says was already fixed once,
			// reintroduced by the mode that does not use the field it counts.
			remainingClaims := budget.MaxClaimsPerSource - len(summary.Claims) - summary.Rows
			if remainingClaims <= 0 {
				break
			}

			// Dataset mode extracts rows in place of claims (M9, §13). One model
			// call per chunk either way: asking for both would double the cost of
			// every chunk to produce a report nobody asked for.
			if a.Rows != nil {
				rows, usage, err := a.mineRowChunk(ctx, lead, src, doc, chunk, remainingClaims, res)
				inputTokensUsed += usage.InputTokens + usage.CacheReadTokens
				if err != nil {
					res.Stats.ChunksFailed++
					a.logger().WarnContext(ctx, "chunk row extraction failed",
						"url", src.url, "chunk", chunk.Index, "err", err)
					continue
				}
				res.Rows = append(res.Rows, rows...)
				summary.Rows += len(rows)
				continue
			}

			claims, usage, err := a.mineChunk(ctx, lead, src, doc, chunk, remainingClaims, budget, res)
			inputTokensUsed += usage.InputTokens + usage.CacheReadTokens
			if err != nil {
				// One bad chunk does not fail the lead; the others may still
				// carry the answer. §9.5 classifies this as degraded — but only
				// if it is counted, so the caller can tell "this source was
				// thin" from "nothing worked".
				res.Stats.ChunksFailed++
				a.logger().WarnContext(ctx, "chunk mining failed",
					"url", src.url, "chunk", chunk.Index, "err", err)
				continue
			}
			res.Claims = append(res.Claims, claims...)
			summary.Claims = append(summary.Claims, claims...)
		}

		if len(summary.Claims) > 0 || summary.Rows > 0 || doc.Excerpt != "" {
			perSource = append(perSource, summary)
		}
	}

	// 6. Summarize what was found.
	if res.Summary == "" {
		// Say what actually happened. Claiming no sources were found while
		// printing claims from those sources directly underneath is worse than
		// no summary: it tells a reader the run was empty when it was
		// degraded, and those call for different responses.
		switch {
		case len(res.Claims) > 0:
			res.Summary = fmt.Sprintf(
				"%d verified claim(s) from %d source(s).",
				len(res.Claims), len(perSource))
		case res.Stats.ChunksFailed > 0:
			res.Summary = fmt.Sprintf(
				"No claims extracted: all %d model call(s) over %d source(s) failed.",
				res.Stats.ChunksFailed, res.Stats.Fetched+res.Stats.SkippedFetch)
		default:
			res.Summary = fmt.Sprintf("No usable sources found for %q.", lead.Query)
		}
	}

	// 7. Persist claims. All or nothing (§ store.InsertClaims).
	//
	// On an uncancellable context, for the same reason Settle uses one: by this
	// point the searches, fetches and mining have happened and the ledger will
	// charge for them whether or not this write lands. A cancellation here — a
	// sibling lead failing fatally and cancelling the batch, a caller pressing
	// Ctrl-C — would discard evidence that has already been paid for, and the
	// report is generated from the store rather than from the returned Result,
	// so those claims would vanish from the answer while still appearing in the
	// spend. Measured before this: Result.Claims=4, rows stored=0.
	if len(res.Claims) > 0 && a.Store != nil {
		if err := a.Store.WithTx(context.WithoutCancel(ctx), func(ctx context.Context, tx store.Tx) error {
			return tx.InsertClaims(ctx, res.Claims)
		}); err != nil {
			return res, fmt.Errorf("actors/web: persist claims: %w", err)
		}
	}
	// Rows, on the same uncancellable context and for the same reason: the
	// dataset is assembled from the STORE at the end of the session, so a row
	// discarded here is one that was paid for and does not appear in the answer.
	if len(res.Rows) > 0 && a.Store != nil {
		if err := a.Store.WithTx(context.WithoutCancel(ctx), func(ctx context.Context, tx store.Tx) error {
			return tx.InsertRows(ctx, a.SessionID, res.Rows)
		}); err != nil {
			return res, fmt.Errorf("actors/web: persist rows: %w", err)
		}
	}

	return res, nil
}

// sourceSummary tracks one source's contribution: claims and rows toward
// MaxClaimsPerSource, and whether it counted toward the run summary.
type sourceSummary struct {
	Claims []core.Claim
	// Rows counts what dataset mode extracted from this source, not the rows
	// themselves.
	Rows int
}

// source is a read source: its text, and the URL that text actually came from.
type source struct {
	doc    *extract.Document
	url    string
	domain string
}

// readSource turns one search hit into extracted text.
//
// The fetch is skipped when the search provider already returned usable
// content. That is the efficiency §10.4 identifies before any headless-browser
// question: no HTTP request, no robots round-trip, no rate-limit pressure, and
// no js_required outcome to explain.
func (a *WebActor) readSource(ctx context.Context, lead core.Lead, hit search.Result, budget Budget, res *Result) (source, bool) {
	pageURL, err := url.Parse(hit.URL)
	if err != nil {
		return source{}, false
	}

	// Check the URL cache after search, before fetch (§9.3). A page another
	// lead already read is free, and converging queries are the common case
	// once a planner is decomposing one question several ways.
	key := cache.URLKey(hit.URL)
	if e, ok := a.Cache.Get(key); ok && len(e.Text) >= fetch.MinUsableText {
		res.Stats.CacheHits++
		return source{
			doc: &extract.Document{
				Text:        e.Text,
				Title:       e.Title,
				PublishedAt: e.PublishedAt,
				Source:      extract.SourcePlainText,
			},
			url:    hit.URL,
			domain: fetch.DomainOf(hit.URL),
		}, true
	}

	if hit.HasUsableContent() && !budget.AlwaysFetch {
		res.Stats.SkippedFetch++
		a.recordOutcome(ctx, lead, hit.URL, fetch.OutcomeProviderContent, 0, 0, "")
		a.Cache.Put(&cache.Entry{
			Key: key, Text: hit.Content, Title: hit.Title, PublishedAt: hit.PublishedAt,
		})
		return source{
			doc: &extract.Document{
				Text:        hit.Content,
				Title:       hit.Title,
				Excerpt:     hit.Snippet,
				PublishedAt: hit.PublishedAt,
				Source:      extract.SourcePlainText,
			},
			url:    hit.URL,
			domain: fetch.DomainOf(hit.URL),
		}, true
	}

	fr, ferr := a.Fetch.Fetch(ctx, hit.URL)
	res.Stats.Fetched++
	res.Costs = append(res.Costs, core.ToolCall{
		SessionID:  a.SessionID,
		LeadID:     &lead.ID,
		Role:       core.RoleExecutor,
		Type:       core.CallFetch,
		Input:      hit.URL,
		DurationMS: fr.Duration.Milliseconds(),
		Err:        fr.Err,
	})

	// The fetcher already reduced the host; recomputing it here from the raw
	// URL produced a second, subtly different answer for the same row.
	domain := fr.Domain
	if domain == "" {
		domain = fetch.DomainOf(hit.URL)
	}
	// Attribute to where the bytes came from. A claim cited to a URL that 301s
	// elsewhere sends a reader to the redirect, not to the evidence.
	finalURL := fr.FinalURL
	if finalURL == "" {
		finalURL = hit.URL
	}

	if ferr != nil || !fr.Outcome.Usable() {
		a.recordOutcomeFor(ctx, lead, hit.URL, domain, fr.Outcome, fr.StatusCode, fr.Bytes, fr.Err)
		return source{}, false
	}

	doc, eerr := a.Extract.Extract(ctx, fr.Content, fr.ContentType, pageURL)
	if eerr != nil {
		a.recordOutcomeFor(ctx, lead, hit.URL, domain, fetch.OutcomeExtractFailed, fr.StatusCode, fr.Bytes, eerr.Error())
		return source{}, false
	}

	// The transport said OK; whether the bytes held readable text is only
	// knowable now. This is what keeps structured_only a distinct cause.
	extract.Refine(fr, doc)
	a.recordOutcomeFor(ctx, lead, hit.URL, domain, fr.Outcome, fr.StatusCode, fr.Bytes, fr.Err)

	if !doc.Usable() {
		return source{}, false
	}
	if doc.PublishedAt == nil {
		doc.PublishedAt = hit.PublishedAt
	}
	// Key on the URL we were given AND the one we landed on: a later lead may
	// surface either, and a redirect chain that costs one fetch should not cost
	// a second because the search provider phrased the link differently.
	a.Cache.Put(&cache.Entry{
		Key: key, Text: doc.Text, Title: doc.Title, PublishedAt: doc.PublishedAt,
	})
	if fk := cache.URLKey(finalURL); fk != key {
		a.Cache.Put(&cache.Entry{
			Key: fk, Text: doc.Text, Title: doc.Title, PublishedAt: doc.PublishedAt,
		})
	}
	return source{doc: doc, url: finalURL, domain: domain}, true
}

// mineChunk delegates to the shared Miner.
//
// The mining logic lives in Miner because the AcademicActor needs exactly it —
// a model call, a parse, a verbatim quote check, and §11.4's lineage — and two
// copies would be two places for the quote check to drift. What stays here is
// the web-specific bookkeeping: the chunk's offset within the document, and the
// stats the fetch pipeline keeps.
func (a *WebActor) mineChunk(
	ctx context.Context,
	lead core.Lead,
	src source,
	doc *extract.Document,
	chunk llm.Chunk,
	maxClaims int,
	budget Budget,
	res *Result,
) ([]core.Claim, llm.Usage, error) {
	miner := &Miner{LLM: a.LLM, Pricing: a.Pricing, Log: a.Log, SessionID: a.SessionID}

	out, err := miner.Mine(ctx, MineInput{
		Lead:        lead,
		SourceURL:   src.url,
		Title:       doc.Title,
		PublishedAt: doc.PublishedAt,
		Text:        chunk.Text,
		Offset:      chunk.Start,
		MaxClaims:   maxClaims,
	})
	if out.HasCall {
		// Recorded before the error is returned: the tokens were spent whether
		// or not the call produced anything.
		res.Costs = append(res.Costs, out.Call)
	}
	res.Stats.ClaimsProposed += out.Proposed
	res.Stats.ClaimsRejected += out.Rejected
	return out.Claims, out.Usage, err
}

// toolCall prices a model response into a ledger row.

// recordOutcome writes the §10.4 row for one fetch attempt.
//
// Failures are recorded as diligently as successes: the rate per cause is
// meaningless without a denominator, and §17.1's gate reads both.
func (a *WebActor) recordOutcome(ctx context.Context, lead core.Lead, rawURL string, outcome fetch.Outcome, status int, bytes int64, errText string) {
	a.recordOutcomeFor(ctx, lead, rawURL, fetch.DomainOf(rawURL), outcome, status, bytes, errText)
}

// recordNotSelected records the tail of a result list the actor never reached.
//
// Coverage needs a denominator and there was none: search results are persisted
// nowhere, so "14 sources read" could not be qualified by "of how many". A read
// count presented as coverage reads as a complete denominator, which is the one
// thing a coverage section exists to prevent.
//
// These are not failures. Nothing was wrong with these pages; the source cap or
// the input-token budget ran out before their turn, which is why the outcome is
// excluded from every per-cause rate — see Outcome.Attempted.
func (a *WebActor) recordNotSelected(ctx context.Context, lead core.Lead, skipped []search.Result) {
	for _, hit := range skipped {
		if hit.URL == "" {
			continue
		}
		a.recordOutcome(ctx, lead, hit.URL, fetch.OutcomeNotSelected, 0, 0, "")
	}
}

// recordOutcomeFor is recordOutcome for a caller that already knows the domain.
func (a *WebActor) recordOutcomeFor(ctx context.Context, lead core.Lead, rawURL, domain string, outcome fetch.Outcome, status int, bytes int64, errText string) {
	if a.Store == nil {
		return
	}
	sessionID := a.SessionID
	leadID := lead.ID

	err := a.Store.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.RecordFetchOutcome(ctx, &store.FetchOutcome{
			SessionID:  &sessionID,
			LeadID:     &leadID,
			URL:        rawURL,
			Domain:     domain,
			Outcome:    string(outcome),
			StatusCode: status,
			Bytes:      bytes,
			Err:        errText,
		})
	})
	if err != nil {
		// Losing an outcome row degrades the M2 dataset; it must not fail the
		// lead that was otherwise fine.
		a.logger().WarnContext(ctx, "recording fetch outcome failed", "url", rawURL, "err", err)
	}
}

func clamp01(f float64) float64 {
	switch {
	case f < 0:
		return 0
	case f > 1:
		return 1
	default:
		return f
	}
}

func truncateForLog(s string) string {
	if len(s) <= 60 {
		return s
	}
	return s[:60] + "…"
}

// rootClaimOf returns the claim lineage a lead's output should inherit.
//
// Empty for an ordinary planner lead, so InsertClaims falls back to "this claim is
// its own root" (§11.4). A follow-up lead carries the claim under investigation, and
// every claim it produces joins that chain rather than starting a new one — which is
// exactly what a per-row counter could not express.
func rootClaimOf(lead core.Lead) string {
	if lead.RootClaimID == nil {
		return ""
	}
	return *lead.RootClaimID
}

// mineRowChunk is mineChunk for dataset mode (M9, §13).
//
// Deliberately a sibling rather than a branch inside mineChunk: the two return
// different things and record different stats, and threading a mode flag through
// one function would put a conditional around every line of it. What they share —
// the cost row, the quote check — is shared through RowMiner and Miner both
// calling toolCallFor and FindQuote, which is where the duplication that matters
// was already collapsed.
func (a *WebActor) mineRowChunk(
	ctx context.Context,
	lead core.Lead,
	src source,
	doc *extract.Document,
	chunk llm.Chunk,
	maxRows int,
	res *Result,
) ([]dataset.Row, llm.Usage, error) {
	out, err := a.Rows.Mine(ctx, RowInput{
		Lead:      lead,
		SourceURL: src.url,
		Title:     doc.Title,
		Text:      chunk.Text,
		Offset:    chunk.Start,
		MaxRows:   maxRows,
	})
	if out.HasCall {
		res.Costs = append(res.Costs, out.Call)
	}
	// The same counters the claim path uses, so a dataset run's trace reads the
	// same way: a rising rejection rate is the signal that a model has started
	// inventing table contents.
	res.Stats.ClaimsProposed += out.Proposed
	res.Stats.ClaimsRejected += out.Rejected
	res.Stats.ValuesCoerced += out.Coerced
	if out.Coerced > 0 {
		// Warn, not debug. A schema whose types do not match what the sources
		// write produces a dataset full of empty cells and no error anywhere, and
		// the per-value detail is a debug line inside RowMiner that nobody runs at
		// debug. The count is the part that has to be visible by default.
		a.logger().WarnContext(ctx, "values dropped: the field type could not hold them",
			"lead", lead.ID, "source", src.url, "dropped", out.Coerced,
			"hint", "the schema's declared types may not match what these sources write")
	}
	if err != nil {
		return nil, out.Usage, err
	}
	return out.Rows, out.Usage, nil
}

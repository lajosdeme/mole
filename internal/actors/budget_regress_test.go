package actors_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/actors"
	"github.com/lajosdeme/mole/internal/cache"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/tools/fetch"
	"github.com/lajosdeme/mole/internal/tools/search"
)

// longArticle is big enough to split into several chunks, so per-chunk and
// per-source limits can be told apart.
func longArticle(n int) string {
	var b strings.Builder
	for i := 0; i < n; i++ {
		fmt.Fprintf(&b, "Finding %d: the measured value was %d.%02d units under the stated conditions. ",
			i, i, i%100)
		fmt.Fprintf(&b, "This sentence exists to give paragraph %d enough substance to survive chunking.\n\n", i)
	}
	return b.String()
}

// mineEverySentence answers with as many claims as the prompt allows, quoting
// the chunk verbatim so every one of them verifies. A well-behaved greedy model
// is exactly what a per-source cap has to hold back.
func mineEverySentence(prompt string) string {
	doc := documentBlock(prompt)
	var claims []map[string]any
	for _, line := range strings.Split(doc, "\n") {
		line = strings.TrimSpace(line)
		if len(line) < 40 {
			continue
		}
		claims = append(claims, map[string]any{
			"text":       "A measurement was reported: " + line[:30],
			"quote":      line,
			"confidence": 0.8,
		})
		if len(claims) >= 50 {
			break
		}
	}
	out, _ := json.Marshal(map[string]any{"claims": claims})
	return string(out)
}

// TestClaimCapIsPerSourceNotPerChunk. MaxClaimsPerSource exists so one verbose
// document cannot dominate the graph. Enforced per chunk instead, a ten-chunk
// page contributed ten times its intended share — and long pages are precisely
// the ones that get chunked.
func TestClaimCapIsPerSourceNotPerChunk(t *testing.T) {
	h := newHarness(t, []search.Result{
		{URL: "https://verbose.example/a", Content: longArticle(300), Rank: 1},
	}, nil, mineEverySentence)
	h.actor.Budget = actors.Budget{MaxSources: 1, MaxClaimsPerSource: 4, MaxInputTokens: 1_000_000}

	res, err := h.actor.Run(context.Background(), h.lead)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.Stats.Chunks < 2 {
		t.Fatalf("document produced %d chunk(s); the test needs several to be meaningful", res.Stats.Chunks)
	}
	if len(res.Claims) > 4 {
		t.Errorf("one source contributed %d claims across %d chunks, want at most 4",
			len(res.Claims), res.Stats.Chunks)
	}
	if len(res.Claims) == 0 {
		t.Error("the cap swallowed every claim")
	}
}

// TestSubBudgetIsRecheckedBetweenChunks. Plan sizes a document against an
// ESTIMATE. Checking the ceiling only once per source let a document whose real
// token count ran over spend roughly a whole extra document before anything
// noticed.
func TestSubBudgetIsRecheckedBetweenChunks(t *testing.T) {
	h := newHarness(t, []search.Result{
		{URL: "https://verbose.example/a", Content: longArticle(400), Rank: 1},
	}, nil, mineEverySentence)
	// The fake reports 1200 input tokens per call regardless of the estimate,
	// which is the real-world case: the estimate is deliberately rough.
	h.actor.Budget = actors.Budget{MaxSources: 1, MaxClaimsPerSource: 50, MaxInputTokens: 3000}

	res, err := h.actor.Run(context.Background(), h.lead)
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	var mineCalls int
	for _, c := range res.Costs {
		if c.Type == core.CallLLM && strings.HasPrefix(c.Input, "mine:") {
			mineCalls++
		}
	}
	// 3000 tokens at 1200 reported per call covers two calls and no more.
	if mineCalls > 3 {
		t.Errorf("%d mining calls against a 3000-token ceiling at 1200/call — the ceiling is not being rechecked", mineCalls)
	}
	if !res.Truncated {
		t.Error("the run overran its sub-budget without reporting Truncated")
	}
}

// TestProviderContentIsNotRecordedAsAFetch. Every per-cause rate in §10.4 is a
// fraction of attempted fetches. Filing a skipped fetch as ok padded that
// denominator with requests that were never made, which drags every failure
// rate — including the js_required number §17.1 turns on — toward zero.
func TestProviderContentIsNotRecordedAsAFetch(t *testing.T) {
	ctx := context.Background()

	h := newHarness(t, []search.Result{
		{URL: "https://provider.example/a", Content: articleBody, Rank: 1},
		{URL: "https://provider.example/b", Content: articleBody, Rank: 2},
		{URL: "https://spa.example/c", Rank: 3},
	}, map[string]*fetch.Result{
		"https://spa.example/c": {URL: "https://spa.example/c", Domain: "spa.example",
			Outcome: fetch.OutcomeOK, StatusCode: 200, ContentType: "text/html",
			Content: []byte(`<html><head><meta property="og:title" content="App"></head>` +
				`<body><div id="root"></div><script src="/a.js"></script><script src="/b.js"></script></body></html>`)},
	}, nil)
	h.actor.Budget = actors.Budget{MaxSources: 5, MaxInputTokens: 1_000_000}

	if _, err := h.actor.Run(ctx, h.lead); err != nil {
		t.Fatalf("run: %v", err)
	}

	byOutcome := map[string]int64{}
	if err := h.db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		stats, err := q.FetchOutcomeStats(ctx, time.Now().Add(-time.Hour), 20)
		for _, s := range stats {
			byOutcome[s.Outcome] = s.Count
		}
		return err
	}); err != nil {
		t.Fatal(err)
	}

	if byOutcome[string(fetch.OutcomeOK)] != 0 {
		t.Errorf("skipped fetches were recorded as ok: %v", byOutcome)
	}
	if byOutcome[string(fetch.OutcomeProviderContent)] != 2 {
		t.Errorf("provider_content count = %d, want 2; got %v",
			byOutcome[string(fetch.OutcomeProviderContent)], byOutcome)
	}
	// With the denominator honest, the one real fetch is the whole sample and
	// the SPA shows up as the capability gap it is.
	if byOutcome[string(fetch.OutcomeJSRequired)] != 1 {
		t.Errorf("js_required count = %d, want 1; got %v",
			byOutcome[string(fetch.OutcomeJSRequired)], byOutcome)
	}
}

// TestClaimsCiteTheURLTheBytesCameFrom. A claim cited to a URL that redirects
// elsewhere sends a reader to the redirect rather than to the evidence.
func TestClaimsCiteTheURLTheBytesCameFrom(t *testing.T) {
	const (
		searchURL = "https://old.example/a"
		finalURL  = "https://new.example/a"
	)

	h := newHarness(t, []search.Result{
		{URL: searchURL, Rank: 1},
	}, map[string]*fetch.Result{
		searchURL: {URL: searchURL, FinalURL: finalURL, Domain: "new.example",
			Outcome: fetch.OutcomeOK, StatusCode: 200, ContentType: "text/html",
			Content: []byte(articleHTML())},
	}, nil)

	res, err := h.actor.Run(context.Background(), h.lead)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(res.Claims) == 0 {
		t.Fatal("no claims produced")
	}
	for _, c := range res.Claims {
		if c.Source != finalURL {
			t.Errorf("claim cites %q, want the post-redirect %q", c.Source, finalURL)
		}
	}
}

// TestAlwaysFetchOverridesProviderContent. Provider-supplied text is a
// measurement blind spot: nothing is fetched, so §10.4 has no denominator,
// §17.1's gate reads "no data", and citation accuracy cannot be checked because
// re-reading the page runs a different extractor than the one that produced the
// text. An eval corpus run on Tavily silently collects none of it.
func TestAlwaysFetchOverridesProviderContent(t *testing.T) {
	ctx := context.Background()
	const url = "https://provider.example/a"

	h := newHarness(t, []search.Result{
		{URL: url, Content: articleBody, Rank: 1},
	}, map[string]*fetch.Result{
		url: {URL: url, Domain: "provider.example", Outcome: fetch.OutcomeOK,
			StatusCode: 200, ContentType: "text/html", Content: []byte(articleHTML())},
	}, nil)
	h.actor.Budget = actors.Budget{MaxSources: 1, MaxInputTokens: 1_000_000, AlwaysFetch: true}

	res, err := h.actor.Run(ctx, h.lead)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.Stats.SkippedFetch != 0 {
		t.Errorf("skipped %d fetches with --always-fetch set", res.Stats.SkippedFetch)
	}
	if res.Stats.Fetched != 1 {
		t.Errorf("fetched %d, want 1", res.Stats.Fetched)
	}

	byOutcome := map[string]int64{}
	if err := h.db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		stats, err := q.FetchOutcomeStats(ctx, time.Now().Add(-time.Hour), 0)
		for _, s := range stats {
			byOutcome[s.Outcome] = s.Count
		}
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if byOutcome[string(fetch.OutcomeProviderContent)] != 0 {
		t.Error("a provider_content row was still recorded")
	}
	if byOutcome[string(fetch.OutcomeOK)] != 1 {
		t.Errorf("outcome mix = %v, want one ok — the gate needs a denominator", byOutcome)
	}
}

// TestDefaultStillSkipsTheFetch: the efficiency §10.4 identifies is the default
// for a reason, and --always-fetch must be the opt-in.
func TestDefaultStillSkipsTheFetch(t *testing.T) {
	h := newHarness(t, []search.Result{
		{URL: "https://provider.example/a", Content: articleBody, Rank: 1},
	}, nil, nil)
	h.actor.Budget = actors.Budget{MaxSources: 1, MaxInputTokens: 1_000_000}

	res, err := h.actor.Run(context.Background(), h.lead)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.Stats.SkippedFetch != 1 || res.Stats.Fetched != 0 {
		t.Errorf("skipped=%d fetched=%d, want 1 and 0", res.Stats.SkippedFetch, res.Stats.Fetched)
	}
}

// TestConvergingQueriesPayForOneFetch is §9.3's concrete win, stated in its own
// words: "two distinct queries converging on one page pay for one fetch".
//
// Common once a planner is decomposing one question several ways — the sub-
// questions overlap, and the same authoritative page answers more than one.
func TestConvergingQueriesPayForOneFetch(t *testing.T) {
	ctx := context.Background()
	const shared = "https://arxiv.org/abs/2401.13660"

	h := newHarness(t, []search.Result{{URL: shared, Rank: 1}},
		map[string]*fetch.Result{
			shared: {URL: shared, Domain: "arxiv.org", Outcome: fetch.OutcomeOK,
				StatusCode: 200, ContentType: "text/html", Content: []byte(articleHTML())},
		}, nil)
	h.actor.Cache = cache.New()
	h.actor.Budget = actors.Budget{MaxSources: 1, MaxInputTokens: 1_000_000}

	// Two leads with different wording, same page.
	first := h.lead
	first.Query = "what does MambaByte achieve on PG-19"
	if _, err := h.actor.Run(ctx, first); err != nil {
		t.Fatal(err)
	}

	second := h.lead
	second.ID = core.NewLeadID()
	second.Query = "MambaByte bits per byte benchmark results"
	if err := h.db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.InsertLead(ctx, &second)
	}); err != nil {
		t.Fatal(err)
	}
	res2, err := h.actor.Run(ctx, second)
	if err != nil {
		t.Fatal(err)
	}

	h.fetcher.mu.Lock()
	fetches := len(h.fetcher.fetched)
	h.fetcher.mu.Unlock()

	if fetches != 1 {
		t.Errorf("the shared page was fetched %d times, want 1", fetches)
	}
	if res2.Stats.CacheHits != 1 {
		t.Errorf("cache hits on the second lead = %d, want 1", res2.Stats.CacheHits)
	}
	// The second lead must still produce claims — a hit returns the document,
	// it does not skip the source.
	if len(res2.Claims) == 0 {
		t.Error("a cached source produced no claims; the hit skipped rather than returned")
	}
}

// TestRedirectTargetIsAlsoCached. A later lead may surface either URL, and a
// redirect chain that cost one fetch should not cost a second because the search
// provider phrased the link differently.
func TestRedirectTargetIsAlsoCached(t *testing.T) {
	ctx := context.Background()
	const (
		linked = "https://old.example/paper"
		final  = "https://new.example/paper"
	)

	h := newHarness(t, []search.Result{{URL: linked, Rank: 1}},
		map[string]*fetch.Result{
			linked: {URL: linked, FinalURL: final, Domain: "new.example",
				Outcome: fetch.OutcomeOK, StatusCode: 200, ContentType: "text/html",
				Content: []byte(articleHTML())},
		}, nil)
	h.actor.Cache = cache.New()
	h.actor.Budget = actors.Budget{MaxSources: 1, MaxInputTokens: 1_000_000}

	if _, err := h.actor.Run(ctx, h.lead); err != nil {
		t.Fatal(err)
	}

	// A second lead that surfaces the POST-redirect URL directly.
	h.search.results = []search.Result{{URL: final, Rank: 1}}
	second := h.lead
	second.ID = core.NewLeadID()
	if err := h.db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.InsertLead(ctx, &second)
	}); err != nil {
		t.Fatal(err)
	}
	res2, err := h.actor.Run(ctx, second)
	if err != nil {
		t.Fatal(err)
	}

	if res2.Stats.CacheHits != 1 {
		t.Errorf("the redirect target was not cached: hits = %d", res2.Stats.CacheHits)
	}
}

// TestCacheDoesNotServeUnusableText. An entry too short to work with would turn
// a cache hit into a source that produces nothing, which is worse than a fetch.
func TestCacheDoesNotServeUnusableText(t *testing.T) {
	c := cache.New()
	const u = "https://example.com/thin"
	c.Put(&cache.Entry{Key: cache.URLKey(u), Text: "far too short to be usable"})

	h := newHarness(t, []search.Result{{URL: u, Rank: 1}},
		map[string]*fetch.Result{
			u: {URL: u, Domain: "example.com", Outcome: fetch.OutcomeOK,
				StatusCode: 200, ContentType: "text/html", Content: []byte(articleHTML())},
		}, nil)
	h.actor.Cache = c
	h.actor.Budget = actors.Budget{MaxSources: 1, MaxInputTokens: 1_000_000}

	res, err := h.actor.Run(context.Background(), h.lead)
	if err != nil {
		t.Fatal(err)
	}
	if res.Stats.CacheHits != 0 {
		t.Error("an unusably short cache entry was served")
	}
	if res.Stats.Fetched != 1 {
		t.Errorf("fetched %d times; the thin entry should have fallen through", res.Stats.Fetched)
	}
}

// TestReadTextIsStored. The export receipt checks each quote against the stored
// text; provider-supplied and fetched pages must both be stored, under the URL
// the claims cite.
func TestReadTextIsStored(t *testing.T) {
	const (
		searchURL = "https://old.example/a"
		finalURL  = "https://new.example/a"
	)
	ctx := context.Background()

	h := newHarness(t, []search.Result{
		{URL: "https://provider.example/a", Content: articleBody, Rank: 1},
		{URL: searchURL, Rank: 2},
	}, map[string]*fetch.Result{
		searchURL: {URL: searchURL, FinalURL: finalURL, Domain: "new.example",
			Outcome: fetch.OutcomeOK, StatusCode: 200, ContentType: "text/html",
			Content: []byte(articleHTML())},
	}, nil)
	h.actor.Budget = actors.Budget{MaxSources: 5, MaxInputTokens: 1_000_000}

	if _, err := h.actor.Run(ctx, h.lead); err != nil {
		t.Fatalf("run: %v", err)
	}

	var docs []core.Document
	if err := h.db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		docs, err = q.DocumentsForSession(ctx, h.actor.SessionID, time.Now())
		return err
	}); err != nil {
		t.Fatal(err)
	}
	byURL := map[string]string{}
	for _, d := range docs {
		byURL[d.URL] = d.Text
	}
	if byURL["https://provider.example/a"] != articleBody {
		t.Errorf("provider text not stored verbatim; stored URLs: %v", keys(byURL))
	}
	if byURL[finalURL] == "" {
		t.Errorf("fetched text not stored under %s; stored URLs: %v", finalURL, keys(byURL))
	}
}

func keys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

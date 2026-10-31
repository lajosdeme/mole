package search_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/tools/limiter"
	"github.com/lajosdeme/mole/internal/tools/search"
)

// The fixtures below follow each provider's documented response shape. They pin
// OUR parsing, not the provider's contract — if an API changes, these still
// pass while production breaks. A live smoke test per provider is the only
// thing that catches that, and it belongs in the M2 harness rather than here.

const braveFixture = `{
  "web": {
    "results": [
      {
        "title": "MambaByte: Token-free Selective State Space Model",
        "url": "https://arxiv.org/abs/2401.13660",
        "description": "We show that MambaByte achieves 1.31 BPB on PG-19.",
        "page_age": "2024-01-24T00:00:00"
      },
      {
        "title": "Byte-level models overview",
        "url": "https://example.com/byte-level",
        "description": "A survey of tokenizer-free architectures.",
        "age": "3 days ago"
      },
      { "title": "no url here", "url": "" }
    ]
  }
}`

const tavilyFixture = `{
  "query": "byte level llm",
  "results": [
    {
      "title": "MambaByte",
      "url": "https://arxiv.org/abs/2401.13660",
      "content": "Short query-focused extract.",
      "raw_content": "FULL PAGE TEXT. MambaByte achieves 1.31 bits per byte on PG-19, outperforming comparable subword models at equal compute. This is long enough to be treated as usable content by the actor and therefore skip a fetch entirely, which is the whole point of preferring a provider that returns page text.",
      "score": 0.97,
      "published_date": "2024-01-24"
    },
    {
      "title": "Snippet only",
      "url": "https://example.com/short",
      "content": "tiny",
      "raw_content": "",
      "score": 0.4
    }
  ]
}`

func serve(t *testing.T, status int, body string, capture func(*http.Request)) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if capture != nil {
			capture(r)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func fastLimit() limiter.Limit { return limiter.Limit{Rate: 1000, Burst: 1000} }

// ---------------------------------------------------------------------------
// Brave
// ---------------------------------------------------------------------------

func TestBraveParsesResults(t *testing.T) {
	var got *http.Request
	srv := serve(t, 200, braveFixture, func(r *http.Request) { got = r.Clone(r.Context()) })

	p, err := search.New(search.Config{
		Provider: search.KindBrave, APIKey: "test-key",
		BaseURL: srv.URL, RateLimit: fastLimit(),
	}, srv.Client())
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	resp, err := p.Search(context.Background(), "byte level llm", search.Options{MaxResults: 5})
	if err != nil {
		t.Fatalf("search: %v", err)
	}

	// The empty-URL row is dropped rather than surfacing as a result the actor
	// would then try to fetch.
	if len(resp.Results) != 2 {
		t.Fatalf("got %d results, want 2", len(resp.Results))
	}
	r0 := resp.Results[0]
	if r0.URL != "https://arxiv.org/abs/2401.13660" {
		t.Errorf("url = %q", r0.URL)
	}
	if r0.Rank != 1 {
		t.Errorf("rank = %d, want 1", r0.Rank)
	}
	if r0.PublishedAt == nil || r0.PublishedAt.Year() != 2024 {
		t.Errorf("published = %v, want 2024", r0.PublishedAt)
	}

	// Brave never returns page content, so every result implies a fetch.
	if r0.HasUsableContent() {
		t.Error("Brave result reported usable content")
	}

	// A relative date like "3 days ago" is deliberately not parsed: a guessed
	// timestamp would corrupt the supersedes edges §11.2 builds from it.
	if resp.Results[1].PublishedAt != nil {
		t.Errorf("relative age was parsed into %v, want nil", resp.Results[1].PublishedAt)
	}

	if got.Header.Get("X-Subscription-Token") != "test-key" {
		t.Error("subscription token header missing")
	}
	if q := got.URL.Query().Get("count"); q != "5" {
		t.Errorf("count = %q, want 5", q)
	}
}

func TestBraveExcludeDomainsUsesQueryOperator(t *testing.T) {
	var got *http.Request
	srv := serve(t, 200, braveFixture, func(r *http.Request) { got = r.Clone(r.Context()) })

	p, _ := search.New(search.Config{
		Provider: search.KindBrave, APIKey: "k", BaseURL: srv.URL, RateLimit: fastLimit(),
	}, srv.Client())

	_, err := p.Search(context.Background(), "topic", search.Options{ExcludeDomains: []string{"spam.example"}})
	if err != nil {
		t.Fatal(err)
	}
	if q := got.URL.Query().Get("q"); !strings.Contains(q, "-site:spam.example") {
		t.Errorf("query = %q, want a -site: operator", q)
	}
}

// TestBraveIncludeDomainsFilteredClientSide: Brave has no include-domains
// parameter, so the option is honoured locally rather than silently dropped.
func TestBraveIncludeDomainsFilteredClientSide(t *testing.T) {
	srv := serve(t, 200, braveFixture, nil)
	p, _ := search.New(search.Config{
		Provider: search.KindBrave, APIKey: "k", BaseURL: srv.URL, RateLimit: fastLimit(),
	}, srv.Client())

	resp, err := p.Search(context.Background(), "topic", search.Options{
		IncludeDomains: []string{"arxiv.org"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Results) != 1 {
		t.Fatalf("got %d results, want 1 after include filter", len(resp.Results))
	}
	if !strings.Contains(resp.Results[0].URL, "arxiv.org") {
		t.Errorf("kept the wrong result: %s", resp.Results[0].URL)
	}
}

// ---------------------------------------------------------------------------
// Tavily
// ---------------------------------------------------------------------------

func TestTavilyPrefersRawContent(t *testing.T) {
	var body []byte
	var authHeader string
	srv := serve(t, 200, tavilyFixture, func(r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		authHeader = r.Header.Get("Authorization")
	})

	p, err := search.New(search.Config{
		Provider: search.KindTavily, APIKey: "tvly-test",
		BaseURL: srv.URL, RateLimit: fastLimit(),
	}, srv.Client())
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	resp, err := p.Search(context.Background(), "byte level llm", search.Options{
		MaxResults: 2, IncludeContent: true, Deep: true,
	})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(resp.Results) != 2 {
		t.Fatalf("got %d results, want 2", len(resp.Results))
	}

	// raw_content wins over the query-focused extract: a claim cannot be
	// grounded against text someone else already filtered for relevance.
	r0 := resp.Results[0]
	if !strings.HasPrefix(r0.Content, "FULL PAGE TEXT") {
		t.Errorf("content = %.40q, want raw_content", r0.Content)
	}
	if r0.Snippet != "Short query-focused extract." {
		t.Errorf("snippet = %q", r0.Snippet)
	}

	// This is the efficiency the provider is chosen for: enough text to skip
	// the fetch on result 1, not enough on result 2.
	if !r0.HasUsableContent() {
		t.Error("result 1 should carry usable content and skip a fetch")
	}
	if resp.Results[1].HasUsableContent() {
		t.Error("result 2 has only a tiny snippet and should still need a fetch")
	}

	// Auth goes in a header, not the body — header redaction in the cassette
	// layer covers headers, so a key in the body would land on disk.
	if authHeader != "Bearer tvly-test" {
		t.Errorf("Authorization = %q", authHeader)
	}
	if strings.Contains(string(body), "tvly-test") {
		t.Error("API key appeared in the request body")
	}

	var sent map[string]any
	if err := json.Unmarshal(body, &sent); err != nil {
		t.Fatalf("request body: %v", err)
	}
	if sent["search_depth"] != "advanced" {
		t.Errorf("search_depth = %v, want advanced", sent["search_depth"])
	}
	if sent["include_raw_content"] != true {
		t.Errorf("include_raw_content = %v", sent["include_raw_content"])
	}
}

// TestTavilyTopsUpAShortRawContentAnswer.
func TestTavilyTopsUpAShortRawContentAnswer(t *testing.T) {
	const full = `{"results":[
	  {"url":"https://arxiv.org/abs/2401.13660","content":"a","raw_content":"FULL"},
	  {"url":"https://example.com/b","content":"b"},
	  {"url":"https://example.com/c","content":"c"},
	  {"url":"https://example.com/d","content":"d"}]}`
	var sent []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var m map[string]any
		_ = json.NewDecoder(r.Body).Decode(&m)
		sent = append(sent, m)
		w.Header().Set("Content-Type", "application/json")
		if m["include_raw_content"] == true {
			_, _ = io.WriteString(w, tavilyFixture)
			return
		}
		_, _ = io.WriteString(w, full)
	}))
	t.Cleanup(srv.Close)

	p, err := search.New(search.Config{
		Provider: search.KindTavily, APIKey: "tvly-test",
		BaseURL: srv.URL, RateLimit: fastLimit(),
	}, srv.Client())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	resp, err := p.Search(context.Background(), "q", search.Options{MaxResults: 4, IncludeContent: true})
	if err != nil {
		t.Fatalf("search: %v", err)
	}

	if len(sent) != 2 || sent[1]["include_raw_content"] != nil {
		t.Fatalf("requests = %v, want a second one without raw content", sent)
	}
	var urls []string
	for _, r := range resp.Results {
		urls = append(urls, r.URL)
	}
	want := []string{"https://arxiv.org/abs/2401.13660", "https://example.com/short",
		"https://example.com/b", "https://example.com/c"}
	if strings.Join(urls, " ") != strings.Join(want, " ") {
		t.Errorf("urls = %v, want %v", urls, want)
	}
	if !strings.HasPrefix(resp.Results[0].Content, "FULL PAGE TEXT") {
		t.Error("the raw-content result lost its page text")
	}
	if resp.Results[3].Rank != 4 {
		t.Errorf("rank = %d, want 4", resp.Results[3].Rank)
	}
}

// TestTavilyFullRawContentAnswerMakesOneCall.
func TestTavilyFullRawContentAnswerMakesOneCall(t *testing.T) {
	calls := 0
	srv := serve(t, 200, tavilyFixture, func(*http.Request) { calls++ })
	p, err := search.New(search.Config{
		Provider: search.KindTavily, APIKey: "tvly-test",
		BaseURL: srv.URL, RateLimit: fastLimit(),
	}, srv.Client())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if _, err := p.Search(context.Background(), "q", search.Options{MaxResults: 2, IncludeContent: true}); err != nil {
		t.Fatalf("search: %v", err)
	}
	if calls != 1 {
		t.Errorf("calls = %d, want 1", calls)
	}
}

// ---------------------------------------------------------------------------
// Shared behaviour
// ---------------------------------------------------------------------------

func TestErrorsAreClassifiedForTheRetryPolicy(t *testing.T) {
	cases := []struct {
		status    int
		sentinel  error
		retryable bool
	}{
		{429, search.ErrRateLimited, true},
		{401, search.ErrUnauthorized, false},
		{403, search.ErrUnauthorized, false},
		{402, search.ErrQuotaExceeded, false},
		{500, nil, true},
		{503, nil, true},
	}

	for _, kind := range search.Kinds() {
		for _, c := range cases {
			srv := serve(t, c.status, `{"error":"nope"}`, nil)
			p, _ := search.New(search.Config{
				Provider: kind, APIKey: "k", BaseURL: srv.URL, RateLimit: fastLimit(),
			}, srv.Client())

			_, err := p.Search(context.Background(), "q", search.Options{})
			if err == nil {
				t.Fatalf("%s %d: no error", kind, c.status)
			}
			if c.sentinel != nil && !errors.Is(err, c.sentinel) {
				t.Errorf("%s %d: error = %v, want %v", kind, c.status, err, c.sentinel)
			}

			// The executor's error policy (§9.5) branches on this: a 429 backs
			// off and retries, a bad key must not — retrying a configuration
			// mistake just burns budget.
			var apiErr *search.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("%s %d: error = %T, want *search.APIError", kind, c.status, err)
			}
			if apiErr.Retryable() != c.retryable {
				t.Errorf("%s %d: Retryable = %v, want %v", kind, c.status, apiErr.Retryable(), c.retryable)
			}
		}
	}
}

func TestConfigValidation(t *testing.T) {
	if _, err := search.New(search.Config{Provider: "google", APIKey: "k"}, nil); err == nil {
		t.Error("unknown provider accepted")
	}
	if _, err := search.New(search.Config{Provider: search.KindBrave}, nil); err == nil {
		t.Error("missing API key accepted")
	}
	if _, err := search.New(search.Config{Provider: search.KindBrave, APIKey: "  "}, nil); err == nil {
		t.Error("whitespace API key accepted")
	}
}

func TestProviderIsSelectable(t *testing.T) {
	for _, kind := range search.Kinds() {
		p, err := search.New(search.Config{Provider: kind, APIKey: "k"}, nil)
		if err != nil {
			t.Fatalf("%s: %v", kind, err)
		}
		if p.Kind() != kind {
			t.Errorf("Kind() = %q, want %q", p.Kind(), kind)
		}
	}
}

func TestCostIsChargedPerQuery(t *testing.T) {
	srv := serve(t, 200, braveFixture, nil)
	p, _ := search.New(search.Config{
		Provider: search.KindBrave, APIKey: "k", BaseURL: srv.URL,
		CostPerQueryMicros: 4_200, RateLimit: fastLimit(),
	}, srv.Client())

	resp, err := p.Search(context.Background(), "q", search.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Cost.USDMicros != 4_200 {
		t.Errorf("cost = %d micros, want 4200", resp.Cost.USDMicros)
	}
	// Search costs money and no tokens. In token-mode sessions that makes it
	// invisible to the budget, which is exactly why MaxToolCalls exists (§8.5).
	if resp.Cost.TotalTokens() != 0 {
		t.Errorf("tokens = %d, want 0", resp.Cost.TotalTokens())
	}
}

// TestRateLimitIsEnforced: Brave's free tier is one query per second, so the
// limiter default has to be the safe value rather than the fast one.
func TestRateLimitIsEnforced(t *testing.T) {
	srv := serve(t, 200, braveFixture, nil)
	p, _ := search.New(search.Config{
		Provider: search.KindBrave, APIKey: "k", BaseURL: srv.URL,
		RateLimit: limiter.Limit{Rate: 2, Burst: 1},
	}, srv.Client())

	ctx := context.Background()
	if _, err := p.Search(ctx, "first", search.Options{}); err != nil {
		t.Fatal(err)
	}

	start := time.Now()
	if _, err := p.Search(ctx, "second", search.Options{}); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed < 300*time.Millisecond {
		t.Errorf("second query waited %v, want ~500ms at 2/s", elapsed)
	}
}

func TestMaxResultsIsBounded(t *testing.T) {
	var got *http.Request
	srv := serve(t, 200, braveFixture, func(r *http.Request) { got = r.Clone(r.Context()) })
	p, _ := search.New(search.Config{
		Provider: search.KindBrave, APIKey: "k", BaseURL: srv.URL, RateLimit: fastLimit(),
	}, srv.Client())

	if _, err := p.Search(context.Background(), "q", search.Options{MaxResults: 500}); err != nil {
		t.Fatal(err)
	}
	if c := got.URL.Query().Get("count"); c != "20" {
		t.Errorf("count = %q, want it clamped to 20", c)
	}
}

package search

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/tools/limiter"
)

// Tavily Search API.
//
//	POST https://api.tavily.com/search
//	Authorization: Bearer tvly-...
//
// Tavily is built for LLM consumers and returns extracted page content with
// each result. When that content is usable the WebActor skips the fetch
// entirely — no HTTP request, no robots round-trip, no rate-limit pressure, no
// js_required. It costs more per query and buys back more than it costs on any
// result it saves a fetch on.

const (
	tavilyBaseURL = "https://api.tavily.com"
	// Tavily's free tier is generous per month but throttled per second; two
	// per second is a conservative default rather than a documented ceiling.
	tavilyDefaultRate = 2.0
)

type tavilyProvider struct {
	base
}

func newTavily(cfg Config, client *http.Client) Provider {
	return &tavilyProvider{
		base: newBase(cfg, client, tavilyBaseURL, DefaultTavilyCostMicros,
			limiter.Limit{Rate: tavilyDefaultRate, Burst: 2}),
	}
}

type tavilyRequest struct {
	Query             string   `json:"query"`
	MaxResults        int      `json:"max_results"`
	SearchDepth       string   `json:"search_depth"`
	IncludeRawContent bool     `json:"include_raw_content,omitempty"`
	IncludeDomains    []string `json:"include_domains,omitempty"`
	ExcludeDomains    []string `json:"exclude_domains,omitempty"`
}

type tavilyResponse struct {
	Query   string `json:"query"`
	Results []struct {
		Title         string  `json:"title"`
		URL           string  `json:"url"`
		Content       string  `json:"content"`
		RawContent    string  `json:"raw_content"`
		Score         float64 `json:"score"`
		PublishedDate string  `json:"published_date"`
	} `json:"results"`
}

func (p *tavilyProvider) Search(ctx context.Context, query string, opts Options) (*Response, error) {
	opts = opts.withDefaults()
	start := time.Now()

	depth := "basic"
	if opts.Deep {
		depth = "advanced"
	}
	payload := tavilyRequest{
		Query:             query,
		MaxResults:        opts.MaxResults,
		SearchDepth:       depth,
		IncludeRawContent: opts.IncludeContent,
		IncludeDomains:    opts.IncludeDomains,
		ExcludeDomains:    opts.ExcludeDomains,
	}
	parsed, err := p.post(ctx, payload)
	if err != nil {
		return nil, err
	}
	calls := int64(1)
	// Tavily can return fewer results when asked for raw content; the rest come
	// from a second query without it, and the actor fetches those pages itself.
	if payload.IncludeRawContent && len(parsed.Results) < opts.MaxResults {
		payload.IncludeRawContent = false
		if more, err := p.post(ctx, payload); err == nil {
			calls++
			seen := make(map[string]bool, len(parsed.Results))
			for _, r := range parsed.Results {
				seen[r.URL] = true
			}
			for _, r := range more.Results {
				if len(parsed.Results) >= opts.MaxResults {
					break
				}
				if !seen[r.URL] {
					seen[r.URL] = true
					parsed.Results = append(parsed.Results, r)
				}
			}
		}
	}

	out := &Response{
		Query:    query,
		Provider: KindTavily,
		Cost:     core.Cost{USDMicros: p.queryCost().USDMicros * calls},
	}
	for _, r := range parsed.Results {
		if r.URL == "" {
			continue
		}
		// raw_content is the full page; content is Tavily's query-focused
		// extract. Prefer the full text: the actor's own chunker and quote
		// verification want the whole document, not a pre-summarized slice of
		// it — a claim cannot be grounded against text that was already
		// filtered by someone else's relevance judgement.
		text := r.RawContent
		if strings.TrimSpace(text) == "" {
			text = r.Content
		}

		out.Results = append(out.Results, Result{
			URL:         r.URL,
			Title:       r.Title,
			Snippet:     r.Content,
			Content:     text,
			Score:       r.Score,
			PublishedAt: parseTavilyDate(r.PublishedDate),
			Rank:        len(out.Results) + 1,
		})
	}
	out.Elapsed = time.Since(start)
	return out, nil
}

// post sends one search request and decodes the answer.
func (p *tavilyProvider) post(ctx context.Context, payload tavilyRequest) (*tavilyResponse, error) {
	if err := p.wait(ctx); err != nil {
		return nil, err
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("search: tavily encode: %w", err)
	}

	ctx, cancel := context.WithTimeout(ctx, p.timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/search", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("search: tavily request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	// Bearer is the current auth scheme. Older examples put api_key in the
	// body; that form is not sent here, so the key never lands in a cassette
	// body where header redaction would not reach it.
	req.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("search: tavily: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return nil, fmt.Errorf("search: tavily read: %w", err)
	}
	if err := classifyStatus(KindTavily, resp.StatusCode, string(respBody)); err != nil {
		return nil, err
	}

	var parsed tavilyResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return nil, fmt.Errorf("search: tavily decode: %w", err)
	}
	return &parsed, nil
}

func parseTavilyDate(s string) *time.Time {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	for _, layout := range []string{time.RFC3339, "2006-01-02", "Mon, 02 Jan 2006 15:04:05 MST"} {
		if t, err := time.Parse(layout, s); err == nil {
			return &t
		}
	}
	return nil
}

// ---------------------------------------------------------------------------

// domainFilter applies include-domain restrictions client-side for providers
// that have no server-side equivalent.
type domainFilter struct{ suffixes []string }

func newDomainFilter(domains []string) *domainFilter {
	if len(domains) == 0 {
		return nil
	}
	s := make([]string, 0, len(domains))
	for _, d := range domains {
		if d = strings.ToLower(strings.TrimSpace(strings.TrimPrefix(d, "."))); d != "" {
			s = append(s, d)
		}
	}
	return &domainFilter{suffixes: s}
}

// allows reports whether rawURL is on the include list. A nil filter allows
// everything.
func (f *domainFilter) allows(rawURL string) bool {
	if f == nil || len(f.suffixes) == 0 {
		return true
	}
	host := hostOf(rawURL)
	if host == "" {
		return false
	}
	for _, s := range f.suffixes {
		if host == s || strings.HasSuffix(host, "."+s) {
			return true
		}
	}
	return false
}

func hostOf(rawURL string) string {
	i := strings.Index(rawURL, "://")
	if i < 0 {
		return ""
	}
	rest := rawURL[i+3:]
	if j := strings.IndexAny(rest, "/?#"); j >= 0 {
		rest = rest[:j]
	}
	if j := strings.LastIndex(rest, "@"); j >= 0 {
		rest = rest[j+1:]
	}
	if j := strings.LastIndex(rest, ":"); j >= 0 && !strings.Contains(rest[j:], "]") {
		rest = rest[:j]
	}
	return strings.ToLower(strings.Trim(rest, "[]"))
}

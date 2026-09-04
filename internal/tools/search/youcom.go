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

	"github.com/lajosdeme/mole/internal/tools/limiter"
)

// You.com Search API.
//
//	POST https://ydc-index.io/v1/search
//	X-API-Key: <key>
//
// Like Tavily, You.com can return page content with each result — set
// extraction_mode to "full_page" and each web result carries a contents.markdown
// field. When that text is usable the WebActor skips the fetch entirely, which
// is the trade the package docs describe as the first thing to try (§10.4).
// The extraction is billed per page on top of the per-query price, so the
// default cost below is the search-only tier and the per-page charge is left
// to the user's CostPerQueryMicros override rather than guessed here.
//
// Unlike Brave, domain filters are supported server-side, so include and
// exclude both travel in the request instead of being emulated client-side.

const (
	youcomBaseURL = "https://ydc-index.io"
	// You.com documents 5 USD per 1000 search calls at the entry tier.
	youcomDefaultCostMicros = 5_000
	// No per-second ceiling is documented; two per second is the same
	// conservative default Tavily gets rather than a claimed limit.
	youcomDefaultRate = 2.0
)

type youcomProvider struct {
	base
}

func newYoucom(cfg Config, client *http.Client) Provider {
	return &youcomProvider{
		base: newBase(cfg, client, youcomBaseURL, DefaultYoucomCostMicros,
			limiter.Limit{Rate: youcomDefaultRate, Burst: 2}),
	}
}

type youcomRequest struct {
	Query          string   `json:"query"`
	Count          int      `json:"count,omitempty"`
	ExtractionMode string   `json:"extraction_mode,omitempty"`
	IncludeDomains []string `json:"include_domains,omitempty"`
	ExcludeDomains []string `json:"exclude_domains,omitempty"`
}

// youcomResponse mirrors the documented shape. results.web carries the hits;
// the news channel is skipped, matching Brave's explicit result_filter=web.
// Unknown fields are ignored, so a provider-side addition does not break
// parsing.
type youcomResponse struct {
	Results struct {
		Web []struct {
			URL         string   `json:"url"`
			Title       string   `json:"title"`
			Description string   `json:"description"`
			Snippets    []string `json:"snippets"`
			PageAge     string   `json:"page_age"`
			Contents    struct {
				Markdown string `json:"markdown"`
			} `json:"contents"`
		} `json:"web"`
	} `json:"results"`
}

func (p *youcomProvider) Search(ctx context.Context, query string, opts Options) (*Response, error) {
	opts = opts.withDefaults()
	start := time.Now()

	if err := p.wait(ctx); err != nil {
		return nil, err
	}

	payload := youcomRequest{
		Query:          query,
		Count:          opts.MaxResults,
		IncludeDomains: opts.IncludeDomains,
		ExcludeDomains: opts.ExcludeDomains,
	}
	if opts.IncludeContent {
		payload.ExtractionMode = "full_page"
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("search: youcom encode: %w", err)
	}

	ctx, cancel := context.WithTimeout(ctx, p.timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/v1/search", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("search: youcom request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	// The key goes in a header, never the body — the cassette redaction list
	// covers X-API-Key, so a key in the body would land on disk where header
	// redaction would not reach it.
	req.Header.Set("X-API-Key", p.apiKey)

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("search: youcom: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return nil, fmt.Errorf("search: youcom read: %w", err)
	}
	if err := classifyStatus(KindYoucom, resp.StatusCode, string(respBody)); err != nil {
		return nil, err
	}

	var parsed youcomResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return nil, fmt.Errorf("search: youcom decode: %w", err)
	}

	out := &Response{
		Query:    query,
		Provider: KindYoucom,
		Cost:     p.queryCost(),
	}
	for _, r := range parsed.Results.Web {
		if r.URL == "" {
			continue
		}
		// contents.markdown is the full page, present only when
		// extraction_mode was full_page. A claim cannot be grounded against
		// text someone else already filtered for relevance, so the full text
		// is preferred and the query-focused snippets serve as the result's
		// description when the provider's own description is thin.
		snippet := r.Description
		if snippet == "" && len(r.Snippets) > 0 {
			snippet = r.Snippets[0]
		}
		out.Results = append(out.Results, Result{
			URL:         r.URL,
			Title:       r.Title,
			Snippet:     snippet,
			Content:     r.Contents.Markdown,
			PublishedAt: parseYoucomDate(r.PageAge),
			Rank:        len(out.Results) + 1,
		})
	}
	out.Elapsed = time.Since(start)
	return out, nil
}

// parseYoucomDate reads the page_age field. You.com returns ISO timestamps
// like 2025-11-15T10:30:00; a value that does not parse is left nil rather
// than guessed — a fabricated timestamp would corrupt the supersedes edges
// §11.2 builds from PublishedAt.
func parseYoucomDate(s string) *time.Time {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	for _, layout := range []string{time.RFC3339, "2006-01-02T15:04:05", "2006-01-02"} {
		if t, err := time.Parse(layout, s); err == nil {
			return &t
		}
	}
	return nil
}

// Package search turns a question into candidate URLs.
//
// Two providers ship: Brave and Tavily. They are not interchangeable in one
// respect that matters more than price — Tavily returns extracted page content
// alongside each result, so a result that arrives with usable text skips the
// fetch entirely. §10.4 lists that as the first thing to try before adding any
// fetch capability, because it removes latency, failure modes, and rate-limit
// pressure all at once.
package search

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/tools/fetch"
	"github.com/lajosdeme/mole/internal/tools/limiter"
)

// Kind names a provider.
type Kind string

const (
	KindBrave  Kind = "brave"
	KindTavily Kind = "tavily"
)

func (k Kind) Valid() bool { return k == KindBrave || k == KindTavily }

// Kinds lists the providers this build supports.
func Kinds() []Kind { return []Kind{KindBrave, KindTavily} }

// Result is one search hit.
type Result struct {
	URL     string
	Title   string
	Snippet string

	// Content is the provider's own extraction of the page.
	//
	// Brave never populates it. Tavily does when asked, and when it is present
	// and long enough the WebActor can skip the fetch — which is the single
	// biggest efficiency difference between the two providers.
	Content string

	// PublishedAt feeds Claim.PublishedAt, which §11.2 needs to tell staleness
	// apart from genuine disagreement. Providers report it unevenly.
	PublishedAt *time.Time

	Score float64
	Rank  int
}

// HasUsableContent reports whether the provider handed back enough text to work
// with directly. The floor mirrors the fetcher's own minimum for usable text.
func (r Result) HasUsableContent() bool {
	return len(strings.TrimSpace(r.Content)) >= fetch.MinUsableText
}

// Response is one search call.
type Response struct {
	Query    string
	Provider Kind
	Results  []Result
	Cost     core.Cost
	Elapsed  time.Duration
}

// Options tune a single search.
type Options struct {
	MaxResults int
	// IncludeContent asks the provider for page text. Ignored by providers that
	// cannot supply it.
	IncludeContent bool
	// Deep requests a more thorough (and more expensive) search where the
	// provider offers the choice.
	Deep bool
	// IncludeDomains / ExcludeDomains restrict the result set where supported.
	IncludeDomains []string
	ExcludeDomains []string
	// NoTopUp sends one request even when a raw-content answer comes back short.
	NoTopUp bool
}

func (o Options) withDefaults() Options {
	if o.MaxResults <= 0 {
		o.MaxResults = 10
	}
	if o.MaxResults > 20 {
		o.MaxResults = 20
	}
	return o
}

// Provider is a search backend.
type Provider interface {
	// Search returns ranked results. Cost is reported on the Response so the
	// ledger can settle against the reservation that covered this lead.
	Search(ctx context.Context, query string, opts Options) (*Response, error)
	Kind() Kind
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

var (
	// ErrRateLimited is a 429. The executor's error policy treats it as
	// transient and retries with backoff (§9.5).
	ErrRateLimited = errors.New("search: rate limited")
	// ErrUnauthorized is a bad or missing key — fatal, not transient. Retrying
	// it just burns the budget on a configuration mistake.
	ErrUnauthorized = errors.New("search: unauthorized")
	// ErrQuotaExceeded means the plan's allowance is gone. Also fatal.
	ErrQuotaExceeded = errors.New("search: quota exceeded")
)

// APIError carries provider detail alongside a sentinel.
type APIError struct {
	Provider Kind
	Status   int
	Body     string
	sentinel error
}

func (e *APIError) Error() string {
	msg := fmt.Sprintf("search: %s returned %d", e.Provider, e.Status)
	if e.Body != "" {
		body := e.Body
		if len(body) > 200 {
			body = body[:200] + "…"
		}
		msg += ": " + body
	}
	return msg
}

func (e *APIError) Unwrap() error { return e.sentinel }

// Retryable reports whether the executor should back off and try again rather
// than fail the lead.
func (e *APIError) Retryable() bool {
	return errors.Is(e.sentinel, ErrRateLimited) || (e.Status >= 500 && e.Status < 600)
}

// statusUsageLimit is Tavily's "exceeds your plan's set usage limit".
const statusUsageLimit = 432

func classifyStatus(kind Kind, status int, body string) error {
	if status >= 200 && status < 300 {
		return nil
	}
	e := &APIError{Provider: kind, Status: status, Body: body}
	switch {
	case status == http.StatusTooManyRequests:
		e.sentinel = ErrRateLimited
	case status == http.StatusUnauthorized, status == http.StatusForbidden:
		e.sentinel = ErrUnauthorized
	case status == http.StatusPaymentRequired, status == statusUsageLimit:
		e.sentinel = ErrQuotaExceeded
	}
	return e
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Config selects and configures a provider.
type Config struct {
	// Provider is which backend to use. Required.
	Provider Kind
	// APIKey for the selected provider.
	APIKey string

	// BaseURL overrides the provider endpoint, for tests and proxies.
	BaseURL string

	// CostPerQueryMicros prices one search in micro-dollars. Defaults come from
	// each provider's published entry tier, but plans differ enough that this
	// is configuration rather than a constant — an unpriced call would make the
	// budget ceiling unenforceable for the search half of a session.
	CostPerQueryMicros int64

	// RateLimit throttles calls to the provider. Defaults are the documented
	// free-tier ceilings, which are low enough that exceeding them is the
	// common failure rather than an edge case.
	RateLimit limiter.Limit

	Timeout time.Duration
}

// Default per-query prices, in micro-dollars, from each provider's entry paid
// tier. Verify against the current plan before trusting a budget built on them.
const (
	DefaultBraveCostMicros  = 5_000 // ~$5 / 1000 queries
	DefaultTavilyCostMicros = 8_000 // ~$8 / 1000 queries
)

// New builds the configured provider.
//
// client carries the record/replay cassette transport, so search calls are
// replayable in tests and in the eval harness.
func New(cfg Config, client *http.Client) (Provider, error) {
	if !cfg.Provider.Valid() {
		return nil, fmt.Errorf("search: unknown provider %q (want one of %v)", cfg.Provider, Kinds())
	}
	if strings.TrimSpace(cfg.APIKey) == "" {
		return nil, fmt.Errorf("search: no API key configured for %s", cfg.Provider)
	}
	if client == nil {
		client = &http.Client{}
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 20 * time.Second
	}

	switch cfg.Provider {
	case KindBrave:
		return newBrave(cfg, client), nil
	case KindTavily:
		return newTavily(cfg, client), nil
	default:
		return nil, fmt.Errorf("search: unhandled provider %q", cfg.Provider)
	}
}

// base holds what both providers share.
type base struct {
	kind    Kind
	apiKey  string
	baseURL string
	client  *http.Client
	lim     *limiter.Limiter
	cost    int64
	timeout time.Duration
}

func newBase(cfg Config, client *http.Client, defaultURL string, defaultCost int64, defaultLimit limiter.Limit) base {
	url := cfg.BaseURL
	if url == "" {
		url = defaultURL
	}
	cost := cfg.CostPerQueryMicros
	if cost <= 0 {
		cost = defaultCost
	}
	lim := cfg.RateLimit
	if lim.Rate <= 0 && lim.MinInterval <= 0 {
		lim = defaultLimit
	}

	l := limiter.New(lim)
	return base{
		kind:    cfg.Provider,
		apiKey:  cfg.APIKey,
		baseURL: strings.TrimSuffix(url, "/"),
		client:  client,
		lim:     l,
		cost:    cost,
		timeout: cfg.Timeout,
	}
}

func (b *base) Kind() Kind { return b.kind }

// wait applies the provider rate limit. Brave's free tier is one query per
// second; exceeding it is a 429, not a slow response.
func (b *base) wait(ctx context.Context) error {
	return b.lim.Wait(ctx, string(b.kind))
}

// cost returns the charge for one query. Search costs dollars and no tokens,
// so in token-mode sessions it is bounded by MaxToolCalls rather than by the
// budget — see §8.5.
func (b *base) queryCost() core.Cost {
	return core.Cost{USDMicros: b.cost}
}

// Package pricing converts model token usage into money.
//
// Rates are stored as nano-dollars per token (int64), not dollars per million
// tokens (float64). At $5/MTok an input token is exactly 5000 nano-dollars and
// a cache read is exactly 500 — both integers, so the cost of a call is exact
// rather than a float that drifts once summed across a long session.
//
// The table is data, not code: a self-hosted model registered with all-zero
// rates costs $0 and still records real token counts, which is what makes
// token-mode budgeting meaningful for users who aren't paying per token.
package pricing

import (
	"errors"
	"fmt"
	"net/url"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/lajosdeme/mole/internal/core"
)

// NanoPerMicro is the scale between the rate unit and the money unit.
const NanoPerMicro = 1000

// Rates is the per-token price of one model, in nano-dollars.
type Rates struct {
	Input               int64
	Output              int64
	CacheRead           int64
	CacheWrite          int64 // 5-minute TTL
	CacheWrite1h        int64
	LongPromptThreshold int64  // prompt tokens threshold for tiered pricing (0 = no tiering)
	LongPrompt          *Rates // rates for prompts exceeding threshold
}

// RateConfig is a rate as somebody writes it down: dollars per MILLION tokens,
// the number printed on the vendor's own pricing page.
//
// Rates internally are nano-dollars per token, and nobody should have to know
// that to configure one. The conversion is this type's whole job.
type RateConfig struct {
	Input  float64 `json:"input"`
	Output float64 `json:"output"`

	// Cache rates, when a provider bills them other than by the usual contract.
	// Zero takes the same multipliers the built-in table assumes: a read at a
	// tenth of input, a write at 1.25x for the short TTL and 2x for the long
	// one.
	//
	// A write rate of zero would mean free, which is the one value that must not
	// be the default: a provider that starts reporting cache-creation tokens
	// would get them for nothing.
	CacheRead    float64 `json:"cache_read,omitempty"`
	CacheWrite   float64 `json:"cache_write,omitempty"`
	CacheWrite1h float64 `json:"cache_write_1h,omitempty"`
}

// Rates converts to the internal form.
func (rc RateConfig) Rates() Rates {
	r := perMTok(rc.Input, rc.Output)
	if rc.CacheRead > 0 {
		r.CacheRead = perToken(rc.CacheRead)
	}
	if rc.CacheWrite > 0 {
		r.CacheWrite = perToken(rc.CacheWrite)
	}
	if rc.CacheWrite1h > 0 {
		r.CacheWrite1h = perToken(rc.CacheWrite1h)
	}
	return r
}

// validate refuses a rate that cannot bill correctly.
//
// At load rather than at the first call, for the same reason SetFallbackModel
// refuses an unpriced fallback: finding out during a run means finding out
// after it has started spending.
func (rc RateConfig) validate() error {
	switch {
	case rc.Input <= 0:
		return fmt.Errorf("pricing: input rate must be above zero, got %v", rc.Input)
	case rc.Output < 0:
		return fmt.Errorf("pricing: output rate cannot be negative, got %v", rc.Output)
	case rc.CacheRead < 0 || rc.CacheWrite < 0 || rc.CacheWrite1h < 0:
		return errors.New("pricing: a cache rate cannot be negative")
	}
	return nil
}

// perToken converts dollars per million tokens to nano-dollars per token.
func perToken(usdPerMTok float64) int64 { return int64(usdPerMTok * 1000) }

// Usage is the raw token count returned by a provider.
type Usage struct {
	InputTokens      int64
	OutputTokens     int64
	CacheReadTokens  int64
	CacheWriteTokens int64
	CacheWriteTTL1h  bool
}

// perMTok builds Rates from the published dollars-per-million-tokens figures.
// Cache multipliers are fixed by the API contract: reads bill at 0.1x input,
// writes at 1.25x (5m TTL) and 2x (1h TTL).
func perMTok(inputUSD, outputUSD float64) Rates {
	in := int64(inputUSD * 1000)
	return Rates{
		Input:        in,
		Output:       int64(outputUSD * 1000),
		CacheRead:    in / 10,
		CacheWrite:   in * 125 / 100,
		CacheWrite1h: in * 2,
	}
}

// defaultTable holds first-party API rates. Partner platforms (Bedrock, Vertex)
// price separately and should be registered explicitly.
func defaultTable() map[string]Rates {
	return map[string]Rates{
		// DeepSeek, from api-docs.deepseek.com/quick_start/pricing:
		//
		//	deepseek-v4-flash  $0.14/MTok in   $0.0028/MTok cache hit  $0.28/MTok out
		//	deepseek-v4-pro    $0.435/MTok in  $0.003625/MTok cache hit $0.87/MTok out
		//
		// Aliases are registered alongside the real names: the ledger prices what
		// the API answered with, the pre-flight check reads the configured name,
		// and the two differ. deepseek-reasoner is registered at PRO rates even
		// though it resolves to flash today — an alias is DeepSeek's to repoint,
		// and an estimate that is too high costs an under-used budget where one
		// too low lets work start that cannot be paid for.
		//
		// Not built with perMTok, whose cache multipliers are Anthropic's: a
		// DeepSeek cache hit is 2% of a miss, not 10%. Cache reads round UP to the
		// nano-dollar, and the write rates equal Input because a miss is billed as
		// input — zero would hand cache-creation tokens away free.
		"deepseek-v4-flash": {Input: 140, Output: 280, CacheRead: 3, CacheWrite: 140, CacheWrite1h: 140},
		// What the API began answering with in October 2026, where it had said
		// "deepseek-v4-flash" before. Registered because a rename is theirs to
		// make; the fallback above is what stops the NEXT one costing anything.
		"deepseek-flash": {Input: 140, Output: 280, CacheRead: 3, CacheWrite: 140, CacheWrite1h: 140},
		"deepseek-chat":  {Input: 140, Output: 280, CacheRead: 3, CacheWrite: 140, CacheWrite1h: 140},

		"deepseek-v4-pro":   {Input: 435, Output: 870, CacheRead: 4, CacheWrite: 435, CacheWrite1h: 435},
		"deepseek-reasoner": {Input: 435, Output: 870, CacheRead: 4, CacheWrite: 435, CacheWrite1h: 435},

		"claude-fable-5":    perMTok(10, 50),
		"claude-fable-5-1":  {Input: 10000, Output: 50000, CacheRead: 250, CacheWrite: 12500, CacheWrite1h: 20000},
		"claude-mythos-5":   perMTok(10, 50),
		"claude-mythos-5-1": {Input: 10000, Output: 50000, CacheRead: 250, CacheWrite: 12500, CacheWrite1h: 20000},

		"claude-opus-5":     perMTok(5, 25),
		"claude-opus-5-5":   {Input: 4000, Output: 20000, CacheRead: 200, CacheWrite: 5000, CacheWrite1h: 8000},
		"claude-opus-4-8":   perMTok(5, 25),
		"claude-opus-4-7":   perMTok(5, 25),
		"claude-opus-4-6":   perMTok(5, 25),

		"claude-sonnet-5":   perMTok(2, 10),
		"claude-sonnet-5-5": {Input: 2000, Output: 10000, CacheRead: 100, CacheWrite: 2500, CacheWrite1h: 4000},
		"claude-sonnet-4-6": perMTok(3, 15),

		"claude-haiku-4-5": perMTok(1, 5),
		"claude-haiku-5-5": {
			Input:               100,
			Output:              500,
			CacheRead:           10,
			CacheWrite:          125,
			CacheWrite1h:        200,
			LongPromptThreshold: 100_000,
			LongPrompt: &Rates{
				Input:        500,
				Output:       2500,
				CacheRead:    50,
				CacheWrite:   625,
				CacheWrite1h: 1000,
			},
		},

		// OpenAI, which detect.go selects from an OPENAI_API_KEY. Rates are the
		// published dollars-per-MTok as of 2026-10; an operator who finds them
		// stale overrides them in config rather than waiting for a release.
		//
		// Not perMTok: that helper encodes Anthropic's cache contract, and
		// OpenAI's is different. A cached input token bills at half, not a
		// tenth, and there is no separate charge to CREATE the cache — a miss is
		// billed as ordinary input, so CacheWrite equals Input rather than being
		// zero. Zero would hand free tokens to a provider that starts reporting
		// cache-creation counts.
		"gpt-4o":      {Input: 2500, Output: 10_000, CacheRead: 1250, CacheWrite: 2500, CacheWrite1h: 2500},
		"gpt-4o-mini": {Input: 150, Output: 600, CacheRead: 75, CacheWrite: 150, CacheWrite1h: 150},
	}
}

// Declare registers operator-supplied rates, overriding anything built in — so
// a price that changes is correctable without waiting for a release.
//
// All or nothing: one bad rate rejects the batch.
func (t *Table) Declare(rates map[string]RateConfig) error {
	for model, rc := range rates {
		if model == "" {
			return errors.New("pricing: a declared rate names no model")
		}
		if err := rc.validate(); err != nil {
			return fmt.Errorf("%s: %w", model, err)
		}
	}
	for model, rc := range rates {
		t.Register(model, rc.Rates())
	}
	return nil
}

// Table maps model IDs to rates. Safe for concurrent use.
type Table struct {
	mu    sync.RWMutex
	rates map[string]Rates
	// strict makes an unknown model an error rather than a silent $0. Default
	// true: silently pricing an unknown model at zero would make the budget
	// ceiling unenforceable for exactly the model you forgot to register.
	strict bool
	// fallback prices a response naming a model nobody registered; empty leaves
	// that an error.
	//
	// The name a provider answers with is theirs to repoint without telling
	// anyone, and an unpriced call ledgers at zero, which stops --usd binding at
	// all. An estimated rate is still a budget; zero is not.
	fallback string
}

func NewTable() *Table {
	return &Table{rates: defaultTable(), strict: true}
}

// SetFallbackModel names the model whose rates price an unregistered one.
// Refused when the fallback itself has no rates. Empty clears it.
func (t *Table) SetFallbackModel(model string) error {
	if model == "" {
		t.mu.Lock()
		t.fallback = ""
		t.mu.Unlock()
		return nil
	}
	if _, ok := t.Lookup(normalize(model)); !ok {
		return &ErrUnknownModel{Model: model}
	}
	t.mu.Lock()
	t.fallback = model
	t.mu.Unlock()
	return nil
}

// NewEmptyTable returns a table with no models registered — the starting point
// for a purely self-hosted deployment.
func NewEmptyTable() *Table {
	return &Table{rates: map[string]Rates{}, strict: true}
}

// SetStrict controls whether unknown models error (true) or price at zero.
func (t *Table) SetStrict(strict bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.strict = strict
}

// Register adds or replaces a model's rates.
func (t *Table) Register(model string, r Rates) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.rates[model] = r
}

// RegisterFree registers a model that costs nothing — a local or self-hosted
// model whose tokens should still be counted and budgeted.
func (t *Table) RegisterFree(model string) { t.Register(model, Rates{}) }

// RegisterLocal prices every model served by a local runtime at zero.
//
// A self-hosted model costs electricity, not tokens, and mole's own logs said so
// several hundred times per run: every call against ollama emitted "model not in
// pricing table; USD cost recorded as zero", which is true, unhelpful, and drowns
// the warning it shares a channel with — the one about a HOSTED model nobody
// registered, where zero cost really does mean an unenforceable ceiling.
//
// Registered by base URL rather than by name, because a local runtime serves
// whatever the user pulled and the names are theirs. The rates stay zero and the
// TOKENS are still counted, which is what makes --tokens meaningful for somebody
// paying no money at all.
func (t *Table) RegisterLocal(model string) { t.RegisterFree(model) }

// IsLocalEndpoint reports whether a base URL is a loopback address.
//
// Deliberately narrow: loopback only. A model served from another machine on the
// network may well be metered by whoever runs it, and silently pricing it at zero
// would be mole deciding somebody else's costs are zero.
func IsLocalEndpoint(baseURL string) bool {
	u, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil {
		return false
	}
	host := u.Hostname()
	return host == "localhost" || host == "127.0.0.1" || host == "::1" || host == "[::1]"
}

// Lookup finds rates for a model, falling back to its undated base ID.
//
// Anthropic publishes an alias and a dated snapshot that resolve to the same
// weights and the same price — "claude-haiku-4-5" and "claude-haiku-4-5-20251001"
// are one model, and the table registers the alias. Without the fallback, a config
// naming the snapshot prices at nothing, which is worse than it sounds: --usd
// refuses to run at all rather than bound a session it cannot cost. That is what
// happened the first time this project pointed at a hosted provider.
//
// Exact match first, so a snapshot that ever does diverge in price can be
// registered explicitly and win.
func (t *Table) Lookup(model string) (Rates, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if r, ok := t.rates[model]; ok {
		return r, true
	}
	if base, ok := stripDateSuffix(model); ok {
		r, ok := t.rates[base]
		return r, ok
	}
	return Rates{}, false
}

// stripDateSuffix removes a trailing -YYYYMMDD, reporting whether it found one.
//
// Deliberately strict about the shape: eight digits that parse as a plausible
// date. A looser rule would fold "gpt-4-32768" onto "gpt-4" and price a model
// against a different one's rates, and mispricing silently is the failure this
// whole package exists to prevent.
func stripDateSuffix(model string) (string, bool) {
	i := strings.LastIndexByte(model, '-')
	if i <= 0 || len(model)-i-1 != 8 {
		return "", false
	}
	digits := model[i+1:]
	for _, c := range digits {
		if c < '0' || c > '9' {
			return "", false
		}
	}
	if _, err := time.Parse("20060102", digits); err != nil {
		return "", false
	}
	return model[:i], true
}

// Models lists registered model IDs, sorted.
func (t *Table) Models() []string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make([]string, 0, len(t.rates))
	for m := range t.rates {
		out = append(out, m)
	}
	sort.Strings(out)
	return out
}

// ErrUnknownModel is returned by Cost when a strict table has no rates for the
// model.
type ErrUnknownModel struct{ Model string }

func (e *ErrUnknownModel) Error() string {
	return fmt.Sprintf("pricing: no rates registered for model %q", e.Model)
}

// Cost prices a usage record. The returned Cost carries both the money and the
// token breakdown, so the same row serves USD-mode and token-mode sessions.
func (t *Table) Cost(model string, u Usage) (core.Cost, error) {
	c := core.Cost{
		InputTokens:      u.InputTokens,
		OutputTokens:     u.OutputTokens,
		CacheReadTokens:  u.CacheReadTokens,
		CacheWriteTokens: u.CacheWriteTokens,
	}

	r, ok := t.Lookup(normalize(model))
	if !ok {
		t.mu.RLock()
		strict, fallback := t.strict, t.fallback
		t.mu.RUnlock()
		// The configured model's rate, when there is one. SetFallbackModel has
		// already checked it is registered, so this lookup cannot miss unless
		// somebody unregistered it afterwards — in which case strict applies.
		if fallback != "" {
			if fr, found := t.Lookup(normalize(fallback)); found {
				r, ok = fr, true
			}
		}
		if !ok {
			if strict {
				return c, &ErrUnknownModel{Model: model}
			}
			return c, nil
		}
	}

	// Use long-prompt rates if available and total prompt tokens exceed threshold.
	if r.LongPrompt != nil && r.LongPromptThreshold > 0 {
		promptTokens := u.InputTokens + u.CacheReadTokens + u.CacheWriteTokens
		if promptTokens > r.LongPromptThreshold {
			r = *r.LongPrompt
		}
	}

	write := r.CacheWrite
	if u.CacheWriteTTL1h {
		write = r.CacheWrite1h
	}

	nano := u.InputTokens*r.Input +
		u.OutputTokens*r.Output +
		u.CacheReadTokens*r.CacheRead +
		u.CacheWriteTokens*write

	c.USDMicros = divRoundHalfUp(nano, NanoPerMicro)
	return c, nil
}

// divRoundHalfUp avoids systematically under-billing: plain integer division
// truncates, which over thousands of calls quietly understates spend and lets a
// session drift past its ceiling.
func divRoundHalfUp(n, d int64) int64 {
	if n >= 0 {
		return (n + d/2) / d
	}
	return -((-n + d/2) / d)
}

// normalize strips the context-window suffix some deployments append
// (e.g. "claude-opus-5[1m]") so a tagged model still prices correctly.
func normalize(model string) string {
	if i := strings.IndexByte(model, '['); i > 0 {
		return model[:i]
	}
	return model
}

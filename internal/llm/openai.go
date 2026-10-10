package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// OpenAI-compatible backend.
//
// Covers DeepSeek, Ollama, llama.cpp, vLLM, LiteLLM, Together, and anything
// else speaking /v1/chat/completions. Written against the wire format rather
// than an SDK: the format is small and stable, and every vendor SDK layers its
// own auth and retry behaviour on top of the same three fields.
//
// This path is what makes token-mode budgeting honest for self-hosted users. A
// local model registered at zero rates costs nothing and still reports tokens,
// so the ceiling means something even when no money is involved.

const openAIDefaultBaseURL = "https://api.openai.com/v1"

type openAIProvider struct {
	cfg    Config
	client *http.Client
	source CredentialSource

	// reasoning remembers what a model's chain of thought costs, per model.
	//
	// A reasoning model spends the output allowance on thinking before it writes
	// anything, so a caller asking for 4096 tokens of ANSWER has to be given room
	// for both. The first call that comes back empty pays for the discovery; every
	// call after it starts with the room already added, which is the difference
	// between "supported" and "works if you retry".
	mu        sync.Mutex
	reasoning map[string]int64
}

func newOpenAICompatible(cfg Config, httpClient *http.Client) (*openAIProvider, error) {
	if cfg.StrongModel == "" {
		// Unlike Anthropic there is no knowable default here — the endpoint
		// might be DeepSeek, a local Qwen, or a proxy in front of five models.
		// Guessing would produce a 404 at the first call instead of a clear
		// error now.
		return nil, fmt.Errorf("llm: openai-compatible backend needs a model (set llm.model)")
	}
	if httpClient == nil {
		httpClient = &http.Client{}
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = openAIDefaultBaseURL
	}
	cfg.BaseURL = strings.TrimSuffix(cfg.BaseURL, "/")

	source := CredentialConfig
	if cfg.APIKey == "" {
		// Local runtimes accept any key, or none. That is not a
		// misconfiguration — it is the normal case for Ollama.
		source = CredentialNotNeeded
	}

	return &openAIProvider{
		cfg: cfg, client: httpClient, source: source,
		reasoning: map[string]int64{},
	}, nil
}

func (p *openAIProvider) Name() string                       { return string(KindOpenAICompatible) }
func (p *openAIProvider) CredentialSource() CredentialSource { return p.source }
func (p *openAIProvider) BaseURL() string                    { return p.cfg.BaseURL }

func (p *openAIProvider) ModelFor(tier Tier) string {
	if tier == TierCheap && p.cfg.CheapModel != "" {
		return p.cfg.CheapModel
	}
	return p.cfg.StrongModel
}

type openAIRequest struct {
	Model     string          `json:"model"`
	Messages  []openAIMessage `json:"messages"`
	MaxTokens int             `json:"max_tokens,omitempty"`
	Stream    bool            `json:"stream"`
}

// Reasoning allowance bounds.
//
// MaxReasoningAllowance caps what one model may be given for thinking: a model
// that has not produced an answer in this many tokens is not about to, and a token
// budget that grows without a ceiling is not a budget.
//
// firstReasoningAllowance is what a first empty response jumps to, and it is
// measured rather than chosen. qwen3:4b asked for the JSON `{"ok":true}` — the
// smallest useful prompt there is — spent 1,924 completion tokens on its chain in
// one run and more than 2,128 in the next, on the same prompt. A model that
// reasons at all reasons in thousands, and stepping there 512 tokens at a time
// means paying for four wasted calls to learn what one can.
//
// maxReasoningAttempts bounds the paid discovery at three provider calls.
const (
	MaxReasoningAllowance   = 16384
	firstReasoningAllowance = 4096
	maxReasoningAttempts    = 3
)

type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIResponse struct {
	Model   string `json:"model"`
	Choices []struct {
		Message struct {
			Content string `json:"content"`
			// Reasoning models return their chain separately; it is not part
			// of the answer and is deliberately not concatenated into Text.
			//
			// Two spellings, because two families exist: DeepSeek and vLLM send
			// `reasoning_content`, Ollama's /v1 sends `reasoning`. Reading only
			// the first is why mole saw an empty message and no explanation.
			ReasoningContent string `json:"reasoning_content"`
			Reasoning        string `json:"reasoning"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
		// OpenAI's o-series reports the split; most local runtimes do not, in
		// which case an empty answer means the whole completion was reasoning.
		CompletionTokensDetails struct {
			ReasoningTokens int64 `json:"reasoning_tokens"`
		} `json:"completion_tokens_details"`
		// DeepSeek and some proxies report cache hits under this name.
		PromptCacheHitTokens  int64 `json:"prompt_cache_hit_tokens"`
		PromptCacheMissTokens int64 `json:"prompt_cache_miss_tokens"`
		PromptTokensDetails   struct {
			CachedTokens int64 `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
}

// do issues the request, retrying transient failures.
//
// The Anthropic backend gets retries from its SDK; this one is hand-rolled
// HTTP and had none, so cfg.MaxRetries — a documented field defaulting to 3 —
// was silently ignored for every OpenAI-compatible provider. A single 429 from
// a free tier failed the whole call.
//
// Retry-After is honoured when the server sends it. Guessing an interval is how
// a client that "retries" still fails: a provider asking for 36 seconds will
// refuse three exponential backoffs totalling four.
func (p *openAIProvider) do(ctx context.Context, payload []byte) ([]byte, error) {
	attempts := p.cfg.MaxRetries + 1
	if attempts < 1 {
		attempts = 1
	}

	var lastErr error
	for attempt := 0; attempt < attempts; attempt++ {
		if attempt > 0 {
			t := time.NewTimer(retryDelay(attempt, lastErr))
			select {
			case <-ctx.Done():
				t.Stop()
				// Report why we gave up: "context deadline exceeded" alone
				// hides that a provider asked for a wait we could not afford.
				return nil, fmt.Errorf("llm: %w (last: %v)", ctx.Err(), lastErr)
			case <-t.C:
			}
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
			p.cfg.BaseURL+"/chat/completions", bytes.NewReader(payload))
		if err != nil {
			return nil, fmt.Errorf("llm: request: %w", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")
		if p.cfg.APIKey != "" {
			httpReq.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)
		}

		resp, err := p.client.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("llm: %s: %w", p.cfg.BaseURL, err)
			if ctx.Err() != nil {
				return nil, lastErr
			}
			continue
		}

		body, readErr := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
		retryAfter, hasRetryAfter := parseRetryAfter(resp.Header.Get("Retry-After"))
		resp.Body.Close()
		if readErr != nil {
			lastErr = fmt.Errorf("llm: read: %w", readErr)
			continue
		}

		statusErr := classifyStatus(string(KindOpenAICompatible), resp.StatusCode, string(body))
		if statusErr == nil {
			return body, nil
		}
		if !Retryable(statusErr) {
			// 413 and 401 fail identically however many times they are sent.
			// Returning now keeps a doomed request from burning wall clock the
			// budget is also counting.
			return nil, statusErr
		}
		lastErr = &retryableErr{err: statusErr, after: retryAfter, hasAfter: hasRetryAfter}
	}

	var re *retryableErr
	if errors.As(lastErr, &re) {
		return nil, re.err
	}
	return nil, lastErr
}

// retryableErr carries a server-supplied wait alongside the error.
//
// hasAfter distinguishes "the server said zero" from "the server said nothing".
// They are different instructions: the first means retry now, and treating it
// as absent turns an immediate retry into a second of backoff for no reason.
type retryableErr struct {
	err      error
	after    time.Duration
	hasAfter bool
}

func (e *retryableErr) Error() string { return e.err.Error() }
func (e *retryableErr) Unwrap() error { return e.err }

// retryDelay is the server's requested wait when it gave one, else exponential
// backoff.
func retryDelay(attempt int, lastErr error) time.Duration {
	var re *retryableErr
	if errors.As(lastErr, &re) && re.hasAfter {
		if re.after > 60*time.Second {
			// A wait longer than this is the provider saying "not today".
			// Honour the cap and let the attempt fail rather than parking a
			// worker; the lead's wall clock is a ceiling too.
			return 60 * time.Second
		}
		return re.after
	}
	d := time.Duration(1<<uint(attempt-1)) * time.Second
	if d > 30*time.Second {
		d = 30 * time.Second
	}
	return d
}

// parseRetryAfter reads the header in both permitted forms: seconds, or an HTTP
// date. Providers also send fractional seconds ("36.48"), which the spec does
// not allow but which is plainly a wait in seconds.
func parseRetryAfter(v string) (time.Duration, bool) {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0, false
	}
	if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 {
		return time.Duration(f * float64(time.Second)), true
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := time.Until(t); d > 0 {
			return d, true
		}
		return 0, true // a date in the past means retry now
	}
	return 0, false
}

// Complete calls the endpoint, giving a reasoning model room to think.
//
// A reasoning model emits its chain of thought against the SAME output allowance
// as its answer, so `MaxTokens: 4096` can buy 4096 tokens of thinking and an empty
// message — observed on qwen3 through Ollama's /v1, which is exactly the shape the
// known gaps described: mole's callers see "no JSON object in the reply" and blame
// the prompt.
//
// Ollama ignored every documented way to switch reasoning off (`think:false`,
// `/no_think`, `chat_template_kwargs.enable_thinking`), so this budgets for it
// instead of fighting it. MaxTokens becomes the ANSWER allowance and the provider
// adds a reasoning allowance on top: learned per model, paid for once by the first
// call that comes back empty, and applied up front from then on.
//
// Every attempt's usage is summed into the returned Response. The tokens were
// spent whether or not the answer arrived, and a ledger that charged for one of
// two calls could not enforce a ceiling.
func (p *openAIProvider) Complete(ctx context.Context, req Request) (*Response, error) {
	start := time.Now()

	model := req.Model
	if model == "" {
		model = p.ModelFor(req.Tier)
	}

	answerTokens := req.MaxTokens
	if answerTokens <= 0 {
		answerTokens = 8192
	}

	msgs := make([]openAIMessage, 0, len(req.Messages)+1)
	if req.System != "" {
		msgs = append(msgs, openAIMessage{Role: "system", Content: req.System})
	}
	for _, m := range req.Messages {
		msgs = append(msgs, openAIMessage{Role: string(m.Role), Content: m.Text})
	}

	ctx, cancel := context.WithTimeout(ctx, p.cfg.Timeout)
	defer cancel()

	var (
		total     Usage
		attempts  int
		allowance = p.reasoningAllowance(model)
	)
	for {
		attempts++
		payload, err := json.Marshal(openAIRequest{
			Model:     model,
			Messages:  msgs,
			MaxTokens: answerTokens + int(allowance),
			Stream:    false,
		})
		if err != nil {
			return nil, fmt.Errorf("llm: encode: %w", err)
		}

		body, err := p.do(ctx, payload)
		if err != nil {
			return nil, err
		}

		var parsed openAIResponse
		if err := json.Unmarshal(body, &parsed); err != nil {
			return nil, fmt.Errorf("llm: decode: %w", err)
		}
		if len(parsed.Choices) == 0 {
			return nil, fmt.Errorf("llm: %s returned no choices", p.cfg.BaseURL)
		}
		choice := parsed.Choices[0]
		reasoning := choice.Message.ReasoningContent
		if reasoning == "" {
			reasoning = choice.Message.Reasoning
		}

		usage := p.usageOf(parsed)
		total = total.Add(usage)

		if strings.TrimSpace(choice.Message.Content) != "" || parsed.Usage.CompletionTokens == 0 {
			out := p.response(model, parsed, choice.Message.Content, reasoning, total, start)
			out.ReasoningTokens = p.reasoningTokensOf(parsed, choice.Message.Content, reasoning)
			out.Attempts = attempts
			if out.Usage.IsZero() && out.Text != "" {
				// Some local runtimes omit usage entirely. That would charge zero
				// and make the ceiling unenforceable, so it surfaces rather than
				// passing.
				return out, ErrNoUsageReported
			}
			// Remembered on SUCCESS too. A model that spent 900 tokens thinking
			// and then answered will do it again on the next call, and waiting
			// for a failure to learn that is waiting for a wasted call.
			p.learnReasoning(model, out.ReasoningTokens)
			return out, nil
		}

		// Empty content with tokens spent: the allowance went on reasoning.
		spent := p.reasoningTokensOf(parsed, "", reasoning)
		next := p.raiseAllowance(model, allowance, spent)
		if next <= allowance || attempts >= maxReasoningAttempts {
			// Either the ceiling is reached or the discovery has been paid for
			// often enough. Fail with the precise reason rather than letting the
			// caller see "no JSON object in the reply" and blame the prompt.
			reason := choice.FinishReason
			if reason == "" {
				reason = "unknown"
			}
			return nil, fmt.Errorf("%w: %d completion tokens produced no content "+
				"(finish_reason %q) after %d attempt(s) with a reasoning allowance of "+
				"%d — this model spends its whole output budget on reasoning; raise "+
				"MaxTokens or use a non-reasoning model",
				ErrEmptyOutput, total.OutputTokens, reason, attempts, allowance)
		}
		allowance = next
	}
}

// reasoningAllowance is the extra room this model has needed before, in tokens.
func (p *openAIProvider) reasoningAllowance(model string) int64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.reasoning[model]
}

// learnReasoning records what a model's chain cost, keeping the largest seen.
//
// The largest rather than an average: the allowance exists to stop the answer
// being squeezed out, and sizing it to the mean guarantees that half of all calls
// are squeezed. Capped, because a token budget that grows without a ceiling is not
// a budget.
func (p *openAIProvider) learnReasoning(model string, spent int64) {
	if spent <= 0 {
		return
	}
	// A little over what was seen: a chain that took 900 tokens once will take
	// 950 on a slightly longer prompt, and being one token short costs a whole
	// wasted call.
	want := spent + spent/4
	if want > MaxReasoningAllowance {
		want = MaxReasoningAllowance
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if want > p.reasoning[model] {
		p.reasoning[model] = want
	}
}

// raiseAllowance grows the room for a model that produced nothing, and reports the
// new value. Equal to the old one means the ceiling is reached.
func (p *openAIProvider) raiseAllowance(model string, current, spent int64) int64 {
	next := current * 2
	if next < firstReasoningAllowance {
		next = firstReasoningAllowance
	}
	if spent > next {
		// The provider told us what it burned; jump past it rather than
		// stepping there over several paid calls.
		next = spent + spent/4
	}
	if next > MaxReasoningAllowance {
		next = MaxReasoningAllowance
	}
	if next <= current {
		return current
	}
	p.mu.Lock()
	p.reasoning[model] = next
	p.mu.Unlock()
	return next
}

// reasoningTokensOf attributes the completion between thinking and answering.
//
// Where the provider reports the split, that is used. Where it does not — every
// local runtime seen so far — an empty answer means the whole completion was
// reasoning, and a non-empty one is estimated from the length of the chain. The
// estimate is only ever used to size the next allowance, never to charge: Usage
// carries the provider's own completion count untouched.
func (p *openAIProvider) reasoningTokensOf(parsed openAIResponse, content, reasoning string) int64 {
	if n := parsed.Usage.CompletionTokensDetails.ReasoningTokens; n > 0 {
		return n
	}
	if reasoning == "" {
		return 0
	}
	if strings.TrimSpace(content) == "" {
		return parsed.Usage.CompletionTokens
	}
	est := EstimateTokens(len(reasoning))
	if est > parsed.Usage.CompletionTokens {
		est = parsed.Usage.CompletionTokens
	}
	return est
}

func (p *openAIProvider) usageOf(parsed openAIResponse) Usage {
	// Cache accounting varies by vendor. DeepSeek splits hit/miss; the OpenAI
	// shape nests cached_tokens. Both mean the same thing, and getting it
	// wrong double-counts cached input as fresh input in the ledger.
	cacheRead := parsed.Usage.PromptCacheHitTokens
	if cacheRead == 0 {
		cacheRead = parsed.Usage.PromptTokensDetails.CachedTokens
	}
	inputTokens := parsed.Usage.PromptTokens
	if cacheRead > 0 && inputTokens >= cacheRead {
		// prompt_tokens is the total including cached; subtract so the two
		// fields do not overlap when the ledger sums them.
		inputTokens -= cacheRead
	}
	// Tokens the provider billed for and did not break out.
	//
	// total_tokens is authoritative in the OpenAI spec, and some backends spend
	// tokens that appear in NEITHER part. Gemini's OpenAI-compatible endpoint
	// does it with internal thinking: a call reporting prompt 21 and completion
	// 5 reported a total of 139, and the 113 in between were billed and
	// invisible. mole read the two parts, never looked at the total, and
	// charged a fifth of what the call cost — on a budget whose whole claim is
	// that it is enforced rather than estimated.
	//
	// Not reachable through completion_tokens_details.reasoning_tokens: that is
	// how OpenAI's o-series reports it, and there the reasoning is already
	// inside completion_tokens, so the parts and the total agree and nothing is
	// added here. Gemini reports neither, returning an opaque
	// thought_signature this struct has no field for.
	//
	// Charged as OUTPUT. It is where every provider bills reasoning, and it is
	// the dearer of the two rates — a table whose job is never to under-bill
	// rounds this way deliberately.
	//
	// Only ever upward. A total below the sum of its parts is a provider that
	// cannot add up, and believing it would under-bill.
	outputTokens := parsed.Usage.CompletionTokens
	if reported := parsed.Usage.PromptTokens + parsed.Usage.CompletionTokens; parsed.Usage.TotalTokens > reported {
		outputTokens += parsed.Usage.TotalTokens - reported
	}

	return Usage{
		InputTokens:     inputTokens,
		OutputTokens:    outputTokens,
		CacheReadTokens: cacheRead,
	}
}

// response assembles the result from the final attempt and the summed usage.
func (p *openAIProvider) response(
	model string, parsed openAIResponse, content, reasoning string,
	total Usage, start time.Time,
) *Response {
	out := &Response{
		Text:       content,
		Reasoning:  reasoning,
		Model:      parsed.Model,
		StopReason: parsed.Choices[0].FinishReason,
		Elapsed:    time.Since(start),
		Usage:      total,
	}
	if out.Model == "" {
		out.Model = model
	}
	return out
}

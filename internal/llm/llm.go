// Package llm is the model provider boundary.
//
// One interface, several backends. §10.1 requires this rather than a direct
// Anthropic wrapper: token-mode budgeting is justified partly by users on
// self-hosted models, and hard-wiring one vendor would make that claim false.
//
// Every call returns Usage, because the budget ledger settles against it.
// A provider that cannot report tokens cannot be used in a metered session —
// see §4.2 for the delegated mode where that is true and what replaces the
// budget there.
package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// Role is a message author.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Message is one conversational turn.
type Message struct {
	Role Role
	Text string
}

func User(text string) Message      { return Message{Role: RoleUser, Text: text} }
func Assistant(text string) Message { return Message{Role: RoleAssistant, Text: text} }

// Tier selects how much capability a call needs.
//
// Splitting the two is a real cost lever: chunk mining runs once per chunk and
// is mostly extraction, while planning and synthesis run once per lead or
// session and carry the reasoning. Using one model for both either overpays on
// the many calls or underperforms on the few that matter.
type Tier string

const (
	// TierCheap handles chunk summarization and claim mining.
	TierCheap Tier = "cheap"
	// TierStrong handles planning, verification, and report synthesis.
	TierStrong Tier = "strong"
)

// StrongEffort is the effort planning and report synthesis request.
const StrongEffort = "medium"

// Request is one completion.
type Request struct {
	// Tier picks the model when Model is empty.
	Tier Tier
	// Model overrides the tier's default.
	Model string

	System   string
	Messages []Message

	MaxTokens int
	// Effort maps to the provider's reasoning-depth control where it has one.
	Effort string

	// Thinking asks for extended reasoning. Off for extraction work, where it
	// adds latency and tokens without improving a mechanical task.
	Thinking bool
}

// Usage is the token accounting a provider reports.
//
// This is not optional. The ledger charges against these numbers, and a
// provider that returns zeros silently makes the session's ceiling
// unenforceable — which is why Complete rejects a response with no usage on a
// non-empty completion.
type Usage struct {
	InputTokens      int64
	OutputTokens     int64
	CacheReadTokens  int64
	CacheWriteTokens int64
}

func (u Usage) Total() int64 {
	return u.InputTokens + u.OutputTokens + u.CacheReadTokens + u.CacheWriteTokens
}

func (u Usage) IsZero() bool { return u == Usage{} }

// Add sums two usages.
//
// Needed because one Complete can now be more than one provider call: a reasoning
// model whose first answer was squeezed out by its own chain of thought is retried
// with more room, and both calls spent tokens the ledger has to charge.
func (u Usage) Add(o Usage) Usage {
	return Usage{
		InputTokens:      u.InputTokens + o.InputTokens,
		OutputTokens:     u.OutputTokens + o.OutputTokens,
		CacheReadTokens:  u.CacheReadTokens + o.CacheReadTokens,
		CacheWriteTokens: u.CacheWriteTokens + o.CacheWriteTokens,
	}
}

// Response is one completion result.
type Response struct {
	Text       string
	Model      string
	StopReason string
	Usage      Usage
	Elapsed    time.Duration

	// Reasoning is a reasoning model's chain of thought, kept separate from Text
	// and never concatenated into it: it is not the answer, it is not quotable
	// evidence, and §11.5 would let a model cite its own reasoning as a source if
	// the two were joined.
	//
	// Carried rather than discarded so a caller can log what a model spent its
	// allowance on when the answer came back thin.
	Reasoning string
	// ReasoningTokens is what the chain cost, where the provider reports it or it
	// can be attributed. NOT part of Usage: providers count reasoning inside
	// completion_tokens, so adding it to the ledger's output count would charge
	// the same tokens twice.
	ReasoningTokens int64
	// Attempts is how many provider calls produced this response. More than one
	// means the first was spent entirely on reasoning and the ceiling was raised
	// — Usage covers every attempt, because the tokens were spent either way.
	Attempts int

	// Refused is set when the provider's safety classifiers declined the
	// request. It arrives as a successful HTTP response, so code that reads
	// Text without checking this gets an empty string and no error.
	Refused bool
	// RefusalCategory carries the provider's reason where one is given.
	RefusalCategory string
}

// Provider is a model backend.
type Provider interface {
	Complete(ctx context.Context, req Request) (*Response, error)
	// Name identifies the backend for logs and diagnostics.
	Name() string
	// ModelFor reports which model a tier resolves to, so cost estimates and
	// `doctor` can name it without making a call.
	ModelFor(tier Tier) string
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

var (
	// ErrRateLimited is transient; the executor backs off and retries (§9.5).
	ErrRateLimited = errors.New("llm: rate limited")
	// ErrUnauthorized is fatal. Retrying a bad credential burns wall-clock and
	// never succeeds.
	ErrUnauthorized = errors.New("llm: unauthorized")
	// ErrQuotaExceeded is fatal for this session.
	ErrQuotaExceeded = errors.New("llm: quota exceeded")
	// ErrOverloaded is transient.
	ErrOverloaded = errors.New("llm: provider overloaded")
	// ErrNoUsageReported means the provider returned a completion without
	// token counts. Treated as an error rather than a zero charge: a silent
	// zero would make the budget ceiling unenforceable for that provider.
	ErrNoUsageReported = errors.New("llm: provider reported no token usage")
	// ErrContextTooLong means the request exceeded the model's window. The
	// chunker's job is to prevent this; when it happens anyway the actor
	// re-splits rather than failing the lead.
	ErrContextTooLong = errors.New("llm: context too long")

	// ErrEmptyOutput means the provider returned a successful response with no
	// content.
	//
	// The case that produces it in practice is a reasoning model: qwen3, gemma4
	// and others emit a `reasoning` field that mole does not read, and it is
	// charged against the same output budget. Ask for too few tokens and the
	// whole allowance goes to reasoning, leaving content empty with
	// finish_reason "length" — a successful call that returned nothing.
	//
	// Worth its own sentinel because the symptom is otherwise a JSON parse
	// failure, which sends whoever reads the log looking at the prompt instead
	// of at MaxTokens.
	ErrEmptyOutput = errors.New("llm: provider returned empty content")
)

// APIError carries provider detail alongside a sentinel.
type APIError struct {
	Provider string
	Status   int
	Body     string
	sentinel error
}

func (e *APIError) Error() string {
	msg := fmt.Sprintf("llm: %s returned %d", e.Provider, e.Status)
	if e.Body != "" {
		body := e.Body
		if len(body) > 300 {
			body = body[:300] + "…"
		}
		msg += ": " + body
	}
	return msg
}

func (e *APIError) Unwrap() error { return e.sentinel }

// Retryable reports whether the executor should back off rather than fail.
func (e *APIError) Retryable() bool {
	return errors.Is(e.sentinel, ErrRateLimited) ||
		errors.Is(e.sentinel, ErrOverloaded) ||
		(e.Status >= 500 && e.Status < 600)
}

// Retryable reports whether any error is worth another attempt.
func Retryable(err error) bool {
	var api *APIError
	if errors.As(err, &api) {
		return api.Retryable()
	}
	return errors.Is(err, ErrRateLimited) || errors.Is(err, ErrOverloaded)
}

func classifyStatus(provider string, status int, body string) error {
	if status >= 200 && status < 300 {
		return nil
	}
	e := &APIError{Provider: provider, Status: status, Body: body}
	switch status {
	case 401, 403:
		e.sentinel = ErrUnauthorized
	case 402:
		e.sentinel = ErrQuotaExceeded
	case 413:
		e.sentinel = ErrContextTooLong
	case 429:
		e.sentinel = ErrRateLimited
	case 529:
		e.sentinel = ErrOverloaded
	default:
		if strings.Contains(strings.ToLower(body), "context") &&
			strings.Contains(strings.ToLower(body), "long") {
			e.sentinel = ErrContextTooLong
		}
	}
	return e
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Kind names a backend.
type Kind string

const (
	KindAnthropic Kind = "anthropic"
	// KindOpenAICompatible covers DeepSeek, Ollama, llama.cpp, vLLM, LiteLLM,
	// Together, and anything else speaking the same wire format.
	KindOpenAICompatible Kind = "openai-compatible"
)

func (k Kind) Valid() bool { return k == KindAnthropic || k == KindOpenAICompatible }

func Kinds() []Kind { return []Kind{KindAnthropic, KindOpenAICompatible} }

// Config selects and configures a provider.
type Config struct {
	Kind Kind

	// APIKey may be empty for Anthropic, in which case the SDK's own
	// credential chain resolves it — an env var, then an `ant auth login`
	// profile. That is what lets a machine with a profile need no
	// configuration at all (§4.2).
	APIKey string

	// BaseURL overrides the endpoint. Required for OpenAI-compatible backends.
	BaseURL string

	// StrongModel and CheapModel resolve the two tiers.
	StrongModel string
	CheapModel  string

	MaxRetries int
	Timeout    time.Duration
}

// Defaults for the Anthropic backend. Opus 5 for reasoning, Haiku 4.5 for the
// per-chunk extraction work that runs an order of magnitude more often.
const (
	DefaultAnthropicStrong = "claude-opus-5"
	DefaultAnthropicCheap  = "claude-haiku-4-5"
)

func (c Config) withDefaults() Config {
	if c.Timeout <= 0 {
		// Long: a strong-tier synthesis call over a whole claim graph is not a
		// fast request, and the SDK streams to avoid a transport timeout.
		c.Timeout = 10 * time.Minute
	}
	// Two sequential ifs here made the negative case unreachable: it clamped to
	// zero and the next test promptly turned that back into three. A switch is
	// the difference between "retries cannot be disabled" and "-1 disables
	// them", which is the only way to say so given that 0 means unset.
	switch {
	case c.MaxRetries < 0:
		c.MaxRetries = 0
	case c.MaxRetries == 0:
		c.MaxRetries = 3
	}
	if c.Kind == KindAnthropic {
		if c.StrongModel == "" {
			c.StrongModel = DefaultAnthropicStrong
		}
		if c.CheapModel == "" {
			c.CheapModel = DefaultAnthropicCheap
		}
	}
	// An OpenAI-compatible endpoint has no knowable default model, so a
	// missing one is a configuration error rather than a guess.
	if c.CheapModel == "" {
		c.CheapModel = c.StrongModel
	}
	return c
}

// CredentialSource records how a provider resolved its credential, so
// `doctor` can answer "which one is this actually using" without guessing.
type CredentialSource string

const (
	CredentialConfig    CredentialSource = "config"
	CredentialEnv       CredentialSource = "env"
	CredentialChain     CredentialSource = "sdk credential chain"
	CredentialNotNeeded CredentialSource = "none required"
)

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

// SourcedProvider is a Provider that can report how it authenticated.
type SourcedProvider interface {
	Provider
	CredentialSource() CredentialSource
}

// New builds the configured provider.
//
// httpClient carries the record/replay cassette transport, so model calls are
// deterministic and cost nothing in tests and in the eval harness. Passing nil
// uses the default client.
func New(cfg Config, httpClient *http.Client) (Provider, error) {
	if !cfg.Kind.Valid() {
		return nil, fmt.Errorf("llm: unknown provider %q (want one of %v)", cfg.Kind, Kinds())
	}
	cfg = cfg.withDefaults()

	switch cfg.Kind {
	case KindAnthropic:
		return newAnthropic(cfg, httpClient)
	case KindOpenAICompatible:
		return newOpenAICompatible(cfg, httpClient)
	default:
		return nil, fmt.Errorf("llm: unhandled provider %q", cfg.Kind)
	}
}

// SourceOf reports how a provider authenticated, or CredentialConfig when the
// provider does not track it.
func SourceOf(p Provider) CredentialSource {
	if sp, ok := p.(SourcedProvider); ok {
		return sp.CredentialSource()
	}
	return CredentialConfig
}

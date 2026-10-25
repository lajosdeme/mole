package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// Anthropic backend, on the official SDK.
//
// The SDK is used rather than raw HTTP for one specific reason beyond
// convenience: it resolves credentials through its own chain — env var, then an
// `ant auth login` OAuth profile — so a machine with a profile and no key
// configured anywhere still works. That is the zero-configuration path in §4.2,
// and hand-rolled HTTP would not have it.

type anthropicProvider struct {
	client anthropic.Client
	cfg    Config
	source CredentialSource

	// capsMu guards capsCache and capsInFlight, the per-model thinking and
	// effort capabilities.
	capsMu       sync.Mutex
	capsCache    map[string]modelCaps
	capsInFlight map[string]*capsLookup
}

// modelCaps is what a model accepts for disabled thinking and effort.
type modelCaps struct {
	disabledThinking bool
	effort           effortSupport
}

// effortSupport is the set of effort levels a model accepts; all means any level.
type effortSupport struct {
	all    bool
	levels map[string]bool
}

// capsLookup is one in-progress Models API call, shared by every caller
// that asks about the same model while it is outstanding.
type capsLookup struct {
	done   chan struct{}
	result modelCaps
}

// newAnthropic builds the client.
//
// httpClient carries the record/replay cassette transport, so model calls are
// deterministic and free in tests and in the eval harness.
func newAnthropic(cfg Config, httpClient *http.Client) (*anthropicProvider, error) {
	opts := []option.RequestOption{
		option.WithMaxRetries(cfg.MaxRetries),
		option.WithRequestTimeout(cfg.Timeout),
	}
	if httpClient != nil {
		opts = append(opts, option.WithHTTPClient(httpClient))
	}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}

	source := detectCredentialSource(cfg.APIKey)

	// Pass the key ONLY when there is one. Handing the SDK an empty string
	// looks like "explicitly no credential" and suppresses its own resolution,
	// which would break the profile path that makes zero-config work.
	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}

	return &anthropicProvider{
		client:       anthropic.NewClient(opts...),
		cfg:          cfg,
		source:       source,
		capsCache:    make(map[string]modelCaps),
		capsInFlight: make(map[string]*capsLookup),
	}, nil
}

// detectCredentialSource reports where the credential came from, so `doctor`
// can name it instead of leaving the user to guess which of three sources won.
func detectCredentialSource(configured string) CredentialSource {
	if configured != "" {
		// config.applyEnv has already folded env vars in, so a non-empty value
		// here may have come from either. Distinguish by checking the
		// environment for the same value.
		for _, k := range []string{"MOLE_LLM_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"} {
			if os.Getenv(k) == configured {
				return CredentialEnv
			}
		}
		return CredentialConfig
	}
	return CredentialChain
}

func (p *anthropicProvider) Name() string { return string(KindAnthropic) }

// CredentialSource exposes how this provider authenticated.
func (p *anthropicProvider) CredentialSource() CredentialSource { return p.source }

func (p *anthropicProvider) ModelFor(tier Tier) string {
	if tier == TierCheap {
		return p.cfg.CheapModel
	}
	return p.cfg.StrongModel
}

func (p *anthropicProvider) Complete(ctx context.Context, req Request) (*Response, error) {
	start := time.Now()

	model := req.Model
	if model == "" {
		model = p.ModelFor(req.Tier)
	}
	if model == "" {
		return nil, fmt.Errorf("llm: no model configured for tier %q", req.Tier)
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 8192
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: int64(maxTokens),
		Messages:  toSDKMessages(req.Messages),
	}
	if req.System != "" {
		params.System = []anthropic.TextBlockParam{{Text: req.System}}
	}
	var caps modelCaps
	if !req.Thinking || req.Effort != "" {
		caps = p.capabilities(ctx, model)
	}
	effort := req.Effort
	switch {
	case req.Thinking:
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{},
		}
	case caps.disabledThinking:
		// Chunk mining is mechanical extraction. Thinking there adds latency
		// and tokens without improving the task, and the per-chunk call is the
		// one that runs most often.
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfDisabled: &anthropic.ThinkingConfigDisabledParam{},
		}
	case effort == "":
		// This model rejects the disabled type outright, so the field is left
		// unset entirely and low effort stands in for "don't think hard".
		effort = string(anthropic.OutputConfigEffortLow)
	}
	if effort = caps.effort.clamp(effort); effort != "" {
		params.OutputConfig = anthropic.OutputConfigParam{
			Effort: anthropic.OutputConfigEffort(effort),
		}
	}

	// Stream and accumulate. A strong-tier synthesis over a whole claim graph
	// can run for minutes with a large max_tokens; a non-streaming request at
	// that size risks an idle-connection timeout rather than a slow success.
	stream := p.client.Messages.NewStreaming(ctx, params)
	var msg anthropic.Message
	for stream.Next() {
		if err := msg.Accumulate(stream.Current()); err != nil {
			// Return the partial response alongside the error. A stream that
			// dies mid-flight was still billed for the tokens it delivered, and
			// returning nil here dropped them: the ledger under-counted, and a
			// ceiling enforced from an under-count is not a ceiling. Callers
			// record cost before checking err precisely so this lands.
			return partialResponse(msg, start), fmt.Errorf("llm: accumulate: %w", err)
		}
	}
	if err := stream.Err(); err != nil {
		return partialResponse(msg, start), translateSDKError(err)
	}

	resp := partialResponse(msg, start)

	// A refusal is a successful HTTP response with an empty content array.
	// Code that reads Text without checking gets "" and no error, so the flag
	// is set before the text is assembled.
	if msg.StopReason == anthropic.StopReasonRefusal {
		resp.Refused = true
		resp.RefusalCategory = string(msg.StopDetails.Category)
		return resp, nil
	}

	var b strings.Builder
	for _, block := range msg.Content {
		if t, ok := block.AsAny().(anthropic.TextBlock); ok {
			b.WriteString(t.Text)
		}
	}
	resp.Text = b.String()

	// A completion with no usage would charge zero and quietly make the
	// session ceiling unenforceable, so it is an error rather than a freebie.
	if resp.Usage.IsZero() && resp.Text != "" {
		return resp, ErrNoUsageReported
	}
	return resp, nil
}

// disabledThinkingPrefixes are models known to reject {"type":"disabled"}
// thinking, used only when the Models API lookup itself fails.
var disabledThinkingPrefixes = []string{
	"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5", "claude-mythos-5",
}

// capabilities reports what model accepts for thinking and effort, asking the
// Models API once per model; concurrent callers share the lookup.
func (p *anthropicProvider) capabilities(ctx context.Context, model string) modelCaps {
	p.capsMu.Lock()
	if v, ok := p.capsCache[model]; ok {
		p.capsMu.Unlock()
		return v
	}
	if lookup, ok := p.capsInFlight[model]; ok {
		p.capsMu.Unlock()
		<-lookup.done
		return lookup.result
	}
	lookup := &capsLookup{done: make(chan struct{})}
	p.capsInFlight[model] = lookup
	p.capsMu.Unlock()

	caps := modelCaps{
		disabledThinking: fallbackDisabledThinkingSupported(model),
		effort:           fallbackEffortSupport(model),
	}
	if info, err := p.client.Models.Get(ctx, model, anthropic.ModelGetParams{}); err == nil {
		caps.disabledThinking = disabledSupportedFromCapabilities(info.Capabilities.Thinking.Types.RawJSON())
		if info.Capabilities.JSON.Effort.Valid() {
			caps.effort = effortFromCapability(info.Capabilities.Effort)
		}
	}

	p.capsMu.Lock()
	p.capsCache[model] = caps
	delete(p.capsInFlight, model)
	p.capsMu.Unlock()
	lookup.result = caps
	close(lookup.done)
	return caps
}

// fallbackDisabledThinkingSupported reports the default answer used when the
// Models API lookup fails.
func fallbackDisabledThinkingSupported(model string) bool {
	for _, prefix := range disabledThinkingPrefixes {
		if strings.HasPrefix(model, prefix) {
			return false
		}
	}
	return true
}

// disabledSupportedFromCapabilities reads capabilities.thinking.types.disabled
// out of the Models API response's raw JSON for that field.
func disabledSupportedFromCapabilities(typesRawJSON string) bool {
	var types struct {
		Disabled struct {
			Supported bool `json:"supported"`
		} `json:"disabled"`
	}
	if err := json.Unmarshal([]byte(typesRawJSON), &types); err != nil {
		return false
	}
	return types.Disabled.Supported
}

// effortOrder lists effort levels from lowest to highest.
var effortOrder = []string{"low", "medium", "high", "xhigh", "max"}

// noEffortPrefixes and threeLevelEffortPrefixes are the effort fallbacks used
// only when the Models API lookup fails.
var (
	noEffortPrefixes = []string{
		"claude-haiku-4", "claude-sonnet-4", "claude-opus-4-1", "claude-opus-4-0", "claude-3",
	}
	threeLevelEffortPrefixes = []string{"claude-opus-4-5", "claude-opus-4-6", "claude-sonnet-4-6"}
)

// fallbackEffortSupport reports the effort levels assumed for model when the
// Models API lookup fails.
func fallbackEffortSupport(model string) effortSupport {
	for _, prefix := range threeLevelEffortPrefixes {
		if strings.HasPrefix(model, prefix) {
			return effortSupport{levels: map[string]bool{"low": true, "medium": true, "high": true}}
		}
	}
	for _, prefix := range noEffortPrefixes {
		if strings.HasPrefix(model, prefix) {
			return effortSupport{}
		}
	}
	return effortSupport{all: true}
}

// effortFromCapability converts the Models API effort capability into the
// set of accepted levels.
func effortFromCapability(c anthropic.EffortCapability) effortSupport {
	if !c.Supported {
		return effortSupport{}
	}
	return effortSupport{levels: map[string]bool{
		"low": c.Low.Supported, "medium": c.Medium.Supported, "high": c.High.Supported,
		"xhigh": c.Xhigh.Supported, "max": c.Max.Supported,
	}}
}

// clamp returns the highest accepted level not above requested, or "" when
// none is accepted.
func (e effortSupport) clamp(requested string) string {
	if requested == "" || e.all || e.levels[requested] {
		return requested
	}
	for i := slices.Index(effortOrder, requested) - 1; i >= 0; i-- {
		if e.levels[effortOrder[i]] {
			return effortOrder[i]
		}
	}
	return ""
}

// partialResponse projects whatever the accumulator holds into a Response.
//
// Called on both the success and the failure paths so a half-finished stream
// still reports its usage. Text and refusal are filled in by the caller only
// when the stream completed.
func partialResponse(msg anthropic.Message, start time.Time) *Response {
	return &Response{
		Model:      string(msg.Model),
		StopReason: string(msg.StopReason),
		Elapsed:    time.Since(start),
		Usage: Usage{
			InputTokens:      msg.Usage.InputTokens,
			OutputTokens:     msg.Usage.OutputTokens,
			CacheReadTokens:  msg.Usage.CacheReadInputTokens,
			CacheWriteTokens: msg.Usage.CacheCreationInputTokens,
		},
	}
}

func toSDKMessages(msgs []Message) []anthropic.MessageParam {
	out := make([]anthropic.MessageParam, 0, len(msgs))
	for _, m := range msgs {
		block := anthropic.NewTextBlock(m.Text)
		if m.Role == RoleAssistant {
			out = append(out, anthropic.NewAssistantMessage(block))
			continue
		}
		out = append(out, anthropic.NewUserMessage(block))
	}
	return out
}

// translateSDKError maps the SDK's error to our sentinels, so the executor's
// retry policy (§9.5) can tell a rate limit from a bad key without matching on
// message text.
func translateSDKError(err error) error {
	var apierr *anthropic.Error
	if errors.As(err, &apierr) {
		return classifyStatus(string(KindAnthropic), apierr.StatusCode, apierr.Error())
	}
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
		return err
	}
	return fmt.Errorf("llm: anthropic: %w", err)
}

package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
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

	// thinkingMu guards thinkingCache, the per-model answer to whether the
	// model accepts {"type":"disabled"} thinking, learned from the Models API.
	thinkingMu    sync.Mutex
	thinkingCache map[string]bool
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
		client:        anthropic.NewClient(opts...),
		cfg:           cfg,
		source:        source,
		thinkingCache: make(map[string]bool),
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
	if req.Effort != "" {
		params.OutputConfig = anthropic.OutputConfigParam{
			Effort: anthropic.OutputConfigEffort(req.Effort),
		}
	}
	switch {
	case req.Thinking:
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{},
		}
	case p.disabledThinkingSupported(ctx, model):
		// Chunk mining is mechanical extraction. Thinking there adds latency
		// and tokens without improving the task, and the per-chunk call is the
		// one that runs most often.
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfDisabled: &anthropic.ThinkingConfigDisabledParam{},
		}
	case req.Effort == "":
		// This model rejects the disabled type outright, so the field is left
		// unset entirely and low effort stands in for "don't think hard".
		params.OutputConfig = anthropic.OutputConfigParam{
			Effort: anthropic.OutputConfigEffortLow,
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

// disabledThinkingSupported reports whether model accepts {"type":"disabled"}
// thinking, asking the Models API once per model and caching the answer.
func (p *anthropicProvider) disabledThinkingSupported(ctx context.Context, model string) bool {
	p.thinkingMu.Lock()
	if v, ok := p.thinkingCache[model]; ok {
		p.thinkingMu.Unlock()
		return v
	}
	p.thinkingMu.Unlock()

	supported := fallbackDisabledThinkingSupported(model)
	if info, err := p.client.Models.Get(ctx, model, anthropic.ModelGetParams{}); err == nil {
		supported = disabledSupportedFromCapabilities(info.Capabilities.Thinking.Types.RawJSON())
	}

	p.thinkingMu.Lock()
	p.thinkingCache[model] = supported
	p.thinkingMu.Unlock()
	return supported
}

// fallbackDisabledThinkingSupported is used when the Models API lookup fails:
// known-bad model families are assumed unsupported, everything else assumed
// supported since that was the behaviour before this check existed.
func fallbackDisabledThinkingSupported(model string) bool {
	for _, prefix := range disabledThinkingPrefixes {
		if strings.HasPrefix(model, prefix) {
			return false
		}
	}
	return true
}

// disabledSupportedFromCapabilities reads capabilities.thinking.types.disabled
// from the Models API response. The SDK's ThinkingTypes struct has no field
// for "disabled" yet, so this parses the type's raw JSON instead of a
// hand-rolled HTTP call.
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

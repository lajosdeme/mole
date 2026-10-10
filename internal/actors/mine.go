package actors

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/pricing"
)

// Miner turns a passage of source text into quote-verified claims.
//
// Extracted from WebActor so the AcademicActor does not carry a second copy.
// The M5 review found the same defect fixed twice in two copies of the response
// handler, twice in one milestone; this is the same shape of code — a model
// call, a parse, a verbatim quote check, and the §11.4 lineage bookkeeping —
// and two copies would be two places for the quote check to drift.
//
// It deliberately does NOT know where the text came from. A web page, an
// abstract, and a full-text section are the same problem once the text is in
// hand, and the difference between them belongs to the actor that fetched it.
type Miner struct {
	LLM     llm.Provider
	Pricing *pricing.Table
	Log     *slog.Logger

	// SessionID scopes the claims this miner records.
	SessionID string
}

// MineInput is one passage to mine.
type MineInput struct {
	Lead core.Lead
	// SourceURL is what the claim will cite. For a paper it is the landing page
	// or the DOI, not the API endpoint the text arrived through — a citation has
	// to be something a reader can open.
	SourceURL string
	Title     string
	// PublishedAt is carried onto every claim. Academic sources report it
	// exactly, which is what §11.2's staleness rule has been missing.
	PublishedAt *time.Time

	// Text is the passage. Offset is where it starts within the whole document,
	// so a quote offset indexes the document rather than the chunk — the chunk
	// does not outlive this call and a re-verification has to find the span.
	Text   string
	Offset int

	MaxClaims int
}

// MineOutput is what one passage yielded.
type MineOutput struct {
	Claims []core.Claim
	Usage  llm.Usage
	// Call is the ledger row for the model call, recorded whether or not the
	// call succeeded — the tokens were spent either way.
	Call     core.ToolCall
	HasCall  bool
	Proposed int
	Rejected int
}

func (m *Miner) logger() *slog.Logger {
	if m.Log != nil {
		return m.Log
	}
	return slog.Default()
}

// Mine extracts claims from one passage, keeping only those whose quote is
// genuinely present in it (§11.5).
//
// The quote check is the whole point and is not negotiable: a claim whose quote
// cannot be found in the text it was mined from was not copied from that text,
// and citing it would put mole's name behind something a model composed.
func (m *Miner) Mine(ctx context.Context, in MineInput) (MineOutput, error) {
	var out MineOutput
	if in.MaxClaims <= 0 {
		in.MaxClaims = 8
	}

	prompt := mineUserPrompt(fenceToken(), in.Lead.Query, in.MaxClaims, in.Title, in.SourceURL, in.Text)
	resp, err := m.LLM.Complete(ctx, llm.Request{
		Tier:      llm.TierCheap,
		System:    mineSystemPrompt,
		Messages:  []llm.Message{llm.User(prompt)},
		MaxTokens: 4096,
	})
	if resp == nil {
		return out, err
	}
	out.Usage = resp.Usage
	out.Call = toolCallFor(m.SessionID, in.Lead, resp, "mine:"+in.SourceURL, m.Pricing, m.Log, err)
	out.HasCall = true

	if err != nil {
		return out, err
	}
	if resp.Refused {
		return out, fmt.Errorf("actors: model refused (%s)", resp.RefusalCategory)
	}

	mined, err := parseMined(resp.Text)
	if err != nil {
		return out, err
	}

	now := time.Now().UTC()
	for _, candidate := range mined {
		out.Proposed++

		if strings.TrimSpace(candidate.Text) == "" {
			out.Rejected++
			continue
		}
		match, ok := FindQuote(in.Text, candidate.Quote)
		if !ok {
			out.Rejected++
			m.logger().DebugContext(ctx, "claim rejected: quote not found in source",
				"source", in.SourceURL, "quote", truncateForLog(candidate.Quote))
			continue
		}

		out.Claims = append(out.Claims, core.Claim{
			SessionID:   m.SessionID,
			LeadID:      in.Lead.ID,
			Text:        strings.TrimSpace(candidate.Text),
			Source:      in.SourceURL,
			Quote:       TruncateQuote(match.Text),
			QuoteOffset: int64(in.Offset + match.Offset),
			PublishedAt: in.PublishedAt,
			RetrievedAt: now,
			// What the SOURCE asserts, not how much mole believes it. Writing
			// this to Confidence made an uncalibrated self-report decide which
			// claims led the report; the Verifier derives confidence from the
			// graph (§11.3).
			AssertionStrength: clamp01(candidate.Confidence),
			// §11.4's lineage, inherited from the lead, which is what makes the
			// verification depth cap bind.
			RootClaimID: rootClaimOf(in.Lead),
			VerifyDepth: in.Lead.VerifyDepth,
		})
		if len(out.Claims) >= in.MaxClaims {
			break
		}
	}
	return out, nil
}

// toolCallFor prices one model call into a ledger row.
//
// Package-level because neither receiver contributed anything but a session id
// and a pricing table, and both callers passed RoleExecutor — a parameter with
// one value is an invitation to pass the wrong one later.
func toolCallFor(
	sessionID string,
	lead core.Lead,
	resp *llm.Response,
	input string,
	table *pricing.Table,
	log *slog.Logger,
	callErr error,
) core.ToolCall {
	tc := core.ToolCall{
		SessionID:  sessionID,
		LeadID:     &lead.ID,
		Role:       core.RoleExecutor,
		Type:       core.CallLLM,
		Model:      resp.Model,
		Input:      input,
		DurationMS: resp.Elapsed.Milliseconds(),
	}
	if callErr != nil {
		tc.Err = callErr.Error()
	}

	if table == nil {
		table = pricing.NewTable()
	}
	cost, err := table.Cost(resp.Model, pricing.Usage{
		InputTokens:      resp.Usage.InputTokens,
		OutputTokens:     resp.Usage.OutputTokens,
		CacheReadTokens:  resp.Usage.CacheReadTokens,
		CacheWriteTokens: resp.Usage.CacheWriteTokens,
	})
	if err != nil {
		// An unpriced model still spent tokens. Zero dollars with real token
		// counts keeps token-mode budgets correct; doctor surfaces the gap.
		if log == nil {
			log = slog.Default()
		}
		log.Warn("model not in pricing table; USD cost recorded as zero",
			"model", resp.Model, "tokens", resp.Usage.Total())
	}
	tc.Cost = cost
	return tc
}

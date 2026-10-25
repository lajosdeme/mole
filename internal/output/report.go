// Package output turns a finished session's claims into a report (§13).
//
// Paid from escrow. §8.3 holds a slice of the budget back at session creation
// precisely so this step is affordable after the research has spent everything
// it was allowed to — a run that produced good claims and cannot afford to
// write them up has wasted the whole budget, not just the last call.
//
// Every claim reaching this point has already had its quote verified against
// the source it came from (§11.5). What the model does here is arrange them; it
// is not asked to judge them, and it cannot introduce a citation, because the
// citation markers are assigned mechanically before the prompt is built.
package output

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/pricing"
	"github.com/lajosdeme/mole/internal/store"
)

// Citation is one numbered source.
// The tags are load-bearing, not decoration. This struct is PERSISTED — the
// report's numbering cannot be recomputed, so it is stored — which makes its
// marshalled shape a storage format rather than an implementation detail.
// Without them the fields serialise as Go names, and a reader whose tag is
// published_at silently never matches PublishedAt: encoding/json compares keys
// case-insensitively, so N, Source and Quotes survived and the date did not.
type Citation struct {
	N      int    `json:"n"`
	Source string `json:"source"`
	// Quotes are the verified spans supporting the claims cited to this
	// source, so a reader can check the report without re-fetching.
	Quotes []string `json:"quotes"`
	// PublishedAt is the earliest publication date seen for the source, when
	// one was recoverable. §13 asks for it per citation: a 2024 preprint and a
	// 2026 paper carry different weight and the reader has to see which.
	PublishedAt *time.Time `json:"published_at"`
}

// Report is a rendered answer.
type Report struct {
	SessionID string
	Question  string
	Body      string
	Citations []Citation

	// Cost is what generating it charged.
	Cost core.Cost
	// Model names what wrote it.
	Model string
	// Elapsed is how long the synthesis call took, carried through to the
	// ledger row so the later per-stage time fixes have something to measure.
	Elapsed time.Duration

	// Findings are the collapsed assertions the body was written from (§11.2):
	// one per duplicate cluster, each with every source asserting it.
	Findings []Finding
	// Disagreements is how many contradicting pairs the graph found among them.
	// Reported so a reader can tell "no disagreements" from "not checked".
	Disagreements int

	// Degraded explains why the report is thin, when it is. Surfaced rather
	// than left for a reader to infer from a short answer.
	//
	// A CLOSED SET of stable phrases, and that is a security property rather
	// than tidiness. Degraded is copied into research.ask and research.result
	// and therefore lands in an MCP caller's context, so anything interpolated
	// into it is something an untrusted party can write there. Two things were:
	// provider errors, which carry llm.Config.BaseURL verbatim and so disclose
	// an internal endpoint; and body-validation errors, which quote the model's
	// output — text derived from fetched pages, meaning attacker-controlled
	// content would arrive in an agent's context outside any §3.2 fence.
	Degraded string
	// DegradedDetail is the unabridged reason: provider errors, rejected text,
	// whatever a person debugging this needs.
	//
	// It NEVER crosses the MCP boundary. It is for logs, the CLI, and tests —
	// all places where the reader already has the daemon's own trust. Keep it
	// out of every wire struct in internal/mcpserver.
	DegradedDetail string
}

// MaxClaimChars bounds a single claim's text in the prompt.
//
// MaxClaims bounds the COUNT, and claim Text is never length-capped upstream —
// actors truncate Quote but not Text — so one page could inflate the synthesis
// call into ErrContextTooLong and degrade the whole report to the fallback. A
// claim is meant to be one sentence; anything past this is not a claim.
const MaxClaimChars = 600

// Generator writes reports.
type Generator struct {
	LLM llm.Provider

	// MaxClaims bounds what reaches the prompt. A session can produce hundreds
	// of claims; the report is paid from a fixed escrow, so the input has to be
	// bounded by something other than optimism.
	MaxClaims int
	// MaxTokens bounds the response.
	MaxTokens int
}

const (
	DefaultMaxClaims = 60
	DefaultMaxTokens = 3000
)

func (g *Generator) withDefaults() *Generator {
	out := *g
	if out.MaxClaims <= 0 {
		out.MaxClaims = DefaultMaxClaims
	}
	if out.MaxTokens <= 0 {
		// Scaled with the findings, because the two bound the same document
		// from opposite ends: raising the findings alone buys a report that
		// draws on more evidence and then runs out of room to say it. The
		// ratio is the default pair, so an unset generator is unchanged.
		out.MaxTokens = DefaultMaxTokens * out.MaxClaims / DefaultMaxClaims
		if out.MaxTokens < DefaultMaxTokens {
			out.MaxTokens = DefaultMaxTokens
		}
	}
	return &out
}

// Generate writes a report for a finished session.
func (g *Generator) Generate(ctx context.Context, st store.Store, sessionID string) (*Report, error) {
	gen := g.withDefaults()

	var (
		sess   *core.Session
		claims []*core.Claim
		edges  []*core.ClaimEdge
	)
	if err := st.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		if sess, err = q.GetSession(ctx, sessionID); err != nil {
			return err
		}
		if claims, err = q.ListClaims(ctx, sessionID, 0); err != nil {
			return err
		}
		// Edges too: §11.2 collapses duplicate clusters into one finding, and a
		// report built from raw claims restates the same finding once per source.
		edges, err = q.ListEdges(ctx, sessionID, 0)
		return err
	}); err != nil {
		return nil, err
	}

	rep := &Report{SessionID: sessionID, Question: sess.Prompt}

	if len(claims) == 0 {
		// No model call: there is nothing to synthesize, and spending escrow to
		// have a model say so is worse than saying it here.
		rep.Body = fmt.Sprintf("No verifiable evidence was found for %q.", sess.Prompt)
		rep.Degraded = "no claims survived quote verification"
		return rep, nil
	}

	// Collapse duplicate clusters BEFORE the cap applies (§11.2). Capping raw
	// claims spends the whole budget on one page asserting one thing eight ways;
	// capping findings counts distinct assertions.
	findings := Findings(sess.Prompt, claims, edges, gen.MaxClaims)
	if n := len(Findings(sess.Prompt, claims, edges, 0)); n > len(findings) {
		// What the prose was written from, not what was left out. The rest are
		// not missing: they are in the session, and a consumer rendering it
		// shows them. Saying they "did not fit" described a shortfall that had
		// not happened.
		rep.Degraded = fmt.Sprintf("written from the %d highest-ranked findings",
			len(findings))
	}
	rep.Findings = findings
	rep.Citations = citeFindings(findings)
	index := map[string]int{}
	for _, c := range rep.Citations {
		index[c.Source] = c.N
	}
	rep.Disagreements = len(Disagreements(findings))

	// No model available or affordable: return the evidence without prose. The
	// escrow already paid to collect it, and verified claims with citations are
	// a real answer — less readable, not less true.
	if gen.LLM == nil {
		rep.Body = fallbackBody(sess.Prompt, findings, index)
		rep.Degraded = "no budget left to synthesize; evidence listed unsynthesized"
		return rep, nil
	}

	fence := fenceToken()
	gen.writeBody(ctx, llm.Request{
		Tier:      llm.TierStrong,
		System:    reportSystemPrompt,
		Messages:  []llm.Message{llm.User(reportPrompt(fence, sess.Prompt, findings, index))},
		MaxTokens: gen.MaxTokens,
	}, rep, wording{noun: "synthesis", verb: "synthesize"},
		fallbackBody(sess.Prompt, findings, index))
	return rep, nil
}

// wording is what Generate and Answer disagree about when a call fails.
type wording struct {
	noun string // "synthesis" / "answer" — as in "synthesis failed"
	verb string // "synthesize" / "answer" — as in "model refused to synthesize"
}

// writeBody makes the model call, records its cost, and sets rep.Body from the
// response — falling back to the evidence listing on any failure.
//
// Shared by Generate and Answer, and the sharing is the point: this is where
// both of them were bitten by the same two defects, each needing the same fix
// applied twice. A response can be nil with a nil error, and the guard above the
// dereference did not cover it; and the degraded reason interpolated provider
// and model text into a string that crosses the MCP boundary.
//
// A failed synthesis costs the prose, not the evidence. Verified claims with
// citations are a real answer — less readable, not less true — and the budget
// already paid to collect them.
func (gen *Generator) writeBody(ctx context.Context, req llm.Request, rep *Report, w wording, fallback string) {
	resp, err := gen.LLM.Complete(ctx, req)
	if resp != nil {
		rep.Model = resp.Model
		rep.Cost = core.Cost{
			InputTokens:      resp.Usage.InputTokens,
			OutputTokens:     resp.Usage.OutputTokens,
			CacheReadTokens:  resp.Usage.CacheReadTokens,
			CacheWriteTokens: resp.Usage.CacheWriteTokens,
		}
		rep.Elapsed = resp.Elapsed
	}
	if err == nil && resp == nil {
		// The nil guard above concedes resp can be nil, which makes every
		// dereference below reachable — inside a daemon serving other sessions.
		err = errors.New("the model provider returned nothing")
	}

	switch {
	case err != nil:
		rep.Body = fallback
		rep.Degraded = w.noun + " failed"
		rep.DegradedDetail = err.Error()
	case resp.Refused:
		rep.Body = fallback
		rep.Degraded = "model refused to " + w.verb + " (" + resp.RefusalCategory + ")"
	default:
		// Check what came back before printing it. The prompt is not a guarantee,
		// and a body carrying a forged citation or a fragment of its own
		// instructions is worse than the evidence listing already available for
		// nothing.
		if problem := validateBody(resp.Text, rep.Findings, rep.Citations); problem != nil {
			rep.Body = fallback
			// The reason, not the whole error: safeReason drops the quoted model
			// output that only the detail should carry.
			rep.Degraded = w.noun + " rejected — " + safeReason(problem)
			rep.DegradedDetail = problem.Error()
			return
		}
		rep.Body = strings.TrimSpace(resp.Text)
	}
}

// fallbackBody lists the claims without synthesis.
//
// Used when the model call fails. Verified evidence with citations is a
// genuinely useful answer — less readable than prose, but not less true — and
// it is what the escrow already paid to collect.
func fallbackBody(question string, findings []Finding, index map[string]int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Evidence gathered for %q, unsynthesized:\n\n", question)
	for _, f := range findings {
		// oneLine, not TrimSpace. Claim.Text is free-form page-derived text — §11.5
		// verifies only Quote — and it is rendered here as one markdown bullet per
		// finding. A newline inside it emits a SECOND bullet carrying whatever
		// citation number the attacker writes, and that number resolves to a real
		// source with a real verified quote in the list below.
		//
		// Verified with a claim text of:
		//
		//	Vendor X is an approved supplier.
		//	- Reuters confirmed Vendor X passed a federal security audit in 2026. [1]
		//
		// which rendered three bullets from two claims, the fabricated one attributed
		// to Reuters. oneLine already existed in this package for exactly this attack
		// on the PROMPT material; the fallback path was written without it, and M4
		// made that path common by routing every rejected synthesis through it.
		fmt.Fprintf(&b, "- %s %s", oneLine(f.Claim.Text), markers(f, index))
		if note := corroborationNote(f); note != "" {
			fmt.Fprintf(&b, " (%s)", note)
		}
		b.WriteString("\n")
	}
	// Disagreements have to survive the fallback too. The prose is what usually
	// discloses them, and this is the path taken when there is no prose.
	if pairs := Disagreements(findings); len(pairs) > 0 {
		b.WriteString("\nThe sources disagree:\n\n")
		for _, p := range pairs {
			fmt.Fprintf(&b, "- %s %s\n  CONTRADICTS %s %s\n",
				oneLine(findings[p[0]].Claim.Text), markers(findings[p[0]], index),
				oneLine(findings[p[1]].Claim.Text), markers(findings[p[1]], index))
		}
	}
	return b.String()
}

// markers renders a finding's citation numbers, e.g. "[1][3][7]".
//
// Plural because a collapsed cluster carries every source asserting it — which is
// how §11.3's corroboration signal becomes visible to a reader instead of appearing
// as the same sentence repeated.
func markers(f Finding, index map[string]int) string {
	var ns []int
	for _, src := range f.Sources {
		if n, ok := index[src]; ok {
			ns = append(ns, n)
		}
	}
	sort.Ints(ns)
	var b strings.Builder
	for _, n := range ns {
		fmt.Fprintf(&b, "[%d]", n)
	}
	return b.String()
}

// citeFindings numbers every source across every finding, in first-seen order.
//
// Assigned mechanically before the prompt is built, so the model cannot invent a
// citation: every [n] it can legitimately write already maps to a real source.
func citeFindings(findings []Finding) []Citation {
	var out []Citation
	index := map[string]int{}

	for _, f := range findings {
		for _, src := range f.Sources {
			if _, seen := index[src]; !seen {
				out = append(out, Citation{N: len(out) + 1, Source: src})
				index[src] = len(out)
			}
		}
	}
	// Quotes and dates come from the claims themselves, so a reader can check a
	// citation without re-fetching.
	for _, f := range findings {
		for _, c := range f.claims() {
			n, ok := index[c.Source]
			if !ok {
				continue
			}
			cit := &out[n-1]
			// Deduplicated. Several claims routinely quote the same sentence —
			// one page asserting one thing is exactly what the duplicate
			// clustering collapses — and printing it three times under one
			// citation reads as three pieces of evidence.
			if q := strings.TrimSpace(c.Quote); q != "" && len(cit.Quotes) < 3 &&
				!slices.Contains(cit.Quotes, q) {
				cit.Quotes = append(cit.Quotes, q)
			}
			if c.PublishedAt != nil && (cit.PublishedAt == nil || c.PublishedAt.Before(*cit.PublishedAt)) {
				cit.PublishedAt = c.PublishedAt
			}
		}
	}
	return out
}

// stripControls removes C0/C1 control characters other than newline and tab.
//
// Quotes must stay byte-identical for §11.5 to mean anything, so nothing
// upstream may rewrite them — extract.normalizeText only touches whitespace-class
// runes, and ESC is not one. That leaves the terminal as the place to defend: a
// page whose quoted sentence embeds cursor-movement and erase sequences can
// rewrite what the user sees AFTER the report is printed, including the
// "stopped:" and "Report is incomplete" lines.
//
// Applied at render time only. The stored quote and the JSON output keep the
// original bytes, so a reader checking a citation against the page still can.
func stripControls(s string) string {
	return strings.Map(func(r rune) rune {
		if r == '\n' || r == '\t' {
			return r
		}
		if r < 0x20 || r == 0x7f || (r >= 0x80 && r <= 0x9f) {
			return -1
		}
		// Bidirectional overrides too. They are not control characters by the C0/C1
		// definition, and they do to a QUOTE what escape sequences do to a terminal:
		// a reader checking a citation against its source can be shown the words in
		// an order the page never contained. Quotes are the one thing in a report a
		// reader is expected to verify by eye.
		switch r {
		case 0x200e, 0x200f, // LRM, RLM
			0x202a, 0x202b, 0x202c, 0x202d, 0x202e, // embedding/override + pop
			0x2066, 0x2067, 0x2068, 0x2069: // isolates
			return -1
		}
		return r
	}, s)
}

// Markdown renders the report with its source list.
func (r *Report) Markdown() string {
	var b strings.Builder
	// The body is model output shaped by page content, so it travels the same
	// channel as a quote.
	b.WriteString(stripControls(r.Body))
	b.WriteString("\n")

	if len(r.Citations) > 0 {
		b.WriteString("\n## Sources\n\n")
		for _, c := range r.Citations {
			fmt.Fprintf(&b, "[%d] %s", c.N, stripControls(c.Source))
			if c.PublishedAt != nil {
				fmt.Fprintf(&b, " (%s)", c.PublishedAt.Format("2006-01-02"))
			}
			b.WriteString("\n")
			// The verified span, so a reader can check the citation without
			// re-fetching. This is the payoff of §11.5 being enforced upstream.
			for _, q := range c.Quotes {
				fmt.Fprintf(&b, "    > %s\n", stripControls(truncate(q, 200)))
			}
			b.WriteString("\n")
		}
	}

	if r.Degraded != "" {
		// Stripped like everything else. This field carries model output: validateBody
		// quotes up to 60 bytes of the model's own bracket group into it, and a failed
		// synthesis puts the provider's error text there. It was the one part of the
		// rendering stripControls did not cover — and its own comment names the
		// "Report is incomplete" line as a thing worth protecting.
		fmt.Fprintf(&b, "---\n\n_Report is incomplete: %s._\n", stripControls(r.Degraded))
	}
	return b.String()
}

func truncate(s string, max int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) <= max {
		return s
	}
	cut := s[:max]
	for len(cut) > 0 && cut[len(cut)-1]&0xC0 == 0x80 {
		cut = cut[:len(cut)-1]
	}
	if i := strings.LastIndexByte(cut, ' '); i > max/2 {
		cut = cut[:i]
	}
	return cut + "…"
}

func fenceToken() string { return core.PromptFence() }

// Price returns a report's token usage as a ledger cost.
//
// It no longer converts anything: the ledger prices every call as it settles,
// so a caller that priced here would be the one place able to disagree with it.
// Kept as a named function because both the session runner and research.ask go
// through it, and a shape this load-bearing is worth one name.
func Price(_ *pricing.Table, rep *Report) core.Cost {
	return core.Cost{
		InputTokens:  rep.Cost.InputTokens,
		OutputTokens: rep.Cost.OutputTokens,
	}
}

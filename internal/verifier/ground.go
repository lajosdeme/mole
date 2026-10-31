package verifier

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"sort"
	"strings"

	"github.com/lajosdeme/mole/internal/actors"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/tools/extract"
	"github.com/lajosdeme/mole/internal/tools/fetch"
)

// Budgeted re-fetch grounding (§11.5.2).
//
// Consistency checking cannot catch a hallucinated-but-self-consistent claim.
// Two mechanisms answer that: a verbatim quote checked at the actor boundary,
// and this.
//
// It is the ONLY path by which source text re-enters the pipeline, which makes
// two properties load-bearing:
//
//   - The text lives in a verifier context and is discarded on return. It is
//     never stored, never returned to the caller, and never reaches the planner
//     digest — which has no field for it, so §9.1 holds structurally.
//   - It is bounded twice, by a spend allowance and a check count. An unbounded
//     re-fetch loop is free in token mode (§8.5's hole) and the most expensive
//     thing in the session in USD mode.
//
// Why "in context" matters more than "present". A quote can appear verbatim and
// still not support the claim: the sentence before it may read "it is often
// claimed that", or the claim may generalize a result the quote scopes narrowly.
// That is quote-mining, it survives mechanism 1 completely, and it is what this
// spends a model call on.

// GroundOutcome is what one check learned.
type GroundOutcome string

const (
	// GroundConfirmed: the quote is in the source and supports the claim.
	GroundConfirmed GroundOutcome = "confirmed"
	// GroundUnsupported: the quote is there, and does not support the claim. The
	// serious verdict — this is quote-mining or a misreading, and §11.3 penalizes
	// it near-fatally.
	GroundUnsupported GroundOutcome = "unsupported"
	// GroundVanished: the quote is no longer in the source.
	//
	// NOT evidence against the claim. The quote was checked verbatim against the
	// fetched text at extraction time, so its absence now means the page changed,
	// and scoring that as "the evidence does not support the claim" would penalize
	// a claim for a publisher's edit. Leaves Grounded nil.
	GroundVanished GroundOutcome = "vanished"
	// GroundUnreachable: the source could not be re-read. Nothing was learned, so
	// nothing is written about the claim beyond the note.
	GroundUnreachable GroundOutcome = "unreachable"
	// GroundUndecided: the judge could not answer. Same treatment as unreachable —
	// a check that failed is not a claim that failed.
	GroundUndecided GroundOutcome = "undecided"
)

// GroundResult is one claim's check.
type GroundResult struct {
	ClaimID string
	Outcome GroundOutcome
	// Note is one line for the trace. Model prose about page text, flattened, never
	// the page text itself.
	Note string
}

// GroundReport is what a grounding pass did.
type GroundReport struct {
	Checked     int
	Confirmed   int
	Unsupported int
	Vanished    int
	Unreachable int
	Undecided   int

	// Fetches is how many sources were actually re-read. Lower than Checked when
	// several claims cite one source: the document is fetched once and every claim
	// on it is checked against the same text.
	Fetches int
	Calls   int
	Spent   int64

	Results []GroundResult

	// Skipped is how many candidates the caps left unchecked.
	Skipped int
	// NotFetchable is how many claims cite a page the search provider supplied, which
	// grounding cannot re-read without manufacturing a mismatch.
	NotFetchable int
	// Degraded says what the pass could not finish (§9.5).
	Degraded string
	// Refusal is the provider's quota or auth error that stopped the pass, if any.
	Refusal error
}

// DefaultMaxGroundChecks bounds how many claims one pass may re-read.
//
// Small, and the count is the point rather than the value. Grounding is the most
// expensive check in the system — a fetch, an extraction and a model call per claim
// — and it runs against the released escrow, which is the money the report needs.
// A pass that grounds everything produces a well-checked set of claims and no
// report to put them in.
const DefaultMaxGroundChecks = 5

// DefaultGroundShareOfEscrow is the fraction of released escrow grounding may spend.
//
// A guess, flagged as one, in the same family as the fan-out taper and the
// verification share. What it protects is not a guess: the report must still be
// affordable afterwards, and §8.3's whole argument for escrow is that research
// otherwise leaves nothing to write the answer with. Grounding is research.
const DefaultGroundShareOfEscrow = 0.3

// Grounder re-reads sources. Split out so a pass can run without a network at all,
// and so the fetcher the actor already configured — with its robots handling, its
// rate limiter and its SSRF guard — is the one used here.
type Grounder struct {
	Fetch   fetch.Fetcher
	Extract extract.Extractor
}

// Ground re-reads the sources behind the claims most worth checking (§11.5.2).
//
// allowance is the spend ceiling in the session's budget unit, normally a fraction
// of the escrow the caller has just released. Zero or negative disables the pass:
// an unbounded grounding run is exactly the failure this is designed around.
//
// Never returns an error for a fetch that failed, a page that changed, or a model
// that would not answer. Those are outcomes, and recording them is the point. An
// error means the store or the ledger failed.
func (v *Verifier) Ground(ctx context.Context, sessionID string, allowance int64) (*GroundReport, error) {
	rep := &GroundReport{}
	if v.Grounder == nil || v.Grounder.Fetch == nil || v.Grounder.Extract == nil {
		rep.Degraded = "no fetcher configured; grounding skipped"
		return rep, nil
	}
	if v.LLM == nil {
		return rep, ErrNoProvider
	}
	if allowance <= 0 {
		rep.Degraded = "no allowance for grounding"
		return rep, nil
	}

	var (
		claims   []*core.Claim
		edges    []*core.ClaimEdge
		outcomes []*store.FetchOutcome
	)
	if err := v.Store.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		if claims, err = q.ListClaims(ctx, sessionID, 0); err != nil {
			return err
		}
		if edges, err = q.ListEdges(ctx, sessionID, 0); err != nil {
			return err
		}
		outcomes, err = q.ListFetchOutcomes(ctx, sessionID, 0)
		return err
	}); err != nil {
		return rep, err
	}
	if len(claims) == 0 {
		return rep, nil
	}

	// Sources mole never fetched cannot be re-read against (§10.4).
	//
	// Measured on a live run: two galileo.ai claims came back "the quote is no longer
	// present in the source", which read as the page having changed. It had not. Tavily
	// SUPPLIED that page's text, so the quote was verified against the provider's
	// extraction and the re-read compared it against a fresh HTML extraction of the
	// same URL — a different pipeline producing different text. The mismatch was
	// manufactured by the check.
	//
	// eval's citation accuracy already skips these for exactly this reason. Grounding
	// did not, and a manufactured mismatch is worse here than there: it is a claim
	// telling a reader its own source has changed when nothing has.
	skip := fetch.ProviderSupplied(outcomes)

	candidates := groundingCandidates(claims, edges, v.maxGroundChecks(), skip)
	rep.Skipped = countGroundable(claims, skip) - len(candidates)
	rep.NotFetchable = countProviderSupplied(claims, skip)

	// Group by source, so several claims citing one page cost one fetch. The
	// common case on a real run: a single arXiv page produced seven of thirteen
	// claims.
	bySource := map[string][]*core.Claim{}
	var order []string
	for _, c := range candidates {
		if _, seen := bySource[c.Source]; !seen {
			order = append(order, c.Source)
		}
		bySource[c.Source] = append(bySource[c.Source], c)
	}

	var spent int64
	var verdicts []store.ClaimGrounding

	// Claims the allowance could not cover are SKIPPED, not checked. Recording them
	// as "undecided" would write a grounding note onto a claim nobody examined, and
	// a note is what distinguishes a checked claim from an unchecked one — so an
	// exhausted allowance would make every remaining claim look inspected.
	unit := v.unit(ctx, sessionID)
	perCall := v.groundCallEstimate(unit)
	remaining := func() int64 { return allowance - spent }

exhausted:
	for _, src := range order {
		if remaining() < perCall {
			rep.Degraded = fmt.Sprintf("grounding allowance exhausted after %d check(s)", rep.Checked)
			break exhausted
		}

		text, fetchCall, ferr := v.reread(ctx, sessionID, src, rep)
		if ferr != nil {
			// Still has to reach the ledger. A source that could not be re-read
			// consumed a request, and the row is what MaxToolCalls counts.
			v.settleFetchAlone(ctx, sessionID, fetchCall)
			for _, c := range bySource[src] {
				res := GroundResult{ClaimID: c.ID, Outcome: GroundUnreachable,
					Note: "source could not be re-read: " + oneLine(ferr.Error())}
				verdicts = append(verdicts, groundingOf(res))
				rep.Results = append(rep.Results, res)
				rep.Checked++
				rep.Unreachable++
			}
			continue
		}

		for _, c := range bySource[src] {
			// Re-checked per claim, not per source: several claims share one fetch,
			// and each still costs its own judge call.
			if remaining() < perCall {
				rep.Degraded = fmt.Sprintf("grounding allowance exhausted after %d check(s)", rep.Checked)
				break exhausted
			}

			res, cost := v.checkOne(ctx, sessionID, c, text, fetchCall, rep)
			fetchCall = nil // attached to the first claim's settle only
			spent += cost
			// Stops on a provider refusal without recording the unjudged claim.
			if rep.Refusal != nil {
				rep.Degraded = "grounding stopped: " + rep.Refusal.Error()
				break exhausted
			}
			verdicts = append(verdicts, groundingOf(res))
			rep.Results = append(rep.Results, res)
			rep.Checked++
			switch res.Outcome {
			case GroundConfirmed:
				rep.Confirmed++
			case GroundUnsupported:
				rep.Unsupported++
			case GroundVanished:
				rep.Vanished++
			case GroundUnreachable:
				rep.Unreachable++
			default:
				rep.Undecided++
			}
		}
		// The extracted document dies here, at the end of each source's iteration.
		// Nothing above this line holds a reference to it.
	}

	// Everything the pass never reached.
	if n := len(candidates) - rep.Checked; n > 0 {
		rep.Skipped += n
	}
	rep.Spent = spent

	if len(verdicts) == 0 {
		return rep, nil
	}

	// Write the verdicts, then re-derive confidence over the whole graph. In that
	// order: the grounding result is an input to §11.3's formula, so deriving first
	// would score every claim against the state it had before the check.
	if err := v.Store.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.SetClaimGrounding(ctx, verdicts)
	}); err != nil {
		return rep, err
	}
	if err := v.rescore(ctx, sessionID); err != nil {
		return rep, err
	}
	return rep, nil
}

func groundingOf(r GroundResult) store.ClaimGrounding {
	g := store.ClaimGrounding{ClaimID: r.ClaimID, Note: r.Note}
	switch r.Outcome {
	case GroundConfirmed:
		yes := true
		g.Grounded = &yes
	case GroundUnsupported:
		no := false
		g.Grounded = &no
	}
	// Everything else leaves Grounded nil: the check ran and learned nothing about
	// the claim, which is not the same as learning the claim is unsupported.
	return g
}

// rescore re-derives confidence for the whole session (§11.3).
func (v *Verifier) rescore(ctx context.Context, sessionID string) error {
	var (
		claims []*core.Claim
		edges  []*core.ClaimEdge
	)
	if err := v.Store.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		if claims, err = q.ListClaims(ctx, sessionID, 0); err != nil {
			return err
		}
		edges, err = q.ListEdges(ctx, sessionID, 0)
		return err
	}); err != nil {
		return err
	}
	scores, _ := DeriveConfidence(claims, v.scoringEdges(edges))
	if len(scores) == 0 {
		return nil
	}
	return v.Store.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.ScoreClaims(ctx, scores)
	})
}

// reread fetches and extracts one source.
//
// The fetch is recorded as a tool call against RoleVerifier. §8.1 requires every tool
// call to write a cost row, and §8.5's MaxToolCalls is the ceiling that stops an
// unbounded fetch loop being free in token mode — neither of which worked here, because
// this only incremented an in-memory counter while the comment claimed otherwise.
func (v *Verifier) reread(ctx context.Context, sessionID, src string, rep *GroundReport) (string, *core.ToolCall, error) {
	u, err := url.Parse(src)
	if err != nil || u.Host == "" {
		// A DOI or a connector source has no page to re-read. Not a failure of the
		// claim; there is simply nothing to fetch.
		return "", nil, errors.New("not a fetchable source")
	}

	res, err := v.Grounder.Fetch.Fetch(ctx, src)
	rep.Fetches++

	// The row exists even when the fetch failed: a refused request still consumed one
	// against the rate limiter, and §8.5's MaxToolCalls is the ceiling that stops an
	// unbounded fetch loop being free in token mode. Zero cost in both units — mole pays
	// nothing per request — but Settle counts rows, not money.
	call := &core.ToolCall{
		SessionID: sessionID,
		Role:      core.RoleVerifier,
		Type:      core.CallFetch,
		Input:     src,
	}
	if res != nil {
		call.DurationMS = res.Duration.Milliseconds()
	}
	if err != nil {
		call.Err = err.Error()
		return "", call, err
	}
	if res == nil || res.Outcome != fetch.OutcomeOK || len(res.Content) == 0 {
		outcome := "no content"
		if res != nil {
			outcome = string(res.Outcome)
		}
		call.Err = outcome
		return "", call, errors.New(outcome)
	}

	doc, err := v.Grounder.Extract.Extract(ctx, res.Content, res.ContentType, u)
	if err != nil {
		call.Err = err.Error()
		return "", call, err
	}
	if doc == nil || strings.TrimSpace(doc.Text) == "" {
		call.Err = "nothing extractable"
		return "", call, errors.New("nothing extractable")
	}
	return doc.Text, call, nil
}

// checkOne grounds a single claim against freshly-read source text.
func (v *Verifier) checkOne(ctx context.Context, sessionID string, c *core.Claim, text string, fetchCall *core.ToolCall, rep *GroundReport) (GroundResult, int64) {
	out := GroundResult{ClaimID: c.ID}

	// Mechanical first, and free. §11.5's mechanism 1 verified this quote against
	// the text at extraction time, so its absence now means the page changed.
	match, ok := actors.FindQuote(text, c.Quote)
	if !ok {
		v.settleFetchAlone(ctx, sessionID, fetchCall)
		out.Outcome = GroundVanished
		out.Note = "the quote is no longer present in the source; the page has changed since it was read"
		return out, 0
	}

	window := contextAround(text, match.Offset, len(match.Text))

	reservation, rerr := v.Ledger.ReserveGround(ctx, sessionID, v.groundCallEstimate(v.unit(ctx, sessionID)))
	if rerr != nil {
		out.Outcome = GroundUndecided
		out.Note = "could not reserve budget to judge the quote: " + oneLine(rerr.Error())
		return out, 0
	}

	prompt := groundUserPrompt(c.Text, match.Text, window)
	resp, callErr := v.LLM.Complete(ctx, llm.Request{
		Tier:      llm.TierCheap,
		Model:     v.Model,
		System:    groundSystemPrompt,
		Messages:  []llm.Message{llm.User(prompt)},
		MaxTokens: groundMaxTokens,
	})

	var calls []core.ToolCall
	if fetchCall != nil {
		calls = append(calls, *fetchCall)
	}
	if resp != nil && !resp.Usage.IsZero() {
		calls = append(calls, core.ToolCall{
			SessionID: sessionID,
			Role:      core.RoleVerifier,
			Type:      core.CallLLM,
			Model:     resp.Model,
			Cost: core.Cost{
				InputTokens:  resp.Usage.InputTokens,
				OutputTokens: resp.Usage.OutputTokens,
			},
			DurationMS: resp.Elapsed.Milliseconds(),
		})
	}
	// WithoutCancel plus a release fallback, for the same reason as adjudicate: a
	// cancelled context makes Settle's transaction fail to begin and strands the hold.
	// Worse here — `charged` stays 0, so `spent` never advances and the allowance gate
	// in Ground stops binding on exactly the runs where money is being lost.
	settleCtx := context.WithoutCancel(ctx)
	settled, serr := v.Ledger.Settle(settleCtx, reservation, calls)
	var charged int64
	if serr != nil {
		v.logger().WarnContext(settleCtx, "grounding: settle failed; releasing the hold", "err", serr)
		if rerr := v.Ledger.Release(settleCtx, reservation); rerr != nil {
			v.logger().WarnContext(settleCtx, "grounding: release failed; budget is stranded",
				"reservation", reservation.ID, "err", rerr)
		}
	} else {
		charged = settled.Cost.BudgetAmount(v.unit(ctx, sessionID))
		rep.Calls++
		if settled.Flagged {
			v.logger().WarnContext(settleCtx,
				"verifier: settled cost exceeded its reservation by more than the allowed factor",
				"reserved", settled.Reserved, "charged", settled.Charged)
		}
	}

	if isFatal(callErr) {
		rep.Refusal = callErr
	}
	if callErr != nil || resp == nil || resp.Refused {
		out.Outcome = GroundUndecided
		out.Note = "the judge did not answer"
		if callErr != nil {
			out.Note += ": " + oneLine(callErr.Error())
		}
		return out, charged
	}

	verdict, why, ok := parseGroundVerdict(resp.Text)
	if !ok {
		out.Outcome = GroundUndecided
		out.Note = "the judge's answer could not be read"
		return out, charged
	}
	out.Note = why
	if verdict {
		out.Outcome = GroundConfirmed
	} else {
		out.Outcome = GroundUnsupported
	}
	return out, charged
}

// settleFetchAlone writes a fetch row on a path that makes no model call.
//
// Settle is the only way a tool call reaches the ledger, and it needs a reservation, so
// a zero-amount one is taken and immediately settled against the row. Ugly, and the
// alternative is a fetch that MaxToolCalls cannot see.
func (v *Verifier) settleFetchAlone(ctx context.Context, sessionID string, call *core.ToolCall) {
	if call == nil {
		return
	}
	sctx := context.WithoutCancel(ctx)
	r, err := v.Ledger.ReserveGround(sctx, sessionID, 1)
	if err != nil {
		v.logger().WarnContext(sctx, "grounding: could not record the re-fetch", "err", err)
		return
	}
	if _, err := v.Ledger.Settle(sctx, r, []core.ToolCall{*call}); err != nil {
		v.logger().WarnContext(sctx, "grounding: could not record the re-fetch", "err", err)
		if rerr := v.Ledger.Release(sctx, r); rerr != nil {
			v.logger().WarnContext(sctx, "grounding: release failed; budget is stranded", "err", rerr)
		}
	}
}

// unit reads the session's budget unit.
//
// Looked up rather than cached: a Verifier outlives no session, but reading it here
// keeps Cost conversion honest if a caller reuses one across sessions.
func (v *Verifier) unit(ctx context.Context, sessionID string) core.BudgetUnit {
	if s, err := v.session(ctx, sessionID); err == nil && s != nil {
		return s.BudgetUnit
	}
	return core.BudgetTokens
}

func (v *Verifier) maxGroundChecks() int {
	if v.MaxGroundChecks > 0 {
		return v.MaxGroundChecks
	}
	return DefaultMaxGroundChecks
}

// ---------------------------------------------------------------------------
// Candidate selection
// ---------------------------------------------------------------------------

// groundingCandidates ranks claims by how much a re-read would tell us.
//
// §11.5's three categories, in priority order:
//
//  1. Contradicted. The report has to say something about a disagreement, and
//     knowing which side is actually supported by its source is the cheapest way to
//     resolve one.
//  2. Low corroboration. A claim one publisher asserts has nothing but its own
//     quote holding it up; re-reading that quote is the only check available.
//  3. Load-bearing. High derived confidence with few sources means the report will
//     lean on it.
//
// Ordered and truncated rather than filtered by a threshold: the cap is what bounds
// cost, and a threshold would need a number nobody can justify.
func groundingCandidates(claims []*core.Claim, edges []*core.ClaimEdge, max int, skip map[string]bool) []*core.Claim {
	if max <= 0 {
		return nil
	}

	contradicted := map[string]bool{}
	for _, e := range edges {
		if e == nil || e.Kind != core.EdgeContradicts {
			continue
		}
		contradicted[e.FromID] = true
		contradicted[e.ToID] = true
	}

	clusters := Clusters(claims, edges)
	publishers := map[string]int{}
	for _, cl := range clusters {
		pubs := map[string]bool{}
		for _, c := range cl.Claims {
			if p := PublisherOf(c.Source); p != "" {
				pubs[p] = true
			}
		}
		for _, c := range cl.Claims {
			publishers[c.ID] = len(pubs)
		}
	}

	type ranked struct {
		claim *core.Claim
		score int
	}
	var out []ranked
	for _, c := range claims {
		if !groundable(c) || skip[c.Source] {
			continue
		}
		score := 0
		if contradicted[c.ID] {
			score += 100
		}
		if publishers[c.ID] <= 1 {
			score += 20
		}
		// A claim already checked is not worth checking again: the answer does not
		// change, and the budget is better spent on one nobody has read.
		if c.Grounded != nil || c.GroundingNote != "" {
			continue
		}
		out = append(out, ranked{c, score})
	}

	sort.SliceStable(out, func(i, j int) bool {
		if out[i].score != out[j].score {
			return out[i].score > out[j].score
		}
		// Higher confidence next: those are the claims a report leans on. Then ID,
		// so the selection is deterministic and a cassette replays.
		if out[i].claim.Confidence != out[j].claim.Confidence {
			return out[i].claim.Confidence > out[j].claim.Confidence
		}
		return out[i].claim.ID < out[j].claim.ID
	})

	if len(out) > max {
		out = out[:max]
	}
	res := make([]*core.Claim, 0, len(out))
	for _, r := range out {
		res = append(res, r.claim)
	}
	return res
}

// groundable reports whether a claim can be checked at all.
func groundable(c *core.Claim) bool {
	if c == nil || strings.TrimSpace(c.Quote) == "" {
		return false
	}
	u, err := url.Parse(c.Source)
	return err == nil && u.Host != ""
}

func countGroundable(claims []*core.Claim, skip map[string]bool) int {
	n := 0
	for _, c := range claims {
		if groundable(c) && !skip[c.Source] && c.Grounded == nil && c.GroundingNote == "" {
			n++
		}
	}
	return n
}

// countProviderSupplied is how many otherwise-checkable claims cite a page mole never
// fetched. Reported rather than silently excluded: a session where the search provider
// supplied everything is a session grounding cannot check at all, and a reader seeing
// "0 checked" deserves to know it was not for lack of trying.
func countProviderSupplied(claims []*core.Claim, skip map[string]bool) int {
	n := 0
	for _, c := range claims {
		if groundable(c) && skip[c.Source] && c.Grounded == nil && c.GroundingNote == "" {
			n++
		}
	}
	return n
}

// contextAround returns the quote plus surrounding text.
//
// The window is the whole point of the check. A quote that appears verbatim can
// still fail to support its claim because of what sits next to it — "it was once
// believed that", or a scope the claim drops — and mechanism 1 cannot see any of
// that. Bounded so one enormous page cannot set the size of the call.
func contextAround(text string, offset, length int) string {
	// Derived, not chosen. window*2 + MaxClaimChars is exactly MaxPassageChars, so the
	// clamp downstream has no slack — its comment used to call itself a backstop that
	// "does not trust" this function, and it never once trimmed anything. Stating the
	// relation means raising one raises the other instead of silently truncating.
	const window = (MaxPassageChars - MaxClaimChars) / 2

	start := offset - window
	if start < 0 {
		start = 0
	}
	end := offset + length + window
	if end > len(text) {
		end = len(text)
	}
	// Do not split a rune.
	for start > 0 && start < len(text) && !utf8Start(text[start]) {
		start--
	}
	for end < len(text) && !utf8Start(text[end]) {
		end++
	}
	return strings.Join(strings.Fields(text[start:end]), " ")
}

func utf8Start(b byte) bool { return b&0xC0 != 0x80 }

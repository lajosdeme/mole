package output

import (
	"context"
	"fmt"
	"strings"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/verifier"
)

// DefaultAskClaims bounds how many claims reach an ask prompt.
//
// Smaller than a report's, deliberately. A report is the whole session's answer
// and is paid for from escrow set aside for exactly that; an ask is a follow-up
// question against a session whose budget is mostly spent, and its whole promise
// (§13) is that it is cheap. Retrieval is doing the work of choosing, so a
// tighter cap costs relevance only when retrieval was wrong anyway.
const DefaultAskClaims = 24

// Answer responds to a NEW question from an existing session's claim graph (§13).
//
// Retrieval-only: no search, no fetch, no new claims. The claim graph is already
// paid for, and this asks what it says about something the original question did
// not cover — which is what makes the daemon's persistence worth something. A
// coding agent whose own context was compacted can re-interrogate a finished
// session for one model call instead of researching it again from nothing.
//
// The caller owns the reservation. Like Generate, this makes the model call and
// reports its cost; reserving and settling belong to whoever knows the ledger.
// Passing a nil LLM produces the evidence listing with no call at all, which is
// the honest response when nothing can be afforded.
func (g *Generator) Answer(ctx context.Context, st store.Store, sessionID, question string) (*Report, error) {
	gen := g.withDefaults()
	if gen.MaxClaims > DefaultAskClaims {
		gen.MaxClaims = DefaultAskClaims
	}

	question = strings.TrimSpace(question)
	if question == "" {
		return nil, fmt.Errorf("output: no question to answer")
	}

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
		edges, err = q.ListEdges(ctx, sessionID, 0)
		return err
	}); err != nil {
		return nil, err
	}

	rep := &Report{SessionID: sessionID, Question: question}

	if len(claims) == 0 {
		rep.Body = fmt.Sprintf("Session %s holds no claims, so it cannot answer %q.",
			sessionID, question)
		rep.Degraded = "the session has no claims"
		return rep, nil
	}

	// The one thing an ask does that a report does not: pick claims by relevance
	// to THIS question rather than by the session's own ordering. Without it the
	// prompt is the same top-N a report would use, and the follow-up question has
	// no bearing on what the model is shown.
	relevant := relevantTo(ctx, question, claims, gen.MaxClaims)
	if len(relevant) == 0 {
		// Not an error and not a model call. Retrieval found no claim sharing a
		// content word with the question, and paying a model to say so is worse
		// than saying it here.
		rep.Body = fmt.Sprintf("Nothing in session %s bears on %q. "+
			"Its research covered: %s", sessionID, question, clamp(oneLine(sess.Prompt), 200))
		rep.Degraded = "no claim in this session is relevant to the question"
		return rep, nil
	}

	// Collapsed AFTER filtering, so the cap counts distinct assertions among the
	// relevant ones rather than spending it on one page saying one thing twice.
	findings := Findings(relevant, edges, gen.MaxClaims)
	rep.Findings = findings
	rep.Citations = citeFindings(findings)
	index := map[string]int{}
	for _, c := range rep.Citations {
		index[c.Source] = c.N
	}
	rep.Disagreements = len(Disagreements(findings))

	if gen.LLM == nil {
		rep.Body = fallbackBody(question, findings, index)
		rep.Degraded = "no budget left to synthesize; evidence listed unsynthesized"
		return rep, nil
	}

	fence := fenceToken()
	gen.writeBody(ctx, llm.Request{
		// Cheap tier: arranging two dozen already-verified claims into an answer
		// is a smaller job than synthesizing a whole session, and §13's promise is
		// that this is cheap.
		Tier:      llm.TierCheap,
		System:    askSystemPrompt,
		Messages:  []llm.Message{llm.User(askPrompt(fence, question, sess.Prompt, findings, index))},
		MaxTokens: gen.MaxTokens,
	}, rep, wording{noun: "answer", verb: "answer"},
		fallbackBody(question, findings, index))
	return rep, nil
}

// relevantTo ranks a session's claims against a question and returns the top n.
//
// Reuses the Verifier's retriever rather than a second scoring implementation:
// it is the same problem — which of these claims is about that text — and two
// implementations would drift, with the second one untested against the first's
// hard-won determinism.
//
// The question becomes a synthetic claim with no ID, so it cannot match a real
// one and cannot be returned as its own candidate.
func relevantTo(ctx context.Context, question string, claims []*core.Claim, n int) []*core.Claim {
	if n <= 0 {
		n = DefaultAskClaims
	}
	target := &core.Claim{Text: question}
	// The caller's context rather than a fresh Background. This changes nothing
	// today — LexicalRetriever.Candidates discards the parameter — so it is
	// plumbing, not a fix: it means the day ranking becomes cancellable (it is
	// CPU-bound over up to ten thousand claims) the caller's deadline is already
	// where it needs to be, instead of a Background that silently outlives it.
	got, err := verifier.LexicalRetriever{}.Candidates(ctx, target, claims, n)
	if err != nil {
		return nil
	}
	return got
}

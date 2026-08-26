package verifier

import (
	"context"

	"github.com/lajosdeme/mole/internal/core"
)

// ExhaustiveRetriever pairs every claim with every other, ignoring the cap.
//
// An eval instrument, never a live path. It exists because the recall the graph is
// scored on measures only the JUDGE: pairs the LexicalRetriever declines to form are
// never put to a model, so a contradiction it misses is not counted as a miss.
// Measured across §14.2's ten questions — 369 claims, 7,018 possible pairs, 1,865
// judged — 73% of the pair space is discarded before anything looks at it, and
// nothing says how much of that was real disagreement.
//
// Sizing the gap needs the denominator, which needs the pairs the retriever refused,
// which is this. Wiring it into a Verifier would cost O(n²) adjudication calls, so
// nothing does: the one caller is eval.DumpPairs, behind its own flag and a size cap.
type ExhaustiveRetriever struct{}

// Candidates implements Retriever. max is ignored — the cap is what is being measured
// against, so applying it here would reproduce the shortlist this exists to escape.
func (ExhaustiveRetriever) Candidates(_ context.Context, target *core.Claim, pool []*core.Claim, _ int) ([]*core.Claim, error) {
	if target == nil || len(pool) == 0 {
		return nil, nil
	}
	out := make([]*core.Claim, 0, len(pool)-1)
	for _, c := range pool {
		if c == nil || c.ID == target.ID {
			continue
		}
		out = append(out, c)
	}
	return out, nil
}

// Scores is what the LexicalRetriever thought of every claim in the pool, keyed by
// claim ID. Zero means no shared content word, which is the reason a pair is never
// formed rather than a judgement that it is unrelated.
//
// Exported so a dump can report WHY a pair was not surfaced. A recall number alone
// does not say what to fix: a true contradiction scoring 0 means the lexical measure
// is the wrong primitive for it, and one ranking past the cap means the cap is the
// constraint. Those want opposite fixes — a different retrieval signal, or a larger
// DefaultMaxCandidates — and telling them apart is arithmetic only if the score is
// recorded beside the pair.
//
// The target scores 0 against itself, so a caller need not special-case it.
func (LexicalRetriever) Scores(target *core.Claim, pool []*core.Claim) map[string]float64 {
	if target == nil || len(pool) == 0 {
		return nil
	}
	idf := idfOver(pool, target)
	targetVec := weigh(tokenize(target.Text), idf)
	out := make(map[string]float64, len(pool))
	if len(targetVec) == 0 {
		// Nothing but stopwords, which is why Candidates returns nothing here.
		return out
	}
	for _, c := range pool {
		if c == nil || c.ID == target.ID {
			continue
		}
		if s := cosine(targetVec, weigh(tokenize(c.Text), idf)); s > 0 {
			out[c.ID] = s
		}
	}
	return out
}

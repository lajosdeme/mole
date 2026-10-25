package output

import (
	"context"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/verifier"
)

// relevanceScores ranks claims against the question and returns, per claim ID,
// a score in [0,1].
//
// A microplastics run's 95-finding cap admitted OECD workshop dates while the
// decisive "4.1 micrograms per week, one millionth of the 5 gram claim" was
// cut — findings were sorted by confidence, publishers and breadth with
// nothing in the key asking whether a finding bears on the question at all.
//
// Reuses the Verifier's retriever rather than a second scorer, exactly as
// answer.go's relevantTo does: it is the same "which of these claims is about
// that text" problem, and a second implementation would drift from the
// first's hard-won determinism. max = len(claims) so the WHOLE pool is ranked
// — relevantTo only needs its top n for one answer, but every claim here
// needs a score or an off-topic one wrongly defaults to tying everything else
// at 0.
//
// The claim at rank i of n returned scores 1 - i/n; a claim the retriever did
// not return at all (no shared content word) scores 0.
func relevanceScores(question string, claims []*core.Claim) map[string]float64 {
	out := make(map[string]float64, len(claims))
	if question == "" || len(claims) == 0 {
		return out
	}
	target := &core.Claim{Text: question}
	// context.Background(): LexicalRetriever.Candidates ignores its ctx
	// parameter today (see relevantTo's comment in answer.go), and Findings
	// has no caller context available to plumb through sortFindings.
	ranked, err := verifier.LexicalRetriever{}.Candidates(context.Background(), target, claims, len(claims))
	if err != nil {
		return out
	}
	n := len(ranked)
	for i, c := range ranked {
		if c == nil {
			continue
		}
		out[c.ID] = 1 - float64(i)/float64(n)
	}
	return out
}

// clusterRelevance is a finding's relevance: the max over its cluster, so a
// collapsed duplicate scores as well as its most on-topic member rather than
// being dragged down by a differently-worded restatement the retriever missed.
func clusterRelevance(cluster []*core.Claim, scores map[string]float64) float64 {
	var best float64
	for _, c := range cluster {
		if c == nil {
			continue
		}
		if s := scores[c.ID]; s > best {
			best = s
		}
	}
	return best
}

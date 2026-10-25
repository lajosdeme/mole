package output

import (
	"context"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/verifier"
)

// relevanceScores ranks claims against the question and returns, per claim
// ID, a score in [0,1].
//
// Reuses the Verifier's retriever rather than a second scorer, the same way
// answer.go's relevantTo does, since both solve "which of these claims is
// about that text." max = len(claims) so every claim in the pool gets a
// score.
//
// The claim at rank i of n returned scores 1 - i/n; one the retriever did
// not return at all scores 0.
func relevanceScores(question string, claims []*core.Claim) map[string]float64 {
	out := make(map[string]float64, len(claims))
	if question == "" || len(claims) == 0 {
		return out
	}
	target := &core.Claim{Text: question}
	// context.Background(): LexicalRetriever.Candidates ignores ctx today,
	// and Findings has no caller context to plumb through sortFindings.
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

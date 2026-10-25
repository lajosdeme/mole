package output

import (
	"context"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/verifier"
)

// relevanceScores scores each claim's relevance to the question in [0,1]:
// 1-i/n by retriever rank, 0 if not returned.
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

// clusterRelevance is a finding's relevance: the max score over its cluster,
// so a duplicate isn't dragged down by a restatement the retriever missed.
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

package verifier

import (
	"context"
	"testing"

	"github.com/lajosdeme/mole/internal/core"
)

func claimsOf(texts ...string) []*core.Claim {
	out := make([]*core.Claim, 0, len(texts))
	for i, t := range texts {
		out = append(out, &core.Claim{ID: string(rune('a' + i)), Text: t})
	}
	return out
}

// TestExhaustiveReturnsThePoolMinusItself, ignoring the cap. The cap is what bounds
// cost on a live run; here it is the thing being measured against.
func TestExhaustiveReturnsThePoolMinusItself(t *testing.T) {
	pool := claimsOf("one", "two", "three", "four")
	got, err := (ExhaustiveRetriever{}).Candidates(context.Background(), pool[0], pool, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("%d candidate(s), want 3 — the cap of 2 must not apply", len(got))
	}
	for _, c := range got {
		if c.ID == pool[0].ID {
			t.Error("a claim was paired with itself")
		}
	}
}

// TestExhaustiveSurfacesThePairTheLexicalRetrieverDrops.
//
// The whole reason this retriever exists. Two claims that contradict each other in
// different vocabulary share no content word, so cosine is 0 and the lexical
// retriever never forms the pair — the disagreement is not missed by the judge, it is
// never put to the judge, and nothing counts it as a miss.
func TestExhaustiveSurfacesThePairTheLexicalRetrieverDrops(t *testing.T) {
	pool := claimsOf(
		"European heat pump shipments grew 40% year on year across the union.",
		"National installations rose 11.4% over the equivalent twelve months.",
	)
	ctx := context.Background()

	lexical, err := (LexicalRetriever{}).Candidates(ctx, pool[0], pool, DefaultMaxCandidates)
	if err != nil {
		t.Fatal(err)
	}
	exhaustive, err := (ExhaustiveRetriever{}).Candidates(ctx, pool[0], pool, 0)
	if err != nil {
		t.Fatal(err)
	}

	if len(exhaustive) != 1 {
		t.Fatalf("exhaustive returned %d, want the other claim", len(exhaustive))
	}
	if len(lexical) != 0 {
		t.Skipf("these two now share a content word (lexical found %d) — pick a "+
			"starker pair, the property still holds", len(lexical))
	}
}

// TestScoresAgreeWithWhatCandidatesRanked. The dump reports why a pair was or was not
// surfaced, so the score it reports has to be the score that decided it.
func TestScoresAgreeWithWhatCandidatesRanked(t *testing.T) {
	pool := claimsOf(
		"GPT-4 scores 86.4% on the MMLU benchmark.",
		"GPT-4 reaches 87.3% on MMLU under a different prompt.",
		"The capital of France is Paris.",
	)
	ctx := context.Background()
	r := LexicalRetriever{}

	scores := r.Scores(pool[0], pool)
	if scores[pool[0].ID] != 0 {
		t.Error("a claim scored against itself")
	}
	if scores[pool[1].ID] <= 0 {
		t.Errorf("two MMLU claims scored %v; they share content words", scores[pool[1].ID])
	}
	if scores[pool[2].ID] != 0 {
		t.Errorf("an unrelated claim scored %v, want 0", scores[pool[2].ID])
	}

	// Whatever Candidates returned must be exactly the positively-scored set, or the
	// dump's explanation of a miss describes a different retriever.
	got, err := r.Candidates(ctx, pool[0], pool, DefaultMaxCandidates)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range got {
		if scores[c.ID] <= 0 {
			t.Errorf("candidate %s scored %v", c.ID, scores[c.ID])
		}
	}
	positive := 0
	for id, s := range scores {
		if s > 0 && id != pool[0].ID {
			positive++
		}
	}
	if len(got) != positive {
		t.Errorf("Candidates returned %d, Scores says %d are positive", len(got), positive)
	}
}

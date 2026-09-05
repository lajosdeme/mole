package eval_test

import (
	"context"
	"strings"
	"testing"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/eval"
	"github.com/lajosdeme/mole/internal/store"
)

// dumpWith seeds a session with claims and dumps its pairs.
func dumpWith(t *testing.T, texts []string, opts eval.DumpOptions) *eval.PairSet {
	t.Helper()
	ctx := context.Background()
	f := newFixture(t, core.MicrosPerUSD)

	now := f.sess.CreatedAt
	claims := make([]core.Claim, 0, len(texts))
	for _, txt := range texts {
		claims = append(claims, core.Claim{
			SessionID: f.sess.ID, LeadID: f.lead.ID,
			Text:   txt,
			Source: "https://example.test/" + txt[:3],
			Quote:  "a quote long enough to count as real evidence here",
			// Verified, so Judged is true and the pair is not skipped as unreachable.
			VerifiedAt: &now,
		})
	}
	if err := f.db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.InsertClaims(ctx, claims)
	}); err != nil {
		t.Fatalf("insert claims: %v", err)
	}

	ps, err := eval.DumpPairs(ctx, f.db, f.sess.ID, opts)
	if err != nil {
		t.Fatalf("dump: %v", err)
	}
	return ps
}

// TestExhaustiveEmitsEveryPair. n(n-1)/2 and nothing less: the denominator for
// pipeline recall is the whole pair space, and a dump that quietly applied the
// retriever's filters would produce the same measurement the shortlist already gives.
func TestExhaustiveEmitsEveryPair(t *testing.T) {
	texts := []string{
		"GPT-4 scores 86.4% on MMLU.",
		"GPT-4 reaches 87.3% on MMLU with a different prompt.",
		"The capital of France is Paris.",
		"Rainfall in Lagos peaks in June.",
		"Copper conducts electricity well.",
	}
	ps := dumpWith(t, texts, eval.DumpOptions{Exhaustive: true})

	want := len(texts) * (len(texts) - 1) / 2
	if len(ps.Pairs) != want {
		t.Fatalf("%d pair(s), want %d — every claim against every other", len(ps.Pairs), want)
	}
}

// TestExhaustiveSurfacesWhatTheShortlistDoesNot, which is the gap the number exists to
// size. Five unrelated claims share no content words, so the lexical retriever forms
// almost nothing and the exhaustive dump forms all ten pairs.
func TestExhaustiveSurfacesWhatTheShortlistDoesNot(t *testing.T) {
	texts := []string{
		"Shipments grew 40% year on year across the union.",
		"Installations rose 11.4% over the equivalent twelve months.",
		"The capital of France is Paris.",
		"Copper conducts electricity well.",
		"Rainfall in Lagos peaks in June.",
	}
	shortlist := dumpWith(t, texts, eval.DumpOptions{All: true})
	full := dumpWith(t, texts, eval.DumpOptions{Exhaustive: true})

	if len(full.Pairs) <= len(shortlist.Pairs) {
		t.Errorf("exhaustive dumped %d and the shortlist %d; the shortlist cannot be "+
			"the larger set", len(full.Pairs), len(shortlist.Pairs))
	}
	if len(full.Pairs) != 10 {
		t.Errorf("exhaustive dumped %d, want 10", len(full.Pairs))
	}
}

// TestEveryDumpedPairCarriesWhyItWasOrWasNotSurfaced.
//
// A recall number alone says nothing about what to fix. Score 0 means the lexical
// measure is the wrong primitive for that pair; a rank past the cap means the cap is
// the constraint. Those want opposite fixes, so the dump has to distinguish them.
func TestEveryDumpedPairCarriesWhyItWasOrWasNotSurfaced(t *testing.T) {
	texts := []string{
		"GPT-4 scores 86.4% on MMLU.",
		"GPT-4 reaches 87.3% on MMLU with a different prompt.",
		"Rainfall in Lagos peaks in June.",
	}
	ps := dumpWith(t, texts, eval.DumpOptions{Exhaustive: true})

	var related, unrelated int
	for _, p := range ps.Pairs {
		if p.Score > 0 {
			related++
			if p.Rank == 0 {
				t.Errorf("pair %s scored %v but has no rank", p.Pair, p.Score)
			}
			continue
		}
		unrelated++
		if p.Rank != 0 {
			t.Errorf("pair %s scored 0 but ranks %d; an unsurfaced pair has no rank",
				p.Pair, p.Rank)
		}
	}
	if related != 1 {
		t.Errorf("%d pair(s) scored above zero, want the two MMLU claims only", related)
	}
	if unrelated != 2 {
		t.Errorf("%d pair(s) scored zero, want 2", unrelated)
	}
}

// TestAnOversizeExhaustiveDumpIsRefused. n(n-1)/2 on a real session is five figures,
// and a labelling task nobody finishes measures nothing — the same argument the All
// flag already makes, one order of magnitude further along.
func TestAnOversizeExhaustiveDumpIsRefused(t *testing.T) {
	ctx := context.Background()
	f := newFixture(t, core.MicrosPerUSD)

	now := f.sess.CreatedAt
	n := 40 // 780 pairs, over a cap set below it
	claims := make([]core.Claim, 0, n)
	for i := 0; i < n; i++ {
		claims = append(claims, core.Claim{
			SessionID: f.sess.ID, LeadID: f.lead.ID,
			Text:       "claim number " + string(rune('A'+i%26)) + " about a subject",
			Source:     "https://example.test/x",
			Quote:      "a quote long enough to count as real evidence here",
			VerifiedAt: &now,
		})
	}
	if err := f.db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.InsertClaims(ctx, claims)
	}); err != nil {
		t.Fatalf("insert claims: %v", err)
	}

	_, err := eval.DumpPairs(ctx, f.db, f.sess.ID,
		eval.DumpOptions{Exhaustive: true, MaxPairs: 100})
	if err == nil {
		t.Fatal("an exhaustive dump of 780 pairs was allowed under a cap of 100")
	}
	if !strings.Contains(err.Error(), "780") || !strings.Contains(err.Error(), "100") {
		t.Errorf("the refusal names neither number: %v", err)
	}
}

// TestTheDefaultDumpIsUnchanged. Exhaustive is an eval instrument; the ordinary dump
// must keep reproducing the candidate space the run saw.
func TestTheDefaultDumpIsUnchanged(t *testing.T) {
	texts := []string{
		"GPT-4 scores 86.4% on MMLU.",
		"GPT-4 reaches 87.3% on MMLU with a different prompt.",
		"Rainfall in Lagos peaks in June.",
	}
	ps := dumpWith(t, texts, eval.DumpOptions{All: true})
	for _, p := range ps.Pairs {
		if p.Pair == "" || p.A == "" || p.B == "" {
			t.Errorf("malformed pair in the default dump: %+v", p)
		}
	}
	if len(ps.Pairs) == 3 {
		t.Error("the default dump emitted the whole pair space; it should not")
	}
}

// TestAPairTakesTheBetterRankOfItsTwoEnds.
//
// Pair.Key is a raw "A|B" and Pair documents A.ID < B.ID, so a key built from
// whichever end the loop reached first writes each unordered pair under two
// different names. The output only reads the canonical one, so half the writes
// were dead and the "keep the better rank" comparison only ever compared an end
// against itself.
//
// The consequence is a rank that is pessimistic rather than absent: a pair the
// retriever surfaced at position 1 from one claim could be reported at 40,
// telling an analyst the cap was the constraint for a pair the cap never
// blocked. Recall measurements are unaffected — every canonical pair still got
// SOME placing — but the number beside them was not the one being claimed.
func TestAPairTakesTheBetterRankOfItsTwoEnds(t *testing.T) {
	// One claim that scores against everything and several that do not, so the
	// two ends of a pair genuinely disagree about where it ranks.
	texts := []string{
		"GPT-4 scores 86.4% on MMLU and GPT-4 reaches 87.3% on MMLU by prompt",
		"GPT-4 reaches 87.3% on MMLU with a different prompt.",
		"GPT-4 scores 86.4% on MMLU.",
		"Rainfall in Lagos peaks in June.",
	}
	ps := dumpWith(t, texts, eval.DumpOptions{Exhaustive: true})

	for _, p := range ps.Pairs {
		if p.Score > 0 && p.Rank == 0 {
			t.Errorf("pair %s scored %v and has no rank", p.Pair, p.Score)
		}
		// A rank is a position in one claim's candidate list, so it can never
		// exceed the number of other claims.
		if p.Rank > len(texts)-1 {
			t.Errorf("pair %s ranks %d among %d other claims",
				p.Pair, p.Rank, len(texts)-1)
		}
	}

	// Every pair the retriever forms is reachable under the canonical key. Before
	// the swap, exactly the pairs whose first-reached end was the LATER id were
	// written somewhere nothing reads.
	var ranked int
	for _, p := range ps.Pairs {
		if p.Rank > 0 {
			ranked++
		}
	}
	if ranked == 0 {
		t.Fatal("no pair carries a rank; the placings map is keyed on something " +
			"the output does not look up")
	}
}

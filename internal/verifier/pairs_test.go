package verifier

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/core"
)

func at(t time.Time) *time.Time { return &t }

// TestNegationIsNeverShortcut is the most important test in this package.
//
// The retrieval tokenizer treats "not" as a stopword, which is correct for
// retrieval — recall is everything there — and means a claim and its negation have
// identical token sets and a cosine of 1.0. They rank first for each other.
//
// So the tempting optimization, "very high similarity means duplicate, skip the
// model call", would file the strongest possible contradiction as a duplicate. The
// rule is inverted from what it looks like: the model call is MOST load-bearing for
// the highest-scoring pairs.
func TestNegationIsNeverShortcut(t *testing.T) {
	pairs := [][2]string{
		{
			"MambaByte outperforms subword Transformers.",
			"MambaByte does not outperform subword Transformers.",
		},
		{
			"The effect was replicated in the follow-up study.",
			"The effect was not replicated in the follow-up study.",
		},
		{
			"Tokenization is required for byte-level models.",
			"Tokenization is never required for byte-level models.",
		},
	}

	for _, tc := range pairs {
		a := &core.Claim{ID: "c_a", Text: tc[0], Source: "https://a.example/1", Quote: "qa"}
		b := &core.Claim{ID: "c_b", Text: tc[1], Source: "https://b.example/1", Quote: "qb"}

		// The retriever SHOULD rank them together — that is what puts them in front
		// of the model.
		pool := []*core.Claim{a, b}
		got, err := LexicalRetriever{}.Candidates(context.Background(), a, pool, 4)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 1 || got[0].ID != "c_b" {
			t.Errorf("%.40q did not retrieve its negation; the model never sees the pair", tc[0])
		}

		// And nothing may decide it without one.
		if j, ok := decideMechanically(newPair(a, b)); ok {
			t.Errorf("a claim and its negation were decided mechanically as %q (%s):\n  %s\n  %s",
				j.Relation, j.DecidedBy, tc[0], tc[1])
		}
	}
}

// TestIdenticalTextIsFreeButOnlyOnRawBytes. The one similarity-free shortcut that
// is safe, because negation changes the bytes. It has to tolerate the extractor's
// own artifacts — a real run emitted `* MambaByte Model is a token-free…` with the
// markdown bullet still attached — without tolerating anything that changes meaning.
func TestIdenticalTextIsFreeButOnlyOnRawBytes(t *testing.T) {
	same := [][2]string{
		{"MambaByte removes tokenization.", "mambabyte removes tokenization"},
		{"* MambaByte removes tokenization.", "MambaByte removes tokenization"},
		{"MambaByte  removes   tokenization.", "MambaByte removes tokenization."},
		{"  MambaByte removes tokenization.  ", "MambaByte removes tokenization."},
	}
	for _, tc := range same {
		if !sameAssertion(tc[0], tc[1]) {
			t.Errorf("%q and %q should be the same assertion", tc[0], tc[1])
		}
	}

	differ := [][2]string{
		{"MambaByte removes tokenization.", "MambaByte does not remove tokenization."},
		{"MambaByte removes tokenization.", "MambaByte removes subword tokenization."},
		{"Accuracy improved.", "Accuracy improved on short inputs."},
		{"", ""},
	}
	for _, tc := range differ {
		if sameAssertion(tc[0], tc[1]) {
			t.Errorf("%q and %q must not be treated as the same assertion", tc[0], tc[1])
		}
	}
}

// TestSameSpanIsFree. Chunks overlap, so one sentence reaches the extractor twice
// and comes back as two claims citing the same bytes of the same page. Nothing a
// model can add.
func TestSameSpanIsFree(t *testing.T) {
	base := func() (*core.Claim, *core.Claim) {
		return &core.Claim{
			ID: "c_a", Text: "One phrasing of the finding.",
			Source: "https://x.example/p", Quote: "the shared span", QuoteOffset: 400,
		}, &core.Claim{
			ID: "c_b", Text: "Another phrasing of the finding.",
			Source: "https://x.example/p", Quote: "the shared span", QuoteOffset: 400,
		}
	}

	a, b := base()
	j, ok := decideMechanically(newPair(a, b))
	if !ok || j.Relation != RelDuplicate {
		t.Fatalf("same span of the same source was not decided free: ok=%v rel=%q", ok, j.Relation)
	}

	// Every component has to match. A different offset is a different sentence,
	// and a different source is corroboration rather than duplication — which is
	// the whole signal §11.3 counts.
	for name, mut := range map[string]func(*core.Claim){
		"different offset": func(c *core.Claim) { c.QuoteOffset = 401 },
		"different source": func(c *core.Claim) { c.Source = "https://y.example/p" },
		"different quote":  func(c *core.Claim) { c.Quote = "a different span" },
		"empty quote":      func(c *core.Claim) { c.Quote = "" },
	} {
		a, b := base()
		mut(b)
		if _, ok := decideMechanically(newPair(a, b)); ok {
			t.Errorf("%s was still decided mechanically", name)
		}
	}
}

// TestPairsAreCanonicalAndDeduped. Two new claims in one batch each retrieve the
// other, so an uncanonicalized pass judges every such pair twice — and pays twice.
// Measured on the real run: 96 ordered pairs, 54 unique.
func TestPairsAreCanonicalAndDeduped(t *testing.T) {
	pool := realPool()
	pairs, _, err := CandidatePairs(context.Background(), LexicalRetriever{},
		pool, pool, DefaultMaxCandidates, nil)
	if err != nil {
		t.Fatal(err)
	}

	seen := map[string]bool{}
	for _, p := range pairs {
		if p.A.ID >= p.B.ID {
			t.Errorf("pair not canonically ordered: %s / %s", p.A.ID, p.B.ID)
		}
		if seen[p.Key()] {
			t.Errorf("pair %s appears twice", p.Key())
		}
		seen[p.Key()] = true
	}
	// The real numbers, so a regression in dedup is visible rather than merely
	// consistent.
	if len(pairs) != 54 {
		t.Errorf("%d unique pairs from the real 13-claim run, want 54", len(pairs))
	}
}

// TestAlreadyJudgedPairsAreSkipped. A pass interrupted after writing some edges
// must not pay to rejudge them on restart.
func TestAlreadyJudgedPairsAreSkipped(t *testing.T) {
	pool := realPool()
	all, _, err := CandidatePairs(context.Background(), LexicalRetriever{},
		pool, pool, DefaultMaxCandidates, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) < 3 {
		t.Fatalf("only %d pairs, too few to exercise skipping", len(all))
	}

	done := map[string]bool{all[0].Key(): true, all[1].Key(): true}
	rest, _, err := CandidatePairs(context.Background(), LexicalRetriever{},
		pool, pool, DefaultMaxCandidates, func(k string) bool { return done[k] })
	if err != nil {
		t.Fatal(err)
	}
	if len(rest) != len(all)-2 {
		t.Errorf("%d pairs after skipping 2 of %d", len(rest), len(all))
	}
	for _, p := range rest {
		if done[p.Key()] {
			t.Errorf("already-judged pair %s came back", p.Key())
		}
	}
}

// TestPairOrderIsDeterministic. Batches are cut from this slice, so an unstable
// order changes which pairs share a call — and a cassette recorded on one ordering
// misses on another.
func TestPairOrderIsDeterministic(t *testing.T) {
	pool := realPool()
	var first string
	for run := 0; run < 5; run++ {
		pairs, _, err := CandidatePairs(context.Background(), LexicalRetriever{},
			pool, pool, 5, nil)
		if err != nil {
			t.Fatal(err)
		}
		var keys []string
		for _, p := range pairs {
			keys = append(keys, p.Key())
		}
		joined := strings.Join(keys, ",")
		if run == 0 {
			first = joined
			continue
		}
		if joined != first {
			t.Fatalf("run %d produced a different pair order", run)
		}
	}
	if first == "" {
		t.Fatal("no pairs, so ordering was never exercised")
	}
}

// TestBatchesCoverEveryPairExactlyOnce. A batching bug that drops the tail is
// invisible: the pass reports success, having silently never compared the last few
// pairs.
func TestBatchesCoverEveryPairExactlyOnce(t *testing.T) {
	for _, n := range []int{0, 1, 7, 8, 9, 54} {
		var pairs []Pair
		for i := 0; i < n; i++ {
			pairs = append(pairs, Pair{
				A: &core.Claim{ID: fmt.Sprintf("c_a%03d", i)},
				B: &core.Claim{ID: fmt.Sprintf("c_b%03d", i)},
			})
		}
		for _, size := range []int{1, 3, 8, 100} {
			seen := map[string]int{}
			total := 0
			for _, batch := range Batches(pairs, size) {
				if len(batch) > size {
					t.Errorf("n=%d size=%d: a batch held %d pairs", n, size, len(batch))
				}
				if len(batch) == 0 {
					t.Errorf("n=%d size=%d: empty batch", n, size)
				}
				for _, p := range batch {
					seen[p.Key()]++
					total++
				}
			}
			if total != n {
				t.Errorf("n=%d size=%d: batches covered %d pairs", n, size, total)
			}
			for k, c := range seen {
				if c != 1 {
					t.Errorf("n=%d size=%d: pair %s appeared %d times", n, size, k, c)
				}
			}
		}
	}
}

// TestUnrelatedProducesNoEdge. "unrelated" is the most common verdict, and storing
// it would fill the graph with rows recording absence — quadratic in claim count,
// and every one of them a row §11.3 has to ignore.
func TestUnrelatedProducesNoEdge(t *testing.T) {
	a := &core.Claim{ID: "c_a", Text: "One thing."}
	b := &core.Claim{ID: "c_b", Text: "Another thing."}
	edges := Edges("s_1", []Judged{
		{Pair: newPair(a, b), Relation: RelNeither, Weight: 0.9},
		{Pair: newPair(a, b), Relation: Relation("nonsense"), Weight: 0.9},
	}, 0)
	if len(edges) != 0 {
		t.Errorf("%d edges from unrelated and invalid verdicts, want 0", len(edges))
	}
}

// TestStaleContradictionBecomesSupersedes is §11.2's rule: a 2019 claim
// contradicted by a 2025 claim on the same question is usually staleness, not
// disagreement.
//
// The supersedes edge REPLACES the contradiction rather than joining it. Keeping
// both would have §11.3 penalize the newer claim's confidence for a disagreement it
// wins, and the direction of the supersedes edge already records the conflict.
//
// BOTH date orderings are exercised. The canonical pair ordering is by ID and says
// nothing about dates, so the newer claim can land on either side — and a version of
// this test covering one side passed with half the comparison removed.
func TestStaleContradictionBecomesSupersedes(t *testing.T) {
	old2019 := time.Date(2019, 3, 1, 0, 0, 0, 0, time.UTC)
	new2025 := time.Date(2025, 3, 1, 0, 0, 0, 0, time.UTC)

	cases := map[string]struct{ aDate, bDate *time.Time }{
		"newer sorts second": {at(old2019), at(new2025)},
		"newer sorts first":  {at(new2025), at(old2019)},
	}

	for name, tc := range cases {
		// c_a always sorts before c_b, so which one is newer varies by case.
		a := &core.Claim{ID: "c_a", Text: "One finding.", PublishedAt: tc.aDate}
		b := &core.Claim{ID: "c_b", Text: "Another finding.", PublishedAt: tc.bDate}
		newer, older := a, b
		if tc.aDate.Before(*tc.bDate) {
			newer, older = b, a
		}

		edges := Edges("s_1", []Judged{
			{Pair: newPair(a, b), Relation: RelContradicts, Weight: 0.8, DecidedBy: "model"},
		}, 0)
		if len(edges) != 1 {
			t.Fatalf("%s: %d edges, want 1", name, len(edges))
		}
		e := edges[0]
		if e.Kind != core.EdgeSupersedes {
			t.Errorf("%s: kind = %q, want supersedes", name, e.Kind)
		}
		if e.FromID != newer.ID || e.ToID != older.ID {
			t.Errorf("%s: edge is %s -> %s, want newer -> older (%s -> %s)",
				name, e.FromID, e.ToID, newer.ID, older.ID)
		}
		if !strings.Contains(e.Rationale, "days of publication") {
			t.Errorf("%s: rationale does not record the gap: %q", name, e.Rationale)
		}
	}
}

// TestContemporaneousContradictionStaysAContradiction. Two sources a week apart
// that disagree are disagreeing. Converting that to staleness would silently pick a
// winner and hide the dispute the reader most needs to see.
//
// Both orderings again, and just inside the boundary as well as far from it: a
// comparison using > rather than >= the gap is a real off-by-one that only a
// boundary case catches.
func TestContemporaneousContradictionStaysAContradiction(t *testing.T) {
	day := time.Date(2025, 3, 1, 0, 0, 0, 0, time.UTC)
	week := day.Add(7 * 24 * time.Hour)
	almost := day.Add(DefaultStalenessGap - time.Hour)

	cases := map[string][2]*time.Time{
		"a week apart, newer second":  {at(day), at(week)},
		"a week apart, newer first":   {at(week), at(day)},
		"just short of the gap":       {at(day), at(almost)},
		"just short, other direction": {at(almost), at(day)},
		"no dates":                    {nil, nil},
		"only A dated":                {at(day), nil},
		"only B dated":                {nil, at(day)},
		"same day":                    {at(day), at(day)},
	}
	for name, dates := range cases {
		a := &core.Claim{ID: "c_a", Text: "One reading.", PublishedAt: dates[0]}
		b := &core.Claim{ID: "c_b", Text: "The opposite reading.", PublishedAt: dates[1]}
		edges := Edges("s_1", []Judged{
			{Pair: newPair(a, b), Relation: RelContradicts, Weight: 0.8},
		}, 0)
		if len(edges) != 1 {
			t.Fatalf("%s: %d edges, want 1", name, len(edges))
		}
		if edges[0].Kind != core.EdgeContradicts {
			t.Errorf("%s: kind = %q, want contradicts", name, edges[0].Kind)
		}
	}
}

package verifier

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/core"
)

func contradiction(a, b *core.Claim) Judged {
	return Judged{Pair: newPair(a, b), Relation: RelContradicts, Weight: 0.9, DecidedBy: "model"}
}

// TestOnlyContradictionsEarnAFollowUp. A duplicate or a supporting pair is settled;
// spending a lead on it buys a third source for something two already agree on.
func TestOnlyContradictionsEarnAFollowUp(t *testing.T) {
	a := &core.Claim{ID: "c_a", Text: "The effect is large."}
	b := &core.Claim{ID: "c_b", Text: "The effect is absent."}

	for _, rel := range []Relation{RelDuplicate, RelNeither} {
		got := FollowUps([]Judged{{Pair: newPair(a, b), Relation: rel}},
			FollowUpOptions{SessionID: "s_1"})
		if len(got) != 0 {
			t.Errorf("%s produced %d follow-up(s)", rel, len(got))
		}
	}

	got := FollowUps([]Judged{contradiction(a, b)}, FollowUpOptions{SessionID: "s_1"})
	if len(got) != 1 {
		t.Fatalf("a contradiction produced %d follow-ups, want 1", len(got))
	}
	if got[0].Lead.Query == "" {
		t.Error("follow-up has no query")
	}
	if got[0].Lead.RootClaimID == nil {
		t.Error("follow-up carries no root claim, so its output starts a new chain")
	}
	if got[0].Lead.VerifyDepth != 1 {
		t.Errorf("VerifyDepth = %d, want 1", got[0].Lead.VerifyDepth)
	}
	if got[0].Because == "" {
		t.Error("follow-up does not say why it exists")
	}
}

// TestStalenessIsNotADisagreement. §11.2 already resolved a contradiction across a
// long publication gap as staleness. Researching it further buys a second answer to
// a question the dates settled.
func TestStalenessIsNotADisagreement(t *testing.T) {
	old := time.Date(2019, 1, 1, 0, 0, 0, 0, time.UTC)
	recent := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	a := &core.Claim{ID: "c_a", Text: "The old figure.", PublishedAt: &old}
	b := &core.Claim{ID: "c_b", Text: "The new figure.", PublishedAt: &recent}

	if got := FollowUps([]Judged{contradiction(a, b)}, FollowUpOptions{SessionID: "s_1"}); len(got) != 0 {
		t.Errorf("%d follow-ups for a contradiction the dates already settled", len(got))
	}

	// Contemporaneous, same claims: this one IS a disagreement.
	sameDay := recent
	b2 := &core.Claim{ID: "c_b", Text: "The new figure.", PublishedAt: &sameDay}
	a2 := &core.Claim{ID: "c_a", Text: "The old figure.", PublishedAt: &sameDay}
	if got := FollowUps([]Judged{contradiction(a2, b2)}, FollowUpOptions{SessionID: "s_1"}); len(got) != 1 {
		t.Errorf("%d follow-ups for a live disagreement, want 1", len(got))
	}
}

// TestDepthCapBinds is §11.4's chain cap, and the reason this arrived with the
// mechanism rather than before it.
//
// A cap on a counter nothing increments is inert. M1 shipped that shape once
// already: MaxLeads was checked and tested and never bound, because no caller set
// the counter it read. So the test walks the chain rather than asserting on one
// depth.
func TestDepthCapBinds(t *testing.T) {
	const maxDepth = 3

	for depth := 0; depth < 6; depth++ {
		// Two claims already at this depth, both on the same chain.
		a := &core.Claim{ID: "c_a", Text: "One reading.", RootClaimID: "c_root", VerifyDepth: depth}
		b := &core.Claim{ID: "c_b", Text: "The other reading.", RootClaimID: "c_root", VerifyDepth: depth}

		got := FollowUps([]Judged{contradiction(a, b)},
			FollowUpOptions{SessionID: "s_1", MaxDepth: maxDepth})

		wantAllowed := depth+1 < maxDepth
		if wantAllowed && len(got) != 1 {
			t.Errorf("depth %d: %d follow-ups, want 1", depth, len(got))
		}
		if !wantAllowed && len(got) != 0 {
			t.Errorf("depth %d: %d follow-ups past a cap of %d", depth, len(got), maxDepth)
		}
		if wantAllowed && got[0].Lead.VerifyDepth != depth+1 {
			t.Errorf("depth %d: follow-up sits at %d, want %d", depth, got[0].Lead.VerifyDepth, depth+1)
		}
	}
}

// TestTheDeeperChainWins. A contradiction between two already-deep claims must not
// reset the count to the shallower one's depth — that is the loop the cap exists to
// close, and taking either endpoint's depth naively reopens it.
func TestTheDeeperChainWins(t *testing.T) {
	shallow := &core.Claim{ID: "c_a", Text: "One reading.", RootClaimID: "c_r1", VerifyDepth: 0}
	deep := &core.Claim{ID: "c_b", Text: "The other reading.", RootClaimID: "c_r2", VerifyDepth: 2}

	got := FollowUps([]Judged{contradiction(shallow, deep)},
		FollowUpOptions{SessionID: "s_1", MaxDepth: 4})
	if len(got) != 1 {
		t.Fatalf("%d follow-ups, want 1", len(got))
	}
	if got[0].Lead.VerifyDepth != 3 {
		t.Errorf("VerifyDepth = %d, want 3 (deeper chain + 1)", got[0].Lead.VerifyDepth)
	}
	if got[0].Lead.RootClaimID == nil || *got[0].Lead.RootClaimID != "c_r2" {
		t.Errorf("joined chain %v, want the deeper one c_r2", got[0].Lead.RootClaimID)
	}

	// And at the cap, the deeper chain is what stops it — a naive read of the
	// shallow endpoint would let this through forever.
	if got := FollowUps([]Judged{contradiction(shallow, deep)},
		FollowUpOptions{SessionID: "s_1", MaxDepth: 3}); len(got) != 0 {
		t.Errorf("%d follow-ups: the shallower endpoint reset the depth count", len(got))
	}
}

// TestPerRootCapCountsTheSession, not one pass. A disagreement that survives three
// attempts to settle it is a real disagreement, and the honest answer is to report
// it as disputed rather than keep paying for the same search.
func TestPerRootCapCountsTheSession(t *testing.T) {
	a := &core.Claim{ID: "c_a", Text: "One reading.", RootClaimID: "c_root"}
	b := &core.Claim{ID: "c_b", Text: "The other reading.", RootClaimID: "c_root"}
	verdicts := []Judged{contradiction(a, b)}

	for existing, wantAllowed := range map[int]bool{0: true, 1: true, 2: true, 3: false, 9: false} {
		got := FollowUps(verdicts, FollowUpOptions{
			SessionID:       "s_1",
			MaxPerRoot:      3,
			MaxDepth:        10,
			ExistingPerRoot: map[string]int{"c_root": existing},
		})
		if wantAllowed && len(got) != 1 {
			t.Errorf("%d existing follow-ups: got %d, want 1", existing, len(got))
		}
		if !wantAllowed && len(got) != 0 {
			t.Errorf("%d existing follow-ups already past a cap of 3: got %d", existing, len(got))
		}
	}
}

// TestOneLeadPerRootPerPass. Two contradictions about the same claim would otherwise
// produce two nearly identical searches in one pass.
func TestOneLeadPerRootPerPass(t *testing.T) {
	root := &core.Claim{ID: "c_a", Text: "The disputed reading."}
	verdicts := []Judged{
		contradiction(root, &core.Claim{ID: "c_b", Text: "First objection."}),
		contradiction(root, &core.Claim{ID: "c_c", Text: "Second objection."}),
		contradiction(root, &core.Claim{ID: "c_d", Text: "Third objection."}),
	}
	got := FollowUps(verdicts, FollowUpOptions{SessionID: "s_1", MaxDepth: 10, MaxPerRoot: 10})
	if len(got) != 1 {
		t.Errorf("%d follow-ups for three objections to one claim, want 1", len(got))
	}
}

// TestPassTotalIsBounded. A graph full of disagreement must not queue more
// verification work in one go than the session can run.
func TestPassTotalIsBounded(t *testing.T) {
	var verdicts []Judged
	for i := 0; i < 20; i++ {
		verdicts = append(verdicts, contradiction(
			&core.Claim{ID: fmt.Sprintf("c_a%02d", i), Text: fmt.Sprintf("Reading %d.", i)},
			&core.Claim{ID: fmt.Sprintf("c_b%02d", i), Text: fmt.Sprintf("Counter-reading %d.", i)},
		))
	}
	for _, max := range []int{1, 2, 5} {
		got := FollowUps(verdicts, FollowUpOptions{
			SessionID: "s_1", MaxTotal: max, MaxDepth: 10, MaxPerRoot: 10,
		})
		if len(got) != max {
			t.Errorf("MaxTotal=%d produced %d follow-ups", max, len(got))
		}
	}
}

// TestFollowUpGenerationIsDeterministic. The leads go into the queue, so an unstable
// order changes what gets researched — and a cassette recorded on one ordering
// misses on another.
func TestFollowUpGenerationIsDeterministic(t *testing.T) {
	var verdicts []Judged
	for i := 0; i < 8; i++ {
		verdicts = append(verdicts, contradiction(
			&core.Claim{ID: fmt.Sprintf("c_a%02d", i), Text: fmt.Sprintf("Reading %d.", i)},
			&core.Claim{ID: fmt.Sprintf("c_b%02d", i), Text: fmt.Sprintf("Counter-reading %d.", i)},
		))
	}
	var first string
	for run := 0; run < 5; run++ {
		got := FollowUps(verdicts, FollowUpOptions{SessionID: "s_1", MaxTotal: 3, MaxDepth: 10})
		var qs []string
		for _, f := range got {
			qs = append(qs, f.Lead.Query)
		}
		joined := strings.Join(qs, "|")
		if run == 0 {
			first = joined
			continue
		}
		if joined != first {
			t.Fatalf("run %d queued different work:\n %s\n %s", run, joined, first)
		}
	}
	if first == "" {
		t.Fatal("nothing was queued, so determinism was never exercised")
	}
}

// TestTheQueryNamesBothSidesAndIsBounded. It is built mechanically rather than by
// asking a model: the claims already say what is in dispute, and generating the
// query from page-derived text inside a model context would put untrusted text in
// the position that decides what gets searched next.
func TestTheQueryNamesBothSidesAndIsBounded(t *testing.T) {
	a := &core.Claim{ID: "c_a", Text: "MambaByte outperforms subword Transformers on PG-19."}
	b := &core.Claim{ID: "c_b", Text: "MambaByte underperforms subword Transformers on PG-19."}
	got := FollowUps([]Judged{contradiction(a, b)}, FollowUpOptions{SessionID: "s_1"})
	if len(got) != 1 {
		t.Fatal("no follow-up")
	}
	q := got[0].Lead.Query
	for _, want := range []string{"outperforms", "underperforms"} {
		if !strings.Contains(q, want) {
			t.Errorf("query does not name both sides (%q missing): %s", want, q)
		}
	}

	// An enormous claim must not become an enormous search query.
	huge := &core.Claim{ID: "c_z", Text: strings.Repeat("filler ", 5000)}
	got = FollowUps([]Judged{contradiction(a, huge)}, FollowUpOptions{SessionID: "s_1"})
	if len(got) != 1 {
		t.Fatal("no follow-up for the oversized claim")
	}
	if n := len(got[0].Lead.Query); n > 320 {
		t.Errorf("query is %d chars; one oversized claim was not clamped", n)
	}
}

// TestFollowUpsOutrankOrdinaryLeads. A contradiction blocks an honest answer, while
// another sub-question merely adds to one.
func TestFollowUpsOutrankOrdinaryLeads(t *testing.T) {
	a := &core.Claim{ID: "c_a", Text: "One reading."}
	b := &core.Claim{ID: "c_b", Text: "The other reading."}
	got := FollowUps([]Judged{contradiction(a, b)}, FollowUpOptions{SessionID: "s_1"})
	if len(got) != 1 {
		t.Fatal("no follow-up")
	}
	if got[0].Lead.Priority <= 0 {
		t.Errorf("follow-up priority = %d, no higher than an ordinary lead", got[0].Lead.Priority)
	}
}

// TestContradictionsOverTheCapAreDeferred, which is step 0's finding turned
// round: the ones that do not fit come back as deferred work rather than
// vanishing.
//
// Two returns rather than one longer list, because the difference matters to
// the caller. The first is queued now and runs now. The second is owed, and
// the run must not be allowed to end while it is outstanding.
func TestContradictionsOverTheCapAreDeferred(t *testing.T) {
	var verdicts []Judged
	for i := 0; i < 10; i++ {
		a := &core.Claim{ID: fmt.Sprintf("c_a%02d", i), Text: fmt.Sprintf("Finding %d is large.", i)}
		b := &core.Claim{ID: fmt.Sprintf("c_b%02d", i), Text: fmt.Sprintf("Finding %d is absent.", i)}
		verdicts = append(verdicts, contradiction(a, b))
	}

	queued, deferred := FollowUpsWithDeferred(verdicts, FollowUpOptions{SessionID: "s_1", MaxTotal: 2})
	if len(queued) != 2 {
		t.Fatalf("queued %d, want the cap of 2", len(queued))
	}
	if len(deferred) != 8 {
		t.Fatalf("deferred %d, want the 8 that did not fit", len(deferred))
	}

	// Every contradiction is accounted for exactly once.
	seen := map[string]bool{}
	for _, f := range append(append([]FollowUp{}, queued...), deferred...) {
		id := *f.Lead.RootClaimID
		if seen[id] {
			t.Errorf("%s is both queued and deferred", id)
		}
		seen[id] = true
	}
	if len(seen) != 10 {
		t.Errorf("%d contradictions accounted for, want 10", len(seen))
	}

	// A rule that disqualifies a contradiction disqualifies it from BOTH. A
	// duplicate is settled; deferring it would be queueing it later rather than
	// not at all.
	a := &core.Claim{ID: "c_x", Text: "One phrasing."}
	b := &core.Claim{ID: "c_y", Text: "Another phrasing."}
	q2, d2 := FollowUpsWithDeferred([]Judged{{Pair: newPair(a, b), Relation: RelDuplicate}},
		FollowUpOptions{SessionID: "s_1", MaxTotal: 2})
	if len(q2) != 0 || len(d2) != 0 {
		t.Errorf("a duplicate produced %d queued and %d deferred", len(q2), len(d2))
	}
}

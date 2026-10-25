package verifier

// The confirm pass (§11): a second, agreeing judgement before an edge is written.
//
// Measured on 149 hand-labelled pairs: one judgement calls "contradicts" correctly
// 51% of the time, two agreeing judgements 70%. The tests below pin the mechanism,
// not the measurement — what must hold is that disagreement withholds the edge and
// that an unaffordable second opinion never loses the first.

import (
	"context"
	"testing"

	"github.com/lajosdeme/mole/internal/core"
)

func pairOf(a, b string) Pair {
	return Pair{A: &core.Claim{ID: a, Text: a + " text"}, B: &core.Claim{ID: b, Text: b + " text"}}
}

// TestASecondLookThatDisagreesWithholdsTheEdge.
func TestASecondLookThatDisagreesWithholdsTheEdge(t *testing.T) {
	first := []Judged{
		{Pair: pairOf("c_a", "c_b"), Relation: RelContradicts, Weight: 0.9},
		{Pair: pairOf("c_c", "c_d"), Relation: RelDuplicate, Weight: 0.9},
		{Pair: pairOf("c_e", "c_f"), Relation: RelNeither, Weight: 0.9},
	}
	// The second look agrees on the duplicate and not on the contradiction.
	v := &Verifier{Confirm: ConfirmBoth}
	second := map[string]Relation{
		first[0].Pair.Key(): RelNeither,
		first[1].Pair.Key(): RelDuplicate,
	}
	got := v.applyConfirmations(first, second, &Result{})

	if got[0].Relation != RelNeither {
		t.Errorf("the unconfirmed contradiction survived as %q", got[0].Relation)
	}
	if got[0].Rationale == "" {
		t.Error("nothing records why the edge was withheld")
	}
	if got[1].Relation != RelDuplicate {
		t.Errorf("a confirmed verdict was changed to %q", got[1].Relation)
	}
	if got[2].Relation != RelNeither {
		t.Errorf("a neither verdict was touched: %q", got[2].Relation)
	}
}

// TestAnUnaffordableSecondOpinionKeepsTheFirstVerdict.
//
// Degrading to "no edges at all" because the allowance ran out mid-confirmation
// would be worse than the single-judgement graph this replaces.
func TestAnUnaffordableSecondOpinionKeepsTheFirstVerdict(t *testing.T) {
	first := []Judged{{Pair: pairOf("c_a", "c_b"), Relation: RelContradicts, Weight: 0.9}}
	v := &Verifier{Confirm: ConfirmBoth}

	res := &Result{}
	got := v.applyConfirmations(first, map[string]Relation{}, res)
	if got[0].Relation != RelContradicts {
		t.Errorf("relation = %q; a pair that was never re-judged lost its verdict",
			got[0].Relation)
	}
	if res.PairsUnconfirmed != 0 {
		t.Errorf("unconfirmed = %d; not asking is not the same as disagreeing",
			res.PairsUnconfirmed)
	}
}

// TestConfirmationIsCountedBothWays, so a run can report what the second pass cost
// it rather than silently shrinking the graph.
func TestConfirmationIsCountedBothWays(t *testing.T) {
	first := []Judged{
		{Pair: pairOf("c_a", "c_b"), Relation: RelContradicts},
		{Pair: pairOf("c_c", "c_d"), Relation: RelContradicts},
	}
	v := &Verifier{Confirm: ConfirmBoth}
	res := &Result{}
	v.applyConfirmations(first, map[string]Relation{
		first[0].Pair.Key(): RelContradicts,
		first[1].Pair.Key(): RelNeither,
	}, res)

	if res.PairsConfirmed != 1 || res.PairsUnconfirmed != 1 {
		t.Errorf("confirmed=%d unconfirmed=%d, want 1 and 1",
			res.PairsConfirmed, res.PairsUnconfirmed)
	}
}

// TestConfirmationOffLeavesEveryVerdictAlone.
func TestConfirmationOffLeavesEveryVerdictAlone(t *testing.T) {
	first := []Judged{{Pair: pairOf("c_a", "c_b"), Relation: RelContradicts}}
	v := &Verifier{Confirm: ConfirmNone}
	got, _ := v.confirm(context.Background(), "s1", [][]Judged{first}, &Result{})
	if got[0].Relation != RelContradicts {
		t.Errorf("relation = %q with confirmation off", got[0].Relation)
	}
}

// TestOnlyEdgeBuildingVerdictsAreReJudged. Re-confirming the "neither" verdicts —
// 86% of pairs — would multiply the cost of verification for nothing, since they
// build no edge either way.
func TestOnlyEdgeBuildingVerdictsAreReJudged(t *testing.T) {
	judged := []Judged{
		{Pair: pairOf("c_a", "c_b"), Relation: RelNeither},
		{Pair: pairOf("c_c", "c_d"), Relation: RelContradicts},
		{Pair: pairOf("c_e", "c_f"), Relation: RelNeither},
		{Pair: pairOf("c_g", "c_h"), Relation: RelDuplicate},
	}
	if got := positivesOf(judged, ConfirmBoth); len(got) != 2 {
		t.Errorf("%d pair(s) queued for a second look, want 2", len(got))
	}
}

// TestDuplicatesOnlyKeepsAContradictionASecondLookRejected: the combination one bool
// could not express, and the one a report a person reads needs.
func TestDuplicatesOnlyKeepsAContradictionASecondLookRejected(t *testing.T) {
	first := []Judged{
		{Pair: pairOf("c_a", "c_b"), Relation: RelContradicts, Weight: 0.9},
		{Pair: pairOf("c_c", "c_d"), Relation: RelDuplicate, Weight: 0.9},
	}
	v := &Verifier{Confirm: ConfirmDuplicates}

	// Only duplicates are asked, so the contradiction takes the never-re-judged path.
	second := map[string]Relation{first[1].Pair.Key(): RelNeither}
	got := v.applyConfirmations(first, second, &Result{})

	if got[0].Relation != RelContradicts {
		t.Errorf("contradiction = %q; it was never re-judged and must keep its verdict",
			got[0].Relation)
	}
	if got[1].Relation != RelNeither {
		t.Errorf("duplicate = %q; a rejected duplicate must still lose its edge",
			got[1].Relation)
	}
}

// TestDuplicatesOnlyAsksAboutDuplicatesAlone. Duplicates were 11 of 116 positives,
// so this mode's second pass also costs about a tenth as much.
func TestDuplicatesOnlyAsksAboutDuplicatesAlone(t *testing.T) {
	judged := []Judged{
		{Pair: pairOf("c_a", "c_b"), Relation: RelContradicts},
		{Pair: pairOf("c_c", "c_d"), Relation: RelDuplicate},
		{Pair: pairOf("c_e", "c_f"), Relation: RelNeither},
	}
	got := positivesOf(judged, ConfirmDuplicates)
	if len(got) != 1 || got[0].Key() != judged[1].Pair.Key() {
		t.Fatalf("%d pair(s) queued, want only the duplicate", len(got))
	}
}

// TestConfirmNoneAsksNothing asserts what is queued, not the output: "off" and
// "asked and told nothing" agree on the graph and cost different money.
func TestConfirmNoneAsksNothing(t *testing.T) {
	judged := []Judged{
		{Pair: pairOf("c_a", "c_b"), Relation: RelContradicts},
		{Pair: pairOf("c_c", "c_d"), Relation: RelDuplicate},
	}
	if got := positivesOf(judged, ConfirmNone); len(got) != 0 {
		t.Errorf("%d pair(s) queued with confirmation off, want 0", len(got))
	}
}

// TestNeitherIsNeverReJudgedInAnyMode: 86% of pairs, no edge either way.
func TestNeitherIsNeverReJudgedInAnyMode(t *testing.T) {
	judged := []Judged{{Pair: pairOf("c_a", "c_b"), Relation: RelNeither}}
	for _, mode := range []ConfirmMode{ConfirmBoth, ConfirmDuplicates, ConfirmNone, ""} {
		if got := positivesOf(judged, mode); len(got) != 0 {
			t.Errorf("mode %q queued a neither verdict", mode)
		}
	}
}

// TestTheZeroValueConfirmsNothing. A mode whose zero value confirmed would have
// switched the second pass on for every caller that never asked for it — and paid
// for it. session.Runner opts in; see its confirmMode.
func TestTheZeroValueConfirmsNothing(t *testing.T) {
	judged := []Judged{
		{Pair: pairOf("c_a", "c_b"), Relation: RelContradicts},
		{Pair: pairOf("c_c", "c_d"), Relation: RelDuplicate},
	}
	if got := positivesOf(judged, ""); len(got) != 0 {
		t.Errorf("%d pair(s) queued under the zero value, want 0", len(got))
	}
}

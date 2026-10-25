package eval_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/lajosdeme/mole/internal/eval"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/verifier"
)

// seqJudge answers each adjudication call with the next of a fixed sequence of
// relations, one verdict per pair in the batch.
type seqJudge struct {
	relations [][]string
	calls     int
}

func (f *seqJudge) Complete(_ context.Context, req llm.Request) (*llm.Response, error) {
	var rels []string
	if f.calls < len(f.relations) {
		rels = f.relations[f.calls]
	}
	f.calls++

	verdicts := ""
	for i, r := range rels {
		if i > 0 {
			verdicts += ","
		}
		verdicts += fmt.Sprintf(`{"pair":%d,"relation":%q,"confidence":0.9,"why":"seq"}`, i+1, r)
	}
	return &llm.Response{
		Text:  fmt.Sprintf(`{"verdicts":[%s]}`, verdicts),
		Model: "seq-fake",
		Usage: llm.Usage{InputTokens: 1, OutputTokens: 1},
	}, nil
}
func (f *seqJudge) Name() string             { return "seq-fake" }
func (f *seqJudge) ModelFor(llm.Tier) string { return "seq-fake" }

func judgedSet(verdicts ...string) *eval.PairSet {
	ps := &eval.PairSet{}
	for i, v := range verdicts {
		ps.Pairs = append(ps.Pairs, eval.LabelledPair{
			Pair: fmt.Sprintf("p%d", i), Model: v, Judged: true,
		})
	}
	return ps
}

// TestAgreementSeparatesInertDisagreementsFromRealOnes.
//
// The measurement that prompted this: claude-haiku-4-5 judging 37 pairs twice scored
// 84% raw, and the six disagreements read as a prompt problem. Five were
// supports/refines/unrelated shuffles, which the graph consumes identically —
// nothing derives anything from those edges. One was contradicts-vs-supports, which
// moves a confidence score and queues a research lead. Acting on 84% would have
// meant rewriting a prompt to chase six problems when there was one.
func TestAgreementSeparatesInertDisagreementsFromRealOnes(t *testing.T) {
	a := judgedSet("supports", "supports", "unrelated", "contradicts", "duplicate_of", "contradicts")
	b := judgedSet("supports", "refines", "supports", "supports", "unrelated", "contradicts")
	//              same        inert      inert       REAL         REAL          same

	ag := eval.CompareVerdicts(a, b)
	if ag.Compared != 6 {
		t.Fatalf("compared %d, want 6", ag.Compared)
	}
	if ag.Same != 2 {
		t.Errorf("Same = %d, want 2 (only the byte-identical verdicts)", ag.Same)
	}
	// The two inert shuffles join the two identical pairs; the contradicts and
	// duplicate_of flips do not.
	if ag.SameEffect != 4 {
		t.Errorf("SameEffect = %d, want 4", ag.SameEffect)
	}
	if got := ag.EffectRate(); got < 0.66 || got > 0.67 {
		t.Errorf("EffectRate = %.3f, want ~0.667", got)
	}

	// duplicate_of is NOT inert: it merges a cluster and so changes the publisher
	// count corroboration is computed from. Folding it in with supports would make
	// this measurement wrong in the direction that looks reassuring.
	if verifier.RelDuplicate.EffectOf() == verifier.RelSupports.EffectOf() {
		t.Error("duplicate_of is being treated as inert; it drives clustering")
	}

	// The reverse mistake: a judge agreeing with itself exactly must never score
	// below its raw rate.
	same := eval.CompareVerdicts(a, a)
	if same.Same != 6 || same.SameEffect != 6 {
		t.Errorf("a set compared with itself: Same=%d SameEffect=%d, want 6 and 6",
			same.Same, same.SameEffect)
	}
}

// TestEveryRelationHasADeliberateEffect.
//
// EffectOf returns inert for anything it does not name, so a relation added later
// would quietly stop counting against agreement — the exact failure this measurement
// exists to catch. Pinning all five makes that a test failure instead.
func TestEveryRelationHasADeliberateEffect(t *testing.T) {
	want := map[verifier.Relation]verifier.Effect{
		verifier.RelContradicts: verifier.EffectContradiction,
		verifier.RelDuplicate:   verifier.EffectDuplicate,
		verifier.RelNeither:     verifier.EffectInert,
		// Retired, and still mapped: labels and claim_edges rows written before the
		// taxonomy shrank carry these, and scoring an old labelled set against a new
		// judge run works only because both sides land on the same effect.
		verifier.RelSupports:  verifier.EffectInert,
		verifier.RelRefines:   verifier.EffectInert,
		verifier.RelUnrelated: verifier.EffectInert,
	}
	for rel, eff := range want {
		if got := rel.EffectOf(); got != eff {
			t.Errorf("%s has effect %q, want %q", rel, got, eff)
		}
	}
	if len(want) != 6 {
		t.Fatal("a relation was added or removed without updating this table")
	}
}

// TestScoreSeparatesInertErrorsFromRealOnes is TestAgreementSeparates... for the
// labelled path.
//
// Measured on claude-haiku-4-5 against 37 blind labels: 76% raw, 97% by effect.
// Eight of nine errors were calling an unrelated pair "supports" or "refines" —
// an edge nothing reads. The ninth was a false contradiction, which spends a
// confidence penalty and a research lead. Those are not the same mistake and a
// single accuracy figure cannot tell them apart.
// Rewritten when the scorer began normalizing the retired vocabulary. Rows b and c
// used to be "inert errors" — supports/refines/unrelated were three names for one
// effect, which is why the taxonomy was shrunk to three relations in the first
// place. Now they normalize onto "neither" and are simply CORRECT, so the fixture
// states the live vocabulary and keeps one genuinely inert case: a stored edge
// carrying a retired kind, which claim_edges rows still do.
func TestScoreSeparatesInertErrorsFromRealOnes(t *testing.T) {
	ps := &eval.PairSet{Pairs: []eval.LabelledPair{
		{Pair: "a", Model: "neither", Label: "neither", Judged: true},      // correct
		{Pair: "b", Model: "supports", Label: "neither", Judged: true},     // correct: a retired name
		{Pair: "c", Model: "unrelated", Label: "neither", Judged: true},    // correct: a retired name
		{Pair: "d", Model: "contradicts", Label: "neither", Judged: true},  // REAL error
		{Pair: "e", Model: "neither", Label: "duplicate_of", Judged: true}, // REAL error
	}}

	s := eval.ScorePairs(ps)
	if s.Labelled != 5 {
		t.Fatalf("Labelled = %d, want 5", s.Labelled)
	}
	if s.Correct != 3 {
		t.Errorf("Correct = %d, want 3 — a retired name is the same verdict", s.Correct)
	}
	if s.CorrectEffect != 3 {
		t.Errorf("CorrectEffect = %d, want 3 — every remaining error changes the graph, "+
			"which is what shrinking the taxonomy to one relation per effect bought",
			s.CorrectEffect)
	}
	if got := s.EffectAccuracy(); got < 0.59 || got > 0.61 {
		t.Errorf("EffectAccuracy = %.3f, want 0.6", got)
	}

	// A missed duplicate_of is a real error, not an inert one: it leaves two claims
	// in separate clusters and inflates the publisher count corroboration is built
	// from. Scoring it as inert would hide a confidence inflation bug.
	only := &eval.PairSet{Pairs: []eval.LabelledPair{
		{Pair: "e", Model: "unrelated", Label: "duplicate_of", Judged: true},
	}}
	if s := eval.ScorePairs(only); s.CorrectEffect != 0 {
		t.Errorf("a missed duplicate_of scored as effect-correct (CorrectEffect=%d)", s.CorrectEffect)
	}

	// Unlabelled and unjudged pairs must stay excluded from both counts, or the
	// effect figure becomes the softer number it must never be.
	padded := &eval.PairSet{Pairs: append([]eval.LabelledPair{
		{Pair: "x", Model: "supports", Label: "", Judged: true},   // unlabelled
		{Pair: "y", Model: "", Label: "unrelated", Judged: false}, // never asked
	}, ps.Pairs...)}
	p := eval.ScorePairs(padded)
	if p.Labelled != 5 || p.CorrectEffect != 3 {
		t.Errorf("padding changed the score: Labelled=%d CorrectEffect=%d, want 5 and 3",
			p.Labelled, p.CorrectEffect)
	}
}

// TestTheRetiredVocabularyIsNotScoredAsAnError.
//
// `pairs dump --all` writes "unrelated" for every pair the graph holds no edge
// for; a labeller writes "neither", which is one of the three words the prompt
// offers. Same verdict, two names — and the scorer compared raw strings, so on the
// first real labelled set it charged the judge with 33 errors out of 149 for a
// rename. Relation.Normalize exists precisely so nothing downstream has to know
// the old vocabulary existed; this was the one place downstream that skipped it.
func TestTheRetiredVocabularyIsNotScoredAsAnError(t *testing.T) {
	set := &eval.PairSet{Pairs: []eval.LabelledPair{
		{Pair: "a|b", Model: "unrelated", Label: "neither", Judged: true},
		{Pair: "c|d", Model: "supports", Label: "neither", Judged: true},
		{Pair: "e|f", Model: "refines", Label: "neither", Judged: true},
		{Pair: "g|h", Model: "neither", Label: "neither", Judged: true},
	}}
	s := eval.ScorePairs(set)
	if s.Correct != 4 {
		t.Errorf("correct = %d of 4; a retired name is being scored as a wrong verdict",
			s.Correct)
	}
	if got := s.Accuracy(); got != 1 {
		t.Errorf("accuracy = %v, want 1", got)
	}
}

// TestARealDisagreementIsStillAnError. The normalisation must fold names, not
// verdicts: contradicts and neither are different answers whatever they are called.
func TestARealDisagreementIsStillAnError(t *testing.T) {
	set := &eval.PairSet{Pairs: []eval.LabelledPair{
		{Pair: "a|b", Model: "contradicts", Label: "neither", Judged: true},
		{Pair: "c|d", Model: "unrelated", Label: "contradicts", Judged: true},
		{Pair: "e|f", Model: "duplicate_of", Label: "neither", Judged: true},
	}}
	s := eval.ScorePairs(set)
	if s.Correct != 0 {
		t.Errorf("correct = %d of 0; a real disagreement was folded away", s.Correct)
	}
}

// TestConfirmKeepsOnlyAgreeingPositives: the first look says "contradicts",
// the second says "neither", and the output withholds the verdict.
func TestConfirmKeepsOnlyAgreeingPositives(t *testing.T) {
	ps := &eval.PairSet{Pairs: []eval.LabelledPair{
		{Pair: "a|b", A: "X is true.", B: "X is false.", Label: "contradicts"},
	}}
	fake := &seqJudge{relations: [][]string{{"contradicts"}, {"neither"}}}

	out, err := eval.JudgePairFile(context.Background(), fake, ps, eval.FileJudgeOptions{
		Model: "fake", Confirm: true,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if fake.calls != 2 {
		t.Fatalf("model called %d time(s), want 2 (confirm judges twice)", fake.calls)
	}
	if got := out.Pairs[0].Model; got != "neither" {
		t.Errorf("Model = %q, want %q (the second look did not agree)", got, "neither")
	}
	if got := out.Pairs[0].Label; got != "contradicts" {
		t.Errorf("Label = %q, want the input label carried through unchanged", got)
	}
}

// TestRetiredLabelsNormalizeInFileMode. A model answering "supports" out of
// habit normalizes to "neither" the same way the production path does.
func TestRetiredLabelsNormalizeInFileMode(t *testing.T) {
	ps := &eval.PairSet{Pairs: []eval.LabelledPair{
		{Pair: "a|b", A: "claim a", B: "claim b", Label: "neither"},
	}}
	fake := &seqJudge{relations: [][]string{{"supports"}}}

	out, err := eval.JudgePairFile(context.Background(), fake, ps, eval.FileJudgeOptions{
		Model: "fake",
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := out.Pairs[0].Model; got != "neither" {
		t.Errorf("Model = %q, want the retired name normalized to %q", got, "neither")
	}
	s := eval.ScorePairs(out)
	if s.Correct != 1 {
		t.Errorf("Correct = %d, want 1: scoring must not penalise a renamed verdict", s.Correct)
	}
}

package output

import (
	"testing"

	"github.com/lajosdeme/mole/internal/core"
)

// relClaim builds a claim with a unique ID, for tests that call Findings
// directly rather than going through a store (report_test.go's claim() helper
// returns a value with no ID, which Findings cannot cluster or rank by).
func relClaim(text, source string, confidence float64) *core.Claim {
	return &core.Claim{
		ID:         core.NewClaimID(),
		Text:       text,
		Source:     source,
		Quote:      "a quote long enough to be real evidence",
		Confidence: confidence,
	}
}

// TestRelevantFindingOutranksEquallyConfidentOffTopicOne is the measured bug: a
// microplastics run's 95-finding cap admitted OECD workshop dates while "4.1
// micrograms per week, one millionth of the 5 gram claim" was cut. Same
// confidence; only one of them bears on the question.
func TestRelevantFindingOutranksEquallyConfidentOffTopicOne(t *testing.T) {
	question := "How much microplastic does the average person ingest per week?"
	offTopic := relClaim("The OECD workshop was held on 12-14 November 2025.", "https://oecd.example/a", 0.45)
	onTopic := relClaim("Adults ingest about 4.1 micrograms of microplastic per week.", "https://science.example/b", 0.45)

	findings := Findings(question, []*core.Claim{offTopic, onTopic}, nil, 0)
	if len(findings) != 2 {
		t.Fatalf("%d findings, want 2", len(findings))
	}
	if findings[0].Claim.ID != onTopic.ID {
		t.Fatalf("first finding is %.60q, want the microplastic finding", findings[0].Claim.Text)
	}
}

// TestStrongEvidenceStillBeatsWeakRelevantEvidence pins that relevance tempers
// confidence rather than overriding it: 0.95*(0.5+0.5*0) = 0.475 still beats
// 0.30*(0.5+0.5*1) = 0.30.
func TestStrongEvidenceStillBeatsWeakRelevantEvidence(t *testing.T) {
	question := "How much microplastic does the average person ingest per week?"
	offTopicStrong := relClaim("The OECD workshop was held on 12-14 November 2025.", "https://oecd.example/a", 0.95)
	onTopicWeak := relClaim("Adults ingest about 4.1 micrograms of microplastic per week.", "https://science.example/b", 0.30)

	findings := Findings(question, []*core.Claim{offTopicStrong, onTopicWeak}, nil, 0)
	if len(findings) != 2 {
		t.Fatalf("%d findings, want 2", len(findings))
	}
	if findings[0].Claim.ID != offTopicStrong.ID {
		t.Fatalf("first finding is %.60q, want the strong off-topic finding", findings[0].Claim.Text)
	}
}

// TestCapAdmitsTheRelevantFinding reproduces the measured failure directly: at a
// tied confidence, a cap of 2 must keep the finding relevant to the question
// over an off-topic one the old key (confidence, publishers, breadth, order)
// would have kept instead.
func TestCapAdmitsTheRelevantFinding(t *testing.T) {
	question := "How much microplastic does the average person ingest per week?"
	off1 := relClaim("The OECD workshop was held on 12-14 November 2025.", "https://oecd.example/a", 0.45)
	off2 := relClaim("The workshop agenda included three panel sessions.", "https://oecd.example/b", 0.45)
	onTopic := relClaim("Adults ingest about 4.1 micrograms of microplastic per week.", "https://science.example/c", 0.45)

	findings := Findings(question, []*core.Claim{off1, off2, onTopic}, nil, 2)
	if len(findings) != 2 {
		t.Fatalf("%d findings, want 2", len(findings))
	}
	for _, f := range findings {
		if f.Claim.ID == onTopic.ID {
			return
		}
	}
	t.Errorf("the relevant finding was cut by the cap: %+v", findings)
}

// TestNoOverlapLeavesOrderUnchanged: a question sharing no content word with any
// claim gives everyone relevance 0, so Findings must order exactly as the old
// key (confidence, publishers, breadth, order) did.
func TestNoOverlapLeavesOrderUnchanged(t *testing.T) {
	a := relClaim("Adults ingest about 4.1 micrograms of microplastic per week.", "https://a.example/x", 0.30)
	b := relClaim("The OECD workshop was held on 12-14 November 2025.", "https://b.example/y", 0.80)
	c := relClaim("A third, unrelated claim about kernels.", "https://c.example/z", 0.80)

	got := Findings("¿Cuánto?", []*core.Claim{a, b, c}, nil, 0)
	if len(got) != 3 {
		t.Fatalf("%d findings, want 3", len(got))
	}
	// Confidence desc; b and c tie at 0.80 so input order (b before c) decides.
	want := []string{b.ID, c.ID, a.ID}
	for i, id := range want {
		if got[i].Claim.ID != id {
			t.Errorf("position %d: got %.40q, want claim %s", i, got[i].Claim.Text, id)
		}
	}
}

package output_test

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/output"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/verifier"
)

func at(t time.Time) *time.Time { return &t }

// realDuplicates are the two claims a live run produced from ONE arXiv page. The
// report rendered them as two separate bullets, both cited [1] — the same finding
// twice, reading as two independent facts.
var realDuplicates = [2]string{
	"MambaByte compares favorably to various subword baselines in terms of performance while handling significantly longer sequences.",
	"Compared to existing subword models, MambaByte is competitive and even outperforms them in some cases.",
}

// TestDuplicatesRenderAsOneFindingWithEverySource is §11.2's rule.
func TestDuplicatesRenderAsOneFindingWithEverySource(t *testing.T) {
	claims := []core.Claim{
		claim(realDuplicates[0], "https://arxiv.org/abs/1", "a quote long enough to be real evidence one"),
		claim(realDuplicates[1], "https://nature.com/x", "a quote long enough to be real evidence two"),
		claim("An unrelated finding about kernels.", "https://other.example/k", "a quote long enough to be real evidence three"),
	}
	st, sid := newStore(t, claims)
	stored := storedClaims(t, st, sid)
	link(t, st, sid, stored[realDuplicates[0]].ID, stored[realDuplicates[1]].ID, core.EdgeDuplicateOf)

	f := &fakeLLM{reply: func(string) string { return "Summary [1]." }}
	rep, err := (&output.Generator{LLM: f}).Generate(context.Background(), st, sid)
	if err != nil {
		t.Fatal(err)
	}

	if len(rep.Findings) != 2 {
		for _, fd := range rep.Findings {
			t.Logf("  %.60q sources=%v", fd.Claim.Text, fd.Sources)
		}
		t.Fatalf("%d findings from 3 claims with one duplicate pair, want 2", len(rep.Findings))
	}

	// The collapsed finding carries both sources, and both citation markers reach
	// the model — that is how corroboration becomes visible instead of appearing as
	// a repeated sentence.
	var collapsed *output.Finding
	for i := range rep.Findings {
		if len(rep.Findings[i].Sources) == 2 {
			collapsed = &rep.Findings[i]
		}
	}
	if collapsed == nil {
		t.Fatal("no finding carries both sources")
	}
	if collapsed.Publishers != 2 {
		t.Errorf("Publishers = %d, want 2 (arxiv.org and nature.com)", collapsed.Publishers)
	}
	if !strings.Contains(f.prompt, "[1][2]") && !strings.Contains(f.prompt, "[2][1]") {
		t.Errorf("the collapsed finding's two citations were not offered together:\n%s", f.prompt)
	}
	// And only ONE of the two duplicate texts reaches the model.
	both := strings.Contains(f.prompt, realDuplicates[0][:40]) &&
		strings.Contains(f.prompt, realDuplicates[1][:40])
	if both {
		t.Errorf("both duplicate wordings reached the model:\n%s", f.prompt)
	}
}

// TestCollapseHappensBeforeTheCap. A cap on raw claims spends its whole budget on one
// page saying one thing eight ways; a cap on findings counts distinct assertions.
func TestCollapseHappensBeforeTheCap(t *testing.T) {
	var claims []core.Claim
	// Eight restatements of one finding, plus two genuinely different ones.
	for i := 0; i < 8; i++ {
		claims = append(claims, claim(
			fmt.Sprintf("Byte-level modelling removes subword tokenization, phrasing %d.", i),
			fmt.Sprintf("https://s%d.example/p", i),
			fmt.Sprintf("a quote long enough to be real evidence %d", i)))
	}
	claims = append(claims,
		claim("An entirely separate finding about CUDA kernels.", "https://k.example/1",
			"a quote long enough to be real evidence k"),
		claim("A third finding about inference latency.", "https://l.example/1",
			"a quote long enough to be real evidence l"))

	st, sid := newStore(t, claims)
	stored := storedClaims(t, st, sid)
	// Link the eight restatements into one cluster.
	first := stored["Byte-level modelling removes subword tokenization, phrasing 0."].ID
	for i := 1; i < 8; i++ {
		other := stored[fmt.Sprintf("Byte-level modelling removes subword tokenization, phrasing %d.", i)].ID
		link(t, st, sid, first, other, core.EdgeDuplicateOf)
	}

	f := &fakeLLM{reply: func(string) string { return "Summary [1]." }}
	rep, err := (&output.Generator{LLM: f, MaxClaims: 3}).Generate(context.Background(), st, sid)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Findings) != 3 {
		t.Fatalf("%d findings under a cap of 3", len(rep.Findings))
	}
	// All three distinct assertions survive, because the cap counted findings.
	for _, want := range []string{"CUDA kernels", "inference latency", "removes subword tokenization"} {
		if !strings.Contains(f.prompt, want) {
			t.Errorf("%q was dropped by the cap:\n%s", want, f.prompt)
		}
	}
}

// TestDisagreementsAreDisclosedNotResolved. §11.3 lowers both sides' confidence, but
// that is a number the prose never shows: a reader looking at two cited sentences has
// no way to know the pipeline found them incompatible.
func TestDisagreementsAreDisclosedNotResolved(t *testing.T) {
	claims := []core.Claim{
		claim("Subword tokenization improves accuracy.", "https://a.example/1",
			"a quote long enough to be real evidence one"),
		claim("Subword tokenization does not improve accuracy.", "https://b.example/1",
			"a quote long enough to be real evidence two"),
	}
	st, sid := newStore(t, claims)
	stored := storedClaims(t, st, sid)
	link(t, st, sid, stored["Subword tokenization improves accuracy."].ID,
		stored["Subword tokenization does not improve accuracy."].ID, core.EdgeContradicts)

	f := &fakeLLM{reply: func(string) string { return "The sources disagree [1][2]." }}
	rep, err := (&output.Generator{LLM: f}).Generate(context.Background(), st, sid)
	if err != nil {
		t.Fatal(err)
	}

	if rep.Disagreements != 1 {
		t.Errorf("Disagreements = %d, want 1", rep.Disagreements)
	}
	// The model must be TOLD, not merely instructed to notice. A rule to disclose
	// something it has to infer is a rule a small model reading forty bullets misses.
	// Asserted on the MATERIAL, not the whole prompt. The rules section explains what
	// a conflict annotation means and so contains overlapping words; an assertion over
	// the whole prompt would pass with the annotation removed.
	mat := material(f.prompt)
	if !strings.Contains(mat, "pairs conflict") {
		t.Errorf("the disagreement was not named in the material:\n%s", mat)
	}
	if !strings.Contains(mat, "cannot both be true") {
		t.Errorf("the material does not say the two conflict:\n%s", mat)
	}
	// And each finding is flagged inline.
	if !strings.Contains(mat, "disputed") {
		t.Errorf("neither side was flagged as disputed:\n%s", mat)
	}
}

// TestDisagreementsSurviveTheFallback. The prose is what usually discloses them, and
// the fallback is the path taken when there is no prose — so a report with no budget
// left must not silently drop the one thing a reader most needs.
func TestDisagreementsSurviveTheFallback(t *testing.T) {
	claims := []core.Claim{
		claim("The effect is large.", "https://a.example/1", "a quote long enough to be real evidence one"),
		claim("The effect is absent.", "https://b.example/1", "a quote long enough to be real evidence two"),
	}
	st, sid := newStore(t, claims)
	stored := storedClaims(t, st, sid)
	link(t, st, sid, stored["The effect is large."].ID, stored["The effect is absent."].ID,
		core.EdgeContradicts)

	// No LLM: the fallback path.
	rep, err := (&output.Generator{}).Generate(context.Background(), st, sid)
	if err != nil {
		t.Fatal(err)
	}
	body := rep.Markdown()
	if !strings.Contains(body, "disagree") {
		t.Errorf("the fallback body hides the disagreement:\n%s", body)
	}
	for _, want := range []string{"The effect is large", "The effect is absent", "CONTRADICTS"} {
		if !strings.Contains(body, want) {
			t.Errorf("fallback omits %q:\n%s", want, body)
		}
	}
}

// TestAFailedGroundingCheckIsFlaggedToTheModelAndTheReader. A claim whose own source
// does not support it on re-reading is §11.5's headline finding, and it is invisible
// in a confidence number.
func TestAFailedGroundingCheckIsFlaggedToTheModelAndTheReader(t *testing.T) {
	no := false
	c := claim("A claim its source does not support.", "https://a.example/1",
		"a quote long enough to be real evidence one")
	c.Grounded = &no
	st, sid := newStore(t, []core.Claim{c})

	f := &fakeLLM{reply: func(string) string { return "Summary [1]." }}
	if _, err := (&output.Generator{LLM: f}).Generate(context.Background(), st, sid); err != nil {
		t.Fatal(err)
	}
	// Inside the MATERIAL, not anywhere in the prompt. The rules section explains
	// what the flag means and therefore contains the same words — so asserting on
	// the whole prompt passed with the flag removed entirely.
	if !strings.Contains(material(f.prompt), "failed source re-read") {
		t.Errorf("the failed grounding check was not flagged on the claim's own line:\n%s",
			material(f.prompt))
	}
}

// material returns just the fenced claim block, so an assertion cannot be satisfied
// by the instructions that describe the thing being asserted.
func material(prompt string) string {
	i := strings.Index(prompt, "<claims-")
	if i < 0 {
		return ""
	}
	return prompt[i:]
}

// TestManyPagesFromOnePublisherIsNotCorroboration. §11.3's inflation risk, made
// visible: a finding backed by five URLs on one site must say so rather than looking
// like five publishers agreeing.
func TestManyPagesFromOnePublisherIsNotCorroboration(t *testing.T) {
	var claims []core.Claim
	for i := 0; i < 3; i++ {
		claims = append(claims, claim(
			fmt.Sprintf("One finding, phrasing %d.", i),
			fmt.Sprintf("https://oneblog.example/post%d", i),
			fmt.Sprintf("a quote long enough to be real evidence %d", i)))
	}
	st, sid := newStore(t, claims)
	stored := storedClaims(t, st, sid)
	first := stored["One finding, phrasing 0."].ID
	for i := 1; i < 3; i++ {
		link(t, st, sid, first, stored[fmt.Sprintf("One finding, phrasing %d.", i)].ID,
			core.EdgeDuplicateOf)
	}

	f := &fakeLLM{reply: func(string) string { return "Summary [1]." }}
	rep, err := (&output.Generator{LLM: f}).Generate(context.Background(), st, sid)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Findings) != 1 {
		t.Fatalf("%d findings, want 1", len(rep.Findings))
	}
	if rep.Findings[0].Publishers != 1 {
		t.Errorf("Publishers = %d for three pages of one site", rep.Findings[0].Publishers)
	}
	if !strings.Contains(f.prompt, "from one publisher") {
		t.Errorf("three pages of one site were not distinguished from corroboration:\n%s", f.prompt)
	}
}

// TestCitationNumbersFollowInsertionOrder is what the input-position tie-break buys.
//
// Claim IDs carry a millisecond timestamp and a random suffix, so a batch inserted in
// one transaction shares the prefix and orders arbitrarily. An ID sort is stable for a
// fixed database — which is why asserting only "five runs agree" passed with the ID
// tie-break restored — but the order it produces bears no relation to anything, so
// two sessions over the same sources number their citations differently and a
// cassette recorded against one misses the other.
//
// ListClaims returns created_at order. Following it makes [1] the first source found.
func TestCitationNumbersFollowInsertionOrder(t *testing.T) {
	var claims []core.Claim
	for i := 0; i < 6; i++ {
		claims = append(claims, claim(
			fmt.Sprintf("Finding number %d.", i),
			fmt.Sprintf("https://s%d.example/p", i),
			fmt.Sprintf("a quote long enough to be real evidence %d", i)))
	}
	st, sid := newStore(t, claims)

	var first string
	for run := 0; run < 5; run++ {
		f := &fakeLLM{reply: func(string) string { return "Summary [1]." }}
		rep, err := (&output.Generator{LLM: f}).Generate(context.Background(), st, sid)
		if err != nil {
			t.Fatal(err)
		}
		var order []string
		for _, c := range rep.Citations {
			order = append(order, fmt.Sprintf("%d=%s", c.N, c.Source))
		}
		joined := strings.Join(order, ",")
		if run == 0 {
			first = joined
			continue
		}
		if joined != first {
			t.Fatalf("run %d numbered citations differently:\n %s\n %s", run, joined, first)
		}
	}

	// And the order is the one claims were inserted in, not an arbitrary one.
	f := &fakeLLM{reply: func(string) string { return "Summary [1]." }}
	rep, err := (&output.Generator{LLM: f}).Generate(context.Background(), st, sid)
	if err != nil {
		t.Fatal(err)
	}
	for i, c := range rep.Citations {
		want := fmt.Sprintf("https://s%d.example/p", i)
		if c.Source != want {
			t.Errorf("citation [%d] is %s, want %s — numbering does not follow the "+
				"order the claims were found in", c.N, c.Source, want)
		}
	}
}

// TestSupersededFindingsAreFlagged (§11.2). A stale claim presented with confidence is
// the failure the supersedes rule exists to catch.
func TestSupersededFindingsAreFlagged(t *testing.T) {
	old := claim("The 2019 figure.", "https://a.example/1", "a quote long enough to be real evidence one")
	old.PublishedAt = at(time.Date(2019, 1, 1, 0, 0, 0, 0, time.UTC))
	recent := claim("The 2025 figure.", "https://b.example/1", "a quote long enough to be real evidence two")
	recent.PublishedAt = at(time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC))

	st, sid := newStore(t, []core.Claim{old, recent})
	stored := storedClaims(t, st, sid)
	// Newer supersedes older.
	link(t, st, sid, stored["The 2025 figure."].ID, stored["The 2019 figure."].ID, core.EdgeSupersedes)

	f := &fakeLLM{reply: func(string) string { return "Summary [1]." }}
	rep, err := (&output.Generator{LLM: f}).Generate(context.Background(), st, sid)
	if err != nil {
		t.Fatal(err)
	}
	var stale, fresh *output.Finding
	for i := range rep.Findings {
		if rep.Findings[i].Claim.Text == "The 2019 figure." {
			stale = &rep.Findings[i]
		} else {
			fresh = &rep.Findings[i]
		}
	}
	if stale == nil || fresh == nil {
		t.Fatalf("expected both findings, got %d", len(rep.Findings))
	}
	if !stale.Superseded {
		t.Error("the older finding is not flagged superseded")
	}
	if fresh.Superseded {
		t.Error("the newer finding was flagged superseded; the edge direction was lost")
	}
	// Material only: the rules use the word "outdated" too.
	if !strings.Contains(material(f.prompt), "outdated") {
		t.Errorf("staleness was not flagged to the model:\n%s", material(f.prompt))
	}
}

// storedClaims reads the session's claims back by text, so a test can reach the IDs
// the store assigned.
func storedClaims(t *testing.T, st store.Store, sid string) map[string]*core.Claim {
	t.Helper()
	out := map[string]*core.Claim{}
	if err := st.Read(context.Background(), func(ctx context.Context, q store.Queries) error {
		list, err := q.ListClaims(ctx, sid, 0)
		if err != nil {
			return err
		}
		for _, c := range list {
			out[c.Text] = c
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	return out
}

// link writes one graph edge.
func link(t *testing.T, st store.Store, sid, from, to string, kind core.EdgeKind) {
	t.Helper()
	if err := st.WithTx(context.Background(), func(ctx context.Context, tx store.Tx) error {
		return tx.InsertEdges(ctx, []core.ClaimEdge{{
			SessionID: sid, FromID: from, ToID: to, Kind: kind, Weight: 1,
		}})
	}); err != nil {
		t.Fatal(err)
	}
}

// TestDisagreementsSurviveReordering is the regression that found a real bug.
//
// Contradictions are discovered against the pre-sort order and the report is sorted by
// confidence, so the two orders differ. An earlier version recorded indices, sorted in
// place, and then translated them using the SORTED slice as the pre-sort reference —
// incoherent, and it passed whenever the sort happened not to reorder anything. When it
// did, a disagreement pointed at whatever finding had moved into the old position:
// worse than silence, because it attributes a dispute to an unrelated claim.
//
// Constructed so the sort definitely reorders: the two disputing claims have the
// LOWEST confidence, so they end up last, and three unrelated findings move ahead of
// them.
func TestDisagreementsSurviveReordering(t *testing.T) {
	mk := func(text, host string, conf float64) core.Claim {
		c := claim(text, "https://"+host+"/p", "a quote long enough to be real evidence "+host)
		c.Confidence = conf
		return c
	}
	claims := []core.Claim{
		// Inserted FIRST, lowest confidence: sorted last.
		mk("Subword tokenization improves accuracy.", "a.example", 0.10),
		mk("Subword tokenization does not improve accuracy.", "b.example", 0.10),
		// Shares no content word with the question, so relevance isn't why
		// this claim outranks the disputed pair — confidence is.
		mk("Kernel fusion improves GPU throughput.", "c.example", 0.90),
		mk("Inference latency drops with speculative decoding.", "d.example", 0.80),
		mk("The PG-19 corpus measures long-document perplexity.", "e.example", 0.70),
	}
	st, sid := newStore(t, claims)
	stored := storedClaims(t, st, sid)
	link(t, st, sid,
		stored["Subword tokenization improves accuracy."].ID,
		stored["Subword tokenization does not improve accuracy."].ID,
		core.EdgeContradicts)

	f := &fakeLLM{reply: func(string) string { return "Summary [1]." }}
	rep, err := (&output.Generator{LLM: f}).Generate(context.Background(), st, sid)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Findings) != 5 {
		t.Fatalf("%d findings, want 5", len(rep.Findings))
	}
	// The sort must actually have moved them, or this proves nothing.
	if rep.Findings[0].Claim.Text == "Subword tokenization improves accuracy." {
		t.Fatal("the sort did not reorder; the case under test was not constructed")
	}
	if rep.Disagreements != 1 {
		t.Errorf("Disagreements = %d, want 1", rep.Disagreements)
	}

	// Each side must point at the OTHER side, by content, not at whichever finding
	// took over its old slot.
	for i, fd := range rep.Findings {
		want := ""
		switch fd.Claim.Text {
		case "Subword tokenization improves accuracy.":
			want = "Subword tokenization does not improve accuracy."
		case "Subword tokenization does not improve accuracy.":
			want = "Subword tokenization improves accuracy."
		default:
			if len(fd.Contradicts) != 0 {
				t.Errorf("finding %d (%.40q) has a dispute it should not",
					i, fd.Claim.Text)
			}
			continue
		}
		if len(fd.Contradicts) != 1 {
			t.Errorf("%.40q has %d disputes, want 1", fd.Claim.Text, len(fd.Contradicts))
			continue
		}
		if got := rep.Findings[fd.Contradicts[0]].Claim.Text; got != want {
			t.Errorf("%.40q points at %.40q, want %.40q", fd.Claim.Text, got, want)
		}
	}
}

// TestTheCapNeverShowsOneSideOfADisagreement.
//
// The naive order — truncate, then drop references past the cap — selects for erasing
// exactly what §11.2 promises a reader always sees. A contradiction LOWERS confidence
// (§11.3), the list is sorted by confidence, so the weaker side of every disputed pair
// sorts to the bottom and is among the first cut. The survivor is then presented as
// unqualified and Report.Disagreements reads 0.
//
// No test covered the truncation branch at all, which is how it survived.
func TestTheCapNeverShowsOneSideOfADisagreement(t *testing.T) {
	mk := func(text, host string, conf float64) core.Claim {
		c := claim(text, "https://"+host+"/p", "a quote long enough to be real evidence "+host)
		c.Confidence = conf
		return c
	}
	// The disputed pair scores lowest, so the cap reaches it first; neither
	// other claim shares a content word with the question.
	claims := []core.Claim{
		mk("Subword tokenization improves accuracy.", "a.example", 0.10),
		mk("Subword tokenization does not improve accuracy.", "b.example", 0.10),
		mk("Kernel fusion improves GPU throughput.", "c.example", 0.90),
		mk("Inference latency drops with speculative decoding.", "d.example", 0.80),
	}
	st, sid := newStore(t, claims)
	stored := storedClaims(t, st, sid)
	link(t, st, sid,
		stored["Subword tokenization improves accuracy."].ID,
		stored["Subword tokenization does not improve accuracy."].ID,
		core.EdgeContradicts)

	for _, cap := range []int{2, 3, 4} {
		f := &fakeLLM{reply: func(string) string { return "Summary [1]." }}
		rep, err := (&output.Generator{LLM: f, MaxClaims: cap}).Generate(context.Background(), st, sid)
		if err != nil {
			t.Fatal(err)
		}
		if len(rep.Findings) > cap {
			t.Errorf("cap=%d: %d findings — the cap stopped being hard", cap, len(rep.Findings))
		}

		// Whichever side survives, the other must be there too.
		present := map[string]bool{}
		for _, fd := range rep.Findings {
			present[fd.Claim.Text] = true
		}
		a := present["Subword tokenization improves accuracy."]
		b := present["Subword tokenization does not improve accuracy."]
		if a != b {
			t.Errorf("cap=%d: one side of a disagreement was shown alone (a=%v b=%v)", cap, a, b)
		}
		if a && b {
			if rep.Disagreements != 1 {
				t.Errorf("cap=%d: both sides present but Disagreements=%d", cap, rep.Disagreements)
			}
			if !strings.Contains(material(f.prompt), "disputed") {
				t.Errorf("cap=%d: the surviving pair was not flagged disputed:\n%s",
					cap, material(f.prompt))
			}
		}
	}
}

// linkBy is link with the edge's created_by set.
func linkBy(t *testing.T, st store.Store, sid, from, to, by string) {
	t.Helper()
	if err := st.WithTx(context.Background(), func(ctx context.Context, tx store.Tx) error {
		return tx.InsertEdges(ctx, []core.ClaimEdge{{
			SessionID: sid, FromID: from, ToID: to, Kind: core.EdgeContradicts, Weight: 1, CreatedBy: by,
		}})
	}); err != nil {
		t.Fatal(err)
	}
}

// TestUncheckedContradictionsAreWithheldFromTheReport checks the report and ask drop unchecked contradictions only when asked.
func TestUncheckedContradictionsAreWithheldFromTheReport(t *testing.T) {
	for _, tc := range []struct {
		name     string
		by       string
		model    string
		toolkit  bool
		want     int
		degraded bool
	}{
		{"unchecked, withheld", "fake-model", "strict", false, 0, true},
		{"re-checked, kept", verifier.RecheckedBy + "strict", "strict", false, 1, false},
		{"unchecked, re-check off", "fake-model", "", false, 1, false},
		{"unchecked, toolkit session", "toolkit", "strict", true, 1, false},
	} {
		for _, ask := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s ask=%v", tc.name, ask), func(t *testing.T) {
				claims := []core.Claim{
					claim("The effect is large.", "https://a.example/1", "a quote long enough to be real evidence one"),
					claim("The effect is absent.", "https://b.example/1", "a quote long enough to be real evidence two"),
				}
				actors := []core.ActorType{core.ActorWeb}
				if tc.toolkit {
					actors = append(actors, core.ActorToolkit)
				}
				st, sid := newStoreWith(t, claims, actors...)
				stored := storedClaims(t, st, sid)
				linkBy(t, st, sid, stored["The effect is large."].ID, stored["The effect is absent."].ID, tc.by)

				gen := &output.Generator{RecheckModel: tc.model}
				var rep *output.Report
				var err error
				if ask {
					rep, err = gen.Answer(context.Background(), st, sid, "is the effect large")
				} else {
					rep, err = gen.Generate(context.Background(), st, sid)
				}
				if err != nil {
					t.Fatal(err)
				}
				if rep.Disagreements != tc.want {
					t.Errorf("Disagreements = %d, want %d", rep.Disagreements, tc.want)
				}
				if got := strings.Contains(rep.Degraded, "1 contradiction(s) withheld: not re-checked"); got != tc.degraded {
					t.Errorf("degraded = %q; want the withheld note %v", rep.Degraded, tc.degraded)
				}
			})
		}
	}
}

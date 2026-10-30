package verifier_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/budget"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/store/sqlite"
	"github.com/lajosdeme/mole/internal/verifier"
)

// recheckClaims is six claims, one contradicting pair per two.
func recheckClaims() []core.Claim {
	out := make([]core.Claim, 0, 6)
	for i := range 6 {
		out = append(out, core.Claim{
			Text:   fmt.Sprintf("Finding number %d about battery fires.", i),
			Source: fmt.Sprintf("https://r%d.example/p", i),
		})
	}
	return out
}

// seedContradictions writes a contradicts edge per pair plus one duplicate_of
// edge, and scores every claim over that graph.
func seedContradictions(t *testing.T, r *rig, claims []core.Claim, pairs [][2]int) {
	t.Helper()
	ctx := context.Background()
	var edges []core.ClaimEdge
	for _, p := range pairs {
		edges = append(edges, core.ClaimEdge{
			SessionID: r.sess.ID, FromID: claims[p[0]].ID, ToID: claims[p[1]].ID,
			Kind: core.EdgeContradicts, Weight: 1, CreatedBy: "seed",
		})
	}
	edges = append(edges, core.ClaimEdge{
		SessionID: r.sess.ID, FromID: claims[0].ID, ToID: claims[5].ID,
		Kind: core.EdgeDuplicateOf, Weight: 1, CreatedBy: "seed",
	})
	if err := r.db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.InsertEdges(ctx, edges)
	}); err != nil {
		t.Fatal(err)
	}
	rescoreAll(t, r)
}

// rescoreAll derives confidence over the stored graph and writes it.
func rescoreAll(t *testing.T, r *rig) []store.ClaimScore {
	t.Helper()
	ctx := context.Background()
	var list []*core.Claim
	if err := r.db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		list, err = q.ListClaims(ctx, r.sess.ID, 0)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	scores, _ := verifier.DeriveConfidence(list, r.edges(t))
	if err := r.db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.ScoreClaims(ctx, scores)
	}); err != nil {
		t.Fatal(err)
	}
	return scores
}

func countKind(edges []*core.ClaimEdge, kind core.EdgeKind) int {
	n := 0
	for _, e := range edges {
		if e.Kind == kind {
			n++
		}
	}
	return n
}

// judgeByText answers "neither" for a pair whose prompt mentions withhold, else "contradicts".
func judgeByText(withhold string) func(int, string) (string, error) {
	return func(call int, prompt string) (string, error) {
		if strings.Contains(prompt, withhold) {
			return judgeEveryPair(verifier.RelNeither)(call, prompt)
		}
		return judgeEveryPair(verifier.RelContradicts)(call, prompt)
	}
}

// TestRecheckWithholdsUnconfirmedContradictions checks unconfirmed edges are removed and confidence re-derived.
func TestRecheckWithholdsUnconfirmedContradictions(t *testing.T) {
	claims := recheckClaims()
	r := newRig(t, 2_000_000, claims, judgeByText("Finding number 2 "))
	r.v.RecheckModel = "strict"
	r.v.BatchSize = 1
	seedContradictions(t, r, claims, [][2]int{{0, 1}, {2, 3}, {4, 5}})
	before := r.claims(t)

	res, err := r.v.Recheck(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatalf("recheck: %v", err)
	}
	if res.Rechecked != 3 || res.Withheld != 1 {
		t.Errorf("rechecked %d, withheld %d; want 3 and 1", res.Rechecked, res.Withheld)
	}
	if res.Contradictions != 2 {
		t.Errorf("contradictions = %d, want 2", res.Contradictions)
	}
	edges := r.edges(t)
	if got := countKind(edges, core.EdgeContradicts); got != 2 {
		t.Errorf("%d contradicts edges remain, want 2", got)
	}
	if got := countKind(edges, core.EdgeDuplicateOf); got != 1 {
		t.Errorf("%d duplicate_of edges remain, want 1", got)
	}
	for _, e := range edges {
		if e.Kind == core.EdgeContradicts && (e.FromID == claims[2].ID || e.ToID == claims[2].ID) {
			t.Errorf("the withheld contradiction %s -> %s is still stored", e.FromID, e.ToID)
		}
	}

	after := r.claims(t)
	want := map[string]float64{}
	for _, s := range rescoreAll(t, r) {
		want[s.ClaimID] = s.Confidence
	}
	for _, i := range []int{2, 3} {
		text := claims[i].Text
		if after[text].Confidence == before[text].Confidence {
			t.Errorf("claim %d kept confidence %.4f after its contradiction was withheld", i, after[text].Confidence)
		}
		if after[text].Confidence != want[claims[i].ID] {
			t.Errorf("claim %d confidence %.4f, want %.4f", i, after[text].Confidence, want[claims[i].ID])
		}
	}
}

// TestRecheckUsesTheRecheckModel checks every re-check request carries RecheckModel.
func TestRecheckUsesTheRecheckModel(t *testing.T) {
	claims := recheckClaims()
	r := newRig(t, 2_000_000, claims, judgeEveryPair(verifier.RelContradicts))
	r.v.Model = "fast"
	r.v.RecheckModel = "strict"
	seedContradictions(t, r, claims, [][2]int{{0, 1}, {2, 3}})

	if _, err := r.v.Recheck(context.Background(), r.sess.ID); err != nil {
		t.Fatal(err)
	}
	models := r.llm.requestedModels()
	if len(models) == 0 {
		t.Fatal("no re-check call was made")
	}
	for i, m := range models {
		if m != "strict" {
			t.Errorf("call %d used %q, want the re-check model", i, m)
		}
	}
}

// TestNoRecheckModelMakesNoCall checks an empty RecheckModel leaves the graph alone.
func TestNoRecheckModelMakesNoCall(t *testing.T) {
	claims := recheckClaims()
	r := newRig(t, 2_000_000, claims, judgeEveryPair(verifier.RelNeither))
	seedContradictions(t, r, claims, [][2]int{{0, 1}, {2, 3}})

	res, err := r.v.Recheck(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if n := r.llm.count(); n != 0 {
		t.Errorf("%d calls with no re-check model", n)
	}
	if res.Withheld != 0 || countKind(r.edges(t), core.EdgeContradicts) != 2 {
		t.Errorf("withheld %d; the graph changed without a re-check model", res.Withheld)
	}
}

// TestRecheckWithNoContradictionsMakesNoCall checks other edge kinds are never re-checked.
func TestRecheckWithNoContradictionsMakesNoCall(t *testing.T) {
	claims := recheckClaims()
	r := newRig(t, 2_000_000, claims, judgeEveryPair(verifier.RelNeither))
	r.v.RecheckModel = "strict"
	seedContradictions(t, r, claims, nil)
	if err := r.db.WithTx(context.Background(), func(ctx context.Context, tx store.Tx) error {
		return tx.InsertEdges(ctx, []core.ClaimEdge{{
			SessionID: r.sess.ID, FromID: claims[1].ID, ToID: claims[2].ID,
			Kind: core.EdgeSupersedes, Weight: 1, CreatedBy: "seed",
		}})
	}); err != nil {
		t.Fatal(err)
	}

	res, err := r.v.Recheck(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if n := r.llm.count(); n != 0 {
		t.Errorf("%d calls with no contradiction edges", n)
	}
	if res.Rechecked != 0 || len(r.edges(t)) != 2 {
		t.Errorf("rechecked %d, %d edges; want 0 and 2", res.Rechecked, len(r.edges(t)))
	}
}

// TestRecheckRespectsTheDeadline checks a deadline inside the report allowance makes no call.
func TestRecheckRespectsTheDeadline(t *testing.T) {
	claims := recheckClaims()
	r := newRigFromSpec(t, budget.SessionSpec{
		Prompt: "q", Mode: core.ModeReport,
		ActorTypes: []core.ActorType{core.ActorWeb},
		BudgetUnit: core.BudgetTokens, Budget: 2_000_000,
		MaxLeads: 50, MaxToolCalls: 500, MaxWallClock: time.Hour,
	}, claims, judgeEveryPair(verifier.RelNeither))
	r.sess = r.reload(t)
	clock := &fakeClock{t: r.sess.CreatedAt.Add(time.Hour - 30*time.Second)}
	r.v.SetClock(clock.now)
	r.v.RecheckModel = "strict"
	seedContradictions(t, r, claims, [][2]int{{0, 1}, {2, 3}})

	res, err := r.v.Recheck(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if n := r.llm.count(); n != 0 {
		t.Errorf("%d calls inside the report allowance", n)
	}
	if !strings.Contains(res.Degraded, "wall clock") {
		t.Errorf("degraded = %q, want the wall clock named", res.Degraded)
	}
	if res.Withheld != 0 || countKind(r.edges(t), core.EdgeContradicts) != 2 {
		t.Errorf("withheld %d; edges changed without a re-check", res.Withheld)
	}
}

// insertEdges writes edges straight to the store.
func insertEdges(t *testing.T, r *rig, edges ...core.ClaimEdge) {
	t.Helper()
	if err := r.db.WithTx(context.Background(), func(ctx context.Context, tx store.Tx) error {
		return tx.InsertEdges(ctx, edges)
	}); err != nil {
		t.Fatal(err)
	}
}

// contradiction is a contradicts edge between claims i and j, created by by.
func contradiction(r *rig, claims []core.Claim, i, j int, by string) core.ClaimEdge {
	return core.ClaimEdge{
		SessionID: r.sess.ID, FromID: claims[i].ID, ToID: claims[j].ID,
		Kind: core.EdgeContradicts, Weight: 1, CreatedBy: by,
	}
}

// comparesPair reports whether a prompt names both claim i and claim j.
func comparesPair(prompt string, i, j int) bool {
	return strings.Contains(prompt, fmt.Sprintf("Finding number %d ", i)) &&
		strings.Contains(prompt, fmt.Sprintf("Finding number %d ", j))
}

// promptsFor returns the prompts sent with model.
func (s *scriptedLLM) promptsFor(model string) []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []string
	for i, m := range s.models {
		if m == model {
			out = append(out, s.prompts[i])
		}
	}
	return out
}

// firstCallThen answers call 0 with first and every later call with later.
func firstCallThen(first verifier.Relation, later func(int, string) (string, error)) func(int, string) (string, error) {
	return func(call int, prompt string) (string, error) {
		if call == 0 {
			return judgeEveryPair(first)(call, prompt)
		}
		return later(call, prompt)
	}
}

// storedConfidence maps claim ID to its stored confidence.
func storedConfidence(t *testing.T, r *rig) map[string]float64 {
	t.Helper()
	out := map[string]float64{}
	for _, c := range r.claims(t) {
		out[c.ID] = c.Confidence
	}
	return out
}

// derivedConfidence maps claim ID to the confidence derived over edges.
func derivedConfidence(t *testing.T, r *rig, edges []*core.ClaimEdge) map[string]float64 {
	t.Helper()
	var list []*core.Claim
	for _, c := range r.claims(t) {
		list = append(list, c)
	}
	scores, _ := verifier.DeriveConfidence(list, edges)
	out := map[string]float64{}
	for _, s := range scores {
		out[s.ClaimID] = s.Confidence
	}
	return out
}

// TestRunRechecksOnlyTheContradictionsItWrote checks a pass re-checks its own new edges and no older ones.
func TestRunRechecksOnlyTheContradictionsItWrote(t *testing.T) {
	claims := recheckClaims()
	r := newRig(t, 2_000_000, claims, judgeEveryPair(verifier.RelContradicts))
	r.v.RecheckModel = "strict"
	r.v.BatchSize = 1
	insertEdges(t, r, contradiction(r, claims, 0, 1, "seed"))

	res, err := r.v.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	written := 0
	for _, e := range r.edges(t) {
		if e.Kind != core.EdgeContradicts {
			continue
		}
		if e.FromID == claims[0].ID && e.ToID == claims[1].ID || e.FromID == claims[1].ID && e.ToID == claims[0].ID {
			if verifier.IsRechecked(e) {
				t.Errorf("the older edge was re-checked: created_by %q", e.CreatedBy)
			}
			continue
		}
		written++
		if !verifier.IsRechecked(e) {
			t.Errorf("new edge %s -> %s not marked re-checked: %q", e.FromID, e.ToID, e.CreatedBy)
		}
	}
	if written == 0 {
		t.Fatal("the pass wrote no contradictions")
	}
	strict := r.llm.promptsFor("strict")
	if len(strict) != written {
		t.Errorf("%d re-check calls, want one per new contradiction (%d)", len(strict), written)
	}
	for _, p := range strict {
		if comparesPair(p, 0, 1) {
			t.Error("the older contradiction was sent for re-check")
		}
	}
	if res.Rechecked != written || res.Withheld != 0 {
		t.Errorf("rechecked %d, withheld %d; want %d and 0", res.Rechecked, res.Withheld, written)
	}
}

// TestRunWithholdsWhatItsRecheckRejects checks a pass removes rejected edges, their count and their follow-ups.
func TestRunWithholdsWhatItsRecheckRejects(t *testing.T) {
	claims := recheckClaims()
	r := newRig(t, 2_000_000, claims, firstCallThen(verifier.RelContradicts, judgeEveryPair(verifier.RelNeither)))
	r.v.RecheckModel = "strict"
	r.v.BatchSize = 100

	res, err := r.v.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if res.Withheld == 0 || res.Withheld != res.Rechecked {
		t.Errorf("rechecked %d, withheld %d; want every contradiction withheld", res.Rechecked, res.Withheld)
	}
	if res.Contradictions != 0 {
		t.Errorf("contradictions = %d after every one was withheld", res.Contradictions)
	}
	if n := countKind(r.edges(t), core.EdgeContradicts); n != 0 {
		t.Errorf("%d contradicts edges stored after the re-check rejected them", n)
	}
	if len(res.FollowUps)+len(res.Deferred) != 0 {
		t.Errorf("%d follow-up(s) for withheld contradictions", len(res.FollowUps)+len(res.Deferred))
	}
}

// TestRecheckSkipsEdgesAlreadyRechecked checks a marked edge is never sent again.
func TestRecheckSkipsEdgesAlreadyRechecked(t *testing.T) {
	claims := recheckClaims()
	r := newRig(t, 2_000_000, claims, judgeEveryPair(verifier.RelContradicts))
	r.v.RecheckModel = "strict"
	r.v.BatchSize = 1
	insertEdges(t, r,
		contradiction(r, claims, 0, 1, verifier.RecheckedBy+"strict"),
		contradiction(r, claims, 2, 3, "seed"))

	res, err := r.v.Recheck(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	strict := r.llm.promptsFor("strict")
	if len(strict) != 1 || comparesPair(strict[0], 0, 1) {
		t.Errorf("%d re-check call(s); want one, for the unmarked edge", len(strict))
	}
	if res.Rechecked != 1 {
		t.Errorf("rechecked %d, want 1", res.Rechecked)
	}

	if _, err := r.v.Recheck(context.Background(), r.sess.ID); err != nil {
		t.Fatal(err)
	}
	if n := r.llm.count(); n != 1 {
		t.Errorf("%d calls after a second re-check, want still 1", n)
	}
}

// TestRecheckMarkerSurvivesReopen checks a re-opened store still knows which edges were re-checked.
func TestRecheckMarkerSurvivesReopen(t *testing.T) {
	claims := recheckClaims()
	r := newRig(t, 2_000_000, claims, judgeEveryPair(verifier.RelContradicts))
	r.v.RecheckModel = "strict"
	insertEdges(t, r, contradiction(r, claims, 0, 1, "seed"), contradiction(r, claims, 2, 3, "seed"))
	if _, err := r.v.Recheck(context.Background(), r.sess.ID); err != nil {
		t.Fatal(err)
	}
	if err := r.db.Close(); err != nil {
		t.Fatal(err)
	}

	db, err := sqlite.Open(r.path, sqlite.Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.Migrate(context.Background()); err != nil {
		t.Fatal(err)
	}
	fl := &scriptedLLM{reply: judgeEveryPair(verifier.RelNeither)}
	v := &verifier.Verifier{Store: db, Ledger: budget.New(db, budget.DefaultConfig()), LLM: fl,
		RecheckModel: "strict", Log: r.v.Log}
	res, err := v.Recheck(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if n := fl.count(); n != 0 || res.Rechecked != 0 {
		t.Errorf("%d calls, %d rechecked after reopening; want 0 and 0", n, res.Rechecked)
	}
	r.db = db
	for _, e := range r.edges(t) {
		if e.Kind == core.EdgeContradicts && !verifier.IsRechecked(e) {
			t.Errorf("edge %s -> %s lost its marker: %q", e.FromID, e.ToID, e.CreatedBy)
		}
	}
}

// TestUncheckedContradictionsDoNotScoreWhenRecheckIsOn checks confidence ignores unchecked contradictions.
func TestUncheckedContradictionsDoNotScoreWhenRecheckIsOn(t *testing.T) {
	claims := recheckClaims()
	r := newRig(t, 2_000_000, claims, firstCallThen(verifier.RelContradicts,
		func(int, string) (string, error) { return "", errors.New("provider hiccup") }))
	r.v.RecheckModel = "strict"
	r.v.BatchSize = 100

	if _, err := r.v.Run(context.Background(), r.sess.ID); err != nil {
		t.Fatal(err)
	}
	all := r.edges(t)
	if countKind(all, core.EdgeContradicts) == 0 {
		t.Fatal("the pass wrote no contradictions")
	}
	var kept []*core.ClaimEdge
	for _, e := range all {
		if verifier.IsRechecked(e) {
			t.Errorf("edge %s -> %s marked re-checked without a verdict", e.FromID, e.ToID)
		}
		if e.Kind != core.EdgeContradicts {
			kept = append(kept, e)
		}
	}
	got, want, counted := storedConfidence(t, r), derivedConfidence(t, r, kept), derivedConfidence(t, r, all)
	differs := false
	for id, w := range want {
		if got[id] != w {
			t.Errorf("claim %s confidence %.4f, want %.4f without the unchecked edges", id, got[id], w)
		}
		differs = differs || counted[id] != w
	}
	if !differs {
		t.Fatal("the contradictions would not have changed confidence; the test proves nothing")
	}
}

// TestUncheckedContradictionsScoreWhenRecheckIsOff checks confidence counts every contradiction without RecheckModel.
func TestUncheckedContradictionsScoreWhenRecheckIsOff(t *testing.T) {
	claims := recheckClaims()
	r := newRig(t, 2_000_000, claims, judgeEveryPair(verifier.RelContradicts))
	r.v.BatchSize = 100

	if _, err := r.v.Run(context.Background(), r.sess.ID); err != nil {
		t.Fatal(err)
	}
	all := r.edges(t)
	if countKind(all, core.EdgeContradicts) == 0 {
		t.Fatal("the pass wrote no contradictions")
	}
	got, want := storedConfidence(t, r), derivedConfidence(t, r, all)
	for id, w := range want {
		if got[id] != w {
			t.Errorf("claim %s confidence %.4f, want %.4f", id, got[id], w)
		}
	}
}

// TestRunCountsOnlyRecheckedContradictions checks a pass counts no contradiction its re-check never reached.
func TestRunCountsOnlyRecheckedContradictions(t *testing.T) {
	claims := recheckClaims()
	r := newRig(t, 2_000_000, claims, firstCallThen(verifier.RelContradicts,
		func(int, string) (string, error) { return "", errors.New("provider hiccup") }))
	r.v.RecheckModel = "strict"
	r.v.BatchSize = 100

	res, err := r.v.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if countKind(r.edges(t), core.EdgeContradicts) == 0 {
		t.Fatal("the pass wrote no contradictions")
	}
	if res.Contradictions != 0 {
		t.Errorf("contradictions = %d with none re-checked, want 0", res.Contradictions)
	}
}

// TestRechecksIsOffForToolkitSessions checks a toolkit session never withholds.
func TestRechecksIsOffForToolkitSessions(t *testing.T) {
	web := &core.Session{ActorTypes: []core.ActorType{core.ActorWeb}}
	kit := &core.Session{ActorTypes: []core.ActorType{core.ActorWeb, core.ActorToolkit}}
	if !verifier.Rechecks(web, "strict") {
		t.Error("an autonomous session with a re-check model does not withhold")
	}
	if verifier.Rechecks(web, "") || verifier.Rechecks(kit, "strict") || verifier.Rechecks(nil, "strict") {
		t.Error("withholds with no re-check model, for a toolkit session, or for no session")
	}
}

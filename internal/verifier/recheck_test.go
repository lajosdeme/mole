package verifier_test

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/budget"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/store"
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

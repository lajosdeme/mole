package verifier_test

// Adjudication batches run several at a time; these tests check that
// overlapping them changes nothing else about the result.

import (
	"context"
	"fmt"
	"hash/fnv"
	"math/rand/v2"
	"regexp"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/verifier"
)

// TestBatchesRunConcurrently checks that batches run at a concurrency of
// four rather than serially.
func TestBatchesRunConcurrently(t *testing.T) {
	const want = 4

	var (
		mu       sync.Mutex
		inFlight int
		peak     int
		gate     = make(chan struct{})
	)
	reply := func(call int, prompt string) (string, error) {
		mu.Lock()
		inFlight++
		peak = max(peak, inFlight)
		g := gate
		if inFlight >= want {
			close(gate)
			gate = make(chan struct{})
		}
		mu.Unlock()

		select {
		case <-g:
		case <-time.After(time.Second):
		}

		mu.Lock()
		inFlight--
		mu.Unlock()
		return judgeEveryPair(verifier.RelNeither)(call, prompt)
	}

	r := newRig(t, 2_000_000, distinctClaims(5), reply)
	r.v.BatchSize = 1 // one pair per batch: five related claims make ten batches

	res, err := r.v.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if batches := res.PairsRetrieved - res.PairsDecidedFree; batches != 10 {
		t.Fatalf("%d batches, want 10 — the rig no longer makes the work this test needs", batches)
	}

	mu.Lock()
	got := peak
	mu.Unlock()
	if got != want {
		t.Errorf("at most %d adjudication calls were in flight at once, want %d", got, want)
	}
	if res.PairsJudged != 10 {
		t.Errorf("%d pairs judged, want 10", res.PairsJudged)
	}
}

// promptPairRE reads the numbered data pairs out of an adjudication prompt,
// capturing each side's "claim" field and ignoring its other members. It
// requires "b", which the instructions' shape example does not carry.
var promptPairRE = regexp.MustCompile(
	`\{"pair":(\d+),"a":\{"claim":"((?:[^"\\]|\\.)*)"[^}]*\},"b":\{"claim":"((?:[^"\\]|\\.)*)"[^}]*\}\}`)

// pairAnswerer answers each pair by its text, not call order; the second
// time a pair is asked (confirm), some answers flip.
type pairAnswerer struct {
	mu    sync.Mutex
	asked map[string]int
	// prompts is every prompt received, fence removed, in arrival order.
	prompts []string
}

var answerRelations = []verifier.Relation{
	verifier.RelContradicts, verifier.RelDuplicate, verifier.RelSupports,
	verifier.RelNeither, verifier.RelRefines, verifier.RelNeither,
}

func (p *pairAnswerer) reply(_ int, prompt string) (string, error) {
	p.mu.Lock()
	p.prompts = append(p.prompts, unfenced(prompt))
	p.mu.Unlock()
	time.Sleep(time.Duration(rand.IntN(21)) * time.Millisecond)

	var b strings.Builder
	b.WriteString(`{"verdicts":[`)
	for i, m := range promptPairRE.FindAllStringSubmatch(prompt, -1) {
		key := m[2] + "|" + m[3]
		p.mu.Lock()
		p.asked[key]++
		nth := p.asked[key]
		p.mu.Unlock()

		h := fnv.New32a()
		_, _ = h.Write([]byte(key))
		sum := h.Sum32()
		rel := answerRelations[sum%uint32(len(answerRelations))]
		if nth > 1 && (sum/7)%3 == 0 {
			rel = verifier.RelNeither // the second look disagrees
		}
		if i > 0 {
			b.WriteString(",")
		}
		fmt.Fprintf(&b, `{"pair":%s,"relation":%q,"confidence":0.8,"why":"because"}`, m[1], rel)
	}
	b.WriteString("]}")
	return b.String(), nil
}

// fenceRE matches core.PromptFence's output: random per call, so two runs'
// prompts can only be compared with it removed.
var fenceRE = regexp.MustCompile(`[0-9a-f]{16}`)

func unfenced(prompt string) string { return fenceRE.ReplaceAllString(prompt, "FENCE") }

// idedClaims is distinctClaims with fixed IDs. Pair order and orientation follow
// claim IDs, so two rigs must share IDs for their batches to match.
func idedClaims(n int) []core.Claim {
	cs := distinctClaims(n)
	for i := range cs {
		cs[i].ID = fmt.Sprintf("c_%02d", i)
	}
	return cs
}

// TestConcurrentRunMatchesSerialRun checks that a concurrent pass writes the
// same edges as a serial one.
func TestConcurrentRunMatchesSerialRun(t *testing.T) {
	type outcome struct {
		edges   []string
		prompts []string
		res     *verifier.Result
	}
	run := func(concurrency int) outcome {
		ans := &pairAnswerer{asked: map[string]int{}}
		r := newRig(t, 4_000_000, idedClaims(9), ans.reply)
		r.v.BatchSize = 3
		r.v.Concurrency = concurrency
		r.v.Confirm = verifier.ConfirmBoth

		res, err := r.v.Run(context.Background(), r.sess.ID)
		if err != nil {
			t.Fatalf("concurrency %d: run: %v", concurrency, err)
		}
		if res.Degraded != "" {
			t.Fatalf("concurrency %d: degraded: %s", concurrency, res.Degraded)
		}
		var edges []string
		for _, e := range r.edges(t) {
			edges = append(edges, fmt.Sprintf("%s %s %s", e.Kind, e.FromID, e.ToID))
		}
		sort.Strings(edges)
		prompts := append([]string(nil), ans.prompts...)
		sort.Strings(prompts)
		return outcome{edges: edges, prompts: prompts, res: res}
	}

	serial := run(1)
	concurrent := run(4)

	if len(serial.edges) == 0 {
		t.Fatal("the serial run wrote no edges, so there is nothing to compare")
	}
	if serial.res.PairsConfirmed == 0 || serial.res.PairsUnconfirmed == 0 {
		t.Fatalf("confirmed=%d unconfirmed=%d; the answerer should exercise both",
			serial.res.PairsConfirmed, serial.res.PairsUnconfirmed)
	}
	if strings.Join(serial.edges, "\n") != strings.Join(concurrent.edges, "\n") {
		t.Errorf("edge sets differ.\nserial:\n  %s\nconcurrent:\n  %s",
			strings.Join(serial.edges, "\n  "), strings.Join(concurrent.edges, "\n  "))
	}
	// The same prompts too, in any order.
	if len(serial.prompts) != len(concurrent.prompts) {
		t.Errorf("%d prompts serially, %d concurrently", len(serial.prompts), len(concurrent.prompts))
	} else {
		for i := range serial.prompts {
			if serial.prompts[i] != concurrent.prompts[i] {
				t.Errorf("prompt sets differ; first difference:\nserial:\n%s\nconcurrent:\n%s",
					pairsIn(serial.prompts[i]), pairsIn(concurrent.prompts[i]))
				break
			}
		}
	}
	if serial.res.ClaimsVerified != concurrent.res.ClaimsVerified {
		t.Errorf("claims verified: serial %d, concurrent %d",
			serial.res.ClaimsVerified, concurrent.res.ClaimsVerified)
	}
	for name, pair := range map[string][2]int{
		"calls":       {serial.res.Calls, concurrent.res.Calls},
		"judged":      {serial.res.PairsJudged, concurrent.res.PairsJudged},
		"unjudged":    {serial.res.PairsUnjudged, concurrent.res.PairsUnjudged},
		"confirmed":   {serial.res.PairsConfirmed, concurrent.res.PairsConfirmed},
		"unconfirmed": {serial.res.PairsUnconfirmed, concurrent.res.PairsUnconfirmed},
	} {
		if pair[0] != pair[1] {
			t.Errorf("pairs %s: serial %d, concurrent %d", name, pair[0], pair[1])
		}
	}
}

// TestAllowanceBoundsTheWave checks that a wave's calls together, not
// individually, respect the allowance.
func TestAllowanceBoundsTheWave(t *testing.T) {
	// The allowance covers exactly two calls. Token mode, so the estimate
	// doesn't depend on the Verifier's pricing.
	est := verifier.EstimateBatch(&verifier.Verifier{}, core.BudgetTokens, 1)
	r := newRig(t, 4*est, distinctClaims(4), judgeEveryPair(verifier.RelNeither))
	r.v.BatchSize = 1 // four related claims make six one-pair batches
	r.v.MaxShareOfBudget = 0.5
	r.llm.usage = llm.Usage{InputTokens: est}

	res, err := r.v.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if batches := res.PairsRetrieved - res.PairsDecidedFree; batches != 6 {
		t.Fatalf("%d batches, want 6 — the rig no longer makes the work this test needs", batches)
	}
	if got := r.llm.count(); got != 2 {
		t.Errorf("%d calls against an allowance for 2", got)
	}
	if !strings.Contains(res.Degraded, "allowance exhausted") {
		t.Errorf("degraded = %q, want the allowance named", res.Degraded)
	}
	if res.PairsUnjudged != 4 {
		t.Errorf("%d pairs unjudged, want the 4 never sent", res.PairsUnjudged)
	}
	r.assertNothingHeld(t, "allowance")
}

// TestFatalErrorStopsLaterWaves checks that a fatal error stops every wave
// after the one already in flight, including the confirm pass.
func TestFatalErrorStopsLaterWaves(t *testing.T) {
	r := newRig(t, 2_000_000, distinctClaims(6), func(call int, prompt string) (string, error) {
		if call == 0 {
			return "", llm.ErrUnauthorized
		}
		return judgeEveryPair(verifier.RelContradicts)(call, prompt)
	})
	r.v.BatchSize = 1 // fifteen batches: four waves' worth
	r.v.Confirm = verifier.ConfirmBoth

	res, err := r.v.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if batches := res.PairsRetrieved - res.PairsDecidedFree; batches <= 2*verifier.DefaultConcurrency {
		t.Fatalf("%d batches; too few for a later wave to exist", batches)
	}
	if got := r.llm.count(); got > verifier.DefaultConcurrency {
		t.Errorf("%d calls after a fatal error; at most the first wave of %d may run",
			got, verifier.DefaultConcurrency)
	}
	if !strings.Contains(res.Degraded, "unauthorized") {
		t.Errorf("degraded = %q, want the fatal cause named", res.Degraded)
	}
	r.assertNothingHeld(t, "fatal")
}

// TestABatchPanicReachesTheCaller checks that a panic inside a wave goroutine
// is carried back to the caller of Run rather than crashing the process.
func TestABatchPanicReachesTheCaller(t *testing.T) {
	r := newRig(t, 2_000_000, distinctClaims(5), func(call int, prompt string) (string, error) {
		if call == 1 {
			panic("malformed model output")
		}
		return judgeEveryPair(verifier.RelNeither)(call, prompt)
	})
	r.v.BatchSize = 1

	var recovered any
	func() {
		defer func() { recovered = recover() }()
		_, _ = r.v.Run(context.Background(), r.sess.ID)
	}()
	if recovered == nil {
		t.Fatal("Run returned normally; the batch's panic was swallowed")
	}
	if !strings.Contains(fmt.Sprint(recovered), "malformed model output") {
		t.Errorf("recovered %v, want the original panic carried through", recovered)
	}
}

// pairsIn is a prompt's data pairs, for a failure message shorter than the prompt.
func pairsIn(prompt string) string {
	return strings.Join(promptPairRE.FindAllString(prompt, -1), "\n")
}

// fixedPairRetriever maps a claim ID to its one partner, so a test can pin
// exactly which pairs and batches form.
type fixedPairRetriever map[string]string

func (f fixedPairRetriever) Candidates(_ context.Context, target *core.Claim, pool []*core.Claim, _ int) ([]*core.Claim, error) {
	partner, ok := f[target.ID]
	if !ok {
		return nil, nil
	}
	for _, c := range pool {
		if c.ID == partner {
			return []*core.Claim{c}, nil
		}
	}
	return nil, nil
}

// TestUnreachedPairsLeaveTheirClaimsUnverified checks that claims in a pair
// the pass never reached stay unverified rather than marked settled.
func TestUnreachedPairsLeaveTheirClaimsUnverified(t *testing.T) {
	ctx := context.Background()
	claims := idedClaims(6)

	// One reservation's worth of allowance, so exactly one of the three disjoint
	// pairs below gets adjudicated.
	est := verifier.EstimateBatch(&verifier.Verifier{}, core.BudgetTokens, 1)
	r := newRig(t, 4*est, claims, judgeEveryPair(verifier.RelNeither))
	r.v.BatchSize = 1
	r.v.MaxShareOfBudget = 0.25 // ceiling = est: room for exactly the first batch
	r.v.Retriever = fixedPairRetriever{
		"c_00": "c_01", "c_01": "c_00",
		"c_02": "c_03", "c_03": "c_02",
		"c_04": "c_05", "c_05": "c_04",
	}
	r.llm.usage = llm.Usage{InputTokens: est}

	res, err := r.v.Run(ctx, r.sess.ID)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.PairsUnjudged != 2 {
		t.Fatalf("%d pairs unjudged, want 2 — the rig no longer sets up three disjoint "+
			"batches with room for only one", res.PairsUnjudged)
	}
	if res.ClaimsVerified != 2 {
		t.Errorf("ClaimsVerified = %d, want 2 (only the judged pair's claims)", res.ClaimsVerified)
	}

	judged := map[string]bool{"c_00": true, "c_01": true}
	byText := r.claims(t)
	for i := 0; i < 6; i++ {
		c := byText[fmt.Sprintf("Subword tokenization affects byte-level scaling in variant %d.", i)]
		if c == nil {
			t.Fatalf("claim %d not found", i)
		}
		want := judged[c.ID]
		if got := c.VerifiedAt != nil; got != want {
			t.Errorf("claim %s: VerifiedAt set = %v, want %v (judged=%v)", c.ID, got, want, want)
		}
	}

	// A second pass re-retrieves the unjudged pairs, since CandidatePairs only
	// skips a pair that already carries an edge.
	r.v.MaxShareOfBudget = 1
	second, err := r.v.Run(ctx, r.sess.ID)
	if err != nil {
		t.Fatalf("second run: %v", err)
	}
	if second.PairsJudged != 2 {
		t.Errorf("second pass judged %d pairs, want exactly the 2 left over", second.PairsJudged)
	}
	if second.PairsUnjudged != 0 {
		t.Errorf("second pass left %d pairs unjudged", second.PairsUnjudged)
	}
	if n := r.unverified(t); n != 0 {
		t.Errorf("%d claims still unverified after the second pass", n)
	}
}

// TestUnreachedPairsKeepTheirDerivedConfidence checks that an unjudged
// claim keeps its derived confidence instead of reverting to 0.
func TestUnreachedPairsKeepTheirDerivedConfidence(t *testing.T) {
	ctx := context.Background()
	claims := idedClaims(6)

	// Allowance for one of three disjoint pairs.
	est := verifier.EstimateBatch(&verifier.Verifier{}, core.BudgetTokens, 1)
	r := newRig(t, 4*est, claims, judgeEveryPair(verifier.RelNeither))
	r.v.BatchSize = 1
	r.v.MaxShareOfBudget = 0.25
	r.v.Retriever = fixedPairRetriever{
		"c_00": "c_01", "c_01": "c_00",
		"c_02": "c_03", "c_03": "c_02",
		"c_04": "c_05", "c_05": "c_04",
	}
	r.llm.usage = llm.Usage{InputTokens: est}

	res, err := r.v.Run(ctx, r.sess.ID)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.PairsUnjudged != 2 {
		t.Fatalf("%d pairs unjudged, want 2 — the rig no longer stops after one batch", res.PairsUnjudged)
	}
	derived := map[string]float64{}
	for _, s := range res.Scores {
		for _, c := range s.Cluster.Claims {
			derived[c.ID] = s.Confidence
		}
	}

	skipped := map[string]bool{"c_02": true, "c_03": true, "c_04": true, "c_05": true}
	byID := map[string]*core.Claim{}
	for _, c := range r.claims(t) {
		byID[c.ID] = c
	}
	for id := range skipped {
		c := byID[id]
		if c == nil {
			t.Fatalf("claim %s not found", id)
		}
		if c.VerifiedAt != nil {
			t.Errorf("claim %s: marked verified, but its pair was never judged", id)
		}
		want, ok := derived[id]
		if !ok {
			t.Fatalf("claim %s: the derivation did not cover it; the test proves nothing", id)
		}
		if want == 0 {
			t.Fatalf("claim %s: derived confidence is 0, so persisting it is indistinguishable from dropping it", id)
		}
		if c.Confidence != want {
			t.Errorf("claim %s: confidence = %v, want the derived %v", id, c.Confidence, want)
		}
	}

	// Still re-judged: keeping the score did not mark the claims.
	r.v.MaxShareOfBudget = 1
	second, err := r.v.Run(ctx, r.sess.ID)
	if err != nil {
		t.Fatalf("second run: %v", err)
	}
	if second.PairsJudged != 2 {
		t.Errorf("second pass judged %d pairs, want the 2 left over", second.PairsJudged)
	}
	if n := r.unverified(t); n != 0 {
		t.Errorf("%d claims still unverified after the second pass", n)
	}
}

// TestAFatalSecondLookStopsLaterWaves checks that a fatal error on a second
// look stops later waves, the same as a fatal first look.
func TestAFatalSecondLookStopsLaterWaves(t *testing.T) {
	var (
		mu    sync.Mutex
		asked = map[string]int{}
		first int
	)
	r := newRig(t, 2_000_000, idedClaims(6), func(call int, prompt string) (string, error) {
		mu.Lock()
		defer mu.Unlock()
		for _, m := range promptPairRE.FindAllStringSubmatch(prompt, -1) {
			key := m[2] + "|" + m[3]
			asked[key]++
			if asked[key] > 1 {
				return "", llm.ErrUnauthorized
			}
		}
		first++
		return judgeEveryPair(verifier.RelContradicts)(call, prompt)
	})
	r.v.BatchSize = 1
	r.v.Concurrency = 1 // one first-look batch per wave: three waves
	r.v.Confirm = verifier.ConfirmBoth
	r.v.Retriever = fixedPairRetriever{
		"c_00": "c_01", "c_01": "c_00",
		"c_02": "c_03", "c_03": "c_02",
		"c_04": "c_05", "c_05": "c_04",
	}

	res, err := r.v.Run(context.Background(), r.sess.ID)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if first != 1 {
		t.Errorf("%d first-look call(s); a fatal second look must stop the waves after it", first)
	}
	if got := r.llm.count(); got != 2 {
		t.Errorf("%d calls, want 2: the first look and the second look that failed", got)
	}
	if !strings.Contains(res.Degraded, "unauthorized") {
		t.Errorf("degraded = %q, want the fatal cause named", res.Degraded)
	}
	if res.PairsUnjudged != 2 {
		t.Errorf("%d pairs unjudged, want the 2 never sent", res.PairsUnjudged)
	}
	r.assertNothingHeld(t, "fatal second look")
}

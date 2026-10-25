package verifier_test

// Adjudication batches run several at a time. Measured: one Standard run spent
// its last ~5 of 12 minutes in 291 serial verifier calls, each an independent
// model call. These tests pin that the batches overlap, and that overlapping them
// changes nothing else — not the graph, not the allowance, not the fatal stop.

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

// TestBatchesRunConcurrently. Each call blocks until four are in flight (or
// 200 ms pass), so a serial loop shows a peak of one and an unbounded one more
// than four.
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
		case <-time.After(200 * time.Millisecond):
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

// promptPairRE reads the numbered data pairs out of an adjudication prompt. It
// requires "b", which the instructions' shape example does not carry.
var promptPairRE = regexp.MustCompile(`\{"pair":(\d+),"a":"((?:[^"\\]|\\.)*)","b":"((?:[^"\\]|\\.)*)"\}`)

// pairAnswerer answers each pair by its text alone, never by call order, after a
// random 0-20 ms sleep — so a run that judges batches in a different order, or
// batches its second looks differently, still gets the same answer for each pair.
//
// The second time a pair is asked about (the confirm pass), some pairs change
// their answer, so both the confirmed and the withheld path are exercised.
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

// TestConcurrentRunMatchesSerialRun. Results are appended in batch order, not
// completion order, so the graph a concurrent pass writes is the serial pass's
// graph exactly.
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
	// The same prompts, first looks and second looks alike, in any order. A real
	// model's answer depends on what else shares its batch, so a second look that
	// mixed pairs from several first-look batches could change the graph even
	// though this answerer, judging each pair alone, would not show it.
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

// TestAllowanceBoundsTheWave. Every call in a wave reserves before any settles,
// so the wave itself must fit the allowance: four launched against room for two
// would all pass a per-batch check and spend twice the share.
func TestAllowanceBoundsTheWave(t *testing.T) {
	// Each call charges exactly its reservation, and the share is exactly two of
	// them, so the allowance is gone after two calls whatever order they run in.
	// Token mode, so the estimate does not depend on the Verifier's pricing.
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

// TestFatalErrorStopsLaterWaves. A bad credential fails every batch alike; the
// wave already in flight cannot be recalled, but nothing after it may start —
// neither a later wave nor the confirm pass over the wave's own verdicts.
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

// TestABatchPanicReachesTheCaller. Serially, a panic in adjudication unwound into
// the session supervisor's recover, which fails that one session and releases its
// holds. A bare wave goroutine is invisible to that recover — the executor's lead
// pool hit exactly this — so the panic must be carried back to the goroutine that
// called Run rather than taking the process down.
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

// fixedPairRetriever maps a claim ID to the one partner it is ever candidated
// against, so a test can pin exactly which pairs — and so which batches — form,
// independent of the lexical retriever's scoring.
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

// TestUnreachedPairsLeaveTheirClaimsUnverified. A pass that stops partway through
// its batches — allowance, wall clock, or a fatal error — must not mark the claims
// in a pair it never adjudicated. Marking them anyway would be permanent: the
// pair has no edge, but persist's old behaviour treated every target as settled,
// so CandidatePairs would never surface that pair's claims again either (their
// VerifiedAt disqualifies them as targets) and the comparison that should have
// happened never does.
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

	// A second pass with an ample allowance re-retrieves exactly the pairs left
	// unjudged: CandidatePairs only skips a pair that already carries an edge, and
	// an unjudged pair has none.
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

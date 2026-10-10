package stats

import (
	"fmt"
	"math"
	"sort"
)

// Comparing more than two groups (§4).
//
// Every pair is tested and every p adjusted by Holm-Bonferroni before a verdict
// is derived. Ten groups at alpha = 0.05 give a 90% chance of at least one
// spurious "significant", which is what corrections exist for — comparing only
// the two largest buckets is not the only alternative.
//
// Holm rather than plain Bonferroni because it dominates it: uniformly more
// powerful at the same family-wise error rate. Not Benjamini-Hochberg, which
// controls the false-discovery RATE — right for screening a hundred hypotheses,
// wrong when a research tool is about to state a difference as a finding. A
// reader wants the probability that ANY stated difference is spurious to be
// small.
//
// The unadjusted p is kept beside the adjusted one, because a reader who knows
// statistics will want to see both and one who does not should see the number the
// verdict was actually derived from.

// MaxGroups bounds how many buckets are compared pairwise.
//
// Six groups is fifteen comparisons, at which point Holm's threshold for the
// smallest p is alpha/15 = 0.0033 and a genuine moderate effect stops being
// detectable — the correction has eaten the power the extra groups were supposed
// to buy. Past this the largest groups are compared and the envelope says how many
// were left out, which is the honest version of the old behaviour rather than a
// silent one.
const MaxGroups = 6

// Pairwise tests every pair of groups, adjusting for the number of comparisons.
//
// Returns the tests in the order they should be read — most significant first —
// and a sentence describing what was and was not compared. Groups that cannot
// support a test (fewer than two records, no resolvable spread) are skipped, and
// the count of comparisons for the correction is the number actually RUN rather
// than the number of pairs that exist: correcting for tests that never happened
// would inflate every p for nothing.
func Pairwise(measure string, groups []Group) ([]Test, string) {
	if len(groups) < 2 {
		return nil, ""
	}
	// Largest first, so a cap keeps the groups a reader would expect it to.
	ordered := append([]Group(nil), groups...)
	sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].N > ordered[j].N })

	var note string
	if len(ordered) > MaxGroups {
		note = fmt.Sprintf("%d groups were found; the %d largest were compared pairwise "+
			"and %d smaller group(s) were not tested — past %d groups the "+
			"multiple-comparison correction removes the power the extra comparisons "+
			"would add", len(ordered), MaxGroups, len(ordered)-MaxGroups, MaxGroups)
		ordered = ordered[:MaxGroups]
	}

	var tests []Test
	for i := 0; i < len(ordered); i++ {
		for j := i + 1; j < len(ordered); j++ {
			t, err := WelchOrReason(measure, ordered[i], ordered[j])
			if err != nil {
				continue
			}
			tests = append(tests, t)
		}
	}
	if len(tests) == 0 {
		return nil, note
	}

	Holm(tests)
	// Most significant first: an envelope carrying fifteen comparisons is read
	// from the top, and the adjusted p is what decides the order because it is
	// what decides the verdict.
	sort.SliceStable(tests, func(i, j int) bool {
		if tests[i].PAdjusted != tests[j].PAdjusted {
			return tests[i].PAdjusted < tests[j].PAdjusted
		}
		return tests[i].P < tests[j].P
	})

	if len(tests) > 1 {
		correction := fmt.Sprintf("%d pairwise comparisons were run, so every p-value "+
			"is Holm-adjusted for multiple testing; the unadjusted value is kept beside "+
			"it and the verdict is derived from the adjusted one", len(tests))
		if note != "" {
			note += ". " + correction
		} else {
			note = correction
		}
	}
	return tests, note
}

// Holm applies the Holm–Bonferroni step-down adjustment in place, and re-derives
// every verdict and sentence from the adjusted p.
//
// The procedure: order the m p-values ascending, multiply the i-th by (m − i), and
// enforce monotonicity so an adjusted value can never fall below the one before
// it. Capped at 1, because a probability cannot exceed it and an "adjusted p =
// 3.4" in a passage a model quotes would be read as a number rather than as
// arithmetic.
//
// A single comparison is the identity, which matters: it means the two-group path
// and the k-group path are the same code, and there is no version of the verdict
// rule that only the one-comparison case takes.
func Holm(tests []Test) {
	m := len(tests)
	if m == 0 {
		return
	}
	idx := make([]int, m)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return tests[idx[a]].P < tests[idx[b]].P })

	var running float64
	for rank, i := range idx {
		adj := float64(m-rank) * tests[i].P
		// Monotone: the step-down procedure requires the adjusted sequence to be
		// non-decreasing, and without this a later, larger raw p can produce a
		// smaller adjusted one — which would make a weaker result look stronger.
		running = math.Max(running, adj)
		tests[i].PAdjusted = math.Min(1, running)
		tests[i].Comparisons = m
		tests[i].Verdict = VerdictFor(tests[i].PAdjusted, tests[i].NA, tests[i].NB)
		tests[i].Summary = summarize(tests[i])
	}
}

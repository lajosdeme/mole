package stats

import (
	"fmt"
	"sort"
)

// Stability across holdout windows (§4).
//
// A significant result on one sample is one draw, and the failure this tool is
// most exposed to is a difference driven by a subset — one month, one region,
// one batch of imports — reported as a property of the data. Re-testing on
// disjoint subsets catches that; no choice of alpha does.
//
// A difference is STABLE when it points the same way in every window with
// enough data to test. Direction, not significance: a third of the rows has a
// third of the power, so requiring each window to clear alpha would call almost
// every real effect unstable — a statement about sample size dressed as one
// about the data. The count of individually significant windows is reported
// beside it.
//
// The per-window tests are NOT Holm-corrected against each other: they are one
// hypothesis re-examined on disjoint data, not three hypotheses. See Pairwise
// for the case where correction is right.

// HoldoutWindows is how many disjoint subsets a comparison is re-tested on.
//
// Three, because §4 says three. It is also the smallest number for which
// "consistent across windows" says anything: two windows agreeing is a coin
// landing the same way twice.
const HoldoutWindows = 3

// Holdout is what the re-tests found.
type Holdout struct {
	// Windows is how many windows could be tested at all. Fewer than
	// HoldoutWindows means a window fell below the k-anonymity floor or had no
	// resolvable spread — which is a fact about the split, not about the effect.
	Windows int
	// Agreed is how many of those pointed the same way as the pooled difference.
	Agreed int
	// Significant is how many cleared alpha on their own third of the data.
	Significant int
	// Stable is Agreed == Windows == HoldoutWindows.
	Stable bool
	// Clause is appended to the sentence a claim must quote.
	Clause string
}

// CheckStability re-tests one comparison on the holdout windows.
//
// a and b map a window label to that window's slice of each group. A window is
// tested only if both sides are present and testable.
func CheckStability(measure string, pooled Test, a, b map[string]Group) Holdout {
	var h Holdout
	labels := make([]string, 0, len(a))
	for label := range a {
		if _, ok := b[label]; ok {
			labels = append(labels, label)
		}
	}
	sort.Strings(labels)

	for _, label := range labels {
		t, err := WelchOrReason(measure, a[label], b[label])
		if err != nil {
			continue
		}
		h.Windows++
		// Same direction as the pooled difference. Sign comparison rather than a
		// threshold: the magnitudes will differ between thirds of a sample, and
		// requiring them to be close would test the sample size again.
		if (t.Difference > 0) == (pooled.Difference > 0) {
			h.Agreed++
		}
		if t.P < Alpha {
			h.Significant++
		}
	}

	h.Stable = h.Windows == HoldoutWindows && h.Agreed == h.Windows
	h.Clause = h.clause()
	return h
}

func (h Holdout) clause() string {
	switch {
	case h.Windows == 0:
		// Said, not omitted. "No stability information" and "stable" must not look
		// alike in a passage a claim is quoted from.
		return fmt.Sprintf("; stability across %d holdout windows could NOT be checked "+
			"— no window held enough records on both sides", HoldoutWindows)
	case h.Stable:
		return fmt.Sprintf("; the difference points the same way in all %d holdout "+
			"windows (%d of them significant on their own third of the data)",
			h.Windows, h.Significant)
	case h.Windows < HoldoutWindows:
		return fmt.Sprintf("; only %d of %d holdout windows could be tested, and the "+
			"difference points the same way in %d of them — NOT established as stable",
			h.Windows, HoldoutWindows, h.Agreed)
	default:
		return fmt.Sprintf("; the difference does NOT hold across holdout windows — it "+
			"points the same way in only %d of %d, so it may be driven by a subset "+
			"of the records rather than being a property of the data",
			h.Agreed, h.Windows)
	}
}

// WithHoldout returns t carrying the stability result, with the clause folded into
// the sentence a claim must quote.
//
// Folded into the SENTENCE rather than left as a field, for the reason the Holm
// correction is: §11.5 makes a claim reproduce the passage verbatim, and that is
// the only lever that stops a model reporting a subset-driven difference as a
// finding. A field it never reads is not a lever.
func WithHoldout(t Test, h Holdout) Test {
	t.Holdout = &h
	t.Summary += h.Clause
	return t
}

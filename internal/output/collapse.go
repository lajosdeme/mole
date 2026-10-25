package output

import (
	"fmt"
	"sort"
	"strings"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/verifier"
)

// Duplicate collapse and disagreement disclosure (§11.2, §11.3).
//
// §11.2's rule: a duplicate_of cluster renders as ONE claim with N sources. Two
// things follow, and both were visibly wrong in the last live run.
//
// The report stops restating itself. That run produced
//
//	[1] MambaByte compares favorably to various subword baselines
//	[1] Compared to existing subword models, MambaByte is competitive
//
// as two separate bullets from one source — the same finding twice, which reads as
// two independent facts and is neither.
//
// And corroboration becomes visible. Three publishers agreeing is the strongest
// signal §11.3 has, and rendering them as three bullets hides it behind repetition
// rather than showing it as one claim with three citations.

// Finding is one assertion put in front of the synthesis model.
//
// A collapsed cluster, so Sources may hold several. Contradicts names the findings
// that dispute it — surfaced rather than resolved, because a reader who cannot see a
// disagreement cannot judge it, and §11.3's lowered confidence is a number the prose
// never shows.
type Finding struct {
	Claim *core.Claim
	// Sources are every distinct source asserting this, representative first.
	Sources []string
	// Publishers is how many DISTINCT publishers those sources represent (§11.3's
	// corroboration signal, which is not the same as the URL count).
	Publishers int
	// breadth is how many claims in the session came from this finding's sources.
	// A pre-Verifier tie-break only — see the ordering note on Findings.
	breadth int
	// Confidence is the cluster's derived figure.
	Confidence float64
	// Contradicts holds the indices, into the Findings slice, of findings that
	// dispute this one.
	Contradicts []int
	// Superseded is set when a later-published finding supersedes this one.
	Superseded bool

	// cluster is every claim collapsed into this finding, so a citation can carry
	// the quote its own source supplied.
	cluster []*core.Claim
	// order is the earliest position its claims held in the input, which arrives
	// from the store in created_at order. Used as the final tie-break.
	order int
	// disputes holds the representative claim IDs of findings that contradict this
	// one, recorded before the sort and resolved to indices after it.
	disputes []string
}

// Findings collapses claims into what the report should say.
//
// Ordered by DERIVED confidence (§11.3), never by the extractor's self-report. This
// ordering replaced a function that sorted by the latter under the same field name, so
// an uncalibrated number — one §11.3 describes as "mostly encoding fluency" — decided
// which claims led the answer and which were dropped at the cap. A fluent claim from
// one anonymous page outranked a claim three independent publishers agreed on.
//
// Before the Verifier scores a session every derived confidence is 0 and the sort has
// nothing to work with. Assertion strength is the wrong tie-break — preferring it is
// the exact behaviour that was removed — so ties fall back to publisher breadth, which
// is mechanical and claims nothing.
//
// Truncated to max AFTER collapsing. Collapsing first is what lets the cap count
// distinct assertions rather than repetitions; a cap applied to raw claims spends its
// whole budget on one page saying one thing eight ways.
func Findings(claims []*core.Claim, edges []*core.ClaimEdge, max int) []Finding {
	if len(claims) == 0 {
		return nil
	}

	// Built FROM the Verifier's scores rather than re-derived.
	//
	// This function used to recompute distinct publishers, superseded-as-target and
	// cross-cluster contradictions from the same (claims, edges) inputs that
	// verifier.scoreCluster had already reduced — two implementations of one derivation,
	// agreeing only because both were written carefully. Meanwhile Score held all of it
	// and was discarded.
	_, scores := verifier.DeriveConfidence(claims, edges)

	// Input position per claim: the final tie-break. Claim IDs carry a millisecond
	// timestamp and a random suffix, so a batch inserted in one transaction orders
	// RANDOMLY, and sorting on ID made citation numbering differ between runs of one
	// session. ListClaims returns created_at order, which is stable and meaningful.
	inputPos := make(map[string]int, len(claims))
	perSource := map[string]int{}
	for i, c := range claims {
		if c == nil {
			continue
		}
		inputPos[c.ID] = i
		perSource[c.Source]++
	}

	out := make([]Finding, 0, len(scores))
	for _, sc := range scores {
		rep := sc.Cluster.Representative()
		if rep == nil {
			continue
		}
		f := Finding{
			Claim: rep,
			// The STORED confidence, not the one just recomputed. They agree in
			// production — same function, same inputs — but the stored number is what
			// ScoreClaims persisted, what `mole trace` prints, and what a reader can
			// look up. Ordering the report by a second, freshly computed figure would
			// mean two numbers for one claim, which is worse than the duplicated
			// derivation this change removes.
			//
			// Score is used for the STRUCTURE it already worked out: cluster membership,
			// distinct publishers, superseded-as-target.
			Confidence: rep.Confidence,
			Publishers: sc.Publishers,
			Superseded: sc.Superseded,
			cluster:    sc.Cluster.Claims,
			order:      len(claims),
		}

		seen := map[string]bool{}
		// Representative's source first, so the citation a reader checks first is the
		// strongest-classed one.
		for _, c := range append([]*core.Claim{rep}, sc.Cluster.Claims...) {
			if c.Source == "" || seen[c.Source] {
				continue
			}
			seen[c.Source] = true
			f.Sources = append(f.Sources, c.Source)
			f.breadth += perSource[c.Source]
		}
		for _, c := range sc.Cluster.Claims {
			if p, ok := inputPos[c.ID]; ok && p < f.order {
				f.order = p
			}
		}
		out = append(out, f)
	}

	// Disagreements, recorded by the opposing finding's REPRESENTATIVE CLAIM ID rather
	// than by index. Indices do not survive the sort below, and an earlier version
	// translated them using the sorted slice as the pre-sort reference — incoherent, and
	// it passed whenever the sort happened not to reorder anything.
	repOf := make(map[string]string, len(claims))
	for _, f := range out {
		for _, c := range f.cluster {
			repOf[c.ID] = f.Claim.ID
		}
	}
	disputes := map[string][]string{}
	for _, e := range edges {
		if e == nil || e.Kind != core.EdgeContradicts {
			continue
		}
		from, okFrom := repOf[e.FromID]
		to, okTo := repOf[e.ToID]
		if !okFrom || !okTo || from == to {
			continue
		}
		disputes[from] = appendUnique(disputes[from], to)
		disputes[to] = appendUnique(disputes[to], from)
	}
	for i := range out {
		out[i].disputes = disputes[out[i].Claim.ID]
	}

	sortFindings(out)

	// Cap BEFORE resolving IDs to indices, and keep both sides of a disagreement
	// together. The naive order — truncate, then drop references past the cap — selects
	// for erasing exactly what §11.2 promises a reader will always see, because a
	// contradiction LOWERS confidence and the list is sorted by it.
	if max > 0 && len(out) > max {
		out = capKeepingPairs(out, max)
	}

	// Now that membership and order are both final, resolve IDs to indices.
	position := make(map[string]int, len(out))
	for i, f := range out {
		position[f.Claim.ID] = i
	}
	for i := range out {
		var idx []int
		for _, id := range out[i].disputes {
			if p, ok := position[id]; ok {
				idx = appendUnique(idx, p)
			}
		}
		sort.Ints(idx)
		out[i].Contradicts = idx
	}
	return out
}

// sortFindings ranks by derived confidence, then publisher breadth, then source breadth,
// then input position.
//
// Ordered by DERIVED confidence (§11.3), never by the extractor's self-report — a
// function that sorted by the latter under the same field name let an uncalibrated number
// decide which claims led the answer. Before the Verifier scores a session every derived
// confidence is 0, and assertion strength is the wrong tie-break, so ties fall back to
// breadth: mechanical, and it claims nothing.
func sortFindings(out []Finding) {
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Confidence != out[j].Confidence {
			return out[i].Confidence > out[j].Confidence
		}
		if out[i].Publishers != out[j].Publishers {
			return out[i].Publishers > out[j].Publishers
		}
		if out[i].breadth != out[j].breadth {
			return out[i].breadth > out[j].breadth
		}
		return out[i].order < out[j].order
	})
}

// capKeepingPairs truncates to max without splitting a disputed pair.
//
// Walks in rank order and admits a finding together with everything it disputes. A group
// that does not fit is skipped entirely and the walk continues, so the cap is still hard
// and a lower-ranked finding can take the freed slot.
func capKeepingPairs(out []Finding, max int) []Finding {
	byID := make(map[string]int, len(out))
	for i, f := range out {
		byID[f.Claim.ID] = i
	}

	kept := make(map[int]bool, max)
	var order []int
	for i := range out {
		if len(kept) >= max {
			break
		}
		if kept[i] {
			continue
		}
		group := []int{i}
		for _, id := range out[i].disputes {
			if j, ok := byID[id]; ok && !kept[j] && j != i {
				group = append(group, j)
			}
		}
		if len(kept)+len(group) > max {
			// Cannot show both sides. Skip rather than present a disputed claim as
			// settled; a later, smaller group may still fit.
			continue
		}
		for _, j := range group {
			kept[j] = true
			order = append(order, j)
		}
	}

	sort.Ints(order) // back into rank order
	res := make([]Finding, 0, len(order))
	for _, i := range order {
		res = append(res, out[i])
	}
	return res
}

// appendUnique appends v unless it is already present. Generic: the int and string
// versions were separate functions for no reason on Go 1.25.
func appendUnique[T comparable](xs []T, v T) []T {
	for _, x := range xs {
		if x == v {
			return xs
		}
	}
	return append(xs, v)
}

// Disagreements returns the pairs a report must disclose, most confident first.
//
// Each pair appears once. §11.3 already lowers both sides' confidence, but that is a
// number the prose never shows, and a reader looking at two cited sentences has no
// way to know the pipeline found them incompatible.
func Disagreements(findings []Finding) [][2]int {
	var out [][2]int
	seen := map[[2]int]bool{}
	for i, f := range findings {
		for _, j := range f.Contradicts {
			if j < 0 || j >= len(findings) {
				continue
			}
			key := [2]int{min(i, j), max(i, j)}
			if seen[key] {
				continue
			}
			seen[key] = true
			out = append(out, key)
		}
	}
	return out
}

// corroborationNote describes a finding's support for the trailing source list.
func corroborationNote(f Finding) string {
	var parts []string
	switch {
	case f.Publishers > 1:
		// Not "independent publishers": distinct domains are not evidence the
		// coverage is independent, only that it is not one site (task 7).
		parts = append(parts, fmt.Sprintf("%d publishers", f.Publishers))
	case len(f.Sources) > 1:
		// Several URLs, one publisher. Worth saying: it looks like corroboration
		// and is not, which is §11.3's inflation risk in a form a reader can see.
		parts = append(parts, fmt.Sprintf("%d pages from one publisher", len(f.Sources)))
	}
	if f.Superseded {
		parts = append(parts, "outdated")
	}
	if len(f.Contradicts) > 0 {
		parts = append(parts, "disputed")
	}
	if f.Claim != nil && f.Claim.Grounded != nil && !*f.Claim.Grounded {
		// The one flag a reader must not miss: the claim's own source, re-read,
		// does not support it (§11.5).
		//
		// Worded as a fragment, not a sentence. The earlier phrasing was a complete
		// clause — "source does NOT support this on re-reading" — and a 3B model
		// lifted it straight into the report body as the whole answer, asserting a
		// re-read verdict on claims that were never checked.
		parts = append(parts, "failed source re-read")
	}
	return strings.Join(parts, "; ")
}

// claims returns every claim in the finding's cluster.
//
// The representative is what the report quotes, but a citation's verified spans
// come from whichever claim cited that source — so a reader checking source [3]
// sees the quote that source actually supplied, not the representative's.
func (f Finding) claims() []*core.Claim {
	if f.cluster != nil {
		return f.cluster
	}
	if f.Claim == nil {
		return nil
	}
	return []*core.Claim{f.Claim}
}

package verifier

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/lajosdeme/mole/internal/core"
)

// Verification-driven follow-up leads (§11.4).
//
// A contradiction is the one finding worth spending more budget on: it means two
// sources disagree and the report cannot honestly assert either without saying so.
// A follow-up lead goes looking for what settles it.
//
// The cap has to arrive with the mechanism. §11.4's whole argument is that a
// per-row RecheckCount cannot bind, because a follow-up produces a NEW claim whose
// counter starts at zero — so lineage is carried on the lead, inherited by the
// claims it produces, and the cap reads the chain rather than the row. A cap with
// nothing incrementing it is the shape that left MaxLeads inert from M0 to M3.

// DefaultMaxVerifyDepth is §11.4's chain cap: a follow-up of a follow-up of a
// follow-up is drift, not diligence.
const DefaultMaxVerifyDepth = 3

// DefaultMaxFollowUpsPerRoot is §11.4's per-root cap, so one stubborn claim cannot
// consume the session.
//
// Counted across the whole session rather than per pass. A disagreement that
// survives three attempts to settle it is a real disagreement, and the honest thing
// is to report it as disputed rather than to keep paying for the same answer.
const DefaultMaxFollowUpsPerRoot = 3

// FollowUp is a lead the Verifier wants run, and why.
type FollowUp struct {
	Lead core.Lead
	// Because names the contradiction that prompted it, for the trace.
	Because string
}

// FollowUpOptions bounds lead generation.
type FollowUpOptions struct {
	SessionID string
	ActorType core.ActorType

	// MaxDepth is the chain cap. Zero takes DefaultMaxVerifyDepth.
	MaxDepth int
	// MaxPerRoot bounds follow-ups per root claim. Zero takes the default.
	MaxPerRoot int
	// StalenessGap must match the one Edges uses, or a pair the graph resolved as
	// staleness is still researched as a live disagreement. Zero takes the default.
	StalenessGap time.Duration

	// MaxTotal bounds one pass's output, so a graph full of disagreement cannot
	// queue more work than the session can run. Zero means unbounded here — the
	// budget ceiling still applies downstream.
	MaxTotal int

	// ExistingPerRoot is how many follow-ups each root already has, so the
	// per-root cap counts the session rather than the pass.
	ExistingPerRoot map[string]int
}

// FollowUps proposes leads to settle the contradictions in a set of verdicts.
//
// Deterministic: verdicts are considered in canonical pair order and the output is
// sorted, so a cassette replays and two runs of one session queue the same work.
// Callers that enforce a per-pass budget want FollowUpsWithDeferred: this
// returns only what fits and says nothing about the rest, which is how 28 of
// one session's 32 contradictions came to be dropped without a trace.
func FollowUps(verdicts []Judged, opts FollowUpOptions) []FollowUp {
	queued, _ := FollowUpsWithDeferred(verdicts, opts)
	return queued
}

// FollowUpsWithDeferred proposes leads and says which of them the pass's budget
// could not take.
//
// Two returns because the difference matters to the caller. The first set is
// queued now. The second is OWED: contradictions that earned a follow-up and
// lost the per-pass cap.
//
// They used to be dropped. FollowUps was called with one pass's verdicts,
// returned the first MaxTotal in canonical order, and said nothing about the
// rest — and a later pass never saw them again, because Verifier.Run judges
// only unverified claims and skips any pair that already carries an edge. The
// cap's own comment said follow-ups "accumulate across passes anyway". Nothing
// accumulated. One Deep run found 32 contradictions, followed up 4, discarded
// 28, and then reported that the planner had nothing left to add — which was
// true, and was the problem.
//
// Deferring is not queueing later by another name: a deferred lead is real
// work the session still owes, and §11.4's caps apply to it exactly as they do
// to a queued one. A contradiction excluded by depth, staleness or the
// per-root cap is excluded from BOTH, because those rules say it should not be
// researched at all.
func FollowUpsWithDeferred(verdicts []Judged, opts FollowUpOptions) (queued, deferred []FollowUp) {
	maxDepth := opts.MaxDepth
	if maxDepth <= 0 {
		maxDepth = DefaultMaxVerifyDepth
	}
	maxPerRoot := opts.MaxPerRoot
	if maxPerRoot <= 0 {
		maxPerRoot = DefaultMaxFollowUpsPerRoot
	}
	actor := opts.ActorType
	if actor == "" {
		actor = core.ActorWeb
	}
	gap := opts.StalenessGap
	if gap <= 0 {
		gap = DefaultStalenessGap
	}

	perRoot := map[string]int{}
	for k, v := range opts.ExistingPerRoot {
		perRoot[k] = v
	}

	ordered := append([]Judged(nil), verdicts...)
	sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].Key() < ordered[j].Key() })

	seen := map[string]bool{}
	var out []FollowUp
	for _, v := range ordered {
		if v.Relation != RelContradicts {
			continue
		}
		// A superseded pair is not a live disagreement — §11.2 already resolved it
		// as staleness, and researching it further buys a second answer to a
		// question the dates settled.
		if _, _, stale := stalePair(v.Pair, gap); stale {
			continue
		}

		// The chain the follow-up would join. Both claims may already sit on
		// chains; take the deeper, so a contradiction between two follow-up claims
		// cannot restart the count at the shallower one's depth.
		root, depth := chainOf(v.Pair)
		if depth+1 >= maxDepth {
			continue
		}
		if perRoot[root] >= maxPerRoot {
			continue
		}
		query := disambiguationQuery(v.Pair)
		// One lead per root per pass. Two contradictions about the same claim would
		// otherwise produce two nearly identical searches.
		if seen[root] {
			continue
		}
		seen[root] = true
		perRoot[root]++

		rootID := root
		out = append(out, FollowUp{
			Lead: core.Lead{
				SessionID:   opts.SessionID,
				ActorType:   actor,
				Query:       query,
				RootClaimID: &rootID,
				VerifyDepth: depth + 1,
				// Priority above an ordinary lead: a contradiction blocks an honest
				// answer, while another sub-question merely adds to one.
				Priority: 1,
			},
			Because: fmt.Sprintf("%.60q contradicts %.60q", v.A.Text, v.B.Text),
		})
	}
	// The budget cuts the list; it does not decide the list. Everything above
	// this line is a question of whether a contradiction DESERVES a follow-up,
	// and everything below is a question of how many fit now.
	if opts.MaxTotal > 0 && len(out) > opts.MaxTotal {
		return out[:opts.MaxTotal], out[opts.MaxTotal:]
	}
	return out, nil
}

// chainOf picks the lineage a follow-up should join.
//
// The deeper of the two claims' chains. Taking the shallower would let a
// contradiction between two already-deep claims reset the depth count, which is the
// loop §11.4's cap exists to close.
func chainOf(p Pair) (root string, depth int) {
	ra, da := rootOf(p.A), p.A.VerifyDepth
	rb, db := rootOf(p.B), p.B.VerifyDepth
	if db > da {
		return rb, db
	}
	if da > db {
		return ra, da
	}
	// Equal depth: lowest root ID, so the choice is stable across runs.
	if rb < ra {
		return rb, db
	}
	return ra, da
}

func rootOf(c *core.Claim) string {
	if c.RootClaimID != "" {
		return c.RootClaimID
	}
	return c.ID
}

// disambiguationQuery turns a disagreement into something searchable.
//
// Built mechanically from the two claim texts rather than by asking a model. Two
// reasons: a query is not worth a call when the claims already say what is in
// dispute, and claim text is page-derived, so generating the query from it in a
// model context would put untrusted text into a position that decides what gets
// searched next.
func disambiguationQuery(p Pair) string {
	a := strings.Join(strings.Fields(p.A.Text), " ")
	b := strings.Join(strings.Fields(p.B.Text), " ")
	q := "evidence resolving whether " + trimTo(a, 120) + " or instead " + trimTo(b, 120)
	return trimTo(q, 300)
}

func trimTo(s string, n int) string {
	if len(s) <= n {
		return strings.TrimRight(s, " .")
	}
	cut := s[:n]
	// Break on a word boundary so the query does not end mid-token.
	if i := strings.LastIndexByte(cut, ' '); i > n/2 {
		cut = cut[:i]
	}
	return strings.TrimRight(cut, " .,;:")
}

package eval

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/verifier"
	"log/slog"
)

// Adjudicator test sets.
//
// §11.2's edge inference is the stage whose errors are hardest to see: a bad extractor
// produces no claims, a bad judge produces a confident graph of wrong edges and everything
// downstream reads it as fact. A live run produced 16 contradictions out of 37 edges, and
// the model's own rationales described the pairs as being about different topics — which
// is "unrelated". Every false positive spent a follow-up lead, docked two claims'
// confidence, and put a disagreement in the report that was not there.
//
// None of the existing metrics can see that. "disagreement rate" counts what was FOUND and
// says so in its own comment; measuring what is CORRECT needs someone to say what the right
// answer was.
//
// So: dump the pairs a session judged, with both claim texts, for a person to label. After
// that, any prompt or model change is measurable in seconds, offline, against real pairs a
// real run produced — rather than against questions someone invented to be measured.

// LabelledPair is one judgement, before and after a human looks at it.
type LabelledPair struct {
	// Pair is the canonical key, stable across dumps of the same session.
	Pair string `json:"pair"`

	A string `json:"a"`
	B string `json:"b"`

	SourceA string `json:"source_a,omitempty"`
	SourceB string `json:"source_b,omitempty"`

	QuoteA string `json:"quote_a,omitempty"`
	QuoteB string `json:"quote_b,omitempty"`

	// Model is what the adjudicator said. "unrelated" means no edge was stored, which
	// covers both "judged unrelated" and "never reached" — Judged distinguishes them.
	Model string `json:"model"`
	// Baseline is what the STORED graph said, when this set came from re-judging with a
	// different model. Kept so a comparison shows both verdicts on one line.
	Baseline string `json:"baseline,omitempty"`
	// Why is the model's own rationale, which is often the clearest evidence that a
	// verdict is wrong.
	Why string `json:"why,omitempty"`

	// Judged is false when one of the claims was never verified, so the pair was never
	// put to the model at all. Scoring skips those: they say nothing about the judge.
	Judged bool `json:"judged"`

	// Score is what the lexical retriever thought of the pair, and Rank is its
	// position in the target claim's candidate list (1-based). Zero and zero mean it
	// shares no content word and was never a candidate at all.
	//
	// Recorded so a missed contradiction says what to fix: score 0 is the lexical
	// measure being the wrong primitive, and a rank past DefaultMaxCandidates is the
	// cap. Only filled on an exhaustive dump, which is the only one measuring this.
	Score float64 `json:"score,omitempty"`
	Rank  int     `json:"rank,omitempty"`

	// Label is the truth, filled in by hand. Empty means unlabelled.
	Label string `json:"label"`
	// Note is for the labeller.
	Note string `json:"note,omitempty"`
}

// PairSet is a dumped set awaiting labels.
type PairSet struct {
	Session string         `json:"session"`
	Model   string         `json:"adjudicator,omitempty"`
	Pairs   []LabelledPair `json:"pairs"`
}

// DumpOptions filter what is written.
type DumpOptions struct {
	// Kinds restricts the dump to pairs the model gave these verdicts. Empty means every
	// pair that carries an edge.
	Kinds []string
	// All includes pairs with no edge — the ones the model called unrelated.
	//
	// Off by default because they dominate: a 25-claim session produces a few dozen
	// edges and several hundred candidate pairs, and a labelling task nobody finishes
	// measures nothing. Turn it on to measure RECALL — whether real disagreements were
	// missed — once precision is known.
	All bool
	// MaxCandidatesPerClaim mirrors the Verifier's retrieval cap, so the candidate space
	// is regenerated exactly as the run saw it.
	MaxCandidatesPerClaim int

	// Exhaustive pairs every claim with every other, retrieval bypassed, and records
	// each pair's lexical score and rank.
	//
	// This is the denominator for PIPELINE recall. --all measures the judge: it adds
	// the no-edge pairs from among the retriever's candidates. A contradiction the
	// retriever never formed is in neither set, so it is missed by nothing and counted
	// by nothing — 73% of the pair space on §14.2's corpus. Implies All.
	Exhaustive bool

	// MaxPairs refuses an exhaustive dump larger than this. Zero takes
	// DefaultMaxDumpPairs.
	MaxPairs int
}

// DefaultMaxDumpPairs bounds an exhaustive dump.
//
// n(n-1)/2 grows fast enough that a real session is five figures — 500 claims is
// 124,750 pairs — and a labelling task nobody finishes measures nothing, which is the
// argument the All flag already makes one order of magnitude earlier. Sized so the
// corpus's smallest sessions fit whole: 18 claims is 153 pairs, 46 is 1,035.
const DefaultMaxDumpPairs = 1200

// DumpPairs regenerates a session's candidate pairs and pairs them with what the
// adjudicator said.
//
// Retrieval is deterministic and costs nothing, so the candidate space is reproduced
// rather than stored — which also means a dump taken today reflects today's retriever.
func DumpPairs(ctx context.Context, st store.Store, sessionID string, opts DumpOptions) (*PairSet, error) {
	var (
		claims []*core.Claim
		edges  []*core.ClaimEdge
	)
	if err := st.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		if claims, err = q.ListClaims(ctx, sessionID, 0); err != nil {
			return err
		}
		edges, err = q.ListEdges(ctx, sessionID, 0)
		return err
	}); err != nil {
		return nil, err
	}
	if len(claims) == 0 {
		return nil, fmt.Errorf("eval: session %s has no claims", sessionID)
	}

	byID := make(map[string]*core.Claim, len(claims))
	for _, c := range claims {
		byID[c.ID] = c
	}

	// What the model said, keyed the way pairs are keyed.
	type verdict struct{ kind, why string }
	said := make(map[string]verdict, len(edges))
	for _, e := range edges {
		a, b := e.FromID, e.ToID
		if a > b {
			a, b = b, a
		}
		said[a+"|"+b] = verdict{string(e.Kind), e.Rationale}
	}

	// Every pair that CARRIES AN EDGE, regardless of whether retrieval would still
	// surface it.
	//
	// Regenerating the candidate space does not reproduce what the run judged. The
	// Verifier works incrementally (§11.1): when a claim was new it was retrieved against
	// the handful of claims that existed then, and its top-8 against the finished set of
	// 25 is a different eight. Measured on a real session: regeneration found 11 of the
	// 16 contradictions the run actually produced, so a test set built from it would have
	// silently excluded five real judgements — and they are judgements, which is the only
	// thing this file is for.
	//
	// Regeneration is still used for the no-edge pairs, since there is no other record of
	// them; that set is approximate and only matters when measuring recall.
	all := make([]verifier.Pair, 0, len(edges))
	for _, e := range edges {
		a, b := byID[e.FromID], byID[e.ToID]
		if a == nil || b == nil {
			continue
		}
		if a.ID > b.ID {
			a, b = b, a
		}
		all = append(all, verifier.Pair{A: a, B: b})
	}

	// Refused before any work: the point of a cap is that an oversize dump never
	// happens, not that it happens and is then discarded.
	if opts.Exhaustive {
		limit := opts.MaxPairs
		if limit <= 0 {
			limit = DefaultMaxDumpPairs
		}
		if n := len(claims) * (len(claims) - 1) / 2; n > limit {
			return nil, fmt.Errorf("eval: an exhaustive dump of session %s is %d pair(s), "+
				"over the limit of %d — raise it deliberately or dump a smaller session",
				sessionID, n, limit)
		}
	}

	// Scores and ranks the lexical retriever would have given, computed once per claim
	// so an exhaustive dump can say why each pair was or was not surfaced.
	type placing struct {
		score float64
		rank  int
	}
	placings := map[string]placing{}
	if opts.Exhaustive {
		lex := verifier.LexicalRetriever{}
		for _, target := range claims {
			scores := lex.Scores(target, claims)
			ranked := make([]string, 0, len(scores))
			for id := range scores {
				ranked = append(ranked, id)
			}
			sort.SliceStable(ranked, func(i, j int) bool {
				if scores[ranked[i]] != scores[ranked[j]] {
					return scores[ranked[i]] > scores[ranked[j]]
				}
				return ranked[i] < ranked[j]
			})
			for i, id := range ranked {
				// Canonical order, because Pair.Key is a raw "A|B" with no
				// normalisation and Pair itself documents A.ID < B.ID.
				//
				// Without the swap each unordered pair was written under two
				// different keys — one per end — so the "keep the better rank"
				// comparison below only ever compared an end against itself, and
				// the output loop, which looks up the canonical key, silently
				// discarded half the writes. A pair ranked 40 from one claim and 1
				// from the other was reported as 40: "the cap is the constraint"
				// for a pair the cap never blocked, which is the exact
				// misdiagnosis Rank was added to prevent.
				a, b := target, byID[id]
				if a.ID > b.ID {
					a, b = b, a
				}
				key := verifier.Pair{A: a, B: b}.Key()
				// Both ends rank the pair. Keep the better one: a pair is surfaced if
				// EITHER claim put it in its candidate list.
				if cur, ok := placings[key]; ok && cur.rank <= i+1 {
					continue
				}
				placings[key] = placing{score: scores[id], rank: i + 1}
			}
		}
	}

	if opts.All || opts.Exhaustive {
		r := verifier.Retriever(verifier.LexicalRetriever{})
		if opts.Exhaustive {
			r = verifier.ExhaustiveRetriever{}
		}
		pairs, free, err := verifier.CandidatePairs(ctx, r,
			claims, claims, opts.MaxCandidatesPerClaim, nil)
		if err != nil {
			return nil, err
		}
		all = append(all, pairs...)
		// Mechanically-decided pairs count too: they are judgements the pipeline made,
		// and a wrong one is as damaging as a wrong model verdict.
		for _, f := range free {
			all = append(all, f.Pair)
		}
	}

	keep := map[string]bool{}
	for _, k := range opts.Kinds {
		keep[strings.ToLower(strings.TrimSpace(k))] = true
	}

	out := &PairSet{Session: sessionID}
	seen := map[string]bool{}
	for _, p := range all {
		key := p.Key()
		if seen[key] {
			continue
		}
		seen[key] = true

		v, hasEdge := said[key]
		kind := v.kind
		if !hasEdge {
			kind = "unrelated"
		}
		if len(keep) > 0 && !keep[kind] {
			continue
		}
		out.Pairs = append(out.Pairs, LabelledPair{
			Pair: key,
			A:    oneLine(p.A.Text), B: oneLine(p.B.Text),
			SourceA: p.A.Source, SourceB: p.B.Source,
			QuoteA: p.A.Quote, QuoteB: p.B.Quote,
			Model: kind, Why: v.why,
			// Both ends verified means the pair was reachable by the pass that ran.
			Judged: p.A.VerifiedAt != nil && p.B.VerifiedAt != nil,
			Score:  placings[key].score, Rank: placings[key].rank,
		})
	}
	sort.SliceStable(out.Pairs, func(i, j int) bool { return out.Pairs[i].Pair < out.Pairs[j].Pair })
	return out, nil
}

func oneLine(s string) string { return strings.Join(strings.Fields(s), " ") }

// LoadPairs reads a labelled set.
func LoadPairs(path string) (*PairSet, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var ps PairSet
	if err := json.Unmarshal(raw, &ps); err != nil {
		return nil, fmt.Errorf("eval: parse %s: %w", path, err)
	}
	return &ps, nil
}

// WritePairs writes a set for labelling.
func WritePairs(path string, ps *PairSet) error {
	raw, err := json.MarshalIndent(ps, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(raw, '\n'), 0o644)
}

// PairScore is how the adjudicator did against the labels.
type PairScore struct {
	Labelled   int `json:"labelled"`
	Unlabelled int `json:"unlabelled"`
	// Skipped is pairs that were never put to the model.
	Skipped int `json:"skipped"`
	Correct int `json:"correct"`

	// Confusion[modelVerdict][trueLabel] is how often the model said one thing and the
	// truth was another.
	Confusion map[string]map[string]int `json:"confusion"`

	// Precision and Recall per relation. Precision is what matters for a judge that
	// over-reports: of the pairs it called X, how many were X.
	Precision map[string]float64 `json:"precision"`
	Recall    map[string]float64 `json:"recall"`

	// CorrectEffect counts verdicts that do the right thing to the graph even when
	// they used the wrong word, on the same reasoning as Agreement.SameEffect.
	//
	// Measured on claude-haiku-4-5 over 37 labelled pairs: 76% raw, 97% by effect.
	// Eight of its nine errors were calling an unrelated pair "supports" or
	// "refines", which builds an edge nothing reads. The ninth was a false
	// contradiction, which costs a confidence penalty and a research lead.
	CorrectEffect int `json:"correct_effect"`
}

// EffectAccuracy is accuracy over what the graph will do, ignoring vocabulary.
func (s PairScore) EffectAccuracy() float64 {
	if s.Labelled == 0 {
		return 0
	}
	return float64(s.CorrectEffect) / float64(s.Labelled)
}

// ScorePairs compares the model's verdicts against the labels.
//
// Unlabelled pairs are counted and excluded, never guessed at — a score computed over
// whichever pairs someone got round to labelling, reported as the score, is how a test set
// starts lying.
func ScorePairs(ps *PairSet) PairScore {
	s := PairScore{
		Confusion: map[string]map[string]int{},
		Precision: map[string]float64{},
		Recall:    map[string]float64{},
	}
	said := map[string]int{}
	truth := map[string]int{}
	hit := map[string]int{}

	for _, p := range ps.Pairs {
		label := strings.ToLower(strings.TrimSpace(p.Label))
		if label == "" {
			s.Unlabelled++
			continue
		}
		if !p.Judged {
			// Never put to the model, so it says nothing about the judge.
			s.Skipped++
			continue
		}
		// Both sides NORMALIZED before comparison, because both carry the retired
		// vocabulary. `pairs dump --all` writes "unrelated" for every pair the graph
		// holds no edge for, and a labeller writes "neither" from the three words the
		// prompt offers — the same verdict under two names, scored as an error 33
		// times in 149 on the first real labelled set.
		//
		// Relation.Normalize exists for exactly this and says so ("nothing past the
		// boundary has to know the old vocabulary existed"); the scorer was the one
		// place downstream that skipped it, so the measurement built to check the
		// judge was penalising it for a rename.
		model := normalizeVerdict(p.Model)
		label = normalizeVerdict(label)

		s.Labelled++
		said[model]++
		truth[label]++
		if s.Confusion[model] == nil {
			s.Confusion[model] = map[string]int{}
		}
		s.Confusion[model][label]++
		if model == label {
			s.Correct++
			hit[label]++
		}
		if verifier.Relation(model).EffectOf() == verifier.Relation(label).EffectOf() {
			s.CorrectEffect++
		}
	}

	for k, n := range said {
		if n > 0 {
			s.Precision[k] = float64(hit[k]) / float64(n)
		}
	}
	for k, n := range truth {
		if n > 0 {
			s.Recall[k] = float64(hit[k]) / float64(n)
		}
	}
	return s
}

// normalizeVerdict folds a relation onto the live vocabulary for comparison.
func normalizeVerdict(s string) string {
	return string(verifier.Relation(strings.ToLower(strings.TrimSpace(s))).Normalize())
}

// Accuracy is the fraction of labelled, judged pairs the model got right.
func (s PairScore) Accuracy() float64 {
	if s.Labelled == 0 {
		return 0
	}
	return float64(s.Correct) / float64(s.Labelled)
}

// pairsFromEdges rebuilds the pairs a session actually judged.
//
// From the edges, not from re-running retrieval. The Verifier works incrementally (§11.1),
// so a claim retrieved against the seven claims that existed when it was new has a
// different top-8 than against the finished twenty-five: regeneration found 11 of the 16
// contradictions one real session produced. These are judgements, and a judgement that is
// no longer a candidate is still a judgement.
func pairsFromEdges(claims []*core.Claim, edges []*core.ClaimEdge) ([]verifier.Pair, map[string]string, map[string]string) {
	byID := make(map[string]*core.Claim, len(claims))
	for _, c := range claims {
		byID[c.ID] = c
	}

	var pairs []verifier.Pair
	kind := map[string]string{}
	why := map[string]string{}
	seen := map[string]bool{}

	for _, e := range edges {
		a, b := byID[e.FromID], byID[e.ToID]
		if a == nil || b == nil {
			continue
		}
		if a.ID > b.ID {
			a, b = b, a
		}
		key := a.ID + "|" + b.ID
		if seen[key] {
			continue
		}
		seen[key] = true
		pairs = append(pairs, verifier.Pair{A: a, B: b})
		kind[key] = string(e.Kind)
		why[key] = e.Rationale
	}
	sort.SliceStable(pairs, func(i, j int) bool { return pairs[i].Key() < pairs[j].Key() })
	return pairs, kind, why
}

// JudgeOptions configure a re-judge.
type JudgeOptions struct {
	Model     string
	BatchSize int
	// Kinds restricts which stored verdicts get re-judged. Empty means all of them.
	Kinds []string
	// Labels carries labels across from an earlier set, matched by pair key.
	//
	// The reason the whole harness is worth having: label once, and every future model
	// or prompt is scored against the same judgements rather than needing the work done
	// again.
	Labels map[string]string
}

// JudgeSession re-adjudicates a session's pairs with a different model.
//
// Reads the store and writes nothing to it. The stored graph is the thing being compared
// against, so a tool that overwrote it while measuring it would have nothing left to
// measure.
func JudgeSession(ctx context.Context, st store.Store, p llm.Provider, sessionID string, opts JudgeOptions, log *slog.Logger) (*PairSet, error) {
	var (
		claims []*core.Claim
		edges  []*core.ClaimEdge
	)
	if err := st.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		if claims, err = q.ListClaims(ctx, sessionID, 0); err != nil {
			return err
		}
		edges, err = q.ListEdges(ctx, sessionID, 0)
		return err
	}); err != nil {
		return nil, err
	}

	pairs, storedKind, _ := pairsFromEdges(claims, edges)
	if len(opts.Kinds) > 0 {
		keep := map[string]bool{}
		for _, k := range opts.Kinds {
			keep[strings.ToLower(strings.TrimSpace(k))] = true
		}
		var filtered []verifier.Pair
		for _, pr := range pairs {
			if keep[storedKind[pr.Key()]] {
				filtered = append(filtered, pr)
			}
		}
		pairs = filtered
	}
	if len(pairs) == 0 {
		return nil, fmt.Errorf("eval: session %s has no judged pairs matching the filter", sessionID)
	}

	judged, unjudged := verifier.JudgePairs(ctx, p, opts.Model, pairs, opts.BatchSize, log)

	verdict := make(map[string]verifier.Judged, len(judged))
	for _, j := range judged {
		verdict[j.Key()] = j
	}
	missed := map[string]bool{}
	for _, pr := range unjudged {
		missed[pr.Key()] = true
	}

	out := &PairSet{Session: sessionID, Model: opts.Model}
	for _, pr := range pairs {
		key := pr.Key()
		lp := LabelledPair{
			Pair: key,
			A:    oneLine(pr.A.Text), B: oneLine(pr.B.Text),
			SourceA: pr.A.Source, SourceB: pr.B.Source,
			QuoteA: pr.A.Quote, QuoteB: pr.B.Quote,
			Baseline: storedKind[key],
			Label:    opts.Labels[key],
		}
		if j, ok := verdict[key]; ok {
			lp.Model = string(j.Relation)
			lp.Why = j.Rationale
			lp.Judged = true
		} else {
			// It answered nothing for this pair. Not "unrelated" — recording a
			// non-answer as a verdict would credit the model with a judgement it did
			// not make, and scoring skips unjudged pairs precisely so it cannot.
			lp.Model = ""
			lp.Judged = false
			if missed[key] {
				lp.Note = "the model returned no verdict for this pair"
			}
		}
		out.Pairs = append(out.Pairs, lp)
	}
	return out, nil
}

// LabelsFrom extracts the labels already applied to a set, for carrying forward.
func LabelsFrom(ps *PairSet) map[string]string {
	out := map[string]string{}
	for _, p := range ps.Pairs {
		if l := strings.TrimSpace(p.Label); l != "" {
			out[p.Pair] = l
		}
	}
	return out
}

// Agreement is how often two verdict sets say the same thing.
type Agreement struct {
	Compared int `json:"compared"`
	Same     int `json:"same"`
	// Unanswered counts pairs one side or the other did not judge.
	Unanswered int `json:"unanswered"`
	// Confusion[a][b] is how often the first set said a and the second said b.
	Confusion map[string]map[string]int `json:"confusion"`

	// SameEffect counts pairs where the two verdicts do the same thing to the graph,
	// whether or not they used the same word. See verifier.Relation.EffectOf: five
	// relations collapse to three effects, because "supports", "refines" and
	// "unrelated" are all read by nothing.
	//
	// This is the number to judge a judge by, and it is not a softer version of
	// Rate. Measured on claude-haiku-4-5 over 37 pairs judged twice: 84% raw, 97%
	// by effect. Five of the six disagreements were supports/refines/unrelated
	// shuffles that change nothing downstream; one was contradicts-vs-supports,
	// which changes a confidence score and queues a research lead. Acting on 84%
	// would mean rewriting a prompt to chase six problems when there was one.
	SameEffect int `json:"same_effect"`
}

// FileJudgeOptions configure judging a labelled pair file.
type FileJudgeOptions struct {
	Model     string
	BatchSize int
	// Confirm judges twice and keeps a contradicts or duplicate_of verdict only
	// when both runs agree.
	Confirm bool
}

// JudgePairFile re-judges every pair in a labelled set with no store and no
// session, carrying each pair's label through unchanged.
func JudgePairFile(ctx context.Context, p llm.Provider, ps *PairSet, opts FileJudgeOptions, log *slog.Logger) (*PairSet, error) {
	pairs := make([]verifier.Pair, 0, len(ps.Pairs))
	for _, lp := range ps.Pairs {
		a, b, ok := strings.Cut(lp.Pair, "|")
		if !ok {
			return nil, fmt.Errorf("eval: pair key %q is not \"idA|idB\"", lp.Pair)
		}
		pairs = append(pairs, verifier.Pair{
			A: &core.Claim{ID: a, Text: lp.A, Quote: lp.QuoteA, Source: lp.SourceA},
			B: &core.Claim{ID: b, Text: lp.B, Quote: lp.QuoteB, Source: lp.SourceB},
		})
	}

	judged, _ := verifier.JudgePairs(ctx, p, opts.Model, pairs, opts.BatchSize, log)
	if opts.Confirm {
		judged = confirmPositives(ctx, p, opts.Model, judged, opts.BatchSize, log)
	}

	verdict := make(map[string]verifier.Judged, len(judged))
	for _, j := range judged {
		verdict[j.Key()] = j
	}

	out := &PairSet{Session: ps.Session, Model: opts.Model}
	for _, lp := range ps.Pairs {
		if j, ok := verdict[lp.Pair]; ok {
			lp.Baseline = lp.Model
			lp.Model = string(j.Relation)
			lp.Why = j.Rationale
			lp.Judged = true
		} else {
			lp.Model = ""
			lp.Judged = false
		}
		out.Pairs = append(out.Pairs, lp)
	}
	return out, nil
}

// confirmPositives re-judges contradicts and duplicate_of verdicts, keeping
// only the ones a second judgement agrees with and withholding the rest.
func confirmPositives(ctx context.Context, p llm.Provider, model string, judged []verifier.Judged, batchSize int, log *slog.Logger) []verifier.Judged {
	var positives []verifier.Pair
	for _, j := range judged {
		if j.Relation.EffectOf() != verifier.EffectInert {
			positives = append(positives, j.Pair)
		}
	}
	if len(positives) == 0 {
		return judged
	}
	second, _ := verifier.JudgePairs(ctx, p, model, positives, batchSize, log)
	again := make(map[string]verifier.Relation, len(second))
	for _, j := range second {
		again[j.Key()] = j.Relation.Normalize()
	}

	out := make([]verifier.Judged, 0, len(judged))
	for _, j := range judged {
		if j.Relation.EffectOf() == verifier.EffectInert {
			out = append(out, j)
			continue
		}
		got, asked := again[j.Key()]
		if !asked || got == j.Relation.Normalize() {
			out = append(out, j)
			continue
		}
		j.Relation = verifier.RelNeither
		j.Rationale = "withheld: a second judgement did not agree (" + string(got) + ")"
		out = append(out, j)
	}
	return out
}

func (a Agreement) Rate() float64 {
	if a.Compared == 0 {
		return 0
	}
	return float64(a.Same) / float64(a.Compared)
}

// EffectRate is agreement on what the graph will do, ignoring vocabulary.
//
// Always at least Rate: identical verdicts have identical effects.
func (a Agreement) EffectRate() float64 {
	if a.Compared == 0 {
		return 0
	}
	return float64(a.SameEffect) / float64(a.Compared)
}

// Compare two verdict sets over the pairs both judged.
//
// The cheapest useful measurement of a judge, and the only one that needs no labels: run
// the SAME model over the SAME pairs twice and see how often it agrees with itself. A judge
// that does not is not measuring anything, and no quantity of labelling will fix it —
// qwen2.5:3b re-judging sixteen of its own contradictions kept three of them, calling seven
// "unrelated" and four "supports".
//
// Self-consistency is a ceiling, not a score: a judge cannot be more accurate than it is
// reproducible. Screening on it first is far cheaper than labelling.
func CompareVerdicts(a, b *PairSet) Agreement {
	out := Agreement{Confusion: map[string]map[string]int{}}
	other := make(map[string]LabelledPair, len(b.Pairs))
	for _, p := range b.Pairs {
		other[p.Pair] = p
	}

	for _, p := range a.Pairs {
		q, ok := other[p.Pair]
		if !ok {
			continue
		}
		// A pair neither side answered says nothing about agreement, and counting a
		// mutual non-answer as agreement is how a broken judge scores perfectly.
		if !p.Judged || !q.Judged || p.Model == "" || q.Model == "" {
			out.Unanswered++
			continue
		}
		out.Compared++
		if out.Confusion[p.Model] == nil {
			out.Confusion[p.Model] = map[string]int{}
		}
		out.Confusion[p.Model][q.Model]++
		if p.Model == q.Model {
			out.Same++
		}
		if verifier.Relation(p.Model).EffectOf() == verifier.Relation(q.Model).EffectOf() {
			out.SameEffect++
		}
	}
	return out
}

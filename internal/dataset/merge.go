package dataset

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"unicode"
)

// Cross-source merge (M9, §13).
//
// Whether two rows are the same company is a judgement, not a fact, so the
// thresholds below are calibrated rather than chosen: merge_test.go builds
// datasets whose duplicate structure is known and reports pairwise precision
// and recall, and those numbers are in `mole eval`.
//
// Three choices:
//
// Multi-pass blocking, because comparing every pair is quadratic over thousands
// of rows and a single block key loses true matches at its edges.
//
// Complete linkage, not transitive closure. A row joins a cluster only if it
// matches EVERY member — union-find would chain "Acme Ltd", "Acme" and "Acme
// Foods" into one company.
//
// Conflicts preserved, never resolved. Two sources giving one company two
// revenues is §11's contradiction problem with a column header, and the merge
// records both.

// Options tune the merge.
type Options struct {
	// Threshold is the similarity a pair of key values must reach.
	//
	// Calibrated, not chosen: see TestMergeQualityOnKnownDuplicates, which
	// reports precision and recall across a range and fails if the default
	// drops below what is recorded there.
	Threshold float64
	// MaxRows bounds the input. A merge is at worst quadratic within a block, so
	// an unbounded row set is an unbounded run.
	MaxRows int
}

// Defaults for the merge.
const (
	// DefaultThreshold sits in the gap the token measure opens up: on the
	// ground-truth set every true pair scores 0.67 or above and every false one
	// 0.50 or below, so anything in between separates them. The test prints the
	// curve and fails if the recorded precision and recall regress.
	//
	// It is a gap rather than a knife edge because the measure counts words. Two
	// names sharing one word out of three cannot reach it; two names differing
	// only by a typo or a reordering cannot fall below it.
	DefaultThreshold = 0.60
	DefaultMaxRows   = 20000
)

func (o Options) withDefaults() Options {
	if o.Threshold <= 0 {
		o.Threshold = DefaultThreshold
	}
	if o.MaxRows <= 0 {
		o.MaxRows = DefaultMaxRows
	}
	return o
}

// Merge folds rows describing the same entity together.
//
// Deterministic: the input is sorted before clustering, so the same rows in a
// different order produce the same dataset. §14.1 keys cassettes on request
// bodies and a report is generated from this, so an order-dependent merge would
// make a replay a coin toss.
func Merge(schema Schema, rows []Row, opts Options) Dataset {
	return MergeContext(context.Background(), schema, rows, opts)
}

// MergeContext is Merge, interruptible.
//
// Clustering is quadratic within a block and a block can be large, so a big row
// set is minutes of uninterruptible work — measured at 66 seconds for 20,000 keys
// sharing a prefix. It runs at session finish, on every `mole dataset` read and
// inside `mole eval`, so a caller that has given up needs to be able to say so.
func MergeContext(ctx context.Context, schema Schema, rows []Row, opts Options) Dataset {
	opts = opts.withDefaults()
	out := Dataset{Schema: schema, Extracted: len(rows)}

	if len(rows) > opts.MaxRows {
		out.Notes = append(out.Notes, describeTruncation(len(rows), opts.MaxRows))
		rows = rows[:opts.MaxRows]
	}

	keyed := make([]keyedRow, 0, len(rows))
	for _, r := range rows {
		// Counted here because this is the only place that sees every extracted
		// row: after folding, a row's evidence is a per-source quote and the
		// count of inputs that had none is unrecoverable. §14.3's row-integrity
		// metric is a hard regression, so it needs the number rather than an
		// approximation of it.
		if r.Quote == "" || r.Source == "" {
			out.Unquoted++
		}
		k := schema.keyOf(r)
		if k.normalised == "" {
			// No usable key. Extraction already refuses these, so reaching here
			// means a row arrived from somewhere else — a stored dataset from an
			// older schema, say. Counted rather than dropped in silence.
			continue
		}
		keyed = append(keyed, keyedRow{row: r, key: k})
	}
	if n := len(rows) - len(keyed); n > 0 {
		out.Notes = append(out.Notes, describeUnkeyed(n))
	}

	// Sorted by key then source, so clustering order does not depend on
	// extraction order.
	sort.SliceStable(keyed, func(i, j int) bool {
		if keyed[i].key.normalised != keyed[j].key.normalised {
			return keyed[i].key.normalised < keyed[j].key.normalised
		}
		return keyed[i].row.Source < keyed[j].row.Source
	})

	clusters, interrupted := cluster(ctx, keyed, opts.Threshold)
	if interrupted {
		out.Notes = append(out.Notes, "the merge was interrupted; the rows below are "+
			"what it had grouped, and duplicates may remain")
	}
	for _, c := range clusters {
		out.Rows = append(out.Rows, schema.fold(c))
	}
	return out
}

func describeTruncation(got, limit int) string {
	return fmt.Sprintf("the merge was given more rows than it will compare "+
		"(%d against a limit of %d); the excess was dropped", got, limit)
}

func describeUnkeyed(n int) string {
	return fmt.Sprintf("%d row(s) had no usable key value and could not be merged", n)
}

// -----------------------------------------------------------------------------
// Keys
// -----------------------------------------------------------------------------

type key struct {
	// normalised is what similarity is computed on: lower-cased, folded,
	// punctuation removed, trailing legal-form words taken off.
	normalised string
	// families are the legal forms that were taken off, canonicalised — {"ltd"}
	// for both "Ltd" and "Limited". Two names that are identical once stripped
	// but carry DIFFERENT forms are usually different entities.
	families map[string]bool
	// blocks are the cheap signatures used to decide which pairs to compare.
	blocks []string
}

type keyedRow struct {
	row Row
	key key
}

// MaxKeyCompareLen bounds what similarity is computed on.
//
// jaro is O(len_a × len_b) and nothing caps an extracted value's length, so forty
// rows carrying an 8,000-character key took nine seconds. No entity name is this
// long; a value that is, is a page's paragraph in a key field.
const MaxKeyCompareLen = 200

func (s Schema) keyOf(r Row) key {
	var parts []string
	for _, name := range s.Keys() {
		if v, ok := r.Get(name); ok {
			parts = append(parts, v)
		}
	}
	kept, families := splitSuffixes(strings.Fields(foldForCompare(strings.Join(parts, " "))))
	norm := strings.Join(kept, " ")
	if len(norm) > MaxKeyCompareLen {
		norm = norm[:MaxKeyCompareLen]
	}
	return key{normalised: norm, families: families, blocks: blocksFor(norm)}
}

// corporateSuffixes maps a legal-form word to its FAMILY.
//
// "Acme Ltd" and "Acme Limited" are one company, and a similarity computed on the
// full strings puts them further apart than "Acme Ltd" and "Acme Foods" — which is
// the wrong way round and the single biggest source of both false positives and
// false negatives on company names.
//
// A family rather than a set, because stripping alone over-merges. Measured:
// "Acme Holdings" and "Acme Group" both reduced to "acme" and merged into one row.
// They are usually different legal entities, and merging two companies is the
// precision failure this package says is the one to hold. The family lets the
// comparison ask a sharper question — do these differ ONLY by their legal form,
// and is that form the same form written differently — so "ltd"/"limited" still
// merge while "holdings"/"group" do not.
var corporateSuffixes = map[string]string{
	"ltd": "ltd", "limited": "ltd",
	"plc": "plc",
	"llp": "llp", "lp": "llp",
	"inc": "inc", "incorporated": "inc",
	"llc":  "llc",
	"corp": "corp", "corporation": "corp",
	"co": "co", "company": "co",
	"group":    "group",
	"holdings": "holdings", "holding": "holdings",
	"gmbh": "gmbh", "ug": "gmbh",
	"ag": "ag",
	"sa": "sa", "sas": "sa",
	"srl": "srl", "spa": "spa",
	"bv": "bv", "nv": "nv",
	"ab": "ab", "as": "as", "oy": "oy", "aps": "aps",
	"pty": "pty", "pte": "pte",
	"kk": "kk", "kg": "kg", "se": "se",

	// Country and region qualifiers, added after a live run split three of eleven
	// rows on them: "Aldi" and "Aldi UK" became two entities, as did "Lidl" and
	// "Lidl GB", each with half the figures. The constructed ground-truth set had
	// no case like it, so the measure said 1.000 while real data was 27%
	// duplicated — which is the difference between a fixture and the world.
	//
	// They belong in this table rather than in a second mechanism because they
	// behave identically: a trailing qualifier that names the same entity, where
	// two DIFFERENT ones ("Aldi UK" against "Aldi US") should not merge. The family
	// rule already does exactly that.
	//
	// Deliberately narrow. Unambiguous country codes and words only — no "group",
	// no "international", no "holdings" beyond what is already above, because
	// "Acme Group" and "Acme International" are routinely different legal entities
	// and merging them is the precision failure this package holds hardest.
	"uk": "uk", "gb": "uk", "britain": "uk",
	"us": "us", "usa": "us", "america": "us",
	"ie": "ie", "ireland": "ie",
	"de": "de", "germany": "de",
	"fr": "fr", "france": "fr",
	"es": "es", "spain": "es",
	"it": "it", "italy": "it",
	"nl": "nl", "netherlands": "nl",
	"ca": "ca", "canada": "ca",
	"au": "au", "australia": "au",
	"nz": "nz",
	"jp": "jp", "japan": "jp",
	"cn": "cn", "china": "cn",
	"in": "in", "india": "in",
}

// Normalise reduces a key value to what should be compared.
//
// Exported because the eval harness computes the same normalisation when scoring,
// and a scorer using a different rule from the merge would be measuring something
// else.
func Normalise(s string) string {
	kept, _ := splitSuffixes(strings.Fields(foldForCompare(s)))
	return strings.Join(kept, " ")
}

// foldForCompare lower-cases, folds accents and removes punctuation, leaving the
// tokens for splitSuffixes.
func foldForCompare(s string) string {
	// Case and accents first: "Nestlé" and "Nestle" are the same company, and a
	// source that lost the accent should not become a second entity.
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			b.WriteRune(fold(r))
		case unicode.IsSpace(r), r == '-', r == '_', r == '/', r == '&':
			b.WriteByte(' ')
		default:
			// Punctuation is removed rather than spaced: "St.Gallen" is one
			// token, and "acme, inc." must not become "acme  inc".
		}
	}

	return strings.Join(strings.Fields(b.String()), " ")
}

// splitSuffixes removes a TRAILING run of legal-form words and returns them as
// canonical families.
//
// Trailing only. Stripping them wherever they appeared was positionally blind, so
// "AS Bank" became "bank" and "The Company" became "the" — the table contains
// short words (as, se, co, ab, lp, kk) that are ordinary leading words in other
// names.
//
// When everything is a suffix the original tokens are kept: "Ltd" as a whole key
// value is not a company, but dropping it silently would lose the row rather than
// report it.
func splitSuffixes(tokens []string) (kept []string, families map[string]bool) {
	families = map[string]bool{}
	end := len(tokens)
	for end > 0 {
		fam, ok := corporateSuffixes[tokens[end-1]]
		if !ok {
			break
		}
		families[fam] = true
		end--
	}
	if end == 0 {
		return tokens, map[string]bool{}
	}
	return tokens[:end], families
}

// foldTable maps the accented Latin letters that appear in entity names onto
// their base form.
//
// A map, and it started as two parallel strings — which silently misaligned. The
// "from" side had 75 runes and the "to" side 79, so every letter past the drift
// mapped to the wrong base: "Łódź" normalised to "dodz" rather than "lodz". Two
// parallel sequences whose correspondence nothing checks is the construct that
// caused it, so the correspondence is now written out and cannot drift.
//
// A table rather than golang.org/x/text/unicode/norm: this binary has no
// third-party dependency it does not need, and the set that matters for entity
// names on the web is small and stable.
var foldTable = map[rune]rune{
	'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a', 'ā': 'a', 'ă': 'a', 'ą': 'a',
	'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e', 'ē': 'e', 'ĕ': 'e', 'ė': 'e', 'ę': 'e', 'ě': 'e',
	'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i', 'ĩ': 'i', 'ī': 'i', 'ĭ': 'i', 'į': 'i',
	'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'ø': 'o', 'ō': 'o', 'ŏ': 'o', 'ő': 'o',
	'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u', 'ũ': 'u', 'ū': 'u', 'ŭ': 'u', 'ů': 'u', 'ű': 'u', 'ų': 'u',
	'ç': 'c', 'ć': 'c', 'ĉ': 'c', 'ċ': 'c', 'č': 'c',
	'ñ': 'n', 'ń': 'n', 'ņ': 'n', 'ň': 'n',
	'ý': 'y', 'ÿ': 'y', 'ŷ': 'y',
	'ž': 'z', 'ź': 'z', 'ż': 'z',
	'š': 's', 'ś': 's', 'ŝ': 's', 'ş': 's', 'ß': 's',
	'ğ': 'g', 'ĝ': 'g',
	'ď': 'd', 'đ': 'd', 'ð': 'd',
	'ł': 'l', 'ļ': 'l', 'ľ': 'l',
	'ŕ': 'r', 'ř': 'r',
	'ť': 't', 'ţ': 't', 'þ': 't',
	'æ': 'a', 'œ': 'o',
}

// fold maps one rune onto its base form, or returns it unchanged.
func fold(r rune) rune {
	if base, ok := foldTable[r]; ok {
		return base
	}
	return r
}

// blockPrefix is how many characters of a token form a block signature.
const blockPrefix = 4

// blocksFor produces the cheap signatures that decide which pairs are compared.
//
// Multi-pass: the whole key's prefix AND each token's prefix, so a pair is
// compared if they agree anywhere. A single block key would lose "British
// Airways" against "Airways, British" and every case where the distinguishing
// word is not first.
func blocksFor(norm string) []string {
	seen := map[string]bool{}
	var out []string
	add := func(s string) {
		if s == "" || seen[s] {
			return
		}
		seen[s] = true
		out = append(out, s)
	}

	flat := strings.ReplaceAll(norm, " ", "")
	add(prefix(flat, blockPrefix))
	add(suffix(flat, blockPrefix))
	for _, tok := range strings.Fields(norm) {
		add(prefix(tok, blockPrefix))
		// The last four characters as well as the first. A prefix block silently
		// cancelled the typo rule for a typo in position 0-3: "Softbank Corp" and
		// "Sofbank Corp" score 1.000 as strings and were never compared, because
		// "soft" and "sofb" are disjoint blocks. tokenMatch exists precisely to
		// absorb that typo, so blocking must not decide the answer before the
		// measure is consulted.
		add(suffix(tok, blockPrefix))
	}
	return out
}

func suffix(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return string(r)
	}
	return string(r[len(r)-n:])
}

func prefix(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return string(r)
	}
	return string(r[:n])
}

// -----------------------------------------------------------------------------
// Clustering
// -----------------------------------------------------------------------------

// cluster groups rows by key similarity, under complete linkage.
func cluster(ctx context.Context, rows []keyedRow, threshold float64) ([][]keyedRow, bool) {
	// Which rows share a block with which. Only these pairs are ever scored.
	byBlock := map[string][]int{}
	for i, r := range rows {
		for _, b := range r.key.blocks {
			byBlock[b] = append(byBlock[b], i)
		}
	}

	assigned := make([]int, len(rows))
	for i := range assigned {
		assigned[i] = -1
	}
	var clusters [][]keyedRow
	var members [][]int

	for i := range rows {
		if ctx.Err() != nil {
			// Everything not yet assigned becomes its own cluster, so the result
			// is under-merged rather than short: losing rows would be worse than
			// leaving duplicates.
			for j := range rows {
				if assigned[j] < 0 {
					clusters = append(clusters, []keyedRow{rows[j]})
					assigned[j] = len(clusters) - 1
				}
			}
			return clusters, true
		}
		if assigned[i] >= 0 {
			continue
		}
		// A new cluster, then everything that matches ALL of it.
		clusters = append(clusters, []keyedRow{rows[i]})
		members = append(members, []int{i})
		ci := len(clusters) - 1
		assigned[i] = ci

		// Best first. Admitting candidates in index order let a partial match
		// arriving earlier raise the complete-linkage bar and lock out a later
		// EXACT one: "JP Morgan", "JP Morgan Chase" and "JPMorgan" put the first
		// two together and left the third alone, even though the code scores
		// "JP Morgan" and "JPMorgan" as identical. Adding an unrelated third row
		// must not un-merge two keys.
		for _, j := range candidatesByScore(rows, byBlock, i) {
			if j <= i || assigned[j] >= 0 {
				continue
			}
			if matchesAll(rows, members[ci], j, threshold) {
				clusters[ci] = append(clusters[ci], rows[j])
				members[ci] = append(members[ci], j)
				assigned[j] = ci
			}
		}
	}
	return clusters, false
}

// candidatesByScore returns the rows sharing a block with seed, most similar
// first. Ties break on index, so the order is total and the merge stays
// deterministic.
func candidatesByScore(rows []keyedRow, byBlock map[string][]int, seed int) []int {
	seen := map[int]bool{}
	var out []int
	for _, b := range rows[seed].key.blocks {
		for _, j := range byBlock[b] {
			if !seen[j] {
				seen[j] = true
				out = append(out, j)
			}
		}
	}
	score := make(map[int]float64, len(out))
	for _, j := range out {
		score[j] = Similarity(rows[seed].key.normalised, rows[j].key.normalised)
	}
	sort.Slice(out, func(a, b int) bool {
		if score[out[a]] != score[out[b]] {
			return score[out[a]] > score[out[b]]
		}
		return out[a] < out[b]
	})
	return out
}

// matchesAll is the complete-linkage test.
//
// Every existing member, not just the one that pulled the candidate in. With
// transitive closure "Acme", "Acme Ltd" and "Acme Foods" collapse into one row
// because the middle matches both ends — and the result is a dataset that has
// quietly merged two companies.
func matchesAll(rows []keyedRow, members []int, j int, threshold float64) bool {
	for _, m := range members {
		if !keysMatch(rows[m].key, rows[j].key, threshold) {
			return false
		}
	}
	return true
}

// keysMatch is Similarity plus the legal-form rule.
//
// Two names identical once stripped but carrying DIFFERENT legal forms are
// usually different entities: "Acme Holdings" and "Acme Group" both reduce to
// "acme" and were merging into one row, which is the precision failure this
// package holds hardest. One side carrying no form at all is the case stripping
// exists for — "Vodafone Group" and "Vodafone" are one company — so it still
// matches.
func keysMatch(a, b key, threshold float64) bool {
	if Similarity(a.normalised, b.normalised) < threshold {
		return false
	}
	if len(a.families) == 0 || len(b.families) == 0 {
		return true
	}
	for fam := range a.families {
		if b.families[fam] {
			return true
		}
	}
	// Both named a legal form and no form is shared. Only a distinction when the
	// rest of the name is otherwise the same — "Acme Ltd" against "Beta GmbH"
	// never reaches here, having failed the similarity test above.
	return false
}

// -----------------------------------------------------------------------------
// Similarity
// -----------------------------------------------------------------------------

// Similarity scores two normalised key values in [0, 1].
//
// Token-level, calibrated against the ground-truth set in merge_test.go.
// Jaro-Winkler is not used: its prefix boost is built for short personal names,
// and on company names the distinguishing word comes second, so "deutschebank"
// and "deutschetelekom" score 0.88 while being different companies.
//
// Token overlap divides by the UNION, not the smaller set — otherwise "Acme"
// against "Acme Bakery Holdings" scores 1.0, and a shared first word is the
// commonest thing two different companies have.
func Similarity(a, b string) float64 {
	if a == b {
		return 1
	}
	if a == "" || b == "" {
		return 0
	}
	if strings.ReplaceAll(a, " ", "") == strings.ReplaceAll(b, " ", "") {
		return 1
	}

	return fuzzyJaccard(a, b)
}

// tokenMatch is how similar two tokens must be to count as the same word.
//
// High, because this absorbs a typo and nothing else. At 0.90 "bank" and "bnak"
// are one word (0.925) while "bank" and "bond" are not.
const tokenMatch = 0.90

// fuzzyJaccard is the intersection over the union of the token sets, where two
// tokens intersect if they are nearly the same word.
//
// Greedy pairing, best match first. Exact for the sizes involved — a key value
// with more than a handful of tokens is not an entity name — and an optimal
// assignment would be more code for no measurable difference.
func fuzzyJaccard(a, b string) float64 {
	ta, tb := strings.Fields(a), strings.Fields(b)
	if len(ta) == 0 || len(tb) == 0 {
		return 0
	}

	used := make([]bool, len(tb))
	var shared int
	for _, x := range ta {
		bestScore, bestIdx := 0.0, -1
		for j, y := range tb {
			if used[j] {
				continue
			}
			s := 1.0
			if x != y {
				s = fuzzyToken(x, y)
			}
			if s > bestScore {
				bestScore, bestIdx = s, j
			}
		}
		if bestIdx >= 0 && bestScore >= tokenMatch {
			used[bestIdx] = true
			shared++
		}
	}
	if shared == 0 {
		return 0
	}
	// The union counts each side's unmatched tokens once plus the shared ones.
	union := shared + (len(ta) - shared) + (len(tb) - shared)
	return float64(shared) / float64(union)
}

// fuzzyToken scores two tokens, refusing to be fuzzy about identifiers.
//
// tokenMatch is calibrated on words: "bank" against "bnak" is 0.925, which is
// what absorbs a typo. On a long token one substitution barely moves the score at
// all, so two ISINs differing in their last digit came out at 1.000 and merged —
// and because the key field's alternatives are filed as SPELLINGS, nothing in the
// CSV or the JSON flagged that two securities had been fused into one row.
//
// A digit is the signal. A typo in a word is common and means nothing; a changed
// character in a ticker, ISIN, CIK, SKU or catalogue number changes which thing
// is being identified. So a token containing a digit must match exactly.
func fuzzyToken(a, b string) float64 {
	if hasDigit(a) || hasDigit(b) {
		return 0
	}
	return jaroWinkler(a, b)
}

func hasDigit(s string) bool {
	for _, r := range s {
		if r >= '0' && r <= '9' {
			return true
		}
	}
	return false
}

// jaroWinkler is the standard measure, with the standard prefix boost.
func jaroWinkler(a, b string) float64 {
	j := jaro(a, b)
	if j <= 0.7 {
		return j
	}
	ra, rb := []rune(a), []rune(b)
	var l int
	for l < len(ra) && l < len(rb) && l < 4 && ra[l] == rb[l] {
		l++
	}
	return j + float64(l)*0.1*(1-j)
}

func jaro(a, b string) float64 {
	ra, rb := []rune(a), []rune(b)
	la, lb := len(ra), len(rb)
	if la == 0 || lb == 0 {
		return 0
	}

	window := la
	if lb > window {
		window = lb
	}
	window = window/2 - 1
	if window < 0 {
		window = 0
	}

	matchedA := make([]bool, la)
	matchedB := make([]bool, lb)
	var matches int
	for i := 0; i < la; i++ {
		lo := i - window
		if lo < 0 {
			lo = 0
		}
		hi := i + window + 1
		if hi > lb {
			hi = lb
		}
		for k := lo; k < hi; k++ {
			if matchedB[k] || ra[i] != rb[k] {
				continue
			}
			matchedA[i], matchedB[k] = true, true
			matches++
			break
		}
	}
	if matches == 0 {
		return 0
	}

	var transpositions int
	k := 0
	for i := 0; i < la; i++ {
		if !matchedA[i] {
			continue
		}
		for !matchedB[k] {
			k++
		}
		if ra[i] != rb[k] {
			transpositions++
		}
		k++
	}
	m := float64(matches)
	return (m/float64(la) + m/float64(lb) + (m-float64(transpositions)/2)/m) / 3
}

// -----------------------------------------------------------------------------
// Folding a cluster into one row
// -----------------------------------------------------------------------------

// fold assembles one Merged from the rows that matched.
func (s Schema) fold(cluster []keyedRow) Merged {
	out := Merged{Cells: map[string]Cell{}, Members: len(cluster), Quotes: map[string]Quote{}}

	sources := map[string]bool{}
	for _, kr := range cluster {
		sources[kr.row.Source] = true
		if _, seen := out.Quotes[kr.row.Source]; !seen && kr.row.Quote != "" {
			out.Quotes[kr.row.Source] = Quote{Text: kr.row.Quote, Offset: kr.row.QuoteOffset}
		}
	}
	for src := range sources {
		out.Sources = append(out.Sources, src)
	}
	sort.Strings(out.Sources)

	for _, f := range s.Fields {
		// Values in the order the sources are sorted, so a tie resolves the same
		// way every run.
		counts := map[string]int{}
		srcOf := map[string][]string{}
		var order []string
		for _, kr := range cluster {
			v, ok := kr.row.Get(f.Name)
			if !ok {
				continue
			}
			if counts[v] == 0 {
				order = append(order, v)
			}
			counts[v]++
			srcOf[v] = append(srcOf[v], kr.row.Source)
		}
		if len(order) == 0 {
			continue
		}
		// Most corroborated first, then the value from the alphabetically first
		// source: a deterministic rule, and one that prefers agreement over
		// arrival order.
		sort.SliceStable(order, func(i, j int) bool {
			if counts[order[i]] != counts[order[j]] {
				return counts[order[i]] > counts[order[j]]
			}
			return order[i] < order[j]
		})

		cell := Cell{Text: order[0], Sources: dedupe(srcOf[order[0]])}
		alts := make([]Alt, 0, len(order)-1)
		for _, v := range order[1:] {
			alts = append(alts, Alt{Text: v, Sources: dedupe(srcOf[v])})
		}
		if len(order) > 1 {
			// A key field's alternatives are SPELLINGS, not disagreements: the
			// merge grouped these rows because it judged the key values to name
			// one entity, so calling them a conflict would contradict its own
			// decision. Everything else is a genuine disagreement.
			if f.Key {
				cell.Variants = alts
			} else {
				cell.Others = alts
			}
		}
		out.Cells[f.Name] = cell
	}
	return out
}

func dedupe(in []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, s := range in {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	sort.Strings(out)
	return out
}

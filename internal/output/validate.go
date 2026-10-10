package output

import (
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// Body validation.
//
// Citation NUMBERS are assigned mechanically before the prompt is built, which
// is what §13 rests on. The brackets around them are not. A live run emitted
//
//	[subword tokenizers often need a massive vocabulary to cover diverse scripts…]
//
// which reads as a citation and points at nothing; another emitted a fragment
// of its own instructions followed by a bare "[1][4]". Both are model failures
// that produced a report worse than the unsynthesized evidence listing already
// available for free.
//
// So the body is checked, and a body that fails falls back to that listing with the
// reason declared. Rejecting costs readability; accepting costs a reader's trust in
// every citation mole prints.

// bodyProblem describes why a synthesis was rejected.
//
// Split in two because the reason travels further than the evidence for it.
// Reason is a closed vocabulary — fixed phrasing, and any interpolation is a
// number or a rune from the lookalike list — so it is safe to hand an MCP
// caller. Detail may quote the model's own output, which is text derived from
// fetched pages; putting that in a caller's context outside a §3.2 fence would
// deliver exactly the content validation just refused.
type bodyProblem struct {
	Reason string
	Detail string
}

func (p bodyProblem) Error() string {
	if p.Detail != "" {
		return p.Reason + ": " + p.Detail
	}
	return p.Reason
}

// safeReason extracts the caller-safe half of a validation failure.
func safeReason(err error) string {
	var p bodyProblem
	if errors.As(err, &p) && p.Reason != "" {
		return p.Reason
	}
	return "the answer did not pass validation"
}

// validateBody checks a synthesized body against the citations it was given.
//
// Returns nil when the body is usable.
func validateBody(body string, findings []Finding, citations []Citation) error {
	trimmed := strings.TrimSpace(body)
	if trimmed == "" {
		return bodyProblem{Reason: "synthesis returned nothing"}
	}

	// Lookalike brackets first. bracketGroups scans ASCII '[' only, so a citation
	// written with fullwidth or CJK brackets was never range-checked at all — a
	// reader sees a citation to source 9 where one source exists. A report has no
	// legitimate reason to contain any of these.
	if r, found := lookalikeBracket(trimmed); found {
		return bodyProblem{Reason: fmt.Sprintf(
			"the answer uses %q as a bracket, which is not a citation and is not checked as one", r)}
	}

	groups := bracketGroups(trimmed)
	valid := 0
	for _, g := range groups {
		if editorialMarks[strings.ToLower(strings.TrimSpace(g))] {
			continue
		}
		ns, ok := citationNumbers(g)
		if !ok {
			// A bracket holding anything but citation numbers. This is the forged
			// citation: it looks like evidence and is not.
			//
			// It also rejects a markdown link, deliberately. The prompt asks for
			// numbered citations and nothing else, and a link in the body is
			// page-derived text reaching the answer outside the citation mechanism
			// — which is the one thing §13's numbering exists to prevent.
			// The only problem here whose evidence is model output. The reason
			// says what happened; the marker itself stays in Detail.
			return bodyProblem{
				Reason: "the answer contains a citation-shaped marker that is not a citation",
				Detail: fmt.Sprintf("[%.60s]", g),
			}
		}
		for _, n := range ns {
			if n < 1 || n > len(citations) {
				return bodyProblem{Reason: fmt.Sprintf(
					"the answer cites source [%d], and only %d source(s) exist", n, len(citations))}
			}
		}
		valid += len(ns)
	}
	if valid == 0 {
		return bodyProblem{Reason: "the answer cites nothing; every factual sentence was required to carry a citation"}
	}

	// Degeneracy takes TWO weak signals together, because either alone produced a
	// measured false rejection.
	//
	// A length floor alone cannot work: the degenerate body that prompted this check is
	// 39 characters over nine findings, and a perfectly good summary of two claims is 52
	// characters. Length does not separate them. Coverage does — the first cites two of
	// nine findings, the second two of two.
	//
	// Coverage alone cannot work either: a large session legitimately gets a short answer
	// citing a fraction of its material.
	//
	// So a body is rejected only when it ignores most of the material AND says less than
	// a typical finding. The median, not the minimum: one ordinary short claim
	// ("MambaByte is token-free." is 24 characters) drops a minimum-based floor below the
	// degenerate body it exists to reject.
	cited := citedFindings(groups, findings, citations)
	prose := len(strings.TrimSpace(stripMarkers(trimmed)))
	floor := medianFinding(findings)
	if len(findings) > 1 && cited*2 < len(findings) && floor > 0 && prose < floor {
		return bodyProblem{Reason: fmt.Sprintf(
			"the answer cites %d of %d findings in %d characters of prose, less than a single "+
				"typical finding (%d) — it has not arranged the material",
			cited, len(findings), prose, floor)}
	}
	return nil
}

// bracketGroups returns the contents of every [...] in the text, plus a marker for
// any group used as markdown link or reference syntax.
//
// The two syntaxes matter because both HIJACK a citation number the pipeline assigned
// mechanically, which is the one thing §13's numbering exists to prevent:
//
//	[1](https://evil.example)   a link labelled "1" pointing anywhere
//	[1]: https://evil.example   a reference definition, after which every [1] in the
//	                            prose resolves to the attacker's URL while the source
//	                            list still shows the real one
//
// An earlier version of this file claimed markdown links were rejected. They were not:
// only non-numeric LABELS were, and "1" is numeric.
func bracketGroups(s string) []string {
	var out []string
	for i := 0; i < len(s); i++ {
		if s[i] != '[' {
			continue
		}
		end := strings.IndexByte(s[i:], ']')
		if end < 0 {
			// An unclosed bracket is not a marker; nothing to check.
			break
		}
		out = append(out, s[i+1:i+end])

		// What follows the close decides whether this is a citation or a link.
		if rest := s[i+end+1:]; strings.HasPrefix(rest, "(") {
			out = append(out, markdownLinkGroup)
		} else if strings.HasPrefix(rest, ":") && atLineStart(s, i) {
			out = append(out, markdownRefGroup)
		}
		i += end
	}
	return out
}

// Sentinels for the two syntaxes, chosen so citationNumbers can never parse them.
const (
	markdownLinkGroup = "\x00link"
	markdownRefGroup  = "\x00ref"
)

// atLineStart reports whether index i is the first non-space byte of its line, which
// is what makes a bracket a markdown reference DEFINITION rather than a citation
// followed by a colon mid-sentence.
func atLineStart(s string, i int) bool {
	for j := i - 1; j >= 0; j-- {
		switch s[j] {
		case '\n':
			return true
		case ' ', '\t':
			continue
		default:
			return false
		}
	}
	return true
}

// lookalikeBracket finds the first non-ASCII bracket in the text.
func lookalikeBracket(s string) (rune, bool) {
	for _, r := range s {
		switch r {
		case '［', '］', // fullwidth
			'【', '】', // CJK lenticular
			'〔', '〕',
			'⟦', '⟧', // mathematical white square
			'⁅', '⁆', // tortoise shell
			'〖', '〗',
			'❲', '❳':
			return r, true
		}
	}
	return 0, false
}

// citationNumbers parses a bracket group's contents as citation numbers.
//
// Accepts "3", "3, 4" and "1-3" — models write all three, and a rejected group discards
// the WHOLE synthesis, so being needlessly strict costs a report. Nothing else: strconv
// .Atoi alone would take "+1", "-0" and Unicode digits, and the promise here is stricter
// than "parses as an int".
func citationNumbers(g string) ([]int, bool) {
	g = strings.TrimSpace(g)
	if g == "" || g == markdownLinkGroup || g == markdownRefGroup {
		return nil, false
	}
	fields := strings.FieldsFunc(g, func(r rune) bool { return r == ',' || r == ' ' })
	if len(fields) == 0 {
		return nil, false
	}
	var out []int
	for _, f := range fields {
		f = strings.TrimSpace(f)
		// A range. "1-3" is three citations, and every one of them still has to be in
		// range for the group to pass.
		if lo, hi, ok := strings.Cut(f, "-"); ok {
			a, aok := parseCitation(lo)
			bnum, bok := parseCitation(hi)
			if !aok || !bok || a > bnum {
				return nil, false
			}
			for n := a; n <= bnum; n++ {
				out = append(out, n)
			}
			continue
		}
		n, ok := parseCitation(f)
		if !ok {
			return nil, false
		}
		out = append(out, n)
	}
	return out, len(out) > 0
}

func parseCitation(s string) (int, bool) {
	s = strings.TrimSpace(s)
	if !allASCIIDigits(s) {
		return 0, false
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0, false
	}
	return n, true
}

// editorialMarks are bracketed asides that are ordinary prose, not citation attempts.
//
// The forged-citation check exists because "[subword tokenizers often need a massive
// vocabulary…]" READS as a citation. "[sic]" does not, and discarding a whole synthesis
// over one is a false rejection with a real cost. Short, closed, and lower-cased before
// lookup — anything longer than these is a sentence in brackets, which is the thing being
// caught.
var editorialMarks = map[string]bool{
	"sic": true, "e.g.": true, "i.e.": true, "eg": true, "ie": true,
	"...": true, "…": true, "citation needed": true, "emphasis added": true,
	"ibid": true, "cf.": true, "cf": true,
}

func allASCIIDigits(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

// stripMarkers removes citation markers, leaving the prose.
func stripMarkers(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		if s[i] == '[' {
			if end := strings.IndexByte(s[i:], ']'); end >= 0 {
				i += end
				continue
			}
		}
		b.WriteByte(s[i])
	}
	return b.String()
}

// citedFindings counts how many findings the body actually cites.
//
// By citation NUMBER, mapped back through the source list: a finding is cited when any
// of its sources carries a number the body used. Coverage is the signal a length floor
// cannot supply.
func citedFindings(groups []string, findings []Finding, citations []Citation) int {
	used := map[int]bool{}
	for _, g := range groups {
		if ns, ok := citationNumbers(g); ok {
			for _, n := range ns {
				used[n] = true
			}
		}
	}
	sourceOf := make(map[string]int, len(citations))
	for _, c := range citations {
		sourceOf[c.Source] = c.N
	}
	n := 0
	for _, f := range findings {
		for _, src := range f.Sources {
			if used[sourceOf[src]] {
				n++
				break
			}
		}
	}
	return n
}

// medianFinding is the median length of the finding texts put in front of the model, or
// 0 when there were none.
//
// The median, not the minimum. The minimum was wrong in a way that re-admitted the exact
// body this check exists to reject: one ordinary short claim in the session — "MambaByte
// is token-free." is 24 characters — drops the floor below a 38-character degenerate
// answer, and the answer is accepted. Measured; and the test that was supposed to catch
// it passed only because all six of its fixtures happened to be about 75 characters.
//
// A median cannot be dragged down by one short claim, and it still scales with the
// material rather than being a constant.
func medianFinding(findings []Finding) int {
	var lens []int
	for _, f := range findings {
		if f.Claim == nil {
			continue
		}
		if n := len(strings.TrimSpace(f.Claim.Text)); n > 0 {
			lens = append(lens, n)
		}
	}
	if len(lens) == 0 {
		return 0
	}
	sort.Ints(lens)
	mid := len(lens) / 2
	if len(lens)%2 == 1 {
		return lens[mid]
	}
	return (lens[mid-1] + lens[mid]) / 2
}

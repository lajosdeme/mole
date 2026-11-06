package verifier

import (
	"regexp"
	"strconv"
	"strings"
)

// figure is one number from a claim, scaled, with its unit.
type figure struct {
	v    float64
	unit string
}

// span is a stated range of one unit.
type span struct {
	lo, hi float64
	unit   string
}

var (
	numPart = `(\d[\d,]*(?:\.\d+)?)\s*(%|percent|per cent)?\s*(thousand|million|billion|trillion|bn)?\s*([a-zA-Zµμ]+)?`
	numRe   = regexp.MustCompile(numPart)
	rangeRe = regexp.MustCompile(`(?i)(?:between\s+)?(\d[\d,]*(?:\.\d+)?)\s*(%|percent|per cent)?\s*(thousand|million|billion|trillion|bn)?\s*(?:-|–|—|to|and)\s*` + numPart)
)

var scales = map[string]float64{
	"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12, "bn": 1e9,
}

// units folds unit words to one spelling; a value with scale sets a multiplier.
var units = map[string]struct {
	unit  string
	scale float64
}{
	"mt": {"t", 1e6}, "mmt": {"t", 1e6}, "gt": {"t", 1e9},
	"t": {"t", 1}, "tonnes": {"t", 1}, "tonne": {"t", 1}, "tons": {"t", 1}, "ton": {"t", 1}, "metric": {"t", 1},
	"g": {"g", 1}, "grams": {"g", 1}, "gram": {"g", 1},
	"µg": {"µg", 1}, "μg": {"µg", 1}, "ug": {"µg", 1}, "micrograms": {"µg", 1},
	"mg": {"mg", 1}, "milligrams": {"mg", 1}, "kg": {"kg", 1}, "kilograms": {"kg", 1},
}

// stopWords are words that follow a number without naming its unit.
var stopWords = map[string]bool{
	"of": true, "and": true, "or": true, "to": true, "in": true, "per": true, "a": true,
	"the": true, "each": true, "every": true, "annually": true, "from": true, "for": true,
}

// read turns one regexp match's groups into a scaled value and unit; ok is false
// for a year (an integer 1900–2100 with no scale, percent or known unit).
func read(num, pct, scale, word string) (float64, string, bool) {
	v, err := strconv.ParseFloat(strings.ReplaceAll(num, ",", ""), 64)
	if err != nil {
		return 0, "", false
	}
	unit := ""
	if pct != "" {
		unit = "%"
	}
	if s, ok := scales[strings.ToLower(scale)]; ok {
		v *= s
	}
	w := strings.ToLower(word)
	_, known := units[w]
	if pct == "" && scale == "" && !known && !strings.ContainsAny(num, ".,") && v >= 1900 && v <= 2100 {
		return 0, "", false
	}
	if u, ok := units[w]; ok && unit == "" {
		unit, v = u.unit, v*u.scale
	} else if unit == "" && w != "" && !stopWords[w] {
		unit = w
	}
	return v, unit, true
}

// spansIn returns the ranges a text states.
func spansIn(s string) []span {
	var out []span
	for _, m := range rangeRe.FindAllStringSubmatch(s, -1) {
		hi, unit, ok := read(m[4], m[5], m[6], m[7])
		if !ok {
			continue
		}
		loScale := m[3]
		if loScale == "" {
			loScale = m[6]
		}
		loPct := m[2]
		if loPct == "" {
			loPct = m[5]
		}
		lo, _, ok := read(m[1], loPct, loScale, m[7])
		if !ok || lo >= hi {
			continue
		}
		out = append(out, span{lo, hi, unit})
	}
	return out
}

// figuresIn returns every number a text states, years left out.
func figuresIn(s string) []figure {
	var out []figure
	for _, m := range numRe.FindAllStringSubmatch(s, -1) {
		if v, unit, ok := read(m[1], m[2], m[3], m[4]); ok {
			out = append(out, figure{v, unit})
		}
	}
	return out
}

// PointInRange reports whether one text states a range and every figure in the
// other lies inside a range of the same unit, so the two can both be true.
func PointInRange(a, b string) bool {
	return inside(spansIn(a), b) || inside(spansIn(b), a)
}

func inside(spans []span, text string) bool {
	if len(spans) == 0 {
		return false
	}
	figs := figuresIn(text)
	if len(figs) == 0 {
		return false
	}
	for _, f := range figs {
		ok := false
		for _, s := range spans {
			if f.unit == s.unit && f.v >= s.lo && f.v <= s.hi {
				ok = true
				break
			}
		}
		if !ok {
			return false
		}
	}
	return true
}

// Package stats is the arithmetic behind §4's statistical-validity check.
//
// A web page asserts and a paper asserts; a query over local data MEASURES, so a
// local claim is verified on n, effect size, significance and holdout stability
// rather than on credibility signals.
//
// Everything works from group SUMMARIES — a count, a sum and a sum of squares.
// No sample is held and no row is read, which is what lets this sit behind the
// aggregation gate at all. The output is a sentence a model can quote, so a
// claim about a difference is checkable the way §11.5 makes a claim about a page
// checkable.
package stats

import (
	"errors"
	"fmt"
	"math"
)

// SumColumn and SumSqColumn are the measure names a query must supply for a
// test to be computable.
//
// A contract between the hypothesis templates that render the query and the
// aggregation gate that reads the result, written down here because it is the
// package that consumes both ends. n comes from COUNT(*), which the gate
// already requires for the k-anonymity floor; these two are what turn a pair of
// means into a comparison with a p-value attached.
const (
	SumColumn   = "sum_measure"
	SumSqColumn = "sum_sq_measure"
	// CountColumn is the count of NON-NULL measure values.
	//
	// Required, and separate from COUNT(*): SUM skips NULL, so using COUNT(*) as n
	// desynchronises them and every blank cell in a CSV drags the mean down. A
	// group with half its measure NULL reported a mean half its real value, and a
	// large significant difference from it.
	CountColumn = "n_measure"
	// OffsetColumn is the constant subtracted from the measure before summing.
	// See Group.Offset — it is what keeps Σx² from cancelling.
	OffsetColumn = "measure_offset"
)

// Group is one bucket's summary. n, Σx and Σx² are sufficient statistics for
// everything below — which is the reason this fits behind the gate at all.
type Group struct {
	Name  string
	N     int64
	Sum   float64
	SumSq float64
	// Offset is a constant subtracted from every value before Sum and SumSq
	// were accumulated.
	//
	// Variance and differences of means are shift-invariant, so a query can
	// centre its measure and hand over sums that do not cancel — which is the
	// only real fix for the precision failure documented on Variance. The mean
	// is recovered by adding it back.
	Offset float64
}

// Mean of the group.
func (g Group) Mean() float64 {
	if g.N == 0 {
		return 0
	}
	return g.Sum/float64(g.N) + g.Offset
}

// Variance is the SAMPLE variance (n−1).
//
// Not the population variance the envelope's column summaries use, and the
// difference is not pedantry: those describe the result set, which is all there
// is of it, while this treats a group as a sample drawn from a process. A
// t-test on a population variance overstates its own confidence.
func (g Group) Variance() float64 {
	v, _ := g.variance()
	return v
}

// MaxMagnitudeRatio is how far the mean may sit from zero, in standard
// deviations, before a variance computed from Σx and Σx² is treated as
// unresolvable.
//
// Σx² − (Σx)²/n subtracts two nearly equal large numbers, and the surviving
// precision depends on the ratio of magnitude to spread. Measured on 1000-row
// groups of known variance:
//
//	spread 1    mean 1e6    relative error 1.5e-06
//	spread 10   mean 1e7    relative error 0.026
//	spread 10   mean 3e7    relative error 0.178
//	spread 100  mean 3e8    relative error 0.099
//
// The ratio separates them; a first-order error estimate does not — eps·Σx²/ss
// is 3.4e-4 for both the 1.5e-06 and the 0.026 row. A million is four orders
// inside float64's ~16 digits, and every row above 1% error exceeds it.
//
// It over-refuses the accurate spread-1/mean-1e6 case. That is the right way to
// be wrong: the fix is subtracting a constant in the query, and a refusal
// naming it costs a retry, where a variance ten times too small costs a
// fabricated p-value of 6e-10.
const MaxMagnitudeRatio = 1e6

// variance reports the SAMPLE variance (n−1) and whether float64 could resolve
// it.
//
// Not the population variance the envelope's column summaries use: those
// describe the result set, which is all there is of it, while this treats a
// group as a sample drawn from a process. A t-test on a population variance
// overstates its own confidence.
//
// The second return exists because the old guard — clamp a negative result to
// zero — caught only the end state and missed the regime that matters. Measured,
// on 1000-row groups with true variance 0.667 and a true p of 0.171:
//
//	mean 1e6   variance 0.667668   not significant, p = 0.177   correct
//	mean 1e7   variance 0.064064   SIGNIFICANT,     p = 6.2e-10  fabricated
//	mean 3e7   variance 0          not significant, p = 0.83
//	mean 1e8   variance 0          refused
//
// At values around ten million — revenue in cents, populations, byte counts —
// the gate reported a large significant effect for data with none. Unresolvable
// and zero are different facts and are now reported as different facts.
func (g Group) variance() (float64, bool) {
	if g.N < 2 {
		return 0, false
	}
	n := float64(g.N)
	ss := g.SumSq - (g.Sum*g.Sum)/n
	if ss <= 0 {
		return 0, ss == 0 && g.SumSq == 0
	}
	v := ss / (n - 1)
	// After centring, Σx/n is near zero and this ratio is small whatever the
	// original magnitude was — so a query that subtracts a constant passes
	// without a special case, and one that does not is measured on its own
	// numbers.
	if sd := math.Sqrt(v); sd > 0 && math.Abs(g.Sum/n)/sd > MaxMagnitudeRatio {
		return 0, false
	}
	return v, true
}

// Verdict is what the test supports.
type Verdict string

const (
	// Significant: the difference is unlikely under the null, and there were
	// enough records for that to mean something.
	Significant Verdict = "significant"
	// NotSignificant: the data does not distinguish the groups.
	NotSignificant Verdict = "not significant"
	// Underpowered: too few records to conclude either way. Reported as its own
	// verdict rather than folded into "not significant", because the two lead
	// to opposite next actions — one says the effect is absent, the other says
	// nobody looked hard enough.
	Underpowered Verdict = "underpowered"
)

// Alpha and MinGroupN are the thresholds.
//
// MinGroupN is a judgement and is stated as one. Twenty per group is where a
// t-test on a non-normal sample stops being badly behaved in practice; it is
// not a rule anyone can derive. It exists because the alternative is reporting
// p = 0.03 from six records against five as though it settled something, which
// is the exact failure §4's row is written to prevent.
const (
	Alpha     = 0.05
	MinGroupN = 20
)

// Test is a two-sample comparison and its verdict.
type Test struct {
	Kind string `json:"kind"`
	// Measure names the column compared.
	Measure string `json:"measure"`

	GroupA string `json:"group_a"`
	GroupB string `json:"group_b"`
	NA     int64  `json:"n_a"`
	NB     int64  `json:"n_b"`

	MeanA      float64 `json:"mean_a"`
	MeanB      float64 `json:"mean_b"`
	Difference float64 `json:"difference"`

	Statistic float64 `json:"statistic"`
	DF        float64 `json:"df"`
	P         float64 `json:"p"`
	// PAdjusted is P after the Holm–Bonferroni correction for however many
	// pairwise comparisons were run, and it is the value every verdict is derived
	// from. Equal to P when there was one comparison, which is what makes the
	// two-group and k-group paths the same code rather than two rules.
	PAdjusted float64 `json:"p_adjusted,omitempty"`
	// Comparisons is how many pairs were tested together. 0 or 1 means no
	// correction was needed; a reader seeing 15 knows why the adjusted value is
	// so much larger than the raw one.
	Comparisons int `json:"comparisons,omitempty"`
	// EffectSize is Cohen's d on the pooled standard deviation. §4 asks for it
	// by name, and for the reason it is usually asked for: a significant
	// difference can still be too small to act on, and p alone cannot say so.
	EffectSize float64 `json:"effect_size"`
	CILow      float64 `json:"ci_low"`
	CIHigh     float64 `json:"ci_high"`

	Verdict Verdict `json:"verdict"`
	// Holdout is §4's stability check, present when the caller ran it. Nil means
	// it was not attempted, which is different from having failed — see
	// Holdout.Clause.
	Holdout *Holdout `json:"holdout,omitempty"`
	// Summary is the sentence that reaches a model. It is the whole point of
	// the package: a model handed two means will describe a trend, and a model
	// handed "not distinguishable from chance (p = 0.41)" has to quote that or
	// lose the claim.
	Summary string `json:"summary"`
}

// Welch compares two groups without assuming equal variances.
//
// Welch rather than Student because equal variances is an assumption nobody
// checks and this has no way to check it either: the groups are two summaries,
// arriving with whatever spread they have. Welch costs a slightly smaller
// degrees of freedom and removes the assumption.
//
// Reports false when the groups cannot support a test at all — fewer than two
// records, or no variation in either.
func Welch(measure string, a, b Group) (Test, bool) {
	t, err := WelchOrReason(measure, a, b)
	return t, err == nil
}

// ErrNoTest is returned by WelchOrReason when the groups cannot support a test.
var ErrNoTest = errors.New("stats: no test")

// WelchOrReason is Welch with the reason it could not run.
//
// "No test" with no reason is the sort of silence this codebase exists to avoid:
// a caller cannot tell "both groups are constant" from "the values are too large
// for float64 to resolve their spread", and those lead to opposite actions —
// one says there is nothing to find, the other says the query should centre its
// measure.
func WelchOrReason(measure string, a, b Group) (Test, error) {
	if a.N < 2 || b.N < 2 {
		return Test{}, fmt.Errorf("%w: a group has fewer than two records", ErrNoTest)
	}
	va, okA := a.variance()
	vb, okB := b.variance()
	if !okA || !okB {
		return Test{}, fmt.Errorf("%w: the spread of %q could not be resolved at this "+
			"magnitude; the query should subtract a constant from the measure before "+
			"summing it", ErrNoTest, measure)
	}
	na, nb := float64(a.N), float64(b.N)

	se2 := va/na + vb/nb
	if se2 <= 0 {
		// Both groups constant. Either they are identical, in which case there
		// is nothing to test, or they differ with zero spread, which is a
		// property of a tiny sample rather than evidence about a process.
		//
		// Overlapping with the degrees-of-freedom check below rather than
		// distinct from it: a zero standard error also makes the
		// Welch–Satterthwaite denominator zero, so removing either guard alone
		// changes nothing. Both stay — this one names the condition, and the
		// other catches whatever else can make df undefined.
		return Test{}, fmt.Errorf("%w: neither group varies", ErrNoTest)
	}
	se := math.Sqrt(se2)

	t := (a.Mean() - b.Mean()) / se
	// Welch–Satterthwaite.
	df := (se2 * se2) / (
	//
	(va*va)/(na*na*(na-1)) + (vb*vb)/(nb*nb*(nb-1)))
	if math.IsNaN(df) || df <= 0 {
		return Test{}, fmt.Errorf("%w: the degrees of freedom are undefined", ErrNoTest)
	}

	p := twoTailedT(t, df)

	// Cohen's d on the pooled standard deviation.
	pooled := math.Sqrt(((na-1)*va + (nb-1)*vb) / (na + nb - 2))
	var d float64
	if pooled > 0 {
		d = (a.Mean() - b.Mean()) / pooled
	}

	// A 95% interval on the difference of means, using the t quantile for THIS
	// df rather than the normal's 1.96.
	//
	// The approximation was justified in a comment as "under 3%", which was both
	// wrong and the wrong thing to care about: it made the interval disagree with
	// the verdict beside it. Measured at n=20/20 (df=38, t* = 2.0244):
	//
	//	NOT distinguishable from chance (Welch t = 1.96, p = 0.057), 95% CI 0.00 to 6.36
	//
	// an interval excluding zero next to a null verdict, in one sentence a claim
	// has to quote. The quantile is inverted from the same distribution the
	// p-value comes from, so the two cannot disagree by construction.
	diff := a.Mean() - b.Mean()
	half := tQuantile(df) * se

	test := Test{
		Kind: "welch t-test", Measure: measure,
		GroupA: a.Name, GroupB: b.Name, NA: a.N, NB: b.N,
		MeanA: a.Mean(), MeanB: b.Mean(), Difference: diff,
		Statistic: t, DF: df, P: p, EffectSize: d,
		CILow: diff - half, CIHigh: diff + half,
	}
	// PAdjusted equals P until something corrects it. Set here rather than left
	// zero, so a caller reading PAdjusted on a single test gets the p-value the
	// verdict came from instead of a zero that looks overwhelmingly significant.
	test.PAdjusted = p
	test.Verdict = VerdictFor(p, a.N, b.N)
	test.Summary = summarize(test)
	return test, nil
}

// VerdictFor is the one place the thresholds are applied.
//
// Variadic over the group sizes because a two-sample test has two and a
// sandbox-computed result has one. It used to be written twice — here over both
// groups, and again in coderunner over a single n — under a comment claiming the
// copy "mirrors the SQL path's rule so the two cannot disagree". They already
// did: this required BOTH groups to clear MinGroupN and the copy checked one.
func VerdictFor(p float64, ns ...int64) Verdict {
	for _, n := range ns {
		if n < MinGroupN {
			return Underpowered
		}
	}
	if p < Alpha {
		return Significant
	}
	return NotSignificant
}

// summarize writes the sentence that becomes evidence.
//
// Phrased so the honest reading is the easy one. "not distinguishable from
// chance" rather than "no difference" — the test cannot show absence — and
// underpowered results say what is missing rather than reporting a p-value
// somebody would quote.
// Sentence renders a verdict for a passage a claim will quote.
//
// Shared, because there were two: Welch's and the sandbox's, under a comment
// claiming they were "phrased like the SQL path's so that a reader cannot tell
// from the wording which one produced it". A reader could — the two differed in
// the underpowered clause, in whether a test statistic was named, and in whether
// an effect size or an interval appeared at all.
//
// detail is the part only the caller knows: Welch adds its statistic and
// interval, a sandbox result adds what its script reported.
func Sentence(head string, v Verdict, p float64, detail string) string {
	switch v {
	case Underpowered:
		return head + fmt.Sprintf("; UNDERPOWERED — fewer than %d records, so this is "+
			"not evidence either way", MinGroupN)
	case Significant:
		return head + fmt.Sprintf("; statistically significant (p = %s)%s", PValue(p), detail)
	default:
		return head + fmt.Sprintf("; NOT distinguishable from chance (p = %s)%s",
			PValue(p), detail)
	}
}

func summarize(t Test) string {
	dir := "higher"
	if t.Difference < 0 {
		dir = "lower"
	}
	head := fmt.Sprintf("%s in %q is %s than in %q by %s (means %s and %s; n = %d and %d)",
		t.Measure, t.GroupA, dir, t.GroupB, sig(math.Abs(t.Difference)),
		sig(t.MeanA), sig(t.MeanB), t.NA, t.NB)

	detail := fmt.Sprintf(", Welch t = %s, effect size %s (%s), 95%% CI %s to %s",
		Num(t.Statistic), Num(t.EffectSize), Magnitude(t.EffectSize),
		Num(t.CILow), Num(t.CIHigh))
	if t.Comparisons > 1 {
		// Stated in the sentence a claim must quote, not only in the JSON. A model
		// handed "p = 0.03" out of fifteen comparisons will write "significant",
		// and the only defence is for the correction to be part of the text it has
		// to reproduce verbatim.
		detail += fmt.Sprintf(", unadjusted p = %s over %d pairwise comparisons",
			PValue(t.P), t.Comparisons)
	}
	// The ADJUSTED p is the one in the sentence, because it is the one the verdict
	// came from. The raw value is in the detail clause above when there was more
	// than one comparison.
	return Sentence(head, t.Verdict, t.PAdjusted, detail)
}

// magnitude labels an effect size, because "d = 0.21" means nothing to a reader
// who has not memorised the conventions and everything to one who has.
// Magnitude labels an effect size. Exported so both evidence paths use the same
// words for the same number.
func Magnitude(d float64) string {
	switch a := math.Abs(d); {
	case a < 0.2:
		return "negligible"
	case a < 0.5:
		return "small"
	case a < 0.8:
		return "medium"
	default:
		return "large"
	}
}

// Num formats a figure for a passage a claim will quote.
//
// One formatter, and there were three: the gate rendered at two decimals, the
// sandbox at four, and this at two. The gate's destroyed the figures it
// rendered — a rate column of 0.0001 to 0.003 reached the model as
//
//	lowest 0.00, highest 0.00, mean 0.00, median 0.00
//
// so §11.5 then permitted only "0.00" as a citation. Small magnitudes keep
// significant digits instead of decimal places, which is what makes a rate
// quotable at all.
//
// The gate's copy also used float64(int64(f)), undefined above 2^63, where the
// other two guarded the range.
func Num(f float64) string {
	switch a := math.Abs(f); {
	case f == math.Trunc(f) && a < 1e15:
		return fmt.Sprintf("%.0f", f)
	case a >= 0.01:
		return fmt.Sprintf("%.2f", f)
	case a > 0:
		// Four significant digits, however small. %g keeps them without
		// committing to an exponent until one is needed.
		return fmt.Sprintf("%.4g", f)
	default:
		return "0"
	}
}

// PValue formats a p-value, with a floor rather than a run of zeros.
func PValue(p float64) string {
	if p < 0.001 {
		return "<0.001"
	}
	return fmt.Sprintf("%.3f", p)
}

func sig(f float64) string  { return Num(f) }
func pval(p float64) string { return PValue(p) }

// tQuantile is the two-sided 95% critical value for df degrees of freedom —
// the t solving twoTailedT(t, df) = Alpha.
//
// Found by bisection on the same function that produces the p-value, so the
// interval and the verdict are answers about one distribution rather than two.
// Twenty iterations over [0, 400] resolves it to ~4e-4, well inside the
// precision anything downstream prints.
func tQuantile(df float64) float64 {
	lo, hi := 0.0, 400.0
	for i := 0; i < 60; i++ {
		mid := (lo + hi) / 2
		if twoTailedT(mid, df) > Alpha {
			lo = mid
		} else {
			hi = mid
		}
	}
	return (lo + hi) / 2
}

// -----------------------------------------------------------------------------
// The t distribution
// -----------------------------------------------------------------------------

// twoTailedT is P(|T| >= |t|) for df degrees of freedom.
//
//	p = I_{df/(df+t²)}(df/2, 1/2)
//
// where I is the regularized incomplete beta function. Written out rather than
// taken from a dependency: it is forty lines, the alternative pulls a whole
// statistics library into a binary that is one static file on purpose, and a
// wrong p-value is the sort of error that would be believed.
func twoTailedT(t, df float64) float64 {
	if math.IsNaN(t) || math.IsInf(t, 0) {
		return 1
	}
	x := df / (df + t*t)
	p := incompleteBeta(df/2, 0.5, x)
	return math.Min(1, math.Max(0, p))
}

// incompleteBeta is the regularized incomplete beta function I_x(a, b).
func incompleteBeta(a, b, x float64) float64 {
	switch {
	case x <= 0:
		return 0
	case x >= 1:
		return 1
	}
	lbeta := lgamma(a+b) - lgamma(a) - lgamma(b) +
		a*math.Log(x) + b*math.Log(1-x)
	front := math.Exp(lbeta)

	// The continued fraction converges quickly on one side of the symmetry
	// point and slowly on the other, so the far side is evaluated through the
	// identity I_x(a,b) = 1 − I_{1−x}(b,a).
	if x < (a+1)/(a+b+2) {
		return front * betaCF(a, b, x) / a
	}
	return 1 - math.Exp(lgamma(a+b)-lgamma(a)-lgamma(b)+
		b*math.Log(1-x)+a*math.Log(x))*betaCF(b, a, 1-x)/b
}

// betaCF evaluates the continued fraction for the incomplete beta function by
// the modified Lentz method.
func betaCF(a, b, x float64) float64 {
	const (
		maxIter = 200
		epsilon = 3e-14
		tiny    = 1e-300
	)
	qab, qap, qam := a+b, a+1, a-1

	c := 1.0
	d := 1 - qab*x/qap
	if math.Abs(d) < tiny {
		d = tiny
	}
	d = 1 / d
	h := d

	for m := 1; m <= maxIter; m++ {
		fm := float64(m)
		m2 := 2 * fm

		// Even step.
		num := fm * (b - fm) * x / ((qam + m2) * (a + m2))
		d = 1 + num*d
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = 1 + num/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1 / d
		h *= d * c

		// Odd step.
		num = -(a + fm) * (qab + fm) * x / ((a + m2) * (qap + m2))
		d = 1 + num*d
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = 1 + num/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1 / d
		del := d * c
		h *= del

		if math.Abs(del-1) < epsilon {
			break
		}
	}
	return h
}

func lgamma(x float64) float64 {
	v, _ := math.Lgamma(x)
	return v
}

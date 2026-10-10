package verifier

import (
	"fmt"
	"net/url"
	"regexp"
	"sort"
	"strings"

	"golang.org/x/net/publicsuffix"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/store"
)

// Derived confidence (§11.3).
//
// Computed from graph structure, not asked of the model: self-reported
// confidence is uncalibrated and mostly encodes fluency, and a fluent claim
// from one anonymous page outranked a claim three publishers agreed on.
// Deterministic, reproducible, explainable line by line in a trace.
//
// §11.3 names five inputs. Four are implemented:
//
//	independent corroborating publishers   — Score.Publishers
//	source class weight                    — Score.Class
//	presence and weight of contradictions  — Score.Contradictors
//	grounding result                       — Score.Grounded
//
// The fifth, "recency vs. the question's volatility", is deliberately absent. It
// needs to know how fast the answer to THIS question changes, and nothing here can
// measure that — a fixed half-life would be a number invented to look rigorous.
// §14.2's corpus is what would supply it. Confidence is honest about being a
// four-term score rather than pretending to a fifth.

// SourceClass is how much weight a publisher's output carries on its own.
type SourceClass string

const (
	ClassPeerReviewed SourceClass = "peer-reviewed"
	ClassPrimary      SourceClass = "primary"
	ClassSecondary    SourceClass = "secondary"
	// ClassUnknown is the default, and defaulting here rather than to something
	// optimistic is the point: an unlisted domain is treated as the weakest kind of
	// evidence, so failing to recognize a good source understates confidence rather
	// than inventing it.
	ClassUnknown SourceClass = "unknown"
)

// Weight is the multiplier a class contributes.
func (c SourceClass) Weight() float64 {
	switch c {
	case ClassPeerReviewed:
		return 1.0
	case ClassPrimary:
		return 0.95
	case ClassSecondary:
		return 0.8
	default:
		return 0.6
	}
}

// rank orders classes so a cluster can be described by its strongest source.
func (c SourceClass) rank() int {
	switch c {
	case ClassPeerReviewed:
		return 3
	case ClassPrimary:
		return 2
	case ClassSecondary:
		return 1
	default:
		return 0
	}
}

// Score is one cluster's derived confidence and the terms that produced it.
//
// The breakdown is not decoration. §11.3 requires the number be explainable in the
// trace viewer, and a bare 0.62 is exactly the kind of figure people either trust
// blindly or dismiss. Every field here is something a reader can check.
type Score struct {
	Confidence float64

	// Publishers is the number of DISTINCT publishers supporting the assertion —
	// registrable domains, not URLs. Five pages on one site is one publisher, and
	// counting them as five is the corroboration inflation the sketch names as an
	// open risk.
	Publishers int
	// Class is the strongest source class in the cluster.
	Class SourceClass
	// Contradictors is the number of distinct publishers whose claims contradict
	// this one. Distinct publishers again, for the same reason.
	Contradictors int
	// Superseded is set when a later-published claim supersedes this one (§11.2).
	Superseded bool
	// Grounded mirrors the cluster's grounding result: nil where §11.5's budgeted
	// re-fetch never ran, which is most claims.
	Grounded *bool

	// Explain is the derivation, one line, for a trace.
	Explain string

	// Cluster is the claims this score covers, representative resolvable via
	// Cluster.Representative.
	//
	// Carried so a caller does not re-derive what scoreCluster already computed.
	// output.Findings independently recomputed distinct publishers, superseded-as-target
	// and cross-cluster contradictions from the same inputs; the two agreed only because
	// both were written carefully, and nothing enforced it.
	Cluster Cluster
}

// Tuning constants. Every one of them is a judgement rather than a measurement,
// and the shape matters more than the values: corroboration saturates, a
// contradiction costs more than a corroboration gains, and an unrecognized source
// is penalized rather than ignored. §14.2's corpus is what would calibrate the
// numbers; the ordering between them is what the tests pin.
const (
	// singlePublisherBase is what one publisher alone can reach, before its class
	// weight applies. Deliberately below half: one source saying something is a
	// lead, not a finding.
	singlePublisherBase = 0.45

	// corroborationDecay gives diminishing returns per additional publisher. The
	// 1→2 step is the largest because independent confirmation is the single
	// biggest change in evidential status; 4→5 barely moves.
	corroborationDecay = 0.6

	// contradictionPenalty is applied per contradicting publisher, multiplicatively
	// and scaled by the edge's weight. One confident contradiction roughly halves
	// confidence, which is intended: a disputed claim should not read as settled.
	contradictionPenalty = 0.5

	// supersededPenalty applies when a later claim supersedes this one. Severe,
	// because the claim is not wrong so much as no longer current, and presenting
	// stale information as confident is the failure mode §11.2 exists to catch.
	supersededPenalty = 0.4

	// ungroundedPenalty applies when §11.5's re-fetch found the quote does not
	// support the claim. Near-fatal: the claim already passed the extraction-time
	// verbatim check, so failing the re-read means the evidence is not there.
	ungroundedPenalty = 0.15

	// groundedBonus applies when a re-fetch confirmed the quote in context. Small,
	// and it makes a checked claim rank above an unchecked one of otherwise equal
	// standing — which is the right ordering, since the check was paid for.
	groundedBonus = 1.05
)

// Cluster is a set of claims that duplicate_of edges say are one assertion.
type Cluster struct {
	// Claims, ordered by ID so a cluster's identity does not depend on traversal
	// order.
	Claims []*core.Claim
}

// IDs returns the cluster's claim IDs.
func (c Cluster) IDs() []string {
	out := make([]string, 0, len(c.Claims))
	for _, cl := range c.Claims {
		out = append(out, cl.ID)
	}
	return out
}

// Representative is the claim that stands for the cluster in a report.
//
// §11.2: a duplicate_of cluster renders as one claim with N sources. The pick is
// the strongest source class, then the earliest claim, so it is deterministic and
// prefers the best-attested wording.
func (c Cluster) Representative() *core.Claim {
	if len(c.Claims) == 0 {
		return nil
	}
	best := c.Claims[0]
	bestRank := ClassOf(best.Source).rank()
	for _, cl := range c.Claims[1:] {
		if r := ClassOf(cl.Source).rank(); r > bestRank {
			best, bestRank = cl, r
		}
	}
	return best
}

// Clusters groups claims by duplicate_of edges.
//
// Union-find over the duplicate edges only. `supports` deliberately does not
// merge: two claims that support each other are two pieces of evidence, and
// merging them would delete exactly the corroboration §11.3 counts.
func Clusters(claims []*core.Claim, edges []*core.ClaimEdge) []Cluster {
	parent := make(map[string]string, len(claims))
	byID := make(map[string]*core.Claim, len(claims))
	for _, c := range claims {
		if c == nil {
			continue
		}
		parent[c.ID] = c.ID
		byID[c.ID] = c
	}

	var find func(string) string
	find = func(x string) string {
		p, ok := parent[x]
		if !ok {
			return ""
		}
		if p != x {
			parent[x] = find(p)
		}
		return parent[x]
	}
	union := func(a, b string) {
		ra, rb := find(a), find(b)
		if ra == "" || rb == "" || ra == rb {
			return
		}
		// Lower ID wins, so the root — and therefore the grouping — does not depend
		// on the order edges came back in.
		if ra < rb {
			parent[rb] = ra
		} else {
			parent[ra] = rb
		}
	}

	for _, e := range edges {
		if e != nil && e.Kind == core.EdgeDuplicateOf {
			union(e.FromID, e.ToID)
		}
	}

	groups := map[string][]*core.Claim{}
	for _, c := range claims {
		if c == nil {
			continue
		}
		root := find(c.ID)
		groups[root] = append(groups[root], c)
	}

	out := make([]Cluster, 0, len(groups))
	for _, members := range groups {
		sort.SliceStable(members, func(i, j int) bool { return members[i].ID < members[j].ID })
		out = append(out, Cluster{Claims: members})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Claims[0].ID < out[j].Claims[0].ID })
	return out
}

// DeriveConfidence scores every claim in a session from the graph.
//
// Every member of a duplicate cluster gets the cluster's score, because they are
// the same assertion: giving them different numbers would make the report's choice
// of wording change its stated confidence.
func DeriveConfidence(claims []*core.Claim, edges []*core.ClaimEdge) ([]store.ClaimScore, []Score) {
	clusters := Clusters(claims, edges)

	// Which cluster each claim belongs to, so edge endpoints can be resolved to
	// clusters rather than claims.
	clusterOf := map[string]int{}
	for i, cl := range clusters {
		for _, c := range cl.Claims {
			clusterOf[c.ID] = i
		}
	}

	scores := make([]Score, 0, len(clusters))
	out := make([]store.ClaimScore, 0, len(claims))
	for _, cl := range clusters {
		s := scoreCluster(cl, clusterOf, edges)
		s.Cluster = cl
		scores = append(scores, s)
		for _, c := range cl.Claims {
			out = append(out, store.ClaimScore{
				ClaimID:    c.ID,
				Confidence: s.Confidence,
				Grounded:   c.Grounded,
			})
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].ClaimID < out[j].ClaimID })
	return out, scores
}

func scoreCluster(cl Cluster, clusterOf map[string]int, edges []*core.ClaimEdge) Score {
	member := map[string]bool{}
	for _, c := range cl.Claims {
		member[c.ID] = true
	}

	// Distinct supporting publishers, and the strongest class among them.
	publishers := map[string]bool{}
	class := ClassUnknown
	var grounded *bool
	for _, c := range cl.Claims {
		if p := PublisherOf(c.Source); p != "" {
			publishers[p] = true
		}
		if k := ClassOf(c.Source); k.rank() > class.rank() {
			class = k
		}
		// A single failed grounding check condemns the cluster; otherwise a
		// confirmed one carries.
		if c.Grounded != nil {
			if !*c.Grounded {
				v := false
				grounded = &v
			} else if grounded == nil {
				v := true
				grounded = &v
			}
		}
	}

	// Contradictions, counted by distinct opposing PUBLISHER rather than by edge.
	// Five contradicting claims from one site is one site disagreeing, and counting
	// them five times is the same inflation the corroboration side guards against.
	contradictors := map[string]bool{}
	var maxContradictionWeight float64
	superseded := false
	for _, e := range edges {
		if e == nil {
			continue
		}
		fromMine, toMine := member[e.FromID], member[e.ToID]
		if !fromMine && !toMine {
			continue
		}

		switch e.Kind {
		case core.EdgeContradicts:
			// Intra-cluster contradictions are the graph disagreeing with itself —
			// the same pair called both duplicate and contradictory. Penalizing a
			// cluster for contradicting itself would be arithmetic on an
			// inconsistency, so it is skipped.
			if fromMine && toMine {
				continue
			}
			other := e.ToID
			if toMine {
				other = e.FromID
			}
			// Group by the opposing cluster, falling back to the claim, since a
			// contradicting duplicate-cluster is still one disagreement.
			key := fmt.Sprintf("cluster:%d", clusterOf[other])
			if _, ok := clusterOf[other]; !ok {
				key = "claim:" + other
			}
			contradictors[key] = true
			if w := edgeWeight(e); w > maxContradictionWeight {
				maxContradictionWeight = w
			}
		case core.EdgeSupersedes:
			// Only being the TARGET is a penalty. Superseding something else is not
			// a mark against the newer claim.
			if toMine && !fromMine {
				superseded = true
			}
		}
	}

	k := len(publishers)
	conf := corroboration(k) * class.Weight()

	for range contradictors {
		conf *= 1 - contradictionPenalty*maxContradictionWeight
	}
	if superseded {
		conf *= supersededPenalty
	}
	if grounded != nil {
		if *grounded {
			conf *= groundedBonus
		} else {
			conf *= ungroundedPenalty
		}
	}

	s := Score{
		Confidence:    clamp01(conf),
		Publishers:    k,
		Class:         class,
		Contradictors: len(contradictors),
		Superseded:    superseded,
		Grounded:      grounded,
	}
	s.Explain = explain(s, len(cl.Claims))
	return s
}

// corroboration is the base score for k independent publishers.
//
// Saturating: 1 publisher reaches singlePublisherBase, and each further one closes
// a fixed fraction of the remaining gap to 1. So the second publisher is worth far
// more than the fifth, which is how evidence actually works — and it means a
// scraper farm republishing one claim across many domains cannot drive confidence
// to certainty.
func corroboration(k int) float64 {
	if k <= 0 {
		// No resolvable publisher at all. Not zero — the claim still carries a
		// verified quote — but below what one identifiable source earns.
		return singlePublisherBase * 0.5
	}
	gap := 1 - singlePublisherBase
	for i := 1; i < k; i++ {
		gap *= corroborationDecay
	}
	return 1 - gap
}

func edgeWeight(e *core.ClaimEdge) float64 {
	if e.Weight <= 0 {
		return 1
	}
	if e.Weight > 1 {
		return 1
	}
	return e.Weight
}

func explain(s Score, members int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%d publisher(s)", s.Publishers)
	if members > 1 {
		fmt.Fprintf(&b, " across %d duplicate claim(s)", members)
	}
	fmt.Fprintf(&b, ", strongest source %s", s.Class)
	if s.Contradictors > 0 {
		fmt.Fprintf(&b, ", contradicted by %d", s.Contradictors)
	}
	if s.Superseded {
		b.WriteString(", superseded by a later claim")
	}
	switch {
	case s.Grounded == nil:
		b.WriteString(", grounding not checked")
	case *s.Grounded:
		b.WriteString(", grounding confirmed")
	default:
		b.WriteString(", grounding FAILED")
	}
	fmt.Fprintf(&b, " → %.2f", s.Confidence)
	return b.String()
}

// ---------------------------------------------------------------------------
// Publisher identity and source class
// ---------------------------------------------------------------------------

// PublisherOf reduces a source to the entity that published it.
//
// eTLD+1 via the public suffix list, which is the whole point: §11.3 counts
// "distinct domains/publishers, not distinct URLs", and the naive last-two-labels
// shortcut gets that wrong in both directions. It merges bbc.co.uk and
// theguardian.co.uk into one publisher named "co.uk", and it splits
// alice.github.io from bob.github.io — which the PSL correctly treats as separate
// publishers, since github.io is a suffix.
//
// fetch.DomainOf deliberately keeps the naive version: it is a grouping key for
// the §10.4 failure ranking, where merging a country's sites is untidy rather than
// wrong, and changing it would alter the meaning of already-stored rows. Here a
// wrong answer inflates confidence, so it is worth the list.
func PublisherOf(source string) string {
	// Scholarly sources first, because eTLD+1 is the wrong unit for them and
	// silently defeats corroboration. A PubMed citation is
	// pubmed.ncbi.nlm.nih.gov/<PMID>/ and a PMC one is
	// pmc.ncbi.nlm.nih.gov/articles/PMC<id>/, so EVERY academic claim folded to
	// nih.gov — two independent studies agreeing counted as one publisher, and
	// §11.3's corroboration term could never rise for academic evidence at all.
	//
	// The unit chosen is the PAPER, not the journal or the index. That is a
	// judgement: two studies in one journal share an editor, so counting them
	// separately is less conservative than eTLD+1 is for news. It is still the
	// better answer, because two independent studies corroborating each other is
	// real corroboration in a way that two articles on one news site is not —
	// there the shared editorial control is the whole reason to discount them.
	if k := scholarlyKey(source); k != "" {
		return k
	}

	host := hostOf(source)
	if host == "" {
		return ""
	}
	etld1, err := publicsuffix.EffectiveTLDPlusOne(host)
	if err != nil {
		// Happens for a bare public suffix ("com") or an IP literal. The host is
		// the most specific honest answer.
		return host
	}
	return etld1
}

func hostOf(source string) string {
	s := strings.TrimSpace(strings.ToLower(source))
	if s == "" {
		return ""
	}
	// A bare "doi:" source with no parseable registrant. Handled by
	// scholarlyKey when it has one; this is the fallback, and it does fold every
	// such source into one publisher — which the comment here used to deny while
	// the code did it.
	if strings.HasPrefix(s, "doi:") {
		return "doi.org"
	}
	// Connector sources ("connector:<name>#<hash>") are local data, not a web
	// publisher.
	if strings.HasPrefix(s, "connector:") {
		name, _, _ := strings.Cut(strings.TrimPrefix(s, "connector:"), "#")
		return "connector." + name
	}
	u, err := url.Parse(s)
	if err != nil || u.Hostname() == "" {
		return ""
	}
	return strings.TrimSuffix(u.Hostname(), ".")
}

// ClassOf classifies a source.
//
// A hand-written list, and it cannot classify the web — that is not the claim. The
// claim is that it is better than treating every source alike, that it is
// deterministic and testable, and that its default is the WEAK class, so an
// unrecognized publisher understates confidence rather than inventing it.
//
// Matched against the registrable domain, never a substring of the URL. That
// distinction is load-bearing: "https://nih.gov.attacker.example/paper" contains
// ".gov" and must not be classified as a government source, and a substring match
// is exactly how it would be.
func ClassOf(source string) SourceClass {
	pub := registrableDomain(source)
	if pub == "" {
		return ClassUnknown
	}
	if c, ok := publisherClass[pub]; ok {
		return c
	}
	// Suffix rules on the registrable domain. Not exhaustive — every country has
	// its own government suffix — and an unmatched one lands in ClassUnknown,
	// which is the safe direction.
	for _, suffix := range primarySuffixes {
		if strings.HasSuffix(pub, suffix) {
			return ClassPrimary
		}
	}
	return ClassUnknown
}

var primarySuffixes = []string{".gov", ".mil", ".gov.uk", ".gov.au", ".gc.ca", ".europa.eu"}

// publisherClass is a starting list, not a taxonomy.
//
// Preprint servers are ClassPrimary rather than ClassPeerReviewed on purpose: an
// arXiv paper is primary research that has not been refereed, and filing it as
// peer-reviewed would be the single most misleading entry this table could hold.
var publisherClass = map[string]SourceClass{
	// Peer-reviewed venues and their registries.
	"doi.org":           ClassPeerReviewed,
	"nature.com":        ClassPeerReviewed,
	"science.org":       ClassPeerReviewed,
	"sciencemag.org":    ClassPeerReviewed,
	"cell.com":          ClassPeerReviewed,
	"plos.org":          ClassPeerReviewed,
	"pnas.org":          ClassPeerReviewed,
	"springer.com":      ClassPeerReviewed,
	"wiley.com":         ClassPeerReviewed,
	"sciencedirect.com": ClassPeerReviewed,
	"tandfonline.com":   ClassPeerReviewed,
	"ieee.org":          ClassPeerReviewed,
	"acm.org":           ClassPeerReviewed,
	"jstor.org":         ClassPeerReviewed,
	"bmj.com":           ClassPeerReviewed,
	"thelancet.com":     ClassPeerReviewed,
	"nejm.org":          ClassPeerReviewed,
	"frontiersin.org":   ClassPeerReviewed,
	"mdpi.com":          ClassPeerReviewed,
	"elifesciences.org": ClassPeerReviewed,

	// Preprints: primary research, not refereed.
	"arxiv.org":   ClassPrimary,
	"biorxiv.org": ClassPrimary,
	"medrxiv.org": ClassPrimary,
	"ssrn.com":    ClassPrimary,
	"osf.io":      ClassPrimary,

	// Standards bodies and registries: primary sources for what they define.
	"ietf.org":       ClassPrimary,
	"rfc-editor.org": ClassPrimary,
	"w3.org":         ClassPrimary,
	"iso.org":        ClassPrimary,
	"unicode.org":    ClassPrimary,
	"whatwg.org":     ClassPrimary,
	"who.int":        ClassPrimary,
	"un.org":         ClassPrimary,
	"worldbank.org":  ClassPrimary,
	"oecd.org":       ClassPrimary,

	// Encyclopaedic and press: secondary by construction — they report on primary
	// work rather than producing it.
	"wikipedia.org":      ClassSecondary,
	"britannica.com":     ClassSecondary,
	"reuters.com":        ClassSecondary,
	"apnews.com":         ClassSecondary,
	"bbc.co.uk":          ClassSecondary,
	"bbc.com":            ClassSecondary,
	"nytimes.com":        ClassSecondary,
	"washingtonpost.com": ClassSecondary,
	"theguardian.com":    ClassSecondary,
	"economist.com":      ClassSecondary,
	"ft.com":             ClassSecondary,
	"wsj.com":            ClassSecondary,
	"arstechnica.com":    ClassSecondary,
	"theverge.com":       ClassSecondary,
	"wired.com":          ClassSecondary,
}

// registrableDomain is the eTLD+1 of a source, for questions about the HOST.
//
// Split from PublisherOf when the latter started answering "which paper" for
// scholarly sources. ClassOf keys a hand-written list on domains — arxiv.org is
// primary, doi.org is peer-reviewed — and pointing it at a paper identifier made
// every academic source classify as unknown, quietly downgrading confidence for
// exactly the sources §11.3 trusts most. Caught by the existing tests, which is
// what they are for.
func registrableDomain(source string) string {
	host := hostOf(source)
	if host == "" {
		return ""
	}
	etld1, err := publicsuffix.EffectiveTLDPlusOne(host)
	if err != nil {
		return host
	}
	return etld1
}

// scholarlyKey identifies the PAPER behind an academic citation, or "".
//
// Three shapes, all produced by internal/actors' academic path:
//
//	https://doi.org/10.1093/bioinformatics/btaa073   -> doi:10.1093 (the registrant)
//	https://pubmed.ncbi.nlm.nih.gov/32003791/        -> pmid:32003791
//	https://pmc.ncbi.nlm.nih.gov/articles/PMC7214034/ -> pmc:PMC7214034
//	https://arxiv.org/abs/2401.13660v3               -> arxiv:2401.13660v3
//
// A DOI resolves to its REGISTRANT prefix rather than the whole identifier,
// because the prefix is assigned per publisher — 10.1093 is Oxford University
// Press, 10.1038 is Nature. So two papers from different publishers corroborate
// and two from the same one do not, which is exactly what §11.3 is asking. The
// index hosts carry no publisher information at all, so there the paper is the
// most specific honest answer.
func scholarlyKey(source string) string {
	s := strings.TrimSpace(strings.ToLower(source))
	if s == "" {
		return ""
	}
	if m := doiRegistrantPattern.FindStringSubmatch(s); m != nil {
		return "doi:" + m[1]
	}
	for _, sp := range scholarlyPatterns {
		if m := sp.pattern.FindStringSubmatch(s); m != nil {
			return sp.prefix + m[1]
		}
	}
	return ""
}

var doiRegistrantPattern = regexp.MustCompile(`(?:doi\.org/|^doi:)(10\.\d{4,9})/`)

var scholarlyPatterns = []struct {
	prefix  string
	pattern *regexp.Regexp
}{
	{"pmid:", regexp.MustCompile(`pubmed\.ncbi\.nlm\.nih\.gov/(\d+)`)},
	{"pmc:", regexp.MustCompile(`ncbi\.nlm\.nih\.gov/(?:pmc/)?articles/(pmc?\d+)`)},
	{"arxiv:", regexp.MustCompile(`arxiv\.org/(?:abs|html|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)`)},
}

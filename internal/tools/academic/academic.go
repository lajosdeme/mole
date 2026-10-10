// Package academic is the provider boundary for scholarly sources (§10.2).
//
// arXiv and PubMed/PMC are the keyless defaults; Unpaywall turns a DOI into a
// legal open-access copy.
//
// Unpaywall is worth about 5%, not the "single highest-leverage call" §10.2
// calls it: over a 97-paper sweep, 46 papers carried no DOI at all, and of the
// 28 unreadable ones that did, Unpaywall placed none. The binding constraint on
// academic coverage is missing identifiers, not unresolved ones.
//
// Semantic Scholar and OpenAlex are absent deliberately — they serve
// citation-graph queries, a different capability from finding papers about a
// question. So is ScienceDirect/Elsevier, whose full text needs an
// institutional entitlement the API does not grant.
//
// Rate limiting and identification (§10.3) are enforced here rather than in
// each provider, so a new one cannot forget. A worker pool without them gets
// the user banned in week one.
package academic

import (
	"context"
	"errors"
	"fmt"
	"net/mail"
	"net/url"
	"strings"
	"time"

	"github.com/lajosdeme/mole/internal/tools/limiter"
)

// Kind names a provider.
type Kind string

const (
	KindArXiv     Kind = "arxiv"
	KindPubMed    Kind = "pubmed"
	KindUnpaywall Kind = "unpaywall"
)

func Kinds() []Kind { return []Kind{KindArXiv, KindPubMed, KindUnpaywall} }

func (k Kind) Valid() bool {
	switch k {
	case KindArXiv, KindPubMed, KindUnpaywall:
		return true
	}
	return false
}

// ErrNoContact is returned when a provider is built without a contact address.
//
// A sentinel rather than a string because §10.3 makes this a startup check
// rather than a README line, and a caller has to be able to tell "you have not
// configured this yet" from "the provider is down".
var ErrNoContact = errors.New("academic: no contact email configured")

// ErrRateLimited is a 429 or a provider's documented throttle response.
//
// executor.Classify has an arm for it, so §9.5 treats it as transient — retried
// with backoff, never aborting the lead — and DeadEndCause labels it
// rate_limited rather than a generic failure. That arm was missing when this
// sentinel was introduced, and the doc claimed the classification anyway: the
// error fell through to Degraded and was never retried.
var ErrRateLimited = errors.New("academic: rate limited")

// LimitFor is §10.3's table, in code.
//
// The numbers are the providers' own published constraints, not guesses, and
// they are conservative where the provider is vague. Getting these wrong is not
// a performance question: these are shared public services run by libraries and
// government agencies, and the failure mode is a block on the user's address.
//
// The keys are prefixed rather than bare, because this limiter is shared with
// the web fetcher, which keys by hostname — an unprefixed "arxiv" would be a
// different bucket from the "arxiv.org" the fetcher uses, and a prefixed one
// cannot silently collide with either.
func LimitFor(k Kind) limiter.Limit {
	switch k {
	case KindArXiv:
		// arXiv's Terms of Use ask for no more than one request every three
		// seconds and a single connection at a time. Burst 1 is the "single
		// connection" part: a burst of 2 is two simultaneous requests, which is
		// the thing they ask callers not to do.
		return limiter.Limit{Rate: 1.0 / 3.0, Burst: 1, MinInterval: 3 * time.Second}

	case KindPubMed:
		// NCBI E-utilities: 3 requests/second without an API key, 10 with one.
		// The keyless figure is used because mole ships keyless — a key is an
		// optimization, and encoding the higher number here would mean the
		// default configuration exceeds the limit it was written for.
		return limiter.Limit{Rate: 3, Burst: 3}

	case KindUnpaywall:
		// Unpaywall asks for an email on every call and publishes a daily
		// allowance rather than a per-second one. Held to a modest rate because
		// there is no documented burst tolerance to rely on.
		return limiter.Limit{Rate: 5, Burst: 5}
	}
	return limiter.Unlimited
}

// LimiterKey is the bucket a provider's calls are charged to.
func LimiterKey(k Kind) string { return "academic:" + string(k) }

// Register installs every provider's limit on a limiter.
//
// Called once where the limiter is built, so the constraints are in force before
// any provider exists to violate them. Registering per provider construction
// would mean a provider built twice resets its own bucket — which is exactly the
// Crawl-delay bug M5's review found in the fetcher.
func Register(l *limiter.Limiter) {
	if l == nil {
		return
	}
	for _, k := range Kinds() {
		l.Set(LimiterKey(k), LimitFor(k))
	}
}

// Options bound one search.
type Options struct {
	// MaxResults caps papers returned. Zero takes the provider's default.
	MaxResults int
}

// Paper is one result, normalized across providers.
//
// The full-text locations are recorded from METADATA, not by fetching. Every
// provider knows which formats exist for a paper without downloading any of
// them, and that is what makes the escalation ladder affordable: the decision
// to read more than the abstract is taken on free information.
type Paper struct {
	Title    string
	Abstract string

	// Identifiers. A paper often has several; whichever the provider knows is
	// recorded, because Unpaywall keys on DOI and PMC keys on PMCID.
	DOI     string
	ArXivID string
	PMID    string
	PMCID   string

	// PublishedAt is the paper's date, which academic sources report exactly.
	//
	// Worth more here than anywhere else in mole: §11.2's staleness detection
	// needs publication dates on both sides of a contradiction to tell "these
	// disagree" from "this one is older", and web sources supply them rarely
	// and unreliably. This is the first place they arrive as fact.
	PublishedAt *time.Time

	// Where the text can be read, discovered from metadata.
	//
	// HTMLURL is the escalation target: arXiv's LaTeXML HTML and PMC's XML are
	// full text the existing extractor already handles. PDFURL is recorded but
	// not yet readable — mole has no PDF extraction, so a paper with only a PDF
	// is counted rather than parsed, which is what turns "do we need a PDF
	// library?" into a number (§10.4).
	HTMLURL    string
	PDFURL     string
	LandingURL string
	Source     Kind
}

// FullTextFormat reports the best text mole can currently READ for a paper.
//
// Deliberately about mole's capability rather than the paper's availability,
// because the two are different questions and conflating them is how a PDF
// library gets built for a corpus that turns out to be mostly closed access.
type FullTextFormat string

const (
	// FormatHTML is full text mole can read today, through the existing fetcher
	// and extractor.
	FormatHTML FullTextFormat = "html"
	// FormatPDFOnly is open access mole cannot read yet. This is the only
	// bucket a PDF extractor would serve.
	FormatPDFOnly FullTextFormat = "pdf_only"
	// FormatClosed has no open-access copy at all. No amount of parsing helps;
	// the abstract is the whole of what can be used.
	FormatClosed FullTextFormat = "closed"
)

func (p Paper) FullTextFormat() FullTextFormat {
	switch {
	case strings.TrimSpace(p.HTMLURL) != "":
		return FormatHTML
	case strings.TrimSpace(p.PDFURL) != "":
		return FormatPDFOnly
	default:
		return FormatClosed
	}
}

// Response is what a search returned.
type Response struct {
	Query    string
	Provider Kind
	Papers   []Paper

	// No Cost field. arXiv, PubMed and Unpaywall are free, so there was never
	// anything to report — it was declared, never written and never read, while
	// its own comment claimed the ledger settled against it. A paid provider
	// would need one back; adding it then, with a consumer, is cheaper than
	// carrying a field that documents behaviour nobody implements.
	//
	// So what bounds an academic session is the money spent MINING, plus §8.5's
	// MaxLeads and MaxWallClock. MaxToolCalls counts only the mine calls.
}

// Resolver turns an identifier into a paper.
//
// Split out from Provider because Unpaywall is one and is not the other. It has
// no topical search — it answers "where can I legally read this DOI" and
// nothing else — and giving it a Search method that always errored would be an
// interface lying about what its implementations do. §10's sketch shows a single
// AcademicProvider with both; this is the one place the implementation
// deliberately differs, because the split is what the real APIs are.
type Resolver interface {
	// Resolve looks a paper up by identifier — a DOI for Unpaywall, an arXiv ID
	// for arXiv, either for PubMed. Each answers for the identifiers it indexes
	// and refuses the rest, rather than reporting "not found" for a paper that
	// exists somewhere it cannot see.
	Resolve(ctx context.Context, id string) (*Paper, error)
	Kind() Kind
}

// Provider is a Resolver that can also search by topic.
type Provider interface {
	Resolver
	Search(ctx context.Context, query string, opts Options) (*Response, error)
}

// Config is what every provider needs.
type Config struct {
	Kind Kind
	// ContactEmail identifies the caller. NOT optional — see Validate.
	ContactEmail string
	// BaseURL overrides the provider's endpoint, for tests.
	BaseURL string
}

// Validate refuses a provider that cannot identify itself.
//
// §10.3: "Contact email is a required config value before any academic provider
// is enabled. Ship it as a startup check, not a README line." Unpaywall requires
// an email parameter and NCBI requires tool and email; sending requests without
// them is against both providers' documented terms, and the consequence lands on
// the user's address rather than on mole.
//
// Enforced at construction rather than at call time so a misconfigured install
// fails before it has made a single request.
func (c Config) Validate() error {
	if !c.Kind.Valid() {
		return fmt.Errorf("academic: unknown provider %q (want one of %v)", c.Kind, Kinds())
	}
	return CheckContact(c.ContactEmail)
}

// CheckContact reports whether an address is usable for provider identification.
//
// Exported so `mole doctor` reports readiness from the same rule that gates the
// providers. A second, hand-written check in the CLI would be free to drift, and
// the whole point of a startup check is that it agrees with the thing it gates.
func CheckContact(email string) error {
	email = strings.TrimSpace(email)
	if email == "" {
		return ErrNoContact
	}
	addr, err := mail.ParseAddress(email)
	if err != nil || !strings.Contains(addr.Address, "@") {
		return fmt.Errorf("%w: %q is not a usable address", ErrNoContact, email)
	}
	return nil
}

// scrubURLError removes the query string from a transport error.
//
// Go's *url.Error stringifies the full URL, and NCBI and Unpaywall both require
// the contact address as a QUERY PARAMETER — so every dial failure produced an
// error carrying the user's personal email, which then reached a WARN log on an
// ordinary run and, from the coverage command, a JSON report of the kind that
// gets committed next to testdata. url.Error redacts userinfo passwords and
// nothing else.
//
// The path is kept: knowing which endpoint failed is the whole value of the
// message. Only the parameters go.
func scrubURLError(err error) string {
	if err == nil {
		return ""
	}
	var ue *url.Error
	if !errors.As(err, &ue) {
		return err.Error()
	}
	scrubbed := ue.URL
	if u, perr := url.Parse(ue.URL); perr == nil {
		u.RawQuery = ""
		u.User = nil
		scrubbed = u.String()
	}
	return fmt.Sprintf("%s %q: %v", ue.Op, scrubbed, ue.Err)
}

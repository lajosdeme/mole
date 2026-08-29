package fetch

import (
	"github.com/lajosdeme/mole/internal/store"
	"regexp"
	"strings"
	"time"
)

// Outcome classifies why a fetch did or did not yield usable text.
//
// This taxonomy is cheap to record and is the only thing that turns "do we need
// a headless browser?" from an argument into a number. Recording it is an M1
// deliverable precisely because adding it later means re-running the whole eval
// corpus to get the data.
type Outcome string

const (
	OutcomeOK Outcome = "ok"

	// OutcomeProviderContent is a source read without fetching it, because the
	// search provider already returned usable page text.
	//
	// Distinct from ok on purpose. Every per-cause rate in §10.4 is a fraction
	// of attempted fetches, and filing these as ok inflated that denominator
	// with requests never made — dragging js_required and bot_block down by
	// however much of the corpus the provider happened to cover.
	OutcomeProviderContent Outcome = "provider_content"

	// OutcomeJSRequired is the SPA signature: the extractor produced almost
	// nothing, but the document is script-heavy and has an app-root element.
	// This is the ONLY outcome that counts as evidence for a headless browser.
	OutcomeJSRequired Outcome = "js_required"

	// OutcomeStructuredOnly means readability failed but __NEXT_DATA__,
	// JSON-LD, or OpenGraph carried the content. A day of parser work converts
	// these into OK — which is why they must not be lumped in with js_required.
	OutcomeStructuredOnly Outcome = "structured_only"

	OutcomeConsentWall Outcome = "consent_wall"
	OutcomeBotBlock    Outcome = "bot_block"
	OutcomePaywall     Outcome = "paywall"
	OutcomeNotFound    Outcome = "not_found"
	OutcomeTimeout     Outcome = "timeout"

	// OutcomeRobotsDenied and OutcomeGuardDenied are the system working
	// correctly. They are separate constants so they can be excluded from any
	// "failure rate" that informs a capability decision; folding them in would
	// inflate the apparent case for a browser.
	OutcomeRobotsDenied Outcome = "robots_denied"
	OutcomeGuardDenied  Outcome = "guard_denied"

	// OutcomeNotSelected is a search result the actor never asked for, because
	// the source cap or the input-token budget was already spent.
	//
	// Recorded so coverage has a denominator. Search results are otherwise not
	// persisted anywhere, so "read 14 sources" had no "of how many" — and a count
	// of what was read, presented as coverage, reads as a complete denominator.
	// Not a failure and not a refusal: nothing was wrong with these pages, the
	// budget simply ran out before their turn.
	OutcomeNotSelected Outcome = "not_selected"

	OutcomeExtractFailed Outcome = "extract_failed"
	OutcomeServerError   Outcome = "server_error"
	OutcomeTooLarge      Outcome = "too_large"
	OutcomeUnsupported   Outcome = "unsupported_type"
	OutcomeNetworkError  Outcome = "network_error"
)

// CapabilityGap reports whether this outcome is evidence that Mole lacks a
// capability, as opposed to the system correctly declining or the server
// failing. Only js_required qualifies — see §17.1's gate.
func (o Outcome) CapabilityGap() bool { return o == OutcomeJSRequired }

// SystemWorking reports whether the outcome is a deliberate refusal by Mole
// rather than a failure.
func (o Outcome) SystemWorking() bool {
	return o == OutcomeRobotsDenied || o == OutcomeGuardDenied
}

// Usable reports whether the fetch produced text an actor can work with.
func (o Outcome) Usable() bool {
	return o == OutcomeOK || o == OutcomeStructuredOnly || o == OutcomeProviderContent
}

// Attempted reports whether a request was actually made. It is the denominator
// for every per-cause rate.
//
// Two outcomes never touched the network: provider_content, where the search
// provider had already returned the text, and not_selected, where the actor never
// reached that result. Counting either would pad the denominator with requests
// nobody made and drag every per-cause rate down by however much of the result
// list happened to go unread.
func (o Outcome) Attempted() bool {
	return o != OutcomeProviderContent && o != OutcomeNotSelected
}

// Result is one fetch attempt, recorded whether or not it succeeded.
type Result struct {
	URL        string
	Domain     string
	Outcome    Outcome
	StatusCode int
	Bytes      int64
	Duration   time.Duration
	Err        string

	// ContentType and Content are populated on a usable fetch.
	ContentType string
	Content     []byte
	// FinalURL differs from URL when redirects were followed.
	FinalURL string
}

// ---------------------------------------------------------------------------
// Classification heuristics
// ---------------------------------------------------------------------------
//
// These are first-pass signatures. They will be wrong at the margins, which is
// expected and fine: M2's eval corpus is what calibrates them, and the point of
// recording the outcome at all is to find out where they are wrong.

var (
	appRootRe = regexp.MustCompile(`(?i)<(div|main|body)[^>]+\bid=["'](root|app|__next|__nuxt|application|react-root|svelte)["']`)
	scriptRe  = regexp.MustCompile(`(?i)<script\b`)

	nextDataRe = regexp.MustCompile(`(?i)<script[^>]+id=["']__NEXT_DATA__["']`)
	jsonLDRe   = regexp.MustCompile(`(?i)<script[^>]+type=["']application/ld\+json["']`)

	consentMarkers = []string{
		"onetrust", "cookiebot", "trustarc", "usercentrics", "quantcast choice",
		"cookie-consent", "cookieconsent", "gdpr-consent", "consent-manager",
		"didomi", "sourcepoint",
	}
	botBlockMarkers = []string{
		"cf-browser-verification", "cf_chl_", "just a moment", "attention required",
		"checking your browser", "ddos protection", "access denied",
		"perimeterx", "px-captcha", "datadome", "incapsula", "are you a robot",
		"enable javascript and cookies to continue",
	}
	paywallMarkers = []string{
		`"isaccessibleforfree":false`, `"isaccessibleforfree": false`,
		"paywall", "subscribe to continue", "subscribers only",
		"this article is for subscribers", "metered-content", "premium-content",
	}
)

// MinUsableText is the extracted-character floor below which a page is treated
// as having produced nothing. Short legitimate pages exist, but below this a
// summarizer has nothing to work with either way.
//
// Exported because the extractor and the search providers apply the same floor;
// three independent copies of this number would drift.
const MinUsableText = 250

// ClassifyBody decides the outcome for a 2xx response whose extraction has
// already been attempted.
//
// extractedLen is the length of the text a readability pass produced. Order
// matters: bot-block and consent markers are checked before the SPA signature,
// because a challenge page is also script-heavy with an empty body and would
// otherwise be miscounted as js_required — inflating the one number §17.1's
// decision actually turns on.
func ClassifyBody(html string, extractedLen int) Outcome {
	if extractedLen >= MinUsableText {
		return OutcomeOK
	}

	lower := strings.ToLower(html)

	if containsAny(lower, botBlockMarkers) {
		return OutcomeBotBlock
	}
	if containsAny(lower, consentMarkers) {
		return OutcomeConsentWall
	}
	if containsAny(lower, paywallMarkers) {
		return OutcomePaywall
	}
	if HasStructuredData(html) {
		return OutcomeStructuredOnly
	}
	if looksJSRequired(html) {
		return OutcomeJSRequired
	}
	return OutcomeExtractFailed
}

// HasStructuredData reports whether the document carries content in a
// machine-readable block that a parser could recover without executing JS.
//
// OpenGraph deliberately does not qualify. og:title and og:description are
// metadata a page emits alongside its content, capped at a sentence or two —
// no parser turns them into the MinUsableText characters that would make this
// fetch OK. Counting them here filed every SPA as structured_only and held
// js_required near zero, which is the one number §17.1's decision reads.
func HasStructuredData(html string) bool {
	return nextDataRe.MatchString(html) || jsonLDRe.MatchString(html)
}

// looksJSRequired is the SPA signature: an app-root element plus enough script
// tags that the page plainly renders client-side.
func looksJSRequired(html string) bool {
	if !appRootRe.MatchString(html) {
		return false
	}
	return len(scriptRe.FindAllStringIndex(html, 4)) >= 2
}

// ClassifyStatus maps a non-2xx response to an outcome.
func ClassifyStatus(status int, body string) Outcome {
	lower := strings.ToLower(body)
	switch {
	case status == 401 || status == 402:
		return OutcomePaywall
	case status == 403 || status == 429:
		// A 403 is ambiguous: a bot challenge, or a genuine paywall. The body
		// markers separate them, and getting this wrong would put paywalls in
		// the bucket a browser is supposed to fix.
		if containsAny(lower, paywallMarkers) {
			return OutcomePaywall
		}
		return OutcomeBotBlock
	case status == 404 || status == 410:
		return OutcomeNotFound
	case status == 408 || status == 504:
		return OutcomeTimeout
	case status >= 500:
		return OutcomeServerError
	case status >= 400:
		if containsAny(lower, botBlockMarkers) {
			return OutcomeBotBlock
		}
		return OutcomeExtractFailed
	case status >= 200 && status < 300:
		return OutcomeOK
	}
	// 1xx and 3xx reach here only when the client did not follow or consume the
	// response. Neither carries a document, so reporting ok would enter a fetch
	// that produced nothing into the denominator as a success.
	return OutcomeExtractFailed
}

func containsAny(haystack string, needles []string) bool {
	for _, n := range needles {
		if strings.Contains(haystack, n) {
			return true
		}
	}
	return false
}

// ProviderSupplied is the set of URLs whose text came from the search provider rather
// than a fetch (§10.4).
//
// Shared because re-reading one is a mistake two callers independently have to avoid:
// the text was extracted by the provider, so a fresh HTML extraction of the same URL
// produces different bytes and any comparison against it manufactures a mismatch. Both
// eval's citation accuracy and §11.5's grounding check need exactly this set, and the
// verifier's copy was written citing eval's reasoning without sharing its code.
func ProviderSupplied(outcomes []*store.FetchOutcome) map[string]bool {
	out := map[string]bool{}
	for _, o := range outcomes {
		if o != nil && Outcome(o.Outcome) == OutcomeProviderContent {
			out[o.URL] = true
		}
	}
	return out
}

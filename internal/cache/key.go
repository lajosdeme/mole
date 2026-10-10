// Package cache stops a session paying twice for the same artifact (§9.3).
//
// Rev 1 did `if cache.SeenRecently(lead) { continue }`, which had two problems.
// The planner asked for something and got nothing back, so the next replan
// spawned an equivalent lead and the loop livelocked. And keying at the LEAD
// level missed the common case: two differently-worded queries surfacing the
// same page still paid for two fetches.
//
// Both fixes follow from keying on the ARTIFACT — a normalized URL, a DOI, a
// query hash — and from returning the stored result rather than skipping. A hit
// is an answer, not an absence.
package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"net/url"
	"strings"
	"unicode"
)

// trackingParams are dropped from a URL before it is used as a key.
//
// Two links to the same article that differ only by campaign tag are the same
// artifact, and a search provider will happily return both. Without this the
// cache misses precisely where it would help most: aggregators and newsletters,
// which is where duplicate links come from.
//
// "ref" and "source" are deliberately absent. They look like tracking tags and
// often are, but they also SELECT content: GitHub uses ?ref=<branch> and many
// viewers use ?source=<document>. Folding them merges two different documents,
// which attributes evidence to a page that never carried it — the failure this
// normalization is supposed to avoid, caused by the normalization.
var trackingParams = map[string]bool{
	"utm_source": true, "utm_medium": true, "utm_campaign": true,
	"utm_term": true, "utm_content": true, "utm_id": true,
	"gclid": true, "fbclid": true, "msclkid": true, "dclid": true,
	"mc_cid": true, "mc_eid": true, "igshid": true, "twclid": true,
	"referrer": true,
	"_hsenc":   true, "_hsmi": true, "hsCtaTracking": true,
	"spm": true, "scid": true, "yclid": true,
}

// NormalizeURL reduces a URL to a stable identity for the same document.
//
// Deliberately conservative about what it drops. Over-normalizing is worse than
// under-normalizing here: a cache that treats two different pages as one
// returns evidence from a page the claim does not cite, which is a correctness
// failure, whereas a missed hit only costs a fetch.
//
// That is why the path is left alone apart from a trailing slash, and why query
// parameters other than known tracking tags are kept and sorted rather than
// stripped — `?id=42` and `?id=43` are different documents, and nothing here
// can tell which parameters are load-bearing for a given site.
func NormalizeURL(raw string) string {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || u.Host == "" {
		// Unparseable: fall back to the literal string, lowercased. Still a
		// usable key, just a less forgiving one.
		return strings.ToLower(strings.TrimSpace(raw))
	}

	// Scheme is NOT folded. It is tempting — the two often serve the same
	// document, and a redirect between them is invisible by the time a URL
	// reaches here — but content fetched over plaintext http can be rewritten
	// by anyone on the path, and storing it under the https key means a later
	// hit on the https URL serves those bytes with no fetch at all. Quotes then
	// verify against an attacker's text and claims cite an HTTPS URL that was
	// never retrieved. TLS would protect the document and the cache would
	// discard the protection.
	u.Scheme = strings.ToLower(u.Scheme)

	host := strings.ToLower(u.Hostname())
	host = strings.TrimSuffix(host, ".")
	// "www." is a prefix, not a document.
	host = strings.TrimPrefix(host, "www.")
	if port := u.Port(); port != "" && !defaultPort(u.Scheme, port) {
		host = host + ":" + port
	}
	u.Host = host

	// A fragment addresses a position within a document, not a document.
	u.Fragment = ""
	u.RawFragment = ""
	// Credentials are not identity, and keeping them would put them in a key.
	u.User = nil

	q := u.Query()
	for k := range q {
		if trackingParams[strings.ToLower(k)] {
			q.Del(k)
		}
	}
	// Encode sorts keys, so parameter order stops mattering.
	u.RawQuery = q.Encode()

	if u.Path == "" {
		u.Path = "/"
	} else if len(u.Path) > 1 {
		u.Path = strings.TrimSuffix(u.Path, "/")
	}

	return u.String()
}

func defaultPort(scheme, port string) bool {
	return (scheme == "https" && port == "443") || (scheme == "http" && port == "80")
}

// QueryKey is the key for a research query.
//
// Folds casing, punctuation and whitespace. It does NOT fold word order, which
// carries the direction of a relation — and a research question is mostly
// relations:
//
//	"did Acme acquire Beta"        /  "did Beta acquire Acme"
//	"is drug A safer than drug B"  /  "is drug B safer than drug A"
//	"does smoking cause cancer"    /  "does cancer cause smoking"
//
// A sorted key makes each pair identical, and the cost is not a missed saving:
// the second question completes as skipped_cache, is credited with the first
// one's claims, and the report never addresses it.
//
// Keeping word order costs the occasional missed hit on a genuinely reordered
// query, which is the cheap direction to be wrong in.
func QueryKey(query string) string {
	raw := strings.FieldsFunc(strings.ToLower(query), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '-'
	})
	// Hyphens are kept inside a word ("pg-19") but a run of them is punctuation,
	// so trim them at the edges and drop what is left empty. Otherwise "PG-19 --
	// results" carries a "--" token that "PG-19 results" does not.
	words := make([]string, 0, len(raw))
	for _, w := range raw {
		if w = strings.Trim(w, "-"); w != "" {
			words = append(words, w)
		}
	}
	if len(words) == 0 {
		return ""
	}
	sum := sha256.Sum256([]byte(strings.Join(words, " ")))
	return "q:" + hex.EncodeToString(sum[:12])
}

// URLKey is the key for a fetched document.
func URLKey(raw string) string {
	n := NormalizeURL(raw)
	if n == "" {
		return ""
	}
	return "u:" + n
}

// DOIKey is the key for an academic identifier.
//
// Present now because §9.3 names it and because AcademicActor (M6) will resolve
// the same DOI from several routes — a publisher page, an Unpaywall copy, an
// arXiv mirror — which is the case URL keying cannot fold.
func DOIKey(doi string) string {
	d := strings.ToLower(strings.TrimSpace(doi))
	d = strings.TrimPrefix(d, "https://doi.org/")
	d = strings.TrimPrefix(d, "http://doi.org/")
	d = strings.TrimPrefix(d, "doi:")
	if d == "" {
		return ""
	}
	return "d:" + d
}

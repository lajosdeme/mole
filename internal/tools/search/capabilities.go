package search

// What a provider supplies, beyond a ranked list of URLs.
//
// Declared in one table because two mechanisms downstream go quiet without it and
// neither says so on its own. Measured on a ten-question corpus: 369 claims, not one
// with a publication date, and §11.2's supersedes edges therefore never fired —
// stalePair returns early when either side has no date, so every conflict across a
// decade of publication was written as a live disagreement.
//
// The gap is per provider and the two providers have opposite halves of it, so
// "does the search provider do X" cannot be answered anywhere but here.

// Capabilities is what one provider returns.
type Capabilities struct {
	// PageContent means results carry usable body text, so a session can mine
	// without fetching. It changes how many fetches a run makes, and it means the
	// extractor — the only other source of a publication date — never runs.
	PageContent bool

	// PublicationDates means results carry a date mole is willing to believe.
	// Without it, and without a fetch, §11.2 cannot separate staleness from
	// disagreement.
	PublicationDates bool
}

// capabilities is the declared table. A provider missing from it claims nothing,
// which is the safe direction: reporting a capability nothing supplies is how a
// silent mechanism stays silent.
var capabilities = map[Kind]Capabilities{
	// page_age on ordinary web results; snippets only, so pages are fetched.
	KindBrave: {PageContent: false, PublicationDates: true},

	// raw_content on every result, and no date at all: published_date is populated
	// for topic=news, and asking for that topic would narrow a research query to
	// news. Confirmed against recorded responses — the key is absent, not empty.
	KindTavily: {PageContent: true, PublicationDates: false},
}

// CapabilitiesOf reports what a provider supplies. An unknown provider claims
// nothing.
func CapabilitiesOf(k Kind) Capabilities { return capabilities[k] }

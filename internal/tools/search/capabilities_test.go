package search

import "testing"

// TestEveryProviderDeclaresItsCapabilities. A gap nobody declared is a mechanism
// that goes quiet when a provider is added, which is how the staleness rule came to
// be inert on every Tavily install with nothing saying so.
func TestEveryProviderDeclaresItsCapabilities(t *testing.T) {
	for _, k := range Kinds() {
		if _, ok := capabilities[k]; !ok {
			t.Errorf("provider %q declares no capabilities", k)
		}
	}
}

// TestTavilySuppliesContentButNoDates is the combination that made §11.2 unreachable:
// no date in the search response, and page content good enough that nothing is
// fetched, so the extractor that could have read one never runs.
func TestTavilySuppliesContentButNoDates(t *testing.T) {
	c := CapabilitiesOf(KindTavily)
	if !c.PageContent {
		t.Error("tavily is the provider that supplies page content")
	}
	if c.PublicationDates {
		t.Error("tavily returns published_date only for topic=news; general search has no such field")
	}
}

// TestBraveSuppliesDatesButNotContent, which is why the same install on Brave can
// separate staleness from disagreement.
func TestBraveSuppliesDatesButNotContent(t *testing.T) {
	c := CapabilitiesOf(KindBrave)
	if !c.PublicationDates {
		t.Error("brave returns page_age on ordinary web results")
	}
	if c.PageContent {
		t.Error("brave returns snippets, so pages are fetched")
	}
}

// TestAnUnknownProviderClaimsNothing. Defaulting to "has everything" would report a
// capability nothing supplies.
func TestAnUnknownProviderClaimsNothing(t *testing.T) {
	c := CapabilitiesOf(Kind("not-a-provider"))
	if c.PageContent || c.PublicationDates {
		t.Errorf("an unknown provider claimed %+v", c)
	}
}

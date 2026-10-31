package actors_test

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/actors"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/tools/academic"
	"github.com/lajosdeme/mole/internal/tools/extract"
	"github.com/lajosdeme/mole/internal/tools/fetch"
	"github.com/lajosdeme/mole/internal/tools/search"
)

// mineKinds answers with regular then qualifier claims quoting the chunk's lines.
func mineKinds(regular, qualifiers int) func(string) string {
	return func(prompt string) string {
		var claims []map[string]any
		for _, line := range strings.Split(documentBlock(prompt), "\n") {
			line = strings.TrimSpace(line)
			if len(line) < 40 || len(claims) >= regular+qualifiers {
				continue
			}
			kind, prefix := "claim", "regular: "
			if len(claims) >= regular {
				kind, prefix = "qualifier", "qualifier: "
			}
			claims = append(claims, map[string]any{
				"text": prefix + line[:30], "quote": line, "confidence": 0.8, "kind": kind,
			})
		}
		out, _ := json.Marshal(map[string]any{"claims": claims})
		return string(out)
	}
}

// countKinds counts kept claims by their text prefix.
func countKinds(res *actors.Result) (regular, qualifiers int) {
	for _, c := range res.Claims {
		if strings.HasPrefix(c.Text, "qualifier: ") {
			qualifiers++
		} else {
			regular++
		}
	}
	return regular, qualifiers
}

// TestQualifiersHaveTheirOwnCeiling checks 8 regular and 4 qualifier claims are kept per source.
func TestQualifiersHaveTheirOwnCeiling(t *testing.T) {
	h := newHarness(t, []search.Result{
		{URL: "https://verbose.example/a", Content: longArticle(20), Rank: 1},
	}, nil, mineKinds(9, 5))
	h.actor.Budget = actors.Budget{MaxSources: 1, MaxInputTokens: 1_000_000}

	res, err := h.actor.Run(context.Background(), h.lead)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if regular, qualifiers := countKinds(res); regular != 8 || qualifiers != 4 {
		t.Errorf("kept %d regular and %d qualifier claims, want 8 and 4", regular, qualifiers)
	}
}

// TestASourceAtTheRegularCapStillRecordsItsQualifiers checks later chunks still yield qualifiers.
func TestASourceAtTheRegularCapStillRecordsItsQualifiers(t *testing.T) {
	var (
		mu    sync.Mutex
		calls int
	)
	mine := func(prompt string) string {
		mu.Lock()
		calls++
		first := calls == 1
		mu.Unlock()
		if first {
			return mineKinds(10, 0)(prompt)
		}
		return mineKinds(0, 3)(prompt)
	}
	h := newHarness(t, []search.Result{
		{URL: "https://verbose.example/a", Content: longArticle(300), Rank: 1},
	}, nil, mine)
	h.actor.Budget = actors.Budget{MaxSources: 1, MaxInputTokens: 1_000_000}

	res, err := h.actor.Run(context.Background(), h.lead)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.Stats.Chunks < 3 {
		t.Fatalf("document produced %d chunk(s); the test needs several", res.Stats.Chunks)
	}
	if regular, qualifiers := countKinds(res); regular != 8 || qualifiers != 4 {
		t.Errorf("kept %d regular and %d qualifier claims, want 8 and 4", regular, qualifiers)
	}
}

// TestMinePromptAsksForQualifiersAndNamesTheRootQuestion checks the extraction prompt's content.
func TestMinePromptAsksForQualifiersAndNamesTheRootQuestion(t *testing.T) {
	h := newHarness(t, []search.Result{
		{URL: "https://a.example/p", Content: articleBody, Rank: 1},
	}, nil, nil)
	h.actor.Question = "Is MambaByte better than subword transformers?"
	if _, err := h.actor.Run(context.Background(), h.lead); err != nil {
		t.Fatalf("run: %v", err)
	}
	h.llm.mu.Lock()
	defer h.llm.mu.Unlock()
	if len(h.llm.calls) == 0 {
		t.Fatal("no mining call")
	}
	prompt := h.llm.calls[0].Messages[0].Text
	for _, want := range []string{
		h.actor.Question, h.lead.Query, `"kind":"qualifier"`,
		"wrong", "outdated", "unmeasured", "misattributed", "cited from elsewhere",
		"dataset", "denominator", "years", "population",
	} {
		if !strings.Contains(prompt, want) {
			t.Errorf("mine prompt lacks %q", want)
		}
	}
}

// TestASourceWithBothCapsFullIsNotMinedFurther checks no chunk is mined once both ceilings are reached.
func TestASourceWithBothCapsFullIsNotMinedFurther(t *testing.T) {
	h := newHarness(t, []search.Result{
		{URL: "https://verbose.example/a", Content: longArticle(300), Rank: 1},
	}, nil, mineKinds(10, 5))
	h.actor.Budget = actors.Budget{MaxSources: 1, MaxInputTokens: 1_000_000}

	res, err := h.actor.Run(context.Background(), h.lead)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.Stats.Chunks < 2 {
		t.Fatalf("document produced %d chunk(s); the test needs several", res.Stats.Chunks)
	}
	if n := h.llm.callCount(); n != 1 {
		t.Errorf("%d mining calls, want 1", n)
	}
}

// academicKinds answers abstract and full-text passages with regular then qualifier claims quoting q.
func academicKinds(q string, abstractRegular, abstractQualifiers int) func(string) string {
	return func(prompt string) string {
		if !strings.Contains(prompt, q) {
			return `{"claims":[]}`
		}
		regular, qualifiers := 5, 5
		if strings.Contains(prompt, "Abstract with") {
			regular, qualifiers = abstractRegular, abstractQualifiers
		}
		var claims []map[string]any
		for i := 0; i < regular+qualifiers; i++ {
			kind, prefix := "claim", "regular: "
			if i >= regular {
				kind, prefix = "qualifier", "qualifier: "
			}
			claims = append(claims, map[string]any{
				"text": prefix + string(rune('a'+i)), "quote": q, "confidence": 0.5, "kind": kind,
			})
		}
		out, _ := json.Marshal(map[string]any{"claims": claims})
		return string(out)
	}
}

// newQualifierPaperActor builds an academic actor over one paper whose full text repeats q.
func newQualifierPaperActor(t *testing.T, q string, reply func(string) string) *actors.AcademicActor {
	t.Helper()
	p := paper("A", "10.1234/q", "Abstract with "+q+" inside.", at(2025, time.January, 1))
	p.HTMLURL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC9/"
	prov := &fakeAcademic{kind: academic.KindPubMed, papers: []academic.Paper{p}}
	a, _ := newAcademic(t, []academic.Provider{prov}, reply)
	a.Budget = actors.Budget{MaxSources: 1, MaxClaimsPerSource: 3, MaxQualifiersPerSource: 2}
	a.Fetch = fetcherFunc(func(_ context.Context, u string) (*fetch.Result, error) {
		seg := strings.Repeat("Filler prose that does not answer anything. ", 200)
		body := seg + q + ". " + seg + q + ". " + seg + q + "."
		return &fetch.Result{URL: u, Outcome: fetch.OutcomeOK, StatusCode: 200,
			ContentType: "text/html", Content: []byte("<html><body><article><p>" + body + "</p></article></body></html>")}, nil
	})
	a.Extract = extract.New()
	return a
}

// TestAbstractQualifiersDoNotSuppressEscalation checks abstract qualifiers leave the regular allowance for full text.
func TestAbstractQualifiersDoNotSuppressEscalation(t *testing.T) {
	const q = "a quotable sentence that appears in every passage here"
	a := newQualifierPaperActor(t, q, academicKinds(q, 1, 2))
	res, err := a.Run(context.Background(), core.Lead{ID: "l1", Query: "totally different wording"})
	if err != nil {
		t.Fatal(err)
	}
	if res.Stats.Fetched != 1 {
		t.Errorf("full text was not read (Fetched=%d) though the abstract left regular allowance", res.Stats.Fetched)
	}
}

// TestFullTextKeepsQualifiersWithinWhatTheAbstractLeft checks full-text regular and qualifier caps are counted apart.
func TestFullTextKeepsQualifiersWithinWhatTheAbstractLeft(t *testing.T) {
	const q = "a quotable sentence that appears in every passage here"
	a := newQualifierPaperActor(t, q, academicKinds(q, 1, 1))
	res, err := a.Run(context.Background(), core.Lead{ID: "l1", Query: "totally different wording"})
	if err != nil {
		t.Fatal(err)
	}
	if res.Stats.Fetched != 1 {
		t.Fatalf("tier 1 did not run (Fetched=%d)", res.Stats.Fetched)
	}
	if regular, qualifiers := countKinds(res); regular != 3 || qualifiers != 2 {
		t.Errorf("kept %d regular and %d qualifier claims, want 3 and 2", regular, qualifiers)
	}
}

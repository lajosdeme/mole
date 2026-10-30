package actors_test

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"github.com/lajosdeme/mole/internal/actors"
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

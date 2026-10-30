package actors

import (
	"strings"
	"testing"
)

// TestParseMinedKeepsKind checks the claim kind survives parsing.
func TestParseMinedKeepsKind(t *testing.T) {
	got, err := parseMined(`{"claims":[{"text":"t","quote":"q","confidence":0.5,"kind":"qualifier"}]}`)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(got) != 1 || got[0].Kind != "qualifier" {
		t.Errorf("parsed %+v, want kind qualifier", got)
	}
}

// TestMinePromptOmitsQualifiersWithoutAnAllowance checks qualifier instructions appear only when some are allowed.
func TestMinePromptOmitsQualifiersWithoutAnAllowance(t *testing.T) {
	if p := mineUserPrompt("f", "", "q", 8, 0, "", "", "text"); strings.Contains(p, "qualifier") {
		t.Errorf("prompt with no qualifier allowance mentions qualifiers:\n%s", p)
	}
	if p := mineUserPrompt("f", "", "q", 8, 4, "", "", "text"); !strings.Contains(p, `"kind":"qualifier"`) {
		t.Errorf("prompt with a qualifier allowance does not ask for them:\n%s", p)
	}
}

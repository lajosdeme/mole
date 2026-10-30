package actors

import "testing"

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

package output

import (
	"encoding/json"
	"testing"
	"time"
)

// TestACitationSurvivesBeingStored.
//
// Citations are persisted, because the numbering is assigned over a findings set
// capped at generation time and cannot be recomputed. That makes this struct's
// marshalled shape a storage format, and it was left as whatever Go's field
// names happened to be — so a consumer reading published_at silently never
// matched PublishedAt while N, Source and Quotes survived on encoding/json's
// case-insensitive fallback. The date vanished and nothing said so.
func TestACitationSurvivesBeingStored(t *testing.T) {
	at := time.Date(2025, 3, 11, 0, 0, 0, 0, time.UTC)
	in := []Citation{{
		N: 1, Source: "https://example.test/a",
		Quotes: []string{"a quote"}, PublishedAt: &at,
	}}

	b, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}

	// Decoded through a reader that spells the fields the way the export does,
	// which is the shape that actually failed.
	var out []struct {
		N           int        `json:"n"`
		Source      string     `json:"source"`
		Quotes      []string   `json:"quotes"`
		PublishedAt *time.Time `json:"published_at"`
	}
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 {
		t.Fatalf("%d citation(s) survived", len(out))
	}
	if out[0].N != 1 || out[0].Source == "" || len(out[0].Quotes) != 1 {
		t.Errorf("citation lost a field: %+v", out[0])
	}
	if out[0].PublishedAt == nil {
		t.Fatalf("the publication date did not survive storage: %s", b)
	}
	if !out[0].PublishedAt.Equal(at) {
		t.Errorf("date = %v, want %v", out[0].PublishedAt, at)
	}
}

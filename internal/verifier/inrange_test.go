package verifier

import (
	"testing"

	"github.com/lajosdeme/mole/internal/core"
)

func TestPointInRange(t *testing.T) {
	cases := []struct {
		a, b string
		want bool
	}{
		{"Jambeck et al. (2015) concluded that 4.8 to 12.7 Mt of plastics reached the oceans in 2010 from populations living within 50 km of the coastline.",
			"Scientists think about 8 million metric tons of plastic entered the ocean in 2010.", true},
		{"Around 8 million tonnes of plastic enter the oceans every year.",
			"Between 4.8 and 12.7 million tonnes of plastic enter the ocean each year.", true},
		{"Approximately 8–11 million tons of plastic enter the ocean annually.",
			"Around 10 million tonnes of plastic enter the ocean.", true},
		{"Between 20 and 30 percent of projects fail.", "25% of projects fail.", true},
		// Outside the range.
		{"Between 8 and 11 million tons of plastic enter the ocean annually.",
			"The IUCN estimates that 14 million tons of plastic enter the seas annually.", false},
		// Different units.
		{"Senathirajah et al. predicted 0.1–5 g of microplastics per week.",
			"Mohamed Nor et al. estimated a median of 4.1 μg per week.", false},
		{"Around 33 billion pounds of plastic enter the oceans each year.",
			"Between 4.8 and 12.7 million tonnes of plastic enter the ocean each year.", false},
		// Percent against a count.
		{"Between 20 and 30 percent of projects fail.", "25 projects failed.", false},
		// Point side states a range reaching outside.
		{"Between 4.8 and 12.7 million tonnes enter the ocean.", "5 to 13 million tonnes enter the ocean.", false},
		// No range on either side.
		{"8 million tonnes enter the ocean.", "10 million tonnes enter the ocean.", false},
		// Years alone are not figures.
		{"Between 2010 and 2020 emissions rose.", "In 2015 emissions rose.", false},
	}
	for _, c := range cases {
		if got := PointInRange(c.a, c.b); got != c.want {
			t.Errorf("PointInRange(%q, %q) = %v, want %v", c.a, c.b, got, c.want)
		}
		if got := PointInRange(c.b, c.a); got != c.want {
			t.Errorf("PointInRange(%q, %q) = %v, want %v (swapped)", c.b, c.a, got, c.want)
		}
	}
}

func TestAFigureInsideTheOtherRangeIsNotAContradiction(t *testing.T) {
	batch := []Pair{
		{A: &core.Claim{ID: "a", Text: "Between 4.8 and 12.7 million tonnes of plastic enter the ocean each year."},
			B: &core.Claim{ID: "b", Text: "Around 8 million tonnes of plastic enter the oceans every year."}},
		{A: &core.Claim{ID: "c", Text: "Between 8 and 11 million tons of plastic enter the ocean annually."},
			B: &core.Claim{ID: "d", Text: "14 million tons of plastic enter the seas annually."}},
	}
	raw := `{"verdicts":[{"pair":1,"same_quantity":true,"relation":"contradicts","confidence":0.9},` +
		`{"pair":2,"same_quantity":true,"relation":"contradicts","confidence":0.9}]}`
	judged, _, err := parseVerdicts(raw, batch)
	if err != nil {
		t.Fatal(err)
	}
	if judged[0].Relation != RelNeither {
		t.Errorf("point inside range judged %s, want neither", judged[0].Relation)
	}
	if judged[1].Relation != RelContradicts {
		t.Errorf("point outside range judged %s, want contradicts", judged[1].Relation)
	}
}

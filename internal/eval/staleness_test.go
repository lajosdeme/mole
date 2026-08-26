package eval

import (
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/core"
)

func contradictEdge() *core.ClaimEdge {
	return &core.ClaimEdge{Kind: core.EdgeContradicts}
}

// TestNoDatedClaimSaysSoRatherThanAskingTheReaderToCheck.
//
// The rule needs PublishedAt on both claims. When nothing carried one it did not run
// at all, and reporting "0 of N resolved" implies it ran and found nothing — a
// different fact, and the one that hid §11.2 being inert across a whole corpus.
func TestNoDatedClaimSaysSoRatherThanAskingTheReaderToCheck(t *testing.T) {
	claims := []*core.Claim{{ID: "c1"}, {ID: "c2"}}
	m := stalenessSeparation(claims, []*core.ClaimEdge{contradictEdge()})

	if m.Status != Blocked {
		t.Errorf("status = %v; with no dated claim the rule could not run", m.Status)
	}
	// Reason, not Detail: Blocked is what "the rule could not run" means, and Reason
	// is the field that carries why.
	if !contains(m.Reason, "0 of 2") {
		t.Errorf("reason does not count the dated claims: %q", m.Reason)
	}
}

// TestDatedClaimsThatResolveNothingIsADifferentReport: here the rule DID run.
func TestDatedClaimsThatResolveNothingIsADifferentReport(t *testing.T) {
	at := time.Now()
	claims := []*core.Claim{{ID: "c1", PublishedAt: &at}, {ID: "c2", PublishedAt: &at}}
	m := stalenessSeparation(claims, []*core.ClaimEdge{contradictEdge()})

	if m.Status != Measured {
		t.Errorf("status = %v; the dates were there, so this is a measurement", m.Status)
	}
	if m.Reason != "" {
		t.Errorf("blamed missing dates when 2 claims carried them: %q", m.Reason)
	}
}

// TestSupersedesIsStillCountedWhenDatesExist.
func TestSupersedesIsStillCountedWhenDatesExist(t *testing.T) {
	at := time.Now()
	claims := []*core.Claim{{ID: "c1", PublishedAt: &at}}
	m := stalenessSeparation(claims, []*core.ClaimEdge{
		{Kind: core.EdgeSupersedes}, contradictEdge(),
	})
	if m.Value != 1 {
		t.Errorf("value = %v, want 1 supersedes edge", m.Value)
	}
	if m.Status != Measured {
		t.Errorf("status = %v, want Measured", m.Status)
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (func() bool {
		for i := 0; i+len(sub) <= len(s); i++ {
			if s[i:i+len(sub)] == sub {
				return true
			}
		}
		return false
	})()
}

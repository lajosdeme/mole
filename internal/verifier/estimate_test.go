package verifier

import (
	"testing"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/pricing"
)

// TestBatchReservationCoversARealCall checks that an 8-pair batch's
// reservation covers the real call's cost without being wildly oversized.
func TestBatchReservationCoversARealCall(t *testing.T) {
	tbl := pricing.NewEmptyTable()
	tbl.Register("m", pricing.RateConfig{Input: 1, Output: 5}.Rates())
	v := &Verifier{Model: "m", Pricing: tbl}

	const pairs = 8
	got := v.estimateBatch(core.BudgetUSD, pairs)

	real, err := tbl.Cost("m", pricing.Usage{
		InputTokens:  350 + pairs*120,
		OutputTokens: pairs * 60,
	})
	if err != nil {
		t.Fatalf("Cost: %v", err)
	}

	if got < real.USDMicros {
		t.Errorf("estimateBatch(%d pairs) = %d micros, want >= the real call's %d — "+
			"a reservation smaller than the call it covers trips the overshoot detector",
			pairs, got, real.USDMicros)
	}
	if max := 4 * real.USDMicros; got > max {
		t.Errorf("estimateBatch(%d pairs) = %d micros, want <= 4x the real call's %d (%d) — "+
			"too generous starves every other batch under the share cap",
			pairs, got, real.USDMicros, max)
	}
}

// TestTokenModeEstimateIgnoresPricing checks that in token mode the estimate
// ignores the pricing table.
func TestTokenModeEstimateIgnoresPricing(t *testing.T) {
	v := &Verifier{Model: "m", Pricing: pricing.NewEmptyTable()}
	const pairs = 8
	got := v.estimateBatch(core.BudgetTokens, pairs)
	want := int64(2 * (350 + pairs*180))
	if got != want {
		t.Errorf("estimateBatch(tokens, %d) = %d, want %d", pairs, got, want)
	}
}

// TestFreeModelStillReservesSomething checks that a model registered at $0
// still reserves a nonzero amount.
func TestFreeModelStillReservesSomething(t *testing.T) {
	tbl := pricing.NewEmptyTable()
	tbl.RegisterFree("m")
	v := &Verifier{Model: "m", Pricing: tbl}

	got := v.estimate(core.BudgetUSD, 1000, 500)
	if got != 1 {
		t.Errorf("estimate for a free model = %d, want 1", got)
	}
}

// TestUnpricedModelUsesConservativeRate checks that a Verifier with no
// pricing table falls back to the $5/$25 per-MTok conservative rate.
func TestUnpricedModelUsesConservativeRate(t *testing.T) {
	v := &Verifier{Model: "m"} // no Pricing at all
	const in, out = int64(1000), int64(500)

	got := v.estimate(core.BudgetUSD, in, out)

	fallback := pricing.NewEmptyTable()
	fallback.Register("fallback", pricing.RateConfig{Input: 5, Output: 25}.Rates())
	want, err := fallback.Cost("fallback", pricing.Usage{InputTokens: in, OutputTokens: out})
	if err != nil {
		t.Fatalf("Cost: %v", err)
	}

	if got != want.USDMicros {
		t.Errorf("estimate with no pricing table = %d, want the $5/$25 fallback rate %d",
			got, want.USDMicros)
	}
}

// TestUnknownModelInANonNilTableAlsoFallsBack checks that a model missing
// from a non-nil pricing table also falls back to the conservative rate.
func TestUnknownModelInANonNilTableAlsoFallsBack(t *testing.T) {
	tbl := pricing.NewEmptyTable() // "m" never registered
	v := &Verifier{Model: "m", Pricing: tbl}
	const in, out = int64(1000), int64(500)

	got := v.estimate(core.BudgetUSD, in, out)

	fallback := pricing.NewEmptyTable()
	fallback.Register("fallback", pricing.RateConfig{Input: 5, Output: 25}.Rates())
	want, err := fallback.Cost("fallback", pricing.Usage{InputTokens: in, OutputTokens: out})
	if err != nil {
		t.Fatalf("Cost: %v", err)
	}
	if got != want.USDMicros {
		t.Errorf("estimate with an unknown model = %d, want the fallback rate %d", got, want.USDMicros)
	}
}

// TestGroundCallEstimateMatchesTokenTotal checks that groundCallEstimate's
// token total in token mode matches the input/output split used for pricing.
func TestGroundCallEstimateMatchesTokenTotal(t *testing.T) {
	v := &Verifier{Model: "m", Pricing: pricing.NewEmptyTable()}
	got := v.groundCallEstimate(core.BudgetTokens)
	want := int64(2 * ((MaxPassageChars+MaxClaimChars)/4 + 400 + groundMaxTokens))
	if got != want {
		t.Errorf("groundCallEstimate(tokens) = %d, want %d", got, want)
	}
}

package verifier

import (
	"testing"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/pricing"
)

// TestBatchReservationCoversARealCall prices an 8-pair batch against the same
// table a real session would carry, and checks the reservation sits between
// the actual call's cost and a generous multiple of it — covering §8.2's hold
// without reserving so much the share cap starves every other batch.
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

// TestTokenModeEstimateIgnoresPricing. In token mode the reservation IS the
// budgeted quantity; pricing has no vote in it, priced or not.
func TestTokenModeEstimateIgnoresPricing(t *testing.T) {
	v := &Verifier{Model: "m", Pricing: pricing.NewEmptyTable()}
	const pairs = 8
	got := v.estimateBatch(core.BudgetTokens, pairs)
	want := int64(2 * (350 + pairs*180))
	if got != want {
		t.Errorf("estimateBatch(tokens, %d) = %d, want %d", pairs, got, want)
	}
}

// TestFreeModelStillReservesSomething. A model registered at $0 (a self-hosted
// deployment, RegisterFree) must still take a nonzero hold: a reservation of 0
// reserves nothing, and the ledger's ReserveVerify treats it as "nothing to
// verify with".
func TestFreeModelStillReservesSomething(t *testing.T) {
	tbl := pricing.NewEmptyTable()
	tbl.RegisterFree("m")
	v := &Verifier{Model: "m", Pricing: tbl}

	got := v.estimate(core.BudgetUSD, 1000, 500)
	if got != 1 {
		t.Errorf("estimate for a free model = %d, want 1", got)
	}
}

// TestUnpricedModelUsesConservativeRate. A Verifier with no pricing table at
// all — or a table that does not know this model — cannot look up a real rate,
// so it reserves against fallbackRates: $5/MTok in, $25/MTok out. Deliberately
// high, so an unidentified model errs toward over-reserving rather than
// under-reserving and tripping the overshoot detector.
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

// TestUnknownModelInANonNilTableAlsoFallsBack. The table exists but has never
// heard of this model — the common case once a provider renames a model out
// from under a config. Same fallback applies, logged once rather than erroring
// the reservation.
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

// TestGroundCallEstimateMatchesTokenTotal. The USD/token split must not change
// the total tokens reserved in token mode — only how the total is PRICED in USD
// mode, by giving input and output their own rates instead of one blended one.
func TestGroundCallEstimateMatchesTokenTotal(t *testing.T) {
	v := &Verifier{Model: "m", Pricing: pricing.NewEmptyTable()}
	got := v.groundCallEstimate(core.BudgetTokens)
	want := int64(2 * ((MaxPassageChars+MaxClaimChars)/4 + 400 + groundMaxTokens))
	if got != want {
		t.Errorf("groundCallEstimate(tokens) = %d, want %d", got, want)
	}
}

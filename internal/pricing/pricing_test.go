package pricing_test

import (
	"errors"
	"testing"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/pricing"
)

func TestOpus5RatesAreExact(t *testing.T) {
	tbl := pricing.NewTable()

	// Claude Opus 5: $5 / MTok input, $25 / MTok output.
	// 1M input tokens must be exactly $5.000000, with no float drift.
	c, err := tbl.Cost("claude-opus-5", pricing.Usage{InputTokens: 1_000_000})
	if err != nil {
		t.Fatalf("cost: %v", err)
	}
	if want := int64(5 * core.MicrosPerUSD); c.USDMicros != want {
		t.Errorf("1M input = %d micros, want %d", c.USDMicros, want)
	}

	c, err = tbl.Cost("claude-opus-5", pricing.Usage{OutputTokens: 1_000_000})
	if err != nil {
		t.Fatalf("cost: %v", err)
	}
	if want := int64(25 * core.MicrosPerUSD); c.USDMicros != want {
		t.Errorf("1M output = %d micros, want %d", c.USDMicros, want)
	}
}

// TestCacheTokensPriceDifferently is why Cost carries a breakdown rather than
// one flat token count: reads bill at 0.1x input and writes at 1.25x, so the
// same token volume can differ in price by more than an order of magnitude.
func TestCacheTokensPriceDifferently(t *testing.T) {
	tbl := pricing.NewTable()
	const n = 1_000_000

	plain, err := tbl.Cost("claude-opus-5", pricing.Usage{InputTokens: n})
	if err != nil {
		t.Fatal(err)
	}
	read, err := tbl.Cost("claude-opus-5", pricing.Usage{CacheReadTokens: n})
	if err != nil {
		t.Fatal(err)
	}
	write, err := tbl.Cost("claude-opus-5", pricing.Usage{CacheWriteTokens: n})
	if err != nil {
		t.Fatal(err)
	}
	write1h, err := tbl.Cost("claude-opus-5", pricing.Usage{CacheWriteTokens: n, CacheWriteTTL1h: true})
	if err != nil {
		t.Fatal(err)
	}

	if want := plain.USDMicros / 10; read.USDMicros != want {
		t.Errorf("cache read = %d, want %d (0.1x input)", read.USDMicros, want)
	}
	if want := plain.USDMicros * 125 / 100; write.USDMicros != want {
		t.Errorf("cache write 5m = %d, want %d (1.25x input)", write.USDMicros, want)
	}
	if want := plain.USDMicros * 2; write1h.USDMicros != want {
		t.Errorf("cache write 1h = %d, want %d (2x input)", write1h.USDMicros, want)
	}

	// All four record the same token volume even though they cost differently.
	for name, c := range map[string]core.Cost{"plain": plain, "read": read, "write": write} {
		if c.TotalTokens() != n {
			t.Errorf("%s: total tokens = %d, want %d", name, c.TotalTokens(), n)
		}
	}
}

func TestUnknownModelIsAnErrorNotFree(t *testing.T) {
	tbl := pricing.NewTable()

	// A silent $0 for an unregistered model would make the budget ceiling
	// unenforceable for exactly the model someone forgot to register.
	_, err := tbl.Cost("gpt-definitely-not-a-claude-model", pricing.Usage{InputTokens: 1000})
	var unknown *pricing.ErrUnknownModel
	if !errors.As(err, &unknown) {
		t.Fatalf("error = %v, want ErrUnknownModel", err)
	}

	tbl.SetStrict(false)
	c, err := tbl.Cost("gpt-definitely-not-a-claude-model", pricing.Usage{InputTokens: 1000})
	if err != nil {
		t.Fatalf("non-strict cost: %v", err)
	}
	if c.USDMicros != 0 {
		t.Errorf("non-strict unknown model cost = %d, want 0", c.USDMicros)
	}
	if c.InputTokens != 1000 {
		t.Errorf("token count lost for unknown model: %d", c.InputTokens)
	}
}

// TestSelfHostedModelCostsNothingButCountsTokens is the case that justifies
// token-mode budgeting: no dollars change hands, but the tokens are still real
// and still need bounding.
func TestSelfHostedModelCostsNothingButCountsTokens(t *testing.T) {
	tbl := pricing.NewEmptyTable()
	tbl.RegisterFree("local/llama-70b")

	c, err := tbl.Cost("local/llama-70b", pricing.Usage{InputTokens: 9000, OutputTokens: 1200})
	if err != nil {
		t.Fatalf("cost: %v", err)
	}
	if c.USDMicros != 0 {
		t.Errorf("self-hosted cost = %d, want 0", c.USDMicros)
	}
	if got := c.BudgetAmount(core.BudgetTokens); got != 10_200 {
		t.Errorf("token budget amount = %d, want 10200", got)
	}
}

func TestContextSuffixIsNormalized(t *testing.T) {
	tbl := pricing.NewTable()
	plain, err := tbl.Cost("claude-opus-5", pricing.Usage{InputTokens: 1000})
	if err != nil {
		t.Fatal(err)
	}
	tagged, err := tbl.Cost("claude-opus-5[1m]", pricing.Usage{InputTokens: 1000})
	if err != nil {
		t.Fatalf("tagged model: %v", err)
	}
	if plain.USDMicros != tagged.USDMicros {
		t.Errorf("tagged model priced differently: %d vs %d", tagged.USDMicros, plain.USDMicros)
	}
}

func TestRoundingDoesNotUnderBill(t *testing.T) {
	tbl := pricing.NewTable()

	// One Haiku input token is 1000 nano-dollars = exactly 1 micro-dollar.
	c, err := tbl.Cost("claude-haiku-4-5", pricing.Usage{InputTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	if c.USDMicros != 1 {
		t.Errorf("1 haiku input token = %d micros, want 1", c.USDMicros)
	}

	// A single Opus cache-read token is 500 nano = 0.5 micro. Truncation would
	// bill 0 and, repeated across a session, understate spend.
	c, err = tbl.Cost("claude-opus-5", pricing.Usage{CacheReadTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	if c.USDMicros != 1 {
		t.Errorf("1 opus cache-read token = %d micros, want 1 (rounded up, not truncated)", c.USDMicros)
	}
}

func TestRegisterOverridesDefault(t *testing.T) {
	tbl := pricing.NewTable()
	tbl.Register("claude-opus-5", pricing.Rates{Input: 1, Output: 1})

	c, err := tbl.Cost("claude-opus-5", pricing.Usage{InputTokens: 1_000_000})
	if err != nil {
		t.Fatal(err)
	}
	if want := int64(1000); c.USDMicros != want {
		t.Errorf("overridden rate produced %d micros, want %d", c.USDMicros, want)
	}
}

// TestADatedSnapshotPricesAsItsBaseModel.
//
// Anthropic ships an alias and a dated snapshot for the same weights at the same
// price. The table registers the alias, so a config naming the snapshot found no
// rates — and an unpriced model does not merely cost zero, it makes --usd refuse
// to start: "no price is registered for claude-haiku-4-5-20251001, so every call
// would cost nothing". That is exactly what happened on this project's first
// hosted run.
func TestADatedSnapshotPricesAsItsBaseModel(t *testing.T) {
	tbl := pricing.NewTable()
	u := pricing.Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000}

	base, err := tbl.Cost("claude-haiku-4-5", u)
	if err != nil {
		t.Fatalf("base model is unpriced: %v", err)
	}
	dated, err := tbl.Cost("claude-haiku-4-5-20251001", u)
	if err != nil {
		t.Fatalf("dated snapshot is unpriced: %v", err)
	}
	if dated.USDMicros != base.USDMicros {
		t.Errorf("dated snapshot costs %d micros, base costs %d — same weights, same price",
			dated.USDMicros, base.USDMicros)
	}
	// And the rate itself is right: $1/MTok in + $5/MTok out = $6.
	if want := int64(6_000_000); base.USDMicros != want {
		t.Errorf("haiku-4-5 at 1M in + 1M out = %d micros, want %d", base.USDMicros, want)
	}

	// A suffix that is not a date must NOT be stripped. Folding "-32768" onto a
	// base name would price one model at another's rates, which is worse than
	// reporting it unknown.
	for _, m := range []string{
		"claude-haiku-4-5-32768",   // context size, not a date
		"claude-haiku-4-5-2025100", // seven digits
		"claude-haiku-4-5-99999999",
		"claude-haiku-4-5-2025ab01",
	} {
		if _, ok := tbl.Lookup(m); ok {
			t.Errorf("%q resolved to a price; only a real -YYYYMMDD suffix may be stripped", m)
		}
	}

	// An exact registration must win over the fallback, so a snapshot that ever
	// does diverge in price can be pinned.
	tbl.Register("claude-haiku-4-5-20251001", pricing.Rates{Input: 9_000, Output: 9_000})
	if r, _ := tbl.Lookup("claude-haiku-4-5-20251001"); r.Input != 9_000 {
		t.Errorf("explicit registration lost to the base-model fallback (input=%d)", r.Input)
	}
}

// TestAnUnregisteredModelFallsBackToTheConfiguredOne.
//
// The model a provider ANSWERS with is not the model you configured, and it is
// theirs to rename without telling anyone. DeepSeek did exactly that: a run
// configured as "deepseek-chat" started coming back as "deepseek-flash", which
// no table knew, so every call priced at $0.00 and a --usd ceiling could not
// bind. The pre-flight check did not catch it, because it validates the name in
// the config while the ledger prices the name in the response.
//
// A rate that is an estimate is a budget that still works. Zero is a budget
// that has been switched off by a third party.
func TestAnUnregisteredModelFallsBackToTheConfiguredOne(t *testing.T) {
	tbl := pricing.NewTable()
	tbl.SetFallbackModel("deepseek-chat")

	const n = 1_000_000
	want, err := tbl.Cost("deepseek-chat", pricing.Usage{InputTokens: n})
	if err != nil {
		t.Fatal(err)
	}
	got, err := tbl.Cost("deepseek-some-name-nobody-registered", pricing.Usage{InputTokens: n})
	if err != nil {
		t.Fatalf("an unknown model with a fallback configured is still an error: %v", err)
	}
	if got.USDMicros != want.USDMicros {
		t.Errorf("unknown model priced at %d micros, want the configured model's %d",
			got.USDMicros, want.USDMicros)
	}
	if got.USDMicros == 0 {
		t.Error("priced at zero, so a --usd ceiling could never bind")
	}
}

// TestNoFallbackKeepsTheErrorStrict, so a deployment that registered nothing is
// still told rather than quietly billed at a guess.
func TestNoFallbackKeepsTheErrorStrict(t *testing.T) {
	tbl := pricing.NewTable()
	if _, err := tbl.Cost("nobody-registered-this", pricing.Usage{InputTokens: 1}); err == nil {
		t.Error("an unknown model with no fallback was accepted")
	}
}

// TestAFallbackNobodyRegisteredIsRefusedAtOnce, rather than at the first call.
func TestAFallbackNobodyRegisteredIsRefusedAtOnce(t *testing.T) {
	tbl := pricing.NewTable()
	if err := tbl.SetFallbackModel("not-a-model"); err == nil {
		t.Error("an unpriced fallback was accepted")
	}
}

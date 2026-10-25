package pricing_test

import (
	"errors"
	"testing"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/llm"
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

// TestEveryModelMoleCanChooseIsPriced.
//
// internal/llm/detect.go selects a model when the user names none — gpt-4o from
// an OPENAI_API_KEY, deepseek-chat from a DEEPSEEK_API_KEY, and so on. A model
// mole picks FOR you and then cannot price is a configuration mole built and
// then refuses to budget: checkUSDIsEnforceable rejects the run, and --usd is
// unavailable to a user who never chose anything.
//
// The rule this pins is narrow on purpose. Not "price every model in the
// world" — this table's job is never to under-bill, and a list nobody can
// maintain goes stale in the direction of billing too little. Only: mole prices
// what mole itself selects.
func TestEveryModelMoleCanChooseIsPriced(t *testing.T) {
	tbl := pricing.NewTable()
	for _, m := range llm.AutoSelectedModels() {
		if _, ok := tbl.Lookup(m); !ok {
			t.Errorf("detect.go can select %q and the table cannot price it", m)
		}
	}
}

// TestRatesComeFromThePageTheVendorPublishes.
//
// Internally a rate is nano-dollars per token, which nobody should have to know
// to configure one. The config takes dollars per MILLION tokens — the number
// printed on the vendor's own pricing page, copied across without arithmetic.
func TestRatesComeFromThePageTheVendorPublishes(t *testing.T) {
	// $2.50/MTok in, $10.00/MTok out, the way OpenAI prints it.
	got := pricing.RateConfig{Input: 2.50, Output: 10.00}.Rates()

	want, _ := pricing.NewTable().Lookup("gpt-4o")
	if got.Input != want.Input || got.Output != want.Output {
		t.Errorf("declared %+v, built-in %+v — the same page should give the same rate",
			got, want)
	}
}

// TestDeclaredCacheRatesDefaultToTheContract, so two numbers are enough for a
// provider that bills cache the ordinary way.
func TestDeclaredCacheRatesDefaultToTheContract(t *testing.T) {
	r := pricing.RateConfig{Input: 3, Output: 15}.Rates()
	if r.CacheRead != 300 {
		t.Errorf("CacheRead = %d, want a tenth of input", r.CacheRead)
	}
	if r.CacheWrite != 3750 || r.CacheWrite1h != 6000 {
		t.Errorf("cache writes = %d/%d, want 1.25x and 2x input", r.CacheWrite, r.CacheWrite1h)
	}

	// And a provider that bills differently says so.
	ex := pricing.RateConfig{Input: 3, Output: 15, CacheRead: 1.5}.Rates()
	if ex.CacheRead != 1500 {
		t.Errorf("an explicit cache read was overridden: %d", ex.CacheRead)
	}
}

// TestADeclaredRateBeatsTheBuiltIn.
//
// A vendor changing a price must be correctable without waiting for a release —
// which is the situation that started all this, when a rename left every call
// priced at zero and nobody could fix it from outside the binary.
func TestADeclaredRateBeatsTheBuiltIn(t *testing.T) {
	tbl := pricing.NewTable()
	if err := tbl.Declare(map[string]pricing.RateConfig{
		"gpt-4o": {Input: 99, Output: 99},
	}); err != nil {
		t.Fatal(err)
	}
	r, _ := tbl.Lookup("gpt-4o")
	if r.Input != 99_000 {
		t.Errorf("Input = %d, want the declared rate to win", r.Input)
	}
}

// TestANonsenseRateIsRefusedAtLoad, not at the first call: discovering it
// mid-run means discovering it after the run has started spending.
func TestANonsenseRateIsRefusedAtLoad(t *testing.T) {
	for name, rc := range map[string]pricing.RateConfig{
		"negative input":  {Input: -1, Output: 1},
		"negative output": {Input: 1, Output: -1},
		"no input rate":   {Output: 1},
	} {
		if err := pricing.NewTable().Declare(map[string]pricing.RateConfig{"m": rc}); err == nil {
			t.Errorf("%s was accepted", name)
		}
	}
}

// TestDeclaringAPriceMakesUsdUsable, which is the point of the whole feature: an
// exotic provider is budgetable because its rates were declared, and the
// fallback then covers a response naming something unexpected.
func TestDeclaringAPriceMakesUsdUsable(t *testing.T) {
	tbl := pricing.NewTable()
	const model = "some-vendor/some-model-v3"

	if _, ok := tbl.Lookup(model); ok {
		t.Fatal("the premise is wrong; that model is registered")
	}
	if err := tbl.Declare(map[string]pricing.RateConfig{model: {Input: 1, Output: 2}}); err != nil {
		t.Fatal(err)
	}
	if err := tbl.SetFallbackModel(model); err != nil {
		t.Fatalf("a declared model cannot be a fallback: %v", err)
	}
	c, err := tbl.Cost("whatever-the-api-actually-answered", pricing.Usage{InputTokens: 1_000_000})
	if err != nil {
		t.Fatal(err)
	}
	if c.USDMicros == 0 {
		t.Error("priced at zero; --usd would not bind")
	}
}

// TestCurrentClaudeRates validates that current Claude models are priced correctly.
func TestCurrentClaudeRates(t *testing.T) {
	tbl := pricing.NewTable()

	// Claude Sonnet 5: $2/MTok input, $10/MTok output.
	// 1M input + 1M output = $2 + $10 = $12.
	c, err := tbl.Cost("claude-sonnet-5", pricing.Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000})
	if err != nil {
		t.Fatalf("claude-sonnet-5 cost: %v", err)
	}
	if want := int64(12_000_000); c.USDMicros != want {
		t.Errorf("sonnet-5 1M in + 1M out = %d micros, want %d", c.USDMicros, want)
	}

	// Claude Opus 5-5: $4/MTok input, $20/MTok output.
	// 1M input + 1M output = $4 + $20 = $24.
	c, err = tbl.Cost("claude-opus-5-5", pricing.Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000})
	if err != nil {
		t.Fatalf("claude-opus-5-5 cost: %v", err)
	}
	if want := int64(24_000_000); c.USDMicros != want {
		t.Errorf("opus-5-5 1M in + 1M out = %d micros, want %d", c.USDMicros, want)
	}

	// Opus 5-5: 1M cache-read tokens at 200 nano-dollars/token = 200_000 micros.
	c, err = tbl.Cost("claude-opus-5-5", pricing.Usage{CacheReadTokens: 1_000_000})
	if err != nil {
		t.Fatalf("opus-5-5 cache read: %v", err)
	}
	if want := int64(200_000); c.USDMicros != want {
		t.Errorf("opus-5-5 1M cache-read = %d micros, want %d", c.USDMicros, want)
	}

	// Claude Sonnet 5-5: $2/MTok input, $10/MTok output.
	// 1M input + 1M output = $2 + $10 = $12.
	c, err = tbl.Cost("claude-sonnet-5-5", pricing.Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000})
	if err != nil {
		t.Fatalf("claude-sonnet-5-5 cost: %v", err)
	}
	if want := int64(12_000_000); c.USDMicros != want {
		t.Errorf("sonnet-5-5 1M in + 1M out = %d micros, want %d", c.USDMicros, want)
	}

	// Claude Fable 5-1: $10/MTok input, $50/MTok output.
	// 1M input + 1M output = $10 + $50 = $60.
	c, err = tbl.Cost("claude-fable-5-1", pricing.Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000})
	if err != nil {
		t.Fatalf("claude-fable-5-1 cost: %v", err)
	}
	if want := int64(60_000_000); c.USDMicros != want {
		t.Errorf("fable-5-1 1M in + 1M out = %d micros, want %d", c.USDMicros, want)
	}
}

// TestHaikuFiveFiveTiersByPromptLength validates prompt-length tiered pricing.
func TestHaikuFiveFiveTiersByPromptLength(t *testing.T) {
	tbl := pricing.NewTable()

	// Short prompt (≤100k): 50k input + 1k output = 50*100 + 1*500 = 5_500_000 nano = 5_500 micros.
	c, err := tbl.Cost("claude-haiku-5-5", pricing.Usage{InputTokens: 50_000, OutputTokens: 1_000})
	if err != nil {
		t.Fatalf("haiku-5-5 short prompt cost: %v", err)
	}
	if want := int64(5_500); c.USDMicros != want {
		t.Errorf("haiku-5-5 short (50k in + 1k out) = %d micros, want %d", c.USDMicros, want)
	}

	// Long prompt (>100k): 150k input + 1k output uses long tier: 150*500 + 1*2500 = 77_500_000 nano = 77_500 micros.
	c, err = tbl.Cost("claude-haiku-5-5", pricing.Usage{InputTokens: 150_000, OutputTokens: 1_000})
	if err != nil {
		t.Fatalf("haiku-5-5 long prompt cost: %v", err)
	}
	if want := int64(77_500); c.USDMicros != want {
		t.Errorf("haiku-5-5 long (150k in + 1k out) = %d micros, want %d", c.USDMicros, want)
	}

	// Mixed cache: 90k input + 20k cache-read + 1k output; total prompt = 110k > 100k, uses long tier.
	// 90*500 + 20*50 + 1*2500 = 45_000_000 + 1_000_000 + 2_500_000 = 48_500_000 nano = 48_500 micros.
	c, err = tbl.Cost("claude-haiku-5-5", pricing.Usage{InputTokens: 90_000, CacheReadTokens: 20_000, OutputTokens: 1_000})
	if err != nil {
		t.Fatalf("haiku-5-5 mixed cache cost: %v", err)
	}
	if want := int64(48_500); c.USDMicros != want {
		t.Errorf("haiku-5-5 mixed (90k in + 20k cache + 1k out) = %d micros, want %d", c.USDMicros, want)
	}
}

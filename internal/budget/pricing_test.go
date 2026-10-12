package budget_test

import (
	"context"
	"testing"

	"github.com/lajosdeme/mole/internal/budget"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/pricing"
	"github.com/lajosdeme/mole/internal/store"
)

// TestEveryRoleIsBilled.
//
// The planner and the verifier recorded tokens and no dollars for as long as
// both existed. Pricing lived in whichever function happened to build the call
// — the executor's miner and the report writer did it, the planner and the
// verifier did not — so two of the four parts of mole that spend money spent it
// for free.
//
// Measured on one Quick run: 141 verifier calls and 2 planner calls carrying
// 127,054 tokens, all ledgered at $0.00, against $0.0614 recorded. A quarter of
// the bill was invisible, and `--usd` could not bind on the phase that makes
// most of the calls.
//
// So the table is consulted HERE, in the one place every call passes through on
// its way to the database, and this test walks all four roles because the bug
// was never about a role — it was about which call site remembered.
func TestEveryRoleIsBilled(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)

	table := pricing.NewTable()
	table.Register("priced-model", pricing.Rates{Input: 1000, Output: 2000})
	cfg := noEscrow()
	cfg.Pricing = table

	l := budget.New(db, cfg)
	s := newSession(t, l, 10_000_000)

	for _, role := range []core.Role{
		core.RolePlanner, core.RoleExecutor, core.RoleVerifier, core.RoleOutput,
	} {
		t.Run(string(role), func(t *testing.T) {
			r, err := l.Reserve(ctx, s.ID, 1_000_000)
			if err != nil {
				t.Fatalf("reserve: %v", err)
			}
			res, err := l.Settle(ctx, r, []core.ToolCall{{
				SessionID: s.ID,
				Role:      role,
				Type:      core.CallLLM,
				Model:     "priced-model",
				// Tokens and no dollars: what every caller records, and what
				// only two of them used to convert.
				Cost: core.Cost{InputTokens: 1000, OutputTokens: 500},
			}})
			if err != nil {
				t.Fatalf("settle: %v", err)
			}
			// 1000 in at 1000 nano + 500 out at 2000 nano = 2,000,000 nano.
			const want = 2_000
			if got := res.Charged; got != want {
				t.Errorf("%s settled %d micros, want %d — its tokens were billed "+
					"by the provider either way", role, got, want)
			}
		})
	}
}

// TestACallThatAlreadyKnowsItsPriceIsLeftAlone.
//
// A search costs dollars and no tokens: the provider bills per query, and the
// charge arrives on the call already. Pricing it again from a token count of
// zero would silently zero it, and search is the largest single line on a run —
// $0.040 of the $0.083 one measured Quick run really cost.
func TestACallThatAlreadyKnowsItsPriceIsLeftAlone(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)
	l := budget.New(db, noEscrow())
	s := newSession(t, l, 10_000_000)

	r, err := l.Reserve(ctx, s.ID, 1_000_000)
	if err != nil {
		t.Fatalf("reserve: %v", err)
	}
	res, err := l.Settle(ctx, r, []core.ToolCall{{
		SessionID: s.ID,
		Role:      core.RoleExecutor,
		Type:      core.CallSearch,
		Cost:      core.Cost{USDMicros: 8_000},
	}})
	if err != nil {
		t.Fatalf("settle: %v", err)
	}
	if res.Charged != 8_000 {
		t.Errorf("a search settled %d micros, want the 8000 it arrived with", res.Charged)
	}
}

// TestAnUnpricedModelStillRecordsItsTokens.
//
// Unchanged behaviour, moved: a model with no rate spends real tokens, and
// zeroing those too would break token-mode budgets as well as dollar ones.
// `mole doctor` is what reports the gap, before a run rather than after it.
func TestAnUnpricedModelStillRecordsItsTokens(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)
	l := budget.New(db, noEscrow())
	s := newSession(t, l, 10_000_000)

	r, err := l.Reserve(ctx, s.ID, 1_000_000)
	if err != nil {
		t.Fatalf("reserve: %v", err)
	}
	res, err := l.Settle(ctx, r, []core.ToolCall{{
		SessionID: s.ID,
		Role:      core.RoleVerifier,
		Type:      core.CallLLM,
		Model:     "a-model-nobody-registered",
		Cost:      core.Cost{InputTokens: 1000, OutputTokens: 500},
	}})
	if err != nil {
		t.Fatalf("settle: %v", err)
	}
	if res.Charged != 0 {
		t.Errorf("an unpriced model settled %d micros; a rate was invented", res.Charged)
	}

	var tokens int64
	if err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		calls, err := q.ListToolCalls(ctx, s.ID, 10_000)
		if err != nil {
			return err
		}
		for _, c := range calls {
			tokens += c.Cost.TotalTokens()
		}
		return nil
	}); err != nil {
		t.Fatalf("list: %v", err)
	}
	if tokens != 1500 {
		t.Errorf("recorded %d tokens, want 1500 — the tokens were spent whether "+
			"or not the model had a price", tokens)
	}
}

// TestASessionLeavesNoTokensUnbilled.
//
// The shape of one measured Quick run — a planner call, executor mining, a
// search, a verifier batch, the report — settled together, and then every row
// is read back. A row carrying tokens and no dollars is the bug this file
// exists for, whichever role put it there.
func TestASessionLeavesNoTokensUnbilled(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)

	table := pricing.NewTable()
	table.Register("m", pricing.Rates{Input: 140, Output: 280, CacheRead: 3})
	cfg := noEscrow()
	cfg.Pricing = table

	l := budget.New(db, cfg)
	s := newSession(t, l, 10_000_000)

	r, err := l.Reserve(ctx, s.ID, 5_000_000)
	if err != nil {
		t.Fatalf("reserve: %v", err)
	}
	llmCall := func(role core.Role, in, out int64) core.ToolCall {
		return core.ToolCall{
			SessionID: s.ID, Role: role, Type: core.CallLLM, Model: "m",
			Cost: core.Cost{InputTokens: in, OutputTokens: out},
		}
	}
	if _, err := l.Settle(ctx, r, []core.ToolCall{
		llmCall(core.RolePlanner, 745, 179),
		llmCall(core.RoleExecutor, 115_159, 16_701),
		{SessionID: s.ID, Role: core.RoleExecutor, Type: core.CallSearch,
			Cost: core.Cost{USDMicros: 40_000}},
		llmCall(core.RoleVerifier, 97_078, 29_052),
		llmCall(core.RoleOutput, 1_974, 1_217),
	}); err != nil {
		t.Fatalf("settle: %v", err)
	}

	if err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		calls, err := q.ListToolCalls(ctx, s.ID, 10_000)
		if err != nil {
			return err
		}
		if len(calls) != 5 {
			t.Fatalf("%d rows stored, want 5", len(calls))
		}
		for _, c := range calls {
			if c.Cost.TotalTokens() > 0 && c.Cost.USDMicros == 0 {
				t.Errorf("%s/%s recorded %d tokens and no money",
					c.Role, c.Type, c.Cost.TotalTokens())
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}

	// 214,956 in at 140 nano and 47,149 out at 280 nano is 43,295 micros, plus
	// the search's 40,000. The run really cost 83,295 micros, and 21,879 of that
	// was the planner and the verifier — the quarter that used to be free.
	var spent int64
	if err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		cost, err := q.SumCosts(ctx, s.ID)
		spent = cost.USDMicros
		return err
	}); err != nil {
		t.Fatal(err)
	}
	const want = 83_295
	if spent != want {
		t.Errorf("session spent %d micros, want %d", spent, want)
	}
}

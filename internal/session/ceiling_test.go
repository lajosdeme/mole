package session_test

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/session"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/store/sqlite"
)

func ceilingDB(t *testing.T) *sqlite.DB {
	t.Helper()
	db, err := sqlite.Open(filepath.Join(t.TempDir(), "t.db"), sqlite.Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.Migrate(context.Background()); err != nil {
		t.Fatal(err)
	}
	return db
}

func created(t *testing.T, spec session.Spec) *core.Session {
	t.Helper()
	db := ceilingDB(t)
	r := &session.Runner{Store: db}
	sess, err := r.Create(context.Background(), spec)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	var got *core.Session
	if err := db.WithTx(context.Background(), func(ctx context.Context, tx store.Tx) error {
		var e error
		got, e = tx.GetSession(ctx, sess.ID)
		return e
	}); err != nil {
		t.Fatal(err)
	}
	return got
}

// TestAnExplicitToolCallCeilingIsHonoured.
func TestAnExplicitToolCallCeilingIsHonoured(t *testing.T) {
	got := created(t, session.Spec{
		Question: "q", Mode: core.ModeReport, BudgetUnit: core.BudgetUSD,
		Budget: 1_000_000, MaxLeads: 9, MaxSources: 5, Timeout: time.Minute,
		MaxToolCalls: 1234,
	})
	if got.MaxToolCalls != 1234 {
		t.Errorf("MaxToolCalls = %d, want the 1234 the caller asked for", got.MaxToolCalls)
	}
}

// TestTheDefaultToolCallCeilingIsABackstopNotABudget.
//
// It was maxLeads * sources * 4 — an assertion that a lead costs four calls per
// source, so twenty calls at five sources. Measured across fourteen recorded
// sessions the median is 42.6 and the maximum 62, so the ceiling cut runs in
// half: one measured run wanted 431 calls and was stopped at 240, delivering 134
// claims where the same question uncapped delivered 272.
//
// What stops a run is meant to be the clock and the money. This is the backstop
// for when the money estimate is wrong, so it must sit far above what a healthy
// run needs rather than inside it.
func TestTheDefaultToolCallCeilingIsABackstopNotABudget(t *testing.T) {
	got := created(t, session.Spec{
		Question: "q", Mode: core.ModeReport, BudgetUnit: core.BudgetUSD,
		Budget: 1_000_000, MaxLeads: 9, MaxSources: 5, Timeout: time.Minute,
	})

	// The worst per-lead rate yet measured, with room above it.
	const worstMeasuredCallsPerLead = 62
	least := int64(9 * worstMeasuredCallsPerLead)
	if got.MaxToolCalls <= least {
		t.Errorf("default ceiling %d does not clear %d — the worst measured rate for 9 leads",
			got.MaxToolCalls, least)
	}
	// And it is still a ceiling.
	if got.MaxToolCalls == 0 {
		t.Error("no tool-call ceiling at all; a runaway has nothing to stop it")
	}
}

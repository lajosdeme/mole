package sqlite_test

import (
	"context"
	"database/sql"
	"strings"
	"testing"
	"time"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/store"
)

// TestLeadQueriesRoundTrip checks a lead's search queries survive insert, get and lease.
func TestLeadQueriesRoundTrip(t *testing.T) {
	ctx := context.Background()
	db, _ := open(t)
	insertSession(t, db, "s_q", 1_000_000)

	want := []string{"EV fire rate per 100,000", "1,530 fires source"}
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.InsertLead(ctx, &core.Lead{ID: "l_q", SessionID: "s_q",
			ActorType: core.ActorWeb, Query: "How often do EVs burn?", Queries: want})
	}); err != nil {
		t.Fatal(err)
	}

	var got, leased *core.Lead
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		var err error
		if got, err = tx.GetLead(ctx, "l_q"); err != nil {
			return err
		}
		leased, err = tx.LeaseNextLead(ctx, "s_q", "w", time.Now().Add(time.Minute))
		return err
	}); err != nil {
		t.Fatal(err)
	}
	for name, l := range map[string]*core.Lead{"get": got, "lease": leased} {
		if l == nil || strings.Join(l.Queries, "|") != strings.Join(want, "|") {
			t.Errorf("%s: queries = %+v, want %q", name, l, want)
		}
	}
}

// TestLeadWithoutQueriesReadsBackEmpty checks a row written without the queries column has none.
func TestLeadWithoutQueriesReadsBackEmpty(t *testing.T) {
	ctx := context.Background()
	db, path := open(t)
	insertSession(t, db, "s_old", 1_000_000)

	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer raw.Close()
	now := time.Now().UnixMicro()
	if _, err := raw.Exec(`INSERT INTO leads (id, session_id, actor_type, query, depth, priority,
		status, created_at, updated_at) VALUES ('l_old','s_old','web','old lead',0,0,'queued',?,?)`,
		now, now); err != nil {
		t.Fatal(err)
	}

	var got *core.Lead
	if err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		got, err = q.GetLead(ctx, "l_old")
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if len(got.Queries) != 0 {
		t.Errorf("queries = %q, want none", got.Queries)
	}
}

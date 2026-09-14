package mcpserver_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/mcpserver"
	"github.com/lajosdeme/mole/internal/session"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/store/sqlite"
)

// wedgedLLM accepts the call and never answers — the failure mode a read
// deadline exists for, and the one a connect timeout does not cover.
type wedgedLLM struct{ entered chan struct{} }

func (p *wedgedLLM) Name() string               { return "wedged" }
func (p *wedgedLLM) ModelFor(t llm.Tier) string { return "wedged-model" }
func (p *wedgedLLM) Complete(ctx context.Context, _ llm.Request) (*llm.Response, error) {
	select {
	case p.entered <- struct{}{}:
	default:
	}
	<-ctx.Done()
	return nil, ctx.Err()
}

// TestAWedgedProviderCannotHangAnAskForever.
//
// AskTimeout was written into the session spec as MaxWallClock and read by
// nothing: an ask never enters the executor loop, which is what checks that
// field. So a provider that accepted the connection and then stopped responding
// held the ask's reservation — budget neither spent nor available — until the
// abandonment sweep half an hour later, with the caller blocked the whole time.
func TestAWedgedProviderCannotHangAnAskForever(t *testing.T) {
	db := openTestDB(t)
	sid := seedFinishedSession(t, db)

	wedged := &wedgedLLM{entered: make(chan struct{}, 1)}
	deps := mcpserver.Deps{Store: db, LLM: wedged, AskTimeout: 150 * time.Millisecond}

	done := make(chan mcpserver.AskOut, 1)
	go func() {
		out, err := deps.Ask(context.Background(), mcpserver.AskIn{
			SessionID: sid, Question: "what does MambaByte achieve on PG-19?",
		})
		if err == nil {
			done <- out
		}
		close(done)
	}()

	select {
	case out, ok := <-done:
		if !ok {
			t.Fatal("Ask returned an error; it should degrade to the evidence listing")
		}
		if out.Degraded == "" {
			t.Fatal("a timed-out provider produced no degradation reason")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Ask did not return; the timeout is not enforced")
	}

	// The precondition, asserted rather than assumed. Without it this test passes
	// on a retrieval short-circuit — the question sharing no content word with any
	// claim, so Answer returns before the provider is ever called — and says
	// nothing whatever about the timeout. That is exactly what it did first.
	select {
	case <-wedged.entered:
	default:
		t.Fatal("the provider was never called; this test did not exercise the timeout")
	}

	// And the hold is resolved, not left for the sweep.
	var held int64
	if err := db.Read(context.Background(), func(ctx context.Context, q store.Queries) error {
		sessions, err := q.ListSessions(ctx, 50)
		if err != nil {
			return err
		}
		for _, s := range sessions {
			if s.Mode == core.ModeAsk {
				held += s.Held
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if held != 0 {
		t.Fatalf("%d micro-dollars still held after a timed-out ask", held)
	}
}

// TestTheServerReportsTheBinarysVersion. It was hardcoded "0.1.0", which stayed
// put through every build and told a client something false. Asserted through a
// real handshake, because the version only matters at the point a client reads
// it out of serverInfo.
func TestTheServerReportsTheBinarysVersion(t *testing.T) {
	for _, tc := range []struct{ set, want string }{
		{"1.4.2-abc", "1.4.2-abc"},
		{"", "dev"}, // unset says so, rather than inventing a number
	} {
		srv := mcpserver.New(mcpserver.Deps{Store: openTestDB(t), Version: tc.set})

		st, ct := mcp.NewInMemoryTransports()
		ctx := context.Background()
		if _, err := srv.Connect(ctx, st, nil); err != nil {
			t.Fatal(err)
		}
		cs, err := mcp.NewClient(&mcp.Implementation{Name: "test", Version: "0"}, nil).Connect(ctx, ct, nil)
		if err != nil {
			t.Fatal(err)
		}
		info := cs.InitializeResult().ServerInfo
		if info == nil {
			t.Fatal("the handshake carried no serverInfo")
		}
		if info.Version != tc.want {
			t.Fatalf("serverInfo.version=%q with Deps.Version=%q, want %q", info.Version, tc.set, tc.want)
		}
		_ = cs.Close()
	}
}

// --- fixtures ---------------------------------------------------------------

func openTestDB(t *testing.T) *sqlite.DB {
	t.Helper()
	db, err := sqlite.Open(t.TempDir()+"/t.db", sqlite.Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.Migrate(context.Background()); err != nil {
		t.Fatal(err)
	}
	return db
}

// seedFinishedSession writes a terminal session with one claim, directly. The
// rig in mcpserver_test.go drives the whole tool surface to get one; this needs
// only something for an ask to read.
func seedFinishedSession(t *testing.T, db *sqlite.DB) string {
	t.Helper()
	ctx := context.Background()

	runner := &session.Runner{Store: db, Owner: "test"}
	sess, err := runner.Create(ctx, session.Spec{
		Question: "what does MambaByte achieve on PG-19", Mode: core.ModeReport,
		BudgetUnit: core.BudgetUSD, Budget: core.MicrosPerUSD,
		MaxSources: 3, MaxDepth: 1, MaxLeads: 4, Timeout: time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}

	lead := core.Lead{ID: core.NewLeadID(), SessionID: sess.ID,
		ActorType: core.ActorWeb, Query: "q", Status: core.LeadDone}
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		if err := tx.InsertLead(ctx, &lead); err != nil {
			return err
		}
		if err := tx.InsertClaims(ctx, []core.Claim{{
			SessionID: sess.ID, LeadID: lead.ID,
			Text:   "MambaByte reaches 0.930 bits per byte on PG-19.",
			Source: "https://arxiv.org/abs/2401.13660",
			Quote:  "0.930 bits per byte", Confidence: 0.8,
		}}); err != nil {
			return err
		}
		return tx.FinishSession(ctx, sess.ID, core.StatusDone, "")
	}); err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(sess.ID) == "" {
		t.Fatal("no session id")
	}
	return sess.ID
}

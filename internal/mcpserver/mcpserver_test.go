package mcpserver_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/lajosdeme/mole/internal/actors"
	"github.com/lajosdeme/mole/internal/compute/connector"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/mcpserver"
	"github.com/lajosdeme/mole/internal/pricing"
	"github.com/lajosdeme/mole/internal/session"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/store/sqlite"
	"github.com/lajosdeme/mole/internal/tools/extract"
	"github.com/lajosdeme/mole/internal/tools/fetch"
	"github.com/lajosdeme/mole/internal/tools/search"
)

// --- stubs -----------------------------------------------------------------

type idlePlanner struct{}

func (idlePlanner) Name() string               { return "stub" }
func (idlePlanner) ModelFor(t llm.Tier) string { return "stub-model" }
func (idlePlanner) Complete(context.Context, llm.Request) (*llm.Response, error) {
	return &llm.Response{
		Text:  `{"questions":[],"rationale":"none"}`,
		Model: "stub-model",
		Usage: llm.Usage{InputTokens: 1, OutputTokens: 1},
	}, nil
}

type emptySearch struct{}

func (emptySearch) Kind() search.Kind { return search.KindTavily }
func (emptySearch) Search(context.Context, string, search.Options) (*search.Response, error) {
	return &search.Response{}, nil
}

// loudSearch answers with more results than were asked for, each carrying a long
// provider-supplied title and snippet — which is what an untrusted provider
// relaying an untrusted page looks like.
type loudSearch struct {
	asked  search.Options
	cost   core.Cost
	titleN int
}

func (l *loudSearch) Kind() search.Kind { return search.KindTavily }

func (l *loudSearch) Search(_ context.Context, q string, o search.Options) (*search.Response, error) {
	l.asked = o
	n := l.titleN
	if n == 0 {
		n = 4000
	}
	var results []search.Result
	for i := 0; i < 25; i++ {
		results = append(results, search.Result{
			Title:   strings.Repeat("T", n),
			URL:     fmt.Sprintf("https://example.org/%d", i),
			Snippet: strings.Repeat("S", n),
		})
	}
	return &search.Response{Query: q, Results: results, Cost: l.cost}, nil
}

// connectToolkitSearch is the toolkit with a search provider that answers.
func connectToolkitSearch(t *testing.T, prov *loudSearch) *rig {
	t.Helper()
	r := connectToolkitActor(t, &actors.WebActor{
		LLM:     idlePlanner{},
		Search:  prov,
		Fetch:   stubFetcher{body: testPage},
		Extract: extract.New(),
	})
	r.pageURL = "https://example.org/review"
	return r
}

// --- harness ----------------------------------------------------------------

type rig struct {
	client *mcp.ClientSession
	db     store.Store
	sup    *session.Supervisor
	// pageURL is a local page the toolkit's fetch tool can read, so those tests
	// exercise the real fetch and extract path without touching the network.
	pageURL string
	// lastJSON is the raw result of the most recent call, so a test can assert on
	// what a client would actually receive rather than on a decoded struct that
	// silently drops fields.
	lastJSON string
}

// tools lists what the server advertises, which is how the toolkit tests check
// that the surface is gated rather than inspecting the Deps they set themselves.
func (r *rig) tools(t *testing.T) []string {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	res, err := r.client.ListTools(ctx, nil)
	if err != nil {
		t.Fatalf("list tools: %v", err)
	}
	var out []string
	for _, tool := range res.Tools {
		out = append(out, tool.Name)
	}
	return out
}

// toolDefs is tools() with the descriptions attached, for the tests that assert a
// tool tells the model something it cannot be made to do — advice only helps if it
// reaches the model, and the description is the only channel that carries it.
func (r *rig) toolDefs(t *testing.T) []*mcp.Tool {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	res, err := r.client.ListTools(ctx, nil)
	if err != nil {
		t.Fatalf("list tools: %v", err)
	}
	return res.Tools
}

func timeNow() time.Time { return time.Now().UTC() }

// expireDocument backdates a stored document past its TTL, so a test can check what
// happens when retention has taken text a caller still holds an id for.
func expireDocument(t *testing.T, r *rig, docID string) {
	t.Helper()
	if err := r.db.WithTx(context.Background(), func(ctx context.Context, tx store.Tx) error {
		return tx.ExpireDocumentForTest(ctx, docID)
	}); err != nil {
		t.Fatal(err)
	}
}

// connectToolkit brings up the server with the toolkit surface enabled and a local
// HTTP page to fetch.
func connectToolkit(t *testing.T) *rig {
	t.Helper()
	page := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprint(w, `<html><head><title>A review</title></head><body><article>`+
			`<p>Across ten randomised trials, intermittent fasting reduced fasting `+
			`glucose in adults with prediabetes. Effects on HbA1c were inconsistent.</p>`+
			`<p>Limitations include short follow-up and heterogeneous eating windows.</p>`+
			`</article></body></html>`)
	}))
	t.Cleanup(page.Close)

	r := connectToolkitActor(t, &actors.WebActor{
		LLM:     idlePlanner{},
		Search:  emptySearch{},
		Fetch:   fetch.NewHTTP(fetch.Config{UserAgent: "mole-test"}, fetch.Options{}),
		Extract: extract.New(),
	})
	r.pageURL = page.URL
	return r
}

// connectToolkitStubFetch is connectToolkit with the network step stubbed.
//
// The real fetcher refuses a loopback address on a high port, which is the egress
// guard doing its job and is asserted directly in TestFetchIsBehindTheEgressGuard.
// Tests about what happens AFTER a successful fetch — storage, fencing, the
// document count — need a fetch that succeeds, so they replace the fetcher and keep
// the real extractor.
func connectToolkitStubFetch(t *testing.T, body string) *rig {
	t.Helper()
	r := connectToolkitActor(t, &actors.WebActor{
		LLM:     idlePlanner{},
		Search:  emptySearch{},
		Fetch:   stubFetcher{body: body},
		Extract: extract.New(),
	})
	r.pageURL = "https://example.org/review"
	return r
}

// csvConnectorForTest ingests the same fixture connectToolkitLocal uses, for the
// tests that call the compute pipeline directly rather than through a tool.
func csvConnectorForTest(t *testing.T) connector.Connector {
	t.Helper()
	dir := t.TempDir()
	csv := filepath.Join(dir, "tickets.csv")
	var b strings.Builder
	b.WriteString("region,tickets,spend,note\n")
	for i := 0; i < 40; i++ {
		region := "north"
		if i%2 == 1 {
			region = "south"
		}
		fmt.Fprintf(&b, "%s,%d,%d,\"Shipment held at the depot until the quarter opened\"\n",
			region, i%9+1, 100+i)
	}
	if err := os.WriteFile(csv, []byte(b.String()), 0o600); err != nil {
		t.Fatal(err)
	}
	c, err := connector.Ingest(context.Background(), "sales", csv, filepath.Join(dir, "s.db"))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// connectToolkitLocal registers a small CSV so the aggregate tools have real data
// behind them — the same ingest path `mole connect add` uses, so the profile and
// the free-text flags are the real ones rather than a fixture's idea of them.
func connectToolkitLocal(t *testing.T) *rig {
	t.Helper()
	dir := t.TempDir()
	csv := filepath.Join(dir, "tickets.csv")
	var b strings.Builder
	b.WriteString("region,tickets,spend,note\n")
	for i := 0; i < 40; i++ {
		region := "north"
		if i%2 == 1 {
			region = "south"
		}
		fmt.Fprintf(&b, "%s,%d,%d,\"Shipment held at the depot until the quarter opened\"\n",
			region, i%9+1, 100+i)
	}
	if err := os.WriteFile(csv, []byte(b.String()), 0o600); err != nil {
		t.Fatal(err)
	}
	c, err := connector.Ingest(context.Background(), "sales", csv, filepath.Join(dir, "s.db"))
	if err != nil {
		t.Fatal(err)
	}

	connectorsForTest = registryOf{c}
	t.Cleanup(func() { connectorsForTest = nil })
	return connectToolkitActor(t, &actors.WebActor{LLM: idlePlanner{}, Search: emptySearch{}})
}

// connectorsForTest is set by connectToolkitLocal, the same way toolkitOn is: Deps
// are built inside connectActor before a rig exists to hang them off.
var connectorsForTest interface{ List() []connector.Connector }

type registryOf []connector.Connector

func (r registryOf) List() []connector.Connector { return r }

// redirectFetcher answers as if the request had been redirected elsewhere, which
// is what the real fetcher reports in FinalURL after following one.
type redirectFetcher struct{ body, finalURL string }

func (f redirectFetcher) Fetch(_ context.Context, _ string) (*fetch.Result, error) {
	return &fetch.Result{
		Content: []byte(f.body), ContentType: "text/html",
		Outcome: fetch.OutcomeOK, StatusCode: 200, FinalURL: f.finalURL,
	}, nil
}

// connectToolkitRedirect is connectToolkitStubFetch where every fetch lands
// somewhere other than the URL that was asked for.
func connectToolkitRedirect(t *testing.T, body string) *rig {
	t.Helper()
	r := connectToolkitActor(t, &actors.WebActor{
		LLM:     idlePlanner{},
		Search:  emptySearch{},
		Fetch:   redirectFetcher{body: body, finalURL: "https://elsewhere.example/served"},
		Extract: extract.New(),
	})
	r.pageURL = "https://example.org/review"
	return r
}

type stubFetcher struct{ body string }

func (s stubFetcher) Fetch(_ context.Context, rawURL string) (*fetch.Result, error) {
	return &fetch.Result{
		Content:     []byte(s.body),
		ContentType: "text/html",
		Outcome:     fetch.OutcomeOK,
		StatusCode:  200,
		FinalURL:    rawURL,
	}, nil
}

// connect wires a real MCP client to the server over an in-memory transport.
//
// A real client rather than calling the handlers directly, because the contract
// this slice ships is a wire format: schemas inferred from Go structs, arguments
// unmarshalled by the SDK, results marshalled back. Calling the Go functions
// would test none of that and would stay green through a field rename that
// breaks every caller.
func connect(t *testing.T, maxUSD int64) *rig {
	t.Helper()
	return connectWith(t, maxUSD, nil)
}

func connectWith(t *testing.T, maxUSD int64, answerer llm.Provider) *rig {
	t.Helper()
	return connectFull(t, maxUSD, 0, answerer)
}

func connectCeilings(t *testing.T, maxUSD, maxTokens int64) *rig {
	t.Helper()
	return connectFull(t, maxUSD, maxTokens, nil)
}

// connectHeld is connectFull with a planner that blocks until the returned
// function is called, so a test that needs a session to be RUNNING can hold it
// there instead of hoping.
//
// TestAskRefusesARunningSession used to check sup.Running() and skip if the
// session had already finished. That made it a coin flip: it passed alone,
// skipped sometimes, and failed under load when the session finished between the
// check and the ask — the wrong assertion firing rather than a real regression.
func connectHeld(t *testing.T, maxUSD int64, answerer llm.Provider) (*rig, func()) {
	t.Helper()
	held := make(chan struct{})
	var once sync.Once
	release := func() { once.Do(func() { close(held) }) }
	// Always released, so a failing test cannot leave the supervisor's shutdown
	// waiting on a planner nobody let go.
	t.Cleanup(release)
	return connectActor(t, maxUSD, 0, answerer,
		&actors.WebActor{LLM: heldPlanner{held}, Search: emptySearch{}}), release
}

// heldPlanner answers only after the test says so.
type heldPlanner struct{ held chan struct{} }

func (heldPlanner) Name() string               { return "stub" }
func (heldPlanner) ModelFor(t llm.Tier) string { return "stub-model" }
func (p heldPlanner) Complete(ctx context.Context, _ llm.Request) (*llm.Response, error) {
	select {
	case <-p.held:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return &llm.Response{
		Text:  `{"questions":[],"rationale":"none"}`,
		Model: "stub-model",
		Usage: llm.Usage{InputTokens: 1, OutputTokens: 1},
	}, nil
}

func connectFull(t *testing.T, maxUSD, maxTokens int64, answerer llm.Provider) *rig {
	t.Helper()
	return connectActor(t, maxUSD, maxTokens, answerer,
		&actors.WebActor{LLM: idlePlanner{}, Search: emptySearch{}})
}

// connectToolkitActor is connectActor with the toolkit surface registered and a
// real fetcher and extractor, so those tools run the path they run in production.
func connectToolkitActor(t *testing.T, actor *actors.WebActor) *rig {
	t.Helper()
	return connectToolkitCapped(t, 0, 0, actor)
}

// connectToolkitCapped is connectToolkitActor with the daemon's per-session
// ceilings set, for the tests that check a toolkit session cannot exceed them.
func connectToolkitCapped(t *testing.T, maxUSD, maxTokens int64, actor *actors.WebActor) *rig {
	t.Helper()
	toolkitOn = true
	t.Cleanup(func() { toolkitOn = false })
	return connectActor(t, maxUSD, maxTokens, nil, actor)
}

// toolkitOn gates registration for the test rig. A package-level flag rather than
// another parameter on four constructors, and reset by Cleanup so it cannot leak
// between tests.
var toolkitOn bool

func connectActor(
	t *testing.T, maxUSD, maxTokens int64, answerer llm.Provider, actor *actors.WebActor,
) *rig {
	t.Helper()
	db, err := sqlite.Open(filepath.Join(t.TempDir(), "t.db"), sqlite.Options{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.Migrate(context.Background()); err != nil {
		t.Fatalf("migrate: %v", err)
	}

	runner := &session.Runner{Store: db, Actor: actor, Owner: "test"}
	sup := session.NewSupervisor(runner, 4, nil)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = sup.Shutdown(ctx)
	})

	srv := mcpserver.New(mcpserver.Deps{
		Supervisor:       sup,
		Toolkit:          toolkitOn,
		Connectors:       connectorsForTest,
		Search:           actor.Search,
		Fetch:            actor.Fetch,
		Extract:          actor.Extract,
		Store:            db,
		LLM:              answerer,
		Pricing:          stubPricing(),
		MaxSessionUSD:    maxUSD,
		MaxSessionTokens: maxTokens,
		MaxSources:       3,
		MaxDepth:         1,
		MaxLeads:         4,
		Timeout:          time.Minute,
	})

	st, ct := mcp.NewInMemoryTransports()
	ctx := context.Background()
	if _, err := srv.Connect(ctx, st, nil); err != nil {
		t.Fatalf("server connect: %v", err)
	}
	cs, err := mcp.NewClient(&mcp.Implementation{Name: "test", Version: "0"}, nil).Connect(ctx, ct, nil)
	if err != nil {
		t.Fatalf("client connect: %v", err)
	}
	t.Cleanup(func() { _ = cs.Close() })

	return &rig{client: cs, db: db, sup: sup}
}

// call invokes a tool and decodes its structured result.
func (r *rig) call(t *testing.T, name string, args any, out any) *mcp.CallToolResult {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	res, err := r.client.CallTool(ctx, &mcp.CallToolParams{Name: name, Arguments: args})
	if err != nil {
		t.Fatalf("%s: transport error: %v", name, err)
	}
	if res.IsError {
		return res
	}
	if raw, err := json.Marshal(res.StructuredContent); err == nil {
		r.lastJSON = string(raw)
	}
	if out != nil {
		raw, err := json.Marshal(res.StructuredContent)
		if err != nil {
			t.Fatalf("%s: re-marshal: %v", name, err)
		}
		if err := json.Unmarshal(raw, out); err != nil {
			t.Fatalf("%s: decode into %T: %v\n%s", name, out, err, raw)
		}
	}
	return res
}

func errText(res *mcp.CallToolResult) string {
	var b strings.Builder
	for _, c := range res.Content {
		if tc, ok := c.(*mcp.TextContent); ok {
			b.WriteString(tc.Text)
		}
	}
	return b.String()
}

// stubPricing prices the test model.
//
// Without it an ask costs exactly zero: an unknown model records its tokens and
// prices them at nothing, which is the right production behaviour and makes any
// assertion about spend vacuous.
func stubPricing() *pricing.Table {
	tbl := pricing.NewTable()
	tbl.Register("stub-model", pricing.Rates{Input: 1_000, Output: 5_000})
	return tbl
}

// --- tests ------------------------------------------------------------------

// TestTheAdvertisedToolsAreTheOnesWeAgreedToShip.
//
// §5.1 lists nine tools; three of them need M8 or M9. Advertising a tool that
// errors is worse than not advertising it, because an agent plans around its
// existence — it will tell a user it can analyze their database and then fail.
func TestTheAdvertisedToolsAreTheOnesWeAgreedToShip(t *testing.T) {
	r := connect(t, 0)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	res, err := r.client.ListTools(ctx, nil)
	if err != nil {
		t.Fatalf("list tools: %v", err)
	}
	var got []string
	for _, tool := range res.Tools {
		got = append(got, tool.Name)
	}
	sort.Strings(got)

	want := []string{
		"research.ask", "research.cancel", "research.report", "research.result",
		"research.sessions.list", "research.status",
	}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("advertised tools:\n  got  %v\n  want %v", got, want)
	}
	// No separate absence loop: the exact-match above already fails if any of
	// research.dataset, research.analyze_local or research.connectors.list
	// appears. A second check that cannot fire unless the first one already did
	// reads like a property and pins nothing.
	// Every tool needs a description an agent can route on.
	for _, tool := range res.Tools {
		if len(tool.Description) < 40 {
			t.Errorf("%s has a %d-character description; an agent cannot tell when to call it",
				tool.Name, len(tool.Description))
		}
	}
}

// TestReportStartsASessionAndReturnsImmediately is §5.3's async contract.
func TestReportStartsASessionAndReturnsImmediately(t *testing.T) {
	r := connect(t, 0)

	var out mcpserver.ReportOut
	res := r.call(t, "research.report", map[string]any{
		"prompt": "what is a byte-level language model",
		"budget": map[string]any{"unit": "usd", "amount": "0.50"},
	}, &out)
	if res.IsError {
		t.Fatalf("report failed: %s", errText(res))
	}
	if !strings.HasPrefix(out.SessionID, "s_") {
		t.Errorf("session_id = %q, want an s_ id", out.SessionID)
	}
	if out.Budget != 500_000 {
		t.Errorf("budget = %d micro-dollars, want 500000 ($0.50)", out.Budget)
	}
	if out.Note == "" {
		t.Error("no note telling the caller to poll; the async contract is invisible otherwise")
	}

	// The session is real and findable by the other tools.
	var st mcpserver.StatusOut
	r.call(t, "research.status", map[string]any{"session_id": out.SessionID}, &st)
	if st.SessionID != out.SessionID {
		t.Errorf("status returned a different session: %q", st.SessionID)
	}
	if st.Budget != out.Budget {
		t.Errorf("status budget %d != report budget %d", st.Budget, out.Budget)
	}
}

// TestTheBudgetCeilingIsTheDaemonsNotTheCallers.
//
// On the command line the budget is a number a person typed. Over MCP it is a
// number an agent chose, and nothing else bounds it. This is the one limit the
// caller cannot raise.
func TestTheBudgetCeilingIsTheDaemonsNotTheCallers(t *testing.T) {
	r := connect(t, 1_000_000) // $1.00

	res := r.call(t, "research.report", map[string]any{
		"prompt": "expensive question",
		"budget": map[string]any{"unit": "usd", "amount": "50.00"},
	}, nil)
	if !res.IsError {
		t.Fatal("a $50 request was accepted under a $1 ceiling")
	}
	msg := errText(res)
	for _, want := range []string{"$1.00", "daemon.max-session-usd"} {
		if !strings.Contains(msg, want) {
			t.Errorf("the refusal does not mention %q, so the caller cannot act on it: %s", want, msg)
		}
	}

	// Nothing was created. A refused request must not leave a session behind.
	var list mcpserver.ListOut
	r.call(t, "research.sessions.list", map[string]any{}, &list)
	if len(list.Sessions) != 0 {
		t.Errorf("a refused request created %d session(s)", len(list.Sessions))
	}

	// And a request under the ceiling still works.
	var ok mcpserver.ReportOut
	if res := r.call(t, "research.report", map[string]any{
		"prompt": "affordable question",
		"budget": map[string]any{"unit": "usd", "amount": "0.25"},
	}, &ok); res.IsError {
		t.Fatalf("a request under the ceiling was refused: %s", errText(res))
	}
}

// TestABudgetWithoutAUnitIsRefused. §8 makes the unit semantically load-bearing:
// a caller that means tokens and gets dollars overspends by six orders of
// magnitude, so there is no bare-number form here any more than on the CLI.
//
// Two layers refuse these, and the split is worth knowing. The SDK infers the
// input schema from the Go struct, and neither Budget field carries omitempty —
// so a MISSING unit or amount is rejected by schema validation before any of our
// code runs. The remaining cases (unknown unit, zero, negative, unparseable)
// reach parseBudget, and mutating either of its checks fails this test.
func TestABudgetWithoutAUnitIsRefused(t *testing.T) {
	r := connect(t, 0)
	for name, budget := range map[string]map[string]any{
		"no unit":      {"amount": "1.00"},
		"no amount":    {"unit": "usd"},
		"unknown unit": {"unit": "euros", "amount": "1.00"},
		"zero":         {"unit": "usd", "amount": "0"},
		"negative":     {"unit": "usd", "amount": "-5"},
		"not a number": {"unit": "tokens", "amount": "lots"},
	} {
		res := r.call(t, "research.report", map[string]any{
			"prompt": "a question", "budget": budget,
		}, nil)
		if !res.IsError {
			t.Errorf("%s: accepted %v", name, budget)
		}
	}
}

// TestResultTruncationIsVisible.
//
// The result goes straight into an agent's context and nothing upstream bounds
// how many claims a session gathers. Silently returning a prefix is how a caller
// concludes a session found less than it did — the same defect as a search that
// quietly drops results.
func TestResultTruncationIsVisible(t *testing.T) {
	r := connect(t, 0)

	var rep mcpserver.ReportOut
	r.call(t, "research.report", map[string]any{
		"prompt": "a question",
		"budget": map[string]any{"unit": "usd", "amount": "0.50"},
	}, &rep)

	// More claims than the cap, written directly: the point is the boundary, not
	// how the claims got there.
	total := mcpserver.MaxClaimsReturned + 25
	writeClaims(t, r.db, rep.SessionID, total)

	var out mcpserver.ResultOut
	r.call(t, "research.result", map[string]any{"session_id": rep.SessionID}, &out)

	if len(out.Claims) != mcpserver.MaxClaimsReturned {
		t.Errorf("returned %d claims, want the cap of %d", len(out.Claims), mcpserver.MaxClaimsReturned)
	}
	if out.TotalClaims != total {
		t.Errorf("total_claims = %d, want %d — the caller cannot tell what it is missing",
			out.TotalClaims, total)
	}
	if !out.Truncated {
		t.Error("truncated is false on a truncated result")
	}
	if !strings.Contains(out.Note, "of") {
		t.Errorf("the note does not say how much was withheld: %q", out.Note)
	}
}

// TestUnknownSessionsAreRefusedByEveryTool. A daemon is long-lived and ids come
// from an agent's context, which may be stale or invented.
func TestUnknownSessionsAreRefusedByEveryTool(t *testing.T) {
	r := connect(t, 0)
	for _, tool := range []string{"research.status", "research.result", "research.cancel"} {
		res := r.call(t, tool, map[string]any{"session_id": "s_NOPE"}, nil)
		if !res.IsError {
			t.Errorf("%s accepted an unknown session id", tool)
		}
		if !strings.Contains(errText(res), "s_NOPE") {
			t.Errorf("%s does not name the id it rejected: %s", tool, errText(res))
		}
	}
	// An empty id is a caller bug, not a lookup miss, and should say so.
	res := r.call(t, "research.status", map[string]any{"session_id": ""}, nil)
	if !res.IsError {
		t.Error("an empty session_id was accepted")
	}
}

// TestCancellingAFinishedSessionIsNotAnError.
//
// "No such session" and "that one already finished" are different answers, and
// an agent polling a session that completed between its status call and its
// cancel call should get the second.
func TestCancellingAFinishedSessionIsNotAnError(t *testing.T) {
	r := connect(t, 0)

	var rep mcpserver.ReportOut
	r.call(t, "research.report", map[string]any{
		"prompt": "a question",
		"budget": map[string]any{"unit": "usd", "amount": "0.50"},
	}, &rep)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	_, _ = r.sup.Wait(ctx, rep.SessionID)
	for i := 0; i < 100 && len(r.sup.Running()) > 0; i++ {
		time.Sleep(20 * time.Millisecond)
	}

	var out mcpserver.CancelOut
	res := r.call(t, "research.cancel", map[string]any{"session_id": rep.SessionID}, &out)
	if res.IsError {
		t.Fatalf("cancelling a finished session errored: %s", errText(res))
	}
	if out.Cancelled {
		t.Error("reported cancelling a session that had already finished")
	}
	if !strings.Contains(out.Note, "not running") {
		t.Errorf("the note does not explain why nothing happened: %q", out.Note)
	}
}

func TestSessionsListFindsWhatWasStarted(t *testing.T) {
	r := connect(t, 0)
	var rep mcpserver.ReportOut
	r.call(t, "research.report", map[string]any{
		"prompt": "findable question",
		"budget": map[string]any{"unit": "usd", "amount": "0.50"},
	}, &rep)

	var list mcpserver.ListOut
	r.call(t, "research.sessions.list", map[string]any{}, &list)
	for _, s := range list.Sessions {
		if s.SessionID == rep.SessionID {
			if s.Prompt != "findable question" {
				t.Errorf("prompt = %q", s.Prompt)
			}
			return
		}
	}
	t.Errorf("the session just started is not in the list: %+v", list.Sessions)
}

// writeClaims inserts n claims directly, with a lead of its own.
//
// It creates the lead rather than looking one up. The first version searched for
// the lead the executor makes when the planner proposes nothing — which it
// creates asynchronously, so the test passed or failed depending on whether the
// goroutine had got there yet. It failed roughly one run in two, and `go test`
// without -v reported "ok" on the run I first looked at.
func writeClaims(t *testing.T, db store.Store, sessionID string, n int) {
	t.Helper()
	ctx := context.Background()

	lead := core.Lead{
		ID: core.NewLeadID(), SessionID: sessionID,
		ActorType: core.ActorWeb, Query: "fixture", Status: core.LeadQueued,
	}
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.InsertLead(ctx, &lead)
	}); err != nil {
		t.Fatalf("insert lead: %v", err)
	}

	claims := make([]core.Claim, 0, n)
	for i := 0; i < n; i++ {
		claims = append(claims, core.Claim{
			ID:          core.NewClaimID(),
			SessionID:   sessionID,
			LeadID:      lead.ID,
			Text:        strings.Repeat("claim ", 3) + string(rune('a'+i%26)),
			Source:      "https://example.com/x",
			Quote:       "a quote",
			RetrievedAt: time.Now().UTC(),
		})
	}
	if err := db.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.InsertClaims(ctx, claims)
	}); err != nil {
		t.Fatalf("insert claims: %v", err)
	}
}

// TestResultReturnsTheReportThatWasPaidFor.
//
// §5.1 defines research.result as returning {report_md, claims[], edges[]}, and
// the MCP flow — report, poll status, result — has no other moment where the
// answer reaches the caller. It shipped returning "" for every session: the
// handler declared a `report` variable, never assigned it, and no test looked at
// report_md at all. Underneath, nothing persisted a report anywhere, so the
// daemon generated one, paid for it out of escrow, and dropped it.
func TestResultReturnsTheReportThatWasPaidFor(t *testing.T) {
	r := connect(t, 0)

	var rep mcpserver.ReportOut
	r.call(t, "research.report", map[string]any{
		"prompt": "a question",
		"budget": map[string]any{"unit": "usd", "amount": "0.50"},
	}, &rep)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	_, _ = r.sup.Wait(ctx, rep.SessionID)
	for i := 0; i < 200 && len(r.sup.Running()) > 0; i++ {
		time.Sleep(20 * time.Millisecond)
	}

	var out mcpserver.ResultOut
	r.call(t, "research.result", map[string]any{"session_id": rep.SessionID}, &out)

	if out.ReportMD == "" {
		t.Error("report_md is empty for a finished session; the answer the session " +
			"paid to produce never reaches the caller")
	}
	// And it must be the session's own answer, not a placeholder.
	if !strings.Contains(out.ReportMD, "a question") {
		t.Errorf("report_md does not mention the question it answers:\n%.300s", out.ReportMD)
	}
	// A degraded report says so, so an empty answer can be told from a failed one.
	if out.Note == "" && out.Status != string(core.StatusDone) {
		t.Errorf("status %q with no note explaining the report", out.Status)
	}
}

// TestTheCeilingCannotBeBypassedByChangingUnit.
//
// The original check read `unit == core.BudgetUSD && ...`, so a caller asking in
// TOKENS skipped it entirely: a 99,999,999,999-token session was accepted under
// a $1.00 ceiling documented as "the one limit the caller cannot raise". The
// test that was supposed to cover this only ever sent usd.
func TestTheCeilingCannotBeBypassedByChangingUnit(t *testing.T) {
	r := connectCeilings(t, 1_000_000, 100_000) // $1.00 / 100k tokens

	for name, budget := range map[string]map[string]any{
		"usd over":    {"unit": "usd", "amount": "50.00"},
		"tokens over": {"unit": "tokens", "amount": "99999999999"},
	} {
		if res := r.call(t, "research.report", map[string]any{
			"prompt": "expensive", "budget": budget,
		}, nil); !res.IsError {
			t.Errorf("%s: accepted %v past the ceiling", name, budget)
		}
	}

	// Both units still work under their ceilings.
	for name, budget := range map[string]map[string]any{
		"usd under":    {"unit": "usd", "amount": "0.25"},
		"tokens under": {"unit": "tokens", "amount": "50000"},
	} {
		if res := r.call(t, "research.report", map[string]any{
			"prompt": "fine", "budget": budget,
		}, nil); res.IsError {
			t.Errorf("%s: refused %v under the ceiling: %s", name, budget, errText(res))
		}
	}

	// Configuring one ceiling and not the other is a misconfiguration, and the
	// safe reading of a misconfigured limit is refusal — otherwise the unset unit
	// is exactly the open door this test exists for.
	half := connectCeilings(t, 1_000_000, 0)
	if res := half.call(t, "research.report", map[string]any{
		"prompt": "x", "budget": map[string]any{"unit": "tokens", "amount": "5"},
	}, nil); !res.IsError {
		t.Error("a token session was accepted on a daemon that caps only dollars")
	}
}

// TestATokenAmountIsParsedWholeOrRefused.
//
// fmt.Sscanf stops at the first character it cannot use and reports success for
// what it read, so "1,000,000" scanned as 1 with a nil error. A caller asking for
// a million tokens got a one-token session, watched it end instantly as
// exhausted, and was told research was running.
func TestATokenAmountIsParsedWholeOrRefused(t *testing.T) {
	r := connect(t, 0)
	for _, amount := range []string{"1e6", "1,000,000", "2.50", "100 000", "12abc"} {
		res := r.call(t, "research.report", map[string]any{
			"prompt": "a question",
			"budget": map[string]any{"unit": "tokens", "amount": amount},
		}, nil)
		if !res.IsError {
			t.Errorf("%q was accepted; it would silently truncate", amount)
		}
	}
	var out mcpserver.ReportOut
	if res := r.call(t, "research.report", map[string]any{
		"prompt": "a question",
		"budget": map[string]any{"unit": "tokens", "amount": "50000"},
	}, &out); res.IsError {
		t.Fatalf("a plain integer was refused: %s", errText(res))
	}
	if out.Budget != 50_000 {
		t.Errorf("budget = %d, want 50000", out.Budget)
	}
}

// TestCallerFanOutIsClamped. max_sources and max_depth multiply into §8.5's
// MaxToolCalls, so an unclamped million disables the ceiling that exists to bind
// when the money estimate is wrong.
func TestCallerFanOutIsClamped(t *testing.T) {
	r := connect(t, 0)
	var out mcpserver.ReportOut
	if res := r.call(t, "research.report", map[string]any{
		"prompt":      "a question",
		"budget":      map[string]any{"unit": "usd", "amount": "0.50"},
		"max_sources": 1_000_000,
		"max_depth":   1_000,
	}, &out); res.IsError {
		t.Fatalf("report: %s", errText(res))
	}

	var sess *core.Session
	if err := r.db.Read(context.Background(), func(ctx context.Context, q store.Queries) error {
		var err error
		sess, err = q.GetSession(ctx, out.SessionID)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	// The ceiling is sized from the clamped LEAD count now; fan-out no longer
	// multiplies into it at all, so a caller asking for a million sources cannot
	// inflate it by even one call.
	if sess.MaxLeads > int64(mcpserver.MaxDepthCeiling)*10 {
		t.Errorf("MaxLeads = %d, which a caller should not be able to reach", sess.MaxLeads)
	}
	if sess.MaxToolCalls > 10_000 {
		t.Errorf("MaxToolCalls = %d — a caller disabled §8.5's ceiling", sess.MaxToolCalls)
	}
}

// TestACallerMayAskForTimeButNotUnlimitedTime.
//
// The second half of #2: an agent calling over MCP could not say how long a
// session may run, and took whatever the daemon was configured with.
//
// Clamped, like max_sources and max_depth and for the stronger version of the
// same reason. On a command line the number is one a person typed. Over MCP it
// is one an agent chose, and MaxSessionUSD exists in this file precisely
// because "a coding agent that misreads its own instructions can ask for a
// thousand dollars as easily as three" — an agent that asks for a thousand
// hours is the same mistake in the other unit.
func TestACallerMayAskForTimeButNotUnlimitedTime(t *testing.T) {
	r := connect(t, 0)

	var out mcpserver.ReportOut
	if res := r.call(t, "research.report", map[string]any{
		"prompt":  "a question",
		"budget":  map[string]any{"unit": "usd", "amount": "0.50"},
		"timeout": "1000h",
	}, &out); res.IsError {
		t.Fatalf("report: %s", errText(res))
	}

	var sess *core.Session
	if err := r.db.Read(context.Background(), func(ctx context.Context, q store.Queries) error {
		var err error
		sess, err = q.GetSession(ctx, out.SessionID)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if sess.MaxWallClock > mcpserver.MaxWallClockCeiling {
		t.Errorf("MaxWallClock = %s, over the daemon's ceiling of %s",
			sess.MaxWallClock, mcpserver.MaxWallClockCeiling)
	}

	// And a reasonable ask is honoured rather than clamped to a default.
	var out2 mcpserver.ReportOut
	if res := r.call(t, "research.report", map[string]any{
		"prompt":  "another question",
		"budget":  map[string]any{"unit": "usd", "amount": "0.50"},
		"timeout": "90s",
	}, &out2); res.IsError {
		t.Fatalf("report: %s", errText(res))
	}
	var sess2 *core.Session
	if err := r.db.Read(context.Background(), func(ctx context.Context, q store.Queries) error {
		var err error
		sess2, err = q.GetSession(ctx, out2.SessionID)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if sess2.MaxWallClock != 90*time.Second {
		t.Errorf("MaxWallClock = %s, want the 90s the caller asked for", sess2.MaxWallClock)
	}
}

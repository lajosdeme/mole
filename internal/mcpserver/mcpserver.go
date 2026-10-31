// Package mcpserver exposes the daemon's research surface over MCP (§5.1).
//
// Separate from internal/daemon so that package stays transport-only: the socket
// and its permissions are one concern, what a caller can ask for is another, and
// this one is testable over an in-memory transport with no socket at all.
//
// The tools are asynchronous by design (§5.3). research.report returns a session
// id and nothing else; the work continues in the daemon long after the calling
// agent's context is gone, and the caller polls research.status the way it would
// wait on a CI job.
package mcpserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/lajosdeme/mole/internal/compute/connector"
	"github.com/lajosdeme/mole/internal/compute/gate"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/pricing"
	"github.com/lajosdeme/mole/internal/session"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/tools/extract"
	"github.com/lajosdeme/mole/internal/tools/fetch"
	"github.com/lajosdeme/mole/internal/tools/search"
)

// MaxClaimsReturned caps how many claims research.result hands back.
//
// The result goes straight into a coding agent's context window, and nothing
// upstream bounds how many claims a session gathers. A cap the caller can see —
// reported as Truncated with the real Total — beats both silently sending five
// thousand and silently sending two hundred of them.
const MaxClaimsReturned = 200

// MaxEdgesReturned caps the edges alongside them, for the same reason.
const MaxEdgesReturned = 500

// DefaultSessionsListed is the page size for research.sessions.list.
const DefaultSessionsListed = 20

// Deps is what the tools operate on.
type Deps struct {
	// Supervisor starts and cancels sessions.
	Supervisor *session.Supervisor
	// Store answers every read. Deliberately not the supervisor: the store is
	// authoritative and survives a restart, while the supervisor only remembers
	// the sessions it is running plus a bounded tail of finished ones. A status
	// query that worked before a daemon restart and failed after would be the
	// worst possible behaviour for a caller told to poll.
	Store store.Store

	// MaxSessionUSD and MaxSessionTokens cap a single session's budget in their
	// respective units. Zero means no ceiling. See config.MaxSessionUSD for why
	// these exist, and checkCeiling for why having only one set is refused.
	MaxSessionUSD    int64
	MaxSessionTokens int64

	// Version is the binary's version, reported to MCP clients in the server
	// handshake. Empty reports "dev".
	Version string

	// LLM answers research.ask. Nil disables synthesis: an ask then returns the
	// relevant claims with citations and no prose, which is the honest response
	// when no call can be made.
	LLM llmProvider
	// Pricing turns an ask's token usage into a ledger cost. Nil takes the
	// default table.
	Pricing *pricing.Table

	// RecheckModel, when set, withholds contradictions not yet re-checked from
	// research.ask.
	RecheckModel string

	// AskTimeout bounds one research.ask. Zero takes DefaultAskTimeout.
	AskTimeout time.Duration

	// Workers is how many leads each session runs at once. Zero takes
	// executor.DefaultWorkers.
	Workers int

	// Defaults applied when a call does not specify.
	MaxSources int
	MaxDepth   int
	MaxLeads   int
	Timeout    time.Duration

	// Search, Fetch and Extract back toolkit mode. Nil leaves the matching tool
	// reporting that it is not configured rather than failing at call time.
	Search  search.Provider
	Fetch   fetch.Fetcher
	Extract extract.Extractor

	// Connectors is the registered local data, for the toolkit's aggregate tools.
	// Nil leaves them reporting that nothing is registered.
	Connectors interface{ List() []connector.Connector }
	// Gate tunes the aggregation gate. The zero value is §12.1's defaults.
	Gate gate.Options

	// Toolkit registers the toolkit tool surface. Off by default: those tools cost
	// every client their definitions in its context window, and a caller who wants
	// autonomous research should not read past them to find research.report.
	Toolkit bool

	Log *slog.Logger
}

// New builds the MCP server with the tool surface attached.
//
// One server for the whole daemon, connected to each accepted connection
// separately. The tools close over shared state, so everything they touch is
// concurrent by construction.
func New(d Deps) *mcp.Server {
	if d.Log == nil {
		d.Log = slog.Default()
	}
	srv := mcp.NewServer(&mcp.Implementation{
		Name:    "mole",
		Title:   "mole — deep research",
		Version: d.version(),
	}, &mcp.ServerOptions{
		// Sent at initialize. In a client that folds these into the model's system
		// context, this is mole's only lever on a prompt it does not assemble —
		// and the measured difference between a fenced document alone and a fenced
		// document plus a system-side rule was one injection in twenty-five
		// against none. Whether a given client honours it is untested per client,
		// so the fence is applied regardless.
		Instructions: serverInstructions,
	})

	mcp.AddTool(srv, &mcp.Tool{
		Name: "research.report",
		Description: "Start a research session and return its id immediately. The work " +
			"continues in the daemon; poll research.status and then research.result. " +
			"Budget is required and is spent on real API calls.",
	}, d.report)

	mcp.AddTool(srv, &mcp.Tool{
		Name: "research.status",
		Description: "How a session is progressing: status, spend against its ceiling, " +
			"and what it has found so far.",
	}, d.status)

	mcp.AddTool(srv, &mcp.Tool{
		Name: "research.result",
		Description: "The finished answer for a session: the report, plus the claims and " +
			"graph edges behind it. Each claim carries its source, the verbatim quote " +
			"supporting it, and a derived confidence — weigh them rather than taking the " +
			"prose alone.",
	}, d.result)

	mcp.AddTool(srv, &mcp.Tool{
		Name: "research.cancel",
		Description: "Stop a running session. Budget already spent is not refunded, but " +
			"nothing further is spent: the remaining research, the grounding pass and the " +
			"report synthesis are all skipped.",
	}, d.cancel)

	mcp.AddTool(srv, &mcp.Tool{
		Name: "research.ask",
		Description: "Ask a NEW question against a finished session's research, without " +
			"researching again. Retrieval-only and cheap: it re-reads claims that were " +
			"already gathered and paid for. Use it when your own context was compacted, " +
			"or to follow up on something the original question did not cover.",
	}, d.ask)

	mcp.AddTool(srv, &mcp.Tool{
		Name: "research.sessions.list",
		Description: "Recent sessions with their status and spend. Use it to find a " +
			"session id again after your own context has been compacted.",
	}, d.listSessions)

	if d.Toolkit {
		registerToolkit(srv, d)
	}
	return srv
}

// ---------------------------------------------------------------------------
// research.report
// ---------------------------------------------------------------------------

// Budget is the amount a session may spend. Both fields are required: §8 makes
// the unit semantically load-bearing, and a bare number is ambiguous between
// dollars and tokens.
type Budget struct {
	Unit   string `json:"unit" jsonschema:"the budget unit: usd or tokens"`
	Amount string `json:"amount" jsonschema:"how much to spend: dollars like 2.50 for usd, a whole number of tokens for tokens"`
}

type ReportIn struct {
	Prompt string `json:"prompt" jsonschema:"the research question"`
	Budget Budget `json:"budget" jsonschema:"what this session may spend"`

	MaxSources int    `json:"max_sources,omitempty" jsonschema:"sources to read per lead; omit for the daemon default"`
	MaxDepth   int    `json:"max_depth,omitempty" jsonschema:"rounds of follow-up questions the planner may add; omit for the daemon default"`
	Timeout    string `json:"timeout,omitempty" jsonschema:"wall-clock ceiling for the session, e.g. 20m; omit for the daemon default. Clamped to the daemon's ceiling"`
}

type ReportOut struct {
	SessionID string `json:"session_id"`
	Status    string `json:"status"`
	Unit      string `json:"budget_unit"`
	Budget    int64  `json:"budget"`
	Note      string `json:"note,omitempty"`
}

func (d Deps) report(ctx context.Context, _ *mcp.CallToolRequest, in ReportIn) (*mcp.CallToolResult, ReportOut, error) {
	prompt := strings.TrimSpace(in.Prompt)
	if prompt == "" {
		return nil, ReportOut{}, errors.New("prompt is empty")
	}

	unit, amount, err := parseBudget(in.Budget)
	if err != nil {
		return nil, ReportOut{}, err
	}
	// The daemon's ceiling, not the caller's. Checked before the session row is
	// written so a refused request costs nothing and leaves nothing behind.
	if err := d.checkCeiling(unit, amount); err != nil {
		return nil, ReportOut{}, err
	}

	spec := session.Spec{
		Question:   prompt,
		Mode:       core.ModeReport,
		BudgetUnit: unit,
		Budget:     amount,
		// Clamped, not just defaulted. These multiply into §8.5's MaxToolCalls
		// (leads x sources x 4), so an unclamped max_sources of a million disables
		// the one ceiling that is supposed to bind when the money estimate is
		// wrong — and points a million fetches at arbitrary hosts from the user's
		// address while doing it.
		MaxSources: clampToDefault(in.MaxSources, d.MaxSources, MaxSourcesCeiling),
		MaxDepth:   clampToDefault(in.MaxDepth, d.MaxDepth, MaxDepthCeiling),
		MaxLeads:   d.MaxLeads,
		Timeout:    clampDuration(in.Timeout, d.Timeout, MaxWallClockCeiling),
		Workers:    d.Workers,
	}

	// ctx is the MCP CALL's context and dies when this response is sent. Start
	// uses it only to create the session row; the run hangs off the supervisor's
	// own context, which is the whole reason the supervisor exists.
	sess, err := d.Supervisor.Start(ctx, spec)
	if err != nil {
		return nil, ReportOut{}, err
	}
	d.Log.Info("session started over MCP", "session", sess.ID, "budget", amount, "unit", unit)

	return nil, ReportOut{
		SessionID: sess.ID,
		Status:    string(sess.Status),
		Unit:      string(unit),
		Budget:    sess.Budget,
		Note: "Research is running. Poll research.status; when it reports done, " +
			"call research.result.",
	}, nil
}

// ---------------------------------------------------------------------------
// research.status
// ---------------------------------------------------------------------------

type SessionRef struct {
	SessionID string `json:"session_id" jsonschema:"the id returned by research.report"`
}

type StatusOut struct {
	SessionID string `json:"session_id"`
	Status    string `json:"status"`
	Prompt    string `json:"prompt"`

	Unit      string `json:"budget_unit"`
	Budget    int64  `json:"budget"`
	Spent     int64  `json:"spent"`
	Remaining int64  `json:"remaining"`

	Claims   int  `json:"claims"`
	Verified int  `json:"claims_verified"`
	Edges    int  `json:"edges"`
	Running  bool `json:"running"`
}

func (d Deps) status(ctx context.Context, _ *mcp.CallToolRequest, in SessionRef) (*mcp.CallToolResult, StatusOut, error) {
	sess, err := d.loadSession(ctx, in.SessionID)
	if err != nil {
		return nil, StatusOut{}, err
	}

	var claims []*core.Claim
	var edges []*core.ClaimEdge
	if err := d.Store.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var rerr error
		if claims, rerr = q.ListClaims(ctx, sess.ID, 10_000); rerr != nil {
			return rerr
		}
		edges, rerr = q.ListEdges(ctx, sess.ID, 0)
		return rerr
	}); err != nil {
		return nil, StatusOut{}, err
	}

	verified := 0
	for _, c := range claims {
		if c != nil && c.VerifiedAt != nil {
			verified++
		}
	}

	return nil, StatusOut{
		SessionID: sess.ID,
		Status:    string(sess.Status),
		Prompt:    sess.Prompt,
		Unit:      string(sess.BudgetUnit),
		Budget:    sess.Budget,
		Spent:     sess.Spent,
		Remaining: sess.Available(),
		Claims:    len(claims),
		Verified:  verified,
		Edges:     len(edges),
		Running:   d.isRunning(sess.ID),
	}, nil
}

// ---------------------------------------------------------------------------
// ResultCitation is one numbered source behind the report's prose.
type ResultCitation struct {
	N      int      `json:"n"`
	Source string   `json:"source"`
	Quotes []string `json:"quotes,omitempty"`
}

// resultCitations decodes the stored bibliography, dropping it whole if it will
// not parse: a marker resolving to the wrong source is worse than one resolving
// to nothing, because only the second is visible.
// Empty rather than nil in every case, so the field is an array and never
// JSON null. An MCP caller iterating it should not have to check first, and
// every session recorded before the citations column — plus every dataset-mode
// session, which cites nothing by design — takes one of these paths.
func resultCitations(raw string) []ResultCitation {
	if raw == "" {
		return []ResultCitation{}
	}
	var cs []ResultCitation
	if err := json.Unmarshal([]byte(raw), &cs); err != nil {
		return []ResultCitation{}
	}
	if cs == nil {
		// "null" parses without error and yields a nil slice.
		return []ResultCitation{}
	}
	return cs
}

// research.result
// ---------------------------------------------------------------------------

// Claim is one finding, in the shape §5.3 says an agent caller wants: sourced,
// dated, quoted, and scored, rather than a paragraph.
type Claim struct {
	ID          string  `json:"id"`
	Text        string  `json:"text"`
	Source      string  `json:"source"`
	Quote       string  `json:"quote,omitempty"`
	PublishedAt string  `json:"published_at,omitempty"`
	Confidence  float64 `json:"confidence"`
	Grounded    *bool   `json:"grounded,omitempty"`
}

// Edge is one relation in the claim graph (§11.2).
type Edge struct {
	From      string  `json:"from"`
	To        string  `json:"to"`
	Kind      string  `json:"kind"`
	Weight    float64 `json:"weight,omitempty"`
	Rationale string  `json:"rationale,omitempty"`
}

type ResultOut struct {
	SessionID string `json:"session_id"`
	Status    string `json:"status"`
	ReportMD  string `json:"report_md"`

	// Citations resolve the [n] markers in ReportMD. Without them a caller reads
	// prose citing [4] with no way to learn what [4] was — the numbering is
	// assigned over a findings set capped at generation time and cannot be
	// recomputed, so it is carried from the session or it is lost.
	Citations []ResultCitation `json:"citations"`

	Claims []Claim `json:"claims"`
	Edges  []Edge  `json:"edges"`

	// TotalClaims and TotalEdges are what the session holds; the arrays above may
	// be shorter. Reported so a truncated result is visibly truncated — silently
	// returning a prefix is how a caller concludes a session found less than it
	// did.
	TotalClaims int  `json:"total_claims"`
	TotalEdges  int  `json:"total_edges"`
	Truncated   bool `json:"truncated"`

	Note string `json:"note,omitempty"`
}

func (d Deps) result(ctx context.Context, _ *mcp.CallToolRequest, in SessionRef) (*mcp.CallToolResult, ResultOut, error) {
	sess, err := d.loadSession(ctx, in.SessionID)
	if err != nil {
		return nil, ResultOut{}, err
	}

	var (
		claims []*core.Claim
		edges  []*core.ClaimEdge
	)
	if err := d.Store.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var rerr error
		if claims, rerr = q.ListClaims(ctx, sess.ID, 10_000); rerr != nil {
			return rerr
		}
		edges, rerr = q.ListEdges(ctx, sess.ID, 0)
		return rerr
	}); err != nil {
		return nil, ResultOut{}, err
	}

	// Totals count non-nil entries, for the same reason the loops below do: a nil
	// is not a claim, and "showing 500 of 502" when two of those were nil invites
	// a caller to go looking for two claims that never existed.
	out := ResultOut{
		SessionID:   sess.ID,
		Status:      string(sess.Status),
		ReportMD:    sess.Report,
		Citations:   resultCitations(sess.ReportCitationsJSON),
		TotalClaims: countNonNil(claims),
		TotalEdges:  countNonNil(edges),
	}
	switch {
	case sess.Status == core.StatusRunning:
		out.Note = "This session is still running; the claims below are what it has " +
			"gathered so far and the report is not written yet."
	case sess.ReportDegraded != "":
		// Said plainly rather than left as an empty or thin report_md. A caller
		// that cannot tell "nothing was found" from "synthesis failed and the
		// evidence is still here" will draw the wrong conclusion from the same
		// empty string.
		out.Note = "The report is incomplete: " + sess.ReportDegraded +
			". The claims below are the evidence it was built from."
	}

	// Count what is EMITTED, not the slice index. A nil entry is skipped but still
	// advances i, so indexing the cap returns fewer than MaxClaimsReturned claims
	// and reports the shortfall as truncation — under-delivering and mislabelling
	// it in the same step.
	for _, c := range claims {
		if c == nil {
			continue
		}
		if len(out.Claims) >= MaxClaimsReturned {
			out.Truncated = true
			break
		}
		cl := Claim{
			ID:         c.ID,
			Text:       c.Text,
			Source:     c.Source,
			Quote:      c.Quote,
			Confidence: c.Confidence,
			Grounded:   c.Grounded,
		}
		if c.PublishedAt != nil {
			cl.PublishedAt = c.PublishedAt.UTC().Format(time.RFC3339)
		}
		out.Claims = append(out.Claims, cl)
	}
	for _, e := range edges {
		if e == nil {
			continue
		}
		if len(out.Edges) >= MaxEdgesReturned {
			out.Truncated = true
			break
		}
		out.Edges = append(out.Edges, Edge{
			From: e.FromID, To: e.ToID, Kind: string(e.Kind),
			Weight: e.Weight, Rationale: e.Rationale,
		})
	}
	if out.Truncated {
		out.Note = strings.TrimSpace(out.Note + fmt.Sprintf(
			" Showing %d of %d claims and %d of %d edges.",
			len(out.Claims), out.TotalClaims, len(out.Edges), out.TotalEdges))
	}
	return nil, out, nil
}

// ---------------------------------------------------------------------------
// research.cancel
// ---------------------------------------------------------------------------

type CancelOut struct {
	SessionID string `json:"session_id"`
	Cancelled bool   `json:"cancelled"`
	Note      string `json:"note,omitempty"`
}

func (d Deps) cancel(ctx context.Context, _ *mcp.CallToolRequest, in SessionRef) (*mcp.CallToolResult, CancelOut, error) {
	// Confirm it exists before asking the supervisor, so "no such session" and
	// "that session already finished" are different answers. Only the store can
	// tell them apart.
	sess, err := d.loadSession(ctx, in.SessionID)
	if err != nil {
		return nil, CancelOut{}, err
	}

	if err := d.Supervisor.Cancel(sess.ID); err != nil {
		if errors.Is(err, session.ErrNoSuchSession) {
			return nil, CancelOut{
				SessionID: sess.ID,
				Cancelled: false,
				Note: fmt.Sprintf("this session is not running (status %s); nothing to cancel",
					sess.Status),
			}, nil
		}
		return nil, CancelOut{}, err
	}
	d.Log.Info("session cancelled over MCP", "session", sess.ID)

	return nil, CancelOut{
		SessionID: sess.ID,
		Cancelled: true,
		Note: "Stopping. Budget already spent is not refunded; nothing further is " +
			"spent. Poll research.status to see it settle.",
	}, nil
}

// ---------------------------------------------------------------------------
// research.sessions.list
// ---------------------------------------------------------------------------

type ListIn struct {
	Limit int `json:"limit,omitempty" jsonschema:"how many recent sessions to return; omit for 20"`
}

type SessionSummary struct {
	SessionID string `json:"session_id"`
	Status    string `json:"status"`
	Prompt    string `json:"prompt"`
	Unit      string `json:"budget_unit"`
	Budget    int64  `json:"budget"`
	Spent     int64  `json:"spent"`
	Running   bool   `json:"running"`
}

type ListOut struct {
	Sessions []SessionSummary `json:"sessions"`
}

func (d Deps) listSessions(ctx context.Context, _ *mcp.CallToolRequest, in ListIn) (*mcp.CallToolResult, ListOut, error) {
	limit := in.Limit
	if limit <= 0 {
		limit = DefaultSessionsListed
	}

	var sessions []*core.Session
	if err := d.Store.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var rerr error
		sessions, rerr = q.ListSessions(ctx, limit)
		return rerr
	}); err != nil {
		return nil, ListOut{}, err
	}

	out := ListOut{Sessions: make([]SessionSummary, 0, len(sessions))}
	for _, s := range sessions {
		if s == nil {
			continue
		}
		out.Sessions = append(out.Sessions, SessionSummary{
			SessionID: s.ID,
			Status:    string(s.Status),
			Prompt:    s.Prompt,
			Unit:      string(s.BudgetUnit),
			Budget:    s.Budget,
			Spent:     s.Spent,
			Running:   d.isRunning(s.ID),
		})
	}
	return nil, out, nil
}

// ---------------------------------------------------------------------------
// Shared
// ---------------------------------------------------------------------------

func (d Deps) loadSession(ctx context.Context, id string) (*core.Session, error) {
	id = strings.TrimSpace(id)
	if id == "" {
		return nil, errors.New("session_id is empty")
	}
	var sess *core.Session
	err := d.Store.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var rerr error
		sess, rerr = q.GetSession(ctx, id)
		return rerr
	})
	if err != nil {
		return nil, fmt.Errorf("no session %s: %w", id, err)
	}
	return sess, nil
}

func (d Deps) isRunning(id string) bool {
	if d.Supervisor == nil {
		return false
	}
	for _, r := range d.Supervisor.Running() {
		if r == id {
			return true
		}
	}
	return false
}

// parseBudget turns the wire form into a unit and an amount.
//
// The unit is required. §8 makes it semantically load-bearing and there is no
// bare-number form on the command line either, for the same reason: a caller
// that means tokens and gets dollars overspends by six orders of magnitude.
func parseBudget(b Budget) (core.BudgetUnit, int64, error) {
	unit := core.BudgetUnit(strings.ToLower(strings.TrimSpace(b.Unit)))
	amount := strings.TrimSpace(b.Amount)
	if amount == "" {
		return "", 0, errors.New("budget.amount is required")
	}

	switch unit {
	case core.BudgetUSD:
		v, err := core.ParseUSD(amount)
		if err != nil {
			return "", 0, fmt.Errorf("budget.amount %q is not a dollar amount: %w", amount, err)
		}
		if v <= 0 {
			return "", 0, errors.New("budget.amount must be positive")
		}
		return unit, v, nil
	case core.BudgetTokens:
		// ParseInt, not fmt.Sscanf. Sscanf stops at the first character it cannot
		// use and reports success for what it read: "1,000,000" scans as 1 and
		// "1e6" as 1, with a nil error. A caller asking for a million tokens would
		// get a one-token session, watch it end instantly as exhausted, and be
		// told research was running. The USD branch never had this because
		// core.ParseUSD rejects trailing junk.
		v, err := strconv.ParseInt(amount, 10, 64)
		if err != nil {
			return "", 0, fmt.Errorf("budget.amount %q is not a whole number of tokens", amount)
		}
		if v <= 0 {
			return "", 0, errors.New("budget.amount must be positive")
		}
		return unit, v, nil
	case "":
		return "", 0, errors.New("budget.unit is required: usd or tokens")
	default:
		return "", 0, fmt.Errorf("budget.unit %q is not usd or tokens", b.Unit)
	}
}

// Ceilings on what a caller may ask for per lead and per session.
//
// Generous — a legitimate agent will never reach them — and finite, which is the
// point: every value here is multiplied into a resource bound.
const (
	MaxSourcesCeiling = 25
	MaxDepthCeiling   = 5
	// MaxWallClockCeiling bounds how long a caller may ask a session to run.
	// Generous and finite: here the number is one an agent chose, so the
	// ceiling is the daemon's, like MaxSessionUSD.
	MaxWallClockCeiling = time.Hour
)

// checkCeiling enforces the daemon's per-session budget ceiling.
//
// Per UNIT, because dollars and tokens are not convertible without knowing which
// model will run, and a guessed exchange rate is a ceiling nobody can reason
// about.
//
// The last branch is the one that matters. Enforcing only the unit that happens
// to have a ceiling leaves the other as an open door: the first version checked
// USD alone, and a caller asking for 99,999,999,999 TOKENS sailed past a $1.00
// ceiling documented as "the one limit the caller cannot raise". Configuring one
// ceiling and not the other is a misconfiguration, and the safe reading of a
// misconfigured limit is refusal.
func (d Deps) checkCeiling(unit core.BudgetUnit, amount int64) error {
	switch unit {
	case core.BudgetUSD:
		if d.MaxSessionUSD > 0 && amount > d.MaxSessionUSD {
			return fmt.Errorf("budget %s is over this daemon's per-session ceiling of %s "+
				"(raise it with: mole config set daemon.max-session-usd)",
				core.FormatUSD(amount), core.FormatUSD(d.MaxSessionUSD))
		}
		if d.MaxSessionUSD == 0 && d.MaxSessionTokens > 0 {
			return errors.New("this daemon caps token budgets but not dollar ones; " +
				"set daemon.max-session-usd before starting a usd session")
		}
	case core.BudgetTokens:
		if d.MaxSessionTokens > 0 && amount > d.MaxSessionTokens {
			return fmt.Errorf("budget %d tokens is over this daemon's per-session "+
				"ceiling of %d (raise it with: mole config set daemon.max-session-tokens)",
				amount, d.MaxSessionTokens)
		}
		if d.MaxSessionTokens == 0 && d.MaxSessionUSD > 0 {
			return errors.New("this daemon caps dollar budgets but not token ones; " +
				"set daemon.max-session-tokens before starting a token session")
		}
	}
	return nil
}

// clampToDefault takes the caller's value when it is set and sane, the daemon's
// default when it is not, and never more than max.
// clampDuration parses a caller's timeout and holds it under the ceiling. An
// unparseable or oversized value takes the default rather than failing the
// call — the caller asked for research.
func clampDuration(v string, def, max time.Duration) time.Duration {
	d, err := time.ParseDuration(strings.TrimSpace(v))
	if err != nil || d <= 0 {
		d = def
	}
	if d <= 0 || d > max {
		if def > 0 && def < max {
			return def
		}
		return max
	}
	return d
}

func clampToDefault(v, def, max int) int {
	if v <= 0 {
		v = def
	}
	if v > max {
		return max
	}
	if v <= 0 {
		return 1
	}
	return v
}

// version reports what the daemon tells an MCP client it is.
//
// Deps.Version carries the binary's real version, stamped by the linker in
// cmd/mole. The fallback is "dev" rather than a number: a hardcoded "0.1.0"
// stayed put through every release and told a client something false, and an
// unset version that says so is more useful than a stale one that does not.
func (d Deps) version() string {
	if v := strings.TrimSpace(d.Version); v != "" {
		return v
	}
	return "dev"
}

// countNonNil counts the non-nil entries of a slice of pointers.
func countNonNil[T any](xs []*T) int {
	n := 0
	for _, x := range xs {
		if x != nil {
			n++
		}
	}
	return n
}

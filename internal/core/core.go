// Package core holds the domain types shared by every other package.
//
// Two conventions matter here and are load-bearing everywhere else:
//
//   - Money is integer micro-dollars (int64), never float64. Budget adherence is
//     a tested metric (max overshoot must be ~0); float accumulation over
//     thousands of reserve/settle operations drifts, so the architecture sketch's
//     `float64` USD fields are represented as `USDMicros` throughout.
//   - A session's budget, spend, holds, and escrow are all expressed in the
//     session's own BudgetUnit, so they are directly comparable int64s.
package core

import (
	"fmt"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

type BudgetUnit string

const (
	BudgetUSD    BudgetUnit = "usd"
	BudgetTokens BudgetUnit = "tokens"
)

func (u BudgetUnit) Valid() bool { return u == BudgetUSD || u == BudgetTokens }

type Mode string

const (
	ModeReport  Mode = "report"
	ModeDataset Mode = "dataset"
	ModeChain   Mode = "chain"
	ModeAsk     Mode = "ask"
)

func (m Mode) Valid() bool {
	switch m {
	case ModeReport, ModeDataset, ModeChain, ModeAsk:
		return true
	}
	return false
}

type ActorType string

const (
	ActorWeb          ActorType = "web"
	ActorAcademic     ActorType = "academic"
	ActorLocalCompute ActorType = "local_compute"
	// ActorToolkit marks a session an external agent drives through the toolkit
	// tools. No actor of this type exists — that is the point. mole dispatches no
	// lead for it, so the executor, the planner and the abandonment sweep all
	// have to be able to tell it apart from a session mole is running itself.
	ActorToolkit ActorType = "toolkit"
)

func (a ActorType) Valid() bool {
	switch a {
	case ActorWeb, ActorAcademic, ActorLocalCompute, ActorToolkit:
		return true
	}
	return false
}

type SessionStatus string

const (
	StatusRunning   SessionStatus = "running"
	StatusDone      SessionStatus = "done"
	StatusExhausted SessionStatus = "budget_exhausted"
	StatusCancelled SessionStatus = "cancelled"
	StatusFailed    SessionStatus = "failed"
)

func (s SessionStatus) Valid() bool {
	switch s {
	case StatusRunning, StatusDone, StatusExhausted, StatusCancelled, StatusFailed:
		return true
	}
	return false
}

// Terminal reports whether the session can no longer spend.
func (s SessionStatus) Terminal() bool { return s != StatusRunning }

type LeadStatus string

const (
	LeadQueued       LeadStatus = "queued"
	LeadLeased       LeadStatus = "leased"
	LeadDone         LeadStatus = "done"
	LeadFailed       LeadStatus = "failed"
	LeadSkippedCache LeadStatus = "skipped_cached"
)

// Role records which stage of the loop spent a tool call. It is what makes
// "how much went to verification vs. execution" a GROUP BY rather than an
// archaeology project.
type Role string

const (
	RolePlanner  Role = "planner"
	RoleExecutor Role = "executor"
	RoleVerifier Role = "verifier"
	RoleOutput   Role = "output"
)

func (r Role) Valid() bool {
	switch r {
	case RolePlanner, RoleExecutor, RoleVerifier, RoleOutput:
		return true
	}
	return false
}

type CallType string

const (
	CallSearch    CallType = "search"
	CallFetch     CallType = "fetch"
	CallLLM       CallType = "llm"
	CallLocalQry  CallType = "local_query"
	CallAcademic  CallType = "academic_query"
	CallSandbox   CallType = "sandbox"
	CallEmbedding CallType = "embedding"
)

type ReservationStatus string

const (
	ReservationHeld     ReservationStatus = "held"
	ReservationSettled  ReservationStatus = "settled"
	ReservationReleased ReservationStatus = "released"
)

type EdgeKind string

const (
	EdgeSupports    EdgeKind = "supports"
	EdgeContradicts EdgeKind = "contradicts"
	EdgeDuplicateOf EdgeKind = "duplicate_of"
	EdgeSupersedes  EdgeKind = "supersedes"
	EdgeRefines     EdgeKind = "refines"
)

func (k EdgeKind) Valid() bool {
	switch k {
	case EdgeSupports, EdgeContradicts, EdgeDuplicateOf, EdgeSupersedes, EdgeRefines:
		return true
	}
	return false
}

// Symmetric reports whether the edge asserts the same thing in both directions.
//
// A storage constraint, not just semantics. `UNIQUE (from_id, to_id, kind)`
// treats A→B and B→A as different rows, so one disagreement discovered from both
// ends is stored twice — and §11.3 penalizes confidence per contradicting edge,
// which would count that disagreement twice. InsertEdges orders the endpoints of
// a symmetric edge so the UNIQUE constraint can see the duplicate.
//
// `supports`, `supersedes` and `refines` stay directional: a specific finding
// supporting a general conclusion is not the same statement reversed, and
// supersedes is decided from PublishedAt, which has an arrow in it.
func (k EdgeKind) Symmetric() bool {
	return k == EdgeContradicts || k == EdgeDuplicateOf
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

// Session is one research task.
//
// Spent, Held, and Escrow are NOT independent state. Spent is a materialized
// sum of the tool_calls ledger, written in the same transaction as each cost
// row, so it can always be recomputed after a crash (see budget.Ledger.Verify).
type Session struct {
	ID         string
	Prompt     string
	Mode       Mode
	ActorTypes []ActorType

	BudgetUnit BudgetUnit
	Budget     int64 // micro-dollars, or tokens, per BudgetUnit
	Spent      int64 // materialized sum of the ledger
	Held       int64 // sum of outstanding reservations
	Escrow     int64 // held back for output + final verify

	// Unit-independent ceilings. Without these, token mode has a hole: search
	// and fetch calls cost no tokens, so an unbounded fetch loop would be free.
	MaxToolCalls  int64
	MaxLeads      int64
	MaxWallClock  time.Duration
	ToolCallCount int64
	LeadCount     int64

	Status SessionStatus
	// StoppedBecause names the ceiling or condition that ended the run, or is
	// empty for one that simply finished.
	//
	// Status cannot say it. StatusExhausted is the string "budget_exhausted"
	// and five different endings reach it: the budget running out, and the four
	// ceilings that have nothing to do with money — max_tool_calls, max_leads,
	// max_wallclock and a context deadline. HitCeiling already returns which,
	// and this is where that answer is kept.
	StoppedBecause string
	CreatedAt      time.Time
	UpdatedAt      time.Time
	// Report is the session's rendered answer, written when it finalizes.
	//
	// Persisted because §5.1's research.result is defined as returning it, and
	// the MCP flow — report, poll status, result — has no other moment where the
	// answer reaches the caller. Before this it lived only in memory: the CLI
	// printed it, and the daemon paid for it out of escrow and dropped it.
	Report string
	// ReportDegraded says why the prose is missing or unsynthesized, so an empty
	// Report can be told from a failed one.
	ReportDegraded string

	// ReportCitationsJSON is the report's bibliography as opaque JSON: the [n]
	// markers in Report resolved to sources. Carried rather than recomputed
	// because the numbering depends on a generation-time cap nobody persists.
	ReportCitationsJSON string

	// ReceiptJSON is the citation receipt as opaque JSON, empty when verification
	// never ran. Opaque because the store carries it without owning its shape, the
	// same way it carries Report without parsing markdown — and because an absent
	// check and a receipt of zeroes are different facts a consumer must tell apart.
	ReceiptJSON string
}

// Available is the spendable balance: everything not already spent, held by an
// outstanding reservation, or escrowed for output.
func (s *Session) Available() int64 {
	a := s.Budget - s.Spent - s.Held - s.Escrow
	if a < 0 {
		return 0
	}
	return a
}

// HitCeiling reports whether any unit-independent limit has been reached.
// Toolkit reports whether this session is driven by an external agent rather
// than by mole's own executor.
//
// It changes what several things mean. No lead is ever dispatched, so an idle
// session is not an abandoned one; nothing mole runs writes to it, so the
// executor's assumptions about tool calls and leads do not hold; and closing it
// is the agent's job rather than a runner's.
func (s *Session) Toolkit() bool {
	for _, a := range s.ActorTypes {
		if a == ActorToolkit {
			return true
		}
	}
	return false
}

func (s *Session) HitCeiling(now time.Time) (bool, string) {
	if s.MaxToolCalls > 0 && s.ToolCallCount >= s.MaxToolCalls {
		return true, "max_tool_calls"
	}
	if s.MaxLeads > 0 && s.LeadCount >= s.MaxLeads {
		return true, "max_leads"
	}
	if s.MaxWallClock > 0 && now.Sub(s.CreatedAt) >= s.MaxWallClock {
		return true, "max_wallclock"
	}
	return false, ""
}

// RemainingFraction is how much of the session's allowance is left, in [0,1].
//
// HitCeiling's continuous twin, and it takes the minimum over the same limits
// for the same reason HitCeiling checks all of them: a session two minutes from
// max_wallclock with 90% of its dollars unspent has 10% left, not 90%. Reporting
// spend alone would promise room that cannot be used.
//
// Exists because §9.1's replan prompt asks the planner to judge whether the open
// sub-questions are "worth more budget" while the digest reported nothing about
// budget — an instruction the planner had no way to answer.
//
// Available() rather than Budget-Spent, so escrow is excluded: tokens held back
// for the report are not spendable on more research, and counting them would
// overstate what is left at exactly the moment the answer matters.
func (s *Session) RemainingFraction(now time.Time) float64 {
	frac := 1.0
	if s.Budget > 0 {
		frac = min(frac, float64(s.Available())/float64(s.Budget))
	}
	if s.MaxToolCalls > 0 {
		frac = min(frac, float64(s.MaxToolCalls-s.ToolCallCount)/float64(s.MaxToolCalls))
	}
	if s.MaxLeads > 0 {
		frac = min(frac, float64(s.MaxLeads-s.LeadCount)/float64(s.MaxLeads))
	}
	if s.MaxWallClock > 0 {
		frac = min(frac, float64(s.MaxWallClock-now.Sub(s.CreatedAt))/float64(s.MaxWallClock))
	}
	return max(0, frac)
}

func (s *Session) Validate() error {
	if s.ID == "" {
		return fmt.Errorf("session: empty id")
	}
	if !s.Mode.Valid() {
		return fmt.Errorf("session: invalid mode %q", s.Mode)
	}
	if !s.BudgetUnit.Valid() {
		return fmt.Errorf("session: invalid budget unit %q", s.BudgetUnit)
	}
	if !s.Status.Valid() {
		return fmt.Errorf("session: invalid status %q", s.Status)
	}
	if s.Budget < 0 {
		return fmt.Errorf("session: negative budget")
	}
	if len(s.ActorTypes) == 0 {
		return fmt.Errorf("session: no actor types")
	}
	for _, a := range s.ActorTypes {
		if !a.Valid() {
			return fmt.Errorf("session: invalid actor type %q", a)
		}
	}
	return nil
}

// EncodeActorTypes / DecodeActorTypes keep the list in one TEXT column rather
// than a join table; the set is tiny and never queried by membership.
func EncodeActorTypes(as []ActorType) string {
	parts := make([]string, len(as))
	for i, a := range as {
		parts[i] = string(a)
	}
	return strings.Join(parts, ",")
}

func DecodeActorTypes(s string) []ActorType {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]ActorType, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, ActorType(p))
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Lead
// ---------------------------------------------------------------------------

// Lead is a unit of work. Leases (not a bare status column) are what make the
// daemon crash-recoverable: a worker takes a lease and heartbeats it, and a
// boot-time sweep requeues leads whose lease expired.
type Lead struct {
	ID        string
	SessionID string
	ActorType ActorType
	Query     string
	ParentID  *string
	Depth     int
	Priority  int
	Status    LeadStatus

	// RootClaimID marks a lead spawned to resolve something about a claim —
	// §11.4's verification lineage. Nil on an ordinary planner lead.
	//
	// Claims the lead produces inherit it, so a chain of follow-ups all trace to
	// the claim that started the investigation. §11.4's point is that this cannot
	// be a per-claim counter: a follow-up produces a NEW claim, and a counter on
	// the row would start over every time.
	RootClaimID *string
	// VerifyDepth is how many follow-ups deep this lead sits. Claims inherit it.
	VerifyDepth int

	LeaseOwner   *string
	LeaseExpires *time.Time

	CreatedAt time.Time
	UpdatedAt time.Time
}

// ---------------------------------------------------------------------------
// Claim + edges
// ---------------------------------------------------------------------------

// Claim is an atomic extracted fact.
//
// Quote is required at extraction time and checked verbatim against the
// extracted text before the claim is accepted. That check is cheap,
// deterministic, and rejects fabricated citations at the actor boundary rather
// than letting the verifier discover them later.
type Claim struct {
	ID        string
	SessionID string
	LeadID    string
	Text      string

	Source      string // URL, DOI, or "connector:<name>#<query_hash>"
	ToolCallID  string
	Quote       string
	QuoteOffset int64

	PublishedAt *time.Time
	RetrievedAt time.Time

	// Verification lineage. A per-row recheck counter cannot work: a follow-up
	// lead produces a NEW claim, which would start the counter over. Depth is
	// inherited and incremented so the cap actually binds.
	RootClaimID string
	VerifyDepth int

	// AssertionStrength is how clearly the SOURCE states this, as reported by
	// the extracting model. A property of the document, not of the world, and
	// that is all the mine prompt asks for.
	//
	// Not confidence, and kept separate from it because it was serving as
	// confidence: §11.3 rejects self-reported confidence as uncalibrated and
	// "mostly encoding fluency", and the report was ordered by it.
	AssertionStrength float64

	// Confidence is derived from graph structure (§11.3) — independent
	// corroborating publishers, source class, contradicting edges, grounding,
	// recency. Never asked of a model.
	//
	// Zero until the Verifier has scored the claim, which is the honest value:
	// nothing corroborates a claim that has not been compared to anything.
	Confidence float64
	Grounded   *bool
	// GroundingNote says how a grounding check reached its verdict, including the
	// two outcomes Grounded cannot express: the quote vanished from its source, or
	// the source could not be reached. Both leave Grounded nil, because neither is
	// evidence about the claim (§11.5).
	GroundingNote string

	// VerifiedAt records that the Verifier has scored this claim. Nil means it
	// has not, which Confidence cannot express on its own: §11.3's formula
	// legitimately returns 0 for an uncorroborated claim carrying a contradiction,
	// so a zero there means "scored badly" as often as "never examined".
	VerifiedAt *time.Time

	CreatedAt time.Time
	// Seq is this claim's position within the batch it was written in, assigned
	// by the store. It exists to make claim order total and reproducible.
	//
	// CreatedAt cannot do it alone: one timestamp is stamped for a whole batch,
	// so every claim from a lead ties, and the tie was being broken by SQLite's
	// unspecified row order. Order matters because citation numbers are assigned
	// by it, and under replay the resulting prompt is the cassette key.
	Seq int
}

type ClaimEdge struct {
	ID        string
	SessionID string
	FromID    string
	ToID      string
	Kind      EdgeKind
	Weight    float64
	CreatedBy string
	Rationale string
	CreatedAt time.Time
}

// ---------------------------------------------------------------------------
// ToolCall
// ---------------------------------------------------------------------------

// ToolCall is one row of the append-only cost ledger. The ledger is the source
// of truth for spend; Session.Spent is a materialized sum of it.
type ToolCall struct {
	ID         string
	SessionID  string
	LeadID     *string
	Role       Role
	Type       CallType
	Model      string
	Input      string // redacted or hashed for local queries
	Cost       Cost
	Err        string
	DurationMS int64
	CreatedAt  time.Time
}

func (t *ToolCall) Validate() error {
	if t.SessionID == "" {
		return fmt.Errorf("toolcall: empty session id")
	}
	if !t.Role.Valid() {
		return fmt.Errorf("toolcall: invalid role %q", t.Role)
	}
	if t.Type == "" {
		return fmt.Errorf("toolcall: empty type")
	}
	return t.Cost.Validate()
}

// ---------------------------------------------------------------------------
// Reservation
// ---------------------------------------------------------------------------

// Reservation is a hold placed on budget before dispatch. Charging after the
// fact lets a pool of N workers overshoot the ceiling by N lead-costs; holding
// first bounds the overshoot to estimate error on a single lead.
type Reservation struct {
	ID         string
	SessionID  string
	LeadID     *string
	Amount     int64
	Status     ReservationStatus
	CreatedAt  time.Time
	ExpiresAt  time.Time
	ResolvedAt *time.Time
}

// ---------------------------------------------------------------------------
// Span
// ---------------------------------------------------------------------------

// Span is one trace record. One span per lead is the M0 requirement; the type
// is general enough for nested spans inside an actor run.
type Span struct {
	ID        string
	SessionID string
	LeadID    *string
	ParentID  *string
	Name      string
	StartedAt time.Time
	EndedAt   *time.Time
	Status    string
	Attrs     map[string]string
}

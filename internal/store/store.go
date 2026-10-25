// Package store defines the persistence boundary.
//
// Everything above this package is dialect-agnostic. The interface is split
// into Queries (reads) and Tx (reads + writes inside one transaction) so that
// the budget invariants — check available, insert hold, update session — happen
// atomically without the ledger logic being duplicated per backend.
package store

import (
	"context"
	"errors"
	"time"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/dataset"
)

var (
	// ErrNotFound is returned by every getter when the row does not exist.
	ErrNotFound = errors.New("store: not found")
	// ErrConflict is returned when a write violates an invariant the schema
	// enforces (e.g. a settle that would drive held negative).
	ErrConflict = errors.New("store: conflict")
)

// Store is the entry point. Reads and writes are deliberately separate calls:
// on SQLite the write path is a single serialized connection while reads run
// concurrently under WAL, and callers should not have to know that.
type Store interface {
	// WithTx runs fn inside a write transaction, committing on nil and rolling
	// back on error. fn must be idempotent under retry.
	WithTx(ctx context.Context, fn func(context.Context, Tx) error) error

	// Read runs fn against a read-only view.
	Read(ctx context.Context, fn func(context.Context, Queries) error) error

	// Migrate applies any pending schema migrations.
	Migrate(ctx context.Context) error

	Close() error
}

// ClaimScore is the Verifier's verdict on one claim (§11.3).
//
// A separate type from core.Claim so a scoring write cannot touch the claim's
// text, quote, or source. Those are evidence: the quote is what §11.5 checks a
// citation against, and a verification pass that could rewrite it would make the
// grounding check circular.
type ClaimScore struct {
	ClaimID string

	// Confidence is derived from graph structure. Never a model's self-report.
	Confidence float64

	// Grounded is three-state: nil where no grounding check ran, which is most
	// claims — §11.5's re-fetch is budgeted and reserved for load-bearing ones.
	Grounded *bool

	// KeepUnverified writes Confidence and Grounded but leaves verified_at
	// unset. The zero value marks the claim verified.
	KeepUnverified bool
}

// ClaimGrounding is one §11.5 grounding verdict.
//
// Separate from ClaimScore so a grounding write cannot touch confidence. The two
// run at different times — edge inference during the loop, grounding once after it
// — and confidence has to be re-derived AFTER grounding rather than alongside it,
// since the grounding result is an input to the formula.
type ClaimGrounding struct {
	ClaimID string
	// Grounded is nil when the check ran but learned nothing about the claim: the
	// quote had vanished, or the source was unreachable. Note says which.
	Grounded *bool
	Note     string
}

// BudgetDelta is an atomic adjustment to a session's counters. Every field is
// a delta, never an absolute, so concurrent settles compose correctly.
type BudgetDelta struct {
	Spent         int64
	Held          int64
	Escrow        int64
	ToolCallCount int64
	LeadCount     int64
}

// FetchOutcome is one recorded fetch attempt (§10.4). It lives in this package
// rather than in tools/fetch so the store does not depend on the fetcher.
type FetchOutcome struct {
	ID         string
	SessionID  *string
	LeadID     *string
	URL        string
	Domain     string
	Outcome    string
	StatusCode int
	Bytes      int64
	Duration   time.Duration
	Err        string
	CreatedAt  time.Time
}

// FetchStat is one row of the outcome mix: how often a cause occurred and
// which domains it concentrated in.
type FetchStat struct {
	Outcome string
	Count   int64
	Domains []DomainCount
}

type DomainCount struct {
	Domain string
	Count  int64
}

// Queries is the read surface.
type Queries interface {
	// ListRows reads a session's extracted dataset rows in insertion order
	// (M9, §13). A read, so it lives here rather than on Tx.
	ListRows(ctx context.Context, sessionID string) ([]dataset.Row, error)

	// DatasetSchema is the schema a session's rows were extracted against.
	// Reports false when the session is not a dataset session.
	DatasetSchema(ctx context.Context, sessionID string) (dataset.Schema, bool, error)

	// ListCrossings reads the audit trail of data that left the machine for this
	// session, oldest first (§12.1, M8).
	ListCrossings(ctx context.Context, sessionID string) ([]core.Crossing, error)

	// Document reads stored source text by id, reporting false when it is absent
	// or past its TTL (toolkit mode).
	//
	// Expiry is applied HERE as well as by the sweep, because a retention promise
	// that depends on a background job having run is not a retention promise.
	Document(ctx context.Context, id string, now time.Time) (core.Document, bool, error)

	// DocumentsForSession lists a session's stored source text, newest last and
	// excluding anything past its TTL.
	DocumentsForSession(ctx context.Context, sessionID string, now time.Time) ([]core.Document, error)

	// RecentLeadCosts returns finished leads' settled costs with the actor type
	// and depth that produced them, OLDEST first, across every session.
	//
	// Feeds the estimator's warm start (§8). Across sessions on purpose: the
	// question it answers — what does a web lead at depth 2 cost on this install
	// — is not a per-session one, and a fresh session is exactly when a cold
	// estimator hurts.
	RecentLeadCosts(ctx context.Context, limit int) ([]core.LeadCost, error)

	GetSession(ctx context.Context, id string) (*core.Session, error)
	ListSessions(ctx context.Context, limit int) ([]*core.Session, error)

	GetReservation(ctx context.Context, id string) (*core.Reservation, error)
	SumHeldReservations(ctx context.Context, sessionID string) (int64, error)

	SumCosts(ctx context.Context, sessionID string) (core.Cost, error)
	SumCostsByRole(ctx context.Context, sessionID string) (map[core.Role]core.Cost, error)
	CountToolCalls(ctx context.Context, sessionID string) (int64, error)
	ListToolCalls(ctx context.Context, sessionID string, limit int) ([]*core.ToolCall, error)

	ListSpans(ctx context.Context, sessionID string) ([]*core.Span, error)

	// FetchOutcomeStats returns the outcome mix, most frequent first, with the
	// top domains per cause. topDomains <= 0 omits the domain breakdown.
	FetchOutcomeStats(ctx context.Context, since time.Time, topDomains int) ([]FetchStat, error)

	// ListFetchOutcomes returns one session's rows. The aggregate above cannot
	// answer per-URL questions, and citation verification needs to know which
	// sources were never fetched — re-reading those compares against a
	// different extraction and manufactures mismatches.
	ListFetchOutcomes(ctx context.Context, sessionID string, limit int) ([]*FetchOutcome, error)

	GetLead(ctx context.Context, id string) (*core.Lead, error)
	ListLeads(ctx context.Context, sessionID string, limit int) ([]*core.Lead, error)

	// CountLeadsByStatus is how the loop knows whether work remains. A read, so
	// a progress display never contends with the writer (§7.1).
	CountLeadsByStatus(ctx context.Context, sessionID string) (map[core.LeadStatus]int, error)

	ListClaims(ctx context.Context, sessionID string, limit int) ([]*core.Claim, error)
	CountClaims(ctx context.Context, sessionID string) (int64, error)

	// ListUnverifiedClaims returns claims the Verifier has not scored, oldest
	// first. §11.1 runs it incrementally over the session's whole claim set
	// rather than the batch one lead produced, so it has to be able to ask what
	// it has not seen.
	ListUnverifiedClaims(ctx context.Context, sessionID string, limit int) ([]*core.Claim, error)

	// ListEdges returns the session's whole graph. Not paged by claim: deriving
	// one claim's confidence needs every edge touching it in either direction,
	// and a session's graph is hundreds of edges, not millions.
	ListEdges(ctx context.Context, sessionID string, limit int) ([]*core.ClaimEdge, error)
}

// Tx is the write surface. It embeds Queries so a transaction can read its own
// uncommitted state — required for check-then-write budget operations.
type Tx interface {
	Queries

	InsertSession(ctx context.Context, s *core.Session) error
	// FinishSession marks a session terminal and records why it stopped.
	//
	// The reason is a ceiling name — "max_tool_calls", "max_leads",
	// "max_wallclock" — or empty for a session that simply finished. It is
	// taken alongside the status rather than in a separate call because the two
	// are one fact: a status of "budget_exhausted" written without its reason is
	// the defect this replaced.
	FinishSession(ctx context.Context, id string, status core.SessionStatus, because string) error
	// SetSessionReport stores the rendered answer and why it is degraded, if it is.
	SetSessionReport(ctx context.Context, id, reportMD, degraded string) error

	// SetSessionReportCitations stores the report's bibliography as opaque JSON:
	// the [n] markers in the prose resolved to sources. Written with the report,
	// because the numbering cannot be recomputed afterwards.
	SetSessionReportCitations(ctx context.Context, id, citationsJSON string) error

	// SetSessionReceipt stores the citation receipt as opaque JSON. Empty means
	// verification never ran, which is distinct from a receipt of zeroes.
	SetSessionReceipt(ctx context.Context, id, receiptJSON string) error

	// InsertRows writes a batch of extracted dataset rows (M9, §13).
	//
	// All or nothing, for the same reason as InsertClaims: a partial batch would
	// leave the merge reasoning over evidence that was never fully recorded, and
	// a dataset is precisely a thing somebody counts.
	InsertRows(ctx context.Context, sessionID string, rows []dataset.Row) error

	// DeleteSession removes a session and everything filed under it — claims,
	// edges, rows, crossings, documents — by foreign-key cascade.
	//
	// Added with the document store, because "documents are deleted with their
	// session" was not a promise anything could keep while nothing could delete a
	// session.
	DeleteSession(ctx context.Context, id string) error

	// InsertDocument stores source text a quote can later be checked against.
	InsertDocument(ctx context.Context, doc core.Document) error

	// ExpireDocumentForTest backdates a document's expiry.
	//
	// On the interface because retention is a property worth testing through the
	// real path rather than by reaching around the store, and there is no
	// legitimate production caller — a document's TTL is set when it is stored.
	ExpireDocumentForTest(ctx context.Context, id string) error

	// PurgeExpiredDocuments deletes source text past its TTL and reports how many
	// rows went. Reclaims disk; correctness does not depend on it, since reads
	// apply expiry themselves.
	PurgeExpiredDocuments(ctx context.Context, now time.Time) (int64, error)

	// InsertCrossings records what crossed the aggregation gate, and what was
	// refused or withheld (§12.1).
	//
	// All or nothing, like the others: an audit trail missing half a batch invites
	// exactly the wrong conclusion about what left the machine.
	InsertCrossings(ctx context.Context, crossings []core.Crossing) error

	// SetDatasetSchema records the schema a session's rows were extracted
	// against. Reading a dataset back needs it: field order for the header, and
	// which fields are keys for the merge.
	SetDatasetSchema(ctx context.Context, sessionID string, schema dataset.Schema) error
	ApplyBudgetDelta(ctx context.Context, sessionID string, d BudgetDelta) error

	InsertReservation(ctx context.Context, r *core.Reservation) error
	ResolveReservation(ctx context.Context, id string, status core.ReservationStatus, at time.Time) error
	ExpireStaleReservations(ctx context.Context, now time.Time) (int, error)
	// ReleaseSessionHolds releases every held reservation for one session,
	// regardless of TTL. For a session that ended without running its own settle
	// path.
	ReleaseSessionHolds(ctx context.Context, sessionID string) (int, error)

	InsertToolCall(ctx context.Context, tc *core.ToolCall) error

	RecordFetchOutcome(ctx context.Context, o *FetchOutcome) error

	InsertLead(ctx context.Context, l *core.Lead) error

	// PromoteDeferredLeads makes owed follow-up work eligible to lease, oldest
	// and highest priority first, and reports how many it moved.
	//
	// limit <= 0 promotes all of them. The pacing that held them back was one
	// verification pass's budget; what bounds the session is §8.5's ceilings,
	// and they are checked on every turn of the loop whether a lead was
	// promoted or not.
	PromoteDeferredLeads(ctx context.Context, sessionID string, limit int) (int, error)
	// SetLeadStatus moves a lead to a terminal state. owner must match the
	// lease holder; an empty owner skips the check, for callers that legitimately
	// have no lease. Without it a worker that lost its lease could terminalize a
	// lead another worker now holds.
	SetLeadStatus(ctx context.Context, id, owner string, status core.LeadStatus) error

	// LeaseNextLead atomically claims the highest-priority queued lead.
	//
	// Dequeue and lease must be one statement. Two workers that SELECT then
	// UPDATE will both see the same row and both run it — paying twice for one
	// lead, which the budget cannot detect because both charges are real.
	// Returns nil when the queue is empty.
	LeaseNextLead(ctx context.Context, sessionID, owner string, expires time.Time) (*core.Lead, error)

	// RenewLease extends a lease the worker still holds. Reports false when the
	// lease was taken by the recovery sweep, which is how a worker learns it
	// stalled long enough to be presumed dead.
	RenewLease(ctx context.Context, leadID, owner string, expires time.Time) (bool, error)

	// ReleaseLease returns a lead to the queue without completing it.
	ReleaseLease(ctx context.Context, leadID, owner string) error

	// SweepAbandonedSessions marks running sessions whose process is gone.
	//
	// The third half of §9.4. Leases and reservations are reclaimed at boot,
	// but nothing touched the session row, so a killed process left a session
	// `running` forever — visible in `sessions` and reconciled by `doctor` for
	// the life of the database.
	//
	// Liveness is judged by updated_at plus the absence of a live lease, which
	// is the same signal leases already use. A live run touches updated_at on
	// every settle and every counted lead, so silence past the threshold with
	// nothing leased means the process is gone.
	SweepAbandonedSessions(ctx context.Context, idleSince time.Time) (int, error)

	// SweepExpiredLeases requeues leads whose worker died. Without it a crash
	// strands them as leased forever, with no way out (§9.4).
	//
	// sessionID scopes the sweep. Empty means every session, which is what boot
	// recovery needs; a running executor must pass its own, or it requeues
	// another live process's in-flight leads and both end up running the lead.
	SweepExpiredLeases(ctx context.Context, sessionID string, now time.Time) (int, error)

	// InsertClaims writes a batch in one transaction. Claims from one actor
	// run land together or not at all: a partial batch would leave the graph
	// citing a lead that reported failure.
	InsertClaims(ctx context.Context, claims []core.Claim) error

	// InsertEdges writes graph edges, upserting on (from_id, to_id, kind).
	//
	// Upsert rather than insert because §11.1 re-verifies: a pair judged again
	// with better context should update its weight and rationale, not fail on the
	// UNIQUE constraint or accumulate a second row saying the same thing.
	InsertEdges(ctx context.Context, edges []core.ClaimEdge) error

	// ScoreClaims writes the Verifier's output for a batch of claims.
	ScoreClaims(ctx context.Context, scores []ClaimScore) error

	// SetClaimGrounding records §11.5 grounding verdicts, leaving confidence
	// alone. Confidence is re-derived afterwards, because the grounding result
	// feeds the formula that produces it.
	SetClaimGrounding(ctx context.Context, results []ClaimGrounding) error

	StartSpan(ctx context.Context, s *core.Span) error
	EndSpan(ctx context.Context, id string, endedAt time.Time, status string) error
}

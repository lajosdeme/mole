package sqlite

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/dataset"
	"github.com/lajosdeme/mole/internal/store"
)

func ensureDir(dir string) error {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("sqlite: create %s: %w", dir, err)
	}
	return nil
}

// execer is satisfied by both *sql.DB and *sql.Tx, so one implementation
// serves the read pool and the write transaction.
type execer interface {
	ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error)
	QueryContext(ctx context.Context, query string, args ...any) (*sql.Rows, error)
	QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row
}

type queries struct{ q execer }

var (
	_ store.Queries = (*queries)(nil)
	_ store.Tx      = (*queries)(nil)
)

// ---------------------------------------------------------------------------
// Time helpers — timestamps are unix microseconds UTC.
// ---------------------------------------------------------------------------

func toMicros(t time.Time) int64 { return t.UTC().UnixMicro() }

func fromMicros(v int64) time.Time { return time.UnixMicro(v).UTC() }

func nullMicros(t *time.Time) sql.NullInt64 {
	if t == nil {
		return sql.NullInt64{}
	}
	return sql.NullInt64{Int64: toMicros(*t), Valid: true}
}

func micrasPtr(v sql.NullInt64) *time.Time {
	if !v.Valid {
		return nil
	}
	t := fromMicros(v.Int64)
	return &t
}

func nullStr(s *string) sql.NullString {
	if s == nil {
		return sql.NullString{}
	}
	return sql.NullString{String: *s, Valid: true}
}

func strPtr(v sql.NullString) *string {
	if !v.Valid {
		return nil
	}
	s := v.String
	return &s
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

const sessionCols = `id, prompt, mode, actor_types, budget_unit, budget, spent, held, escrow,
	max_tool_calls, max_leads, max_wallclock_ns, tool_call_count, lead_count,
	status, stopped_because, created_at, updated_at, report_md, report_degraded,
	receipt_json, report_citations_json`

func (t *queries) InsertSession(ctx context.Context, s *core.Session) error {
	if err := s.Validate(); err != nil {
		return err
	}
	now := time.Now().UTC()
	if s.CreatedAt.IsZero() {
		s.CreatedAt = now
	}
	s.UpdatedAt = now

	_, err := t.q.ExecContext(ctx, `
		INSERT INTO sessions (`+sessionCols+`)
		VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		s.ID, s.Prompt, string(s.Mode), core.EncodeActorTypes(s.ActorTypes),
		string(s.BudgetUnit), s.Budget, s.Spent, s.Held, s.Escrow,
		s.MaxToolCalls, s.MaxLeads, int64(s.MaxWallClock), s.ToolCallCount, s.LeadCount,
		string(s.Status), s.StoppedBecause, toMicros(s.CreatedAt), toMicros(s.UpdatedAt),
		s.Report, s.ReportDegraded, s.ReceiptJSON, s.ReportCitationsJSON)
	if err != nil {
		return fmt.Errorf("sqlite: insert session: %w", err)
	}
	return nil
}

func scanSession(sc interface{ Scan(...any) error }) (*core.Session, error) {
	var (
		s           core.Session
		mode        string
		actorTypes  string
		budgetUnit  string
		status      string
		wallclockNS int64
		created     int64
		updated     int64
	)
	err := sc.Scan(&s.ID, &s.Prompt, &mode, &actorTypes, &budgetUnit,
		&s.Budget, &s.Spent, &s.Held, &s.Escrow,
		&s.MaxToolCalls, &s.MaxLeads, &wallclockNS, &s.ToolCallCount, &s.LeadCount,
		&status, &s.StoppedBecause, &created, &updated,
		&s.Report, &s.ReportDegraded, &s.ReceiptJSON, &s.ReportCitationsJSON)
	if err != nil {
		return nil, err
	}
	s.Mode = core.Mode(mode)
	s.ActorTypes = core.DecodeActorTypes(actorTypes)
	s.BudgetUnit = core.BudgetUnit(budgetUnit)
	s.Status = core.SessionStatus(status)
	s.MaxWallClock = time.Duration(wallclockNS)
	s.CreatedAt = fromMicros(created)
	s.UpdatedAt = fromMicros(updated)
	return &s, nil
}

func (t *queries) GetSession(ctx context.Context, id string) (*core.Session, error) {
	row := t.q.QueryRowContext(ctx, `SELECT `+sessionCols+` FROM sessions WHERE id = ?`, id)
	s, err := scanSession(row)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, store.ErrNotFound
	}
	if err != nil {
		return nil, fmt.Errorf("sqlite: get session: %w", err)
	}
	return s, nil
}

func (t *queries) ListSessions(ctx context.Context, limit int) ([]*core.Session, error) {
	if limit <= 0 {
		limit = 50
	}
	rows, err := t.q.QueryContext(ctx,
		`SELECT `+sessionCols+` FROM sessions ORDER BY created_at DESC LIMIT ?`, limit)
	if err != nil {
		return nil, fmt.Errorf("sqlite: list sessions: %w", err)
	}
	defer rows.Close()

	var out []*core.Session
	for rows.Next() {
		s, err := scanSession(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, rows.Err()
}

func (t *queries) SetSessionReport(ctx context.Context, id, reportMD, degraded string) error {
	res, err := t.q.ExecContext(ctx,
		`UPDATE sessions SET report_md = ?, report_degraded = ?, updated_at = ? WHERE id = ?`,
		reportMD, degraded, toMicros(time.Now()), id)
	if err != nil {
		return fmt.Errorf("sqlite: set report: %w", err)
	}
	return requireOneRow(res, "session")
}

// SetSessionReportCitations stores the report's bibliography as opaque JSON.
func (t *queries) SetSessionReportCitations(ctx context.Context, id, citationsJSON string) error {
	res, err := t.q.ExecContext(ctx,
		`UPDATE sessions SET report_citations_json = ?, updated_at = ? WHERE id = ?`,
		citationsJSON, toMicros(time.Now()), id)
	if err != nil {
		return fmt.Errorf("sqlite: set report citations: %w", err)
	}
	return requireOneRow(res, "session")
}

// SetSessionReceipt stores the citation receipt as opaque JSON.
func (t *queries) SetSessionReceipt(ctx context.Context, id, receiptJSON string) error {
	res, err := t.q.ExecContext(ctx,
		`UPDATE sessions SET receipt_json = ?, updated_at = ? WHERE id = ?`,
		receiptJSON, toMicros(time.Now()), id)
	if err != nil {
		return fmt.Errorf("sqlite: set receipt: %w", err)
	}
	return requireOneRow(res, "session")
}

func (t *queries) FinishSession(
	ctx context.Context, id string, status core.SessionStatus, because string,
) error {
	if !status.Valid() {
		return fmt.Errorf("sqlite: invalid status %q", status)
	}
	res, err := t.q.ExecContext(ctx,
		`UPDATE sessions SET status = ?, stopped_because = ?, updated_at = ? WHERE id = ?`,
		string(status), because, toMicros(time.Now()), id)
	if err != nil {
		return fmt.Errorf("sqlite: finish session: %w", err)
	}
	return requireOneRow(res, "session")
}

// ApplyBudgetDelta adjusts the session counters atomically.
//
// The WHERE clause carries the invariants rather than trusting the caller:
// spent, held, and escrow may never go negative. A settle that would drive held
// below zero — a double-settle, or a settle against a released reservation —
// matches no rows and surfaces as ErrConflict instead of silently corrupting
// the ledger.
func (t *queries) ApplyBudgetDelta(ctx context.Context, sessionID string, d store.BudgetDelta) error {
	res, err := t.q.ExecContext(ctx, `
		UPDATE sessions SET
			spent           = spent           + ?,
			held            = held            + ?,
			escrow          = escrow          + ?,
			tool_call_count = tool_call_count + ?,
			lead_count      = lead_count      + ?,
			updated_at      = ?
		WHERE id = ?
		  AND spent           + ? >= 0
		  AND held            + ? >= 0
		  AND escrow          + ? >= 0
		  AND tool_call_count + ? >= 0
		  AND lead_count      + ? >= 0`,
		d.Spent, d.Held, d.Escrow, d.ToolCallCount, d.LeadCount, toMicros(time.Now()), sessionID,
		d.Spent, d.Held, d.Escrow, d.ToolCallCount, d.LeadCount)
	if err != nil {
		return fmt.Errorf("sqlite: apply budget delta: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return fmt.Errorf("%w: budget delta %+v would violate a non-negative invariant on session %s",
			store.ErrConflict, d, sessionID)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Reservations
// ---------------------------------------------------------------------------

const reservationCols = `id, session_id, lead_id, amount, status, created_at, expires_at, resolved_at`

func (t *queries) InsertReservation(ctx context.Context, r *core.Reservation) error {
	if r.Amount <= 0 {
		return fmt.Errorf("sqlite: reservation amount must be positive, got %d", r.Amount)
	}
	_, err := t.q.ExecContext(ctx, `
		INSERT INTO reservations (`+reservationCols+`) VALUES (?,?,?,?,?,?,?,?)`,
		r.ID, r.SessionID, nullStr(r.LeadID), r.Amount, string(r.Status),
		toMicros(r.CreatedAt), toMicros(r.ExpiresAt), nullMicros(r.ResolvedAt))
	if err != nil {
		return fmt.Errorf("sqlite: insert reservation: %w", err)
	}
	return nil
}

func (t *queries) GetReservation(ctx context.Context, id string) (*core.Reservation, error) {
	var (
		r        core.Reservation
		leadID   sql.NullString
		status   string
		created  int64
		expires  int64
		resolved sql.NullInt64
	)
	err := t.q.QueryRowContext(ctx,
		`SELECT `+reservationCols+` FROM reservations WHERE id = ?`, id).
		Scan(&r.ID, &r.SessionID, &leadID, &r.Amount, &status, &created, &expires, &resolved)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, store.ErrNotFound
	}
	if err != nil {
		return nil, fmt.Errorf("sqlite: get reservation: %w", err)
	}
	r.LeadID = strPtr(leadID)
	r.Status = core.ReservationStatus(status)
	r.CreatedAt = fromMicros(created)
	r.ExpiresAt = fromMicros(expires)
	r.ResolvedAt = micrasPtr(resolved)
	return &r, nil
}

// ResolveReservation moves a held reservation to settled or released. The
// status guard in the WHERE clause makes the transition idempotent-safe: a
// second settle affects no rows and reports a conflict rather than
// double-releasing the hold.
func (t *queries) ResolveReservation(ctx context.Context, id string, status core.ReservationStatus, at time.Time) error {
	if status != core.ReservationSettled && status != core.ReservationReleased {
		return fmt.Errorf("sqlite: cannot resolve reservation to %q", status)
	}
	res, err := t.q.ExecContext(ctx,
		`UPDATE reservations SET status = ?, resolved_at = ? WHERE id = ? AND status = 'held'`,
		string(status), toMicros(at), id)
	if err != nil {
		return fmt.Errorf("sqlite: resolve reservation: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return fmt.Errorf("%w: reservation %s is not held", store.ErrConflict, id)
	}
	return nil
}

func (t *queries) SumHeldReservations(ctx context.Context, sessionID string) (int64, error) {
	var total sql.NullInt64
	err := t.q.QueryRowContext(ctx,
		`SELECT SUM(amount) FROM reservations WHERE session_id = ? AND status = 'held'`, sessionID).
		Scan(&total)
	if err != nil {
		return 0, fmt.Errorf("sqlite: sum held: %w", err)
	}
	return total.Int64, nil
}

// ExpireStaleReservations releases holds whose lease has passed. This is the
// reservation half of crash recovery: a worker that died mid-run leaves budget
// held forever otherwise.
// ReleaseSessionHolds releases every held reservation for one session.
//
// Not TTL-gated, unlike ExpireStaleReservations. It exists for the case where a
// session ends abruptly and its own settle path never ran — a panic in the
// pipeline, principally — where waiting out the reservation TTL means the money
// reads as held on a session that is already finished.
func (t *queries) ReleaseSessionHolds(ctx context.Context, sessionID string) (int, error) {
	return t.releaseHeld(ctx,
		`SELECT id, session_id, amount FROM reservations
		 WHERE status = 'held' AND session_id = ?`, sessionID)
}

func (t *queries) ExpireStaleReservations(ctx context.Context, now time.Time) (int, error) {
	return t.releaseHeld(ctx,
		`SELECT id, session_id, amount FROM reservations
		 WHERE status = 'held' AND expires_at < ?`, toMicros(now))
}

// releaseHeld resolves and credits back every reservation the query returns.
func (t *queries) releaseHeld(ctx context.Context, query string, args ...any) (int, error) {
	now := time.Now()
	rows, err := t.q.QueryContext(ctx, query, args...)
	if err != nil {
		return 0, fmt.Errorf("sqlite: scan held reservations: %w", err)
	}

	type stale struct {
		id, sessionID string
		amount        int64
	}
	var list []stale
	for rows.Next() {
		var s stale
		if err := rows.Scan(&s.id, &s.sessionID, &s.amount); err != nil {
			rows.Close()
			return 0, err
		}
		list = append(list, s)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return 0, err
	}

	// Per-reservation, and one bad row must not abort the batch.
	//
	// This is boot recovery: it runs before anything else and it sweeps EVERY
	// session, so a single unreleasable hold used to take the whole transaction
	// down and leave every other session's stale reservations in place. Seen
	// for real, from reservations orphaned by a session deleted with foreign
	// keys disabled.
	var released int
	var firstErr error
	for _, s := range list {
		if err := t.ResolveReservation(ctx, s.id, core.ReservationReleased, now); err != nil {
			if firstErr == nil {
				firstErr = fmt.Errorf("sqlite: resolve reservation %s: %w", s.id, err)
			}
			continue
		}
		err := t.ApplyBudgetDelta(ctx, s.sessionID, store.BudgetDelta{Held: -s.amount})
		if errors.Is(err, store.ErrConflict) || errors.Is(err, store.ErrNotFound) {
			// The session is gone, or its counters cannot absorb the credit.
			// The reservation is resolved either way — there is no counter left
			// to give the money back to, and leaving it held would make it
			// unreleasable forever.
			released++
			continue
		}
		if err != nil {
			if firstErr == nil {
				firstErr = fmt.Errorf("sqlite: release hold on %s: %w", s.sessionID, err)
			}
			continue
		}
		released++
	}
	return released, firstErr
}

// ---------------------------------------------------------------------------
// Tool calls (the ledger)
// ---------------------------------------------------------------------------

const toolCallCols = `id, session_id, lead_id, role, type, model, input,
	usd_micros, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
	err, duration_ms, created_at`

func (t *queries) InsertToolCall(ctx context.Context, tc *core.ToolCall) error {
	if err := tc.Validate(); err != nil {
		return err
	}
	if tc.ID == "" {
		tc.ID = core.NewToolCallID()
	}
	if tc.CreatedAt.IsZero() {
		tc.CreatedAt = time.Now().UTC()
	}
	_, err := t.q.ExecContext(ctx, `
		INSERT INTO tool_calls (`+toolCallCols+`) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		tc.ID, tc.SessionID, nullStr(tc.LeadID), string(tc.Role), string(tc.Type), tc.Model, tc.Input,
		tc.Cost.USDMicros, tc.Cost.InputTokens, tc.Cost.OutputTokens,
		tc.Cost.CacheReadTokens, tc.Cost.CacheWriteTokens,
		tc.Err, tc.DurationMS, toMicros(tc.CreatedAt))
	if err != nil {
		return fmt.Errorf("sqlite: insert tool call: %w", err)
	}
	return nil
}

const costSumExpr = `COALESCE(SUM(usd_micros),0), COALESCE(SUM(input_tokens),0),
	COALESCE(SUM(output_tokens),0), COALESCE(SUM(cache_read_tokens),0),
	COALESCE(SUM(cache_write_tokens),0)`

func (t *queries) SumCosts(ctx context.Context, sessionID string) (core.Cost, error) {
	var c core.Cost
	err := t.q.QueryRowContext(ctx,
		`SELECT `+costSumExpr+` FROM tool_calls WHERE session_id = ?`, sessionID).
		Scan(&c.USDMicros, &c.InputTokens, &c.OutputTokens, &c.CacheReadTokens, &c.CacheWriteTokens)
	if err != nil {
		return c, fmt.Errorf("sqlite: sum costs: %w", err)
	}
	return c, nil
}

// SumCostsByRole answers "how much of this session went to verification vs.
// execution" as one GROUP BY. This is the entire reason ToolCall.Role exists.
func (t *queries) SumCostsByRole(ctx context.Context, sessionID string) (map[core.Role]core.Cost, error) {
	rows, err := t.q.QueryContext(ctx,
		`SELECT role, `+costSumExpr+` FROM tool_calls WHERE session_id = ? GROUP BY role`, sessionID)
	if err != nil {
		return nil, fmt.Errorf("sqlite: sum costs by role: %w", err)
	}
	defer rows.Close()

	out := map[core.Role]core.Cost{}
	for rows.Next() {
		var (
			role string
			c    core.Cost
		)
		if err := rows.Scan(&role, &c.USDMicros, &c.InputTokens, &c.OutputTokens,
			&c.CacheReadTokens, &c.CacheWriteTokens); err != nil {
			return nil, err
		}
		out[core.Role(role)] = c
	}
	return out, rows.Err()
}

func (t *queries) CountToolCalls(ctx context.Context, sessionID string) (int64, error) {
	var n int64
	err := t.q.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM tool_calls WHERE session_id = ?`, sessionID).Scan(&n)
	if err != nil {
		return 0, fmt.Errorf("sqlite: count tool calls: %w", err)
	}
	return n, nil
}

func (t *queries) ListToolCalls(ctx context.Context, sessionID string, limit int) ([]*core.ToolCall, error) {
	if limit <= 0 {
		limit = 200
	}
	rows, err := t.q.QueryContext(ctx,
		`SELECT `+toolCallCols+` FROM tool_calls WHERE session_id = ? ORDER BY created_at LIMIT ?`,
		sessionID, limit)
	if err != nil {
		return nil, fmt.Errorf("sqlite: list tool calls: %w", err)
	}
	defer rows.Close()

	var out []*core.ToolCall
	for rows.Next() {
		var (
			tc      core.ToolCall
			leadID  sql.NullString
			role    string
			typ     string
			created int64
		)
		if err := rows.Scan(&tc.ID, &tc.SessionID, &leadID, &role, &typ, &tc.Model, &tc.Input,
			&tc.Cost.USDMicros, &tc.Cost.InputTokens, &tc.Cost.OutputTokens,
			&tc.Cost.CacheReadTokens, &tc.Cost.CacheWriteTokens,
			&tc.Err, &tc.DurationMS, &created); err != nil {
			return nil, err
		}
		tc.LeadID = strPtr(leadID)
		tc.Role = core.Role(role)
		tc.Type = core.CallType(typ)
		tc.CreatedAt = fromMicros(created)
		out = append(out, &tc)
	}
	return out, rows.Err()
}

// ---------------------------------------------------------------------------
// Fetch outcomes (§10.4)
// ---------------------------------------------------------------------------

func (t *queries) RecordFetchOutcome(ctx context.Context, o *store.FetchOutcome) error {
	if o.Outcome == "" {
		return fmt.Errorf("sqlite: fetch outcome is empty")
	}
	if o.ID == "" {
		o.ID = core.NewSpanID()
	}
	if o.CreatedAt.IsZero() {
		o.CreatedAt = time.Now().UTC()
	}
	_, err := t.q.ExecContext(ctx, `
		INSERT INTO fetch_outcomes
			(id, session_id, lead_id, url, domain, outcome, status_code, bytes, duration_ms, err, created_at)
		VALUES (?,?,?,?,?,?,?,?,?,?,?)`,
		o.ID, nullStr(o.SessionID), nullStr(o.LeadID), o.URL, o.Domain, o.Outcome,
		o.StatusCode, o.Bytes, o.Duration.Milliseconds(), o.Err, toMicros(o.CreatedAt))
	if err != nil {
		return fmt.Errorf("sqlite: record fetch outcome: %w", err)
	}
	return nil
}

// ListFetchOutcomes returns one session's fetch rows, oldest first.
func (t *queries) ListFetchOutcomes(ctx context.Context, sessionID string, limit int) ([]*store.FetchOutcome, error) {
	if limit <= 0 {
		limit = 1000
	}
	rows, err := t.q.QueryContext(ctx, `
		SELECT id, session_id, lead_id, url, domain, outcome, status_code, bytes, duration_ms, err, created_at
		FROM fetch_outcomes WHERE session_id = ? ORDER BY created_at ASC LIMIT ?`, sessionID, limit)
	if err != nil {
		return nil, fmt.Errorf("sqlite: list fetch outcomes: %w", err)
	}
	defer rows.Close()

	var out []*store.FetchOutcome
	for rows.Next() {
		var (
			o          store.FetchOutcome
			sid, lid   sql.NullString
			durationMS int64
			createdAt  int64
		)
		if err := rows.Scan(&o.ID, &sid, &lid, &o.URL, &o.Domain, &o.Outcome,
			&o.StatusCode, &o.Bytes, &durationMS, &o.Err, &createdAt); err != nil {
			return nil, err
		}
		if sid.Valid {
			v := sid.String
			o.SessionID = &v
		}
		if lid.Valid {
			v := lid.String
			o.LeadID = &v
		}
		o.Duration = time.Duration(durationMS) * time.Millisecond
		o.CreatedAt = fromMicros(createdAt)
		out = append(out, &o)
	}
	return out, rows.Err()
}

// FetchOutcomeStats produces the outcome mix plus the ranked domains per cause.
//
// Both halves matter to §17.1: the rate says whether a capability gap is worth
// closing, and the domain list often says it is really three sites rather than
// an architecture problem.
func (t *queries) FetchOutcomeStats(ctx context.Context, since time.Time, topDomains int) ([]store.FetchStat, error) {
	rows, err := t.q.QueryContext(ctx, `
		SELECT outcome, COUNT(*) FROM fetch_outcomes
		WHERE created_at >= ?
		GROUP BY outcome ORDER BY COUNT(*) DESC`, toMicros(since))
	if err != nil {
		return nil, fmt.Errorf("sqlite: fetch outcome stats: %w", err)
	}

	var stats []store.FetchStat
	for rows.Next() {
		var s store.FetchStat
		if err := rows.Scan(&s.Outcome, &s.Count); err != nil {
			rows.Close()
			return nil, err
		}
		stats = append(stats, s)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}

	if topDomains <= 0 {
		return stats, nil
	}

	for i := range stats {
		dRows, err := t.q.QueryContext(ctx, `
			SELECT domain, COUNT(*) FROM fetch_outcomes
			WHERE outcome = ? AND created_at >= ? AND domain != ''
			GROUP BY domain ORDER BY COUNT(*) DESC LIMIT ?`,
			stats[i].Outcome, toMicros(since), topDomains)
		if err != nil {
			return nil, fmt.Errorf("sqlite: fetch outcome domains: %w", err)
		}
		for dRows.Next() {
			var dc store.DomainCount
			if err := dRows.Scan(&dc.Domain, &dc.Count); err != nil {
				dRows.Close()
				return nil, err
			}
			stats[i].Domains = append(stats[i].Domains, dc)
		}
		dRows.Close()
		if err := dRows.Err(); err != nil {
			return nil, err
		}
	}
	return stats, nil
}

// ---------------------------------------------------------------------------
// Spans
// ---------------------------------------------------------------------------

func (t *queries) StartSpan(ctx context.Context, s *core.Span) error {
	if s.ID == "" {
		s.ID = core.NewSpanID()
	}
	if s.StartedAt.IsZero() {
		s.StartedAt = time.Now().UTC()
	}
	attrs := "{}"
	if len(s.Attrs) > 0 {
		b, err := json.Marshal(s.Attrs)
		if err != nil {
			return fmt.Errorf("sqlite: marshal span attrs: %w", err)
		}
		attrs = string(b)
	}
	_, err := t.q.ExecContext(ctx, `
		INSERT INTO spans (id, session_id, lead_id, parent_id, name, started_at, ended_at, status, attrs)
		VALUES (?,?,?,?,?,?,?,?,?)`,
		s.ID, s.SessionID, nullStr(s.LeadID), nullStr(s.ParentID), s.Name,
		toMicros(s.StartedAt), nullMicros(s.EndedAt), s.Status, attrs)
	if err != nil {
		return fmt.Errorf("sqlite: insert span: %w", err)
	}
	return nil
}

func (t *queries) EndSpan(ctx context.Context, id string, endedAt time.Time, status string) error {
	res, err := t.q.ExecContext(ctx,
		`UPDATE spans SET ended_at = ?, status = ? WHERE id = ? AND ended_at IS NULL`,
		toMicros(endedAt), status, id)
	if err != nil {
		return fmt.Errorf("sqlite: end span: %w", err)
	}
	return requireOneRow(res, "span")
}

func (t *queries) ListSpans(ctx context.Context, sessionID string) ([]*core.Span, error) {
	rows, err := t.q.QueryContext(ctx, `
		SELECT id, session_id, lead_id, parent_id, name, started_at, ended_at, status, attrs
		FROM spans WHERE session_id = ? ORDER BY started_at`, sessionID)
	if err != nil {
		return nil, fmt.Errorf("sqlite: list spans: %w", err)
	}
	defer rows.Close()

	var out []*core.Span
	for rows.Next() {
		var (
			s        core.Span
			leadID   sql.NullString
			parentID sql.NullString
			started  int64
			ended    sql.NullInt64
			attrs    string
		)
		if err := rows.Scan(&s.ID, &s.SessionID, &leadID, &parentID, &s.Name,
			&started, &ended, &s.Status, &attrs); err != nil {
			return nil, err
		}
		s.LeadID = strPtr(leadID)
		s.ParentID = strPtr(parentID)
		s.StartedAt = fromMicros(started)
		s.EndedAt = micrasPtr(ended)
		if attrs != "" && attrs != "{}" {
			if err := json.Unmarshal([]byte(attrs), &s.Attrs); err != nil {
				return nil, fmt.Errorf("sqlite: unmarshal span attrs: %w", err)
			}
		}
		out = append(out, &s)
	}
	return out, rows.Err()
}

// ---------------------------------------------------------------------------

func requireOneRow(res sql.Result, what string) error {
	n, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return fmt.Errorf("%s: %w", what, store.ErrNotFound)
	}
	return nil
}

// isUniqueViolation reports whether err is a UNIQUE/PRIMARY KEY conflict.
// The driver does not export a typed error for this, so the check is on the
// message — narrow enough to be safe, and it is only used to convert a
// duplicate insert into ErrConflict.
func isUniqueViolation(err error) bool {
	return err != nil && strings.Contains(err.Error(), "UNIQUE constraint failed")
}

// ---------------------------------------------------------------------------
// Leads
// ---------------------------------------------------------------------------

const leadCols = `id, session_id, actor_type, query, parent_id, depth, priority,
	status, lease_owner, lease_expires, root_claim_id, verify_depth,
	created_at, updated_at`

func (t *queries) InsertLead(ctx context.Context, l *core.Lead) error {
	if l.ID == "" {
		l.ID = core.NewLeadID()
	}
	now := time.Now().UTC()
	if l.CreatedAt.IsZero() {
		l.CreatedAt = now
	}
	l.UpdatedAt = now
	if l.Status == "" {
		l.Status = core.LeadQueued
	}

	_, err := t.q.ExecContext(ctx, `
		INSERT INTO leads (`+leadCols+`) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		l.ID, l.SessionID, string(l.ActorType), l.Query, nullStr(l.ParentID),
		l.Depth, l.Priority, string(l.Status),
		nullStr(l.LeaseOwner), nullMicros(l.LeaseExpires),
		nullStr(l.RootClaimID), l.VerifyDepth,
		toMicros(l.CreatedAt), toMicros(l.UpdatedAt))
	if err != nil {
		return fmt.Errorf("sqlite: insert lead: %w", err)
	}
	return nil
}

func scanLead(sc interface{ Scan(...any) error }) (*core.Lead, error) {
	var (
		l         core.Lead
		actorType string
		parentID  sql.NullString
		status    string
		owner     sql.NullString
		expires   sql.NullInt64
		rootClaim sql.NullString
		created   int64
		updated   int64
	)
	err := sc.Scan(&l.ID, &l.SessionID, &actorType, &l.Query, &parentID,
		&l.Depth, &l.Priority, &status, &owner, &expires,
		&rootClaim, &l.VerifyDepth, &created, &updated)
	if err != nil {
		return nil, err
	}
	l.ActorType = core.ActorType(actorType)
	l.ParentID = strPtr(parentID)
	l.Status = core.LeadStatus(status)
	l.LeaseOwner = strPtr(owner)
	l.LeaseExpires = micrasPtr(expires)
	l.RootClaimID = strPtr(rootClaim)
	l.CreatedAt = fromMicros(created)
	l.UpdatedAt = fromMicros(updated)
	return &l, nil
}

func (t *queries) GetLead(ctx context.Context, id string) (*core.Lead, error) {
	row := t.q.QueryRowContext(ctx, `SELECT `+leadCols+` FROM leads WHERE id = ?`, id)
	l, err := scanLead(row)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, store.ErrNotFound
	}
	if err != nil {
		return nil, fmt.Errorf("sqlite: get lead: %w", err)
	}
	return l, nil
}

func (t *queries) ListLeads(ctx context.Context, sessionID string, limit int) ([]*core.Lead, error) {
	if limit <= 0 {
		limit = 200
	}
	rows, err := t.q.QueryContext(ctx,
		`SELECT `+leadCols+` FROM leads WHERE session_id = ? ORDER BY created_at LIMIT ?`,
		sessionID, limit)
	if err != nil {
		return nil, fmt.Errorf("sqlite: list leads: %w", err)
	}
	defer rows.Close()

	var out []*core.Lead
	for rows.Next() {
		l, err := scanLead(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, l)
	}
	return out, rows.Err()
}

func (t *queries) SetLeadStatus(ctx context.Context, id, owner string, status core.LeadStatus) error {
	// Clear the lease alongside the status. Leaving lease_owner set on a
	// terminal lead let ReleaseLease — which matches on owner alone — flip a
	// finished lead back to queued, to be re-run and re-charged.
	query := `UPDATE leads SET status = ?, lease_owner = NULL, lease_expires = NULL, updated_at = ?
	           WHERE id = ?`
	args := []any{string(status), toMicros(time.Now()), id}
	if owner != "" {
		// A worker that lost its lease must not terminalize a lead someone else
		// now holds.
		query += ` AND lease_owner = ?`
		args = append(args, owner)
	}

	res, err := t.q.ExecContext(ctx, query, args...)
	if err != nil {
		return fmt.Errorf("sqlite: set lead status: %w", err)
	}
	return requireOneRow(res, "lead")
}

// ---------------------------------------------------------------------------
// Lead queue (§9.2, §9.4)
// ---------------------------------------------------------------------------

// LeaseNextLead atomically claims the highest-priority queued lead.
//
// The UPDATE ... WHERE id = (SELECT ...) is deliberately one statement, but be
// clear about what that buys TODAY: nothing. §7.1's single-writer pool already
// serializes the enclosing transaction, so a SELECT-then-UPDATE would be just
// as safe — verified by writing the naive version and watching the concurrency
// test still pass.
//
// It is written this way because the safety currently comes from the
// deployment rather than the query, and that is a fragile place for it. M5
// turns on a worker pool; a future backend may allow concurrent writers. When
// either happens, two workers seeing the same row both run the lead and pay
// for it twice — and the ledger cannot detect that, because both charges are
// real and correctly recorded. The budget simply drains faster than the work
// justifies, which reads as an expensive model.
//
// No test covers the difference. One cannot, against a store that serializes.
//
// Priority first, then insertion order, so a verifier follow-up (§11.1) jumps
// the queue without starving the original leads.
func (t *queries) LeaseNextLead(ctx context.Context, sessionID, owner string, expires time.Time) (*core.Lead, error) {
	now := toMicros(time.Now())
	row := t.q.QueryRowContext(ctx, `
		UPDATE leads
		   SET status = ?, lease_owner = ?, lease_expires = ?, updated_at = ?
		 WHERE id = (
		       SELECT id FROM leads
		        WHERE session_id = ? AND status = ?
		        ORDER BY priority DESC, created_at ASC
		        LIMIT 1
		 )
		RETURNING `+leadCols,
		string(core.LeadLeased), owner, toMicros(expires), now,
		sessionID, string(core.LeadQueued))

	l, err := scanLead(row)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil // queue drained; not an error
	}
	if err != nil {
		return nil, fmt.Errorf("sqlite: lease next lead: %w", err)
	}
	return l, nil
}

// RenewLease extends a lease the worker still holds.
func (t *queries) RenewLease(ctx context.Context, leadID, owner string, expires time.Time) (bool, error) {
	res, err := t.q.ExecContext(ctx, `
		UPDATE leads SET lease_expires = ?, updated_at = ?
		 WHERE id = ? AND lease_owner = ? AND status = ?`,
		toMicros(expires), toMicros(time.Now()), leadID, owner, string(core.LeadLeased))
	if err != nil {
		return false, fmt.Errorf("sqlite: renew lease: %w", err)
	}
	n, err := res.RowsAffected()
	// false rather than an error: losing a lease to the sweep is how a worker
	// learns it stalled long enough to be presumed dead, which is a normal
	// event it has to handle, not a failure of this call.
	return n == 1, err
}

// ReleaseLease returns a lead to the queue without completing it.
//
// The status guard matters as much as the owner one: without it a Release
// arriving after a Complete would flip a finished lead back to queued, to be
// re-run and re-charged. SetLeadStatus clears lease_owner on completion, which
// closes the same hole from the other side — both, because this is the failure
// the package exists to prevent.
func (t *queries) ReleaseLease(ctx context.Context, leadID, owner string) error {
	_, err := t.q.ExecContext(ctx, `
		UPDATE leads SET status = ?, lease_owner = NULL, lease_expires = NULL, updated_at = ?
		 WHERE id = ? AND lease_owner = ? AND status = 'leased'`,
		string(core.LeadQueued), toMicros(time.Now()), leadID, owner)
	if err != nil {
		return fmt.Errorf("sqlite: release lease: %w", err)
	}
	return nil
}

// SweepAbandonedSessions marks running sessions whose process is gone (§9.4).
//
// Conservative on purpose: a session with ANY unexpired lease is live by
// definition, and idleSince has to be well past the lease TTL or a slow-but-
// working run gets marked dead underneath itself.
func (t *queries) SweepAbandonedSessions(ctx context.Context, idleSince time.Time) (int, error) {
	// Toolkit sessions are excluded, and the reason is what the sweep is for: it
	// marks sessions whose PROCESS is gone. A toolkit session has no mole process
	// — an external agent drives it — so thirty idle minutes means the agent is
	// thinking, or the person went to lunch, not that anything died. Marking it
	// failed broke a live session: every later reservation was refused with
	// "session is done" while claim_add kept succeeding.
	res, err := t.q.ExecContext(ctx, `
		UPDATE sessions SET status = 'failed', updated_at = ?
		 WHERE status = 'running'
		   AND updated_at < ?
		   AND actor_types NOT LIKE '%toolkit%'
		   AND id NOT IN (
		       SELECT session_id FROM leads
		        WHERE status = 'leased' AND lease_expires IS NOT NULL AND lease_expires > ?
		   )`,
		toMicros(time.Now()), toMicros(idleSince), toMicros(time.Now()))
	if err != nil {
		return 0, fmt.Errorf("sqlite: sweep abandoned sessions: %w", err)
	}
	n, err := res.RowsAffected()
	return int(n), err
}

// SweepExpiredLeases requeues leads whose worker died (§9.4).
func (t *queries) SweepExpiredLeases(ctx context.Context, sessionID string, now time.Time) (int, error) {
	// Terminal sessions are excluded: requeueing their leads produces rows no
	// executor will ever lease (LeaseNextLead filters by session), so the CLI
	// reports "recovered N leads" for work that is permanently stuck.
	query := `
		UPDATE leads SET status = ?, lease_owner = NULL, lease_expires = NULL, updated_at = ?
		 WHERE status = ? AND lease_expires IS NOT NULL AND lease_expires < ?
		   AND session_id IN (SELECT id FROM sessions WHERE status = 'running')`
	args := []any{string(core.LeadQueued), toMicros(now), string(core.LeadLeased), toMicros(now)}
	if sessionID != "" {
		query += ` AND session_id = ?`
		args = append(args, sessionID)
	}

	res, err := t.q.ExecContext(ctx, query, args...)
	if err != nil {
		return 0, fmt.Errorf("sqlite: sweep expired leases: %w", err)
	}
	n, err := res.RowsAffected()
	return int(n), err
}

// CountLeadsByStatus is how the loop knows whether work remains.
func (t *queries) CountLeadsByStatus(ctx context.Context, sessionID string) (map[core.LeadStatus]int, error) {
	rows, err := t.q.QueryContext(ctx,
		`SELECT status, COUNT(*) FROM leads WHERE session_id = ? GROUP BY status`, sessionID)
	if err != nil {
		return nil, fmt.Errorf("sqlite: count leads: %w", err)
	}
	defer rows.Close()

	out := map[core.LeadStatus]int{}
	for rows.Next() {
		var status string
		var n int
		if err := rows.Scan(&status, &n); err != nil {
			return nil, err
		}
		out[core.LeadStatus(status)] = n
	}
	return out, rows.Err()
}

// ---------------------------------------------------------------------------
// Claims
// ---------------------------------------------------------------------------

const claimCols = `id, session_id, lead_id, text, source, tool_call_id, quote,
	quote_offset, published_at, retrieved_at, root_claim_id, verify_depth,
	assertion_strength, confidence, grounded, grounding_note, verified_at, created_at, seq`

// InsertClaims writes a batch.
//
// All or nothing: a partial batch would leave the claim graph citing a lead
// that went on to report failure, and §11's clustering would then be reasoning
// over evidence that was never fully recorded.
func (t *queries) InsertClaims(ctx context.Context, claims []core.Claim) error {
	now := time.Now().UTC()
	for i := range claims {
		c := &claims[i]
		if c.ID == "" {
			c.ID = core.NewClaimID()
		}
		// A claim with no verification ancestry is its own root. Setting this
		// here rather than at the call site is what keeps the lineage cap in
		// §11.4 from silently starting over on every new claim.
		if c.RootClaimID == "" {
			c.RootClaimID = c.ID
		}
		if c.CreatedAt.IsZero() {
			c.CreatedAt = now
		}
		if c.RetrievedAt.IsZero() {
			c.RetrievedAt = now
		}
		// Position within THIS batch, assigned unconditionally rather than only
		// when unset. A caller-supplied seq would have to be unique per batch to
		// mean anything, and nothing upstream is in a position to guarantee that
		// — the actor returns claims in the order it mined them, which is exactly
		// what this is recording.
		c.Seq = i

		var grounded sql.NullInt64
		if c.Grounded != nil {
			grounded = sql.NullInt64{Int64: b2i(*c.Grounded), Valid: true}
		}

		_, err := t.q.ExecContext(ctx, `
			INSERT INTO claims (`+claimCols+`) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
			c.ID, c.SessionID, c.LeadID, c.Text, c.Source, c.ToolCallID, c.Quote,
			c.QuoteOffset, nullMicros(c.PublishedAt), toMicros(c.RetrievedAt),
			c.RootClaimID, c.VerifyDepth, c.AssertionStrength, c.Confidence, grounded,
			c.GroundingNote, nullMicros(c.VerifiedAt), toMicros(c.CreatedAt), c.Seq)
		if err != nil {
			return fmt.Errorf("sqlite: insert claim %d/%d: %w", i+1, len(claims), err)
		}
	}
	return nil
}

func (t *queries) ListClaims(ctx context.Context, sessionID string, limit int) ([]*core.Claim, error) {
	if limit <= 0 {
		limit = 500
	}
	rows, err := t.q.QueryContext(ctx,
		`SELECT `+claimCols+` FROM claims WHERE session_id = ? ORDER BY created_at, seq, id LIMIT ?`,
		sessionID, limit)
	if err != nil {
		return nil, fmt.Errorf("sqlite: list claims: %w", err)
	}
	return scanClaims(rows)
}

// ListUnverifiedClaims returns what the Verifier has not scored yet.
//
// `verified_at IS NULL`, not `confidence = 0`: §11.3's formula returns 0 for an
// uncorroborated claim carrying a contradiction, so the zero would make a
// correctly-scored claim look unexamined and get rescored on every pass.
func (t *queries) ListUnverifiedClaims(ctx context.Context, sessionID string, limit int) ([]*core.Claim, error) {
	if limit <= 0 {
		limit = 500
	}
	rows, err := t.q.QueryContext(ctx,
		`SELECT `+claimCols+` FROM claims
		 WHERE session_id = ? AND verified_at IS NULL
		 ORDER BY created_at, seq, id LIMIT ?`,
		sessionID, limit)
	if err != nil {
		return nil, fmt.Errorf("sqlite: list unverified claims: %w", err)
	}
	return scanClaims(rows)
}

// scanClaims is shared by every claim reader.
//
// Extracted rather than duplicated: the column list is sixteen wide and the scan
// order has to match it exactly, so a second hand-written copy is a second place
// for two adjacent float64 fields to silently swap.
func scanClaims(rows *sql.Rows) ([]*core.Claim, error) {
	defer rows.Close()

	var out []*core.Claim
	for rows.Next() {
		var (
			c         core.Claim
			published sql.NullInt64
			retrieved int64
			grounded  sql.NullInt64
			verified  sql.NullInt64
			created   int64
		)
		if err := rows.Scan(&c.ID, &c.SessionID, &c.LeadID, &c.Text, &c.Source,
			&c.ToolCallID, &c.Quote, &c.QuoteOffset, &published, &retrieved,
			&c.RootClaimID, &c.VerifyDepth, &c.AssertionStrength, &c.Confidence,
			&grounded, &c.GroundingNote, &verified, &created, &c.Seq); err != nil {
			return nil, err
		}
		c.PublishedAt = micrasPtr(published)
		c.RetrievedAt = fromMicros(retrieved)
		c.VerifiedAt = micrasPtr(verified)
		c.CreatedAt = fromMicros(created)
		if grounded.Valid {
			v := grounded.Int64 != 0
			c.Grounded = &v
		}
		out = append(out, &c)
	}
	return out, rows.Err()
}

// ---------------------------------------------------------------------------
// Claim graph (§11.2)
// ---------------------------------------------------------------------------

const edgeCols = `id, session_id, from_id, to_id, kind, weight, created_by,
	rationale, created_at`

// InsertEdges upserts a batch of graph edges.
//
// Two things happen here that the schema cannot do on its own.
//
// Symmetric kinds get their endpoints ordered (§core.EdgeKind.Symmetric). The
// UNIQUE constraint is on (from_id, to_id, kind), so "A contradicts B" and "B
// contradicts A" are two rows describing one disagreement — and §11.3 penalizes
// confidence per contradicting edge, so the same disagreement would be counted
// twice against both claims. Ordering the pair lets the constraint see them as
// one fact.
//
// Self-edges are rejected. A claim cannot corroborate or contradict itself, the
// UNIQUE constraint does not stop it, and one would inflate its own corroboration
// count. The edge table also carries no foreign key to claims, so a mistyped ID
// would otherwise persist as a dangling edge with no error.
func (t *queries) InsertEdges(ctx context.Context, edges []core.ClaimEdge) error {
	now := time.Now().UTC()
	for i := range edges {
		e := edges[i]
		if e.FromID == "" || e.ToID == "" {
			return fmt.Errorf("sqlite: edge %d/%d has an empty endpoint", i+1, len(edges))
		}
		if e.FromID == e.ToID {
			return fmt.Errorf("sqlite: edge %d/%d is a self-edge on %s", i+1, len(edges), e.FromID)
		}
		if !e.Kind.Valid() {
			return fmt.Errorf("sqlite: edge %d/%d has unknown kind %q", i+1, len(edges), e.Kind)
		}
		if e.Kind.Symmetric() && e.FromID > e.ToID {
			e.FromID, e.ToID = e.ToID, e.FromID
		}
		if e.ID == "" {
			e.ID = core.NewEdgeID()
		}
		if e.CreatedAt.IsZero() {
			e.CreatedAt = now
		}
		if e.Weight == 0 {
			e.Weight = 1.0
		}

		// On conflict the newer judgement wins, but created_at is left alone:
		// when the edge was first discovered is history, and re-verification is
		// not a new discovery.
		_, err := t.q.ExecContext(ctx, `
			INSERT INTO claim_edges (`+edgeCols+`) VALUES (?,?,?,?,?,?,?,?,?)
			ON CONFLICT (from_id, to_id, kind) DO UPDATE SET
				weight = excluded.weight,
				created_by = excluded.created_by,
				rationale = excluded.rationale`,
			e.ID, e.SessionID, e.FromID, e.ToID, e.Kind, e.Weight, e.CreatedBy,
			e.Rationale, toMicros(e.CreatedAt))
		if err != nil {
			return fmt.Errorf("sqlite: insert edge %d/%d: %w", i+1, len(edges), err)
		}
	}
	return nil
}

func (t *queries) ListEdges(ctx context.Context, sessionID string, limit int) ([]*core.ClaimEdge, error) {
	if limit <= 0 {
		limit = 5000
	}
	rows, err := t.q.QueryContext(ctx,
		`SELECT `+edgeCols+` FROM claim_edges WHERE session_id = ? ORDER BY created_at LIMIT ?`,
		sessionID, limit)
	if err != nil {
		return nil, fmt.Errorf("sqlite: list edges: %w", err)
	}
	defer rows.Close()

	var out []*core.ClaimEdge
	for rows.Next() {
		var (
			e       core.ClaimEdge
			created int64
		)
		if err := rows.Scan(&e.ID, &e.SessionID, &e.FromID, &e.ToID, &e.Kind,
			&e.Weight, &e.CreatedBy, &e.Rationale, &created); err != nil {
			return nil, err
		}
		e.CreatedAt = fromMicros(created)
		out = append(out, &e)
	}
	return out, rows.Err()
}

// SetClaimGrounding records §11.5 grounding verdicts.
//
// Touches grounded and grounding_note only. Confidence is NOT written here: the
// grounding result is an input to §11.3's formula, so it has to land first and the
// derivation has to run afterwards. Writing both at once would score the claim
// against the grounding state it had before the check.
func (t *queries) SetClaimGrounding(ctx context.Context, results []store.ClaimGrounding) error {
	for i, g := range results {
		if g.ClaimID == "" {
			return fmt.Errorf("sqlite: grounding %d/%d has no claim id", i+1, len(results))
		}
		var grounded sql.NullInt64
		if g.Grounded != nil {
			grounded = sql.NullInt64{Int64: b2i(*g.Grounded), Valid: true}
		}
		res, err := t.q.ExecContext(ctx, `
			UPDATE claims SET grounded = ?, grounding_note = ? WHERE id = ?`,
			grounded, g.Note, g.ClaimID)
		if err != nil {
			return fmt.Errorf("sqlite: set grounding %d/%d: %w", i+1, len(results), err)
		}
		if n, err := res.RowsAffected(); err == nil && n == 0 {
			return fmt.Errorf("sqlite: set grounding on %s: %w", g.ClaimID, store.ErrNotFound)
		}
	}
	return nil
}

// ScoreClaims writes the Verifier's verdicts.
//
// Touches confidence, grounded, and verified_at only. The claim's text, quote and
// source are evidence — the quote is what §11.5 checks a citation against — and a
// verification pass able to rewrite them would make its own grounding check
// circular.
func (t *queries) ScoreClaims(ctx context.Context, scores []store.ClaimScore) error {
	now := toMicros(time.Now().UTC())
	for i, s := range scores {
		if s.ClaimID == "" {
			return fmt.Errorf("sqlite: score %d/%d has no claim id", i+1, len(scores))
		}

		var grounded sql.NullInt64
		if s.Grounded != nil {
			grounded = sql.NullInt64{Int64: b2i(*s.Grounded), Valid: true}
		}

		res, err := t.q.ExecContext(ctx, `
			UPDATE claims SET confidence = ?, grounded = ?, verified_at = ?
			WHERE id = ?`, s.Confidence, grounded, now, s.ClaimID)
		if err != nil {
			return fmt.Errorf("sqlite: score claim %d/%d: %w", i+1, len(scores), err)
		}
		// A score for a claim that does not exist means the Verifier is reasoning
		// over something the store never had. Silently affecting zero rows would
		// leave the claim unverified forever and the pass reporting success.
		if n, err := res.RowsAffected(); err == nil && n == 0 {
			return fmt.Errorf("sqlite: score claim %s: %w", s.ClaimID, store.ErrNotFound)
		}
	}
	return nil
}

func (t *queries) CountClaims(ctx context.Context, sessionID string) (int64, error) {
	var n int64
	err := t.q.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM claims WHERE session_id = ?`, sessionID).Scan(&n)
	if err != nil {
		return 0, fmt.Errorf("sqlite: count claims: %w", err)
	}
	return n, nil
}

func b2i(b bool) int64 {
	if b {
		return 1
	}
	return 0
}

// -----------------------------------------------------------------------------
// Dataset rows (M9, §13)
// -----------------------------------------------------------------------------

// optional turns an empty string into a nil pointer, so an absent lead id is
// stored as NULL rather than as "".
func optional(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// InsertRows writes a batch of extracted rows.
//
// All or nothing, matching InsertClaims: a dataset is a thing somebody counts, so
// a half-written batch is worse than none — the merge would report a row count
// that never existed.
func (t *queries) InsertRows(ctx context.Context, sessionID string, rows []dataset.Row) error {
	if len(rows) == 0 {
		return nil
	}
	now := time.Now().UTC()
	// No prepared statement: execer is the interface both a *sql.DB and a *sql.Tx
	// satisfy here, and it deliberately exposes only the three Context methods —
	// adding Prepare to it for one insert would widen the seam every other query
	// in this file works through.
	const insert = `
		INSERT INTO dataset_rows
		  (id, session_id, lead_id, seq, values_json, source, quote, quote_offset, retrieved_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`

	// Continues the session's existing sequence, so two batches from two leads do
	// not both start at zero and lose their order against each other.
	var base int64
	if err := t.q.QueryRowContext(ctx,
		`SELECT COALESCE(MAX(seq) + 1, 0) FROM dataset_rows WHERE session_id = ?`,
		sessionID).Scan(&base); err != nil {
		return err
	}

	for i, r := range rows {
		values, err := json.Marshal(r.Values)
		if err != nil {
			return fmt.Errorf("store: encode row values: %w", err)
		}
		at := r.RetrievedAt
		if at.IsZero() {
			at = now
		}
		if _, err := t.q.ExecContext(ctx, insert, core.NewRowID(), sessionID,
			nullStr(optional(r.LeadID)), base+int64(i), string(values),
			r.Source, r.Quote, r.QuoteOffset, toMicros(at)); err != nil {
			return err
		}
	}
	return nil
}

// ListRows reads a session's rows in insertion order.
func (t *queries) ListRows(ctx context.Context, sessionID string) ([]dataset.Row, error) {
	rows, err := t.q.QueryContext(ctx, `
		SELECT lead_id, values_json, source, quote, quote_offset, retrieved_at
		  FROM dataset_rows WHERE session_id = ? ORDER BY seq`, sessionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []dataset.Row
	for rows.Next() {
		var (
			leadID sql.NullString
			values string
			at     int64
			r      dataset.Row
		)
		if err := rows.Scan(&leadID, &values, &r.Source, &r.Quote,
			&r.QuoteOffset, &at); err != nil {
			return nil, err
		}
		r.RetrievedAt = fromMicros(at)
		if err := json.Unmarshal([]byte(values), &r.Values); err != nil {
			return nil, fmt.Errorf("store: decode row values: %w", err)
		}
		r.LeadID = leadID.String
		out = append(out, r)
	}
	return out, rows.Err()
}

// DeleteSession removes a session and, by cascade, everything filed under it.
//
// The cascade is the point: listing the child tables here would be a list that
// drifts the next time one is added. It depends on foreign_keys(1), which
// sqlite.Open sets and TestDeletingASessionTakesItsDocuments verifies is actually
// in force — an unenforced pragma would turn this into a row deletion that orphans
// everything else.
func (t *queries) DeleteSession(ctx context.Context, id string) error {
	_, err := t.q.ExecContext(ctx, `DELETE FROM sessions WHERE id = ?`, id)
	if err != nil {
		return fmt.Errorf("sqlite: delete session: %w", err)
	}
	return nil
}

// InsertDocument stores source text for later quote verification (toolkit mode).
func (t *queries) InsertDocument(ctx context.Context, doc core.Document) error {
	id := doc.ID
	if id == "" {
		id = core.NewDocumentID()
	}
	fetched := doc.FetchedAt
	if fetched.IsZero() {
		fetched = time.Now().UTC()
	}
	expires := doc.ExpiresAt
	if expires.IsZero() {
		expires = fetched.Add(core.DefaultDocumentTTL)
	}
	_, err := t.q.ExecContext(ctx, `
		INSERT INTO documents (id, session_id, url, title, text, truncated, published_at, fetched_at, expires_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		id, doc.SessionID, doc.URL, doc.Title, doc.Text, doc.Truncated,
		publishedMicros(doc.PublishedAt), toMicros(fetched), toMicros(expires))
	return err
}

// publishedMicros keeps "unstated" distinct from "the epoch": a zero time is
// stored as NULL, because most pages carry no date and pretending they claim 1970
// would feed the staleness rule a comparison it has no business making.
func publishedMicros(t time.Time) any {
	if t.IsZero() {
		return nil
	}
	return toMicros(t)
}

// Document reads stored source text, treating an expired row as absent.
//
// The expiry check is here rather than only in the sweep: a caller must not be
// able to verify a quote against text that was supposed to be gone, whether or not
// a background job has run recently.
func (t *queries) Document(ctx context.Context, id string, now time.Time) (core.Document, bool, error) {
	var (
		d                core.Document
		fetched, expires int64
		published        sql.NullInt64
	)
	err := t.q.QueryRowContext(ctx, `
		SELECT id, session_id, url, title, text, truncated, published_at, fetched_at, expires_at
		  FROM documents WHERE id = ?`, id).
		Scan(&d.ID, &d.SessionID, &d.URL, &d.Title, &d.Text, &d.Truncated, &published,
			&fetched, &expires)
	if errors.Is(err, sql.ErrNoRows) {
		return core.Document{}, false, nil
	}
	if err != nil {
		return core.Document{}, false, fmt.Errorf("sqlite: read document: %w", err)
	}
	d.FetchedAt, d.ExpiresAt = fromMicros(fetched), fromMicros(expires)
	if published.Valid {
		d.PublishedAt = fromMicros(published.Int64)
	}
	if d.Expired(now) {
		return core.Document{}, false, nil
	}
	return d, true, nil
}

// DocumentsForSession lists a session's live documents.
//
// Expiry is applied here for the same reason it is applied in Document: a caller
// must not see text the retention rule says is gone, sweep or no sweep.
func (t *queries) DocumentsForSession(ctx context.Context, sessionID string, now time.Time) ([]core.Document, error) {
	rows, err := t.q.QueryContext(ctx, `
		SELECT id, url, title, text, truncated, published_at, fetched_at, expires_at
		  FROM documents WHERE session_id = ? AND expires_at > ?
		 ORDER BY fetched_at, id`, sessionID, toMicros(now))
	if err != nil {
		return nil, fmt.Errorf("sqlite: list documents: %w", err)
	}
	defer rows.Close()

	var out []core.Document
	for rows.Next() {
		var (
			d                core.Document
			fetched, expires int64
			published        sql.NullInt64
		)
		if err := rows.Scan(&d.ID, &d.URL, &d.Title, &d.Text, &d.Truncated,
			&published, &fetched, &expires); err != nil {
			return nil, err
		}
		d.SessionID = sessionID
		d.FetchedAt, d.ExpiresAt = fromMicros(fetched), fromMicros(expires)
		if published.Valid {
			d.PublishedAt = fromMicros(published.Int64)
		}
		out = append(out, d)
	}
	return out, rows.Err()
}

// ExpireDocumentForTest backdates a document's expiry. See the interface comment.
func (t *queries) ExpireDocumentForTest(ctx context.Context, id string) error {
	_, err := t.q.ExecContext(ctx,
		`UPDATE documents SET expires_at = ? WHERE id = ?`,
		toMicros(time.Now().UTC().Add(-time.Hour)), id)
	return err
}

// PurgeExpiredDocuments reclaims disk taken by source text past its TTL.
func (t *queries) PurgeExpiredDocuments(ctx context.Context, now time.Time) (int64, error) {
	res, err := t.q.ExecContext(ctx,
		`DELETE FROM documents WHERE expires_at <= ?`, toMicros(now))
	if err != nil {
		return 0, fmt.Errorf("sqlite: purge documents: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return 0, err
	}
	return n, nil
}

// InsertCrossings records what left the machine (§12.1).
//
// No value from the data is written: the statement, its hash, the counts and one
// of mole's own reason strings. See migration 0010 for why keeping the statement
// is safe where keeping a value would not be.
func (t *queries) InsertCrossings(ctx context.Context, crossings []core.Crossing) error {
	if len(crossings) == 0 {
		return nil
	}
	const insert = `
		INSERT INTO compute_crossings
		  (id, session_id, lead_id, connector, query, query_hash, outcome, detail,
		   rows_described, columns, columns_withheld, buckets, buckets_suppressed,
		   buckets_beyond_topk, tests, truncated, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	now := time.Now().UTC()
	for _, c := range crossings {
		id := c.ID
		if id == "" {
			id = core.NewCrossingID()
		}
		at := c.CreatedAt
		if at.IsZero() {
			at = now
		}
		if _, err := t.q.ExecContext(ctx, insert, id, c.SessionID,
			nullStr(optional(c.LeadID)), c.Connector, c.Query, c.QueryHash,
			string(c.Outcome), c.Detail, c.RowsDescribed, c.Columns, c.ColumnsWithheld,
			c.Buckets, c.Suppressed, c.BeyondTopK, c.Tests, c.Truncated,
			toMicros(at)); err != nil {
			return err
		}
	}
	return nil
}

// RecentLeadCosts totals each finished lead's calls and attributes them to the
// lead's actor type and depth.
//
// Only leads that reached `done`: a lead that failed halfway spent real money and
// is a sample of a FAILURE, and mixing those into the distribution would teach the
// estimator to reserve for the average of working and broken.
//
// Errored calls are counted, though. The money was spent, the ledger charged it,
// and a reservation that ignored retries would under-reserve exactly the leads
// that need the room.
//
// Ordered oldest first, which is what the caller's rolling window expects: it
// keeps the last N observations, so feeding newest-first would keep the oldest.
func (t *queries) RecentLeadCosts(ctx context.Context, limit int) ([]core.LeadCost, error) {
	if limit <= 0 {
		return nil, nil
	}
	rows, err := t.q.QueryContext(ctx, `
		SELECT actor_type, depth, usd, in_tok, out_tok, cr_tok, cw_tok FROM (
			SELECT l.actor_type AS actor_type, l.depth AS depth,
			       COALESCE(SUM(tc.usd_micros),0)         AS usd,
			       COALESCE(SUM(tc.input_tokens),0)       AS in_tok,
			       COALESCE(SUM(tc.output_tokens),0)      AS out_tok,
			       COALESCE(SUM(tc.cache_read_tokens),0)  AS cr_tok,
			       COALESCE(SUM(tc.cache_write_tokens),0) AS cw_tok,
			       MAX(tc.created_at)                     AS last_at
			  FROM tool_calls tc
			  JOIN leads l ON l.id = tc.lead_id
			 WHERE l.status = 'done'
			 GROUP BY tc.lead_id
			 ORDER BY last_at DESC
			 LIMIT ?
		) ORDER BY last_at ASC`, limit)
	if err != nil {
		return nil, fmt.Errorf("sqlite: recent lead costs: %w", err)
	}
	defer rows.Close()

	var out []core.LeadCost
	for rows.Next() {
		var (
			actor string
			lc    core.LeadCost
		)
		if err := rows.Scan(&actor, &lc.Depth, &lc.Cost.USDMicros, &lc.Cost.InputTokens,
			&lc.Cost.OutputTokens, &lc.Cost.CacheReadTokens,
			&lc.Cost.CacheWriteTokens); err != nil {
			return nil, err
		}
		lc.ActorType = core.ActorType(actor)
		out = append(out, lc)
	}
	return out, rows.Err()
}

// ListCrossings reads a session's audit trail, oldest first.
func (t *queries) ListCrossings(ctx context.Context, sessionID string) ([]core.Crossing, error) {
	rows, err := t.q.QueryContext(ctx, `
		SELECT id, lead_id, connector, query, query_hash, outcome, detail,
		       rows_described, columns, columns_withheld, buckets, buckets_suppressed,
		       buckets_beyond_topk, tests, truncated, created_at
		  FROM compute_crossings WHERE session_id = ? ORDER BY created_at, id`, sessionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []core.Crossing
	for rows.Next() {
		var (
			leadID  sql.NullString
			outcome string
			at      int64
			c       core.Crossing
		)
		if err := rows.Scan(&c.ID, &leadID, &c.Connector, &c.Query, &c.QueryHash,
			&outcome, &c.Detail, &c.RowsDescribed, &c.Columns, &c.ColumnsWithheld,
			&c.Buckets, &c.Suppressed, &c.BeyondTopK, &c.Tests, &c.Truncated,
			&at); err != nil {
			return nil, err
		}
		c.SessionID = sessionID
		c.LeadID = leadID.String
		c.Outcome = core.CrossingOutcome(outcome)
		c.CreatedAt = fromMicros(at)
		out = append(out, c)
	}
	return out, rows.Err()
}

// SetDatasetSchema records the schema the rows were extracted against.
func (t *queries) SetDatasetSchema(ctx context.Context, sessionID string, schema dataset.Schema) error {
	raw, err := json.Marshal(schema)
	if err != nil {
		return fmt.Errorf("store: encode schema: %w", err)
	}
	_, err = t.q.ExecContext(ctx,
		`UPDATE sessions SET dataset_schema = ? WHERE id = ?`, string(raw), sessionID)
	return err
}

// DatasetSchema reads it back, reporting false for a session that is not a
// dataset session.
func (t *queries) DatasetSchema(ctx context.Context, sessionID string) (dataset.Schema, bool, error) {
	var raw sql.NullString
	err := t.q.QueryRowContext(ctx,
		`SELECT dataset_schema FROM sessions WHERE id = ?`, sessionID).Scan(&raw)
	if err != nil {
		return dataset.Schema{}, false, err
	}
	if !raw.Valid || strings.TrimSpace(raw.String) == "" {
		return dataset.Schema{}, false, nil
	}
	var s dataset.Schema
	if err := json.Unmarshal([]byte(raw.String), &s); err != nil {
		return dataset.Schema{}, false, fmt.Errorf("store: decode schema: %w", err)
	}
	return s, true, nil
}

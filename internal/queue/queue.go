// Package queue is the lead queue: dispatch, leases, and recovery.
//
// Leads are LEASED rather than marked running (§9.4). The difference only shows
// up when something goes wrong, which is when it matters: a worker that dies
// mid-lead leaves a `running` row nothing will ever revisit, and rev 1 had no
// way out of that. A lease expires, and an expired lease is a lead the next
// sweep puts back.
//
// The single property everything else rests on: a lead is dispatched to at most
// one worker. Two workers running the same lead pay for it twice, and the
// ledger cannot detect that — both charges are real, both are correctly
// recorded, and the budget simply drains faster than the work justifies.
package queue

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/store"
)

// DefaultLeaseTTL is how long a worker may hold a lead without a heartbeat.
//
// Long enough that a slow fetch plus a slow model call does not lose the lease,
// short enough that a crash does not strand work for the length of a session.
const DefaultLeaseTTL = 5 * time.Minute

// Queue dispatches leads for one session.
type Queue struct {
	st  store.Store
	ttl time.Duration
	now func() time.Time
}

// New creates a queue. A zero ttl uses DefaultLeaseTTL.
func New(st store.Store, ttl time.Duration) *Queue {
	if ttl <= 0 {
		ttl = DefaultLeaseTTL
	}
	return &Queue{st: st, ttl: ttl, now: time.Now}
}

// TTL is the lease duration in force, so a caller can size a heartbeat from it.
func (q *Queue) TTL() time.Duration { return q.ttl }

// SetClock overrides time, for tests.
func (q *Queue) SetClock(now func() time.Time) { q.now = now }

// Lease is a worker's claim on one lead.
type Lease struct {
	Lead  *core.Lead
	Owner string
}

// Push adds leads to the queue.
//
// Depth is carried from the caller. Note that the depth CAP is enforced by the
// executor's own round counter, not by walking the tree, so Lead.ParentID is
// currently informational and is not set — the tree is for the trace view, and
// M4's lineage guard (§11.4) is what will need it populated.
func (q *Queue) Push(ctx context.Context, leads []core.Lead) error {
	if len(leads) == 0 {
		return nil
	}
	return q.st.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		for i := range leads {
			l := &leads[i]
			if l.ID == "" {
				l.ID = core.NewLeadID()
			}
			if l.Status == "" {
				l.Status = core.LeadQueued
			}
			if err := tx.InsertLead(ctx, l); err != nil {
				return err
			}
		}
		return nil
	})
}

// LeaseNext claims the highest-priority queued lead, or returns nil when the
// queue is empty.
func (q *Queue) LeaseNext(ctx context.Context, sessionID, owner string) (*Lease, error) {
	var lease *Lease
	expires := q.now().Add(q.ttl)

	err := q.st.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		lead, err := tx.LeaseNextLead(ctx, sessionID, owner, expires)
		if err != nil || lead == nil {
			return err
		}
		lease = &Lease{Lead: lead, Owner: owner}
		return nil
	})
	return lease, err
}

// Renew extends a lease.
//
// Reports false when the lease is gone — swept as expired, and possibly already
// re-leased to another worker. A worker that keeps going after that is racing
// the one that now owns the lead, so the caller must stop rather than finish.
func (q *Queue) Renew(ctx context.Context, l *Lease) (bool, error) {
	expires := q.now().Add(q.ttl)
	var ok bool
	err := q.st.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		var err error
		ok, err = tx.RenewLease(ctx, l.Lead.ID, l.Owner, expires)
		return err
	})
	return ok, err
}

// Complete marks a lead finished.
func (q *Queue) Complete(ctx context.Context, l *Lease, status core.LeadStatus) error {
	switch status {
	case core.LeadDone, core.LeadFailed, core.LeadSkippedCache:
	default:
		return fmt.Errorf("queue: %q is not a terminal lead status", status)
	}
	err := q.st.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.SetLeadStatus(ctx, l.Lead.ID, l.Owner, status)
	})
	if errors.Is(err, store.ErrNotFound) {
		// The lease is gone — swept, and possibly re-leased. Distinguishable so
		// the caller can stop rather than treat a lost lead as completed.
		return fmt.Errorf("%w: lease for lead %s is no longer held by %s",
			ErrLeaseLost, l.Lead.ID, l.Owner)
	}
	return err
}

// ErrLeaseLost means the lease was taken before the operation could complete.
var ErrLeaseLost = errors.New("queue: lease lost")

// Release returns a lead to the queue without completing it.
func (q *Queue) Release(ctx context.Context, l *Lease) error {
	return q.st.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		return tx.ReleaseLease(ctx, l.Lead.ID, l.Owner)
	})
}

// Sweep requeues leads whose lease has expired.
//
// sessionID scopes it. A running executor must pass its own: an unscoped sweep
// requeues another live process's in-flight leases, and since nothing stops that
// process finishing its lead, both end up running it and both settle a charge.
// Confirmed reachable with a single worker — the database has no exclusive lock.
//
// Empty sessionID sweeps every running session, which is what boot recovery
// needs (§9.4): after a crash every lead the dead process held is leased with an
// expiry in the past, and without this they stay that way.
func (q *Queue) Sweep(ctx context.Context, sessionID string) (int, error) {
	var n int
	err := q.st.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		var err error
		n, err = tx.SweepExpiredLeases(ctx, sessionID, q.now())
		return err
	})
	return n, err
}

// Stats reports the lead counts by status.
type Stats struct {
	Queued, Leased, Done, Failed, Cached int
	// Deferred is work the session owes that cannot be leased yet: follow-ups a
	// verification pass judged worth running and its budget could not take.
	//
	// Apart from Queued deliberately. The loop reads Queued as "work I can take
	// now", and counting deferred there makes it spin on leads the queue will
	// never hand out.
	Deferred int
}

// Pending reports work that is not finished — queued plus in flight.
//
// The loop does NOT use this to decide when it is done: it stops when LeaseNext
// returns nil, because a lead another worker holds is not work this one can take.
// Pending is for reporting.
func (s Stats) Pending() int { return s.Queued + s.Leased }

// PromoteDeferred makes owed follow-up work eligible to lease.
//
// Deferred leads are contradictions a verification pass judged worth settling
// and its own budget could not take. They are not a wish list: §11.4's rules
// have already been applied, and the only thing between them and the queue was
// MaxTotal.
//
// limit <= 0 promotes all of them. What bounds the session is §8.5's ceilings,
// checked on every turn of the loop; this only decides when work the session
// already owes becomes takeable.
func (q *Queue) PromoteDeferred(ctx context.Context, sessionID string, limit int) (int, error) {
	var n int
	err := q.st.WithTx(ctx, func(ctx context.Context, tx store.Tx) error {
		var err error
		n, err = tx.PromoteDeferredLeads(ctx, sessionID, limit)
		return err
	})
	return n, err
}

// Stats counts the session's leads by status.
func (q *Queue) Stats(ctx context.Context, sessionID string) (Stats, error) {
	var out Stats
	err := q.st.Read(ctx, func(ctx context.Context, qs store.Queries) error {
		counts, err := qs.CountLeadsByStatus(ctx, sessionID)
		if err != nil {
			return err
		}
		out = Stats{
			Queued:   counts[core.LeadQueued],
			Leased:   counts[core.LeadLeased],
			Done:     counts[core.LeadDone],
			Failed:   counts[core.LeadFailed],
			Cached:   counts[core.LeadSkippedCache],
			Deferred: counts[core.LeadDeferred],
		}
		return nil
	})
	return out, err
}

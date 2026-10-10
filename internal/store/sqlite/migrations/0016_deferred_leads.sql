-- Work the session owes and has not queued yet.
--
-- A verification pass proposes a follow-up for every contradiction worth
-- settling and then takes the first few, because a graph full of disagreement
-- could otherwise queue more research than the budget can pay for. The ones
-- that did not fit were discarded: FollowUps returned the winners and nothing
-- about the losers, and no later pass could find them again, since
-- Verifier.Run judges only unverified claims and skips any pair that already
-- carries an edge.
--
-- Measured on one Deep run: 32 contradictions found, 4 followed up, 28 gone.
-- The run then ended with the planner reporting nothing left to add, which was
-- true from where the planner stood and wrong about the session.
--
-- A deferred lead is an ordinary queued lead that is not yet eligible to be
-- leased. Not a separate table: it is the same work, in the same queue, with
-- the same ceilings — and the thing that was missing was never a place to put
-- it, only the admission that it was owed.
--
-- Default 0, so every existing lead stays immediately eligible.
ALTER TABLE leads ADD COLUMN deferred INTEGER NOT NULL DEFAULT 0;

-- The dispatch index covers the lease query's WHERE and ORDER BY. Deferred
-- joins it rather than filtering after the fact, so a session holding hundreds
-- of deferred follow-ups does not pay for them on every lease.
DROP INDEX IF EXISTS idx_leads_dispatch;
CREATE INDEX idx_leads_dispatch ON leads(session_id, status, deferred, priority DESC, created_at);

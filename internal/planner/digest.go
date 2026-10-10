// Package planner decomposes a question into leads and replans as evidence
// arrives.
//
// The digest is what the planner sees between rounds: a rolling summary of which
// sub-questions have evidence and which do not. It is bounded on purpose — the
// planner reads it every replan, so an unbounded digest makes planning cost grow
// with the length of the run.
package planner

import (
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/lajosdeme/mole/internal/core"
)

// DefaultDigestChars bounds the serialized digest.
//
// Characters rather than tokens for the same reason as chunking (§4.1): the
// tokenizer differs per provider, and an estimate that runs long produces a
// rejected request rather than a slightly larger bill. Roughly 1k tokens.
const DefaultDigestChars = 4000

// SubQuestion is one thread of the research.
type SubQuestion struct {
	// ID is short and stable so the planner can refer to a question across
	// replans without the full text being echoed back each time.
	ID   string
	Text string

	// Leads counts dispatches against this question; Claims counts what came
	// back. The pair is the coverage signal: leads without claims is a
	// question being asked badly, not a question with no answer.
	Leads  int
	Claims int

	// Answered is set by the planner, not inferred from a claim count. A
	// question can accumulate claims that do not answer it.
	Answered bool
}

// DeadEnd records a lead that produced nothing, collapsed by cause.
//
// Collapsed rather than listed: twenty bot_block failures are one fact for the
// planner ("this route is blocked"), and listing each would spend the digest's
// whole budget on the least informative part of it.
type DeadEnd struct {
	Cause string
	Count int
	// Example is one query that hit this cause, so the planner can tell a
	// blocked domain from a badly phrased search.
	Example string
}

// Digest is the planner's entire view of a session.
//
// Its METHODS are safe for concurrent use. That matters from M5: a worker pool
// records leads, claims and dead ends as they complete, and without this the
// counters lose updates — measured at 3 lost out of 400 across 8 goroutines, with
// the race detector reporting 36 races on the same run.
//
// Its exported FIELDS are not protected and must not be touched by a worker.
// BudgetRemaining and Contradictions are written between batches, by the
// coordinator, at the replan and verify points — which is where they belong
// anyway: both are current state read fresh for a planner call, not something a
// lead produces.
type Digest struct {
	// mu guards every field below. Held across compact(), which is why the
	// unlocked renderLocked/openLocked variants exist: compact serializes the
	// digest to decide whether it is over budget, and String taking the lock
	// again would deadlock on the first oversized digest.
	mu sync.Mutex

	// Question is the original prompt. Never dropped by compaction: without it
	// the planner does not know what it is researching.
	Question string

	Questions []SubQuestion
	DeadEnds  []DeadEnd

	// Contradictions is how many live disagreements the Verifier has found (§11).
	//
	// A count, so §9.1's rule holds: no page-derived text reaches the planner. It
	// still changes the decision, and it is the one signal that separates "this
	// sub-question has evidence" from "this sub-question has an argument" — a
	// planner marking a disputed question answered is the failure this prevents.
	Contradictions int

	// LeadsRun and ClaimsFound are session totals, kept even when the
	// per-question detail is compacted away.
	LeadsRun    int
	ClaimsFound int

	// BudgetRemaining is the fraction of the session's allowance still
	// available, from core.Session.RemainingFraction.
	//
	// Set fresh before each replan rather than accumulated: it is current state,
	// not history, and a stale figure is worse than none. Zero means "not
	// reported" and is omitted from the serialized form — unambiguous in
	// practice because a session with genuinely nothing left has already
	// stopped and will not replan.
	BudgetRemaining float64

	// MaxChars bounds the serialized form. Zero uses DefaultDigestChars.
	MaxChars int

	// nextID is the monotonic source of sub-question IDs. Monotonic across the
	// whole session, so an ID a replan refers to always means what it meant
	// when it was assigned.
	nextID int

	// openElided counts OPEN questions compaction had to drop. Surfaced in the
	// serialized form: the planner must know its view is partial, or it will
	// read a truncated list as the complete set of remaining work.
	openElided int
}

// NewDigest starts a digest for a session.
func NewDigest(question string, maxChars int) *Digest {
	if maxChars <= 0 {
		maxChars = DefaultDigestChars
	}
	return &Digest{Question: question, MaxChars: maxChars}
}

// AddQuestions registers sub-questions the planner proposed, and returns them
// with the IDs the digest assigned.
//
// IDs are assigned here, never taken from the model. Model-authored IDs are
// numbered per batch, so every replan that omits them restarts at q1 and
// collides with the initial decomposition — and the old dedupe-by-ID then
// DROPPED the new question while the executor still queued its lead and
// credited its claims to the question it collided with. Confirmed: two rounds
// both using q1 left one question holding both rounds' leads, and after
// MarkAnswered the digest read "No open sub-questions" while a new question was
// running. That is §9.3's livelock arriving from the other direction.
//
// Returning the assigned IDs is what lets the caller map leads correctly;
// positional correspondence with the input is not enough once duplicates are
// dropped.
//
// Dedupe is by normalized TEXT, because that is what actually identifies a
// research thread. A replan re-proposing the same wording should not open a
// second one.
func (d *Digest) AddQuestions(qs []SubQuestion) []SubQuestion {
	d.mu.Lock()
	defer d.mu.Unlock()

	out := make([]SubQuestion, 0, len(qs))
	for _, q := range qs {
		text := oneLine(q.Text)
		if text == "" {
			continue
		}
		if existing := d.findByText(text); existing != nil {
			out = append(out, *existing)
			continue
		}
		d.nextID++
		added := SubQuestion{ID: fmt.Sprintf("q%d", d.nextID), Text: text}
		d.Questions = append(d.Questions, added)
		out = append(out, added)
	}
	d.compact()
	return out
}

func (d *Digest) findByText(text string) *SubQuestion {
	for i := range d.Questions {
		if d.Questions[i].Text == text {
			return &d.Questions[i]
		}
	}
	return nil
}

// RecordLead notes a dispatch against a sub-question.
func (d *Digest) RecordLead(questionID string) {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.LeadsRun++
	if q := d.find(questionID); q != nil {
		q.Leads++
	}
	// Compact here too. RecordLead grows a serialized per-question line, so
	// skipping it let the digest sit over MaxChars until some other method
	// happened to be called.
	d.compact()
}

// RecordClaims notes evidence found for a sub-question.
func (d *Digest) RecordClaims(questionID string, n int) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if n <= 0 {
		return
	}
	d.ClaimsFound += n
	if q := d.find(questionID); q != nil {
		q.Claims += n
	}
	d.compact()
}

// RecordDeadEnd notes a lead that produced no evidence (§9.5 degraded).
func (d *Digest) RecordDeadEnd(cause, exampleQuery string) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if cause == "" {
		cause = "no_evidence"
	}
	for i := range d.DeadEnds {
		if d.DeadEnds[i].Cause == cause {
			d.DeadEnds[i].Count++
			d.compact()
			return
		}
	}
	d.DeadEnds = append(d.DeadEnds, DeadEnd{Cause: cause, Count: 1, Example: oneLine(exampleQuery)})
	d.compact()
}

// MarkAnswered closes a sub-question.
func (d *Digest) MarkAnswered(questionID string) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if q := d.find(questionID); q != nil {
		q.Answered = true
	}
	d.compact()
}

// Open returns the sub-questions still without an answer.
func (d *Digest) Open() []SubQuestion {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.openLocked()
}

func (d *Digest) openLocked() []SubQuestion {
	var out []SubQuestion
	for _, q := range d.Questions {
		if !q.Answered {
			out = append(out, q)
		}
	}
	return out
}

// Complete reports whether every sub-question has been answered. A digest with
// no questions at all is not complete — nothing has been planned yet.
func (d *Digest) Complete() bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.Questions) > 0 && len(d.openLocked()) == 0
}

func (d *Digest) find(id string) *SubQuestion {
	for i := range d.Questions {
		if d.Questions[i].ID == id {
			return &d.Questions[i]
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Compaction
// ---------------------------------------------------------------------------

// compact shrinks the digest until it serializes within MaxChars, dropping
// what the planner stops needing first:
//
//  1. Dead ends beyond the worst few — already collapsed by cause, so the rest
//     is a long tail of one-offs.
//  2. Sub-question text, truncated. Losing a question's tail beats losing the
//     question.
//  3. Open questions, most-attempted first, with a count of what was elided.
//
// Answered questions are never dropped: String() serializes only open ones, so
// removing them cannot reduce the size.
//
// Step 3 loses real information — a planner that cannot see an open question
// may re-plan it — and is last for that reason. Without it a session where
// nothing is answered grows the digest without limit: 43k characters over 500
// open questions, ten times the budget, which is the quadratic planner cost
// §9.1 forbids. Most-attempted-first goes because a question with leads and no
// claims is closest to a dead end, and one never attempted is the actionable
// one.
func (d *Digest) compact() {
	limit := d.MaxChars
	if limit <= 0 {
		limit = DefaultDigestChars
	}
	if len(d.renderLocked()) <= limit {
		return
	}

	// 1. Keep only the highest-count dead ends.
	if len(d.DeadEnds) > 4 {
		sort.SliceStable(d.DeadEnds, func(i, j int) bool { return d.DeadEnds[i].Count > d.DeadEnds[j].Count })
		d.DeadEnds = d.DeadEnds[:4]
		if len(d.renderLocked()) <= limit {
			return
		}
	}

	// 2. Truncate question text. Bounded by a floor: a question shortened past
	// recognition is worse than a slightly oversized digest, because the
	// planner would re-plan it as new.
	for width := 160; width >= 40; width -= 40 {
		for i := range d.Questions {
			d.Questions[i].Text = truncate(d.Questions[i].Text, width)
		}
		if len(d.renderLocked()) <= limit {
			return
		}
	}

	// 3. Elide open questions, most-attempted first. Genuinely lossy; see the
	// doc comment for why it still beats an unbounded digest.
	for len(d.renderLocked()) > limit {
		idx, worst := -1, -1
		for i, q := range d.Questions {
			if !q.Answered && q.Leads > worst {
				idx, worst = i, q.Leads
			}
		}
		// Never elide the last open question: a digest with none reads as a
		// finished session and would stop the loop early.
		if idx < 0 || len(d.openLocked()) <= 1 {
			return
		}
		d.openElided++
		d.Questions = append(d.Questions[:idx], d.Questions[idx+1:]...)
	}
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

// String renders the digest for a prompt.
//
// Stable ordering: two identical states must produce identical text, or the
// cassette key changes between runs and every replay misses.
func (d *Digest) String() string {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.renderLocked()
}

func (d *Digest) renderLocked() string {
	var b strings.Builder

	fmt.Fprintf(&b, "Research question: %s\n\n", d.Question)
	fmt.Fprintf(&b, "Progress: %d lead(s) run, %d claim(s) found.\n", d.LeadsRun, d.ClaimsFound)
	if d.Contradictions > 0 {
		fmt.Fprintf(&b, "%d contradiction(s) found between sources — evidence on at "+
			"least one sub-question is disputed, not settled.\n", d.Contradictions)
	}
	if d.BudgetRemaining > 0 {
		fmt.Fprintf(&b, "Roughly %.0f%% of the session's allowance remains.\n", d.BudgetRemaining*100)
	}

	answered := 0
	for _, q := range d.Questions {
		if q.Answered {
			answered++
		}
	}
	if answered > 0 {
		fmt.Fprintf(&b, "%d sub-question(s) answered.\n", answered)
	}

	open := d.openLocked()
	if len(open) == 0 {
		b.WriteString("\nNo open sub-questions.\n")
	} else {
		b.WriteString("\nOpen sub-questions:\n")
		for _, q := range open {
			fmt.Fprintf(&b, "  [%s] %s\n", q.ID, q.Text)
			// Leads-without-claims is the signal that separates "no answer
			// exists" from "we are asking badly", and the planner needs it to
			// decide between rephrasing and giving up.
			fmt.Fprintf(&b, "        %d lead(s) run, %d claim(s) found\n", q.Leads, q.Claims)
		}
		if d.openElided > 0 {
			// Say so. A planner reading a truncated list as the complete set
			// of remaining work would call the session finished.
			fmt.Fprintf(&b, "  (%d further open sub-question(s) elided to fit; "+
				"the list above is partial)\n", d.openElided)
		}
	}

	if len(d.DeadEnds) > 0 {
		b.WriteString("\nDead ends so far:\n")
		for _, de := range d.DeadEnds {
			fmt.Fprintf(&b, "  %s ×%d", de.Cause, de.Count)
			if de.Example != "" {
				fmt.Fprintf(&b, " (e.g. %q)", truncate(de.Example, 60))
			}
			b.WriteString("\n")
		}
	}
	return b.String()
}

// oneLine collapses text to a single line.
//
// Digest.String() is a line-structured format the planner reads back, so any
// multi-line field can forge additional entries — a sub-question whose text
// contains "\n  [q9] fabricated" renders as a second, fully-formed question
// with its own coverage counts. Sub-question wording is model output, so it is
// flattened on the way in.
func oneLine(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

func truncate(s string, max int) string {
	s = oneLine(s)
	if len(s) <= max {
		return s
	}
	cut := s[:max]
	// Back off to a rune boundary so a truncated digest stays valid UTF-8.
	for len(cut) > 0 && cut[len(cut)-1]&0xC0 == 0x80 {
		cut = cut[:len(cut)-1]
	}
	if i := strings.LastIndexByte(cut, ' '); i > max/2 {
		cut = cut[:i]
	}
	return cut + "…"
}

// LeadsFor turns open sub-questions into dispatchable leads.
func LeadsFor(sessionID string, actor core.ActorType, qs []SubQuestion, depth int, parent *string) []core.Lead {
	return LeadsAcross(sessionID, []core.ActorType{actor}, qs, depth, parent)
}

// LeadsAcross distributes sub-questions across the session's actors, round-robin.
//
// Round-robin, and not "every actor researches every question": that would
// double the cost of a two-actor session for evidence largely about the same
// thing. And not "one actor for the whole session", which is what this did — a
// session asking for web AND academic got web for every lead, because the actor
// was chosen once from the session rather than per question. The academic actor
// was built, registered, and never handed a lead, and `--actors web,academic`
// silently researched nothing academically.
//
// Deterministic by position, so a replay assigns the same actor to the same
// question. Not model-chosen: the planner deciding per question is a separate
// change with its own prompt and its own failure mode, and this is the
// mechanical version that makes the flag mean what it says.
func LeadsAcross(sessionID string, actors []core.ActorType, qs []SubQuestion, depth int, parent *string) []core.Lead {
	if len(actors) == 0 {
		actors = []core.ActorType{core.ActorWeb}
	}
	out := make([]core.Lead, 0, len(qs))
	for i, q := range qs {
		out = append(out, core.Lead{
			SessionID: sessionID,
			ActorType: actors[i%len(actors)],
			Query:     q.Text,
			Depth:     depth,
			ParentID:  parent,
			Status:    core.LeadQueued,
		})
	}
	return out
}

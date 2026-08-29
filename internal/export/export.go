// Package export writes one finished session as a single JSON document.
//
// Everything a consumer needs to render or audit a session, in one read, with no
// knowledge of mole's schema: the sub-questions it asked, the report, the
// disagreements it found with both sides inlined, every claim and its quote, what
// it could not read, and what it spent.
//
// It exists because the pieces were reachable only as five commands printing five
// formats — a consumer had to shell out repeatedly and parse text tables. Nothing
// here is new information; it is the information mole already had, in one shape.
//
// No model calls. The export is derived from stored state, so re-exporting a
// finished session is free and deterministic, and a consumer that improves how it
// renders improves every session already run. That is the same property
// dataset.Merge has, for the same reason.
package export

import (
	"context"
	"fmt"
	"sort"
	"time"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/tools/fetch"
)

// SchemaVersion is the envelope's version.
//
// Consumers outside this repository parse this document, so a renamed or removed
// field is an outage rather than a refactor. Additions are a minor matter and do
// not move this number; anything else does. TestTheEnvelopeIsPinned is where the
// contract actually lives.
const SchemaVersion = 1

// Export is one session, whole.
type Export struct {
	SchemaVersion int       `json:"schema_version"`
	MoleVersion   string    `json:"mole_version"`
	ExportedAt    time.Time `json:"exported_at"`

	Session   Session    `json:"session"`
	Scope     []Sub      `json:"scope"`
	Report    Report     `json:"report"`
	Conflicts []Conflict `json:"conflicts"`
	Evidence  []Evidence `json:"evidence"`
	Coverage  Coverage   `json:"coverage"`
	Cost      Cost       `json:"cost"`

	// Receipt is nil when verification did not run, and is deliberately NOT
	// omitempty: a consumer must be able to tell "not checked" from "checked and
	// found nothing", and an absent key reads as the second.
	Receipt *Receipt `json:"receipt"`
}

// Session is what was asked and what it was allowed to spend.
type Session struct {
	ID         string    `json:"id"`
	Question   string    `json:"question"`
	Mode       string    `json:"mode"`
	Status     string    `json:"status"`
	Actors     []string  `json:"actors"`
	BudgetUnit string    `json:"budget_unit"`
	Budget     int64     `json:"budget"`
	Spent      int64     `json:"spent"`
	CreatedAt  time.Time `json:"created_at"`
	UpdatedAt  time.Time `json:"updated_at"`
}

// Sub is one sub-question, as executed.
//
// The leads that ran, not a plan that was approved: mole decomposes and dispatches
// in one step, so this says what was asked rather than what someone agreed to.
type Sub struct {
	ID       string `json:"id"`
	Query    string `json:"query"`
	Actor    string `json:"actor"`
	Depth    int    `json:"depth"`
	ParentID string `json:"parent_id"`
	Status   string `json:"status"`
}

// Report is the synthesized answer, as written.
type Report struct {
	Markdown string `json:"markdown"`
	// Degraded is mole's own account of why the report is thin, when it is.
	Degraded string `json:"degraded"`
}

// Claim is one assertion and the text it was mined from.
type Claim struct {
	ID          string     `json:"id"`
	Text        string     `json:"text"`
	Source      string     `json:"source"`
	Quote       string     `json:"quote"`
	PublishedAt *time.Time `json:"published_at"`
	Confidence  float64    `json:"confidence"`
}

// Evidence is a Claim plus which sub-question produced it.
type Evidence struct {
	Claim
	LeadID string `json:"lead_id"`
}

// Conflict is two claims the graph found in disagreement, with BOTH SIDES INLINED.
//
// Inlined rather than referenced by id: the consumer renders a document and should
// not have to join back into the evidence array, which would put mole's schema
// into a codebase that only wants to lay out two quotes side by side.
type Conflict struct {
	// Kind is "contradicts", or "supersedes" where publication dates resolved the
	// conflict as staleness (§11.2).
	Kind      string `json:"kind"`
	Rationale string `json:"rationale"`
	DecidedBy string `json:"decided_by"`
	A         Claim  `json:"a"`
	B         Claim  `json:"b"`
}

// Coverage is what was read and what was not.
type Coverage struct {
	// Considered is how many distinct sources the session had available to read:
	// what it read, what refused to be read, and what the budget never reached.
	Considered int `json:"considered"`
	// Read is how many distinct sources produced a claim.
	Read int `json:"read"`
	// NotSelected is how many results the source cap or the token budget passed
	// over. Not failures — nothing was wrong with those pages.
	NotSelected int `json:"not_selected"`

	// ConsideredKnown says whether Considered means anything.
	//
	// False for every session recorded before the not_selected outcome existed,
	// where search results were persisted nowhere and the denominator cannot be
	// reconstructed. Reporting considered == read for those would be the exact
	// overstatement a coverage section exists to prevent, and it would be
	// invisible — so the flag is carried rather than inferred from a zero.
	ConsideredKnown bool   `json:"considered_known"`
	Note            string `json:"note"`

	// Unreachable names every source that could not be read, and why. A result the
	// budget never reached is NOT in here: "nobody asked" and "it refused" are
	// different facts, and a pack printing the first as the second would name pages
	// nothing was wrong with.
	Unreachable []Unreachable `json:"unreachable"`
}

// Unreachable is one source mole could not read.
type Unreachable struct {
	URL        string `json:"url"`
	Domain     string `json:"domain"`
	Outcome    string `json:"outcome"`
	StatusCode int    `json:"status_code"`
}

// Cost is what the session spent, in the unit its budget was set in.
type Cost struct {
	USDMicros int64      `json:"usd_micros"`
	Calls     int64      `json:"calls"`
	Tokens    Tokens     `json:"tokens"`
	ByRole    []RoleCost `json:"by_role"`
}

// Tokens counts every token that moved. Cache reads are cheaper, not free.
type Tokens struct {
	Input      int64 `json:"input"`
	Output     int64 `json:"output"`
	CacheRead  int64 `json:"cache_read"`
	CacheWrite int64 `json:"cache_write"`
}

// RoleCost is one role's share.
type RoleCost struct {
	Role      string `json:"role"`
	USDMicros int64  `json:"usd_micros"`
}

// Receipt is the post-hoc check that every quote is still in the source it cites.
type Receipt struct {
	Verified    int `json:"verified"`
	Mismatch    int `json:"mismatch"`
	OffsetDrift int `json:"offset_drift"`
	Unreachable int `json:"unreachable"`
	Skipped     int `json:"skipped"`
}

// Options controls what Build gathers.
type Options struct {
	// MoleVersion is stamped into the envelope. Empty leaves it blank rather than
	// guessing, so a consumer can tell an unversioned export from an old one.
	MoleVersion string
	// Now overrides the export timestamp. Zero takes time.Now; tests set it.
	Now time.Time
	// Receipt, when non-nil, is the verification result to carry. Build does not
	// run it: verification re-fetches every cited source, and a function documented
	// as deriving from stored state must not quietly open sockets.
	Receipt *Receipt
}

// Build assembles the export from stored state.
func Build(ctx context.Context, st store.Store, sessionID string, opts Options) (*Export, error) {
	var (
		sess     *core.Session
		leads    []*core.Lead
		claims   []*core.Claim
		edges    []*core.ClaimEdge
		fetches  []*store.FetchOutcome
		byRole   map[core.Role]core.Cost
		calls    []*core.ToolCall
		callsErr error
	)
	if err := st.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		if sess, err = q.GetSession(ctx, sessionID); err != nil {
			return err
		}
		if leads, err = q.ListLeads(ctx, sessionID, 0); err != nil {
			return err
		}
		if claims, err = q.ListClaims(ctx, sessionID, 0); err != nil {
			return err
		}
		if edges, err = q.ListEdges(ctx, sessionID, 0); err != nil {
			return err
		}
		if fetches, err = q.ListFetchOutcomes(ctx, sessionID, 0); err != nil {
			return err
		}
		if byRole, err = q.SumCostsByRole(ctx, sessionID); err != nil {
			return err
		}
		calls, callsErr = q.ListToolCalls(ctx, sessionID, 0)
		return callsErr
	}); err != nil {
		return nil, fmt.Errorf("export %s: %w", sessionID, err)
	}

	now := opts.Now
	if now.IsZero() {
		now = time.Now().UTC()
	}

	ex := &Export{
		SchemaVersion: SchemaVersion,
		MoleVersion:   opts.MoleVersion,
		ExportedAt:    now.UTC(),
		Session:       sessionOf(sess),
		Scope:         scopeOf(leads),
		Report:        Report{Markdown: sess.Report, Degraded: sess.ReportDegraded},
		Evidence:      evidenceOf(claims),
		Receipt:       opts.Receipt,
	}
	ex.Conflicts = conflictsOf(edges, claims)
	ex.Coverage = coverageOf(claims, fetches)
	ex.Cost = costOf(byRole, calls)
	return ex, nil
}

func sessionOf(s *core.Session) Session {
	actors := make([]string, 0, len(s.ActorTypes))
	for _, a := range s.ActorTypes {
		actors = append(actors, string(a))
	}
	return Session{
		ID: s.ID, Question: s.Prompt, Mode: string(s.Mode), Status: string(s.Status),
		Actors: actors, BudgetUnit: string(s.BudgetUnit),
		Budget: s.Budget, Spent: s.Spent,
		CreatedAt: s.CreatedAt.UTC(), UpdatedAt: s.UpdatedAt.UTC(),
	}
}

func scopeOf(leads []*core.Lead) []Sub {
	out := make([]Sub, 0, len(leads))
	for _, l := range leads {
		if l == nil {
			continue
		}
		var parent string
		if l.ParentID != nil {
			parent = *l.ParentID
		}
		out = append(out, Sub{
			ID: l.ID, Query: l.Query, Actor: string(l.ActorType),
			Depth: l.Depth, ParentID: parent, Status: string(l.Status),
		})
	}
	// Depth then id: the decomposition reads as a tree rather than in whatever
	// order the queue happened to lease.
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Depth != out[j].Depth {
			return out[i].Depth < out[j].Depth
		}
		return out[i].ID < out[j].ID
	})
	return out
}

func claimOf(c *core.Claim) Claim {
	return Claim{
		ID: c.ID, Text: c.Text, Source: c.Source, Quote: c.Quote,
		PublishedAt: c.PublishedAt, Confidence: c.Confidence,
	}
}

func evidenceOf(claims []*core.Claim) []Evidence {
	out := make([]Evidence, 0, len(claims))
	for _, c := range claims {
		if c == nil {
			continue
		}
		out = append(out, Evidence{Claim: claimOf(c), LeadID: c.LeadID})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// conflictsOf carries the edges a reader has to see: a live disagreement, and one
// that publication dates resolved as staleness. Duplicates and supports are how the
// graph reached its confidence, not something a document argues from.
func conflictsOf(edges []*core.ClaimEdge, claims []*core.Claim) []Conflict {
	byID := make(map[string]*core.Claim, len(claims))
	for _, c := range claims {
		if c != nil {
			byID[c.ID] = c
		}
	}
	out := make([]Conflict, 0, len(edges))
	for _, e := range edges {
		if e == nil {
			continue
		}
		if e.Kind != core.EdgeContradicts && e.Kind != core.EdgeSupersedes {
			continue
		}
		a, b := byID[e.FromID], byID[e.ToID]
		if a == nil || b == nil {
			// A conflict missing a side cannot be rendered, and half of one is
			// worse than none.
			continue
		}
		out = append(out, Conflict{
			Kind: string(e.Kind), Rationale: e.Rationale, DecidedBy: e.CreatedBy,
			A: claimOf(a), B: claimOf(b),
		})
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].A.ID != out[j].A.ID {
			return out[i].A.ID < out[j].A.ID
		}
		return out[i].B.ID < out[j].B.ID
	})
	return out
}

// The note travels in the document, so a consumer rendering coverage cannot
// present a partial count as a complete one by accident.
const (
	coverageKnown = "considered counts every distinct source the session had: read, " +
		"refused, and passed over when the source cap or token budget ran out"
	coverageUnknown = "sources considered is not recorded for this session: it predates " +
		"the not_selected outcome, so only what was read and what could not be read are known"
)

func coverageOf(claims []*core.Claim, fetches []*store.FetchOutcome) Coverage {
	read := map[string]bool{}
	for _, c := range claims {
		if c != nil && c.Source != "" {
			read[c.Source] = true
		}
	}

	considered := map[string]bool{}
	for u := range read {
		considered[u] = true
	}

	seenBad := map[string]bool{}
	skipped := map[string]bool{}
	var bad []Unreachable
	for _, f := range fetches {
		if f == nil || f.Outcome == "" || f.URL == "" {
			continue
		}
		considered[f.URL] = true

		switch fetch.Outcome(f.Outcome) {
		case fetch.OutcomeNotSelected:
			skipped[f.URL] = true
			continue
		case fetch.OutcomeOK, fetch.OutcomeProviderContent, fetch.OutcomeStructuredOnly:
			// Read, by one route or another.
			continue
		}
		if seenBad[f.URL] {
			continue
		}
		seenBad[f.URL] = true
		bad = append(bad, Unreachable{
			URL: f.URL, Domain: f.Domain, Outcome: f.Outcome, StatusCode: f.StatusCode,
		})
	}
	sort.SliceStable(bad, func(i, j int) bool { return bad[i].URL < bad[j].URL })
	if bad == nil {
		bad = []Unreachable{}
	}

	// Known only when something actually recorded what went unread. A session with
	// no not_selected rows either predates the outcome or read everything it found,
	// and those are indistinguishable from here — so neither claims a denominator.
	known := len(skipped) > 0
	c := Coverage{
		Read: len(read), NotSelected: len(skipped),
		ConsideredKnown: known, Note: coverageUnknown, Unreachable: bad,
	}
	if known {
		c.Considered, c.Note = len(considered), coverageKnown
	}
	return c
}

func costOf(byRole map[core.Role]core.Cost, calls []*core.ToolCall) Cost {
	var c Cost
	for _, call := range calls {
		if call == nil {
			continue
		}
		c.Calls++
		c.USDMicros += call.Cost.USDMicros
		c.Tokens.Input += call.Cost.InputTokens
		c.Tokens.Output += call.Cost.OutputTokens
		c.Tokens.CacheRead += call.Cost.CacheReadTokens
		c.Tokens.CacheWrite += call.Cost.CacheWriteTokens
	}

	c.ByRole = make([]RoleCost, 0, len(byRole))
	for role, cost := range byRole {
		c.ByRole = append(c.ByRole, RoleCost{Role: string(role), USDMicros: cost.USDMicros})
	}
	// A map iterates in a random order, and two exports of one session must agree.
	sort.SliceStable(c.ByRole, func(i, j int) bool { return c.ByRole[i].Role < c.ByRole[j].Role })
	return c
}

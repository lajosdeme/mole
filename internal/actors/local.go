package actors

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"

	"github.com/lajosdeme/mole/internal/compute"
	"github.com/lajosdeme/mole/internal/compute/coderunner"
	"github.com/lajosdeme/mole/internal/compute/connector"
	"github.com/lajosdeme/mole/internal/compute/gate"
	"github.com/lajosdeme/mole/internal/compute/hypothesis"
	"github.com/lajosdeme/mole/internal/compute/sqlguard"
	"github.com/lajosdeme/mole/internal/compute/stats"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/pricing"
	"github.com/lajosdeme/mole/internal/store"
)

// LocalComputeActor answers a lead from the user's own data (M8, §12).
//
// The shape is the same as the other two actors — take a lead, do the work,
// return summary plus claims plus costs — and the evidence is different in one
// way that decides the whole implementation: it never leaves the machine.
//
// The run is four steps, and the model is only in the first and the last:
//
//	profile  →  the model picks a template and columns   (§12.3: it cannot author SQL)
//	         →  hypothesis.Render turns that into SQL     (identifiers from the profile)
//	         →  sqlguard + the aggregation gate run it    (§12.2, §12.1)
//	         →  the model mines claims from the envelope  (§11.5 quote check, unchanged)
//
// The middle two steps are deterministic. What the model influences is which
// question gets asked; what it never touches is the data, the statement, or
// whether the answer may cross.
type LocalComputeActor struct {
	// Connectors is the registered set. The model chooses among them by name,
	// so a run with none configured is a no-op rather than an error — the same
	// way a web run with no search provider is a configuration problem and not
	// a crash.
	Connectors ConnectorSource

	LLM     llm.Provider
	Pricing *pricing.Table
	Log     *slog.Logger
	Budget  Budget

	// Gate tunes the aggregation gate. The zero value is §12.1's defaults, and
	// raising KFloor is the only knob a privacy-conscious user needs.
	Gate gate.Options

	// Store persists the §12.1 audit trail. Nil disables the durable record and
	// nothing else — the log line still happens, and a run without a store (a
	// test, a dry probe) should not lose its evidence over an audit table.
	//
	// No SessionID field beside it: a crossing is filed under the lead's session,
	// which this actor already has on every call.
	Store store.Store

	// Code runs model-authored analysis in the sandbox (§12.1). Nil disables it,
	// which is the state of any machine without a container runtime — and a
	// supported one: the SQL path needs no sandbox, so a nil Code costs the
	// hypotheses SQL cannot express and nothing else.
	Code coderunner.Runner
}

// ConnectorSource is the registry, narrowed to what the actor needs.
type ConnectorSource interface {
	List() []connector.Connector
}

func (a *LocalComputeActor) Type() core.ActorType { return core.ActorLocalCompute }

func (a *LocalComputeActor) logger() *slog.Logger {
	if a.Log != nil {
		return a.Log
	}
	return slog.Default()
}

// defaultHypotheses caps how many questions one lead may ask when the budget
// names no ceiling of its own.
const defaultHypotheses = 3

func (a *LocalComputeActor) Run(ctx context.Context, lead core.Lead) (*Result, error) {
	budget := a.Budget
	if sub, ok := SubBudgetFrom(ctx); ok {
		budget = budget.tighten(sub)
	}
	maxHypotheses := budget.MaxSources
	if maxHypotheses <= 0 {
		maxHypotheses = defaultHypotheses
	}
	maxClaims := budget.MaxClaimsPerSource
	if maxClaims <= 0 {
		maxClaims = 8
	}
	// The input ceiling, which this actor ignored entirely.
	//
	// WebActor rechecks it per chunk and reports Truncated; this read only
	// MaxSources and MaxClaimsPerSource, so the executor's per-lead reservation
	// was silently unenforced and reserved-versus-actual diverged for every local
	// lead. The planning prompt is the reason it matters: renderSchema emits every
	// column of every registered connector, uncapped.
	maxInput := budget.MaxInputTokens
	if maxInput <= 0 {
		maxInput = 12000
	}
	var inputUsed int64

	res := &Result{}
	sources := a.Connectors.List()
	if len(sources) == 0 {
		res.Summary = "No local data is registered, so this lead was not researched. " +
			"`mole connect add <name> <path>` registers a file or a folder."
		return res, nil
	}

	plans, spent, err := a.plan(ctx, lead, sources, maxHypotheses, maxInput, res)
	inputUsed += spent
	if err != nil {
		return res, err
	}
	if len(plans) == 0 {
		res.Summary = "No hypothesis could be formed over the registered data for this lead."
		return res, nil
	}

	miner := &Miner{LLM: a.LLM, Pricing: a.Pricing, Log: a.Log, SessionID: lead.SessionID}
	var findings []string
	var usedGate, usedSandbox bool
	var crossings []core.Crossing
	var refusal error

	for _, p := range plans {
		if err := ctx.Err(); err != nil {
			res.Truncated = true
			break
		}
		if remaining := maxInput - inputUsed; remaining <= 0 {
			// Out of allowance. Reported, not silently skipped: a run that asked
			// three questions and could afford one is a different result from one
			// that found nothing.
			res.Truncated = true
			break
		}
		text, cite, note, err := a.evidence(ctx, lead, sources, p)
		crossings = append(crossings, note.crossings...)
		if err != nil {
			// A refused or unanswerable hypothesis is a normal outcome, not a
			// failed run: the gate exists to say no. It is recorded so the
			// trace shows what was tried.
			a.logger().InfoContext(ctx, "hypothesis not answered",
				"connector", p.Connector, "table", p.Table, "template", p.Template, "err", err)
			res.Stats.ChunksSkipped++
			continue
		}
		// Chunks only. This run performs no search, so SearchResults is not
		// touched — the sibling AcademicActor does the same.
		res.Stats.Chunks++

		inputUsed += llm.EstimateTokens(len(text))
		out, err := miner.Mine(ctx, MineInput{
			Lead:          lead,
			SourceURL:     cite,
			Title:         evidenceTitle(p),
			Text:          text,
			MaxClaims:     maxClaims,
			MaxQualifiers: budget.withDefaults().MaxQualifiersPerSource,
		})
		if out.HasCall {
			res.Costs = append(res.Costs, out.Call)
		}
		res.Stats.ClaimsProposed += out.Proposed
		res.Stats.ClaimsRejected += out.Rejected
		if llm.Refused(err) {
			refusal = err
			break
		}
		if err != nil {
			res.Stats.ChunksFailed++
			a.logger().WarnContext(ctx, "mining a local result failed",
				"connector", p.Connector, "err", err)
			continue
		}
		res.Claims = append(res.Claims, note.cap(widenToQualifiedLine(text, out.Claims))...)
		findings = append(findings, note.finding)
		if p.Code != nil {
			usedSandbox = true
		} else {
			usedGate = true
		}
	}

	// Claims are PERSISTED here, and were not until a live run made it obvious.
	//
	// WebActor and AcademicActor each write their own claims; this actor returned
	// them in Result and wrote nothing, and the executor only carries Result.Claims
	// in memory for the digest. So every local claim was lost: a live local-only
	// session reported "1 claim(s)" on its way past and then produced a report
	// saying "No verifiable evidence was found", `mole eval` said "no claims", and
	// `mole ask` had nothing to answer from — because the report and the scorecard
	// are both built from the STORE.
	//
	// Uncancellable, for the reason web.go gives: by this point the model calls
	// have happened and the ledger will charge for them whether or not this write
	// lands.
	if len(res.Claims) > 0 && a.Store != nil {
		if err := a.Store.WithTx(context.WithoutCancel(ctx),
			func(ctx context.Context, tx store.Tx) error {
				return tx.InsertClaims(ctx, res.Claims)
			}); err != nil {
			return res, fmt.Errorf("actors/local: persist claims: %w", err)
		}
	}

	// The audit trail, on an uncancellable context and for a reason stronger than
	// the one that applies to claims: the data has ALREADY left the machine by
	// this point, so a cancelled write loses the record of a crossing that
	// happened. §12.1's promise is that the user can audit what left, and a trail
	// that drops its rows when a sibling lead fails is not that promise.
	a.recordCrossings(ctx, crossings)

	res.Summary = localSummary(lead, findings, res, usedGate, usedSandbox)
	if refusal != nil {
		return res, fmt.Errorf("actors/local: mine: %w", refusal)
	}
	return res, nil
}

// recordCrossings writes the audit trail, and never fails the run for it.
//
// A crossing has already happened when this is called. Returning an error would
// discard evidence the user has paid for over a bookkeeping failure — so the
// failure is logged loudly (a missing audit row is a real problem) and the run
// keeps its findings.
func (a *LocalComputeActor) recordCrossings(ctx context.Context, crossings []core.Crossing) {
	if len(crossings) == 0 || a.Store == nil {
		return
	}
	if err := a.Store.WithTx(context.WithoutCancel(ctx),
		func(ctx context.Context, tx store.Tx) error {
			return tx.InsertCrossings(ctx, crossings)
		}); err != nil {
		a.logger().ErrorContext(ctx, "the audit trail was not written; §12.1's record "+
			"of what left this machine is incomplete", "crossings", len(crossings), "err", err)
	}
}

// widenToQualifiedLine extends a quote that landed inside a statistical sentence
// to the whole sentence.
//
// A model asked to quote verbatim may stop at the first clause — "...higher by
// 63.67 (means 141.97 and 78.30; n = 100 and 100); statistically significant
// (p = <0.001)" — dropping the Holm correction, the effect size and the holdout
// stability that follow it. §11.5 passes, because a prefix is a verbatim
// substring, and the qualification never reaches the claim or the report.
//
// The quote is widened rather than the claim refused: the widened text is still
// verbatim, which is the property §11.5 protects, and refusing would discard a
// true claim over its packaging. The claim's own Text is left alone — that is
// the model's assertion, and rewriting it would put words in its mouth.
//
// Only lines carrying a verdict are widened. Widening a bucket line a reader
// checks figure by figure would make a short quote look long for no gain.
func widenToQualifiedLine(passage string, claims []core.Claim) []core.Claim {
	if len(claims) == 0 {
		return claims
	}
	out := make([]core.Claim, 0, len(claims))
	for _, c := range claims {
		if line, offset, ok := qualifiedLineFor(passage, c.Quote); ok {
			c.Quote = TruncateQuote(line)
			c.QuoteOffset = int64(offset)
		}
		out = append(out, c)
	}
	return out
}

// qualifiedLineFor finds the verdict line a quote sits inside, when it sits inside
// one and does not already carry the whole of it.
func qualifiedLineFor(passage, quote string) (string, int, bool) {
	quote = strings.TrimSpace(quote)
	if quote == "" {
		return "", 0, false
	}
	var at int
	for _, raw := range strings.Split(passage, "\n") {
		line := strings.TrimSpace(raw)
		start := at
		at += len(raw) + 1

		if !carriesAVerdict(line) || len(line) <= len(quote) {
			continue
		}
		i := strings.Index(line, quote)
		if i < 0 {
			continue
		}
		// The offset of the trimmed line within the passage.
		return line, start + strings.Index(raw, line), true
	}
	return "", 0, false
}

// carriesAVerdict reports whether a line is one of stats' rendered sentences.
//
// Matched on the verdict wording rather than on a marker character, because the
// wording is what stats.Sentence produces and a marker would be one more thing two
// packages have to agree about.
func carriesAVerdict(line string) bool {
	for _, s := range []string{
		"statistically significant",
		"NOT distinguishable from chance",
		"UNDERPOWERED",
	} {
		if strings.Contains(line, s) {
			return true
		}
	}
	return false
}

// verdictNote is what a run learned about its own evidence, kept separately from
// the passage so the two paths can share the capping rule.
type verdictNote struct {
	finding     string
	unsupported bool
	// crossings are the audit rows for this hypothesis — one for the statement,
	// and one more for §4's holdout statement when it ran. Empty means no gate
	// ran at all.
	//
	// Carried out through the return value rather than accumulated on the actor:
	// one actor instance serves a session's leads CONCURRENTLY, so a field
	// appended to during Run is a data race, and the race would be on the record
	// of what left the user's machine.
	crossings []core.Crossing
}

func (n *verdictNote) add(c core.Crossing) { n.crossings = append(n.crossings, c) }

// cap applies §4's statistical-validity check.
func (n verdictNote) cap(claims []core.Claim) []core.Claim {
	if !n.unsupported {
		return claims
	}
	for i := range claims {
		if claims[i].AssertionStrength > unsupportedAssertionCap {
			claims[i].AssertionStrength = unsupportedAssertionCap
		}
	}
	return claims
}

// evidence turns one plan into the passage claims are mined from.
//
// Two routes, and which one is taken is the plan's choice rather than a
// fallback: a template renders SQL through both gates, and a code plan runs in
// the sandbox. Both end at a passage of figures and a citation, so everything
// downstream — mining, the quote check, the claim graph — cannot tell them
// apart. That is the point: the evidence differs and the standard does not.
func (a *LocalComputeActor) evidence(
	ctx context.Context, lead core.Lead, sources []connector.Connector, p hypothesis.Plan,
) (text, cite string, note verdictNote, err error) {
	c, ok := findConnector(sources, p.Connector)
	if !ok {
		return "", "", note, fmt.Errorf("no connector named %q is registered", p.Connector)
	}
	if p.Code != nil {
		return a.runCode(ctx, lead, c, p)
	}
	return a.runQuery(ctx, lead, c, p)
}

// crossing turns one gate outcome into an audit row.
//
// Every outcome, not only the successful ones. A trail that recorded what crossed
// and not what was refused would let a user conclude that the two questions mole
// asked are all it tried — and the refusals are the more interesting half, because
// they are the gate doing the thing the user is trusting it to do.
func (a *LocalComputeActor) crossing(lead core.Lead, c connector.Connector, query string,
	rec gate.Record, outcome core.CrossingOutcome, detail string) core.Crossing {
	if rec.QueryHash == "" {
		// A refusal never became an envelope, so there is no hash to read off one.
		// A refusal from BEFORE rendering has no statement either — the plan never
		// became SQL — so the hash is over whatever identifies the attempt, and the
		// query column is honestly empty rather than filled with a fiction.
		rec.Query, rec.QueryHash = query, gate.HashQuery(query)
	}
	return core.Crossing{
		SessionID:       lead.SessionID,
		LeadID:          lead.ID,
		Connector:       c.Name,
		Query:           rec.Query,
		QueryHash:       rec.QueryHash,
		Outcome:         outcome,
		Detail:          detail,
		RowsDescribed:   rec.RowsDescribed,
		Columns:         rec.Columns,
		ColumnsWithheld: rec.ColumnsWithheld,
		Buckets:         rec.Buckets,
		Suppressed:      rec.Suppressed,
		BeyondTopK:      rec.BeyondTopK,
		Tests:           rec.Tests,
		Truncated:       rec.Truncated,
	}
}

// runQuery is the SQL route: render, both gates, envelope.
func (a *LocalComputeActor) runQuery(
	ctx context.Context, lead core.Lead, c connector.Connector, p hypothesis.Plan,
) (string, string, verdictNote, error) {
	var note verdictNote

	// The pipeline lives in internal/compute so toolkit mode runs the identical
	// one. Two copies would be two places for §12's rules to drift.
	opts := a.Gate
	res := compute.Run(ctx, c, p, opts, a.logger())
	note.add(res.Crossing(lead.SessionID, lead.ID, c.Name))
	if res.Err != nil {
		return "", "", note, res.Err
	}
	env := res.Envelope

	// §4's fourth column, when there is a comparison to check.
	//
	// A second statement rather than one cleverer one: the pooled comparison and
	// the per-window statistics are different groupings of the same table, and
	// folding them together would make every bucket a (group, window) pair — so
	// the k-anonymity floor would apply to thirds of a group and the envelope a
	// reader sees would be three times as long and about nothing they asked.
	env = a.checkStability(ctx, lead, c, p, env, &note)

	note.finding = describeFinding(p, env)
	note.unsupported = unsupported(envelopeVerdicts(env))
	// §4's table: a local claim's source is "connector name + query hash", and a
	// claim nobody can trace back to a statement is not evidence.
	cite := fmt.Sprintf("connector:%s#%s", c.Name, env.QueryHash[:16])
	return env.Text(), cite, note, nil
}

// checkStability re-tests each comparison on deterministic holdout windows (§4).
//
// Best effort, and quiet about it: a failure here is one qualifying clause missing
// from evidence that is otherwise complete and already paid for. The clause says
// which of "stable", "not stable" and "could not be checked" happened, so a
// missing check is never read as a passed one.
//
// The second statement is audited like the first. It reads the same data and the
// figures it produces cross to the same model, so a trail that recorded one and
// not the other would understate what left the machine.
func (a *LocalComputeActor) checkStability(
	ctx context.Context, lead core.Lead, c connector.Connector, p hypothesis.Plan,
	env gate.AggregateEnvelope, note *verdictNote,
) gate.AggregateEnvelope {
	if len(env.TestResults) == 0 {
		return env
	}
	query, err := hypothesis.RenderHoldout(c, p, stats.HoldoutWindows)
	if err != nil {
		return env
	}
	if err := sqlguard.Check(query); err != nil {
		a.logger().WarnContext(ctx, "the holdout statement did not pass the parse gate",
			"connector", c.Name, "err", err)
		return env
	}

	db, err := c.Open()
	if err != nil {
		return env
	}
	defer db.Close()

	opts := a.Gate
	opts.Log = a.logger()
	opts.FreeTextColumns = freeTextColumns(c)
	// The pairwise comparison is suppressed: over (group, window) buckets it would
	// compare "north in window 1" against "south in window 2".
	opts.SkipTests = true

	windowed, err := gate.Aggregate(ctx, db, query, opts)
	if err != nil {
		// Recorded as a refusal like any other. The commonest cause is the floor:
		// a third of a group is often under five records, which is the privacy
		// guarantee holding rather than a fault.
		note.add(a.crossing(lead, c, query, gate.Record{}, core.CrossingRefused, err.Error()))
		env.Notes = append(env.Notes, "stability across holdout windows could not be "+
			"checked: "+err.Error())
		return env
	}
	note.add(a.crossing(lead, c, query, gate.Describe(windowed), core.CrossingCrossed, ""))

	byGroup := windowGroups(windowed)
	measure := "the mean"
	for i, t := range env.TestResults {
		h := stats.CheckStability(measure, t, byGroup[t.GroupA], byGroup[t.GroupB])
		env.TestResults[i] = stats.WithHoldout(t, h)
	}
	return env
}

// windowGroups indexes a windowed envelope by group, then by window label.
//
// The bucket key is [group, window] because the statement groups by both, in that
// order. Buckets the gate folded or suppressed are absent, which CheckStability
// reports as an untestable window rather than a passing one.
func windowGroups(env gate.AggregateEnvelope) map[string]map[string]stats.Group {
	out := map[string]map[string]stats.Group{}
	for _, b := range env.TopK {
		if b.Other || len(b.Key) != 2 {
			continue
		}
		g, ok := gate.GroupFrom(b)
		if !ok {
			continue
		}
		group, window := b.Key[0], b.Key[1]
		// The group's own name, so it matches the pooled test's GroupA/GroupB —
		// which gate.groupFrom builds by joining the whole key.
		g.Name = group
		if out[group] == nil {
			out[group] = map[string]stats.Group{}
		}
		out[group][window] = g
	}
	return out
}

// runCode is the sandbox route (§12.1).
//
// The script never touches the aggregation gate, because it is on the other side
// of it: the gate exists to stop rows reaching a model, and inside the sandbox
// there is no model. What constrains this route is the container and the output
// contract — see internal/compute/coderunner.
func (a *LocalComputeActor) runCode(
	ctx context.Context, lead core.Lead, c connector.Connector, p hypothesis.Plan,
) (string, string, verdictNote, error) {
	var note verdictNote
	if a.Code == nil {
		return "", "", note, errors.New(
			"no container runtime is available, so a code hypothesis cannot run " +
				"(mole doctor reports what is missing)")
	}
	// Rendered and checked, or not sent at all. The query is only a hint to the
	// script — the container is the control — but an unchecked hint is still a
	// string mole built from a model's choice, and unfilled template slots would
	// reach the container as MOLE_QUERY="".
	var query string
	if p.Template != "" {
		rendered, err := hypothesis.Render(c, p)
		if err != nil {
			return "", "", note, err
		}
		if err := sqlguard.Check(rendered); err != nil {
			return "", "", note, err
		}
		query = rendered
	}

	// The sandbox route crosses too, and is recorded as such. §12.1 puts the
	// control in the container rather than the gate here — but the promise is that
	// a user can audit what left their machine, and "the container held it" is not
	// the same as "nothing left". The declared metrics did.
	codeHash := "code:" + shortHash(p.Code.Script)

	out, err := a.Code.Analyze(ctx, coderunner.Request{
		DBPath: c.DBPath,
		Query:  query,
		Script: p.Code.Script,
		Contract: coderunner.Contract{
			Metrics: p.Code.Metrics,
			Tests:   p.Code.Tests,
		},
	})
	if err != nil {
		note.add(a.crossing(lead, c, query, gate.Record{QueryHash: codeHash, Query: query},
			core.CrossingRefused, err.Error()))
		return "", "", note, err
	}
	if out.Empty() {
		refusal := fmt.Errorf(
			"the analysis produced nothing that may cross (%d declared and %d undeclared "+
				"output(s) refused)", len(out.Dropped), out.Undeclared)
		note.add(a.crossing(lead, c, query, gate.Record{QueryHash: codeHash, Query: query},
			core.CrossingRefused, refusal.Error()))
		return "", "", note, refusal
	}
	note.add(a.crossing(lead, c, query, gate.Record{
		QueryHash: codeHash,
		Query:     query,
		// The counts a script's output has: how many declared numbers crossed, and
		// how many outputs the contract refused. Undeclared is a COUNT rather than
		// a list for the reason coderunner gives — a key is text the script chose.
		Columns:         len(out.Metrics),
		ColumnsWithheld: len(out.Dropped) + out.Undeclared,
		Tests:           len(out.Findings),
	}, core.CrossingCrossed, ""))

	note.finding = describeCodeFinding(p, out)
	note.unsupported = unsupported(outputVerdicts(out))
	// The script is what produced the evidence, so the script is what the
	// citation identifies. A query hash would name a statement that only
	// suggested what to look at.
	cite := fmt.Sprintf("connector:%s#code:%s", c.Name, shortHash(p.Code.Script))
	return out.Text(p.Code.Script), cite, note, nil
}

// planDescriptor identifies an attempt that never became a statement.
//
// Not SQL and not pretending to be: the template and the columns the model chose,
// which is what a reader auditing a refusal needs and all that exists at that
// point. Identifiers only — the columns are names from the profile, which already
// crossed when the model was shown the schema.
func planDescriptor(p hypothesis.Plan) string {
	var b strings.Builder
	fmt.Fprintf(&b, "-- not rendered: template=%s table=%s", p.Template, p.Table)
	names := make([]string, 0, len(p.Columns))
	for role := range p.Columns {
		names = append(names, role)
	}
	sort.Strings(names)
	for _, role := range names {
		fmt.Fprintf(&b, " %s=%s", role, p.Columns[role])
	}
	return b.String()
}

func shortHash(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])[:16]
}

// unsupported reports whether a run produced verdicts and none supported a
// difference.
//
// One predicate over verdicts, because there were two — the same three lines
// over two element types — and §4's cap is exactly the rule that must not
// diverge by route. Empty means no test was computable, which is not the same as
// a test that failed and is deliberately not capped: a distribution has no
// comparison to fail.
func unsupported(verdicts []stats.Verdict) bool {
	if len(verdicts) == 0 {
		return false
	}
	for _, v := range verdicts {
		if v == stats.Significant {
			return false
		}
	}
	return true
}

func envelopeVerdicts(env gate.AggregateEnvelope) []stats.Verdict {
	out := make([]stats.Verdict, 0, len(env.TestResults))
	for _, t := range env.TestResults {
		out = append(out, t.Verdict)
	}
	return out
}

func outputVerdicts(o coderunner.Output) []stats.Verdict {
	out := make([]stats.Verdict, 0, len(o.Findings))
	for _, f := range o.Findings {
		out = append(out, f.Verdict)
	}
	return out
}

func describeCodeFinding(p hypothesis.Plan, out coderunner.Output) string {
	headline := fmt.Sprintf("ran in the sandbox, %d metric(s) and %d statistical result(s)",
		len(out.Metrics), len(out.Findings))
	if n := len(out.Dropped) + out.Undeclared; n > 0 {
		headline += fmt.Sprintf(", %d output(s) refused", n)
	}
	return finding(p.Question, "a sandboxed analysis over "+p.Connector,
		headline, outputVerdicts(out))
}

func findConnector(sources []connector.Connector, name string) (connector.Connector, bool) {
	for _, c := range sources {
		if strings.EqualFold(c.Name, name) {
			return c, true
		}
	}
	return connector.Connector{}, false
}

// freeTextColumns is what the profile already decided, handed to the gate so it
// can union that with what it derives from the result.
func freeTextColumns(c connector.Connector) []string {
	var out []string
	for _, t := range c.Tables {
		for _, col := range t.Columns {
			if col.FreeText {
				out = append(out, col.Name)
			}
		}
	}
	return out
}

// unsupportedAssertionCap is where a claim lands when the envelope it was mined
// from ran a comparison and the comparison did not support one.
//
// A cap and not a rejection, because the claim may be perfectly true — "the
// north region has the most records" is a fact about the counts whether or not
// the means differ significantly. What it must not do is enter the graph
// asserting as much as a claim the data actually supports. §11.3 derives
// confidence from the graph, and AssertionStrength is what the source says
// about itself; an underpowered result should not say much.
const unsupportedAssertionCap = 0.3

// -----------------------------------------------------------------------------
// Planning
// -----------------------------------------------------------------------------

// plan asks the model which questions to put to the data.
//
// One call for ALL the hypotheses rather than one per hypothesis: the schema is
// the expensive part of the prompt and it does not change between them.
func (a *LocalComputeActor) plan(
	ctx context.Context, lead core.Lead, sources []connector.Connector,
	max int, maxInput int64, res *Result,
) ([]hypothesis.Plan, int64, error) {
	fence := fenceToken()
	prompt := planPrompt(fence, lead.Query, sources, max, a.Code != nil)

	// The schema is unbounded in principle — every column of every registered
	// connector — so it is capped here rather than discovered to be too large by
	// the provider. Truncated is set so a thin plan reads as a budget outcome.
	if limit := int(maxInput) * 3; len(prompt) > limit && limit > 0 {
		prompt = prompt[:limit] + "\n[schema truncated: the input allowance for this " +
			"lead does not cover every registered column]"
		res.Truncated = true
	}
	spent := llm.EstimateTokens(len(prompt))

	resp, err := a.LLM.Complete(ctx, llm.Request{
		Tier:      llm.TierCheap,
		System:    planSystemPrompt,
		Messages:  []llm.Message{llm.User(prompt)},
		MaxTokens: 2048,
	})
	if resp == nil {
		return nil, spent, err
	}
	res.Costs = append(res.Costs,
		toolCallFor(lead.SessionID, lead, resp, "local:plan", a.Pricing, a.Log, err))
	if err != nil {
		return nil, spent, err
	}
	if resp.Refused {
		return nil, spent, fmt.Errorf("actors: model refused to plan (%s)", resp.RefusalCategory)
	}

	plans, err := parsePlans(resp.Text)
	if err != nil {
		return nil, spent, err
	}
	if len(plans) > max {
		plans = plans[:max]
	}
	return plans, spent, nil
}

func parsePlans(raw string) ([]hypothesis.Plan, error) {
	body := extractJSONArray(raw)
	if body == "" {
		return nil, fmt.Errorf("actors: no hypothesis list in the model's reply: %s",
			truncateForLog(raw))
	}
	var plans []hypothesis.Plan
	if err := json.Unmarshal([]byte(body), &plans); err != nil {
		// The reply is quoted, and it needs to be. A weak model fails here in
		// ways the error alone does not identify — a live run against a 3B
		// model returned a doubly-wrapped array, which reads only as "cannot
		// unmarshal array into ...Plan" unless the text is in front of you.
		return nil, fmt.Errorf("actors: parse hypotheses: %w; reply was: %s",
			err, truncateForLog(body))
	}
	return plans, nil
}

// extractJSONArray finds the outermost bracketed array in a reply.
func extractJSONArray(raw string) string {
	start := strings.Index(raw, "[")
	end := strings.LastIndex(raw, "]")
	if start < 0 || end <= start {
		return ""
	}
	return raw[start : end+1]
}

// -----------------------------------------------------------------------------
// Summary
// -----------------------------------------------------------------------------

// finding is one line of the planner's summary.
//
// One builder for both routes. There were two, sharing three of four decisions
// in different words and a different order — and the §4 verdict clause is the
// part that must read the same whichever route produced it, since a replan that
// cannot tell an underpowered result from a settled one builds on it.
func finding(question, fallback, headline string, verdicts []stats.Verdict) string {
	q := strings.TrimSpace(question)
	if q == "" {
		q = fallback
	}
	line := q + " — " + headline
	for _, v := range verdicts {
		line += fmt.Sprintf("; comparison %s", v)
	}
	return line
}

func describeFinding(p hypothesis.Plan, env gate.AggregateEnvelope) string {
	headline := fmt.Sprintf("%d row(s) described", env.RowCount)
	if env.Suppressed > 0 {
		headline += fmt.Sprintf(", %d group(s) below the reporting floor", env.Suppressed)
	}
	if env.BeyondTopK > 0 {
		headline += fmt.Sprintf(", %d beyond the reported limit", env.BeyondTopK)
	}
	return finding(p.Question,
		fmt.Sprintf("%s over %s.%s", p.Template, p.Connector, p.Table),
		headline, envelopeVerdicts(env))
}

// localSummary describes the run for the planner, and names the boundary that
// actually applied.
//
// The gate claim is appended only when the gate ran. A sandbox-only run never
// touches it, and the summary is read by both the planner and the report.
func localSummary(lead core.Lead, findings []string, res *Result, usedGate, usedSandbox bool) string {
	if len(findings) == 0 {
		// ChunksSkipped plus ChunksFailed: the first counts hypotheses refused
		// before mining, the second those whose mining call errored. Reporting
		// only the first under-reported what was attempted.
		return fmt.Sprintf(
			"No hypothesis over the registered data could be answered for %q. "+
				"%d were tried.", lead.Query,
			res.Stats.ChunksSkipped+res.Stats.ChunksFailed)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Queried local data for %q. %d hypothes%s answered, yielding %d claim(s).\n\n",
		lead.Query, len(findings), plural(len(findings)), len(res.Claims))
	for _, f := range findings {
		b.WriteString("- " + f + "\n")
	}
	b.WriteString("\nNo row from the data left the machine. ")
	switch {
	case usedGate && usedSandbox:
		b.WriteString("The queried figures crossed the aggregation gate as aggregates " +
			"(§12.1); the sandboxed ones were computed in a container with no network " +
			"and returned only the values their plan declared.")
	case usedSandbox:
		b.WriteString("The figures were computed inside a container with no network, " +
			"no writable filesystem and a read-only view of the data, and only the " +
			"values the plan declared were returned (§12.1).")
	default:
		b.WriteString("Every figure above came through the aggregation gate as an " +
			"aggregate (§12.1).")
	}
	return b.String()
}

func plural(n int) string {
	if n == 1 {
		return "is"
	}
	return "es"
}

// evidenceTitle names the source in the mining prompt.
func evidenceTitle(p hypothesis.Plan) string {
	if p.Code != nil {
		return fmt.Sprintf("%s.%s — sandboxed analysis", p.Connector, p.Table)
	}
	return fmt.Sprintf("%s.%s — %s", p.Connector, p.Table, p.Template)
}

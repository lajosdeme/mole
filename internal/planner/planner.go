package planner

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/llm/jsonish"
)

// Planner decomposes a question and decides what to research next.
//
// Two model calls exist here and nowhere else in the loop: an initial
// decomposition, and a replan against the digest. Everything between them is
// mechanical bookkeeping, which is what keeps planner cost proportional to the
// number of REPLANS rather than the number of leads (§9.1).
type Planner struct {
	LLM llm.Provider

	// MaxInitialLeads bounds the first fan-out. A model asked to decompose
	// without a bound will happily produce thirty sub-questions and spend the
	// whole budget before the first replan can react.
	MaxInitialLeads int

	// MaxNewLeadsPerReplan bounds each subsequent fan-out.
	MaxNewLeadsPerReplan int

	// ReplanEvery is §9.1's batching: replan after this many completed leads
	// rather than after every one, which cuts planner calls by roughly k.
	ReplanEvery int

	// MaxDepth caps the lead tree independently of budget (§9.1). A follow-up
	// of a follow-up of a follow-up is usually drift, not depth.
	//
	// Zero means unset, per Go convention, and takes DefaultMaxDepth. Use
	// DepthNone to mean "no follow-up rounds at all" — the two have to be
	// distinguishable, because the obvious `<= 0` treatment made
	// `--max-depth 0` silently plan two extra rounds against ceilings that had
	// been sized for none.
	MaxDepth int
}

// Defaults chosen to spend a small budget usefully rather than exhaust it on
// the first fan-out.
const (
	DefaultMaxInitialLeads      = 4
	DefaultMaxNewLeadsPerReplan = 3
	DefaultReplanEvery          = 3
	DefaultMaxDepth             = 2
)

// plannerMaxTokens is the output allowance for a planning call.
//
// Far more than the JSON needs, because a reasoning model spends this budget on
// its own reasoning first and only then emits content. qwen3:4b was measured at
// 1000-1800 characters of reasoning on a trivial prompt; at 300 tokens it
// returned nothing at all. The planner is the worst place to be short — a
// failure here ends the session before any research happens — and unused output
// tokens are not billed.
const plannerMaxTokens = 4000

// DepthNone disables follow-up rounds: the initial decomposition and nothing
// more. Distinct from the zero value, which means "unset".
const DepthNone = -1

func (p *Planner) withDefaults() *Planner {
	out := *p
	if out.MaxInitialLeads <= 0 {
		out.MaxInitialLeads = DefaultMaxInitialLeads
	}
	if out.MaxNewLeadsPerReplan <= 0 {
		out.MaxNewLeadsPerReplan = DefaultMaxNewLeadsPerReplan
	}
	if out.ReplanEvery <= 0 {
		out.ReplanEvery = DefaultReplanEvery
	}
	switch {
	case out.MaxDepth == 0:
		out.MaxDepth = DefaultMaxDepth
	case out.MaxDepth < 0:
		out.MaxDepth = 0
	}
	return &out
}

// Plan is what a planning call produced.
type Plan struct {
	Questions []SubQuestion
	Leads     []core.Lead
	Usage     llm.Usage
	Model     string
	// Elapsed is how long the model call took, carried through to the ledger
	// row.
	Elapsed time.Duration
	// Done is set when the planner judges the research complete. Only Replan
	// can set it.
	Done bool

	// Answered names open sub-questions the planner now considers closed.
	// Returned rather than applied: the caller owns the digest, and a planner
	// that mutated it would make a replan unrepeatable against a cassette.
	Answered []string
	// Rationale is one line for the trace, not for the model.
	Rationale string
}

// InitialLeads decomposes the prompt into the first round of sub-questions.
//
// The prompt is the user's, so it is trusted input in the §3.2 sense — nothing
// fetched has been seen yet. It is still delimited, because a research question
// can legitimately quote a web page, and the difference between "the user typed
// this" and "a page said this" stops being visible once it is concatenated.
func (p *Planner) InitialLeads(ctx context.Context, sess *core.Session) (*Plan, error) {
	pl := p.withDefaults()

	fence := fenceToken()
	req := llm.Request{
		Tier:      llm.TierStrong,
		Effort:    llm.StrongEffort,
		System:    plannerSystemPrompt,
		Messages:  []llm.Message{llm.User(decomposePrompt(fence, sess.Prompt, pl.MaxInitialLeads))},
		MaxTokens: plannerMaxTokens,
	}

	resp, err := pl.LLM.Complete(ctx, req)
	if resp == nil {
		return nil, err
	}
	out := &Plan{Usage: resp.Usage, Model: resp.Model, Elapsed: resp.Elapsed}
	if err != nil {
		return out, err
	}
	if resp.Refused {
		return out, fmt.Errorf("planner: model refused to plan (%s)", resp.RefusalCategory)
	}

	parsed, err := parsePlan(resp.Text)
	if err != nil {
		return out, err
	}

	out.Questions = toSubQuestions(parsed.Questions, pl.MaxInitialLeads)
	if len(out.Questions) == 0 {
		// A decomposition that produced nothing would end the session before
		// it began. The original question is always a valid lead.
		out.Questions = []SubQuestion{{ID: "q1", Text: sess.Prompt}}
	}
	out.Leads = LeadsAcross(sess.ID, sessionActors(sess), out.Questions, 0, nil)
	out.Rationale = parsed.Rationale
	return out, nil
}

// ShouldReplan implements §9.1's batching.
//
// Replanning after every lead is what made rev 1 quadratic. Replanning never is
// a fixed plan, which cannot react to a dead end. Every k, or when the queue
// drains, is the compromise.
func (p *Planner) ShouldReplan(completedSinceLastReplan int, queueEmpty bool) bool {
	pl := p.withDefaults()
	if queueEmpty {
		// The queue draining is a branch of the lead tree closing, and it is
		// the last moment a replan can add anything.
		return true
	}
	return completedSinceLastReplan >= pl.ReplanEvery
}

// DepthExhausted reports whether the lead tree has reached the depth cap.
//
// Exported so the caller can take that decision for free. Replan checks it too,
// but only after the caller has reserved budget for a model call it turns out
// not to make — and when a ceiling has just fired, the reservation is refused
// and the session reports the ceiling as its stop reason instead of the cap that
// actually ended it. Same outcome, wrong explanation, and the stop reason is an
// eval metric (§14.3).
func (p *Planner) DepthExhausted(depth int) bool {
	return depth >= p.withDefaults().MaxDepth
}

// DepthCapReason describes the cap for a trace, without a model call.
func (p *Planner) DepthCapReason() string {
	return fmt.Sprintf("lead tree reached the depth cap of %d", p.withDefaults().MaxDepth)
}

// Replan proposes follow-up leads from the digest.
//
// The digest carries counts and the planner's own prior wording — no page text
// (see the package comment). That closes the §3.2 injection path completely,
// and costs something real: the planner can see that a sub-question found no
// evidence, but not that what WAS found suggests a different angle. A
// content-aware replan needs the summaries fenced back in, and is deferred
// until there is an eval corpus to show whether it pays for itself.
func (p *Planner) Replan(ctx context.Context, sess *core.Session, d *Digest, depth int) (*Plan, error) {
	pl := p.withDefaults()

	out := &Plan{}
	if depth >= pl.MaxDepth {
		// The cap binds independently of budget (§9.1): a follow-up of a
		// follow-up of a follow-up is usually drift, not depth.
		//
		// Kept as a backstop even though the executor now checks DepthExhausted
		// before reserving: this is exported, and a caller that forgets must not
		// get an unbounded lead tree.
		out.Done = true
		out.Rationale = pl.DepthCapReason()
		return out, nil
	}

	fence := fenceToken()
	resp, err := pl.LLM.Complete(ctx, llm.Request{
		Tier:      llm.TierStrong,
		Effort:    llm.StrongEffort,
		System:    plannerSystemPrompt,
		Messages:  []llm.Message{llm.User(replanPrompt(fence, d, pl.MaxNewLeadsPerReplan))},
		MaxTokens: plannerMaxTokens,
	})
	if resp == nil {
		return nil, err
	}
	out.Usage, out.Model, out.Elapsed = resp.Usage, resp.Model, resp.Elapsed
	if err != nil {
		return out, err
	}
	if resp.Refused {
		return out, fmt.Errorf("planner: model refused to replan (%s)", resp.RefusalCategory)
	}

	parsed, err := parsePlan(resp.Text)
	if err != nil {
		return out, err
	}

	out.Done = parsed.Done
	out.Rationale = parsed.Rationale
	out.Questions = toSubQuestions(parsed.Questions, pl.MaxNewLeadsPerReplan)

	// Answered questions the planner named. Applied by the caller, which owns
	// the digest.
	out.Answered = parsed.Answered

	if !out.Done {
		out.Leads = LeadsAcross(sess.ID, sessionActors(sess), out.Questions, depth+1, nil)
	}
	return out, nil
}

// sessionActors is the set of actors a session's leads may be assigned to.
//
// Every actor the session asked for, in the order it asked. This used to return
// ONE actor and prefer web whenever web was present, which meant a session
// asking for web and academic got web for every lead — the academic actor was
// built, registered, and never given work. LeadsAcross distributes them.
//
// Not a model choice per sub-question. That is a real improvement and a separate
// change: it needs a prompt, a schema field, and a failure mode for when the
// model names an actor the session did not enable.
func sessionActors(sess *core.Session) []core.ActorType {
	if sess == nil || len(sess.ActorTypes) == 0 {
		return []core.ActorType{core.ActorWeb}
	}
	return sess.ActorTypes
}

func toSubQuestions(in []plannedQuestion, max int) []SubQuestion {
	var out []SubQuestion
	for i, q := range in {
		text := strings.TrimSpace(q.Question)
		if text == "" {
			continue
		}
		id := strings.TrimSpace(q.ID)
		if id == "" {
			id = fmt.Sprintf("q%d", i+1)
		}
		out = append(out, SubQuestion{ID: id, Text: text})
		if len(out) >= max {
			break
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

type plannedQuestion struct {
	ID       string `json:"id"`
	Question string `json:"question"`
}

type planResponse struct {
	Questions []plannedQuestion `json:"questions"`
	Answered  []string          `json:"answered"`
	Done      bool              `json:"done"`
	Rationale string            `json:"rationale"`
}

func parsePlan(raw string) (planResponse, error) {
	body := jsonish.ExtractObject(raw)
	if body == "" {
		return planResponse{}, fmt.Errorf("planner: no JSON object in model response (%.80q)", raw)
	}
	var out planResponse
	if err := json.Unmarshal([]byte(body), &out); err != nil {
		return planResponse{}, fmt.Errorf("planner: parse plan: %w", err)
	}
	return out, nil
}

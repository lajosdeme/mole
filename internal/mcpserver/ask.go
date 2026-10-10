package mcpserver

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/lajosdeme/mole/internal/budget"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/output"
)

// AskAllowance bounds what one research.ask call may spend, in micro-dollars.
//
// §13's promise is that an ask is cheap: one model call against claims already
// paid for. The allowance is what makes that true rather than aspirational —
// without it an ask draws on whatever the session has left, and a session that
// stopped early with most of its budget intact could spend it all answering one
// follow-up.
const AskAllowance = 50_000 // $0.05

type AskIn struct {
	SessionID string `json:"session_id" jsonschema:"the session whose research should be queried"`
	Question  string `json:"question" jsonschema:"the follow-up question to answer from that session's claims"`
}

type AskOut struct {
	SessionID string `json:"session_id"`
	Question  string `json:"question"`
	Answer    string `json:"answer"`

	// Claims are the ones the answer is built from, so a caller can weigh the
	// evidence rather than the prose (§5.3).
	Claims []Claim `json:"claims"`
	// Citations maps each [n] in the answer to its source.
	Citations []Citation `json:"citations"`

	// AskSessionID is the small session this answer was charged to. Present so
	// the cost is traceable: it will not appear against the session that was
	// queried, whose accounts are closed.
	AskSessionID string `json:"ask_session_id,omitempty"`
	Spent        int64  `json:"spent"`
	// Degraded says why the answer is not synthesized prose, when it is not.
	Degraded string `json:"degraded,omitempty"`
}

// Citation is one numbered source behind an answer.
//
// Quotes plural: several claims can cite one source, and §13 asks for the
// verified span per citation so a reader can check the answer without
// re-fetching the page.
type Citation struct {
	N           int      `json:"n"`
	Source      string   `json:"source"`
	Quotes      []string `json:"quotes,omitempty"`
	PublishedAt string   `json:"published_at,omitempty"`
}

// ask answers a new question from a finished session's graph (§13).
//
// The ask runs as its OWN session, mode "ask", with a small budget, reading the
// source session's claims. That is forced rather than chosen: §8 refuses a
// reservation against a terminal session, because a finished ledger is settled
// and reconciled and adding cost rows later rewrites a closed account.
//
// A session of its own keeps every §8 invariant — each model call belongs to a
// live session, each cost row has a reservation — and makes the ask appear in
// sessions.list with its own cost.
//
// Exported because `mole ask` is the same operation from a terminal, and the
// budgeting is the part that must not be written twice. The CLI builds a Deps
// with no Supervisor and calls this directly.
func (d Deps) Ask(ctx context.Context, in AskIn) (AskOut, error) {
	if d.Log == nil {
		// New() fills this in; a caller constructing Deps by hand (the CLI) may
		// not, and every warning path below would nil-panic.
		d.Log = slog.Default()
	}
	question := strings.TrimSpace(in.Question)
	if question == "" {
		return AskOut{}, errors.New("question is empty")
	}
	src, err := d.loadSession(ctx, in.SessionID)
	if err != nil {
		return AskOut{}, err
	}
	if src.Status == core.StatusRunning {
		return AskOut{}, fmt.Errorf(
			"session %s is still running; ask it once it has finished, or its claim "+
				"graph will change under the answer", src.ID)
	}

	allowance := int64(AskAllowance)
	if d.MaxSessionUSD > 0 && allowance > d.MaxSessionUSD {
		allowance = d.MaxSessionUSD
	}

	led := budget.New(d.Store, budget.DefaultConfig())
	gen := &output.Generator{LLM: d.LLM}

	var (
		askSess     *core.Session
		reservation *core.Reservation
	)
	if d.LLM != nil {
		askSess, err = led.CreateSession(ctx, budget.SessionSpec{
			Prompt:     question,
			Mode:       core.ModeAsk,
			ActorTypes: []core.ActorType{core.ActorWeb},
			BudgetUnit: core.BudgetUSD,
			Budget:     allowance,
			// One call, no leads, no fetches. The ceilings say so rather than
			// relying on the loop never being started.
			MaxToolCalls: 1,
			MaxLeads:     0,
			MaxWallClock: d.askTimeout(),
		})
		if err != nil {
			return AskOut{}, fmt.Errorf("could not open an ask session: %w", err)
		}
		// The escrow a session reserves at creation is for a report this session
		// will never write; releasing it makes the whole allowance spendable on the
		// one call the ask actually makes.
		if _, rerr := led.ReleaseEscrow(ctx, askSess.ID); rerr != nil {
			d.Log.Warn("could not release ask escrow", "session", askSess.ID, "err", rerr)
		}
		if cur, lerr := d.loadSession(ctx, askSess.ID); lerr == nil {
			if avail := cur.Available(); avail > 0 {
				reservation, err = led.ReserveOutput(ctx, askSess.ID, avail)
				if err != nil {
					d.Log.Warn("could not reserve for an ask", "session", askSess.ID, "err", err)
				}
			}
		}
	}
	if reservation == nil {
		// No provider, or nothing reservable. Answer then lists the relevant claims
		// with citations rather than making a call whose cost cannot be recorded.
		gen.LLM = nil
	}

	// Reads the SOURCE session's claims; spends the ask session's budget.
	//
	// The deadline is applied HERE and not only written into the session spec.
	// MaxWallClock is checked by the executor loop, and an ask never enters it —
	// so until this line the constant below documented a bound that nothing
	// enforced, and a provider that accepted the connection and then stopped
	// responding would hang the call forever with the reservation still held.
	askCtx, cancel := context.WithTimeout(ctx, d.askTimeout())
	defer cancel()

	rep, err := gen.Answer(askCtx, d.Store, src.ID, question)
	if err != nil {
		d.closeAsk(ctx, led, askSess, reservation, nil, core.StatusFailed)
		return AskOut{}, err
	}

	// Logged, never returned. The detail can carry the provider's endpoint or the
	// model output that validation just rejected, and this reply lands verbatim in
	// a caller's context. See output.Report.Degraded.
	if rep.DegradedDetail != "" {
		d.Log.Warn("ask degraded", "session", src.ID,
			"reason", rep.Degraded, "detail", rep.DegradedDetail)
	}

	spent := d.closeAsk(ctx, led, askSess, reservation, rep, core.StatusDone)

	out := AskOut{
		SessionID: src.ID,
		Question:  question,
		Answer:    rep.Body,
		Spent:     spent,
		Degraded:  rep.Degraded,
	}
	if askSess != nil {
		out.AskSessionID = askSess.ID
	}
	for _, c := range rep.Citations {
		cit := Citation{N: c.N, Source: c.Source, Quotes: c.Quotes}
		if c.PublishedAt != nil {
			cit.PublishedAt = c.PublishedAt.UTC().Format(time.RFC3339)
		}
		out.Citations = append(out.Citations, cit)
	}
	for _, f := range rep.Findings {
		if f.Claim == nil {
			continue
		}
		out.Claims = append(out.Claims, Claim{
			ID:         f.Claim.ID,
			Text:       f.Claim.Text,
			Source:     f.Claim.Source,
			Quote:      f.Claim.Quote,
			Confidence: f.Confidence,
			Grounded:   f.Claim.Grounded,
		})
	}
	return out, nil
}

// ask is the MCP tool. It carries no logic of its own so the CLI and an agent
// cannot drift apart.
func (d Deps) ask(ctx context.Context, _ *mcp.CallToolRequest, in AskIn) (*mcp.CallToolResult, AskOut, error) {
	out, err := d.Ask(ctx, in)
	return nil, out, err
}

// DefaultAskTimeout bounds an ask's wall clock. One model call; five minutes is
// generous for it and short enough that a wedged provider does not pin a
// reservation until the abandonment sweep. Enforced on the context in Ask.
const DefaultAskTimeout = 5 * time.Minute

// askTimeout is Deps.AskTimeout, or the default. Overridable so the enforcement
// can be tested in milliseconds instead of five minutes — a bound nothing can
// afford to exercise is a bound nobody checks.
func (d Deps) askTimeout() time.Duration {
	if d.AskTimeout > 0 {
		return d.AskTimeout
	}
	return DefaultAskTimeout
}

// closeAsk settles the reservation and finalizes the ask session.
//
// Called on every path including the failing one: an unresolved hold is budget
// neither spent nor available, and an ask session left running would be swept as
// abandoned half an hour later rather than closed now.
func (d Deps) closeAsk(
	ctx context.Context,
	led *budget.Ledger,
	askSess *core.Session,
	reservation *core.Reservation,
	rep *output.Report,
	status core.SessionStatus,
) int64 {
	if askSess == nil {
		return 0
	}
	ctx = context.WithoutCancel(ctx)

	var spent int64
	if reservation != nil {
		var calls []core.ToolCall
		if rep != nil && !rep.Cost.IsZero() {
			calls = append(calls, core.ToolCall{
				SessionID: askSess.ID,
				Role:      core.RoleOutput,
				Type:      core.CallLLM,
				Model:     rep.Model,
				Input:     "ask",
				Cost:      output.Price(d.Pricing, rep),
			})
		}
		settled, err := led.Settle(ctx, reservation, calls)
		if err != nil {
			d.Log.Warn("settling an ask failed", "session", askSess.ID, "err", err)
			if rerr := led.Release(ctx, reservation); rerr != nil {
				d.Log.Warn("could not release an ask hold", "session", askSess.ID, "err", rerr)
			}
		} else {
			spent = settled.Cost.BudgetAmount(core.BudgetUSD)
		}
	}
	// No reason: an ask is one call with no ceilings to hit, so there is never
	// anything to explain beyond the status.
	if err := led.Finish(ctx, askSess.ID, status, ""); err != nil {
		d.Log.Warn("could not finalize an ask session", "session", askSess.ID, "err", err)
	}
	return spent
}

// llmProvider is the subset of the provider an ask needs. Declared so Deps can
// hold one without the whole actor.
type llmProvider = llm.Provider

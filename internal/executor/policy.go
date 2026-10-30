package executor

import (
	"context"
	"errors"
	"net"
	"time"

	"github.com/lajosdeme/mole/internal/budget"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/tools/academic"
	"github.com/lajosdeme/mole/internal/tools/search"
)

// Error policy (§9.5).
//
// actor.Run fails routinely, and the useful question is never "did it fail" but
// "does failing again help". Three answers:
//
//   - Transient — the same request might work shortly. Retry, then give up.
//   - Degraded — this lead is not going to produce anything, but the session is
//     fine. Record why and move on.
//   - Fatal — nothing will work. Stop, and spend the escrow on a partial report
//     rather than continuing to burn budget on calls that cannot succeed.
//
// Partial cost is settled in every case. The tokens were spent whether or not
// the call succeeded, and a ledger of successes cannot enforce a ceiling.

// Class is what to do about an error.
type Class int

const (
	// Degraded is the default. An unrecognized error is more likely a dead
	// link or a parse failure than an outage, and treating the unknown as
	// fatal would abort a session over one bad page.
	Degraded Class = iota
	Transient
	Fatal
)

func (c Class) String() string {
	switch c {
	case Transient:
		return "transient"
	case Fatal:
		return "fatal"
	default:
		return "degraded"
	}
}

// MaxAttempts caps retries of a whole lead (§9.5).
//
// Deliberately small, and smaller than it looks. A retry re-runs the entire
// lead — search, fetches, every model call — and all of it is charged again.
// The LLM layer already retries individual calls with backoff, so reaching
// here means something outside a single model call failed, and three full
// re-runs of a lead is already an expensive way to find that out.
const MaxAttempts = 3

// Classify decides what an actor error means.
func Classify(err error) Class {
	if err == nil {
		return Degraded
	}

	switch {
	// Nothing will work. Continuing spends budget on calls that cannot
	// succeed, which is worse than stopping with a partial report.
	case errors.Is(err, llm.ErrUnauthorized),
		errors.Is(err, llm.ErrQuotaExceeded),
		errors.Is(err, search.ErrUnauthorized),
		errors.Is(err, search.ErrQuotaExceeded),
		errors.Is(err, budget.ErrInsufficientBudget):
		return Fatal

	// The same request might work shortly.
	case errors.Is(err, academic.ErrRateLimited),
		errors.Is(err, llm.ErrRateLimited),
		errors.Is(err, llm.ErrOverloaded),
		errors.Is(err, context.DeadlineExceeded):
		return Transient
	}

	// A search provider speaks for itself about whether a retry is worth it.
	var searchErr *search.APIError
	if errors.As(err, &searchErr) {
		if searchErr.Retryable() {
			return Transient
		}
		return Degraded
	}

	// Connection resets and DNS blips are transient by nature.
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return Transient
	}

	// llm.ErrContextTooLong is deliberately NOT transient: the same request
	// will be too long next time. It is a chunking failure for this lead, and
	// the digest should record it as one.
	return Degraded
}

// Backoff is the wait before attempt n (1-based), with jitter.
//
// Jitter matters more than the curve here. Without it, several leads that hit
// the same rate limit retry in lockstep and hit it again together, which is how
// a brief throttle becomes a sustained one.
func Backoff(attempt int, jitter func() float64) time.Duration {
	if attempt < 1 {
		attempt = 1
	}
	// Cap the shift before it happens: 1<<63 is negative, and a negative
	// duration slips past a `> 30s` check and returns immediately. Unreachable
	// through MaxAttempts, but Backoff is exported.
	shift := attempt - 1
	if shift > 5 {
		shift = 5
	}
	base := time.Duration(1<<uint(shift)) * time.Second
	if jitter == nil {
		return base
	}
	// Full jitter: uniform in [0, base]. Decorrelates retries better than
	// adding a small fraction, at the cost of sometimes retrying very soon.
	return time.Duration(jitter() * float64(base))
}

// DeadEndCause maps an error to the short label the digest groups by (§9.1).
//
// Grouped, so the planner sees "bot_block ×20" rather than twenty lines. The
// label has to be an enum-ish constant for that to work — an error string with
// a URL in it groups into a population of one.
func DeadEndCause(err error) string {
	switch {
	case err == nil:
		return "no_evidence"
	case errors.Is(err, llm.ErrContextTooLong):
		return "context_too_long"
	case errors.Is(err, academic.ErrRateLimited), errors.Is(err, llm.ErrRateLimited):
		return "rate_limited"
	case errors.Is(err, context.DeadlineExceeded):
		return "timeout"
	case errors.Is(err, context.Canceled):
		return "cancelled"
	}
	var searchErr *search.APIError
	if errors.As(err, &searchErr) {
		return "search_failed"
	}
	return "lead_failed"
}

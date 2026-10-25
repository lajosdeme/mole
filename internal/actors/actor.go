// Package actors executes leads.
//
// An Actor is the only component that touches raw source material. It takes one
// lead, does the work, and returns a summary plus claims — never the pages,
// never the raw text. That boundary is what keeps planner cost from compounding
// as research deepens (§2), and it is also where fabricated citations die
// (§11.5).
package actors

import (
	"context"

	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/dataset"
	"github.com/lajosdeme/mole/internal/llm"
)

// Result is what one lead produced.
type Result struct {
	// Summary is a one-line count of claims and sources, or why there were
	// none.
	Summary string

	// Claims are atomic facts, each carrying a source and a verbatim quote
	// that has already been checked against the text it came from.
	Claims []core.Claim

	// Rows are schema-shaped extractions, in dataset mode (M9, §13).
	//
	// Instead of Claims, not alongside them. The extraction is one model call per
	// chunk either way, and asking for both would double the cost of every chunk
	// to produce a report nobody asked for — dataset mode is an output MODE, so
	// the output it produces is the dataset.
	Rows []dataset.Row

	// Costs are the tool calls this run made, for the ledger to settle. Every
	// call is recorded whether or not it succeeded — the money was spent
	// either way, and a ledger of successes only cannot enforce a ceiling.
	Costs []core.ToolCall

	// Truncated reports that the sub-budget could not cover the whole
	// document. Surfaced rather than silently dropping content (§4.1).
	Truncated bool

	// Stats describe what happened, for the trace view and for M2's metrics.
	Stats RunStats
}

// RunStats is per-run accounting that is not money.
type RunStats struct {
	SearchResults int
	Fetched       int
	// SkippedFetch counts results whose content the search provider already
	// supplied. The whole reason for preferring a provider that returns page
	// text (§10.4), and worth measuring rather than assuming.
	SkippedFetch  int
	Chunks        int
	ChunksSkipped int

	// ChunksFailed counts chunks whose model call errored. A run where every
	// chunk failed still returns whatever it scraped together, so without this
	// the result looks thin rather than broken.
	ChunksFailed int

	// CacheHits counts sources served from the session cache instead of the
	// network (§9.3).
	CacheHits int

	ClaimsProposed int
	// ClaimsRejected counts claims discarded because their quote did not
	// appear in the source. A rising rate here is the signal that a model or
	// prompt has started fabricating.
	ClaimsRejected int

	// ValuesCoerced counts dataset values dropped because the schema's declared
	// type could not hold them (M9): a `number` field given "roughly $1.2m".
	//
	// Its own counter rather than folded into ClaimsRejected, because it means
	// something different and has a different fix. A rejected row means the model
	// invented; a coerced-away value usually means the SCHEMA is wrong for what
	// the sources write, and the user is the only one who can change that. Rows
	// that lost every non-key value still counted as accepted, so without this
	// the run reported a clean extraction and delivered empty cells.
	ValuesCoerced int
}

// Actor executes a lead.
type Actor interface {
	Run(ctx context.Context, lead core.Lead) (*Result, error)
	Type() core.ActorType
}

// Budget is what an actor may spend on one lead.
//
// The actor does not reserve or settle — the executor owns that. It receives a
// ceiling and reports what it used, so the reservation and the actual cost stay
// in one place.
type Budget struct {
	// MaxInputTokens caps the text fed to the model across the whole run.
	MaxInputTokens int64
	// MaxSources caps how many search results are read.
	MaxSources int
	// MaxClaimsPerSource bounds a single page's contribution, so one verbose
	// document cannot dominate the graph.
	MaxClaimsPerSource int

	// AlwaysFetch ignores content the search provider supplied and fetches the
	// page itself.
	//
	// Slower and costlier, and off by default for exactly that reason. But
	// provider-supplied text is a measurement blind spot: nothing was fetched,
	// so §10.4 has no denominator and §17.1's gate reads "no data", and
	// citation accuracy cannot be checked because re-reading the page runs a
	// different extractor than the one that produced the text. An eval corpus
	// run on a content-supplying provider silently collects none of it.
	AlwaysFetch bool

	// MaxChunkTokens caps a SINGLE request, which is a different limit from
	// MaxInputTokens and binds first.
	//
	// The default chunk targets ~8k tokens, which fits any current context
	// window — but a context window is not the only ceiling. A provider's
	// per-minute token allowance can be far smaller (Groq's free tier is 6k
	// TPM), and a request over it fails with 413 no matter how much budget is
	// left. Sized wrong, every chunk of every source fails and the run reports
	// success having read nothing.
	MaxChunkTokens int64
}

// ChunkOptions derives the split from the per-request cap.
func (b Budget) ChunkOptions() llm.ChunkOptions {
	o := llm.DefaultChunkOptions()
	if b.MaxChunkTokens <= 0 {
		return o
	}
	// Leave room for the prompt itself and for the estimator running short:
	// EstimateTokens is deliberately rough, and overshooting here is a hard
	// failure rather than a slightly larger bill.
	usable := b.MaxChunkTokens * 2 / 3
	chars := int(usable * 3)
	if chars < o.MinChars*2 {
		chars = o.MinChars * 2
	}
	if chars < o.MaxChars {
		o.MaxChars = chars
		if o.OverlapChars >= o.MaxChars {
			o.OverlapChars = o.MaxChars / 8
		}
	}
	return o
}

func (b Budget) withDefaults() Budget {
	if b.MaxInputTokens <= 0 {
		b.MaxInputTokens = 60_000
	}
	if b.MaxSources <= 0 {
		b.MaxSources = 5
	}
	if b.MaxClaimsPerSource <= 0 {
		b.MaxClaimsPerSource = 8
	}
	return b
}

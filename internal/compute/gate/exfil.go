package gate

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// The exfil check (§14.3: "Assert no row-level data crosses the aggregation
// gate (§12.1)").
//
// An invariant rather than the metric §14.3 describes: the accumulator still
// holds every value it read, so an envelope is checked against its own source
// before being returned, and one carrying a value it should not is withheld.
//
// The rule: a text value may appear only as a bucket key that survived the
// k-anonymity floor, or a range bound the gate chose to report. Numeric values
// are exempt — an aggregate IS a number, and excluding them would make the
// check report the answer as the leak.
//
// No input currently reaches it: the structural rules refuse or withhold first,
// every time, so only positive controls cover it. It is a backstop for a rule
// not yet written, and the day it fires is the day something else broke.
//
// It cannot catch a column that should have been withheld and was not. If the
// gate believes a column holds categories, it believes its values are
// reportable, and nothing self-referential can say otherwise — that case
// belongs to the profiler, which is why connector.IsFreeText has one
// implementation.

// ErrLeak is returned when an envelope carries a value it should not.
//
// Deliberately not an ErrRefused: this is a bug in this package, not a rejected
// query, and a caller retrying with a different statement would be responding
// to the wrong thing.
var ErrLeak = errors.New("gate: envelope carries row-level data")

// leakMinLen is the shortest value the check will look for.
//
// Substring matching cannot tell a short value from a coincidence — a note
// column holding "n/a" matches the letter sequence in half the envelope's own
// field names. Shorter values are left to the structural rules, which do not
// depend on recognising them: a withheld column carries no range and
// contributes no buckets whatever its contents look like.
const leakMinLen = 8

// verifyNoLeak reports whether env carries a value it was not entitled to.
//
// described is the set of rows the envelope was allowed to summarize, and the
// permitted values are derived from IT rather than from the envelope. That
// distinction is the whole check: an earlier version read the allowed set out
// of the envelope's own bucket keys and range bounds, which meant a value that
// leaked into either of them authorised itself and the check reported nothing.
// Both of its positive controls failed, which is what positive controls are
// for.
func (a *accumulator) verifyNoLeak(env AggregateEnvelope, described []int) error {
	allowed := map[string]bool{}
	for _, r := range described {
		if r >= len(a.rows) {
			continue
		}
		for i := range a.names {
			// A grouping key of a bucket that survived the floor describes at
			// least KFloor records, so it may be named. Nothing else may.
			if a.shape.grouped && i < len(a.shape.isKey) && a.shape.isKey[i] && !a.freeTextKey[i] {
				allowed[a.rows[r][i].text] = true
			}
		}
	}

	needles := map[string]int{} // value -> column index
	for _, row := range a.rows {
		for i, c := range row {
			if c.null || c.isNum || len(c.text) < leakMinLen || allowed[c.text] {
				continue
			}
			needles[c.text] = i
		}
	}
	if len(needles) == 0 {
		return nil
	}

	// The whole envelope, not the fields currently known to hold text. A field
	// added later is exactly the case worth catching, and enumerating fields
	// here would mean the check quietly stops covering the envelope the moment
	// somebody extends it.
	//
	// Query is excluded: it is the statement mole rendered rather than
	// something read out of the data, and a template filtering on a literal
	// would otherwise report itself as a leak.
	scanned := env
	scanned.Query = ""
	raw, err := json.Marshal(scanned)
	if err != nil {
		return fmt.Errorf("gate: exfil check: %w", err)
	}
	hay := string(raw)

	for needle, col := range needles {
		// Compared in encoded form, so a value carrying a quote or a newline is
		// still found after JSON escaped it.
		encoded, err := json.Marshal(needle)
		if err != nil {
			continue
		}
		if strings.Contains(hay, strings.Trim(string(encoded), `"`)) {
			// The value itself is not in the message. An error about data
			// escaping should not be another copy of it.
			name := "(unknown)"
			if col < len(a.names) {
				name = a.names[col]
			}
			return fmt.Errorf("%w: a value from column %q reached the envelope", ErrLeak, name)
		}
	}
	return nil
}

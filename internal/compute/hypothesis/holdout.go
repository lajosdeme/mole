package hypothesis

import (
	"fmt"

	"github.com/lajosdeme/mole/internal/compute/connector"
	"github.com/lajosdeme/mole/internal/compute/stats"
)

// The holdout statement (§4's stability column).
//
// Not a Template, and deliberately not one: Templates is "the whole vocabulary" a
// MODEL may choose from, and this is not a question anybody asks — it is the same
// question asked again on disjoint thirds of the table. Putting it in the list
// would offer a model a template whose output is meaningless on its own, and
// §12.3's rule is about what the model may select, not about what mole may run.
//
// It renders through the same slot validation as its parent, so the columns are
// still looked up in the profile rather than escaped into a statement.

// HoldoutColumn is the alias the window index arrives under.
//
// `holdout`, not `window`: WINDOW is a keyword in SQLite and a result column
// called it parses as the start of a window clause.
const HoldoutColumn = "holdout"

// RenderHoldout renders the group comparison split into deterministic windows.
//
// The split is `rowid % n`, for two reasons.
//
// Deterministic: the same table gives the same windows every run, so a
// stability result is reproducible and a replayed session sees the same
// evidence. A random split would make §4's fourth column a coin toss.
//
// Interleaved rather than sliced: contiguous thirds of a file sorted by date
// turn the stability check into a comparison of three time periods, which fails
// for a seasonal measure that is perfectly stable. Every third row spans the
// whole table.
//
// It is not independent of insertion order — a file whose rows repeat with
// period three puts one phase in every window. The alternative, hashing the
// row's values, needs a function the sqlguard allowlist does not include, for a
// threat that is a pathological export rather than an adversary.
func RenderHoldout(c connector.Connector, p Plan, windows int) (string, error) {
	if p.Template != KindGroupCompare {
		return "", invalid("the holdout statement applies to %s, not %q",
			KindGroupCompare, p.Template)
	}
	if windows < 2 {
		return "", invalid("a holdout split needs at least two windows, not %d", windows)
	}
	cols, table, err := resolve(c, p)
	if err != nil {
		return "", err
	}

	centre := fmt.Sprintf("(SELECT AVG(%s) FROM %s)", cols["measure"], table)
	return fmt.Sprintf(
		"SELECT %[1]s AS bucket, (rowid %% %[9]d) AS %[10]s, COUNT(*) AS n, "+
			"COUNT(%[2]s) AS %[6]s, AVG(%[2]s) AS mean, "+
			"MIN(%[7]s) AS %[8]s, "+
			"SUM(%[2]s - %[7]s) AS %[4]s, "+
			"SUM((%[2]s - %[7]s) * (%[2]s - %[7]s)) AS %[5]s "+
			"FROM %[3]s GROUP BY 1, 2 ORDER BY n DESC",
		cols["key"], cols["measure"], table,
		stats.SumColumn, stats.SumSqColumn, stats.CountColumn,
		centre, stats.OffsetColumn, windows, HoldoutColumn), nil
}

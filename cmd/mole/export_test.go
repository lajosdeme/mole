package main

import (
	"strings"
	"testing"
)

// TestExportRefusesAnUnknownSession, rather than emitting an envelope describing
// nothing. A consumer parsing a well-formed export of a session that does not
// exist would render an empty pack and bill for it.
func TestExportRefusesAnUnknownSession(t *testing.T) {
	out, err := exec(t, "export", "s_06NOSUCHSESSION")
	if err == nil {
		t.Fatalf("exporting an unknown session succeeded: %s", out)
	}
}

// TestExportIsInTheVisibleCommandSet. It is the contract a paying consumer reads,
// not an evaluation tool, so it belongs beside research and trace rather than in
// the hidden group with eval and pairs.
func TestExportIsInTheVisibleCommandSet(t *testing.T) {
	out, err := exec(t, "--help")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "export") {
		t.Errorf("export is missing from the help screen:\n%s", out)
	}
}

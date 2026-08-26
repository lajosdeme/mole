package session

import (
	"testing"

	"github.com/lajosdeme/mole/internal/verifier"
)

// TestASessionConfirmsUnlessToldNotTo. The Verifier's zero value confirms nothing, so
// without this the default session would silently stop asking twice.
func TestASessionConfirmsUnlessToldNotTo(t *testing.T) {
	if got := (&Runner{}).confirmMode(); got != verifier.ConfirmBoth {
		t.Errorf("default session confirm mode = %q, want %q", got, verifier.ConfirmBoth)
	}
	for _, want := range []verifier.ConfirmMode{
		verifier.ConfirmDuplicates, verifier.ConfirmNone, verifier.ConfirmBoth,
	} {
		if got := (&Runner{Confirm: want}).confirmMode(); got != want {
			t.Errorf("confirmMode() = %q, want %q", got, want)
		}
	}
}

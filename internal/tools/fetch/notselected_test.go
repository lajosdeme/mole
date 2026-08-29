package fetch

import "testing"

// TestNotSelectedIsNotAnAttemptedFetch.
//
// Attempted() is the denominator for every per-cause rate, so a result the actor
// passed over without asking for it would drag js_required and bot_block down by
// however much of the result list the source cap happened to cut — the same
// padding provider_content was separated out to avoid.
func TestNotSelectedIsNotAnAttemptedFetch(t *testing.T) {
	if OutcomeNotSelected.Attempted() {
		t.Error("a result nobody fetched counts toward the fetch denominator")
	}
	if OutcomeNotSelected.Usable() {
		t.Error("an unread result is reported as usable text")
	}
	if OutcomeNotSelected.CapabilityGap() {
		t.Error("not reading a result is evidence for a headless browser")
	}
	if OutcomeNotSelected.SystemWorking() {
		t.Error("a source cap is not a refusal; it is a budget")
	}
}

// TestTheOnlyUnattemptedOutcomesAreTheTwoThatNeverAskedForAPage. A new outcome
// added without a decision about the denominator defaults to being counted, and a
// rate that silently changes meaning is worse than one that is wrong loudly.
func TestTheOnlyUnattemptedOutcomesAreTheTwoThatNeverAskedForAPage(t *testing.T) {
	all := []Outcome{
		OutcomeOK, OutcomeProviderContent, OutcomeJSRequired, OutcomeStructuredOnly,
		OutcomeConsentWall, OutcomeBotBlock, OutcomePaywall, OutcomeNotFound,
		OutcomeTimeout, OutcomeRobotsDenied, OutcomeGuardDenied, OutcomeExtractFailed,
		OutcomeServerError, OutcomeTooLarge, OutcomeUnsupported, OutcomeNetworkError,
		OutcomeNotSelected,
	}
	var unattempted []Outcome
	for _, o := range all {
		if !o.Attempted() {
			unattempted = append(unattempted, o)
		}
	}
	if len(unattempted) != 2 {
		t.Errorf("unattempted outcomes are %v, want exactly provider_content and not_selected",
			unattempted)
	}
}

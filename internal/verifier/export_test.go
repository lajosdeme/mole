package verifier

// EstimateBatch exposes estimateBatch to the external test package, so a test
// can size a budget in whole batch reservations rather than hard-coding the
// estimate's arithmetic.
var EstimateBatch = (*Verifier).estimateBatch

package config_test

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"time"

	"github.com/lajosdeme/mole/internal/config"
)

// isolate points config at a temp dir so tests never touch a real one.
func isolate(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("MOLE_CONFIG_DIR", dir)
	// Clear anything the developer's shell might be exporting, or the env
	// overrides would leak into assertions about file contents.
	for _, k := range []string{
		"MOLE_SEARCH_PROVIDER", "MOLE_BRAVE_API_KEY", "BRAVE_API_KEY",
		"MOLE_TAVILY_API_KEY", "TAVILY_API_KEY", "MOLE_LLM_API_KEY",
		"ANTHROPIC_API_KEY", "MOLE_LLM_BASE_URL", "MOLE_LLM_MODEL",
		"MOLE_CONTACT_EMAIL",
	} {
		t.Setenv(k, "")
	}
	return dir
}

func TestLoadWithoutConfigIsNotAHardError(t *testing.T) {
	isolate(t)
	cfg, err := config.Load()
	if !errors.Is(err, config.ErrNotConfigured) {
		t.Fatalf("error = %v, want ErrNotConfigured", err)
	}
	if cfg == nil {
		t.Fatal("Load returned nil config; callers need a usable zero value")
	}
}

func TestSetSaveRoundTrip(t *testing.T) {
	isolate(t)

	cfg := &config.Config{}
	for _, kv := range [][2]string{
		{"search.provider", "tavily"},
		{"search.tavily-key", "tvly-secret-value"},
		{"search.brave-key", "brave-secret-value"},
		{"llm.provider", "anthropic"},
		{"llm.api-key", "sk-ant-secret"},
		{"llm.model", "claude-opus-5"},
		{"contact-email", "researcher@example.com"},
	} {
		if err := cfg.Set(kv[0], kv[1]); err != nil {
			t.Fatalf("set %s: %v", kv[0], err)
		}
	}
	if err := cfg.Save(); err != nil {
		t.Fatalf("save: %v", err)
	}

	loaded, err := config.Load()
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if loaded.Search.Provider != "tavily" {
		t.Errorf("provider = %q", loaded.Search.Provider)
	}
	if loaded.Search.ActiveKey() != "tvly-secret-value" {
		t.Errorf("active key = %q, want the Tavily key", loaded.Search.ActiveKey())
	}

	// Both keys persist, so switching providers does not mean re-entering a
	// key that was already supplied.
	if err := loaded.Set("search.provider", "brave"); err != nil {
		t.Fatal(err)
	}
	if loaded.Search.ActiveKey() != "brave-secret-value" {
		t.Errorf("after switch, active key = %q", loaded.Search.ActiveKey())
	}
}

// TestConfigFileIsOwnerOnly: the file holds API keys and nothing else protects
// them until the keyring lands with the daemon work.
func TestConfigFileIsOwnerOnly(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("permissions are not meaningful on windows")
	}
	dir := isolate(t)

	cfg := &config.Config{}
	if err := cfg.Set("llm.api-key", "sk-secret"); err != nil {
		t.Fatal(err)
	}
	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}

	info, err := os.Stat(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if mode := info.Mode().Perm(); mode != 0o600 {
		t.Errorf("config mode = %04o, want 0600", mode)
	}

	ok, detail := config.CheckPermissions()
	if !ok {
		t.Errorf("CheckPermissions failed on a freshly written file: %s", detail)
	}

	// A loosened file is reported rather than silently tolerated.
	if err := os.Chmod(filepath.Join(dir, "config.json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if ok, _ := config.CheckPermissions(); ok {
		t.Error("world-readable config passed the permission check")
	}
}

// TestSaveOverLooseFileTightensIt covers the upgrade path: a file created by an
// earlier version, or by a user's editor, must not stay world-readable.
func TestSaveOverLooseFileTightensIt(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("permissions are not meaningful on windows")
	}
	dir := isolate(t)
	path := filepath.Join(dir, "config.json")

	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg := &config.Config{}
	_ = cfg.Set("llm.api-key", "sk-secret")
	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}

	info, _ := os.Stat(path)
	if mode := info.Mode().Perm(); mode != 0o600 {
		t.Errorf("mode after save = %04o, want 0600", mode)
	}
}

func TestSecretsAreMaskedForDisplay(t *testing.T) {
	isolate(t)

	cfg := &config.Config{}
	_ = cfg.Set("llm.api-key", "sk-ant-api03-verylongsecretvalue")
	_ = cfg.Set("llm.model", "claude-opus-5")

	var sawSecret, sawPlain bool
	for _, f := range config.Fields() {
		shown := config.Display(f, cfg)
		if f.Name == "llm.api-key" {
			sawSecret = true
			if strings.Contains(shown, "verylongsecret") {
				t.Errorf("secret leaked into display: %q", shown)
			}
			if !strings.Contains(shown, "*") {
				t.Errorf("secret not masked: %q", shown)
			}
		}
		if f.Name == "llm.model" {
			sawPlain = true
			if shown != "claude-opus-5" {
				t.Errorf("non-secret was mangled: %q", shown)
			}
		}
	}
	if !sawSecret || !sawPlain {
		t.Fatal("field table is missing expected keys")
	}
}

func TestMaskNeverRevealsShortSecrets(t *testing.T) {
	for _, s := range []string{"", "a", "abc", "12345678"} {
		if got := config.Mask(s); strings.Contains(got, s) && s != "" {
			t.Errorf("Mask(%q) = %q leaks the value", s, got)
		}
	}
	long := config.Mask("sk-ant-1234567890abcdef")
	if !strings.HasPrefix(long, "sk-a") {
		t.Errorf("Mask lost the identifying prefix: %q", long)
	}
}

func TestValidationRejectsBadValues(t *testing.T) {
	isolate(t)
	cfg := &config.Config{}

	if err := cfg.Set("search.provider", "google"); err == nil {
		t.Error("unknown search provider accepted")
	}
	if err := cfg.Set("llm.provider", "hotdog"); err == nil {
		t.Error("unknown llm provider accepted")
	}
	if err := cfg.Set("contact-email", "not-an-address"); err == nil {
		t.Error("malformed contact email accepted")
	}
	if err := cfg.Set("nonexistent.key", "x"); err == nil {
		t.Error("unknown key accepted")
	}

	// Provider names are normalized rather than rejected on case.
	if err := cfg.Set("search.provider", "  BRAVE "); err != nil {
		t.Errorf("case/whitespace variant rejected: %v", err)
	}
	if cfg.Search.Provider != "brave" {
		t.Errorf("provider = %q, want normalized to brave", cfg.Search.Provider)
	}
}

// TestEnvOverridesFile: containers and CI supply secrets through the
// environment, and a live env var should win over a stale file.
func TestEnvOverridesFile(t *testing.T) {
	isolate(t)

	cfg := &config.Config{}
	_ = cfg.Set("search.provider", "brave")
	_ = cfg.Set("search.brave-key", "from-file")
	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}

	t.Setenv("BRAVE_API_KEY", "from-env")

	loaded, err := config.Load()
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Search.BraveKey != "from-env" {
		t.Errorf("key = %q, want the environment value", loaded.Search.BraveKey)
	}
}

func TestEnvAloneIsEnoughToBeConfigured(t *testing.T) {
	isolate(t)
	t.Setenv("MOLE_SEARCH_PROVIDER", "tavily")
	t.Setenv("TAVILY_API_KEY", "tvly-env")

	cfg, err := config.Load()
	if err != nil {
		t.Fatalf("env-only config reported an error: %v", err)
	}
	if cfg.Search.ActiveKey() != "tvly-env" {
		t.Errorf("active key = %q", cfg.Search.ActiveKey())
	}
}

// TestTheSlowModelKnobsRoundTrip.
//
// Both exist because a live run needed them and neither was reachable: a 12B model on an
// integrated GPU, adjudicating eight claim pairs in one call after ollama reloaded 8.4GB
// of weights, exceeded the hardcoded ten-minute ceiling and the whole batch was skipped.
// A ceiling that cannot be raised turns a slow model into a broken one.
func TestTheSlowModelKnobsRoundTrip(t *testing.T) {
	isolate(t)

	cfg, err := config.Load()
	if err != nil && !errors.Is(err, config.ErrNotConfigured) {
		t.Fatal(err)
	}
	for _, tc := range []struct{ key, val string }{
		{"llm.timeout", "25m"},
		{"llm.verifier-batch-size", "3"},
	} {
		if err := cfg.Set(tc.key, tc.val); err != nil {
			t.Fatalf("set %s: %v", tc.key, err)
		}
	}
	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}
	cfg, err = config.Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.LLM.Timeout != 25*time.Minute {
		t.Errorf("timeout = %v, want 25m", cfg.LLM.Timeout)
	}
	if cfg.LLM.VerifierBatchSize != 3 {
		t.Errorf("batch size = %d, want 3", cfg.LLM.VerifierBatchSize)
	}

	// Both clear back to "use the default", or a value set once can never be undone.
	for _, key := range []string{"llm.timeout", "llm.verifier-batch-size"} {
		if err := cfg.Set(key, ""); err != nil {
			t.Fatalf("clear %s: %v", key, err)
		}
	}
	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}
	cfg, err = config.Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.LLM.Timeout != 0 || cfg.LLM.VerifierBatchSize != 0 {
		t.Errorf("not cleared: timeout=%v batch=%d", cfg.LLM.Timeout, cfg.LLM.VerifierBatchSize)
	}

	// Nonsense is refused rather than silently becoming zero, which would read as
	// "default" and hide the typo.
	for _, tc := range []struct{ key, val string }{
		{"llm.timeout", "twenty minutes"},
		{"llm.timeout", "-5m"},
		{"llm.verifier-batch-size", "0"},
		{"llm.verifier-batch-size", "-2"},
		{"llm.verifier-batch-size", "lots"},
	} {
		if err := cfg.Set(tc.key, tc.val); err == nil {
			t.Errorf("%s=%q was accepted", tc.key, tc.val)
		}
	}
}

// TestADeclaredRateIsCheckedAgainstTheFileItIsIn.
//
// Validated at load, while the file is still the thing being talked about, so a
// typo names the file rather than surfacing much later as a model that
// mysteriously bills nothing.
func TestADeclaredRateIsCheckedAgainstTheFileItIsIn(t *testing.T) {
	dir := isolate(t)
	write := func(t *testing.T, body string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}

	write(t, `{"search":{"provider":"tavily"},"pricing":{"some/model":{"input":-1,"output":2}}}`)
	_, err := config.Load()
	if err == nil {
		t.Fatal("a negative rate loaded without complaint")
	}
	if !strings.Contains(err.Error(), "config.json") {
		t.Errorf("the error does not name the file it is in: %v", err)
	}

	// And a good one loads, in the units a vendor publishes.
	write(t, `{"search":{"provider":"tavily"},"pricing":{"some/model":{"input":2.5,"output":10}}}`)
	cfg, err := config.Load()
	if err != nil {
		t.Fatalf("a valid rate was refused: %v", err)
	}
	if got := cfg.Pricing["some/model"].Input; got != 2.5 {
		t.Errorf("input rate = %v, want 2.5 dollars per MTok", got)
	}
}

// Package config stores machine-local settings and provider credentials.
//
// Secrets live here and nowhere else. §5.2 is explicit that keys must not
// appear in .mcp.json — that file gets committed, and a key in it is a leaked
// key. Everything Mole needs to authenticate is read from this file at
// daemon start.
//
// Format is JSON rather than the TOML the sketch mentions, to keep the
// dependency count at one. The file is written by `mole config set` rather
// than hand-edited, so the ergonomic difference is small; revisit if users end
// up editing it directly.
package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/lajosdeme/mole/internal/core"
	"time"

	"github.com/lajosdeme/mole/internal/pricing"
)

// Config is the on-disk settings file.
// Duration reads "20m" or a count of nanoseconds, and writes the readable one.
// A plain time.Duration accepts only the latter, so such a field is settable
// through `mole config set` and not by editing the file.
//
// llm.timeout predates this and still stores nanoseconds: changing it would
// make files written by a new binary unreadable by an old one.
type Duration time.Duration

func (d Duration) Duration() time.Duration { return time.Duration(d) }

func (d Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(time.Duration(d).String())
}

func (d *Duration) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		parsed, err := time.ParseDuration(s)
		if err != nil {
			return fmt.Errorf("not a duration (try 20m): %w", err)
		}
		*d = Duration(parsed)
		return nil
	}
	var n int64
	if err := json.Unmarshal(b, &n); err != nil {
		return fmt.Errorf("not a duration: %s", b)
	}
	*d = Duration(n)
	return nil
}

type Config struct {
	// Search selects and authenticates the search backend.
	Search SearchConfig `json:"search"`
	// LLM selects and authenticates the model provider.
	LLM LLMConfig `json:"llm"`

	// MaxWallClock is how long a session may run when --timeout says nothing.
	// The SESSION's clock, not a model call's — llm.timeout is that one.
	//
	// A default, not a ceiling: --timeout overrides it. The ceiling a caller
	// cannot raise is the daemon's, in mcpserver.
	MaxWallClock Duration `json:"max_wallclock,omitempty"`

	// Pricing declares model rates, in dollars per MILLION tokens. mole ships
	// rates only for the models it can select itself; declare anything else
	// here, and a declared rate wins over a built-in one.
	//
	//	"pricing": {
	//	  "gpt-4o":        {"input": 2.50, "output": 10.00},
	//	  "some/model-v3": {"input": 1.00, "output": 3.00, "cache_read": 0.50}
	//	}
	Pricing map[string]pricing.RateConfig `json:"pricing,omitempty"`

	// ContactEmail is sent to academic providers that require identification
	// (Unpaywall requires it; NCBI requires tool+email). Checked at startup
	// rather than documented in a README, so a missing value is caught before
	// it becomes a ban.
	ContactEmail string `json:"contact_email,omitempty"`

	// MaxSessionUSD caps what one session started over MCP may be given, in
	// micro-dollars. Zero means no ceiling.
	//
	// On the command line the budget is a number a person typed. Over MCP it is a
	// number an agent chose, and nothing else bounds it — a coding agent that
	// misreads its own instructions can ask for a thousand dollars as easily as
	// three. The ceiling is the daemon's, not the caller's, which is the point:
	// it is the one limit the caller cannot raise.
	MaxSessionUSD int64 `json:"max_session_usd,omitempty"`

	// MaxSessionTokens caps a token-unit session started over MCP. Zero means no
	// ceiling.
	//
	// Separate from MaxSessionUSD because the two units are not convertible
	// without knowing which model will run: §8 makes the unit load-bearing, and a
	// guessed exchange rate would be a ceiling nobody could reason about. Having
	// only one of the two set is treated as a misconfiguration rather than an
	// open door — see mcpserver.checkCeiling.
	MaxSessionTokens int64 `json:"max_session_tokens,omitempty"`

	// Budget defaults applied when a session does not specify.
	DefaultBudgetUnit string `json:"default_budget_unit,omitempty"`
	DefaultBudgetUSD  string `json:"default_budget_usd,omitempty"`
	DefaultBudgetToks int64  `json:"default_budget_tokens,omitempty"`
}

type SearchConfig struct {
	// Provider is "brave" or "tavily".
	Provider string `json:"provider,omitempty"`
	// BraveKey and TavilyKey are stored separately so switching providers does
	// not require re-entering a key you already gave.
	BraveKey  string `json:"brave_key,omitempty"`
	TavilyKey string `json:"tavily_key,omitempty"`
	// CostPerQueryMicros overrides the default price for the active provider,
	// for users on a plan that differs from the published entry tier.
	CostPerQueryMicros int64 `json:"cost_per_query_micros,omitempty"`
}

// ActiveKey returns the key for the selected provider.
func (s SearchConfig) ActiveKey() string {
	switch s.Provider {
	case "brave":
		return s.BraveKey
	case "tavily":
		return s.TavilyKey
	}
	return ""
}

type LLMConfig struct {
	// Provider is "anthropic" or "openai-compatible".
	Provider string `json:"provider,omitempty"`
	APIKey   string `json:"api_key,omitempty"`
	// BaseURL points at a self-hosted or proxied endpoint.
	BaseURL string `json:"base_url,omitempty"`
	// Model is the default model id.
	Model string `json:"model,omitempty"`
	// CheapModel handles chunk mining and extraction, where the strong model
	// is not worth its price (§10.1).
	CheapModel string `json:"cheap_model,omitempty"`

	// Timeout bounds a single model call. Zero takes the provider default.
	//
	// Ten minutes was hardcoded, which is generous for an API and reachable for a local
	// one: a 12B model on an integrated GPU, adjudicating eight claim pairs in one call
	// after ollama has just reloaded 8.4GB of weights, exceeded it and the whole batch
	// was skipped. A ceiling that cannot be raised turns a slow model into a broken one.
	Timeout time.Duration `json:"timeout,omitempty"`

	// VerifierBatchSize is how many claim pairs go into one adjudication call. Zero takes
	// the Verifier's default.
	//
	// The other half of the same problem. Batching is what makes verification affordable
	// — one call per pair costs a third of a session — but a batch is also the unit that
	// has to finish inside Timeout, and a slow model wants a smaller one.
	VerifierBatchSize int `json:"verifier_batch_size,omitempty"`

	// VerifierModel judges claim pairs and grounding (§11). Empty uses CheapModel.
	//
	// A third setting rather than a third tier, because the argument for it is
	// measured rather than architectural. On a live 25-claim run, adjudication made 7
	// calls against 31 for chunk mining — so a model too expensive to mine with can be
	// affordable to judge with, and the two workloads are not alike: mining is
	// extraction, judging is deciding whether two sentences can both be true.
	//
	// It is also the stage whose errors corrupt everything downstream. That same run
	// produced 16 contradictions out of 37 edges, and the model's own rationales
	// described the pairs as being about different topics — which is "unrelated". Every
	// false positive spent a follow-up lead, docked two claims' confidence, and put a
	// disagreement in the report that was not there.
	VerifierModel string `json:"verifier_model,omitempty"`

	// RecheckModel re-judges contradiction edges before the report. Empty disables it.
	RecheckModel string `json:"recheck_model,omitempty"`

	// MaxInputTokens caps a SINGLE request's input.
	//
	// Not the same limit as a context window, and often much smaller. A
	// provider's per-minute token allowance rejects an oversized request with
	// 413 no matter how much budget is left — a Groq free tier is 6000 TPM
	// against a default chunk of roughly 8000. Left unset, chunking targets a
	// context window and every chunk fails.
	MaxInputTokens int64 `json:"max_input_tokens,omitempty"`

	// Strong, Cheap and Verifier override the settings above for one role each.
	Strong   *LLMRole `json:"strong,omitempty"`
	Cheap    *LLMRole `json:"cheap,omitempty"`
	Verifier *LLMRole `json:"verifier,omitempty"`
}

// LLMRole is one role's provider and model; empty fields inherit the top level,
// except that a role on another provider inherits nothing and another endpoint no key.
type LLMRole struct {
	Provider string `json:"provider,omitempty"`
	APIKey   string `json:"api_key,omitempty"`
	BaseURL  string `json:"base_url,omitempty"`
	Model    string `json:"model,omitempty"`
}

// RoleNames lists the roles that accept an llm.<role>.* block.
var RoleNames = []string{"strong", "cheap", "verifier"}

// block returns the address of a role's override pointer.
func (l *LLMConfig) block(role string) **LLMRole {
	switch role {
	case "strong":
		return &l.Strong
	case "cheap":
		return &l.Cheap
	case "verifier":
		return &l.Verifier
	}
	return nil
}

// Resolve returns a role's settings: the top level overlaid with its block.
func (l LLMConfig) Resolve(role string) LLMRole {
	r := LLMRole{Provider: l.Provider, APIKey: l.APIKey, BaseURL: l.BaseURL}
	switch role {
	case "strong":
		r.Model = l.Model
	case "cheap":
		r.Model = l.CheapModel
	case "verifier":
		r.Model = l.VerifierModel
	}
	if p := l.block(role); p != nil && *p != nil {
		b := **p
		// A role on another provider takes nothing from the top level.
		if b.Provider != "" && providerKind(b.Provider) != providerKind(l.Provider) {
			return b
		}
		r.Provider = firstNonEmpty(b.Provider, r.Provider)
		r.BaseURL = firstNonEmpty(b.BaseURL, r.BaseURL)
		r.Model = firstNonEmpty(b.Model, r.Model)
		// A role on another provider or endpoint uses only its own key, never the top-level one.
		if providerKind(r.Provider) != providerKind(l.Provider) || r.BaseURL != l.BaseURL {
			r.APIKey = b.APIKey
		} else {
			r.APIKey = firstNonEmpty(b.APIKey, r.APIKey)
		}
	}
	return r
}

// providerKind returns the provider name with the anthropic default applied.
func providerKind(p string) string {
	if p == "" {
		return "anthropic"
	}
	return p
}

// Block returns a role's override block, or nil when it has none.
func (l LLMConfig) Block(role string) *LLMRole {
	if p := l.block(role); p != nil {
		return *p
	}
	return nil
}

// HasRoleBlocks reports whether any role overrides the top level.
func (l LLMConfig) HasRoleBlocks() bool {
	return l.Strong != nil || l.Cheap != nil || l.Verifier != nil
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

// Dir is the config directory, honouring XDG.
func Dir() string {
	if p := os.Getenv("MOLE_CONFIG_DIR"); p != "" {
		return p
	}
	base := os.Getenv("XDG_CONFIG_HOME")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return ".mole"
		}
		base = filepath.Join(home, ".config")
	}
	return filepath.Join(base, "mole")
}

// Path is the config file location.
func Path() string { return filepath.Join(Dir(), "config.json") }

// ---------------------------------------------------------------------------
// Load / Save
// ---------------------------------------------------------------------------

// ErrNotConfigured is returned by Load when no config file exists.
var ErrNotConfigured = errors.New("config: not configured (run: mole init)")

// Load reads the config, applying environment overrides.
func Load() (*Config, error) {
	c := &Config{}

	b, err := os.ReadFile(Path())
	switch {
	case os.IsNotExist(err):
		// Not fatal on its own: the environment may carry everything needed,
		// which is how a container deployment supplies credentials.
		c.applyEnv()
		if c.Search.Provider == "" && c.LLM.APIKey == "" {
			return c, ErrNotConfigured
		}
		return c, nil
	case err != nil:
		return nil, fmt.Errorf("config: read %s: %w", Path(), err)
	}

	if err := json.Unmarshal(b, c); err != nil {
		return nil, fmt.Errorf("config: parse %s: %w", Path(), err)
	}
	// Declared rates are checked HERE, while the file is still the thing being
	// talked about, so a typo names the file it is in rather than surfacing
	// later as a model that mysteriously bills nothing.
	if err := pricing.NewTable().Declare(c.Pricing); err != nil {
		return nil, fmt.Errorf("config: %s: %w", Path(), err)
	}
	c.applyEnv()
	return c, nil
}

// applyEnv lets environment variables override the file. Containers and CI
// supply secrets this way, and an env var should win over a stale file.
func (c *Config) applyEnv() {
	setIf := func(dst *string, keys ...string) {
		for _, k := range keys {
			if v := os.Getenv(k); v != "" {
				*dst = v
				return
			}
		}
	}
	setIf(&c.Search.Provider, "MOLE_SEARCH_PROVIDER")
	setIf(&c.Search.BraveKey, "MOLE_BRAVE_API_KEY", "BRAVE_API_KEY")
	setIf(&c.Search.TavilyKey, "MOLE_TAVILY_API_KEY", "TAVILY_API_KEY")
	setIf(&c.LLM.Provider, "MOLE_LLM_PROVIDER")
	// ANTHROPIC_API_KEY fills the top-level key only for an anthropic or unset provider.
	llmKeys := []string{"MOLE_LLM_API_KEY"}
	if providerKind(c.LLM.Provider) == "anthropic" {
		llmKeys = append(llmKeys, "ANTHROPIC_API_KEY")
	}
	setIf(&c.LLM.APIKey, llmKeys...)
	setIf(&c.LLM.BaseURL, "MOLE_LLM_BASE_URL")
	setIf(&c.LLM.Model, "MOLE_LLM_MODEL")
	setIf(&c.LLM.CheapModel, "MOLE_LLM_CHEAP_MODEL")
	setIf(&c.LLM.VerifierModel, "MOLE_LLM_VERIFIER_MODEL")
	setIf(&c.LLM.RecheckModel, "MOLE_LLM_RECHECK_MODEL")
	setIf(&c.ContactEmail, "MOLE_CONTACT_EMAIL")
}

// Save writes the config with owner-only permissions.
//
// The file holds API keys, so the mode is 0600 and the directory 0700. An OS
// keyring is the better home for these and is planned with the daemon work in
// M7; until then the permissions are the whole protection, which is why they
// are set explicitly on every write rather than left to the umask.
func (c *Config) Save() error {
	dir := Dir()
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("config: create %s: %w", dir, err)
	}

	b, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')

	// Write-then-rename so an interrupted save cannot truncate a good config.
	tmp := Path() + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return fmt.Errorf("config: write: %w", err)
	}
	if err := os.Rename(tmp, Path()); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("config: commit: %w", err)
	}
	// Rename preserves the temp file's mode, but be explicit in case the
	// destination already existed with a looser one.
	if err := os.Chmod(Path(), 0o600); err != nil {
		return fmt.Errorf("config: chmod: %w", err)
	}
	return nil
}

// CheckPermissions reports whether the config file is readable by anyone other
// than its owner. Windows has no meaningful equivalent, so it is skipped there.
func CheckPermissions() (ok bool, detail string) {
	if runtime.GOOS == "windows" {
		return true, "not checked on windows"
	}
	info, err := os.Stat(Path())
	if err != nil {
		return true, "no config file"
	}
	mode := info.Mode().Perm()
	if mode&0o077 != 0 {
		return false, fmt.Sprintf("%s is mode %04o — readable by others; run: chmod 600 %s", Path(), mode, Path())
	}
	return true, fmt.Sprintf("mode %04o", mode)
}

// ---------------------------------------------------------------------------
// Field access for the CLI
// ---------------------------------------------------------------------------

// Field describes a settable key.
type Field struct {
	Name   string
	Secret bool
	Help   string
	get    func(*Config) string
	set    func(*Config, string) error
}

// Fields enumerates everything `mole config set` accepts.
func Fields() []Field {
	fields := []Field{
		{
			Name: "search.provider", Help: "search backend: brave | tavily",
			get: func(c *Config) string { return c.Search.Provider },
			set: func(c *Config, v string) error {
				v = strings.ToLower(strings.TrimSpace(v))
				if v != "brave" && v != "tavily" {
					return fmt.Errorf("search.provider must be brave or tavily, got %q", v)
				}
				c.Search.Provider = v
				return nil
			},
		},
		{
			Name: "search.brave-key", Secret: true, Help: "Brave Search API subscription token",
			get: func(c *Config) string { return c.Search.BraveKey },
			set: func(c *Config, v string) error { c.Search.BraveKey = strings.TrimSpace(v); return nil },
		},
		{
			Name: "search.tavily-key", Secret: true, Help: "Tavily API key (tvly-…)",
			get: func(c *Config) string { return c.Search.TavilyKey },
			set: func(c *Config, v string) error { c.Search.TavilyKey = strings.TrimSpace(v); return nil },
		},
		{
			Name: "llm.provider", Help: "model backend: anthropic | openai-compatible",
			get: func(c *Config) string { return c.LLM.Provider },
			set: func(c *Config, v string) error {
				v = strings.ToLower(strings.TrimSpace(v))
				if v != "anthropic" && v != "openai-compatible" {
					return fmt.Errorf("llm.provider must be anthropic or openai-compatible, got %q", v)
				}
				c.LLM.Provider = v
				return nil
			},
		},
		{
			Name: "llm.api-key", Secret: true, Help: "model provider API key",
			get: func(c *Config) string { return c.LLM.APIKey },
			set: func(c *Config, v string) error { c.LLM.APIKey = strings.TrimSpace(v); return nil },
		},
		{
			Name: "llm.base-url", Help: "override endpoint for a self-hosted model",
			get: func(c *Config) string { return c.LLM.BaseURL },
			set: func(c *Config, v string) error { c.LLM.BaseURL = strings.TrimSpace(v); return nil },
		},
		{
			Name: "llm.model", Help: "default model id",
			get: func(c *Config) string { return c.LLM.Model },
			set: func(c *Config, v string) error { c.LLM.Model = strings.TrimSpace(v); return nil },
		},
		{
			Name: "llm.cheap-model", Help: "model for chunk mining and extraction",
			get: func(c *Config) string { return c.LLM.CheapModel },
			set: func(c *Config, v string) error { c.LLM.CheapModel = strings.TrimSpace(v); return nil },
		},
		{
			Name: "llm.verifier-model",
			Help: "model for judging claim pairs and grounding; defaults to the cheap model",
			get:  func(c *Config) string { return c.LLM.VerifierModel },
			set: func(c *Config, v string) error {
				c.LLM.VerifierModel = strings.TrimSpace(v)
				return nil
			},
		},
		{
			Name: "llm.recheck-model",
			Help: "model that re-judges contradiction edges before the report; empty disables. " +
				"sent to the provider of the role whose model it names, else the cheap model's provider",
			get: func(c *Config) string { return c.LLM.RecheckModel },
			set: func(c *Config, v string) error {
				c.LLM.RecheckModel = strings.TrimSpace(v)
				return nil
			},
		},
		{
			Name: "daemon.max-session-usd",
			Help: "ceiling on what one MCP-started session may spend, e.g. 2.50; unset means no ceiling",
			get: func(c *Config) string {
				if c.MaxSessionUSD == 0 {
					return ""
				}
				return core.FormatUSD(c.MaxSessionUSD)
			},
			set: func(c *Config, v string) error {
				v = strings.TrimSpace(v)
				if v == "" {
					c.MaxSessionUSD = 0
					return nil
				}
				amount, err := core.ParseUSD(v)
				if err != nil {
					return err
				}
				if amount < 0 {
					return errors.New("a ceiling cannot be negative")
				}
				c.MaxSessionUSD = amount
				return nil
			},
		},
		{
			Name: "daemon.max-session-tokens",
			Help: "ceiling on tokens for one MCP-started session; unset means no ceiling",
			get: func(c *Config) string {
				if c.MaxSessionTokens == 0 {
					return ""
				}
				return strconv.FormatInt(c.MaxSessionTokens, 10)
			},
			set: func(c *Config, v string) error {
				v = strings.TrimSpace(v)
				if v == "" {
					c.MaxSessionTokens = 0
					return nil
				}
				n, err := strconv.ParseInt(v, 10, 64)
				if err != nil {
					return fmt.Errorf("not a whole number of tokens: %w", err)
				}
				if n < 0 {
					return errors.New("a ceiling cannot be negative")
				}
				c.MaxSessionTokens = n
				return nil
			},
		},
		{
			Name: "max-wallclock",
			Help: "default wall-clock ceiling for a session, e.g. 20m; --timeout overrides it per run",
			get: func(c *Config) string {
				if c.MaxWallClock == 0 {
					return ""
				}
				return c.MaxWallClock.Duration().String()
			},
			set: func(c *Config, v string) error {
				v = strings.TrimSpace(v)
				if v == "" {
					c.MaxWallClock = 0
					return nil
				}
				d, err := time.ParseDuration(v)
				if err != nil {
					return fmt.Errorf("not a duration (try 20m): %w", err)
				}
				if d <= 0 {
					return errors.New("a session clock must be above zero")
				}
				c.MaxWallClock = Duration(d)
				return nil
			},
		},
		{
			Name: "llm.timeout",
			Help: "ceiling on a single model call, e.g. 20m; local models on modest hardware need more than the 10m default",
			get: func(c *Config) string {
				if c.LLM.Timeout == 0 {
					return ""
				}
				return c.LLM.Timeout.String()
			},
			set: func(c *Config, v string) error {
				v = strings.TrimSpace(v)
				if v == "" {
					c.LLM.Timeout = 0
					return nil
				}
				d, err := time.ParseDuration(v)
				if err != nil {
					return fmt.Errorf("not a duration (try 20m): %w", err)
				}
				if d < 0 {
					return errors.New("timeout cannot be negative")
				}
				c.LLM.Timeout = d
				return nil
			},
		},
		{
			Name: "llm.verifier-batch-size",
			Help: "claim pairs per adjudication call; lower it if a slow model times out",
			get: func(c *Config) string {
				if c.LLM.VerifierBatchSize == 0 {
					return ""
				}
				return strconv.Itoa(c.LLM.VerifierBatchSize)
			},
			set: func(c *Config, v string) error {
				v = strings.TrimSpace(v)
				if v == "" {
					c.LLM.VerifierBatchSize = 0
					return nil
				}
				n, err := strconv.Atoi(v)
				if err != nil || n < 1 {
					return errors.New("must be a positive number of pairs")
				}
				c.LLM.VerifierBatchSize = n
				return nil
			},
		},
		{
			Name: "llm.max-input-tokens",
			Help: "per-REQUEST input cap; set to your provider's TPM limit if it is small (e.g. 6000 on a Groq free tier)",
			get: func(c *Config) string {
				if c.LLM.MaxInputTokens == 0 {
					return ""
				}
				return strconv.FormatInt(c.LLM.MaxInputTokens, 10)
			},
			set: func(c *Config, v string) error {
				v = strings.TrimSpace(v)
				if v == "" {
					c.LLM.MaxInputTokens = 0
					return nil
				}
				n, err := strconv.ParseInt(v, 10, 64)
				if err != nil || n < 0 {
					return fmt.Errorf("llm.max-input-tokens must be a non-negative integer, got %q", v)
				}
				c.LLM.MaxInputTokens = n
				return nil
			},
		},
		{
			Name: "contact-email", Help: "contact address sent to academic providers (required by Unpaywall and NCBI)",
			get: func(c *Config) string { return c.ContactEmail },
			set: func(c *Config, v string) error {
				v = strings.TrimSpace(v)
				if v != "" && !strings.Contains(v, "@") {
					return fmt.Errorf("contact-email does not look like an address: %q", v)
				}
				c.ContactEmail = v
				return nil
			},
		},
	}
	for _, role := range RoleNames {
		fields = append(fields, roleFields(role)...)
	}
	return fields
}

// roleFields declares the llm.<role>.* keys for one role.
func roleFields(role string) []Field {
	prefix := "llm." + role + "."
	field := func(name, help string, secret bool, pick func(*LLMRole) *string, check func(string) (string, error)) Field {
		return Field{
			Name: prefix + name, Help: help, Secret: secret,
			get: func(c *Config) string {
				if b := c.LLM.Block(role); b != nil {
					return *pick(b)
				}
				return ""
			},
			set: func(c *Config, v string) error {
				v, err := check(strings.TrimSpace(v))
				if err != nil {
					return err
				}
				p := c.LLM.block(role)
				if *p == nil {
					*p = &LLMRole{}
				}
				*pick(*p) = v
				if **p == (LLMRole{}) {
					*p = nil
				}
				return nil
			},
		}
	}
	keep := func(v string) (string, error) { return v, nil }
	return []Field{
		field("provider", role+" role backend: anthropic | openai-compatible; empty inherits llm.provider", false,
			func(r *LLMRole) *string { return &r.Provider },
			func(v string) (string, error) {
				v = strings.ToLower(v)
				if v != "" && v != "anthropic" && v != "openai-compatible" {
					return "", fmt.Errorf("%sprovider must be anthropic or openai-compatible, got %q", prefix, v)
				}
				return v, nil
			}),
		field("api-key", role+" role API key; empty inherits llm.api-key", true,
			func(r *LLMRole) *string { return &r.APIKey }, keep),
		field("base-url", role+" role endpoint; empty inherits llm.base-url", false,
			func(r *LLMRole) *string { return &r.BaseURL }, keep),
		field("model", role+" role model; wins over the top-level model setting", false,
			func(r *LLMRole) *string { return &r.Model }, keep),
	}
}

func lookupField(name string) (Field, bool) {
	name = strings.ToLower(strings.TrimSpace(name))
	for _, f := range Fields() {
		if f.Name == name {
			return f, true
		}
	}
	return Field{}, false
}

// Set assigns a field by name.
func (c *Config) Set(name, value string) error {
	f, ok := lookupField(name)
	if !ok {
		return fmt.Errorf("config: unknown key %q", name)
	}
	return f.set(c, value)
}

// Get reads a field by name.
func (c *Config) Get(name string) (string, error) {
	f, ok := lookupField(name)
	if !ok {
		return "", fmt.Errorf("config: unknown key %q", name)
	}
	return f.get(c), nil
}

// Display returns the value safe for printing: secrets are masked, because
// `mole config list` output ends up in terminal scrollback, screenshots, and
// bug reports.
func Display(f Field, c *Config) string {
	v := f.get(c)
	if v == "" {
		return "(unset)"
	}
	if !f.Secret {
		return v
	}
	return Mask(v)
}

// Mask reduces a secret to a recognizable stub. Enough to confirm which key is
// configured, not enough to use.
func Mask(s string) string {
	if len(s) <= 8 {
		return "********"
	}
	return s[:4] + strings.Repeat("*", 8) + s[len(s)-2:]
}

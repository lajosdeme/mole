// Command mole is the daemon and CLI.
//
// `research` and `eval` drive and score the planner loop; the rest operate on
// persisted state or configuration — migrate, doctor, sessions, stats, trace,
// config — plus a dev seeder.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime/debug"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/lajosdeme/mole/internal/budget"
	"github.com/lajosdeme/mole/internal/compute/coderunner"
	"github.com/lajosdeme/mole/internal/compute/sandbox"
	"github.com/lajosdeme/mole/internal/config"
	"github.com/lajosdeme/mole/internal/core"
	"github.com/lajosdeme/mole/internal/daemon"
	"github.com/lajosdeme/mole/internal/llm"
	"github.com/lajosdeme/mole/internal/obs"
	"github.com/lajosdeme/mole/internal/pricing"
	"github.com/lajosdeme/mole/internal/store"
	"github.com/lajosdeme/mole/internal/store/sqlite"
	"github.com/lajosdeme/mole/internal/tools/academic"
	"github.com/lajosdeme/mole/internal/tools/search"
	"github.com/lajosdeme/mole/internal/verifier"
	"github.com/spf13/cobra"
)

// version is overridden at build time via -ldflags.
// version is set by the release build's ldflags. `go install` sets no ldflags, so
// it falls back to the module version the toolchain embedded — otherwise every
// installed copy reports "dev" and a bug report cannot say which build it came
// from.
var version = "dev"

func moleVersion() string {
	if version != "dev" {
		return version
	}
	if info, ok := debug.ReadBuildInfo(); ok && info.Main.Version != "" &&
		info.Main.Version != "(devel)" {
		return info.Main.Version
	}
	return version
}

// ---------------------------------------------------------------------------
// Shared plumbing
// ---------------------------------------------------------------------------

// defaultDBPath follows the XDG data spec so the database lands somewhere a
// backup tool will find it, not in a dotfile at $HOME.
func defaultDBPath() string {
	if p := os.Getenv("MOLE_DB"); p != "" {
		return p
	}
	base := os.Getenv("XDG_DATA_HOME")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "mole.db"
		}
		base = filepath.Join(home, ".local", "share")
	}
	return filepath.Join(base, "mole", "mole.db")
}

// openDBMigrate opens and applies any pending migrations.
//
// The migration is the reason this is a separate door. Ordinary writes from two
// processes are fine — see openDBNoMigrate — but changing the schema underneath
// a daemon that is mid-run is not, because its prepared statements and its Go
// structs were built against the old one. So migrating is reserved for the
// commands whose job it is (`migrate`) or that own the database for their whole
// run (`serve`).
func openDBMigrate(ctx context.Context, path string) (*sqlite.DB, error) {
	db, err := sqlite.Open(path, sqlite.Options{})
	if err != nil {
		return nil, err
	}
	if err := db.Migrate(ctx); err != nil {
		_ = db.Close()
		return nil, err
	}
	return db, nil
}

// openDBNoMigrate opens an existing database at the expected schema version,
// leaving the schema alone. The returned DB can still write.
//
// This is what lets the CLI coexist with a running daemon, and the reason is
// narrower than it looks. SQLite in WAL mode allows any number of readers
// alongside a writer, and serialises writers per TRANSACTION via a file lock
// held across processes — not per connection. The daemon opens the database and
// keeps it open, but between transactions it holds no lock at all, so a second
// process writing short transactions is not blocked in any way a person could
// perceive; busy_timeout absorbs the overlap. `mole ask` writes on this basis.
//
// What is genuinely unsafe is migrating under a live daemon, so this does not:
// it checks the schema version and says what to run.
// errNoDatabase distinguishes a fresh install from a broken one, so `doctor` can
// tell a first-time user "nothing is wrong yet" instead of exiting non-zero at
// them.
var errNoDatabase = errors.New("no database")

func openDBNoMigrate(ctx context.Context, path string) (*sqlite.DB, error) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, fmt.Errorf("%w at %s (run: mole migrate)", errNoDatabase, path)
	}

	db, err := sqlite.Open(path, sqlite.Options{})
	if err != nil {
		return nil, err
	}

	have, err := db.SchemaVersion(ctx)
	if err != nil {
		_ = db.Close()
		return nil, err
	}
	want, err := sqlite.ExpectedSchemaVersion()
	if err != nil {
		_ = db.Close()
		return nil, err
	}
	if have < want {
		_ = db.Close()
		return nil, fmt.Errorf("database schema is at version %d, this binary expects %d (run: mole migrate)", have, want)
	}
	if have > want {
		// A newer daemon wrote this database. Reading it with older code could
		// misinterpret columns, so refuse rather than print something wrong.
		_ = db.Close()
		return nil, fmt.Errorf("database schema is at version %d, newer than this binary's %d (upgrade mole)", have, want)
	}
	return db, nil
}

func newTabWriter() *tabwriter.Writer {
	return tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
}

// ---------------------------------------------------------------------------
// migrate
// ---------------------------------------------------------------------------

func newMigrateCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "migrate",
		Short: "Apply pending database migrations",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			return cmdMigrate(cmd.Context(), dbPath(cmd))
		},
	}
}

func cmdMigrate(ctx context.Context, path string) error {
	db, err := openDBMigrate(ctx, path)
	if err != nil {
		return err
	}
	defer db.Close()

	fmt.Printf("migrations applied — %s\n", db.Path())
	return nil
}

// ---------------------------------------------------------------------------
// doctor
// ---------------------------------------------------------------------------

// cmdDoctor is where the "check this at startup, not in the README"
// requirements live: the database, the schema, the ledger's own consistency,
// the credentials M1 needs to make a call, the sandbox M8 uses when it is there,
// and the MCP socket M7 listens on.
//
// It exits non-zero when something is actually broken, so a setup script can
// branch on it.
func newDoctorCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "doctor",
		Short: "Check configuration and environment",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			return cmdDoctor(cmd.Context(), dbPath(cmd))
		},
	}
}

func cmdDoctor(ctx context.Context, path string) error {
	r := &checks{}
	report := r.require

	db, err := openDBNoMigrate(ctx, path)
	if err != nil {
		if errors.Is(err, errNoDatabase) {
			// A fresh install, not a broken one. `research` creates the database on
			// first use, so there is nothing for the user to repair — and exiting
			// non-zero here tells a setup script the opposite.
			r.note(false, "state db", "not created yet; the first run makes it")
			fmt.Println("\nnothing is wrong — run a research question to get started")
			return nil
		}
		report(false, "state db", err.Error())
		return errors.New("doctor found problems")
	}
	defer db.Close()
	report(true, "state db", db.Path()+" (sqlite, WAL, single writer)")

	var sessionCount int
	err = db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		list, err := q.ListSessions(ctx, 1000)
		sessionCount = len(list)
		return err
	})
	if err != nil {
		report(false, "schema", err.Error())
	} else {
		report(true, "schema", fmt.Sprintf("readable, %d session(s)", sessionCount))
	}

	table := pricing.NewTable()
	report(true, "pricing", fmt.Sprintf("%d models registered", len(table.Models())))

	// Reconcile every session's materialized counters against the ledger.
	// A divergence here is the single most important thing this command can
	// find: it means the budget ceiling is no longer trustworthy.
	led := budget.New(db, budget.DefaultConfig())
	drifted := 0
	err = db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		list, err := q.ListSessions(ctx, 1000)
		if err != nil {
			return err
		}
		for _, s := range list {
			v, err := led.Verify(ctx, s.ID)
			if err != nil {
				return err
			}
			if !v.Consistent() {
				drifted++
				fmt.Printf("    session %s: spent recorded=%d ledger=%d, held recorded=%d rows=%d\n",
					s.ID, v.SpentRecorded, v.SpentFromLedger, v.HeldRecorded, v.HeldFromRows)
			}
		}
		return nil
	})
	if err != nil {
		report(false, "ledger", err.Error())
	} else {
		report(drifted == 0, "ledger", fmt.Sprintf("%d session(s) reconciled, %d drifted", sessionCount, drifted))
	}

	reportConfig(r)
	reportSandbox(ctx, r)
	reportSocket(r)

	if r.pending > 0 {
		fmt.Println("\nsome checks are informational until their milestone lands")
	}
	if r.blocked > 0 {
		// Exit non-zero. doctor is what a setup script and a CI job run to find
		// out whether this install works; printing "!" and returning success
		// tells both of them it does.
		return fmt.Errorf("doctor found %d problem(s)", r.blocked)
	}
	return nil
}

// checks accumulates doctor's findings.
//
// Two severities, because they mean different things to a caller: require is
// "this install is broken", note is "a later milestone will need this". Folding
// them together is what made the exit code useless.
type checks struct {
	blocked int
	pending int
}

func (c *checks) require(good bool, label, detail string) {
	if !good {
		c.blocked++
	}
	c.print(good, label, detail)
}

func (c *checks) note(good bool, label, detail string) {
	if !good {
		c.pending++
	}
	c.print(good, label, detail)
}

func (c *checks) print(good bool, label, detail string) {
	mark := "✓"
	if !good {
		mark = "!"
	}
	fmt.Printf("%s %-18s %s\n", mark, label, detail)
}

// reportSandbox reports the container runtime (§3.6), and reports it as a
// CAPABILITY rather than a requirement.
//
// note, not require: mole is one static binary and a container runtime is
// needed by CodeRunner alone. The SQL path needs none — §12.2 says outright
// that "the sandbox is not the control here" — so an absent runtime costs one
// feature and must not fail a setup script.
//
// The second line is the point of the whole check. "sandbox: not found" tells
// somebody something is missing without telling them whether it matters, and
// the answer is that it usually does not.
func reportSandbox(ctx context.Context, r *checks) {
	rep := sandbox.Detect(ctx)

	switch {
	case rep.Usable:
		r.note(true, "sandbox", rep.Detail)
		r.print(true, "", "  "+sandbox.DefaultLimits().Summary())
		// A usable runtime with no interpreter image is ready for everything
		// except the one thing that needs one, so the two are reported apart.
		if rep.HasImage(ctx, coderunner.DefaultImage) {
			r.print(true, "", "  image "+coderunner.DefaultImage+" present")
		} else {
			r.note(false, "sandbox image", coderunner.DefaultImage+" not pulled")
			r.print(false, "", "  code analysis needs it: "+
				string(rep.Runtime)+" pull "+coderunner.DefaultImage)
		}
		if len(rep.Missing) > 0 {
			// Usable and imperfect. Said out loud, because the alternative is a
			// tick beside a runtime that cannot report memory limits reliably.
			r.print(true, "", "  note: "+strings.Join(rep.Missing, "; "))
		}
	case rep.Found:
		// Present and not trustworthy, which is a different problem from absent
		// and needs a different sentence. Trusting it would be worse than
		// having nothing.
		r.note(false, "sandbox", rep.Detail)
		r.print(false, "", "  cannot run model-authored code: "+strings.Join(rep.Missing, "; "))
		r.print(false, "", "  local SQL analysis is unaffected (§12.2)")
	default:
		r.note(false, "sandbox", rep.Detail)
		r.print(false, "", "  local code analysis unavailable; install podman or docker to enable it")
		r.print(false, "", "  local SQL analysis is unaffected (§12.2)")
	}
}

// reportSocket checks the MCP socket's §3.5 properties.
//
// require, not note: M7 has landed, and an exposed socket is a real problem with
// this install rather than a milestone that has not arrived — anything that can
// connect can spend the user's budget and read every claim they have collected.
// A socket that does not exist yet is not a problem, and says so.
//
// The check calls daemon.InspectSocket, which calls the same checkPrivateDir the
// daemon enforces. A doctor line that reported a property nothing enforced, or
// enforced a property doctor did not report, is the failure this project has
// already made once (§12.1's "every crossing is logged", which emitted nothing
// for a whole milestone).
func reportSocket(r *checks) {
	st := daemon.InspectSocket(defaultSocket())
	r.require(st.OK(), "mcp socket", st.Summary())
	// The first problem is already in the summary line.
	for _, p := range st.Problems[min(1, len(st.Problems)):] {
		r.print(false, "", "  "+p)
	}
	if st.Present {
		// Notes about an absent socket would only repeat the summary.
		for _, n := range st.Notes {
			r.print(st.OK(), "", "  "+n)
		}
	}
	if st.DirMode != 0 {
		owner := "owned by this user"
		if !st.DirOwned {
			owner = "owned by another user"
		}
		r.print(st.DirOK, "", fmt.Sprintf("  directory %#o, %s", st.DirMode, owner))
	}
	// The other half of §3.5 that a user can get wrong on their own: a key in
	// .mcp.json is a leaked key, because that file gets committed.
	if path, found := mcpConfigWithSecret(); found {
		r.require(false, "mcp config", path+" contains what looks like an API key")
		r.print(false, "", "  .mcp.json is committed; credentials belong in "+
			config.Path()+" (mode 0600), and the shim passes none")
	}
}

// mcpConfigWithSecret looks for a credential in the MCP config files a coding
// agent reads.
//
// Cheap and specific: the shim takes no credentials at all (§5.2), so anything
// key-shaped in there was put there by hand and is both useless and exposed. Only
// the files in the current directory and the user's home are read — this is a
// check, not a filesystem scan.
func mcpConfigWithSecret() (string, bool) {
	home, _ := os.UserHomeDir()
	candidates := []string{".mcp.json"}
	if home != "" {
		candidates = append(candidates,
			filepath.Join(home, ".mcp.json"),
			filepath.Join(home, ".claude.json"))
	}
	for _, path := range candidates {
		raw, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		if !strings.Contains(string(raw), "mole") {
			continue
		}
		if secretPattern.MatchString(string(raw)) {
			return path, true
		}
	}
	return "", false
}

// secretPattern matches the key shapes the providers mole uses actually issue,
// rather than the word "key": a config naming `MOLE_DB_PATH` is not a leak, and a
// pattern that flagged it would train the user to ignore this line.
var secretPattern = regexp.MustCompile(`(sk-[A-Za-z0-9_-]{16,}|tvly-[A-Za-z0-9_-]{16,}|BSA[A-Za-z0-9_-]{16,})`)

// reportConfig checks credentials and settings. These are the "verify at
// startup, not in a README" requirements: a missing search key is a session
// that fails on its first lead, and a missing contact address is a ban from an
// academic provider that requires identification.
func reportConfig(r *checks) {
	report := r.require
	cfg, err := config.Load()
	if err != nil && !errors.Is(err, config.ErrNotConfigured) {
		report(false, "config", err.Error())
		return
	}

	if permsOK, detail := config.CheckPermissions(); !permsOK {
		// The config file holds API keys and nothing else protects them yet.
		report(false, "config perms", detail)
	} else {
		report(true, "config perms", detail)
	}

	switch {
	case cfg.Search.Provider == "":
		report(false, "search provider", "not selected (run: mole config set search.provider brave|tavily)")
	case cfg.Search.ActiveKey() == "":
		report(false, "search provider",
			fmt.Sprintf("%s selected but no key (run: mole config set search.%s-key ...)",
				cfg.Search.Provider, cfg.Search.Provider))
	default:
		// Read from the declared table rather than a provider name written out
		// here, so adding a provider cannot leave this reporting the wrong thing.
		caps := search.CapabilitiesOf(search.Kind(cfg.Search.Provider))
		note := ""
		if caps.PageContent {
			// Changes how many fetches a session makes.
			note = " — returns page content, skips fetches"
		}
		report(true, "search provider",
			fmt.Sprintf("%s, key %s%s", cfg.Search.Provider, config.Mask(cfg.Search.ActiveKey()), note))

		// A note, not a failure: research works fine without dates. Said out loud
		// because the alternative is a mechanism that is silently inert — measured
		// on a ten-question corpus, 369 claims carried no date between them and
		// every conflict across a decade of publication was written as a live
		// disagreement, with nothing anywhere saying why.
		if !caps.PublicationDates {
			detail := "supplies no publication dates, so §11.2 cannot tell a stale " +
				"source from a disagreeing one"
			if caps.PageContent {
				detail += "; page content means nothing is fetched either, and the " +
					"extractor is the only other place a date comes from"
			}
			r.note(false, "source dates", detail)
			r.note(false, "", "--always-fetch restores it at one fetch per source, "+
				"or use a provider that dates its results")
		}
	}

	reportLLM(report, cfg)

	// Reported from the same rule that GATES the providers, not a second
	// hand-written check beside it. §10.3 makes this a startup check rather than
	// a README line, and a check that is free to drift from the thing it gates
	// is worth nothing. Still a note rather than a requirement: an install with
	// no academic provider in use is not broken for want of an address.
	if err := academic.CheckContact(cfg.ContactEmail); err != nil {
		r.note(false, "contact email",
			"not usable — Unpaywall and NCBI require it, so academic providers will refuse to start "+
				"(set it with: mole config set contact-email you@example.com)")
	} else {
		r.note(true, "contact email", cfg.ContactEmail)
	}
}

// ---------------------------------------------------------------------------
// sessions
// ---------------------------------------------------------------------------

func newSessionsCmd() *cobra.Command {
	var limit int
	c := &cobra.Command{
		Use:   "sessions",
		Short: "List recent sessions",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			return cmdSessions(cmd.Context(), dbPath(cmd), limit)
		},
	}
	c.Flags().IntVar(&limit, "limit", 20, "maximum sessions to list")
	return c
}

func cmdSessions(ctx context.Context, path string, limit int) error {
	db, err := openDBNoMigrate(ctx, path)
	if err != nil {
		return err
	}
	defer db.Close()

	var list []*core.Session
	if err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		list, err = q.ListSessions(ctx, limit)
		return err
	}); err != nil {
		return err
	}

	if len(list) == 0 {
		fmt.Println("no sessions")
		return nil
	}

	w := newTabWriter()
	fmt.Fprintln(w, "ID\tSTATUS\tMODE\tSPENT\tBUDGET\tAVAILABLE\tCREATED")
	for _, s := range list {
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			s.ID, s.Status, s.Mode,
			core.FormatAmount(s.Spent, s.BudgetUnit),
			core.FormatAmount(s.Budget, s.BudgetUnit),
			core.FormatAmount(s.Available(), s.BudgetUnit),
			s.CreatedAt.Format(time.RFC3339))
	}
	return w.Flush()
}

// ---------------------------------------------------------------------------
// trace
// ---------------------------------------------------------------------------

func newTraceCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "trace <session-id>",
		Short: "Show the cost and span breakdown for one session",
		Args:  exactArgs(1, "mole trace <session-id>"),
		RunE: func(cmd *cobra.Command, args []string) error {
			return cmdTrace(cmd.Context(), dbPath(cmd), args[0])
		},
	}
}

func cmdTrace(ctx context.Context, path, sessionID string) error {
	db, err := openDBNoMigrate(ctx, path)
	if err != nil {
		return err
	}
	defer db.Close()

	var (
		sess   *core.Session
		byRole map[core.Role]core.Cost
		spans  []*core.Span
		claims []*core.Claim
		edges  []*core.ClaimEdge
	)
	if err := db.Read(ctx, func(ctx context.Context, q store.Queries) error {
		var err error
		if sess, err = q.GetSession(ctx, sessionID); err != nil {
			return err
		}
		if byRole, err = q.SumCostsByRole(ctx, sessionID); err != nil {
			return err
		}
		if claims, err = q.ListClaims(ctx, sessionID, 0); err != nil {
			return err
		}
		if edges, err = q.ListEdges(ctx, sessionID, 0); err != nil {
			return err
		}
		spans, err = q.ListSpans(ctx, sessionID)
		return err
	}); err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return fmt.Errorf("no session %s", sessionID)
		}
		return err
	}

	unit := sess.BudgetUnit
	fmt.Printf("%s  %s / %s  ·  status=%s  mode=%s\n",
		sess.ID,
		core.FormatAmount(sess.Spent, unit),
		core.FormatAmount(sess.Budget, unit),
		sess.Status, sess.Mode)
	if sess.Escrow > 0 {
		fmt.Printf("escrow held: %s\n", core.FormatAmount(sess.Escrow, unit))
	}
	fmt.Println()

	// Role breakdown — the reason ToolCall.Role exists.
	roles := []core.Role{core.RolePlanner, core.RoleExecutor, core.RoleVerifier, core.RoleOutput}
	total := sess.Spent
	w := newTabWriter()
	fmt.Fprintln(w, "ROLE\tSPEND\tSHARE\tTOKENS")
	for _, r := range roles {
		c, ok := byRole[r]
		if !ok {
			continue
		}
		amount := c.BudgetAmount(unit)
		share := 0.0
		if total > 0 {
			share = float64(amount) / float64(total) * 100
		}
		fmt.Fprintf(w, "%s\t%s\t%.0f%%\t%d\n", r, core.FormatAmount(amount, unit), share, c.TotalTokens())
	}
	if err := w.Flush(); err != nil {
		return err
	}

	// Cache accounting, which a flat token count would hide entirely.
	var all core.Cost
	for _, c := range byRole {
		all = all.Add(c)
	}
	fmt.Printf("\ntokens: %d in · %d out · %d cache-read · %d cache-write\n",
		all.InputTokens, all.OutputTokens, all.CacheReadTokens, all.CacheWriteTokens)

	// Ledger reconciliation, shown inline so a drift is impossible to miss.
	led := budget.New(db, budget.DefaultConfig())
	v, err := led.Verify(ctx, sessionID)
	if err != nil {
		return err
	}
	if v.Consistent() {
		fmt.Printf("ledger: consistent (%d calls)\n", v.CallsFromRows)
	} else {
		fmt.Printf("ledger: DRIFT — spent recorded=%d ledger=%d, held recorded=%d rows=%d, calls recorded=%d rows=%d\n",
			v.SpentRecorded, v.SpentFromLedger, v.HeldRecorded, v.HeldFromRows, v.CallsRecorded, v.CallsFromRows)
	}

	printClaimGraph(os.Stdout, claims, edges)

	if len(spans) > 0 {
		fmt.Printf("\nspans (%d):\n", len(spans))
		sort.Slice(spans, func(i, j int) bool { return spans[i].StartedAt.Before(spans[j].StartedAt) })
		sw := newTabWriter()
		fmt.Fprintln(sw, "  NAME\tDURATION\tSTATUS")
		for _, s := range spans {
			dur := "open"
			if s.EndedAt != nil {
				dur = s.EndedAt.Sub(s.StartedAt).Round(time.Millisecond).String()
			}
			status := s.Status
			if status == "" {
				status = "-"
			}
			fmt.Fprintf(sw, "  %s\t%s\t%s\n", s.Name, dur, status)
		}
		if err := sw.Flush(); err != nil {
			return err
		}
	}

	return nil
}

// ---------------------------------------------------------------------------
// dev
// ---------------------------------------------------------------------------

func newDevCmd() *cobra.Command {
	c := &cobra.Command{
		Use:   "dev",
		Short: "Development helpers",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			return cmd.Help()
		},
	}
	c.AddCommand(newCoverageCmd())
	c.AddCommand(&cobra.Command{
		Use:   "seed",
		Short: "Write a synthetic session through the real ledger",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			return cmdDevSeed(cmd.Context(), dbPath(cmd))
		},
	})
	return c
}

func cmdDevSeed(ctx context.Context, path string) error {
	db, err := openDBMigrate(ctx, path)
	if err != nil {
		return err
	}
	defer db.Close()

	log := obs.NewLogger(obs.LogOptions{Level: "info", Format: "text"})
	tracer := obs.NewTracer(db, log)
	table := pricing.NewTable()
	led := budget.New(db, budget.DefaultConfig())

	sess, err := led.CreateSession(ctx, budget.SessionSpec{
		Prompt:       "seed: what is the current consensus on byte-level LLMs?",
		Mode:         core.ModeReport,
		ActorTypes:   []core.ActorType{core.ActorWeb},
		BudgetUnit:   core.BudgetUSD,
		Budget:       3 * core.MicrosPerUSD,
		MaxToolCalls: 200,
	})
	if err != nil {
		return err
	}

	type step struct {
		role  core.Role
		model string
		usage pricing.Usage
	}
	steps := []step{
		{core.RolePlanner, "claude-opus-5", pricing.Usage{InputTokens: 3200, OutputTokens: 700}},
		{core.RoleExecutor, "claude-opus-5", pricing.Usage{InputTokens: 14000, OutputTokens: 1800, CacheReadTokens: 9000}},
		{core.RoleExecutor, "claude-opus-5", pricing.Usage{InputTokens: 11000, OutputTokens: 1500, CacheReadTokens: 9000}},
		{core.RoleVerifier, "claude-opus-5", pricing.Usage{InputTokens: 5200, OutputTokens: 900, CacheReadTokens: 4000}},
	}

	for i, st := range steps {
		ctx, span := tracer.Start(ctx, sess.ID, string(st.role))

		cost, err := table.Cost(st.model, st.usage)
		if err != nil {
			span.End(ctx, "error")
			return err
		}

		amount := cost.BudgetAmount(sess.BudgetUnit)
		res, err := led.Reserve(ctx, sess.ID, amount)
		if err != nil {
			span.End(ctx, "error")
			return fmt.Errorf("step %d: %w", i, err)
		}

		if _, err := led.Settle(ctx, res, []core.ToolCall{{
			Role:  st.role,
			Type:  core.CallLLM,
			Model: st.model,
			Cost:  cost,
		}}); err != nil {
			span.End(ctx, "error")
			return err
		}
		span.End(ctx, "ok")
	}

	// Output is paid from escrow — release it first, exactly as the real loop
	// will.
	released, err := led.ReleaseEscrow(ctx, sess.ID)
	if err != nil {
		return err
	}
	ctx, outSpan := tracer.Start(ctx, sess.ID, "output")
	outCost, err := table.Cost("claude-opus-5", pricing.Usage{InputTokens: 18000, OutputTokens: 4200, CacheReadTokens: 12000})
	if err != nil {
		outSpan.End(ctx, "error")
		return err
	}
	outRes, err := led.Reserve(ctx, sess.ID, outCost.BudgetAmount(sess.BudgetUnit))
	if err != nil {
		outSpan.End(ctx, "error")
		return err
	}
	if _, err := led.Settle(ctx, outRes, []core.ToolCall{{
		Role: core.RoleOutput, Type: core.CallLLM, Model: "claude-opus-5", Cost: outCost,
	}}); err != nil {
		outSpan.End(ctx, "error")
		return err
	}
	outSpan.End(ctx, "ok")

	if err := led.Finish(ctx, sess.ID, core.StatusDone); err != nil {
		return err
	}

	fmt.Printf("seeded session %s (escrow released: %s)\n",
		sess.ID, core.FormatAmount(released, sess.BudgetUnit))
	fmt.Printf("inspect: mole trace %s\n", sess.ID)
	return nil
}

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

const configUsage = `mole config — settings and credentials

Usage:
  mole config list              Show all settings (secrets masked)
  mole config get <key>         Print one value
  mole config set <key> <value> Assign a value
  mole config path              Print the config file location
  mole config test-llm          Make one cheap call and report model + cost

Keys:
`

func newConfigCmd() *cobra.Command {
	c := &cobra.Command{
		Use:   "config",
		Short: "Get, set, and list settings",
		Long:  configUsage + configKeyHelp(),
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			printConfigUsage()
			return nil
		},
	}
	c.AddCommand(
		&cobra.Command{
			Use: "list", Short: "Show all settings (secrets masked)", Args: cobra.NoArgs,
			RunE: func(*cobra.Command, []string) error { return cmdConfigList() },
		},
		&cobra.Command{
			Use: "get <key>", Short: "Print one value", Args: exactArgs(1, "mole config get <key>"),
			RunE: func(_ *cobra.Command, a []string) error { return cmdConfigGet(a[0]) },
		},
		&cobra.Command{
			Use: "set <key> <value>", Short: "Assign a value", Args: exactArgs(2, "mole config set <key> <value>"),
			RunE: func(_ *cobra.Command, a []string) error { return cmdConfigSet(a[0], a[1]) },
		},
		&cobra.Command{
			Use: "path", Short: "Print the config file location", Args: cobra.NoArgs,
			RunE: func(*cobra.Command, []string) error { fmt.Println(config.Path()); return nil },
		},
		&cobra.Command{
			Use: "test-llm", Short: "Make one cheap call and report model + cost", Args: cobra.NoArgs,
			RunE: func(cmd *cobra.Command, _ []string) error { return cmdConfigTestLLM(cmd.Context()) },
		},
	)
	return c
}

func printConfigUsage() {
	fmt.Print(configUsage)
	fmt.Print(configKeyHelp())
}

// configKeyHelp lists the settable keys. Generated from config.Fields() rather
// than written out, so a new setting cannot be added without appearing here.
func configKeyHelp() string {
	var b strings.Builder
	w := tabwriter.NewWriter(&b, 0, 0, 2, ' ', 0)
	for _, f := range config.Fields() {
		secret := ""
		if f.Secret {
			secret = "  (secret)"
		}
		fmt.Fprintf(w, "  %s\t%s%s\n", f.Name, f.Help, secret)
	}
	_ = w.Flush()
	return b.String()
}

// loadConfigForEdit tolerates a missing file: `mole config set` on a fresh
// machine has to work, and that is the only way the file ever gets created.
func loadConfigForEdit() (*config.Config, error) {
	cfg, err := config.Load()
	if err != nil && !errors.Is(err, config.ErrNotConfigured) {
		return nil, err
	}
	return cfg, nil
}

func cmdConfigList() error {
	cfg, err := loadConfigForEdit()
	if err != nil {
		return err
	}

	fmt.Printf("%s\n\n", config.Path())
	w := newTabWriter()
	fmt.Fprintln(w, "KEY\tVALUE")
	for _, f := range config.Fields() {
		fmt.Fprintf(w, "%s\t%s\n", f.Name, config.Display(f, cfg))
	}
	if err := w.Flush(); err != nil {
		return err
	}

	if ok, detail := config.CheckPermissions(); !ok {
		fmt.Printf("\n! %s\n", detail)
	}
	return nil
}

func cmdConfigGet(key string) error {
	cfg, err := loadConfigForEdit()
	if err != nil {
		return err
	}
	v, err := cfg.Get(key)
	if err != nil {
		return err
	}
	// Print secrets in full here: `get` on a single named key is a deliberate
	// act, unlike `list`, which is browsing.
	fmt.Println(v)
	return nil
}

func cmdConfigSet(key, value string) error {
	cfg, err := loadConfigForEdit()
	if err != nil {
		return err
	}
	if err := cfg.Set(key, value); err != nil {
		return err
	}
	if err := cfg.Save(); err != nil {
		return err
	}

	f, _ := lookupConfigField(key)
	fmt.Printf("set %s = %s\n", key, config.Display(f, cfg))
	return nil
}

func lookupConfigField(name string) (config.Field, bool) {
	for _, f := range config.Fields() {
		if f.Name == name {
			return f, true
		}
	}
	return config.Field{}, false
}

// reportLLM says which model backend is in force and, critically, WHERE its
// credential came from.
//
// Three sources can supply it — config, environment, or the SDK's own chain
// resolving an `ant auth login` profile — and "which one is this actually
// using" otherwise becomes a debugging session. Auto-detection means the answer
// is frequently one the user never typed.
func reportLLM(report func(bool, string, string), cfg *config.Config) {
	built, reason, err := buildLLM(cfg)
	if err != nil {
		report(false, "llm provider", err.Error())
		return
	}
	if built == nil {
		report(false, "llm provider",
			"none found — set one with `mole config set llm.api-key ...`, "+
				"run `ant auth login`, or start a local model (ollama serve)")
		return
	}

	detail := fmt.Sprintf("%s via %s", built.ModelFor(llm.TierStrong), llm.SourceOf(built))
	if cheap := built.ModelFor(llm.TierCheap); cheap != built.ModelFor(llm.TierStrong) {
		detail += fmt.Sprintf(" (cheap tier: %s)", cheap)
	}
	if v := cfg.LLM.VerifierModel; v != "" {
		// Named separately because it is the stage whose errors are hardest to see: a
		// bad adjudicator does not fail, it produces a confident graph of wrong edges.
		detail += fmt.Sprintf(" (verifier: %s)", v)
	}
	if reason != "" {
		detail += " — " + reason
	}

	// A local endpoint can be verified for free, so a green tick here is
	// earned rather than assumed. A hosted one cannot be checked without
	// spending money; say so instead of implying it works.
	if base := cfg.LLM.BaseURL; base != "" && llm.IsLoopback(base) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		if !llm.Reachable(ctx, base, nil) {
			report(false, "llm provider", detail+" — NOT REACHABLE (is the runtime started?)")
			return
		}
		report(true, "llm provider", detail+", reachable")
		return
	}
	// An unpriced model is not fatal, but USD budgeting cannot work for it —
	// and a model name left over from a different provider shows up here
	// rather than at the first call, twenty leads into a session.
	if warn := unpricedModels(built); warn != "" {
		report(false, "llm provider", detail+" — "+warn)
		return
	}
	report(true, "llm provider", detail+" (unverified — run: mole config test-llm)")
}

// checkKeyMatchesProvider refuses a key that plainly belongs elsewhere.
//
// llm.provider defaults to Anthropic when unset, so a Groq or OpenAI key set on
// its own yielded "✓ llm provider claude-opus-5 via config" from doctor and
// then an auth failure at the first model call. A green tick that is wrong is
// worse than no tick: it is the check being run and passing.
//
// Only an unambiguous mismatch is refused. An unrecognized prefix is fine —
// self-hosted and proxy keys look like anything.
func checkKeyMatchesProvider(cfg *config.Config, kind llm.Kind) error {
	vendor, known := llm.VendorFromKey(cfg.LLM.APIKey)
	if !known || vendor.Kind == kind {
		return nil
	}
	// An explicitly chosen provider is the user's decision to make; only the
	// silent default is worth blocking on.
	if cfg.LLM.Provider != "" {
		return nil
	}

	var b strings.Builder
	fmt.Fprintf(&b, "llm.api-key looks like a %s key, but llm.provider is unset so it defaults to anthropic.\n",
		vendor.Name)
	b.WriteString("Set the provider explicitly:\n")
	fmt.Fprintf(&b, "  mole config set llm.provider %s\n", vendor.Kind)
	if vendor.BaseURL != "" {
		fmt.Fprintf(&b, "  mole config set llm.base-url %s\n", vendor.BaseURL)
	}
	b.WriteString("  mole config set llm.model <name>\n")
	b.WriteString("  mole config set llm.cheap-model <name>")
	if vendor.Models != "" {
		fmt.Fprintf(&b, "        # e.g. %s", vendor.Models)
	}
	// USD budgeting needs a price, and a third-party model will not be in the
	// table. Say so now rather than at the first settle.
	b.WriteString("\n\nNote: models outside the pricing table cannot be budgeted in USD — use --tokens.")
	return errors.New(b.String())
}

// unpricedModels reports models the pricing table does not know.
//
// Returns "" when everything is priced, or when the provider is a local one
// where zero cost is the correct answer rather than a missing entry.
// unpricedModelList returns the configured models the pricing table does not
// know.
//
// One implementation for both callers: checkUSDIsEnforceable used to gate on
// unpricedModels() being non-empty and then recompute the list itself, so if the
// two ever disagreed the error rendered as "no price is registered for " with
// nothing after it.
func unpricedModelList(p llm.Provider) []string {
	if llm.SourceOf(p) == llm.CredentialNotNeeded {
		return nil // local model, genuinely free
	}
	table := pricing.NewTable()
	var missing []string
	for _, m := range []string{p.ModelFor(llm.TierStrong), p.ModelFor(llm.TierCheap)} {
		if m == "" {
			continue
		}
		if _, ok := table.Lookup(m); !ok && !contains(missing, m) {
			missing = append(missing, m)
		}
	}
	return missing
}

func unpricedModels(p llm.Provider) string {
	if llm.SourceOf(p) == llm.CredentialNotNeeded {
		return "" // local model, genuinely free
	}
	table := pricing.NewTable()
	var missing []string
	for _, m := range []string{p.ModelFor(llm.TierStrong), p.ModelFor(llm.TierCheap)} {
		if m == "" {
			continue
		}
		if _, ok := table.Lookup(m); !ok && !contains(missing, m) {
			missing = append(missing, m)
		}
	}
	if len(missing) == 0 {
		return ""
	}
	return fmt.Sprintf("no price registered for %s; USD budgets will not work (use --tokens, or check llm.model matches the provider)",
		strings.Join(missing, ", "))
}

func contains(hay []string, needle string) bool {
	for _, h := range hay {
		if h == needle {
			return true
		}
	}
	return false
}

// buildLLM resolves a provider from config, falling back to auto-detection.
//
// Precedence: an explicit setting always wins over an ambient one. A user who
// configured a key meant to use it, and silently preferring a detected local
// model would spend their session on the wrong thing.
func buildLLM(cfg *config.Config) (llm.Provider, string, error) {
	return buildLLMWithClient(cfg, nil)
}

// buildLLMWithClient is buildLLM with an explicit HTTP client, so the cassette
// layer can sit under the model calls too (§14.1). A nil client means the SDK's
// own default. Detection still probes with the real client: a cassette of
// "is ollama listening on this machine" would be meaningless to replay.
func buildLLMWithClient(cfg *config.Config, client *http.Client) (provider llm.Provider, reason string, err error) {
	// Any explicit llm.* setting counts as a configured provider, including the
	// model names. Omitting them meant `mole config set llm.model …` on its own
	// was silently discarded in favour of whatever autodetection found — the
	// exact surprise this precedence rule exists to prevent.
	if cfg.LLM.APIKey != "" || cfg.LLM.BaseURL != "" || cfg.LLM.Provider != "" ||
		cfg.LLM.Model != "" || cfg.LLM.CheapModel != "" {
		kind := llm.Kind(cfg.LLM.Provider)
		if kind == "" {
			kind = llm.KindAnthropic
		}
		if err := checkKeyMatchesProvider(cfg, kind); err != nil {
			return nil, "", err
		}
		p, err := llm.New(llm.Config{
			Kind:        kind,
			APIKey:      cfg.LLM.APIKey,
			BaseURL:     cfg.LLM.BaseURL,
			StrongModel: cfg.LLM.Model,
			CheapModel:  cfg.LLM.CheapModel,
			Timeout:     cfg.LLM.Timeout,
		}, client)
		if err != nil {
			return nil, "", err
		}
		return p, "", nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	detected, ok := llm.Detect(ctx, nil)
	if !ok {
		return nil, "", nil
	}
	// A detected local runtime reports no model until one is chosen, and
	// naming it wrongly is worse than saying so.
	if detected.Config.StrongModel == "" && detected.Config.Kind == llm.KindOpenAICompatible {
		return nil, "", fmt.Errorf(
			"found %s but no model selected (run: mole config set llm.model <name>)", detected.Reason)
	}
	p, err := llm.New(detected.Config, client)
	if err != nil {
		return nil, "", err
	}
	suffix := "auto-detected: " + detected.Reason
	if detected.Free {
		suffix += ", no cost"
	}
	return p, suffix, nil
}

// cmdConfigTestLLM makes the smallest possible real call.
//
// Auth, model name, and endpoint are all things that fail at the first call
// rather than at configuration time. Discovering that twenty leads into a
// session wastes the search spend already incurred, so this exists to fail
// early and cheaply.
func cmdConfigTestLLM(ctx context.Context) error {
	cfg, err := loadConfigForEdit()
	if err != nil {
		return err
	}

	provider, reason, err := buildLLM(cfg)
	if err != nil {
		return err
	}
	if provider == nil {
		return errors.New("no model provider configured or detected (run: mole doctor)")
	}

	model := provider.ModelFor(llm.TierCheap)
	src := llm.SourceOf(provider)
	if reason != "" {
		fmt.Printf("using %s via %s (%s)\n", model, src, reason)
	} else {
		fmt.Printf("using %s via %s\n", model, src)
	}

	ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()

	resp, err := provider.Complete(ctx, llm.Request{
		Tier:      llm.TierCheap,
		MaxTokens: 16,
		Messages:  []llm.Message{llm.User("Reply with the single word: ok")},
	})
	if err != nil {
		// Distinguish "your credential is wrong" from "the service hiccuped",
		// because the fix is completely different.
		switch {
		case errors.Is(err, llm.ErrUnauthorized):
			return fmt.Errorf("credential rejected by the provider: %w", err)
		case errors.Is(err, llm.ErrQuotaExceeded):
			return fmt.Errorf("account has no remaining quota: %w", err)
		case llm.Retryable(err):
			return fmt.Errorf("provider temporarily unavailable, credential may still be fine: %w", err)
		}
		return err
	}
	if resp.Refused {
		return fmt.Errorf("provider refused the request (category %q)", resp.RefusalCategory)
	}

	cost, perr := pricing.NewTable().Cost(resp.Model, pricing.Usage{
		InputTokens:      resp.Usage.InputTokens,
		OutputTokens:     resp.Usage.OutputTokens,
		CacheReadTokens:  resp.Usage.CacheReadTokens,
		CacheWriteTokens: resp.Usage.CacheWriteTokens,
	})

	fmt.Printf("✓ reply: %q\n", strings.TrimSpace(resp.Text))
	fmt.Printf("  model: %s · %d in / %d out tokens · %v\n",
		resp.Model, resp.Usage.InputTokens, resp.Usage.OutputTokens,
		resp.Elapsed.Round(time.Millisecond))

	switch {
	case perr != nil:
		// An unpriced model is not a failure, but it does mean USD budgeting
		// cannot work for it — better said now than discovered mid-session.
		fmt.Printf("  cost:  unpriced — %v\n", perr)
		fmt.Printf("         USD budgets will not work with this model; use --tokens\n")
	case cost.USDMicros == 0:
		fmt.Printf("  cost:  free (local model)\n")
	default:
		fmt.Printf("  cost:  %s\n", core.FormatUSD(cost.USDMicros))
	}
	return nil
}

// printClaimGraph shows the §11 graph: how many claims, how many are scored, and
// what the edges say.
//
// The unverified count is the load-bearing number. §11.3's derived confidence is
// 0 until the Verifier scores a claim, and a reader who saw only confidences would
// read a whole session of unscored claims as a session of worthless ones.
func printClaimGraph(w io.Writer, claims []*core.Claim, edges []*core.ClaimEdge) {
	if len(claims) == 0 {
		return
	}

	verified := 0
	for _, c := range claims {
		if c.VerifiedAt != nil {
			verified++
		}
	}

	fmt.Fprintf(w, "\nclaims: %d", len(claims))
	switch {
	case verified == 0:
		fmt.Fprint(w, " · none verified — derived confidence is 0 until the Verifier runs")
	case verified < len(claims):
		fmt.Fprintf(w, " · %d verified, %d not", verified, len(claims)-verified)
	default:
		fmt.Fprint(w, " · all verified")
	}
	fmt.Fprintln(w)

	if len(edges) == 0 {
		return
	}

	byKind := map[core.EdgeKind]int{}
	for _, e := range edges {
		byKind[e.Kind]++
	}
	// Fixed order, so two traces of the same session read the same way.
	order := []core.EdgeKind{core.EdgeDuplicateOf, core.EdgeSupports,
		core.EdgeContradicts, core.EdgeSupersedes, core.EdgeRefines}

	fmt.Fprintf(w, "graph: %d edge(s)\n", len(edges))
	gw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(gw, "  KIND\tCOUNT")
	for _, k := range order {
		if n := byKind[k]; n > 0 {
			fmt.Fprintf(gw, "  %s\t%d\n", k, n)
		}
	}
	if err := gw.Flush(); err != nil {
		return
	}

	// How each confidence was arrived at. §11.3 requires the number be explainable in
	// the trace, verifier.Score has computed that string on every pass since M4, and
	// nothing printed it — so a reader saw a bare 0.62 and had to trust or dismiss it.
	if scored := scoredClusters(claims, edges); len(scored) > 0 {
		fmt.Println("\nconfidence:")
		for _, s := range scored {
			fmt.Printf("  %.2f  %.58q\n        %s\n", s.conf, s.text, s.explain)
		}
	}

	// Contradictions are the one kind worth naming individually: they are what a
	// reader most needs to see, and §11.3 penalizes confidence for them.
	text := map[string]string{}
	for _, c := range claims {
		text[c.ID] = c.Text
	}
	shown := 0
	for _, e := range edges {
		if e.Kind != core.EdgeContradicts || shown >= 5 {
			continue
		}
		if shown == 0 {
			fmt.Fprintln(w, "\ncontradictions:")
		}
		shown++
		fmt.Fprintf(w, "  %.60q\n  ⚡ %.60q\n", text[e.FromID], text[e.ToID])
		if e.Rationale != "" {
			fmt.Fprintf(w, "     %.100s\n", e.Rationale)
		}
	}
	if n := byKind[core.EdgeContradicts]; n > shown {
		fmt.Fprintf(w, "  (%d more)\n", n-shown)
	}
}

// scoredCluster is one line of the confidence breakdown.
type scoredCluster struct {
	conf    float64
	text    string
	explain string
}

// scoredClusters re-derives the graph's confidence breakdown for display.
//
// Recomputed rather than stored: the derivation is a pure function of the claims and
// edges already in the database, so a trace of an old session explains itself with
// today's formula rather than showing a number whose reasoning was thrown away. It also
// means `mole trace` needs no schema of its own for this.
func scoredClusters(claims []*core.Claim, edges []*core.ClaimEdge) []scoredCluster {
	verified := false
	for _, c := range claims {
		if c.VerifiedAt != nil {
			verified = true
			break
		}
	}
	if !verified {
		// Nothing has been scored, so every explanation would read "0 publishers → 0.00"
		// and say only that the Verifier has not run — which the line above already says.
		return nil
	}

	clusters := verifier.Clusters(claims, edges)
	_, scores := verifier.DeriveConfidence(claims, edges)
	if len(scores) != len(clusters) {
		return nil
	}

	out := make([]scoredCluster, 0, len(scores))
	for i, s := range scores {
		rep := clusters[i].Representative()
		if rep == nil {
			continue
		}
		out = append(out, scoredCluster{conf: s.Confidence, text: rep.Text, explain: s.Explain})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].conf > out[j].conf })
	return out
}

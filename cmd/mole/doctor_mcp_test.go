package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// `mole doctor`'s M7 checks, which were "informational until their milestone
// lands" for two milestones after M7 landed.
//
// Both are real problems with an install rather than absent features: anything
// that can reach the socket can spend the user's budget and read every claim they
// have collected, and a key in .mcp.json is a leaked key because that file gets
// committed.

// privateRuntimeDir points defaultSocket() at a directory this test owns.
func privateRuntimeDir(t *testing.T, mode os.FileMode) string {
	t.Helper()
	dir := t.TempDir()
	// t.TempDir() inherits the umask — 0755 on the machine this was written on —
	// so the mode is always set explicitly rather than assumed.
	if err := os.Chmod(dir, mode); err != nil {
		t.Fatal(err)
	}
	t.Setenv("XDG_RUNTIME_DIR", dir)
	return dir
}

// noRuntimePath is a PATH with no container runtime, so doctor's sandbox probe
// never starts podman inside the test's XDG_RUNTIME_DIR.
func noRuntimePath(t *testing.T) string {
	t.Helper()
	return t.TempDir()
}

// TestDoctorReportsTheSocketDirectory, in the ordinary case.
func TestDoctorReportsTheSocketDirectory(t *testing.T) {
	privateRuntimeDir(t, 0o700)
	out, err := doctorRun(t, t.TempDir(), noRuntimePath(t))
	if !strings.Contains(out, "mcp socket") {
		t.Fatalf("doctor says nothing about the socket:\n%s", out)
	}
	if !strings.Contains(out, "not created yet") {
		t.Errorf("a fresh install is not described:\n%s", out)
	}
	if err != nil && strings.Contains(err.Error(), "socket") {
		t.Errorf("a fresh install failed doctor: %v", err)
	}
}

// TestAnExposedSocketDirectoryFailsDoctor.
//
// require, not note: this one is broken now, not pending. The exit code is what a
// setup script branches on, so it has to move.
func TestAnExposedSocketDirectoryFailsDoctor(t *testing.T) {
	privateRuntimeDir(t, 0o700)
	_, baseline := doctorRun(t, t.TempDir(), noRuntimePath(t))

	privateRuntimeDir(t, 0o777)
	out, err := doctorRun(t, t.TempDir(), noRuntimePath(t))

	if !strings.Contains(out, "group or other can reach the socket") {
		t.Errorf("an exposed socket directory is not reported:\n%s", out)
	}
	if err == nil {
		t.Fatal("doctor succeeded with a world-reachable socket directory")
	}
	if baseline != nil && err.Error() == baseline.Error() {
		t.Errorf("the problem count did not change: %v", err)
	}
}

// TestAKeyInTheMCPConfigIsReported.
//
// The shim passes no credentials at all (§5.2), so anything key-shaped in an MCP
// config was put there by hand: useless to mole, and committed to a repository.
func TestAKeyInTheMCPConfigIsReported(t *testing.T) {
	privateRuntimeDir(t, 0o700)

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, ".mcp.json"), []byte(`{
  "mcpServers": {"mole": {"command": "mole-mcp",
    "env": {"TAVILY_API_KEY": "tvly-dev-0123456789abcdefghij"}}}
}`), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Chdir(dir)

	out, err := doctorRun(t, t.TempDir(), noRuntimePath(t))
	if !strings.Contains(out, "mcp config") {
		t.Fatalf("a key in .mcp.json is not reported:\n%s", out)
	}
	if !strings.Contains(out, "credentials belong in") {
		t.Errorf("the report does not say where a key should live:\n%s", out)
	}
	if err == nil {
		t.Error("doctor succeeded with a credential in a committed file")
	}
}

// TestAnMCPConfigWithNoKeyIsNotReported. A check that flags `MOLE_DB_PATH`
// trains the user to ignore the line.
func TestAnMCPConfigWithNoKeyIsNotReported(t *testing.T) {
	privateRuntimeDir(t, 0o700)

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, ".mcp.json"), []byte(`{
  "mcpServers": {"mole": {"command": "mole-mcp",
    "env": {"MOLE_SOCKET": "/run/user/1000/mole.sock", "MOLE_DB_PATH": "/x/mole.db"}}}
}`), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Chdir(dir)

	out, _ := doctorRun(t, t.TempDir(), noRuntimePath(t))
	if strings.Contains(out, "mcp config") {
		t.Errorf("an ordinary MCP config was flagged as holding a key:\n%s", out)
	}
}

// TestAnUnrelatedMCPConfigIsNotRead. A key belonging to another server is not
// mole's to comment on, and flagging it would make the check noise.
func TestAnUnrelatedMCPConfigIsNotRead(t *testing.T) {
	privateRuntimeDir(t, 0o700)

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, ".mcp.json"), []byte(`{
  "mcpServers": {"other": {"command": "x", "env": {"K": "sk-0123456789abcdefghij"}}}
}`), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Chdir(dir)

	out, _ := doctorRun(t, t.TempDir(), noRuntimePath(t))
	if strings.Contains(out, "mcp config") {
		t.Errorf("another server's key was reported as mole's problem:\n%s", out)
	}
}

// TestDoctorOnAFreshInstallIsNotAFailure.
//
// The database is created by the first run, so a first-time user has none — and
// doctor exited non-zero at them, which tells a setup script the install is broken
// when nothing is wrong yet.
func TestDoctorOnAFreshInstallIsNotAFailure(t *testing.T) {
	t.Setenv("MOLE_CONFIG_DIR", t.TempDir())
	db := filepath.Join(t.TempDir(), "absent", "mole.db")

	stdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	done := make(chan string, 1)
	go func() {
		var b strings.Builder
		_, _ = io.Copy(&b, r)
		done <- b.String()
	}()

	doctorErr := cmdDoctor(context.Background(), db)
	_ = w.Close()
	os.Stdout = stdout
	out := <-done

	if doctorErr != nil {
		t.Errorf("doctor failed on a fresh install: %v", doctorErr)
	}
	if !strings.Contains(out, "nothing is wrong") {
		t.Errorf("doctor does not tell a new user they are fine:\n%s", out)
	}
}

// TestAMissingDatabaseIsStillReportedWhenItShouldExist. The relaxation must not
// hide a database that was deleted out from under a real install.
func TestAMissingDatabaseIsStillReportedWhenItShouldExist(t *testing.T) {
	if !errors.Is(fmt.Errorf("%w at /x", errNoDatabase), errNoDatabase) {
		t.Fatal("errNoDatabase does not match through wrapping")
	}
	if errors.Is(errors.New("permission denied"), errNoDatabase) {
		t.Error("an unrelated failure matches errNoDatabase")
	}
}

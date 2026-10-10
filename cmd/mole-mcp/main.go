// Command mole-mcp forwards MCP over stdio to a running mole daemon.
//
// Deliberately almost nothing: it copies bytes both ways and does not parse,
// buffer, validate or log a frame (§5). An editor spawns it, it forwards, and
// it dies with the editor session while the research continues in the daemon —
// the asymmetry the daemon/shim split exists for.
//
// Its failures surface as "the MCP server would not start", so the one thing it
// invests in is saying what went wrong and what to do about it.
//
//	// .mcp.json — safe to commit; contains no secrets (§5.2)
//	{
//	  "mcpServers": {
//	    "mole": {
//	      "command": "mole-mcp",
//	      "args": ["--socket", "${XDG_RUNTIME_DIR}/mole.sock"]
//	    }
//	  }
//	}
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"time"
)

// DefaultSocketName matches the daemon's.
const DefaultSocketName = "mole.sock"

func defaultSocket() string {
	if dir := os.Getenv("XDG_RUNTIME_DIR"); dir != "" {
		return filepath.Join(dir, DefaultSocketName)
	}
	base := os.Getenv("XDG_DATA_HOME")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return DefaultSocketName
		}
		base = filepath.Join(home, ".local", "share")
	}
	return filepath.Join(base, "mole", DefaultSocketName)
}

func main() {
	socket := flag.String("socket", defaultSocket(), "unix socket of the mole daemon")
	timeout := flag.Duration("connect-timeout", 5*time.Second, "how long to wait for the daemon")
	flag.Usage = func() {
		fmt.Fprint(os.Stderr, `mole-mcp — forwards MCP over stdio to a running mole daemon.

Spawned by an editor or coding agent; not usually run by hand. Start the daemon
first with "mole serve".

`)
		flag.PrintDefaults()
	}
	flag.Parse()

	if err := run(*socket, *timeout); err != nil {
		// stderr, never stdout: stdout is the MCP channel and a diagnostic
		// written there is a protocol error the client cannot parse.
		fmt.Fprintf(os.Stderr, "mole-mcp: %v\n", err)
		os.Exit(1)
	}
}

func run(socket string, timeout time.Duration) error {
	conn, err := net.DialTimeout("unix", socket, timeout)
	if err != nil {
		return dialAdvice(socket, err)
	}
	defer func() { _ = conn.Close() }()

	return pump(conn, os.Stdin, os.Stdout)
}

// dialAdvice turns a connection failure into something actionable.
//
// This runs inside an editor, where the whole failure a user sees is "the MCP
// server would not start". "connect: no such file or directory" does not tell
// them the daemon is not running, and the socket path is the first thing to
// check when two components have to agree on it.
func dialAdvice(socket string, err error) error {
	if _, serr := os.Stat(socket); errors.Is(serr, os.ErrNotExist) {
		return fmt.Errorf("no daemon at %s — start one with: mole serve\n"+
			"(if it is running elsewhere, pass its path with --socket)", socket)
	}
	return fmt.Errorf("could not reach the daemon at %s: %w\n"+
		"the socket exists, so a daemon may have exited without cleaning up; "+
		"try: mole serve", socket, err)
}

// halfCloser is the half-close a unix connection supports.
type halfCloser interface {
	CloseWrite() error
}

// pump copies stdin to the connection and the connection to stdout.
//
// Returns when the DAEMON side finishes, not when stdin does. The difference
// matters: an editor closing its end of stdin means "no more requests", not
// "discard the reply you are mid-way through sending". So stdin EOF half-closes
// the connection — which is what lets the daemon see EOF and end its MCP session
// cleanly — and this keeps reading until the daemon has said everything it
// intends to.
func pump(conn net.Conn, stdin io.Reader, stdout io.Writer) error {
	go func() {
		_, _ = io.Copy(conn, stdin)
		// Half-close so the daemon reads EOF and shuts its session down.
		//
		// Deliberately nothing if the connection cannot half-close. A full Close
		// here would also tear down the direction carrying the reply, truncating
		// a response already on its way — the exact failure this function is
		// arranged to avoid, and one a first version of this code shipped. The
		// only transport in use is a unix socket, which always supports it; the
		// caller's deferred Close still cleans up.
		if hc, ok := conn.(halfCloser); ok {
			_ = hc.CloseWrite()
		}
	}()

	if _, err := io.Copy(stdout, conn); err != nil {
		// A closed connection is how this ends normally, not a failure.
		if errors.Is(err, net.ErrClosed) || errors.Is(err, io.EOF) {
			return nil
		}
		return fmt.Errorf("forwarding from the daemon: %w", err)
	}
	return nil
}

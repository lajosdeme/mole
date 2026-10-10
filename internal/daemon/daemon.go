// Package daemon is the long-lived local server (§5).
//
// Research sessions run past an hour; a bare stdio subprocess dies when the
// coding agent's session closes. So the daemon holds the session, claim and
// budget state, and a disposable shim forwards to it over a unix socket.
//
// The socket is the trust boundary. §5's named-connector design keeps
// credentials out of the tool call by keeping them daemon-side — which only
// helps if reaching the daemon is itself privileged, so the listener setup in
// this file is a security control rather than plumbing (§3.5).
package daemon

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/lajosdeme/mole/internal/session"
)

// Handler serves one accepted connection. Slice 5 supplies the MCP one.
//
// An interface rather than the MCP server directly, so the listener's security
// properties can be tested without a protocol, and so a future transport does
// not have to reimplement them.
type Handler interface {
	Serve(ctx context.Context, conn net.Conn)
}

// HandlerFunc adapts a function to Handler.
type HandlerFunc func(ctx context.Context, conn net.Conn)

func (f HandlerFunc) Serve(ctx context.Context, conn net.Conn) { f(ctx, conn) }

// Server accepts connections on a unix socket and hands them to a Handler.
type Server struct {
	// Socket is the path to listen on.
	Socket string
	// Supervisor runs the sessions. Shutdown stops it and waits for holds to be
	// released before returning.
	Supervisor *session.Supervisor
	Handler    Handler
	Log        *slog.Logger

	// ShutdownGrace bounds how long Serve waits for in-flight connections and
	// running sessions after its context is cancelled. Zero takes a default.
	ShutdownGrace time.Duration

	mu    sync.Mutex
	conns map[net.Conn]struct{}
}

// DefaultShutdownGrace is how long a stopping daemon waits.
//
// Long enough for a session to settle its open reservation, which is what the
// wait is for: exiting early leaves budget neither spent nor available until the
// next boot's sweep reclaims it (§9.4).
const DefaultShutdownGrace = 30 * time.Second

// maxSocketPath is the kernel's limit on a unix socket path.
//
// sun_path is 108 bytes on Linux including the terminator. Kept conservative
// rather than platform-specific: the failure it prevents is a confusing error,
// and refusing a 100-byte path costs nothing.
const maxSocketPath = 104

func (s *Server) logger() *slog.Logger {
	if s.Log != nil {
		return s.Log
	}
	return slog.Default()
}

// Listen creates the socket with the properties §3.5 requires.
//
// Two independent controls, because each alone has a hole:
//
//   - The parent directory is created 0700. net.Listen applies the process
//     umask, so the socket can appear as 0755 and be connectable in the window
//     between Listen and Chmod. A private directory closes that window, because
//     the path is unreachable whatever the socket's own mode says.
//   - The socket is then chmod 0600, so loosening the directory later — by a
//     careless install script, or by putting the socket somewhere shared — does
//     not silently expose it.
//
// XDG_RUNTIME_DIR is already 0700 and user-owned on Linux, which is why §5.2
// puts the socket there; this does not assume it.
func (s *Server) Listen() (net.Listener, error) {
	if s.Socket == "" {
		return nil, errors.New("daemon: no socket path")
	}
	// sun_path is a fixed-size field in the kernel's address struct — 108 bytes on
	// Linux, less on some BSDs — and exceeding it fails as "bind: invalid
	// argument", which says nothing about the length. Worth catching here: the
	// path comes from a flag or XDG_RUNTIME_DIR, and a long one is a
	// configuration mistake the user can fix once they know what it is.
	if n := len(s.Socket); n >= maxSocketPath {
		return nil, fmt.Errorf(
			"daemon: socket path is %d bytes, over the %d the kernel allows: %s\n"+
				"choose a shorter path with --socket", n, maxSocketPath, s.Socket)
	}

	dir := filepath.Dir(s.Socket)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("daemon: socket directory %s: %w", dir, err)
	}
	if err := checkPrivateDir(dir); err != nil {
		return nil, err
	}

	if err := s.clearStaleSocket(); err != nil {
		return nil, err
	}

	ln, err := net.Listen("unix", s.Socket)
	if err != nil {
		return nil, fmt.Errorf("daemon: listen on %s: %w", s.Socket, err)
	}
	if err := os.Chmod(s.Socket, 0o600); err != nil {
		_ = ln.Close()
		return nil, fmt.Errorf("daemon: securing %s: %w", s.Socket, err)
	}
	return ln, nil
}

// checkPrivateDir refuses a socket directory other users can reach.
//
// MkdirAll above does NOT make an existing directory private: it returns nil
// without touching the mode or the owner. So the comment on Listen — that a
// private directory closes the window between net.Listen and Chmod — held only
// for a directory this process created, and the common case is one that already
// exists. `mole serve --socket /tmp/mole.sock` on a 1777 /tmp got a socket at
// 0755 until Chmod ran, world-connectable in between; and a world-writable
// parent also lets another user win the race between clearStaleSocket's Remove
// and net.Listen, and squat the path the shim then connects to.
//
// Refusing rather than chmod-ing: a directory that is not ours to begin with is
// not one to silently take over, and a user who pointed the socket somewhere
// shared should be told rather than have it quietly changed underneath them.
func checkPrivateDir(dir string) error {
	fi, err := os.Stat(dir)
	if err != nil {
		return fmt.Errorf("daemon: socket directory %s: %w", dir, err)
	}
	return checkDirPrivate(dir, fi)
}

// clearStaleSocket removes a socket file left by a dead daemon, and refuses if
// one is still alive.
//
// Told apart by dialing it. Unlinking without checking would silently steal the
// socket from a running daemon: the old one keeps its listener on an unlinked
// inode, every client reaches the new one, and two processes then hold write
// locks on the same database.
func (s *Server) clearStaleSocket() error {
	info, err := os.Stat(s.Socket)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("daemon: %s: %w", s.Socket, err)
	}
	if info.Mode()&os.ModeSocket == 0 {
		return fmt.Errorf("daemon: %s exists and is not a socket", s.Socket)
	}

	conn, derr := net.DialTimeout("unix", s.Socket, 2*time.Second)
	if derr == nil {
		_ = conn.Close()
		return fmt.Errorf("daemon: %s is already served by a running daemon", s.Socket)
	}
	// Nothing accepting: the file outlived its process.
	if err := os.Remove(s.Socket); err != nil {
		return fmt.Errorf("daemon: removing stale socket %s: %w", s.Socket, err)
	}
	s.logger().Info("removed a stale socket left by a previous daemon", "socket", s.Socket)
	return nil
}

// Serve accepts until ctx is cancelled, then shuts down.
//
// Owns the listener: closes it on the way out and removes the socket file, so a
// clean stop does not leave a path that the next start has to reason about.
func (s *Server) Serve(ctx context.Context, ln net.Listener) error {
	if s.Handler == nil {
		return errors.New("daemon: no handler")
	}
	// Owns the listener on every path. Without this an Accept error returns
	// through shutdown with the listener still bound, and the context goroutine
	// below leaks waiting for a cancellation that already happened.
	defer func() { _ = ln.Close() }()

	s.mu.Lock()
	s.conns = map[net.Conn]struct{}{}
	s.mu.Unlock()

	var wg sync.WaitGroup
	accepting := make(chan struct{})

	// Unblock Accept on cancellation. A unix listener has no deadline that
	// interacts with context, so closing it is what makes the loop return.
	go func() {
		<-ctx.Done()
		close(accepting)
		_ = ln.Close()
	}()

	for {
		conn, err := ln.Accept()
		if err != nil {
			select {
			case <-accepting:
				// Expected: the goroutine above closed the listener.
			default:
				return s.shutdown(&wg, fmt.Errorf("daemon: accept: %w", err))
			}
			return s.shutdown(&wg, nil)
		}

		if err := s.authorize(conn); err != nil {
			s.logger().Warn("refused a connection", "err", err)
			_ = conn.Close()
			continue
		}

		s.track(conn)
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer s.untrack(conn)
			defer func() { _ = conn.Close() }()
			s.Handler.Serve(ctx, conn)
		}()
	}
}

// authorize checks the peer is the user who started the daemon (§3.5).
//
// Belt and braces over the socket's file permissions. They are the primary
// control and this is the one that still holds if the socket is moved somewhere
// shared, if a umask surprise widens it, or if a future transport forgets. Root
// is refused like anyone else: the daemon's own uid is the only one that
// matches, because a session started by root would run with root's config and
// credentials.
func (s *Server) authorize(conn net.Conn) error {
	uid, ok, err := peerUID(conn)
	return authorizeUID(uid, ok, err, uint32(os.Getuid()))
}

// authorizeUID is the decision, separated from the syscall that feeds it.
//
// Split out because the refusal — the actual §3.5 control — cannot be reached
// from a test running as one user: every connection a test makes carries its own
// uid. The test that claimed to cover it asserted an ALLOW, on a non-unix
// connection, and never touched this comparison.
func authorizeUID(uid uint32, ok bool, err error, self uint32) error {
	if err != nil {
		// Could not ask. Refusing is the safe default for an authorization check
		// that failed to run.
		return fmt.Errorf("could not read peer credentials: %w", err)
	}
	if !ok {
		// The platform does not support it. The file mode is still in force.
		return nil
	}
	if uid != self {
		return fmt.Errorf("peer uid %d is not %d", uid, self)
	}
	return nil
}

func (s *Server) track(c net.Conn) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.conns[c] = struct{}{}
}

func (s *Server) untrack(c net.Conn) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.conns, c)
}

// shutdown closes live connections, waits for their handlers, and stops the
// supervisor.
//
// The supervisor is stopped LAST and waited on, because that is where the money
// is: its Shutdown returns once every running session has released its budget
// holds. Closing connections first is what lets those handlers notice and
// unblock.
func (s *Server) shutdown(wg *sync.WaitGroup, cause error) error {
	grace := s.ShutdownGrace
	if grace <= 0 {
		grace = DefaultShutdownGrace
	}
	deadline := time.Now().Add(grace)

	s.mu.Lock()
	for c := range s.conns {
		_ = c.Close()
	}
	s.mu.Unlock()

	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(time.Until(deadline)):
		s.logger().Warn("connection handlers did not finish within the shutdown grace")
	}

	if s.Supervisor != nil {
		// Its OWN window, not the handlers' leftovers. Sharing one deadline meant
		// a slow tool handler could consume the whole grace and hand the
		// supervisor an already-expired context — so the participant the wait
		// exists for, the one releasing budget holds, got zero time.
		ctx, cancel := context.WithTimeout(context.Background(), grace)
		defer cancel()
		if err := s.Supervisor.Shutdown(ctx); err != nil {
			s.logger().Warn("supervisor shutdown incomplete", "err", err)
			if cause == nil {
				cause = err
			}
		}
	}

	// No unlink here, deliberately.
	//
	// net.Listen("unix") sets unlink-on-close, so ln.Close() — which happens at
	// the very START of shutdown, in the context goroutine — has already removed
	// the path. Everything after that point runs while the path is free, and this
	// wait is up to ShutdownGrace long because it is waiting for research to
	// release its budget holds.
	//
	// So a second Remove here does not clean up our socket; it deletes whatever
	// is at that path now. On a fast restart that is the SUCCESSOR daemon's live
	// socket: B binds while A is still draining, A unlinks B, B keeps the SQLite
	// single writer while being unreachable, mole-mcp reports "no daemon", and
	// the user starts C — at which point two processes hold write access to one
	// database, the exact disaster clearStaleSocket exists to prevent.
	//
	// The previous version did this unconditionally under a comment claiming it
	// checked ownership first. It did not.
	return cause
}

// -----------------------------------------------------------------------------
// Inspecting an install (for `mole doctor`)
// -----------------------------------------------------------------------------

// SocketState is what `mole doctor` can say about the MCP socket without starting
// a daemon.
type SocketState struct {
	Path string
	// DirMode and DirOwned describe the parent directory — the control that
	// closes the window between net.Listen and Chmod.
	DirMode  os.FileMode
	DirOwned bool
	DirOK    bool
	// Present is whether a socket file exists at all; Live is whether something
	// accepts on it.
	Present bool
	Live    bool
	Mode    os.FileMode
	// Problems are the reasons this install would be refused or is exposed,
	// phrased for a user rather than a caller.
	Problems []string
	// Notes are true statements that are not problems.
	Notes []string
}

// InspectSocket reports the §3.5 properties of a socket path.
//
// It calls the SAME checkPrivateDir the daemon enforces, rather than a second
// hand-written check beside it: `mole doctor` existed for two milestones saying
// socket permissions were "informational until M7 lands", M7 landed, and a check
// that is free to drift from the thing it gates is worth nothing. This is the
// M8 lesson (§12.1's "every crossing is logged" emitted nothing for a milestone)
// applied before it costs anything.
//
// Reports rather than fixes. A directory that is not ours is not ours to chmod,
// and a user who pointed the socket somewhere shared should be told.
func InspectSocket(path string) SocketState {
	st := SocketState{Path: path}
	if path == "" {
		st.Problems = append(st.Problems, "no socket path")
		return st
	}
	if n := len(path); n >= maxSocketPath {
		st.Problems = append(st.Problems, fmt.Sprintf(
			"the path is %d bytes, over the %d the kernel allows; choose a shorter one",
			n, maxSocketPath))
	}

	dir := filepath.Dir(path)
	if fi, err := os.Stat(dir); err == nil {
		st.DirMode = fi.Mode().Perm()
		st.DirOwned, _ = dirOwnedBySelf(fi)
		if err := checkPrivateDir(dir); err != nil {
			st.Problems = append(st.Problems, err.Error())
		} else {
			st.DirOK = true
		}
	} else {
		// Absent is fine: the daemon creates it 0700. Said out loud so a reader
		// does not take a tick as "checked and private".
		st.Notes = append(st.Notes,
			"the directory does not exist yet; mole serve will create it 0700")
		st.DirOK = true
	}

	info, err := os.Stat(path)
	switch {
	case err != nil:
		st.Notes = append(st.Notes, "no socket yet — the daemon is not running")
		return st
	case info.Mode()&os.ModeSocket == 0:
		st.Present = true
		st.Problems = append(st.Problems, "the path exists and is not a socket")
		return st
	}
	st.Present = true
	st.Mode = info.Mode().Perm()
	if modeBitsMeaningful && st.Mode&0o077 != 0 {
		st.Problems = append(st.Problems, fmt.Sprintf(
			"the socket is mode %#o — group or other can connect", st.Mode))
	}

	// Dialled, because a socket file that nothing accepts on is the state a dead
	// daemon leaves and the state a squatter creates, and doctor should not report
	// a stale file as a running daemon.
	conn, derr := net.DialTimeout("unix", path, 2*time.Second)
	if derr == nil {
		_ = conn.Close()
		st.Live = true
	} else {
		st.Notes = append(st.Notes,
			"the socket file exists but nothing accepts on it; mole serve clears it")
	}
	return st
}

// OK reports whether this install would be accepted.
func (s SocketState) OK() bool { return len(s.Problems) == 0 }

// Summary is the one-line form for doctor.
func (s SocketState) Summary() string {
	switch {
	case !s.OK():
		return s.Path + " — " + s.Problems[0]
	case s.Live:
		return fmt.Sprintf("%s, mode %#o, daemon listening", s.Path, s.Mode)
	case s.Present:
		return fmt.Sprintf("%s, mode %#o, no daemon accepting", s.Path, s.Mode)
	default:
		return s.Path + " — not created yet (mole serve makes it 0600 in a 0700 directory)"
	}
}

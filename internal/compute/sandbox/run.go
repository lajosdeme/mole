package sandbox

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"time"
)

// Running code inside the sandbox (§3.6).
//
// Runs one container with the flags Limits.Flags renders, feeds it a script on
// stdin, and returns what the script printed — bounded in size, bounded in
// time, with no route out except stdout.
//
// The caller does not trust the script: it is model-authored code over real
// data, which is the reason for the container. What it trusts is that the
// runtime applies the flags, and slice 6's container test establishes that — no
// network, read-only rootfs, no capabilities, non-root uid, noexec scratch.
//
// What crosses out:
//
// stdout, capped. That is a channel and it is the caller's job to constrain what
// may travel down it — see internal/compute/coderunner, which accepts only
// declared names with numeric values. This package deliberately does not
// interpret the bytes: mixing "run it safely" with "decide what may leave" in
// one place is how one of them ends up implicit.

// ErrNoRuntime is returned when Run is called without a usable runtime.
var ErrNoRuntime = errors.New("sandbox: no usable container runtime")

// Mount is a host path made visible inside the container, always read-only.
//
// There is no writable mount and no field to ask for one. The script's scratch
// space is the tmpfs at /tmp, which does not outlive the container — a writable
// bind mount would be a way for model-authored code to leave something behind
// on the host, and no analysis needs one.
type Mount struct {
	HostPath      string
	ContainerPath string
}

// Spec is one run.
type Spec struct {
	// Image is the interpreter image, and should be pinned by digest. A tag is
	// accepted because a tag is what a user will have pulled.
	Image string
	// Command is the entrypoint, e.g. []string{"python", "-"}. The script
	// arrives on stdin rather than as an argument: an argument goes through the
	// runtime's own quoting, and a script is exactly the kind of text that
	// finds the edges of it.
	Command []string
	Script  string

	Mounts []Mount
	// Env is passed as KEY=VALUE. Nothing secret belongs here — the container
	// has no network, but a credential in an environment variable is still a
	// credential handed to model-authored code.
	Env []string

	Limits         Limits
	MaxOutputBytes int64
}

// ErrMountPath is returned for a path the runtime cannot express in a mount
// spec.
var ErrMountPath = errors.New("sandbox: unusable mount path")

// defaultMaxOutputBytes bounds stdout. Unexported: nothing outside this package
// has a reason to name it.
//
// Generous for a set of statistics and far too small for a data dump, which is
// the distinction that matters: a script that tries to print its input hits
// this rather than filling memory.
const defaultMaxOutputBytes = 256 << 10

// Result is what the run produced.
type Result struct {
	Stdout   string
	Stderr   string
	ExitCode int
	Duration time.Duration

	// TimedOut means the wallclock limit killed it.
	TimedOut bool
	// Cancelled means the CALLER gave up — Ctrl-C, a session deadline, a batch
	// abandoned. Distinct from TimedOut because they mean opposite things about
	// the script: one ran too long, the other never got the chance.
	Cancelled bool
	// Truncated means the script printed more than MaxOutputBytes.
	Truncated bool
}

// Run executes spec in a container and returns what it printed.
//
// A non-zero exit is not an error: model-authored code failing is an ordinary
// outcome, and the caller needs the stderr to decide what to do. The error
// return is for the run not happening at all.
func (r Report) Run(ctx context.Context, spec Spec) (Result, error) {
	if !r.Usable {
		return Result{}, fmt.Errorf("%w: %s", ErrNoRuntime, r.Detail)
	}
	if strings.TrimSpace(spec.Image) == "" {
		return Result{}, errors.New("sandbox: no image")
	}
	if len(spec.Command) == 0 {
		return Result{}, errors.New("sandbox: no command")
	}

	limits := spec.Limits
	if limits.Wallclock <= 0 {
		limits.Wallclock = DefaultLimits().Wallclock
	}
	maxOut := spec.MaxOutputBytes
	if maxOut <= 0 {
		maxOut = defaultMaxOutputBytes
	}

	name, err := containerName()
	if err != nil {
		return Result{}, err
	}

	args := []string{"run", "--rm", "--name", name, "--interactive"}
	args = append(args, limits.Flags()...)
	for _, m := range spec.Mounts {
		// A colon cannot appear in either half: the spec is
		// host:container:options, so a data folder named `2024:Q1` produced
		//
		//	docker: invalid spec: /…/2024:Q1/connector.sqlite:/data/…:ro: too many colons
		//
		// The runtime refuses rather than mounting the wrong path, so this was
		// never a hole — but the failure reached the user as a raw runtime error
		// nobody could act on. Refused here, with the fix in the message.
		if strings.ContainsAny(m.HostPath, ":") || strings.ContainsAny(m.ContainerPath, ":") {
			return Result{}, fmt.Errorf("%w: %q contains a colon, which a container "+
				"mount specification cannot express; move or rename it",
				ErrMountPath, m.HostPath)
		}
		// Always :ro. Not a default that a caller can override — the struct has
		// no field for it.
		args = append(args, "--volume", m.HostPath+":"+m.ContainerPath+":ro")
	}
	for _, e := range spec.Env {
		args = append(args, "--env", e)
	}
	args = append(args, spec.Image)
	args = append(args, spec.Command...)

	// The wallclock limit is enforced here rather than inside the container,
	// because a limit the script could choose to ignore is not a limit.
	runCtx, cancel := context.WithTimeout(ctx, limits.Wallclock)
	defer cancel()

	cmd := exec.CommandContext(runCtx, string(r.Runtime), args...)
	cmd.Stdin = strings.NewReader(spec.Script)
	// Without this, killing the CLI leaves Run blocked until every pipe holder
	// closes — measured at 30 seconds on a cancelled run, before a further 10 in
	// killContainer. With a worker pool each cancelled lead paid it.
	cmd.WaitDelay = 2 * time.Second

	var out, errBuf cappedBuffer
	out.limit, errBuf.limit = maxOut, 8<<10
	cmd.Stdout = &out
	cmd.Stderr = &errBuf

	start := time.Now()
	runErr := cmd.Run()
	res := Result{
		Stdout:    out.String(),
		Stderr:    errBuf.String(),
		Duration:  time.Since(start),
		Truncated: out.truncated,
	}

	if runCtx.Err() != nil {
		// The wallclock limit and the caller giving up are different outcomes and
		// were reported as the same one, so Ctrl-C surfaced to the user as "the
		// analysis exceeded its wallclock limit".
		res.TimedOut = errors.Is(runCtx.Err(), context.DeadlineExceeded)
		res.Cancelled = !res.TimedOut

		// Killing the CLI does not stop the container: `docker run --rm` leaves
		// it running when its client goes away, so the name exists to be able to
		// reach it. Detached, because a cancelled caller must not then wait on a
		// ten-second cleanup — with a worker pool that cost is paid per lead.
		go killContainer(string(r.Runtime), name)
	}

	var ee *exec.ExitError
	switch {
	case runErr == nil:
		res.ExitCode = 0
	case errors.As(runErr, &ee):
		res.ExitCode = ee.ExitCode()
	case res.TimedOut:
		res.ExitCode = -1
	default:
		return res, fmt.Errorf("sandbox: run: %w", runErr)
	}
	return res, nil
}

// HasImage reports whether an image is already pulled.
//
// Separate from Detect because it answers a different question: a usable runtime
// with no interpreter image is ready for everything except the one thing that
// needs one, and `doctor` should say which of the two is missing. Never pulls —
// downloading a few hundred megabytes is a decision for whoever runs the
// command, not a side effect of asking.
func (r Report) HasImage(ctx context.Context, image string) bool {
	if !r.Usable || strings.TrimSpace(image) == "" {
		return false
	}
	ctx, cancel := context.WithTimeout(ctx, probeTimeout)
	defer cancel()
	return exec.CommandContext(ctx, string(r.Runtime), "image", "inspect", image).Run() == nil
}

func killContainer(runtime, name string) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = exec.CommandContext(ctx, runtime, "kill", name).Run()
}

// containerName is unique per run, so a timeout can reach the right container.
func containerName() (string, error) {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", fmt.Errorf("sandbox: name: %w", err)
	}
	return "mole-compute-" + hex.EncodeToString(b[:]), nil
}

// cappedBuffer stops reading after limit bytes and records that it did.
//
// A cap rather than a truncation after the fact: a script that prints its input
// would otherwise be held in memory in full before anybody decided it was too
// large, which is the same mistake as reading a result set before deciding
// whether it may cross.
type cappedBuffer struct {
	buf       bytes.Buffer
	limit     int64
	written   int64
	truncated bool
}

func (c *cappedBuffer) Write(p []byte) (int, error) {
	room := c.limit - c.written
	if room <= 0 {
		c.truncated = true
		// Reporting the full length keeps the writer from treating this as a
		// short write and erroring; the bytes are dropped on purpose.
		return len(p), nil
	}
	if int64(len(p)) > room {
		c.truncated = true
		c.buf.Write(p[:room])
		c.written = c.limit
		return len(p), nil
	}
	c.buf.Write(p)
	c.written += int64(len(p))
	return len(p), nil
}

func (c *cappedBuffer) String() string { return c.buf.String() }

var _ io.Writer = (*cappedBuffer)(nil)

//go:build !windows

package daemon

import (
	"fmt"
	"os"
	"syscall"
)

// modeBitsMeaningful: unix permission bits are the access control here.
const modeBitsMeaningful = true

// checkDirPrivate enforces the unix contract: mode 0700 and owned by this uid.
func checkDirPrivate(dir string, fi os.FileInfo) error {
	if perm := fi.Mode().Perm(); perm&0o077 != 0 {
		return fmt.Errorf("daemon: socket directory %s is mode %#o — group or other "+
			"can reach the socket\nuse a private directory, e.g. --socket $XDG_RUNTIME_DIR/mole.sock",
			dir, perm)
	}
	if st, ok := fi.Sys().(*syscall.Stat_t); ok {
		if self := uint32(os.Getuid()); st.Uid != self {
			return fmt.Errorf("daemon: socket directory %s is owned by uid %d, not %d",
				dir, st.Uid, self)
		}
	}
	return nil
}

// dirOwnedBySelf reports whether fi is owned by the current uid, and whether
// the platform could tell.
func dirOwnedBySelf(fi os.FileInfo) (owned, known bool) {
	st, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return true, false
	}
	return st.Uid == uint32(os.Getuid()), true
}

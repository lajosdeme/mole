//go:build windows

package daemon

import "os"

// modeBitsMeaningful: Windows synthesises permission bits; ACLs are the control.
const modeBitsMeaningful = false

// checkDirPrivate is a no-op on Windows. os.FileInfo.Mode().Perm() is
// synthesised (0777 for every directory) and carries no ACL information, so the
// unix 0o077 test would refuse every path. The per-user %LOCALAPPDATA% tree the
// default socket lives under is ACL-restricted to the owning account, which is
// the control the daemon relies on here.
func checkDirPrivate(string, os.FileInfo) error { return nil }

// dirOwnedBySelf cannot be answered from os.FileInfo on Windows.
func dirOwnedBySelf(os.FileInfo) (owned, known bool) { return true, false }

// SPDX-License-Identifier: Apache-2.0
//go:build localembed && (linux || darwin)

package local

import (
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

// downloadLockName is the lock file held for the duration of Ensure's
// exists-check-then-download sequence on a model's cache directory.
const downloadLockName = ".lock"

// lockDir takes a BLOCKING exclusive advisory lock on dir and returns the
// closer that releases it.
//
// Unlike the data-dir lock in cmd/rostam-server/mcplock_unix.go — which
// refuses a second holder immediately, because two processes mapping the same
// live store would corrupt it — two processes racing to download the same
// model must simply wait their turn: the loser should block until the winner
// finishes, then find the artifacts already cached and skip the download
// entirely. So this uses unix.LOCK_EX with no LOCK_NB.
//
// The lock is flock(2)-based: advisory, released automatically by the kernel
// if the process dies while holding it, and per-open-file-description (so it
// does not protect against the same process locking twice, which Ensure's
// single call site doesn't do).
func lockDir(dir string) (func() error, error) {
	path := filepath.Join(dir, downloadLockName)
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, fmt.Errorf("opening lock file %s: %w", path, err)
	}
	if err := unix.Flock(int(f.Fd()), unix.LOCK_EX); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("locking %s: %w", path, err)
	}
	// Closing the file drops the flock. The lock file itself is left behind on
	// purpose: unlinking it would let a second process create a fresh one and
	// lock that instead, which is the same as having no lock at all.
	return f.Close, nil
}

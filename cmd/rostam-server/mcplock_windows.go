// SPDX-License-Identifier: Apache-2.0
//go:build windows

package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/sys/windows"
)

// mcpLockName is the lock file an embedded `mcp` session holds inside its data
// directory for the lifetime of the process. Same name as the unix variant:
// the two never coexist, but a shared data directory copied between platforms
// should not grow a second lock file.
const mcpLockName = ".mcp.lock"

// Win32 LockFileEx flags. golang.org/x/sys/windows declares the call but not
// these constants, so they are spelled out here from the Win32 documentation.
const (
	lockfileFailImmediately = 0x00000001
	lockfileExclusiveLock   = 0x00000002
)

// lockDataDir takes an exclusive lock on dir and returns the closer that
// releases it. See the unix variant for why an embedded data directory needs
// one at all: two rostam-server mcp processes on the same -data dir both map
// the same cache files and both believe they own them, which corrupts the store
// rather than failing.
//
// The Windows implementation is LockFileEx over one byte of the lock file, with
// LOCKFILE_FAIL_IMMEDIATELY so a second holder is refused instead of blocking.
// Like flock(2), the lock is tied to the open handle: closing it releases the
// lock, and the kernel closes the handle if this process dies, so a crash
// cannot leave the data dir permanently unopenable.
//
// This used to be the no-op !unix stub, which meant the single-writer contract
// documented in mcp.go was simply not enforced on Windows — a platform the
// release builds ship a binary for.
func lockDataDir(dir string) (func() error, error) {
	path := filepath.Join(dir, mcpLockName)
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, fmt.Errorf("opening lock file %s: %w", path, err)
	}
	// A zero Overlapped locks from offset 0; one byte is enough, since every
	// participant locks exactly the same range.
	var ol windows.Overlapped
	if err := windows.LockFileEx(windows.Handle(f.Fd()),
		lockfileExclusiveLock|lockfileFailImmediately, 0, 1, 0, &ol); err != nil {
		_ = f.Close()
		return nil, lockFileErr(path, err)
	}
	// Closing the handle drops the lock. The lock file itself is left behind on
	// purpose: deleting it would let a second process create a fresh one and
	// lock that instead, which is the same as having no lock at all.
	return f.Close, nil
}

// lockFileErr classifies a LockFileEx failure. Only genuine contention carries
// errDataDirBusy — runMcpCmd turns that sentinel into "another rostam-server
// mcp process is using this data directory", and sending someone whose real
// problem is a bad path hunting for a second process wastes their time.
func lockFileErr(path string, err error) error {
	if errors.Is(err, windows.ERROR_LOCK_VIOLATION) || errors.Is(err, windows.ERROR_SHARING_VIOLATION) {
		return fmt.Errorf("%w: %w", errDataDirBusy, err)
	}
	return fmt.Errorf("locking %s: %w", path, err)
}

// SPDX-License-Identifier: Apache-2.0
//go:build unix

package main

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// TestLockDataDirExcludesSecondHolder is the whole point of the lock: a second
// embedded mcp session must be refused the data dir the first one holds. flock
// is per open-file-description, so two opens in one process contend exactly as
// two processes would.
func TestLockDataDirExcludesSecondHolder(t *testing.T) {
	dir := t.TempDir()

	unlock, err := lockDataDir(dir)
	if err != nil {
		t.Fatalf("first lock: %v", err)
	}

	if _, err := lockDataDir(dir); err == nil {
		t.Fatal("second lock succeeded; the data dir is not actually exclusive")
	}

	// Releasing must hand the directory over, not leave it wedged for good — a
	// finished session cannot make the dir unusable by the next one.
	if err := unlock(); err != nil {
		t.Fatalf("unlock: %v", err)
	}
	unlock2, err := lockDataDir(dir)
	if err != nil {
		t.Fatalf("relock after release: %v", err)
	}
	if err := unlock2(); err != nil {
		t.Fatalf("second unlock: %v", err)
	}
}

// TestLockDataDirDistinctDirs guards against a lock that is accidentally global
// (e.g. keyed on something other than the directory): two different data dirs
// are independent stores and must both be openable at once.
func TestLockDataDirDistinctDirs(t *testing.T) {
	a, b := t.TempDir(), t.TempDir()
	ua, err := lockDataDir(a)
	if err != nil {
		t.Fatalf("lock a: %v", err)
	}
	defer func() { _ = ua() }()
	ub, err := lockDataDir(b)
	if err != nil {
		t.Fatalf("lock b while a is held: %v", err)
	}
	defer func() { _ = ub() }()
}

// TestLockDataDirMissingDir reports a failure to create the lock file rather
// than silently proceeding unlocked -- and reports it as what it is. Only a
// real flock conflict may carry errDataDirBusy: runMcpCmd turns that sentinel
// into "another rostam-server mcp process is using this data directory", and
// telling someone with a mistyped path to go hunt for a second process sends
// them the wrong way entirely.
func TestLockDataDirMissingDir(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "does-not-exist")
	_, err := lockDataDir(missing)
	if err == nil {
		t.Fatal("locking a nonexistent directory should fail")
	}
	if errors.Is(err, errDataDirBusy) {
		t.Fatalf("a missing directory must not be reported as a lock conflict: %v", err)
	}
}

// TestLockDataDirConflictIsBusy is the other half of that contract: a genuine
// second holder DOES carry errDataDirBusy, so the actionable "use -connect"
// message still reaches the one caller it is meant for.
func TestLockDataDirConflictIsBusy(t *testing.T) {
	dir := t.TempDir()
	unlock, err := lockDataDir(dir)
	if err != nil {
		t.Fatalf("first lock: %v", err)
	}
	defer func() { _ = unlock() }()

	_, err = lockDataDir(dir)
	if !errors.Is(err, errDataDirBusy) {
		t.Fatalf("a real second holder must report errDataDirBusy, got %v", err)
	}
}

// TestClaimDataDirCreatesMissingPath is the regression the lock introduced:
// -data pointing at a path that does not exist yet is an ordinary first run
// and must succeed. Taking the lock moved a file open ahead of the engine's
// own directory creation, so for one revision this failed ENOENT -- and
// reported it as a lock conflict.
func TestClaimDataDirCreatesMissingPath(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "nested", "does-not-exist-yet")

	unlock, err := claimDataDir(dir)
	if err != nil {
		t.Fatalf("claiming a not-yet-existing -data path: %v", err)
	}
	defer func() { _ = unlock() }()

	if fi, serr := os.Stat(dir); serr != nil || !fi.IsDir() {
		t.Fatalf("the -data directory should have been created: stat = %v, err = %v", fi, serr)
	}
}

// TestClaimDataDirHeapModeClaimsNothing checks the "" (heap) case leaves no
// lock file anywhere and still returns a usable closer -- heap mode has no
// directory to be a second writer to.
func TestClaimDataDirHeapModeClaimsNothing(t *testing.T) {
	unlock, err := claimDataDir("")
	if err != nil {
		t.Fatalf("heap mode should claim nothing and succeed: %v", err)
	}
	if unlock == nil {
		t.Fatal("heap mode must still return a callable release func")
	}
	if err := unlock(); err != nil {
		t.Fatalf("heap-mode release: %v", err)
	}
}

// TestClaimDataDirConflictSurvivesTheHelper makes sure the sentinel is not lost
// on the way through claimDataDir: runMcpCmd switches on it, so a wrapper
// that flattened it would silently downgrade the actionable message.
func TestClaimDataDirConflictSurvivesTheHelper(t *testing.T) {
	dir := t.TempDir()
	unlock, err := claimDataDir(dir)
	if err != nil {
		t.Fatalf("first claim: %v", err)
	}
	defer func() { _ = unlock() }()

	if _, err := claimDataDir(dir); !errors.Is(err, errDataDirBusy) {
		t.Fatalf("conflict through claimDataDir must stay errDataDirBusy, got %v", err)
	}
}

// TestClaimDataDirUncreatablePathIsNotBusy pins the misdiagnosis directly: a
// path that cannot be created (a file sits where the directory should go) is
// an I/O error, never a conflict.
func TestClaimDataDirUncreatablePathIsNotBusy(t *testing.T) {
	base := t.TempDir()
	blocker := filepath.Join(base, "iam-a-file")
	if err := os.WriteFile(blocker, []byte("x"), 0o600); err != nil {
		t.Fatalf("write blocker: %v", err)
	}

	_, err := claimDataDir(filepath.Join(blocker, "under-a-file"))
	if err == nil {
		t.Fatal("creating a directory under a regular file should fail")
	}
	if errors.Is(err, errDataDirBusy) {
		t.Fatalf("an uncreatable path must not be reported as a lock conflict: %v", err)
	}
}

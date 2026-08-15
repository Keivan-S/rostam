// SPDX-License-Identifier: Apache-2.0
//go:build unix

package main

import (
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
// than silently proceeding unlocked.
func TestLockDataDirMissingDir(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "does-not-exist")
	if _, err := lockDataDir(missing); err == nil {
		t.Fatal("locking a nonexistent directory should fail")
	}
}

// SPDX-License-Identifier: Apache-2.0
//go:build linux

package cache

import (
	"path/filepath"
	"testing"
)

func TestMmapFileRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.dat")
	f, region, err := mmapFile(path, 4096, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(region) != 4096 {
		t.Fatalf("region size = %d, want 4096", len(region))
	}
	copy(region, []byte("HELLO"))
	if err := msync(region); err != nil {
		t.Fatalf("msync: %v", err)
	}
	if err := munmapAndClose(f, region); err != nil {
		t.Fatalf("close: %v", err)
	}

	// Reopen, verify marker survives.
	f2, region2, err := mmapFile(path, 4096, false)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = munmapAndClose(f2, region2) }()
	if string(region2[:5]) != "HELLO" {
		t.Errorf("marker = %q, want HELLO", region2[:5])
	}
}

func TestMmapFileCreatesIfMissing(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "new.dat")
	f, region, err := mmapFile(path, 1024, false)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = munmapAndClose(f, region) }()
	if len(region) != 1024 {
		t.Errorf("region = %d bytes, want 1024", len(region))
	}
}

func TestMmapFileSizesExisting(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "grow.dat")
	f, region, err := mmapFile(path, 512, false)
	if err != nil {
		t.Fatal(err)
	}
	_ = munmapAndClose(f, region)
	f2, region2, err := mmapFile(path, 2048, false)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = munmapAndClose(f2, region2) }()
	if len(region2) != 2048 {
		t.Errorf("region = %d, want 2048", len(region2))
	}
}

func TestMlockGracefulFailure(t *testing.T) {
	// mlock often fails in CI due to RLIMIT_MEMLOCK; the function must
	// log + continue, not error.
	dir := t.TempDir()
	path := filepath.Join(dir, "mlock.dat")
	f, region, err := mmapFile(path, 4096, true)
	if err != nil {
		t.Fatalf("mmapFile with mlock=true should not error: %v", err)
	}
	defer func() { _ = munmapAndClose(f, region) }()
}

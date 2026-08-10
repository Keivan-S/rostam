//go:build cgo

// SPDX-License-Identifier: Apache-2.0

package wasm

import (
	"errors"
	"os"
	"testing"
)

func TestCompileAcceptsAllowedImports(t *testing.T) {
	bytes, err := os.ReadFile("testdata/incr.wasm")
	if err != nil {
		t.Fatal(err)
	}
	m, err := Compile(bytes)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	if m == nil {
		t.Fatal("Compile returned nil module")
	}
	defer func() { _ = m.Close() }()
}

func TestCompileRejectsBannedImports(t *testing.T) {
	// Hand-built WASM that imports wasi_snapshot_preview1.clock_time_get.
	// Magic + version + import "wasi_snapshot_preview1" "clock_time_get"
	// with a (param i32 i32) -> i32 signature. The implementer can either
	// hand-construct these 50 bytes or precompile a .wat file and check
	// it in as testdata/clock.wasm. The test demands a clear rejection
	// from the determinism gate.
	bytes, err := os.ReadFile("testdata/clock.wasm")
	if err != nil {
		t.Skip("testdata/clock.wasm not present; skipping banned-import test")
		return
	}
	_, err = Compile(bytes)
	if err == nil {
		t.Fatal("Compile accepted a module importing clock_time_get")
	}
	if !errors.Is(err, ErrBannedImport) {
		t.Fatalf("err = %v, want ErrBannedImport", err)
	}
}

func TestCompileRejectsEmptyBytes(t *testing.T) {
	_, err := Compile(nil)
	if err == nil {
		t.Fatal("Compile(nil): expected error")
	}
}

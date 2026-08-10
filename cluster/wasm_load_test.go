// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/ops"
)

func TestLoadWASMModulesFromConfig(t *testing.T) {
	dir := t.TempDir()

	wasmBytes, err := os.ReadFile("../wasm/testdata/incr.wasm")
	if err != nil {
		t.Fatalf("read testdata/incr.wasm: %v", err)
	}

	cfgs := []WASMModuleConfig{
		{
			Name:       "incr",
			Kind:       ops.OpReadWrite,
			Bytes:      wasmBytes,
			ExportName: "incr",
		},
	}

	loaded, err := loadWASMModules(dir, cfgs, nil, newWASMState())
	if err != nil {
		t.Fatalf("loadWASMModules: %v", err)
	}
	if len(loaded) != 1 {
		t.Fatalf("loaded len = %d, want 1", len(loaded))
	}
	if loaded[0].Name != "incr" {
		t.Fatalf("loaded[0].Name = %q, want incr", loaded[0].Name)
	}

	blobPath := blobPathFor(t, dir, wasmBytes)
	if _, err := os.Stat(blobPath); err != nil {
		t.Errorf("expected the persisted module blob at %s: %v", blobPath, err)
	}
	// The per-name module file is gone; only the metadata sidecar keeps the name.
	if _, err := os.Stat(filepath.Join(dir, "wasm", "incr.wasm")); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("a name-addressed <op>.wasm was written; the blob store replaced it (stat err = %v)", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "wasm", "incr.json")); err != nil {
		t.Errorf("expected the metadata sidecar: %v", err)
	}
}

func TestLoadWASMModulesRejectsConflict(t *testing.T) {
	dir := t.TempDir()
	wasmBytes := []byte("\x00asm\x01\x00\x00\x00") // minimal magic header

	cfgs := []WASMModuleConfig{
		{Name: "dup", Kind: ops.OpReadWrite, Bytes: wasmBytes, ExportName: "run"},
		{Name: "dup", Kind: ops.OpReadOnly, Bytes: wasmBytes, ExportName: "run"},
	}

	_, err := loadWASMModules(dir, cfgs, nil, newWASMState())
	if err == nil {
		t.Fatal("expected error for duplicate Name, got nil")
	}
	if !strings.Contains(err.Error(), "dup") {
		t.Errorf("err message %q does not mention duplicate name 'dup'", err.Error())
	}
}

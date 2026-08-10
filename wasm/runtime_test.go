//go:build cgo

// SPDX-License-Identifier: Apache-2.0

package wasm

import (
	"os"
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
)

func TestRuntimeInvokeIncrModule(t *testing.T) {
	wasmBytes, err := os.ReadFile("testdata/incr.wasm")
	if err != nil {
		t.Fatal(err)
	}
	m, err := Compile(wasmBytes)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	defer func() { _ = m.Close() }()

	rt, err := NewRuntime()
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	defer func() { _ = rt.Close() }()

	id, err := rt.AddModule(m, "apply", 0)
	if err != nil {
		t.Fatalf("AddModule: %v", err)
	}

	c, err := cache.New(cache.DefaultConfig())
	if err != nil {
		t.Fatalf("cache.New: %v", err)
	}
	defer func() { _ = c.Close() }()
	tx := ops.NewTxContext(c)

	_, err = rt.Invoke(id, tx, nil)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
}

func TestRuntimeInvokeUnknownOpReturnsError(t *testing.T) {
	rt, err := NewRuntime()
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	defer func() { _ = rt.Close() }()

	c, err := cache.New(cache.DefaultConfig())
	if err != nil {
		t.Fatalf("cache.New: %v", err)
	}
	defer func() { _ = c.Close() }()
	tx := ops.NewTxContext(c)

	_, err = rt.Invoke(ModuleIDFor([]byte("no such module"), "apply", 0), tx, nil)
	if err == nil {
		t.Fatal("expected error for unknown op, got nil")
	}
}

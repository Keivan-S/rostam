// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"os"
	"testing"

	"github.com/rostamlabs/rostam/ops"
)

func TestNodeRegisterWASMViaCall(t *testing.T) {
	n := newTestNode(t, 1)

	wasmBytes, err := os.ReadFile("../wasm/testdata/incr.wasm")
	if err != nil {
		t.Fatalf("read incr.wasm: %v", err)
	}

	// The CLIENT EDGE carries the marker AND the module: the node stores the bytes
	// and pushes them to the cluster before it broadcasts the bare marker.
	payload := ops.EncodeWASMRegistrationRequest(ops.WASMRegistration{
		Name:       "wasm_incr",
		Kind:       ops.OpReadWrite,
		Blob:       ops.WASMBlobFingerprint(wasmBytes),
		ExportName: "apply",
	}, wasmBytes)

	if _, err := n.Call("__register_wasm__", payload); err != nil {
		t.Fatalf("Call __register_wasm__: %v", err)
	}

	if _, err := n.Call("wasm_incr", stdArgs([]byte("k"))); err != nil {
		t.Fatalf("Call wasm_incr after registration: %v", err)
	}
}

func TestNodeReloadsWASMOnRestart(t *testing.T) {
	dir := t.TempDir()

	wasmBytes, err := os.ReadFile("../wasm/testdata/incr.wasm")
	if err != nil {
		t.Fatalf("read incr.wasm: %v", err)
	}

	// First node: register the module via Raft.
	n1 := newTestNodeAt(t, dir, 1)

	// The CLIENT EDGE carries the marker AND the module: the node stores the bytes
	// and pushes them to the cluster before it broadcasts the bare marker.
	payload := ops.EncodeWASMRegistrationRequest(ops.WASMRegistration{
		Name:       "wasm_incr",
		Kind:       ops.OpReadWrite,
		Blob:       ops.WASMBlobFingerprint(wasmBytes),
		ExportName: "apply",
	}, wasmBytes)

	if _, err := n1.Call("__register_wasm__", payload); err != nil {
		t.Fatalf("n1 Call __register_wasm__: %v", err)
	}

	// Close n1 explicitly (Cleanup also calls Close, but idempotent).
	if err := n1.Close(); err != nil {
		t.Fatalf("n1 Close: %v", err)
	}

	// Second node on the same dir — must reload from disk.
	n2 := newTestNodeAt(t, dir, 1)

	if _, err := n2.Call("wasm_incr", stdArgs([]byte("k"))); err != nil {
		t.Fatalf("n2 Call wasm_incr after restart: %v", err)
	}
}

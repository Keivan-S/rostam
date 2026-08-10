// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"os"
	"testing"

	"github.com/rostamlabs/rostam/ops"
)

// encodeKey wraps a raw key in the [keyLen u16][key] wire format every WASM op's
// args must use (ops.WASMKeyExtractorHandle), so the cluster can route the call
// to the correct shard.
func encodeKey(key []byte) []byte { return ops.EncodeKeyArgs(key) }

// TestEmbeddedStoreRegisterWASMRoundtrip registers the incr WASM module via
// the Embedded store, then calls the registered op to verify it executes.
func TestEmbeddedStoreRegisterWASMRoundtrip(t *testing.T) {
	wasmBytes, err := os.ReadFile("wasm/testdata/incr.wasm")
	if err != nil {
		t.Skip("wasm/testdata/incr.wasm not found:", err)
	}

	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)

	ctx := context.Background()
	reg := WASMRegistration{
		Name:       "wasm_incr",
		Kind:       ops.OpReadWrite,
		Blob:       ops.WASMBlobFingerprint(wasmBytes),
		ExportName: "apply",
		MaxFuel:    0,
	}
	if _, err := s.RegisterWASM(ctx, reg, wasmBytes); err != nil {
		t.Fatalf("RegisterWASM: %v", err)
	}

	// incr.wasm ignores the value at the key but the cluster still needs a
	// routing key in [keyLen u16][key] format to dispatch to the right shard.
	if _, err = s.Call(ctx, "wasm_incr", encodeKey([]byte("k"))); err != nil {
		t.Fatalf("Call wasm_incr: %v", err)
	}
}

// TestDirectStoreRegisterWASMRoundtrip registers the incr WASM module via the
// Direct store and invokes the op.
func TestDirectStoreRegisterWASMRoundtrip(t *testing.T) {
	wasmBytes, err := os.ReadFile("wasm/testdata/incr.wasm")
	if err != nil {
		t.Skip("wasm/testdata/incr.wasm not found:", err)
	}

	s := newSingleDirect(t)
	ctx := context.Background()

	reg := WASMRegistration{
		Name:       "wasm_incr",
		Kind:       ops.OpReadWrite,
		Blob:       ops.WASMBlobFingerprint(wasmBytes),
		ExportName: "apply",
		MaxFuel:    0,
	}
	if _, err := s.RegisterWASM(ctx, reg, wasmBytes); err != nil {
		t.Fatalf("RegisterWASM: %v", err)
	}

	// incr.wasm ignores the value; pass a std-encoded key so any extractor works.
	if _, err = s.Call(ctx, "wasm_incr", encodeKey([]byte("k"))); err != nil {
		t.Fatalf("Call wasm_incr: %v", err)
	}
}

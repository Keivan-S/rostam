//go:build cgo

// SPDX-License-Identifier: Apache-2.0

package wasm

import (
	"os"
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
)

// BenchmarkInvokeStub measures the pure WASM call overhead. The
// testdata/incr.wasm module just calls set_result(0,0) and returns 0
// — no cache access, no real work. This is the floor for any WASM op.
func BenchmarkInvokeStub(b *testing.B) {
	bs, err := os.ReadFile("testdata/incr.wasm")
	if err != nil {
		b.Fatal(err)
	}
	m, err := Compile(bs)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = m.Close() }()

	rt, err := NewRuntime()
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = rt.Close() }()

	id, err := rt.AddModule(m, "apply", 0)
	if err != nil {
		b.Fatal(err)
	}

	c, err := cache.New(cache.DefaultConfig())
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	tx := ops.NewTxContext(c)
	b.ResetTimer()
	for b.Loop() {
		_, _ = rt.Invoke(id, tx, nil)
	}
}

// BenchmarkNativeNoop is the apples-to-apples Go baseline: a registered
// Go op handler that does the same set_result-equivalent — return
// empty bytes, nil error. Subtract this from BenchmarkInvokeStub to get
// the wazero call overhead alone.
func BenchmarkNativeNoop(b *testing.B) {
	c, err := cache.New(cache.DefaultConfig())
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	reg := ops.NewRegistry()
	_ = reg.Register("noop", ops.OpReadOnly, func(_ *ops.TxContext, _ []byte) ([]byte, error) {
		return nil, nil
	})
	handler, _, _, _ := reg.Lookup("noop")
	tx := ops.NewTxContext(c)
	b.ResetTimer()
	for b.Loop() {
		_, _ = handler(tx, nil)
	}
}

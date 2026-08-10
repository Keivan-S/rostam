// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"encoding/binary"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

func newDirectForTest(t *testing.T) Store {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	s, err := NewDirect(DirectConfig{Ops: reg, DataDir: t.TempDir()})
	if err != nil {
		t.Fatalf("NewDirect: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

// readCounter reads an "incr" counter without mutating it (delta 0) and decodes
// the int64 big-endian result the incr handler returns.
func readCounter(t *testing.T, s Store, key []byte) int64 {
	t.Helper()
	got, err := s.Call(context.Background(), "incr", ops.EncodeIncrArgs(key, 0))
	if err != nil {
		t.Fatalf("read counter %q: %v", key, err)
	}
	if len(got) < 8 {
		t.Fatalf("incr result %d bytes, want >=8", len(got))
	}
	return int64(binary.BigEndian.Uint64(got))
}

// TestDirectPerShardSameKeySerialized proves the per-shard op lock keeps
// concurrent read-modify-write on the SAME key atomic: G*M concurrent incrs of
// one key must end at exactly G*M with no lost updates (all collide on that
// key's single shard lock). Run with -race for the data-race check too.
func TestDirectPerShardSameKeySerialized(t *testing.T) {
	s := newDirectForTest(t)
	ctx := context.Background()
	key := []byte("counter")
	const G, M = 16, 500

	var wg sync.WaitGroup
	for g := 0; g < G; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < M; i++ {
				if _, err := s.Call(ctx, "incr", ops.EncodeIncrArgs(key, 1)); err != nil {
					t.Errorf("incr: %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()

	if v := readCounter(t, s, key); v != int64(G*M) {
		t.Fatalf("counter = %d, want %d (lost updates ⇒ per-shard serialization broken)", v, G*M)
	}
}

// TestDirectPerShardDistinctKeysConcurrent hammers many DISTINCT keys (which
// fan out across shards) concurrently — the case the per-shard lock is meant to
// parallelize. Each key gets M incrs and must end at exactly M. Under -race this
// proves different-shard ops running in parallel never corrupt each other.
func TestDirectPerShardDistinctKeysConcurrent(t *testing.T) {
	s := newDirectForTest(t)
	ctx := context.Background()
	const K, M = 256, 64

	keys := make([][]byte, K)
	for i := range keys {
		keys[i] = []byte(fmt.Sprintf("player:%d", i))
	}

	var wg sync.WaitGroup
	for i := 0; i < K; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for m := 0; m < M; m++ {
				if _, err := s.Call(ctx, "incr", ops.EncodeIncrArgs(keys[i], 1)); err != nil {
					t.Errorf("incr key %d: %v", i, err)
					return
				}
			}
		}(i)
	}
	wg.Wait()

	for _, k := range keys {
		if v := readCounter(t, s, k); v != int64(M) {
			t.Fatalf("key %q = %d, want %d", k, v, M)
		}
	}
}

// TestDirectCrossShardOpExcludesEveryOp proves a CROSS-SHARD op (e.g. a WASM op,
// whose handler may touch arbitrary keys) takes the all-shards barrier rather
// than a single shard lock: while it runs, no other read-write op runs — not
// even another cross-shard op with a DIFFERENT routing key (which under naive
// per-shard locking would grab a different shard lock and overlap), and not a
// single-shard op on any shard. A shared "active" counter must never exceed 1
// while a cross-shard op holds it. This is the regression the per-shard change
// would otherwise introduce for WASM ops.
func TestDirectCrossShardOpExcludesEveryOp(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	std := ops.KeyExtractorByHandle("std")

	var active, violations int32
	enter := func(assertAlone bool) {
		n := atomic.AddInt32(&active, 1)
		if assertAlone && n != 1 {
			atomic.AddInt32(&violations, 1)
		}
		time.Sleep(40 * time.Microsecond) // widen the window so overlaps surface
		atomic.AddInt32(&active, -1)
	}
	// xs: cross-shard — must run alone (asserts active==1 on entry).
	if err := reg.RegisterRoutableCrossShard("xsop", ops.OpReadWrite,
		func(_ *ops.TxContext, _ []byte) ([]byte, error) { enter(true); return nil, nil }, std); err != nil {
		t.Fatalf("register xsop: %v", err)
	}
	// ss: ordinary single-shard op — bumps active so an overlapping xs sees >1.
	if err := reg.RegisterRoutable("ssop", ops.OpReadWrite,
		func(_ *ops.TxContext, _ []byte) ([]byte, error) { enter(false); return nil, nil }, std); err != nil {
		t.Fatalf("register ssop: %v", err)
	}

	s, err := NewDirect(DirectConfig{Ops: reg, DataDir: t.TempDir()})
	if err != nil {
		t.Fatalf("NewDirect: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	ctx := context.Background()

	const G = 8
	var wg sync.WaitGroup
	for g := 0; g < G; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			// distinct routing keys per goroutine: naive per-shard locking would
			// let these fan out across shards and overlap.
			xsKey := ops.EncodeKeyArgs([]byte(fmt.Sprintf("xs-%d", g)))
			ssKey := ops.EncodeKeyArgs([]byte(fmt.Sprintf("ss-%d", g)))
			for i := 0; i < 150; i++ {
				if i%2 == 0 {
					if _, err := s.Call(ctx, "xsop", xsKey); err != nil {
						t.Errorf("xsop: %v", err)
						return
					}
				} else {
					if _, err := s.Call(ctx, "ssop", ssKey); err != nil {
						t.Errorf("ssop: %v", err)
						return
					}
				}
			}
		}(g)
	}
	wg.Wait()

	if v := atomic.LoadInt32(&violations); v != 0 {
		t.Fatalf("cross-shard op overlapped another op %d times — all-shards barrier not taken", v)
	}
}

// TestDirectPutSerializedWithRMW proves finding 017: a raw Store.Put on a key is
// serialized against a multi-step read-modify-write op on the SAME key, so a Put
// can never land BETWEEN an RMW's read and write and be silently clobbered (lost
// update). The custom "rmw" op reads the key, WIDENS its window with a short
// sleep, re-reads, and flags a violation if the value changed underneath it. With
// the per-shard opMu held across the whole RMW (Call's read-write path) AND taken
// by Store.Put (the fix), a concurrent Put blocks until the RMW completes, so the
// two reads always agree. WITHOUT the fix, Put wrote straight to the cache shard
// mid-window and the reads diverge — the exact tear that clobbers handleIncr's
// Get-then-Put. This is the same window-overlap detector TestDirectCrossShardOp
// ExcludesEveryOp uses, applied to the raw Put surface. Run under -race.
func TestDirectPutSerializedWithRMW(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	std := ops.KeyExtractorByHandle("std")

	var violations int32
	rmw := func(tx *ops.TxContext, args []byte) ([]byte, error) {
		key, ok := std(args)
		if !ok {
			return nil, fmt.Errorf("rmw: bad key args")
		}
		read := func() string { b, _ := tx.Get(key); return string(b) }
		v1 := read()
		time.Sleep(40 * time.Microsecond) // widen the RMW window so a racing Put surfaces
		v2 := read()
		if v1 != v2 {
			atomic.AddInt32(&violations, 1)
		}
		return nil, tx.Put(key, []byte("rmw-done"), 0)
	}
	if err := reg.RegisterRoutable("rmw", ops.OpReadWrite, rmw, std); err != nil {
		t.Fatalf("register rmw: %v", err)
	}

	s, err := NewDirect(DirectConfig{Ops: reg, DataDir: t.TempDir()})
	if err != nil {
		t.Fatalf("NewDirect: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	ctx := context.Background()

	key := []byte("shared")
	args := ops.EncodeKeyArgs(key)
	if err := s.Put(ctx, key, []byte("seed"), 0); err != nil {
		t.Fatalf("seed Put: %v", err)
	}

	var wg sync.WaitGroup
	const rmwG, putG = 4, 4
	for g := 0; g < rmwG; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 60; i++ {
				if _, err := s.Call(ctx, "rmw", args); err != nil {
					t.Errorf("rmw: %v", err)
					return
				}
			}
		}()
	}
	for g := 0; g < putG; g++ {
		wg.Add(1)
		go func(gi int) {
			defer wg.Done()
			for i := 0; i < 500; i++ {
				if err := s.Put(ctx, key, []byte(fmt.Sprintf("p%d-%d", gi, i)), 0); err != nil {
					t.Errorf("Put: %v", err)
					return
				}
			}
		}(g)
	}
	wg.Wait()

	if v := atomic.LoadInt32(&violations); v != 0 {
		t.Fatalf("Store.Put tore an in-flight RMW %d times — Put bypasses per-shard opMu (lost update)", v)
	}
}

//go:build cgo && linux

// SPDX-License-Identifier: Apache-2.0

package wasm

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
)

// Host-boundary error tests.
//
// A state-mutating host function (cache_put / cache_del / cache_expire) can only
// tell the guest "-1" when the cache refuses the mutation. That is the whole
// signal the WASM contract has. If the host stops there, the error dies at the
// boundary: Invoke returns the guest's own status with a NIL error, the FSM sees
// ApplyResponse.Err == nil, classifyApplyErr never runs, and the applied index
// advances on a replica whose state no longer matches its peers'.
//
// These tests pin the fix from the outside: a real guest module, a real cache
// that genuinely cannot take the write, and the requirement that
// errors.Is(err, cache.ErrFull) hold on what Invoke returns — because that is
// exactly the predicate shard/apply_class.go tests to decide classFatal.

// stdArgs frames key as the invoke args every WASM op now receives:
// [keyLen u16][key], with an empty payload.
//
// The put/del testdata modules SKIP that 2-byte prefix before handing the rest
// to their host call (see wasm/testdata/put.wat), because
// ops.WASMKeyExtractorHandle pinned every WASM op to the "std" key extractor —
// the extractor picks the ROUTING key but does not rewrite the module's args.
// Invoking them with a bare key would therefore address key[2:], and these tests
// would be measuring a host call against the wrong key instead of the host-error
// boundary they exist to pin.
func stdArgs(key []byte) []byte {
	out := make([]byte, 2+len(key))
	out[0] = byte(len(key) >> 8)
	out[1] = byte(len(key))
	copy(out[2:], key)
	return out
}

// loadHostErrModule compiles a testdata module and registers it on a fresh
// Runtime under opName. Both are torn down with the test.
func loadHostErrModule(t *testing.T, path, opName string) (*Runtime, ModuleID) {
	t.Helper()
	wasmBytes, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	m, err := Compile(wasmBytes)
	if err != nil {
		t.Fatalf("Compile %s: %v", path, err)
	}
	t.Cleanup(func() { _ = m.Close() })

	rt, err := NewRuntime()
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	t.Cleanup(func() { _ = rt.Close() })

	id, err := rt.AddModule(m, "apply", 0)
	if err != nil {
		t.Fatalf("AddModule %s: %v", opName, err)
	}
	return rt, id
}

// fullPersistentCache returns a single-shard, single-page PERSISTENT
// reject-writes cache filled until it rejects writes, along with a key that is
// live inside it. Persistent + reject-writes is the profile replication forces
// (shard/store.go), so this is the configuration whose ErrFull actually reaches
// a replicated apply.
//
// The victim key is deliberately LONG: deleting it on a persistent shard must
// append a 26+len(key)-byte tombstone, which cannot fit in the scraps left over
// once the short padding keys stop fitting. That is what makes Del — not just
// Put — hit ErrFull.
func fullPersistentCache(t *testing.T) (*cache.Cache, []byte) {
	t.Helper()
	cfg := cache.DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 1 << 20 // exactly one page
	cfg.AtCapPolicy = cache.PolicyRejectWrites
	cfg.TTLSweepIntervalMs = 0
	cfg.DataDir = t.TempDir()

	c, err := cache.New(cfg)
	if err != nil {
		t.Fatalf("cache.New: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })

	victim := []byte(strings.Repeat("v", 512))
	if err := c.Put(victim, []byte("x"), 0); err != nil {
		t.Fatalf("seeding the victim key: %v", err)
	}
	for i := 0; ; i++ {
		err := c.Put([]byte(fmt.Sprintf("pad-%08d", i)), []byte("x"), 0)
		if errors.Is(err, cache.ErrFull) {
			break
		}
		if err != nil {
			t.Fatalf("padding write %d: %v", i, err)
		}
		if i > 1<<20 {
			t.Fatal("the shard never filled; the test config is wrong")
		}
	}
	return c, victim
}

// TestWASMHostDeleteErrorReachesInvoke is the case this branch introduced: a
// delete on a persistent shard now needs room for its tombstone, so cache_del
// can fail with ErrFull. The guest sees -1 and returns it; the caller must still
// get an error that errors.Is(cache.ErrFull), or the node advances past a delete
// its peers performed and it did not.
func TestWASMHostDeleteErrorReachesInvoke(t *testing.T) {
	rt, id := loadHostErrModule(t, "testdata/del.wasm", "del")
	c, victim := fullPersistentCache(t)

	_, err := rt.Invoke(id, ops.NewTxContext(c), stdArgs(victim))
	if err == nil {
		t.Fatal("Invoke returned nil error after cache_del was refused: the host error " +
			"died at the WASM boundary and this apply would silently advance (divergence)")
	}
	if !errors.Is(err, cache.ErrFull) {
		t.Fatalf("Invoke error = %v; want one that errors.Is(cache.ErrFull) so "+
			"classifyApplyErr can rate it classFatal", err)
	}
	// The delete really did not happen — which is precisely why the error has to
	// escape: a peer with room DID delete this key.
	if _, gerr := c.Get(victim); gerr != nil {
		t.Fatalf("the victim key is gone (%v) even though the delete was refused", gerr)
	}
}

// TestWASMHostPutErrorReachesInvoke is the same boundary for cache_put, which has
// always been able to fail this way. It is not new, but it diverges replicas by
// the identical mechanism, so it is pinned identically.
func TestWASMHostPutErrorReachesInvoke(t *testing.T) {
	rt, id := loadHostErrModule(t, "testdata/put.wasm", "put")
	c, _ := fullPersistentCache(t)

	_, err := rt.Invoke(id, ops.NewTxContext(c), stdArgs([]byte("a-key-with-nowhere-to-go")))
	if err == nil {
		t.Fatal("Invoke returned nil error after cache_put was refused: the host error " +
			"died at the WASM boundary and this apply would silently advance (divergence)")
	}
	if !errors.Is(err, cache.ErrFull) {
		t.Fatalf("Invoke error = %v; want one that errors.Is(cache.ErrFull)", err)
	}
}

// TestWASMHostErrorDoesNotLeakToNextInvoke covers the other half of the contract:
// the holder is allocated once per module and reused for every Invoke, so a
// latched error must be cleared on entry. If it leaked, the first refused write
// would poison every later invoke of that op — turning one full shard into a
// permanently unusable stored procedure.
func TestWASMHostErrorDoesNotLeakToNextInvoke(t *testing.T) {
	rt, id := loadHostErrModule(t, "testdata/del.wasm", "del")
	full, victim := fullPersistentCache(t)

	if _, err := rt.Invoke(id, ops.NewTxContext(full), stdArgs(victim)); !errors.Is(err, cache.ErrFull) {
		t.Fatalf("setup: first Invoke error = %v, want cache.ErrFull", err)
	}

	healthy, err := cache.New(cache.DefaultConfig())
	if err != nil {
		t.Fatalf("cache.New: %v", err)
	}
	defer func() { _ = healthy.Close() }()

	// Deleting an absent key is a clean no-op: the guest returns 0 and no host
	// error is recorded, so Invoke must report success.
	if _, err := rt.Invoke(id, ops.NewTxContext(healthy), stdArgs([]byte("never-written"))); err != nil {
		t.Fatalf("second Invoke on a healthy cache = %v, want nil: the previous "+
			"invoke's host error leaked through the reused holder", err)
	}
}

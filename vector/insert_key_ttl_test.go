// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"os"
	"testing"
)

// TestInsertKeyTTLEngineDrop: a per-key payload TTL supplied AT INSERT is present
// before its deadline and DROPPED after, while the point and its permanent keys
// survive — the engine computed the absolute deadline now+ttl at insert (mirrors
// set_payload). Exercises hnsw.Insert → insertLocked → arena.keyExpires.
func TestInsertKeyTTLEngineDrop(t *testing.T) {
	h, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 2_000_000
	h.now = func() int64 { return fakeNow }

	// Insert with a permanent key + a 1000ms key TTL, both in the payload.
	_, ke, err := h.Insert(1, []float32{1, 0, 0, 0}, 0,
		Metadata{"perm": NewString("keep"), "temp": NewString("bye")},
		nil, map[string]int64{"temp": 1000}, CASCond{})
	if err != nil {
		t.Fatalf("Insert: %v", err)
	}
	wantDeadline := uint64(fakeNow) + 1000
	if ke["temp"] != wantDeadline {
		t.Fatalf("returned keyExpires[temp] = %d, want %d (absolute now+ttl)", ke["temp"], wantDeadline)
	}
	// The arena stored the absolute deadline.
	slot, _ := h.arena.Slot(1)
	if got := h.arena.KeyExpires(slot)["temp"]; got != wantDeadline {
		t.Fatalf("arena keyExpires[temp] = %d, want %d", got, wantDeadline)
	}

	// Before the deadline: both keys present.
	_, meta, _, _, _, ok := h.Get(1)
	if !ok {
		t.Fatal("Get(1) ok=false before key expiry")
	}
	if meta["temp"].Str != "bye" || meta["perm"].Str != "keep" {
		t.Fatalf("pre-expiry meta = %v, want both keys", meta)
	}

	// Past the deadline: temp dropped (lazy-drop on read), point + perm survive.
	fakeNow += 1500
	_, meta, _, _, _, ok = h.Get(1)
	if !ok {
		t.Fatal("Get(1) ok=false after key expiry (the POINT must still live)")
	}
	if hasKey(meta, "temp") {
		t.Fatalf("post-expiry meta still has expired key 'temp': %v", meta)
	}
	if meta["perm"].Str != "keep" {
		t.Fatalf("post-expiry lost permanent key: %v", meta)
	}
}

// TestInsertKeyTTLOnlyPresentKeys: a key TTL whose key is NOT in the payload sets
// no deadline; a ttl<=0 is ignored (the key is permanent). Mirrors
// computeInsertKeyExpires / set_payload semantics.
func TestInsertKeyTTLOnlyPresentKeys(t *testing.T) {
	h, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 100
	h.now = func() int64 { return fakeNow }

	_, ke, err := h.Insert(1, []float32{1, 0, 0, 0}, 0,
		Metadata{"a": NewInt(1), "b": NewInt(2)},
		nil, map[string]int64{
			"a":       1000, // present → deadline
			"b":       0,    // ttl<=0 → permanent, no deadline
			"missing": 1000, // absent from payload → skipped
		}, CASCond{})
	if err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if len(ke) != 1 {
		t.Fatalf("keyExpires = %v, want exactly {a}", ke)
	}
	if ke["a"] != uint64(fakeNow)+1000 {
		t.Fatalf("keyExpires[a] = %d, want %d", ke["a"], uint64(fakeNow)+1000)
	}
}

// TestInsertNoKeyTTLLeavesSlotClear: a plain insert (nil key TTL) leaves the
// slot's keyExpires nil — the zero-overhead path.
func TestInsertNoKeyTTLLeavesSlotClear(t *testing.T) {
	h, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	_, ke, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, Metadata{"a": NewInt(1)}, nil, nil, CASCond{})
	if err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if ke != nil {
		t.Fatalf("returned keyExpires = %v, want nil for a no-key_ttl insert", ke)
	}
	slot, _ := h.arena.Slot(1)
	if got := h.arena.KeyExpires(slot); got != nil {
		t.Fatalf("arena keyExpires = %v, want nil for a no-key_ttl insert", got)
	}
}

// TestUpsertKeyTTLOnFreshPoint: Upsert (delete-then-insert) sets a per-key TTL on
// the FRESH point. Drives the Collection.Upsert path (heap, no WAL).
func TestUpsertKeyTTLOnFreshPoint(t *testing.T) {
	cfg := keyTTLCfg()
	c, err := NewCollection("docs", cfg)
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	defer c.Stop()
	var fakeNow int64 = 7_000
	c.idx.(*hnsw).now = func() int64 { return fakeNow }

	if err := c.UpsertKeyTTL(1, []float32{1, 0, 0, 0}, "", 0,
		Metadata{"temp": NewInt(5), "perm": NewInt(1)},
		nil, map[string]int64{"temp": 500}); err != nil {
		t.Fatalf("UpsertKeyTTL: %v", err)
	}
	slot, _ := c.idx.(*hnsw).arena.Slot(1)
	if got := c.idx.(*hnsw).arena.KeyExpires(slot)["temp"]; got != uint64(fakeNow)+500 {
		t.Fatalf("upserted keyExpires[temp] = %d, want %d", got, uint64(fakeNow)+500)
	}
	// Pre-deadline present, post-deadline dropped.
	if _, meta, _, _, _, ok := c.Get(1); !ok || meta["temp"].Int != 5 {
		t.Fatalf("pre-deadline meta=%v ok=%v, want temp:5", meta, ok)
	}
	fakeNow += 600
	if _, meta, _, _, _, ok := c.Get(1); !ok || hasKey(meta, "temp") {
		t.Fatalf("post-deadline meta=%v ok=%v, want temp DROPPED", meta, ok)
	}
}

// TestAppendInsertByteIdenticalWhenEmptyKeyTTL: appendInsertStaged with an empty
// key TTL map appends only the constant 0 flag byte (like writeOptVersion) — the
// keyExpires block contributes NO per-key data, so two inserts that differ only in
// (nil vs empty map) key TTL produce identical record bytes. Proves the WAL gap is
// closed at zero overhead for the no-key_ttl path.
func TestAppendInsertByteIdenticalWhenEmptyKeyTTL(t *testing.T) {
	dir := t.TempDir()
	w1, err := openWAL(dir+"/a.wal", true)
	if err != nil {
		t.Fatal(err)
	}
	w2, err := openWAL(dir+"/b.wal", true)
	if err != nil {
		t.Fatal(err)
	}
	// Staged write: the record BYTES are what this test compares, and the staged
	// pair emits byte-identical bytes (both wals are noSync, so no commit-wait is
	// arranged and commitWaitStaged would be a no-op anyway).
	if _, err := w1.appendInsertStaged(1, []float32{1, 2, 3, 4}, 0, Metadata{"a": NewInt(1)}, nil, nil, 1); err != nil {
		t.Fatal(err)
	}
	if _, err := w2.appendInsertStaged(1, []float32{1, 2, 3, 4}, 0, Metadata{"a": NewInt(1)}, nil, map[string]uint64{}, 1); err != nil {
		t.Fatal(err)
	}
	_ = w1.close()
	_ = w2.close()
	b1, err := os.ReadFile(dir + "/a.wal")
	if err != nil {
		t.Fatal(err)
	}
	b2, err := os.ReadFile(dir + "/b.wal")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(b1, b2) {
		t.Fatalf("appendInsertStaged(nil) vs appendInsertStaged(empty map) differ:\n nil=%x\n empty=%x", b1, b2)
	}
}

// TestInsertKeyTTLWALReplayAbsolute: a per-key payload TTL set AT INSERT in a
// WAL-mode collection survives a CRASH (reopen = checkpoint + WAL tail, NO flush)
// as an ABSOLUTE deadline — NOT recomputed on replay. This is the dense
// appendInsertStaged keyExpires-block gap closing: without the WAL block the deadline
// would be lost on a crash-before-snapshot. We freeze the clock, insert with a key
// TTL, crash, reopen, and assert the key is present pre-deadline and dropped past
// the ORIGINAL absolute deadline.
func TestInsertKeyTTLWALReplayAbsolute(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	cfg := walCfg()
	cfg.Dim = 4
	if err := cs.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	c, _ := cs.Get("docs")
	var fakeNow int64 = 9_000_000
	c.idx.(*hnsw).now = func() int64 { return fakeNow }

	// Insert with a permanent key + a 1000ms key TTL → absolute deadline 9_001_000.
	if err := c.InsertKeyTTL(1, []float32{1, 0, 0, 0}, 0,
		Metadata{"perm": NewInt(1), "temp": NewInt(9)},
		nil, map[string]int64{"temp": 1000}); err != nil {
		t.Fatalf("InsertKeyTTL: %v", err)
	}
	wantDeadline := uint64(fakeNow) + 1000
	// Crash: ONLY the WAL holds the insert (no flush).
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = cs2.Close() }()
	c2, _ := cs2.Get("docs")
	c2.idx.(*hnsw).now = func() int64 { return fakeNow }

	h2 := c2.idx.(*hnsw)
	slot, ok := h2.arena.Slot(1)
	if !ok {
		t.Fatal("point id=1 missing after WAL replay (insert not recovered)")
	}
	if got := h2.arena.KeyExpires(slot)["temp"]; got != wantDeadline {
		t.Fatalf("replayed keyExpires[temp] = %d, want %d (absolute deadline not preserved / WAL gap open)", got, wantDeadline)
	}
	if _, meta, _, _, _, ok := c2.Get(1); !ok || meta["temp"].Int != 9 {
		t.Fatalf("post-replay pre-deadline meta=%v ok=%v, want temp:9", meta, ok)
	}
	// Advance past the ORIGINAL absolute deadline: the key drops (proves it was NOT
	// recomputed as now+ttl at replay time — that would push it to fakeNow+2000).
	fakeNow += 1500
	if _, meta, _, _, _, ok := c2.Get(1); !ok || hasKey(meta, "temp") {
		t.Fatalf("post-replay post-deadline meta=%v ok=%v, want temp DROPPED", meta, ok)
	}
	if _, meta, _, _, _, _ := c2.Get(1); meta["perm"].Int != 1 {
		t.Fatalf("post-replay lost permanent key: %v", meta)
	}
}

// TestInsertKeyTTLSnapshotAbsolute: the insert-time absolute key deadline survives
// a snapshot round-trip VERBATIM (time-stable) — the snapshot serializes
// arena.keyExpires wholesale, so persistence is automatic for the insert path.
func TestInsertKeyTTLSnapshotAbsolute(t *testing.T) {
	h, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 3_000_000
	h.now = func() int64 { return fakeNow }
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0,
		Metadata{"perm": NewInt(1), "temp": NewInt(9)},
		nil, map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	wantDeadline := uint64(fakeNow) + 1000

	var buf bytes.Buffer
	if err := h.Snapshot(&buf); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	h2, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	if err := h2.Restore(&buf); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	h2.now = func() int64 { return fakeNow }
	slot, _ := h2.arena.Slot(1)
	if got := h2.arena.KeyExpires(slot)["temp"]; got != wantDeadline {
		t.Fatalf("restored keyExpires[temp] = %d, want %d (snapshot not time-stable)", got, wantDeadline)
	}
	// Advance past the original absolute deadline: key drops.
	h2.now = func() int64 { return fakeNow + 1500 }
	if _, meta, _, _, _, ok := h2.Get(1); !ok || hasKey(meta, "temp") {
		t.Fatalf("post-restore post-deadline meta=%v ok=%v, want temp DROPPED", meta, ok)
	}
}

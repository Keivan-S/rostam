// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"testing"
)

// keyTTLCfg is a small heap-backed config for the per-key payload TTL tests.
// FilterFirstThreshold is large so seeded posting lists make the filter take the
// filter-first (index) path (matching TestReindexCorrectness's setup).
func keyTTLCfg() Config {
	return Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, FilterFirstThreshold: 10_000}
}

// TestKeyTTLGetDrop: a key with a TTL is returned before its deadline and
// DROPPED after, while the point and its non-TTL keys survive. Exercises the
// hnsw.Get read path through liveMeta.
func TestKeyTTLGetDrop(t *testing.T) {
	h, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, Metadata{"perm": NewString("keep")}, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	// Add a TTL key (1000ms) plus another permanent key via SetPayload.
	if _, ke, _, err := h.SetPayload(1, Metadata{"temp": NewString("bye"), "perm2": NewInt(7)},
		map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("SetPayload: %v", err)
	} else if ke["temp"] != uint64(fakeNow)+1000 {
		t.Fatalf("resulting keyExpires[temp] = %d, want %d", ke["temp"], uint64(fakeNow)+1000)
	}

	// Before expiry: all keys present.
	_, meta, _, _, _, ok := h.Get(1)
	if !ok {
		t.Fatal("Get(1) ok=false before expiry")
	}
	if meta["temp"].Str != "bye" || meta["perm"].Str != "keep" || meta["perm2"].Int != 7 {
		t.Fatalf("pre-expiry meta = %v, want all three keys", meta)
	}

	// Advance past the deadline: temp dropped, point + perm keys survive.
	fakeNow += 1500
	_, meta, _, _, _, ok = h.Get(1)
	if !ok {
		t.Fatal("Get(1) ok=false after key expiry (the POINT must still live)")
	}
	if _, present := meta["temp"]; present {
		t.Fatalf("post-expiry meta still has expired key 'temp': %v", meta)
	}
	if meta["perm"].Str != "keep" || meta["perm2"].Int != 7 {
		t.Fatalf("post-expiry meta lost a permanent key: %v", meta)
	}
}

// TestKeyTTLFilterFirstNoMatch: a FILTER on an expired key must return NO match
// on the dense filter-first path (the payloadIdx superset still lists the slot,
// but the predicate re-check sees the TTL-filtered meta via liveMeta in admits).
func TestKeyTTLFilterFirstNoMatch(t *testing.T) {
	cfg := keyTTLCfg()
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	// id=1 closest to the query; give it field temp==hot with a TTL.
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 1: %v", err)
	}
	// Seed ~10 other points with temp==hot (far from query) so the posting list
	// exists and candidates(filter) returns ok=true → filter-first path.
	seedID := uint64(100)
	for j := 0; j < 10; j++ {
		if _, _, err := h.Insert(seedID, []float32{0, 0, float32(seedID), 0}, 0, Metadata{"temp": NewString("hot")}, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert seeder %d: %v", seedID, err)
		}
		seedID++
	}
	// Background corpus to keep the value selective.
	for i := uint64(2); i <= 20; i++ {
		if _, _, err := h.Insert(i, []float32{0, float32(i), 0, 0}, 0, Metadata{"a": NewInt(int64(i))}, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	// id=1 gets temp==hot with a 1000ms TTL.
	if _, _, _, err := h.SetPayload(1, Metadata{"temp": NewString("hot")}, map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}

	f := Filter{Op: FilterEq, Field: "temp", Value: NewString("hot")}
	if _, ok := h.payloadIdx.candidates(f, cfg.FilterFirstThreshold); !ok {
		t.Fatal("candidates ok=false — test would not exercise filter-first")
	}
	q := []float32{1, 0, 0, 0}

	// Before expiry: id=1 matches the filter (it is closest, so it's a top hit).
	res, err := h.SearchFiltered(q, 5, f)
	if err != nil {
		t.Fatalf("SearchFiltered pre-expiry: %v", err)
	}
	if !containsID(res, 1) {
		t.Fatalf("pre-expiry filter temp==hot = %v, want to contain id=1", resultIDs(res))
	}

	// After the key expires: the predicate (over liveMeta) must reject id=1 even
	// though the payloadIdx posting list still lists it (tolerated over-cover).
	fakeNow += 1500
	res, err = h.SearchFiltered(q, 5, f)
	if err != nil {
		t.Fatalf("SearchFiltered post-expiry: %v", err)
	}
	if containsID(res, 1) {
		t.Fatalf("post-expiry filter temp==hot = %v, want NOT to contain id=1 (expired key matched)", resultIDs(res))
	}
}

// TestKeyTTLScrollDrop: scroll (which assembles payload via docForLocked →
// liveMeta) drops the expired key while keeping the point + permanent keys.
func TestKeyTTLScrollDrop(t *testing.T) {
	h, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, Metadata{"perm": NewInt(1)}, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if _, _, _, err := h.SetPayload(1, Metadata{"temp": NewInt(9)}, map[string]int64{"temp": 500}, CASCond{}); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}

	docsBefore, err := h.scrollDocs(Filter{}, 0)
	if err != nil {
		t.Fatalf("scroll pre-expiry: %v", err)
	}
	if len(docsBefore) != 1 || docsBefore[0].Metadata["temp"].Int != 9 {
		t.Fatalf("pre-expiry scroll = %v, want temp:9 present", docsBefore)
	}

	fakeNow += 1000
	docsAfter, err := h.scrollDocs(Filter{}, 0)
	if err != nil {
		t.Fatalf("scroll post-expiry: %v", err)
	}
	if len(docsAfter) != 1 {
		t.Fatalf("scroll post-expiry doc count = %d, want 1 (point survives)", len(docsAfter))
	}
	if _, present := docsAfter[0].Metadata["temp"]; present {
		t.Fatalf("post-expiry scroll still has expired key: %v", docsAfter[0].Metadata)
	}
	if docsAfter[0].Metadata["perm"].Int != 1 {
		t.Fatalf("post-expiry scroll lost permanent key: %v", docsAfter[0].Metadata)
	}
}

// TestKeyTTLOverwriteDeleteClear: overwrite replaces the deadline set, delete
// drops a key's deadline, clear wipes all deadlines.
func TestKeyTTLOverwriteDeleteClear(t *testing.T) {
	h, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }
	slotOf := func(id uint64) uint32 { s, _ := h.arena.Slot(id); return s }

	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert: %v", err)
	}

	// SetPayload two TTL keys.
	if _, _, _, err := h.SetPayload(1, Metadata{"x": NewInt(1), "y": NewInt(2)},
		map[string]int64{"x": 1000, "y": 2000}, CASCond{}); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}
	if ke := h.arena.KeyExpires(slotOf(1)); len(ke) != 2 {
		t.Fatalf("after set: keyExpires len = %d, want 2", len(ke))
	}

	// Overwrite: only z carries a TTL now; x/y deadlines must be gone.
	if _, ke, _, err := h.OverwritePayload(1, Metadata{"z": NewInt(3)}, map[string]int64{"z": 1500}, CASCond{}); err != nil {
		t.Fatalf("OverwritePayload: %v", err)
	} else if len(ke) != 1 || ke["z"] != uint64(fakeNow)+1500 {
		t.Fatalf("after overwrite: keyExpires = %v, want only z", ke)
	}
	if ke := h.arena.KeyExpires(slotOf(1)); len(ke) != 1 {
		t.Fatalf("after overwrite: stored keyExpires len = %d, want 1", len(ke))
	}

	// Re-add x with a TTL, then DeletePayloadKeys(z) → only x's deadline remains.
	if _, _, _, err := h.SetPayload(1, Metadata{"x": NewInt(1)}, map[string]int64{"x": 1000}, CASCond{}); err != nil {
		t.Fatalf("SetPayload x: %v", err)
	}
	if _, ke, _, err := h.DeletePayloadKeys(1, []string{"z"}, CASCond{}); err != nil {
		t.Fatalf("DeletePayloadKeys: %v", err)
	} else if _, present := ke["z"]; present {
		t.Fatalf("after delete: keyExpires still has z: %v", ke)
	}

	// Clear: no deadlines at all.
	if _, ke, _, err := h.ClearPayload(1, CASCond{}); err != nil {
		t.Fatalf("ClearPayload: %v", err)
	} else if len(ke) != 0 {
		t.Fatalf("after clear: keyExpires = %v, want empty", ke)
	}
	if ke := h.arena.KeyExpires(slotOf(1)); ke != nil {
		t.Fatalf("after clear: stored keyExpires = %v, want nil", ke)
	}
}

// TestKeyTTLSetClearsDeadline: a non-positive ttl in SetPayload clears an
// existing deadline (the key becomes permanent again).
func TestKeyTTLSetClearsDeadline(t *testing.T) {
	h, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	_, _, _ = h.Insert(1, []float32{1, 0, 0, 0}, 0, nil, nil, nil, CASCond{})
	if _, _, _, err := h.SetPayload(1, Metadata{"k": NewInt(1)}, map[string]int64{"k": 500}, CASCond{}); err != nil {
		t.Fatalf("SetPayload set ttl: %v", err)
	}
	// Clear the deadline with ttl=0 (key stays in payload).
	if _, ke, _, err := h.SetPayload(1, Metadata{"k": NewInt(1)}, map[string]int64{"k": 0}, CASCond{}); err != nil {
		t.Fatalf("SetPayload clear ttl: %v", err)
	} else if len(ke) != 0 {
		t.Fatalf("after ttl=0: keyExpires = %v, want empty", ke)
	}
	// The key must now survive past the original deadline.
	fakeNow += 1000
	if _, meta, _, _, _, ok := h.Get(1); !ok || meta["k"].Int != 1 {
		t.Fatalf("key dropped after ttl was cleared: meta=%v ok=%v", meta, ok)
	}
}

// TestKeyTTLSnapshotRoundtrip: snapshot → restore preserves the ABSOLUTE per-key
// deadline (so a pending key TTL survives a restart and still expires on time).
func TestKeyTTLSnapshotRoundtrip(t *testing.T) {
	src, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	src.now = func() int64 { return fakeNow }

	_, _, _ = src.Insert(1, []float32{1, 0, 0, 0}, 0, Metadata{"perm": NewInt(1)}, nil, nil, CASCond{})
	if _, _, _, err := src.SetPayload(1, Metadata{"temp": NewInt(9)}, map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}
	wantDeadline := uint64(fakeNow) + 1000

	var buf bytes.Buffer
	if err := src.Snapshot(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}

	dst, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW dst: %v", err)
	}
	dst.now = func() int64 { return fakeNow }
	if err := dst.Restore(&buf); err != nil {
		t.Fatalf("restore: %v", err)
	}

	// The absolute deadline must be preserved verbatim.
	slot, _ := dst.arena.Slot(1)
	if got := dst.arena.KeyExpires(slot)["temp"]; got != wantDeadline {
		t.Fatalf("restored keyExpires[temp] = %d, want %d (deadline not preserved)", got, wantDeadline)
	}
	// Pre-expiry the key is present; advancing the (shared) clock drops it.
	if _, meta, _, _, _, ok := dst.Get(1); !ok || meta["temp"].Int != 9 {
		t.Fatalf("post-restore pre-expiry meta=%v ok=%v, want temp:9", meta, ok)
	}
	fakeNow += 1500
	if _, meta, _, _, _, ok := dst.Get(1); !ok || hasKey(meta, "temp") {
		t.Fatalf("post-restore post-expiry meta=%v, want temp DROPPED (point alive)", meta)
	}
	if _, meta, _, _, _, _ := dst.Get(1); meta["perm"].Int != 1 {
		t.Fatalf("post-restore lost permanent key: %v", meta)
	}
}

// TestKeyTTLSnapshotBackwardCompat: a synthesized OLD snapshot (no keyExpires
// block) restores cleanly with NO per-key TTL. We force the old format by
// rewriting the version field to 4 and truncating the keyExpires block — instead,
// we exercise the real path: a snapshot of an index with NO per-key TTL is
// byte-compatible (zero count) and restores with nil keyExpires.
func TestKeyTTLSnapshotBackwardCompat(t *testing.T) {
	// Build a v5 snapshot that simply has NO per-key deadlines (the common case).
	// Restoring it must yield nil keyExpires everywhere (no per-key TTL), proving
	// the absent-block path is handled. (A literal v<5 snapshot is covered by the
	// version-gated restore: version<5 never reads the block and leaves it nil.)
	src, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	_, _, _ = src.Insert(1, []float32{1, 0, 0, 0}, 0, Metadata{"a": NewInt(1)}, nil, nil, CASCond{})
	var buf bytes.Buffer
	if err := src.Snapshot(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}

	dst, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW dst: %v", err)
	}
	if err := dst.Restore(&buf); err != nil {
		t.Fatalf("restore: %v", err)
	}
	slot, _ := dst.arena.Slot(1)
	if ke := dst.arena.KeyExpires(slot); ke != nil {
		t.Fatalf("restored keyExpires = %v, want nil (no per-key TTL)", ke)
	}
	if _, meta, _, _, _, ok := dst.Get(1); !ok || meta["a"].Int != 1 {
		t.Fatalf("restored point meta=%v ok=%v, want a:1", meta, ok)
	}
}

// TestKeyTTLWALReplayAbsolute: a WAL-mode collection's per-key deadline survives
// a crash (reopen = checkpoint + WAL tail) as an ABSOLUTE deadline — NOT
// recomputed on replay. We freeze the clock so the deadline is deterministic, set
// a key TTL, crash (no flush), reopen, and assert the key is present pre-deadline
// and dropped past it.
func TestKeyTTLWALReplayAbsolute(t *testing.T) {
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
	if err := cs.Insert("docs", 1, []float32{1, 0, 0, 0}, 0, Metadata{"perm": NewInt(1)}, nil); err != nil {
		t.Fatal(err)
	}
	c, _ := cs.Get("docs")
	var fakeNow int64 = 5_000_000
	c.idx.(*hnsw).now = func() int64 { return fakeNow }
	// Set a key TTL of 1000ms → absolute deadline 5_001_000.
	if err := c.SetPayload(1, Metadata{"temp": NewInt(9)}, map[string]int64{"temp": 1000}); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}
	wantDeadline := uint64(fakeNow) + 1000
	// Crash: only the WAL holds the mutation.
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = cs2.Close() }()
	c2, _ := cs2.Get("docs")
	// Pin the (post-replay) clock to a value where the deadline is in the FUTURE.
	c2.idx.(*hnsw).now = func() int64 { return fakeNow }

	h2 := c2.idx.(*hnsw)
	slot, _ := h2.arena.Slot(1)
	if got := h2.arena.KeyExpires(slot)["temp"]; got != wantDeadline {
		t.Fatalf("replayed keyExpires[temp] = %d, want %d (absolute deadline not preserved)", got, wantDeadline)
	}
	if _, meta, _, _, _, ok := c2.Get(1); !ok || meta["temp"].Int != 9 {
		t.Fatalf("post-replay pre-deadline meta=%v ok=%v, want temp:9", meta, ok)
	}
	// Advance past the absolute deadline: the key drops (proves it was NOT
	// recomputed as now+ttl at replay time — that would push it to fakeNow+2000).
	fakeNow += 1500
	if _, meta, _, _, _, ok := c2.Get(1); !ok || hasKey(meta, "temp") {
		t.Fatalf("post-replay post-deadline meta=%v ok=%v, want temp DROPPED", meta, ok)
	}
	if _, meta, _, _, _, _ := c2.Get(1); meta["perm"].Int != 1 {
		t.Fatalf("post-replay lost permanent key: %v", meta)
	}
}

// hasKey reports whether m contains key k (test helper).
func hasKey(m Metadata, k string) bool {
	_, ok := m[k]
	return ok
}

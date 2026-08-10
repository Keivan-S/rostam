// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"testing"
	"time"
)

// getPayloadCfg is a small heap-backed config for the Get/payload tests.
func getPayloadCfg() Config {
	return Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1}
}

// TestGetLive proves Get returns the correct vector, payload, and remaining TTL
// for a live point, and that the returned data is a DEEP COPY (mutating it does
// not corrupt the arena).
func TestGetLive(t *testing.T) {
	h, err := newHNSW(getPayloadCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	meta := Metadata{"a": NewInt(1), "tag": NewString("x")}
	if _, _, err := h.Insert(7, []float32{1, 2, 3, 4}, 500*time.Millisecond, meta, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert: %v", err)
	}

	vec, got, ttl, sparse, _, ok := h.Get(7)
	if !ok {
		t.Fatal("Get(7) ok=false, want true")
	}
	if len(vec) != 4 || vec[0] != 1 || vec[3] != 4 {
		t.Fatalf("Get vec = %v, want [1 2 3 4]", vec)
	}
	if got["a"].Int != 1 || got["tag"].Str != "x" {
		t.Fatalf("Get meta = %v, want {a:1, tag:x}", got)
	}
	if ttl <= 0 || ttl > 500*time.Millisecond {
		t.Fatalf("Get ttl = %v, want (0, 500ms]", ttl)
	}
	if sparse != nil {
		t.Fatalf("Get sparse = %v, want nil", sparse)
	}

	// Deep-copy isolation: mutating the returned map/slice must not affect the arena.
	got["a"] = NewInt(999)
	vec[0] = -1
	_, got2, _, _, _, _ := h.Get(7)
	if got2["a"].Int != 1 {
		t.Fatalf("arena meta corrupted by caller mutation: a=%d, want 1", got2["a"].Int)
	}
	slot, _ := h.arena.Slot(7)
	if h.arena.Vec(slot)[0] != 1 {
		t.Fatalf("arena vec corrupted by caller mutation: %v", h.arena.Vec(slot))
	}
}

// TestGetAbsentTombstonedExpired covers the three not-found liveness cases.
func TestGetAbsentTombstonedExpired(t *testing.T) {
	h, err := newHNSW(getPayloadCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	// Absent.
	if _, _, _, _, _, ok := h.Get(1); ok {
		t.Fatal("Get(absent) ok=true, want false")
	}

	// Tombstoned (deleted).
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	h.Delete(1, CASCond{})
	if _, _, _, _, _, ok := h.Get(1); ok {
		t.Fatal("Get(tombstoned) ok=true, want false")
	}

	// TTL-expired (fake clock).
	if _, _, err := h.Insert(2, []float32{0, 1, 0, 0}, 50*time.Millisecond, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if _, _, _, _, _, ok := h.Get(2); !ok {
		t.Fatal("Get(2) pre-expiry ok=false, want true")
	}
	fakeNow += 100
	if _, _, _, _, _, ok := h.Get(2); ok {
		t.Fatal("Get(expired) ok=true, want false")
	}
}

// TestReindexCorrectness is THE landmine-#1 test: a merge that ADDS a field must
// make that field visible to filter-first search. Without the mandatory
// payloadIdx.reindex inside SetPayload, the new field would be invisible
// (under-cover → missed result). Overwrite/delete-keys/clear are checked to
// correctly REMOVE a field from the index too.
//
// CRITICAL — the test must genuinely exercise the FILTER-FIRST (index) path, not
// the graph fallback. payloadIdx.candidates(filter) only returns ok=true (→
// filter-first) when a posting list for the queried field-value EXISTS; if the
// queried value lived on id=1 alone, candidates would return ok=false and the
// search would fall to GRAPH traversal, which predicate-evals against the arena
// metadata (which SetMetadata updates regardless of reindex) — so the test would
// pass even with reindex removed and would NOT guard the landmine. To force
// filter-first we SEED ~10 OTHER points carrying the SAME field-value BEFORE the
// mutation, so the posting list already exists and candidates returns ok=true.
// The query vector is closest to id=1, so a filter-first candidate set MISSING
// id=1 (stale index) would fail to return it. We assert candidates ok=true to
// pin the path, then assert id=1 is among the (filter-first) results.
//
// Filler points (b/c/d held by ~10 seeders) keep the candidate set small and
// selective relative to N, so the planner's preferFilterFirst fires.
func TestReindexCorrectness(t *testing.T) {
	cfg := getPayloadCfg()
	cfg.FilterFirstThreshold = 10_000
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}

	// id=1 is the point we mutate; its vector is closest to the query so a
	// filter-first set that includes it returns it as the top hit.
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, Metadata{"a": NewInt(1)}, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 1: %v", err)
	}
	// Background corpus (a==i, distinct values, far from the query) → makes N
	// large so the seeded value-sets stay selective.
	for i := uint64(2); i <= 20; i++ {
		if _, _, err := h.Insert(i, []float32{0, float32(i), 0, 0}, 0, Metadata{"a": NewInt(int64(i))}, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	// SEEDERS — ~10 OTHER points carrying EACH value we will later filter on
	// (b==2, c==3, d==4), so those posting lists ALREADY EXIST in the index and
	// candidates(filter) returns ok=true → filter-first is the path taken. Their
	// vectors are far from the query, so they never out-rank id=1; they only
	// populate the posting lists. ids 100..149.
	seedID := uint64(100)
	for _, kv := range []struct {
		field string
		val   Value
	}{
		{"b", NewInt(2)},
		{"c", NewInt(3)},
		{"d", NewInt(4)},
	} {
		for j := 0; j < 10; j++ {
			meta := Metadata{kv.field: kv.val}
			if _, _, err := h.Insert(seedID, []float32{0, 0, float32(seedID), 0}, 0, meta, nil, nil, CASCond{}); err != nil {
				t.Fatalf("Insert seeder %d: %v", seedID, err)
			}
			seedID++
		}
	}

	q := []float32{1, 0, 0, 0}

	// pinFilterFirst asserts the payload index can narrow `f` (ok=true), i.e. the
	// search will take the FILTER-FIRST path — without this guard the test could
	// silently fall back to graph and stop guarding the landmine.
	pinFilterFirst := func(name string, f Filter) {
		t.Helper()
		if _, ok := h.payloadIdx.candidates(f, cfg.FilterFirstThreshold); !ok {
			t.Fatalf("%s: candidates ok=false (graph fallback) — test would not exercise filter-first", name)
		}
	}

	// Merge ADD field b==2 to id=1. The b==2 posting list already holds the 10
	// seeders; reindex must ADD id=1 to it. Without reindex, filter-first's
	// candidate set for b==2 = {seeders}, MISSING id=1 → id=1 not returned.
	if _, _, _, err := h.SetPayload(1, Metadata{"b": NewInt(2)}, nil, CASCond{}); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}
	fb := Filter{Op: FilterEq, Field: "b", Value: NewInt(2)}
	pinFilterFirst("b==2", fb)
	// Filter b==2 MUST find id=1 via filter-first (proves reindex ran). id=1's
	// vector is closest to q, so it is the top result among the matched set.
	res, err := h.SearchFiltered(q, 5, fb)
	if err != nil {
		t.Fatalf("SearchFiltered b==2: %v", err)
	}
	if !containsID(res, 1) {
		t.Fatalf("filter b==2 = %v, want to contain id=1 (reindex of added field missing)", resultIDs(res))
	}
	// The original field a==1 is still present (merge retained it).
	pinFilterFirst("a==1", Filter{Op: FilterEq, Field: "a", Value: NewInt(1)})
	res, _ = h.SearchFiltered(q, 5, Filter{Op: FilterEq, Field: "a", Value: NewInt(1)})
	if len(res) != 1 || res[0].ID != 1 {
		t.Fatalf("filter a==1 after merge = %v, want [id=1]", resultIDs(res))
	}

	// Overwrite id=1 with only c==3 → old fields a,b no longer matched (id=1 must
	// be DROPPED from the a and b posting lists), c==3 now matched (id=1 ADDED to
	// the c posting list which the seeders already populate).
	if _, _, _, err := h.OverwritePayload(1, Metadata{"c": NewInt(3)}, nil, CASCond{}); err != nil {
		t.Fatalf("OverwritePayload: %v", err)
	}
	pinFilterFirst("a==1 post-overwrite", Filter{Op: FilterEq, Field: "a", Value: NewInt(1)})
	if res, _ := h.SearchFiltered(q, 5, Filter{Op: FilterEq, Field: "a", Value: NewInt(1)}); containsID(res, 1) {
		t.Fatalf("filter a==1 after overwrite = %v, want NOT to contain id=1", resultIDs(res))
	}
	pinFilterFirst("b==2 post-overwrite", fb)
	if res, _ := h.SearchFiltered(q, 5, fb); containsID(res, 1) {
		t.Fatalf("filter b==2 after overwrite = %v, want NOT to contain id=1", resultIDs(res))
	}
	fc := Filter{Op: FilterEq, Field: "c", Value: NewInt(3)}
	pinFilterFirst("c==3 post-overwrite", fc)
	if res, _ := h.SearchFiltered(q, 5, fc); !containsID(res, 1) {
		t.Fatalf("filter c==3 after overwrite = %v, want to contain id=1", resultIDs(res))
	}

	// Delete-keys c → c no longer matched (id=1 DROPPED from the c posting list;
	// the seeders remain so the list — and ok=true — persist).
	if _, _, _, err := h.DeletePayloadKeys(1, []string{"c"}, CASCond{}); err != nil {
		t.Fatalf("DeletePayloadKeys: %v", err)
	}
	pinFilterFirst("c==3 post-delete", fc)
	if res, _ := h.SearchFiltered(q, 5, fc); containsID(res, 1) {
		t.Fatalf("filter c==3 after delete-keys = %v, want NOT to contain id=1", resultIDs(res))
	}

	// Re-add d==4 then clear → no field matches (clear must DROP id=1 from the d
	// posting list, which the seeders keep populated so the path stays filter-first).
	if _, _, _, err := h.SetPayload(1, Metadata{"d": NewInt(4)}, nil, CASCond{}); err != nil {
		t.Fatalf("SetPayload d: %v", err)
	}
	fd := Filter{Op: FilterEq, Field: "d", Value: NewInt(4)}
	pinFilterFirst("d==4 pre-clear", fd)
	if res, _ := h.SearchFiltered(q, 5, fd); !containsID(res, 1) {
		t.Fatalf("filter d==4 pre-clear = %v, want to contain id=1", resultIDs(res))
	}
	if _, _, _, err := h.ClearPayload(1, CASCond{}); err != nil {
		t.Fatalf("ClearPayload: %v", err)
	}
	pinFilterFirst("d==4 post-clear", fd)
	if res, _ := h.SearchFiltered(q, 5, fd); containsID(res, 1) {
		t.Fatalf("filter d==4 after clear = %v, want NOT to contain id=1", resultIDs(res))
	}
}

// containsID reports whether any result has the given user id.
func containsID(rs []Result, id uint64) bool {
	for _, r := range rs {
		if r.ID == id {
			return true
		}
	}
	return false
}

// TestPayloadSemantics is the merge/overwrite/delete-keys/clear semantics table:
// each op produces exactly the right resulting payload (verified via Get).
func TestPayloadSemantics(t *testing.T) {
	h, err := newHNSW(getPayloadCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	base := Metadata{"a": NewInt(1), "b": NewString("x")}
	mustInsert := func() {
		h.Delete(99, CASCond{})
		if _, _, err := h.Insert(99, []float32{1, 0, 0, 0}, 0, cloneMeta(base), nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert: %v", err)
		}
	}

	// Merge: overwrite a, add c; b retained.
	mustInsert()
	got, _, _, err := h.SetPayload(99, Metadata{"a": NewInt(2), "c": NewInt(3)}, nil, CASCond{})
	if err != nil {
		t.Fatalf("SetPayload: %v", err)
	}
	wantMerge := Metadata{"a": NewInt(2), "b": NewString("x"), "c": NewInt(3)}
	if !metaEqual(got, wantMerge) {
		t.Fatalf("merge resulting = %v, want %v", got, wantMerge)
	}
	if _, g, _, _, _, _ := h.Get(99); !metaEqual(g, wantMerge) {
		t.Fatalf("merge stored = %v, want %v", g, wantMerge)
	}

	// Overwrite: whole payload replaced.
	mustInsert()
	got, _, _, _ = h.OverwritePayload(99, Metadata{"z": NewInt(9)}, nil, CASCond{})
	if !metaEqual(got, Metadata{"z": NewInt(9)}) {
		t.Fatalf("overwrite resulting = %v, want {z:9}", got)
	}

	// Delete-keys: a removed, b retained; absent key is a no-op.
	mustInsert()
	got, _, _, _ = h.DeletePayloadKeys(99, []string{"a", "missing"}, CASCond{})
	if !metaEqual(got, Metadata{"b": NewString("x")}) {
		t.Fatalf("delete-keys resulting = %v, want {b:x}", got)
	}

	// Delete-keys removing ALL keys → nil payload.
	mustInsert()
	got, _, _, _ = h.DeletePayloadKeys(99, []string{"a", "b"}, CASCond{})
	if len(got) != 0 {
		t.Fatalf("delete-keys all resulting = %v, want empty", got)
	}

	// Clear: empty payload.
	mustInsert()
	got, _, _, _ = h.ClearPayload(99, CASCond{})
	if len(got) != 0 {
		t.Fatalf("clear resulting = %v, want empty", got)
	}
	if _, g, _, _, _, _ := h.Get(99); len(g) != 0 {
		t.Fatalf("clear stored = %v, want empty", g)
	}
}

// TestPayloadPreservesVectorAndTTL proves payload ops never touch the vector or TTL.
func TestPayloadPreservesVectorAndTTL(t *testing.T) {
	h, err := newHNSW(getPayloadCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }
	if _, _, err := h.Insert(5, []float32{9, 8, 7, 6}, time.Second, Metadata{"a": NewInt(1)}, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if _, _, _, err := h.SetPayload(5, Metadata{"b": NewInt(2)}, nil, CASCond{}); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}
	vec, _, ttl, _, _, ok := h.Get(5)
	if !ok {
		t.Fatal("Get ok=false")
	}
	if vec[0] != 9 || vec[3] != 6 {
		t.Fatalf("vector changed by payload op: %v", vec)
	}
	if ttl <= 0 || ttl > time.Second {
		t.Fatalf("ttl changed by payload op: %v", ttl)
	}
}

// TestPayloadMutationDeadPoint covers ErrIDNotFound for absent/tombstoned/expired.
func TestPayloadMutationDeadPoint(t *testing.T) {
	h, err := newHNSW(getPayloadCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	// Absent.
	if _, _, _, err := h.SetPayload(1, Metadata{"a": NewInt(1)}, nil, CASCond{}); err != ErrIDNotFound {
		t.Fatalf("SetPayload(absent) err = %v, want ErrIDNotFound", err)
	}

	// Tombstoned.
	_, _, _ = h.Insert(1, []float32{1, 0, 0, 0}, 0, nil, nil, nil, CASCond{})
	h.Delete(1, CASCond{})
	if _, _, _, err := h.OverwritePayload(1, Metadata{"a": NewInt(1)}, nil, CASCond{}); err != ErrIDNotFound {
		t.Fatalf("OverwritePayload(tombstoned) err = %v, want ErrIDNotFound", err)
	}
	if _, _, _, err := h.DeletePayloadKeys(1, []string{"a"}, CASCond{}); err != ErrIDNotFound {
		t.Fatalf("DeletePayloadKeys(tombstoned) err = %v, want ErrIDNotFound", err)
	}

	// Expired.
	_, _, _ = h.Insert(2, []float32{0, 1, 0, 0}, 50*time.Millisecond, nil, nil, nil, CASCond{})
	fakeNow += 100
	if _, _, _, err := h.ClearPayload(2, CASCond{}); err != ErrIDNotFound {
		t.Fatalf("ClearPayload(expired) err = %v, want ErrIDNotFound", err)
	}
}

// TestCollectionPayloadDelegates covers the Collection-level Get + payload methods
// (no-WAL path) and their ErrIDNotFound behavior.
func TestCollectionPayloadDelegates(t *testing.T) {
	c, err := NewCollection("t", getPayloadCfg())
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	defer func() { _ = c.Close() }()
	if err := c.Insert(1, []float32{1, 0, 0, 0}, 0, Metadata{"a": NewInt(1)}, nil); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if err := c.SetPayload(1, Metadata{"b": NewInt(2)}, nil); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}
	_, meta, _, _, _, ok := c.Get(1)
	if !ok || meta["a"].Int != 1 || meta["b"].Int != 2 {
		t.Fatalf("Get after merge = %v ok=%v, want {a:1,b:2}", meta, ok)
	}
	if err := c.ClearPayload(2); err != ErrIDNotFound {
		t.Fatalf("ClearPayload(absent) = %v, want ErrIDNotFound", err)
	}
	if _, _, _, _, _, ok := c.Get(2); ok {
		t.Fatal("Get(absent) ok=true, want false")
	}
}

// metaEqual compares two Metadata maps by key set and Value equality.
func metaEqual(a, b Metadata) bool {
	if len(a) != len(b) {
		return false
	}
	for k, va := range a {
		vb, ok := b[k]
		if !ok || !va.Equal(vb) {
			return false
		}
	}
	return true
}

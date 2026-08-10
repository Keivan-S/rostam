// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"sort"
	"testing"
	"time"
)

// orderEngine wraps a dense/ivf index with the few operations + test hooks the
// cached-order_by tests need, so the assertions run identically against both.
type orderEngine struct {
	name           string
	insert         func(t *testing.T, id uint64, meta Metadata)
	del            func(t *testing.T, id uint64)
	setPayload     func(t *testing.T, id uint64, patch Metadata)
	scrollPage     func(pred Predicate, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) ([]Document, uint64, bool)
	orderRebuilds  func() uint64
	scrollRebuilds func() uint64
	setNow         func(fn func() int64)
}

func denseOrderEngine(t *testing.T) *orderEngine {
	t.Helper()
	h, err := newHNSW(Config{Dim: 4, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 7, Metric: L2})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	vecFor := func(id uint64) []float32 { return []float32{float32(id), float32(id % 7), float32(id % 3), 1} }
	return &orderEngine{
		name: "dense",
		insert: func(t *testing.T, id uint64, meta Metadata) {
			if _, _, err := h.Insert(id, vecFor(id), 0, meta, nil, nil, CASCond{}); err != nil {
				t.Fatalf("dense Insert %d: %v", id, err)
			}
		},
		del: func(t *testing.T, id uint64) {
			if ok, err := h.Delete(id, CASCond{}); err != nil || !ok {
				t.Fatalf("dense Delete %d: ok=%v err=%v", id, ok, err)
			}
		},
		setPayload: func(t *testing.T, id uint64, patch Metadata) {
			if _, _, _, err := h.SetPayload(id, patch, nil, CASCond{}); err != nil {
				t.Fatalf("dense SetPayload %d: %v", id, err)
			}
		},
		scrollPage: func(pred Predicate, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) ([]Document, uint64, bool) {
			return h.scrollPage(Filter{}, pred, nil, order, afterID, afterKey, hasAfter, limit)
		},
		orderRebuilds:  func() uint64 { return h.orderRebuilds },
		scrollRebuilds: func() uint64 { return h.scrollRebuilds },
		setNow:         func(fn func() int64) { h.now = fn },
	}
}

func ivfOrderEngine(t *testing.T) *orderEngine {
	t.Helper()
	ix, err := newIVF(ivfTestConfig(4))
	if err != nil {
		t.Fatalf("newIVF: %v", err)
	}
	vecFor := func(id uint64) []float32 { return []float32{float32(id), float32(id % 7), float32(id % 3), 1} }
	return &orderEngine{
		name: "ivf",
		insert: func(t *testing.T, id uint64, meta Metadata) {
			if _, _, err := ix.Insert(id, vecFor(id), 0, meta, nil, nil, CASCond{}); err != nil {
				t.Fatalf("ivf Insert %d: %v", id, err)
			}
		},
		del: func(t *testing.T, id uint64) {
			if ok, err := ix.Delete(id, CASCond{}); err != nil || !ok {
				t.Fatalf("ivf Delete %d: ok=%v err=%v", id, ok, err)
			}
		},
		setPayload: func(t *testing.T, id uint64, patch Metadata) {
			if _, _, _, err := ix.SetPayload(id, patch, nil, CASCond{}); err != nil {
				t.Fatalf("ivf SetPayload %d: %v", id, err)
			}
		},
		scrollPage: func(pred Predicate, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) ([]Document, uint64, bool) {
			return ix.scrollPage(Filter{}, pred, nil, order, afterID, afterKey, hasAfter, limit)
		},
		orderRebuilds:  func() uint64 { return ix.orderRebuilds },
		scrollRebuilds: func() uint64 { return ix.scrollRebuilds },
		setNow:         func(fn func() int64) { ix.now = fn },
	}
}

func eachOrderEngine(t *testing.T, fn func(t *testing.T, e *orderEngine)) {
	t.Helper()
	t.Run("dense", func(t *testing.T) { fn(t, denseOrderEngine(t)) })
	t.Run("ivf", func(t *testing.T) { fn(t, ivfOrderEngine(t)) })
}

// bruteOrder returns the ground-truth (value, id) order for the given direction over
// (id -> value), EXCLUDING ids whose value is absent (mirrors the EXCLUDE policy).
func bruteOrder(vals map[uint64]float64, desc bool) []uint64 {
	type kv struct {
		id  uint64
		val float64
	}
	rows := make([]kv, 0, len(vals))
	for id, v := range vals {
		rows = append(rows, kv{id, v})
	}
	sort.Slice(rows, func(i, j int) bool {
		return OrderLess(rows[i].val, rows[i].id, rows[j].val, rows[j].id, desc)
	})
	out := make([]uint64, len(rows))
	for i, r := range rows {
		out[i] = r.id
	}
	return out
}

// scrollAllOrdered pages through an order_by scroll with the given limit and returns
// the concatenated ids across pages (the warm-path reuse exercises the cache).
func scrollAllOrdered(e *orderEngine, order *OrderBy, limit int) []uint64 {
	var collected []uint64
	var afterID uint64
	var afterKey float64
	hasAfter := false
	for {
		docs, nextAfter, hasMore := e.scrollPage(nil, order, afterID, afterKey, hasAfter, limit)
		for _, d := range docs {
			collected = append(collected, d.ID)
		}
		if !hasMore {
			break
		}
		afterID = nextAfter
		// Recover the cursor key from the last doc's ORDER field (the v2 cursor) — must
		// use order.Key, not a hardcoded field, or the seek never advances.
		last := docs[len(docs)-1]
		if v, ok := last.Metadata[order.Key]; ok {
			if f, ok2 := numericValue(v); ok2 {
				afterKey = f
			}
		}
		hasAfter = true
	}
	return collected
}

// TestOrderSnapshotIdenticalToBruteForce: a paged order_by scroll (asc + desc) yields
// EXACTLY the brute-force (value, id) sorted order — the cache changes perf not output.
func TestOrderSnapshotIdenticalToBruteForce(t *testing.T) {
	eachOrderEngine(t, func(t *testing.T, e *orderEngine) {
		vals := map[uint64]float64{}
		ids := []uint64{50, 3, 17, 8, 99, 42, 7, 1, 25, 12, 60, 33}
		for i, id := range ids {
			v := float64((i*37)%11) - 5 // many value ties to stress the id tiebreak
			vals[id] = v
			e.insert(t, id, Metadata{"v": NewFloat(v)})
		}
		for _, desc := range []bool{false, true} {
			order := &OrderBy{Key: "v", Desc: desc}
			want := bruteOrder(vals, desc)
			for _, limit := range []int{0, 1, 3, 5, 100} {
				got := scrollAllOrdered(e, order, limit)
				if len(got) != len(want) {
					t.Fatalf("desc=%v limit=%d: got %d ids, want %d (%v vs %v)", desc, limit, len(got), len(want), got, want)
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("desc=%v limit=%d: order[%d]=%d, want %d (got %v want %v)", desc, limit, i, got[i], want[i], got, want)
					}
				}
			}
		}
	})
}

// TestOrderSnapshotWarmAcrossPages: paging through one scroll rebuilds the snapshot
// EXACTLY ONCE — page-to-page reuse (warmth), mirroring the scrollRebuilds assertion.
func TestOrderSnapshotWarmAcrossPages(t *testing.T) {
	eachOrderEngine(t, func(t *testing.T, e *orderEngine) {
		for i := uint64(0); i < 30; i++ {
			e.insert(t, i, Metadata{"v": NewFloat(float64(i % 9))})
		}
		order := &OrderBy{Key: "v"}
		// Page 1 builds the snapshot (one rebuild).
		_, nextAfter, hasMore := e.scrollPage(nil, order, 0, 0, false, 5)
		afterRebuilds := e.orderRebuilds()
		if !hasMore {
			t.Fatalf("expected more than one page with limit 5 over 30 ids")
		}
		_ = nextAfter
		// Walk all remaining pages of THIS scroll; no mutation occurs between pages,
		// so the warm snapshot must be reused (zero further rebuilds).
		got := scrollAllOrdered(e, order, 5)
		if len(got) != 30 {
			t.Fatalf("warm scroll returned %d ids, want 30", len(got))
		}
		if e.orderRebuilds() != afterRebuilds {
			t.Fatalf("order snapshot rebuilt across pages: %d -> %d (must reuse warm)", afterRebuilds, e.orderRebuilds())
		}
	})
}

// TestOrderSnapshotInvalidateInsertDelete: after an insert or delete the next page
// rebuilds and reflects the change.
func TestOrderSnapshotInvalidateInsertDelete(t *testing.T) {
	eachOrderEngine(t, func(t *testing.T, e *orderEngine) {
		vals := map[uint64]float64{}
		for i := uint64(1); i <= 10; i++ {
			v := float64(i)
			vals[i] = v
			e.insert(t, i, Metadata{"v": NewFloat(v)})
		}
		order := &OrderBy{Key: "v"}
		_ = scrollAllOrdered(e, order, 0) // warm
		before := e.orderRebuilds()

		// Insert id 11 with a value that lands in the middle.
		vals[11] = 5.5
		e.insert(t, 11, Metadata{"v": NewFloat(5.5)})
		got := scrollAllOrdered(e, order, 4)
		if e.orderRebuilds() == before {
			t.Fatalf("snapshot not rebuilt after insert")
		}
		want := bruteOrder(vals, false)
		assertOrder(t, "after insert", got, want)

		before = e.orderRebuilds()
		// Delete id 5.
		delete(vals, 5)
		e.del(t, 5)
		got = scrollAllOrdered(e, order, 4)
		if e.orderRebuilds() == before {
			t.Fatalf("snapshot not rebuilt after delete")
		}
		assertOrder(t, "after delete", got, bruteOrder(vals, false))
	})
}

// TestOrderSnapshotInvalidateOnPayloadWrite is THE KEY test: a set_payload that moves
// a point's order-field value must rebuild the order snapshot and place the point in
// its NEW sorted position (this is exactly what reusing idSetVersion would get WRONG).
func TestOrderSnapshotInvalidateOnPayloadWrite(t *testing.T) {
	eachOrderEngine(t, func(t *testing.T, e *orderEngine) {
		vals := map[uint64]float64{}
		for i := uint64(1); i <= 10; i++ {
			v := float64(i)
			vals[i] = v
			e.insert(t, i, Metadata{"v": NewFloat(v)})
		}
		order := &OrderBy{Key: "v"}
		_ = scrollAllOrdered(e, order, 0) // warm the snapshot
		before := e.orderRebuilds()

		// Move id 1's value from 1 (first) to 100 (last).
		vals[1] = 100
		e.setPayload(t, 1, Metadata{"v": NewFloat(100)})

		got := scrollAllOrdered(e, order, 3)
		if e.orderRebuilds() == before {
			t.Fatalf("order snapshot NOT rebuilt after a payload write to the order field (stale ordering)")
		}
		want := bruteOrder(vals, false)
		assertOrder(t, "after payload reorder", got, want)
		if got[len(got)-1] != 1 {
			t.Fatalf("after raising id 1's value it must sort LAST: got %v", got)
		}
	})
}

// TestOrderPayloadWriteDoesNotInvalidateIDScroll proves the separate-counter design:
// a set_payload must NOT rebuild the id-scroll scrollSnap (no regression). Dense only
// here via the existing scrollRebuilds hook; ivf shares the identical mechanism.
func TestOrderPayloadWriteDoesNotInvalidateIDScroll(t *testing.T) {
	eachOrderEngine(t, func(t *testing.T, e *orderEngine) {
		for i := uint64(1); i <= 10; i++ {
			e.insert(t, i, Metadata{"v": NewFloat(float64(i))})
		}
		// Warm the id-scroll snapshot (no order).
		_, _, _ = e.scrollPage(nil, nil, 0, 0, false, 0)
		beforeID := e.scrollRebuilds()
		// A payload write to the order field.
		e.setPayload(t, 1, Metadata{"v": NewFloat(100)})
		// The id-scroll snapshot must still be warm.
		_, _, _ = e.scrollPage(nil, nil, 0, 0, false, 0)
		if e.scrollRebuilds() != beforeID {
			t.Fatalf("payload write invalidated the id-scroll snapshot: %d -> %d (regression)", beforeID, e.scrollRebuilds())
		}
	})
}

// TestOrderSnapshotCapEviction: scrolling by more than orderCacheCap distinct
// (field, direction) combos stays bounded and every result stays correct.
func TestOrderSnapshotCapEviction(t *testing.T) {
	eachOrderEngine(t, func(t *testing.T, e *orderEngine) {
		// Each point carries several distinct numeric fields.
		fields := []string{"a", "b", "c", "d", "e", "f"} // 6 > cap(4)
		valsByField := map[string]map[uint64]float64{}
		for _, f := range fields {
			valsByField[f] = map[uint64]float64{}
		}
		for i := uint64(1); i <= 12; i++ {
			meta := Metadata{}
			for fi, f := range fields {
				v := float64((int(i)*7 + fi*3) % 13)
				meta[f] = NewFloat(v)
				valsByField[f][i] = v
			}
			e.insert(t, i, meta)
		}
		// Scroll asc by each field; with 6 fields and cap 4 the map must stay bounded.
		for _, f := range fields {
			order := &OrderBy{Key: f}
			got := scrollAllOrdered(e, order, 5)
			assertOrder(t, "cap field "+f, got, bruteOrder(valsByField[f], false))
		}
		// Re-scroll the first field: it was likely evicted, so a rebuild is fine, but
		// the result must still be exactly correct.
		got := scrollAllOrdered(e, &OrderBy{Key: fields[0]}, 5)
		assertOrder(t, "cap re-scroll", got, bruteOrder(valsByField[fields[0]], false))
	})
}

// TestOrderSnapshotTTLDroppedByWalk: a point whose POINT TTL expires WITHOUT a sweep
// stays in the cached snapshot rows (lazy aging bumps nothing) but is dropped by the
// walk's TTL gate (mirrors the id-scroll lazy-TTL contract). Dense + ivf.
func TestOrderSnapshotTTLDroppedByWalk(t *testing.T) {
	t.Run("dense", func(t *testing.T) {
		h, err := newHNSW(Config{Dim: 4, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 7, Metric: L2})
		if err != nil {
			t.Fatalf("newHNSW: %v", err)
		}
		now := int64(1000)
		h.now = func() int64 { return now }
		// id 5 gets a short TTL; the rest are permanent.
		for i := uint64(1); i <= 6; i++ {
			ttl := time.Duration(0)
			if i == 5 {
				ttl = 100 * time.Millisecond
			}
			if _, _, err := h.Insert(i, []float32{float32(i), 0, 0, 1}, ttl, Metadata{"v": NewFloat(float64(i))}, nil, nil, CASCond{}); err != nil {
				t.Fatalf("Insert %d: %v", i, err)
			}
		}
		order := &OrderBy{Key: "v"}
		// Warm the snapshot while id 5 is still live.
		docs, _, _ := h.scrollPage(Filter{}, nil, nil, order, 0, 0, false, 0)
		if !docsContainID(docs, 5) {
			t.Fatalf("id 5 should be present before TTL: %v", docIDs(docs))
		}
		rebuildsAfterWarm := h.orderRebuilds
		// Advance the clock past id 5's deadline WITHOUT sweeping (no mutation/bump).
		now = 2000
		docs2, _, _ := h.scrollPage(Filter{}, nil, nil, order, 0, 0, false, 0)
		if h.orderRebuilds != rebuildsAfterWarm {
			t.Fatalf("TTL aging triggered a snapshot rebuild (must be lazy): %d -> %d", rebuildsAfterWarm, h.orderRebuilds)
		}
		if docsContainID(docs2, 5) {
			t.Fatalf("TTL-expired id 5 leaked into results: %v", docIDs(docs2))
		}
		// It must still be in the (un-rebuilt) snapshot rows — dropped by the WALK.
		snap := h.orderSnaps[orderCacheKey{"v", false}]
		if snap == nil || !rowsContain(snap.rows, 5) {
			t.Fatalf("TTL-expired id 5 should remain in the lazy snapshot rows")
		}
	})
	t.Run("ivf", func(t *testing.T) {
		ix, err := newIVF(ivfTestConfig(4))
		if err != nil {
			t.Fatalf("newIVF: %v", err)
		}
		now := int64(1000)
		ix.now = func() int64 { return now }
		for i := uint64(1); i <= 6; i++ {
			ttl := time.Duration(0)
			if i == 5 {
				ttl = 100 * time.Millisecond
			}
			if _, _, err := ix.Insert(i, []float32{float32(i), 0, 0, 1}, ttl, Metadata{"v": NewFloat(float64(i))}, nil, nil, CASCond{}); err != nil {
				t.Fatalf("Insert %d: %v", i, err)
			}
		}
		order := &OrderBy{Key: "v"}
		docs, _, _ := ix.scrollPage(Filter{}, nil, nil, order, 0, 0, false, 0)
		if !docsContainID(docs, 5) {
			t.Fatalf("id 5 should be present before TTL: %v", docIDs(docs))
		}
		rebuildsAfterWarm := ix.orderRebuilds
		now = 2000
		docs2, _, _ := ix.scrollPage(Filter{}, nil, nil, order, 0, 0, false, 0)
		if ix.orderRebuilds != rebuildsAfterWarm {
			t.Fatalf("TTL aging triggered a snapshot rebuild (must be lazy): %d -> %d", rebuildsAfterWarm, ix.orderRebuilds)
		}
		if docsContainID(docs2, 5) {
			t.Fatalf("TTL-expired id 5 leaked into results: %v", docIDs(docs2))
		}
		snap := ix.orderSnaps[orderCacheKey{"v", false}]
		if snap == nil || !rowsContain(snap.rows, 5) {
			t.Fatalf("TTL-expired id 5 should remain in the lazy snapshot rows")
		}
	})
}

func docsContainID(docs []Document, id uint64) bool {
	for _, d := range docs {
		if d.ID == id {
			return true
		}
	}
	return false
}

func rowsContain(rows []OrderedID, id uint64) bool {
	for _, r := range rows {
		if r.ID == id {
			return true
		}
	}
	return false
}

func assertOrder(t *testing.T, label string, got, want []uint64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d ids, want %d (%v vs %v)", label, len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s: [%d]=%d want %d (got %v want %v)", label, i, got[i], want[i], got, want)
		}
	}
}

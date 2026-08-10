// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"testing"
	"time"
)

// namedMVOrderEngine wraps a named/MV index with the operations + test hooks the
// cached-order_by tests need, so the assertions run identically against both
// families (mirroring orderEngine for dense/ivf in order_snapshot_test.go).
type namedMVOrderEngine struct {
	name          string
	insert        func(t *testing.T, id uint64, meta Metadata)
	del           func(t *testing.T, id uint64)
	setPayload    func(t *testing.T, id uint64, patch Metadata)
	scrollPage    func(pred Predicate, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) ([]Document, uint64, bool)
	orderRebuilds func() uint64
	dataVersion   func() uint64
	setNow        func(fn func() int64)
	snapRows      func(field string, desc bool) ([]OrderedID, bool)
	insertKeyTTL  func(t *testing.T, id uint64, meta Metadata, keyTTLMs map[string]int64)
}

func namedTestEngine(t *testing.T) *namedMVOrderEngine {
	t.Helper()
	nc, err := NewNamedCollection("c", map[string]NamedVectorParams{
		"v": {Dim: 4, M: 16, EfConstruction: 200, EfSearch: 64, Metric: L2},
	})
	if err != nil {
		t.Fatalf("newNamedCollection: %v", err)
	}
	vecFor := func(id uint64) []float32 { return []float32{float32(id), float32(id % 7), float32(id % 3), 1} }
	return &namedMVOrderEngine{
		name: "named",
		insert: func(t *testing.T, id uint64, meta Metadata) {
			if err := nc.Insert(id, map[string][]float32{"v": vecFor(id)}, meta, 0); err != nil {
				t.Fatalf("named Insert %d: %v", id, err)
			}
		},
		del: func(t *testing.T, id uint64) {
			if ok, err := nc.Delete(id); err != nil || !ok {
				t.Fatalf("named Delete %d: ok=%v err=%v", id, ok, err)
			}
		},
		setPayload: func(t *testing.T, id uint64, patch Metadata) {
			if err := nc.SetPayload(id, patch, nil); err != nil {
				t.Fatalf("named SetPayload %d: %v", id, err)
			}
		},
		scrollPage: func(pred Predicate, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) ([]Document, uint64, bool) {
			return nc.scrollPage(Filter{}, pred, order, afterID, afterKey, hasAfter, limit)
		},
		orderRebuilds: func() uint64 { return nc.orderRebuilds },
		dataVersion:   func() uint64 { return nc.dataVersion },
		setNow:        func(fn func() int64) { nc.now = fn },
		snapRows: func(field string, desc bool) ([]OrderedID, bool) {
			s, ok := nc.orderSnaps[orderCacheKey{field, desc}]
			if !ok {
				return nil, false
			}
			return s.rows, true
		},
		insertKeyTTL: func(t *testing.T, id uint64, meta Metadata, keyTTLMs map[string]int64) {
			if _, err := nc.InsertCASKeyTTL(id, map[string][]float32{"v": vecFor(id)}, nil, meta, 0, keyTTLMs, CASCond{}); err != nil {
				t.Fatalf("named InsertCASKeyTTL %d: %v", id, err)
			}
		},
	}
}

func mvTestEngine(t *testing.T) *namedMVOrderEngine {
	t.Helper()
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4})
	if err != nil {
		t.Fatalf("NewMultiVectorIndex: %v", err)
	}
	tokFor := func(id uint64) [][]float32 {
		return [][]float32{{float32(id), float32(id % 7), float32(id % 3), 1}}
	}
	return &namedMVOrderEngine{
		name: "mv",
		insert: func(t *testing.T, id uint64, meta Metadata) {
			if err := m.Add(id, tokFor(id), meta); err != nil {
				t.Fatalf("mv Add %d: %v", id, err)
			}
		},
		del: func(t *testing.T, id uint64) {
			if ok := m.Delete(id); !ok {
				t.Fatalf("mv Delete %d: not present", id)
			}
		},
		setPayload: func(t *testing.T, id uint64, patch Metadata) {
			if err := m.SetPayload(id, patch, nil); err != nil {
				t.Fatalf("mv SetPayload %d: %v", id, err)
			}
		},
		scrollPage: func(pred Predicate, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) ([]Document, uint64, bool) {
			return m.scrollPage(Filter{}, pred, order, afterID, afterKey, hasAfter, limit)
		},
		orderRebuilds: func() uint64 { return m.orderRebuilds },
		dataVersion:   func() uint64 { return m.dataVersion },
		setNow:        func(fn func() int64) { m.now = fn },
		snapRows: func(field string, desc bool) ([]OrderedID, bool) {
			s, ok := m.orderSnaps[orderCacheKey{field, desc}]
			if !ok {
				return nil, false
			}
			return s.rows, true
		},
		insertKeyTTL: func(t *testing.T, id uint64, meta Metadata, keyTTLMs map[string]int64) {
			if _, err := m.AddCASKeyTTL(id, tokFor(id), meta, keyTTLMs, CASCond{}); err != nil {
				t.Fatalf("mv AddCASKeyTTL %d: %v", id, err)
			}
		},
	}
}

func eachNamedMVEngine(t *testing.T, fn func(t *testing.T, e *namedMVOrderEngine)) {
	t.Helper()
	t.Run("named", func(t *testing.T) { fn(t, namedTestEngine(t)) })
	t.Run("mv", func(t *testing.T) { fn(t, mvTestEngine(t)) })
}

// scrollAllNamedMV pages through an order_by scroll with the given limit and returns
// the concatenated ids across pages (the warm-path reuse exercises the cache).
func scrollAllNamedMV(e *namedMVOrderEngine, order *OrderBy, limit int) []uint64 {
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

// TestNamedMVOrderSnapshotIdenticalToBruteForce: a paged order_by scroll (asc + desc)
// yields EXACTLY the brute-force (value, id) sorted order for both families.
func TestNamedMVOrderSnapshotIdenticalToBruteForce(t *testing.T) {
	eachNamedMVEngine(t, func(t *testing.T, e *namedMVOrderEngine) {
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
				got := scrollAllNamedMV(e, order, limit)
				assertOrder(t, e.name, got, want)
			}
		}
	})
}

// TestNamedMVOrderSnapshotWarmAcrossPages: paging through one scroll rebuilds the
// snapshot EXACTLY ONCE — page-to-page reuse (warmth).
func TestNamedMVOrderSnapshotWarmAcrossPages(t *testing.T) {
	eachNamedMVEngine(t, func(t *testing.T, e *namedMVOrderEngine) {
		for i := uint64(0); i < 30; i++ {
			e.insert(t, i, Metadata{"v": NewFloat(float64(i % 9))})
		}
		order := &OrderBy{Key: "v"}
		_, _, hasMore := e.scrollPage(nil, order, 0, 0, false, 5)
		if !hasMore {
			t.Fatalf("expected more than one page with limit 5 over 30 ids")
		}
		afterRebuilds := e.orderRebuilds()
		got := scrollAllNamedMV(e, order, 5)
		if len(got) != 30 {
			t.Fatalf("warm scroll returned %d ids, want 30", len(got))
		}
		if e.orderRebuilds() != afterRebuilds {
			t.Fatalf("order snapshot rebuilt across pages: %d -> %d (must reuse warm)", afterRebuilds, e.orderRebuilds())
		}
	})
}

// TestNamedMVOrderSnapshotInvalidateInsertDelete: after an insert or delete the next
// page rebuilds and reflects the change.
func TestNamedMVOrderSnapshotInvalidateInsertDelete(t *testing.T) {
	eachNamedMVEngine(t, func(t *testing.T, e *namedMVOrderEngine) {
		vals := map[uint64]float64{}
		for i := uint64(1); i <= 10; i++ {
			v := float64(i)
			vals[i] = v
			e.insert(t, i, Metadata{"v": NewFloat(v)})
		}
		order := &OrderBy{Key: "v"}
		_ = scrollAllNamedMV(e, order, 0) // warm
		before := e.orderRebuilds()

		vals[11] = 5.5
		e.insert(t, 11, Metadata{"v": NewFloat(5.5)})
		got := scrollAllNamedMV(e, order, 4)
		if e.orderRebuilds() == before {
			t.Fatalf("snapshot not rebuilt after insert")
		}
		assertOrder(t, "after insert", got, bruteOrder(vals, false))

		before = e.orderRebuilds()
		delete(vals, 5)
		e.del(t, 5)
		got = scrollAllNamedMV(e, order, 4)
		if e.orderRebuilds() == before {
			t.Fatalf("snapshot not rebuilt after delete")
		}
		assertOrder(t, "after delete", got, bruteOrder(vals, false))
	})
}

// TestNamedMVOrderSnapshotInvalidateOnPayloadWrite is THE KEY test: a set_payload that
// moves a point's order-field value must rebuild the order snapshot and place the
// point in its NEW sorted position (exactly what a version-counter that ignored
// payloads would get WRONG).
func TestNamedMVOrderSnapshotInvalidateOnPayloadWrite(t *testing.T) {
	eachNamedMVEngine(t, func(t *testing.T, e *namedMVOrderEngine) {
		vals := map[uint64]float64{}
		for i := uint64(1); i <= 10; i++ {
			v := float64(i)
			vals[i] = v
			e.insert(t, i, Metadata{"v": NewFloat(v)})
		}
		order := &OrderBy{Key: "v"}
		_ = scrollAllNamedMV(e, order, 0) // warm
		before := e.orderRebuilds()

		// Move id 1's value from 1 (first) to 100 (last).
		vals[1] = 100
		e.setPayload(t, 1, Metadata{"v": NewFloat(100)})

		got := scrollAllNamedMV(e, order, 3)
		if e.orderRebuilds() == before {
			t.Fatalf("order snapshot NOT rebuilt after a payload write to the order field (stale ordering)")
		}
		assertOrder(t, "after payload reorder", got, bruteOrder(vals, false))
		if got[len(got)-1] != 1 {
			t.Fatalf("after raising id 1's value it must sort LAST: got %v", got)
		}
	})
}

// TestNamedMVOrderSnapshotExactlyOnceBump proves a single set_payload bumps
// dataVersion by EXACTLY 1 (not 2 via the CAS-wrapper double-call): the public method
// funnels through one *Locked body that bumps once.
func TestNamedMVOrderSnapshotExactlyOnceBump(t *testing.T) {
	eachNamedMVEngine(t, func(t *testing.T, e *namedMVOrderEngine) {
		e.insert(t, 1, Metadata{"v": NewFloat(1)})
		// A single insert above bumped once already; measure deltas around each op.
		v0 := e.dataVersion()
		e.setPayload(t, 1, Metadata{"v": NewFloat(2)})
		if d := e.dataVersion() - v0; d != 1 {
			t.Fatalf("set_payload bumped dataVersion by %d, want exactly 1", d)
		}
		v1 := e.dataVersion()
		e.insert(t, 2, Metadata{"v": NewFloat(3)})
		if d := e.dataVersion() - v1; d != 1 {
			t.Fatalf("insert bumped dataVersion by %d, want exactly 1", d)
		}
		v2 := e.dataVersion()
		e.del(t, 2)
		if d := e.dataVersion() - v2; d != 1 {
			t.Fatalf("delete bumped dataVersion by %d, want exactly 1", d)
		}
	})
}

// TestNamedMVOrderSnapshotCapEviction: scrolling by more than orderCacheCap distinct
// (field, direction) combos stays bounded and every result stays correct.
func TestNamedMVOrderSnapshotCapEviction(t *testing.T) {
	eachNamedMVEngine(t, func(t *testing.T, e *namedMVOrderEngine) {
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
		for _, f := range fields {
			order := &OrderBy{Key: f}
			got := scrollAllNamedMV(e, order, 5)
			assertOrder(t, "cap field "+f, got, bruteOrder(valsByField[f], false))
		}
		got := scrollAllNamedMV(e, &OrderBy{Key: fields[0]}, 5)
		assertOrder(t, "cap re-scroll", got, bruteOrder(valsByField[fields[0]], false))
	})
}

// TestNamedOrderSnapshotPerKeyTTLDroppedByWalk: a per-key TTL on the ORDER field
// expiring (without a mutation/bump) leaves the id in the snapshot rows but the walk
// drops it (the order field is gone from the live view ⇒ no longer numeric, excluded)
// — the named family's lazy-TTL-in-the-walk contract.
func TestNamedOrderSnapshotPerKeyTTLDroppedByWalk(t *testing.T) {
	e := namedTestEngine(t)
	now := int64(1000)
	e.setNow(func() int64 { return now })
	for i := uint64(1); i <= 6; i++ {
		if i == 5 {
			// id 5's order field "v" carries a short per-key TTL.
			e.insertKeyTTL(t, i, Metadata{"v": NewFloat(float64(i))}, map[string]int64{"v": 100})
		} else {
			e.insert(t, i, Metadata{"v": NewFloat(float64(i))})
		}
	}
	order := &OrderBy{Key: "v"}
	docs, _, _ := e.scrollPage(nil, order, 0, 0, false, 0)
	if !docsContainID(docs, 5) {
		t.Fatalf("id 5 should be present before per-key TTL: %v", docIDs(docs))
	}
	rebuildsAfterWarm := e.orderRebuilds()
	// Advance past id 5's per-key deadline WITHOUT a mutation (no bump).
	now = 2000
	docs2, _, _ := e.scrollPage(nil, order, 0, 0, false, 0)
	if e.orderRebuilds() != rebuildsAfterWarm {
		t.Fatalf("per-key TTL aging triggered a rebuild (must be lazy): %d -> %d", rebuildsAfterWarm, e.orderRebuilds())
	}
	if docsContainID(docs2, 5) {
		t.Fatalf("per-key-TTL-expired id 5 leaked into results: %v", docIDs(docs2))
	}
	// It must remain in the (un-rebuilt) snapshot rows — dropped by the WALK.
	rows, ok := e.snapRows("v", false)
	if !ok || !rowsContain(rows, 5) {
		t.Fatalf("per-key-TTL-expired id 5 should remain in the lazy snapshot rows")
	}
}

// TestNamedOrderSnapshotPointTTLDroppedByWalk: a POINT TTL expiring (named only, MV
// has no point TTL) leaves the id in the snapshot rows but the walk's TTL gate drops
// it (mirrors the dense id-scroll lazy-TTL contract).
func TestNamedOrderSnapshotPointTTLDroppedByWalk(t *testing.T) {
	nc, err := NewNamedCollection("c", map[string]NamedVectorParams{
		"v": {Dim: 4, M: 16, EfConstruction: 200, EfSearch: 64, Metric: L2},
	})
	if err != nil {
		t.Fatalf("newNamedCollection: %v", err)
	}
	now := int64(1000)
	nc.now = func() int64 { return now }
	for i := uint64(1); i <= 6; i++ {
		ttl := time.Duration(0)
		if i == 5 {
			ttl = 100 * time.Millisecond
		}
		if err := nc.Insert(i, map[string][]float32{"v": {float32(i), 0, 0, 1}}, Metadata{"v": NewFloat(float64(i))}, ttl); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	order := &OrderBy{Key: "v"}
	docs, _, _ := nc.scrollPage(Filter{}, nil, order, 0, 0, false, 0)
	if !docsContainID(docs, 5) {
		t.Fatalf("id 5 should be present before point TTL: %v", docIDs(docs))
	}
	rebuildsAfterWarm := nc.orderRebuilds
	now = 2000
	docs2, _, _ := nc.scrollPage(Filter{}, nil, order, 0, 0, false, 0)
	if nc.orderRebuilds != rebuildsAfterWarm {
		t.Fatalf("point TTL aging triggered a rebuild (must be lazy): %d -> %d", rebuildsAfterWarm, nc.orderRebuilds)
	}
	if docsContainID(docs2, 5) {
		t.Fatalf("point-TTL-expired id 5 leaked into results: %v", docIDs(docs2))
	}
	snap := nc.orderSnaps[orderCacheKey{"v", false}]
	if snap == nil || !rowsContain(snap.rows, 5) {
		t.Fatalf("point-TTL-expired id 5 should remain in the lazy snapshot rows")
	}
}

// TestMVOrderSnapshotDeleteBumpsRebuilds: MV has no TTL/tombstone — a docID in
// docTokens is live; a delete bumps dataVersion ⇒ the next scroll rebuilds and the
// doc disappears.
func TestMVOrderSnapshotDeleteBumpsRebuilds(t *testing.T) {
	e := mvTestEngine(t)
	vals := map[uint64]float64{}
	for i := uint64(1); i <= 6; i++ {
		vals[i] = float64(i)
		e.insert(t, i, Metadata{"v": NewFloat(float64(i))})
	}
	order := &OrderBy{Key: "v"}
	_ = scrollAllNamedMV(e, order, 0) // warm
	before := e.orderRebuilds()
	delete(vals, 4)
	e.del(t, 4)
	got := scrollAllNamedMV(e, order, 0)
	if e.orderRebuilds() == before {
		t.Fatalf("MV delete did not bump+rebuild the order snapshot")
	}
	assertOrder(t, "mv after delete", got, bruteOrder(vals, false))
	if docsHasID(got, 4) {
		t.Fatalf("deleted MV doc 4 still present: %v", got)
	}
}

func docsHasID(ids []uint64, id uint64) bool {
	for _, x := range ids {
		if x == id {
			return true
		}
	}
	return false
}

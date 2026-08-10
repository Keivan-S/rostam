// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"testing"
	"time"
)

func TestNewHNSWValidatesConfig(t *testing.T) {
	if _, err := newHNSW(Config{Dim: 0, M: 16, EfConstruction: 200, EfSearch: 64}); err == nil {
		t.Fatal("newHNSW(invalid config) should return an error")
	}
}

func TestAssignLevelDistribution(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	const n = 100_000
	var sum int
	for i := 0; i < n; i++ {
		sum += h.assignLevel()
	}
	mean := float64(sum) / float64(n)
	// Expected: E[floor(-ln(U)*mL)] where mL = 1/ln(M)
	// = sum_{k=1}^inf e^(-k*ln(M)) = e^(-ln(M)) / (1 - e^(-ln(M))) = (1/M) / (1 - 1/M) = 1/(M-1)
	expected := 1.0 / (16.0 - 1.0)
	if mean < expected*0.8 || mean > expected*1.2 {
		t.Errorf("assignLevel mean = %v, want within 20%% of %v", mean, expected)
	}
}

func TestNewHNSWInitialState(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, M: 16, EfConstruction: 200, EfSearch: 64})
	if err != nil {
		t.Fatal(err)
	}
	if h.maxLevel != -1 {
		t.Errorf("empty index maxLevel = %d, want -1", h.maxLevel)
	}
	if h.Stats().Size != 0 {
		t.Errorf("empty index Size = %d, want 0", h.Stats().Size)
	}
}

func TestSearchLayerFindsNearest(t *testing.T) {
	h, _ := newHNSW(Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2})
	// Manually populate three vectors at level 0. We're testing search in
	// isolation, so we set up neighbor edges by hand rather than via Insert.
	insertManual := func(id uint64, v []float32) uint32 {
		slot, _ := h.arena.Insert(id, v)
		h.setNode(slot, &node{slot: slot, level: 0})
		return slot
	}
	a := insertManual(1, []float32{0, 0})
	b := insertManual(2, []float32{1, 0})
	c := insertManual(3, []float32{5, 5})
	// Fully connected at level 0 so the search can reach every node.
	h.writeNbrs(h.nodes[a], 0, []uint32{b, c})
	h.writeNbrs(h.nodes[b], 0, []uint32{a, c})
	h.writeNbrs(h.nodes[c], 0, []uint32{a, b})
	h.entryPoint = a
	h.maxLevel = 0

	// Query at (0.9, 0); ef=2 should return b and a (the two closest).
	got := h.searchLayer(&layerScratch{}, h.exactScorer([]float32{0.9, 0}), []uint32{a}, 2, 0, nil, uint64(h.now()))
	if len(got) != 2 {
		t.Fatalf("searchLayer returned %d, want 2: %+v", len(got), got)
	}
	// Sorted ascending: b (dist 0.01) then a (dist 0.81).
	if got[0].slot != b {
		t.Errorf("nearest slot = %d, want %d (b)", got[0].slot, b)
	}
	if got[1].slot != a {
		t.Errorf("second slot = %d, want %d (a)", got[1].slot, a)
	}
}

func TestInsertSingleNodeSetsEntryPoint(t *testing.T) {
	h, _ := newHNSW(Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2})
	if _, _, err := h.Insert(1, []float32{1, 2}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatal(err)
	}
	if h.maxLevel < 0 {
		t.Errorf("after first insert maxLevel = %d, want >= 0", h.maxLevel)
	}
	if slot, ok := h.arena.Slot(1); !ok {
		t.Fatal("id 1 missing from arena after Insert")
	} else if h.entryPoint != slot {
		t.Errorf("entryPoint = %d, want %d (first node's slot)", h.entryPoint, slot)
	}
}

func TestInsertDimMismatch(t *testing.T) {
	h, _ := newHNSW(Config{Dim: 3, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2})
	_, _, err := h.Insert(1, []float32{1, 2}, 0, nil, nil, nil, CASCond{})
	if err != ErrDimMismatch {
		t.Errorf("got %v, want ErrDimMismatch", err)
	}
}

func TestInsertManyAndConnectivity(t *testing.T) {
	h, _ := newHNSW(Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2})
	for i := uint64(0); i < 50; i++ {
		x := float32(i % 10)
		y := float32(i / 10)
		if _, _, err := h.Insert(i, []float32{x, y}, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	if h.Stats().Size != 50 {
		t.Errorf("size after 50 inserts = %d, want 50", h.Stats().Size)
	}
	// Every node at level 0 must have at least one neighbor (otherwise the
	// graph is disconnected and search will fail).
	for slot, nd := range h.nodes {
		if nd == nil {
			continue
		}
		if h.nbrLen(nd, 0) == 0 {
			t.Errorf("slot %d has no level-0 neighbors", slot)
		}
	}
}

func TestSelectNeighborsKeepsClosestM(t *testing.T) {
	h, _ := newHNSW(Config{Dim: 2, M: 3, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2})
	insertManual := func(id uint64, v []float32) uint32 {
		slot, _ := h.arena.Insert(id, v)
		h.setNode(slot, &node{slot: slot, level: 0})
		return slot
	}
	insertManual(1, []float32{1, 0}) // dist 1
	insertManual(2, []float32{2, 0}) // dist 4
	insertManual(3, []float32{3, 0}) // dist 9
	insertManual(4, []float32{4, 0}) // dist 16
	insertManual(5, []float32{5, 0}) // dist 25

	candidates := []slotDist{
		{slot: 0, dist: 1},
		{slot: 1, dist: 4},
		{slot: 2, dist: 9},
		{slot: 3, dist: 16},
		{slot: 4, dist: 25},
	}
	kept := h.selectNeighbors([]float32{0, 0}, candidates, 3)
	if len(kept) != 3 {
		t.Fatalf("selectNeighbors len = %d, want 3", len(kept))
	}
	// The heuristic should keep at least the absolute closest.
	if kept[0].slot != 0 {
		t.Errorf("closest neighbor slot = %d, want 0", kept[0].slot)
	}
}

func TestSearchTrivialNearest(t *testing.T) {
	h, _ := newHNSW(Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2})
	pts := []struct {
		id uint64
		v  []float32
	}{
		{1, []float32{0, 0}},
		{2, []float32{10, 10}},
		{3, []float32{0, 1}},
		{4, []float32{1, 0}},
		{5, []float32{100, 100}},
	}
	for _, p := range pts {
		if _, _, err := h.Insert(p.id, p.v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	results, err := h.Search([]float32{0.1, 0.1}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 3 {
		t.Fatalf("got %d results, want 3", len(results))
	}
	if results[0].ID != 1 {
		t.Errorf("nearest = %d, want 1", results[0].ID)
	}
	if results[1].ID != 3 && results[1].ID != 4 {
		t.Errorf("second nearest = %d, want 3 or 4", results[1].ID)
	}
}

func TestSearchEmptyIndexReturnsEmpty(t *testing.T) {
	h, _ := newHNSW(Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2})
	results, err := h.Search([]float32{0, 0}, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 0 {
		t.Errorf("empty index returned %d results, want 0", len(results))
	}
}

func TestSearchDimMismatch(t *testing.T) {
	h, _ := newHNSW(Config{Dim: 3, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2})
	if _, err := h.Search([]float32{0, 0}, 5); err != ErrDimMismatch {
		t.Errorf("got %v, want ErrDimMismatch", err)
	}
}

func TestDeleteHidesFromSearch(t *testing.T) {
	h, _ := newHNSW(Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2})
	for i := uint64(1); i <= 5; i++ {
		_, _, _ = h.Insert(i, []float32{float32(i), 0}, 0, nil, nil, nil, CASCond{})
	}
	// id 1 is at (1,0); query at (1.1, 0) — id 1 is the obvious nearest.
	results, _ := h.Search([]float32{1.1, 0}, 1)
	if len(results) != 1 || results[0].ID != 1 {
		t.Fatalf("pre-delete search returned %+v, want id 1", results)
	}
	if ok, _ := h.Delete(1, CASCond{}); !ok {
		t.Fatal("Delete(1) should return true")
	}
	if ok, _ := h.Delete(1, CASCond{}); ok {
		t.Fatal("second Delete(1) should return false")
	}
	results, _ = h.Search([]float32{1.1, 0}, 1)
	if len(results) != 1 || results[0].ID == 1 {
		t.Errorf("post-delete search returned %+v, must not contain id 1", results)
	}
}

func TestDeleteAbsentIDReturnsFalse(t *testing.T) {
	h, _ := newHNSW(Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2})
	if ok, _ := h.Delete(999, CASCond{}); ok {
		t.Error("Delete of unknown id should return false")
	}
}

func TestCloseIsIdempotent(t *testing.T) {
	h, _ := newHNSW(Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2})
	if err := h.Close(); err != nil {
		t.Errorf("first Close: %v", err)
	}
	if err := h.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
}

func TestHNSWInsertTTLFiltering(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}

	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 50*time.Millisecond, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 1: %v", err)
	}
	if _, _, err := h.Insert(2, []float32{0, 1, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 2: %v", err)
	}

	// Before expiry: both visible.
	got, err := h.Search([]float32{1, 0, 0, 0}, 10)
	if err != nil {
		t.Fatalf("Search before expiry: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("before expiry got %d results, want 2", len(got))
	}

	// Advance clock past id=1's deadline.
	fakeNow += 100

	got, err = h.Search([]float32{1, 0, 0, 0}, 10)
	if err != nil {
		t.Fatalf("Search after expiry: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("after expiry got %d results, want 1", len(got))
	}
	if got[0].ID != 2 {
		t.Fatalf("after expiry id=%d, want 2", got[0].ID)
	}
}

func TestHNSWInsertMetadata(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	meta := Metadata{
		"tenant": NewString("acme"),
		"score":  NewInt(95),
		"tags":   NewStrings([]string{"prod"}),
	}
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, meta, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert with metadata: %v", err)
	}
	if _, _, err := h.Insert(2, []float32{0, 1, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert no metadata: %v", err)
	}

	slot1, _ := h.arena.Slot(1)
	got := h.arena.Metadata(slot1)
	if got["tenant"].Str != "acme" || got["score"].Int != 95 {
		t.Errorf("Metadata(slot1) = %+v, want tenant=acme score=95", got)
	}
	if len(got["tags"].Strs) != 1 || got["tags"].Strs[0] != "prod" {
		t.Errorf("Metadata(slot1) tags = %+v, want [prod]", got["tags"].Strs)
	}

	slot2, _ := h.arena.Slot(2)
	if got := h.arena.Metadata(slot2); got != nil {
		t.Errorf("Metadata(slot2) = %+v, want nil", got)
	}
}

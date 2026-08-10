// SPDX-License-Identifier: Apache-2.0

package vector

import "testing"

func TestSparseIndexAddSearch(t *testing.T) {
	si := newSparseIndex()
	// slot 0: dims {1:1.0, 3:2.0}; slot 1: dims {3:0.5, 5:4.0}; slot 2: dims {7:1.0}
	si.add(0, SparseVector{Indices: []uint32{1, 3}, Values: []float32{1.0, 2.0}})
	si.add(1, SparseVector{Indices: []uint32{3, 5}, Values: []float32{0.5, 4.0}})
	si.add(2, SparseVector{Indices: []uint32{7}, Values: []float32{1.0}})

	// query {3:10, 5:1} → slot0: 2*10=20; slot1: 0.5*10 + 4*1 = 9; slot2: 0
	scores := si.search(SparseVector{Indices: []uint32{3, 5}, Values: []float32{10, 1}}, nil)
	if scores[0] != 20 {
		t.Errorf("slot0 score = %v, want 20", scores[0])
	}
	if scores[1] != 9 {
		t.Errorf("slot1 score = %v, want 9", scores[1])
	}
	if _, ok := scores[2]; ok {
		t.Errorf("slot2 should not appear (no shared dim), got %v", scores[2])
	}
}

func TestSparseIndexAdmitGating(t *testing.T) {
	si := newSparseIndex()
	si.add(0, SparseVector{Indices: []uint32{1}, Values: []float32{1.0}})
	si.add(1, SparseVector{Indices: []uint32{1}, Values: []float32{1.0}})

	// admit only slot 0.
	scores := si.search(SparseVector{Indices: []uint32{1}, Values: []float32{5}}, func(slot uint32) bool {
		return slot == 0
	})
	if scores[0] != 5 {
		t.Errorf("slot0 = %v, want 5", scores[0])
	}
	if _, ok := scores[1]; ok {
		t.Error("slot1 should be gated out by admit")
	}
}

func TestTopKSparse(t *testing.T) {
	scores := map[uint32]float32{10: 1.0, 20: 5.0, 30: 3.0, 40: 5.0}
	got := topKSparse(scores, 2)
	if len(got) != 2 {
		t.Fatalf("topK len = %d, want 2", len(got))
	}
	// Highest scores 5.0 (slots 20 and 40); tie breaks by lower slot → 20 then 40.
	if got[0].slot != 20 || got[0].score != 5.0 {
		t.Errorf("top[0] = %+v, want slot 20 score 5", got[0])
	}
	if got[1].slot != 40 {
		t.Errorf("top[1] = %+v, want slot 40", got[1])
	}
	// k larger than set → all, sorted.
	all := topKSparse(scores, 10)
	if len(all) != 4 {
		t.Errorf("topK(10) len = %d, want 4", len(all))
	}
	// k <= 0 or empty → nil.
	if topKSparse(scores, 0) != nil {
		t.Error("topK(0) should be nil")
	}
	if topKSparse(map[uint32]float32{}, 5) != nil {
		t.Error("topK(empty) should be nil")
	}
}

func TestSparseIndexRebuild(t *testing.T) {
	a := newArena(2, 0)
	for i := uint64(1); i <= 3; i++ {
		slot, _ := a.Insert(i, []float32{float32(i), 0})
		a.SetSparse(slot, &SparseVector{Indices: []uint32{uint32(i)}, Values: []float32{1.0}})
	}
	si := newSparseIndex()
	si.rebuild(a, map[uint32]bool{})

	// Each slot indexed under its own dim.
	for i := uint64(1); i <= 3; i++ {
		slot, _ := a.Slot(i)
		scores := si.search(SparseVector{Indices: []uint32{uint32(i)}, Values: []float32{2.0}}, nil)
		if scores[slot] != 2.0 {
			t.Errorf("after rebuild, dim %d slot %d score = %v, want 2.0", i, slot, scores[slot])
		}
	}

	// Rebuild excluding a tombstoned slot.
	slot2, _ := a.Slot(2)
	si.rebuild(a, map[uint32]bool{slot2: true})
	scores := si.search(SparseVector{Indices: []uint32{2}, Values: []float32{1.0}}, nil)
	if _, ok := scores[slot2]; ok {
		t.Error("tombstoned slot should be excluded from rebuilt index")
	}
}

func TestHNSWInsertPopulatesSparseIndex(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	sv := &SparseVector{Indices: []uint32{2, 5}, Values: []float32{1.0, 2.0}}
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, nil, sv, nil, CASCond{}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	slot, _ := h.arena.Slot(1)
	scores := h.sparseIdx.search(SparseVector{Indices: []uint32{5}, Values: []float32{3.0}}, nil)
	if scores[slot] != 6.0 { // 2.0 * 3.0
		t.Errorf("indexed slot score = %v, want 6.0", scores[slot])
	}
}

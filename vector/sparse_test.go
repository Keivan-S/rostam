// SPDX-License-Identifier: Apache-2.0

package vector

import "testing"

func TestSparseVectorValidate(t *testing.T) {
	cases := []struct {
		name string
		sv   SparseVector
		ok   bool
	}{
		{"empty", SparseVector{}, true},
		{"valid", SparseVector{Indices: []uint32{1, 5, 9}, Values: []float32{0.1, 0.2, 0.3}}, true},
		{"len mismatch", SparseVector{Indices: []uint32{1, 2}, Values: []float32{0.1}}, false},
		{"unsorted", SparseVector{Indices: []uint32{5, 1}, Values: []float32{0.1, 0.2}}, false},
		{"dup", SparseVector{Indices: []uint32{1, 1}, Values: []float32{0.1, 0.2}}, false},
		{"single", SparseVector{Indices: []uint32{7}, Values: []float32{1.0}}, true},
	}
	for _, c := range cases {
		err := c.sv.Validate()
		if c.ok && err != nil {
			t.Errorf("%s: Validate = %v, want nil", c.name, err)
		}
		if !c.ok && err == nil {
			t.Errorf("%s: Validate = nil, want error", c.name)
		}
	}
}

func TestSparseDot(t *testing.T) {
	a := SparseVector{Indices: []uint32{1, 3, 5, 7}, Values: []float32{1, 2, 3, 4}}
	b := SparseVector{Indices: []uint32{2, 3, 5, 8}, Values: []float32{10, 20, 30, 40}}
	// overlap at 3 (2*20=40) and 5 (3*30=90) → 130
	if got := sparseDot(a, b); got != 130 {
		t.Errorf("sparseDot = %v, want 130", got)
	}
	// disjoint → 0
	c := SparseVector{Indices: []uint32{100, 200}, Values: []float32{1, 1}}
	if got := sparseDot(a, c); got != 0 {
		t.Errorf("disjoint sparseDot = %v, want 0", got)
	}
	// empty → 0
	if got := sparseDot(a, SparseVector{}); got != 0 {
		t.Errorf("empty sparseDot = %v, want 0", got)
	}
	// self
	if got := sparseDot(a, a); got != 1+4+9+16 {
		t.Errorf("self sparseDot = %v, want 30", got)
	}
}

func TestSparseClone(t *testing.T) {
	if (SparseVector{}).clone() != nil {
		t.Error("clone of empty sparse should be nil")
	}
	orig := SparseVector{Indices: []uint32{1, 2}, Values: []float32{0.5, 0.6}}
	cl := orig.clone()
	if cl == nil {
		t.Fatal("clone returned nil for non-empty")
	}
	// Mutating the clone must not touch the original.
	cl.Indices[0] = 99
	cl.Values[0] = -1
	if orig.Indices[0] != 1 || orig.Values[0] != 0.5 {
		t.Error("clone aliases original storage")
	}
}

func TestArenaSparse(t *testing.T) {
	a := newArena(4, 0)
	slot, err := a.Insert(1, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if a.Sparse(slot) != nil {
		t.Errorf("default Sparse = %+v, want nil", a.Sparse(slot))
	}
	sv := &SparseVector{Indices: []uint32{3, 7}, Values: []float32{0.5, 0.9}}
	a.SetSparse(slot, sv)
	if got := a.Sparse(slot); got == nil || got.Indices[1] != 7 {
		t.Errorf("Sparse after Set = %+v", got)
	}
	// Slot reuse clears sparse.
	a.Delete(1)
	slot2, _ := a.Insert(2, []float32{5, 6, 7, 8})
	if slot2 != slot {
		t.Fatalf("expected slot reuse: %d vs %d", slot2, slot)
	}
	if a.Sparse(slot2) != nil {
		t.Errorf("reused slot Sparse = %+v, want nil", a.Sparse(slot2))
	}
}

func TestHNSWInsertSparse(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	sv := &SparseVector{Indices: []uint32{2, 5}, Values: []float32{1.0, 2.0}}
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, nil, sv, nil, CASCond{}); err != nil {
		t.Fatalf("Insert with sparse: %v", err)
	}
	if _, _, err := h.Insert(2, []float32{0, 1, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert no sparse: %v", err)
	}
	slot1, _ := h.arena.Slot(1)
	got := h.arena.Sparse(slot1)
	if got == nil || len(got.Indices) != 2 || got.Indices[1] != 5 {
		t.Errorf("Sparse(slot1) = %+v", got)
	}
	slot2, _ := h.arena.Slot(2)
	if h.arena.Sparse(slot2) != nil {
		t.Errorf("Sparse(slot2) = %+v, want nil", h.arena.Sparse(slot2))
	}
}

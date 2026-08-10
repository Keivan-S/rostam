// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"testing"
)

func TestReclaimRemovesTombstonedFromGraph(t *testing.T) {
	h, _ := newHNSW(Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2})
	for i := uint64(1); i <= 10; i++ {
		_, _, _ = h.Insert(i, []float32{float32(i), 0}, 0, nil, nil, nil, CASCond{})
	}
	ok3, _ := h.Delete(3, CASCond{})
	ok7, _ := h.Delete(7, CASCond{})
	if !ok3 || !ok7 {
		t.Fatal("setup deletes should succeed")
	}
	slot3, _ := h.arena.Slot(3)
	slot7, _ := h.arena.Slot(7)
	if h.nodeAt(slot3) == nil {
		t.Fatal("pre-reclaim: tombstoned slot 3 should still be in the graph")
	}

	removed := h.Reclaim()
	if removed != 2 {
		t.Errorf("Reclaim returned %d, want 2", removed)
	}
	if h.nodeAt(slot3) != nil {
		t.Error("post-reclaim: slot 3 still in the graph")
	}
	if h.nodeAt(slot7) != nil {
		t.Error("post-reclaim: slot 7 still in the graph")
	}
	if _, ok := h.arena.Slot(3); ok {
		t.Error("post-reclaim: id 3 still in arena")
	}
	if len(h.tombstoned) != 0 {
		t.Errorf("post-reclaim: tombstoned has %d entries, want 0", len(h.tombstoned))
	}
	// Search must still find the non-deleted neighbors.
	results, _ := h.Search([]float32{5, 0}, 3)
	for _, r := range results {
		if r.ID == 3 || r.ID == 7 {
			t.Errorf("post-reclaim: deleted id %d appeared in results", r.ID)
		}
	}
}

func TestReclaimDanglingEdgesRemoved(t *testing.T) {
	h, _ := newHNSW(Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2})
	for i := uint64(1); i <= 10; i++ {
		_, _, _ = h.Insert(i, []float32{float32(i), 0}, 0, nil, nil, nil, CASCond{})
	}
	deletedSlot, _ := h.arena.Slot(5)
	_, _ = h.Delete(5, CASCond{})
	_ = h.Reclaim()

	// No surviving node should reference deletedSlot in its neighbor list.
	for slot, nd := range h.nodes {
		if nd == nil {
			continue
		}
		for lvl := 0; lvl <= nd.level; lvl++ {
			for _, m := range h.nbrsAt(nd, lvl) {
				if m == deletedSlot {
					t.Errorf("dangling edge: slot %d level %d -> %d", slot, lvl, m)
				}
			}
		}
	}
}

func TestReclaimReassignsEntryPointIfNeeded(t *testing.T) {
	h, _ := newHNSW(Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2})
	for i := uint64(1); i <= 5; i++ {
		_, _, _ = h.Insert(i, []float32{float32(i), 0}, 0, nil, nil, nil, CASCond{})
	}
	epID := uint64(0)
	for id, slot := range h.arena.idMap {
		if slot == h.entryPoint {
			epID = id
			break
		}
	}
	if epID == 0 {
		t.Fatal("could not find id for entryPoint slot")
	}
	if ok, _ := h.Delete(epID, CASCond{}); !ok {
		t.Fatalf("Delete(%d) failed", epID)
	}
	_ = h.Reclaim()

	// Entry point must point to a live slot, or maxLevel must be -1.
	if h.maxLevel >= 0 {
		if h.nodeAt(h.entryPoint) == nil {
			t.Errorf("post-reclaim: entryPoint %d is not in the graph", h.entryPoint)
		}
	}
	// Search must still work.
	results, _ := h.Search([]float32{1, 0}, 1)
	if len(results) != 1 {
		t.Errorf("post-reclaim search returned %d results, want 1", len(results))
	}
}

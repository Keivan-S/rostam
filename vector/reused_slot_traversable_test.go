// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"testing"
)

// A REUSED SLOT MUST STAY TRAVERSABLE.
//
// An upsert is delete-then-insert, and the insert lands on the SAME slot the
// deleted point held (placeLockedAt's reclaim branch frees it and the arena's
// LIFO free list hands it straight back). Every other node's in-edges into that
// slot therefore survive the upsert — they point at a slot that is about to hold
// the same id again.
//
// Under Option B the new point is not linked when placement returns, so those
// in-edges lead to a node whose own forward edges have not been written yet. The
// question is what they find there. Leaving the PREDECESSOR's level-0 edges in
// place until the forward write replaces them makes the slot a working waypoint
// for the whole window — they point at real, near neighbours, since the
// predecessor held the same id. Clearing them at placement instead makes it a
// dead end.
//
// That distinction is not cosmetic. Measured by single-line ablation at the
// commit that introduced the clearing, on the cluster reshard test:
//
//	                        partitioned-graph events   test failures
//	clearing at placement            63 / 25 runs          7 / 25
//	clearing removed                  1 / 25 runs          0 / 25
//
// Under churn the whole upserted band passes through this window, and on a
// line-shaped corpus an id-ordered sweep of dead ends cuts the graph in two: the
// searches that build the graph stop being able to cross the band, so the halves
// link only within themselves. Points end up present in the arena, byte-correct,
// returned by scans, and unreachable by vector search.
//
// This test pins the invariant at the exact instant it matters, through the real
// insert path.
func TestReusedSlotStaysTraversableDuringLinkWindow(t *testing.T) {
	const n, dim = 400, 8
	cfg := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 64, Seed: 12}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids, vecs := siftLikeCorpus(n, dim, 21)
	insertAllHNSW(t, h, ids, vecs)

	// Upsert an id from the middle of the corpus, so it has real neighbours whose
	// in-edges point at its slot.
	victim := ids[n/2]
	slotBefore, ok := h.arena.Slot(victim)
	if !ok {
		t.Fatal("victim not present")
	}
	h.mu.RLock()
	edgesBefore := h.nbrLen(h.nodeAt(slotBefore), 0)
	h.mu.RUnlock()
	if edgesBefore == 0 {
		t.Fatal("victim has no level-0 edges to begin with — nothing to preserve")
	}

	var checked bool
	defer func() { linkGapHook = nil }()
	linkGapHook = func() {
		checked = true
		h.mu.RLock()
		defer h.mu.RUnlock()
		slot, present := h.arena.Slot(victim)
		if !present {
			t.Error("victim absent from the arena mid-insert")
			return
		}
		if slot != slotBefore {
			t.Errorf("upsert moved the point from slot %d to %d — the reclaim branch did not reuse the slot, so this test is not exercising the window it claims", slotBefore, slot)
			return
		}
		nd := h.nodeAt(slot)
		if nd == nil {
			t.Error("no node at the reused slot mid-insert")
			return
		}
		// The predecessor's edges must still be there.
		if got := h.nbrLen(nd, 0); got == 0 {
			t.Error("the reused slot has NO level-0 edges during the placement/link window — every in-edge pointing at it is now a dead end, which is what partitions the graph under churn")
		}
		// And it must be visible to traversal, or those edges cannot be used.
		if nd.unlinked.Load() {
			t.Error("the reused slot is hidden from traversal during the window — its inherited edges are unreachable, which is the same dead end by another route")
		}
		// The point of all of it: a search must still cross this slot.
		res, serr := h.Search(vecs[n/2], 5)
		if serr != nil {
			t.Errorf("search mid-window: %v", serr)
			return
		}
		if len(res) == 0 {
			t.Error("search returned nothing mid-window")
		}
	}

	// The real upsert: delete then insert, exactly as Collection.Upsert does.
	if _, err := h.Delete(victim, CASCond{}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := h.Insert(victim, vecs[n/2], 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatal(err)
	}
	linkGapHook = nil
	if !checked {
		t.Fatal("the link-gap hook never fired — the upsert did not defer a link phase, so nothing was verified")
	}

	// After the insert the node owns its edges: the predecessor's list must have
	// been replaced, not merged into (which would inflate degree with a dead
	// point's neighbours).
	h.mu.RLock()
	nd := h.nodeAt(slotBefore)
	after := h.nbrLen(nd, 0)
	stale := nd.staleLevel0
	h.mu.RUnlock()
	if stale {
		t.Error("staleLevel0 still set after linking — the flag never got cleared, so a later merge could inherit dead edges")
	}
	if after > h.maxM0() {
		t.Errorf("level-0 degree %d exceeds the cap %d after upsert — the predecessor's edges were merged in rather than replaced", after, h.maxM0())
	}

	// And the whole graph is still whole.
	if bad := unreachableLivePoints(h); len(bad) != 0 {
		t.Fatalf("%d live point(s) unreachable after the upsert", len(bad))
	}
}

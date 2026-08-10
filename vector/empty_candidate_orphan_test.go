// SPDX-License-Identifier: Apache-2.0

package vector

import "testing"

// THE EMPTY CANDIDATE SET.
//
// searchLayer admits only LIVE points, and it has no early exit while its result
// set is empty — the `nearest.len() >= ef` termination can never fire — so it
// walks the ENTIRE component reachable from its frontier before giving up. An
// empty return therefore means something exact and total: at this level, every
// node reachable from the entry point is tombstoned, TTL-expired, or still inside
// its own placement/link window.
//
// linkNode used to accept that as an answer. It wrote the empty list as the
// node's forward edges and — because back-edges are only ever added to the
// neighbours just chosen — gave it no in-edges either. The node is then in
// h.nodes, in the arena, live, returned by Get and by Scroll, with node.unlinked
// cleared so every guard built for exactly this shape (the placement
// window, the reused-slot marker) reports it healthy — and unreachable from
// the entry point forever, because nothing ever re-links a node after its link
// phase. Points present, byte-correct, and permanently absent from vector search.
//
// The blast radius is worse than one point. An edgeless node is a dead end for
// every traversal that steps onto it, so the next insert's descent collapses onto
// it and links blind; and if it draws a taller level it promotes ITSELF to entry
// point, at which stroke the whole live graph is orphaned at once. That is the
// shape the reshard boundary repro produces: 23 and 24 of 31 live points
// unreachable, the healthy mass severed from a starved pocket around a new entry
// point — while 23 of the 31 possible roots still reached all 31 nodes. The graph
// was whole. The ROOT was an edgeless node.
//
// THE TESTS BELOW USE NO CONCURRENCY, WHICH IS THE POINT. The interleaving is not
// what produces this; it only decides how often the precondition occurs. All it
// takes is an index whose entire live set is tombstoned at the moment something
// is inserted — and a lazy delete of a whole band followed by a backfill is a
// reshard's ordinary shape. The bug predates Option B and the concurrent link
// window entirely.

// TestBackfillIntoAnAllTombstonedIndexStaysReachable tombstones every point in an
// index without reclaiming it, then backfills — and asserts the new points join
// the graph rather than being written into it edgeless.
//
// The seed count is chosen so the tombstoned graph keeps maxLevel >= 1. That is
// load-bearing: a backfilled node drawing level 0 cannot then promote itself to
// entry point, so it has no way to become reachable by accident, and the failure
// is the pure one. With a one-level seed graph the first backfilled node still
// links with zero edges but becomes the entry point, and the ones after it link
// to IT and repair it — the damage is real but self-healing, which is exactly why
// this needs the taller seed to be caught reliably.
func TestBackfillIntoAnAllTombstonedIndexStaysReachable(t *testing.T) {
	cfg := Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	vecOf := func(id uint64) []float32 { return []float32{float32(id), 1, 0, 0} }

	const seeded = 40
	for id := uint64(0); id < seeded; id++ {
		if _, _, err := h.Insert(id, vecOf(id), 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("seed %d: %v", id, err)
		}
	}
	// A lazy band delete: no Reclaim, so every slot is still a graph vertex with
	// its edges intact and the entry point is still one of them.
	for id := uint64(0); id < seeded; id++ {
		if _, err := h.Delete(id, CASCond{}); err != nil {
			t.Fatalf("delete %d: %v", id, err)
		}
	}

	// The preconditions this test is about, asserted rather than assumed — if a
	// future change stops the setup reaching this state the test must fail loudly
	// instead of passing vacuously.
	if n := h.Stats().Size; n != 0 {
		t.Fatalf("live size %d, want 0 — the setup did not tombstone the whole index", n)
	}
	if h.maxLevel < 1 {
		t.Fatalf("maxLevel %d, want >= 1 — a backfilled level-0 node could promote itself to entry point and mask the failure", h.maxLevel)
	}
	if !h.tombstoned[h.entryPoint] {
		t.Fatalf("entry point slot %d is live — the setup did not leave the traversal rooted in the dead band", h.entryPoint)
	}

	// The backfill. Each point is checked THE MOMENT it is inserted: a later
	// insert that happens to pick it as a neighbour would append a back-edge and
	// hide the fact that this one linked blind.
	const backfill = 4
	for id := uint64(100); id < 100+backfill; id++ {
		if _, _, err := h.Insert(id, vecOf(id), 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("backfill %d: %v", id, err)
		}
		slot, ok := h.arena.Slot(id)
		if !ok {
			t.Fatalf("id %d: not in the arena", id)
		}
		nd := h.nodeAt(slot)
		if nd == nil {
			t.Fatalf("id %d: no graph node", id)
		}
		if h.nbrLen(nd, 0) == 0 {
			t.Errorf("id %d (slot %d): linked with ZERO level-0 edges — searchLayer found no LIVE candidate and linkNode wrote the empty set",
				id, slot)
		}
	}

	// Out-edges alone are not the property search depends on. Reachability is.
	if bad := unreachableLivePoints(h); len(bad) != 0 {
		ids := make([]uint64, 0, len(bad))
		for _, slot := range bad {
			ids = append(ids, h.arena.ID(slot))
		}
		t.Fatalf("%d of %d live point(s) unreachable from the entry point after backfilling an all-tombstoned index: slots=%v ids=%v (entryPoint=%d maxLevel=%d)",
			len(bad), backfill, bad, ids, h.entryPoint, h.maxLevel)
	}

	// The consequence stated the way a caller meets it.
	for id := uint64(100); id < 100+backfill; id++ {
		res, err := h.Search(vecOf(id), 1)
		if err != nil {
			t.Fatal(err)
		}
		if len(res) == 0 || res[0].ID != id {
			t.Errorf("id %d: searching for its own vector returned %v — the point is stored but invisible", id, res)
		}
	}
}

// TestReclaimedThenBackfilledIndexStaysReachable is the same emptying reached the
// other way: the band is RECLAIMED, so the nodes are gone and maxLevel is back to
// -1. The backfill then takes the anchor path, which has always been correct.
// It is the control — it says the hole above is the ALL-TOMBSTONED state
// specifically, not backfilling an emptied index in general.
func TestReclaimedThenBackfilledIndexStaysReachable(t *testing.T) {
	cfg := Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	vecOf := func(id uint64) []float32 { return []float32{float32(id), 1, 0, 0} }
	for id := uint64(0); id < 40; id++ {
		if _, _, err := h.Insert(id, vecOf(id), 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("seed %d: %v", id, err)
		}
	}
	for id := uint64(0); id < 40; id++ {
		if _, err := h.Delete(id, CASCond{}); err != nil {
			t.Fatalf("delete %d: %v", id, err)
		}
	}
	h.Reclaim()
	if h.maxLevel >= 0 {
		t.Fatalf("maxLevel %d after reclaiming every point, want -1 — this control is meant to take the anchor path", h.maxLevel)
	}
	for id := uint64(100); id < 104; id++ {
		if _, _, err := h.Insert(id, vecOf(id), 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("backfill %d: %v", id, err)
		}
	}
	if bad := unreachableLivePoints(h); len(bad) != 0 {
		t.Fatalf("%d live point(s) unreachable after reclaim-then-backfill: %v", len(bad), bad)
	}
}

// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"testing"
)

// ELECTING AN UNLINKED NODE AS THE ENTRY POINT.
//
// electEntryPoint scans h.nodes for the highest-level survivor. That was a total
// function of the graph while placement and linking were one critical section:
// being in h.nodes meant being linked. Option B broke the implication — between
// placement and the link phase a node is in h.nodes, carries its final level, and
// has no edges at all.
//
// Elect such a node and the graph does not merely lose a point, it loses
// EVERYTHING. The entry point now has no out-edges, so the linker that is about
// to run for that very node reads itself back from entryState, descends from
// itself, finds only itself, and writes itself as its own neighbour at every
// level. Every subsequent insert then descends from that isolated node and links
// to it, so the index slowly re-forms AROUND the new entry point while every point
// inserted before the incident stays permanently unreachable — present in the
// arena, returned by Get and by scans, invisible to vector search.
//
// That is the reshard failure's exact signature: the oracle scan passes (the
// points are there, byte-correct), and a search for one of them returns a
// different id at its own true distance, with a variable-length prefix of the
// nearest band missing.
//
// electEntryPoint is reachable from two places that the reshard workload hits
// constantly: placeLockedAt's upsert-over-tombstone reclaim calls it whenever the
// re-inserted id happens to own the entry point, and Reclaim re-elects when the
// entry point's slot is freed. A churn worker upserting the same 80-id band in a
// loop, doubled by the reshard dual-write, drives both.

// TestElectEntryPointSkipsUnlinkedNode reproduces the mechanism deterministically
// by doing, at the placement/link gap, exactly what the reclaim branch does at
// that moment: elect a new entry point.
//
// It inserts until one insert draws a level taller than the whole graph — the
// case where the scan would pick it — elects at the gap, lets the link finish,
// and then asserts the graph is still whole. Before the fix, everything inserted
// earlier becomes unreachable.
func TestElectEntryPointSkipsUnlinkedNode(t *testing.T) {
	const n, dim = 600, 8
	cfg := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 64, Seed: 3}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids, vecs := siftLikeCorpus(n, dim, 11)
	insertAllHNSW(t, h, ids, vecs)

	rng := rand.New(rand.NewSource(4242))
	v := make([]float32, dim)

	// entered: an election ran while a taller PENDING node existed — the exact
	// state the bug needs. electedPending: that election actually chose it, which
	// is the bug itself.
	var entered, electedPending bool
	var pendingSlot uint32
	var pendingLevel int
	defer func() { linkGapHook = nil }()

	for i := 0; i < 600 && !entered; i++ {
		id := uint64(n + i + 1000)
		for d := range v {
			v[d] = rng.Float32()
		}
		linkGapHook = func() {
			h.mu.Lock()
			defer h.mu.Unlock()
			slot, ok := h.arena.Slot(id)
			if !ok {
				return
			}
			nd := h.nodeAt(slot)
			// Only interesting when the pending node is taller than everything
			// else, which is exactly when the scan would pick it.
			if nd == nil || nd.level <= h.maxLevel {
				return
			}
			entered, pendingSlot, pendingLevel = true, slot, nd.level
			// EXACTLY what placeLockedAt's reclaim branch does when the id being
			// re-inserted owns the entry point.
			h.electEntryPoint()
			electedPending = h.entryPoint == slot
		}
		if _, _, err := h.Insert(id, v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}
	linkGapHook = nil

	if !entered {
		t.Fatal("no insert ever drew a new top level — the scenario under test was never entered")
	}
	t.Logf("election ran during the placement/link gap of slot %d (level %d, taller than the graph)", pendingSlot, pendingLevel)
	if electedPending {
		t.Error("the election chose a node that had not been linked yet — it has no edges, so it isolates the entry point")
	}

	// The elected node must not have ended up as its own only neighbour.
	h.mu.RLock()
	nd := h.nodeAt(pendingSlot)
	selfOnly := false
	if nd != nil {
		nbrs := h.nbrsAt(nd, 0)
		selfOnly = len(nbrs) > 0
		for _, nb := range nbrs {
			if nb != pendingSlot {
				selfOnly = false
				break
			}
		}
	}
	h.mu.RUnlock()
	if selfOnly {
		t.Error("the elected node linked only to itself — it descended from itself because it was the entry point before it had edges")
	}

	// And the graph must still be whole.
	if bad := unreachableLivePoints(h); len(bad) != 0 {
		t.Fatalf("%d of %d nodes became unreachable from the entry point (first few: %v) — an unlinked node was elected and orphaned the graph",
			len(bad), len(h.nodes), bad[:min(len(bad), 8)])
	}

	// The corpus inserted before the incident must still be findable by SEARCH,
	// not merely reachable in the abstract.
	misses := 0
	for i := 0; i < n; i += 17 {
		res, err := h.Search(vecs[i], 1)
		if err != nil {
			t.Fatal(err)
		}
		if len(res) == 0 || res[0].ID != ids[i] {
			misses++
		}
	}
	if misses != 0 {
		t.Errorf("%d seeded points are no longer their own nearest neighbour after the election", misses)
	}
}

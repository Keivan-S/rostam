// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"testing"
)

// THE ANCHOR INSERT INHERITS NOTHING.
//
// Every placement onto a reused slot deliberately keeps the previous occupant's
// level-0 edges, so the slot stays traversable until that node's own forward
// write replaces them (see placeLockedAt). Exactly one placement has no forward
// write to follow: the anchor, taken when maxLevel < 0 because the index is
// empty. It links nothing and returns.
//
// So for the anchor — and only the anchor — "kept until replaced" means KEPT
// FOREVER. The route in is ordinary: Reclaim frees slots without clearing
// level0Len, and reclaiming every node re-elects maxLevel to -1, so the next
// insert lands on the anchor path holding a free-list slot that still carries an
// unrelated point's edge list. That list then belongs permanently to the index's
// ENTRY POINT, is reported by nbrLen, and is written verbatim into any snapshot
// — a durable artifact carrying edges to points that no longer exist.
//
// Two independent guards, because either alone would do it and both are cheap:
// the anchor branch clears the slot, and Reclaim clears the slots it frees.

// TestAnchorInsertInheritsNoEdges drives the exact sequence: fill, delete
// everything, Reclaim, then insert into the now-empty index and inspect the
// anchor — in memory and after a snapshot round-trip.
func TestAnchorInsertInheritsNoEdges(t *testing.T) {
	const n, dim = 300, 8
	cfg := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 64, Seed: 31}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids, vecs := siftLikeCorpus(n, dim, 44)
	insertAllHNSW(t, h, ids, vecs)

	// Empty the index completely, then physically reclaim.
	for _, id := range ids {
		if _, err := h.Delete(id, CASCond{}); err != nil {
			t.Fatalf("delete %d: %v", id, err)
		}
	}
	if got := h.Reclaim(); got != n {
		t.Fatalf("Reclaim freed %d slots, want %d", got, n)
	}
	h.mu.RLock()
	empty := h.maxLevel
	h.mu.RUnlock()
	if empty >= 0 {
		t.Fatalf("maxLevel = %d after reclaiming every node, want -1 — the next insert would not take the anchor path and this test would prove nothing", empty)
	}

	// The next insert is the anchor. It lands on a recycled slot.
	const anchorID = uint64(1 << 40)
	if _, _, err := h.Insert(anchorID, vecs[0], 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("anchor insert: %v", err)
	}

	h.mu.RLock()
	slot, present := h.arena.Slot(anchorID)
	nd := h.nodeAt(slot)
	var deg int
	var stale bool
	if nd != nil {
		deg = h.nbrLen(nd, 0)
		stale = nd.staleLevel0
	}
	isEntry := h.entryPoint == slot
	h.mu.RUnlock()

	if !present || nd == nil {
		t.Fatal("anchor missing from the index")
	}
	if !isEntry {
		t.Fatalf("anchor at slot %d is not the entry point — the test is not inspecting what it claims", slot)
	}
	if stale {
		t.Error("anchor still flagged staleLevel0 — it never links, so nothing will ever clear it or replace the edges it names")
	}
	if deg != 0 {
		t.Errorf("anchor has %d inherited level-0 edge(s) from the slot's previous occupant — it is the entry point of a one-node index and must have none", deg)
	}

	// A snapshot must not carry them into a durable artifact either.
	var buf bytes.Buffer
	if err := h.Snapshot(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	restored, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := restored.Restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("restore: %v", err)
	}
	rslot, ok := restored.arena.Slot(anchorID)
	if !ok {
		t.Fatal("anchor missing after restore")
	}
	rnd := restored.nodeAt(rslot)
	if rnd == nil {
		t.Fatal("anchor has no node after restore")
	}
	if got := restored.nbrLen(rnd, 0); got != 0 {
		t.Errorf("restored anchor carries %d level-0 edge(s) — a dead point's neighbours were serialized into the snapshot", got)
	}
	// And the restored index is usable.
	res, err := restored.Search(vecs[0], 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].ID != anchorID {
		t.Fatalf("restored one-node index returned %+v, want the anchor", res)
	}
}

// TestReclaimClearsFreedSlotEdgeCounts pins the companion guard directly: after
// Reclaim, no freed slot may still report edges. This is what keeps a later
// free-list reuse from inheriting an unrelated point's neighbours regardless of
// which placement path picks the slot up.
func TestReclaimClearsFreedSlotEdgeCounts(t *testing.T) {
	const n, dim = 200, 8
	cfg := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 64, Seed: 7}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids, vecs := siftLikeCorpus(n, dim, 8)
	insertAllHNSW(t, h, ids, vecs)

	// Delete a subset, so the index survives and the reclaim is partial.
	freed := map[uint32]bool{}
	for i := 0; i < n; i += 3 {
		slot, ok := h.arena.Slot(ids[i])
		if !ok {
			continue
		}
		if _, err := h.Delete(ids[i], CASCond{}); err != nil {
			t.Fatal(err)
		}
		freed[slot] = true
	}
	if h.Reclaim() == 0 {
		t.Fatal("Reclaim freed nothing")
	}

	h.mu.RLock()
	defer h.mu.RUnlock()
	for slot := range freed {
		if h.nodes[slot] != nil {
			continue // slot already handed to a new point
		}
		if h.level0Len[slot] != 0 {
			t.Fatalf("freed slot %d still reports %d level-0 edges after Reclaim — the next occupant would inherit an unrelated point's neighbours",
				slot, h.level0Len[slot])
		}
	}
}

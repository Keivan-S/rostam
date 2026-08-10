// SPDX-License-Identifier: Apache-2.0

package vector

// Reclaim physically removes tombstoned slots from the graph and arena,
// pruning dangling neighbor edges. Returns the number of slots reclaimed.
//
// Safe to call at any time — takes the write lock. searchLayer already
// tolerates dangling neighbor entries (h.nodes[slot] == nil → continue),
// so we just clear the references; recall stays within 1-2% of baseline.
func (h *hnsw) Reclaim() int {
	h.mu.Lock()
	defer h.mu.Unlock()

	if len(h.tombstoned) == 0 {
		return 0
	}
	deletedSlots := make(map[uint32]bool, len(h.tombstoned))
	for slot := range h.tombstoned {
		deletedSlots[slot] = true
	}

	// 1. Remove dangling edges from surviving nodes by compacting each neighbor
	// list in place (filtering only ever shrinks it).
	for _, nd := range h.nodes {
		if nd == nil || deletedSlots[nd.slot] {
			continue
		}
		for lc := 0; lc <= nd.level; lc++ {
			cur := h.nbrsAt(nd, lc)
			w := 0
			for _, m := range cur {
				if !deletedSlots[m] {
					cur[w] = m
					w++
				}
			}
			h.setNbrLen(nd, lc, w)
		}
	}

	// 2. Remove tombstoned nodes from the graph and arena.
	idsToDelete := make([]uint64, 0, len(deletedSlots))
	for id, slot := range h.arena.idMap {
		if deletedSlots[slot] {
			idsToDelete = append(idsToDelete, id)
		}
	}
	for _, id := range idsToDelete {
		h.arena.Delete(id)
	}
	for slot := range deletedSlots {
		h.nodes[slot] = nil
		// Clear the freed slot's edge count too. Nothing reads it while the slot is
		// free (nodeAt returns nil, so no traversal reaches it), but the next
		// occupant inherits whatever is left here — ensureGraphSlot only zeroes
		// slots it GROWS into, not ones already within the slab. That inheritance
		// is deliberate and useful for the same-id upsert reclaim, which is a
		// separate branch in placeLockedAt and never comes through here; for a
		// general free-list reuse the previous occupant is an unrelated point, so
		// leaving its edges buys nothing and costs a stale list to reason about.
		h.level0Len[slot] = 0
	}
	count := len(deletedSlots)
	h.tombstoned = make(map[uint32]bool)

	// Reclaim freed tombstoned slots from idMap (membership of the underlying id
	// map changed). The LIVE id set is unchanged (these ids were already tombstoned
	// and thus already excluded from the snapshot's live set), but bump anyway so
	// any future rebuild reflects the now-smaller idMap exactly. count > 0 here (an
	// empty tombstone set returned early above). Single increment under h.mu.
	h.idSetVersion++
	h.bumpData() // id-set change also invalidates the order_by snapshot

	// Release the byte-quota accounting for reclaimed slots so the running
	// total stays in sync with what's actually live. Symmetric with the
	// h.bytesUsed += estimateInsertBytes call in Insert.
	h.bytesUsed -= estimateInsertBytes(h.cfg.Dim, h.cfg.M) * int64(count)
	if h.bytesUsed < 0 {
		h.bytesUsed = 0
	}

	// Rebuild the sparse inverted index: reclaim removed slots whose postings
	// would otherwise dangle. tombstoned is now empty, so rebuild re-adds every
	// surviving slot's sparse vector.
	if h.sparseIdx != nil {
		h.sparseIdx.rebuild(h.arena, h.tombstoned)
	}

	// Rebuild the BM25 index for the same reason: reclaim frees slots whose text
	// postings would otherwise dangle (and could be reused). tombstoned is now
	// empty, so rebuild re-derives every surviving slot's $content stats.
	if h.bm25 != nil {
		h.bm25.rebuild(h.arena, h.tombstoned, h.az)
	}

	// Rebuild the payload index for the same reason: reclaim frees slots whose
	// equality postings would otherwise dangle (and could be reused).
	if h.payloadIdx != nil {
		h.payloadIdx.rebuild(h.arena)
	}

	// 3. If entryPoint was tombstoned, elect a new one (highest-level
	// surviving node). If no nodes survive, reset maxLevel to -1.
	if deletedSlots[h.entryPoint] {
		h.entryPoint = 0
		h.maxLevel = -1
		for slot, nd := range h.nodes {
			// Same exclusion as electEntryPoint: a node still inside its
			// placement/link window has no edges, and electing it isolates the
			// entry point and orphans the graph. See node.unlinked.
			if nd == nil || nd.unlinked.Load() {
				continue
			}
			if nd.level > h.maxLevel {
				h.maxLevel = nd.level
				h.entryPoint = uint32(slot) //nolint:gosec // slot index < arena capacity (< 2^32)
			}
		}
	}

	return count
}

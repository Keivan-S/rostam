// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"time"
)

// defaultSweepInterval is the period at which a collection's TTL sweeper
// scans the arena for expired entries when Config.SweepInterval is 0.
const defaultSweepInterval = 60 * time.Second

// sweepOnce scans arena.expires for POINTS whose deadline has passed and
// tombstones them, then sweeps expired PER-KEY payload TTL entries off the
// surviving slots (physically dropping the keys + reindexing the dense
// payloadIdx, so an expired key no longer lingers as a stale posting). Must be
// called without holding h.mu — it takes the write lock for the duration of the
// scan. Returns the number of POINTS tombstoned (the per-key drops are counted
// separately in h.keysSwept; the point return value is unchanged for callers).
func (h *hnsw) sweepOnce() int {
	// Fast path: no slot carries a point deadline or a per-key deadline, so the
	// scan below would touch every id and mutate nothing. Checked WITHOUT the
	// write lock (arena.DeadlineSlots is atomic) — a collection that never uses
	// TTLs pays no lock contention on the periodic sweeper at all. A racing
	// concurrent insert that sets a fresh deadline right after this read is
	// caught by the NEXT sweep tick (the existing sweep-interval latency, not a
	// correctness gap).
	if h.arena.DeadlineSlots() == 0 {
		return 0
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	now := uint64(h.now())
	swept := 0
	keysSwept := 0
	for id, slot := range h.arena.idMap {
		// POINT-TTL pass (fires first). A point-expired or already-tombstoned slot
		// is removed wholesale; we never bother dropping its individual keys.
		exp := h.arena.expires[slot]
		if exp != 0 && exp <= now {
			if h.tombstoned[slot] {
				continue
			}
			// Clear the slot's deadlines (point + any per-key) through the setters
			// so arena.deadlinePoints/deadlineKeys drop it: it is now tombstoned and
			// never needs sweeping again.
			h.arena.SetExpires(slot, 0)
			h.arena.SetKeyExpires(slot, nil)
			h.tombstoned[slot] = true
			h.expiredCount.Add(1)
			swept++
			_ = id // id retained for symmetry with Delete; we tombstone by slot.
			continue
		}
		if h.tombstoned[slot] {
			continue
		}

		// PER-KEY pass: the point lives — physically reclaim any expired payload
		// keys so the dense payloadIdx drops their stale postings. Fast path: a
		// slot with no per-key deadlines is skipped without allocating.
		ke := h.arena.KeyExpires(slot)
		if len(ke) == 0 {
			continue
		}
		meta := h.arena.Metadata(slot)
		var expiredAny bool
		for k, dl := range ke {
			if keyExpired(dl, now) {
				if _, present := meta[k]; present {
					expiredAny = true
					break
				}
			}
		}
		if !expiredAny {
			continue // deadlines exist but none reached / none still present
		}
		// Mirror the SetPayload mutation pattern (write lock held): clone meta, drop
		// the expired keys (same keyExpired predicate as liveMeta — single source of
		// truth, so the swept slot reads identically to a lazily-dropped one), set the
		// metadata, prune the deadlines, reindex the dense payloadIdx.
		newMeta := cloneMeta(meta)
		ke = cloneKeyExpires(ke)
		for k, dl := range ke {
			if keyExpired(dl, now) {
				delete(newMeta, k)
				keysSwept++
			}
		}
		if len(newMeta) == 0 {
			newMeta = nil
		}
		ke = pruneKeyExpires(ke, newMeta)
		h.arena.SetMetadata(slot, newMeta)
		h.arena.SetKeyExpires(slot, ke)
		h.payloadIdx.reindex(slot, newMeta)
		// NOTE: deliberately NO idSetVersion bump — a key drop does not change the
		// live id set (the scroll snapshot caches ids, not payload), so bumping would
		// needlessly invalidate it. And NO WAL/snapshot: the absolute deadline is
		// already durable, so a swept key is re-derived (re-dropped / re-swept) on
		// restart, exactly like the point-TTL sweep.
	}
	if keysSwept > 0 {
		h.keysSwept.Add(uint64(keysSwept))
	}
	if swept > 0 {
		// Swept POINTS are now tombstoned → excluded from the snapshot's live set.
		// (The forward walk's admits already filters expired ids via isExpired, so
		// scroll correctness does not depend on this bump; we bump to keep the cached
		// snapshot's tombstone-exclusion consistent.) Single increment under h.mu.
		// Note this is gated on POINT sweeps only — key-only drops never bump it.
		h.idSetVersion++
		h.bumpData() // id-set change also invalidates the order_by snapshot
	}
	return swept
}

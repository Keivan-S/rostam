// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"sync"
	"unsafe"
)

// READER-NON-BLOCKING INSERT (Option B) — the locking contract.
//
// An Insert has two phases with wildly different costs. PLACEMENT (arena slot,
// version/TTL/metadata/payload-index/sparse/BM25 bookkeeping, node allocation,
// slab growth) is O(1) and takes microseconds. LINKING (an EfConstruction-wide
// searchLayer at every level, neighbor selection, forward writes and reverse
// back-edges) walks the graph and takes hundreds of microseconds. Historically
// BOTH ran under h.mu's write lock, so every query stalled behind the whole
// thing.
//
// Option B keeps placement under the write lock — which is what preserves the
// slab-growth invariant verbatim (see grow_race_test.go: growth is still
// mutually exclusive with every reader) — and moves ONLY the link phase to the
// READ lock, where it runs alongside queries. What the link phase mutates while
// holding a mere read lock is exactly three things, each with its own guard:
//
//	level0 / level0Len / node.upper   →  the per-slot link STRIPE below
//	entryPoint / maxLevel             →  globalMu (the promotion is rare)
//	the graph as a whole, vs. a
//	  serializer walking every node   →  linkMu (snapshot / persist barrier)
//
// THE GATE. Readers must pay nothing when nothing is being linked, which is the
// steady state of a read-mostly index. h.linkers counts in-flight linkers, and
// every neighbor read takes the unsynchronized fast path when it is zero. That
// is sound — not merely likely — because of WHERE the counter is incremented:
// under h.mu's WRITE lock, during placement. A reader holding the read lock
// therefore cannot have the count change under it (the increment needs
// exclusivity the reader denies), so a reader that observes zero is guaranteed
// that no linker can start before it releases. The DEcrement is unsynchronized,
// which is safe in the only direction it matters: a reader that observes a
// stale non-zero merely takes the locked path, and a reader that observes the
// fresh zero is reading a graph the linker has finished mutating (the atomic
// orders its writes before the decrement).
//
// This replaces the older `linkLocks == nil` gate, which could only express
// "a bulk build is running" because its per-slot lock array was allocated for
// the duration of BuildConcurrent and sized to that build's node count. An
// incremental linker cannot own such an array: the slot space grows as it runs,
// and sync.Mutex cannot be moved by append. Fixed stripes solve exactly that.
//
// THE SERIALIZER BARRIER, AND ITS LOCK ORDER. linkMu is not just "held during
// linking" — it is held from BEFORE placement until AFTER linking, by the insert
// body, spanning the interval in which the insert holds no lock at all. That gap
// is the whole reason it exists. A point becomes visible to a graph serializer
// the moment placement writes h.nodes[slot]; if a Snapshot or SavePersist ran in
// the gap it would record that point with an EMPTY edge list at every level, and
// the restored index would hold a point retrievable by id and permanently
// unreachable by vector search. Covering only the link phase leaves exactly that
// window open.
//
// The global order is linkMu THEN h.mu, and it is not interchangeable. The
// insert path (linkMu.R → h.mu.W → h.mu.R) and the serializers (linkMu.W →
// h.mu.R) both obey it. Taking them the other way round in either place closes a
// three-way cycle, because Go's RWMutex stops admitting readers once a writer
// waits: a Snapshot holding h.mu.R and waiting on linkMu.W, a TTL sweep or
// Reclaim — neither of which goes through Collection.opMu — queued on h.mu.W
// behind that Snapshot, and a linker holding linkMu.R that can no longer
// re-enter h.mu.R. Each waits on the next. Acquiring linkMu strictly outside
// h.mu removes the cycle by construction.
//
// WHAT A CONCURRENT READER CAN SEE MID-INSERT. Between placement and the end of
// linking, the point is fully present in the arena and in every posting list —
// so Get, Scroll, and the filter-first query path can return it — while the
// graph traversal cannot yet reach it. That asymmetry is invisible to clients,
// because Collection.opMu means Insert has not returned and no caller has been
// told the point exists. It is NOT invisible to code inside this package, and it
// is the kind of thing a future reader rediscovers as a bug: if a new internal
// path ever needs the two views to agree, it must take the barrier, not assume.

// linkStripeCount is the number of permanent per-slot link locks. Slot s is
// guarded by stripe s&linkStripeMask, so the stripe set of any node is a
// SUPERSET of "that node's own lock" — coarser, never weaker, which is all the
// concurrent-build correctness argument needs (it only requires that two
// mutations of the same node's lists cannot overlap).
//
// 4096 keeps false conflicts negligible for both workloads that link: the
// incremental path has ONE linker at a time (writers are serialized upstream by
// Collection.opMu), and BuildConcurrent has GOMAXPROCS of them, for a collision
// probability per acquisition of about GOMAXPROCS/4096 — under 1% at 32 cores.
// The array costs 256 KiB, so it is allocated LAZILY on the first link (see
// ensureLinkStripes): an index that is only ever read pays nothing, and any
// index that is written is already carrying a graph orders of magnitude larger.
const (
	linkStripeCount = 4096
	linkStripeMask  = linkStripeCount - 1
)

// paddedMutex is a sync.Mutex occupying a full cache line, so two stripes that
// happen to be acquired at once by different cores do not ping-pong the same
// line. Without the padding eight stripes share one line and the striping buys
// far less than its index arithmetic costs.
type paddedMutex struct {
	mu sync.Mutex
	_  [cacheLineSize - unsafe.Sizeof(sync.Mutex{})]byte
}

const cacheLineSize = 64

// Compile-time proof that the padding actually lands on a cache line; a future
// change to sync.Mutex's size would otherwise silently reintroduce sharing.
var _ [0]struct{} = [unsafe.Sizeof(paddedMutex{}) - cacheLineSize]struct{}{}

// ensureLinkStripes allocates the permanent stripe array on first use. Must hold
// h.mu (write) — it runs inside placement, before the linker count is raised, so
// any reader that can observe a non-zero count necessarily observes the array
// too (both writes are published by the same lock release).
func (h *hnsw) ensureLinkStripes() {
	if h.linkStripes == nil {
		h.linkStripes = make([]paddedMutex, linkStripeCount)
	}
}

// stripe returns the link lock guarding slot's neighbor lists.
func (h *hnsw) stripe(slot uint32) *sync.Mutex {
	return &h.linkStripes[slot&linkStripeMask].mu
}

// linking reports whether any linker may currently be mutating neighbor lists.
// See the gate discussion above for why a `false` here licenses an entirely
// unsynchronized read.
func (h *hnsw) linking() bool { return h.linkers.Load() != 0 }

// notYetLinked reports whether slot holds a node that is placed but not yet
// linked, which the admission gate treats exactly like a tombstone: not part of
// the graph, so not a search result and — the load-bearing half — not eligible
// to become the width-1 descent frontier. A frontier of one edgeless node makes
// the next level down find nothing, which is how a linker ends up writing a node
// with three edges instead of M. See node.unlinked.
//
// The h.linking() guard is what keeps this free. A node is flagged inside
// placement, which raises the linker count in the same critical section, and
// unflagged before that count drops (linkRead clears the flag in its body, ahead
// of the deferred decrement). So a zero count proves no node carries the flag,
// and a read-mostly index never pays the node lookup at all — which matters
// because this sits in the per-candidate admission path.
func (h *hnsw) notYetLinked(slot uint32) bool {
	if !h.linking() {
		return false
	}
	nd := h.nodeAt(slot)
	return nd != nil && nd.unlinked.Load()
}

// entryState returns the (entryPoint, maxLevel) pair every traversal starts
// from. The pair MUST be read together: a linker promoting a taller node writes
// both under globalMu, and a descent that started at the old entry point with
// the new maxLevel would walk levels the entry node does not have.
//
// The fast path is the same gate as neighborsAt and rests on the same argument:
// with no linker in flight the only writers are write-lock holders, whom this
// caller's read lock already excludes, so the plain read is safe.
func (h *hnsw) entryState() (uint32, int) {
	if !h.linking() {
		return h.entryPoint, h.maxLevel
	}
	h.globalMu.Lock()
	ep, ml := h.entryPoint, h.maxLevel
	h.globalMu.Unlock()
	return ep, ml
}

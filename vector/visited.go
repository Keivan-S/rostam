// SPDX-License-Identifier: Apache-2.0

package vector

import "sync"

// visitedSet is an allocation-free "have I seen this slot?" set for one
// searchLayer traversal. Instead of a fresh map[uint32]bool per call (the old
// hot-path allocation), it uses an epoch-stamped array: stamps[slot] holds the
// epoch at which slot was last marked, and a slot counts as visited iff its
// stamp equals the current epoch. prepare() bumps the epoch, which clears the
// whole set in O(1) without touching memory — only an epoch wraparound (every
// 2^32 traversals) pays an O(capacity) zeroing.
type visitedSet struct {
	stamps []uint32
	cur    uint32
}

// prepare sizes the set for n slots and starts a fresh epoch. Growing reuses
// the backing array when capacity allows; a new array is zero-filled (so all
// slots read as unvisited regardless of cur).
func (v *visitedSet) prepare(n int) {
	if cap(v.stamps) < n {
		v.stamps = make([]uint32, n, max(n, 2*cap(v.stamps)))
		v.cur = 1
		return
	}
	v.stamps = v.stamps[:n]
	v.cur++
	if v.cur == 0 { // epoch wrapped — reset stamps and restart at 1
		for i := range v.stamps {
			v.stamps[i] = 0
		}
		v.cur = 1
	}
}

// seen reports whether slot was already marked this epoch.
func (v *visitedSet) seen(slot uint32) bool { return v.stamps[slot] == v.cur }

// mark records slot as visited this epoch.
func (v *visitedSet) mark(slot uint32) { v.stamps[slot] = v.cur }

// layerScratch bundles every reusable buffer one searchLayer traversal needs:
// the visited-set and the two search heaps. Pooling the whole bundle means a
// single Get/Put per searchLayer recycles all three backing arrays — steady
// state allocates nothing here except the returned result slice. Each scratch
// is owned by one goroutine for the duration of a searchLayer call, so no
// synchronization is needed beyond the pool itself.
type layerScratch struct {
	visited    visitedSet
	candidates minHeap
	nearest    maxHeap
	out        []slotDist // searchLayer's result buffer (reused; aliased by its return)
	cur        []uint32   // descent frontier between searchLayer calls (reused)
	qbuf       []float32  // normalized-query buffer for the cosine metric (reused)
	qcode      []byte     // encoded query-node code for symmetric build navigation (reused)
	nbrScratch []uint32   // neighbor-list copy buffer for concurrent build (reused)

	// Batched neighbor-expansion buffers (expandBatched): the unvisited slots
	// filtered out of one neighbor list, and the block of distances the batched
	// kernel writes for them. Both are bounded by the largest neighbor list
	// (m0 = 2M at level 0), so they reach their steady-state capacity within the
	// first few traversals and never allocate again. They are distinct from
	// nbrScratch because a concurrent build holds a neighbor-list copy there
	// while this expansion runs over it.
	batchSlots []uint32
	batchDist  []float32

	preScratch []uint32 // pre-existing back-edges preserved across a concurrent-build forward write (reused)

	// Back-edge prune scratch (addBackEdge → reselect). These MUST be
	// per-goroutine rather than per-index: BuildConcurrent prunes a neighbor's
	// list under that neighbor's link lock, so several workers reselect at the
	// same time. All three are distinct from out/extPool because a prune runs
	// while the caller is still iterating the searchLayer/pickNeighbors result.
	pruneSlots []uint32   // level-0 "existing edges + the new one" private copy
	pruneCands []slotDist // (slot, dist) pairs fed to the diversity heuristic
	pruneKept  []slotDist // the heuristic's output

	// extendCandidates build-time scratch (Config.ExtendCandidates): a dedup set
	// over slots already in the pool and the reused extended-pool buffer.
	extVisited visitedSet
	extPool    []slotDist

	// filterRejects accumulates this traversal's predicate rejections, which
	// searchLayerCore flushes to the index's shared atomic once on return. The
	// counter used to be bumped atomically per rejected candidate — on a
	// selective filter that is most of the traversal, so every concurrent query
	// contended on a single cache line for a statistic none of them reads.
	filterRejects uint64

	// gate is the per-QUERY bitset admission gate (filter_bitset.go). Unlike
	// everything above it spans a whole query rather than one traversal, so
	// reset() must NOT clear it — the descent and every ef-doubling pass of the
	// bottom-level expansion share one gate, exactly as they share cur/qbuf.
	// Its lifetime is owned by searchIntoWith, the only place that arms it;
	// getLayerScratch disarms on every checkout so a gate can never be inherited
	// by the next query to borrow this scratch.
	gate admitGate

	sparseAcc sparseAccumulator // pooled score scratch for the sparse lane (reused)
}

// reset prepares the per-traversal buffers for a fresh searchLayer call: the
// visited-set, both heaps, and the result buffer. cur/qbuf are owned by the
// enclosing search (they persist across the searchLayer calls of one query)
// and are NOT reset here.
func (s *layerScratch) reset(n int) {
	s.visited.prepare(n)
	s.candidates = s.candidates[:0]
	s.nearest = s.nearest[:0]
	s.out = s.out[:0]
	s.filterRejects = 0
}

// layerScratchPool recycles layerScratch instances across searchLayer calls.
var layerScratchPool = sync.Pool{New: func() any { return &layerScratch{} }}

// getLayerScratch checks a scratch out of the pool with the admission gate
// DISARMED. Every caller goes through here rather than the pool directly: a gate
// is a compiled answer to one specific filter, and a scratch that carried one
// into an unrelated query would silently return that query's results filtered by
// the wrong predicate. The gate's own lifetime is already bracketed by a defer
// at its single arming site (searchIntoWith) — this is the second lock, so that
// the invariant holds for the paths that never heard of gates (build workers,
// BM25, discover, group, MMR, IVF, and the incremental LINKER) even if that
// defer is ever lost. The linker is the one that made this worth the second
// lock: since the insert split it runs under the READ lock, concurrently with
// queries, so it can be handed a scratch a filtered search returned moments ago.
// Disarming retains the bitset's backing array, so pooled reuse still allocates
// nothing.
func getLayerScratch() *layerScratch {
	s := layerScratchPool.Get().(*layerScratch)
	s.gate.disable()
	return s
}

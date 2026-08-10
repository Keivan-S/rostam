// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"fmt"
	"os"
	"unsafe"
)

// Graph adjacency storage.
//
// HNSW edges are dominated by level 0: every node lives there and its cap is
// 2*M (vs M at higher levels), so level 0 holds the large majority of edges and
// — critically — all of the per-node allocation. Representing it as one
// [][]uint32-per-node (a slice header + a separately-allocated, append-grown
// backing array per node) cost ~4x the edge bytes in practice: millions of tiny
// heap objects, 24-byte slice headers, and append-doubling capacity slack (see
// TestMemBreakdownSIFT).
//
// Instead, level 0 lives in a single flat slab indexed by slot with a fixed
// stride of m0 = 2*M: node `slot`'s neighbors are level0[slot*m0 : slot*m0 +
// level0Len[slot]]. One allocation total, zero per-node headers, zero slack
// beyond the (inherent) per-node cap. This mirrors hnswlib's contiguous
// linkLists block. Levels >= 1 are rare (~1/M of nodes) and stay as small
// per-node slices (node.upper), where the overhead is negligible.
//
// Concurrency: the slab is pre-sized before the parallel build (no reallocation
// during it), and each node's region (level0[slot*m0:...], level0Len[slot],
// node.upper) is mutated only under that slot's link lock — exactly the
// invariant the [][]uint32 version relied on. Distinct slots touch disjoint
// slab elements, so there is no data race between workers on different nodes.

// makeUpper allocates the per-node slice-of-slices for levels 1..level. Returns
// nil for a level-0-only node (the common case), so those nodes carry no
// graph allocation beyond their slab region.
func makeUpper(level int) [][]uint32 {
	if level <= 0 {
		return nil
	}
	return make([][]uint32, level)
}

// ensureGraphSlot grows the node table and the level-0 slab so `slot` is
// addressable. Geometric growth keeps amortized insert O(1); BuildConcurrent
// and Restore pre-size up front so this never reallocates mid-build. Returns an
// error only when the level-0 slab fails to grow. Must hold the write lock
// (search holds the read lock, so growth never races a reader).
//
// Only the level-0 slab grows in place. h.nodes ([]*node) deliberately stays on
// the Go heap and keeps its doubling copy: off-heap memory is invisible to the
// GC, so a *node stored there would be collected out from under the graph. It is
// also cheap — 8 B/slot against 3072 B/slot for the vectors at 768d.
// h.level0Len (2 B/slot) stays for the same reason: not worth a reservation.
// See reserve.go for the measured split.
func (h *hnsw) ensureGraphSlot(slot uint32) error {
	need := int(slot) + 1
	if len(h.nodes) < need {
		// One geometric grow, not `need - len` successive appends: the append loop
		// this replaces could reallocate (and copy) several times inside a single
		// call when a Reclaim-reused slot jumped the table forward.
		if cap(h.nodes) < need {
			grown := make([]*node, len(h.nodes), max(need, 2*cap(h.nodes)))
			copy(grown, h.nodes)
			h.nodes = grown
		}
		h.nodes = h.nodes[:need]
	}
	if len(h.level0Len) < need {
		if cap(h.level0Len) < need {
			grown := make([]uint16, len(h.level0Len), max(need, 2*cap(h.level0Len)))
			copy(grown, h.level0Len)
			h.level0Len = grown
		}
		h.level0Len = h.level0Len[:need]
	}
	want := need * h.m0
	if len(h.level0) >= want {
		return nil
	}
	if cap(h.level0) < want {
		if err := h.growLevel0(want); err != nil {
			return err
		}
	}
	h.level0 = h.level0[:want]
	return nil
}

// uint32sOver reinterprets a byte region as a []uint32 spanning its full length
// (len == cap == len(region)/4). The region must outlive the slice; the header
// is rebuilt whenever the mapping moves. Mirrors floatsOver.
func uint32sOver(region []byte) []uint32 {
	if len(region) == 0 {
		return nil
	}
	n := len(region) / 4
	//nolint:gosec // G103: reviewed unsafe reinterpretation of mmap bytes as uint32.
	return unsafe.Slice((*uint32)(unsafe.Pointer(&region[0])), n)
}

// useGraphMmap backs the level-0 slab with a memory-mapped file at path. Called
// once from newHNSW when Config.GraphMmapPath is set, before any insert/build.
// The initial mapping reserves mmapInitVectors slots' worth of edges; the slab
// grows geometrically on demand (growGraphMmap).
func (h *hnsw) useGraphMmap(path string) error {
	size := int64(mmapInitVectors * h.m0 * 4)
	f, region, err := openVecMmap(path, size)
	if err != nil {
		return err
	}
	h.releaseGraphBacking() // attaching a new backing must not strand an old reservation
	h.graphMmapF = f
	h.graphRegion = region
	h.level0 = uint32sOver(region)[:0]
	return nil
}

// releaseGraphBacking hands the level-0 slab's address-space reservation back to
// the OS. Same ownership rule as the arena's releaseVecsBacking: the reservation
// is off-heap, the GC cannot reclaim it, and there is no finalizer — so every
// path that ATTACHES a different backing to h.level0 has to come through here
// first, or the old range stays mapped with nothing referencing it. Both callers
// (useGraphMmap, loadGraphMmap) are documented as running on a fresh index, so
// this is belt-and-braces; it is here because the failure mode is a silent leak
// that no test of program output would ever catch.
func (h *hnsw) releaseGraphBacking() {
	if h.graphRes == nil {
		return
	}
	_ = h.graphRes.release()
	h.graphRes = nil
	h.graphRegion = nil
}

// growLevel0 ensures the level-0 slab has capacity for at least needU32 edges,
// growing geometrically. It is the single growth chokepoint for the slab across
// every backing, and mirrors arena.growVecs exactly — see the strategy ladder
// there, and reserve.go for why the reserved path leaves the base address (and
// therefore every reader's aliases) untouched. The logical length is preserved;
// the caller reslices up to its target.
func (h *hnsw) growLevel0(needU32 int) error {
	if cap(h.level0) >= needU32 {
		return nil
	}
	newCap := cap(h.level0) * 2
	if newCap < needU32 {
		newCap = needU32
	}
	oldLen := len(h.level0)
	newBytes := int64(newCap) * 4

	if h.graphRes != nil {
		err := h.graphRes.commitTo(newBytes)
		if err == nil {
			h.adoptLevel0(h.graphRes.region(), oldLen)
			return nil
		}
		if !errors.Is(err, errSlabReserveExhausted) {
			return err
		}
	}
	if newBytes >= slabReserveThreshold {
		ok, err := h.reserveLevel0(newBytes, oldLen)
		if err != nil {
			return err
		}
		if ok {
			return nil
		}
	}
	if h.graphMmapF != nil {
		region, err := growVecMmap(h.graphMmapF, h.graphRegion, newBytes)
		if err != nil {
			return err
		}
		h.graphRegion = region
		// File grew in place (bytes 0..oldLen retained, the tail zero-filled); keep
		// the logical length and let the caller reslice up to its target.
		h.level0 = uint32sOver(region)[:oldLen]
		return nil
	}
	grown := make([]uint32, oldLen, newCap)
	copy(grown, h.level0)
	h.level0 = grown
	return nil
}

// adoptLevel0 rebuilds the level0 header over region, keeping logicalLen edges.
// For a file-backed slab the region is also what persist msync's, so graphRegion
// tracks it.
func (h *hnsw) adoptLevel0(region []byte, logicalLen int) {
	if h.graphMmapF != nil {
		h.graphRegion = region
	}
	h.level0 = uint32sOver(region)[:logicalLen]
}

// reserveLevel0 moves the slab onto a fresh reservation sized for newBytes,
// carrying the first logicalLen edges across. Reports false (with no error) when
// no reservation could be made — not a failure, just the legacy path.
func (h *hnsw) reserveLevel0(newBytes int64, logicalLen int) (bool, error) {
	hint := slabHintBytes(h.cfg.MaxVectors, h.m0*4)
	res, err := newSlabReservation(h.graphMmapF, slabReserveSize(newBytes, hint), newBytes)
	if err != nil {
		return false, nil //nolint:nilerr // a reservation is best-effort; the caller has a correct slower path
	}
	old := h.graphRes
	if h.graphMmapF != nil {
		// File-backed: the edges are in the FILE, which the new reservation maps
		// from offset 0 — they are already there at the new base, nothing copied.
		if old != nil {
			_ = old.release()
		} else if uerr := unmapVecMmap(h.graphRegion); uerr != nil {
			_ = res.release()
			return false, uerr
		}
		h.graphRes = res
		h.adoptLevel0(res.region(), logicalLen)
		return true, nil
	}
	dst := uint32sOver(res.region())
	copy(dst[:logicalLen], h.level0[:logicalLen])
	h.graphRes = res
	h.level0 = dst[:logicalLen]
	if old != nil {
		_ = old.release()
	}
	return true, nil
}

// loadGraphMmap attaches an EXISTING level-0 graph mmap file (written by a prior
// run), mapping it at its current on-disk size without truncating, and slices
// the slab to n slots. The level-0 adjacency comes back zero-copy; the caller
// restores level0Len separately. Used by instant-restart (persist.go).
func (h *hnsw) loadGraphMmap(path string, n int) error {
	fi, err := os.Stat(path)
	if err != nil {
		return err
	}
	if fi.Size() < int64(n*h.m0*4) {
		return fmt.Errorf("vector: graph file %s too small (%d bytes) for %d slots x m0=%d", path, fi.Size(), n, h.m0)
	}
	f, region, err := openVecMmap(path, fi.Size())
	if err != nil {
		return err
	}
	h.releaseGraphBacking() // attaching a new backing must not strand an old reservation
	h.graphMmapF = f
	h.graphRegion = region
	h.level0 = uint32sOver(region)[:n*h.m0]
	return nil
}

// closeGraphMmap releases the level-0 backing: the address-space reservation
// and/or the mmap. Idempotent; a no-op for a heap-backed graph that never grew
// past the reservation threshold.
func (h *hnsw) closeGraphMmap() error {
	if h.graphRes == nil && h.graphMmapF == nil {
		return nil
	}
	var err error
	if h.graphRes != nil {
		if serr := h.graphRes.sync(); serr != nil {
			err = serr
		}
		if rerr := h.graphRes.release(); rerr != nil && err == nil {
			err = rerr
		}
		// Cleared inline rather than via releaseGraphBacking because Close, unlike
		// an attach, has to surface the sync/release errors.
		h.graphRes = nil
		if h.graphMmapF != nil {
			if cerr := h.graphMmapF.Close(); cerr != nil && err == nil {
				err = fmt.Errorf("vector: close: %w", cerr)
			}
		}
	} else {
		err = closeVecMmap(h.graphMmapF, h.graphRegion)
	}
	h.graphMmapF = nil
	h.graphRegion = nil
	h.level0 = nil
	return err
}

// presizeGraphSlab sizes the level-0 slab to exactly n slots (n*m0 edges) up
// front — used by BuildConcurrent and Restore so the parallel/restore phase only
// writes into a stable region. Routes through growLevel0, so a large presize
// lands directly on a reservation and the index starts life on the in-place
// growth path rather than switching onto it later.
//
// Contents of the presized range are NOT cleared: level0Len is freshly zeroed
// just above, and nbrsAt bounds every read by it, so whatever bytes the slab
// already held are unobservable until the caller writes the node's real edges.
// That was already true of the mmap backing (a grown file keeps its bytes); the
// heap backing now shares the property instead of paying a full-slab memclr.
func (h *hnsw) presizeGraphSlab(n int) error {
	h.level0Len = make([]uint16, n)
	want := n * h.m0
	if err := h.growLevel0(want); err != nil {
		return err
	}
	h.level0 = h.level0[:want]
	return nil
}

// nbrsAt returns node nd's level-lc neighbor slots. For level 0 the returned
// slice aliases the slab (valid until the next write to this slot); for upper
// levels it is the node's own slice. Caller must ensure lc <= nd.level.
func (h *hnsw) nbrsAt(nd *node, lc int) []uint32 {
	if lc == 0 {
		base := int(nd.slot) * h.m0
		return h.level0[base : base+int(h.level0Len[nd.slot])]
	}
	return nd.upper[lc-1]
}

// nbrLen returns the number of level-lc neighbors of nd.
func (h *hnsw) nbrLen(nd *node, lc int) int {
	if lc == 0 {
		return int(h.level0Len[nd.slot])
	}
	return len(nd.upper[lc-1])
}

// writeNbrs overwrites nd's level-lc neighbor list with slots. len(slots) must
// be <= the level cap (m0 at level 0, M above); callers satisfy this because
// the list comes from selectNeighbors / a prune, both capped.
func (h *hnsw) writeNbrs(nd *node, lc int, slots []uint32) {
	if lc == 0 {
		base := int(nd.slot) * h.m0
		n := copy(h.level0[base:base+h.m0], slots)
		h.level0Len[nd.slot] = uint16(n) //nolint:gosec // n <= m0 <= 256
		return
	}
	nd.upper[lc-1] = append(nd.upper[lc-1][:0], slots...)
}

// writeNbrsFromDist is writeNbrs taking slotDist (the form selectNeighbors and
// searchLayer produce), avoiding an intermediate []uint32 in the hot build path.
func (h *hnsw) writeNbrsFromDist(nd *node, lc int, sd []slotDist) {
	if lc == 0 {
		base := int(nd.slot) * h.m0
		for i, s := range sd {
			h.level0[base+i] = s.slot
		}
		h.level0Len[nd.slot] = uint16(len(sd)) //nolint:gosec // len(sd) <= M <= m0
		return
	}
	row := nd.upper[lc-1][:0]
	for _, s := range sd {
		row = append(row, s.slot)
	}
	nd.upper[lc-1] = row
}

// setNbrLen truncates nd's level-lc list to n entries (used by in-place edge
// compaction in Reclaim, which only ever shrinks a list).
func (h *hnsw) setNbrLen(nd *node, lc, n int) {
	if lc == 0 {
		h.level0Len[nd.slot] = uint16(n) //nolint:gosec // n <= m0
		return
	}
	nd.upper[lc-1] = nd.upper[lc-1][:n]
}

// addBackEdge appends `from` to nd's level-lc list. If that would exceed maxM,
// it re-selects maxM neighbors from the existing list plus `from` via the
// heuristic — identical to the old "append then prune if > maxM", so the
// resulting edge set is unchanged. Must hold the write lock (serial Insert) or
// nd's link lock (concurrent build). s carries the caller's per-goroutine prune
// buffers; it is never shared between workers, so the buffers are safe even
// though several nodes are pruned concurrently under their own link locks.
func (h *hnsw) addBackEdge(s *layerScratch, nd *node, lc int, from uint32, maxM int) {
	if lc == 0 {
		base := int(nd.slot) * h.m0
		ln := int(h.level0Len[nd.slot])
		if ln < maxM {
			h.level0[base+ln] = from
			h.level0Len[nd.slot] = uint16(ln + 1) //nolint:gosec // ln < maxM <= m0
			return
		}
		// Full: re-select among the maxM existing edges + the new one. The copy is
		// mandatory (reselect's writeNbrs overwrites the slab region it aliases);
		// only the ALLOCATION is pooled.
		tmp := s.pruneSlots[:0]
		if cap(tmp) < ln+1 {
			tmp = make([]uint32, 0, ln+1)
		}
		tmp = append(tmp, h.level0[base:base+ln]...)
		tmp = append(tmp, from)
		s.pruneSlots = tmp
		h.reselect(s, nd, 0, tmp, maxM)
		return
	}
	nd.upper[lc-1] = append(nd.upper[lc-1], from)
	if len(nd.upper[lc-1]) > maxM {
		h.pruneNeighborsTo(s, nd, lc, maxM)
	}
}

// pruneNeighborsTo reduces nd's level-lc list back to `keep` entries via the
// heuristic. Used for the M_max0 = 2*M cap at level 0 and M_max = M above. Must
// hold the write lock (serial) or nd's link lock (concurrent build). Safe for
// upper levels (the list is the node's own slice); for level 0 it goes through
// reselect with the slab alias, which is fine because reselect copies the slot
// ids into its candidate buffer before writeNbrs overwrites the slab.
func (h *hnsw) pruneNeighborsTo(s *layerScratch, nd *node, lc int, keep int) {
	h.reselect(s, nd, lc, h.nbrsAt(nd, lc), keep)
}

// reselect recomputes nd's level-lc list as the heuristic's pick of `keep`
// neighbors from `slots`, then writes it back. `slots` must be a private copy
// for level 0 (writeNbrs overwrites the slab region it would otherwise alias).
// The candidate/kept buffers come from s (per-goroutine) so a steady-state
// prune allocates nothing.
func (h *hnsw) reselect(s *layerScratch, nd *node, lc int, slots []uint32, keep int) {
	// Node→neighbor distances in the same space selectNeighbors compares against
	// (code-space under QuantizedBuild, else exact float32).
	pf := h.buildPairFn()
	cands := s.pruneCands[:0]
	if cap(cands) < len(slots) {
		cands = make([]slotDist, 0, len(slots))
	}
	for _, sl := range slots {
		cands = append(cands, slotDist{slot: sl, dist: pf(nd.slot, sl)})
	}
	s.pruneCands = cands
	insertionSortByDist(cands)
	// Pre-size the kept buffer so selectNeighborsInto never reallocates; it then
	// returns either s.pruneKept or cands, both live until writeNbrsFromDist.
	if cap(s.pruneKept) < keep {
		s.pruneKept = make([]slotDist, 0, keep)
	}
	kept := h.selectNeighborsInto(s.pruneKept[:0], h.arena.Vec(nd.slot), cands, keep)
	h.writeNbrsFromDist(nd, lc, kept)
}

// insertionSortByDist sorts ascending by distance — selectNeighbors requires
// its input in that order. Insertion sort matches the original prune path and
// is fast for the small (<= 2*M+1) lists involved.
func insertionSortByDist(c []slotDist) {
	for i := 1; i < len(c); i++ {
		for j := i; j > 0 && c[j-1].dist > c[j].dist; j-- {
			c[j-1], c[j] = c[j], c[j-1]
		}
	}
}

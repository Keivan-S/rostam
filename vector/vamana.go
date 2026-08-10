// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// effectiveVamanaR resolves the Vamana out-degree R from cfg, applying the
// default (defaultVamanaR) for the zero value. This is the SINGLE source of truth
// for the level-0 slab stride of a Vamana index — newVamana sets h.m0 from it at
// construction, and the persist/snapshot reconstruction paths derive m0 from it on
// reopen/restore so build and restore can never drift.
func effectiveVamanaR(cfg Config) int {
	r := cfg.VamanaR
	if r == 0 {
		r = defaultVamanaR
	}
	return r
}

// effectiveM0 is the level-0 adjacency slab stride for cfg's index type: Vamana ⇒
// VamanaR (the out-degree cap), every other index type (HNSW) ⇒ 2*M. It mirrors the
// stride set at construction (newVamana: h.m0 = R; newHNSW: m0 = 2*M) so the
// persist/snapshot reconstruction paths (newPersistShell, snapshot restore) size the
// restored slab IDENTICALLY to how it was built. A reopen with the wrong stride
// corrupts every node's neighbor list — this helper is the linchpin that prevents it.
func effectiveM0(cfg Config) int {
	if cfg.IndexType == IndexVamana {
		return effectiveVamanaR(cfg)
	}
	return 2 * cfg.M
}

// newVamana constructs an IndexVamana index: the HNSW engine constrained to a
// SINGLE LAYER (every node is level 0, upper nil) with an R-bounded out-degree
// (the level-0 slab stride m0 = VamanaR) and an α-RobustPrune neighbor rule
// (pruneAlpha = VamanaAlpha). It reuses the entire hnsw VectorIndex surface — the
// genuinely-new code is the level pinning here, the α on selectNeighbors, the
// two-pass buildVamana, and the medoid entry point.
//
// Single-layer pinning: mL is forced to 0 so assignLevel always returns
// floor(-ln(r)*0) = 0 — every node lands at level 0, upper is always nil, and the
// search/insert level-descent loops (lc := maxLevel; lc > 0/level) are no-ops. The
// Vamana defaults (R=64, L=100, alpha=1.2) are applied for zero fields; L drives
// both the build beam (EfConstruction) and the search ef floor (EfSearch) when the
// caller did not set them.
// applyVamanaDefaults resolves the Vamana geometry defaults into cfg: R = out-degree
// (slab stride), L = beam width, alpha = pass-2 prune α, with EfConstruction/EfSearch
// defaulting to L and M defaulting to 16 (unused for the single-layer geometry — m0
// comes from R — but it must pass Validate, M ∈ [1,128]). This is the SINGLE place the
// defaults are applied, shared by construction (newVamana) and the persistent reopen
// (openPersist), so a reopened Vamana validates and header-matches a zero-field user
// config exactly as construction did (e.g. the persisted M=16 header matches a cfg with
// M unset). Returns the normalized config; callers pass it to newHNSW / cfg.Validate.
func applyVamanaDefaults(cfg Config) Config {
	cfg.VamanaR = effectiveVamanaR(cfg)
	if cfg.VamanaL == 0 {
		cfg.VamanaL = defaultVamanaL
	}
	if cfg.VamanaAlpha == 0 {
		cfg.VamanaAlpha = defaultVamanaAlpha
	}
	if cfg.M == 0 {
		cfg.M = 16
	}
	if cfg.EfConstruction == 0 {
		cfg.EfConstruction = cfg.VamanaL
	}
	if cfg.EfSearch == 0 {
		cfg.EfSearch = cfg.VamanaL
	}
	return cfg
}

func newVamana(cfg Config) (*hnsw, error) {
	// Resolve Vamana geometry defaults BEFORE Validate (which is also called inside
	// newHNSW). Shared with the persistent reopen so build and restore normalize a
	// zero-field user config identically (M, R, L, alpha, ef knobs).
	cfg = applyVamanaDefaults(cfg)
	r := cfg.VamanaR
	alpha := cfg.VamanaAlpha

	h, err := newHNSW(cfg)
	if err != nil {
		return nil, err
	}
	// Pin to a single layer: mL = 0 ⇒ assignLevel returns 0 for every node. Set the
	// level-0 slab stride from R (the Vamana out-degree cap) instead of 2*M; m0 is
	// the stride AND, via maxM0(), the reverse-edge/re-prune cap. If a GraphMmap slab
	// was presized in newHNSW with the old m0 it is empty at this point (no nodes
	// yet), so resetting m0 here is safe — buildVamana/Insert presize against the new
	// stride.
	h.mL = 0
	h.m0 = r
	h.pruneAlpha = alpha
	h.vamana = true
	return h, nil
}

// openVamana reopens a Vamana index. v1 Vamana (in-RAM) is snapshot-only
// like a non-persistent IVF/HNSW: it returns a fresh empty index the caller then
// Restores. The Persistent/instant-restart reopen is not yet implemented. metaPath is
// accepted for openIndex signature symmetry and ignored.
func openVamana(cfg Config, _ string) (*hnsw, error) {
	return newVamana(cfg)
}

// medoidSlot returns the live slot whose vector is nearest to the sample mean of
// all LIVE placed vectors — the Vamana graph's entry point. The mean is computed
// over the stored (cosine-normalized when Metric==Cosine) vectors, then a single
// linear scan finds the closest live point under the configured metric. O(N·dim);
// called once per build AND on entry-point re-election after a deletion. Tombstoned
// and TTL-expired slots are skipped from both the mean and the scan so the elected
// medoid is over the genuinely-live set (at build time there are no tombstones, so
// the build path is unaffected). Must hold h.mu (callers hold the write lock).
// Returns false when no live points remain.
func (h *hnsw) medoidSlot() (uint32, bool) {
	n := h.arena.Size()
	if n == 0 {
		return 0, false
	}
	mean := make([]float32, h.cfg.Dim)
	count := 0
	for _, nd := range h.nodes {
		// nd.unlinked: a node still inside its placement/link window is not an
		// entry-point candidate (see node.unlinked), and medoidSlot's only caller
		// is electEntryPoint's Vamana branch. Excluding it from the MEAN too keeps
		// the medoid a pure function of the linked graph rather than of insert
		// timing.
		if nd == nil || nd.unlinked.Load() || h.tombstoned[nd.slot] || h.isExpired(nd.slot) {
			continue
		}
		v := h.arena.Vec(nd.slot)
		for i, x := range v {
			mean[i] += x
		}
		count++
	}
	if count == 0 {
		return 0, false
	}
	inv := float32(1) / float32(count)
	for i := range mean {
		mean[i] *= inv
	}
	// NOTE: for Cosine the mean of unit vectors is generally NOT unit-length, but do
	// not "fix" this by normalizing — only the argmin over this FIXED mean selects the
	// medoid, and uniformly scaling the mean does not change which point is nearest.
	dist := h.metricDist()
	var best uint32
	bestD := float32(math.MaxFloat32)
	found := false
	for _, nd := range h.nodes {
		if nd == nil || nd.unlinked.Load() || h.tombstoned[nd.slot] || h.isExpired(nd.slot) {
			continue
		}
		d := dist(mean, h.arena.Vec(nd.slot))
		if !found || d < bestD {
			found = true
			bestD = d
			best = nd.slot
		}
	}
	return best, found
}

// buildVamana bulk-loads an EMPTY Vamana index from ids/vecs with the canonical
// two-pass α-RobustPrune algorithm, reusing the concurrent build machinery:
//
//	pass 1: α = 1            (build a connected single-layer graph)
//	pass 2: α = VamanaAlpha  (re-link to add the long-range diversity edges)
//
// Each pass, in a seeded randomized order, links every point p: greedy-search from
// the MEDOID collecting the visited set V (beam = EfConstruction = L),
// RobustPrune(p, V, α, R) → Nout(p) (h.pickNeighbors → selectNeighbors at
// h.pruneAlpha, capped at forwardM=R), write p's forward edges, and add reverse
// edges p→j with an R-cap re-prune (addBackEdge → maxM0 = R). This is exactly
// linkOneNode at level 0; buildVamana drives it across two passes with the medoid
// as the fixed entry point and flips h.pruneAlpha between passes.
//
// Constraints mirror BuildConcurrentMeta: the index must be EMPTY ON ENTRY (no
// vectors, no graph, nothing in the payload index) and entries carry no TTL and no
// sparse vector. metas is optional and, when present, is applied by the placement
// loop below exactly as the HNSW build applies it. After it returns the index is
// an ordinary serial Vamana index (Search/Insert/Delete/Snapshot all work).
func (h *hnsw) buildVamana(ids []uint64, vecs [][]float32, metas []Metadata, workers int) error {
	start := time.Now()
	defer func() { h.insertLat.observe(time.Since(start)) }()

	if len(ids) != len(vecs) {
		return ErrBuildLenMismatch
	}
	for _, v := range vecs {
		if len(v) != h.cfg.Dim {
			return ErrDimMismatch
		}
	}
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	// payloadIdx.isEmpty() is part of "empty" here for the same reason it is in
	// BuildConcurrentMeta, and for a reason this build did not previously have: the
	// placement loop below now WRITES payload postings, so starting from a
	// pre-populated index would mix this build's slot space with a previous
	// occupant's postings. Added rather than assumed — the HNSW build has always
	// checked it, and the two bulk builders must not disagree about what empty means.
	if h.arena.Size() != 0 || h.maxLevel >= 0 || !h.payloadIdx.isEmpty() {
		return ErrBuildNonEmpty
	}
	n := len(ids)
	if n == 0 {
		return nil
	}
	// Quota, on the same terms as the HNSW bulk build — the two builders must not
	// disagree about what a collection is allowed to hold any more than they
	// disagree about what empty means.
	if err := bulkQuotaErr(h.cfg, n); err != nil {
		h.quotaRejects.Add(1)
		return err
	}
	if err := h.arena.Reserve(n); err != nil {
		return err
	}

	// ---- single-threaded setup: place every vector + node (all level 0) ----
	h.nodes = make([]*node, n)
	if err := h.presizeGraphSlab(n); err != nil {
		return err
	}
	// Quantizer training samples, collected during placement and trained BELOW (after
	// placement, before the link passes) — exactly as BuildConcurrent does for HNSW.
	// Without this a QuantPQ/SQ/PRQ Vamana navigates on UNTRAINED (zero) codes that
	// degrade to exact-float, and the Persistent sidecar would write an untrained
	// codec; training here makes codes-in-RAM ADC navigation real for the disk-native
	// path. nil for every non-quantized / fixed-scale build (no-op).
	var pqSample, sqSample, prqSample [][]float32
	if _, isPQ := h.quant.(*pqQuantizer); isPQ {
		pqSample = make([][]float32, n)
	}
	if _, isSQ := h.quant.(*trainedSQ); isSQ {
		sqSample = make([][]float32, n)
	}
	if _, isPRQ := h.quant.(*prqQuantizer); isPRQ {
		prqSample = make([][]float32, n)
	}
	for i := range ids {
		slot := uint32(i) //nolint:gosec // i < n, bounded by 2^32 vectors per arena
		v := vecs[i]
		if h.cfg.Metric == Cosine {
			v = append([]float32(nil), v...)
			normalize(v)
		}
		h.arena.PutAt(slot, ids[i], v)
		h.arena.idMap[ids[i]] = slot
		if len(metas) != 0 { // see BuildConcurrentMeta: length, not nilness
			h.applyBulkMeta(slot, metas[i])
		}
		if pqSample != nil {
			pqSample[i] = v
		}
		if sqSample != nil {
			sqSample[i] = v
		}
		if prqSample != nil {
			prqSample[i] = v
		}
		// Single-layer: assignLevel returns 0 (mL == 0), so every node is level 0,
		// upper nil. Call it anyway to advance the RNG identically to the HNSW build
		// (keeps the seeded build deterministic and the level invariant explicit).
		_ = h.assignLevel()
		h.nodes[slot] = &node{slot: slot, level: 0, upper: nil}
	}
	// Train the quantizer codebooks/ranges on the placed vectors and encode every
	// slot, BEFORE the link passes (which navigate on exact floats — quantizedBuild()
	// is false here). After this, search navigates on real ADC/SQ codes + exact
	// rescore. Mirrors BuildConcurrent; no-op when the sample is nil.
	if pqSample != nil {
		if err := h.trainAndEncodePQ(pqSample, workers); err != nil {
			return err
		}
	}
	if sqSample != nil {
		if err := h.trainAndEncodeSQ(sqSample); err != nil {
			return err
		}
	}
	if prqSample != nil {
		if err := h.trainAndEncodePRQ(prqSample, workers); err != nil {
			return err
		}
	}
	h.bytesUsed += estimateInsertBytes(h.cfg.Dim, h.cfg.M) * int64(n)
	h.insertOps.Add(uint64(n)) //nolint:gosec // n >= 0
	h.idSetVersion++
	h.bumpData()

	// Single-layer index: maxLevel is 0 once any node exists. Seed the entry point
	// to the medoid (nearest live point to the sample mean) — the fixed start for
	// every greedy search in both passes and at query time.
	h.maxLevel = 0
	medoid, ok := h.medoidSlot()
	if !ok {
		medoid = 0
	}
	h.entryPoint = medoid

	// Deterministic per-pass insertion order from the configured seed (independent of
	// the build RNG so the order is stable regardless of assignLevel draws).
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	seed := h.cfg.Seed
	if seed == 0 {
		seed = 1 // buildVamana needs a stable order even when Seed is unset
	}

	// ---- two link passes: α=1 then α=VamanaAlpha ----
	h.ensureLinkStripes()
	h.linkers.Add(1)
	defer h.linkers.Add(-1) // gate OFF: revert to unsynchronized reads

	alphas := [2]float32{1.0, h.cfg.VamanaAlpha}
	for pass, a := range alphas {
		h.pruneAlpha = a
		// Shuffle the order per pass (seeded, pass-indexed) so the two passes visit in
		// different randomized orders — the standard Vamana build.
		rng := rand.New(rand.NewSource(seed + int64(pass)))
		rng.Shuffle(n, func(i, j int) { order[i], order[j] = order[j], order[i] })

		var idx atomic.Int64
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				s := getLayerScratch()
				defer layerScratchPool.Put(s)
				for {
					oi := int(idx.Add(1)) - 1
					if oi >= n {
						return
					}
					slot := uint32(order[oi]) //nolint:gosec // order[oi] < n < 2^32
					h.linkOneNode(s, slot, 0)
				}
			}()
		}
		wg.Wait()
	}

	// Restore the steady-state prune α (pass-2 VamanaAlpha) for any post-build
	// incremental Insert, which links at the configured α on the single layer.
	h.pruneAlpha = h.cfg.VamanaAlpha
	return nil
}

// SPDX-License-Identifier: Apache-2.0
//go:build cuda

package vector

// gpuIndex — the -tags cuda GPU-accelerated dense index. It is an EXACT
// (brute-force) KNN: every query is scored against every live vector on the GPU
// and an exact top-k is selected, so a GPU result is identical (within float
// tolerance) to a CPU exact brute force under the same metric. There is NO
// recall loss — the GPU buys throughput, not approximation.
//
// Architecture (embed-hnsw + resident-buffer):
//
//   - gpuIndex EMBEDS *hnsw, so it inherits the ENTIRE VectorIndex surface for
//     free: storage (the arena), insert/delete/upsert, payload/TTL/metadata,
//     snapshot/restore, persistence, stats, hybrid/sparse, scroll, etc. The
//     inner hnsw still builds its graph on insert (the arena is populated as a
//     side effect); the graph is simply UNUSED for the overridden dense search.
//     (v1 tradeoff: we pay the graph-build cost without using it for dense KNN.
//     A follow-up could skip graph construction for a pure-GPU collection.)
//
//   - We OVERRIDE only the dense-KNN entry points (Search / SearchFiltered /
//     SearchInto / SearchFilteredWith). Everything else dispatches to *hnsw.
//
//   - GPU-EXACT SURFACE (divergence note): ONLY those four core dense-KNN entry
//     points are GPU-exact. HybridSearch, SearchMMR, SearchGroups, DiscoverVecs
//     and RecommendVecs are INHERITED unchanged from the embedded *hnsw and serve
//     APPROXIMATE results from the inner HNSW graph (not GPU-exact). They are not
//     re-routed through the GPU kernel in v1; if you need GPU-exact behavior, use
//     the four core entry points. (Re-routing them is a deliberate follow-up, not
//     part of this index's contract today.)
//
//   - EXACTNESS ACROSS THE KERNEL LIMIT: a single kernel dispatch emits at most
//     cuda.MaxK (256) candidates per query. The GPU FAST PATH covers k <= 256 on
//     a non-selective query. When the admitted top-k cannot be drawn from the GPU
//     top-256 (a highly selective filter, > MaxK tombstones, or k > MaxK) the host
//     transparently FALLS BACK to a CPU-exact brute force over the live arena, so
//     the returned top-k is always exact — never silently truncated to 256.
//
//   - RESIDENT GPU BUFFER: the corpus float32 matrix is kept resident on the
//     device (vector/cuda.Handle). We track the inner hnsw's idSetVersion
//     (bumped on every id-set mutation: insert/delete/sweep/reclaim/build/
//     restore) as a generation stamp; on a search after a change, we lazily
//     re-upload the full slot range [0, Capacity()) once (slot index == GPU row
//     index — an identity mapping) and reuse it across queries until the next
//     mutation. Tombstoned / expired / filtered slots are NOT excluded from the
//     upload; instead the SAME admission gate the CPU path uses (h.admits) is
//     applied host-side to the GPU candidates. Because brute force is exact (all
//     rows scanned), filtering after selection over the full corpus yields the
//     exact admitted top-k. (Rebuild-on-dirty is the v1 strategy; an incremental
//     device-side update is a follow-up.)
//
// The cgo binding lives in the sibling package vector/cuda (NOT here): the
// vector package has hand-written Go assembly (distance_amd64.s, ...), and the
// toolchain forbids mixing cgo with Go-syntax assembly in one package. gpuIndex
// therefore calls a Go-typed cuda.Handle.
//
// Concurrency: the resident-buffer (re)upload mutates gpu state, so the search
// path takes h.mu as a WRITE lock (not RLock) around the upload+search. This is
// coarser than the CPU read-parallel search but correct for v1; a finer design
// would double-buffer the device corpus under RLock.

import (
	"errors"
	"sort"

	"github.com/rostamlabs/rostam/vector/cuda"
)

// gpuState holds the device handle and the resident-buffer generation tracking.
// It is guarded by the embedded hnsw's mu.
type gpuState struct {
	dev *cuda.Handle
	dim int

	// uploadedVer is the inner hnsw idSetVersion the resident corpus was last
	// (re)built at; uploadedRows is the slot count it covers. A search re-uploads
	// when the live idSetVersion differs (a mutation happened) — the lazy
	// rebuild-on-dirty signal.
	uploadedVer  uint64
	uploadedRows int
	built        bool // whether dev holds a valid uploaded corpus at all
}

// gpuIndex is an hnsw whose dense search is served by the GPU exact-KNN kernel.
type gpuIndex struct {
	*hnsw
	gpu *gpuState
}

// Compile-time assertion that *gpuIndex satisfies VectorIndex (it inherits all
// of hnsw's methods and overrides the dense-search ones).
var _ VectorIndex = (*gpuIndex)(nil)

// metricCode maps vector.Metric to the cuda package's metric code (identical
// numeric values, but make the dependency explicit).
func metricCode(m Metric) int {
	switch m {
	case Cosine:
		return cuda.MetricCosine
	case L2:
		return cuda.MetricL2
	case DotProduct:
		return cuda.MetricDotProduct
	default:
		return cuda.MetricL2
	}
}

// newGPUIndex builds a GPU index: an inner hnsw (so inserts populate the arena
// and the full VectorIndex surface works) plus a device handle sized to the
// configured dimension. Fails loud if the device handle cannot be created (no
// GPU / OOM), mirroring the fail-loud philosophy of the !cuda stub.
func newGPUIndex(cfg Config) (VectorIndex, error) {
	h, err := newHNSW(cfg)
	if err != nil {
		return nil, err
	}
	dev, err := cuda.New(cfg.Dim)
	if err != nil {
		_ = h.Close()
		return nil, err
	}
	return &gpuIndex{
		hnsw: h,
		gpu:  &gpuState{dev: dev, dim: cfg.Dim},
	}, nil
}

// openGPUIndex reopens a GPU index. A GPU collection is snapshot-only (heap-
// backed, like a non-persistent index), so reopen returns a fresh empty inner
// index the caller then Restores (mirroring openIVF/openVamana). A fresh device
// handle is created; the resident buffer rebuilds lazily on the first search
// after Restore (idSetVersion is bumped by Restore, so the generation check
// triggers an upload).
func openGPUIndex(cfg Config, metaPath string) (VectorIndex, error) {
	h, err := newHNSW(cfg)
	if err != nil {
		return nil, err
	}
	dev, err := cuda.New(cfg.Dim)
	if err != nil {
		_ = h.Close()
		return nil, err
	}
	return &gpuIndex{
		hnsw: h,
		gpu:  &gpuState{dev: dev, dim: cfg.Dim},
	}, nil
}

// ensureResidentLocked (re)uploads the resident corpus to the device when the
// live id-set generation has advanced since the last upload (or nothing has
// been uploaded yet). Must hold h.mu (write). Uploads the FULL slot range
// [0, Capacity()) so the GPU row index equals the arena slot index (an identity
// mapping); tombstoned/expired slots are uploaded too and filtered at admission.
func (g *gpuIndex) ensureResidentLocked() error {
	h := g.hnsw
	if h.arena.vecsDropped {
		return errors.New("vector: gpuIndex: resident floats dropped (unsupported)")
	}
	capacity := h.arena.Capacity()
	if g.gpu.built && g.gpu.uploadedVer == h.idSetVersion && g.gpu.uploadedRows == capacity {
		return nil // resident buffer is current
	}
	// arena.vecs is laid out row-major (slot*dim) exactly as the kernel expects.
	if err := g.gpu.dev.Upload(h.arena.vecs, capacity); err != nil {
		g.gpu.built = false
		return err
	}
	g.gpu.uploadedVer = h.idSetVersion
	g.gpu.uploadedRows = capacity
	g.gpu.built = true
	return nil
}

// gpuScoreToDist converts the kernel's raw score (dot for Cosine/Dot, squared
// distance for L2) to the metric's distFunc value (smaller = nearer), matching
// pickDist exactly: Cosine -> 1 - dot, L2 -> the squared distance as-is,
// DotProduct -> -dot.
func gpuScoreToDist(m Metric, score float32) float32 {
	switch m {
	case Cosine:
		return 1.0 - score
	case L2:
		return score
	case DotProduct:
		return -score
	default:
		return score
	}
}

// admitFn is the admission gate variant: plain (admits) or external-provider
// (admitsWith). It abstracts the only difference between the single-vector and
// named/MV search paths.
type admitFn func(slot uint32, pred Predicate) bool

// gpuSearchLocked runs the exact GPU KNN for a single query under h.mu (write),
// applying admit (the same tombstone/TTL/predicate gate the CPU exact path uses)
// and returns the admitted top-k as []Result with distances reoriented to the
// metric's distFunc convention. q must already be cosine-normalized when the
// metric is Cosine. When pred != nil the GPU still scores all rows; over-fetch +
// host admission yields the exact admitted top-k.
//
// Exactness across the MAX_K kernel limit: the kernel can only emit up to
// cuda.MaxK (256) outputs per query, so we dispatch with kGPU = min(kfetch,
// MaxK). The GPU returns the top-kGPU by raw score; we apply admit to those
// candidates. If that yields >= k admitted results we return them (the common
// FAST PATH: k <= 256, non-selective). Otherwise — a highly selective filter,
// >MaxK tombstones, or k > MaxK so the GPU could not fetch enough — we FALL BACK
// to a CPU-exact brute force over the live arena vectors. This preserves the
// exact top-k contract for the pathological cases at CPU cost; the top-k is
// never silently truncated.
func (g *gpuIndex) gpuSearchLocked(dst []Result, q []float32, k int, pred Predicate, admit admitFn) ([]Result, error) {
	h := g.hnsw
	if err := g.ensureResidentLocked(); err != nil {
		return dst, err
	}
	capacity := h.arena.Capacity()
	if capacity == 0 || k <= 0 {
		return dst, nil
	}

	// Over-fetch so that, after dropping tombstoned/expired/filtered rows, we
	// still have k admitted results. With a predicate active, fetch the full
	// corpus rank so the admitted top-k is exact regardless of selectivity
	// (brute force is exact — the simplest correct approach). With no predicate,
	// fetch k plus the tombstone count.
	kfetch := k
	if pred != nil {
		kfetch = capacity
	} else if dead := len(h.tombstoned) + g.numExpiredLocked(); dead > 0 {
		// Widen by BOTH tombstoned and TTL-expired-but-unswept slots. Expired
		// slots are dropped at admission just like tombstones, so a TTL workload
		// between sweeps would otherwise under-fetch and fall back to the CPU
		// exact path on every query (see audit finding). Counting expired live
		// slots is an O(capacity) scan, still far cheaper than the fallback.
		kfetch = k + dead
	}
	if kfetch > capacity {
		kfetch = capacity
	}

	// The kernel caps a single dispatch at cuda.MaxK outputs per query. Clamp the
	// GPU fetch to that and use the same stride for the result buffers.
	kGPU := kfetch
	if kGPU > cuda.MaxK {
		kGPU = cuda.MaxK
	}

	outIdx := make([]int32, kGPU)
	outScore := make([]float32, kGPU)
	if err := g.gpu.dev.Search(q, 1, kGPU, metricCode(h.cfg.Metric), outIdx, outScore); err != nil {
		return dst, err
	}

	base := len(dst)
	added := 0
	for i := 0; i < kGPU && added < k; i++ {
		row := int(outIdx[i])
		if row < 0 {
			break // sentinel padding (fewer rows than kGPU)
		}
		slot := uint32(row)
		if !admit(slot, pred) {
			continue
		}
		// The GPU scores the whole uploaded row range [0, Capacity()) by slot
		// index, so unlike the CPU lanes it never consults h.nodes and cannot rely
		// on the graph to keep dead slots out of reach. Two kinds of dead slot land
		// here, and admit() rejects NEITHER:
		//
		//   - Reserved but never written (Reserve pre-sizes the arena).
		//   - FREED BY Reclaim — the likelier one in production. Reclaim empties
		//     h.tombstoned wholesale and arena.Delete clears the expiry, so nothing
		//     is left for admit() to catch, and notYetLinked is false once the node
		//     is nil. The freed slot keeps a STALE id, usually NON-ZERO, so the old
		//     `id == 0` guard let it straight through: these lanes have been
		//     emitting ghost results for reclaimed points.
		//
		// Allocated()'s idMap round-trip rejects both. This is a real behaviour
		// change for the GPU lanes, not just the id-0 repair the CPU sites got.
		if !h.arena.Allocated(slot) {
			continue
		}
		dst = append(dst, Result{ID: h.slotID(slot), Distance: gpuScoreToDist(h.cfg.Metric, outScore[i])})
		added++
	}

	// Fast path: the GPU top-kGPU yielded enough admitted candidates.
	if added >= k {
		// Re-sort the admitted candidates by the SAME deterministic (distance, id)
		// comparator the CPU fallback uses. The GPU kernel breaks score ties by row
		// (slot) order, so without this an identical query+corpus could return a
		// different id ordering on the fast path than on the fallback path. Sorting
		// the admitted window makes the two paths byte-for-byte order-identical.
		out := dst[base:]
		sort.Slice(out, func(a, b int) bool {
			if out[a].Distance != out[b].Distance {
				return out[a].Distance < out[b].Distance
			}
			return out[a].ID < out[b].ID
		})
		return dst, nil
	}
	// The GPU top-kGPU did not contain k admitted candidates (selective filter,
	// >MaxK tombstones, or k > MaxK). Discard the partial GPU result and recompute
	// the exact admitted top-k on the CPU over the full live arena.
	dst = dst[:base]
	return g.cpuExactSearchLocked(dst, q, k, pred, admit)
}

// numExpiredLocked counts the live arena slots whose TTL deadline has passed but
// which have not yet been swept/tombstoned. These slots are dropped at admission
// exactly like tombstones, so the no-predicate GPU over-fetch must account for
// them or a TTL workload silently runs on the CPU-exact fallback between sweeps.
// O(capacity); only meaningful when the collection uses TTL. Must hold h.mu.
func (g *gpuIndex) numExpiredLocked() int {
	h := g.hnsw
	now := uint64(h.now())
	capacity := h.arena.Capacity()
	n := 0
	for s := 0; s < capacity; s++ {
		slot := uint32(s) //nolint:gosec // slot < capacity < 2^32
		if h.tombstoned[slot] {
			continue // already counted via len(h.tombstoned)
		}
		if exp := h.arena.ExpiresAt(slot); exp != 0 && exp <= now {
			n++
		}
	}
	return n
}

// cpuExactSearchLocked computes the exact admitted top-k by scoring every live
// arena vector against q with the metric's distFunc and applying the same admit
// gate the GPU path uses. It is the production exact fallback for the cases the
// GPU kernel cannot satisfy in a single dispatch (selective filters, >MaxK
// tombstones, k > cuda.MaxK). Must hold h.mu. q must already be cosine-normalized
// when the metric is Cosine (it is scored directly against the stored arena
// vectors, which are likewise normalized for Cosine). Distances are in the
// metric's distFunc convention (smaller = nearer), so results are returned
// nearest-first without any reorientation.
func (g *gpuIndex) cpuExactSearchLocked(dst []Result, q []float32, k int, pred Predicate, admit admitFn) ([]Result, error) {
	h := g.hnsw
	dist := pickDistDim(h.cfg.Metric, h.cfg.Dim)
	capacity := h.arena.Capacity()

	type cand struct {
		id uint64
		d  float32
	}
	cands := make([]cand, 0, capacity)
	for s := 0; s < capacity; s++ {
		slot := uint32(s)
		if !admit(slot, pred) {
			continue
		}
		// Full-capacity scan by slot index: [0, Capacity()) includes both
		// reserved-but-unwritten slots and slots FREED BY Reclaim, and admit()
		// rejects neither (Reclaim empties h.tombstoned, arena.Delete clears the
		// expiry, notYetLinked is false for a nil node). A reclaimed slot keeps a
		// STALE, usually NON-ZERO id, so the old `id == 0` guard never caught it —
		// this lane has been emitting ghost results for reclaimed points.
		// Allocated()'s idMap round-trip rejects both kinds.
		if !h.arena.Allocated(slot) {
			continue
		}
		cands = append(cands, cand{id: h.slotID(slot), d: dist(q, h.arena.Vec(slot))})
	}
	sort.Slice(cands, func(a, b int) bool {
		if cands[a].d != cands[b].d {
			return cands[a].d < cands[b].d
		}
		return cands[a].id < cands[b].id
	})
	for i := 0; i < len(cands) && i < k; i++ {
		dst = append(dst, Result{ID: cands[i].id, Distance: cands[i].d})
	}
	return dst, nil
}

// --- Overridden dense-search entry points ---

// Search returns the k nearest neighbors (no filter) via the GPU exact KNN.
func (g *gpuIndex) Search(query []float32, k int) ([]Result, error) {
	return g.SearchInto(nil, query, k, Filter{})
}

// SearchFiltered returns the k nearest neighbors satisfying filter via the GPU.
func (g *gpuIndex) SearchFiltered(query []float32, k int, filter Filter) ([]Result, error) {
	return g.SearchInto(nil, query, k, filter)
}

// SearchInto appends up to k GPU-exact nearest neighbors matching filter onto
// dst. Mirrors hnsw.SearchInto's contract (dim check, k<=0 no-op, cosine
// normalization, filter compile) but serves the dense KNN from the GPU.
func (g *gpuIndex) SearchInto(dst []Result, query []float32, k int, filter Filter) ([]Result, error) {
	return g.searchIntoGPU(dst, query, k, filter, nil)
}

// SearchFilteredWith is SearchInto with an OPTIONAL external metadata provider.
// metaOf == nil is byte-identical to SearchInto. When metaOf != nil the
// predicate is evaluated against the external payload (admitsWith) — the
// named/MV hook. The GPU still scores all rows; admission is rerouted to the
// provider.
func (g *gpuIndex) SearchFilteredWith(dst []Result, query []float32, k int, filter Filter, metaOf func(id uint64) Metadata) ([]Result, error) {
	return g.searchIntoGPU(dst, query, k, filter, metaOf)
}

// searchIntoGPU is the shared GPU dense-search body. It reproduces hnsw's query
// preparation (dim validation, cosine normalization, filter compilation), then
// runs the GPU exact KNN under h.mu and applies the CPU-identical admission gate
// (admits, or admitsWith when metaOf != nil).
func (g *gpuIndex) searchIntoGPU(dst []Result, query []float32, k int, filter Filter, metaOf func(id uint64) Metadata) ([]Result, error) {
	h := g.hnsw
	if len(query) != h.cfg.Dim {
		return dst, ErrDimMismatch
	}
	if k <= 0 {
		return dst, nil
	}
	pred, err := CompileFilter(filter)
	if err != nil {
		return dst, err
	}

	q := query
	if h.cfg.Metric == Cosine {
		qbuf := make([]float32, len(query))
		copy(qbuf, query)
		normalize(qbuf)
		q = qbuf
	}

	now := uint64(h.now()) // one clock read for the whole admission scan
	var admit admitFn
	if metaOf == nil {
		admit = func(slot uint32, p Predicate) bool { return h.admits(slot, p, now) }
	} else {
		admit = func(slot uint32, p Predicate) bool { return h.admitsWith(slot, p, metaOf, now) }
	}

	h.mu.Lock()
	defer h.mu.Unlock()
	h.searchOps.Add(1)

	return g.gpuSearchLocked(dst, q, k, pred, admit)
}

// Close releases the device handle and then the inner hnsw.
func (g *gpuIndex) Close() error {
	if g.gpu != nil && g.gpu.dev != nil {
		g.gpu.dev.Free()
		g.gpu.dev = nil
	}
	return g.hnsw.Close()
}

// gpuSearchBatch runs the GPU exact KNN for a BATCH of queries in one kernel
// dispatch (the throughput path used by the host-side bench). It returns, per
// query, the admitted top-k []Result (no filter). Queries must be already
// prepared (cosine-normalized when the metric is Cosine). Takes h.mu (write).
func (g *gpuIndex) gpuSearchBatch(queries [][]float32, k int) ([][]Result, error) {
	h := g.hnsw
	nq := len(queries)
	if nq == 0 || k <= 0 {
		return nil, nil
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if err := g.ensureResidentLocked(); err != nil {
		return nil, err
	}
	capacity := h.arena.Capacity()
	if capacity == 0 {
		return make([][]Result, nq), nil
	}
	kfetch := k
	if dead := len(h.tombstoned); dead > 0 {
		kfetch = k + dead
	}
	if kfetch > capacity {
		kfetch = capacity
	}
	// The kernel caps each query at cuda.MaxK outputs; the result buffers are
	// laid out with this SAME stride (kGPU), so index per query by kGPU — NOT
	// kfetch (which may exceed MaxK).
	kGPU := kfetch
	if kGPU > cuda.MaxK {
		kGPU = cuda.MaxK
	}
	flat := make([]float32, nq*h.cfg.Dim)
	for i, qv := range queries {
		copy(flat[i*h.cfg.Dim:(i+1)*h.cfg.Dim], qv)
	}
	outIdx := make([]int32, nq*kGPU)
	outScore := make([]float32, nq*kGPU)
	if err := g.gpu.dev.Search(flat, nq, kGPU, metricCode(h.cfg.Metric), outIdx, outScore); err != nil {
		return nil, err
	}
	now := uint64(h.now()) // one clock read for the whole batch admission scan
	admit := func(slot uint32, p Predicate) bool { return h.admits(slot, p, now) }
	results := make([][]Result, nq)
	for qi := 0; qi < nq; qi++ {
		var rs []Result
		added := 0
		base := qi * kGPU
		for i := 0; i < kGPU && added < k; i++ {
			row := int(outIdx[base+i])
			if row < 0 {
				break
			}
			slot := uint32(row)
			if !admit(slot, nil) {
				continue
			}
			// Row range [0, Capacity()) scanned by slot index: includes both
			// reserved-but-unwritten slots and slots FREED BY Reclaim, and admit()
			// rejects neither (Reclaim empties h.tombstoned, arena.Delete clears
			// the expiry, notYetLinked is false for a nil node). A reclaimed slot
			// keeps a STALE, usually NON-ZERO id, so the old `id == 0` guard never
			// caught it — this lane has been emitting ghost results for reclaimed
			// points. Allocated()'s idMap round-trip rejects both kinds.
			if !h.arena.Allocated(slot) {
				continue
			}
			rs = append(rs, Result{ID: h.slotID(slot), Distance: gpuScoreToDist(h.cfg.Metric, outScore[base+i])})
			added++
		}
		// Fall back to a CPU-exact top-k when the GPU top-kGPU could not supply k
		// admitted candidates (>MaxK tombstones or k > cuda.MaxK).
		if added < k {
			exact, err := g.cpuExactSearchLocked(nil, queries[qi], k, nil, admit)
			if err != nil {
				return nil, err
			}
			rs = exact
		}
		results[qi] = rs
	}
	return results, nil
}

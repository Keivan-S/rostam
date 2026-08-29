// SPDX-License-Identifier: Apache-2.0

package vtypes

import (
	"errors"
	"time"
)

// Metric identifies a distance / similarity function.
type Metric uint8

const (
	// Cosine treats vectors as directions and minimizes 1 - dot(a,b)/(|a|*|b|).
	// Vectors are normalized on insert so the hot path is a single dot product.
	Cosine Metric = iota
	// L2 minimizes squared Euclidean distance. Results expose the squared
	// distance directly; callers take sqrt at the API boundary if needed.
	L2
	// DotProduct returns -dot(a,b) (negated so smaller = more similar). Caller
	// is responsible for any normalization.
	DotProduct
)

// IndexType selects the nearest-neighbor index implementation behind a
// Collection. The zero value is IndexHNSW so an unset config (and every existing
// persisted/wire config) keeps the historical HNSW behavior — backward-compatible
// by construction.
type IndexType uint8

const (
	// IndexHNSW is the default graph index (Hierarchical Navigable Small World).
	IndexHNSW IndexType = iota
	// IndexIVF is the IVF-Flat index: a k-means coarse quantizer over inverted
	// lists, trained at BuildConcurrent. nlist/nprobe come from Config.IVFNlist /
	// Config.IVFNprobe (0 = the index's computed/default values).
	IndexIVF
	// IndexVamana is the DiskANN-family Vamana graph: a SINGLE-LAYER graph with
	// bounded out-degree R, built by the α-RobustPrune two-pass algorithm. It
	// reuses the entire HNSW engine constrained to one layer (every node level 0)
	// with an α-RobustPrune neighbor rule and a two-pass build; search enters at
	// the medoid and runs the level-0 ef-beam (no multi-level descent). R/L/alpha
	// come from Config.VamanaR / Config.VamanaL / Config.VamanaAlpha.
	IndexVamana
	// IndexGPU is an OPTIONAL GPU-accelerated (brute-force / exact-KNN) index,
	// available ONLY in a build with -tags cuda (which requires CGO + the CUDA
	// toolkit + an NVIDIA GPU). The DEFAULT pure-Go, CGO_ENABLED=0 build never
	// compiles the GPU code path: selecting IndexGPU there fails LOUD at config
	// time with ErrGPUNotCompiled (never a silent HNSW fallback). The gpuSupported
	// constant (build-tag split) gates this; dispatch goes through
	// newGPUIndex/openGPUIndex (also build-tag split).
	//
	// GPU-exact surface: ONLY the four core dense-KNN entry points (Search,
	// SearchFiltered, SearchInto, SearchFilteredWith) are GPU-exact. MMR / Groups /
	// Hybrid / Discover / Recommend (SearchMMR, SearchGroups, HybridSearch,
	// DiscoverVecs, RecommendVecs) fall back to the APPROXIMATE inner HNSW graph —
	// they are inherited from the embedded engine and not re-routed through the GPU
	// kernel. The dense fast path serves k <= 256 on the GPU and transparently
	// falls back to a CPU-exact brute force above that (selective filters, > 256
	// tombstones, or k > 256), so the core top-k is always exact, never truncated.
	IndexGPU
)

// QuantMode selects the vector quantization scheme for an index.
type QuantMode uint8

const (
	// QuantNone stores full-precision float32 vectors with no quantization.
	// This is the default.
	QuantNone QuantMode = iota
	// QuantSQ8 stores scalar int8 codes (4× smaller than float32) and rescores
	// the over-collected candidate set on full-precision float32. Cosine-scope
	// for v1 (see the quantization design spec).
	QuantSQ8
	// QuantBQ1 stores 1-bit-per-dimension sign codes (32× smaller than float32),
	// navigates the graph by Hamming distance, and rescores on float32. Best for
	// high-dimensional normalized embeddings. Cosine-scope for v1.
	QuantBQ1
	// QuantPQ stores product-quantization codes (m bytes/vector, nbits=8). Used by
	// the IVF-PQ index mode: the residual codebooks + per-cell ADC LUT live on the
	// pq codec and are driven by the IVF (see pq.go). The newQuantizer adapter for
	// this mode exists so the arena sizes the codes side-array to CodeLen()==m and
	// can Encode on insert; the residual LUT scoring path is IVF-driven, not via
	// the centroid-agnostic Distance below.
	QuantPQ
	// QuantSQ is the TRAINED, metric-agnostic scalar quantizer (trainedSQ). Unlike
	// the legacy fixed-scale QuantSQ8 (a 1/127 Cosine-only fast-path), it learns a
	// per-dimension [min,max] range from a build sample and scores asymmetrically
	// under the index's ACTUAL metric (Cosine/L2/DotProduct). SQBits selects the
	// bit-depth (8-bit only). Added AFTER QuantPQ so existing on-disk
	// QuantNone/SQ8/BQ1/PQ enum values are unchanged. See sq.go.
	QuantSQ
	// QuantPRQ is PRODUCT-RESIDUAL quantization: a stack of PRQLayers (default 2)
	// product-quantizer layers where each layer quantizes the RESIDUAL of the
	// previous one. Code = PRQLayers·m bytes; reconstruction = Σ layer
	// reconstructions; ADC = sum of the per-layer LUTs (the additive approximation,
	// since HNSW's exact float rescore fixes the final ranking). Strictly higher
	// accuracy than plain PQ at the same m. Added AFTER QuantSQ so existing on-disk
	// QuantNone/SQ8/BQ1/PQ/SQ enum values are unchanged. See prq.go.
	QuantPRQ
)

// FullTextConfig enables and parameterizes the BM25 full-text lane. The zero
// value (Analyzer="", K1=0, B=0) resolves to the default English analyzer and
// the standard BM25 knobs (k1=1.2, b=0.75). A nil *FullTextConfig on Config means
// full text is DISABLED.
type FullTextConfig struct {
	// Analyzer is the registered analyzer name (analysis.ByName). "" → the default
	// English analyzer ("english").
	Analyzer string `json:"analyzer,omitempty"`
	// K1 is the BM25 term-frequency saturation knob; 0 → 1.2.
	K1 float32 `json:"k1,omitempty"`
	// B is the BM25 length-normalization knob; 0 → 0.75.
	B float32 `json:"b,omitempty"`
}

// Config configures a VectorIndex. Dim and Metric are required; the rest
// have sensible HNSW defaults (matching Qdrant).
type Config struct {
	Dim            int // dimensionality, required
	Metric         Metric
	M              int   // graph degree (default 16)
	EfConstruction int   // build-time width (default 200)
	EfSearch       int   // query-time width (default 64)
	Seed           int64 // RNG seed for level assignment

	// SweepInterval is the period between TTL sweeps. 0 means use the
	// package default (60 s). Set to a small value in tests for
	// deterministic timing.
	SweepInterval time.Duration

	// SuppressSweep disables the background wall-clock TTL sweeper entirely (the
	// startSweeper goroutine becomes a no-op). Set by the persistent-cluster policy
	// (effectiveClusterConfig, alongside WAL=false): under Raft replication the
	// sweeper's wall-clock physical removal (point tombstones + per-key drops) would
	// diverge committed state across replicas whose clocks differ, so it is turned
	// off and expired entries are instead filtered lazily at read time (client
	// staleness only, never a state mutation) — the vector analog of the KV
	// replicated no-wall-clock-removal rule (#4 vector TTL determinism, B3a). Space
	// reclamation of expired vectors on a replicated collection is therefore
	// deferred to a future deterministic stamped reclaimer (B3b analog). Default
	// false = single-node behavior (sweeper on), byte-identical.
	SuppressSweep bool

	// MaxVectors is the upper bound on live (non-tombstoned) vectors. 0 = unlimited.
	// Insert returns ErrCollectionFull when arena.Size() >= MaxVectors.
	MaxVectors int64

	// MaxBytes is the upper bound on estimated memory use (arena + graph).
	// 0 = unlimited. Insert returns ErrCollectionFull when the next insert
	// would push usage past the limit.
	MaxBytes int64

	// MaxInsertsPerSecond rate-limits inserts via a token bucket. 0 = unlimited.
	// Insert returns ErrCollectionRateLimited when the bucket is empty.
	MaxInsertsPerSecond int

	// MaxEfSearch bounds the dynamic ef widening used by filtered search.
	// When a filter rejects most candidates, search doubles ef (starting at
	// max(EfSearch, 2*k)) up to this cap to recover recall. 0 = package
	// default (1024). Unfiltered search ignores this field.
	MaxEfSearch int

	// Quant selects vector quantization. QuantNone (default) stores full
	// float32. QuantSQ8 stores int8 codes (4× smaller) for graph traversal and
	// rescores the top candidates on float32. Cosine-scope for v1.
	Quant QuantMode

	// RescoreFactor sets how many candidates the exact rescore stage re-ranks
	// when quantization is active: rescored = RescoreFactor * k. 0 = package
	// default (3). Ignored when Quant is QuantNone.
	RescoreFactor int

	// QuantPQM is the number of PQ sub-quantizers (subspaces) for a QuantPQ dense
	// (HNSW) index; Dim must be divisible by it. 0 = defaultPQM(Dim). Ignored
	// unless Quant == QuantPQ. (The IVF-PQ index uses IVFPQM, a separate field, so
	// the two PQ families never share a knob.)
	QuantPQM int `json:"quant_pq_m,omitempty"`

	// PQNBits is the per-subspace PQ code width for a QuantPQ dense (HNSW) index:
	// 0/8 ⇒ the 8-bit codec (256 sub-centroids/subspace, m bytes/code — the existing
	// path, byte-identical), 4 ⇒ the 4-bit LUT16 fast-scan codec (16 sub-centroids/
	// subspace, ceil(m/2) bytes/code, scored by the AVX2 VPSHUFB in-register table
	// lookup). 4-bit is lossier candidate-gen ADC; the exact float32 rescore recovers
	// the final ranking. Ignored unless Quant == QuantPQ. omitempty keeps a non-4-bit
	// collection's create-args JSON byte-identical (the key is absent when 0).
	// Validated 0/4/8 (rejected otherwise). CodeLen depends on it, so it is carried
	// in snapColCfg + every restore path (the geometry-on-restore lesson).
	PQNBits int `json:"pq_nbits,omitempty"`

	// SQBits is the bit-depth of the TRAINED scalar quantizer (Quant == QuantSQ).
	// 0 ⇒ 8 (the only depth currently supported; 4/6 may follow). Ignored unless
	// Quant == QuantSQ. omitempty keeps a non-SQ collection's create-args JSON
	// byte-identical (the key is absent when zero).
	SQBits int `json:"sq_bits,omitempty"`

	// PRQLayers is the number of product-residual quantization layers
	// (Quant == QuantPRQ). 0 ⇒ defaultPRQLayers (2). Code = PRQLayers·QuantPQM
	// bytes/vector; more layers = finer residual = better recall at proportionally
	// more bytes. Ignored unless Quant == QuantPRQ. omitempty keeps a non-PRQ
	// collection's create-args JSON byte-identical (the key is absent when zero).
	PRQLayers int `json:"prq_layers,omitempty"`

	// QuantStorage selects where float32 vectors live when quantization is
	// enabled. QuantInRAM (default) keeps them on the heap; QuantMmap stores
	// them in a memory-mapped file (MmapPath) so only codes are resident.
	QuantStorage QuantStore

	// MmapPath is the backing file for float32 vectors when QuantStorage is
	// QuantMmap. Required in that mode; ignored otherwise.
	MmapPath string

	// Persistent makes a CollectionStore manage this collection's storage as
	// mmap-backed files (vectors + level-0 graph) so the collection reopens via
	// instant restart (map files, no graph rebuild) instead of replaying a
	// snapshot. The store owns the file paths (it overrides QuantStorage,
	// MmapPath, and GraphMmapPath); the caller only sets Persistent + a
	// quantizer. Requires Quant != QuantNone (mmap-backed vectors need a
	// quantizer). Scope matches SavePersist: dense (no tombstones) — metadata,
	// TTL, and sparse vectors are all persisted.
	Persistent bool

	// WAL enables a write-ahead log so each Insert/Delete is crash-durable
	// between Flush checkpoints (without it, ops since the last Flush are lost on
	// a hard crash). The CollectionStore manages the log: it fsyncs each op
	// (unless WALNoSync), checkpoints via a snapshot on Flush, replays the log on
	// open, and rotates it. WAL collections are heap-backed; WAL and Persistent
	// are mutually exclusive (Persistent is mmap instant-restart, durable only at
	// Flush — choose one durability strategy).
	WAL bool

	// WALNoSync skips the per-op fsync, trading crash-durability for throughput
	// (ops can be lost on power loss, but a clean process exit is still durable).
	// Ignored unless WAL is set.
	WALNoSync bool

	// GraphMmapPath, when non-empty, backs the level-0 adjacency slab (the bulk
	// of the HNSW graph) with a memory-mapped file instead of the Go heap, so
	// those bytes are off-heap and OS-reclaimable. Orthogonal to QuantStorage —
	// combine with QuantMmap to push both the vectors and the graph's largest
	// array off-heap. The file is a native-endian runtime backing store (like
	// MmapPath), not a portable artifact, and must differ from MmapPath.
	// Linux and Windows only; other platforms return an mmap-unsupported error.
	GraphMmapPath string

	// FilterFirstThreshold caps the candidate-set size for which an
	// equality-filtered search is answered by exact brute force over the
	// payload index instead of graph traversal. 0 = package default (10000).
	// Larger values favor the exact (and, for selective filters, faster)
	// filter-first path over approximate graph search.
	FilterFirstThreshold int

	// FilterFirstRelativeBP is an OPT-IN relative selectivity gate (basis points
	// of the LIVE collection size, 0..10000; 0 = OFF = current absolute behavior,
	// BYTE-IDENTICAL). When > 0 the effective filter-first materialization limit
	// becomes max(absThreshold, min(FilterFirstRelativeBP*liveCount/10000,
	// hardCap=1_000_000)), so a relatively-selective filter (e.g. 50k matches on a
	// 100M-doc collection = 0.05%) can use exact filter-first BEYOND the absolute
	// 10k cap, bounded by a hard ceiling so a non-selective filter on a billion-doc
	// collection cannot blow memory. The EXISTING preferFilterFirst cost model
	// still makes the final filter-first-vs-graph decision — raising the ceiling
	// only widens WHICH filters are CONSIDERED. 0 (the default) returns absThreshold
	// exactly at every gate site (see effectiveFilterFirstLimit), so an unset field
	// is byte/behaviour-identical to today.
	FilterFirstRelativeBP int

	// ExtendCandidates enables the Malkov-Yashunin extendCandidates variant of
	// the neighbor-selection heuristic at build time: before the diversity pick,
	// the candidate pool is enriched with the (already-linked) neighbors of each
	// candidate. The richer, second-hop pool yields more directionally-diverse
	// edges, which raises the graph's recall ceiling on clustered/anisotropic
	// data (real embeddings) — the high-ef end of the recall/QPS curve. It costs
	// only build time (more distance computations during construction); query
	// latency and the on-disk graph layout are unchanged. Off by default.
	// Measured on SIFT (200k x 128d, k=10, m=16, efc=200) it moved QPS at
	// matched recall by +0.6% / -4.2% — the sign flips between runs, i.e. no
	// measurable effect — for ~2.5x build time (see recall_levers_test.go).
	// For recall-critical collections prefer Level0FullDegree, which measured
	// a real Pareto gain on the same harness; verify on the target corpus
	// before enabling this.
	ExtendCandidates bool

	// ExtendCandidatesMax bounds the enriched pool when ExtendCandidates is set:
	// once the pool (base candidates plus second-hop neighbors, added
	// closest-first) reaches this many entries, no further neighbors are pulled
	// in. This caps the extra build-time distance work to ~this many
	// computations per node per level, so extension stays affordable on large
	// collections while keeping most of its recall benefit. 0 = unbounded (the
	// full second-hop pool). Ignored unless ExtendCandidates is set.
	ExtendCandidatesMax int

	// Level0FullDegree selects up to M0 = 2*M *forward* neighbors for each new
	// node at level 0 (the dense bottom layer), instead of M. The level-0 cap is
	// already 2*M; by default Rostam fills only M forward edges and lets reverse
	// edges grow degree opportunistically, but those reverse edges are chosen for
	// the neighbor's neighborhood, not this node's. Selecting 2*M forward edges
	// via the heuristic gives every node a full set of directionally-diverse
	// bottom-layer links — matching Qdrant's m0 convention. The level-0 slab
	// already reserves 2*M edge slots per node, so this costs no extra memory —
	// only ~1.6x build time and a modestly wider bottom-layer search. Measured
	// on SIFT (200k x 128d, k=10, m=16, efc=200): +8.5% / +12.2% QPS at matched
	// recall, every measured point positive in both runs (see
	// recall_levers_test.go). Recommended for high-recall workloads; confirm on
	// the target corpus, since graph-quality effects are geometry-dependent.
	Level0FullDegree bool

	// QuantizedBuild navigates and selects neighbors on the int8 quantizer codes
	// during the bulk build (BuildConcurrent) and serial Insert, instead of exact
	// float32. At high dimension the build is memory-bandwidth-bound — codes are
	// 4x smaller, so this is the dominant lever for build/ingest speed (it is how
	// Qdrant's quantized indexing is fast). The graph is built from approximate
	// distances, so recall is slightly lower; the query-time rescore recovers the
	// final ranking. Requires Quant != QuantNone. Off by default (the default
	// build navigates on exact float32 for maximum graph quality).
	QuantizedBuild bool

	// Partitions is the number of partitions a collection is split into on the
	// clustered backend. 0 or 1 = single-partition (default; routes by bare
	// collection name exactly as before). >1 distributes vectors by hash(id)%P
	// across partitions and makes search a scatter-gather fan-out. Ignored by the
	// single-node directStore (always treated as 1). Immutable after creation
	// except via the offline resplit op.
	Partitions int

	// NamedVectors, when non-empty, makes this a Qdrant-style NAMED-VECTOR
	// collection: a MAP of named dense vector spaces (e.g. {"title":…,"image":…}),
	// each its own HNSW sub-index, all sharing one per-point payload + point-id
	// namespace. It is an OPT-IN new collection family alongside dense + MV; when
	// set, the CollectionStore routes creation to a NamedCollection and the
	// per-space index params come from each NamedVectorParams. The dense
	// collection-level fields above (Partitions/WAL/Persistent/…) still apply at
	// the collection level; per-space index params live in NamedVectorParams.
	NamedVectors map[string]NamedVectorParams `json:"named_vectors,omitempty"`

	// IndexType selects the backing index implementation. IndexHNSW (0, default)
	// keeps the historical graph index; IndexIVF builds an IVF-Flat index (k-means
	// coarse quantizer + inverted lists, trained at BuildConcurrent). The zero
	// value is HNSW so every existing config restores/decodes unchanged.
	IndexType IndexType `json:"index_type,omitempty"`

	// IVFNlist is the number of IVF inverted lists (Voronoi cells). 0 = computed at
	// train time as max(1, 4*sqrt(N)). Ignored unless IndexType == IndexIVF.
	IVFNlist int `json:"ivf_nlist,omitempty"`

	// IVFNprobe is the number of nearest cells an IVF query probes. 0 = the index
	// default (8). Ignored unless IndexType == IndexIVF.
	IVFNprobe int `json:"ivf_nprobe,omitempty"`

	// IVFPQ enables IVF-PQ mode: the IVF stores compact residual product-
	// quantization codes (m bytes/vector, nbits=8) instead of full float vectors
	// and scores candidates by asymmetric distance (ADC). PQ-only by default
	// (IVFRerank=false) drops the resident floats for maximum compression; set
	// IVFRerank to keep full vectors for an exact final rescore. Ignored unless
	// IndexType == IndexIVF. The zero value (false) keeps the exact IVF-Flat path
	// byte/behavior-identical.
	IVFPQ bool `json:"ivf_pq,omitempty"`

	// IVFPQM is the number of PQ sub-quantizers (subspaces); dim must be divisible
	// by it. 0 = defaultPQM(dim). Ignored unless IVFPQ is set.
	IVFPQM int `json:"ivf_pq_m,omitempty"`

	// IVFRerank, when set with IVFPQ, keeps the full float vectors resident and
	// exact-rescores the ADC shortlist (rerankFactor*k candidates) for near-exact
	// recall at a higher memory cost. Off by default = PQ-only (ADC scores are the
	// final ranking, floats dropped).
	IVFRerank bool `json:"ivf_rerank,omitempty"`

	// VamanaR is the max out-degree (R) of a Vamana graph: the level-0 adjacency
	// slab stride and the RobustPrune neighbor cap. 0 ⇒ defaultVamanaR (64).
	// Ignored unless IndexType == IndexVamana. omitempty keeps a non-Vamana
	// collection's create-args JSON byte-identical (the key is absent when zero).
	VamanaR int `json:"vamana_r,omitempty"`

	// VamanaL is the Vamana build/search beam width (the visited-set size the
	// greedy search collects, and the search ef floor). 0 ⇒ defaultVamanaL (100).
	// Ignored unless IndexType == IndexVamana. omitempty keeps a non-Vamana
	// collection's create-args JSON byte-identical.
	VamanaL int `json:"vamana_l,omitempty"`

	// VamanaAlpha is the pass-2 RobustPrune α (≥ 1): a candidate v is dropped when
	// α·dist(kept*, v) < dist(p, v), so a larger α keeps more long-range edges
	// (better recall, denser graph). Pass 1 always uses α=1. 0 ⇒ defaultVamanaAlpha
	// (1.2). Ignored unless IndexType == IndexVamana. omitempty keeps a non-Vamana
	// collection's create-args JSON byte-identical.
	VamanaAlpha float32 `json:"vamana_alpha,omitempty"`

	// OPQ enables an optional orthogonal rotation R (d×d) applied inside the PQ
	// codec BEFORE the sub-space split (and un-applied by Rᵀ in reconstruct), so
	// the M PQ sub-spaces carry balanced variance → higher recall at the same
	// M/nbits. v1 is a seeded RANDOM orthogonal rotation (Gram-Schmidt of a
	// Gaussian matrix; deterministic in Seed). It composes with both IVF-PQ
	// (rotates the residual) and HNSW-PQ. Requires a PQ mode (QuantPQ or IVFPQ);
	// default false ⇒ rotation nil ⇒ byte/behaviour-identical to plain PQ. The
	// IVF/HNSW build wiring is not yet done; the codec + flag ship now.
	OPQ bool `json:"opq,omitempty"`

	// OPQIters drives FULL-OPQ iterative Procrustes refinement (Ge et al. 2013).
	// Ignored unless OPQ is set. 0 (default) means 1 = the v1 single-random-rotation
	// behavior, BYTE-IDENTICAL to before (no SVD, no refinement). A value > 1 runs
	// that many train→reconstruct→solve-rotation→re-rotate iterations: iter-0 is the
	// same seeded random R + train; each subsequent iter solves the optimal rotation
	// R = V·Uᵀ from a deterministic Jacobi SVD of M = Σ x·x̂ᵀ (see vector/svd.go) and
	// re-trains, lowering PQ reconstruction error on imbalanced/correlated data.
	//
	// MONOTONICITY: more iterations ≈ better reconstruction quality overall, but NOT
	// guaranteed strictly monotone per-iteration — k-means re-seeding at each iter
	// can transiently raise error by 0.1–0.2% before converging. The net trend is
	// downward; the result is deterministic regardless of the intermediate trajectory.
	//
	// CLUSTER DETERMINISM: the refinement is a PURE function of applied state — the
	// slot-ordered training sample, the fixed Seed, and a deterministic SVD (fixed
	// sweep order + fixed sweep count, no random, no map iteration) — so every Raft
	// replica converges to a BIT-IDENTICAL rotation + codebooks. Validated to
	// [0, maxOPQIters] (fail-loud), mirroring the IVFTrainThreshold range check.
	OPQIters int `json:"opq_iters,omitempty"`

	// PQDropVecs (HNSW-PQ only) DROPS the resident float32 vectors after the bulk
	// BuildConcurrent (once the PQ codebooks are trained + every slot encoded), so
	// only the M-byte codes stay in RAM — maximum compression (≈4·D bytes/vec → M
	// bytes/vec). The graph is still LINKED on EXACT floats (the drop fires at the
	// END of the build), so graph quality is unaffected; only the query-time exact
	// rescore is traded away — search becomes ADC-only (the ADC ordering is the
	// result). Get/MMR/Recommend/Discover/reshard then reconstruct an APPROXIMATE
	// vector from the code (vecFor). Requires Quant == QuantPQ (else
	// ErrInvalidPQDropVecs). Default false ⇒ byte/behaviour-identical to today's
	// exact-rescore PQ-HNSW. Mirrors the IVF-PQ-only float-drop pattern. After the
	// drop the index is read-mostly: an incremental Insert returns
	// ErrPQDropVecsReadOnly (no floats to navigate on; rebuild to add).
	PQDropVecs bool `json:"pq_drop_vecs,omitempty"`

	// IVFTrainThreshold is the number of LIVE vectors at which an INCREMENTALLY
	// built QUANTIZED index (one populated via Insert/InsertIfAbsent*/RestoreInsert,
	// not BuildConcurrent) DETERMINISTICALLY auto-trains under the write lock,
	// synchronously, inside the insert that crosses the threshold. It governs two
	// families (a collection is EITHER IVF OR HNSW per IndexType, so the field is
	// unambiguous per-collection):
	//   - IndexIVF: trains the coarse quantizer (and, for IVF-PQ, the residual
	//     codebooks).
	//   - IndexHNSW with Quant == QuantPQ: trains the PQ codebooks + encodes the
	//     backlog (and, if PQDropVecs, drops the resident floats) — the
	//     incremental analogue of the BuildConcurrent train hook, so an
	//     incrementally-built HNSW-PQ index (named/MV, reshard, WAL replay) is no
	//     longer inert.
	// Until the threshold is crossed the index is exact brute-force (correct but
	// uncompressed/unpruned — the untrained-PQ search falls back to exact float).
	// 0 = defaultIVFTrainThreshold. Ignored for non-quantized HNSW and IVF/HNSW
	// built via BuildConcurrent (which trains at build).
	//
	// CLUSTER DETERMINISM: this is a create-time Config field, persisted and
	// replicated like the other index knobs, so every Raft replica sees the
	// IDENTICAL threshold and trains at the IDENTICAL applied insert with the
	// IDENTICAL (slot/index-ordered) vector set and the same Seed ⇒ bit-identical
	// centroids and codebooks across replicas. Negative values are rejected by
	// Validate.
	IVFTrainThreshold int `json:"ivf_train_threshold,omitempty"`

	// IVFDriftRetrain opts an IVF index into DETERMINISTIC auto-retrain-on-DRIFT.
	// After the index auto-trains once (its live count crosses IVFTrainThreshold),
	// the original centroids become stale as the collection grows/shifts. With this
	// set, a TWO-STAGE trigger inside the same deterministic insertLocked apply hook
	// retrains the coarse quantizer (and IVF-PQ residual codebooks) when the live
	// distribution has drifted: stage 1 is an O(1) geometric growth checkpoint
	// (liveCount >= lastTrainCount*IVFDriftGrowthFactor); stage 2, at the checkpoint,
	// compares the mean nearest-centroid distance of the slot-ordered live sample to
	// its train-time reference and retrains iff it grew past IVFDriftFactor (else it
	// DEFERS, bumping the checkpoint without retraining). Like the auto-train trigger
	// it is a PURE function of applied (Raft-replicated) state, so every replica
	// retrains at the IDENTICAL applied insert with the IDENTICAL sample + Seed ⇒
	// bit-identical centroids. Ignored unless IndexType == IndexIVF. Default false ⇒
	// the IVF train-once behaviour is BYTE-IDENTICAL to before (no extra branch taken).
	IVFDriftRetrain bool `json:"ivf_drift_retrain,omitempty"`
	// IVFDriftGrowthFactor is the geometric live-count multiple between stage-1 drift
	// checkpoints (the next checkpoint = lastTrainCount * this). Must be > 1.0; 0 =
	// default 2.0. Ignored unless IVFDriftRetrain is set.
	IVFDriftGrowthFactor float64 `json:"ivf_drift_growth_factor,omitempty"`
	// IVFDriftFactor is how much the mean nearest-centroid distance must exceed its
	// train-time reference (stage 2) before a retrain fires. Must be > 1.0; 0 =
	// default 1.5. Ignored unless IVFDriftRetrain is set.
	IVFDriftFactor float64 `json:"ivf_drift_factor,omitempty"`

	// AnisotropicEta is the ScaNN-style score-aware PQ quantization weight (η ≥ 1).
	// It weights reconstruction error PARALLEL to each datapoint's direction (which
	// dominates the inner-product/MIPS score estimate) by η relative to orthogonal
	// error when training the PQ codebooks. 0 or 1 ⇒ the EXISTING isotropic L2
	// k-means (byte-identical to before — the codebooks are bit-for-bit unchanged);
	// η > 1 (e.g. 4.0) opts into the anisotropic trainer. It changes only the learned
	// codebooks, NOT the codec (encode/ADC/snapshot format), so it is a TRAIN-time
	// knob that rides Config for retrain reproducibility — no wire/snapshot change.
	// The benefit concentrates on DotProduct/Cosine (MIPS) candidate generation;
	// Rostam's exact rescore already fixes final ranking, so this lifts the recall of
	// the retrieved candidate SET at a fixed m. Applies to HNSW-PQ and IVF-PQ.
	//
	// v1 SIMPLIFICATION: the parallel/orthogonal split is taken per-subspace (the
	// sub-vector's own direction) rather than ScaNN's full-vector direction; see
	// kmeansAnisotropic. Validated >= 0 and non-NaN (reject negative/NaN). omitempty
	// keeps a non-anisotropic collection's create-args JSON byte-identical.
	AnisotropicEta float32 `json:"anisotropic_eta,omitempty"`

	// SOAR opts an IVF index into ScaNN-style multi-assignment (Spilled
	// Orthogonality-Amplified Residuals). With SOAR on, each point joins its
	// PRIMARY cell (nearestCentroid) AND a SECONDARY cell chosen to minimize an
	// orthogonality-amplified residual loss — the secondary residual is most
	// ORTHOGONAL to the primary one, so the two encodings are complementary and
	// IVF recall@k RISES at a fixed nprobe (the true neighbor is reachable via
	// either cell). The slot joins BOTH cells' inverted lists; query gather dedups
	// it so it is scored once. Ignored unless IndexType == IndexIVF. Default false
	// ⇒ single assignment, BYTE-IDENTICAL to the pre-SOAR IVF (the existing path is
	// untouched and the snapshot/sidecar formats stay byte-identical when off).
	SOAR bool `json:"soar,omitempty"`
	// SOARLambda is the orthogonality-amplification weight λ in the SOAR secondary-
	// assignment loss: loss(c) = ‖r_c‖² + λ·(r_c · r̂1)² where r_c = x−centroid(c),
	// r̂1 is the unit primary residual. Larger λ penalizes a secondary residual that
	// is PARALLEL to the primary one harder (favouring complementary cells). 0 ⇒
	// the engine default (1.5). Must be >= 0. Ignored unless SOAR is set. omitempty
	// keeps a non-SOAR collection's create-args JSON byte-identical.
	SOARLambda float32 `json:"soar_lambda,omitempty"`

	// FullText, when non-nil, enables a server-side BM25 full-text lane: the
	// reserved $content of each upserted record is tokenized (Config.FullText.
	// Analyzer) into a dedicated bm25Index alongside the dense/sparse lanes, and
	// SearchText/HybridText score it with BM25 over live corpus stats. nil (the
	// default) DISABLES it — no bm25Index is allocated and every dense/sparse/RAG
	// path is byte/behavior-identical. HNSW-only in v1 (an IVF collection ignores
	// it). See FullTextConfig.
	FullText *FullTextConfig `json:"full_text,omitempty"`
}

// NamedVectorParams is the per-named-space index configuration: the per-index
// subset of Config (everything that parameterizes a single HNSW), WITHOUT the
// collection-level fields (Partitions/WAL/tenant/quota/sweep) which are owned by
// the collection, not the individual space. One NamedVectorParams configures one
// sub-index of a NamedCollection.
type NamedVectorParams struct {
	Dim            int    `json:"dim"`                       // dimensionality, required (> 0)
	Metric         Metric `json:"metric,omitempty"`          // distance/similarity; Cosine default (zero value)
	M              int    `json:"m,omitempty"`               // graph degree (0 = default 16)
	EfConstruction int    `json:"ef_construction,omitempty"` // build-time width (0 = default 200)
	EfSearch       int    `json:"ef_search,omitempty"`       // query-time width (0 = default 64)
	Seed           int64  `json:"seed,omitempty"`            // RNG seed for level assignment

	MaxEfSearch   int       `json:"max_ef_search,omitempty"`  // filtered-search ef cap (0 = default 1024)
	Quant         QuantMode `json:"quant,omitempty"`          // vector quantization (QuantNone default)
	RescoreFactor int       `json:"rescore_factor,omitempty"` // quantized rescore multiple (0 = default 3)
	// NOTE: there is intentionally no SQBits/PRQLayers/QuantPQM here (unlike the
	// dense Config), so a named QuantSQ space is locked to 8-bit and a named QuantPRQ
	// space to defaults. This is internally consistent across save+restore (no
	// corruption) — it is a feature limit; named-vector SQ bit-depth / PRQ params are
	// a follow-up to the HNSW dense path (see the design doc's Follow-ups).

	// FilterFirstRelativeBP mirrors the dense Config knob: the opt-in relative
	// selectivity gate (basis points of the live count; 0 = off = byte-identical).
	// Threaded into the per-space Config so the named search filter-first gate honors
	// it. Validated by newIndex (cfg.Validate enforces 0..10000).
	// Honored by named/MV SEARCH only; scroll and order_by remain bound to the
	// absolute cap (a pre-existing limitation).
	FilterFirstRelativeBP int `json:"filter_first_relative_bp,omitempty"`

	// QuantizedBuild navigates/selects on int8 codes during build (requires
	// Quant != QuantNone). Off by default.
	QuantizedBuild bool `json:"quantized_build,omitempty"`

	// IndexType selects this dense space's backing index. IndexHNSW (0, default)
	// keeps the historical per-space graph index — a zero-value (beyond Dim)
	// NamedVectorParams is byte/behaviour-identical to before. IndexIVF builds an
	// IVF-Flat / IVF-PQ index for this space (mirrors the dense Config fields).
	IndexType IndexType `json:"index_type,omitempty"`
	// IVFNlist / IVFNprobe / IVFPQ / IVFPQM / IVFRerank / OPQ mirror the dense
	// Config IVF knobs (see Config). Ignored unless IndexType == IndexIVF.
	IVFNlist  int  `json:"ivf_nlist,omitempty"`
	IVFNprobe int  `json:"ivf_nprobe,omitempty"`
	IVFPQ     bool `json:"ivf_pq,omitempty"`
	IVFPQM    int  `json:"ivf_pq_m,omitempty"`
	IVFRerank bool `json:"ivf_rerank,omitempty"`
	OPQ       bool `json:"opq,omitempty"`
	// OPQIters mirrors the dense Config knob: full-OPQ iterative Procrustes
	// refinement (0 = 1 = the v1 single-random-rotation behavior, byte-identical;
	// > 1 = that many refine iterations). Ignored unless OPQ is set. Validated to
	// [0, maxOPQIters] by ValidateConfig(namedToConfig(...)) at NewNamedCollection.
	OPQIters int `json:"opq_iters,omitempty"`
	// IVFTrainThreshold mirrors the dense Config knob: the live-vector count at
	// which this space's incrementally-built QUANTIZED index (IVF coarse/residual
	// codebooks, or HNSW-PQ codebooks) deterministically auto-trains. 0 =
	// defaultIVFTrainThreshold. Ignored for a non-quantized HNSW space.
	IVFTrainThreshold int `json:"ivf_train_threshold,omitempty"`

	// IVFDriftRetrain / IVFDriftGrowthFactor / IVFDriftFactor mirror the dense Config
	// drift-retrain knobs (see Config): an IVF space opts into deterministic
	// auto-retrain-on-drift. Ignored unless IndexType == IndexIVF. The named create
	// path is JSON-carried (these flow through the spaces config JSON), validated by
	// ValidateConfig(namedToConfig(...)) at NewNamedCollection.
	IVFDriftRetrain      bool    `json:"ivf_drift_retrain,omitempty"`
	IVFDriftGrowthFactor float64 `json:"ivf_drift_growth_factor,omitempty"`
	IVFDriftFactor       float64 `json:"ivf_drift_factor,omitempty"`

	// PQDropVecs mirrors the dense Config knob (HNSW-PQ only, Quant == QuantPQ):
	// once this space's INCREMENTALLY-built HNSW-PQ index auto-trains (its live
	// count crosses IVFTrainThreshold), the resident float32 vectors are DROPPED so
	// only the M-byte codes stay resident (maximum compression; search becomes
	// ADC-only). Honored because the named inner HNSW auto-trains incrementally
	// (the float-drop folds into that auto-train). Requires Quant == QuantPQ (else
	// ErrInvalidPQDropVecs at create, via ValidateConfig(namedToConfig(...))). Default false =>
	// byte/behaviour-identical to today's exact-rescore named PQ-HNSW.
	PQDropVecs bool `json:"pq_drop_vecs,omitempty"`

	// Sparse marks this space as a SPARSE space (backed by a standalone id-keyed
	// inverted index, NOT an HNSW). Default false = a DENSE space (unchanged
	// behavior). A sparse space stores a *SparseVector per point and ignores the
	// dense-only knobs (Dim/Metric/M/Ef…/Quant) — they are not validated for it. The
	// field is omitempty so a dense-only collection's create-args JSON is
	// byte-identical (the Sparse key is absent), preserving the dense wire.
	Sparse bool `json:"sparse,omitempty"`
}

// DefaultConfig returns a Config with HNSW defaults, leaving Dim and Metric
// for the caller to fill in.
func DefaultConfig() Config {
	return Config{
		M:              16,
		EfConstruction: 200,
		EfSearch:       64,
	}
}

// QuantStore selects where the full-precision float32 vectors live when
// quantization is enabled. Codes always stay resident in RAM.
type QuantStore uint8

const (
	// QuantInRAM keeps float32 vectors on the Go heap alongside the codes.
	// Fastest rescore, but uses more memory than no quantization. Default.
	QuantInRAM QuantStore = iota
	// QuantMmap stores float32 vectors in a memory-mapped file so only the
	// int8 codes are resident; the rescore stage pages vectors in on demand.
	// This is the configuration that reduces resident memory.
	QuantMmap
)

// Config validation error sentinels, returned by vector.ValidateConfig. They
// live in the leaf so a moved Config's validation errors are comparable without
// importing the engine; vector re-exports each via a value alias so errors.Is
// identity is preserved and every existing call site is unchanged.
var (
	ErrInvalidDim                   = errors.New("vector: invalid Dim (must be > 0)")
	ErrInvalidMetric                = errors.New("vector: invalid Metric")
	ErrInvalidM                     = errors.New("vector: invalid M (must be > 0 and <= 128)")
	ErrInvalidEf                    = errors.New("vector: invalid Ef (must be > 0)")
	ErrInvalidQuant                 = errors.New("vector: invalid Quant mode")
	ErrInvalidRescoreFactor         = errors.New("vector: invalid RescoreFactor (must be >= 0)")
	ErrInvalidQuantStorage          = errors.New("vector: invalid QuantStorage (QuantMmap requires a quantizer and MmapPath)")
	ErrInvalidGraphMmap             = errors.New("vector: invalid GraphMmapPath (must differ from MmapPath)")
	ErrInvalidPersistent            = errors.New("vector: Persistent requires a quantizer (set Quant to QuantSQ8 or QuantBQ1)")
	ErrInvalidQuantizedBuild        = errors.New("vector: QuantizedBuild requires a quantizer (set Quant to QuantSQ8 or QuantBQ1)")
	ErrInvalidWAL                   = errors.New("vector: WAL and Persistent are mutually exclusive (WAL is heap-backed with snapshot+replay durability; Persistent is mmap instant-restart)")
	ErrInvalidPartitions            = errors.New("vector: Partitions must be >= 0 (0 or 1 = single-partition)")
	ErrInvalidIndexType             = errors.New("vector: invalid IndexType (must be IndexHNSW, IndexIVF, IndexVamana, or IndexGPU)")
	ErrGPUNotCompiled               = errors.New("vector: IndexGPU requires a build with -tags cuda (GPU support not compiled in)")
	ErrInvalidIVFParams             = errors.New("vector: IVFNlist and IVFNprobe must be >= 0")
	ErrInvalidVamanaParams          = errors.New("vector: invalid Vamana params (need R > 0, L >= R, alpha >= 1; 0 = default R=64/L=100/alpha=1.2)")
	ErrInvalidIVFTrainThreshold     = errors.New("vector: IVFTrainThreshold must be >= 0")
	ErrInvalidFilterFirstRelativeBP = errors.New("vector: FilterFirstRelativeBP must be in [0, 10000] (basis points; 0 = off)")
	ErrInvalidIVFDriftFactor        = errors.New("vector: IVFDriftGrowthFactor and IVFDriftFactor must be > 1.0 (0 = default)")
	ErrInvalidIVFPQ                 = errors.New("vector: IVFPQ requires IndexType == IndexIVF")
	ErrInvalidIVFPQM                = errors.New("vector: IVFPQM must be >= 0 and divide Dim evenly")
	ErrInvalidQuantPQM              = errors.New("vector: QuantPQM must be >= 0 and divide Dim evenly")
	ErrInvalidPQNBits               = errors.New("vector: PQNBits must be 0, 4, or 8")
	ErrInvalidPRQLayers             = errors.New("vector: PRQLayers must be >= 0 (0 = default 2)")
	ErrInvalidOPQ                   = errors.New("vector: OPQ requires a PQ mode (set Quant to QuantPQ for HNSW-PQ, or IVFPQ for IVF-PQ)")
	ErrInvalidOPQIters              = errors.New("vector: OPQIters must be in [0, 20] (0 = 1 = the v1 single-rotation behavior)")
	ErrInvalidPQDropVecs            = errors.New("vector: PQDropVecs requires HNSW-PQ (set Quant to QuantPQ)")
	ErrInvalidAnisotropicEta        = errors.New("vector: AnisotropicEta must be >= 0 and not NaN (0 or 1 = isotropic)")
	ErrInvalidSOAR                  = errors.New("vector: SOAR requires IndexType == IndexIVF")
	ErrInvalidSOARLambda            = errors.New("vector: SOARLambda must be >= 0 and not NaN (0 = default 1.5)")
	ErrInvalidFullText              = errors.New("vector: invalid FullText config (HNSW-only; Analyzer must be a registered analyzer name)")
)

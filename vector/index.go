// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"io"
	"math"
	"time"

	"github.com/rostamlabs/rostam/vector/analysis"
)

// Metric, IndexType, and their Cosine/L2/DotProduct and IndexHNSW/IndexIVF/
// IndexVamana/IndexGPU constants now live in the engine-free vtypes leaf package
// and are re-exported via vtypes_aliases.go.

// Vamana defaults (applied when the corresponding Config field is 0).
const (
	defaultVamanaR     = 64  // max out-degree (level-0 slab stride)
	defaultVamanaL     = 100 // build/search beam width
	defaultVamanaAlpha = 1.2 // pass-2 RobustPrune α (pass 1 is always α=1)
)

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
	// Linux-only; other platforms return an mmap-unsupported error.
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

// FullTextConfig now lives in the engine-free vtypes leaf package and is
// re-exported via vtypes_aliases.go.

// NamedConfig is the collection-level config carrier for a named-vector
// collection: the per-space index params (Spaces) plus the collection-level
// single-node durability flags. It is the named analogue of the dense Config's
// {NamedVectors, WAL, WALNoSync} subset — the named family has no Persistent
// (mmap) mode, so WAL is its only single-node disk-durability option.
//
// WAL enables the named single-node WAL lifecycle (apply-then-log under opMu,
// open = restore-snapshot + replay-WAL-tail + open-WAL, Flush = snapshot +
// truncate). It is FORCED OFF on the cluster path (durability is Raft/SnapshotAll
// there, exactly like dense effectiveClusterConfig). Heap-only when WAL is false.
type NamedConfig struct {
	Spaces    map[string]NamedVectorParams `json:"spaces"`
	WAL       bool                         `json:"wal,omitempty"`
	WALNoSync bool                         `json:"wal_no_sync,omitempty"`
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
	// [0, maxOPQIters] by toConfig().Validate() at NewNamedCollection.
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
	// toConfig().Validate() at NewNamedCollection.
	IVFDriftRetrain      bool    `json:"ivf_drift_retrain,omitempty"`
	IVFDriftGrowthFactor float64 `json:"ivf_drift_growth_factor,omitempty"`
	IVFDriftFactor       float64 `json:"ivf_drift_factor,omitempty"`

	// PQDropVecs mirrors the dense Config knob (HNSW-PQ only, Quant == QuantPQ):
	// once this space's INCREMENTALLY-built HNSW-PQ index auto-trains (its live
	// count crosses IVFTrainThreshold), the resident float32 vectors are DROPPED so
	// only the M-byte codes stay resident (maximum compression; search becomes
	// ADC-only). Honored because the named inner HNSW auto-trains incrementally
	// (the float-drop folds into that auto-train). Requires Quant == QuantPQ (else
	// ErrInvalidPQDropVecs at create, via toConfig().Validate()). Default false =>
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

// toConfig builds the Config for this space's sub-HNSW. Named sub-indexes are
// always heap-backed/in-memory in v1 (no per-space WAL/Persistent/mmap — see the
// named-vectors plan non-goals); only the per-index params carry over. M /
// EfConstruction / EfSearch defaults match DefaultConfig so a zero-value
// NamedVectorParams (beyond Dim) yields the standard HNSW.
func (p NamedVectorParams) toConfig() Config {
	c := DefaultConfig()
	c.Dim = p.Dim
	c.Metric = p.Metric
	if p.M > 0 {
		c.M = p.M
	}
	if p.EfConstruction > 0 {
		c.EfConstruction = p.EfConstruction
	}
	if p.EfSearch > 0 {
		c.EfSearch = p.EfSearch
	}
	c.Seed = p.Seed
	c.MaxEfSearch = p.MaxEfSearch
	c.Quant = p.Quant
	c.RescoreFactor = p.RescoreFactor
	c.FilterFirstRelativeBP = p.FilterFirstRelativeBP
	c.QuantizedBuild = p.QuantizedBuild
	// IVF / IVF-PQ knobs (ignored by newHNSW when IndexType == IndexHNSW). The
	// per-space Config is Validated by newIndex (newHNSW/newIVF both call
	// cfg.Validate()), so a bad IVF param fails loud at NewNamedCollection.
	c.IndexType = p.IndexType
	c.IVFNlist = p.IVFNlist
	c.IVFNprobe = p.IVFNprobe
	c.IVFPQ = p.IVFPQ
	c.IVFPQM = p.IVFPQM
	c.IVFRerank = p.IVFRerank
	c.OPQ = p.OPQ
	c.OPQIters = p.OPQIters
	c.IVFTrainThreshold = p.IVFTrainThreshold
	c.IVFDriftRetrain = p.IVFDriftRetrain
	c.IVFDriftGrowthFactor = p.IVFDriftGrowthFactor
	c.IVFDriftFactor = p.IVFDriftFactor
	// PQDropVecs (HNSW-PQ only) folds the float-drop into this space's incremental
	// auto-train. Validated QuantPQ-only by the toConfig().Validate() in
	// validateNamedVectors / newIndex (ErrInvalidPQDropVecs otherwise).
	c.PQDropVecs = p.PQDropVecs
	return c
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

// Validate returns an error if the config is malformed.
func (c Config) Validate() error {
	if c.Dim <= 0 {
		return ErrInvalidDim
	}
	if c.Metric > DotProduct {
		return ErrInvalidMetric
	}
	if c.M <= 0 || c.M > 128 {
		return ErrInvalidM
	}
	if c.EfConstruction <= 0 {
		return ErrInvalidEf
	}
	if c.EfSearch <= 0 {
		return ErrInvalidEf
	}
	// QuantPQ, QuantSQ and QuantPRQ are allowed for the dense/HNSW family (navigate
	// on approximate codes, exact-rescore the over-collected shortlist on the kept
	// floats). Any quant mode above QuantPRQ is unknown. The allowance was extended
	// from QuantSQ to QuantPRQ (product-residual quantization); it admits ONLY the
	// known dense modes, no garbage. (The IVF index does NOT set cfg.Quant ==
	// QuantPQ/QuantSQ/QuantPRQ — it drives PQ via IVFPQ/enableCodes — so this
	// allowance is purely a new dense capability and leaves IVF's validation
	// untouched.)
	if c.Quant > QuantPRQ {
		return ErrInvalidQuant
	}
	// SQBits selects the trained scalar quantizer's bit-depth: 0 (⇒ 8), 4, 6, or 8.
	// 8-bit is one byte/dim (the Task-1 fast path); 4/6 pack levels LSB-first into a
	// bit stream (ceil(dim*bits/8) bytes). Anything else is rejected. Validated only
	// for QuantSQ; a non-SQ config ignores the field.
	if c.Quant == QuantSQ {
		// SQ on the graph engine (HNSW or Vamana): both share the trained-SQ auto-train
		// path (BuildConcurrent / buildVamana train + encode). An IVF index has no SQ
		// auto-train path, so IVF+QuantSQ would pass Validate but never train its
		// quantizer — reject it (only IVF is excluded here).
		if c.IndexType != IndexHNSW && c.IndexType != IndexVamana {
			return ErrInvalidQuant
		}
		switch c.SQBits {
		case 0, 4, 6, 8:
		default:
			return ErrInvalidQuant
		}
	}
	if c.Quant == QuantPQ {
		// PQ on the graph engine (HNSW or Vamana): both train + encode via
		// trainAndEncodePQ (BuildConcurrent / buildVamana) and navigate on ADC + exact
		// rescore — the disk-native Vamana path. An IVF index never carries Quant ==
		// QuantPQ (it uses IVFPQ), so only IVF is excluded here.
		if c.IndexType != IndexHNSW && c.IndexType != IndexVamana {
			return ErrInvalidQuant
		}
		// PQNBits selects the per-subspace code width: 0/8 (256 sub-centroids, the
		// 8-bit codec) or 4 (16 sub-centroids, the LUT16 fast-scan codec). Any other
		// value is rejected.
		switch c.PQNBits {
		case 0, 4, 8:
		default:
			return ErrInvalidPQNBits
		}
		// The configured M (QuantPQM, or defaultPQM(Dim) when 0) must divide Dim evenly.
		m := c.QuantPQM
		if m == 0 {
			m = defaultPQM(c.Dim)
		}
		if m <= 0 || c.Dim%m != 0 {
			return ErrInvalidQuantPQM
		}
	}
	if c.Quant == QuantPRQ {
		// PRQ on the graph engine (HNSW or Vamana): both train + encode via
		// trainAndEncodePRQ and navigate on the summed-LUT ADC + exact rescore. An IVF
		// index never carries Quant == QuantPRQ, so only IVF is excluded here.
		if c.IndexType != IndexHNSW && c.IndexType != IndexVamana {
			return ErrInvalidQuant
		}
		// PRQLayers must be >= 0 (0 ⇒ defaultPRQLayers). Negative is nonsensical.
		if c.PRQLayers < 0 {
			return ErrInvalidPRQLayers
		}
		// Each layer is an m-subquantizer PQ; m (QuantPQM, or defaultPQM(Dim) when 0)
		// must divide Dim evenly, exactly like QuantPQ.
		m := c.QuantPQM
		if m == 0 {
			m = defaultPQM(c.Dim)
		}
		if m <= 0 || c.Dim%m != 0 {
			return ErrInvalidQuantPQM
		}
	}
	if c.QuantPQM < 0 {
		return ErrInvalidQuantPQM
	}
	if c.PRQLayers < 0 {
		return ErrInvalidPRQLayers
	}
	if c.RescoreFactor < 0 {
		return ErrInvalidRescoreFactor
	}
	// Checked before QuantStorage so a Persistent config without a quantizer
	// reports the actionable cause (the store derives QuantStorage from it). IVF is
	// EXEMPT: its instant-restart sidecar mmaps the full-precision float vectors
	// directly (cfg.MmapPath), so an IVF-Flat (QuantNone) collection can be Persistent
	// — the mmap holds floats, not quantized codes. HNSW still requires a quantizer
	// (its mmap storage holds codes).
	if c.Persistent && c.Quant == QuantNone && c.IndexType != IndexIVF {
		return ErrInvalidPersistent
	}
	if c.QuantizedBuild && c.Quant == QuantNone {
		return ErrInvalidQuantizedBuild
	}
	if c.Partitions < 0 {
		return ErrInvalidPartitions
	}
	if c.WAL && c.Persistent {
		return ErrInvalidWAL
	}
	if c.QuantStorage > QuantMmap {
		return ErrInvalidQuantStorage
	}
	if c.QuantStorage == QuantMmap && (c.Quant == QuantNone || c.MmapPath == "") {
		return ErrInvalidQuantStorage
	}
	if c.GraphMmapPath != "" && c.GraphMmapPath == c.MmapPath {
		return ErrInvalidGraphMmap
	}
	if c.IndexType > IndexGPU {
		return ErrInvalidIndexType
	}
	// IndexGPU is only usable in a -tags cuda build. In the DEFAULT (pure-Go,
	// CGO_ENABLED=0) build gpuSupported is false, so fail LOUD here at config time
	// rather than silently falling back to HNSW.
	if c.IndexType == IndexGPU && !gpuSupported {
		return ErrGPUNotCompiled
	}
	if c.IVFNlist < 0 || c.IVFNprobe < 0 {
		return ErrInvalidIVFParams
	}
	// Vamana geometry: R (max out-degree), L (beam width), alpha (pass-2 prune α).
	// Defaults (R=64, L=100, alpha=1.2) apply for 0, so the constraints are checked
	// against the effective values: R>0, L>=R, alpha>=1. Negative R/L and alpha in
	// (0,1) are nonsensical (fail loud, mirroring the IVF/SQ param gates). Validated
	// for every config (cheap) but only meaningful for IndexVamana — a non-Vamana
	// config leaves the fields 0, which resolves to the defaults and passes.
	{
		r, l, alpha := c.VamanaR, c.VamanaL, c.VamanaAlpha
		if r == 0 {
			r = defaultVamanaR
		}
		if l == 0 {
			l = defaultVamanaL
		}
		if alpha == 0 {
			alpha = defaultVamanaAlpha
		}
		if r <= 0 || l < r || alpha < 1 {
			return ErrInvalidVamanaParams
		}
	}
	if c.IVFTrainThreshold < 0 {
		return ErrInvalidIVFTrainThreshold
	}
	// FilterFirstRelativeBP is basis points (0..10000); fail loud on out-of-range so
	// a nonsensical value never silently rides (mirrors the ivf_nlist negative reject).
	// 0 (default) is the OFF state (byte-identical absolute behavior).
	if c.FilterFirstRelativeBP < 0 || c.FilterFirstRelativeBP > 10000 {
		return ErrInvalidFilterFirstRelativeBP
	}
	// Drift-retrain factors must be > 1.0 (a checkpoint/threshold below 1 would
	// retrain on every insert or never). 0 is allowed — it resolves to the engine
	// default (2.0 growth / 1.5 drift) at use. Fail loud on an explicit <= 1.0,
	// mirroring the ivf_nlist negative reject. The factors are validated unconditionally
	// (even when IVFDriftRetrain is off) so a nonsensical value never silently rides.
	if c.IVFDriftGrowthFactor != 0 && c.IVFDriftGrowthFactor <= 1.0 {
		return ErrInvalidIVFDriftFactor
	}
	if c.IVFDriftFactor != 0 && c.IVFDriftFactor <= 1.0 {
		return ErrInvalidIVFDriftFactor
	}
	// IVF + Persistent is supported via the instant-restart mmap sidecar
	// (SavePersist/openPersistIVF): the float vectors live in the mmap'd vecs file
	// and the IVF structures (centroids, lists, slotCell, codes, PQ trailer) are
	// written to the .meta sidecar at Flush. Both vecs-present (IVF-Flat / IVFRerank)
	// and the float-dropped PQ-only state round-trip. No gate needed here; the
	// store wires cfg.MmapPath for a Persistent IVF (effectiveConfig) and the
	// cluster path still forces Persistent=false (snapshot/Raft durable). The MV
	// inner-IVF rejection (multivector_persist.go) is separate and stays.
	// IVF-PQ: residual product quantization is a mode of the IVF index. PQNBits
	// selects the residual code width — 0/8 (256 sub-centroids/subspace, the
	// byte-per-subspace default) or 4 (16 sub-centroids, the nibble-packed LUT16
	// fast-scan codec scored by the in-register kernel on the query path). IVFPQM
	// must divide Dim evenly; 0 is allowed and resolved to defaultPQM(Dim) at
	// construction.
	if (c.IVFPQ || c.IVFRerank) && c.IndexType != IndexIVF {
		return ErrInvalidIVFPQ
	}
	if c.IVFPQM < 0 {
		return ErrInvalidIVFPQM
	}
	if c.IVFPQ && c.IVFPQM > 0 && c.Dim%c.IVFPQM != 0 {
		return ErrInvalidIVFPQM
	}
	// PQNBits on IVF-PQ: only 0/4/8 are valid (mirrors the QuantPQ gate above). Any
	// other value is rejected fail-loud so a nonsensical width never silently rides.
	if c.IVFPQ {
		switch c.PQNBits {
		case 0, 4, 8:
		default:
			return ErrInvalidPQNBits
		}
	}
	// OPQ is a rotation that lives inside the PQ codec, so it is only meaningful
	// with a PQ-family mode active: HNSW-PQ (Quant == QuantPQ), HNSW-PRQ
	// (Quant == QuantPRQ — the rotation is applied once before layer 0), or IVF-PQ
	// (IVFPQ). Fail loud otherwise so an OPQ flag on a non-PQ index is not silently
	// ignored.
	if c.OPQ && c.Quant != QuantPQ && c.Quant != QuantPRQ && !c.IVFPQ {
		return ErrInvalidOPQ
	}
	// OPQIters drives full-OPQ iterative refinement (see Config.OPQIters). Fail
	// loud on an out-of-range value [0, maxOPQIters] so a nonsensical iteration
	// count never silently rides (mirrors the OPQ / IVFTrainThreshold range checks).
	// 0 (= 1, the v1 behavior) and 1 are byte-identical to before. The cap bounds
	// the one-time O(d²·sweeps·iters) training cost on large dim.
	if c.OPQIters < 0 || c.OPQIters > maxOPQIters {
		return ErrInvalidOPQIters
	}
	// AnisotropicEta is the ScaNN score-aware PQ weight (η ≥ 1; 0/1 = isotropic).
	// Reject a negative or NaN value fail-loud so a nonsensical knob never silently
	// rides (mirrors the OPQIters / IVFDriftFactor gates). 0 and 1 are the
	// byte-identical isotropic state; any η > 0 is accepted (η in (0,1) is treated as
	// isotropic by trainCodebooks, which only switches the anisotropic trainer for
	// η > 1). Validated unconditionally (cheap) — it only affects PQ training, so a
	// non-PQ collection leaves it 0 and passes.
	if c.AnisotropicEta < 0 || math.IsNaN(float64(c.AnisotropicEta)) {
		return ErrInvalidAnisotropicEta
	}
	// SOAR is an IVF-only multi-assignment mode (it adds a second inverted-list
	// membership per point). Reject it on a non-IVF index fail-loud so a SOAR flag
	// on HNSW/Vamana is not silently ignored. SOARLambda must be >= 0 and non-NaN
	// (0 resolves to the engine default 1.5 at assignment); validated
	// unconditionally so a nonsensical λ never silently rides even when SOAR is off.
	if c.SOAR && c.IndexType != IndexIVF {
		return ErrInvalidSOAR
	}
	if c.SOARLambda < 0 || math.IsNaN(float64(c.SOARLambda)) {
		return ErrInvalidSOARLambda
	}
	// PQDropVecs is the HNSW-PQ maximum-compression mode (drop resident floats →
	// ADC-only search). It is only meaningful when the dense HNSW PQ codec is
	// active (Quant == QuantPQ); fail loud so a PQDropVecs flag on a non-PQ index
	// (or on the separate IVFPQ family) is not silently ignored. Mirrors the OPQ
	// gate above.
	if c.PQDropVecs && c.Quant != QuantPQ {
		return ErrInvalidPQDropVecs
	}
	// FullText (BM25) is HNSW-only in v1 and must name a registered analyzer.
	// nil (default) is the byte-identical disabled state. The analyzer is resolved
	// at construction; validate the name here so a bad config fails loud at create.
	if c.FullText != nil {
		if c.IndexType != IndexHNSW {
			return ErrInvalidFullText
		}
		if _, err := analysis.ByName(c.FullText.Analyzer); err != nil {
			return ErrInvalidFullText
		}
	}
	return nil
}

// CASCond is the optimistic-concurrency precondition threaded into the dense
// mutators. It is the "expected version" half of a compare-and-set:
//
//   - Has == false (the zero value): NO precondition — an unconditional write.
//     The version STILL bumps on success. Every existing non-CAS caller passes
//     the zero value, so its behavior is unchanged (today's write + a bump).
//   - Has == true: the mutation applies ONLY when the point's current version
//     (0 if absent) equals Expected; otherwise it returns ErrVersionConflict
//     with no mutation and no bump. Expected == 0 means "expect the point to be
//     absent/new" — an insert-if-absent CAS (etcd/Qdrant-style).
//
// The check + bump are atomic under the engine write lock (the FSM-Apply
// serialization point), so the result is deterministic across Raft replicas.
type CASCond struct {
	Expected uint64 // the version the caller expects the point to currently have
	Has      bool   // whether to enforce the precondition at all
}

// check evaluates the precondition against the point's current version (0 if
// absent). It returns ErrVersionConflict when Has is set and current !=
// Expected; otherwise nil (no precondition, or a match). Callers MUST hold the
// engine write lock so the read-of-current and the subsequent apply+bump are one
// atomic step.
func (c CASCond) check(current uint64) error {
	if c.Has && current != c.Expected {
		return ErrVersionConflict
	}
	return nil
}

// Result is one search result: the id of the vector and its distance from
// the query under the index's metric. Score is the fusion score for hybrid
// search (higher = more relevant); it is 0 for plain dense KNN results.
type Result struct {
	ID       uint64  `json:"id"`
	Distance float32 `json:"distance"`
	Score    float32 `json:"score"`
}

// Stats is a snapshot of an index's runtime statistics.
type Stats struct {
	Size            int
	Tombstoned      int
	SearchOps       uint64
	InsertOps       uint64
	AvgSearchDepth  float32
	Expired         uint64           // count of entries that aged out via TTL
	QuotaRejects    uint64           // cumulative inserts rejected by quota or rate limit
	FilterRejects   uint64           // cumulative candidates rejected by an active search filter
	FilterGates     uint64           // cumulative filtered searches that armed the payload-index bitset admission gate
	ComplementGates uint64           // the subset of FilterGates armed from the filter's REJECTION side (high-pass-rate ranges)
	ColumnGates     uint64           // the subset of FilterGates answered by the numeric column sidecar
	ColumnDrops     uint64           // times an insert reclaimed the column sidecar to stay inside MaxBytes
	SparseVectors   int              // live slots carrying a sparse vector
	SearchLatency   LatencyHistogram // per-search wall time, log-scale buckets
	InsertLatency   LatencyHistogram // per-insert wall time, log-scale buckets
}

// HybridOpts configures a hybrid (dense + sparse) search. The zero value is
// valid: RRF fusion, rrfK=60, dense/sparse candidate pools sized from k.
type HybridOpts struct {
	Filter  Filter       // metadata predicate; zero = no filter
	Method  FusionMethod // FusionRRF (default) or FusionWeighted
	Alpha   float64      // weighted only: dense weight in [0,1] (0 → treated as 0.5 default by HybridSearch)
	RRFK    int          // RRF constant; 0 = default 60
	DenseK  int          // dense-lane candidate pool; 0 = max(k, 50)
	SparseK int          // sparse-lane candidate pool; 0 = max(k, 50)
}

// VectorIndex is the abstraction over any nearest-neighbor index. It is the full
// contract Collection requires of its backing index: every Collection method
// delegates to one of these. HNSW is the only implementation today; a future
// IVF-Flat implementation must satisfy this same interface. Several methods are
// unexported — that is fine because all implementations live in package vector.
type VectorIndex interface {
	// --- Writes ---
	// Insert upserts id. cas is the optimistic-CAS precondition (CASCond{} = no
	// precondition, an unconditional write that still bumps the version). On
	// success it returns the resulting version (1 for a fresh insert, current+1
	// for an in-place upsert); on a CAS mismatch it returns ErrVersionConflict
	// with no mutation.
	// keyTTLMs is an OPTIONAL per-key payload TTL map (key -> RELATIVE ms); the
	// engine computes the ABSOLUTE deadline now+ttl at insert (mirroring
	// set_payload) and stores it in the slot's keyExpires. Empty/nil = no per-key
	// TTL (byte-identical / zero-overhead).
	// It returns the resulting version AND the resulting ABSOLUTE per-key deadline
	// map (key -> now+ttl) the engine computed, so the Collection layer WAL-logs
	// exactly what was applied (time-stable). keyExpires is nil when no per-key TTL.
	Insert(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyTTLMs map[string]int64, cas CASCond) (version uint64, keyExpires map[string]uint64, err error)
	// InsertAt is Insert with every TTL deadline computation AND liveness check
	// (CAS/reclaim) judged against the EXPLICIT leader-stamped clock nowMs (unix
	// millis) instead of the wall clock, so a replicated apply stamps byte-identical
	// absolute point/per-key deadlines and agrees on liveness on every replica (#4
	// vector TTL determinism, mirroring cache.PutAt). Insert is byte-identical to
	// before this method existed.
	InsertAt(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (version uint64, keyExpires map[string]uint64, err error)
	// RestoreInsert inserts id with an EXACT version set verbatim (NOT bumped) —
	// the version-preserving primitive used by WAL replay and reshard copy so a
	// restored/copied point keeps its original version. No CAS, no rate limit.
	// keyExpires is the ABSOLUTE per-key deadline map restored VERBATIM (NOT
	// recomputed now+ttl) so pending key deadlines survive a crash time-stable.
	RestoreInsert(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64) error
	// RestoreInsertAt is RestoreInsert stamping the POINT ttl deadline and judging
	// reclaim liveness against the EXPLICIT leader-stamped clock nowMs (keyExpires is
	// still installed VERBATIM) — the replicated version-preserving insert path (#4
	// vector TTL determinism).
	RestoreInsertAt(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64, nowMs int64) error
	// InsertIfAbsent inserts only if id is not currently live (returns
	// inserted=false on a no-op), atomically (one critical section, no
	// check-then-act gap). The online-copy primitive that, with Raft
	// serialization, closes Race A (value clobber).
	InsertIfAbsent(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector) (bool, error)
	// InsertIfAbsentVersion is InsertIfAbsent that, on a real insert, sets the
	// point's version VERBATIM to `version` (NOT 1) — the version-PRESERVING
	// online-copy primitive used by the reshard copy pass so a copied point keeps
	// its per-point CAS version while still never clobbering a concurrent live
	// dual-write. version==0 behaves exactly like InsertIfAbsent (version 1 on a
	// fresh insert). keyExpires is the ABSOLUTE per-key payload deadline map set
	// VERBATIM (NOT recomputed now+ttl) on a real insert so the online reshard copy
	// keeps the point's original key deadlines time-stable; version==0 && nil
	// keyExpires reproduces today's InsertIfAbsent behavior byte-identically.
	InsertIfAbsentVersion(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64) (bool, error)
	// InsertIfAbsentVersionAt is InsertIfAbsentVersion whose liveness OUTCOME (insert
	// vs no-op), reclaim, and point-TTL deadline are judged against the EXPLICIT
	// leader-stamped clock nowMs, so skewed replicas agree on whether an expired id
	// resurrects and stamp identical deadlines (#4 vector TTL determinism).
	InsertIfAbsentVersionAt(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64, nowMs int64) (bool, error)
	// Delete removes id. cas is the optimistic-CAS precondition; on a mismatch it
	// returns ErrVersionConflict and removed=false with no mutation. A no-CAS
	// Delete (CASCond{}) behaves exactly as before (removed reports whether id was
	// live). The version is dropped with the point (a later reinsert starts at 1).
	Delete(id uint64, cas CASCond) (removed bool, err error)
	// DeleteAt is Delete judging the dead-slot liveness gate against the EXPLICIT
	// leader-stamped clock nowMs, so replicas agree on already-dead vs live and their
	// tombstone sets stay identical (#4 vector TTL determinism). Delete is
	// byte-identical to before.
	DeleteAt(id uint64, cas CASCond, nowMs int64) (removed bool, err error)
	// BuildConcurrent bulk-loads an empty index from ids/vecs using `workers`
	// goroutines (0 = GOMAXPROCS).
	BuildConcurrent(ids []uint64, vecs [][]float32, workers int) error
	// BuildConcurrentMeta is BuildConcurrent carrying an OPTIONAL per-point
	// payload: metas is nil/empty (identical to BuildConcurrent) or exactly
	// len(ids) long with a nil entry per payload-less point. It is on the
	// INTERFACE rather than behind a type assertion on purpose — an optional
	// interface would let a family that never implemented it silently drop the
	// payloads of a bulk load, which is the exact failure this path exists to
	// avoid.
	BuildConcurrentMeta(ids []uint64, vecs [][]float32, metas []Metadata, workers int) error

	// --- Liveness / point reads ---
	// Exists reports whether id is currently live, using the same liveness
	// definition as search admission (tombstoned + TTL-expired = not live). O(1)
	// idMap probe; the resurrection-guard liveness check (Race B).
	Exists(id uint64) bool
	// Get returns the live record plus the point's current version (>= 1 for a
	// live point; an absent/dead point returns ok=false and version 0).
	Get(id uint64) (vec []float32, meta Metadata, ttl time.Duration, sparse *SparseVector, version uint64, ok bool)
	// GetInto is Get that APPENDS the stored vector into the caller-owned scratch
	// buffer dst (passed as dst[:0]) instead of allocating a fresh []float32 each
	// call — the dense-read analogue of cache.GetInto. The returned vec aliases
	// dst's backing array when dst has the capacity, so a hot-loop caller reusing
	// one buffer incurs ZERO allocations for the vector copy. Everything else (meta,
	// ttl, sparse, version, ok) is identical to Get, including the same liveness
	// gate and torn-read safety — only the destination of the vector copy changes.
	// On a miss (absent/tombstoned/expired) it returns vec=nil like Get. NOTE: when
	// the storage path must RECONSTRUCT the vector (e.g. a PQDropVecs index where
	// vecFor allocates), that inner allocation is unavoidable and GetInto cannot
	// elide it — the zero-alloc guarantee holds only when vecFor aliases the arena.
	GetInto(dst []float32, id uint64) (vec []float32, meta Metadata, ttl time.Duration, sparse *SparseVector, version uint64, ok bool)

	// --- Search ---
	Search(query []float32, k int) ([]Result, error)
	SearchFiltered(query []float32, k int, filter Filter) ([]Result, error)
	SearchInto(dst []Result, query []float32, k int, filter Filter) ([]Result, error)
	// SearchFilteredWith is SearchInto with an OPTIONAL external metadata provider
	// (the named/MV hook). metaOf == nil is byte-identical to SearchInto. When
	// metaOf != nil the filter predicate is evaluated against the EXTERNAL per-point
	// payload via metaOf(id) (the sub-arena carries no metadata) — predicate-eval
	// only, the payload-index/filter-first planner is bypassed.
	SearchFilteredWith(dst []Result, query []float32, k int, filter Filter, metaOf func(id uint64) Metadata) ([]Result, error)
	// filterFirstByID computes exact top-k over an index-narrowed candidate ID set
	// whose payload lives in an EXTERNAL map (the named/MV families: the sub-arena
	// carries the VECTORS but no metadata). For each candidate id it resolves the
	// vector via the sub-arena, applies the tombstone/TTL liveness gate AND the
	// external predicate RE-CHECK (pred evaluated against metaOf(id)), scores under
	// the active metric, and returns the k closest. Takes the index's own read lock.
	filterFirstByID(dst []Result, cands []uint64, query []float32, k int, pred Predicate, metaOf func(id uint64) Metadata) []Result
	// preferFilterFirst is the cost-based planner decision: should an index-narrowed
	// filtered search brute-force the ncand candidates (exact) rather than traverse
	// the index? It compares estimated distance-computation costs.
	preferFilterFirst(ncand, k int) bool
	// filterFirstCrossover returns the largest ncand in [0, limit] for which
	// preferFilterFirst(ncand, k) is true — i.e. the biggest candidate set the
	// planner could still choose filter-first for. A caller that materializes a
	// candidate superset passes it as the early-abort ceiling, so a set that grows
	// past the cap is abandoned mid-build instead of being finished and discarded.
	// Each index computes it from ITS OWN cost model.
	filterFirstCrossover(k, limit int) int
	// filterFirstThreshold is the maximum candidate-set size for which the planner
	// uses brute-force filter-first search instead of index traversal.
	filterFirstThreshold() int
	// effectiveFilterFirstLimit folds the OPT-IN relative selectivity gate
	// (Config.FilterFirstRelativeBP) into filterFirstThreshold for the given live
	// count. When FilterFirstRelativeBP == 0 (default) it returns
	// filterFirstThreshold() EXACTLY (byte-identical); when > 0 it may raise the
	// limit up to a hard cap so a relatively-selective filter can use filter-first
	// beyond the absolute cap. The named/MV search gates call this on the inner
	// index with the inner arena's live count.
	effectiveFilterFirstLimit(liveCount int) int
	// Dim returns the configured vector dimensionality (cfg.Dim).
	Dim() int
	// vecsForIDs returns, for each live id present in this index, a COPY of its
	// float vector (exact arena floats when present, reconstructed from the PQ code
	// + coarse centroid when the floats were dropped). Absent/dead ids are omitted.
	// Takes the index's own read lock internally and never aliases arena storage.
	vecsForIDs(ids []uint64) map[uint64][]float32
	// withVecAccess holds the index's read lock for the duration of fn and passes a
	// getter that returns each live id's float vector as an arena VIEW (no per-id
	// copy/allocation — the MaxSim hot path resolves thousands of token vectors per
	// query). The view is valid ONLY for the duration of fn (the lock is released on
	// return). PQ-dropped floats are reconstructed into a fresh slice (rare). Absent
	// ids report ok=false. The allocation-free analogue of vecsForIDs.
	withVecAccess(fn func(get func(id uint64) ([]float32, bool)))
	HybridSearch(dense []float32, sparse SparseVector, k int, opts HybridOpts) ([]Result, error)
	HybridLanes(dense []float32, sparse SparseVector, k int, opts HybridOpts) ([]Result, []Result, error)
	SearchMMR(query []float32, k int, opts MMROpts) ([]Result, error)
	Recommend(k int, opts RecommendOpts) ([]Result, error)
	Discover(k int, opts DiscoverOpts) ([]Result, error)
	// DiscoverVecs is the RESOLVED-vectors discovery path (the Query API leaf
	// form): the target + context-pair example VECTORS are supplied directly (the
	// coordinator resolved the ids and embedded them), so no per-call id resolution
	// runs. It executes the IDENTICAL seed + per-candidate context-pair scorer as
	// Discover (the equivalence oracle), score-descending.
	DiscoverVecs(k int, opts DiscoverVecsOpts) ([]Result, error)
	// RecommendVecs is the BEST_SCORE recommend path (the Query API leaf form): the
	// positive/negative example VECTORS are supplied directly (the coordinator
	// resolved the ids and embedded them). It seeds a candidate pool from the
	// positives' centroid, scores each candidate by the BEST_SCORE merge (bestScore —
	// max-sim-to-positive vs max-sim-to-negative), and sorts score-descending. The
	// recommend analogue of DiscoverVecs and the equivalence oracle for BEST_SCORE.
	RecommendVecs(k int, opts RecommendVecsOpts) ([]Result, error)
	SearchGroups(query []float32, k int, opts GroupOpts) ([]Group, error)
	GroupCandidates(query []float32, opts GroupOpts) ([]Document, error)

	// --- Payload mutations (return the resulting payload + per-key deadlines
	// for the WAL, mirroring how Collection logs them, PLUS the resulting version
	// after the bump). cas is the optimistic-CAS precondition (CASCond{} = no
	// precondition); a mismatch returns ErrVersionConflict with no mutation. ---
	SetPayload(id uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond) (Metadata, map[string]uint64, uint64, error)
	OverwritePayload(id uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond) (Metadata, map[string]uint64, uint64, error)
	DeletePayloadKeys(id uint64, keys []string, cas CASCond) (Metadata, map[string]uint64, uint64, error)
	ClearPayload(id uint64, cas CASCond) (Metadata, map[string]uint64, uint64, error)
	// The ...At variants judge the per-key deadline computation (Set/Overwrite) AND
	// the dead-point liveness gate against the EXPLICIT leader-stamped clock nowMs,
	// so a replicated payload mutation stamps byte-identical per-key deadlines and
	// agrees on liveness on every replica (#4 vector TTL determinism). The non-At
	// forms use the wall clock and are byte-identical to before.
	SetPayloadAt(id uint64, patch Metadata, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (Metadata, map[string]uint64, uint64, error)
	OverwritePayloadAt(id uint64, meta Metadata, keyTTLMs map[string]int64, cas CASCond, nowMs int64) (Metadata, map[string]uint64, uint64, error)
	DeletePayloadKeysAt(id uint64, keys []string, cas CASCond, nowMs int64) (Metadata, map[string]uint64, uint64, error)
	ClearPayloadAt(id uint64, cas CASCond, nowMs int64) (Metadata, map[string]uint64, uint64, error)
	// RestorePayload restores a logged payload + per-key deadlines AND the exact
	// version verbatim (NOT bumped) — the WAL-replay primitive.
	RestorePayload(id uint64, meta Metadata, keyExpires map[string]uint64, version uint64) error

	// --- Doc enrichment / scan / scroll (RAG + resplit + fan-out primitives) ---
	fetchDocs(results []Result) []Document
	matchingIDs(filter Filter, pred Predicate) ([]uint64, error)
	// matchingIDsAt is matchingIDs judging the tombstone/TTL admission gate against
	// the EXPLICIT leader-stamped clock nowMs, so a replicated delete-by-filter
	// selects the SAME id set on every replica (#4 vector TTL determinism).
	matchingIDsAt(filter Filter, pred Predicate, nowMs uint64) ([]uint64, error)
	scanVectors() []ScanRecord
	scrollDocs(filter Filter, limit int) ([]Document, error)
	// scrollPage walks the live id set and returns up to `limit` documents.
	//
	//   - order == nil: today's deterministic id-ASCENDING resume-after-id scroll
	//     (the cached scrollSnap path, byte-identical / zero-overhead). afterKey is
	//     ignored.
	//   - order != nil: order_by scroll — the admitted live set is sorted per call by
	//     the (value, id) total order for order.Key/Desc (see vector.OrderLess), ids
	//     whose order field is missing/non-numeric are EXCLUDED, and the page resumes
	//     STRICTLY PAST the cursor (afterKey, afterID) — or past order.StartFrom on
	//     page 1. Returned docs carry the order field in Metadata so the coordinator
	//     can read the last doc's order value for the next (v2) cursor.
	//
	// filter is the ORIGINAL (uncompiled) filter; pred is its compiled form. When a
	// filter is present (pred != nil) AND order == nil (the id-ascending path), the
	// implementation MAY consult the payload index for a candidate SUPERSET and walk
	// only those ids (filter-first narrowing) instead of the full id snapshot — the
	// SAME predicate recheck still runs per candidate, so the page (ids/order/cursor)
	// is identical to the predicate-eval walk. Falls back to the full-snapshot walk
	// when the filter is not index-narrowable or not selective. filter is unused on
	// the order_by path (it stays predicate-eval) and on the no-filter path.
	scrollPage(filter Filter, pred Predicate, metaOf metaProvider, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool)

	// --- Maintenance ---
	sweepOnce() int
	Reclaim() int

	// --- Persistence ---
	Snapshot(w io.Writer) error
	Restore(r io.Reader) error
	SavePersist(metaPath string) error

	// --- Observability / lifecycle ---
	Stats() Stats
	Close() error

	// SetNowFunc overrides the wall-clock source (unix millis) the index's non-apply
	// expiry sites consult (sweeper + read/query filter + the wall-clock branch of
	// the write paths). nil restores the real clock. TEST/advanced seam mirroring
	// cache.Cache.SetNowFunc; production never calls it, so the default is
	// byte-identical to time.Now. It does NOT affect the stamped apply path (InsertAt
	// et al. take an explicit stamp).
	SetNowFunc(fn func() int64)
}

// Error sentinels returned by VectorIndex implementations.
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
	ErrInvalidIVFPersistent         = errors.New("vector: IVF index is snapshot-only in v1 and cannot be combined with Persistent (mmap instant-restart); use snapshot/WAL durability instead")
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
	ErrPQDropVecsReadOnly           = errors.New("vector: index is read-only after PQDropVecs dropped the float vectors (rebuild to add points)")
	ErrInvalidFullText              = errors.New("vector: invalid FullText config (HNSW-only; Analyzer must be a registered analyzer name)")
	ErrFullTextDisabled             = errors.New("vector: full-text search requires a collection created with FullText enabled")
	ErrEmptyFilter                  = errors.New("vector: DeleteByFilter requires a non-empty filter (refusing to delete the whole collection)")
	ErrEmptyGroupBy                 = errors.New("vector: SearchGroups requires a non-empty GroupBy field")
	ErrEmptyDocument                = errors.New("vector: multi-vector Add requires at least one token vector")

	// Named-vector (NamedCollection) sentinels — fail-loud on a malformed config
	// or a request that names a space that does not exist.
	ErrEmptyNamedVectors  = errors.New("vector: named collection requires at least one named vector space")
	ErrEmptyVectorName    = errors.New("vector: named vector name must be non-empty")
	ErrReservedVectorName = errors.New("vector: named vector name must not contain reserved characters ('#', '@', '/')")
	ErrUnknownVectorName  = errors.New("vector: unknown named vector space")
	// ErrSpaceModalityMismatch is returned when a request supplies the wrong value
	// kind for a named space's modality: a sparse value for a DENSE space, or a
	// dense value for a SPARSE space. Fail loud (the wire fail-loud mapping follows).
	ErrSpaceModalityMismatch = errors.New("vector: value modality does not match named space (dense vs sparse)")
	ErrNoRecommendExamples   = errors.New("vector: recommend requires at least one positive example")
	// ErrRecommendBestScoreL2Negatives rejects a BEST_SCORE recommend that combines
	// the L2 (Euclidean) metric with negative examples. BEST_SCORE's piecewise merge
	// (max_pos > max_neg ? max_pos : -max_neg) is a SCORE-STYLE strategy: it assumes
	// sign-meaningful, higher-is-better similarities, which hold for the inner-product
	// metrics (Cosine: 1-dist; DotProduct: -dist). For L2 the similarity is -squared-L2,
	// which is ALWAYS <= 0, so the -max_neg sign-flip in the else branch produces a
	// POSITIVE score that outranks every (non-positive) positive-dominated candidate —
	// inverting the ranking (a candidate ON a positive ranks last; a candidate NEAR a
	// negative ranks first). Fail loud rather than return an inverted ranking. L2 +
	// BEST_SCORE with NO negatives is well-defined (score = nearest-positive similarity,
	// monotonic) and stays allowed; use Cosine/DotProduct for L2-shaped data with
	// negatives, or AVERAGE_VECTOR (which steers via mean(pos)-mean(neg) and is sound for L2).
	ErrRecommendBestScoreL2Negatives = errors.New("vector: best_score recommend with negative examples requires a score-style metric (cosine/dot-product), not L2")
	ErrNoContextPairs                = errors.New("vector: discover requires at least one context pair")
	ErrReserveNonEmpty               = errors.New("vector: Reserve requires an empty arena")
	ErrBuildNonEmpty                 = errors.New("vector: BuildConcurrent requires an empty index")
	ErrBuildLenMismatch              = errors.New("vector: BuildConcurrent ids and vecs length mismatch")
	ErrBuildMetaLenMismatch          = errors.New("vector: BuildConcurrentMeta payloads must be empty or one per id")
	ErrDimMismatch                   = errors.New("vector: vector length does not match index Dim")
	ErrDuplicateID                   = errors.New("vector: id already present (delete first)")
	ErrIDNotFound                    = errors.New("vector: id not found")

	// ErrVersionConflict is returned by an optimistic-CAS mutation whose
	// CASCond.Expected does not match the point's current version (0 if absent).
	// On a conflict the mutation is NOT applied and the version is NOT bumped, so
	// it is safe to retry after re-reading the current version (Get returns it).
	ErrVersionConflict = errors.New("vector: version conflict (optimistic CAS expected_version mismatch)")
	ErrEmptyIndex      = errors.New("vector: index is empty")
	ErrSnapshotFormat  = errors.New("vector: snapshot has invalid format or version")
	ErrNotImplemented  = errors.New("vector: not implemented in this version")

	// ErrCollectionFull is returned when an insert would exceed the collection's
	// MaxVectors or MaxBytes quota.
	ErrCollectionFull = errors.New("vector: collection full (quota exceeded)")

	// ErrCollectionRateLimited is returned when MaxInsertsPerSecond is configured
	// and the token bucket is empty. Callers can retry after the bucket refills.
	ErrCollectionRateLimited = errors.New("vector: collection insert rate limited")
)

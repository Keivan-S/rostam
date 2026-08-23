// SPDX-License-Identifier: Apache-2.0

package vtypes

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

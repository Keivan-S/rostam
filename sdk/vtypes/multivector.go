// SPDX-License-Identifier: Apache-2.0

package vtypes

// MultiVectorConfig configures a MultiVectorIndex. Only the token dimensionality
// is required; the rest take the standard HNSW defaults. The metric is always
// Cosine — MaxSim is defined over normalized vectors.
type MultiVectorConfig struct {
	Dim            int   // token vector dimensionality, required
	M              int   // graph degree (0 = default 16)
	EfConstruction int   // build width (0 = default 200)
	EfSearch       int   // query width (0 = default 64)
	Seed           int64 // RNG seed for level assignment

	// Quant selects token-vector quantization for the first-stage graph
	// (QuantNone | QuantSQ8 | QuantBQ1). A Persistent index defaults Quant to
	// QuantSQ8 when unset.
	Quant         QuantMode
	RescoreFactor int // over-fetch multiple for quantized first-stage rescore (0 = default)

	// FilterFirstRelativeBP mirrors the dense Config knob: the opt-in relative
	// selectivity gate (basis points of the live DOCUMENT count; 0 = off).
	FilterFirstRelativeBP int `json:"filter_first_relative_bp,omitempty"`

	// IndexType selects the inner token index's backing engine. IndexHNSW (0, the
	// default) keeps the historical per-token graph index. IndexIVF builds an
	// IVF-Flat / IVF-PQ inner index.
	IndexType IndexType `json:"index_type,omitempty"`
	// IVFNlist / IVFNprobe / IVFPQ / IVFPQM / IVFRerank / OPQ mirror the dense
	// Config IVF knobs. Ignored unless IndexType == IndexIVF.
	IVFNlist  int  `json:"ivf_nlist,omitempty"`
	IVFNprobe int  `json:"ivf_nprobe,omitempty"`
	IVFPQ     bool `json:"ivf_pq,omitempty"`
	IVFPQM    int  `json:"ivf_pq_m,omitempty"`
	IVFRerank bool `json:"ivf_rerank,omitempty"`
	OPQ       bool `json:"opq,omitempty"`
	// OPQIters mirrors the dense Config knob. Ignored unless OPQ is set.
	OPQIters int `json:"opq_iters,omitempty"`
	// IVFTrainThreshold mirrors the dense Config knob: the live token count at which
	// the incrementally-built QUANTIZED inner index deterministically auto-trains.
	IVFTrainThreshold int `json:"ivf_train_threshold,omitempty"`

	// IVFDriftRetrain / IVFDriftGrowthFactor / IVFDriftFactor mirror the dense Config
	// drift-retrain knobs. Ignored unless IndexType == IndexIVF.
	IVFDriftRetrain      bool    `json:"ivf_drift_retrain,omitempty"`
	IVFDriftGrowthFactor float64 `json:"ivf_drift_growth_factor,omitempty"`
	IVFDriftFactor       float64 `json:"ivf_drift_factor,omitempty"`

	// PQDropVecs mirrors the dense Config knob (HNSW-PQ only, Quant == QuantPQ).
	PQDropVecs bool `json:"pq_drop_vecs,omitempty"`

	// Persistent enables the mmap-backed, instant-restart mode. Persistent requires
	// Quant != QuantNone (mmap needs codes).
	Persistent    bool
	MmapPath      string // store-managed: token float32 vectors (mmap)
	GraphMmapPath string // store-managed: level-0 graph slab (mmap)

	// WAL enables the single-node WAL HEAP-checkpoint durability mode. Mutually
	// exclusive with Persistent. WALNoSync skips the per-op fsync.
	WAL       bool
	WALNoSync bool

	// SuppressSweep disables the background per-key-TTL sweeper. Set by the
	// persistent-cluster policy.
	SuppressSweep bool

	// Partitions is the number of partitions the collection is split into on the
	// clustered backend. 0 or 1 = single-partition. Immutable after creation.
	Partitions int
}

// MultiResult is one scored document from a multi-vector search. Score is the
// MaxSim relevance (higher = better), not a distance.
type MultiResult struct {
	ID       uint64   `json:"id"`
	Score    float32  `json:"score"`
	Metadata Metadata `json:"metadata,omitempty"`
}

// MultiScanRecord is a complete, live multi-vector document exported by
// ScanDocuments: everything an offline resplit needs to re-insert it into a
// re-hashed generation.
type MultiScanRecord struct {
	ID       uint64
	Tokens   [][]float32 // one row per token vector; owned copies of the stored (normalized) vectors
	Metadata Metadata    // owned copy; nil if none
	Version  uint64      // per-document CAS version (0 if absent)
	// KeyExpires is the document's per-key payload TTL map (payload key -> ABSOLUTE
	// unix-millis deadline), an OWNED clone. nil/empty when the document has no
	// per-key TTL.
	KeyExpires map[string]uint64
	// Sparse is the document's OPTIONAL doc-level sparse vector, an OWNED clone. nil
	// when the document has no sparse vector.
	Sparse *SparseVector
}

// SPDX-License-Identifier: Apache-2.0

package vtypes

import "errors"

// --- Unified Query API surface (engine-free data types) ---
//
// This is the proto-FREE Go-struct surface for the unified vector_query op. The
// ops layer converts a marshaled pb.QuerySpec into these structs; the engine
// consumes them. A query carries a ROOT leaf plus N single-level PREFETCH leaves,
// combined at the root by either FUSION (RRF/Weighted/DBSF over the prefetch
// lanes) or RERANK (the root leaf re-scores the union of the prefetch candidates).

// LeafKind tags a QueryLeaf as a dense or a sparse query node.
type LeafKind uint8

const (
	// LeafDense is a dense (float vector) query node.
	LeafDense LeafKind = iota
	// LeafSparse is a sparse (indices/values) query node.
	LeafSparse
	// LeafMVMaxSim is a multi-vector late-interaction (MaxSim) query node for the
	// MV family, executed over the token query matrix (Tokens), score-descending.
	LeafMVMaxSim
	// LeafRecommend is a RECOMMEND (Qdrant-parity) query node: search by positive/
	// negative EXAMPLE point-ids (Positive/Negative). Under AVERAGE_VECTOR it is
	// rewritten into a LeafDense leaf by a coordinator pre-pass; under BEST_SCORE it
	// stays a real execLeaf.
	LeafRecommend
	// LeafDiscover is a DISCOVER (Qdrant-parity) query node: a target + context
	// positive/negative example PAIRS guide a CUSTOM per-candidate scorer.
	LeafDiscover
)

// QueryMode selects how the root combines the prefetch lanes.
type QueryMode uint8

const (
	// ModeFusion fuses the N prefetch lanes via the configured fusion method.
	ModeFusion QueryMode = iota
	// ModeRerank re-scores the UNION of the prefetch candidate ids by the ROOT
	// leaf, restricted to that candidate set, returning the reranked top-k.
	ModeRerank
)

// RecommendStrategy selects how a recommend query scores candidates. It mirrors
// Qdrant's recommend strategy enum: AverageVector derives ONE query vector
// mean(positive) - mean(negative) (the default — a dense rewrite); BestScore
// scores each candidate by a custom per-candidate max-similarity merge over the
// positive/negative example vectors (a real scorer, like Discover). The zero
// value is RecommendAverageVector so an un-strategied recommend leaf is
// byte/behaviour-identical to the original AVERAGE_VECTOR path.
type RecommendStrategy uint8

const (
	// RecommendAverageVector (default, 0) is the original recommend: derive
	// normalize(mean(positive) - mean(negative)) → a dense query.
	RecommendAverageVector RecommendStrategy = iota
	// RecommendBestScore scores each candidate by merge(max_sim_to_any_positive,
	// max_sim_to_any_negative) — a custom per-candidate scorer, score-descending.
	RecommendBestScore
)

// ContextPair is one discovery constraint: candidates closer to Positive than
// to Negative are favored. Both refer to ids present in the index.
type ContextPair struct {
	Positive uint64
	Negative uint64
}

// DiscoverPair is one RESOLVED discovery constraint: the positive/negative
// example VECTORS (not ids). The Query API leaf carries these resolved vectors
// (the coordinator resolves the context-pair ids → vectors and embeds them), so
// the discover execLeaf scores candidates against the example vectors directly
// without re-resolving ids per partition. It is the resolved analogue of
// ContextPair (which carries ids the engine resolves internally).
type DiscoverPair struct {
	Pos []float32
	Neg []float32
}

// QueryLeaf is one query node: a dense or a sparse leaf. Kind selects which of
// Dense / Sparse is populated. K is the leaf's own top-k; LaneK is the per-lane
// candidate pool pulled before fusion/rerank (0 = the engine default
// max(K,50)). Filter is the optional per-leaf metadata predicate (zero = no
// filter). Space names the target NAMED vector space for the named-collection
// Query API: empty ⇒ a dense-collection leaf, non-empty ⇒ a named-space leaf.
type QueryLeaf struct {
	Kind   LeafKind
	Dense  []float32
	Sparse SparseVector
	Tokens [][]float32
	K      int
	Filter Filter
	LaneK  int
	Space  string
	// Positive / Negative are the RECOMMEND example point-ids (LeafRecommend only):
	// the coordinator resolves them to stored vectors and derives the dense query
	// vector mean(Positive) - mean(Negative).
	Positive []uint64
	Negative []uint64
	// Strategy selects how a RECOMMEND leaf scores candidates (LeafRecommend only):
	// RecommendAverageVector (default, 0) = the derive-to-dense path;
	// RecommendBestScore (1) = a CUSTOM per-candidate max-similarity scorer.
	Strategy RecommendStrategy
	// RecPosVecs / RecNegVecs are the RESOLVED BEST_SCORE recommend example VECTORS
	// (LeafRecommend + RecommendBestScore only).
	RecPosVecs [][]float32
	RecNegVecs [][]float32
	// DiscoverTarget / DiscoverContext are the RESOLVED discover target + context-
	// pair example VECTORS (LeafDiscover only).
	DiscoverTarget  []float32
	DiscoverContext []DiscoverPair
	// DiscoverTargetID / DiscoverContextIDs are the UNRESOLVED discover target +
	// context-pair example POINT-IDS (LeafDiscover only): the input the coordinator
	// resolve pre-pass maps to DiscoverTarget / DiscoverContext.
	DiscoverTargetID   []uint64
	DiscoverContextIDs []ContextPair
	// ScoreDesc tags this leaf's lane ORIENTATION: false = distance-ascending,
	// true = score-descending.
	ScoreDesc bool
}

// QuerySource is one prefetch source in the (bounded) query tree: EITHER a flat
// leaf (Leaf != nil, Spec == nil) OR a nested sub-query (Spec != nil, Leaf ==
// nil). Exactly one arm is set.
type QuerySource struct {
	Leaf *QueryLeaf
	Spec *QuerySpec
}

// LeafSource wraps a plain leaf as a QuerySource{Leaf} — the 1-level form. Every
// existing query (dense/named/MV/recommend/discover) builds its prefetch from leaf
// sources. Exported so the HTTP front end and the ops codec can construct leaf
// sources without reaching into the struct.
func LeafSource(l QueryLeaf) QuerySource {
	return QuerySource{Leaf: &l}
}

// IsLeaf reports whether this source is a flat leaf (the 1-level path).
func (s QuerySource) IsLeaf() bool { return s.Leaf != nil }

// QuerySpec is the full unified query: a root leaf, N prefetch sources, the
// combine mode, the fusion config, and the final top-k. The collection is NOT
// part of the spec (it lives in the op header). A prefetch source is a leaf
// (the 1-level path) or a nested QuerySpec (recursion).
type QuerySpec struct {
	Root     QueryLeaf
	Prefetch []QuerySource
	Mode     QueryMode
	Method   FusionMethod
	Alpha    float64
	RRFK     int
	K        int
	// GroupBy / GroupSize make this a GROUPED query (Qdrant-parity group_by): when
	// GroupBy != "" the final ordered top pool is collapsed by the GroupBy metadata
	// field into the top-K GROUPS (ranked by best member), each with up to GroupSize
	// hits. K is reinterpreted as the number of GROUPS.
	GroupBy   string
	GroupSize int
}

// QueryResult is the mode-tagged result of a query. In ModeFusion, Lanes holds
// one unfused lane per prefetch leaf (in prefetch order) AND Fused holds the
// locally-fused top-k. In ModeRerank, Fused holds the reranked top-k and Lanes
// is nil.
type QueryResult struct {
	Mode  QueryMode
	Fused []Result
	Lanes [][]Result
	// Groups is populated ONLY for a GROUPED query (spec.GroupBy != ""): the top-K
	// groups (ranked by best member) collapsed from the ordered Fused pool. Empty
	// (nil) for a flat query.
	Groups []Group
}

// MaxPrefetchSources bounds the NUMBER of prefetch sources at any single query
// spec node (breadth). A node over this bound is rejected fail-loud
// (ErrTooManyPrefetchSources) at every nesting level. Exported so the ops decode
// reads the SAME const as the engine validation (single source of truth).
const MaxPrefetchSources = 64

// Query API validation errors shared with the ops codec (enforced at the decode
// edge and, defensively, in the engine).
var (
	// ErrQuerySpecTooDeep is returned when a nested query tree exceeds the maximum
	// allowed prefetch nesting depth (anti-DoS).
	ErrQuerySpecTooDeep = errors.New("vector: query spec nesting exceeds the maximum depth")
	// ErrTooManyPrefetchSources is returned when a single query spec node carries
	// more than MaxPrefetchSources prefetch sources (anti-DoS / OOM guard).
	ErrTooManyPrefetchSources = errors.New("vector: query spec node has too many prefetch sources")
)

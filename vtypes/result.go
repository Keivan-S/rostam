// SPDX-License-Identifier: Apache-2.0

package vtypes

import "errors"

// Result is one search result: the id of the vector and its distance from
// the query under the index's metric. Score is the fusion score for hybrid
// search (higher = more relevant); it is 0 for plain dense KNN results.
type Result struct {
	ID       uint64  `json:"id"`
	Distance float32 `json:"distance"`
	Score    float32 `json:"score"`
}

// FusionMethod selects how the dense and sparse lanes are combined in a
// hybrid search.
type FusionMethod uint8

const (
	// FusionRRF is Reciprocal Rank Fusion: score = Σ 1/(rrfK + rank). Rank-based
	// and scale-free, so it needs no score normalization across lanes. Default.
	FusionRRF FusionMethod = iota
	// FusionWeighted is a linear blend of min-max-normalized lane scores:
	// score = alpha*denseNorm + (1-alpha)*sparseNorm.
	FusionWeighted
	// FusionDBSF is Distribution-Based Score Fusion (Qdrant parity). Each lane's
	// relevance values are normalized by their DISTRIBUTION using 3-sigma bounds,
	// then blended with the same alpha-weighted shape as FusionWeighted.
	FusionDBSF
)

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

// ErrDimMismatch is returned when a query vector's length does not match the
// index dimensionality.
var ErrDimMismatch = errors.New("vector: vector length does not match index Dim")

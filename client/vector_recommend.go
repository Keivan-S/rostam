// SPDX-License-Identifier: Apache-2.0
package client

import (
	"context"
	"fmt"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// RecommendRequest recommends points similar to a set of positive examples and
// dissimilar to a set of negative examples, identified by point id. The engine
// resolves the example ids to their stored vectors and excludes them from the
// results.
type RecommendRequest struct {
	Positive    []uint64
	Negative    []uint64
	Strategy    vector.RecommendStrategy // zero = RecommendAverageVector
	K           int
	Filter      vector.Filter
	Consistency Consistency
}

// Recommend runs a RECOMMEND query, returning points similar to the positive
// examples (and dissimilar to any negative examples), ranked by score/distance.
//
// The recommend leaf rides as BOTH the ModeRerank Root and the sole Prefetch
// lane. Two engine constraints force this shape (verified against
// vector/query.go's runQuerySpecAt): every QuerySpec — FUSION or RERANK —
// rejects an empty Prefetch (ErrQueryNoPrefetch), and this package's Query
// decodes the op result via ops.DecodeQueryResultDegraded, which only accepts
// the flat RERANK-tagged wire (a ModeFusion result carries UNFUSED per-lane
// data meant for a cross-partition coordinator to merge, which this direct
// single-shard op path never does). So a plain "just recommend" call must be
// ModeRerank: the Root leaf (rewritten by the engine's recommend pre-pass into
// a derived dense query) supplies the ranking, and the identical leaf
// duplicated into Prefetch supplies the rerank's candidate pool.
func (col *Collection) Recommend(ctx context.Context, req RecommendRequest) (SearchResponse, error) {
	leaf := vector.QueryLeaf{
		Kind:     vector.LeafRecommend,
		Positive: req.Positive,
		Negative: req.Negative,
		Strategy: req.Strategy,
		K:        req.K,
		Filter:   req.Filter,
	}
	spec := vector.QuerySpec{
		Mode:     vector.ModeRerank,
		K:        req.K,
		Root:     leaf,
		Prefetch: []vector.QuerySource{vector.LeafSource(leaf)},
	}
	return col.Query(ctx, QueryRequest{Spec: spec, Consistency: req.Consistency})
}

// QueryRequest is the power-user escape hatch: run any QuerySpec (multi-leaf
// fusion, rerank, discover) directly.
type QueryRequest struct {
	Spec        vector.QuerySpec
	Consistency Consistency
}

// Query runs a raw vector.QuerySpec against the collection, decoding the
// result into the same SearchResponse shape as Search/Recommend.
func (col *Collection) Query(ctx context.Context, req QueryRequest) (SearchResponse, error) {
	specBytes, err := ops.MarshalEngineQuerySpec(req.Spec)
	if err != nil {
		return SearchResponse{}, fmt.Errorf("client: marshal query spec: %w", err)
	}
	body, err := col.c.Call(ctx, "vector_query",
		ops.EncodeQueryArgs(col.name, specBytes, uint8(req.Consistency), 0, 0))
	if err != nil {
		return SearchResponse{}, err
	}
	results, degraded, missing, err := ops.DecodeQueryResultDegraded(body)
	if err != nil {
		return SearchResponse{}, err
	}
	return SearchResponse{Results: results, Degraded: degraded, Missing: missing}, nil
}

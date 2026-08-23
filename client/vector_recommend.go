// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"fmt"

	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// RecommendRequest recommends points similar to a set of positive examples and
// dissimilar to a set of negative examples, identified by point id. The engine
// resolves the example ids to their stored vectors and excludes them from the
// results.
type RecommendRequest struct {
	Positive    []uint64
	Negative    []uint64
	Strategy    vtypes.RecommendStrategy // zero = RecommendAverageVector
	K           int
	Filter      vtypes.Filter
	Consistency Consistency
}

// Recommend runs a RECOMMEND query, returning points similar to the positive
// examples (and dissimilar to any negative examples), ranked by score/distance.
//
// The recommend leaf rides as the sole ModeFusion Prefetch lane (no Root) —
// the canonical shape the HTTP /query handler builds for a recommend request
// (httpapi/vector.go's (queryLeafReq).toLeaf, appended via
// vector.LeafSource(leaf) into spec.Prefetch). With one prefetch lane, the
// fused result is that lane unchanged; the engine's recommend pre-pass
// (resolveRecommendLeaves) resolves the example ids and rewrites the leaf to
// a derived dense query before it runs, excluding the examples from the
// results.
//
// This sends a ModeFusion vector_query, whose single-node reply carries the
// UNFUSED lane(s) (EncodeQueryResult's FUSION tag); only a routed/multi-node
// (coordinator) deployment fuses those lanes and re-tags the reply as the flat
// RERANK result DecodeQueryResultDegraded expects. Against a single
// coordinator-less node, the fused result cannot be decoded.
func (col *Collection) Recommend(ctx context.Context, req RecommendRequest) (SearchResponse, error) {
	leaf := vtypes.QueryLeaf{
		Kind:      vtypes.LeafRecommend,
		Positive:  req.Positive,
		Negative:  req.Negative,
		Strategy:  req.Strategy,
		ScoreDesc: req.Strategy == vtypes.RecommendBestScore,
		K:         req.K,
		Filter:    req.Filter,
	}
	spec := vtypes.QuerySpec{
		Mode:     vtypes.ModeFusion,
		K:        req.K,
		Prefetch: []vtypes.QuerySource{vtypes.LeafSource(leaf)},
	}
	return col.Query(ctx, QueryRequest{Spec: spec, Consistency: req.Consistency})
}

// QueryRequest is the power-user escape hatch: run any QuerySpec (multi-leaf
// fusion, rerank, discover) directly.
type QueryRequest struct {
	Spec        vtypes.QuerySpec
	Consistency Consistency
}

// Query runs a raw vector.QuerySpec against the collection, decoding the
// result into the same SearchResponse shape as Search/Recommend.
//
// If req.Spec.Mode is ModeFusion, this has the same single-node limitation as
// Recommend: the reply carries unfused lanes, and only a routed/multi-node
// (coordinator) deployment fuses and re-tags them into the flat result this
// method decodes. Against a single coordinator-less node, a ModeFusion spec's
// result cannot be decoded.
//
// Returns ErrCollectionNotFound when the collection itself does not exist
// (this also covers Recommend, which delegates to Query).
func (col *Collection) Query(ctx context.Context, req QueryRequest) (SearchResponse, error) {
	specBytes, err := wire.MarshalEngineQuerySpec(req.Spec)
	if err != nil {
		return SearchResponse{}, fmt.Errorf("client: marshal query spec: %w", err)
	}
	body, err := col.c.Call(ctx, "vector_query",
		wire.EncodeQueryArgs(col.name, specBytes, uint8(req.Consistency), 0, 0))
	if err != nil {
		return SearchResponse{}, mapCollErr(err)
	}
	results, degraded, missing, err := wire.DecodeQueryResultDegraded(body)
	if err != nil {
		return SearchResponse{}, err
	}
	return SearchResponse{Results: results, Degraded: degraded, Missing: missing}, nil
}

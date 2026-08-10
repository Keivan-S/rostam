// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// resolveNamedRecommendForFanOut is the CLUSTER named-recommend coordinator
// pre-pass — the per-space analogue of resolveRecommendForFanOut (recommend_fanout.go).
// The single-node (*NamedCollection).Query resolves recommend leaves from the LOCAL
// space index (resolveNamedRecommendLeaves); in a partitioned named
// collection (P>1) the example point-ids may live on OTHER partitions, so the
// coordinator must:
//
//  1. collect the union of every recommend leaf's example ids (positive ∪ negative),
//  2. batch-get their stored PER-SPACE vectors CLUSTER-WIDE (VectorNamedGetBatch
//     route-by-id groups the ids by owning partition and fetches each subset; each
//     present point carries a map[space][]float32),
//  3. rewrite each recommend leaf ONCE on the coordinator (partition-invariant) using
//     the LEAF'S SPACE vectors: AVERAGE_VECTOR → derive normalize(mean(pos)−mean(neg))
//     with the SPACE's metric → a dense leaf (Space PRESERVED); BEST_SCORE → embed the
//     resolved per-space example VECTORS into RecPosVecs/RecNegVecs + CLEAR the ids
//     (the leaf stays a LeafRecommend the named best-score execLeaf runs per partition),
//  4. re-marshal the rewritten (partition-invariant) spec so the EXISTING
//     namedQueryFanOut fans it out verbatim to every partition.
//
// The per-space metric/dim come from the named collection's config
// (VectorNamedGetConfig — identical on every physical partition, read from the
// logical name). It returns the rewritten spec, its re-marshaled bytes, and the
// example-id exclude set (pruned from the merged result AFTER the global fuse/rerank,
// mirroring the single-node path). The over-fetch (k + |examples|) is applied to
// spec.K here so k results survive the post-merge prune. When the spec has no
// recommend leaves it returns ok=false and the caller fans out the original
// spec/bytes unchanged. Fail-loud mirrors the single-node path (per-space metric).
func (e *embedded) resolveNamedRecommendForFanOut(collection string, spec vector.QuerySpec) (vector.QuerySpec, []byte, map[uint64]struct{}, bool, error) {
	if !vector.SpecHasRecommendLeaves(spec) {
		return spec, nil, nil, false, nil
	}

	// 1+2. Resolve every example id cluster-wide with its PER-SPACE vectors.
	ids := vector.RecommendExampleIDs(spec)
	points, _, err := e.VectorNamedGetBatch(context.Background(), collection, ids, true, false)
	if err != nil {
		return spec, nil, nil, false, err
	}
	resolved := make(map[uint64]map[string][]float32, len(points))
	for _, p := range points {
		if len(p.Vectors) > 0 {
			resolved[p.ID] = p.Vectors
		}
	}

	// Per-space metric/dim from the named config (identical on every partition).
	cfg, cerr := e.VectorNamedGetConfig(context.Background(), collection)
	if cerr != nil {
		return spec, nil, nil, false, cerr
	}
	spaceMeta := func(space string) (vector.Metric, int, error) {
		return namedSpaceMeta(cfg, space)
	}

	// 3. Rewrite each recommend leaf ONCE (per-space derive/embed).
	exclude, rerr := vector.RewriteNamedRecommendLeavesWith(&spec, resolved, spaceMeta)
	if rerr != nil {
		return spec, nil, nil, false, rerr
	}

	// Over-fetch so k results remain after the examples are pruned post-merge.
	// Widen EVERY nested sub-spec's K too: collectTreeLanesAt uses each sub-spec's
	// own K as the lane pool for its leaf children, so a recommend leaf buried at any
	// depth (now rewritten) would otherwise under-fetch and leave fewer than wantK
	// survivors after the post-fold prune. vector.WidenNestedSpecsK recurses.
	if len(exclude) > 0 {
		wantK := spec.K
		if wantK <= 0 {
			wantK = 10
		}
		nExclude := len(exclude)
		spec.K = wantK + nExclude
		vector.WidenNestedSpecsK(&spec, nExclude)
	}

	// 4. Re-marshal the rewritten spec for the verbatim fan-out.
	specBytes, merr := ops.MarshalEngineQuerySpec(spec)
	if merr != nil {
		return spec, nil, nil, false, merr
	}
	return spec, specBytes, exclude, true, nil
}

// resolveNamedDiscoverForFanOut is the CLUSTER named-discover coordinator pre-pass —
// the per-space analogue of resolveDiscoverForFanOut (discover_fanout.go). The
// single-node (*NamedCollection).Query resolves discover leaves from the LOCAL space
// index (resolveNamedDiscoverLeaves); in a partitioned named collection
// (P>1) the target + context-pair point-ids may live on OTHER partitions, so the
// coordinator resolves them cluster-wide via VectorNamedGetBatch, EMBEDS each leaf's
// SPACE vectors into the leaf ONCE (RewriteNamedDiscoverLeavesWith) + CLEARS the ids,
// then re-marshals so namedQueryFanOut fans it out verbatim; each partition runs the
// named discover scorer over ITS candidates and the coordinator merges by discover
// score (score-desc — the orientation-aware merge, since the discover leaf carries
// ScoreDesc=true). Discover does NOT exclude the target/context ids, so there is no
// over-fetch and no post-merge prune. When the spec has no id-bearing discover leaves
// it returns ok=false and the caller fans out unchanged. Fail-loud mirrors the
// single-node path (per-space metric validated).
func (e *embedded) resolveNamedDiscoverForFanOut(collection string, spec vector.QuerySpec) (vector.QuerySpec, []byte, bool, error) {
	if !vector.SpecHasDiscoverLeaves(spec) {
		return spec, nil, false, nil
	}

	ids := vector.DiscoverLeafIDs(spec)
	points, _, err := e.VectorNamedGetBatch(context.Background(), collection, ids, true, false)
	if err != nil {
		return spec, nil, false, err
	}
	resolved := make(map[uint64]map[string][]float32, len(points))
	for _, p := range points {
		if len(p.Vectors) > 0 {
			resolved[p.ID] = p.Vectors
		}
	}

	cfg, cerr := e.VectorNamedGetConfig(context.Background(), collection)
	if cerr != nil {
		return spec, nil, false, cerr
	}
	spaceMeta := func(space string) (vector.Metric, int, error) {
		return namedSpaceMeta(cfg, space)
	}

	if rerr := vector.RewriteNamedDiscoverLeavesWith(&spec, resolved, spaceMeta); rerr != nil {
		return spec, nil, false, rerr
	}

	specBytes, merr := ops.MarshalEngineQuerySpec(spec)
	if merr != nil {
		return spec, nil, false, merr
	}
	return spec, specBytes, true, nil
}

// namedSpaceMeta resolves a named DENSE space to its (metric, dim) from the
// collection's named config, fail-loud on an unknown space (ErrUnknownVectorName) or
// a SPARSE space (ErrSpaceModalityMismatch — recommend/discover are dense-vector
// concepts). It is the coordinator's counterpart to (*NamedCollection).denseSpace,
// which the single-node path uses; the named config's per-space Sparse flag is the
// SAME modality marker denseSpace consults (nc.sparseSpaces is built from
// cfg[name].Sparse), so a sparse space is rejected identically on the coordinator.
func namedSpaceMeta(cfg map[string]vector.NamedVectorParams, space string) (vector.Metric, int, error) {
	p, ok := cfg[space]
	if !ok {
		return 0, 0, vector.ErrUnknownVectorName
	}
	if p.Sparse {
		// recommend/discover are dense-vector concepts (mirrors denseSpace's reject).
		return 0, 0, vector.ErrSpaceModalityMismatch
	}
	return p.Metric, p.Dim, nil
}

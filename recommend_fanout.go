// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// resolveRecommendForFanOut is the CLUSTER recommend coordinator pre-pass. The
// single-node (*Collection).Query resolves recommend leaves from the LOCAL
// index; in a partitioned collection (P>1) the example point-ids may live on
// OTHER partitions, so the coordinator must:
//
//  1. collect the union of every recommend leaf's example ids (positive ∪ negative),
//  2. batch-get their stored vectors CLUSTER-WIDE (VectorGetBatch route-by-id groups
//     the ids by owning partition and fetches each subset — so ids spanning
//     partitions are all resolved),
//  3. derive the query vector ONCE on the coordinator (DeriveRecommendVector over the
//     resolved id→vector map — partition-invariant), rewriting each recommend leaf
//     into a dense leaf,
//  4. re-marshal the rewritten (now dense, partition-invariant) spec so the EXISTING
//     queryFanOut fans it out verbatim to every partition (the partition handlers
//     only ever see a dense spec — they reject an un-rewritten recommend leaf).
//
// It returns the rewritten spec, its re-marshaled bytes, and the example-id exclude
// set (the coordinator prunes these from the merged result AFTER the global fuse/
// rerank, mirroring the single-node excludeExamplesFromResult). The over-fetch
// (k + |examples|) is applied to spec.K here so k results survive the post-merge
// prune. When the spec has no recommend leaves it returns ok=false and the caller
// fans out the original spec/bytes unchanged (the dense/named/MV path is untouched).
//
// Fail-loud mirrors the single-node path: no positives in a recommend leaf →
// ErrNoRecommendExamples; NONE of a leaf's positives resolve cluster-wide →
// ErrIDNotFound; a recommend leaf carrying a Space → ErrQueryRecommendHasSpace.
func (e *embedded) resolveRecommendForFanOut(collection string, gen uint32, spec vector.QuerySpec) (vector.QuerySpec, []byte, map[uint64]struct{}, bool, error) {
	// physConfig is the name to use for the vector_get_config call.
	//
	// Key-selection convention (verified against CreateCollection + catalog impls):
	//
	//   P>1 (partitioned, hasCatalogPartitions=true):
	//     physConfig = PartitionKeyGen(collection, gen, 0) — the first physical partition,
	//     which always holds the same dim/metric config as all other partitions.
	//
	//   P<=1 (single-shard, single-node OR multi-node):
	//     CreateCollection takes the DIRECT path for Partitions<=1 (no catalog write,
	//     no logical-collection marker, no SetPartitionsGen call) — the collection is
	//     stored directly under the logical name. As a result hasCatalogPartitions
	//     returns false in BOTH catalog implementations (no entry → ok=false), and
	//     physConfig = collection (the logical name, which IS the collection).
	//     This is verified by TestHasCatalogPartitionsKeySelection.
	var physConfig string
	if gen == 0 && !hasCatalogPartitions(e, collection) {
		physConfig = collection // P<=1: logical collection holds the correct config
	} else {
		physConfig = string(ops.PartitionKeyGen(collection, gen, 0)) // P>1: partition-0
	}
	return e.resolveRecommendWithConfig(collection, physConfig, spec)
}

// hasCatalogPartitions reports whether the collection is registered as P>1 in the
// coordinator's catalog — used to pick the right physical config key in
// resolveRecommendForFanOut. Returns false for both catalog implementations when
// the collection is unpartitioned (P<=1): singleNodeCatalog/ops.Catalog returns
// ok=false when rec.Partitions==0 (the logical marker uses Partitions=0); metaCatalog
// returns ok=false when p<=1.
func hasCatalogPartitions(e *embedded, collection string) bool {
	_, _, ok := e.catalog.PartitionsGen(collection)
	return ok
}

func (e *embedded) resolveRecommendWithConfig(collection, physConfig string, spec vector.QuerySpec) (vector.QuerySpec, []byte, map[uint64]struct{}, bool, error) {
	if !vector.SpecHasRecommendLeaves(spec) {
		return spec, nil, nil, false, nil
	}

	// 1+2. Resolve every example id cluster-wide (with vectors). VectorGetBatch
	// route-by-id fetches ids from their owning partitions, so positives/negatives
	// spanning partitions are all returned; absent ids land in `missing` (skipped,
	// never an error — mirrors vecsForIDs dropping unknown ids).
	ids := vector.RecommendExampleIDs(spec)
	points, _, err := e.VectorGetBatch(context.Background(), collection, ids, true, false)
	if err != nil {
		return spec, nil, nil, false, err
	}
	resolved := make(map[uint64][]float32, len(points))
	for _, p := range points {
		if len(p.Vec) > 0 {
			resolved[p.ID] = p.Vec
		}
	}

	// Coordinator needs the collection dim+metric to derive (the resolved vectors
	// are raw floats; the metric decides normalize). Config is per-physical-partition
	// for a partitioned collection — read it from partition 0 (all partitions share
	// the same dim/metric). For an unpartitioned collection physConfig is the logical
	// collection name (same collection, just without the partition suffix).
	cfgBody, cerr := e.Call(context.Background(), "vector_get_config",
		ops.EncodeGetConfigArgs(physConfig))
	if cerr != nil {
		return spec, nil, nil, false, cerr
	}
	cfg, cerr := ops.DecodeGetConfigResult(cfgBody)
	if cerr != nil {
		return spec, nil, nil, false, cerr
	}

	// 3. Rewrite each recommend leaf ONCE on the coordinator (partition-invariant):
	// AVERAGE_VECTOR → derive mean(pos)-mean(neg) → a dense leaf; BEST_SCORE → embed
	// the resolved positive/negative example VECTORS into the leaf (RecPosVecs/
	// RecNegVecs) and keep it a LeafRecommend the best-score execLeaf runs per
	// partition. RewriteRecommendLeavesWith branches on the leaf's Strategy.
	exclude, rerr := vector.RewriteRecommendLeavesWith(&spec, cfg.Metric, resolved, func(positive, negative []uint64) ([]float32, error) {
		return vector.DeriveRecommendVector(cfg.Dim, cfg.Metric, resolved, positive, negative)
	})
	if rerr != nil {
		return spec, nil, nil, false, rerr
	}

	// Over-fetch so k results remain after the examples are pruned post-merge
	// (mirrors the single-node spec.K = wantK + len(exclude)). Widen EVERY nested
	// sub-spec's K too: collectTreeLanesAt uses each sub-spec's own K as the lane pool
	// for its leaf children, so a recommend leaf buried at any depth (now rewritten to
	// dense/best-score) would otherwise under-fetch. Mirrors named resolveNamedRecommendForFanOut.
	if len(exclude) > 0 {
		wantK := spec.K
		if wantK <= 0 {
			wantK = 10
		}
		nExclude := len(exclude)
		spec.K = wantK + nExclude
		vector.WidenNestedSpecsK(&spec, nExclude)
	}

	// 4. Re-marshal the rewritten (dense) spec for the verbatim fan-out.
	specBytes, merr := ops.MarshalEngineQuerySpec(spec)
	if merr != nil {
		return spec, nil, nil, false, merr
	}
	return spec, specBytes, exclude, true, nil
}

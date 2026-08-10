// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"sort"

	"github.com/rostamlabs/rostam/cluster"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// queryFanOut runs the unified vector_query op on every physical partition and
// reproduces the single-partition (*Collection).Query result EXACTLY for both
// root modes. It is the Query API generalization of hybridFanOut.
//
// FUSION mode: every partition returns its N UNFUSED prefetch lanes (one per
// prefetch leaf, in prefetch order). The coordinator unions lane[i] across all
// partitions, truncates each unioned lane to its GLOBAL per-lane K (the same
// leafLanePool the single-node engine uses), then fuses ONCE via the same
// fold the single-node fuseLanes applies. This is the exactness invariant
// inherited from hybridFanOut: the union is truncated to the per-lane K BEFORE
// fusing so RRF rank / Weighted+DBSF normalization are computed over exactly the
// same candidate lists as the single-partition path. A partition pre-fusing, or
// a wrong per-lane truncation, diverges from the P==1 oracle.
//
// RERANK mode: every partition returns its partition-local reranked top-k
// (QueryResult.Fused). Because point ids are PARTITION-DISJOINT (PartitionOf),
// every doc is prefetched AND reranked on its sole owning partition, so the
// partition-local rerank == the global rerank for that doc. The coordinator just
// merge-sorts the per-partition top-ks by the ROOT leaf's orientation (dense →
// distance ascending, sparse → score descending) and takes the global top-k. No
// re-scoring (the coordinator has no vectors) — correctness relies on id
// disjointness.
//
// rc/opa/bound ride EVERY per-partition arg (via EncodeQueryArgs' opts trailer)
// so a Linearizable query arms each partition leader's readIndex barrier; they
// are never silently dropped on the fan-out path.
func (e *embedded) queryFanOut(collection string, P int, gen uint32, specBytes []byte, spec vector.QuerySpec, rc, opa uint8, bound uint64) ([]VectorResult, cluster.FanResult, error) {
	k := queryK(spec)

	a := cluster.FanArgs{
		Collection:    collection,
		P:             P,
		Generation:    gen,
		K:             k,
		Op:            "vector_query",
		Consistency:   cluster.Consistency(rc),
		OnUnavailable: cluster.OnUnavailable(opa),
		Encode: func(physCol string) []byte {
			// specBytes is reused verbatim for every partition; only the collection
			// name in the header changes. The opts trailer carries rc/opa/bound to
			// each shard (Linearizable arms the per-partition barrier).
			return ops.EncodeQueryArgs(physCol, specBytes, rc, opa, bound)
		},
	}
	decode := func(raw []byte) ([]vector.QueryResult, error) {
		qr, err := ops.DecodeQueryResult(raw)
		if err != nil {
			return nil, err
		}
		return []vector.QueryResult{qr}, nil
	}
	merge := func(parts [][]vector.QueryResult, _ int) []vector.QueryResult {
		var all []vector.QueryResult
		for _, p := range parts {
			all = append(all, p...)
		}
		return all
	}
	parts, fr, err := cluster.FanOut(a, e.node.CallPhysical, decode, merge)
	if err != nil {
		return nil, fr, err
	}

	switch spec.Mode {
	case vector.ModeRerank:
		return rerankMergeFanOut(parts, spec.Root, k), fr, nil
	default: // ModeFusion
		// A spec with a nested MULTI-lane FUSION node shipped per-partition UNFUSED
		// tree-lanes; re-walk the spec tree and fold each FUSION node over the global
		// union (P>1==P1 EXACT). A flat / nested-single-lane / nested-RERANK FUSION spec
		// takes the unchanged top-level fold.
		if vector.SpecHasNestedFusion(spec) {
			res, terr := treeFusionMergeFanOut(parts, spec, k)
			return res, fr, terr
		}
		return fusionMergeFanOut(parts, spec, k), fr, nil
	}
}

// queryK is the effective final top-k for a spec: spec.K if positive, else the
// engine default 10 (matching vector.(*Collection).Query's k default), so the
// coordinator truncates to the same k the single-node engine uses.
func queryK(spec vector.QuerySpec) int {
	if spec.K > 0 {
		return spec.K
	}
	return 10
}

// fusionMergeFanOut unions the per-partition unfused lanes, truncates each
// unioned lane to its global per-lane K, then folds the lanes into the fused
// top-k exactly like the single-node fuseLanes. The per-lane K for lane i is
// vector.SourceLanePool(spec.Prefetch[i], k) — the SAME pool the engine used when
// it produced each partition's lane (a leaf's lane pool, or a nested sub-spec's own
// final top-k), so the unioned-and-truncated lane matches what a single partition
// holding all the data would have produced. A nested sub-spec lane is unioned +
// truncated like a leaf lane (each partition ran the whole sub-spec over its own
// data → a partition-local lane → P>1==P1 recursively).
//
// Lane orientation matches the engine: lane 0 is treated as the dense
// (distance-ascending) lane and each subsequent lane as a score-descending
// (sparse) lane; the fold mirrors fuseLanes (2-lane Fuse, then FuseScoreLanes
// for 3+). v1 cross-partition fusion is exact for the 2-lane dense+sparse case
// (the hybrid generalization); >2 lanes is a documented v2 follow-up but the
// fold here stays consistent with the single-node path so the P==1 oracle holds.
func fusionMergeFanOut(parts []vector.QueryResult, spec vector.QuerySpec, k int) []VectorResult {
	nLanes := len(spec.Prefetch)
	if nLanes == 0 {
		return nil
	}
	// Union each lane across partitions (lane i from every partition).
	unioned := make([][]vector.Result, nLanes)
	for _, qr := range parts {
		for i := 0; i < nLanes && i < len(qr.Lanes); i++ {
			unioned[i] = append(unioned[i], qr.Lanes[i]...)
		}
	}
	// Truncate each unioned lane to its global per-lane K, sorting by the lane's
	// per-leaf ORIENTATION so the truncation keeps the globally-best candidates.
	// Distance-asc leaves (dense / named-dense) truncate distance-ascending;
	// score-desc leaves (sparse / named-sparse / MV-MaxSim / MV-sparse) truncate
	// score-descending. Secondary key = ascending id for cross-partition
	// determinism (partition append order varies; equal scores/distances admit any
	// valid top-k order — the same inherent behavior as single-partition).
	for i := range unioned {
		// Per-lane orientation + pool read the SOURCE: a leaf source uses its leaf's
		// ScoreDesc + leafLanePool; a NESTED sub-spec source uses the sub-spec's effective
		// fused orientation (vector.SourceOrientation) + its own final top-k
		// (vector.SourceLanePool) — so a sub-spec lane is unioned + truncated like a leaf
		// lane (each partition ran the whole sub-spec over its own data → a partition-local
		// lane → P>1==P1 recursively).
		// Sort each unioned lane by its SOURCE's intrinsic orientation, NOT by lane index:
		// a distance-asc source produces distance-ascending Results (Score==0), so it must
		// be truncated distance-ascending to keep the SAME global per-lane top-pool the
		// single partition's distance-truncated lane held; a score-desc source (sparse /
		// MV / a fused sub-spec) produces score-descending Results. Sorting a distance-asc
		// lane by Score (all zero) would tie every candidate and truncate an arbitrary
		// subset → P>1 diverges from P1. truncateLaneByOrientation is the shared per-source
		// union+truncate the recursive tree fold (treeFusionMergeFanOut) reuses verbatim.
		laneScoreDesc := vector.SourceOrientation(spec.Prefetch[i])
		pool := vector.SourceLanePool(spec.Prefetch[i], k)
		unioned[i] = truncateLaneByOrientation(unioned[i], laneScoreDesc, pool)
	}

	// Fold the unioned lanes into the fused top-k, mirroring the single-node
	// vector.fuseLanes contract exactly (it is unexported in the vector package,
	// so the fold is replicated here). 1 lane → that lane truncated to k; 2 lanes
	// → the canonical dense+sparse Fuse; 3+ → fold each remaining score-oriented
	// lane in via FuseScoreLanes (which does not invert the running fused lane).
	// lane0's EFFECTIVE orientation (vector.SourceOrientation — a leaf's ScoreDesc, or
	// a sub-spec's fused orientation) drives the fold start, byte-identical to the
	// single-node runQuerySpec's laneOrient[0].
	lane0ScoreDesc := vector.SourceOrientation(spec.Prefetch[0])
	fused := foldUnionedLanes(unioned, lane0ScoreDesc, spec.Method, spec.Alpha, spec.RRFK, k)
	out := make([]VectorResult, len(fused))
	for i, r := range fused {
		out[i] = VectorResult(r)
	}
	return out
}

// treeFusionMergeFanOut is the RECURSIVE coordinator fold for a spec with a nested
// MULTI-lane FUSION node (vector.SpecHasNestedFusion). Each partition shipped its
// UNFUSED tree-lanes (the flat pre-order spec-tree lane list collectTreeLanesAt walks);
// this re-walks the IDENTICAL spec tree consuming those lanes in the SAME pre-order and
// folds bottom-up: at each FUSION node it produces ONE global lane per prefetch SOURCE
// (a leaf / nested-RERANK / single-lane-FUSION source → union the partition lanes +
// truncate; a nested MULTI-lane FUSION source → RECURSE), then folds the per-source
// global lanes via foldUnionedLanes — STRUCTURALLY MIRRORING the single-node
// runQuerySpecAt fold, but with the cross-partition union+truncate applied at each leaf
// lane FIRST ⇒ global RRF/DBSF normalization at EVERY FUSION node ⇒ P>1==P1 EXACT.
//
// The per-partition emit order (collectTreeLanesAt) and this consume order MUST be the
// IDENTICAL tree walk — both use the SAME predicate (a nested spec source is EXPANDED
// iff sub.Mode==FUSION && len(sub.Prefetch)>=2). A shared cursor (advanced in lockstep
// across all partitions, whose lane lists are structurally identical) guarantees no
// lane mis-association.
func treeFusionMergeFanOut(parts []vector.QueryResult, spec vector.QuerySpec, k int) ([]VectorResult, error) {
	res, _, err := treeFusionMergeFanOutCursor(parts, spec, k)
	return res, err
}

// treeFusionMergeFanOutCursor is the cursor-witnessing variant of treeFusionMergeFanOut:
// it returns the final cursor position alongside the result so the test suite can assert
// emit/consume lane-count agreement (cursor == len(lanes per partition)) without
// modifying the production API.
func treeFusionMergeFanOutCursor(parts []vector.QueryResult, spec vector.QuerySpec, k int) ([]VectorResult, int, error) {
	cursor := 0
	fused, err := mergeTreeFusionNode(parts, spec, 0, &cursor)
	if err != nil {
		return nil, cursor, err
	}
	out := make([]VectorResult, len(fused))
	for i, r := range fused {
		out[i] = VectorResult(r)
	}
	return out, cursor, nil
}

// mergeTreeFusionNode folds ONE FUSION node of the tree-lanes spec into its global
// fused top-k, advancing *cursor over the per-partition flat lane lists in the SAME
// pre-order collectTreeLanesAt emitted. It returns one fused []vector.Result for the
// node (which the parent treats as a single source lane). spec.K bounds the node's own
// fold; depth mirrors collectTreeLanesAt's depth guard (defensive recursion bound — a
// directly-constructed in-memory spec bypasses the decode-side length check).
func mergeTreeFusionNode(parts []vector.QueryResult, spec vector.QuerySpec, depth int, cursor *int) ([]vector.Result, error) {
	if depth > vector.MaxQueryDepthExec {
		// Fail-loud: mirrors the emit-side collectTreeLanesAt which returns
		// ErrQuerySpecTooDeep at the same bound. A directly-constructed in-memory
		// over-deep spec bypasses the decode-side length check; fail-loud here so the
		// coordinator never silently returns an empty/truncated result for a spec that
		// the emit side rejected with an error.
		return nil, vector.ErrQuerySpecTooDeep
	}
	nodeK := spec.K
	if nodeK <= 0 {
		nodeK = 10
	}
	// One GLOBAL lane per prefetch SOURCE (mirroring runQuerySpecAt's lanes[i]).
	srcLanes := make([][]vector.Result, len(spec.Prefetch))
	for i := range spec.Prefetch {
		src := spec.Prefetch[i]
		if src.Spec != nil && src.Spec.Mode == vector.ModeFusion && len(src.Spec.Prefetch) >= 2 {
			// Nested MULTI-lane FUSION source: RECURSE — its global fused top-k is one lane.
			sub, err := mergeTreeFusionNode(parts, *src.Spec, depth+1, cursor)
			if err != nil {
				return nil, err
			}
			srcLanes[i] = sub
			continue
		}
		// Leaf / nested-RERANK / single-lane-FUSION source: ONE lane at *cursor across all
		// partitions — union + truncate by the SOURCE's orientation/pool (identical to
		// fusionMergeFanOut's per-source handling), so a partition-invariant source lane
		// unions losslessly into this node.
		laneScoreDesc := vector.SourceOrientation(src)
		pool := vector.SourceLanePool(src, nodeK)
		var lane []vector.Result
		for p := range parts {
			if *cursor < len(parts[p].Lanes) {
				lane = append(lane, parts[p].Lanes[*cursor]...)
			}
		}
		lane = truncateLaneByOrientation(lane, laneScoreDesc, pool)
		srcLanes[i] = lane
		*cursor++
	}
	lane0ScoreDesc := vector.SourceOrientation(spec.Prefetch[0])
	return foldUnionedLanes(srcLanes, lane0ScoreDesc, spec.Method, spec.Alpha, spec.RRFK, nodeK), nil
}

// truncateLaneByOrientation sorts an UNIONED cross-partition source lane by its
// orientation (score-desc → score descending; else distance ascending), secondary key
// ascending id for cross-partition determinism, then truncates to pool — the SAME
// per-source union+truncate fusionMergeFanOut applies, factored out so the recursive
// tree fold reuses it verbatim.
func truncateLaneByOrientation(lane []vector.Result, scoreDesc bool, pool int) []vector.Result {
	if scoreDesc {
		sort.SliceStable(lane, func(a, b int) bool {
			if lane[a].Score != lane[b].Score {
				return lane[a].Score > lane[b].Score
			}
			return lane[a].ID < lane[b].ID
		})
	} else {
		sort.SliceStable(lane, func(a, b int) bool {
			if lane[a].Distance != lane[b].Distance {
				return lane[a].Distance < lane[b].Distance
			}
			return lane[a].ID < lane[b].ID
		})
	}
	if pool > 0 && len(lane) > pool {
		lane = lane[:pool]
	}
	return lane
}

// foldUnionedLanes folds the unioned prefetch lanes into the fused top-k,
// replicating the single-node vector.fuseLanes (unexported) fold so the
// cross-partition result matches the P==1 path byte-for-byte. lane0ScoreDesc is
// lane0's ORIENTATION (spec.Prefetch[0].ScoreDesc): false (dense / named-dense)
// → the fold starts with Fuse (lane0 distance-ascending, inverted into a score),
// byte-identical to the pre-orientation path; true (MV, both lanes already
// score-descending) → the fold starts with FuseScoreLanes (lane0 is already a
// score, NOT inverted). Either way the running fused lane is score-descending
// after the first fold, so lanes[2:] always fold via FuseScoreLanes.
func foldUnionedLanes(lanes [][]vector.Result, lane0ScoreDesc bool, method vector.FusionMethod, alpha float64, rrfK, k int) []vector.Result {
	switch len(lanes) {
	case 0:
		return nil
	case 1:
		out := lanes[0]
		if k > 0 && len(out) > k {
			out = out[:k]
		}
		return out
	}
	var fused []vector.Result
	if lane0ScoreDesc {
		fused = vector.FuseScoreLanes(lanes[0], lanes[1], method, alpha, rrfK, k)
	} else {
		fused = vector.Fuse(lanes[0], lanes[1], method, alpha, rrfK, k)
	}
	for i := 2; i < len(lanes); i++ {
		fused = vector.FuseScoreLanes(fused, lanes[i], method, alpha, rrfK, k)
	}
	return fused
}

// rerankMergeFanOut merges the per-partition reranked top-ks into the global
// top-k. Each partition's Fused is its partition-local reranked result (exact
// because ids are partition-disjoint), so a merge-sort by the root leaf's
// orientation (root.ScoreDesc) + a top-k truncation reproduces the single-
// partition rerank. Distance-asc root (dense / named-dense) → distance ascending;
// score-desc root (sparse / named-sparse / MV-MaxSim / MV-sparse) → score
// descending. Secondary key = ascending id (cross-partition-stable; ids are
// disjoint so this is a total order with no real ties across partitions).
func rerankMergeFanOut(parts []vector.QueryResult, root vector.QueryLeaf, k int) []VectorResult {
	var all []vector.Result
	for _, qr := range parts {
		all = append(all, qr.Fused...)
	}
	if root.ScoreDesc {
		sort.SliceStable(all, func(a, b int) bool {
			if all[a].Score != all[b].Score {
				return all[a].Score > all[b].Score
			}
			return all[a].ID < all[b].ID
		})
	} else {
		sort.SliceStable(all, func(a, b int) bool {
			if all[a].Distance != all[b].Distance {
				return all[a].Distance < all[b].Distance
			}
			return all[a].ID < all[b].ID
		})
	}
	if k > 0 && len(all) > k {
		all = all[:k]
	}
	out := make([]VectorResult, len(all))
	for i, r := range all {
		out[i] = VectorResult(r)
	}
	return out
}

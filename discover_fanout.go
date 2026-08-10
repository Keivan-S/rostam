// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// resolveDiscoverForFanOut is the CLUSTER discover coordinator pre-pass. The
// single-node (*Collection).Query resolves discover leaves from the LOCAL index
// (resolveDiscoverLeaves); in a partitioned collection (P>1) the target +
// context-pair point-ids may live on OTHER partitions, so the coordinator must:
//
//  1. collect the union of every id-bearing discover leaf's example ids (the
//     optional target id ∪ each context pair's positive/negative),
//  2. batch-get their stored vectors CLUSTER-WIDE (VectorGetBatch route-by-id groups
//     the ids by owning partition and fetches each subset — so ids spanning
//     partitions are all resolved),
//  3. EMBED the resolved target/context vectors into each discover leaf ONCE on the
//     coordinator (RewriteDiscoverLeavesWith) and CLEAR the leaf ids — so the leaf is
//     partition-invariant and each partition handler runs the discover execLeaf
//     (DiscoverVecs) over the embedded vectors without re-resolving against its local
//     index. Unlike recommend, discover stays a LeafDiscover (a custom per-candidate
//     scorer) — it is NOT rewritten to a dense leaf,
//  4. re-marshal the rewritten (now embedded-vectors, partition-invariant) spec so the
//     EXISTING queryFanOut fans it out verbatim to every partition; each partition runs
//     the discover scorer over ITS candidates and the coordinator merges by discover
//     score (score-desc — the orientation-aware fusionMergeFanOut/rerankMergeFanOut,
//     since the discover leaf carries ScoreDesc=true, exactly like MV MaxSim).
//
// Discover does NOT exclude the target/context ids from the result (discover.go ranks
// candidates by its own context score; the example ids are not results to prune — they
// are simply scored like any candidate), so there is no over-fetch and no post-merge
// prune (the recommend exclude path does not apply).
//
// It returns the rewritten spec + its re-marshaled bytes. When the spec has no
// id-bearing discover leaves it returns ok=false and the caller fans out the original
// spec/bytes unchanged (the dense/named/MV/recommend path, and a discover leaf that
// already carries raw vectors, are untouched). Fail-loud mirrors the single-node path:
// a discover leaf carrying a Space → ErrQueryDiscoverHasSpace; a leaf with no context
// at all → ErrQueryDiscoverNoContext; an unresolvable target id (or NO context pair
// resolving) cluster-wide → ErrIDNotFound.
func (e *embedded) resolveDiscoverForFanOut(collection string, spec vector.QuerySpec) (vector.QuerySpec, []byte, bool, error) {
	if !vector.SpecHasDiscoverLeaves(spec) {
		return spec, nil, false, nil
	}

	// 1+2. Resolve every example id cluster-wide (with vectors). VectorGetBatch
	// route-by-id fetches ids from their owning partitions, so a target / context pair
	// spanning partitions is all returned; absent ids land in `missing` (skipped here,
	// surfaced as the fail-loud ErrIDNotFound by RewriteDiscoverLeavesWith when a named
	// target or every pair fails to resolve — mirroring the single-node path).
	ids := vector.DiscoverLeafIDs(spec)
	points, _, err := e.VectorGetBatch(context.Background(), collection, ids, true, false)
	if err != nil {
		return spec, nil, false, err
	}
	resolved := make(map[uint64][]float32, len(points))
	for _, p := range points {
		if len(p.Vec) > 0 {
			resolved[p.ID] = p.Vec
		}
	}

	// 3. Embed the resolved vectors into each discover leaf ONCE on the coordinator and
	// clear the ids (so the partition handlers see an already-resolved leaf).
	if rerr := vector.RewriteDiscoverLeavesWith(&spec, resolved); rerr != nil {
		return spec, nil, false, rerr
	}

	// 4. Re-marshal the rewritten (embedded-vectors) spec for the verbatim fan-out.
	specBytes, merr := ops.MarshalEngineQuerySpec(spec)
	if merr != nil {
		return spec, nil, false, merr
	}
	return spec, specBytes, true, nil
}

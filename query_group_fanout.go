// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"github.com/rostamlabs/rostam/cluster"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// queryGroupPart is one partition's grouped fan-out reply: its UNGROUPED flat query
// result (FUSION unfused lanes / RERANK partition-local reranked top-k) plus its
// per-id group-key map (Metadata[GroupBy] read locally on the owning partition).
type queryGroupPart struct {
	qr   vector.QueryResult
	keys map[uint64]vector.Value
}

// queryGroupFanOut runs the GROUPED Query API (spec.GroupBy != "") across every
// physical partition and reproduces the single-node grouped query (queryGrouped)
// EXACTLY for both root modes — the grouped generalization of queryFanOut.
//
// EXACTNESS INVARIANT (P>1==P1 for FUSION and RERANK):
//   - Each partition runs the flat dense pipeline over the SAME wide candidate pool
//     (vector.GroupFetchK — the single source of truth used by queryGrouped and the
//     standalone SearchGroups), returning its UNGROUPED flat result PLUS a per-id
//     group-key map (vector_query → QueryGroupedFanOut on the leaf).
//   - The coordinator runs its NORMAL fuse/rerank merge (fusionMergeFanOut /
//     rerankMergeFanOut — UNCHANGED) over the wide pool to get the GLOBAL ordered id
//     list. Because the merge is the exact P>1==P1 query merge applied to the wide
//     pool, the ordered pool is identical to the single-node wide ordered pool.
//   - The coordinator maps each ordered id→its group-key (unioned from EVERY
//     partition's per-id key map), reconstructs minimal Documents preserving the merge
//     order, and folds via vector.GroupDocuments ONCE → the top-K groups. Grouping is a
//     deterministic post-process on the exact ordered pool, so the groups, their order,
//     their keys, and the hit order within each group are byte-identical to the
//     single-node grouped query (and to the standalone SearchGroups oracle).
//
// k is the number of GROUPS (spec.K, default 10); the merge truncates to the WIDE pool
// (fetchK), NOT to k, so the ordered pool is large enough to form k groups × groupSize
// — then GroupDocuments truncates to k groups. rc/opa/bound ride every per-partition
// arg (the Linearizable barrier), identical to queryFanOut.
func (e *embedded) queryGroupFanOut(collection string, P int, gen uint32, specBytes []byte, spec vector.QuerySpec, rc, opa uint8, bound uint64) ([]VectorGroup, cluster.FanResult, error) {
	groupsK := queryK(spec) // number of GROUPS (spec.K, default 10)
	groupSize := spec.GroupSize
	if groupSize <= 0 {
		groupSize = 1
	}
	// The WIDE candidate pool the merge must produce before grouping — IDENTICAL formula
	// to queryGrouped / SearchGroups so the ordered pool matches the single-node oracle.
	fetchK := vector.GroupFetchK(groupsK, groupSize, 0)

	a := cluster.FanArgs{
		Collection:    collection,
		P:             P,
		Generation:    gen,
		K:             fetchK,
		Op:            "vector_query",
		Consistency:   cluster.Consistency(rc),
		OnUnavailable: cluster.OnUnavailable(opa),
		Encode: func(physCol string) []byte {
			// specBytes is reused verbatim (GroupBy set ⇒ each partition runs
			// QueryGroupedFanOut, widening to the wide pool itself). rc/opa/bound ride the
			// trailer so a Linearizable grouped query arms each partition leader's barrier.
			return ops.EncodeQueryArgs(physCol, specBytes, rc, opa, bound)
		},
	}
	decode := func(raw []byte) ([]queryGroupPart, error) {
		qr, keys, err := ops.DecodeQueryResultGroupedFanOut(raw)
		if err != nil {
			return nil, err
		}
		return []queryGroupPart{{qr: qr, keys: keys}}, nil
	}
	merge := func(parts [][]queryGroupPart, _ int) []queryGroupPart {
		var all []queryGroupPart
		for _, p := range parts {
			all = append(all, p...)
		}
		return all
	}
	parts, fr, err := cluster.FanOut(a, e.node.CallPhysical, decode, merge)
	if err != nil {
		return nil, fr, err
	}

	return groupMergedQueryParts(parts, spec, fetchK, groupsK, groupSize), fr, nil
}

// groupMergedQueryParts is the SHARED post-merge grouping step for the grouped Query
// API (used by both the P>1 fan-out and the single-shard P==1 path): it runs the
// UNCHANGED fuse/rerank merge over the per-partition flat results to get the global
// ordered pool (to the WIDE fetchK), unions the per-id group-key maps, reconstructs
// minimal Documents in merge order, and folds via vector.GroupDocuments ONCE into the
// top-groupsK groups. The merge is byte-identical to the non-grouped query merge; only
// the final GroupDocuments fold is new — so FUSION and RERANK group P>1==P1 exact.
func groupMergedQueryParts(parts []queryGroupPart, spec vector.QuerySpec, fetchK, groupsK, groupSize int) []VectorGroup {
	// Union every partition's per-id group-key map. IDs are partition-disjoint, so no
	// key collides; an id absent everywhere simply has no group value (GroupDocuments
	// drops such a hit, matching the single-node path).
	keys := make(map[uint64]vector.Value)
	flat := make([]vector.QueryResult, 0, len(parts))
	for _, p := range parts {
		flat = append(flat, p.qr)
		for id, v := range p.keys {
			keys[id] = v
		}
	}

	// Run the NORMAL merge over the WIDE pool (k=fetchK, NOT groupsK) so the ordered pool
	// is large enough to form groupsK groups × groupSize before grouping.
	var ordered []VectorResult
	switch spec.Mode {
	case vector.ModeRerank:
		ordered = rerankMergeFanOut(flat, spec.Root, fetchK)
	default: // ModeFusion
		ordered = fusionMergeFanOut(flat, spec, fetchK)
	}

	// Reconstruct minimal Documents in the GLOBAL ordered order, attaching each id's
	// group-key as Metadata[GroupBy] — the SAME shape GroupDocuments reads. An id with no
	// key carries no Metadata and is skipped by GroupDocuments (matching single-node).
	docs := make([]vector.Document, 0, len(ordered))
	for _, r := range ordered {
		d := vector.Document{ID: r.ID, Distance: r.Distance, Score: r.Score}
		if v, ok := keys[r.ID]; ok {
			d.Metadata = vector.Metadata{spec.GroupBy: v}
		}
		docs = append(docs, d)
	}
	return vector.GroupDocuments(docs, vector.GroupOpts{GroupBy: spec.GroupBy, GroupSize: groupSize}, groupsK)
}

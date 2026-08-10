// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"sort"

	"github.com/rostamlabs/rostam/cluster"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// hybridLanes carries one partition's unfused dense/sparse candidate lanes back
// to the coordinator so they can be unioned across partitions and fused once.
type hybridLanes struct {
	dense  []vector.Result
	sparse []vector.Result
}

// hybridFanOut runs vector_hybrid_lanes on every physical partition, unions the
// per-partition lanes, truncates the unioned lanes to the GLOBAL denseK/sparseK,
// then fuses ONCE — reproducing single-partition hybrid EXACTLY.
//
// EXACTNESS INVARIANT: the union is truncated to denseK/sparseK BEFORE fusing.
// RRF rank and Weighted min-max normalization are computed over exactly the
// denseK/sparseK candidate lists; fusing the full P×denseK union would change
// ranks/normalization and diverge from the single-partition oracle. The denseK/
// sparseK default formula matches HybridSearch/HybridLanes (max(k, 50)).
func (e *embedded) hybridFanOut(collection string, P int, gen uint32, dense []float32, k int, opts VectorHybridOpts) ([]VectorResult, cluster.FanResult, error) {
	hopts := toVectorHybridOpts(opts)
	denseK := hopts.DenseK
	if denseK <= 0 {
		denseK = k
		if denseK < 50 {
			denseK = 50
		}
	}
	sparseK := hopts.SparseK
	if sparseK <= 0 {
		sparseK = k
		if sparseK < 50 {
			sparseK = 50
		}
	}

	a := cluster.FanArgs{
		Collection:    collection,
		P:             P,
		Generation:    gen,
		K:             k,
		Op:            "vector_hybrid_lanes",
		Consistency:   cluster.Consistency(opts.ReadConsistency),
		OnUnavailable: cluster.OnUnavailable(opts.OnPartitionUnavailable),
		Encode: func(physCol string) []byte {
			// rc/opa trailer carries Linearizable to the shard so the readIndex
			// barrier runs on each partition's leader. The vector_hybrid_lanes
			// handler decodes with DecodeHybridSearchArgs, which ignores the
			// trailer, so this is wire-compatible with the lanes handler while
			// ops.ReadConsistencyOf("vector_hybrid_lanes", ...) reads the byte.
			return ops.EncodeHybridSearchArgsOpts(physCol, dense, k, opts.Sparse, hopts, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness)
		},
	}
	decode := func(raw []byte) ([]hybridLanes, error) {
		d, s, err := ops.DecodeHybridLanesResult(raw)
		if err != nil {
			return nil, err
		}
		return []hybridLanes{{dense: d, sparse: s}}, nil
	}
	merge := func(parts [][]hybridLanes, _ int) []hybridLanes {
		var all []hybridLanes
		for _, p := range parts {
			all = append(all, p...)
		}
		return all
	}
	parts, fr, err := cluster.FanOut(a, e.node.CallPhysical, decode, merge)
	if err != nil {
		return nil, fr, err
	}

	var allDense, allSparse []vector.Result
	for _, p := range parts {
		allDense = append(allDense, p.dense...)
		allSparse = append(allSparse, p.sparse...)
	}

	// EXACTNESS INVARIANT: truncate to the global denseK/sparseK BEFORE fusing.
	// Dense lane: ascending Distance, secondary ascending ID for fan-out-internal
	// determinism across runs (partition append order can vary). Equal distances
	// admit any valid top-k order; the secondary key does not reproduce single-
	// partition HNSW tie order (HNSW heap order is not globally reproducible).
	// Sparse lane: descending Score, secondary ascending ID. Single-partition
	// breaks score ties by slot (insertion index), which is partition-local and
	// meaningless globally; ascending ID is the only cross-partition-stable
	// tiebreak and is correct because equal scores admit any valid top-k order.
	// Ties spanning the denseK/sparseK cutoff are kept by sort order — the same
	// inherent top-k-with-ties behavior as single-partition, no new divergence.
	sort.SliceStable(allDense, func(i, j int) bool {
		if allDense[i].Distance != allDense[j].Distance {
			return allDense[i].Distance < allDense[j].Distance
		}
		return allDense[i].ID < allDense[j].ID
	})
	if len(allDense) > denseK {
		allDense = allDense[:denseK]
	}
	sort.SliceStable(allSparse, func(i, j int) bool {
		if allSparse[i].Score != allSparse[j].Score {
			return allSparse[i].Score > allSparse[j].Score
		}
		return allSparse[i].ID < allSparse[j].ID
	})
	if len(allSparse) > sparseK {
		allSparse = allSparse[:sparseK]
	}

	// Mirror HybridSearch's single-lane degradation, then fuse both lanes.
	var fused []vector.Result
	switch {
	case opts.Sparse.IsZero():
		fused = allDense
		if len(fused) > k {
			fused = fused[:k]
		}
	case len(dense) == 0:
		fused = allSparse
		if len(fused) > k {
			fused = fused[:k]
		}
	default:
		fused = vector.Fuse(allDense, allSparse, hopts.Method, hopts.Alpha, hopts.RRFK, k)
	}

	// VectorResult is an alias of vector.Result; the conversion is identity but
	// kept explicit to insulate this path from the alias becoming a distinct type.
	out := make([]VectorResult, len(fused))
	for i, r := range fused {
		out[i] = VectorResult(r)
	}
	return out, fr, nil
}

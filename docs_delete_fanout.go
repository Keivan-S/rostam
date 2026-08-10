// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"sort"

	"github.com/rostamlabs/rostam/cluster"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// docsFanOut scatters vector_search_docs to all partitions and merges the
// enriched documents by ascending distance into the global top-k. Exact: each
// Document carries its own partition-local distance (no global normalization),
// IDs are disjoint, so the global top-k is a subset of the union of per-partition
// top-k. Honors ReadConsistency / OnPartitionUnavailable.
func (e *embedded) docsFanOut(collection string, P int, gen uint32, query []float32, k int, filter VectorFilter, rc, opa uint8, bound uint64) ([]VectorDocument, cluster.FanResult, error) {
	a := cluster.FanArgs{
		Collection:    collection,
		P:             P,
		Generation:    gen,
		K:             k,
		Op:            "vector_search_docs",
		Consistency:   cluster.Consistency(rc),
		OnUnavailable: cluster.OnUnavailable(opa),
		Encode: func(physCol string) []byte {
			// rc/opa trailer carries Linearizable to the shard so the readIndex
			// barrier runs on each partition's leader (vector_search_docs).
			return ops.EncodeVectorSearchArgsOpts(physCol, k, query, filter, rc, opa, bound)
		},
	}
	decode := func(raw []byte) ([]vector.Document, error) {
		return ops.DecodeVectorDocs(raw)
	}
	merge := func(parts [][]vector.Document, k int) []vector.Document {
		var all []vector.Document
		for _, p := range parts {
			all = append(all, p...)
		}
		sort.SliceStable(all, func(i, j int) bool {
			if all[i].Distance != all[j].Distance {
				return all[i].Distance < all[j].Distance
			}
			return all[i].ID < all[j].ID
		})
		if k >= 0 && len(all) > k {
			all = all[:k]
		}
		return all
	}
	return cluster.FanOut(a, e.node.CallPhysical, decode, merge)
}

// deleteByFilterFanOut scatters vector_delete_by_filter to all partitions (each
// deletes its disjoint subset) and sums the counts. Fail-loud: OnUnavailable=Fail
// aborts with an error if any partition is unreachable. Because partitions execute
// concurrently with no cross-partition rollback, any partition that already
// succeeded before the abort has committed its deletes — so a failed call leaves a
// PARTIAL deletion (not all-or-nothing). The op is idempotent (re-running deletes
// only remaining matches; already-deleted are no-ops), so the caller safely retries
// the same filter when the cluster is healthy to complete the deletion. Writes route
// to each partition's Raft leader via CallPhysical.
func (e *embedded) deleteByFilterFanOut(collection string, P int, gen uint32, filter VectorFilter) (int, cluster.FanResult, error) {
	a := cluster.FanArgs{
		Collection:    collection,
		P:             P,
		Generation:    gen,
		K:             -1, // no top-k; sum counts
		Op:            "vector_delete_by_filter",
		Consistency:   cluster.LeaderOnly, // writes go to the leader anyway; explicit
		OnUnavailable: cluster.Fail,       // destructive op: fail-loud on any unreachable partition
		Encode: func(physCol string) []byte {
			return ops.EncodeDeleteByFilterArgs(physCol, filter)
		},
	}
	decode := func(raw []byte) ([]int, error) {
		n, err := ops.DecodeDeleteByFilterResult(raw)
		if err != nil {
			return nil, err
		}
		return []int{n}, nil
	}
	merge := func(parts [][]int, _ int) []int {
		sum := 0
		for _, p := range parts {
			for _, c := range p {
				sum += c
			}
		}
		return []int{sum}
	}
	counts, fr, err := cluster.FanOut(a, e.node.CallPhysical, decode, merge)
	if err != nil {
		return 0, fr, err
	}
	total := 0
	for _, c := range counts {
		total += c
	}
	return total, fr, nil
}

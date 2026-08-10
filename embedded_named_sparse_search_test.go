// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"sort"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// seedNamedSparseCollection creates a P-partition (or P==1) named collection with a
// dense "title" space and a sparse "terms" space, then inserts ids 1..n each with a
// sparse "terms" value derived deterministically from the id, plus a shared payload
// {"id": id}. Returns the Store and the inserted sparse vectors (by id) for a
// brute-force ground truth.
func seedNamedSparseCollection(t *testing.T, coll string, P, n int) (Store, map[uint64]*vector.SparseVector) {
	t.Helper()
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	cfg := map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: vector.L2},
		"terms": {Sparse: true},
	}
	if err := s.VectorNamedCreateCollection(ctx, coll, cfg, P); err != nil {
		t.Fatalf("create named sparse (P=%d): %v", P, err)
	}
	vecs := make(map[uint64]*vector.SparseVector, n)
	emb := s.(*embedded)
	for id := uint64(1); id <= uint64(n); id++ {
		// Deterministic sparse value: terms {id%7, id%11+12, id%13+24} with weights.
		idx := []uint32{uint32(id % 7), uint32(id%11) + 12, uint32(id%13) + 24}
		sort.Slice(idx, func(i, j int) bool { return idx[i] < idx[j] })
		// Drop dups so indices stay strictly ascending (rare collisions).
		uniq := idx[:1]
		for _, v := range idx[1:] {
			if v != uniq[len(uniq)-1] {
				uniq = append(uniq, v)
			}
		}
		sv := &vector.SparseVector{Indices: uniq, Values: make([]float32, len(uniq))}
		for i := range sv.Values {
			// Distinct per-(id,term) weights to minimize score ties (so the top-k id set
			// is well-defined and the fan-out merge order matches brute force).
			sv.Values[i] = float32(id)*0.013 + float32(uniq[i])*0.001 + float32(i)*0.1 + 1
		}
		meta := VectorMetadata{"id": vector.NewInt(int64(id))}
		if err := emb.VectorNamedInsertSparse(ctx, coll, id, nil, map[string]*vector.SparseVector{"terms": sv}, meta, 0); err != nil {
			t.Fatalf("named sparse insert %d: %v", id, err)
		}
		vecs[id] = sv
	}
	return s, vecs
}

// bruteForceSparse ranks vecs by dot(query, v) descending (ties: lower id), top-k.
func bruteForceSparse(vecs map[uint64]*vector.SparseVector, query vector.SparseVector, k int) []uint64 {
	type sc struct {
		id    uint64
		score float32
	}
	var all []sc
	for id, v := range vecs {
		var dot float32
		i, j := 0, 0
		for i < len(query.Indices) && j < len(v.Indices) {
			switch {
			case query.Indices[i] == v.Indices[j]:
				dot += query.Values[i] * v.Values[j]
				i++
				j++
			case query.Indices[i] < v.Indices[j]:
				i++
			default:
				j++
			}
		}
		if dot != 0 {
			all = append(all, sc{id, dot})
		}
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].score != all[j].score {
			return all[i].score > all[j].score
		}
		return all[i].id < all[j].id
	})
	if len(all) > k {
		all = all[:k]
	}
	out := make([]uint64, len(all))
	for i, s := range all {
		out[i] = s.id
	}
	return out
}

// TestNamedSparseSearchFanOutMatchesGroundTruth: a sparse search over a P>1 named
// collection returns the same top-k (by id set, since equal scores are possible) as
// a brute-force dot-product ranking over all inserted sparse vectors.
func TestNamedSparseSearchFanOutMatchesGroundTruth(t *testing.T) {
	const P, n, k = 4, 240, 15
	s, vecs := seedNamedSparseCollection(t, "nsp", P, n)
	ctx := context.Background()

	// Ensure the ids actually span all partitions (else the fan-out isn't exercised).
	touched := map[int]bool{}
	for id := uint64(1); id <= uint64(n); id++ {
		touched[ops.PartitionOf(id, P)] = true
	}
	if len(touched) != P {
		t.Fatalf("ids only touch %d/%d partitions", len(touched), P)
	}

	query := vector.SparseVector{Indices: []uint32{1, 14, 26}, Values: []float32{2, 3, 1}}
	got, err := s.VectorNamedSparseSearch(ctx, "nsp", "terms", query, k, VectorFilter{})
	if err != nil {
		t.Fatalf("sparse search: %v", err)
	}
	want := bruteForceSparse(vecs, query, k)

	// Compare as score-sorted ids; equal-score ties may differ in id order across the
	// fan-out merge vs brute force only when scores tie, so compare the score sequence
	// AND the set membership. The score sequence must match exactly (descending).
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d, want %d (%v vs %v)", len(got), len(want), idsOfResults(got), want)
	}
	gotIDs := idsOfResults(got)
	// Verify scores are non-increasing.
	for i := 1; i < len(got); i++ {
		if got[i].Score > got[i-1].Score {
			t.Fatalf("scores not descending at %d: %v", i, got)
		}
	}
	// Verify the returned id set equals the ground-truth id set.
	if !sameIDSet(gotIDs, want) {
		t.Fatalf("id set mismatch:\n got=%v\nwant=%v", gotIDs, want)
	}
}

// TestNamedSparseSearchFanOutFilter: a filtered sparse search over P>1 returns only
// points matching the payload filter.
func TestNamedSparseSearchFanOutFilter(t *testing.T) {
	const P, n, k = 4, 120, 50
	s, _ := seedNamedSparseCollection(t, "nspf", P, n)
	ctx := context.Background()
	// Filter id <= 30.
	filter := VectorFilter{Op: vector.FilterLte, Field: "id", Value: vector.NewInt(30)}
	query := vector.SparseVector{Indices: []uint32{1, 14, 26}, Values: []float32{2, 3, 1}}
	got, err := s.VectorNamedSparseSearch(ctx, "nspf", "terms", query, k, filter)
	if err != nil {
		t.Fatalf("filtered sparse search: %v", err)
	}
	for _, r := range got {
		if r.ID > 30 {
			t.Fatalf("filter leaked id %d (> 30)", r.ID)
		}
	}
}

func idsOfResults(res []VectorResult) []uint64 {
	out := make([]uint64, len(res))
	for i, r := range res {
		out[i] = r.ID
	}
	return out
}

func sameIDSet(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	m := make(map[uint64]int, len(a))
	for _, x := range a {
		m[x]++
	}
	for _, x := range b {
		m[x]--
	}
	for _, v := range m {
		if v != 0 {
			return false
		}
	}
	return true
}

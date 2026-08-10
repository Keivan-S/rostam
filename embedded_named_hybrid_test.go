// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"sort"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// seedNamedHybridCollection creates a P-partition (or P==1) named collection with a
// dense "title" space (dim 4, L2) and a sparse "terms" space, then inserts ids 1..n
// each with BOTH a deterministic dense vector and a deterministic sparse vector (so
// both hybrid lanes can score every point), plus a shared payload {"id": id}.
func seedNamedHybridCollection(t *testing.T, coll string, P, n int) Store {
	t.Helper()
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	cfg := map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: vector.L2},
		"terms": {Sparse: true},
	}
	if err := s.VectorNamedCreateCollection(ctx, coll, cfg, P); err != nil {
		t.Fatalf("create named hybrid (P=%d): %v", P, err)
	}
	emb := s.(*embedded)
	for id := uint64(1); id <= uint64(n); id++ {
		// Distinct-ish dense vector so L2 distances rarely tie.
		dense := []float32{
			float32(id)*0.01 + 0.1,
			float32(id%5)*0.2 + 0.05,
			float32(id%7)*0.13 + 0.02,
			float32(id%3)*0.31 + 0.07,
		}
		idx := []uint32{uint32(id % 7), uint32(id%11) + 12, uint32(id%13) + 24}
		sort.Slice(idx, func(i, j int) bool { return idx[i] < idx[j] })
		uniq := idx[:1]
		for _, v := range idx[1:] {
			if v != uniq[len(uniq)-1] {
				uniq = append(uniq, v)
			}
		}
		sv := &vector.SparseVector{Indices: uniq, Values: make([]float32, len(uniq))}
		for i := range sv.Values {
			sv.Values[i] = float32(id)*0.013 + float32(uniq[i])*0.001 + float32(i)*0.1 + 1
		}
		meta := VectorMetadata{"id": vector.NewInt(int64(id))}
		if err := emb.VectorNamedInsertSparse(ctx, coll, id, map[string][]float32{"title": dense},
			map[string]*vector.SparseVector{"terms": sv}, meta, 0); err != nil {
			t.Fatalf("named hybrid insert %d: %v", id, err)
		}
	}
	return s
}

// hybridResultKey is a fully comparable hybrid result (id + fused score + dense
// distance) for exact P1-vs-P4 comparison.
type hybridResultKey struct {
	id    uint64
	score float32
	dist  float32
}

func hybridKeys(res []VectorResult) []hybridResultKey {
	out := make([]hybridResultKey, len(res))
	for i, r := range res {
		out[i] = hybridResultKey{r.ID, r.Score, r.Distance}
	}
	return out
}

func eqHybridKeys(a, b []hybridResultKey) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestNamedHybridFanOutMatchesP1 is the partition-invariance oracle: a cross-space
// named hybrid over a P>1 collection (fan lanes → union → Fuse) returns the EXACT
// same fused top-k (id + score + distance, in order) as the same hybrid over a P==1
// collection. This proves the lanes-fan-out + global-Fuse reproduces the single-
// partition path. Run for both RRF and weighted fusion.
func TestNamedHybridFanOutMatchesP1(t *testing.T) {
	const n, k = 200, 12
	ctx := context.Background()
	denseQ := []float32{1.2, 0.6, 0.3, 0.4}
	sparseQ := vector.SparseVector{Indices: []uint32{1, 14, 26}, Values: []float32{2, 3, 1}}

	cases := []struct {
		name string
		opts NamedHybridOpts
	}{
		{"rrf", NamedHybridOpts{Method: FusionRRF}},
		{"weighted", NamedHybridOpts{Method: FusionWeighted, Alpha: 0.35}},
		{"dbsf", NamedHybridOpts{Method: FusionDBSF, Alpha: 0.35}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s1 := seedNamedHybridCollection(t, "nh1_"+tc.name, 1, n)
			got1, err := s1.VectorNamedHybridSearch(ctx, "nh1_"+tc.name, "title", denseQ, "terms", sparseQ, k, tc.opts)
			if err != nil {
				t.Fatalf("P1 hybrid: %v", err)
			}

			const P = 4
			sP := seedNamedHybridCollection(t, "nh4_"+tc.name, P, n)
			// Make sure the ids span all partitions so the fan-out is genuinely exercised.
			touched := map[int]bool{}
			for id := uint64(1); id <= uint64(n); id++ {
				touched[ops.PartitionOf(id, P)] = true
			}
			if len(touched) != P {
				t.Fatalf("ids only touch %d/%d partitions", len(touched), P)
			}
			gotP, err := sP.VectorNamedHybridSearch(ctx, "nh4_"+tc.name, "title", denseQ, "terms", sparseQ, k, tc.opts)
			if err != nil {
				t.Fatalf("P4 hybrid: %v", err)
			}

			if !eqHybridKeys(hybridKeys(gotP), hybridKeys(got1)) {
				t.Fatalf("P4 != P1:\n P4=%v\n P1=%v", hybridKeys(gotP), hybridKeys(got1))
			}
		})
	}
}

// TestNamedHybridModalityMismatchEdge: the edge surfaces a clean error (not a
// silent wrong-lane result) when a dense space is named as the sparse lane.
func TestNamedHybridModalityMismatchEdge(t *testing.T) {
	ctx := context.Background()
	s := seedNamedHybridCollection(t, "nhmm", 1, 10)
	_, err := s.VectorNamedHybridSearch(ctx, "nhmm", "terms", []float32{1, 2, 3, 4}, "terms",
		vector.SparseVector{Indices: []uint32{1}, Values: []float32{1}}, 5, NamedHybridOpts{})
	if err == nil {
		t.Fatal("expected modality-mismatch error, got nil")
	}
}

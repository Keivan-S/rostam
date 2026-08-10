// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"sort"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// seedMVHybridCollection creates a P-partition (or P==1) MV collection (dim 4) and
// adds ids 1..n each with a deterministic token matrix (MaxSim lane) AND a
// deterministic doc-level sparse vector (sparse lane), plus a shared payload
// {"id": id}. The MV analogue of seedNamedHybridCollection.
func seedMVHybridCollection(t *testing.T, coll string, P, n int) Store {
	t.Helper()
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	if err := s.VectorMVCreateCollection(ctx, coll, MultiVectorConfig{Dim: 4, Partitions: P}); err != nil {
		t.Fatalf("create mv hybrid (P=%d): %v", P, err)
	}
	for id := uint64(1); id <= uint64(n); id++ {
		// Two distinct-ish token rows so MaxSim scores rarely tie.
		tokens := [][]float32{
			{float32(id)*0.01 + 0.1, float32(id%5)*0.2 + 0.05, float32(id%7)*0.13 + 0.02, float32(id%3)*0.31 + 0.07},
			{float32(id%4)*0.17 + 0.03, float32(id)*0.005 + 0.2, float32(id%9)*0.11 + 0.01, float32(id%6)*0.23 + 0.04},
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
		if err := s.VectorMVAdd(ctx, coll, id, tokens, meta, WriteOpts{Sparse: sv}); err != nil {
			t.Fatalf("mv hybrid add %d: %v", id, err)
		}
	}
	return s
}

// TestMVHybridFanOutMatchesP1 is the partition-invariance oracle: an MV hybrid over a
// P>1 collection (fan vector_mv_hybrid_lanes → union each lane → truncate to the
// global denseK/sparseK → FuseScoreLanes) returns the EXACT same fused top-k
// (id + score + distance, in order) as the same hybrid over a P==1 collection. This
// proves the lanes-fan-out + global-fuse reproduces the single-partition MVHybrid.
// Run for both RRF and weighted fusion. Mirrors TestNamedHybridFanOutMatchesP1.
func TestMVHybridFanOutMatchesP1(t *testing.T) {
	const n, k = 200, 12
	ctx := context.Background()
	query := [][]float32{{1.2, 0.6, 0.3, 0.4}, {0.2, 0.9, 0.1, 0.5}}
	sparseQ := vector.SparseVector{Indices: []uint32{1, 14, 26}, Values: []float32{2, 3, 1}}

	cases := []struct {
		name string
		opts MVHybridOpts
	}{
		{"rrf", MVHybridOpts{Method: FusionRRF}},
		{"weighted", MVHybridOpts{Method: FusionWeighted, Alpha: 0.35}},
		{"dbsf", MVHybridOpts{Method: FusionDBSF, Alpha: 0.35}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s1 := seedMVHybridCollection(t, "mh1_"+tc.name, 1, n)
			got1, err := s1.VectorMVHybridSearch(ctx, "mh1_"+tc.name, query, sparseQ, k, tc.opts)
			if err != nil {
				t.Fatalf("P1 mv hybrid: %v", err)
			}

			const P = 4
			sP := seedMVHybridCollection(t, "mh4_"+tc.name, P, n)
			touched := map[int]bool{}
			for id := uint64(1); id <= uint64(n); id++ {
				touched[ops.PartitionOf(id, P)] = true
			}
			if len(touched) != P {
				t.Fatalf("ids only touch %d/%d partitions", len(touched), P)
			}
			gotP, err := sP.VectorMVHybridSearch(ctx, "mh4_"+tc.name, query, sparseQ, k, tc.opts)
			if err != nil {
				t.Fatalf("P4 mv hybrid: %v", err)
			}

			if !eqHybridKeys(hybridKeys(gotP), hybridKeys(got1)) {
				t.Fatalf("P4 != P1:\n P4=%v\n P1=%v", hybridKeys(gotP), hybridKeys(got1))
			}
		})
	}
}

// TestMVHybridFanOutSingleLaneDegradation: the fan-out path collapses to a single
// lane exactly as the single-partition MVHybrid does (empty sparse ⇒ MaxSim only;
// empty tokens ⇒ sparse only), and P>1 still equals P1.
func TestMVHybridFanOutSingleLaneDegradation(t *testing.T) {
	const n, k = 120, 10
	ctx := context.Background()
	query := [][]float32{{1.0, 0.5, 0.25, 0.3}}
	sparseQ := vector.SparseVector{Indices: []uint32{2, 14}, Values: []float32{1, 2}}

	s1 := seedMVHybridCollection(t, "mhd1", 1, n)
	s4 := seedMVHybridCollection(t, "mhd4", 4, n)

	// MaxSim only (empty sparse).
	a1, err := s1.VectorMVHybridSearch(ctx, "mhd1", query, vector.SparseVector{}, k, MVHybridOpts{Method: FusionRRF})
	if err != nil {
		t.Fatal(err)
	}
	a4, err := s4.VectorMVHybridSearch(ctx, "mhd4", query, vector.SparseVector{}, k, MVHybridOpts{Method: FusionRRF})
	if err != nil {
		t.Fatal(err)
	}
	if !eqHybridKeys(hybridKeys(a4), hybridKeys(a1)) {
		t.Fatalf("maxsim-only P4 != P1:\n P4=%v\n P1=%v", hybridKeys(a4), hybridKeys(a1))
	}

	// Sparse only (empty tokens).
	b1, err := s1.VectorMVHybridSearch(ctx, "mhd1", nil, sparseQ, k, MVHybridOpts{Method: FusionRRF})
	if err != nil {
		t.Fatal(err)
	}
	b4, err := s4.VectorMVHybridSearch(ctx, "mhd4", nil, sparseQ, k, MVHybridOpts{Method: FusionRRF})
	if err != nil {
		t.Fatal(err)
	}
	if !eqHybridKeys(hybridKeys(b4), hybridKeys(b1)) {
		t.Fatalf("sparse-only P4 != P1:\n P4=%v\n P1=%v", hybridKeys(b4), hybridKeys(b1))
	}
}

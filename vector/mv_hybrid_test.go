// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"testing"
)

// mvHybridFixture builds an MV index with docs carrying BOTH a token matrix (for the
// MaxSim lane) and a doc-level sparse vector (for the sparse lane). Some docs omit
// the sparse vector (sparseNil) so they contribute only to the MaxSim lane.
func mvHybridFixture(t *testing.T) (*MultiVectorIndex, map[uint64]*SparseVector) {
	t.Helper()
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 3, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })

	type doc struct {
		tokens [][]float32
		sparse *SparseVector
		meta   Metadata
	}
	docs := map[uint64]doc{
		1: {tokens: [][]float32{{1, 0, 0}, {0, 1, 0}}, sparse: mvSV(0, 1.0, 2, 2.0), meta: Metadata{"g": NewString("a")}},
		2: {tokens: [][]float32{{0, 1, 0}}, sparse: mvSV(1, 3.0, 2, 1.0), meta: Metadata{"g": NewString("b")}},
		3: {tokens: [][]float32{{0, 0, 1}, {1, 0, 0}}, sparse: mvSV(0, 0.2, 5, 4.0), meta: Metadata{"g": NewString("a")}},
		4: {tokens: [][]float32{{1, 1, 0}}, sparse: nil, meta: Metadata{"g": NewString("a")}}, // MaxSim-only doc
		5: {tokens: [][]float32{{0, 1, 1}}, sparse: mvSV(2, 5.0), meta: Metadata{"g": NewString("b")}},
	}
	sparseMap := map[uint64]*SparseVector{}
	for id, d := range docs {
		if _, err := m.AddCASKeyTTLSparse(id, d.tokens, d.meta, nil, d.sparse, CASCond{}); err != nil {
			t.Fatalf("add %d: %v", id, err)
		}
		if d.sparse != nil {
			sparseMap[id] = d.sparse
		}
	}
	return m, sparseMap
}

func resultsEqual(t *testing.T, got, want []Result, label string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d != want %d\n got=%+v\nwant=%+v", label, len(got), len(want), got, want)
	}
	for i := range want {
		if got[i].ID != want[i].ID {
			t.Fatalf("%s rank %d: id %d != want %d\n got=%+v\nwant=%+v", label, i, got[i].ID, want[i].ID, got, want)
		}
		if d := got[i].Score - want[i].Score; math.Abs(float64(d)) > 1e-6 {
			t.Fatalf("%s rank %d: score %v != want %v", label, i, got[i].Score, want[i].Score)
		}
	}
}

// handFuse reproduces MVHybrid's lane construction independently: it runs the public
// Search (MaxSim lane) and SearchSparse (sparse lane) at the SAME lane sizes the
// engine uses (denseK/sparseK default max(k,50)), converts the MaxSim []MultiResult
// to a score-desc []Result, then fuses with FuseScoreLanes using the same params.
// This is the ground-truth oracle: MVHybrid must equal it exactly.
func handFuse(t *testing.T, m *MultiVectorIndex, query [][]float32, sparseQ *SparseVector, k int, opts HybridOpts) []Result {
	t.Helper()
	laneK := func(knob int) int {
		if knob > 0 {
			return knob
		}
		if k < 50 {
			return 50
		}
		return k
	}
	mr, err := m.Search(query, laneK(opts.DenseK), MultiSearchOpts{Filter: opts.Filter})
	if err != nil {
		t.Fatal(err)
	}
	dense := make([]Result, len(mr))
	for i, r := range mr {
		dense[i] = Result{ID: r.ID, Score: r.Score}
	}
	// SearchSparse applies only the live-doc gate; with no filter that matches the
	// hybrid sparse lane. (Filtered cases are handled in their own test.)
	sparse := m.SearchSparse(sparseQ, laneK(opts.SparseK))
	return FuseScoreLanes(dense, sparse, opts.Method, opts.Alpha, opts.RRFK, k)
}

func TestMVHybridMatchesHandFusedGroundTruth(t *testing.T) {
	m, _ := mvHybridFixture(t)
	query := [][]float32{{1, 0, 0}, {0, 1, 0}}
	sparseQ := mvSV(0, 1.0, 2, 1.0, 5, 1.0)
	k := 5

	for _, tc := range []struct {
		name string
		opts HybridOpts
	}{
		{"rrf", HybridOpts{Method: FusionRRF}},
		{"weighted_default_alpha", HybridOpts{Method: FusionWeighted}},
		{"weighted_alpha_0.3", HybridOpts{Method: FusionWeighted, Alpha: 0.3}},
		{"rrf_rrfk_10", HybridOpts{Method: FusionRRF, RRFK: 10}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want := handFuse(t, m, query, sparseQ, k, tc.opts)
			got, err := m.MVHybrid(query, sparseQ, k, tc.opts)
			if err != nil {
				t.Fatal(err)
			}
			resultsEqual(t, got, want, tc.name)
		})
	}
}

// TestMVHybridSingleLaneDegradation: empty sparse query ⇒ MaxSim lane only; empty
// token matrix ⇒ sparse lane only.
func TestMVHybridSingleLaneDegradation(t *testing.T) {
	m, _ := mvHybridFixture(t)
	query := [][]float32{{1, 0, 0}}
	sparseQ := mvSV(0, 1.0, 2, 1.0)
	k := 3

	// Empty sparse ⇒ MaxSim only (== Search top-k, score-carried).
	t.Run("maxsim_only", func(t *testing.T) {
		mr, err := m.Search(query, k, MultiSearchOpts{})
		if err != nil {
			t.Fatal(err)
		}
		want := make([]Result, len(mr))
		for i, r := range mr {
			want[i] = Result{ID: r.ID, Score: r.Score}
		}
		got, err := m.MVHybrid(query, nil, k, HybridOpts{Method: FusionRRF})
		if err != nil {
			t.Fatal(err)
		}
		resultsEqual(t, got, want, "maxsim_only")
	})

	// Empty tokens ⇒ sparse only (== SearchSparse top-k).
	t.Run("sparse_only", func(t *testing.T) {
		want := m.SearchSparse(sparseQ, k)
		got, err := m.MVHybrid(nil, sparseQ, k, HybridOpts{Method: FusionRRF})
		if err != nil {
			t.Fatal(err)
		}
		resultsEqual(t, got, want, "sparse_only")
	})
}

// TestMVHybridFilterBothLanes: a filter narrows BOTH lanes consistently — a doc that
// fails the filter never appears in the fused result, and a doc with no sparse vector
// still contributes (only) to the MaxSim lane.
func TestMVHybridFilterBothLanes(t *testing.T) {
	m, _ := mvHybridFixture(t)
	query := [][]float32{{1, 0, 0}, {0, 1, 0}}
	sparseQ := mvSV(0, 1.0, 2, 1.0, 5, 1.0)
	k := 5
	filter := Filter{Op: FilterEq, Field: "g", Value: NewString("a")}
	opts := HybridOpts{Method: FusionRRF, Filter: filter}

	got, err := m.MVHybrid(query, sparseQ, k, opts)
	if err != nil {
		t.Fatal(err)
	}
	// Only g==a docs (1,3,4) may appear; 2 and 5 (g==b) must be absent from BOTH lanes.
	allowed := map[uint64]bool{1: true, 3: true, 4: true}
	sawMaxSimOnly4 := false
	for _, r := range got {
		if !allowed[r.ID] {
			t.Fatalf("doc %d (g=b) leaked past the filter: %+v", r.ID, got)
		}
		if r.ID == 4 {
			sawMaxSimOnly4 = true // doc 4 has no sparse vec → MaxSim-lane-only contributor
		}
	}
	if !sawMaxSimOnly4 {
		t.Fatalf("doc 4 (MaxSim-only, g=a) should appear via the MaxSim lane: %+v", got)
	}
}

// TestFuseScoreLanesWeightedNotInverted guards the lane-orientation fix: for the
// weighted path FuseScoreLanes must treat the FIRST lane as a DESC score (higher =
// better), NOT invert it the way Fuse (distance-oriented) would. With alpha=1 (first
// lane only) the highest-Score doc must rank first.
func TestFuseScoreLanesWeightedNotInverted(t *testing.T) {
	first := []Result{{ID: 10, Score: 9.0}, {ID: 20, Score: 1.0}} // 10 is the BEST MaxSim doc
	out := FuseScoreLanes(first, nil, FusionWeighted, 1.0, 0, 2)
	if len(out) != 2 || out[0].ID != 10 {
		t.Fatalf("weighted score lane inverted: want best-score doc 10 first, got %+v", out)
	}
	// Cross-check: the distance-oriented Fuse WOULD invert (treats Score-as-Distance
	// only via the .Distance field; here Distance is 0 for both so it ties). The point
	// is FuseScoreLanes uses .Score, so 10 (higher score) wins decisively.
	if out[0].Score <= out[1].Score {
		t.Fatalf("weighted score: doc 10 must outscore doc 20, got %+v", out)
	}
}

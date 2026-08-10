// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"testing"
)

// newIVFHybridCorpus builds a small IVF-Flat index mirroring newHybridCorpus:
// docs 1-5 cluster near the dense origin (weak shared sparse term), doc 100 is far
// in dense space but carries a strong sparse term. Untrained IVF brute-forces
// exactly, which is fine — the point is the dense+sparse lane build + the fusion.
func newIVFHybridCorpus(t *testing.T) *ivf {
	t.Helper()
	cfg := Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1}
	cfg.IndexType = IndexIVF
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatalf("newIVF: %v", err)
	}
	for i := uint64(1); i <= 5; i++ {
		v := []float32{float32(i) * 0.01, 0, 0, 0}
		sv := &SparseVector{Indices: []uint32{1}, Values: []float32{0.1}}
		if _, _, err := ix.Insert(i, v, 0, nil, sv, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	if _, _, err := ix.Insert(100, []float32{9, 9, 9, 9}, 0, nil,
		&SparseVector{Indices: []uint32{42}, Values: []float32{10.0}}, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 100: %v", err)
	}
	return ix
}

// TestIVFHybridDBSFTakesDBSFPath is the regression guard for the bug where
// ivf.HybridSearch had no FusionDBSF branch and silently fell back to fuseRRF (so
// method=dbsf produced RRF results for any IVF-backed dense collection). It asserts
// that ivf.HybridSearch(FusionDBSF) equals the hand-fused fuseDBSF over the index's
// own lanes — NOT fuseRRF. Before the fix this fails (DBSF result == RRF result).
func TestIVFHybridDBSFTakesDBSFPath(t *testing.T) {
	ix := newIVFHybridCorpus(t)
	dense := []float32{0, 0, 0, 0}
	sparse := SparseVector{Indices: []uint32{42}, Values: []float32{5.0}}
	const k = 5
	const alpha = 0.5

	denseLane, sparseLane, err := ix.HybridLanes(dense, sparse, k, HybridOpts{})
	if err != nil {
		t.Fatalf("HybridLanes: %v", err)
	}
	wantDBSF := fuseDBSF(denseLane, sparseLane, k, alpha)
	wantRRF := fuseRRF(denseLane, sparseLane, k, 0)

	got, err := ix.HybridSearch(dense, sparse, k, HybridOpts{Method: FusionDBSF, Alpha: alpha})
	if err != nil {
		t.Fatalf("HybridSearch DBSF: %v", err)
	}
	if !sameFused(got, wantDBSF) {
		t.Fatalf("ivf.HybridSearch(FusionDBSF) did not take the DBSF path:\n got=%v\nwant=%v", ids(got), ids(wantDBSF))
	}
	// Guard the guard: DBSF and RRF must actually differ on this corpus, else the
	// test above would pass even with the RRF-fallback bug.
	if sameFused(wantDBSF, wantRRF) {
		t.Skip("DBSF and RRF coincide on this corpus; test cannot distinguish the bug")
	}
	for _, r := range got {
		if math.IsNaN(float64(r.Score)) {
			t.Errorf("DBSF IVF hybrid NaN score for id%d", r.ID)
		}
	}
}

func ids(rs []Result) []uint64 {
	out := make([]uint64, len(rs))
	for i, r := range rs {
		out[i] = r.ID
	}
	return out
}

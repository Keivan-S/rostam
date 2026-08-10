// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"testing"
)

// hybrid setup: a small corpus where one doc is a strong sparse (lexical) match
// but a weak dense neighbor, to prove fusion surfaces it.
func newHybridCorpus(t *testing.T) *hnsw {
	t.Helper()
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	// Docs 1-5 cluster near the dense query origin; doc 100 is far in dense
	// space but carries the exact sparse term the query wants.
	for i := uint64(1); i <= 5; i++ {
		v := []float32{float32(i) * 0.01, 0, 0, 0}
		sv := &SparseVector{Indices: []uint32{1}, Values: []float32{0.1}} // weak shared term
		if _, _, err := h.Insert(i, v, 0, nil, sv, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	// Far dense vector, strong sparse term 42.
	if _, _, err := h.Insert(100, []float32{9, 9, 9, 9}, 0, nil,
		&SparseVector{Indices: []uint32{42}, Values: []float32{10.0}}, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 100: %v", err)
	}
	return h
}

func TestHybridSearchSurfacesSparseMatch(t *testing.T) {
	h := newHybridCorpus(t)
	// Query: dense near origin (favors 1-5), sparse strongly hits term 42 (doc 100).
	dense := []float32{0, 0, 0, 0}
	sparse := SparseVector{Indices: []uint32{42}, Values: []float32{5.0}}

	// Pure dense would never surface doc 100 (it's the farthest vector).
	pureDense, err := h.Search(dense, 3)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range pureDense {
		if r.ID == 100 {
			t.Fatal("doc 100 should NOT appear in pure dense top-3")
		}
	}

	// Hybrid: doc 100's exact sparse term should pull it into the results.
	got, err := h.HybridSearch(dense, sparse, 3, HybridOpts{})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, r := range got {
		if r.ID == 100 {
			found = true
		}
	}
	if !found {
		t.Errorf("hybrid search should surface doc 100 via sparse term; got %+v", got)
	}
}

func TestHybridSearchPureDenseDegradation(t *testing.T) {
	h := newHybridCorpus(t)
	dense := []float32{0, 0, 0, 0}

	// Empty sparse → must equal SearchFiltered (pure dense).
	hybrid, err := h.HybridSearch(dense, SparseVector{}, 3, HybridOpts{})
	if err != nil {
		t.Fatal(err)
	}
	plain, err := h.SearchFiltered(dense, 3, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(hybrid) != len(plain) {
		t.Fatalf("hybrid(empty sparse) len %d != dense len %d", len(hybrid), len(plain))
	}
	for i := range plain {
		if hybrid[i].ID != plain[i].ID || hybrid[i].Distance != plain[i].Distance {
			t.Errorf("result %d: hybrid %+v != dense %+v", i, hybrid[i], plain[i])
		}
	}
}

func TestHybridSearchPureSparseDegradation(t *testing.T) {
	h := newHybridCorpus(t)
	// Nil dense → pure sparse. Query term 42 → only doc 100 matches.
	got, err := h.HybridSearch(nil, SparseVector{Indices: []uint32{42}, Values: []float32{1.0}}, 5, HybridOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].ID != 100 {
		t.Errorf("pure sparse search = %+v, want [id=100]", got)
	}
}

func TestHybridSearchFilterAware(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	for i := uint64(1); i <= 10; i++ {
		tenant := "acme"
		if i%2 == 0 {
			tenant = "globex"
		}
		meta := Metadata{"tenant": NewString(tenant)}
		sv := &SparseVector{Indices: []uint32{7}, Values: []float32{1.0}}
		if _, _, err := h.Insert(i, []float32{float32(i) * 0.01, 0, 0, 0}, 0, meta, sv, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	opts := HybridOpts{Filter: Filter{Op: FilterEq, Field: "tenant", Value: NewString("acme")}}
	got, err := h.HybridSearch([]float32{0, 0, 0, 0}, SparseVector{Indices: []uint32{7}, Values: []float32{1.0}}, 10, opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 {
		t.Fatal("filtered hybrid returned nothing")
	}
	for _, r := range got {
		if r.ID%2 == 0 {
			t.Errorf("result id %d is globex (even); filter should exclude it", r.ID)
		}
	}
}

func TestHybridSearchValidatesSparse(t *testing.T) {
	h := newHybridCorpus(t)
	bad := SparseVector{Indices: []uint32{5, 1}, Values: []float32{1, 2}} // unsorted
	if _, err := h.HybridSearch([]float32{0, 0, 0, 0}, bad, 3, HybridOpts{}); err == nil {
		t.Error("HybridSearch should reject an invalid sparse query")
	}
}

func TestHybridSearchWeighted(t *testing.T) {
	h := newHybridCorpus(t)
	dense := []float32{0, 0, 0, 0}
	sparse := SparseVector{Indices: []uint32{42}, Values: []float32{5.0}}
	// alpha=0 → pure sparse weight → doc 100 should rank first.
	got, err := h.HybridSearch(dense, sparse, 5, HybridOpts{Method: FusionWeighted, Alpha: 0.001})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 || got[0].ID != 100 {
		t.Errorf("weighted (sparse-heavy) top = %+v, want doc 100 first", got)
	}
}

func TestHybridLanesMatchHybridSearch(t *testing.T) {
	h := newHybridCorpus(t)
	dense := []float32{0, 0, 0, 0}
	// Sparse query hits term 42 (only doc 100) and term 1 (docs 1-5) — overlap
	// and both lanes contribute results, making this a meaningful fusion test.
	sparse := SparseVector{Indices: []uint32{1, 42}, Values: []float32{0.5, 5.0}}
	for _, method := range []FusionMethod{FusionRRF, FusionWeighted} {
		opts := HybridOpts{Method: method, DenseK: 50, SparseK: 50}
		want, err := h.HybridSearch(dense, sparse, 5, opts)
		if err != nil {
			t.Fatal(err)
		}
		dl, sl, err := h.HybridLanes(dense, sparse, 5, opts)
		if err != nil {
			t.Fatal(err)
		}
		got := Fuse(dl, sl, method, opts.Alpha, opts.RRFK, 5)
		if !sameFused(want, got) {
			t.Fatalf("method=%v: lanes+Fuse %v != HybridSearch %v", method, got, want)
		}
	}
}

func TestHybridLanesShapes(t *testing.T) {
	// Use a dedicated index with 2+ docs matching sparse term 42 so the
	// descending-by-Score assertion on the sparse lane is meaningful.
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	// Background dense docs (term 1 only — won't match sparse query term 42).
	for i := uint64(1); i <= 5; i++ {
		v := []float32{float32(i) * 0.01, 0, 0, 0}
		sv := &SparseVector{Indices: []uint32{1}, Values: []float32{0.1}}
		if _, _, err := h.Insert(i, v, 0, nil, sv, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	// Two docs with different scores on term 42: ensures sparse lane has ≥2
	// elements and makes the descending-sort check non-vacuous.
	if _, _, err := h.Insert(100, []float32{9, 9, 9, 9}, 0, nil,
		&SparseVector{Indices: []uint32{42}, Values: []float32{10.0}}, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 100: %v", err)
	}
	if _, _, err := h.Insert(101, []float32{8, 8, 8, 8}, 0, nil,
		&SparseVector{Indices: []uint32{42}, Values: []float32{3.0}}, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 101: %v", err)
	}

	dense := []float32{0, 0, 0, 0}
	sparse := SparseVector{Indices: []uint32{42}, Values: []float32{5.0}}
	dl, sl, err := h.HybridLanes(dense, sparse, 5, HybridOpts{DenseK: 50, SparseK: 50})
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i < len(dl); i++ {
		if dl[i].Distance < dl[i-1].Distance {
			t.Fatal("dense lane not ascending by Distance")
		}
	}
	if len(sl) < 2 {
		t.Fatalf("sparse lane has %d element(s); need ≥2 to verify sort order", len(sl))
	}
	for i := 1; i < len(sl); i++ {
		if sl[i].Score > sl[i-1].Score {
			t.Fatalf("sparse lane not descending by Score: sl[%d].Score=%v > sl[%d].Score=%v",
				i, sl[i].Score, i-1, sl[i-1].Score)
		}
	}
	if len(dl) > 50 || len(sl) > 50 {
		t.Fatal("lanes exceed DenseK/SparseK")
	}
	if len(dl) == 0 || len(sl) == 0 {
		t.Fatal("expected both lanes populated for a dense+sparse query")
	}
}

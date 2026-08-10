// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"testing"
)

// eqResults compares two []Result by (ID, Score, Distance) in order — the fused
// hybrid result must match the hand-fused ground truth exactly.
func eqResults(a, b []Result) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].ID != b[i].ID || a[i].Score != b[i].Score || a[i].Distance != b[i].Distance {
			return false
		}
	}
	return true
}

// namedHybridFixture builds a collection with a dense space "title" (dim 4 cosine)
// and a sparse space "terms", inserts overlapping + disjoint points across the two
// spaces, and returns it plus the dense + sparse queries used by the tests.
func namedHybridFixture(t *testing.T) (nc *NamedCollection, denseQ []float32, sparseQ *SparseVector) {
	t.Helper()
	nc, err := NewNamedCollection("c", namedSparseTestConfig())
	if err != nil {
		t.Fatal(err)
	}
	// Each point gets BOTH a dense title vector and a sparse terms vector, so both
	// lanes can score it; ids 5/6 are sparse-only / dense-only to exercise union.
	insert := func(id uint64, dense []float32, terms *SparseVector) {
		t.Helper()
		v := map[string][]float32{}
		if dense != nil {
			v["title"] = dense
		}
		sp := map[string]*SparseVector{}
		if terms != nil {
			sp["terms"] = terms
		}
		if err := nc.InsertSparse(id, v, sp, Metadata{"kind": NewString("a")}, 0); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}
	insert(1, []float32{1, 0, 0, 0}, sv([]uint32{0, 2}, []float32{1, 2}))
	insert(2, []float32{0, 1, 0, 0}, sv([]uint32{2, 5}, []float32{3, 1}))
	insert(3, []float32{0.9, 0.1, 0, 0}, sv([]uint32{0, 9}, []float32{2, 2}))
	insert(4, []float32{0, 0, 1, 0}, sv([]uint32{5}, []float32{5}))
	insert(5, nil, sv([]uint32{2, 5}, []float32{4, 4})) // sparse-only
	insert(6, []float32{0.8, 0.2, 0, 0}, nil)           // dense-only
	return nc, []float32{1, 0, 0, 0}, sv([]uint32{2, 5}, []float32{2, 1})
}

// TestNamedHybridMatchesHandFusedRRF runs the two lanes separately, fuses with the
// SAME Fuse params, and asserts NamedHybrid produces the identical fused result.
func TestNamedHybridMatchesHandFusedRRF(t *testing.T) {
	nc, denseQ, sparseQ := namedHybridFixture(t)
	k := 4
	opts := HybridOpts{Method: FusionRRF}

	// Hand-fused ground truth: run each lane via the SAME engine surfaces NamedHybrid
	// uses internally (pooled to the same DenseK/SparseK = max(k,50)), then Fuse.
	dense, err := nc.SearchNamed("title", denseQ, namedHybridK(opts.DenseK, k), Filter{})
	if err != nil {
		t.Fatal(err)
	}
	sparse, err := nc.SearchNamedSparse("terms", sparseQ, namedHybridK(opts.SparseK, k), Filter{})
	if err != nil {
		t.Fatal(err)
	}
	want := Fuse(dense, sparse, opts.Method, opts.Alpha, opts.RRFK, k)

	got, err := nc.NamedHybrid("title", denseQ, "terms", sparseQ, k, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !eqResults(got, want) {
		t.Fatalf("NamedHybrid RRF = %+v\nwant (hand-fused) %+v", got, want)
	}
}

// TestNamedHybridMatchesHandFusedWeighted is the weighted-fusion (alpha) analogue.
func TestNamedHybridMatchesHandFusedWeighted(t *testing.T) {
	nc, denseQ, sparseQ := namedHybridFixture(t)
	k := 5
	opts := HybridOpts{Method: FusionWeighted, Alpha: 0.3}

	dense, err := nc.SearchNamed("title", denseQ, namedHybridK(opts.DenseK, k), Filter{})
	if err != nil {
		t.Fatal(err)
	}
	sparse, err := nc.SearchNamedSparse("terms", sparseQ, namedHybridK(opts.SparseK, k), Filter{})
	if err != nil {
		t.Fatal(err)
	}
	want := Fuse(dense, sparse, opts.Method, opts.Alpha, opts.RRFK, k)

	got, err := nc.NamedHybrid("title", denseQ, "terms", sparseQ, k, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !eqResults(got, want) {
		t.Fatalf("NamedHybrid weighted = %+v\nwant (hand-fused) %+v", got, want)
	}
}

// TestNamedHybridSingleLaneDegradation: an empty sparse query returns the dense
// lane alone; an empty dense query returns the sparse lane alone.
func TestNamedHybridSingleLaneDegradation(t *testing.T) {
	nc, denseQ, sparseQ := namedHybridFixture(t)
	k := 3

	// Empty sparse query ⇒ dense lane only (truncated to k).
	denseOnly, err := nc.NamedHybrid("title", denseQ, "terms", nil, k, HybridOpts{})
	if err != nil {
		t.Fatal(err)
	}
	wantDense, err := nc.SearchNamed("title", denseQ, k, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if !eqIDs(idsOf(denseOnly), idsOf(wantDense)) {
		t.Fatalf("dense-only degradation = %v, want %v", idsOf(denseOnly), idsOf(wantDense))
	}

	// Empty dense query ⇒ sparse lane only (truncated to k).
	sparseOnly, err := nc.NamedHybrid("title", nil, "terms", sparseQ, k, HybridOpts{})
	if err != nil {
		t.Fatal(err)
	}
	wantSparse, err := nc.SearchNamedSparse("terms", sparseQ, k, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if !eqIDs(idsOf(sparseOnly), idsOf(wantSparse)) {
		t.Fatalf("sparse-only degradation = %v, want %v", idsOf(sparseOnly), idsOf(wantSparse))
	}
}

// TestNamedHybridFilterBothLanes: a payload filter must gate BOTH lanes identically
// to the hand-fused ground truth computed with the same filter.
func TestNamedHybridFilterBothLanes(t *testing.T) {
	nc, err := NewNamedCollection("c", namedSparseTestConfig())
	if err != nil {
		t.Fatal(err)
	}
	kinds := map[uint64]string{1: "a", 2: "b", 3: "a", 4: "b"}
	for id := uint64(1); id <= 4; id++ {
		// Distinct dense vectors so L2/cosine distances never tie (a tie would make the
		// two independent SearchNamed calls' rank order non-deterministic across runs).
		dense := []float32{1, float32(id) * 0.1, float32(id) * 0.03, float32(id) * 0.05}
		terms := sv([]uint32{2, 5}, []float32{float32(id), 1})
		if err := nc.InsertSparse(id, map[string][]float32{"title": dense},
			map[string]*SparseVector{"terms": terms}, Metadata{"kind": NewString(kinds[id])}, 0); err != nil {
			t.Fatal(err)
		}
	}
	denseQ := []float32{1, 0, 0, 0}
	sparseQ := sv([]uint32{2, 5}, []float32{2, 1})
	filter := Filter{Op: FilterEq, Field: "kind", Value: NewString("a")}
	k := 4
	opts := HybridOpts{Method: FusionRRF, Filter: filter}

	dense, err := nc.SearchNamed("title", denseQ, namedHybridK(0, k), filter)
	if err != nil {
		t.Fatal(err)
	}
	sparse, err := nc.SearchNamedSparse("terms", sparseQ, namedHybridK(0, k), filter)
	if err != nil {
		t.Fatal(err)
	}
	want := Fuse(dense, sparse, opts.Method, opts.Alpha, opts.RRFK, k)

	got, err := nc.NamedHybrid("title", denseQ, "terms", sparseQ, k, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !eqResults(got, want) {
		t.Fatalf("filtered NamedHybrid = %+v\nwant %+v", got, want)
	}
	for _, r := range got {
		if kinds[r.ID] != "a" {
			t.Fatalf("filter leaked id %d (kind %q)", r.ID, kinds[r.ID])
		}
	}
}

// TestNamedHybridModalityMismatch: a dense space named as the sparse lane (or a
// sparse space named as the dense lane, or an unknown space) is fail-loud.
func TestNamedHybridModalityMismatch(t *testing.T) {
	nc, denseQ, sparseQ := namedHybridFixture(t)
	// denseSpace points at the sparse space "terms" ⇒ modality mismatch.
	if _, err := nc.NamedHybrid("terms", denseQ, "terms", sparseQ, 3, HybridOpts{}); !errors.Is(err, ErrSpaceModalityMismatch) {
		t.Fatalf("dense lane on sparse space: got %v, want ErrSpaceModalityMismatch", err)
	}
	// sparseSpace points at the dense space "title" ⇒ modality mismatch.
	if _, err := nc.NamedHybrid("title", denseQ, "title", sparseQ, 3, HybridOpts{}); !errors.Is(err, ErrSpaceModalityMismatch) {
		t.Fatalf("sparse lane on dense space: got %v, want ErrSpaceModalityMismatch", err)
	}
	// Unknown space ⇒ ErrUnknownVectorName.
	if _, err := nc.NamedHybrid("nope", denseQ, "terms", sparseQ, 3, HybridOpts{}); !errors.Is(err, ErrUnknownVectorName) {
		t.Fatalf("unknown dense space: got %v, want ErrUnknownVectorName", err)
	}
	if _, err := nc.NamedHybrid("title", denseQ, "nope", sparseQ, 3, HybridOpts{}); !errors.Is(err, ErrUnknownVectorName) {
		t.Fatalf("unknown sparse space: got %v, want ErrUnknownVectorName", err)
	}
}

// TestNamedHybridLanesUnfused: NamedHybridLanes returns the two raw lanes that fuse
// (via Fuse) to the same result NamedHybrid produces — the partition-fan-out oracle.
func TestNamedHybridLanesUnfused(t *testing.T) {
	nc, denseQ, sparseQ := namedHybridFixture(t)
	k := 4
	opts := HybridOpts{Method: FusionRRF}

	dense, sparse, err := nc.NamedHybridLanes("title", denseQ, "terms", sparseQ, k, opts)
	if err != nil {
		t.Fatal(err)
	}
	fused := Fuse(dense, sparse, opts.Method, opts.Alpha, opts.RRFK, k)

	got, err := nc.NamedHybrid("title", denseQ, "terms", sparseQ, k, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !eqResults(got, fused) {
		t.Fatalf("Fuse(NamedHybridLanes) = %+v\nwant NamedHybrid %+v", fused, got)
	}
}

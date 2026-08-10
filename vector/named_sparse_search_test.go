// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"testing"
	"time"
)

// idsOf extracts the ids from a []Result in order (the ranked order searchTopK /
// SearchNamedSparse returns).
func idsOf(res []Result) []uint64 {
	out := make([]uint64, len(res))
	for i, r := range res {
		out[i] = r.ID
	}
	return out
}

func eqIDs(a []uint64, b []uint64) bool {
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

// TestSearchNamedSparseTopK inserts sparse vectors into a sparse named space and
// checks SearchNamedSparse returns the correct dot-product top-k versus a
// brute-force ground truth.
func TestSearchNamedSparseTopK(t *testing.T) {
	nc, err := NewNamedCollection("c", namedSparseTestConfig())
	if err != nil {
		t.Fatal(err)
	}
	vecs := map[uint64]*SparseVector{
		1: sv([]uint32{0, 2, 5}, []float32{1, 1, 1}),
		2: sv([]uint32{2, 5}, []float32{3, 1}),
		3: sv([]uint32{0, 9}, []float32{2, 2}),
		4: sv([]uint32{5}, []float32{5}),
	}
	for id, v := range vecs {
		if err := nc.InsertSparse(id, nil, map[string]*SparseVector{"terms": v}, Metadata{"k": NewInt(int64(id))}, 0); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}
	query := sv([]uint32{2, 5}, []float32{2, 1})
	got, err := nc.SearchNamedSparse("terms", query, 3, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	want := bruteForceSparseTopK(vecs, query, 3, nil)
	if !eqIDs(idsOf(got), want) {
		t.Fatalf("top-k = %v, want %v", idsOf(got), want)
	}
}

// TestSearchNamedSparseFilter checks a payload filter on a sparse search admits
// only matching points (same admit gate as dense SearchNamed).
func TestSearchNamedSparseFilter(t *testing.T) {
	nc, err := NewNamedCollection("c", namedSparseTestConfig())
	if err != nil {
		t.Fatal(err)
	}
	vecs := map[uint64]*SparseVector{
		1: sv([]uint32{2, 5}, []float32{2, 2}),
		2: sv([]uint32{2, 5}, []float32{3, 1}),
		3: sv([]uint32{2, 5}, []float32{4, 4}),
	}
	kinds := map[uint64]string{1: "a", 2: "b", 3: "a"}
	for id, v := range vecs {
		if err := nc.InsertSparse(id, nil, map[string]*SparseVector{"terms": v}, Metadata{"kind": NewString(kinds[id])}, 0); err != nil {
			t.Fatal(err)
		}
	}
	query := sv([]uint32{2, 5}, []float32{1, 1})
	filter := Filter{Op: FilterEq, Field: "kind", Value: NewString("a")}
	got, err := nc.SearchNamedSparse("terms", query, 10, filter)
	if err != nil {
		t.Fatal(err)
	}
	// Brute force with the same admit (kind=="a") for ground truth.
	admit := func(id uint64) bool { return kinds[id] == "a" }
	want := bruteForceSparseTopK(vecs, query, 10, admit)
	if !eqIDs(idsOf(got), want) {
		t.Fatalf("filtered top-k = %v, want %v", idsOf(got), want)
	}
	for _, r := range got {
		if kinds[r.ID] != "a" {
			t.Fatalf("filter leaked id %d (kind %q)", r.ID, kinds[r.ID])
		}
	}
}

// TestSearchNamedSparseTTLExpired checks a TTL-expired point is not scored (same
// liveness gate as dense SearchNamed).
func TestSearchNamedSparseTTLExpired(t *testing.T) {
	now := int64(1_000_000)
	nc, err := NewNamedCollection("c", namedSparseTestConfig())
	if err != nil {
		t.Fatal(err)
	}
	nc.now = func() int64 { return now }
	q := map[string]*SparseVector{"terms": sv([]uint32{1}, []float32{1})}
	if err := nc.InsertSparse(1, nil, q, nil, 10*time.Millisecond); err != nil {
		t.Fatal(err)
	}
	if err := nc.InsertSparse(2, nil, q, nil, 0); err != nil {
		t.Fatal(err)
	}
	now += 100 // expire point 1
	got, err := nc.SearchNamedSparse("terms", sv([]uint32{1}, []float32{1}), 10, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if ids := idsOf(got); len(ids) != 1 || ids[0] != 2 {
		t.Fatalf("expired point leaked: got %v, want [2]", ids)
	}
}

// TestSearchNamedSparseModality covers the fail-loud cases: a sparse search against
// a DENSE space is ErrSpaceModalityMismatch; an unknown space is
// ErrUnknownVectorName.
func TestSearchNamedSparseModality(t *testing.T) {
	nc, err := NewNamedCollection("c", namedSparseTestConfig())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nc.SearchNamedSparse("title", sv([]uint32{0}, []float32{1}), 5, Filter{}); !errors.Is(err, ErrSpaceModalityMismatch) {
		t.Fatalf("dense space sparse search: got %v, want ErrSpaceModalityMismatch", err)
	}
	if _, err := nc.SearchNamedSparse("nope", sv([]uint32{0}, []float32{1}), 5, Filter{}); !errors.Is(err, ErrUnknownVectorName) {
		t.Fatalf("unknown space: got %v, want ErrUnknownVectorName", err)
	}
}

// TestSearchNamedSparseDocs checks the docs-shaped variant carries the shared
// payload and the same ranking.
func TestSearchNamedSparseDocs(t *testing.T) {
	nc, err := NewNamedCollection("c", namedSparseTestConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err := nc.InsertSparse(7, nil, map[string]*SparseVector{"terms": sv([]uint32{3}, []float32{2})}, Metadata{"tag": NewString("x")}, 0); err != nil {
		t.Fatal(err)
	}
	docs, err := nc.SearchNamedSparseDocs("terms", sv([]uint32{3}, []float32{1}), 5, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 1 || docs[0].ID != 7 {
		t.Fatalf("docs = %+v, want one doc id 7", docs)
	}
	if got := docs[0].Metadata["tag"]; got.Str != "x" {
		t.Fatalf("doc payload tag = %v, want x", got)
	}
}

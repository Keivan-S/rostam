// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"testing"
	"time"
)

func findDoc(docs []Document, id uint64) (Document, bool) {
	for _, d := range docs {
		if d.ID == id {
			return d, true
		}
	}
	return Document{}, false
}

func ragCollection(t *testing.T) *Collection {
	t.Helper()
	c, err := NewCollection("docs", Config{Dim: 8, Metric: Cosine, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// TestRAGUpsertSearchDocs checks the core retrieval path: Upsert stores content +
// metadata, SearchDocs returns them, and the reserved content field does not leak
// into the caller-visible metadata.
func TestRAGUpsertSearchDocs(t *testing.T) {
	c := ragCollection(t)
	defer func() { _ = c.Close() }()

	_, vecs := siftLikeCorpus(200, 8, 3)
	for i, v := range vecs {
		normalize(v)
		meta := Metadata{"ord": NewInt(int64(i)), "src": NewString("corpus")}
		if err := c.Upsert(uint64(i+1), v, fmt.Sprintf("chunk-%d", i+1), 0, meta, nil); err != nil {
			t.Fatalf("upsert %d: %v", i+1, err)
		}
	}

	docs, err := c.SearchDocs(vecs[0], 5, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) == 0 {
		t.Fatal("SearchDocs returned nothing")
	}
	d1, ok := findDoc(docs, 1) // query == vecs[0] -> id 1 is its own nearest neighbor
	if !ok {
		t.Fatalf("id 1 not in results: %+v", docs)
	}
	if d1.Content != "chunk-1" {
		t.Errorf("content = %q, want chunk-1", d1.Content)
	}
	if d1.Metadata == nil || d1.Metadata["ord"].Int != 0 || d1.Metadata["src"].Str != "corpus" {
		t.Errorf("metadata = %v", d1.Metadata)
	}
	if _, leaked := d1.Metadata[contentField]; leaked {
		t.Error("reserved content field leaked into returned metadata")
	}
}

// findScan returns the ScanRecord for id, or ok=false.
func findScan(recs []ScanRecord, id uint64) (ScanRecord, bool) {
	for _, r := range recs {
		if r.ID == id {
			return r, true
		}
	}
	return ScanRecord{}, false
}

// TestScanVectors verifies the resplit read primitive: ScanVectors returns
// exactly the live records with their exact vectors, metadata, and sparse
// vectors; a deleted id is excluded; and the returned vector is a COPY (mutating
// it does not corrupt the index).
func TestScanVectors(t *testing.T) {
	c, err := NewCollection("scan", Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	const n = 5
	vecs := map[uint64][]float32{
		1: {1, 0, 0, 0},
		2: {0, 1, 0, 0},
		3: {0, 0, 1, 0},
		4: {0, 0, 0, 1},
		5: {1, 1, 0, 0},
	}
	sparse3 := &SparseVector{Indices: []uint32{2, 7}, Values: []float32{0.5, 0.25}}
	for id := uint64(1); id <= n; id++ {
		meta := Metadata{"ord": NewInt(int64(id)), "src": NewString("corpus")}
		var sp *SparseVector
		if id == 3 {
			sp = sparse3
		}
		if err := c.Insert(id, append([]float32(nil), vecs[id]...), 0, meta, sp); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}
	// Delete one — it must not appear in the scan.
	if !c.Delete(2) {
		t.Fatal("delete 2 failed")
	}

	recs := c.ScanVectors()
	if len(recs) != n-1 {
		t.Fatalf("ScanVectors returned %d records, want %d", len(recs), n-1)
	}
	if _, ok := findScan(recs, 2); ok {
		t.Error("deleted id 2 present in scan")
	}

	// Exact vector + metadata for a known id.
	r1, ok := findScan(recs, 1)
	if !ok {
		t.Fatal("id 1 missing from scan")
	}
	for i, want := range vecs[1] {
		if r1.Vec[i] != want {
			t.Errorf("id 1 vec[%d] = %v, want %v", i, r1.Vec[i], want)
		}
	}
	if r1.Metadata["ord"].Int != 1 || r1.Metadata["src"].Str != "corpus" {
		t.Errorf("id 1 metadata = %v", r1.Metadata)
	}
	if r1.Sparse != nil {
		t.Errorf("id 1 sparse = %v, want nil", r1.Sparse)
	}

	// Sparse round-trips for id 3.
	r3, ok := findScan(recs, 3)
	if !ok {
		t.Fatal("id 3 missing from scan")
	}
	if r3.Sparse == nil {
		t.Fatal("id 3 sparse missing")
	}
	if len(r3.Sparse.Indices) != 2 || r3.Sparse.Indices[0] != 2 || r3.Sparse.Indices[1] != 7 ||
		r3.Sparse.Values[0] != 0.5 || r3.Sparse.Values[1] != 0.25 {
		t.Errorf("id 3 sparse = %+v", r3.Sparse)
	}

	// The returned vector must be a copy: mutating it must not corrupt the index.
	r1.Vec[0] = 999
	res, err := c.Search([]float32{1, 0, 0, 0}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].ID != 1 || res[0].Distance != 0 {
		t.Errorf("after mutating scanned vec, search for id 1 = %+v (index corrupted?)", res)
	}
}

// TestScanVectorsTTL verifies a record's remaining TTL is exported as a positive
// duration (reconstructed from the absolute deadline).
func TestScanVectorsTTL(t *testing.T) {
	c, err := NewCollection("scanttl", Config{Dim: 2, Metric: L2, M: 16, EfConstruction: 50, EfSearch: 32, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	if err := c.Insert(1, []float32{1, 0}, time.Hour, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := c.Insert(2, []float32{0, 1}, 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	recs := c.ScanVectors()
	r1, _ := findScan(recs, 1)
	if r1.TTL <= 0 || r1.TTL > time.Hour {
		t.Errorf("id 1 TTL = %v, want (0, 1h]", r1.TTL)
	}
	r2, _ := findScan(recs, 2)
	if r2.TTL != 0 {
		t.Errorf("id 2 TTL = %v, want 0", r2.TTL)
	}
}

// TestRAGUpsertReplaces verifies Upsert on an existing id replaces its vector,
// content, and metadata (no stale content), keeping the live count at 1.
func TestRAGUpsertReplaces(t *testing.T) {
	c := ragCollection(t)
	defer func() { _ = c.Close() }()

	a := []float32{1, 0, 0, 0, 0, 0, 0, 0}
	b := []float32{0, 1, 0, 0, 0, 0, 0, 0}
	normalize(a)
	normalize(b)
	if err := c.Upsert(1, a, "old", 0, Metadata{"v": NewInt(1)}, nil); err != nil {
		t.Fatal(err)
	}
	if err := c.Upsert(1, b, "new", 0, Metadata{"v": NewInt(2)}, nil); err != nil {
		t.Fatal(err)
	}
	if got := c.Stats().Size; got != 1 {
		t.Errorf("live size after replace = %d, want 1", got)
	}
	docs, err := c.SearchDocs(b, 1, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 1 || docs[0].ID != 1 {
		t.Fatalf("results = %+v", docs)
	}
	if docs[0].Content != "new" || docs[0].Metadata["v"].Int != 2 {
		t.Errorf("after replace got content=%q meta=%v, want new/2", docs[0].Content, docs[0].Metadata)
	}
}

// TestRAGDeleteByFilter checks deleting all records matching a filter (a
// document's chunks), the returned count, survivors, and the empty-filter guard.
func TestRAGDeleteByFilter(t *testing.T) {
	c := ragCollection(t)
	defer func() { _ = c.Close() }()

	_, vecs := siftLikeCorpus(300, 8, 5)
	for i, v := range vecs {
		normalize(v)
		doc := int64((i % 3) + 1) // doc 1, 2, 3
		if err := c.Upsert(uint64(i+1), v, fmt.Sprintf("c%d", i+1), 0, Metadata{"doc": NewInt(doc)}, nil); err != nil {
			t.Fatal(err)
		}
	}

	// Empty filter is rejected (no accidental wipe).
	if _, err := c.DeleteByFilter(Filter{}); err != ErrEmptyFilter {
		t.Errorf("empty-filter DeleteByFilter err = %v, want ErrEmptyFilter", err)
	}

	n, err := c.DeleteByFilter(Filter{Op: FilterEq, Field: "doc", Value: NewInt(2)})
	if err != nil {
		t.Fatal(err)
	}
	want := 0
	for i := 0; i < 300; i++ {
		if (i%3)+1 == 2 {
			want++
		}
	}
	if n != want {
		t.Errorf("deleted %d, want %d", n, want)
	}
	if got := c.Stats().Size; got != 300-want {
		t.Errorf("live size after delete = %d, want %d", got, 300-want)
	}
	// No surviving record has doc==2.
	docs, err := c.SearchDocs(vecs[1], 300, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range docs {
		if d.Metadata["doc"].Int == 2 {
			t.Errorf("doc==2 survived delete-by-filter: id %d", d.ID)
		}
	}
}

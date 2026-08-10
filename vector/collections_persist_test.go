// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// TestCollectionStorePersistentRestart exercises the public wrapper end to end:
// create a Persistent collection, bulk-load it, Flush, close the store, then
// reopen the store from the same dir — which must instant-restart the
// collection (map its files, no rebuild) and return byte-identical results.
func TestCollectionStorePersistentRestart(t *testing.T) {
	const n, dim, k = 3000, 32, 10
	dir := t.TempDir()

	ids, vecs := siftLikeCorpus(n, dim, 4)
	for _, v := range vecs {
		normalize(v)
	}
	_, queries := siftLikeCorpus(80, dim, 5)
	for _, q := range queries {
		normalize(q)
	}

	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{
		Dim: dim, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1,
		Quant: QuantSQ8, RescoreFactor: 3, Persistent: true,
	}
	if err := cs.CreateCollection("docs", cfg); err != nil {
		t.Fatalf("create: %v", err)
	}
	coll, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing after create")
	}
	if err := coll.BuildConcurrent(ids, vecs, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatalf("build: %v", err)
	}

	before := make([][]uint64, len(queries))
	for i, q := range queries {
		res, err := coll.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		before[i] = resultIDs(res)
	}

	if err := cs.Flush("docs"); err != nil {
		t.Fatalf("flush: %v", err)
	}
	if err := cs.Close(); err != nil {
		t.Fatalf("close store: %v", err)
	}

	// The store-managed files must exist under default/<col>.*.
	for _, suffix := range []string{".json", ".vecs", ".graph", ".meta"} {
		p := filepath.Join(dir, "vectors", "default", "docs"+suffix)
		if _, err := os.Stat(p); err != nil {
			t.Errorf("expected persisted file %s: %v", p, err)
		}
	}

	// Reopen: this instant-restarts the persistent collection (no rebuild).
	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen store: %v", err)
	}
	defer func() { _ = cs2.Close() }()
	coll2, ok := cs2.Get("docs")
	if !ok {
		t.Fatal("collection missing after reopen")
	}
	for i, q := range queries {
		res, err := coll2.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		if !eqUint64(resultIDs(res), before[i]) {
			t.Errorf("query %d after restart: %v != %v", i, resultIDs(res), before[i])
			break
		}
	}
}

// TestCollectionStorePersistentMetadata checks the public path with metadata:
// insert vectors+metadata into a Persistent collection, Flush, reopen the store,
// and confirm a filtered search returns identical results (payload index rebuilt
// on instant restart).
func TestCollectionStorePersistentMetadata(t *testing.T) {
	const dim, k = 16, 10
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{
		Dim: dim, Metric: Cosine, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1,
		Quant: QuantSQ8, RescoreFactor: 3, Persistent: true,
	}
	if err := cs.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	_, vecs := siftLikeCorpus(400, dim, 2)
	for i, v := range vecs {
		normalize(v)
		meta := Metadata{"cat": NewInt(int64(i % 5))}
		if err := cs.Insert("docs", uint64(i+1), v, 0, meta, nil); err != nil {
			t.Fatal(err)
		}
	}
	flt := Filter{Op: FilterEq, Field: "cat", Value: NewInt(3)}
	before, err := cs.SearchFiltered("docs", vecs[0], k, flt)
	if err != nil {
		t.Fatal(err)
	}
	if len(before) == 0 {
		t.Fatal("pre-restart filtered search returned nothing")
	}
	if err := cs.Flush("docs"); err != nil {
		t.Fatal(err)
	}
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs2.Close() }()
	after, err := cs2.SearchFiltered("docs", vecs[0], k, flt)
	if err != nil {
		t.Fatal(err)
	}
	if !eqUint64(resultIDs(before), resultIDs(after)) {
		t.Errorf("filtered search after restart: %v != %v", resultIDs(after), resultIDs(before))
	}
}

// TestCollectionStorePersistentValidation rejects a Persistent collection
// without a quantizer (mmap-backed vectors need one).
func TestCollectionStorePersistentValidation(t *testing.T) {
	cs, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()
	err = cs.CreateCollection("bad", Config{
		Dim: 8, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Persistent: true, // Quant defaults to QuantNone
	})
	if !errors.Is(err, ErrInvalidPersistent) {
		t.Errorf("create persistent w/o quantizer = %v, want ErrInvalidPersistent", err)
	}
}

// TestCollectionStorePersistentCreatedNotFlushed checks a persistent collection
// created but never flushed reopens as a fresh empty index (no sidecar yet),
// rather than failing.
func TestCollectionStorePersistentCreatedNotFlushed(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{Dim: 8, Metric: Cosine, M: 8, EfConstruction: 50, EfSearch: 50, Quant: QuantSQ8, RescoreFactor: 2, Persistent: true}
	if err := cs.CreateCollection("fresh", cfg); err != nil {
		t.Fatal(err)
	}
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen (no sidecar): %v", err)
	}
	defer func() { _ = cs2.Close() }()
	c, ok := cs2.Get("fresh")
	if !ok {
		t.Fatal("collection missing after reopen")
	}
	res, err := c.Search(make([]float32, 8), 5)
	if err != nil {
		t.Fatalf("search empty restored collection: %v", err)
	}
	if len(res) != 0 {
		t.Errorf("empty collection returned %d results", len(res))
	}
}

// TestCollectionStorePersistentDropRemovesFiles checks DropCollection removes
// the managed mmap + sidecar files.
func TestCollectionStorePersistentDropRemovesFiles(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()
	cfg := Config{Dim: 8, Metric: Cosine, M: 8, EfConstruction: 50, EfSearch: 50, Quant: QuantSQ8, RescoreFactor: 2, Persistent: true}
	if err := cs.CreateCollection("tmp", cfg); err != nil {
		t.Fatal(err)
	}
	v := make([]float32, 8)
	v[0] = 1
	_ = cs.Insert("tmp", 1, v, 0, nil, nil)
	if err := cs.Flush("tmp"); err != nil {
		t.Fatal(err)
	}
	if err := cs.DropCollection("tmp"); err != nil {
		t.Fatal(err)
	}
	for _, suffix := range []string{".vecs", ".graph", ".meta", ".json"} {
		p := filepath.Join(dir, "vectors", "default", "tmp"+suffix)
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Errorf("file %s should be removed after drop (err=%v)", p, err)
		}
	}
}

// TestCollectionStorePersistentDeleteByFilter exercises the lifted constraint
// end to end: a Persistent collection that has had records removed via
// DeleteByFilter (leaving tombstones) is flushed, the store is closed, and on
// reopen the deleted docs are gone while the survivors round-trip.
func TestCollectionStorePersistentDeleteByFilter(t *testing.T) {
	const dim, k = 16, 10
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{
		Dim: dim, Metric: Cosine, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1,
		Quant: QuantSQ8, RescoreFactor: 3, Persistent: true,
	}
	if err := cs.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}

	// 60 chunks across 3 documents (doc = id % 3); content tags the survivor set.
	_, vecs := siftLikeCorpus(60, dim, 7)
	for i, v := range vecs {
		normalize(v)
		id := uint64(i + 1)
		meta := Metadata{"doc": NewInt(int64(id % 3))}
		if err := cs.Upsert("docs", id, v, "chunk", 0, meta, nil); err != nil {
			t.Fatalf("upsert %d: %v", id, err)
		}
	}

	// Purge every chunk of doc 0.
	removed, err := cs.DeleteByFilter("docs", Filter{Op: FilterEq, Field: "doc", Value: NewInt(0)})
	if err != nil {
		t.Fatalf("delete by filter: %v", err)
	}
	if removed == 0 {
		t.Fatal("delete by filter removed nothing")
	}

	if err := cs.Flush("docs"); err != nil {
		t.Fatalf("flush after delete: %v", err)
	}
	if err := cs.Close(); err != nil {
		t.Fatal(err)
	}

	// Reopen (instant restart) and confirm the purge survived persistence.
	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = cs2.Close() }()

	for _, q := range vecs {
		docs, err := cs2.SearchDocs("docs", q, k, Filter{})
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range docs {
			if d.ID%3 == 0 {
				t.Fatalf("deleted doc-0 chunk id %d resurfaced after reopen", d.ID)
			}
			if d.Content != "chunk" {
				t.Errorf("survivor id %d lost content: %q", d.ID, d.Content)
			}
		}
	}

	// A filter for the purged document must now match nothing.
	gone, err := cs2.SearchDocs("docs", vecs[0], k, Filter{Op: FilterEq, Field: "doc", Value: NewInt(0)})
	if err != nil {
		t.Fatal(err)
	}
	if len(gone) != 0 {
		t.Errorf("purged document still returns %d chunks after reopen", len(gone))
	}
}

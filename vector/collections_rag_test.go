// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"testing"
)

// ragCorpus builds n normalized dim-d vectors.
func ragCorpus(n, dim int, seed int64) [][]float32 {
	_, vecs := siftLikeCorpus(n, dim, seed)
	for _, v := range vecs {
		normalize(v)
	}
	return vecs
}

// TestRAGStoreSnapshotPersistsContent: a plain (snapshot-checkpointed)
// collection's document content + metadata survive Flush and reopen.
func TestRAGStoreSnapshotPersistsContent(t *testing.T) {
	const dim, k = 16, 10
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{Dim: dim, Metric: Cosine, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1}
	if err := cs.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	vecs := ragCorpus(200, dim, 4)
	for i, v := range vecs {
		if err := cs.Upsert("docs", uint64(i+1), v, fmt.Sprintf("chunk-%d", i+1), 0, Metadata{"src": NewString("s")}, nil); err != nil {
			t.Fatal(err)
		}
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
	docs, err := cs2.SearchDocs("docs", vecs[0], k, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	d1, ok := findDoc(docs, 1)
	if !ok || d1.Content != "chunk-1" || d1.Metadata["src"].Str != "s" {
		t.Errorf("after snapshot reopen, doc 1 = %+v", d1)
	}
}

// TestRAGStorePersistentContent: a Persistent (instant-restart) collection's
// content survives Flush + reopen via the sidecar.
func TestRAGStorePersistentContent(t *testing.T) {
	const dim, k = 16, 10
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{Dim: dim, Metric: Cosine, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1,
		Quant: QuantSQ8, RescoreFactor: 3, Persistent: true}
	if err := cs.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	vecs := ragCorpus(200, dim, 6)
	for i, v := range vecs {
		if err := cs.Upsert("docs", uint64(i+1), v, fmt.Sprintf("chunk-%d", i+1), 0, Metadata{"doc": NewInt(int64(i % 4))}, nil); err != nil {
			t.Fatal(err)
		}
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
	// filtered SearchDocs after instant restart returns content + metadata.
	docs, err := cs2.SearchDocs("docs", vecs[0], k, Filter{Op: FilterEq, Field: "doc", Value: NewInt(0)})
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) == 0 {
		t.Fatal("no docs after instant restart")
	}
	for _, d := range docs {
		if d.Content == "" {
			t.Errorf("doc %d lost content after instant restart", d.ID)
		}
		if d.Metadata["doc"].Int != 0 {
			t.Errorf("filter leaked: doc %d has doc=%d", d.ID, d.Metadata["doc"].Int)
		}
	}
}

// TestRAGStoreWALRecoversUpsertsAndDeletes: a WAL collection recovers Upserts
// (content) and DeleteByFilter purges across a crash with no Flush.
func TestRAGStoreWALRecoversUpsertsAndDeletes(t *testing.T) {
	const dim, k = 16, 10
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{Dim: dim, Metric: Cosine, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1,
		Quant: QuantSQ8, RescoreFactor: 3, WAL: true}
	if err := cs.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	vecs := ragCorpus(300, dim, 8)
	for i, v := range vecs {
		if err := cs.Upsert("docs", uint64(i+1), v, fmt.Sprintf("chunk-%d", i+1), 0, Metadata{"doc": NewInt(int64((i % 3) + 1))}, nil); err != nil {
			t.Fatal(err)
		}
	}
	// Re-upsert one id with new content (replace), and delete a whole "doc".
	if err := cs.Upsert("docs", 1, vecs[0], "chunk-1-v2", 0, Metadata{"doc": NewInt(1)}, nil); err != nil {
		t.Fatal(err)
	}
	delByDoc := Filter{Op: FilterEq, Field: "doc", Value: NewInt(2)}
	if _, err := cs.DeleteByFilter("docs", delByDoc); err != nil {
		t.Fatal(err)
	}
	// Crash: close WITHOUT Flush — only the WAL is on disk.
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs2.Close() }()

	// The replaced content is recovered.
	docs, err := cs2.SearchDocs("docs", vecs[0], k, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	d1, ok := findDoc(docs, 1)
	if !ok || d1.Content != "chunk-1-v2" {
		t.Errorf("replaced content not recovered: %+v", d1)
	}
	// The deleted doc is gone.
	all, err := cs2.SearchDocs("docs", vecs[1], 300, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range all {
		if d.Metadata["doc"].Int == 2 {
			t.Errorf("doc==2 chunk %d survived delete-by-filter across WAL recovery", d.ID)
		}
	}
}

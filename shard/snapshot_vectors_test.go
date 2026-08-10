// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"bytes"
	"io"
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/vector"
)

// fakeSink is a minimal hraft.SnapshotSink over an in-memory buffer.
type fakeSink struct{ *bytes.Buffer }

func (fakeSink) Close() error  { return nil }
func (fakeSink) Cancel() error { return nil }
func (fakeSink) ID() string    { return "test" }

// TestFSMSnapshotIncludesVectors proves the durability fix: a v2 FSM snapshot
// captures the vector CollectionStore (single- and multi-vector), so restoring
// from snapshot (as a recovering/new node does after log truncation) brings the
// vectors back — not just the cache.
func TestFSMSnapshotIncludesVectors(t *testing.T) {
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()
	c.Put([]byte("k"), []byte("v"), 0)

	src, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	if err := src.CreateCollection("docs", vector.Config{Dim: 3, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1}); err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 4; i++ {
		if err := src.Upsert("docs", uint64(i), []float32{float32(i), 0, 0}, "chunk", 0, vector.Metadata{"d": vector.NewInt(int64(i))}, nil); err != nil {
			t.Fatal(err)
		}
	}
	if err := src.CreateMultiVector("mv", vector.MultiVectorConfig{Dim: 3, Seed: 1}); err != nil {
		t.Fatal(err)
	}
	if err := src.MultiAdd("mv", 1, [][]float32{{1, 0, 0}, {0, 1, 0}}, nil); err != nil {
		t.Fatal(err)
	}

	// Persist the FSM snapshot (cache + vectors).
	var sink fakeSink
	sink.Buffer = &bytes.Buffer{}
	data, err := serializeSnapshot(c, src, 0, nil)
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	snap := &fsmSnapshot{data: data}
	if err := snap.Persist(sink); err != nil {
		t.Fatalf("persist: %v", err)
	}

	// Restore into a fresh cache + vector store (simulating a recovering node).
	c2, _ := cache.New(cache.DefaultConfig())
	defer c2.Close()
	dst, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Close()
	rc := io.NopCloser(bytes.NewReader(sink.Bytes()))
	if _, err := restoreSnapshot(c2, dst, nil, rc); err != nil {
		t.Fatalf("restore: %v", err)
	}

	// Cache restored.
	if got, _ := c2.Get([]byte("k")); string(got) != "v" {
		t.Errorf("cache key not restored: %q", got)
	}
	// Single-vector restored with content + metadata.
	docs, err := dst.SearchDocs("docs", []float32{1, 0, 0}, 4, vector.Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 4 || docs[0].Content != "chunk" {
		t.Fatalf("restored docs = %d %+v, want 4 with content", len(docs), docs)
	}
	// Multi-vector restored.
	mv, ok := dst.GetMultiVector("mv")
	if !ok || mv.NumDocs() != 1 {
		t.Fatalf("multi-vector not restored (ok=%v)", ok)
	}
}

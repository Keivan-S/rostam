// SPDX-License-Identifier: Apache-2.0

package rag

import (
	"context"
	"testing"
)

func TestEmbeddedRetrieverRoundtripBM25(t *testing.T) {
	dir := t.TempDir() // allocate BEFORE Close cleanup is registered
	r, err := NewEmbeddedRetriever(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	ctx := context.Background()

	if err := r.EnsureCorpus(ctx, "docs", 0); err != nil { // BM25-only
		t.Fatal(err)
	}
	chunks := []StoredChunk{
		{ID: 1, Content: "the epoll transport picks its loop count from GOMAXPROCS", Source: "a.md", Index: 0},
		{ID: 2, Content: "raft shards should roughly equal core count", Source: "b.md", Index: 0},
	}
	if err := r.Upsert(ctx, "docs", chunks); err != nil {
		t.Fatal(err)
	}
	hits, err := r.Search(ctx, "docs", "epoll loop count", nil, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 || hits[0].Source != "a.md" {
		t.Fatalf("expected a.md top hit, got %+v", hits)
	}
	if hits[0].Content == "" {
		t.Fatalf("hit content should be populated: %+v", hits[0])
	}
}

func TestEmbeddedRetrieverDeleteBySource(t *testing.T) {
	dir := t.TempDir()
	r, err := NewEmbeddedRetriever(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	ctx := context.Background()
	_ = r.EnsureCorpus(ctx, "docs", 0)
	_ = r.Upsert(ctx, "docs", []StoredChunk{
		{ID: 1, Content: "alpha", Source: "a.md", Index: 0},
		{ID: 2, Content: "beta", Source: "b.md", Index: 0},
	})
	n, err := r.DeleteBySource(ctx, "docs", "a.md")
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("deleted %d, want 1", n)
	}
	hits, _ := r.Search(ctx, "docs", "alpha", nil, 5)
	for _, h := range hits {
		if h.Source == "a.md" {
			t.Fatalf("a.md should be gone: %+v", hits)
		}
	}
}

// SPDX-License-Identifier: Apache-2.0

package rag

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/rostamlabs/rostam/semcache"
)

func TestIngestBM25AndReIngest(t *testing.T) {
	src := t.TempDir()
	corpusDir := t.TempDir()
	writeFile(t, filepath.Join(src, "a.md"), "# Title\n\nepoll transport loop count comes from gomaxprocs\n\nsecond paragraph about raft shards")
	writeFile(t, filepath.Join(src, "skip.bin"), "\x00\x01binary")

	r, err := NewEmbeddedRetriever(corpusDir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	ctx := context.Background()
	if err := r.EnsureCorpus(ctx, "docs", 0); err != nil {
		t.Fatal(err)
	}

	rep, err := Ingest(ctx, r, []string{src}, IngestOptions{Corpus: "docs", ChunkSize: 8, ChunkOverlap: 2})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Files != 1 || rep.Chunks < 1 {
		t.Fatalf("report=%+v", rep)
	}
	if rep.Skipped != 1 {
		t.Fatalf("expected 1 skipped (skip.bin), got %+v", rep)
	}

	before, _ := r.Search(ctx, "docs", "epoll", nil, 10)
	// Re-ingest the same file: chunk count in the corpus must not double.
	if _, err := Ingest(ctx, r, []string{src}, IngestOptions{Corpus: "docs", ChunkSize: 8, ChunkOverlap: 2}); err != nil {
		t.Fatal(err)
	}
	after, _ := r.Search(ctx, "docs", "epoll", nil, 10)
	if len(after) != len(before) {
		t.Fatalf("re-ingest duplicated chunks: before=%d after=%d", len(before), len(after))
	}
}

func TestIngestReIngestEmptiedFilePurges(t *testing.T) {
	src := t.TempDir()
	corpusDir := t.TempDir()
	path := filepath.Join(src, "a.md")
	writeFile(t, path, "epoll transport loop count comes from gomaxprocs and raft shards")

	r, err := NewEmbeddedRetriever(corpusDir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	ctx := context.Background()
	if err := r.EnsureCorpus(ctx, "docs", 0); err != nil {
		t.Fatal(err)
	}

	if _, err := Ingest(ctx, r, []string{src}, IngestOptions{Corpus: "docs", ChunkSize: 8, ChunkOverlap: 2}); err != nil {
		t.Fatal(err)
	}
	hits, err := r.Search(ctx, "docs", "epoll", nil, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 {
		t.Fatalf("expected hits for initial content, got none")
	}

	// Overwrite with whitespace-only content: SplitText will yield zero
	// chunks, but the file's prior chunks must still be purged.
	writeFile(t, path, "   \n\t ")
	if _, err := Ingest(ctx, r, []string{src}, IngestOptions{Corpus: "docs", ChunkSize: 8, ChunkOverlap: 2}); err != nil {
		t.Fatal(err)
	}
	after, err := r.Search(ctx, "docs", "epoll", nil, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(after) != 0 {
		t.Fatalf("expected stale chunks purged after re-ingesting emptied file, got %d hits", len(after))
	}
}

func TestIngestWithStubEmbedder(t *testing.T) {
	src := t.TempDir()
	corpusDir := t.TempDir()
	writeFile(t, filepath.Join(src, "a.md"), "alpha beta gamma delta epsilon")
	emb := semcache.NewStubEmbedder("stub", 8)

	r, err := NewEmbeddedRetriever(corpusDir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	ctx := context.Background()
	if err := r.EnsureCorpus(ctx, "docs", emb.Dim()); err != nil {
		t.Fatal(err)
	}
	rep, err := Ingest(ctx, r, []string{src}, IngestOptions{Corpus: "docs", Embedder: emb})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Chunks < 1 {
		t.Fatalf("want >=1 chunk, got %+v", rep)
	}
	// Dense search should return the chunk (stub vectors are deterministic).
	qv, _ := emb.Embed(ctx, []string{"alpha"})
	hits, err := r.Search(ctx, "docs", "alpha", qv[0], 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 {
		t.Fatalf("dense search returned no hits")
	}
}

// TestIngestReIngestInvalidUTF8Purges: a previously-indexed source that later
// becomes invalid UTF-8 is skipped, but its stale chunks must still be purged
// so search never returns content no longer backed by the file.
func TestIngestReIngestInvalidUTF8Purges(t *testing.T) {
	src := t.TempDir()
	corpusDir := t.TempDir()
	path := filepath.Join(src, "a.md")
	writeFile(t, path, "epoll transport loop count comes from gomaxprocs and raft shards")

	r, err := NewEmbeddedRetriever(corpusDir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	ctx := context.Background()
	if err := r.EnsureCorpus(ctx, "docs", 0); err != nil {
		t.Fatal(err)
	}

	if _, err := Ingest(ctx, r, []string{src}, IngestOptions{Corpus: "docs", ChunkSize: 8, ChunkOverlap: 2}); err != nil {
		t.Fatal(err)
	}
	if hits, err := r.Search(ctx, "docs", "epoll", nil, 10); err != nil {
		t.Fatal(err)
	} else if len(hits) == 0 {
		t.Fatalf("expected hits for initial content, got none")
	}

	// Overwrite a.md with invalid UTF-8 bytes: the ingester skips it, but the
	// prior chunks for this path must be purged.
	writeFile(t, path, "valid prefix \xff\xfe\xfa invalid tail")
	rep, err := Ingest(ctx, r, []string{src}, IngestOptions{Corpus: "docs", ChunkSize: 8, ChunkOverlap: 2})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Skipped != 1 {
		t.Fatalf("expected the invalid-UTF-8 file to be skipped, got report %+v", rep)
	}
	after, err := r.Search(ctx, "docs", "epoll", nil, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(after) != 0 {
		t.Fatalf("expected stale chunks purged after source became invalid UTF-8, got %d hits", len(after))
	}
}

func writeFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
}

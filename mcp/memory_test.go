// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"strings"
	"testing"
)

func TestRememberRecallBM25(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	var r1 struct {
		ID uint64 `json:"id"`
	}
	c.callTool("remember", map[string]any{"content": "the deploy password is stored in vault under ops/deploy"}, &r1, false)
	c.callTool("remember", map[string]any{"content": "the coffee machine is on floor two"}, nil, false)

	var rec struct {
		Hits []struct {
			ID      uint64 `json:"id"`
			Content string `json:"content"`
		} `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "deploy password vault", "k": 1}, &rec, false)
	if len(rec.Hits) != 1 || !strings.Contains(rec.Hits[0].Content, "vault") {
		t.Fatalf("BM25 recall missed: %+v", rec.Hits)
	}
	if rec.Hits[0].ID != r1.ID {
		t.Fatalf("id mismatch: %d vs %d", rec.Hits[0].ID, r1.ID)
	}
}

func TestRememberDedupesSameContent(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	var a, b struct {
		ID uint64 `json:"id"`
	}
	c.callTool("remember", map[string]any{"content": "same fact"}, &a, false)
	c.callTool("remember", map[string]any{"content": "same fact"}, &b, false)
	if a.ID != b.ID {
		t.Fatalf("same content should dedupe: %d vs %d", a.ID, b.ID)
	}
}

func TestNamespaceIsolation(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	c.callTool("remember", map[string]any{"content": "alpha secret", "namespace": "projA"}, nil, false)
	var rec struct {
		Hits []struct{ Content string } `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "alpha secret", "namespace": "projB"}, &rec, false)
	if len(rec.Hits) != 0 {
		t.Fatalf("cross-namespace leak: %+v", rec.Hits)
	}
}

func TestRememberRejectsReservedMetadata(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	msg := c.callTool("remember", map[string]any{"content": "x", "metadata": map[string]any{"__ns": "evil"}}, nil, true)
	if !strings.Contains(msg, "__ns") {
		t.Fatalf("error should name the reserved key, got %q", msg)
	}
}

// fixed-vector embedder: "a"-prefixed texts embed near each other, "b" far.
type fakeEmbedder struct{}

func (fakeEmbedder) Model() string { return "fake" }
func (fakeEmbedder) Dim() int      { return 4 }
func (fakeEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, tx := range texts {
		if strings.HasPrefix(tx, "a") {
			out[i] = []float32{1, 0, 0, 0}
		} else {
			out[i] = []float32{0, 1, 0, 0}
		}
	}
	return out, nil
}

func TestRecallHybridUsesEmbedder(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t), Embedder: fakeEmbedder{}})
	c.initialize()
	c.callTool("remember", map[string]any{"content": "a: dense-close fact"}, nil, false)
	c.callTool("remember", map[string]any{"content": "b: dense-far fact"}, nil, false)
	var rec struct {
		Hits []struct {
			Content string `json:"content"`
		} `json:"hits"`
	}
	// query "a..." shares NO BM25 tokens with either doc; only the dense side ranks it.
	c.callTool("recall", map[string]any{"query": "a unrelated words", "k": 1}, &rec, false)
	if len(rec.Hits) != 1 || !strings.HasPrefix(rec.Hits[0].Content, "a:") {
		t.Fatalf("hybrid recall did not use dense side: %+v", rec.Hits)
	}
}

func TestForgetDeletesAndPrunesEmptyNamespace(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	var a1, a2, b1 struct {
		ID uint64 `json:"id"`
	}
	c.callTool("remember", map[string]any{"content": "a fact one", "namespace": "a"}, &a1, false)
	c.callTool("remember", map[string]any{"content": "a fact two", "namespace": "a"}, &a2, false)
	c.callTool("remember", map[string]any{"content": "b fact one", "namespace": "b"}, &b1, false)

	var nsBefore struct {
		Namespaces []string `json:"namespaces"`
	}
	c.callTool("list_namespaces", map[string]any{}, &nsBefore, false)
	if !containsStr(nsBefore.Namespaces, "a") || !containsStr(nsBefore.Namespaces, "b") {
		t.Fatalf("expected both namespaces before forget: %+v", nsBefore.Namespaces)
	}

	const unknownID = uint64(999999999)
	var fg struct {
		Deleted []uint64 `json:"deleted"`
		Missing []uint64 `json:"missing"`
	}
	c.callTool("forget", map[string]any{"ids": []uint64{b1.ID, unknownID}}, &fg, false)
	if len(fg.Deleted) != 1 || fg.Deleted[0] != b1.ID {
		t.Fatalf("expected deleted=[%d], got %+v", b1.ID, fg.Deleted)
	}
	if len(fg.Missing) != 1 || fg.Missing[0] != unknownID {
		t.Fatalf("expected missing=[%d], got %+v", unknownID, fg.Missing)
	}

	var rec struct {
		Hits []struct{ ID uint64 } `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "b fact one", "namespace": "b"}, &rec, false)
	if len(rec.Hits) != 0 {
		t.Fatalf("forgotten memory still recallable: %+v", rec.Hits)
	}

	var nsAfter struct {
		Namespaces []string `json:"namespaces"`
	}
	c.callTool("list_namespaces", map[string]any{}, &nsAfter, false)
	if containsStr(nsAfter.Namespaces, "b") {
		t.Fatalf("emptied namespace \"b\" should be pruned: %+v", nsAfter.Namespaces)
	}
	if !containsStr(nsAfter.Namespaces, "a") {
		t.Fatalf("namespace \"a\" should remain: %+v", nsAfter.Namespaces)
	}
}

func containsStr(ss []string, want string) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}

func TestListMemoriesPaginates(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	var facts [3]struct {
		ID uint64 `json:"id"`
	}
	c.callTool("remember", map[string]any{"content": "fact one", "namespace": "page"}, &facts[0], false)
	c.callTool("remember", map[string]any{"content": "fact two", "namespace": "page"}, &facts[1], false)
	c.callTool("remember", map[string]any{"content": "fact three", "namespace": "page"}, &facts[2], false)

	var page1 struct {
		Memories []struct {
			ID uint64 `json:"id"`
		} `json:"memories"`
		NextCursor string `json:"next_cursor"`
	}
	c.callTool("list_memories", map[string]any{"namespace": "page", "limit": 2}, &page1, false)
	if len(page1.Memories) != 2 {
		t.Fatalf("expected 2 memories on page 1, got %d", len(page1.Memories))
	}
	if page1.NextCursor == "" {
		t.Fatalf("expected non-empty cursor after a partial page")
	}

	var page2 struct {
		Memories []struct {
			ID uint64 `json:"id"`
		} `json:"memories"`
		NextCursor string `json:"next_cursor"`
	}
	c.callTool("list_memories", map[string]any{"namespace": "page", "limit": 2, "cursor": page1.NextCursor}, &page2, false)
	if len(page2.Memories) != 1 {
		t.Fatalf("expected 1 memory on page 2, got %d", len(page2.Memories))
	}
	if page2.NextCursor != "" {
		t.Fatalf("expected exhausted cursor on page 2, got %q", page2.NextCursor)
	}

	seen := map[uint64]bool{}
	for _, m := range page1.Memories {
		seen[m.ID] = true
	}
	for _, m := range page2.Memories {
		seen[m.ID] = true
	}
	if len(seen) != 3 {
		t.Fatalf("expected union of 3 distinct ids across pages, got %d: %+v", len(seen), seen)
	}
	for _, f := range facts {
		if !seen[f.ID] {
			t.Fatalf("id %d missing from paginated results", f.ID)
		}
	}
}

func TestEmbedderIdentityMismatchFailsStartup(t *testing.T) {
	st := newHeapStore(t)
	c := startServer(t, Config{Store: st}) // BM25-only creates the collection
	c.initialize()
	c.callTool("remember", map[string]any{"content": "seed"}, nil, false)

	// Reopening the same store with a real embedder must fail loudly.
	if _, err := NewServer(context.Background(), Config{Store: st, Embedder: fakeEmbedder{}}); err == nil {
		t.Fatal("expected embedder-identity mismatch error")
	}
}

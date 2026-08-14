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

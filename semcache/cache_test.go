// SPDX-License-Identifier: Apache-2.0

package semcache

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
)

// newTestCache brings up a single-node in-process engine and a fake embedder,
// returning a ready Cache. Shared by every cache test.
func newTestCache(t *testing.T) *Cache {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("register ops: %v", err)
	}
	store, err := rostam.NewEmbedded(rostam.EmbeddedConfig{
		NodeID: "t", DataDir: t.TempDir(), NumShards: 1, Bootstrap: true, Ops: reg,
	})
	if err != nil {
		t.Fatalf("NewEmbedded: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	deadline := time.Now().Add(5 * time.Second)
	for !store.IsLeader([]byte("k")) && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}

	c, err := New(context.Background(), Config{
		Store:      store,
		Embedder:   fakeEmbedder{model: "fake-1", dim: 64},
		Collection: "llm-cache",
		Threshold:  0.97,
		TTL:        time.Hour,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

func TestNewValidates(t *testing.T) {
	_, err := New(context.Background(), Config{}) // no store, no embedder
	if err == nil {
		t.Fatal("expected validation error for empty config")
	}
	if !strings.Contains(err.Error(), "embedder") && !strings.Contains(err.Error(), "store") {
		t.Fatalf("unhelpful validation error: %v", err)
	}
}

func TestNewIsIdempotent(t *testing.T) {
	c := newTestCache(t) // creates the collection once
	// Creating a second Cache over the SAME store + collection must not fail.
	_, err := New(context.Background(), Config{
		Store: c.store, Embedder: c.embedder, Collection: c.collection, Threshold: 0.97,
	})
	if err != nil {
		t.Fatalf("second New over existing collection: %v", err)
	}
}

func TestStorePersistsAnswerAndTokens(t *testing.T) {
	c := newTestCache(t)
	ctx := context.Background()
	scope := Scope{Model: "m", Temperature: 0}

	if err := c.Store(ctx, "what is the capital of france?", scope, "Paris.", 3); err != nil {
		t.Fatalf("Store: %v", err)
	}
	// Storing the SAME (prompt, scope) twice must not error (idempotent upsert).
	if err := c.Store(ctx, "what is the capital of france?", scope, "Paris.", 3); err != nil {
		t.Fatalf("Store (repeat): %v", err)
	}
}

func TestLookupHitMissAndScope(t *testing.T) {
	c := newTestCache(t)
	ctx := context.Background()
	scope := Scope{Model: "m", Temperature: 0}

	// Miss on empty cache.
	if _, ok, err := c.Lookup(ctx, "hello world", scope); err != nil || ok {
		t.Fatalf("expected clean miss, got ok=%v err=%v", ok, err)
	}

	// Store then exact-prompt lookup => hit (identical vector, similarity 1.0).
	if err := c.Store(ctx, "hello world", scope, "hi!", 2); err != nil {
		t.Fatal(err)
	}
	hit, ok, err := c.Lookup(ctx, "hello world", scope)
	if err != nil || !ok {
		t.Fatalf("expected hit, got ok=%v err=%v", ok, err)
	}
	if hit.Answer != "hi!" || hit.OutTokens != 2 {
		t.Fatalf("wrong hit payload: %+v", hit)
	}
	if hit.Score < c.threshold {
		t.Fatalf("hit score %v below threshold %v", hit.Score, c.threshold)
	}

	// Same prompt, DIFFERENT scope => miss (scope isolation).
	if _, ok, _ := c.Lookup(ctx, "hello world", Scope{Model: "other"}); ok {
		t.Fatal("cross-scope lookup must miss")
	}

	// A clearly different prompt => miss (below threshold under fake embedder).
	if _, ok, _ := c.Lookup(ctx, "completely unrelated quantum chromodynamics", scope); ok {
		t.Fatal("dissimilar prompt must miss at 0.97 threshold")
	}
}

func TestCacheableHonorsMaxTemp(t *testing.T) {
	c := newTestCache(t) // MaxTemp defaults to 0
	if !c.Cacheable(0) {
		t.Fatal("temp 0 should be cacheable")
	}
	if c.Cacheable(0.7) {
		t.Fatal("temp above MaxTemp must not be cacheable")
	}
}

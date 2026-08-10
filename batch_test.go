// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"bytes"
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// newMultiShardEmbedded builds a single-node embedded Store with several shards so
// a PutBatch fans across more than one shard's Raft group. It waits for cluster
// readiness (every routed shard has elected a leader) before returning.
func newMultiShardEmbedded(t *testing.T, numShards int) Store {
	t.Helper()
	dir := t.TempDir()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	s, err := NewEmbedded(EmbeddedConfig{
		NodeID:    "test-node",
		DataDir:   dir,
		NumShards: numShards,
		Bootstrap: true,
		Ops:       reg,
	})
	if err != nil {
		t.Fatalf("NewEmbedded: %v", err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Logf("embedded Close: %v", err)
		}
	})

	// Readiness gate: keyed puts across a spread of keys prove every shard group
	// this node routes to has elected a leader (mirrors waitClusterLeaders).
	ctx := context.Background()
	deadline := time.Now().Add(cpuScaled(30 * time.Second))
	for {
		ready := true
		for i := 0; i < numShards*8; i++ {
			key := []byte(fmt.Sprintf("__ready__/%d", i))
			if perr := s.Put(ctx, key, []byte("1"), 0); perr != nil {
				ready = false
				break
			}
		}
		if ready {
			return s
		}
		if time.Now().After(deadline) {
			t.Fatal("newMultiShardEmbedded: timed out waiting for shard leaders")
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// TestEmbeddedPutBatchRoundtrip inserts ~200 entries spanning multiple shards via
// Store.PutBatch, then reads each key back with Get. It also asserts that an empty
// PutBatch is a no-op.
func TestEmbeddedPutBatchRoundtrip(t *testing.T) {
	s := newMultiShardEmbedded(t, 4)
	ctx := context.Background()

	// Empty PutBatch is a no-op.
	if err := s.PutBatch(ctx, nil); err != nil {
		t.Fatalf("PutBatch(nil): %v", err)
	}

	const n = 200
	entries := make([]ops.PutEntry, n)
	for i := range entries {
		entries[i] = ops.PutEntry{
			Key: []byte(fmt.Sprintf("batch/key/%03d", i)),
			Val: []byte(fmt.Sprintf("val-%d", i)),
		}
	}
	if err := s.PutBatch(ctx, entries); err != nil {
		t.Fatalf("PutBatch: %v", err)
	}

	for _, e := range entries {
		got, err := s.Get(ctx, e.Key)
		if err != nil {
			t.Fatalf("Get(%q): %v", e.Key, err)
		}
		if !bytes.Equal(got, e.Val) {
			t.Fatalf("Get(%q) = %q, want %q", e.Key, got, e.Val)
		}
	}
}

// TestDirectPutBatch inserts many entries via a Direct store's PutBatch and reads
// each back. Empty PutBatch is a no-op.
func TestDirectPutBatch(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	s, err := NewDirect(DirectConfig{
		Ops:   reg,
		Cache: CacheConfig{NumShardsPerNode: 4},
	})
	if err != nil {
		t.Fatalf("NewDirect: %v", err)
	}
	t.Cleanup(func() {
		if cerr := s.Close(); cerr != nil {
			t.Logf("direct Close: %v", cerr)
		}
	})

	ctx := context.Background()
	if err := s.PutBatch(ctx, nil); err != nil {
		t.Fatalf("PutBatch(nil): %v", err)
	}

	const n = 200
	entries := make([]ops.PutEntry, n)
	for i := range entries {
		entries[i] = ops.PutEntry{
			Key: []byte(fmt.Sprintf("d/key/%03d", i)),
			Val: []byte(fmt.Sprintf("dval-%d", i)),
		}
	}
	if err := s.PutBatch(ctx, entries); err != nil {
		t.Fatalf("PutBatch: %v", err)
	}

	for _, e := range entries {
		got, err := s.Get(ctx, e.Key)
		if err != nil {
			t.Fatalf("Get(%q): %v", e.Key, err)
		}
		if !bytes.Equal(got, e.Val) {
			t.Fatalf("Get(%q) = %q, want %q", e.Key, got, e.Val)
		}
	}
}

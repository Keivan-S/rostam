// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/rostamlabs/rostam/ops"
)

// TestNodePutBatchAcrossShards verifies Node.PutBatch groups entries by shard,
// applies every entry (one put_batch per shard), and each key reads back from
// the shard its own hash routes to. Single-node so this node leads every shard,
// isolating the fan-out grouping from leader routing.
func TestNodePutBatchAcrossShards(t *testing.T) {
	tc := newTestCluster(t, 1, 16)
	n := tc.nodes[0]

	const count = 200
	entries := make([]ops.PutEntry, count)
	for i := range count {
		entries[i] = ops.PutEntry{
			Key: fmt.Appendf(nil, "key-%03d", i),
			Val: fmt.Appendf(nil, "val-%03d", i),
		}
	}
	// Sanity: these keys really do span multiple shards (else the test proves
	// nothing about grouping).
	shards := map[int]struct{}{}
	for _, e := range entries {
		shards[shardOf(e.Key, 16)] = struct{}{}
	}
	if len(shards) < 2 {
		t.Fatalf("keys landed on %d shard(s); need >1 to exercise grouping", len(shards))
	}

	if err := n.PutBatch(entries); err != nil {
		t.Fatalf("PutBatch: %v", err)
	}

	for _, e := range entries {
		got, err := n.Call("get", ops.EncodeKeyArgs(e.Key))
		if err != nil {
			t.Fatalf("get %q: %v", e.Key, err)
		}
		if !bytes.Equal(got, e.Val) {
			t.Fatalf("get %q = %q, want %q", e.Key, got, e.Val)
		}
	}

	// Empty batch is a no-op, not an error.
	if err := n.PutBatch(nil); err != nil {
		t.Fatalf("empty PutBatch: %v", err)
	}
}

// TestNodeCallPutBatchCrossShardRejected verifies the wire-level guard: a raw
// put_batch whose keys span shards is rejected (ErrPutBatchCrossShard) rather
// than silently storing off-shard keys where they can never be read back.
func TestNodeCallPutBatchCrossShardRejected(t *testing.T) {
	tc := newTestCluster(t, 1, 16)
	n := tc.nodes[0]

	// Find two keys that hash to different shards.
	base := shardOf([]byte("key-000"), 16)
	var other []byte
	for i := 1; i < 10000; i++ {
		k := fmt.Appendf(nil, "key-%03d", i)
		if shardOf(k, 16) != base {
			other = k
			break
		}
	}
	if other == nil {
		t.Fatal("could not find a cross-shard key pair")
	}
	args := ops.EncodePutBatchArgs([]ops.PutEntry{
		{Key: []byte("key-000"), Val: []byte("v")},
		{Key: other, Val: []byte("v")},
	})
	if _, err := n.Call("put_batch", args); err != ErrPutBatchCrossShard {
		t.Fatalf("cross-shard put_batch: got %v, want ErrPutBatchCrossShard", err)
	}
	// The off-shard key must not have been written.
	if _, err := n.Call("get", ops.EncodeKeyArgs(other)); err == nil {
		t.Fatal("rejected cross-shard batch must apply nothing, but off-shard key was written")
	}
}

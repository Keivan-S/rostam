// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/vector"
)

// TestPartitionedCreateSucceedsFirstAttemptOnEveryNode is the multi-node
// regression for partitioned-collection creation at replication-factor < node
// count.
//
// It deliberately does NOT use createCollectionTolerant. That helper retries on
// ErrNotLeader for 30s AND, on "already exists", repairs the catalog itself with
// SetPartitionsGen — compensations that mask precisely the two defects this test
// exists to catch:
//
//  1. The create failed outright. embedded.Call dispatched through the BARE
//     cluster node, so a partition landing on a shard this node hosts as a
//     FOLLOWER came back NotLeader (mapErr flattens it to ErrNotLeader, dropping
//     the leader address) and the per-partition loop aborted. Observed on a live
//     3-node cluster: the create failed from ALL THREE nodes.
//  2. It failed PARTWAY, leaving orphaned physical partitions and — because
//     SetPartitionsGen runs only after the loop — no catalog record at all, so
//     the collection silently behaved as UNPARTITIONED.
//
// A create must therefore succeed on the FIRST attempt from EVERY node, and the
// catalog must record P without anyone patching it up afterwards. Retrying, or
// writing the catalog entry on the test's behalf, would turn both defects back
// into green.
//
// RF=2 with 3 nodes is the shape that matters: it guarantees some shard is hosted
// by a node that does not lead it, which is the case that broke. At RF=N every
// node leads or hosts everything and the bug is unreachable.
func TestPartitionedCreateSucceedsFirstAttemptOnEveryNode(t *testing.T) {
	const (
		nodes     = 3
		numShards = 8
		rf        = 2
		parts     = 8
	)
	stores := newInmemEmbeddedCluster(t, nodes, numShards, rf)
	waitClusterLeadersRF(t, stores, numShards)

	ctx := context.Background()
	cfg := VectorConfig{
		Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64,
		Seed: 1, Partitions: parts,
	}

	for i, s := range stores {
		coll := fmt.Sprintf("parted_n%d", i+1)
		// ONE attempt. No tolerance, no retry.
		if err := s.CreateCollection(ctx, coll, cfg); err != nil {
			t.Fatalf("node %d: CreateCollection(%s, P=%d) failed on the first attempt: %v",
				i+1, coll, parts, err)
		}
		// The catalog must record P by itself. Without this the create can "succeed"
		// while leaving the collection silently unpartitioned — defect 2.
		waitEmbeddedCatalogGen(t, s.(*embedded), coll, parts, 0, cpuScaled(15*time.Second))
	}
}

// TestPartitionedCreateIsUsableFromEveryNode pins the other half: a partitioned
// collection created on one node must accept writes routed from ANY node, since
// its partitions are spread across shards that different nodes lead.
//
// This is the vector analogue of the leader-hint fix — a write for a shard the
// contacted node hosts as a follower must reach the leader rather than surfacing
// NotLeader to the caller.
func TestPartitionedCreateIsUsableFromEveryNode(t *testing.T) {
	const (
		nodes     = 3
		numShards = 8
		rf        = 2
		parts     = 8
	)
	stores := newInmemEmbeddedCluster(t, nodes, numShards, rf)
	waitClusterLeadersRF(t, stores, numShards)

	ctx := context.Background()
	coll := "parted_shared"
	if err := stores[0].CreateCollection(ctx, coll, VectorConfig{
		Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64,
		Seed: 1, Partitions: parts,
	}); err != nil {
		t.Fatalf("create %s: %v", coll, err)
	}
	for _, s := range stores {
		waitEmbeddedCatalogGen(t, s.(*embedded), coll, parts, 0, cpuScaled(15*time.Second))
	}

	// Ids are chosen only to spread across partitions; any id works, the point is
	// that the ORIGIN node varies while the owning shard does not.
	for i, s := range stores {
		for k := range 8 {
			id := uint64(i*100 + k + 1)
			if err := s.VectorInsert(ctx, coll, id, []float32{
				float32(id), 0, 0, 0,
			}); err != nil {
				t.Fatalf("node %d: insert id=%d into partitioned %s: %v", i+1, id, coll, err)
			}
		}
	}
}

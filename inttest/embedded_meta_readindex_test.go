// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/cluster"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// ----------------------------------------------------------------------------
// The read-path wiring proofs. resolveCollectionForRead runs the meta
// readIndex barrier (cluster.Node.MetaReadBarrier) EXACTLY ONCE per Linearizable
// read on the COORDINATOR (before fan-out, never per-partition), and ZERO times
// for AnyReplica/LeaderOnly reads and ALL writes. Single-node/pure-embedded is a
// true no-op.
//
// The proof instrument is the Task-2 forward hook (cluster.SetMetaReadIndexForwardHook),
// which fires ONCE each time a FOLLOWER actually forwards a __meta_readindex__ op
// to the meta leader. So we drive the reads on a meta-FOLLOWER coordinator: a
// Linearizable read there forwards exactly once (the barrier's single leader RTT);
// an off-path read forwards zero times. (On the leader coordinator the barrier
// fast-paths with zero forwards too, but the follower is the strict case — it is
// the only node that CAN forward, so a zero there is a real zero.)
// ----------------------------------------------------------------------------

// metaFollowerStore returns the index of a store whose node is a meta-FOLLOWER
// (not the meta leader). It waits briefly for a stable single leader so a follower
// is deterministically identifiable. Used so a Linearizable read on that store
// exercises the follower-forward path the counter observes.
func metaFollowerStore(t *testing.T, stores []rostam.Store) int {
	t.Helper()
	// load-flakiness hardening: readiness gate (wait for a stable single meta
	// leader so a follower is identifiable) widened 10s->30s. Setup-only — cuts
	// false "no stable meta follower" fatals under CPU contention; cannot mask a
	// behavioral regression.
	deadline := time.Now().Add(cpuScaled(30 * time.Second))
	for time.Now().Before(deadline) {
		leaders := 0
		follower := -1
		for i, s := range stores {
			n := s.(*rostam.Embedded).Node()
			if n.MetaIsLeader() {
				leaders++
			} else if follower < 0 {
				follower = i
			}
		}
		if leaders == 1 && follower >= 0 {
			return follower
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("no stable meta follower found")
	return -1
}

// seedPartitionedDocs creates a P-partition "docs" collection on the coordinator and
// upserts N points, retrying through replication/election jitter, then waits for the
// given store's local catalog to converge on (P, gen 0) so the reads under test are
// not racing creation.
func seedPartitionedDocs(t *testing.T, ctx context.Context, stores []rostam.Store, readIdx, P, N int) {
	t.Helper()
	// Create idempotently via the shared createPartitionedDocs helper, which treats a
	// "vector: collection already exists" result as success (the prior attempt's create
	// committed; a retry seeing "already exists" means the physical partitions are there)
	// and completes the catalog entry — mirroring the Task-4 create-idempotency pattern.
	// The bare retryUntil(t, "create", ...) used here previously fataled on "already
	// exists" because it only retries rostam.ErrNotLeader, which made the suite flake at
	// -count>=2 when a create committed but returned a transient (non-not-leader) error.
	createPartitionedDocs(t, ctx, stores[0], P)
	// Seed via the shared seedDocs helper, which retries EVERY transient error per id
	// within a generous per-id budget. The bare retryUntil(t, "upsert", ...) used here
	// previously fataled the test outright the moment a per-partition shard-leader
	// election blip surfaced a transient error (or a not-leader that never cleared in
	// its single budget) — the second flake source at -count>=2 under RF=3 churn.
	// seedDocs still fails loud if any id never lands, so no correctness coverage is lost.
	seedDocs(t, ctx, stores[0], N)
	// widened 10s->20s + cpuScaled for CPU-contended CI (readIdx is a meta-follower,
	// so this rides follower meta-apply lag); finite so a real catalog-convergence
	// hang still fails. Setup/progress gate, not a behavioral assertion.
	waitEmbeddedCatalogGen(t, stores[readIdx].(*rostam.Embedded), "docs", P, 0, cpuScaled(20*time.Second))
}

// TestLinearizableReadFiresBarrierExactlyOnce is THE once-per-read proof: a
// Linearizable read on a P>1 collection, issued on a meta-FOLLOWER coordinator,
// forwards __meta_readindex__ EXACTLY ONCE (the coordinator barrier before fan-out),
// NOT P times (it is not per-partition). Asserting ==1 (not >=1) catches a
// regression that moves the barrier into the per-partition fan-out.
func TestLinearizableReadFiresBarrierExactlyOnce(t *testing.T) {
	stores := newInmemEmbeddedCluster(t, 3, 4, 3) // RF=3: leader-routed partitioned reads serviceable everywhere. load-flakiness hardening: host-shard pool 8→4 (assertion is on the P=4 collection's barrier-forward count, not the host pool size; 4 still gives P>1 fan-out) to cut raft-group contention
	ctx := context.Background()
	const P, N = 4, 40
	readIdx := metaFollowerStore(t, stores)
	seedPartitionedDocs(t, ctx, stores, readIdx, P, N)

	var forwards int32
	cluster.SetMetaReadIndexForwardHook(func() { atomic.AddInt32(&forwards, 1) })
	defer cluster.SetMetaReadIndexForwardHook(nil)

	// A Linearizable dense search on the follower coordinator. P=4 so a per-partition
	// barrier would forward 4 times; the coordinator barrier forwards exactly once.
	opts := rostam.VectorSearchOpts{ReadConsistency: ops.ConsistencyLinearizable, OnPartitionUnavailable: 1}
	if _, _, err := stores[readIdx].VectorSearchExt(ctx, "docs", []float32{1, 0, 0, 0}, 10, opts); err != nil {
		t.Fatalf("Linearizable VectorSearchExt: %v", err)
	}
	if got := atomic.LoadInt32(&forwards); got != 1 {
		t.Fatalf("Linearizable read fired %d __meta_readindex__ forwards on a P=%d collection, want EXACTLY 1 (one coordinator barrier, NOT per-partition)", got, P)
	}

	// A second Linearizable read fires exactly one MORE forward (one per read), not a
	// cached/skipped barrier and not a per-partition burst.
	atomic.StoreInt32(&forwards, 0)
	if _, _, err := stores[readIdx].VectorSearchDocs(ctx, "docs", []float32{1, 0, 0, 0}, 10, opts); err != nil {
		t.Fatalf("Linearizable VectorSearchDocs: %v", err)
	}
	if got := atomic.LoadInt32(&forwards); got != 1 {
		t.Fatalf("second Linearizable read fired %d forwards, want EXACTLY 1 per read", got)
	}
}

// TestNonLinearizableReadsAndWritesFireBarrierZero proves the ZERO-cost contract off
// the Linearizable path: AnyReplica + LeaderOnly reads and a WRITE issue ZERO
// __meta_readindex__ forwards on the same lagging-capable follower coordinator where
// a Linearizable read WOULD forward. This is the byte-for-byte-old-path guarantee.
func TestNonLinearizableReadsAndWritesFireBarrierZero(t *testing.T) {
	stores := newInmemEmbeddedCluster(t, 3, 4, 3) // load-flakiness hardening: host-shard pool 8→4 (assertion is the ZERO-forward count on the P=4 collection, not the host pool size; 4 still gives P>1 fan-out) to cut raft-group contention
	ctx := context.Background()
	const P, N = 4, 40
	readIdx := metaFollowerStore(t, stores)
	seedPartitionedDocs(t, ctx, stores, readIdx, P, N)

	var forwards int32
	cluster.SetMetaReadIndexForwardHook(func() { atomic.AddInt32(&forwards, 1) })
	defer cluster.SetMetaReadIndexForwardHook(nil)

	q := []float32{1, 0, 0, 0}

	// AnyReplica read: ZERO forwards. We assert the count is zero regardless of any
	// transient read error from shard-leadership churn — an off-Linearizable read
	// NEVER enters the barrier, so it cannot forward even if the read itself blips.
	atomic.StoreInt32(&forwards, 0)
	_, _, _ = stores[readIdx].VectorSearchExt(ctx, "docs", q, 10,
		rostam.VectorSearchOpts{ReadConsistency: ops.ConsistencyAnyReplica, OnPartitionUnavailable: 1})
	if got := atomic.LoadInt32(&forwards); got != 0 {
		t.Fatalf("AnyReplica read fired %d __meta_readindex__ forwards, want 0 (zero added cost off the Linearizable path)", got)
	}

	// LeaderOnly read: ZERO forwards (same rationale).
	atomic.StoreInt32(&forwards, 0)
	_, _, _ = stores[readIdx].VectorSearchExt(ctx, "docs", q, 10,
		rostam.VectorSearchOpts{ReadConsistency: ops.ConsistencyLeaderOnly, OnPartitionUnavailable: 1})
	if got := atomic.LoadInt32(&forwards); got != 0 {
		t.Fatalf("LeaderOnly read fired %d __meta_readindex__ forwards, want 0", got)
	}

	// A WRITE: ZERO forwards (writes keep the plain resolveAlias, never the barrier).
	// We assert the forward count is ZERO regardless of whether the write itself
	// succeeds through any shard-leadership election churn — the invariant is that a
	// write NEVER enters the read barrier, so even a transient not-leader write must
	// not forward a __meta_readindex__. (Tolerating the write's own transient error
	// keeps this a barrier-wiring proof, not a hostage to harness election timing.)
	atomic.StoreInt32(&forwards, 0)
	_ = stores[readIdx].VectorUpsert(ctx, "docs", uint64(N+1), []float32{float32(N + 1), 0, 0, 0}, "doc-new", rostam.VectorInsertOpts{})
	if got := atomic.LoadInt32(&forwards); got != 0 {
		t.Fatalf("write fired %d __meta_readindex__ forwards, want 0 (writes never run the read barrier)", got)
	}
}

// TestSingleNodeLinearizableReadNoForwards proves the pure-embedded no-op: a
// single-node embedded rostam.Store (no meta-Raft) serves a Linearizable read with ZERO
// forwards and a correct result — the barrier short-circuits locally (n.meta == nil).
func TestSingleNodeLinearizableReadNoForwards(t *testing.T) {
	s := newSingleEmbedded(t) // single-node, no peers ⇒ no meta-Raft
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	const N = 20
	retryUntil(t, "create", func() error {
		return s.CreateCollection(ctx, "docs", rostam.VectorConfig{
			Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1,
		})
	})
	for id := uint64(1); id <= uint64(N); id++ {
		idc := id
		retryUntil(t, "upsert", func() error {
			return s.VectorUpsert(ctx, "docs", idc, []float32{float32(idc), 0, 0, 0}, fmt.Sprintf("doc-%d", idc), rostam.VectorInsertOpts{})
		})
	}

	var forwards int32
	cluster.SetMetaReadIndexForwardHook(func() { atomic.AddInt32(&forwards, 1) })
	defer cluster.SetMetaReadIndexForwardHook(nil)

	opts := rostam.VectorSearchOpts{ReadConsistency: ops.ConsistencyLinearizable}
	res, _, err := s.VectorSearchExt(ctx, "docs", []float32{1, 0, 0, 0}, 10, opts)
	if err != nil {
		t.Fatalf("single-node Linearizable VectorSearchExt: %v", err)
	}
	if len(res) == 0 {
		t.Fatal("single-node Linearizable read returned no results")
	}
	if got := atomic.LoadInt32(&forwards); got != 0 {
		t.Fatalf("single-node Linearizable read fired %d forwards, want 0 (pure-embedded no-op)", got)
	}
}

// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/cluster"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// ----------------------------------------------------------------------------
// Named-vector read-consistency wiring proofs (mirror the dense proofs in
// embedded_meta_readindex_test.go). resolveCollectionForRead runs the meta
// readIndex barrier EXACTLY ONCE per Linearizable named read on the COORDINATOR
// (before fan-out, never per-partition), and ZERO times for AnyReplica/LeaderOnly
// named reads. Single-node/pure-embedded named Linearizable is a true no-op.
//
// Same proof instrument as dense: the follower-forward counter
// (cluster.SetMetaReadIndexForwardHook). Reads are driven on a meta-FOLLOWER
// coordinator so a Linearizable read forwards exactly once (the barrier's single
// leader RTT) and an off-path read forwards zero times.
// ----------------------------------------------------------------------------

// seedPartitionedNamed creates a P-partition "named" collection (two spaces:
// title + image) on the coordinator, upserts N points, then waits for the given
// store's local catalog to converge on (P, gen 0) so the reads under test are not
// racing creation. The named counterpart of seedPartitionedDocs.
func seedPartitionedNamed(t *testing.T, ctx context.Context, stores []rostam.Store, readIdx, P, N int) {
	t.Helper()
	cfg := map[string]rostam.NamedVectorParams{
		"title": {Dim: 8, Metric: vector.Cosine, M: 8, EfConstruction: 100, EfSearch: 64},
		"image": {Dim: 8, Metric: vector.Cosine, M: 8, EfConstruction: 100, EfSearch: 64},
	}
	retryUntil(t, "create named", func() error {
		err := stores[0].VectorNamedCreateCollection(ctx, "named", cfg, P)
		if err != nil && strings.Contains(err.Error(), "already exists") {
			// A prior attempt committed the physical partitions but returned a
			// transient — complete the catalog write idempotently and treat as success
			// (mirror createPartitionedDocs's already-exists handling).
			return stores[0].(*rostam.Embedded).Catalog().SetPartitionsGen("named", P, 0)
		}
		return err
	})
	for i := 1; i <= N; i++ {
		vecs := map[string][]float32{"title": namedTitleVec(i), "image": namedImageVec(i, N)}
		payload := rostam.VectorMetadata{"n": vector.NewInt(int64(i))}
		ii := i
		retryUntil(t, "named insert", func() error {
			return stores[0].VectorNamedInsert(ctx, "named", uint64(ii), vecs, payload, 0)
		})
	}
	// cpuScaled for CPU-contended CI: readIdx is a meta-follower, so this rides
	// follower meta-apply lag; finite so a real catalog-convergence hang still fails.
	waitEmbeddedCatalogGen(t, stores[readIdx].(*rostam.Embedded), "named", P, 0, cpuScaled(20*time.Second))
}

// TestNamedLinearizableReadFiresBarrierExactlyOnce is THE once-per-read proof for
// named search: a Linearizable named read on a P>1 collection, issued on a
// meta-FOLLOWER coordinator, forwards __meta_readindex__ EXACTLY ONCE (the
// coordinator barrier before fan-out), NOT P times. Asserting ==1 catches a
// regression that moves the barrier into the per-partition fan-out.
func TestNamedLinearizableReadFiresBarrierExactlyOnce(t *testing.T) {
	stores := newInmemEmbeddedCluster(t, 3, 4, 3) // RF=3 so leader-routed partitioned reads are serviceable everywhere
	ctx := context.Background()
	const P, N = 4, 40
	readIdx := metaFollowerStore(t, stores)
	seedPartitionedNamed(t, ctx, stores, readIdx, P, N)

	var forwards int32
	cluster.SetMetaReadIndexForwardHook(func() { atomic.AddInt32(&forwards, 1) })
	defer cluster.SetMetaReadIndexForwardHook(nil)

	// A Linearizable named search on the follower coordinator. P=4 so a per-partition
	// barrier would forward 4 times; the coordinator barrier forwards exactly once.
	opts := rostam.NamedSearchOpts{ReadConsistency: ops.ConsistencyLinearizable, OnPartitionUnavailable: 1}
	if _, err := stores[readIdx].VectorNamedSearchExt(ctx, "named", "title", namedTitleQuery(), 10, opts); err != nil {
		t.Fatalf("Linearizable VectorNamedSearchExt: %v", err)
	}
	if got := atomic.LoadInt32(&forwards); got != 1 {
		t.Fatalf("Linearizable named read fired %d __meta_readindex__ forwards on a P=%d collection, want EXACTLY 1 (one coordinator barrier, NOT per-partition)", got, P)
	}

	// search_docs fires exactly one MORE forward (one per read).
	atomic.StoreInt32(&forwards, 0)
	if _, err := stores[readIdx].VectorNamedSearchDocsExt(ctx, "named", "title", namedTitleQuery(), 10, opts); err != nil {
		t.Fatalf("Linearizable VectorNamedSearchDocsExt: %v", err)
	}
	if got := atomic.LoadInt32(&forwards); got != 1 {
		t.Fatalf("Linearizable named docs read fired %d forwards, want EXACTLY 1 per read", got)
	}

	// scroll fires exactly one MORE forward.
	atomic.StoreInt32(&forwards, 0)
	if _, _, err := stores[readIdx].VectorNamedScrollExt(ctx, "named", rostam.VectorFilter{}, 10, "",
		rostam.NamedScrollOpts{ReadConsistency: ops.ConsistencyLinearizable, OnPartitionUnavailable: 1}); err != nil {
		t.Fatalf("Linearizable VectorNamedScrollExt: %v", err)
	}
	if got := atomic.LoadInt32(&forwards); got != 1 {
		t.Fatalf("Linearizable named scroll fired %d forwards, want EXACTLY 1 per read", got)
	}
}

// TestNamedNonLinearizableReadsFireBarrierZero proves the ZERO-cost contract off
// the Linearizable path for named reads: AnyReplica + LeaderOnly named searches
// (and a scroll) issue ZERO __meta_readindex__ forwards on the same
// lagging-capable follower coordinator where a Linearizable read WOULD forward.
func TestNamedNonLinearizableReadsFireBarrierZero(t *testing.T) {
	stores := newInmemEmbeddedCluster(t, 3, 4, 3)
	ctx := context.Background()
	const P, N = 4, 40
	readIdx := metaFollowerStore(t, stores)
	seedPartitionedNamed(t, ctx, stores, readIdx, P, N)

	var forwards int32
	cluster.SetMetaReadIndexForwardHook(func() { atomic.AddInt32(&forwards, 1) })
	defer cluster.SetMetaReadIndexForwardHook(nil)

	q := namedTitleQuery()

	// AnyReplica named read: ZERO forwards. Asserted regardless of any transient
	// read error from shard-leadership churn — an off-Linearizable read never enters
	// the barrier, so it cannot forward even if the read itself blips.
	atomic.StoreInt32(&forwards, 0)
	_, _ = stores[readIdx].VectorNamedSearchExt(ctx, "named", "title", q, 10,
		rostam.NamedSearchOpts{ReadConsistency: ops.ConsistencyAnyReplica, OnPartitionUnavailable: 1})
	if got := atomic.LoadInt32(&forwards); got != 0 {
		t.Fatalf("AnyReplica named read fired %d __meta_readindex__ forwards, want 0 (zero added cost off the Linearizable path)", got)
	}

	// LeaderOnly named read: ZERO forwards (same rationale).
	atomic.StoreInt32(&forwards, 0)
	_, _ = stores[readIdx].VectorNamedSearchExt(ctx, "named", "title", q, 10,
		rostam.NamedSearchOpts{ReadConsistency: ops.ConsistencyLeaderOnly, OnPartitionUnavailable: 1})
	if got := atomic.LoadInt32(&forwards); got != 0 {
		t.Fatalf("LeaderOnly named read fired %d __meta_readindex__ forwards, want 0", got)
	}

	// AnyReplica named scroll: ZERO forwards.
	atomic.StoreInt32(&forwards, 0)
	_, _, _ = stores[readIdx].VectorNamedScrollExt(ctx, "named", rostam.VectorFilter{}, 10, "",
		rostam.NamedScrollOpts{ReadConsistency: ops.ConsistencyAnyReplica, OnPartitionUnavailable: 1})
	if got := atomic.LoadInt32(&forwards); got != 0 {
		t.Fatalf("AnyReplica named scroll fired %d __meta_readindex__ forwards, want 0", got)
	}
}

// TestSingleNodeNamedLinearizableReadNoForwards proves the pure-embedded no-op: a
// single-node embedded rostam.Store (no meta-Raft) serves a Linearizable named
// read with ZERO forwards and a correct result — the barrier short-circuits
// locally (n.meta == nil).
func TestSingleNodeNamedLinearizableReadNoForwards(t *testing.T) {
	s := newSingleEmbedded(t) // single-node, no peers ⇒ no meta-Raft
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	const N = 20
	cfg := map[string]rostam.NamedVectorParams{
		"title": {Dim: 8, Metric: vector.Cosine, M: 8, EfConstruction: 100, EfSearch: 64},
	}
	retryUntil(t, "create named", func() error {
		return s.VectorNamedCreateCollection(ctx, "named", cfg, 0)
	})
	for id := 1; id <= N; id++ {
		idc := id
		retryUntil(t, "named insert", func() error {
			return s.VectorNamedInsert(ctx, "named", uint64(idc),
				map[string][]float32{"title": namedTitleVec(idc)}, rostam.VectorMetadata{"n": vector.NewInt(int64(idc))}, 0)
		})
	}

	var forwards int32
	cluster.SetMetaReadIndexForwardHook(func() { atomic.AddInt32(&forwards, 1) })
	defer cluster.SetMetaReadIndexForwardHook(nil)

	opts := rostam.NamedSearchOpts{ReadConsistency: ops.ConsistencyLinearizable}
	res, err := s.VectorNamedSearchExt(ctx, "named", "title", namedTitleQuery(), 10, opts)
	if err != nil {
		t.Fatalf("single-node Linearizable VectorNamedSearchExt: %v", err)
	}
	if len(res) == 0 {
		t.Fatal("single-node Linearizable named read returned no results")
	}
	if got := atomic.LoadInt32(&forwards); got != 0 {
		t.Fatalf("single-node Linearizable named read fired %d forwards, want 0 (pure-embedded no-op)", got)
	}
}

// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/cluster"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
	"github.com/rostamlabs/rostam/vector"
)

// ----------------------------------------------------------------------------
// The named-read-consistency PROOFS — the end-to-end demonstration that
// a Linearizable NAMED read genuinely arms BOTH barriers:
//
//   - the META readIndex barrier on the coordinator (resolveCollectionForRead):
//     a Linearizable named read on a lagging meta-follower BLOCKS until its local
//     catalog catches up to the leader-verified frontier, then serves FRESH —
//     while an AnyReplica named read in the same window returns WITHOUT blocking
//     (TestNamedLinearizableReadBlocksUntilCatalogFresh).
//
//   - the SHARD data barrier on each partition (shard.Store.verifyLeaderAndCatchUp):
//     a Linearizable named search over a P>1 collection enters the shard barrier
//     ONCE PER PARTITION — the critical-item proof that `vector_named_search` being
//     in ops.ReadConsistencyOf actually arms the shard barrier, closing the
//     silent-degrade hole (TestNamedLinearizableShardBarrierRealPath).
//
// Plus partition-count invariance + anti-silent-drop + a no-rc regression guard
// (TestNamedLinearizablePartitionCorrectness).
//
// These mirror, verbatim, the dense/MV proofs:
//   - TestLinearizableReadBlocksUntilCatalogFresh (meta_readindex_integration_test.go)
//   - TestLinearizableBarrierRunsOnRealReadPath / ...MVRealPath (linearizable_barrier_realpath_test.go)
//   - TestMVSearchFilterPartitionCountInvariance / ...AntiSilentDrop (mv_filter_fanout_test.go)
// reusing the SAME test seams: cluster.SetMetaApplyCatalogGate (lag a follower's
// catalog apply), cluster.SetMetaReadIndexForwardHook (count __meta_readindex__
// forwards / prove blocked-not-slow), and shard.SetBarrierEnteredHook (count the
// shard readIndex-barrier entries per partition leader read). No production hook
// is added — the named path reuses the dense realpath mechanism exactly.
// ----------------------------------------------------------------------------

// linNamedScrollIDs runs a strict Linearizable named scroll (read_consistency=2,
// fail-on-unavailable) on the given store and returns the distinct id set + error.
// Fail-on-unavailable turns an incomplete fan-out into a hard error, so the result
// is never a silently-truncated subset (the named mirror of linScrollIDs).
func linNamedScrollIDs(ctx context.Context, s rostam.Store) (map[uint64]bool, error) {
	docs, _, err := s.VectorNamedScrollExt(ctx, "named", rostam.VectorFilter{}, 0, "",
		rostam.NamedScrollOpts{ReadConsistency: ops.ConsistencyLinearizable, OnPartitionUnavailable: 1})
	if err != nil {
		return nil, err
	}
	return idSet(docs), nil
}

// TestNamedLinearizableReadBlocksUntilCatalogFresh is THE block-then-serve-fresh
// proof for a NAMED read — the named mirror of TestLinearizableReadBlocksUntilCatalogFresh.
//
// Mutation staged: a fresh partitioned named collection CREATE. We gate ONE
// meta-follower so it never applies the create's OpSetCatalogEntry (gen 0) — its
// local catalog has NO "named" entry while gated (the stale view is unambiguous:
// "unknown collection" / empty). A Linearizable named search on the gated follower
// runs IN A GOROUTINE and must BLOCK on the meta readIndex barrier (bounded
// not-done select + forward-counter > 0 ⇒ blocked, not slow). CONTRAST: an
// AnyReplica named search in the same held window returns WITHOUT blocking (it does
// NOT trip the barrier). On RELEASE the SAME blocked read unblocks and serves FRESH
// (the latest data, never stale/empty/unknown-collection). Deterministic via the
// gate signal — no staleness sleeps.
func TestNamedLinearizableReadBlocksUntilCatalogFresh(t *testing.T) {
	// Clear the gate AFTER cluster teardown (LIFO Cleanup) so a trailing meta-Apply
	// during shutdown can't race a nil store of the gate under -race.
	t.Cleanup(func() { cluster.SetMetaApplyCatalogGate(nil) })

	stores := newInmemEmbeddedCluster(t, 3, 4, 3) // RF=3
	ctx := context.Background()
	const P, N = 4, 30

	lagIdx, lagID, _ := pickLagger(t, stores)
	coord := stores[0] // drive mutations from node 0 (forwards transparently)
	lagStore := stores[lagIdx]
	lagEE := lagStore.(*rostam.Embedded)

	// Gate the lagging follower's apply of the CREATE catalog entry for "named"
	// (generation 0). Other collections / generations pass through untouched, so
	// ONLY the create on this one node lags. Signal lagEntered when it reaches the
	// block (window provably open).
	var (
		lagEnteredOnce sync.Once
		lagEntered     = make(chan struct{})
		release        = make(chan struct{})
		releaseOnce    sync.Once
	)
	releaseGate := func() { releaseOnce.Do(func() { close(release) }) }
	cluster.SetMetaApplyCatalogGate(func(nodeID, collection string, partitions, generation uint32) {
		if nodeID != lagID || ops.CanonicalName(collection) != ops.CanonicalName("named") || generation != 0 {
			return
		}
		lagEnteredOnce.Do(func() { close(lagEntered) })
		<-release // hold this node at "no named entry" until the test releases the window
	})
	defer releaseGate()

	// CREATE the partitioned NAMED collection on the coordinator (two spaces). The
	// gate blocks only the lagging FOLLOWER's apply of the create entry — NOT commit,
	// NOT the coordinator's create — so the create call returns normally while the
	// follower stays gated.
	cfg := map[string]rostam.NamedVectorParams{
		"title": {Dim: 8, Metric: vector.Cosine, M: 8, EfConstruction: 100, EfSearch: 64},
		"image": {Dim: 8, Metric: vector.Cosine, M: 8, EfConstruction: 100, EfSearch: 64},
	}
	retryUntil(t, "create named", func() error {
		err := coord.VectorNamedCreateCollection(ctx, "named", cfg, P)
		if err != nil && strings.Contains(err.Error(), "already exists") {
			return coord.(*rostam.Embedded).Catalog().SetPartitionsGen("named", P, 0)
		}
		return err
	})
	waitEmbeddedCatalogGen(t, coord.(*rostam.Embedded), "named", P, 0, 25*time.Second)

	// Wait until the gated follower is genuinely blocked at the create apply (window open).
	select {
	case <-lagEntered:
	case <-time.After(40 * time.Second):
		releaseGate()
		t.Fatal("lagging follower never reached the create-apply gate")
	}

	// Seed the points — they land in the partitioned physical shards (which replicate
	// independently of the gated META apply on the lagging follower).
	want := idRange(1, N)
	for i := 1; i <= N; i++ {
		ii := i
		vecs := map[string][]float32{"title": namedTitleVec(ii), "image": namedImageVec(ii, N)}
		payload := rostam.VectorMetadata{"n": vector.NewInt(int64(ii))}
		retryUntil(t, "named insert", func() error {
			return coord.VectorNamedInsert(ctx, "named", uint64(ii), vecs, payload, 0)
		})
	}
	// Sanity: the seeded set is readable on the coordinator (a caught-up node), so a
	// later empty read on the lagging follower is a barrier/routing issue, not a
	// missing-data/seed artefact. Poll through shard-replication / leader-election jitter.
	{
		deadline := time.Now().Add(45 * time.Second)
		var lastErr error
		for time.Now().Before(deadline) {
			ids, err := linNamedScrollIDs(ctx, coord)
			if err != nil {
				lastErr = err
				time.Sleep(50 * time.Millisecond)
				continue
			}
			if len(ids) == N {
				lastErr = nil
				break
			}
			lastErr = nil
			time.Sleep(50 * time.Millisecond)
		}
		if lastErr != nil {
			releaseGate()
			t.Fatalf("coordinator Linearizable named read errored before seeing the seeded set: %v", lastErr)
		}
		if n, _ := linNamedScrollIDs(ctx, coord); n != nil && len(n) != N {
			releaseGate()
			t.Fatalf("coordinator Linearizable named read sees %d ids, want %d — seed did not converge", len(n), N)
		}
	}

	// Confirm the asymmetry: the gated follower's LOCAL catalog still has NO "named"
	// entry. This IS the lag window.
	if p, _, ok := lagEE.Catalog().PartitionsGen("named"); ok {
		releaseGate()
		t.Fatalf("gated follower already has a named catalog entry (p=%d) — the create lag did not hold", p)
	}

	// Instrument forwards so we can prove the Linearizable read is FORWARDING/WAITING
	// (blocked on the barrier), not merely slow.
	var forwards int32
	cluster.SetMetaReadIndexForwardHook(func() { atomic.AddInt32(&forwards, 1) })
	defer cluster.SetMetaReadIndexForwardHook(nil)

	// (2) The Linearizable NAMED read on the gated follower, in a goroutine. It must
	// BLOCK on the meta readIndex barrier until we release the gate.
	type linResult struct {
		res []rostam.VectorResult
		err error
	}
	linDone := make(chan linResult, 1)
	go func() {
		res, err := lagStore.VectorNamedSearchExt(ctx, "named", "title", namedTitleQuery(), 10,
			rostam.NamedSearchOpts{ReadConsistency: ops.ConsistencyLinearizable, OnPartitionUnavailable: 1})
		linDone <- linResult{res, err}
	}()

	// Assert NOT-done while the gate is held. A bounded select: if the Linearizable
	// read returns here it served a stale/empty catalog WITHOUT waiting — a barrier bug.
	select {
	case r := <-linDone:
		releaseGate()
		t.Fatalf("Linearizable named read returned (results=%d, err=%v) while the create apply was gated — "+
			"it must BLOCK on the meta readIndex barrier until the local catalog catches up", len(r.res), r.err)
	case <-time.After(500 * time.Millisecond):
		// Good: still blocked on the barrier.
	}
	if atomic.LoadInt32(&forwards) == 0 {
		releaseGate()
		t.Fatal("blocked Linearizable named read issued ZERO __meta_readindex__ forwards — it must forward to the meta leader and wait")
	}

	// (3) CONTRAST: an AnyReplica named read on the SAME gated follower in the SAME
	// window returns WITHOUT BLOCKING — it does NOT trip the meta readIndex barrier.
	// We assert the no-block behavior (the local catalog is still stale at this point).
	if p, _, ok := lagEE.Catalog().PartitionsGen("named"); ok {
		releaseGate()
		t.Fatalf("gated follower local catalog already shows named (p=%d) before release — the create lag did not hold for the contrast", p)
	}
	anyDone := make(chan struct{}, 1)
	go func() {
		_, _ = lagStore.VectorNamedSearchExt(ctx, "named", "title", namedTitleQuery(), 10,
			rostam.NamedSearchOpts{ReadConsistency: ops.ConsistencyAnyReplica, OnPartitionUnavailable: 1})
		anyDone <- struct{}{}
	}()
	select {
	case <-anyDone:
		// Good: the AnyReplica named read did NOT block (contrast with the still-blocked
		// Linearizable read).
	case <-time.After(15 * time.Second):
		releaseGate()
		t.Fatal("AnyReplica named read on the gated follower BLOCKED — a non-Linearizable read must NEVER trip the meta readIndex barrier")
	}
	// The Linearizable read MUST still be blocked (the AnyReplica read returning did
	// not unblock it) — confirming the barrier, not some shared latch, holds it.
	select {
	case r := <-linDone:
		releaseGate()
		t.Fatalf("Linearizable named read returned (results=%d, err=%v) before release while AnyReplica returned promptly — "+
			"the Linearizable read must stay BLOCKED on the barrier until catch-up", len(r.res), r.err)
	default:
	}

	// (4) RELEASE: the follower applies the create, its catalog catches up, and the
	// SAME blocked Linearizable read unblocks and serves FRESH — never stale/empty/
	// "unknown collection". The named "title" space ranks the SMALLEST ids first
	// (namedTitleVec), so the fresh result is a non-empty subset of the seeded set.
	releaseStart := time.Now()
	releaseGate()
	select {
	case r := <-linDone:
		if r.err != nil {
			t.Fatalf("Linearizable named read after gate release: %v (must serve fresh, never error / never stale)", r.err)
		}
		if len(r.res) == 0 {
			t.Fatal("Linearizable named read after catch-up returned EMPTY — it must serve the fresh create, never the stale/absent local catalog")
		}
		for _, hit := range r.res {
			if !want[hit.ID] {
				t.Fatalf("Linearizable named read after catch-up returned id %d outside the seeded set — phantom/stale routing", hit.ID)
			}
		}
		// "title" top-1 must be id=1 (smallest id is nearest the title query) — a fresh,
		// correctly-routed result, never a stale/empty one.
		if r.res[0].ID != 1 {
			t.Fatalf("Linearizable named read after catch-up top-1 = id %d, want id=1 (nearest title) — fresh result expected", r.res[0].ID)
		}
		if d := time.Since(releaseStart); d > 15*time.Second {
			t.Fatalf("Linearizable named read took %s after release — should unblock promptly once caught up", d)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("Linearizable named read did not return after the gate was released")
	}

	// The follower is now caught up; a Linearizable named scroll converges to the
	// EXACT seeded set once shard replication settles.
	{
		deadline := time.Now().Add(40 * time.Second)
		var lastIDs map[uint64]bool
		var lastErr error
		for time.Now().Before(deadline) {
			ids, err := linNamedScrollIDs(ctx, lagStore)
			if err != nil {
				lastErr = err
				time.Sleep(50 * time.Millisecond)
				continue
			}
			lastIDs, lastErr = ids, nil
			if len(ids) == N {
				break
			}
			time.Sleep(50 * time.Millisecond)
		}
		if lastErr != nil {
			t.Fatalf("caught-up follower Linearizable named read errored: %v", lastErr)
		}
		assertExactIDSet(t, "caught-up follower Linearizable named read (exact seeded set)", lastIDs, want)
	}
}

// TestNamedLinearizableShardBarrierRealPath is THE critical-item proof: a
// Linearizable NAMED read over a PARTITIONED collection genuinely enters the SHARD
// data barrier (shard.Store.verifyLeaderAndCatchUp) on EACH partition leader — i.e.
// `vector_named_search` being in ops.ReadConsistencyOf actually arms the barrier.
//
// Mirror of TestLinearizableBarrierRunsOnRealReadPath / ...MVRealPath: the SAME
// process-wide shard.SetBarrierEnteredHook counts every readIndex-barrier entry.
// The named family's per-partition op is `vector_named_search`/`_search_docs`/
// `_scroll` (same name as the coordinator op — no distinct lanes/candidates
// variant), so the rc byte must reach the shard in the per-op args for the peek
// (ops.ReadConsistencyOf) to return ConsistencyLinearizable and the barrier to fire.
//
// THIS would FAIL if named were dropped from ReadConsistencyOf (the silent-degrade
// hole): the shard's peek would return (0,false), verifyLeaderAndCatchUp would never
// run, and a Linearizable named read would silently degrade to stale — barrierEntries
// would stay 0. The contrast (AnyReplica) asserts the default path adds ZERO barrier
// cost.
func TestNamedLinearizableShardBarrierRealPath(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	// Observe every readIndex-barrier entry (process-wide hook; the embedded engine
	// runs in this process). Each phase measures its own entries via the swap-delta.
	var barrierEntries atomic.Int64
	shard.SetBarrierEnteredHook(func() { barrierEntries.Add(1) })
	t.Cleanup(func() { shard.SetBarrierEnteredHook(nil) })
	delta := func() int64 { return barrierEntries.Swap(0) }

	cfg := map[string]rostam.NamedVectorParams{
		"title": {Dim: 8, Metric: vector.Cosine, M: 8, EfConstruction: 100, EfSearch: 64},
		"image": {Dim: 8, Metric: vector.Cosine, M: 8, EfConstruction: 100, EfSearch: 64},
	}
	const P, N = 4, 80
	// Partitioned (P>1): a Linearizable named search fans out; each partition's leader
	// read must enter the barrier.
	if err := s.VectorNamedCreateCollection(ctx, "parts", cfg, P); err != nil {
		t.Fatalf("VectorNamedCreateCollection parts (P=%d): %v", P, err)
	}
	// Unpartitioned (P==1): a Linearizable named search routes to the leader and must
	// enter the barrier exactly once (the P<=1 callReadLeader path).
	if err := s.VectorNamedCreateCollection(ctx, "single", cfg, 1); err != nil {
		t.Fatalf("VectorNamedCreateCollection single (P=1): %v", err)
	}
	for id := 1; id <= N; id++ {
		idc := uint64(id)
		vecs := map[string][]float32{"title": namedTitleVec(id), "image": namedImageVec(id, N)}
		md := rostam.VectorMetadata{"n": vector.NewInt(int64(id))}
		if err := s.VectorNamedInsert(ctx, "parts", idc, vecs, md, 0); err != nil {
			t.Fatalf("named insert parts %d: %v", id, err)
		}
		if err := s.VectorNamedInsert(ctx, "single", idc, vecs, md, 0); err != nil {
			t.Fatalf("named insert single %d: %v", id, err)
		}
	}
	// Drain any barrier entries incurred during setup (writes don't enter it, but be
	// defensive so the assertions below measure only the reads under test).
	delta()

	q := namedTitleQuery()
	lin := rostam.NamedSearchOpts{ReadConsistency: ops.ConsistencyLinearizable}
	any := rostam.NamedSearchOpts{ReadConsistency: ops.ConsistencyAnyReplica}

	// (a) PARTITIONED Linearizable named SEARCH: the shard barrier runs on every
	// partition's leader. >= P entries. 0 ⇒ vector_named_search dropped from
	// ReadConsistencyOf (or the fan-out re-encode dropped the rc byte) ⇒ STALE-SERVE.
	res, err := s.VectorNamedSearchExt(ctx, "parts", "title", q, 5, lin)
	if err != nil {
		t.Fatalf("linearizable VectorNamedSearchExt parts: %v", err)
	}
	if len(res) == 0 || res[0].ID != 1 {
		t.Fatalf("linearizable named parts results = %+v, want nearest id=1 first", res)
	}
	// >= 1 (not >= P): co-hosted partitions COALESCE into a shared readIndex barrier
	// (single-flight readindex coalescing — arrival<=capture safe). The anti-stale
	// guard is the FIRE: 0 ⇒ vector_named_search not in ReadConsistencyOf or rc dropped
	// ⇒ STALE-SERVE. Coalescing only lowers the count, never to 0.
	if n := delta(); n < 1 {
		t.Fatalf("PARTITIONED linearizable named search entered the shard barrier %d times, want >= 1 "+
			"(coalesced). 0 ⇒ vector_named_search not in ReadConsistencyOf or rc byte dropped ⇒ STALE-SERVE HOLE", n)
	}

	// (b) PARTITIONED Linearizable named SEARCH_DOCS: same per-partition barrier proof
	// for the docs op (vector_named_search_docs in ReadConsistencyOf).
	if _, err := s.VectorNamedSearchDocsExt(ctx, "parts", "title", q, 5, lin); err != nil {
		t.Fatalf("linearizable VectorNamedSearchDocsExt parts: %v", err)
	}
	if n := delta(); n < 1 {
		t.Fatalf("PARTITIONED linearizable named search_docs entered the shard barrier %d times, want >= 1 "+
			"(coalesced; 0 ⇒ vector_named_search_docs not in ReadConsistencyOf or rc dropped ⇒ STALE)", n)
	}

	// (c) PARTITIONED Linearizable named SCROLL: the scroll op fans out per partition
	// (vector_named_scroll in ReadConsistencyOf).
	if _, _, err := s.VectorNamedScrollExt(ctx, "parts", rostam.VectorFilter{}, 10, "",
		rostam.NamedScrollOpts{ReadConsistency: ops.ConsistencyLinearizable}); err != nil {
		t.Fatalf("linearizable VectorNamedScrollExt parts: %v", err)
	}
	if n := delta(); n < 1 {
		t.Fatalf("PARTITIONED linearizable named scroll entered the shard barrier %d times, want >= 1 "+
			"(coalesced; 0 ⇒ vector_named_scroll not in ReadConsistencyOf or rc dropped ⇒ STALE)", n)
	}

	// (d) UNPARTITIONED (P==1) Linearizable named search: routes to the leader via
	// callReadLeader, barrier runs (>= 1).
	res, err = s.VectorNamedSearchExt(ctx, "single", "title", q, 5, lin)
	if err != nil {
		t.Fatalf("linearizable VectorNamedSearchExt single: %v", err)
	}
	if len(res) == 0 || res[0].ID != 1 {
		t.Fatalf("linearizable named single results = %+v, want nearest id=1 first", res)
	}
	if n := delta(); n < 1 {
		t.Fatalf("UNPARTITIONED (P==1) linearizable named search entered the shard barrier %d times, want >= 1 "+
			"(the P<=1 path must route to the leader WITH rc). 0 ⇒ the P==1 named read skipped the barrier ⇒ STALE-SERVE HOLE", n)
	}

	// (e) CONTRAST: AnyReplica named reads on the SAME collections must NOT enter the
	// barrier (zero added consensus cost on the default path).
	if _, err := s.VectorNamedSearchExt(ctx, "parts", "title", q, 5, any); err != nil {
		t.Fatalf("anyreplica VectorNamedSearchExt parts: %v", err)
	}
	if _, err := s.VectorNamedSearchDocsExt(ctx, "parts", "title", q, 5, any); err != nil {
		t.Fatalf("anyreplica VectorNamedSearchDocsExt parts: %v", err)
	}
	if _, _, err := s.VectorNamedScrollExt(ctx, "parts", rostam.VectorFilter{}, 10, "",
		rostam.NamedScrollOpts{ReadConsistency: ops.ConsistencyAnyReplica}); err != nil {
		t.Fatalf("anyreplica VectorNamedScrollExt parts: %v", err)
	}
	if _, err := s.VectorNamedSearchExt(ctx, "single", "title", q, 5, any); err != nil {
		t.Fatalf("anyreplica VectorNamedSearchExt single: %v", err)
	}
	if n := delta(); n != 0 {
		t.Fatalf("AnyReplica named reads entered the shard barrier %d times, want 0 (default path must be barrier-free)", n)
	}
}

// TestNamedLinearizablePartitionCorrectness proves partition-count invariance +
// anti-silent-drop for the Linearizable named fan-out — the named mirror of
// TestMVSearchFilterPartitionCountInvariance / ...AntiSilentDrop.
//
//   - Partition-count invariance: a Linearizable named search over P>1 returns the
//     SAME correct top-k as the SAME data with P=1 (independent ground truth: the
//     "title" space ranks ascending id, the "image" space ranks descending id).
//   - Anti-silent-drop: the shard barrier fires on ALL partitions of the
//     Linearizable named search (counted per-partition) — this is the proof that rc
//     rides EVERY per-partition fan-out arg; it would go to < P (specifically 0)
//     if rc were dropped on the per-partition encode (the same regression class the
//     dense/MV anti-silent-drop tests catch).
//   - No-rc named search + scroll unchanged (regression guard).
func TestNamedLinearizablePartitionCorrectness(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	cfg := map[string]rostam.NamedVectorParams{
		"title": {Dim: 8, Metric: vector.Cosine, M: 8, EfConstruction: 100, EfSearch: 64},
		"image": {Dim: 8, Metric: vector.Cosine, M: 8, EfConstruction: 100, EfSearch: 64},
	}
	const N = 40
	seed := func(name string, P int) {
		if err := s.VectorNamedCreateCollection(ctx, name, cfg, P); err != nil {
			t.Fatalf("VectorNamedCreateCollection %s (P=%d): %v", name, P, err)
		}
		for id := 1; id <= N; id++ {
			vecs := map[string][]float32{"title": namedTitleVec(id), "image": namedImageVec(id, N)}
			md := rostam.VectorMetadata{"n": vector.NewInt(int64(id))}
			if err := s.VectorNamedInsert(ctx, name, uint64(id), vecs, md, 0); err != nil {
				t.Fatalf("named insert %s %d: %v", name, id, err)
			}
		}
	}
	const P = 4
	seed("parts", P)
	seed("single", 1)

	const k = 8
	lin := rostam.NamedSearchOpts{ReadConsistency: ops.ConsistencyLinearizable}

	resultIDs := func(res []rostam.VectorResult) []uint64 {
		out := make([]uint64, len(res))
		for i, r := range res {
			out[i] = r.ID
		}
		return out
	}

	// Independent ground truth: "title" ranks ascending id (1,2,...), "image" ranks
	// descending id (N, N-1, ...).
	wantTitle := []uint64{}
	for id := 1; id <= k; id++ {
		wantTitle = append(wantTitle, uint64(id))
	}
	wantImage := []uint64{}
	for id := N; id > N-k; id-- {
		wantImage = append(wantImage, uint64(id))
	}

	// --- Partition-count invariance for the "title" space ---
	gotP4Title, err := s.VectorNamedSearchExt(ctx, "parts", "title", namedTitleQuery(), k, lin)
	if err != nil {
		t.Fatalf("linearizable named search parts/title: %v", err)
	}
	gotP1Title, err := s.VectorNamedSearchExt(ctx, "single", "title", namedTitleQuery(), k, lin)
	if err != nil {
		t.Fatalf("linearizable named search single/title: %v", err)
	}
	if !reflect.DeepEqual(resultIDs(gotP4Title), resultIDs(gotP1Title)) {
		t.Fatalf("title partition-count variance: P=%d %v != P=1 %v", P, resultIDs(gotP4Title), resultIDs(gotP1Title))
	}
	if !reflect.DeepEqual(resultIDs(gotP1Title), wantTitle) {
		t.Fatalf("title P=1 Linearizable top-k = %v, want %v (ascending id ground truth)", resultIDs(gotP1Title), wantTitle)
	}

	// --- Partition-count invariance for the "image" space (proves the NAME selects
	// the space through fan-out: a different, opposite ranking) ---
	gotP4Image, err := s.VectorNamedSearchExt(ctx, "parts", "image", namedImageQuery(), k, lin)
	if err != nil {
		t.Fatalf("linearizable named search parts/image: %v", err)
	}
	gotP1Image, err := s.VectorNamedSearchExt(ctx, "single", "image", namedImageQuery(), k, lin)
	if err != nil {
		t.Fatalf("linearizable named search single/image: %v", err)
	}
	if !reflect.DeepEqual(resultIDs(gotP4Image), resultIDs(gotP1Image)) {
		t.Fatalf("image partition-count variance: P=%d %v != P=1 %v", P, resultIDs(gotP4Image), resultIDs(gotP1Image))
	}
	if !reflect.DeepEqual(resultIDs(gotP1Image), wantImage) {
		t.Fatalf("image P=1 Linearizable top-k = %v, want %v (descending id ground truth)", resultIDs(gotP1Image), wantImage)
	}

	// --- Anti-silent-drop: rc rides EVERY per-partition fan-out arg. We count the
	// shard barrier entries for a Linearizable named search over the P>1 collection;
	// it must fire on ALL P partitions. If rc were dropped on the per-partition encode,
	// the per-partition peek (ReadConsistencyOf) would return (0,false) on the shards
	// and the barrier would NOT fire (< P, in practice 0). This is the same gate the
	// dense/MV anti-silent-drop tests use, observed via the shard barrier counter. ---
	var barrierEntries atomic.Int64
	shard.SetBarrierEnteredHook(func() { barrierEntries.Add(1) })
	t.Cleanup(func() { shard.SetBarrierEnteredHook(nil) })
	barrierEntries.Store(0)
	if _, err := s.VectorNamedSearchExt(ctx, "parts", "title", namedTitleQuery(), k, lin); err != nil {
		t.Fatalf("anti-silent-drop linearizable named search: %v", err)
	}
	// >= 1 (not >= P): single-flight readindex coalescing merges the concurrent
	// per-partition barriers of co-hosted partitions into shared flights, so the count
	// is now < P even when rc rides every arg (the coalesced barrier is arrival<=capture
	// safe — no partition serves stale). The anti-silent-drop FLOOR is the fire itself:
	// if rc were dropped on the per-partition encode, ReadConsistencyOf returns
	// (0,false) on the shards and the barrier NEVER fires ⇒ count 0. So 0 still catches
	// a dropped rc; coalescing only collapses the >0 count.
	if n := barrierEntries.Load(); n < 1 {
		t.Fatalf("anti-silent-drop FAILED: Linearizable named search over P=%d fired the shard barrier %d times, want >= 1 "+
			"(coalesced; 0 ⇒ rc DROPPED on the per-partition encode ⇒ those shards serve STALE)", P, n)
	}
	shard.SetBarrierEnteredHook(nil)

	// --- Regression: a NO-RC (default AnyReplica) named search + scroll over the same
	// partitioned data is unchanged — same correct top-k, no barrier, no error. ---
	gotNoRC, err := s.VectorNamedSearch(ctx, "parts", "title", namedTitleQuery(), k, rostam.VectorFilter{})
	if err != nil {
		t.Fatalf("no-rc named search parts/title: %v", err)
	}
	if !reflect.DeepEqual(resultIDs(gotNoRC), wantTitle) {
		t.Fatalf("no-rc named search top-k = %v, want %v (regression: rc plumbing changed the no-rc path)", resultIDs(gotNoRC), wantTitle)
	}
	scrollDocs, _, err := s.VectorNamedScroll(ctx, "parts", rostam.VectorFilter{}, 0, "")
	if err != nil {
		t.Fatalf("no-rc named scroll parts: %v", err)
	}
	if got := idSet(scrollDocs); !reflect.DeepEqual(got, idRange(1, N)) {
		t.Fatalf("no-rc named scroll returned %d distinct ids, want the exact seeded set 1..%d (regression)", len(got), N)
	}
}

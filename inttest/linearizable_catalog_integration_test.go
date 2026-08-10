// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/cluster"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// ----------------------------------------------------------------------------
// The linearizability proof — a lagging meta-follower, reading DURING
// the post-cutover/pre-gate window, never reads a stale OR dropped generation.
//
// The invariant under test: the online reshard dual-writes BOTH the
// old and the new generation for as long as Status==Resharding (incl. AFTER the
// Phase-4 cutover flip), and the Phase-4.5 gate (waitAllNodesCatalogGen) keeps
// the old gen ALIVE + state Resharding until EVERY node routes to the new gen.
// So a follower that lags the cutover (still routes to the old gen) reads an old
// gen that is (a) still present (gate hasn't dropped it) and (b) still fresh
// (dual-write keeps mirroring every write, including writes committed AFTER the
// cutover). That is linearizable.
//
// Lagging mechanism (deterministic): a test-only meta-FSM apply gate
// (cluster.SetMetaApplyCatalogGate) BLOCKS exactly ONE node's apply of the
// cutover OpSetCatalogEntry. It fires BEFORE the FSM write lock, so the blocked
// node keeps serving its prior catalog state (the old gen) — it does NOT hang,
// it LAGS. The reshard runs in a goroutine; the coordinator flips the catalog
// and enters the Phase-4.5 gate, which BLOCKS (the lagging node never confirms
// the new gen) — holding the post-cutover window open deterministically. We do
// the proof reads on the lagging node during that window, then release the gate.
// (No sleeps gate the window: lagEntered + the gate's own blocking do.)
//
// Cluster shape: RF=3 (every shard replicated on all nodes) so a LEADER-routed
// Linearizable read (read_consistency=2) can resolve every partition's leader
// locally and run the readIndex barrier — the same RF=3 shape
// TestClusterReshardGateUnreachableNodeProceeds uses. (At RF=1 a non-hosting node
// cannot resolve a remote shard's leader, so leader-routed partitioned reads are
// not serviceable in this in-process harness; that is a pre-existing harness
// limitation, unrelated to the catalog gen-routing gap under test.)
// ----------------------------------------------------------------------------

// lagNodeID is the cluster NodeID we make lag the cutover. The cluster harness
// names peers n1..nN; n3 is a non-coordinator follower (coordinator is n1/node 0).
const lagNodeID = "n3"

// TestReshardLaggingFollowerOldGenFresh is THE reshard-side-gate proof. A 3-node
// (RF=3) cluster, a partitioned collection seeded with data. Node 2 ("n3") is made
// to LAG the reshard cutover (its meta-apply of the new-gen catalog entry is
// blocked). While it lags — in the post-cutover, pre-gate-pass window — a LeaderOnly
// read (read_consistency=1) on that node:
//
//	(1) returns the EXACT seeded set (not stale, not an "unknown collection"
//	    error) — it routes to the old gen, which is still alive + fresh; AND
//	(2) after a NEW value is written DURING the reshard (after the cutover, before
//	    the gate passes), reflects that new value too — because dual-write mirrors
//	    the post-cutover write into the old gen the lagging node is reading.
//
// (2) is the freshness proof: the lagging follower's old-gen read reflects a write
// committed AFTER the cutover. It would FAIL without the dual-write (it would have
// collapsed to new-gen-only at the cutover ⇒ old gen stale, missing the new value)
// or without the Phase-4.5 all-nodes gate (the gate would have dropped the old gen
// on a fixed drain ⇒ the read errors / loses the partition).
//
// Consistency level — why LeaderOnly, not Linearizable: this proof is about the
// lagging follower routing to (and serving fresh from) the OLD gen via its LOCAL
// catalog. LeaderOnly is exactly that read: it resolves (P, gen) from the lagging
// local catalog (still oldP) and reads each partition from its old-gen shard leader
// (which holds every dual-written write ⇒ deterministically fresh), and it does NOT
// trip the Task-3 meta readIndex barrier. Under that barrier a Linearizable read on
// a perpetually-lagging follower instead BLOCKS-then-catches-up to the leader meta
// frontier (the new gen) rather than serving the lagging local catalog — so it is no
// longer the read that exercises lagging old-gen routing. The Linearizable
// block-then-serve-fresh behavior is proven in the Task-4 tests
// (meta_readindex_integration_test.go). Every assertion here keeps its teeth: the
// routing guard still pins oldP during the window / newP after, the exact-set and
// read-your-writes checks are unchanged, and a stale/dropped old gen would still
// surface as a persistent unknown-collection / no-shard-owner error (NOT masked).
func TestReshardLaggingFollowerOldGenFresh(t *testing.T) {
	defer rostam.SetReshardDrainGrace(20 * time.Millisecond)()

	stores := newInmemEmbeddedCluster(t, 3, 8, 3) // RF=3: leader-routed reads serviceable on every node
	ctx := context.Background()
	const oldP, newP, N = 4, 8, 200
	const newID = uint64(N + 1) // the post-cutover write proving freshness

	createCollectionTolerant(t, ctx, stores[0], "docs", rostam.VectorConfig{
		Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: oldP,
	})
	for id := uint64(1); id <= N; id++ {
		idc := id
		retryUntil(t, "upsert", func() error {
			return stores[0].VectorUpsert(ctx, "docs", idc, []float32{float32(idc), 0, 0, 0}, fmt.Sprintf("doc-%d", idc), rostam.VectorInsertOpts{})
		})
	}

	ee0 := stores[0].(*rostam.Embedded)
	ee2 := stores[2].(*rostam.Embedded) // the lagging follower (n3)
	// Every node has converged on the seeded (oldP, gen 0) catalog before we start,
	// so the ONLY lag in the test is the one we induce on the cutover. Also confirm
	// a LeaderOnly read is serviceable on n3 up front (full set, no error) — so a
	// later failure is the gap, not a harness/leader-resolution artefact.
	waitEmbeddedCatalogGen(t, ee2, "docs", oldP, 0, 10*time.Second)
	want1 := idRange(1, N)
	pre := laggingLeaderScrollFull(t, ctx, stores[2], ee2, oldP)
	assertExactIDSet(t, "baseline lagging read (pre-reshard)", pre, want1)

	// Install the deterministic lag: block n3's apply of the cutover entry (the
	// new-gen OpSetCatalogEntry for "docs", generation 1). Gen-0 (create) and other
	// collections pass through untouched, so ONLY the cutover lags. We signal
	// lagEntered when n3 reaches the block (so the test knows the window is open)
	// and release it by closing release.
	var (
		lagEnteredOnce sync.Once
		lagEntered     = make(chan struct{})
		release        = make(chan struct{})
	)
	cluster.SetMetaApplyCatalogGate(func(nodeID, collection string, partitions, generation uint32) {
		if nodeID != lagNodeID || ops.CanonicalName(collection) != ops.CanonicalName("docs") || generation != 1 {
			return // not n3's cutover apply — apply immediately
		}
		lagEnteredOnce.Do(func() { close(lagEntered) })
		<-release // hold n3 at the old gen until the test releases the window
	})
	defer cluster.SetMetaApplyCatalogGate(nil)
	// Guarantee the gate is released even if an assertion fails before we close it,
	// so the reshard goroutine can drain and t.Cleanup (node shutdown) won't hang.
	var releaseOnce sync.Once
	releaseGate := func() { releaseOnce.Do(func() { close(release) }) }
	defer releaseGate()

	// Run the reshard in the background. It flips the catalog (Phase 4 cutover),
	// then BLOCKS in the Phase-4.5 gate because n3 never confirms the new gen.
	reshardErr := make(chan error, 1)
	go func() { reshardErr <- stores[0].VectorReshard(ctx, "docs", newP) }()

	// Wait until n3 is genuinely lagging at the cutover.
	select {
	case <-lagEntered:
	case <-time.After(20 * time.Second):
		t.Fatal("n3 never reached the cutover-apply gate (reshard never cut over?)")
	}

	// The coordinator (n1/node 0) must have applied the cutover (new gen) by now;
	// the gate is what holds the window open. Confirm the asymmetry: node 0 shows
	// the NEW gen, the lagging n3 STILL shows the OLD gen. This IS the window.
	waitEmbeddedCatalogGen(t, ee0, "docs", newP, 1, 5*time.Second)
	if p, gen, ok := ee2.Catalog().PartitionsGen("docs"); !ok || p != oldP || gen != 0 {
		t.Fatalf("lagging n3 catalog = (p=%d, gen=%d, ok=%v), want the OLD gen (p=%d, gen=0) — the lag did not hold", p, gen, ok, oldP)
	}
	// Dual-write must still be on (Status==Resharding) — old gen is being kept fresh.
	if st, on := ee0.Catalog().ReshardState("docs"); !on || st.Status != 1 {
		t.Fatalf("reshard state on coordinator = %+v on=%v, want Status==1 (dual-write must stay on through the gate)", st, on)
	}

	// PROOF PART 1 — the lagging LeaderOnly read returns the EXACT seeded set
	// (never stale, never an unknown-collection error). It routes to the old gen
	// (n3 still shows oldP/gen0; laggingLeaderScrollFull re-asserts that the read
	// routed to oldP), the old gen is alive (gate hasn't dropped it) and fresh
	// (dual-write mirrored every seed write into it).
	got1 := laggingLeaderScrollFull(t, ctx, stores[2], ee2, oldP)
	assertExactIDSet(t, "lagging read (pre-new-write)", got1, want1)

	// PROOF PART 2 (the old-gen freshness proof) — write a NEW value DURING the
	// reshard, AFTER the cutover, BEFORE the gate passes. Dual-write puts it in
	// BOTH gens; the lagging n3, reading the OLD gen, must SEE it.
	retryUntil(t, "post-cutover upsert", func() error {
		return stores[0].VectorUpsert(ctx, "docs", newID, []float32{float32(newID), 0, 0, 0}, fmt.Sprintf("doc-%d", newID), rostam.VectorInsertOpts{})
	})
	// n3 is STILL lagging (we have not released the gate) — re-confirm before the
	// proof read so the assertion is unambiguous.
	if p, gen, ok := ee2.Catalog().PartitionsGen("docs"); !ok || p != oldP || gen != 0 {
		t.Fatalf("n3 cut over before the proof read (p=%d gen=%d ok=%v) — window closed early", p, gen, ok)
	}
	got2 := laggingLeaderScrollFull(t, ctx, stores[2], ee2, oldP)
	if !got2[newID] {
		t.Fatalf("OLD-GEN FRESHNESS VIOLATION: lagging old-gen read MISSING the post-cutover write id=%d. "+
			"The dual-write must mirror post-cutover writes into the old gen; without it the old gen would collapse to new-gen-only here.", newID)
	}
	assertExactIDSet(t, "lagging read (post-new-write, the old-gen freshness proof)", got2, idRange(1, int(newID)))

	// Release the lag: n3 applies the cutover, the gate passes, the reshard finishes.
	releaseGate()
	select {
	case err := <-reshardErr:
		must(t, err)
	case <-time.After(20 * time.Second):
		t.Fatal("reshard did not complete after the lag was released")
	}

	// After completion: every node routes to the new gen and a LeaderOnly read on
	// the (now caught-up) follower sees the full set (seeded + the post-cutover
	// write); the old gen is gone on the coordinator.
	waitEmbeddedCatalogGen(t, ee2, "docs", newP, 1, 10*time.Second)
	final := laggingLeaderScrollFull(t, ctx, stores[2], ee2, newP)
	assertExactIDSet(t, "post-reshard read on the (now caught-up) follower", final, idRange(1, int(newID)))
	for p := 0; p < oldP; p++ {
		if genPartitionExists(t, ee0, "docs", 0, p) {
			t.Fatalf("old gen-0 partition %d still present after the completed reshard", p)
		}
	}
}

// TestReshardGateWaitsForAllNodes asserts the gate genuinely WAITS on the lagging
// node: while n3 lags the cutover apply, the reshard coordinator does NOT clear
// the reshard state and does NOT drop the old gen — old gen survives + Status
// stays Resharding for as long as a node lags. Once n3 catches up the gate passes
// and the reshard completes. The wait is gated on the actual all-applied
// condition (n3's catalog showing the new gen via __catalog_gen__), NOT a fixed
// sleep — releasing the lag is what lets the reshard finish.
func TestReshardGateWaitsForAllNodes(t *testing.T) {
	defer rostam.SetReshardDrainGrace(20 * time.Millisecond)()

	stores := newInmemEmbeddedCluster(t, 3, 8, 3) // RF=3
	ctx := context.Background()
	const oldP, newP, N = 4, 8, 120

	createCollectionTolerant(t, ctx, stores[0], "docs", rostam.VectorConfig{
		Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: oldP,
	})
	for id := uint64(1); id <= N; id++ {
		idc := id
		retryUntil(t, "upsert", func() error {
			return stores[0].VectorUpsert(ctx, "docs", idc, []float32{float32(idc), 0, 0, 0}, fmt.Sprintf("doc-%d", idc), rostam.VectorInsertOpts{})
		})
	}
	ee0 := stores[0].(*rostam.Embedded)
	ee2 := stores[2].(*rostam.Embedded)
	waitEmbeddedCatalogGen(t, ee2, "docs", oldP, 0, 10*time.Second)

	var (
		lagEnteredOnce sync.Once
		lagEntered     = make(chan struct{})
		release        = make(chan struct{})
	)
	cluster.SetMetaApplyCatalogGate(func(nodeID, collection string, partitions, generation uint32) {
		if nodeID != lagNodeID || ops.CanonicalName(collection) != ops.CanonicalName("docs") || generation != 1 {
			return
		}
		lagEnteredOnce.Do(func() { close(lagEntered) })
		<-release
	})
	defer cluster.SetMetaApplyCatalogGate(nil)
	var releaseOnce sync.Once
	releaseGate := func() { releaseOnce.Do(func() { close(release) }) }
	defer releaseGate()

	reshardErr := make(chan error, 1)
	go func() { reshardErr <- stores[0].VectorReshard(ctx, "docs", newP) }()

	select {
	case <-lagEntered:
	case <-time.After(20 * time.Second):
		t.Fatal("n3 never reached the cutover-apply gate")
	}
	// Coordinator cut over; the gate is now blocked waiting on n3.
	waitEmbeddedCatalogGen(t, ee0, "docs", newP, 1, 5*time.Second)

	// While n3 lags, the reshard MUST NOT progress past the gate. Hold the lag for
	// a window that comfortably exceeds the drain grace and any prompt-completion
	// timing, and assert throughout: the reshard goroutine has NOT returned, state
	// stays Resharding, AND the old gen is fully present.
	deadline := time.Now().Add(1 * time.Second)
	for time.Now().Before(deadline) {
		select {
		case err := <-reshardErr:
			t.Fatalf("reshard returned (err=%v) while n3 still lags the cutover — the gate did NOT wait", err)
		default:
		}
		if st, on := ee0.Catalog().ReshardState("docs"); !on || st.Status != 1 {
			t.Fatalf("reshard state cleared while n3 lags (%+v on=%v) — the gate did NOT wait", st, on)
		}
		for p := 0; p < oldP; p++ {
			if !genPartitionExists(t, ee0, "docs", 0, p) {
				t.Fatalf("old gen-0 partition %d dropped while n3 lags — the gate did NOT wait", p)
			}
		}
		time.Sleep(50 * time.Millisecond)
	}

	// Now let n3 catch up. The all-applied condition is met; the gate passes and the
	// reshard completes — proving the wait was gated on catalog-gen convergence, not
	// a fixed sleep.
	releaseGate()
	select {
	case err := <-reshardErr:
		must(t, err)
	case <-time.After(20 * time.Second):
		t.Fatal("reshard did not complete after n3 caught up")
	}
	if st, on := ee0.Catalog().ReshardState("docs"); on || st.Status != 0 {
		t.Fatalf("reshard state still set after the gate passed: %+v on=%v", st, on)
	}
	for p := 0; p < oldP; p++ {
		if genPartitionExists(t, ee0, "docs", 0, p) {
			t.Fatalf("old gen-0 partition %d not dropped after the gate passed", p)
		}
	}
}

// laggingLeaderScrollFull performs a strict LeaderOnly scroll (read_consistency=1)
// on the given node and returns the distinct id set, requiring the read to route
// to the EXPECTED gen's partition count (wantP) — so it genuinely exercises the
// node's catalog routing (oldP during the lag window, newP after catch-up). It
// uses Fail-on-unavailable so an incomplete fan-out is an ERROR (never a silently
// truncated subset — the exact-set assertion would otherwise be undermined), and
// retries through transient raft leader-election jitter (the gate holds the lag
// window open, so the lagging read has time to land complete). The retry can NOT
// mask the gap: a stale/dropped old gen surfaces as a PERSISTENT unknown-
// collection / no-shard-owner error that exhausts the deadline, and the routing
// guard re-checks wantP on every attempt.
//
// Consistency level: LeaderOnly (NOT Linearizable). This helper proves the
// reshard-side gate's guarantee — that a follower routing to the OLD gen serves
// FRESH data because dual-write keeps the old gen alive+current until all nodes
// cut over. That guarantee is exercised by reads that route via the lagging
// follower's LOCAL catalog: LeaderOnly resolves (P, gen) from the lagging local
// catalog (still oldP), then reads each partition from its old-gen SHARD LEADER —
// which holds every dual-written write, so the result is deterministically fresh —
// and it does NOT trip the meta readIndex barrier. Linearizable is NO LONGER such
// a read: under the Task-3 read-side barrier it first BLOCKS until the
// coordinator's local meta-FSM reaches the leader-verified frontier (it catches up
// to the new gen rather than serving the lagging local catalog), so a Linearizable
// read on a perpetually-lagging follower correctly times out instead of routing
// oldP. The Linearizable block-then-serve-fresh behavior is proven separately in
// the Task-4 tests (meta_readindex_integration_test.go).
func laggingLeaderScrollFull(t *testing.T, ctx context.Context, s rostam.Store, ee *rostam.Embedded, wantP int) map[uint64]bool {
	t.Helper()
	opts := rostam.VectorScrollOpts{ReadConsistency: ops.ConsistencyLeaderOnly, OnPartitionUnavailable: 1} // LeaderOnly + Fail
	deadline := time.Now().Add(15 * time.Second)
	var lastErr error
	var lastDegraded bool
	var lastN int
	for time.Now().Before(deadline) {
		if p, _, ok := ee.Catalog().PartitionsGen("docs"); !ok || p != wantP {
			t.Fatalf("laggingLeaderScrollFull: node routes to p=%d (ok=%v), want p=%d — routing changed under the read", p, ok, wantP)
		}
		docs, meta, _, err := s.VectorScroll(ctx, "docs", rostam.VectorFilter{}, 0, opts)
		if err != nil {
			lastErr = err
			time.Sleep(50 * time.Millisecond)
			continue
		}
		if meta.Degraded {
			lastDegraded = true
			lastN = len(idSet(docs))
			time.Sleep(50 * time.Millisecond)
			continue
		}
		return idSet(docs)
	}
	t.Fatalf("laggingLeaderScrollFull: no complete non-degraded LeaderOnly read within budget "+
		"(lastErr=%v lastDegraded=%v lastN=%d, wantP=%d) — a PERSISTENT failure here means the gen the "+
		"node routes to is stale/dropped (the gap), not transient election", lastErr, lastDegraded, lastN, wantP)
	return nil
}

// ----------------------------------------------------------------------------
// Cluster integration — continuous Linearizable reads from ALL nodes
// across a complete online reshard end-to-end. Where the staged-gate test proves ONE staged
// lagging read in the critical window, this drives a CONTINUOUS reader pool on
// every node for the WHOLE reshard duration (before cutover, during the held-open
// post-cutover/pre-gate window, through the Phase-4.5 gate, and after old-gen
// drop), AND mutates the collection mid-reshard, asserting every read is correct
// + linearizable against independent ground truth. Zero errors, zero stale/extra.
// ----------------------------------------------------------------------------

// committedSet is the test's independent ground-truth oracle. Ids are written in
// strict ascending order, so it tracks two contiguous high-water-marks:
//
//   - committed: ids 1..committed have all RETURNED from VectorUpsert ⇒ durably
//     applied ⇒ a correct read MUST observe them (else stale/dropped gen).
//   - inflight: the highest id whose upsert has STARTED (may already be visible to
//     a reader before its call returns) — the upper bound a read may legitimately
//     observe. A read that observes an id > inflight is a true phantom.
//
// A Linearizable read is sound iff it contains every id in 1..committedBefore (no
// stale loss) and no id beyond max(committedAfter, inflightAfter) (no phantom).
// Because a write can become visible to a reader the instant the leader applies it
// — BEFORE the upsert call returns and BEFORE commit() runs — the upper bound uses
// inflight (raised before the call) not committed (raised after); otherwise a
// legitimately-fresh read races the oracle and looks like a phantom. snapshot()
// returns (committed, inflight) under one lock so a reader brackets its read.
type committedSet struct {
	mu        sync.Mutex
	committed uint64 // ids 1..committed have all returned from upsert
	inflight  uint64 // highest id whose upsert has begun (>= committed)
}

func (c *committedSet) snapshot() (committed, inflight uint64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.committed, c.inflight
}

// beginWrite raises the inflight mark BEFORE the upsert is issued (ids written in
// ascending order, so the highest started id bounds what a reader may observe).
func (c *committedSet) beginWrite(id uint64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if id > c.inflight {
		c.inflight = id
	}
}

// commit raises the committed mark AFTER the upsert returns (the id is now durable
// ⇒ every subsequent read must observe it).
func (c *committedSet) commit(id uint64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if id == c.committed+1 {
		c.committed = id
	}
}

// assertLinearizableEnvelope checks a read's id set against the ground-truth
// envelope: it MUST contain every id in 1..before (the committed mark BEFORE the
// read began — else a durable write was lost ⇒ stale/dropped gen), and MUST NOT
// contain any id outside 1..after (the inflight upper bound AFTER the read
// returned — else a phantom beyond any write that had even started). Within the
// (before, after] band the id may or may not appear (the write raced the read).
// before==after collapses to an EXACT-set check. Returns an error (the reader pool
// collects all errors and the test asserts zero) rather than t.Fatal so a single
// bad read does not abort sibling goroutines mid-flight.
func assertLinearizableEnvelope(what string, got map[uint64]bool, before, after uint64) error {
	for id := uint64(1); id <= before; id++ {
		if !got[id] {
			return fmt.Errorf("%s: STALE/DROPPED — missing id %d that was committed before the read (got %d ids, envelope [1..%d]..[1..%d])", what, id, len(got), before, after)
		}
	}
	for id := range got {
		if id == 0 || id > after {
			return fmt.Errorf("%s: PHANTOM — id %d present but highest in-flight write by read-return was %d (envelope [1..%d]..[1..%d])", what, id, after, before, after)
		}
	}
	return nil
}

// TestReshardUnderConcurrentReadsLinearizable is the Task-5 cluster integration
// proof. A 3-node RF=3 cluster, a partitioned collection seeded with a known full
// id set. We run a COMPLETE online reshard end-to-end while driving CONTINUOUS
// reads from ALL THREE nodes concurrently for the whole reshard, and we MUTATE
// mid-reshard (insert new ids after the cutover, before the gate passes). Every
// read is checked against independent ground truth (committedSet) for the EXACT
// routable window:
//
//   - never a stale subset (must contain every id committed before the read),
//   - never a phantom (no id beyond the latest commit at read-return),
//   - never an "unknown collection"/empty/degraded error (fail-on-unavailable
//     turns an incomplete fan-out into a hard error, which the pool records).
//
// Read-consistency split (why not uniformly Linearizable) — under the Task-3
// read-side meta readIndex barrier a Linearizable read first BLOCKS until the
// coordinator's local meta-FSM reaches the leader-verified frontier. The two
// caught-up nodes (n0/n1) satisfy that barrier immediately, so they read
// Linearizable for the whole reshard (the strong "continuous linearizable reads
// stay fresh across a reshard" proof). The perpetually-gated n3, by construction,
// can NEVER catch up to the new-gen frontier while gated, so a Linearizable read
// there would correctly block-then-time-out — it is no longer the read that
// exercises lagging old-gen routing. n3 therefore reads LeaderOnly, the level that
// resolves oldP from its LOCAL catalog and reads from the old-gen shard leader
// (kept fresh by dual-write). This preserves every assertion's teeth: n3's old-gen
// reads must still observe every committed id (dual-write freshness) with no
// phantom, and a stale/dropped old gen still surfaces as a gen-routing-gap error
// (isGenRoutingGap → fatal). The Linearizable block-then-serve-fresh behavior on a
// lagging follower is proven in the Task-4 tests (meta_readindex_integration_test.go).
//
// Determinism: we DO NOT sleep to cover the critical window. We block ONE node's
// (n3) cutover apply with the meta-apply gate (cluster.SetMetaApplyCatalogGate),
// which holds the post-cutover/pre-gate window open until we release it. The
// readers run continuously across that whole held-open window AND across the
// gate-pass + old-gen-drop after release. The mutator inserts the mid-reshard ids
// while n3 still lags (window provably open), so the new ids must reach the old
// gen via dual-write for n3's old-gen-routed reads to observe them.
//
// This FAILS if dual-write collapses to new-gen-only at the cutover (n3's old-gen
// read goes stale / misses the mid-reshard ids), or if the gate drops the old gen
// early (n3's read errors / loses partitions), or if any node ever routes a read to
// a stale-or-dropped gen.
func TestReshardUnderConcurrentReadsLinearizable(t *testing.T) {
	defer rostam.SetReshardDrainGrace(20 * time.Millisecond)()

	stores := newInmemEmbeddedCluster(t, 3, 8, 3) // RF=3: every node hosts every shard ⇒ leader-routed Linearizable reads resolve locally
	ctx := context.Background()
	const oldP, newP, seedN = 4, 8, 200
	const midN = 30 // ids inserted DURING the reshard (after cutover, before the gate)

	createCollectionTolerant(t, ctx, stores[0], "docs", rostam.VectorConfig{
		Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: oldP,
	})
	gt := &committedSet{}
	for id := uint64(1); id <= seedN; id++ {
		idc := id
		gt.beginWrite(idc)
		retryUntil(t, "upsert", func() error {
			return stores[0].VectorUpsert(ctx, "docs", idc, []float32{float32(idc), 0, 0, 0}, fmt.Sprintf("doc-%d", idc), rostam.VectorInsertOpts{})
		})
		gt.commit(idc)
	}

	ee := []*rostam.Embedded{stores[0].(*rostam.Embedded), stores[1].(*rostam.Embedded), stores[2].(*rostam.Embedded)}
	// Every node has converged on the seeded (oldP, gen 0) catalog before we start,
	// so the only catalog lag in the run is the cutover lag we induce on n3. Confirm
	// up front a Linearizable read is serviceable on every node (so a later failure
	// is the gen-routing gap, not a harness/leader-resolution artefact).
	// poolRC picks the read-consistency for a node's continuous reads: the
	// perpetually-lagging n3 (index 2) reads LeaderOnly (a Linearizable read there
	// would block forever on the Task-3 meta barrier — it can never catch up to the
	// new-gen frontier while gated), every caught-up node reads Linearizable (its
	// barrier passes immediately). Both resolve via the node's LOCAL catalog and read
	// from the shard leader, so dual-write keeps n3's old-gen LeaderOnly read fresh.
	// At baseline (below) every node is caught up, so Linearizable is serviceable on
	// all three.
	poolRC := func(node int) uint8 {
		if node == 2 {
			return ops.ConsistencyLeaderOnly
		}
		return ops.ConsistencyLinearizable
	}
	for i, e := range ee {
		waitEmbeddedCatalogGen(t, e, "docs", oldP, 0, 10*time.Second)
		got, _, err := scrollEnvelope(ctx, stores[i], ops.ConsistencyLinearizable)
		if err != nil {
			t.Fatalf("baseline Linearizable read on node %d failed: %v", i, err)
		}
		if err := assertLinearizableEnvelope(fmt.Sprintf("baseline node %d", i), got, seedN, seedN); err != nil {
			t.Fatal(err)
		}
	}

	// Block n3's apply of the cutover entry (new-gen OpSetCatalogEntry for "docs",
	// generation 1). Only the cutover lags; gen-0/create and other ops pass through.
	var (
		lagEnteredOnce sync.Once
		lagEntered     = make(chan struct{})
		release        = make(chan struct{})
	)
	cluster.SetMetaApplyCatalogGate(func(nodeID, collection string, partitions, generation uint32) {
		if nodeID != lagNodeID || ops.CanonicalName(collection) != ops.CanonicalName("docs") || generation != 1 {
			return
		}
		lagEnteredOnce.Do(func() { close(lagEntered) })
		<-release
	})
	defer cluster.SetMetaApplyCatalogGate(nil)
	var releaseOnce sync.Once
	releaseGate := func() { releaseOnce.Do(func() { close(release) }) }
	defer releaseGate()

	// Continuous reader pool: one goroutine per node, looping until stop, each read
	// bracketed by ground-truth snapshots and checked against the envelope. All
	// errors are collected; the test asserts ZERO. A per-node read counter proves
	// the loops actually exercised every phase (not a no-op).
	var (
		stop      = make(chan struct{})
		readersWG sync.WaitGroup
		errMu     sync.Mutex
		readErrs  []error
		readCount [3]int64
	)
	recordErr := func(err error) {
		errMu.Lock()
		readErrs = append(readErrs, err)
		errMu.Unlock()
	}
	for i := range ee {
		node := i
		readersWG.Add(1)
		go func() {
			defer readersWG.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				before, _ := gt.snapshot()
				got, degraded, err := scrollEnvelope(ctx, stores[node], poolRC(node))
				_, after := gt.snapshot()
				if err != nil {
					// Transient raft election jitter (leader re-resolution) can surface as
					// a momentary error on a Linearizable read; it is NOT the gap (the gap is
					// a PERSISTENT stale/dropped gen). Tolerate sparse transient errors but
					// record persistent ones via the final completeness re-read + the
					// post-run exact re-read on every node. To keep the pool strict yet not
					// flaky, treat an error as fatal ONLY if it names the gap (unknown
					// collection / no shard owner) — those mean the node routed to a
					// stale/dropped gen, which the gate+dual-write must prevent.
					if isGenRoutingGap(err) {
						recordErr(fmt.Errorf("node %d: gen-routing gap on Linearizable read: %w", node, err))
						return
					}
					continue
				}
				if degraded {
					// Fail-on-unavailable means a degraded (incomplete) fan-out is itself an
					// error path; but a degraded flag without error can appear during leader
					// churn. A degraded read that DROPS a pre-committed id is the gap; check it.
					if err := assertLinearizableEnvelope(fmt.Sprintf("node %d (degraded)", node), got, before, after); err != nil {
						recordErr(err)
						return
					}
					continue
				}
				atomic.AddInt64(&readCount[node], 1)
				if err := assertLinearizableEnvelope(fmt.Sprintf("node %d", node), got, before, after); err != nil {
					recordErr(err)
					return
				}
			}
		}()
	}

	// Run the reshard in the background; it flips the catalog (Phase 4 cutover) then
	// BLOCKS in the Phase-4.5 gate (n3 never confirms the new gen) — holding the
	// post-cutover window open while the reader pool hammers all three nodes.
	reshardErr := make(chan error, 1)
	go func() { reshardErr <- stores[0].VectorReshard(ctx, "docs", newP) }()

	select {
	case <-lagEntered:
	case <-time.After(20 * time.Second):
		close(stop)
		readersWG.Wait()
		t.Fatal("n3 never reached the cutover-apply gate (reshard never cut over?)")
	}

	// Window is open: coordinator routes new gen, n3 still routes old gen, dual-write
	// on. Assert the asymmetry so the mid-reshard mutation provably lands in the
	// post-cutover/pre-gate window the gate protects.
	waitEmbeddedCatalogGen(t, ee[0], "docs", newP, 1, 5*time.Second)
	if p, gen, ok := ee[2].Catalog().PartitionsGen("docs"); !ok || p != oldP || gen != 0 {
		close(stop)
		readersWG.Wait()
		t.Fatalf("lagging n3 catalog = (p=%d, gen=%d, ok=%v), want OLD gen (p=%d, gen=0)", p, gen, ok, oldP)
	}
	if st, on := ee[0].Catalog().ReshardState("docs"); !on || st.Status != 1 {
		close(stop)
		readersWG.Wait()
		t.Fatalf("reshard state on coordinator = %+v on=%v, want Status==1 (dual-write must stay on through the gate)", st, on)
	}

	// MUTATE mid-reshard: insert midN new ids AFTER the cutover, BEFORE the gate
	// passes. Dual-write must mirror each into the OLD gen so n3's old-gen reads (and
	// the coordinator's new-gen reads) observe it once committed — read-your-writes.
	for k := 1; k <= midN; k++ {
		id := uint64(seedN + k)
		gt.beginWrite(id)
		retryUntil(t, "mid-reshard upsert", func() error {
			return stores[0].VectorUpsert(ctx, "docs", id, []float32{float32(id), 0, 0, 0}, fmt.Sprintf("doc-%d", id), rostam.VectorInsertOpts{})
		})
		gt.commit(id) // committed mark advances ONLY after the write returns
		// n3 must still be lagging while we mutate (so these writes exercise dual-write
		// into the old gen that n3 reads). Re-confirm periodically.
		if k%10 == 0 {
			if p, gen, ok := ee[2].Catalog().PartitionsGen("docs"); !ok || p != oldP || gen != 0 {
				close(stop)
				readersWG.Wait()
				t.Fatalf("n3 cut over mid-mutation (p=%d gen=%d ok=%v) — window closed early", p, gen, ok)
			}
		}
	}

	// All midN ids are committed; n3 STILL lags. A direct LeaderOnly read on the
	// lagging n3 (old gen) MUST now observe the full seeded+mid set — the explicit
	// read-your-writes assertion that dual-write keeps the old gen fresh. LeaderOnly
	// (not Linearizable) is the level that exercises lagging old-gen routing: it
	// resolves oldP from n3's local catalog and reads from the old-gen shard leader
	// (which holds every dual-written write). A Linearizable read here would instead
	// BLOCK on the Task-3 meta barrier waiting to catch up to the new-gen frontier
	// (gated forever) — that block-then-serve-fresh behavior is proven in the Task-4
	// tests (meta_readindex_integration_test.go).
	full := idRange(1, seedN+midN)
	got, _, err := scrollEnvelope(ctx, stores[2], ops.ConsistencyLeaderOnly)
	if err != nil {
		close(stop)
		readersWG.Wait()
		t.Fatalf("lagging n3 read after mid-reshard mutations failed: %v", err)
	}
	if err := assertLinearizableEnvelope("lagging n3 read-your-writes (mid-reshard)", got, seedN+midN, seedN+midN); err != nil {
		close(stop)
		readersWG.Wait()
		t.Fatalf("OLD-GEN FRESHNESS VIOLATION on the lagging old-gen read: %v", err)
	}

	// Release the lag: n3 applies the cutover, the gate passes, the reshard finishes.
	// The reader pool keeps running across the gate-pass + old-gen-drop transition.
	releaseGate()
	select {
	case err := <-reshardErr:
		must(t, err)
	case <-time.After(20 * time.Second):
		close(stop)
		readersWG.Wait()
		t.Fatal("reshard did not complete after the lag was released")
	}

	// Reshard done: every node routes the new gen. Let the readers run a beat across
	// the settled new-gen state, then stop and drain.
	for i, e := range ee {
		waitEmbeddedCatalogGen(t, e, "docs", newP, 1, 10*time.Second)
		_ = i
	}
	close(stop)
	readersWG.Wait()

	// Assert ZERO read errors across the entire reshard from all nodes.
	errMu.Lock()
	defer errMu.Unlock()
	if len(readErrs) != 0 {
		for _, e := range readErrs {
			t.Errorf("concurrent Linearizable read error: %v", e)
		}
		t.Fatalf("%d Linearizable read(s) returned wrong/stale/dropped data during the reshard", len(readErrs))
	}
	// Prove the pool actually exercised every node (not a no-op that vacuously passed).
	for i := range ee {
		if c := atomic.LoadInt64(&readCount[i]); c == 0 {
			t.Fatalf("node %d performed 0 complete Linearizable reads during the reshard", i)
		}
	}

	// Final state: every node's Linearizable read returns the EXACT full set via the
	// new gen, the reshard state is cleared, and the old gen is dropped.
	for i := range ee {
		fgot, _, ferr := scrollEnvelope(ctx, stores[i], ops.ConsistencyLinearizable)
		if ferr != nil {
			t.Fatalf("final Linearizable read on node %d failed: %v", i, ferr)
		}
		if err := assertLinearizableEnvelope(fmt.Sprintf("final node %d (new gen, exact)", i), fgot, seedN+midN, seedN+midN); err != nil {
			t.Fatal(err)
		}
		if !mapsEqual(fgot, full) {
			t.Fatalf("final node %d: read != full set (%d vs %d)", i, len(fgot), len(full))
		}
	}
	if st, on := ee[0].Catalog().ReshardState("docs"); on || st.Status != 0 {
		t.Fatalf("reshard state still set after completion: %+v on=%v", st, on)
	}
	for p := 0; p < oldP; p++ {
		if genPartitionExists(t, ee[0], "docs", 0, p) {
			t.Fatalf("old gen-0 partition %d still present after the completed reshard", p)
		}
	}
}

// scrollEnvelope performs a single strict scroll at the given read-consistency
// level (fail-on-unavailable) and returns its distinct id set, whether the read was
// degraded (incomplete fan-out flagged but not errored), and any error. Unlike
// laggingLeaderScrollFull it does NOT pin a routing gen or retry — the concurrent
// pool calls it in a tight loop and the caller handles the envelope/error policy,
// so it stays a thin single-shot read suitable for high-frequency probing.
//
// The consistency level is a parameter because the concurrent pool routes it
// per-node (see TestReshardUnderConcurrentReadsLinearizable): caught-up nodes read
// Linearizable (the readIndex barrier passes immediately on a fresh meta-FSM), while
// the perpetually-lagging n3 reads LeaderOnly (a Linearizable read there would
// correctly BLOCK on the Task-3 barrier waiting to catch up to the new-gen frontier,
// which never happens while it is gated — that block-then-serve behavior is proven
// in the Task-4 tests, not here). Both levels resolve (P, gen) from the caller's
// LOCAL catalog and read each partition from its shard leader, so dual-write keeps
// the lagging n3's old-gen LeaderOnly read deterministically fresh.
func scrollEnvelope(ctx context.Context, s rostam.Store, rc uint8) (map[uint64]bool, bool, error) {
	opts := rostam.VectorScrollOpts{ReadConsistency: rc, OnPartitionUnavailable: 1} // Fail-on-unavailable
	docs, meta, _, err := s.VectorScroll(ctx, "docs", rostam.VectorFilter{}, 0, opts)
	if err != nil {
		return nil, false, err
	}
	return idSet(docs), meta.Degraded, nil
}

// isGenRoutingGap reports whether err names the gen-routing gap this plan closes:
// a Linearizable read that routed to a stale/dropped generation surfaces as an
// "unknown collection" / "no shard owner" / not-found style failure (the partition
// the node routed to no longer exists or was never the right gen). Transient raft
// leader-election jitter does NOT name these — it is leadership/timeout churn — so
// gating fatality on these substrings keeps the pool strict about the gap while
// tolerating benign election transients.
func isGenRoutingGap(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	for _, sub := range []string{"unknown collection", "no shard owner", "no owner", "not found", "unknown shard", "no such collection"} {
		if strings.Contains(strings.ToLower(s), sub) {
			return true
		}
	}
	return false
}

// mapsEqual reports exact membership equality of two id sets.
func mapsEqual(a, b map[uint64]bool) bool {
	if len(a) != len(b) {
		return false
	}
	for id := range a {
		if !b[id] {
			return false
		}
	}
	return true
}

// idRange returns the set {lo, lo+1, ..., hi} as a membership map.
func idRange(lo, hi int) map[uint64]bool {
	m := make(map[uint64]bool, hi-lo+1)
	for id := lo; id <= hi; id++ {
		m[uint64(id)] = true
	}
	return m
}

// assertExactIDSet fails unless got is exactly want (same membership) — the proof
// asserts the EXACT expected id set, not len>0 / not a non-empty subset, so a
// stale read that drops or omits ids (or returns an unexpected extra) is caught.
func assertExactIDSet(t *testing.T, what string, got, want map[uint64]bool) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d distinct ids, want exactly %d", what, len(got), len(want))
	}
	for id := range want {
		if !got[id] {
			t.Fatalf("%s: missing expected id %d (got %d ids) — STALE read", what, id, len(got))
		}
	}
	for id := range got {
		if !want[id] {
			t.Fatalf("%s: unexpected id %d present — read returned a value outside the committed set", what, id)
		}
	}
}

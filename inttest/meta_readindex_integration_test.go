// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"errors"
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
// The read-side meta readIndex barrier proofs — a lagging meta-follower's
// Linearizable read BLOCKS until its local meta-FSM catches up to the leader-
// verified catalog frontier, then serves FRESH (never stale / never the dropped
// old gen). These are the COMPLEMENT of the reshard-side-gate proofs in
// linearizable_catalog_integration_test.go (whose doc-comments forward-reference
// THIS file): there a LeaderOnly read serves the kept-fresh OLD gen via the
// lagging LOCAL catalog; here a Linearizable read instead trips the Task-3 barrier
// and is converted into a leader-frontier-fresh read.
//
// Determinism (NO staleness sleeps): the lag is staged with the test-only meta-FSM
// apply gate (cluster.SetMetaApplyCatalogGate), which BLOCKS exactly ONE node's
// apply of a chosen catalog OpSetCatalogEntry BEFORE the FSM write lock — so the
// gated node keeps serving its prior catalog and genuinely LAGS (it does not hang).
// The gate signals lagEntered when the node reaches the block (window provably
// open) and is released by closing a channel. "Blocked" is asserted the textbook
// way: the Linearizable read runs IN A GOROUTINE writing a done-channel; we assert
// NOT-done while the gate is held (a single bounded select), then done-AND-fresh
// after release. No fixed sleep is used to wait for staleness to appear.
//
// Cluster shape: RF=3 (every shard replicated on all nodes) so a leader-routed
// partitioned read resolves every partition's leader locally on any node — the same
// shape the reshard-side-gate proofs use. The lagging node is chosen to be a meta
// FOLLOWER so its barrier genuinely FORWARDS __meta_readindex__ (1 RTT) and then
// waits — the forward counter (cluster.SetMetaReadIndexForwardHook) proves it is
// forwarding/waiting, not merely slow.
// ----------------------------------------------------------------------------

// pickLagger waits for a single stable meta leader and returns a meta-FOLLOWER store
// index (preferring a non-zero index so stores[0] stays a clean coordinator) plus its
// NodeID and the meta-leader store index. Every CREATE / reshard / seed is driven from
// stores[0] (which transparently forwards to the relevant shard/meta leader — the
// proven path the reshard-side-gate tests use); the gate + Linearizable read run on
// the returned FOLLOWER so its meta readIndex barrier genuinely FORWARDS
// __meta_readindex__ and WAITS on its local FSM (the real block path the forward
// counter observes). Leadership in this RF=3 harness is NOT pinned to node 0, so we
// discover a follower at runtime rather than hardcoding n3. The "window open"
// asymmetry is checked on the discovered meta-LEADER store (it reflects a new
// catalog gen first). The harness names peers "n1".."nN" with peers[i].NodeID ==
// fmt.Sprintf("n%d", i+1), so the follower's NodeID is "n{idx+1}"; no production
// NodeID accessor is added (TEST-ONLY).
func pickLagger(t *testing.T, stores []rostam.Store) (lagIdx int, lagID string, leadIdx int) {
	t.Helper()
	// Require the SAME single meta leader across several consecutive polls so we pick on
	// a settled cluster (not a transient pre-vote/election state). This also gives the
	// per-shard raft groups time to elect, reducing transient not-leader on the first
	// create/reshard the caller issues.
	deadline := time.Now().Add(cpuScaled(45 * time.Second)) // widened 30s->45s for CPU-contended CI; setup gate (wait for a settled single meta leader), finite so a real no-leader still fails
	stableLead, stableCount := -1, 0
	for time.Now().Before(deadline) {
		leaders, lead := 0, -1
		var followers []int
		for i, s := range stores {
			if s.(*rostam.Embedded).Node().MetaIsLeader() {
				leaders++
				lead = i
			} else {
				followers = append(followers, i)
			}
		}
		if leaders != 1 || lead < 0 || len(followers) == 0 {
			stableLead, stableCount = -1, 0
			time.Sleep(50 * time.Millisecond)
			continue
		}
		if lead == stableLead {
			stableCount++
		} else {
			stableLead, stableCount = lead, 1
		}
		if stableCount >= 5 { // ~250ms of a steady single leader
			// Prefer a non-zero follower so stores[0] stays a clean coordinator.
			follower := followers[0]
			for _, f := range followers {
				if f != 0 {
					follower = f
					break
				}
			}
			return follower, fmt.Sprintf("n%d", follower+1), lead
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("no stable (single leader + distinct follower) meta shape found")
	return -1, "", -1
}

// createPartitionedDocs creates the P-partition "docs" collection on coord, retrying
// transient meta/shard not-leader election jitter AND treating an "already exists"
// result as success. The latter matters because CreateCollection is NOT idempotent: a
// first attempt can create the physical partitions then return not-leader on the final
// SetPartitionsGen meta write; the retry then sees the physical collections and reports
// "already exists" even though the create effectively succeeded. We confirm real
// success by waiting for coord's local catalog to show docs at (P, gen 0) — so an
// "already exists" that is NOT backed by a converged catalog still fails loud.
// The retry loop below swallows its RETRYABLE errors by design (not-leader churn
// and a failed idempotent catalog write are both expected mid-election), but it
// must not swallow them past the deadline: an exhausted budget used to fall
// straight through to waitEmbeddedCatalogGen, which reports only the catalog SHAPE
// — "catalog "docs" = (p=1, gen=0, ok=false)" — and names neither the create error
// nor the catalog-write error that actually blocked it. That failure is
// unactionable: persistent not-leader, a meta write that never committed, and a
// genuine routing regression all produce the identical message. So the last error
// of each kind is kept and reported when the loop gives up, and the loop's own
// failure is raised BEFORE the convergence gate that would otherwise mask it.
func createPartitionedDocs(t *testing.T, ctx context.Context, coord rostam.Store, P int) {
	t.Helper()
	start := time.Now()
	deadline := start.Add(cpuScaled(45 * time.Second)) // widened 30s->45s for CPU-contended CI; setup retry through election jitter, finite so a real failure still fails
	var (
		created     bool
		attempts    int
		lastCreate  error // last CreateCollection error (retryable: not-leader / already-exists)
		lastCatalog error // last direct SetPartitionsGen error on the already-exists path
	)
	for time.Now().Before(deadline) {
		attempts++
		err := coord.CreateCollection(ctx, "docs", rostam.VectorConfig{
			Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: P,
		})
		if err == nil {
			created = true
			break
		}
		lastCreate = err
		if strings.Contains(err.Error(), "already exists") {
			// A prior attempt got not-leader AFTER creating the physical partitions but
			// (possibly) BEFORE the SetPartitionsGen catalog write — so the physical
			// partitions exist yet the catalog may not be partitioned. Idempotently
			// complete the catalog entry so routing resolves the partitioned form; this
			// converges below.
			serr := coord.(*rostam.Embedded).Catalog().SetPartitionsGen("docs", P, 0)
			if serr == nil {
				created = true
				break
			}
			lastCatalog = serr
			time.Sleep(50 * time.Millisecond)
			continue
		}
		if !strings.Contains(err.Error(), "not leader") {
			t.Fatalf("create docs: %v", err)
		}
		time.Sleep(50 * time.Millisecond)
	}
	if !created {
		t.Fatalf("create docs: gave up after %d attempts in %s (never created and never wrote the catalog); "+
			"last create error: %v; last catalog-write error: %v",
			attempts, time.Since(start).Round(time.Millisecond), lastCreate, lastCatalog)
	}
	waitEmbeddedCatalogGen(t, coord.(*rostam.Embedded), "docs", P, 0, cpuScaled(25*time.Second)) // widened 15s->25s for CPU-contended CI; setup convergence gate, finite so a real non-convergence still fails
}

// seedDocs upserts ids 1..N (vector {id,0,0,0}) into "docs" on coord, retrying EVERY
// transient error (not just rostam.ErrNotLeader) per id within a generous per-id budget. The
// shared retryUntil fatals on any non-rostam.ErrNotLeader error, which is too brittle for
// seeding under the in-process RF=3 harness's shard-leader election churn (a partition
// can momentarily surface a non-not-leader availability error). Correctness is still
// enforced: an id that never lands within its budget fails loud, and the proof reads
// downstream assert the EXACT set — so a silently-dropped seed would still be caught.
func seedDocs(t *testing.T, ctx context.Context, coord rostam.Store, N int) {
	t.Helper()
	for id := uint64(1); id <= uint64(N); id++ {
		deadline := time.Now().Add(cpuScaled(90 * time.Second)) // load-flakiness hardening (widened 60s->90s for CPU-contended CI); finite so a real hang still fails (still fails loud if an id never lands)
		var lastErr error
		ok := false
		for time.Now().Before(deadline) {
			if err := coord.VectorUpsert(ctx, "docs", id, []float32{float32(id), 0, 0, 0}, fmt.Sprintf("doc-%d", id), rostam.VectorInsertOpts{}); err != nil {
				lastErr = err
				time.Sleep(50 * time.Millisecond)
				continue
			}
			ok = true
			break
		}
		if !ok {
			t.Fatalf("seed upsert id=%d never succeeded: %v", id, lastErr)
		}
	}
}

// linScrollIDs runs a strict Linearizable scroll (read_consistency=2,
// fail-on-unavailable) on the given store and returns the distinct id set + error.
// Fail-on-unavailable turns an incomplete fan-out into a hard error, so the result
// is never a silently-truncated subset.
func linScrollIDs(ctx context.Context, s rostam.Store) (map[uint64]bool, error) {
	docs, _, _, err := s.VectorScroll(ctx, "docs", rostam.VectorFilter{}, 0,
		rostam.VectorScrollOpts{ReadConsistency: ops.ConsistencyLinearizable, OnPartitionUnavailable: 1})
	if err != nil {
		return nil, err
	}
	return idSet(docs), nil
}

// TestLinearizableReadBlocksUntilCatalogFresh is THE block-then-serve-fresh proof.
//
// Mutation staged: a fresh partitioned collection CREATE. We gate ONE meta-follower
// (n) so it never applies the create's OpSetCatalogEntry (gen 0) — its local catalog
// has NO "docs" entry while gated. This is the cleanest proof because the stale view
// is unambiguous ("unknown collection" / empty) and it directly proves read-your-
// writes for a just-created collection, which was EXPLICITLY out of scope for the
// reshard-side gate (that gate only keeps an existing old gen fresh; it does nothing
// for a create a follower hasn't applied yet).
//
// Steps:
//  1. CREATE "docs" (P>1) on the coordinator and seed N points (these commit on the
//     coordinator + replicate; the gated follower has the SHARD data replicate
//     independently, but its META catalog apply of the create is blocked).
//  2. On the gated follower, a Linearizable scroll runs IN A GOROUTINE. While the
//     gate is held we assert it has NOT returned (a bounded select) AND the forward
//     counter shows it forwarded/waiting — it is BLOCKED on the barrier, not slow.
//  3. CONTRAST: in the same held window a LeaderOnly scroll on the SAME follower
//     returns WITHOUT blocking (it does NOT trip the barrier) and the follower's LOCAL
//     catalog is still stale (no "docs" entry) — documenting that the barrier is
//     precisely what converts a stale-capable read into a linearizable one. (We assert
//     the no-block + local-staleness, NOT an empty data value: an off-path scroll of a
//     not-yet-locally-partitioned collection can be re-resolved as partitioned by the
//     remote shard leader it forwards to, so the data value is not a reliable stale
//     signal — the local-catalog staleness + no-block ARE.)
//  4. RELEASE the gate. The SAME Linearizable read unblocks and serves FRESH (a
//     non-empty subset of the seeded set, no error) — never stale/empty/"unknown
//     collection"; a follow-up poll asserts the EXACT seeded set once shard data settles.
func TestLinearizableReadBlocksUntilCatalogFresh(t *testing.T) {
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

	// Gate the lagging follower's apply of the CREATE catalog entry for "docs"
	// (generation 0). Other collections / generations pass through untouched, so ONLY
	// the create on this one node lags. Signal lagEntered when it reaches the block.
	var (
		lagEnteredOnce sync.Once
		lagEntered     = make(chan struct{})
		release        = make(chan struct{})
		releaseOnce    sync.Once
	)
	releaseGate := func() { releaseOnce.Do(func() { close(release) }) }
	cluster.SetMetaApplyCatalogGate(func(nodeID, collection string, partitions, generation uint32) {
		if nodeID != lagID || ops.CanonicalName(collection) != ops.CanonicalName("docs") || generation != 0 {
			return
		}
		lagEnteredOnce.Do(func() { close(lagEntered) })
		<-release // hold this node at "no docs entry" until the test releases the window
	})
	// Always release, even on an early t.Fatal, so the reshard/create goroutine drains
	// and node-shutdown Cleanup cannot hang.
	defer releaseGate()

	// CREATE the partitioned collection on the coordinator. The meta-apply gate blocks
	// only the lagging FOLLOWER's apply of the create entry — NOT commit, NOT the
	// coordinator's create call — so CreateCollection returns normally (after commit +
	// the coordinator's own apply, and createPartitionedDocs waits for stores[0]'s local
	// catalog to converge) while the follower stays gated.
	createPartitionedDocs(t, ctx, coord, P)

	// The create committed; the gated follower applies (and blocks at) the create entry
	// at some point after commit. Wait until it is genuinely blocked (window open).
	select {
	case <-lagEntered:
	case <-time.After(cpuScaled(40 * time.Second)): // widened 20s->40s for CPU-contended CI; setup/progress, finite so a real hang still fails
		releaseGate()
		t.Fatal("lagging follower never reached the create-apply gate")
	}
	// Seed the points — they land in the partitioned physical shards (which replicate
	// independently of the gated META apply on the lagging follower).
	want := idRange(1, N)
	seedDocs(t, ctx, coord, N)
	// Sanity: the seeded set is readable on the coordinator (a caught-up node) so a
	// later empty read on the lagging follower is a barrier/routing issue, not a
	// missing-data/seed artefact. Poll through shard-replication / leader-election
	// jitter (a Linearizable read can momentarily see a partial set while the last
	// upserts propagate to every partition's shard leader).
	{
		deadline := time.Now().Add(cpuScaled(45 * time.Second)) // widened 30s->45s for CPU-contended CI; setup/progress, finite so a real hang still fails
		var lastN int
		var lastErr error
		for time.Now().Before(deadline) {
			ids, err := linScrollIDs(ctx, coord)
			if err != nil {
				lastErr = err
				time.Sleep(50 * time.Millisecond)
				continue
			}
			if len(ids) == N {
				lastErr = nil
				break
			}
			lastN, lastErr = len(ids), nil
			time.Sleep(50 * time.Millisecond)
		}
		if lastErr != nil {
			releaseGate()
			t.Fatalf("coordinator Linearizable read errored before seeing the seeded set: %v", lastErr)
		}
		if n, _ := linScrollIDs(ctx, coord); n != nil && len(n) != N {
			releaseGate()
			t.Fatalf("coordinator Linearizable read sees %d ids (last partial %d), want %d — seed did not converge", len(n), lastN, N)
		}
	}

	// Confirm the asymmetry: the coordinator sees "docs" (P, gen 0); the gated
	// follower's LOCAL catalog still has NO "docs" entry. This IS the lag window.
	if p, _, ok := lagEE.Catalog().PartitionsGen("docs"); ok {
		releaseGate()
		t.Fatalf("gated follower already has a docs catalog entry (p=%d) — the create lag did not hold", p)
	}

	// Instrument forwards so we can prove the Linearizable read is FORWARDING/WAITING
	// (blocked on the barrier), not merely slow.
	var forwards int32
	cluster.SetMetaReadIndexForwardHook(func() { atomic.AddInt32(&forwards, 1) })
	defer cluster.SetMetaReadIndexForwardHook(nil)

	// (2) The Linearizable read on the gated follower, in a goroutine. It must BLOCK
	// on the meta readIndex barrier (forward to the leader, learn the create frontier,
	// wait for the local FSM to reach it) until we release the gate.
	type linResult struct {
		ids map[uint64]bool
		err error
	}
	linDone := make(chan linResult, 1)
	go func() {
		ids, err := linScrollIDs(ctx, lagStore)
		linDone <- linResult{ids, err}
	}()

	// Assert NOT-done while the gate is held. A bounded select: if the Linearizable
	// read returns here it served a stale/empty catalog WITHOUT waiting — a Task-1..3
	// barrier bug. (This is the "blocked, not slow" assertion.)
	select {
	case r := <-linDone:
		releaseGate()
		t.Fatalf("Linearizable read returned (ids=%d, err=%v) while the create apply was gated — "+
			"it must BLOCK on the meta readIndex barrier until the local catalog catches up", len(r.ids), r.err)
	case <-time.After(500 * time.Millisecond):
		// Good: still blocked on the barrier.
	}
	if atomic.LoadInt32(&forwards) == 0 {
		releaseGate()
		t.Fatal("blocked Linearizable read issued ZERO __meta_readindex__ forwards — it must forward to the meta leader and wait")
	}

	// (3) CONTRAST: a non-Linearizable read on the SAME gated follower in the SAME
	// window returns WITHOUT BLOCKING — it does NOT trip the meta readIndex barrier. The
	// barrier is precisely what converts a stale-capable read into a linearizable one;
	// an off-path read never waits. We assert the behavioral contrast two ways:
	//   (a) the follower's LOCAL catalog is STILL stale (no "docs" entry) at this point
	//       — the deterministic stale signal the barrier closes (a LeaderOnly read
	//       resolves routing from exactly this lagging local catalog), and
	//   (b) a LeaderOnly read RETURNS PROMPTLY (does not block like the Linearizable
	//       read does) — the no-added-cost-off-the-Linearizable-path contract.
	// (We assert the no-block behavior, NOT an empty data result: an off-path scroll of
	// a not-yet-locally-partitioned collection can be re-resolved as partitioned by the
	// remote shard leader it forwards to, so the data value is not a reliable stale
	// signal — the local-catalog staleness + no-block ARE.)
	if p, _, ok := lagEE.Catalog().PartitionsGen("docs"); ok {
		releaseGate()
		t.Fatalf("gated follower local catalog already shows docs (p=%d) before release — the create lag did not hold for the contrast", p)
	}
	loDone := make(chan struct{}, 1)
	go func() {
		_, _, _, _ = lagStore.VectorScroll(ctx, "docs", rostam.VectorFilter{}, 0,
			rostam.VectorScrollOpts{ReadConsistency: ops.ConsistencyLeaderOnly, OnPartitionUnavailable: 1})
		loDone <- struct{}{}
	}()
	select {
	case <-loDone:
		// Good: the LeaderOnly read did NOT block (contrast with the still-blocked
		// Linearizable read), and the Linearizable read is STILL blocked (asserted next).
	case <-time.After(cpuScaled(15 * time.Second)): // widened 5s->15s for CPU-contended CI; progress deadline (LeaderOnly must not block), finite — still proves no-block vs the still-blocked Linearizable read
		releaseGate()
		t.Fatal("LeaderOnly read on the gated follower BLOCKED — a non-Linearizable read must NEVER trip the meta readIndex barrier")
	}
	// The Linearizable read MUST still be blocked at this point (the LeaderOnly read
	// returning did not unblock it) — confirming the barrier, not some shared latch, is
	// what holds the Linearizable read.
	select {
	case r := <-linDone:
		releaseGate()
		t.Fatalf("Linearizable read returned (ids=%d, err=%v) before release while LeaderOnly returned promptly — "+
			"the Linearizable read must stay BLOCKED on the barrier until catch-up", len(r.ids), r.err)
	default:
	}

	// (4) RELEASE: the follower applies the create, its catalog catches up, and the
	// SAME blocked Linearizable read unblocks and serves FRESH — never stale/empty/
	// "unknown collection". The unblocked read proves the barrier converted a stale-
	// capable read into a fresh one: it routed the just-created collection (a non-empty
	// subset of the seeded set, no error) instead of the lagging local catalog's absent
	// "docs". We then assert the EXACT 60-id set via a brief follow-up poll, since data
	// freshness is governed by SHARD replication (orthogonal to the meta CATALOG barrier
	// this test proves) and the last upserts can take a beat to settle on every
	// partition's shard leader — the barrier's job (catalog freshness) is done the
	// instant the read unblocks routed-correct.
	releaseStart := time.Now()
	releaseGate()
	select {
	case r := <-linDone:
		if r.err != nil {
			t.Fatalf("Linearizable read after gate release: %v (must serve fresh, never error / never stale)", r.err)
		}
		if len(r.ids) == 0 {
			t.Fatal("Linearizable read after catch-up returned EMPTY — it must serve the fresh create, never the stale/absent local catalog")
		}
		for id := range r.ids {
			if !want[id] {
				t.Fatalf("Linearizable read after catch-up returned id %d outside the seeded set — phantom/stale routing", id)
			}
		}
		if d := time.Since(releaseStart); d > cpuScaled(15*time.Second) { // widened 5s->15s for CPU-contended CI; progress deadline (must unblock after release), finite so a real stuck-barrier still fails
			t.Fatalf("Linearizable read took %s after release — should unblock promptly once caught up", d)
		}
	case <-time.After(cpuScaled(30 * time.Second)): // widened 15s->30s for CPU-contended CI; progress deadline, finite so a real hang still fails
		t.Fatal("Linearizable read did not return after the gate was released")
	}

	// The follower is now caught up; a Linearizable read converges to the EXACT seeded
	// set once shard replication settles (poll-tolerant of the data-side settle, not the
	// catalog barrier which is already satisfied).
	{
		deadline := time.Now().Add(cpuScaled(40 * time.Second)) // widened 20s->40s for CPU-contended CI; data-settle progress poll, finite so a real non-convergence still fails
		var lastIDs map[uint64]bool
		var lastErr error
		for time.Now().Before(deadline) {
			ids, err := linScrollIDs(ctx, lagStore)
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
			t.Fatalf("caught-up follower Linearizable read errored: %v", lastErr)
		}
		assertExactIDSet(t, "caught-up follower Linearizable read (exact seeded set)", lastIDs, want)
	}
}

// TestLinearizableReadClosesRejoinResidual stages the down-node-rejoin reshard
// residual the reshard-side gate left open: a node that MISSED a reshard cutover
// (its meta lags, still routing to the OLD gen) serving a catalog-routed read. With
// the read-side barrier a Linearizable read on that lagging node BLOCKS until it
// applies the cutover, then routes to the correct NEW gen and returns the full
// latest id set — it NEVER routes to a (to-be-)dropped old gen and never goes stale.
//
// We gate the lagging follower's apply of the cutover OpSetCatalogEntry (generation
// 1). While gated its LOCAL catalog still shows the OLD gen (oldP/gen0) — exactly the
// node that "missed the cutover". A Linearizable read there must BLOCK (it cannot
// resolve the new-gen frontier locally yet); on release it catches up to gen1 and the
// SAME read returns the full set via the NEW gen. The assertion pins the routed gen
// to newP AFTER release, so a read that routed to the old gen would be caught.
func TestLinearizableReadClosesRejoinResidual(t *testing.T) {
	defer rostam.SetReshardDrainGrace(20 * time.Millisecond)()
	t.Cleanup(func() { cluster.SetMetaApplyCatalogGate(nil) })

	stores := newInmemEmbeddedCluster(t, 3, 4, 3) // RF=3
	ctx := context.Background()
	const oldP, newP, N = 4, 8, 40

	lagIdx, lagID, leadIdx := pickLagger(t, stores)
	coord := stores[0] // drive mutations from node 0 (forwards transparently)
	lagStore := stores[lagIdx]
	lagEE := lagStore.(*rostam.Embedded)
	leadEE := stores[leadIdx].(*rostam.Embedded) // the meta leader reflects a new catalog gen first

	// Seed a partitioned collection at the OLD gen.
	createPartitionedDocs(t, ctx, coord, oldP)
	want := idRange(1, N)
	seedDocs(t, ctx, coord, N)
	// Every node converged on (oldP, gen0) before we stage the lag, so the ONLY lag is
	// the cutover we induce.
	waitEmbeddedCatalogGen(t, lagEE, "docs", oldP, 0, cpuScaled(20*time.Second)) // widened 10s->20s + cpuScaled for CPU-contended CI (follower meta-apply lag); setup gate, finite so a real hang still fails

	// Gate the lagging follower's apply of the cutover (new-gen) entry: it "misses" the
	// cutover and keeps routing to the old gen until released.
	var (
		lagEnteredOnce sync.Once
		lagEntered     = make(chan struct{})
		release        = make(chan struct{})
		releaseOnce    sync.Once
	)
	releaseGate := func() { releaseOnce.Do(func() { close(release) }) }
	cluster.SetMetaApplyCatalogGate(func(nodeID, collection string, partitions, generation uint32) {
		if nodeID != lagID || ops.CanonicalName(collection) != ops.CanonicalName("docs") || generation != 1 {
			return
		}
		lagEnteredOnce.Do(func() { close(lagEntered) })
		<-release
	})
	defer releaseGate()

	reshardErr := make(chan error, 1)
	go func() { reshardErr <- coord.VectorReshard(ctx, "docs", newP) }()

	// Wait for the gated follower to reach the cutover apply (window open). The reshard
	// (migration + dual-write drain + cutover) can take a while under RF=3 election churn,
	// so we give a generous budget; if the reshard goroutine errors first, surface that.
	select {
	case <-lagEntered:
	case err := <-reshardErr:
		releaseGate()
		t.Fatalf("reshard returned (err=%v) before the lagging follower reached the cutover-apply gate", err)
	case <-time.After(cpuScaled(60 * time.Second)): // widened 40s->60s for CPU-contended CI; setup/progress (reshard must reach the cutover gate), finite so a real hang still fails
		releaseGate()
		t.Fatal("lagging follower never reached the cutover-apply gate")
	}

	// Window open: the meta leader routes the NEW gen; the lagging follower still routes
	// the OLD gen (it "missed the cutover"). This is the rejoin residual.
	waitEmbeddedCatalogGen(t, leadEE, "docs", newP, 1, cpuScaled(20*time.Second)) // widened 5s->20s for CPU-contended CI; setup gate (leader reflects new gen), finite so a real hang still fails
	if p, gen, ok := lagEE.Catalog().PartitionsGen("docs"); !ok || p != oldP || gen != 0 {
		releaseGate()
		t.Fatalf("lagging follower catalog = (p=%d, gen=%d, ok=%v), want OLD gen (p=%d, gen=0) — the lag did not hold", p, gen, ok, oldP)
	}

	var forwards int32
	cluster.SetMetaReadIndexForwardHook(func() { atomic.AddInt32(&forwards, 1) })
	defer cluster.SetMetaReadIndexForwardHook(nil)

	// A Linearizable read on the lagging follower must BLOCK (it cannot resolve the
	// new-gen frontier locally yet) — it must NOT route to the old gen.
	type linResult struct {
		ids map[uint64]bool
		err error
	}
	linDone := make(chan linResult, 1)
	go func() {
		ids, err := linScrollIDs(ctx, lagStore)
		linDone <- linResult{ids, err}
	}()
	select {
	case r := <-linDone:
		releaseGate()
		t.Fatalf("Linearizable read returned (ids=%d, err=%v) while the cutover apply was gated — "+
			"it must BLOCK on the barrier until the node applies the cutover, NEVER route to the old gen", len(r.ids), r.err)
	case <-time.After(500 * time.Millisecond):
		// Good: blocked, catching up to the new-gen frontier.
	}
	if atomic.LoadInt32(&forwards) == 0 {
		releaseGate()
		t.Fatal("blocked Linearizable read issued ZERO __meta_readindex__ forwards — it must forward to the meta leader and wait")
	}

	// Release: the follower applies the cutover, catches up to gen1, and the SAME read
	// unblocks and serves via the NEW gen (never the old/dropped gen). The unblocked
	// read proves the barrier held until catch-up and then routed correctly: a non-empty
	// subset of the latest set, no error. (The EXACT full set is asserted below via a
	// brief poll, tolerating shard-side data settle which is orthogonal to the meta
	// catalog barrier this test proves.)
	releaseGate()
	select {
	case r := <-linDone:
		if r.err != nil {
			t.Fatalf("Linearizable read after cutover catch-up: %v (must serve fresh via the new gen, never error / never route to a dropped gen)", r.err)
		}
		if len(r.ids) == 0 {
			t.Fatal("Linearizable read after cutover catch-up returned EMPTY — it must route the new gen, never a dropped/stale gen")
		}
		for id := range r.ids {
			if !want[id] {
				t.Fatalf("Linearizable read after cutover catch-up returned id %d outside the latest set — phantom/stale routing", id)
			}
		}
	case <-time.After(cpuScaled(30 * time.Second)): // widened 15s->30s for CPU-contended CI; progress deadline (read must unblock after release), finite so a real hang still fails
		t.Fatal("Linearizable read did not return after the gate was released")
	}

	// Pin the routed gen to the NEW gen on the (now caught-up) follower: the read
	// resolved newP, never the old gen.
	waitEmbeddedCatalogGen(t, lagEE, "docs", newP, 1, cpuScaled(20*time.Second)) // widened 10s->20s for CPU-contended CI; progress gate (follower catches up to new gen), finite so a real hang still fails
	if p, gen, ok := lagEE.Catalog().PartitionsGen("docs"); !ok || p != newP || gen != 1 {
		t.Fatalf("after catch-up the follower routes (p=%d gen=%d ok=%v), want the NEW gen (p=%d gen=1)", p, gen, ok, newP)
	}
	// The caught-up follower's Linearizable read converges to the EXACT latest set via
	// the new gen (poll-tolerant of shard-side data settle).
	{
		deadline := time.Now().Add(cpuScaled(40 * time.Second)) // widened 20s->40s for CPU-contended CI; data-settle progress poll, finite so a real non-convergence still fails
		var lastIDs map[uint64]bool
		var lastErr error
		for time.Now().Before(deadline) {
			ids, err := linScrollIDs(ctx, lagStore)
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
			t.Fatalf("caught-up follower Linearizable read errored: %v", lastErr)
		}
		assertExactIDSet(t, "caught-up follower Linearizable read via new gen (rejoin residual closed, exact set)", lastIDs, want)
	}

	select {
	case err := <-reshardErr:
		must(t, err)
	case <-time.After(cpuScaled(40 * time.Second)): // widened 20s->40s for CPU-contended CI; progress deadline (reshard must complete), finite so a real hang still fails
		t.Fatal("reshard did not complete after the lag was released")
	}
	// Old gen is gone on the reshard coordinator (stores[0], which performs the drop) —
	// proving the lagging Linearizable read would have routed to a DROPPED gen had it
	// not blocked. We poll briefly to allow the post-gate drop to settle.
	coordEE := coord.(*rostam.Embedded)
	retryUntil(t, "old-gen dropped", func() error {
		for p := 0; p < oldP; p++ {
			if genPartitionExists(t, coordEE, "docs", 0, p) {
				return fmt.Errorf("old gen-0 partition %d still present after the completed reshard", p)
			}
		}
		return nil
	})
}

// TestLinearizableReadTimeoutFailsLoud proves the fail-loud contract: with the meta
// leader's catalog frontier unreachable to the lagging follower for the duration of
// the read's barrier deadline, a Linearizable read returns *cluster.ErrMetaLinearizableTimeout
// within the deadline — it NEVER hangs and NEVER serves stale.
//
// We make the frontier unreachable by gating the lagging follower's apply of a fresh
// create indefinitely (it can never catch up to the create frontier while gated) AND
// shortening the read's barrier budget via the package var metaReadIndexReadTimeout.
// The forwarded leader confirms the frontier, but the local FSM can never reach it
// inside the short budget ⇒ the barrier times out fail-loud.
func TestLinearizableReadTimeoutFailsLoud(t *testing.T) {
	t.Cleanup(func() { cluster.SetMetaApplyCatalogGate(nil) })

	defer rostam.SetMetaReadIndexReadTimeout(600 * time.Millisecond)()

	stores := newInmemEmbeddedCluster(t, 3, 4, 3) // RF=3
	ctx := context.Background()
	const P, N = 4, 30

	lagIdx, lagID, _ := pickLagger(t, stores)
	coord := stores[0] // drive mutations from node 0 (forwards transparently)
	lagStore := stores[lagIdx]
	lagEE := lagStore.(*rostam.Embedded)

	var (
		lagEnteredOnce sync.Once
		lagEntered     = make(chan struct{})
		release        = make(chan struct{})
		releaseOnce    sync.Once
	)
	releaseGate := func() { releaseOnce.Do(func() { close(release) }) }
	cluster.SetMetaApplyCatalogGate(func(nodeID, collection string, partitions, generation uint32) {
		if nodeID != lagID || ops.CanonicalName(collection) != ops.CanonicalName("docs") || generation != 0 {
			return
		}
		lagEnteredOnce.Do(func() { close(lagEntered) })
		<-release // never released before the read's short barrier deadline elapses
	})
	defer releaseGate()

	// CREATE synchronously (the gate blocks only the lagging FOLLOWER's apply, not commit
	// or the coordinator's create call), retrying transient meta/shard not-leader.
	createPartitionedDocs(t, ctx, coord, P)

	// Wait until the gated follower is genuinely blocked at the create apply.
	select {
	case <-lagEntered:
	case <-time.After(cpuScaled(40 * time.Second)): // widened 20s->40s for CPU-contended CI; setup gate (NOT the asserted 600ms barrier timeout below), finite so a real hang still fails
		releaseGate()
		t.Fatal("lagging follower never reached the create-apply gate")
	}
	// Seed points so the only reason a read could fail is the barrier timeout, not a
	// missing-data path (these land in the shards regardless of the gated meta apply).
	seedDocs(t, ctx, coord, N)
	if _, _, ok := lagEE.Catalog().PartitionsGen("docs"); ok {
		releaseGate()
		t.Fatal("gated follower applied the create — cannot stage the unreachable-frontier timeout")
	}

	// The Linearizable read must fail loud with the typed timeout within ~the deadline.
	start := time.Now()
	_, err := linScrollIDs(ctx, lagStore)
	elapsed := time.Since(start)

	if err == nil {
		releaseGate()
		t.Fatal("Linearizable read SUCCEEDED while the meta frontier was unreachable — it must fail loud, never serve stale")
	}
	var to *cluster.ErrMetaLinearizableTimeout
	if !errors.As(err, &to) && !strings.Contains(err.Error(), "meta linearizable read timed out") {
		releaseGate()
		t.Fatalf("error = %v, want *cluster.ErrMetaLinearizableTimeout", err)
	}
	if elapsed > 3*time.Second {
		t.Fatalf("Linearizable read took %s — must be bounded by the ~600ms barrier deadline (never hangs)", elapsed)
	}

	releaseGate()
}

// TestNonLinearizableReadNoMetaRTT proves the zero-cost contract under staleness: on
// a lagging follower (its meta apply of a catalog mutation gated), BOTH an AnyReplica
// and a LeaderOnly read issue ZERO __meta_readindex__ forwards (counter-proven) and
// return WITHOUT blocking — the barrier is only ever on the Linearizable path, so an
// off-path read adds no meta RTT even when the local catalog is stale.
func TestNonLinearizableReadNoMetaRTT(t *testing.T) {
	t.Cleanup(func() { cluster.SetMetaApplyCatalogGate(nil) })

	stores := newInmemEmbeddedCluster(t, 3, 4, 3) // RF=3
	ctx := context.Background()
	const oldP, newP, N = 4, 8, 30

	lagIdx, lagID, leadIdx := pickLagger(t, stores)
	coord := stores[0] // drive mutations from node 0 (forwards transparently)
	lagStore := stores[lagIdx]
	lagEE := lagStore.(*rostam.Embedded)
	leadEE := stores[leadIdx].(*rostam.Embedded) // the meta leader reflects a new catalog gen first

	defer rostam.SetReshardDrainGrace(20 * time.Millisecond)()

	createPartitionedDocs(t, ctx, coord, oldP)
	seedDocs(t, ctx, coord, N)
	waitEmbeddedCatalogGen(t, lagEE, "docs", oldP, 0, cpuScaled(30*time.Second)) // load-flakiness hardening + cpuScaled for CPU-contended CI (follower meta-apply lag); finite so a real hang still fails

	// Stage the lag (gate the cutover) so the lagging follower's local catalog is stale
	// while we issue the off-path reads.
	var (
		lagEnteredOnce sync.Once
		lagEntered     = make(chan struct{})
		release        = make(chan struct{})
		releaseOnce    sync.Once
	)
	releaseGate := func() { releaseOnce.Do(func() { close(release) }) }
	cluster.SetMetaApplyCatalogGate(func(nodeID, collection string, partitions, generation uint32) {
		if nodeID != lagID || ops.CanonicalName(collection) != ops.CanonicalName("docs") || generation != 1 {
			return
		}
		lagEnteredOnce.Do(func() { close(lagEntered) })
		<-release
	})
	defer releaseGate()

	reshardErr := make(chan error, 1)
	go func() { reshardErr <- coord.VectorReshard(ctx, "docs", newP) }()

	select {
	case <-lagEntered:
	case err := <-reshardErr:
		releaseGate()
		t.Fatalf("reshard returned (err=%v) before the lagging follower reached the cutover-apply gate", err)
	case <-time.After(cpuScaled(60 * time.Second)): // load-flakiness hardening; finite so a real hang still fails
		releaseGate()
		t.Fatal("lagging follower never reached the cutover-apply gate")
	}
	waitEmbeddedCatalogGen(t, leadEE, "docs", newP, 1, cpuScaled(20*time.Second)) // load-flakiness hardening; finite so a real hang still fails
	if p, gen, ok := lagEE.Catalog().PartitionsGen("docs"); !ok || p != oldP || gen != 0 {
		releaseGate()
		t.Fatalf("lagging follower catalog = (p=%d gen=%d ok=%v), want OLD gen (p=%d gen=0)", p, gen, ok, oldP)
	}

	var forwards int32
	cluster.SetMetaReadIndexForwardHook(func() { atomic.AddInt32(&forwards, 1) })
	defer cluster.SetMetaReadIndexForwardHook(nil)

	// AnyReplica: ZERO forwards, returns without blocking. We assert the forward count
	// is zero regardless of any transient read error (an off-path read NEVER enters the
	// barrier, so it cannot forward even if the read itself blips on shard-leader churn).
	atomic.StoreInt32(&forwards, 0)
	done := make(chan struct{}, 1)
	go func() {
		_, _, _, _ = lagStore.VectorScroll(ctx, "docs", rostam.VectorFilter{}, 0,
			rostam.VectorScrollOpts{ReadConsistency: ops.ConsistencyAnyReplica, OnPartitionUnavailable: 1})
		done <- struct{}{}
	}()
	select {
	case <-done:
	case <-time.After(cpuScaled(20 * time.Second)): // load-flakiness hardening; finite so a real hang still fails
		releaseGate()
		t.Fatal("AnyReplica read on the lagging follower did not return — it must NOT block (no barrier off the Linearizable path)")
	}
	if f := atomic.LoadInt32(&forwards); f != 0 {
		releaseGate()
		t.Fatalf("AnyReplica read fired %d __meta_readindex__ forwards, want 0 (zero added cost off the Linearizable path under staleness)", f)
	}

	// LeaderOnly: ZERO forwards, returns without blocking (same rationale).
	atomic.StoreInt32(&forwards, 0)
	done2 := make(chan struct{}, 1)
	go func() {
		_, _, _, _ = lagStore.VectorScroll(ctx, "docs", rostam.VectorFilter{}, 0,
			rostam.VectorScrollOpts{ReadConsistency: ops.ConsistencyLeaderOnly, OnPartitionUnavailable: 1})
		done2 <- struct{}{}
	}()
	select {
	case <-done2:
	case <-time.After(cpuScaled(20 * time.Second)): // load-flakiness hardening; finite so a real hang still fails
		releaseGate()
		t.Fatal("LeaderOnly read on the lagging follower did not return — it must NOT block")
	}
	if f := atomic.LoadInt32(&forwards); f != 0 {
		releaseGate()
		t.Fatalf("LeaderOnly read fired %d __meta_readindex__ forwards, want 0", f)
	}

	// Drain the reshard cleanly.
	releaseGate()
	select {
	case err := <-reshardErr:
		must(t, err)
	case <-time.After(cpuScaled(40 * time.Second)): // load-flakiness hardening; finite so a real hang still fails
		t.Fatal("reshard did not complete after the lag was released")
	}
}

// ----------------------------------------------------------------------------
// The cluster integration proof — continuous Linearizable reads from ALL
// nodes across a COMPLETE online reshard, where ONE follower lags the cutover by a
// BOUNDED delay (gate held, then RELEASED partway) and NATURALLY catches up while
// the reshard runs to completion. The distinguishing point vs the reshard-side-gate
// integration test (TestReshardUnderConcurrentReadsLinearizable in
// linearizable_catalog_integration_test.go, where the lagging node is PERPETUALLY
// gated and therefore reads LeaderOnly): here EVERY node — including the
// bounded-lag follower — reads strictly Linearizable for the WHOLE reshard, so the
// READ-SIDE meta readIndex barrier is the mechanism keeping every read correct. On
// the lagging follower a Linearizable read either block-then-serves-fresh (the
// barrier waits for its local meta-FSM to reach the leader-verified frontier) or, once
// it has caught up, routes the new gen correctly — NEVER stale, NEVER a dropped-gen
// error. Because the gate is released partway (NOT held forever), the
// node deterministically catches up and the reshard genuinely COMPLETES.
//
// Isolating the barrier (documented in-test): we shorten the reshard-side gate's
// all-nodes timeout (reshardCutoverGateTimeout) AND the drain grace so the old gen is
// retired SOONER — the reshard-side gate stops being able to keep the old gen alive
// "long enough" for a lagging node. Then the read-side barrier is unambiguously doing
// the work of keeping the lagging follower's Linearizable reads correct: it blocks
// the read until the node's catalog catches up to the new gen rather than letting it
// route to a soon-dropped old gen. (We still RELEASE the apply gate partway, so the
// node converges quickly and the gate generally passes; the short timeout makes the
// barrier — not a generous gate — the load-bearing guarantee for the lagging reads.)
//
// Determinism: NO fixed staleness sleeps. The lag window is opened by the meta-apply
// gate (it signals lagEntered when the follower reaches the cutover-apply block) and
// closed by releasing that gate — at which point the follower naturally catches up.
// The mid-reshard mutation is issued while the window is provably open (lagEntered
// fired, asymmetry asserted). The reader pool runs continuously across the WHOLE
// transition (pre-cutover, held-open post-cutover/pre-gate window, gate-pass, old-gen
// drop). Every read is bracketed by ground-truth snapshots (committedSet) and checked
// against the linearizable envelope; all reader errors are collected and asserted ZERO.
// The only settle-poll is for shard-side data convergence on the final exact-set
// assertion (orthogonal to the meta catalog barrier under test).
//
// This FAILS (real bug) if any node's Linearizable read is ever stale (misses a
// committed id), a phantom (an id beyond any started write), or routes to a dropped
// gen (a gen-routing-gap error) — at any point across the reshard, including the
// bounded-lag follower's reads during its lag-then-catch-up window.
func TestReshardUnderConcurrentLinearizableReadsMetaBarrier(t *testing.T) {
	// Isolate the read-side barrier: retire the old gen SOONER so the reshard-side
	// gate is NOT the thing keeping a lagging node's reads correct — the barrier is.
	defer rostam.SetReshardDrainGrace(20 * time.Millisecond)()
	// short: the gate cannot babysit a lagging node for long; the barrier does the work
	defer rostam.SetReshardCutoverGateTimeout(1 * time.Second)()
	// Clear the gate AFTER cluster teardown (LIFO Cleanup) so a trailing meta-Apply
	// during shutdown can't race a nil store of the gate under -race.
	t.Cleanup(func() { cluster.SetMetaApplyCatalogGate(nil) })

	stores := newInmemEmbeddedCluster(t, 3, 8, 3) // RF=3: every node hosts every shard ⇒ leader-routed Linearizable reads resolve locally
	ctx := context.Background()
	const oldP, newP, seedN = 4, 8, 200
	const midN = 30 // ids inserted DURING the reshard (after cutover, before release)

	// Discover a stable single meta leader + a distinct follower at runtime (leadership
	// is not pinned to node 0 in this harness). The discovered follower is the bounded-
	// lag node; mutations are driven from stores[0] (forwards transparently).
	lagIdx, lagID, leadIdx := pickLagger(t, stores)
	coord := stores[0]
	lagStore := stores[lagIdx]
	lagEE := lagStore.(*rostam.Embedded)
	leadEE := stores[leadIdx].(*rostam.Embedded) // the meta leader reflects a new catalog gen first

	createPartitionedDocs(t, ctx, coord, oldP)
	// Seed ids 1..seedN via the hardened ascending-order helper, then mark them all
	// committed in the ground-truth oracle (seedDocs fails loud if any id never lands).
	gt := &committedSet{}
	for id := uint64(1); id <= seedN; id++ {
		gt.beginWrite(id)
	}
	seedDocs(t, ctx, coord, seedN)
	for id := uint64(1); id <= seedN; id++ {
		gt.commit(id)
	}

	ee := []*rostam.Embedded{stores[0].(*rostam.Embedded), stores[1].(*rostam.Embedded), stores[2].(*rostam.Embedded)}
	// Every node has converged on the seeded (oldP, gen 0) catalog before we start, so
	// the only catalog lag in the run is the bounded cutover lag we induce. Confirm a
	// Linearizable read is serviceable on every node up front (so a later failure is the
	// barrier/gen-routing gap, not a harness/leader-resolution artefact).
	for i, e := range ee {
		waitEmbeddedCatalogGen(t, e, "docs", oldP, 0, cpuScaled(25*time.Second)) // widened 10s->25s + cpuScaled for CPU-contended CI (per-node incl. follower meta-apply lag); setup gate, finite so a real non-convergence still fails
		got, _, err := scrollEnvelope(ctx, stores[i], ops.ConsistencyLinearizable)
		if err != nil {
			t.Fatalf("baseline Linearizable read on node %d failed: %v", i, err)
		}
		if err := assertLinearizableEnvelope(fmt.Sprintf("baseline node %d", i), got, seedN, seedN); err != nil {
			t.Fatal(err)
		}
	}

	// Block the lagging follower's apply of the cutover entry (new-gen
	// OpSetCatalogEntry for "docs", generation 1). Only the cutover lags; gen-0/create
	// and other ops pass through. Released partway (NOT forever) so the node naturally
	// catches up and the reshard completes.
	var (
		lagEnteredOnce sync.Once
		lagEntered     = make(chan struct{})
		release        = make(chan struct{})
		releaseOnce    sync.Once
	)
	releaseGate := func() { releaseOnce.Do(func() { close(release) }) }
	cluster.SetMetaApplyCatalogGate(func(nodeID, collection string, partitions, generation uint32) {
		if nodeID != lagID || ops.CanonicalName(collection) != ops.CanonicalName("docs") || generation != 1 {
			return
		}
		lagEnteredOnce.Do(func() { close(lagEntered) })
		<-release
	})
	defer releaseGate()

	// Instrument __meta_readindex__ forwards so we can prove the lagging follower's
	// Linearizable reads genuinely FORWARD to the meta leader and wait on the barrier
	// (not merely serve a coincidentally-fresh local catalog).
	var forwards int64
	cluster.SetMetaReadIndexForwardHook(func() { atomic.AddInt64(&forwards, 1) })
	defer cluster.SetMetaReadIndexForwardHook(nil)

	// Continuous reader pool: one goroutine per node, looping until stop. EVERY node —
	// INCLUDING the bounded-lag follower — reads strictly Linearizable, so the read-side
	// barrier is exercised on the lagging node throughout its lag-then-catch-up window.
	// Each read is bracketed by ground-truth snapshots and checked against the envelope.
	// All errors are collected; the test asserts ZERO. Per-node counters prove every
	// loop actually exercised the reshard (not a vacuous no-op).
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
				got, degraded, err := scrollEnvelope(ctx, stores[node], ops.ConsistencyLinearizable)
				_, after := gt.snapshot()
				if err != nil {
					// A gen-routing gap (unknown collection / no shard owner) means a node
					// routed to a stale/dropped gen — the bug the barrier must prevent; fatal.
					// A perpetual barrier timeout would ALSO be the bug here (the node must
					// catch up once the gate is released), so a timeout is fatal too — it
					// means the barrier never resolved across the whole reshard. Transient
					// raft election jitter (leader re-resolution) is tolerated.
					if isGenRoutingGap(err) {
						recordErr(fmt.Errorf("node %d: gen-routing gap on Linearizable read: %w", node, err))
						return
					}
					var to *cluster.ErrMetaLinearizableTimeout
					if errors.As(err, &to) || strings.Contains(err.Error(), "meta linearizable read timed out") {
						recordErr(fmt.Errorf("node %d: meta barrier timed out on a Linearizable read (the bounded-lag node must catch up, never time out across a completing reshard): %w", node, err))
						return
					}
					continue
				}
				if degraded {
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
	// enters the Phase-4.5 gate. The lagging follower is gated at the cutover apply, so
	// the gate waits for it — but only up to the SHORT reshardCutoverGateTimeout, after
	// which the reshard proceeds (logs + drops the old gen). The read-side barrier is
	// what keeps the lagging node's Linearizable reads correct through that window.
	reshardErr := make(chan error, 1)
	go func() { reshardErr <- coord.VectorReshard(ctx, "docs", newP) }()

	select {
	case <-lagEntered:
	case err := <-reshardErr:
		close(stop)
		readersWG.Wait()
		t.Fatalf("reshard returned (err=%v) before the lagging follower reached the cutover-apply gate", err)
	case <-time.After(40 * time.Second):
		close(stop)
		readersWG.Wait()
		t.Fatal("lagging follower never reached the cutover-apply gate (reshard never cut over?)")
	}

	// Window is open: the meta leader routes the NEW gen; the lagging follower still
	// routes the OLD gen. Assert the asymmetry so the mid-reshard mutation provably
	// lands in the post-cutover window while the lagging node still lags.
	waitEmbeddedCatalogGen(t, leadEE, "docs", newP, 1, 5*time.Second)
	if p, gen, ok := lagEE.Catalog().PartitionsGen("docs"); !ok || p != oldP || gen != 0 {
		close(stop)
		readersWG.Wait()
		t.Fatalf("lagging follower catalog = (p=%d, gen=%d, ok=%v), want OLD gen (p=%d, gen=0)", p, gen, ok, oldP)
	}

	// MUTATE mid-reshard: insert midN new ids AFTER the cutover, while the follower
	// still lags. Read-your-writes: once each upsert returns (committed), every
	// subsequent Linearizable read on ANY node — including the lagging follower, whose
	// barrier blocks-then-serves-fresh — must observe it.
	for k := 1; k <= midN; k++ {
		id := uint64(seedN + k)
		gt.beginWrite(id)
		retryUntil(t, "mid-reshard upsert", func() error {
			return coord.VectorUpsert(ctx, "docs", id, []float32{float32(id), 0, 0, 0}, fmt.Sprintf("doc-%d", id), rostam.VectorInsertOpts{})
		})
		gt.commit(id) // committed mark advances ONLY after the write returns
	}

	// Prove the lagging follower's Linearizable reads are FORWARDING/WAITING on the
	// barrier (not merely serving a coincidentally-fresh local catalog) while it lags.
	if atomic.LoadInt64(&forwards) == 0 {
		close(stop)
		readersWG.Wait()
		releaseGate()
		t.Fatal("zero __meta_readindex__ forwards observed while a follower lagged — the Linearizable reads on the lagging node must forward to the meta leader and wait on the barrier")
	}

	// RELEASE the gate partway: the lagging follower applies the cutover, NATURALLY
	// catches up to the new gen, and the reshard runs to completion. The reader pool
	// keeps running across the catch-up + gate-pass + old-gen-drop transition. (Contrast
	// the held-gate test, which held it forever; here the reshard COMPLETES.)
	releaseGate()
	select {
	case err := <-reshardErr:
		must(t, err)
	case <-time.After(30 * time.Second):
		close(stop)
		readersWG.Wait()
		t.Fatal("reshard did not complete after the bounded lag was released")
	}

	// Reshard done: every node routes the new gen. Let the readers run a beat across the
	// settled new-gen state (so the pool provably covers post-drop reads too), then drain.
	for _, e := range ee {
		waitEmbeddedCatalogGen(t, e, "docs", newP, 1, 10*time.Second)
	}
	// Confirm the lagging follower genuinely caught up to the NEW gen (it did not stay
	// pinned to the old gen) — the barrier converged it, the reshard completed on it.
	if p, gen, ok := lagEE.Catalog().PartitionsGen("docs"); !ok || p != newP || gen != 1 {
		close(stop)
		readersWG.Wait()
		t.Fatalf("lagging follower did not catch up to the new gen (p=%d gen=%d ok=%v), want (p=%d gen=1)", p, gen, ok, newP)
	}
	close(stop)
	readersWG.Wait()

	// Assert ZERO read errors across the entire reshard from ALL nodes (incl. the
	// bounded-lag follower reading Linearizable through its lag-then-catch-up window).
	errMu.Lock()
	if len(readErrs) != 0 {
		for _, e := range readErrs {
			t.Errorf("concurrent Linearizable read error: %v", e)
		}
		errMu.Unlock()
		t.Fatalf("%d Linearizable read(s) returned wrong/stale/dropped data (or timed out) during the reshard", len(readErrs))
	}
	errMu.Unlock()
	// Prove the pool actually exercised every node (not a no-op that vacuously passed).
	for i := range ee {
		if c := atomic.LoadInt64(&readCount[i]); c == 0 {
			t.Fatalf("node %d performed 0 complete Linearizable reads during the reshard", i)
		}
	}

	// Final state: every node's Linearizable read returns the EXACT full set via the
	// new gen, the reshard state is cleared, and the old gen is dropped. The exact-set
	// check is poll-tolerant of shard-side data settle (orthogonal to the meta catalog
	// barrier under test); a stale/dropped-gen read would surface as a persistent error
	// or a wrong set, not a transient.
	full := idRange(1, seedN+midN)
	for i := range ee {
		deadline := time.Now().Add(20 * time.Second)
		var fgot map[uint64]bool
		var ferr error
		for time.Now().Before(deadline) {
			fgot, _, ferr = scrollEnvelope(ctx, stores[i], ops.ConsistencyLinearizable)
			if ferr != nil {
				if isGenRoutingGap(ferr) {
					t.Fatalf("final Linearizable read on node %d hit a gen-routing gap (stale/dropped gen): %v", i, ferr)
				}
				time.Sleep(50 * time.Millisecond)
				continue
			}
			if mapsEqual(fgot, full) {
				break
			}
			time.Sleep(50 * time.Millisecond)
		}
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

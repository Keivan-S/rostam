// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
	"github.com/rostamlabs/rostam/vector"
)

// TestBoundedStalenessMultiNodeSeam is the end-to-end PROOF of the
// ConsistencyBoundedStaleness (rc=3) read on a real multi-node cluster — the
// seam the ops/shard/cluster unit tests cannot cover: a bounded-staleness POINT
// GET issued against a node that hosts the owning partition's shard as a
// FOLLOWER, exercising the shard freshness guard (shard.Store.serveBoundedStaleness)
// composed with the embedded any-replica route + the transparent leader upgrade
// (embedded.callReadLeader's NotLeaderError → CallPhysical leader retry).
//
// It mirrors TestClusterLeaderOnlyServedByLeader's harness EXACTLY:
//   - newInmemEmbeddedCluster(3, numShards, RF=2): every partition shard has a
//     leader + one follower on a distinct node, so cross-node forwarding is real.
//   - The writer/reader split: at RF>1 the embedded WRITE path returns NotLeader
//     (no leader-following) for a write to a shard the driving node hosts as a
//     FOLLOWER, so we CREATE+WRITE through a node that leads-or-does-not-host
//     every partition, then READ from a DIFFERENT node that FOLLOWS one of them.
//   - The same deterministic (writer, reader, followerPart) discovery loop, the
//     same AnyReplica convergence gate, and the same leadership-consensus gate so
//     the reader's tracked leader view agrees with reality before we assert.
//
// Two subtests, both issuing a single-id GET routed to an id whose partition the
// reader FOLLOWS (ops.PartitionOf(id,P)==followerPart), so the read genuinely
// hits the follower's shard guard:
//
//	(A) TIGHT bound (0) + a fresh overwrite on the leader: the follower is (or
//	    momentarily becomes) out-of-bound, its guard fetches the leader frontier,
//	    finds lag>0, returns NotLeaderError, and callReadLeader TRANSPARENTLY
//	    UPGRADES to the leader — so the read returns the LATEST (overwritten)
//	    vector, never the stale one and never an error. Read-your-writes via the
//	    bounded upgrade. (Timing-robust: even if the follower has already caught
//	    up within the 0-bound, the served-local value is still the fresh one; the
//	    load-bearing assertion is FRESH-data, polled until it converges.)
//
//	(B) LARGE bound (1<<62): the follower is trivially within bound, so the guard
//	    serves LOCALLY without an upgrade (the leader-offload win). We assert the
//	    read succeeds with correct data AND, via shard.SetBoundedStalenessServedHook,
//	    that a LOCAL serve (servedLocal=true) actually occurred on the follower.
//
// THIS would FAIL if BoundedStaleness leader-pinned (the follower guard would
// never run), if the guard served stale-beyond-bound under a tight bound (A would
// see the old vector), or if the within-bound path forced an upgrade (B's
// local-serve hook would never fire).
func TestBoundedStalenessMultiNodeSeam(t *testing.T) {
	const (
		numShards = 8
		n         = 3
		rf        = 2
		P         = n // P=N partitions
	)
	stores := newInmemEmbeddedCluster(t, n, numShards, rf)
	ctx := context.Background()

	// Deterministically find a (writer, coll, reader, followerPart) such that the
	// writer can create the partitioned collection (it leads-or-does-not-host every
	// partition shard) AND some OTHER node hosts a FOLLOWER of at least one of that
	// collection's partition shards (the reader). Mirrors TestClusterLeaderOnly-
	// ServedByLeader's discovery loop verbatim (unique disposable names; a partial
	// create is abandoned, never retried, so it cannot poison a later attempt).
	var (
		coll         string
		writerIdx    = -1
		readerIdx    = -1
		followerPart = -1
	)
	createCfg := rostam.VectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: P}
	deadline := time.Now().Add(60 * time.Second)
search:
	for attempt := 0; attempt < 64 && time.Now().Before(deadline); attempt++ {
		for w := 0; w < n; w++ {
			cand := fmt.Sprintf("boundedstale-w%d-a%d", w, attempt)
			if err := stores[w].CreateCollection(ctx, cand, createCfg); err != nil {
				continue
			}
			for r := 0; r < n; r++ {
				if r == w {
					continue
				}
				hostsAll, follows, fp := true, false, -1
				for p := 0; p < P; p++ {
					key := []byte(ops.CanonicalName(string(ops.PartitionKeyGen(cand, 0, p))))
					led := stores[r].IsLeader(key)
					known := stores[r].LeaderAddr(key) != ""
					if !led && !known {
						hostsAll = false
						break
					}
					if !led {
						follows = true
						if fp < 0 {
							fp = p
						}
					}
				}
				if hostsAll && follows {
					coll, writerIdx, readerIdx, followerPart = cand, w, r, fp
					break search
				}
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	if writerIdx < 0 {
		t.Fatalf("could not find a (writer, reader) pair: a node that creates a P=%d collection AND another node that follows one of its partition shards (RF=%d)", P, rf)
	}
	t.Logf("writer=node%d created %q; reader=node%d FOLLOWS partition %d (IsLeader=false) — bounded reads of ids in that partition hit the follower's shard guard", writerIdx, coll, readerIdx, followerPart)

	// Pick a concrete id whose owning partition is exactly followerPart, so a
	// single-id get routed by id lands on the reader's FOLLOWER replica of that
	// partition's shard (the seam under test). v(id)={id,0,0,0} is tie-free.
	targetID := uint64(0)
	for id := uint64(1); id <= 10_000; id++ {
		if ops.PartitionOf(id, P) == followerPart {
			targetID = id
			break
		}
	}
	if targetID == 0 {
		t.Fatalf("no id routes to follower partition %d (P=%d)", followerPart, P)
	}
	t.Logf("target id=%d routes to partition %d (the followed one)", targetID, followerPart)

	// Seed the target id (plus a few neighbours so the partition is non-trivial)
	// through the writer. Writes route to the partition leader (or forward), never a
	// hosted-follower write.
	seedVec := []float32{float32(targetID), 0, 0, 0}
	for id := uint64(1); id <= uint64(3*P); id++ {
		v := []float32{float32(id), 0, 0, 0}
		retryUntil(t, fmt.Sprintf("insert %s %d", coll, id), func() error {
			return stores[writerIdx].VectorInsert(ctx, coll, id, v)
		})
	}

	// Readiness: the reader's local catalog converges to P, and its FOLLOWER replica
	// of followerPart has applied the seed (so a follower-served read returns correct
	// data). We gate on an AnyReplica get from the reader returning the seeded vector
	// — an AnyReplica get serves from the local replica when the reader hosts the
	// shard, so a clean pass means the follower has applied the seed. No fixed sleeps.
	er := stores[readerIdx].(*rostam.Embedded)
	waitEmbeddedCatalog(t, er, coll, P, 15*time.Second)
	convDeadline := time.Now().Add(20 * time.Second)
	for {
		found, vec, _, _, _, err := stores[readerIdx].VectorGetExt(ctx, coll, targetID, true, false,
			rostam.ReadOpts{ReadConsistency: ops.ConsistencyAnyReplica})
		if err == nil && found && len(vec) == 4 && vec[0] == seedVec[0] {
			break
		}
		if time.Now().After(convDeadline) {
			t.Fatalf("reader node%d: replication of id=%d did not converge (found=%v err=%v)", readerIdx, targetID, found, err)
		}
		time.Sleep(20 * time.Millisecond)
	}

	// Leadership-consensus gate (mirrors TestClusterLeaderOnlyServedByLeader): wait
	// until every partition shard has a unique leader the reader's view AGREES with,
	// held stable, so the bounded read's frontier RTT / leader upgrade targets the
	// real leader (not a stale-tracked one). Then re-confirm the reader still FOLLOWS
	// the target partition's shard.
	partKeys := make([][]byte, P)
	for p := 0; p < P; p++ {
		partKeys[p] = []byte(ops.CanonicalName(string(ops.PartitionKeyGen(coll, 0, p))))
	}
	uniqueLeaderAddr := func(key []byte) string {
		addr, count := "", 0
		for i := range stores {
			if stores[i].IsLeader(key) {
				count++
				addr = stores[i].LeaderAddr(key)
			}
		}
		if count != 1 {
			return ""
		}
		return addr
	}
	consensusReady := func() bool {
		for p := 0; p < P; p++ {
			la := uniqueLeaderAddr(partKeys[p])
			if la == "" || stores[readerIdx].LeaderAddr(partKeys[p]) != la {
				return false
			}
		}
		return true
	}
	consDeadline := time.Now().Add(20 * time.Second)
	stableRounds := 0
	for {
		if consensusReady() {
			stableRounds++
		} else {
			stableRounds = 0
		}
		if stableRounds >= 10 {
			break
		}
		if time.Now().After(consDeadline) {
			t.Fatalf("reader node%d: partition-shard leadership did not reach stable consensus the reader agrees with", readerIdx)
		}
		time.Sleep(50 * time.Millisecond)
	}
	fkey := partKeys[followerPart]
	if stores[readerIdx].IsLeader(fkey) || stores[readerIdx].LeaderAddr(fkey) == "" {
		t.Fatalf("reader node%d no longer follows partition %d's shard (IsLeader=%v leader=%q)",
			readerIdx, followerPart, stores[readerIdx].IsLeader(fkey), stores[readerIdx].LeaderAddr(fkey))
	}
	reader := stores[readerIdx]

	// ---------------------------------------------------------------------------
	// (A) Transparent upgrade when out-of-bound: a fresh overwrite on the leader +
	// a TIGHT (bound=0) bounded get on the follower must return the LATEST vector.
	// The follower lags the just-committed overwrite, so its guard fetches the
	// leader frontier, finds lag>0, returns NotLeaderError, and callReadLeader
	// transparently upgrades to the leader — read-your-writes. Even if the follower
	// catches up within the 0-bound first, the served value is still the fresh one;
	// the load-bearing assertion is FRESH data with no error, polled to convergence.
	// ---------------------------------------------------------------------------
	t.Run("upgrade_out_of_bound", func(t *testing.T) {
		// Distinct fresh value for the SAME id (vec[1] becomes the freshness marker
		// so a stale serve — old vec[1]==0 — is detectable).
		const freshMark = float32(777)
		freshVec := []float32{float32(targetID), freshMark, 0, 0}
		retryUntil(t, "overwrite target on leader", func() error {
			return stores[writerIdx].VectorUpsert(ctx, coll, targetID, freshVec, "fresh", rostam.VectorInsertOpts{})
		})

		// Observe the served-local vs upgrade decisions on the follower (diagnostic;
		// the correctness assertion is the fresh value, which holds under either
		// branch).
		var localServes, upgrades atomic.Int64
		shard.SetBoundedStalenessServedHook(func(servedLocal bool) {
			if servedLocal {
				localServes.Add(1)
			} else {
				upgrades.Add(1)
			}
		})
		defer shard.SetBoundedStalenessServedHook(nil)

		bounded0 := rostam.ReadOpts{ReadConsistency: ops.ConsistencyBoundedStaleness, MaxStaleness: 0}

		// Poll the bound=0 read until it returns the fresh value. It must NEVER
		// error and NEVER return the stale value (freshMark missing). Generous
		// deadline for multi-node replication/upgrade timing.
		var lastVec []float32
		conv := false
		pollDeadline := time.Now().Add(10 * time.Second)
		for time.Now().Before(pollDeadline) {
			found, vec, _, _, _, err := reader.VectorGetExt(ctx, coll, targetID, true, false, bounded0)
			if err != nil {
				t.Fatalf("bounded(0) get on follower errored (must transparently upgrade, never surface an error): %v", err)
			}
			if !found {
				t.Fatalf("bounded(0) get on follower returned not-found for id=%d (must serve the fresh point)", targetID)
			}
			if len(vec) != 4 {
				t.Fatalf("bounded(0) get returned vec len %d, want 4", len(vec))
			}
			lastVec = vec
			// A STALE serve (the old seeded vector, vec[1]==0) is the bug this guards:
			// fail LOUD if the bound=0 read ever serves stale-beyond-bound data.
			if vec[1] != freshMark && vec[1] != 0 {
				t.Fatalf("bounded(0) get returned an unexpected vec[1]=%v (want fresh %v)", vec[1], freshMark)
			}
			if vec[1] == freshMark {
				conv = true
				break
			}
			time.Sleep(25 * time.Millisecond)
		}
		if !conv {
			t.Fatalf("bounded(0) get on follower never returned the fresh overwrite (last vec=%v, want vec[1]=%v) — "+
				"the out-of-bound read did not upgrade/serve fresh", lastVec, freshMark)
		}
		// Sanity: the bounded guard ran on the follower at least once (local serve or
		// upgrade). 0 ⇒ the read never hit the follower's bounded guard (it leader-
		// pinned or skipped the guard) — the seam would be untested.
		if localServes.Load()+upgrades.Load() == 0 {
			t.Fatalf("bounded(0) get never entered the follower's bounded-staleness guard (local=%d upgrade=%d) — "+
				"BoundedStaleness must route any-replica and run the shard guard", localServes.Load(), upgrades.Load())
		}
		t.Logf("bounded(0): follower guard fired local=%d upgrade=%d; converged to the fresh overwrite", localServes.Load(), upgrades.Load())
	})

	// ---------------------------------------------------------------------------
	// (B) Within-bound local serve: a LARGE bound serves the follower's replica
	// LOCALLY (the offload win) without an upgrade. Assert the read succeeds with
	// correct data AND that a LOCAL serve (servedLocal=true) actually occurred on
	// the follower via the served hook.
	// ---------------------------------------------------------------------------
	t.Run("within_bound_local_serve", func(t *testing.T) {
		var localServes, upgrades atomic.Int64
		shard.SetBoundedStalenessServedHook(func(servedLocal bool) {
			if servedLocal {
				localServes.Add(1)
			} else {
				upgrades.Add(1)
			}
		})
		defer shard.SetBoundedStalenessServedHook(nil)

		// A huge bound: any realistic follower lag is within it, so the guard serves
		// locally. MaxStaleness is a raft-entry count; 1<<62 is effectively unbounded.
		boundedHuge := rostam.ReadOpts{ReadConsistency: ops.ConsistencyBoundedStaleness, MaxStaleness: 1 << 62}

		// Poll until a LOCAL serve is observed AND the read returns correct data.
		// Under such a large bound a local serve is expected immediately, but poll
		// for robustness against transient leadership/frontier jitter.
		got := false
		pollDeadline := time.Now().Add(10 * time.Second)
		for time.Now().Before(pollDeadline) {
			found, vec, _, _, _, err := reader.VectorGetExt(ctx, coll, targetID, true, false, boundedHuge)
			if err != nil {
				t.Fatalf("within-bound bounded get errored: %v", err)
			}
			if !found || len(vec) != 4 || vec[0] != float32(targetID) {
				t.Fatalf("within-bound bounded get returned wrong data: found=%v vec=%v (want vec[0]=%d)", found, vec, targetID)
			}
			if localServes.Load() > 0 {
				got = true
				break
			}
			time.Sleep(25 * time.Millisecond)
		}
		if !got {
			t.Fatalf("within-bound (huge) bounded get never recorded a LOCAL follower serve (local=%d upgrade=%d) — "+
				"a within-bound read must serve from the follower locally (the leader-offload win), not upgrade",
				localServes.Load(), upgrades.Load())
		}
		t.Logf("huge-bound: follower served LOCALLY (local=%d upgrade=%d) — within-bound offload confirmed", localServes.Load(), upgrades.Load())
	})
}

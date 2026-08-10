// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"fmt"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// TestPBFailoverNoAckedLoss is the failover GO/NO-GO gate: kill a primary under load
// and prove automatic failover loses ZERO acknowledged writes.
//
// Setup: a 3-node RF=full PB cluster with PBAutoFailover ON and every failover
// timing SHRUNK via config (NOT a fake clock — hashicorp's LastContact uses a
// non-injectable real clock, so we drive the real timings small: pbLeaseTTL=1s,
// metaContactStaleness=500ms, failoverTimeout=3s, renewInterval=300ms,
// RaftHeartbeat=200ms). The honor rule still holds:
// 3s > 1s + 500ms + 300ms (renewInterval) + 500ms (tick) = 2.3s.
//
// NOTE this is a CRASH-STOP test (node.Close stops the old primary acking
// instantly), so it proves no-acked-LOSS, not the no-DOUBLE-primary partition
// window (a partitioned-but-alive primary self-fencing before its replacement is
// named). That window is covered by the corrected honor-rule assertion + the
// self-fence unit test; a full network-partition e2e test remains the explicit
// gate item before PBAutoFailover goes default-on (see shard/pbisr/DESIGN.md).
//
// We write K keys to the primary (full-ISR commit ⇒ every acked write is on all
// three nodes), record every acked key, then kill the primary with node.Close()
// (a deterministic crash; the surviving two keep meta quorum). The meta leader's
// failover ticker observes the primary go silent past failoverTimeout and promotes
// the lowest-id ISR survivor. We then assert, on the NEW primary:
//   - every pre-kill acked key is present (no acked-loss), and
//   - the engine's LastApplied equals the acked high-water — i.e. it holds the
//     dense seq run 1..K with no gap (full-ISR commit gap-rejects, so a present
//     high-water implies density).
func TestPBFailoverNoAckedLoss(t *testing.T) {
	const numShards = 1
	const sh = 0
	tc := newPBTestCluster(t, 3, numShards, 1, func(c *Config) {
		c.PBAutoFailover = true
		// MinISR=1 is the FLOOR ONLY — it is deliberately the WEAKEST setting, so this
		// gate exercises the case where the floor provides no protection and the
		// no-acked-loss guarantee must come entirely from full-ISR commit. Raising it
		// to 3 would MASK the bug this test exists to catch: it was a primary acking
		// against an ISR of {self} (which a floor of 3 would have rejected outright)
		// because its LOCAL MetaFSM had applied the seed's epoch entry but not yet the
		// separate ISR entry — an ISR narrower than the COMMITTED one. Do not raise it.
		//
		// Note what full-ISR commit does and does not promise: it waits for every
		// member of the ISR the primary READS, which is only the committed ISR if the
		// primary's view is not stale. The atomic seed (ApplySetShardSeed) is what
		// makes that true at bootstrap.
		c.MinISR = 1
		c.PBLeaseTTLMs = 1000
		c.PBMetaContactStalenessMs = 500
		// Honor-rule floor = leaseTTL + staleness + renewInterval + failoverTick
		// = 1000 + 500 + 300 + 500 = 2300ms; failoverTimeout must strictly exceed it.
		c.PBFailoverTimeoutMs = 3000
		c.PBRenewIntervalMs = 300
		c.ShardCfg.RaftHeartbeatMs = 200
	})

	// Find the primary node and drive acked writes through it. Poll first: on a node
	// that built its shard before the seed replicated, the self-lease is granted
	// only once the lease-keeper observes the FSM primary (expected follower timing).
	findNodeIdx := func(nodeID string) int {
		for i, p := range tc.peers {
			if p.NodeID == nodeID {
				return i
			}
		}
		return -1
	}

	const K = 60
	acked := make(map[string][]byte, K)
	var primaryIdx = -1
	var origPrimary string

	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) && primaryIdx < 0 {
		primary := tc.nodes[0].meta.FSM.ShardPrimary(sh)
		if primary == "" {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		idx := findNodeIdx(primary)
		if idx < 0 {
			t.Fatalf("primary %q not in peer list", primary)
		}
		// Probe once: if the primary can't yet accept a write, keep polling.
		probe := ops.EncodePutArgs([]byte("probe"), []byte("v"), 0)
		if _, err := tc.nodes[idx].Call("put", probe); err != nil {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		primaryIdx = idx
		origPrimary = primary
	}
	if primaryIdx < 0 {
		t.Fatal("PB primary never accepted a write within 20s")
	}

	// (0) PRE-WRITE ISR ASSERTION — the direct detector for the acked-loss defect.
	// The primary has just accepted a write, so it holds a valid lease at the seeded
	// epoch. Its OWN local MetaFSM must therefore already show the FULL placement as
	// the ISR: that is the view proposeSequenced reads to build the per-write peer
	// set, and full-ISR commit is only a durability guarantee if it is complete.
	//
	// This is deliberately a hard assert, not a poll. Under the two-entry seed the
	// primary could be leased and writable while its FSM still read ISR=[self] (the
	// epoch entry applied, the ISR entry not yet), so every write it acked lived on
	// one node. Polling here would paper over exactly that window.
	{
		// Reference set: the node's OWN computed placement — exactly what
		// pbShardControlSeeds derived the seed from. Deliberately NOT
		// MetaState.Placement: that table is filled by OpSetPlacement self-heal on its
		// own schedule and is legitimately still empty this early, which says nothing
		// about the ISR. n.placement is written once at construction and read-only
		// after (node.go), so reading it here is race-free.
		owners := append([]string(nil), tc.nodes[primaryIdx].shardOwners(sh)...)
		isr := append([]string(nil), tc.nodes[primaryIdx].meta.FSM.ShardISR(sh)...)
		sort.Strings(owners)
		sort.Strings(isr)
		if len(owners) == 0 {
			t.Fatalf("shard %d has no placement owners on the primary node", sh)
		}
		if !reflect.DeepEqual(isr, owners) {
			t.Fatalf("primary %q accepted a write while its OWN MetaFSM read ISR=%v, want the full placement %v "+
				"— it is acking against a view NARROWER than the committed ISR, so its acks are not durable",
				origPrimary, isr, owners)
		}
		t.Logf("pre-write: primary %q sees ISR=%v (full placement)", origPrimary, isr)
	}

	// Write K keys; every returned-nil Put is a full-ISR-committed (durable on all
	// three nodes) acked write we must not lose.
	for i := 0; i < K; i++ {
		key := []byte(fmt.Sprintf("key-%03d", i))
		val := []byte(fmt.Sprintf("val-%03d", i))
		if _, err := tc.nodes[primaryIdx].Call("put", ops.EncodePutArgs(key, val, 0)); err != nil {
			t.Fatalf("pre-kill put %d failed: %v", i, err)
		}
		acked[string(key)] = val
	}
	// The engine's dense seq high-water is the probe (seq 1) + K keys = K+1. A
	// lease-fenced probe attempt burns no seq (TestPrimaryApplyFailureDoesNotBurnSeq),
	// so exactly one probe committed.
	expectedHW := uint64(K + 1)

	// (0b) POST-BURST REPLICATION ASSERTION: the acks we just collected must have
	// been paid for by BOTH backups. Assert it on the primary's own per-peer acked
	// high-water — an empty or short Peers list means the primary committed on
	// itself alone (or ahead of a lagging backup), which is the acked-loss defect
	// caught at its source rather than inferred from a missing key after the kill.
	{
		st := tc.nodes[primaryIdx].pbEngines[sh].ReplicationStatus()
		if len(st.Peers) != len(tc.peers)-1 {
			t.Fatalf("primary acked %d writes with %d replicating peers (%+v), want %d — "+
				"it committed without the full ISR", K, len(st.Peers), st.Peers, len(tc.peers)-1)
		}
		for _, p := range st.Peers {
			if p.Acked < expectedHW {
				t.Fatalf("backup %q acked only up to seq %d, want >= %d — the primary acked "+
					"writes this backup does not hold", p.Peer, p.Acked, expectedHW)
			}
		}
		t.Logf("post-burst: %d backups acked through seq >= %d (%+v)", len(st.Peers), expectedHW, st.Peers)
	}

	// Let the primary beacon a couple of times so every node's tracker has observed
	// it alive (so its subsequent silence is real failover evidence, not a
	// never-seen primary).
	time.Sleep(700 * time.Millisecond)

	// KILL the primary (deterministic crash). The surviving two keep meta quorum.
	if err := tc.nodes[primaryIdx].Close(); err != nil {
		t.Logf("primary Close returned: %v (tolerated)", err)
	}

	// Poll a survivor's replicated FSM for the promotion: a higher epoch AND a new
	// primary that is NOT the dead one.
	survivorIdx := -1
	for i := range tc.nodes {
		if i != primaryIdx {
			survivorIdx = i
			break
		}
	}
	var newPrimary string
	var newEpoch uint64
	promoteDeadline := time.Now().Add(25 * time.Second)
	for time.Now().Before(promoteDeadline) {
		ep := tc.nodes[survivorIdx].meta.FSM.ShardEpoch(sh)
		pr := tc.nodes[survivorIdx].meta.FSM.ShardPrimary(sh)
		if ep > 1 && pr != "" && pr != origPrimary {
			newPrimary, newEpoch = pr, ep
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if newPrimary == "" {
		t.Fatalf("no promotion within 25s (orig primary %q, survivor epoch %d, primary %q)",
			origPrimary, tc.nodes[survivorIdx].meta.FSM.ShardEpoch(sh), tc.nodes[survivorIdx].meta.FSM.ShardPrimary(sh))
	}
	t.Logf("failover: %q (epoch 1) -> %q (epoch %d)", origPrimary, newPrimary, newEpoch)

	newIdx := findNodeIdx(newPrimary)
	if newIdx == primaryIdx || newIdx < 0 {
		t.Fatalf("new primary %q resolves to bad node index %d (dead was %d)", newPrimary, newIdx, primaryIdx)
	}
	newNode := tc.nodes[newIdx]

	// The new primary's engine must have adopted the promotion (its lease-keeper
	// tick calls Promote once it sees the higher epoch). Wait for that.
	eng := newNode.pbEngines[sh]
	if eng == nil {
		t.Fatalf("new primary node %d has no engine for shard %d", newIdx, sh)
	}
	engDeadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(engDeadline) && eng.Epoch() < newEpoch {
		time.Sleep(50 * time.Millisecond)
	}

	// (1) NO ACKED-LOSS: every pre-kill acked key is present on the new primary.
	//
	// Collect EVERY miss before failing. `acked` is a Go map, so a per-key t.Fatalf
	// stops on an arbitrary first miss and reports a single key — which badly
	// understated this defect: when it fired, ALL K acked keys were gone on BOTH
	// survivors, and a one-key message read like an isolated straggler rather than
	// total loss of the burst. The count is the diagnosis.
	store := newNode.getShard(sh)
	missing := make([]string, 0, len(acked))
	for key, want := range acked {
		got, err := store.Get([]byte(key))
		if err != nil || string(got) != string(want) {
			missing = append(missing, fmt.Sprintf("%s(got=%q,err=%v)", key, got, err))
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		shown := missing
		if len(shown) > 10 {
			shown = shown[:10]
		}
		t.Fatalf("ACKED-LOSS: %d of %d acked keys missing on new primary %q; first %d: %v",
			len(missing), len(acked), newPrimary, len(shown), shown)
	}

	// (2) DENSE SEQ RUN: with no post-promotion writes, the new primary's applied
	// high-water must equal the pre-kill total (probe + K). Equality proves every
	// acked seq materialized AND there is no gap in 1..HW (a gap would have
	// gap-rejected at commit, so a full high-water with no gap is dense-in-order).
	if la := eng.LastApplied(); la != expectedHW {
		t.Fatalf("new primary LastApplied=%d, want %d (dense 1..%d) — acked-loss or gap", la, expectedHW, expectedHW)
	}
	t.Logf("new primary LastApplied=%d, acked keys=%d — no acked-loss, dense 1..%d", eng.LastApplied(), len(acked), expectedHW)
}

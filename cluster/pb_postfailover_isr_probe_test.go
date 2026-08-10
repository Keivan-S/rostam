// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"fmt"
	"sort"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// TestPBPostFailoverSingletonISRWindow answers one question empirically: does a
// FRESHLY PROMOTED primary ack writes on itself alone, and if so, are those writes
// lost when it in turn fails?
//
// It is a MEASUREMENT, not a gate. Both outcomes are logged; it fails only if the
// cluster does not get far enough to answer (no promotion, no writes accepted).
//
// Why this differs from the bootstrap defect the atomic seed fixed. Failover
// commits OpSetShardEpoch, which RESETS the ISR to {newPrimary} — deliberately.
// So a fresh primary acking alone is acking against the COMMITTED ISR, which is
// exactly what MinISR=1 licenses. The bootstrap case was different in kind: there
// the primary acked against a view NARROWER THAN COMMITTED (its local FSM had the
// epoch entry but not yet the separate ISR entry), which no MinISR setting
// licenses. Do not conflate the two.
//
// Five nodes so that killing two primaries still leaves a 3-node meta quorum.
func TestPBPostFailoverSingletonISRWindow(t *testing.T) {
	const numShards = 1
	const sh = 0
	tc := newPBTestCluster(t, 5, numShards, 1, func(c *Config) {
		c.PBAutoFailover = true
		c.MinISR = 1
		c.PBLeaseTTLMs = 1000
		c.PBMetaContactStalenessMs = 500
		c.PBFailoverTimeoutMs = 3000
		c.PBRenewIntervalMs = 300
		c.ShardCfg.RaftHeartbeatMs = 200
	})

	findNodeIdx := func(nodeID string) int {
		for i, p := range tc.peers {
			if p.NodeID == nodeID {
				return i
			}
		}
		return -1
	}
	// liveIdx is any node still running, used to read the replicated FSM.
	dead := map[int]bool{}
	liveIdx := func() int {
		for i := range tc.nodes {
			if !dead[i] {
				return i
			}
		}
		return -1
	}
	// waitPrimary polls the replicated FSM for a primary at an epoch > minEpoch
	// that is not any already-killed node.
	waitPrimary := func(minEpoch uint64, within time.Duration) (string, uint64) {
		t.Helper()
		deadline := time.Now().Add(within)
		for time.Now().Before(deadline) {
			li := liveIdx()
			ep := tc.nodes[li].meta.FSM.ShardEpoch(sh)
			pr := tc.nodes[li].meta.FSM.ShardPrimary(sh)
			if pr != "" && ep > minEpoch {
				if idx := findNodeIdx(pr); idx >= 0 && !dead[idx] {
					return pr, ep
				}
			}
			time.Sleep(50 * time.Millisecond)
		}
		return "", 0
	}
	writeBatch := func(idx int, prefix string, n int) map[string][]byte {
		t.Helper()
		out := make(map[string][]byte, n)
		for i := 0; i < n; i++ {
			key := []byte(fmt.Sprintf("%s-%03d", prefix, i))
			val := []byte(fmt.Sprintf("%sval-%03d", prefix, i))
			if _, err := tc.nodes[idx].Call("put", ops.EncodePutArgs(key, val, 0)); err != nil {
				t.Fatalf("%s put %d failed: %v", prefix, i, err)
			}
			out[string(key)] = val
		}
		return out
	}

	// --- Bring up and find the original primary. ---
	var p1Idx = -1
	var p1 string
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) && p1Idx < 0 {
		primary := tc.nodes[0].meta.FSM.ShardPrimary(sh)
		if primary == "" {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		idx := findNodeIdx(primary)
		if idx < 0 {
			t.Fatalf("primary %q not in peer list", primary)
		}
		if _, err := tc.nodes[idx].Call("put", ops.EncodePutArgs([]byte("probe"), []byte("v"), 0)); err != nil {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		p1Idx, p1 = idx, primary
	}
	if p1Idx < 0 {
		t.Fatal("PB primary never accepted a write within 20s")
	}
	writeBatch(p1Idx, "batch1", 10)
	time.Sleep(700 * time.Millisecond) // let the beacon establish liveness

	// --- Kill P, wait for Q. ---
	if err := tc.nodes[p1Idx].Close(); err != nil {
		t.Logf("primary Close returned: %v (tolerated)", err)
	}
	dead[p1Idx] = true
	p2, ep2 := waitPrimary(1, 30*time.Second)
	if p2 == "" {
		t.Fatal("no promotion after killing the original primary")
	}
	p2Idx := findNodeIdx(p2)
	t.Logf("failover 1: %q -> %q (epoch %d)", p1, p2, ep2)

	// Wait only for Q's engine to adopt the promotion, then write IMMEDIATELY —
	// the whole point is to sample the window before the grow driver re-widens.
	eng := tc.nodes[p2Idx].pbEngines[sh]
	engDeadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(engDeadline) && eng.Epoch() < ep2 {
		time.Sleep(10 * time.Millisecond)
	}
	isrAtPromotion := append([]string(nil), tc.nodes[p2Idx].meta.FSM.ShardISR(sh)...)
	sort.Strings(isrAtPromotion)

	batch2 := writeBatch(p2Idx, "batch2", 10)
	st := eng.ReplicationStatus()
	isrAfterWrites := append([]string(nil), tc.nodes[p2Idx].meta.FSM.ShardISR(sh)...)
	sort.Strings(isrAfterWrites)

	t.Logf("ANSWER — new primary %q at epoch %d: committed ISR at promotion=%v, after batch2=%v; "+
		"engine LastSeq=%d Committed=%d Peers=%+v",
		p2, ep2, isrAtPromotion, isrAfterWrites, st.LastSeq, st.Committed, st.Peers)
	if len(st.Peers) == 0 {
		t.Logf("ANSWER: YES — the freshly promoted primary acked all %d batch2 writes with an EMPTY peer set "+
			"(resident on itself alone). This is the post-failover singleton-ISR window.", len(batch2))
	} else {
		t.Logf("ANSWER: NO — the freshly promoted primary already replicated to %d peer(s) for batch2.", len(st.Peers))
	}

	// --- Kill Q, wait for R, and see whether batch2 survived. ---
	if err := tc.nodes[p2Idx].Close(); err != nil {
		t.Logf("second primary Close returned: %v (tolerated)", err)
	}
	dead[p2Idx] = true

	// The committed ISR at this instant decides what CAN happen next:
	// decidePBPromotions draws candidates ONLY from the ISR, excluding the dead
	// primary. A singleton ISR of {Q} therefore leaves NO candidate at all.
	isrAtSecondKill := append([]string(nil), tc.nodes[liveIdx()].meta.FSM.ShardISR(sh)...)
	sort.Strings(isrAtSecondKill)
	if len(isrAtSecondKill) == 1 && isrAtSecondKill[0] == p2 {
		t.Logf("ANSWER — consequence: at the second kill the committed ISR is still the singleton %v. "+
			"decidePBPromotions draws candidates only from the ISR minus the dead primary, so there is NO "+
			"promotable member: the shard goes UNAVAILABLE (fail-stop) and batch2 stays pinned to the dead "+
			"node's disk. That is materially different from the bootstrap defect, which SILENTLY promoted a "+
			"survivor and dropped the writes.", isrAtSecondKill)
	} else {
		t.Logf("ANSWER — consequence: at the second kill the committed ISR was %v (the grow driver had already "+
			"re-widened), so a promotion is possible and batch2 survival is a real question.", isrAtSecondKill)
	}

	// Bounded short: when the ISR is a singleton no promotion is possible by
	// construction, so a long wait would only pad the runtime.
	p3, ep3 := waitPrimary(ep2, 10*time.Second)
	if p3 == "" {
		t.Log("no second promotion within 10s — the shard stayed down, as the singleton ISR implies")
		return
	}
	p3Idx := findNodeIdx(p3)
	t.Logf("failover 2: %q -> %q (epoch %d)", p2, p3, ep3)

	eng3 := tc.nodes[p3Idx].pbEngines[sh]
	d3 := time.Now().Add(10 * time.Second)
	for time.Now().Before(d3) && eng3.Epoch() < ep3 {
		time.Sleep(50 * time.Millisecond)
	}
	store := tc.nodes[p3Idx].getShard(sh)
	missing := 0
	for key, want := range batch2 {
		got, err := store.Get([]byte(key))
		if err != nil || string(got) != string(want) {
			missing++
		}
	}
	t.Logf("ANSWER — batch2 survival on the THIRD primary %q: %d of %d acked keys MISSING",
		p3, missing, len(batch2))
}

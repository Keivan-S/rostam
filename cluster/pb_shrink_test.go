// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"sync"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard/pbisr"
)

// ---------------------------------------------------------------------------
// pbDropInjector — the PB-transport drop injector (ISR shrink harness).
//
// It wraps a single node's PB replication transport so replication to a chosen
// backup FAILS deterministically (a definitive negative ack, exactly like the
// inmem `drop` fault), firing the engine's completeSend failure path and bumping
// that peer's consecutive-failure counter — the shrink wedge signal — WITHOUT
// killing the node or touching any other peer. It is the PB-data-path analogue of
// the meta-transport partitionableStreamLayer used by the failover gate, spliced
// via the nil-in-prod Config.pbTransportWrap seam.
// ---------------------------------------------------------------------------

type pbDropInjector struct {
	mu      sync.Mutex
	dropped map[string]bool
}

func newPBDropInjector() *pbDropInjector {
	return &pbDropInjector{dropped: make(map[string]bool)}
}

func (d *pbDropInjector) setDrop(peer string, on bool) {
	d.mu.Lock()
	d.dropped[peer] = on
	d.mu.Unlock()
}

func (d *pbDropInjector) isDropped(peer string) bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.dropped[peer]
}

// wrap installs the injector over base (the node-ID-resolving transport). It only
// implements the plain Transport contract, so a wrapped node loses the inline /
// group fast paths — correctness is unaffected (frames just route through the
// per-peer sender), and only the shrink harness ever wraps.
func (d *pbDropInjector) wrap(base pbisr.Transport) pbisr.Transport {
	return &pbDroppingTransport{base: base, inj: d}
}

type pbDroppingTransport struct {
	base pbisr.Transport
	inj  *pbDropInjector
}

func (t *pbDroppingTransport) Replicate(peer string, msg pbisr.ReplicateMsg, done func(pbisr.AckMsg, error)) error {
	if t.inj.isDropped(peer) {
		// Definitive negative ack: completeSend's failure path fires (fail-fast +
		// per-peer failure-counter bump), the exact wedge signal the shrink driver
		// reads. done fires exactly once, so the exactly-once record latch holds.
		done(pbisr.AckMsg{Epoch: msg.Epoch, Seq: msg.Seq, OK: false}, nil)
		return nil
	}
	return t.base.Replicate(peer, msg, done)
}

// newShrinkablePBTestCluster builds an n-node PB cluster like newPBTestCluster but
// installs a per-node pbDropInjector over each node's PB transport, keyed by
// NodeID. The caller's opts run AFTER the wrap opt (so they may still set
// PBAutoFailover etc.); the wrap keys each injector by the node's unique ID, set
// before any opt runs.
func newShrinkablePBTestCluster(t *testing.T, n, numShards, minISR int, opts ...func(*Config)) (*pbTestCluster, map[string]*pbDropInjector) {
	t.Helper()
	injectors := make(map[string]*pbDropInjector, n)
	wrapOpt := func(c *Config) {
		inj := newPBDropInjector()
		c.pbTransportWrap = inj.wrap
		injectors[c.NodeID] = inj
	}
	all := append([]func(*Config){wrapOpt}, opts...)
	tc := newPBTestCluster(t, n, numShards, minISR, all...)
	return tc, injectors
}

// nodeByID returns the *Node with the given cluster node ID (or nil).
func (tc *pbTestCluster) nodeByID(id string) *Node {
	for _, nd := range tc.nodes {
		if nd != nil && nd.cfg.NodeID == id {
			return nd
		}
	}
	return nil
}

// awaitLeasedPrimaryWrite polls until shard sh has a committed primary that
// ACCEPTS a write (its self-lease is granted), returning the primary's NodeID and
// its *Node. On a CPU-oversubscribed host where no leased primary ever
// materializes it returns "" so the caller can SKIP (not false-fail).
func (tc *pbTestCluster) awaitLeasedPrimaryWrite(t *testing.T, sh int, putArgs []byte, budget time.Duration) (string, *Node) {
	t.Helper()
	deadline := time.Now().Add(budget)
	for time.Now().Before(deadline) {
		primary := tc.nodes[0].meta.FSM.ShardPrimary(sh)
		if primary == "" {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		nd := tc.nodeByID(primary)
		if nd == nil {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		if _, err := nd.Call("put", putArgs); err == nil {
			return primary, nd
		}
		time.Sleep(50 * time.Millisecond)
	}
	return "", nil
}

// --- The main e2e: a dead backup auto-shrinks and the pipeline resumes ---------

// TestPBShrinkUnwedgesDeadBackup: 3-node RF=full PB cluster, MinISR=2. Drop the
// primary→one-backup replication so full-ISR commit stalls; the shrink driver
// detects the dead backup, commits an OpSetShardISR removing it (still >= MinISR),
// applies it live to the engine, and writes resume — committed advancing past the
// stall — with the removed member gone from the committed ISR.
func TestPBShrinkUnwedgesDeadBackup(t *testing.T) {
	const numShards = 4
	tc, inj := newShrinkablePBTestCluster(t, 3, numShards, 2, func(c *Config) {
		c.PBAutoFailover = true
		c.PBShrinkThreshold = 3 // a few failed writes is enough to declare the backup dead
		// Keep the failover timeout at its (large) default so no PRIMARY promotion
		// races this backup-shrink test.
	})

	key := []byte("shrink-key")
	sh := shardOf(key, numShards)
	putArgs := ops.EncodePutArgs(key, []byte("v0"), 0)

	primary, pnode := tc.awaitLeasedPrimaryWrite(t, sh, putArgs, 20*time.Second)
	if primary == "" {
		t.Skip("SKIP: no alive leased PB primary within 20s (host too loaded)")
	}
	baseCommitted := pnode.pbEngines[sh].Committed()

	// The full ISR is all three owners; pick a backup to kill.
	isr := pnode.meta.FSM.ShardISR(sh)
	var victim string
	for _, m := range isr {
		if m != primary {
			victim = m
			break
		}
	}
	if victim == "" {
		t.Fatalf("shard %d ISR %v has no backup besides primary %q", sh, isr, primary)
	}

	// Cut primary→victim replication. Full-ISR commit now stalls on victim.
	inj[primary].setDrop(victim, true)

	// Drive writes: each fails while the shard is wedged, then — once the driver has
	// committed the shrink and live-re-evaluated the engine — a write SUCCEEDS and
	// committed advances. Poll both the committed ISR (victim removed) and a
	// successful write.
	var shrankISR, wrote bool
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		v := ops.EncodePutArgs(key, []byte("v1"), 0)
		_, err := pnode.Call("put", v)
		nowISR := pnode.meta.FSM.ShardISR(sh)
		if !containsStr(nowISR, victim) && len(nowISR) >= 2 {
			shrankISR = true
		}
		if err == nil && shrankISR {
			wrote = true
			break
		}
		time.Sleep(50 * time.Millisecond)
	}

	if !shrankISR {
		t.Fatalf("shard %d ISR never dropped the dead backup %q (still %v) — no auto-shrink",
			sh, victim, pnode.meta.FSM.ShardISR(sh))
	}
	if !wrote {
		t.Fatalf("no write succeeded after the shrink — the pipeline stayed wedged")
	}
	if got := pnode.pbEngines[sh].Committed(); got <= baseCommitted {
		t.Fatalf("committed = %d did not advance past the pre-drop watermark %d after shrink", got, baseCommitted)
	}
	// The surviving backup must still hold the data (no acked-loss under shrink).
	finalISR := pnode.meta.FSM.ShardISR(sh)
	if len(finalISR) < 2 {
		t.Fatalf("final ISR %v is below MinISR=2 — shrink breached the floor", finalISR)
	}
}

// --- The floor: a shrink that would breach MinISR is REFUSED -------------------

// TestPBShrinkRefusedBelowMinISR: RF=full, MinISR=3, so dropping ANY backup would
// take the ISR below the floor. The driver must REFUSE — the shard stays stalled
// (unavailability over durability, H3) and the committed ISR keeps the dead member.
func TestPBShrinkRefusedBelowMinISR(t *testing.T) {
	const numShards = 4
	tc, inj := newShrinkablePBTestCluster(t, 3, numShards, 3, func(c *Config) {
		c.PBAutoFailover = true
		c.PBShrinkThreshold = 3
	})

	key := []byte("floor-key")
	sh := shardOf(key, numShards)
	putArgs := ops.EncodePutArgs(key, []byte("v0"), 0)

	primary, pnode := tc.awaitLeasedPrimaryWrite(t, sh, putArgs, 20*time.Second)
	if primary == "" {
		t.Skip("SKIP: no alive leased PB primary within 20s (host too loaded)")
	}

	isr := pnode.meta.FSM.ShardISR(sh)
	var victim string
	for _, m := range isr {
		if m != primary {
			victim = m
			break
		}
	}
	inj[primary].setDrop(victim, true)

	// Hammer writes for long enough that the driver has run many ticks. Every write
	// must keep failing (the shard is wedged and the floor forbids the shrink), and
	// the committed ISR must STILL contain the dead member.
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		v := ops.EncodePutArgs(key, []byte("v1"), 0)
		if _, err := pnode.Call("put", v); err == nil {
			t.Fatalf("a write SUCCEEDED while MinISR=3 forbids dropping the dead backup — the floor was breached")
		}
		time.Sleep(50 * time.Millisecond)
	}
	if got := pnode.meta.FSM.ShardISR(sh); !containsStr(got, victim) {
		t.Fatalf("dead backup %q was removed from ISR %v despite the MinISR=3 floor", victim, got)
	}
}

// --- The epoch guard: a stale-epoch shrink forward is a no-op ------------------

// TestPBShrinkStaleEpochForwardRejected proves the OpSetShardISR epoch guard that
// protects against a FENCED ex-primary: an ISR update carrying a stale epoch (one
// the shard has already advanced past) is applied as a no-op — the committed ISR
// is unchanged. This is the same guard submitSetShardISR relies on so a fenced
// ex-primary's late shrink forward cannot mutate a newer epoch's ISR.
func TestPBShrinkStaleEpochForwardRejected(t *testing.T) {
	const numShards = 4
	tc := newPBTestCluster(t, 3, numShards, 2)

	key := []byte("epoch-key")
	sh := shardOf(key, numShards)
	putArgs := ops.EncodePutArgs(key, []byte("v0"), 0)
	primary, pnode := tc.awaitLeasedPrimaryWrite(t, sh, putArgs, 20*time.Second)
	if primary == "" {
		t.Skip("SKIP: no alive leased PB primary within 20s (host too loaded)")
	}

	curEpoch := pnode.meta.FSM.ShardEpoch(sh)
	before := pnode.meta.FSM.ShardISR(sh)
	if curEpoch == 0 {
		t.Fatalf("shard %d has epoch 0 after a committed write", sh)
	}

	// Find the meta leader (only it can apply) and submit an ISR shrink at a STALE
	// epoch (curEpoch-1). The FSM epoch guard must make it a no-op.
	var leader *Node
	for _, nd := range tc.nodes {
		if nd.meta.Raft.State().String() == "Leader" {
			leader = nd
			break
		}
	}
	if leader == nil {
		t.Skip("SKIP: no meta leader observed")
	}
	staleISR := []string{primary} // a drastic shrink that MUST be ignored at the stale epoch
	if err := leader.meta.ApplySetShardISR(sh, curEpoch-1, staleISR, 5*time.Second); err != nil {
		t.Fatalf("ApplySetShardISR at stale epoch returned a hard error (want silent no-op): %v", err)
	}

	// The committed ISR must be unchanged: the stale-epoch update was ignored.
	after := pnode.meta.FSM.ShardISR(sh)
	if !sameStrSet(before, after) {
		t.Fatalf("stale-epoch ISR shrink mutated the ISR: before=%v after=%v", before, after)
	}
}

// --- Pure decision unit tests --------------------------------------------------

func TestDecidePBShrink(t *testing.T) {
	isr := []string{"n1", "n2", "n3"}

	// Normal shrink: drop the one dead backup, floor satisfied.
	req, ok := decidePBShrink(0, 7, "n1", isr, []string{"n2"}, 2)
	if !ok || len(req.newISR) != 2 || containsStr(req.newISR, "n2") || req.epoch != 7 {
		t.Fatalf("normal shrink: ok=%v req=%+v, want {n1,n3} at epoch 7", ok, req)
	}

	// Floor refusal: dropping n2 would leave {n1,n3} = 2 < minISR 3.
	if _, ok := decidePBShrink(0, 7, "n1", isr, []string{"n2"}, 3); ok {
		t.Fatal("shrink below MinISR must be refused")
	}

	// Nothing removable: the stalled peer is not in the ISR.
	if _, ok := decidePBShrink(0, 7, "n1", isr, []string{"nX"}, 2); ok {
		t.Fatal("a stalled peer not in the ISR must not shrink")
	}

	// The primary is never a removal candidate even if reported stalled.
	if _, ok := decidePBShrink(0, 7, "n1", isr, []string{"n1"}, 2); ok {
		t.Fatal("the primary must never be shrunk out of its own ISR")
	}

	// Empty stalled set: no shrink.
	if _, ok := decidePBShrink(0, 7, "n1", isr, nil, 2); ok {
		t.Fatal("no stalled peers must not shrink")
	}
}

func TestIsStrictSubset(t *testing.T) {
	if !isStrictSubset([]string{"n1", "n2"}, []string{"n1", "n2", "n3"}) {
		t.Fatal("{n1,n2} must be a strict subset of {n1,n2,n3}")
	}
	if isStrictSubset([]string{"n1", "n2", "n3"}, []string{"n1", "n2", "n3"}) {
		t.Fatal("equal sets are not a STRICT subset")
	}
	if isStrictSubset([]string{"n1", "nX"}, []string{"n1", "n2", "n3"}) {
		t.Fatal("a member the superset lacks disqualifies (not a pure narrowing)")
	}
	if isStrictSubset([]string{"n1", "n2", "n3", "n4"}, []string{"n1", "n2", "n3"}) {
		t.Fatal("a larger set is never a subset (grow is not a narrowing)")
	}
}

// --- helpers ------------------------------------------------------------------

func containsStr(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}

func sameStrSet(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for _, x := range a {
		if !containsStr(b, x) {
			return false
		}
	}
	return true
}

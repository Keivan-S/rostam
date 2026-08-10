// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"testing"
	"time"
)

// TestCatalogGenOpReportsLocalGen exercises the __catalog_gen__ admin op
// directly: a node's reply must mirror its LOCAL CollectionPartitionsGen — the
// committed (P, gen, OK) for a known collection, and OK=false for an unknown one.
func TestCatalogGenOpReportsLocalGen(t *testing.T) {
	c := newTestCluster(t, 3, 8)
	defer c.Close()

	leader := metaLeaderNode(t, c)
	if err := leader.SetCollectionPartitions("default/docs", 6, 3, 5*time.Second); err != nil {
		t.Fatalf("SetCollectionPartitions: %v", err)
	}
	// Let every node converge so the direct handler call below is deterministic.
	for i, n := range c.nodes {
		if err := waitCatalogGen(n, "default/docs", 6, 3, 5*time.Second); err != nil {
			t.Fatalf("node %d converge: %v", i, err)
		}
	}

	// Known collection: handler returns the committed gen with OK=true.
	args, err := gobEncode(catalogGenReq{Collection: "default/docs"})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := leader.handleCatalogGen(args)
	if err != nil {
		t.Fatalf("handleCatalogGen: %v", err)
	}
	var reply catalogGenReply
	if err := gobDecode(raw, &reply); err != nil {
		t.Fatal(err)
	}
	if !reply.OK || reply.P != 6 || reply.Gen != 3 {
		t.Fatalf("__catalog_gen__ known = (P=%d,Gen=%d,OK=%v), want (6,3,true)", reply.P, reply.Gen, reply.OK)
	}

	// Unknown collection: OK=false, P=Gen=0.
	uargs, err := gobEncode(catalogGenReq{Collection: "default/nope"})
	if err != nil {
		t.Fatal(err)
	}
	uraw, err := leader.handleCatalogGen(uargs)
	if err != nil {
		t.Fatalf("handleCatalogGen unknown: %v", err)
	}
	var ureply catalogGenReply
	if err := gobDecode(uraw, &ureply); err != nil {
		t.Fatal(err)
	}
	if ureply.OK || ureply.P != 0 || ureply.Gen != 0 {
		t.Fatalf("__catalog_gen__ unknown = (P=%d,Gen=%d,OK=%v), want (0,0,false)", ureply.P, ureply.Gen, ureply.OK)
	}
}

// TestWaitAllNodesCatalogGenConverges is the normal case: once every node's
// meta-FSM has applied a catalog gen, the gate resolves promptly (nil).
func TestWaitAllNodesCatalogGenConverges(t *testing.T) {
	c := newTestCluster(t, 3, 8)
	defer c.Close()

	leader := metaLeaderNode(t, c)
	if err := leader.SetCollectionPartitions("default/docs", 6, 2, 5*time.Second); err != nil {
		t.Fatalf("SetCollectionPartitions: %v", err)
	}
	// Convergence helper (existing) gates the gate: once every node shows gen 2,
	// waitAllNodesCatalogGen must return nil. We call it from the leader (the
	// coordinator role) but it polls EVERY node including itself.
	for i, n := range c.nodes {
		if err := waitCatalogGen(n, "default/docs", 6, 2, 5*time.Second); err != nil {
			t.Fatalf("node %d converge: %v", i, err)
		}
	}
	if err := leader.waitAllNodesCatalogGen("default/docs", 2, 5*time.Second); err != nil {
		t.Fatalf("waitAllNodesCatalogGen after convergence: %v", err)
	}
	// Canonicalization: passing the BARE name must hit the same catalog key.
	if err := leader.waitAllNodesCatalogGen("docs", 2, 5*time.Second); err != nil {
		t.Fatalf("waitAllNodesCatalogGen(bare name): %v", err)
	}
}

// TestWaitAllNodesCatalogGenWaitsForConvergence asserts the gate does NOT return
// before every node has actually applied the wanted gen. We start the gate
// immediately after committing on the leader (followers may still be replaying)
// and require that, whenever it returns nil, the cluster has in fact converged.
// Run concurrently so a premature return (a bug) would be observable: the gate
// must not resolve until the slowest node reports the gen.
func TestWaitAllNodesCatalogGenWaitsForConvergence(t *testing.T) {
	c := newTestCluster(t, 3, 8)
	defer c.Close()

	leader := metaLeaderNode(t, c)
	if err := leader.SetCollectionPartitions("default/docs", 6, 5, 5*time.Second); err != nil {
		t.Fatalf("SetCollectionPartitions: %v", err)
	}

	// The gate must block until ALL nodes report gen 5. When it returns nil,
	// re-read every node's local gen with NO further wait: each must already show
	// gen 5 (the gate guarantees it). A premature nil would catch a lagging node.
	if err := leader.waitAllNodesCatalogGen("default/docs", 5, 5*time.Second); err != nil {
		t.Fatalf("waitAllNodesCatalogGen: %v", err)
	}
	for i, n := range c.nodes {
		p, gen, ok := n.CollectionPartitionsGen("default/docs")
		if !ok || gen != 5 {
			t.Fatalf("gate returned nil but node %d local gen = (P=%d,Gen=%d,OK=%v), want gen 5 — gate resolved before convergence", i, p, gen, ok)
		}
	}
}

// TestWaitAllNodesCatalogGenTimesOutOnUnreachable: with one node's server
// stopped (so its __catalog_gen__ peer call always errors), the gate must NOT
// hang — it times out and the ErrCutoverGateTimeout names the unreachable node.
func TestWaitAllNodesCatalogGenTimesOutOnUnreachable(t *testing.T) {
	c := newTestCluster(t, 3, 8)
	defer c.Close()

	leader := metaLeaderNode(t, c)
	if err := leader.SetCollectionPartitions("default/docs", 6, 1, 5*time.Second); err != nil {
		t.Fatalf("SetCollectionPartitions: %v", err)
	}
	for i, n := range c.nodes {
		if err := waitCatalogGen(n, "default/docs", 6, 1, 5*time.Second); err != nil {
			t.Fatalf("node %d converge: %v", i, err)
		}
	}

	// Pick a victim node that is NOT the coordinator (so the local read still
	// confirms the coordinator itself). Stop its server so peer calls to it fail.
	var victimIdx = -1
	for i, n := range c.nodes {
		if n != leader {
			victimIdx = i
			break
		}
	}
	if victimIdx < 0 {
		t.Fatal("no non-leader node to stop")
	}
	victimID := c.peers[victimIdx].NodeID
	if c.servers[victimIdx] != nil {
		_ = c.servers[victimIdx].Close()
		c.servers[victimIdx] = nil
	}
	if c.nodes[victimIdx] != nil {
		_ = c.nodes[victimIdx].Close()
		c.nodes[victimIdx] = nil
	}

	// The gate must time out (not hang) and name the unreachable node.
	start := time.Now()
	err := leader.waitAllNodesCatalogGen("default/docs", 1, 600*time.Millisecond)
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("waitAllNodesCatalogGen returned nil despite an unreachable node")
	}
	var gateErr *ErrCutoverGateTimeout
	if !errors.As(err, &gateErr) {
		t.Fatalf("error = %v, want *ErrCutoverGateTimeout", err)
	}
	if elapsed > 3*time.Second {
		t.Fatalf("gate took %s — should be bounded by the ~600ms timeout, not hang", elapsed)
	}
	found := false
	for _, id := range gateErr.Unconfirmed {
		if id == victimID {
			found = true
		}
	}
	if !found {
		t.Fatalf("ErrCutoverGateTimeout.Unconfirmed = %v, want it to name the stopped node %q", gateErr.Unconfirmed, victimID)
	}
}

// TestWaitAllNodesCatalogGenSingleNode: a single-node cluster (len(Peers)<=1) is
// trivially satisfied — the local node IS all nodes — so the gate returns nil
// immediately, with no peers to poll, even for a collection that has no catalog
// entry at all.
func TestWaitAllNodesCatalogGenSingleNode(t *testing.T) {
	c := newTestCluster(t, 1, 8)
	defer c.Close()

	n := c.nodes[0]
	start := time.Now()
	if err := n.waitAllNodesCatalogGen("default/docs", 7, 2*time.Second); err != nil {
		t.Fatalf("single-node gate should be trivially satisfied, got %v", err)
	}
	if elapsed := time.Since(start); elapsed > 200*time.Millisecond {
		t.Fatalf("single-node gate took %s — should return immediately", elapsed)
	}
}

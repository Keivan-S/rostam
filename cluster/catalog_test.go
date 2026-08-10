// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"fmt"
	"testing"
	"time"

	hraft "github.com/hashicorp/raft"
)

func TestNodeCatalogReadWriteOnLeader(t *testing.T) {
	c := newTestCluster(t, 3, 8)
	defer c.Close()

	leader := metaLeaderNode(t, c)
	if err := leader.SetCollectionPartitions("default/docs", 6, 0, 5*time.Second); err != nil {
		t.Fatalf("SetCollectionPartitions on leader: %v", err)
	}
	for i, n := range c.nodes {
		if err := waitCatalog(n, "default/docs", 6, 5*time.Second); err != nil {
			t.Fatalf("node %d: %v", i, err)
		}
	}
}

func TestNodeCatalogForwardsFromNonLeader(t *testing.T) {
	c := newTestCluster(t, 3, 8)
	defer c.Close()

	// Pick a node whose meta-Raft is NOT the leader.
	var follower *Node
	leader := metaLeaderNode(t, c)
	for _, n := range c.nodes {
		if n != leader && n.meta != nil {
			follower = n
			break
		}
	}
	if follower == nil {
		t.Fatal("no non-leader meta node found")
	}

	if err := follower.SetCollectionPartitions("default/docs", 6, 0, 5*time.Second); err != nil {
		t.Fatalf("SetCollectionPartitions on follower (should forward): %v", err)
	}
	for i, n := range c.nodes {
		if err := waitCatalog(n, "default/docs", 6, 5*time.Second); err != nil {
			t.Fatalf("node %d: %v", i, err)
		}
	}
}

func TestSetCollectionPartitionsIsReadYourWrites(t *testing.T) {
	c := newTestCluster(t, 3, 8)
	defer c.Close()

	leader := metaLeaderNode(t, c)
	var follower *Node
	for _, n := range c.nodes {
		if n != leader && n.meta != nil {
			follower = n
			break
		}
	}
	if follower == nil {
		t.Fatal("no follower meta node")
	}
	// Write through the FOLLOWER (forwarded path). On return, the follower's OWN
	// local read must already reflect the value (read-your-writes), with NO poll.
	if err := follower.SetCollectionPartitions("default/docs", 6, 0, 5*time.Second); err != nil {
		t.Fatalf("SetCollectionPartitions: %v", err)
	}
	if p, ok := follower.CollectionPartitions("default/docs"); !ok || p != 6 {
		t.Fatalf("read-your-writes violated on issuing follower: got (%d,%v), want (6,true)", p, ok)
	}
}

func TestSetCollectionReshardReadYourWrites(t *testing.T) {
	c := newTestCluster(t, 3, 8)
	defer c.Close()

	leader := metaLeaderNode(t, c)
	var follower *Node
	for _, n := range c.nodes {
		if n != leader && n.meta != nil {
			follower = n
			break
		}
	}
	if follower == nil {
		t.Fatal("no follower meta node")
	}
	// Write reshard state through the FOLLOWER (forwarded path), INCLUDING the
	// source (old) gen pin. On return, the follower's OWN local read must already
	// reflect it (read-your-writes), no poll — and the Source pin must survive the
	// meta-Raft log + FSM round-trip so a lagging follower's dualTargets can still
	// write the old gen after the cutover (the linearizable-catalog fix).
	want := ReshardEntry{Status: 1, TargetP: 4, TargetGen: 1, SourceP: 2, SourceGen: 0}
	if err := follower.SetCollectionReshard("default/docs", want, 5*time.Second); err != nil {
		t.Fatalf("SetCollectionReshard: %v", err)
	}
	got, ok := follower.CollectionReshard("default/docs")
	if !ok || got != want {
		t.Fatalf("read-your-writes violated on issuing follower: got (%+v,%v), want (%+v,true)", got, ok, want)
	}
	// The leader's local FSM must carry the Source pin too (it applied the entry).
	if lgot, lok := leader.CollectionReshard("default/docs"); !lok || lgot != want {
		t.Fatalf("leader reshard read = (%+v,%v), want (%+v,true) — Source pin lost in meta round-trip", lgot, lok, want)
	}
	// Clearing to Stable (Status 0) must also be read-your-writes.
	if err := follower.SetCollectionReshard("default/docs", ReshardEntry{}, 5*time.Second); err != nil {
		t.Fatalf("SetCollectionReshard clear: %v", err)
	}
	if got, ok := follower.CollectionReshard("default/docs"); ok && got.Status != 0 {
		t.Fatalf("after clear got (%+v,%v), want Stable", got, ok)
	}
}

// TestSetAliasesReadYourWrites mirrors TestSetCollectionReshardReadYourWrites:
// setting aliases through a FOLLOWER (forwarded to the leader) must, on return,
// already be visible to that follower's OWN local ResolveAlias (read-your-writes),
// with no poll. A delete must likewise be immediately visible as a miss.
func TestSetAliasesReadYourWrites(t *testing.T) {
	c := newTestCluster(t, 3, 8)
	defer c.Close()

	leader := metaLeaderNode(t, c)
	var follower *Node
	for _, n := range c.nodes {
		if n != leader && n.meta != nil {
			follower = n
			break
		}
	}
	if follower == nil {
		t.Fatal("no follower meta node")
	}
	// Create an alias through the FOLLOWER (forwarded path). On return the
	// follower's OWN local read must already reflect the SPECIFIC written value.
	if err := follower.SetAliases([]AliasAction{
		{Alias: "prod", Canonical: "default/coll_v1"},
	}, 5*time.Second); err != nil {
		t.Fatalf("SetAliases create: %v", err)
	}
	if got, ok := follower.ResolveAlias("prod"); !ok || got != "default/coll_v1" {
		t.Fatalf("read-your-writes violated on issuing follower: got (%q,%v), want (default/coll_v1,true)", got, ok)
	}
	// Atomic swap through the follower: delete + recreate to a new target in one
	// batch. ResolveAlias must show the NEW target, never undefined.
	if err := follower.SetAliases([]AliasAction{
		{Alias: "prod", Delete: true},
		{Alias: "prod", Canonical: "default/coll_v2"},
	}, 5*time.Second); err != nil {
		t.Fatalf("SetAliases swap: %v", err)
	}
	if got, ok := follower.ResolveAlias("prod"); !ok || got != "default/coll_v2" {
		t.Fatalf("after swap got (%q,%v), want (default/coll_v2,true)", got, ok)
	}
	// A delete must be read-your-writes as a miss.
	if err := follower.SetAliases([]AliasAction{{Alias: "prod", Delete: true}}, 5*time.Second); err != nil {
		t.Fatalf("SetAliases delete: %v", err)
	}
	if got, ok := follower.ResolveAlias("prod"); ok {
		t.Fatalf("after delete got (%q,%v), want miss", got, ok)
	}
}

func TestNodeCatalogGenerationConverges(t *testing.T) {
	c := newTestCluster(t, 3, 8)
	defer c.Close()

	leader := metaLeaderNode(t, c)
	if err := leader.SetCollectionPartitions("default/docs", 6, 2, 5*time.Second); err != nil {
		t.Fatalf("SetCollectionPartitions on leader: %v", err)
	}
	for i, n := range c.nodes {
		if err := waitCatalogGen(n, "default/docs", 6, 2, 5*time.Second); err != nil {
			t.Fatalf("node %d: %v", i, err)
		}
	}
}

func waitCatalogGen(n *Node, collection string, wantP, wantGen uint32, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if p, gen, ok := n.CollectionPartitionsGen(collection); ok && p == wantP && gen == wantGen {
			return nil
		}
		time.Sleep(20 * time.Millisecond)
	}
	p, gen, ok := n.CollectionPartitionsGen(collection)
	return fmt.Errorf("catalog %q = (%d,%d,%v), want (%d,%d)", collection, p, gen, ok, wantP, wantGen)
}

func waitCatalog(n *Node, collection string, want uint32, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if p, ok := n.CollectionPartitions(collection); ok && p == want {
			return nil
		}
		time.Sleep(20 * time.Millisecond)
	}
	got, ok := n.CollectionPartitions(collection)
	return fmt.Errorf("catalog %q = (%d,%v), want %d", collection, got, ok, want)
}

// metaLeaderNode returns the node whose meta-Raft is currently leader.
func metaLeaderNode(t *testing.T, c *testCluster) *Node {
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		for _, n := range c.nodes {
			if n.meta != nil && n.meta.Raft.State() == hraft.Leader {
				return n
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("no meta leader elected")
	return nil
}

// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"fmt"
	"net"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/server"
	"github.com/rostamlabs/rostam/shard"
)

// pbTestCluster owns an N-node in-process primary-backup (ReplicationMode="pb")
// cluster: N cluster.Node instances sharing a static peer list. The test drives
// node.Call directly on the shard's primary and reads back through the local
// stores/engines. It binds a real per-node PB transport listener (PBAddr) so the
// primary can replicate to its ISR backups over TCP, AND a real client-facing
// server (ServerAddr) so inter-node forwarding (a follower-hosted primary's Plan-4
// beacon forwarding its OpShardLeaseRenew to the meta leader) resolves.
type pbTestCluster struct {
	nodes   []*Node
	servers []*server.Server // per-node client-facing server (so beacon forwards resolve)
	peers   []Peer

	// metaParts maps NodeID → that node's meta-Raft partition injector, populated
	// only by newPartitionablePBTestCluster. nil for the plain harness. A test uses
	// it to cut/heal exactly one node's META transport (see pb_partition_test.go).
	metaParts map[string]*partitionableStreamLayer
}

// newPBTestCluster spins up an n-node PB cluster with numShards shards under full
// replication (RF=0 ⇒ every node owns every shard) and the given MinISR. It
// pre-binds each node's Raft and PB listeners (so Peers carries final addrs before
// construction), closes-then-rebinds each just before its owner claims it, and
// waits for every node to observe a meta-Raft leader. Cleanup is registered with t.
func newPBTestCluster(t *testing.T, n, numShards, minISR int, opts ...func(*Config)) *pbTestCluster {
	t.Helper()

	makeReg := func() *ops.Registry {
		r := ops.NewRegistry()
		if err := ops.RegisterBuiltins(r); err != nil {
			t.Fatal(err)
		}
		return r
	}

	raftAddrs := make([]string, n)
	pbAddrs := make([]string, n)
	serverAddrs := make([]string, n)
	raftLn := make([]net.Listener, n)
	pbLn := make([]net.Listener, n)
	serverLn := make([]net.Listener, n)
	cleanupLn := func() {
		for _, ls := range [][]net.Listener{raftLn, pbLn, serverLn} {
			for _, l := range ls {
				if l != nil {
					_ = l.Close()
				}
			}
		}
	}
	for i := 0; i < n; i++ {
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			cleanupLn()
			t.Fatal(err)
		}
		raftLn[i] = ln
		raftAddrs[i] = ln.Addr().String()
	}
	for i := 0; i < n; i++ {
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			cleanupLn()
			t.Fatal(err)
		}
		pbLn[i] = ln
		pbAddrs[i] = ln.Addr().String()
	}
	for i := 0; i < n; i++ {
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			cleanupLn()
			t.Fatal(err)
		}
		serverLn[i] = ln
		serverAddrs[i] = ln.Addr().String()
	}
	defer cleanupLn()

	peers := make([]Peer, n)
	for i := 0; i < n; i++ {
		peers[i] = Peer{
			NodeID:     fmt.Sprintf("n%d", i+1),
			RaftAddr:   raftAddrs[i],
			ServerAddr: serverAddrs[i], // real client-facing server (beacon forwarding resolves here)
			PBAddr:     pbAddrs[i],
		}
	}

	// Allocate every DataDir BEFORE registering tc.Close, because t.Cleanup runs
	// LIFO. With t.TempDir() called inside the loop below, its removal was
	// registered AFTER tc.Close and therefore ran BEFORE it: the directories were
	// deleted while the raft goroutines were still live, and any follower that
	// happened to persist its term in that window died with
	//
	//	panic: failed to save current term: logstore: create
	//	  /tmp/TestX/001/meta/raftlog/stable.tmp: no such file or directory
	//
	// — a package-level flake with a scary, misleading stack (it points at
	// hashicorp/raft, not at the harness). Registering the temp dirs first makes
	// tc.Close the LAST cleanup registered and so the FIRST to run: the cluster is
	// shut down, then its directories go away.
	dataDirs := make([]string, n)
	for i := range dataDirs {
		dataDirs[i] = t.TempDir()
	}

	tc := &pbTestCluster{peers: peers, nodes: make([]*Node, n), servers: make([]*server.Server, n)}
	t.Cleanup(tc.Close)

	for i := 0; i < n; i++ {
		reg := makeReg()
		cc := cache.DefaultConfig()
		cc.NumShards = 1
		cfg := Config{
			NodeID:          peers[i].NodeID,
			DataDir:         dataDirs[i],
			NumShards:       numShards,
			Bootstrap:       true,
			RaftAddr:        raftAddrs[i],
			Peers:           peers,
			ReplicationMode: ReplicationModePB,
			MinISR:          minISR,
			ShardCfg: shard.Config{
				NodeID: "ignored", DataDir: "ignored",
				Cache: cc, Ops: reg,
				Bootstrap:       true,
				RaftHeartbeatMs: 200,
				RaftElectionMs:  1000,
				NoSync:          true,
			},
			Ops: reg,
		}
		// Apply per-test overrides (e.g. PBAutoFailover, shrunk failover timings)
		// before construction. Kept out of the base config so the default cluster
		// stays byte-identical to the pre-Plan-4 static harness.
		for _, opt := range opts {
			opt(&cfg)
		}
		// Hand the pre-bound listeners' addrs off to their real owners: close the
		// raft listener immediately before mux.New re-listens on it, and the PB
		// listener immediately before NewNetTransport re-listens on it.
		_ = raftLn[i].Close()
		raftLn[i] = nil
		_ = pbLn[i].Close()
		pbLn[i] = nil
		node, err := New(cfg)
		if err != nil {
			t.Fatalf("node %d: %v", i, err)
		}
		tc.nodes[i] = node

		// Start the real client-facing server so peer forwarding (the beacon's
		// forward-to-leader path) resolves. Close the pre-bound listener immediately
		// before server.New re-listens on its addr, mirroring the raft harness.
		_ = serverLn[i].Close()
		serverLn[i] = nil
		srv, err := server.New(server.Config{Addr: peers[i].ServerAddr, Dispatcher: node})
		if err != nil {
			t.Fatalf("node %d server: %v", i, err)
		}
		go srv.Serve() //nolint:errcheck // Serve returns nil on clean Close
		tc.servers[i] = srv
	}

	// Wait for every node to observe a meta-Raft leader (the control plane that
	// carries the PB seed). Per-shard primary readiness is polled by the test.
	deadline := time.Now().Add(15 * time.Second)
	for {
		ready := true
		for _, nd := range tc.nodes {
			if addr, _ := nd.meta.Raft.LeaderWithID(); addr == "" {
				ready = false
				break
			}
		}
		if ready {
			break
		}
		if !time.Now().Before(deadline) {
			t.Fatal("PB cluster: no meta-Raft leader within 15s")
		}
		time.Sleep(50 * time.Millisecond)
	}
	return tc
}

func (tc *pbTestCluster) Close() {
	// Close servers before nodes so no in-flight forward races a closing node.
	for i, s := range tc.servers {
		if s != nil {
			_ = s.Close()
			tc.servers[i] = nil
		}
	}
	for i, n := range tc.nodes {
		if n != nil {
			_ = n.Close()
			tc.nodes[i] = nil
		}
	}
}

// TestStaticPBClusterReplicatesAndReads brings up a 3-node PB cluster (RF=full,
// MinISR=2), writes to a shard's primary, and asserts the write is readable on
// the primary AND replicated to the ISR backups. The bootstrap seed sets each
// shard's ISR to all 3 owners, and Propose always commits on the FULL current
// ISR (MinISR is only an ISR-shrink floor — inert in this static cluster, not
// the reason both backups ack), so the Put only returns once BOTH backups
// acked over the real PB transport, and a green here exercises the whole
// Task-4 wiring end to end.
func TestStaticPBClusterReplicatesAndReads(t *testing.T) {
	const numShards = 4
	tc := newPBTestCluster(t, 3, numShards, 2)

	key := []byte("pb-key")
	sh := shardOf(key, numShards)
	val := []byte("pb-value")
	putArgs := ops.EncodePutArgs(key, val, 0)

	// The bootstrap seed makes owners[0] the primary; find its node. Poll the
	// primary's Call until it succeeds: on a node that built its shard before the
	// seed commit replicated, the self-lease is granted only once the leaseKeeper
	// observes the FSM primary — that is EXPECTED (no construction-time retry).
	var primaryIdx = -1
	var lastErr error
	var lastPrimary string
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		primary := tc.nodes[0].meta.FSM.ShardPrimary(sh)
		lastPrimary = primary
		if primary == "" {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		idx := -1
		for i, p := range tc.peers {
			if p.NodeID == primary {
				idx = i
				break
			}
		}
		if idx < 0 {
			t.Fatalf("primary %q not in peer list", primary)
		}
		_, err := tc.nodes[idx].Call("put", putArgs)
		if err == nil {
			primaryIdx = idx
			break
		}
		lastErr = err
		time.Sleep(50 * time.Millisecond)
	}
	if primaryIdx < 0 {
		t.Fatalf("PB write to primary never succeeded within 15s (lastPrimary=%q, lastErr=%v)", lastPrimary, lastErr)
	}

	// Read back on the primary's local store.
	got, err := tc.nodes[primaryIdx].getShard(sh).Get(key)
	if err != nil || string(got) != string(val) {
		t.Fatalf("primary Get = %q, %v; want %q", got, err, val)
	}

	// The primary's engine must have committed the write (full ISR acked).
	if c := tc.nodes[primaryIdx].pbEngines[sh].Committed(); c < 1 {
		t.Fatalf("primary engine Committed()=%d; want >=1", c)
	}

	// Every backup ISR member must have materialized the write via Receive:
	// its local store returns the value AND its engine advanced LastApplied.
	for i, nd := range tc.nodes {
		if i == primaryIdx {
			continue
		}
		eng := nd.pbEngines[sh]
		if eng == nil {
			t.Fatalf("node %d owns no engine for shard %d (full replication expected)", i, sh)
		}
		var replicated bool
		bd := time.Now().Add(5 * time.Second)
		for time.Now().Before(bd) {
			if v, gerr := nd.getShard(sh).Get(key); gerr == nil && string(v) == string(val) && eng.LastApplied() >= 1 {
				replicated = true
				break
			}
			time.Sleep(25 * time.Millisecond)
		}
		if !replicated {
			t.Fatalf("write did not replicate to backup node %d (LastApplied=%d)", i, eng.LastApplied())
		}
	}
}

// requireVisiblePlacement establishes a precondition several tests assert but
// none used to check: that shard sh has a VISIBLE placement in watch's MetaFSM.
//
// MetaState.Placement is normally filled by the bootstrap OpSetMembers entry,
// which can be LOST to bootstrap leadership churn (appended on a leader that
// steps down before it commits) — the reason MetaFSM.Apply carries an explicit
// OpSetPlacement self-heal path at all. A static test cluster never triggers that
// self-heal, so in exactly those runs Placement stays permanently empty and any
// test asserting on it fails against a premise that silently does not hold, with
// the PRODUCT behaviour entirely correct. Measured at 2 failures in 8 runs of
// TestReplMetricsBelowPlacementInvisibleCase ("placement_size = 0, want 3").
//
// So: wait briefly for the bootstrap entry, and if it was lost, drive the same
// OpSetPlacement self-heal production would. This establishes a PRECONDITION and
// weakens no assertion — the distinction that makes a de-flake safe.
//
// TestGrowAbortLoggedAndRateLimited carries an inline copy of this logic
// (inherited from fix/pb-critic-findings); it is left as-is because it is
// verified, but the two could be unified.
func requireVisiblePlacement(t *testing.T, nodes []*Node, watch *Node, sh, numShards int) {
	t.Helper()
	visible := func() bool {
		got := watch.meta.FSM.State().Placement
		return sh < len(got) && len(got[sh]) > 0
	}
	deadline := time.Now().Add(5 * time.Second)
	for !visible() && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	if visible() {
		return
	}
	owners := append([]string(nil), watch.shardOwners(sh)...)
	if len(owners) == 0 {
		t.Fatalf("shard %d has no placement owners on the watched node", sh)
	}
	healed := false
	for _, nd := range nodes {
		if err := nd.meta.ApplySetPlacement(sh, numShards, owners, 5*time.Second); err == nil {
			healed = true
			break
		}
	}
	if !healed {
		t.Fatal("could not self-heal the placement table on any node (no meta leader?)")
	}
	deadline = time.Now().Add(10 * time.Second)
	for !visible() && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	if !visible() {
		t.Fatalf("shard %d placement still not visible after an explicit OpSetPlacement "+
			"— the assertions below cannot mean anything", sh)
	}
}

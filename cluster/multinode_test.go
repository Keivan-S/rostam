// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/server"
	"github.com/rostamlabs/rostam/shard"
)

// testCluster owns a 3-node (or N-node) in-process Rostam cluster:
// N cluster.Node instances, N TCP servers, a shared peer list, the
// dataDirs preserved for restart-style tests, and one client wired to
// all server addresses.
type testCluster struct {
	nodes    []*Node
	servers  []*server.Server
	peers    []Peer
	dataDirs []string // preserved across stops for restart test
	client   *client.Client
}

// newTestCluster spins up an N-node Rostam cluster in-process with
// numShards data shards. It pre-binds Raft and server ports per node
// (so Peers carries final addrs before construction), constructs each
// node and TCP server, then waits for every node to see a meta-Raft
// leader before returning. The Cleanup is registered with t.
func newTestCluster(t *testing.T, n int, numShards int, rf ...int) *testCluster {
	t.Helper()
	replicationFactor := 0
	if len(rf) > 0 {
		replicationFactor = rf[0]
	}
	// Each node gets its own registry so that RegisterTopology (called once
	// per node during newMultiNode) does not collide across nodes that share
	// a single test process.
	makeReg := func() *ops.Registry {
		r := ops.NewRegistry()
		if err := ops.RegisterBuiltins(r); err != nil {
			t.Fatal(err)
		}
		return r
	}

	// Pre-bind Raft ports and server ports per node. We need addrs
	// BEFORE constructing cluster.Node (Peers must list them). To
	// minimize the TOCTOU window between Close and re-Listen (during
	// which the kernel could hand the freed port to e.g. an ephemeral
	// outbound raft dial, leading to a Raft node listening on what we
	// thought was a server port), we hold ALL listeners open until
	// each constructor is about to claim its addr — see the per-node
	// closeAndBuild dance in the construction loop below.
	raftAddrs := make([]string, n)
	serverAddrs := make([]string, n)
	raftLn := make([]net.Listener, n)
	serverLn := make([]net.Listener, n)
	{
		cleanup := func() {
			for _, l := range raftLn {
				if l != nil {
					_ = l.Close()
				}
			}
			for _, l := range serverLn {
				if l != nil {
					_ = l.Close()
				}
			}
		}
		for i := 0; i < n; i++ {
			ln, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				cleanup()
				t.Fatal(err)
			}
			raftLn[i] = ln
			raftAddrs[i] = ln.Addr().String()
		}
		for i := 0; i < n; i++ {
			ln, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				cleanup()
				t.Fatal(err)
			}
			serverLn[i] = ln
			serverAddrs[i] = ln.Addr().String()
		}
	}

	peers := make([]Peer, n)
	for i := 0; i < n; i++ {
		peers[i] = Peer{
			NodeID:     fmt.Sprintf("n%d", i+1),
			RaftAddr:   raftAddrs[i],
			ServerAddr: serverAddrs[i],
		}
	}
	// Defensive uniqueness check on the 2*n pre-bound addresses. A
	// collision here would mean the kernel handed us a duplicate port
	// while listeners were still open — should be impossible, but if
	// it ever happens the surface symptom (client connects to the
	// "wrong" server) is obscure, so fail loud.
	{
		seen := make(map[string]int, 2*n)
		for i, a := range raftAddrs {
			if j, dup := seen[a]; dup {
				t.Fatalf("pre-bind: raftAddrs[%d]=%s duplicates earlier slot %d", i, a, j)
			}
			seen[a] = i
		}
		for i, a := range serverAddrs {
			if j, dup := seen[a]; dup {
				t.Fatalf("pre-bind: serverAddrs[%d]=%s duplicates raft/server slot %d", i, a, j)
			}
			seen[a] = n + i
		}
	}

	tc := &testCluster{peers: peers}
	tc.dataDirs = make([]string, n)
	for i := 0; i < n; i++ {
		tc.dataDirs[i] = t.TempDir()
	}
	// Pre-allocate slices so cleanup paths can index without shifting on
	// partial init.
	tc.nodes = make([]*Node, n)
	tc.servers = make([]*server.Server, n)

	// Ensure that any pre-bind listeners we never get around to handing
	// off (e.g. mid-loop fatal) are released. Slots set to nil have
	// already been transferred to mux/server via Close-and-Listen.
	defer func() {
		for _, l := range raftLn {
			if l != nil {
				_ = l.Close()
			}
		}
		for _, l := range serverLn {
			if l != nil {
				_ = l.Close()
			}
		}
	}()

	for i := 0; i < n; i++ {
		reg := makeReg()
		cc := cache.DefaultConfig()
		cc.NumShards = 1
		cfg := Config{
			NodeID:            peers[i].NodeID,
			DataDir:           tc.dataDirs[i],
			NumShards:         numShards,
			ReplicationFactor: replicationFactor,
			Bootstrap:         true,
			RaftAddr:          raftAddrs[i],
			Peers:             peers,
			// ROSTAM_TEST_RAFT_TRANSPORT=fabric runs the whole multinode suite
			// over the fabric transport instead of the default mux path.
			RaftTransport: os.Getenv("ROSTAM_TEST_RAFT_TRANSPORT"),
			ShardCfg: shard.Config{
				NodeID: "ignored", DataDir: "ignored",
				Cache: cc, Ops: reg,
				Bootstrap: true,
				// Under -race, scheduler jitter can starve a leader's
				// heartbeat goroutine for >100ms, causing followers to
				// start spurious elections that drop the in-flight
				// Apply with ErrLeadershipLost — and, occasionally,
				// abort the TCP pipeline mid-response, surfacing to the
				// client as "connection reset by peer". 200ms heartbeat
				// / 1s election leaves enough headroom for race-mode
				// CI without making the tests slow. (Kept at 1s, NOT the
				// 2500ms used by the inttest/root helpers: this harness
				// backs leader-KILL recovery tests where a wider election
				// window would only SLOW re-election; the flake here is a
				// setup-seed fragility addressed by seed-retry + a widened
				// recovery deadline below, not election storms.)
				RaftHeartbeatMs: 200,
				RaftElectionMs:  1000,
				NoSync:          true,
			},
			Ops: reg,
		}
		// Close the pre-bound Raft listener immediately before mux.New
		// re-listens on its exact address. The other 5 pre-bound
		// listeners stay open during this window, so the kernel cannot
		// reassign the freed port to a different node's listener.
		_ = raftLn[i].Close()
		raftLn[i] = nil
		node, err := New(cfg)
		if err != nil {
			for j := 0; j < i; j++ {
				if tc.servers[j] != nil {
					_ = tc.servers[j].Close()
				}
				if tc.nodes[j] != nil {
					_ = tc.nodes[j].Close()
				}
			}
			t.Fatalf("node %d: %v", i, err)
		}
		// Verify mux bound to the requested address.
		if node.mux != nil {
			if got, want := node.mux.Addr().String(), raftAddrs[i]; got != want {
				_ = node.Close()
				t.Fatalf("node %d: mux bound to %s != requested %s", i, got, want)
			}
		}
		tc.nodes[i] = node

		// Same pattern for the per-node server listener.
		_ = serverLn[i].Close()
		serverLn[i] = nil
		srv, err := server.New(server.Config{Addr: peers[i].ServerAddr, Dispatcher: node})
		if err != nil {
			_ = node.Close()
			for j := 0; j < i; j++ {
				if tc.servers[j] != nil {
					_ = tc.servers[j].Close()
				}
				if tc.nodes[j] != nil {
					_ = tc.nodes[j].Close()
				}
			}
			t.Fatalf("server %d: %v", i, err)
		}
		// Verify the actual bound addr matches what we asked for. A
		// mismatch would mean the OS handed us a different port (e.g.
		// because the requested port was grabbed by an ephemeral dial
		// between our Close and Listen) — surface that loudly.
		if got, want := srv.Addr().String(), peers[i].ServerAddr; got != want {
			_ = srv.Close()
			_ = node.Close()
			t.Fatalf("server %d: bound addr %s != requested %s", i, got, want)
		}
		go srv.Serve() //nolint:errcheck // Serve returns nil on clean Close
		tc.servers[i] = srv
	}

	// Wait for cluster to settle: every node must report a non-empty
	// LeaderAddr (i.e. shard 0 has elected a leader visible to it).
	tc.waitReady(t, 15*time.Second)

	addrs := make([]string, n)
	for i, s := range tc.servers {
		addrs[i] = s.Addr().String()
	}
	// Debug: log every address so flakes that stem from address
	// confusion are easy to diagnose from test output.
	for i := 0; i < n; i++ {
		t.Logf("node %d: raftAddr=%s serverAddr(peers)=%s serverAddr(actual)=%s", i, peers[i].RaftAddr, peers[i].ServerAddr, addrs[i])
	}
	c, err := client.New(client.Config{
		Servers:           addrs,
		MaxConnsPerServer: 8,
		MaxNotLeaderHops:  5, // generous; 3 nodes × numShards may bounce
	})
	if err != nil {
		t.Fatal(err)
	}
	tc.client = c

	t.Cleanup(tc.Close)
	return tc
}

// waitReady blocks until every (non-nil) node reports a Leader state
// for every shard, or the deadline elapses. Checking every shard (not
// just shard 0) is necessary because cluster.Node routes by xxhash and
// a write to k000 may land on shard 3 — if shard 3 hasn't elected a
// leader yet, every node returns NotLeader and the client exhausts its
// hop budget. We accept that a leader exists *somewhere* per shard;
// each node simply needs to see at least one Leader in its PerShard
// view for every shard ID.
func (tc *testCluster) waitReady(t *testing.T, d time.Duration) {
	t.Helper()
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if tc.allShardsHaveLeader() {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("cluster did not become ready within %s", d)
}

// allShardsHaveLeader returns true iff for every shard ID, at least
// one (non-nil) node reports Raft state "Leader" for that shard.
func (tc *testCluster) allShardsHaveLeader() bool {
	if len(tc.nodes) == 0 {
		return false
	}
	// Determine numShards from the first live node.
	var numShards int
	for _, n := range tc.nodes {
		if n != nil {
			numShards = n.Stats().NumShards
			break
		}
	}
	if numShards == 0 {
		return false
	}
	hasLeader := make([]bool, numShards)
	for _, n := range tc.nodes {
		if n == nil {
			continue
		}
		st := n.Stats()
		for s := 0; s < numShards && s < len(st.PerShard); s++ {
			if st.PerShard[s].Raft["state"] == "Leader" {
				hasLeader[s] = true
			}
		}
	}
	for _, ok := range hasLeader {
		if !ok {
			return false
		}
	}
	return true
}

// Close releases all resources owned by the testCluster. Idempotent on
// nil slots (e.g. after a leader-kill test nils a node).
func (tc *testCluster) Close() {
	if tc.client != nil {
		_ = tc.client.Close()
		tc.client = nil
	}
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

// findShardLeader returns the index of the node hosting shardID's leader,
// or (-1, false) if no node currently reports Leader for that shard.
func (tc *testCluster) findShardLeader(shardID int) (idx int, found bool) {
	for i, n := range tc.nodes {
		if n == nil {
			continue
		}
		st := n.Stats()
		if shardID < 0 || shardID >= len(st.PerShard) {
			continue
		}
		if st.PerShard[shardID].Raft["state"] == "Leader" {
			return i, true
		}
	}
	return -1, false
}

// TestThreeNodeBootstrap verifies that a 3-node cluster reaches a state
// where every node sees a leader. newTestCluster's waitReady does the
// load-bearing assertion; this test just exercises construction.
func TestThreeNodeBootstrap(t *testing.T) {
	tc := newTestCluster(t, 3, 4)
	_ = tc // waitReady inside newTestCluster already asserted readiness
}

// TestThreeNodeWriteFromAnyNode writes 50 keys via the multi-server
// client and reads them back. Because the client picks Servers[0] for
// the initial attempt and follows NotLeader hints, this exercises the
// follower-write-then-redirect path.
func TestThreeNodeWriteFromAnyNode(t *testing.T) {
	tc := newTestCluster(t, 3, 4)
	ctx := context.Background()
	for i := 0; i < 50; i++ {
		k := fmt.Appendf(nil, "k%03d", i)
		if _, err := tc.client.Call(ctx, "put", ops.EncodePutArgs(k, []byte{1}, 0)); err != nil {
			t.Fatalf("put %d: %v", i, err)
		}
	}
	// Read-only ops bypass Raft; wait for all replicas to apply before reads.
	for _, n := range tc.nodes {
		if n != nil {
			waitAllApplied(t, n)
		}
	}
	for i := 0; i < 50; i++ {
		k := fmt.Appendf(nil, "k%03d", i)
		got, err := tc.client.Call(ctx, "get", ops.EncodeKeyArgs(k))
		if err != nil {
			t.Fatalf("get %d: %v", i, err)
		}
		if !bytes.Equal(got, []byte{1}) {
			t.Fatalf("get %d = %v, want [1]", i, got)
		}
	}
}

// TestThreeNodeLeaderKill seeds 20 keys, kills the node hosting shard
// 0's leader, then verifies writes resume against a freshly elected
// leader within a deadline. The post-kill put-then-get confirms the
// new leader actually committed.
func TestThreeNodeLeaderKill(t *testing.T) {
	tc := newTestCluster(t, 3, 4)
	ctx := context.Background()

	for i := 0; i < 20; i++ {
		k := fmt.Appendf(nil, "k%02d", i)
		if _, err := tc.client.Call(ctx, "put", ops.EncodePutArgs(k, []byte{1}, 0)); err != nil {
			t.Fatalf("put: %v", err)
		}
	}

	leaderIdx, ok := tc.findShardLeader(0)
	if !ok {
		t.Fatal("could not find shard 0 leader")
	}
	t.Logf("killing leader node %d", leaderIdx)
	deadAddr := tc.peers[leaderIdx].ServerAddr

	// Close the original client first so its pooled conns to the killed
	// server (and others) are torn down. Otherwise:
	//   1. server.Close blocks on in-flight handleConn goroutines whose
	//      readFrame loops won't return until the client end closes
	//      (default IdleTimeout is 5min, so the test would hang),
	//   2. cfg.Servers[0] may be the dead server's addr, and Client.Call
	//      doesn't retry on dial-refused errors — so the post-kill write
	//      would fail immediately. A fresh client with only the surviving
	//      servers avoids both problems.
	_ = tc.client.Close()
	tc.client = nil

	// Close node first so Raft + shards unwind cleanly, then the server.
	// Both should now return promptly because the client released its
	// connections.
	_ = tc.nodes[leaderIdx].Close()
	_ = tc.servers[leaderIdx].Close()
	tc.servers[leaderIdx] = nil
	tc.nodes[leaderIdx] = nil

	// Fresh client wired only to surviving servers.
	survAddrs := make([]string, 0, len(tc.peers)-1)
	for i, p := range tc.peers {
		if i == leaderIdx {
			continue
		}
		survAddrs = append(survAddrs, p.ServerAddr)
	}
	if deadAddr == "" || len(survAddrs) == 0 {
		t.Fatalf("internal: deadAddr=%q survAddrs=%v", deadAddr, survAddrs)
	}
	c2, err := client.New(client.Config{
		Servers:           survAddrs,
		MaxConnsPerServer: 8,
		MaxNotLeaderHops:  5,
	})
	if err != nil {
		t.Fatalf("new client: %v", err)
	}
	defer func() { _ = c2.Close() }()

	// New leader must elect within deadline. Writes resume.
	// 30s budget matches the restart-path deadline — under full-suite
	// CPU contention a single-shard re-election plus the client's
	// 100ms poll interval can exceed a tight 10s window.
	deadline := time.Now().Add(30 * time.Second)
	var lastErr error
	succeeded := false
	for time.Now().Before(deadline) {
		_, err := c2.Call(ctx, "put", ops.EncodePutArgs([]byte("post-kill"), []byte{2}, 0))
		if err == nil {
			succeeded = true
			break
		}
		lastErr = err
		time.Sleep(100 * time.Millisecond)
	}
	if !succeeded {
		t.Fatalf("writes never resumed: %v", lastErr)
	}
	// Get is a read-only op served from local cache without a leader
	// check, so it may land on a follower whose FSM has not yet
	// applied the Put we just made (Raft commit on the new leader
	// does not block on followers' Apply). Retry briefly until the
	// value is visible.
	var got []byte
	getDeadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(getDeadline) {
		var gerr error
		got, gerr = c2.Call(ctx, "get", ops.EncodeKeyArgs([]byte("post-kill")))
		if gerr == nil {
			break
		}
		if !errors.Is(gerr, client.ErrNotFound) {
			t.Fatalf("get post-kill: %v", gerr)
		}
		time.Sleep(50 * time.Millisecond)
	}
	if !bytes.Equal(got, []byte{2}) {
		t.Fatalf("got %v, want [2]", got)
	}
}

// newSmartTestCluster wraps newTestCluster and replaces the dumb
// round-robin client with a smart client that has cfg.Ops set.
// Returns the cluster and the ops.Registry used on the client side
// (each cluster node retains its own registry from newTestCluster).
//
// TopologyRefreshInterval is set to 1s for fast test turnaround.
func newSmartTestCluster(t *testing.T, n int, numShards int) (*testCluster, *ops.Registry) {
	t.Helper()
	tc := newTestCluster(t, n, numShards)
	// Close the plain client created by newTestCluster; the t.Cleanup
	// registered there will call tc.Close() which will close whatever
	// tc.client points to at cleanup time, so we just overwrite the field.
	_ = tc.client.Close()
	addrs := make([]string, len(tc.servers))
	for i, s := range tc.servers {
		addrs[i] = s.Addr().String()
	}
	clientReg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(clientReg); err != nil {
		t.Fatal(err)
	}
	c, err := client.New(client.Config{
		Servers:                 addrs,
		MaxConnsPerServer:       8,
		MaxNotLeaderHops:        5,
		Ops:                     clientReg,
		TopologyRefreshInterval: 1 * time.Second,
	})
	if err != nil {
		t.Fatalf("smart client: %v", err)
	}
	tc.client = c
	return tc, clientReg
}

// TestSmartClientRoutesToShardLeader writes 100 keys and reads them back
// using a smart client. The smart client uses topology-aware routing to
// send each key directly to its shard leader, avoiding NotLeader bounces.
func TestSmartClientRoutesToShardLeader(t *testing.T) {
	tc, _ := newSmartTestCluster(t, 3, 4)
	// Brief pause lets leaders stabilise if any are still electing.
	time.Sleep(500 * time.Millisecond)

	ctx := context.Background()
	for i := 0; i < 100; i++ {
		k := fmt.Appendf(nil, "k%03d", i)
		if _, err := tc.client.Call(ctx, "put", ops.EncodePutArgs(k, []byte{1}, 0)); err != nil {
			t.Fatalf("put %d: %v", i, err)
		}
	}
	// Read-only ops bypass Raft; wait for all replicas to apply.
	for _, n := range tc.nodes {
		if n != nil {
			waitAllApplied(t, n)
		}
	}
	for i := 0; i < 100; i++ {
		k := fmt.Appendf(nil, "k%03d", i)
		got, err := tc.client.Call(ctx, "get", ops.EncodeKeyArgs(k))
		if err != nil {
			t.Fatalf("get %d: %v", i, err)
		}
		if !bytes.Equal(got, []byte{1}) {
			t.Errorf("get %d = %v, want [1]", i, got)
		}
	}
}

// TestSmartClientConvergesAfterLeaderKill seeds 20 keys, kills the
// shard-0 leader, and verifies that writes resume within 30 s as the
// smart client converges on the newly elected leader via topology refresh.
//
// Regression: before the fixup this test was flaky ~50% of
// the time when Servers[0] coincided with the killed leader. The fixes
// were: round-robin nextServer (so the dead Servers[0] isn't always
// picked as the fallback), retry transport errors (dial refused, EOF)
// against other servers within MaxNotLeaderHops, and short-deadline
// refreshTopology attempts (500 ms/server) so a dead Servers[0] does
// not stall the entire refresh budget. Run with -count=10 -race to
// confirm deterministic convergence.
func TestSmartClientConvergesAfterLeaderKill(t *testing.T) {
	tc, _ := newSmartTestCluster(t, 3, 4)
	time.Sleep(500 * time.Millisecond)

	ctx := context.Background()
	for i := 0; i < 20; i++ {
		k := fmt.Appendf(nil, "k%02d", i)
		// Setup seed: tolerate transient errors (shard-leader election jitter, the
		// 500ms warmup above not yet covering every shard's first election) within a
		// bounded budget instead of fataling on the first blip. Under full-./... CPU
		// oversubscription the bare single-shot put fataled ~7s in when a shard had not
		// yet elected. This is setup readiness, not the assertion under test (the
		// post-kill convergence + the read-back value below are); seeding still fails
		// loud if a key never lands within the budget.
		seedDeadline := time.Now().Add(30 * time.Second)
		var perr error
		landed := false
		for time.Now().Before(seedDeadline) {
			if _, perr = tc.client.Call(ctx, "put", ops.EncodePutArgs(k, []byte{1}, 0)); perr == nil {
				landed = true
				break
			}
			time.Sleep(50 * time.Millisecond)
		}
		if !landed {
			t.Fatalf("seed put %d never landed: %v", i, perr)
		}
	}

	leaderIdx, ok := tc.findShardLeader(0)
	if !ok {
		t.Fatal("no shard 0 leader")
	}
	t.Logf("killing shard 0 leader: node %d", leaderIdx)
	_ = tc.servers[leaderIdx].Close()
	_ = tc.nodes[leaderIdx].Close()
	tc.servers[leaderIdx] = nil
	tc.nodes[leaderIdx] = nil

	deadline := time.Now().Add(cpuScaled(45 * time.Second)) // widened 30s->45s for CPU-contended CI; finite progress deadline (writes must resume after leader kill), so a genuine never-converge still fails loud
	var lastErr error
	succeeded := false
	for time.Now().Before(deadline) {
		_, err := tc.client.Call(ctx, "put", ops.EncodePutArgs([]byte("post-kill"), []byte{2}, 0))
		if err == nil {
			succeeded = true
			break
		}
		lastErr = err
		time.Sleep(100 * time.Millisecond)
	}
	if !succeeded {
		t.Fatalf("writes never resumed: %v", lastErr)
	}
	// Read-only ops bypass Raft and hit the local cache directly, so a
	// get after a put can race the FSM apply on followers. Wait for all
	// surviving nodes' shards to catch up before the get.
	for _, n := range tc.nodes {
		if n != nil {
			waitAllApplied(t, n)
		}
	}
	got, err := tc.client.Call(ctx, "get", ops.EncodeKeyArgs([]byte("post-kill")))
	if err != nil {
		t.Fatalf("get post-kill: %v", err)
	}
	if !bytes.Equal(got, []byte{2}) {
		t.Errorf("got %v, want [2]", got)
	}
}

// TestSmartClientCompatWithOpsNil verifies that the Phase-5b-style client
// (Ops=nil) still works correctly against the same multi-node cluster,
// preserving backward compatibility.
func TestSmartClientCompatWithOpsNil(t *testing.T) {
	tc := newTestCluster(t, 3, 4)
	ctx := context.Background()
	for i := 0; i < 10; i++ {
		k := fmt.Appendf(nil, "k%02d", i)
		if _, err := tc.client.Call(ctx, "put", ops.EncodePutArgs(k, []byte{1}, 0)); err != nil {
			t.Fatalf("put: %v", err)
		}
	}
	// Read-only ops bypass Raft; wait for all replicas to apply before reads.
	for _, n := range tc.nodes {
		if n != nil {
			waitAllApplied(t, n)
		}
	}
	for i := 0; i < 10; i++ {
		k := fmt.Appendf(nil, "k%02d", i)
		got, err := tc.client.Call(ctx, "get", ops.EncodeKeyArgs(k))
		if err != nil {
			t.Fatalf("get %d: %v", i, err)
		}
		if !bytes.Equal(got, []byte{1}) {
			t.Errorf("get %d = %v, want [1]", i, got)
		}
	}
}

// TestThreeNodeRestart writes 50 keys, stops all 3 nodes, restarts each
// with Bootstrap=false against the same DataDirs, and verifies the keys
// survive. This is the canonical durability test for the multi-node
// path.
func TestThreeNodeRestart(t *testing.T) {
	tc := newTestCluster(t, 3, 4)
	ctx := context.Background()

	// Write 50 keys. We rely on Raft commit (each Apply returns only
	// after a majority has persisted) for durability rather than fsync,
	// so NoSync=true is fine here.
	for i := 0; i < 50; i++ {
		k := fmt.Appendf(nil, "k%03d", i)
		v := fmt.Appendf(nil, "v%03d", i)
		if _, err := tc.client.Call(ctx, "put", ops.EncodePutArgs(k, v, 0)); err != nil {
			t.Fatalf("put: %v", err)
		}
	}

	// Stop all 3 nodes; keep dataDirs and peers for the restart.
	_ = tc.client.Close()
	tc.client = nil
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

	// Restart with Bootstrap=false on the same DataDirs. Each node gets
	// its own registry so RegisterTopology doesn't collide across nodes.
	newNodes := make([]*Node, len(tc.peers))
	newServers := make([]*server.Server, len(tc.peers))
	for i := range tc.peers {
		reg := ops.NewRegistry()
		if err := ops.RegisterBuiltins(reg); err != nil {
			t.Fatal(err)
		}
		cc := cache.DefaultConfig()
		cc.NumShards = 1
		cfg := Config{
			NodeID:        tc.peers[i].NodeID,
			DataDir:       tc.dataDirs[i],
			NumShards:     4,
			Bootstrap:     false, // existing state
			RaftAddr:      tc.peers[i].RaftAddr,
			Peers:         tc.peers,
			RaftTransport: os.Getenv("ROSTAM_TEST_RAFT_TRANSPORT"),
			ShardCfg: shard.Config{
				NodeID: "ignored", DataDir: "ignored",
				Cache: cc, Ops: reg,
				Bootstrap:       false,
				RaftHeartbeatMs: 200, RaftElectionMs: 1000, NoSync: true,
			},
			Ops: reg,
		}
		node, err := New(cfg)
		if err != nil {
			for j := 0; j < i; j++ {
				if newServers[j] != nil {
					_ = newServers[j].Close()
				}
				if newNodes[j] != nil {
					_ = newNodes[j].Close()
				}
			}
			t.Fatalf("restart node %d: %v", i, err)
		}
		newNodes[i] = node
		srv, err := server.New(server.Config{Addr: tc.peers[i].ServerAddr, Dispatcher: node})
		if err != nil {
			_ = node.Close()
			for j := 0; j < i; j++ {
				if newServers[j] != nil {
					_ = newServers[j].Close()
				}
				if newNodes[j] != nil {
					_ = newNodes[j].Close()
				}
			}
			t.Fatalf("restart server %d: %v", i, err)
		}
		go srv.Serve() //nolint:errcheck // Serve returns nil on clean Close
		newServers[i] = srv
	}
	t.Cleanup(func() {
		for _, s := range newServers {
			if s != nil {
				_ = s.Close()
			}
		}
		for _, n := range newNodes {
			if n != nil {
				_ = n.Close()
			}
		}
	})

	// Wait for restarted cluster to re-elect leaders on every shard.
	// Shard 0's leader alone isn't enough — keys hash to all shards
	// and the client exhausts its hop budget if any shard is still
	// electing. Restart with 13 Raft groups (4 shards + meta) per node
	// can collide on randomized election timeouts; budget 30s.
	restartTC := &testCluster{nodes: newNodes}
	restartDeadline := time.Now().Add(30 * time.Second)
	ready := false
	for time.Now().Before(restartDeadline) {
		if restartTC.allShardsHaveLeader() {
			ready = true
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if !ready {
		t.Fatal("restarted cluster did not re-elect leaders within deadline")
	}

	// Leader election precedes FSM replay; reads bypass Raft and hit
	// the cache directly, so a Get before applied_index catches up
	// can miss recently-applied keys. Wait per-shard.
	for _, n := range newNodes {
		waitAllApplied(t, n)
	}

	addrs := make([]string, len(tc.peers))
	for i, p := range tc.peers {
		addrs[i] = p.ServerAddr
	}
	c, err := client.New(client.Config{
		Servers:           addrs,
		MaxConnsPerServer: 8,
		MaxNotLeaderHops:  5,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	for i := 0; i < 50; i++ {
		k := fmt.Appendf(nil, "k%03d", i)
		wantV := fmt.Appendf(nil, "v%03d", i)
		got, err := c.Call(ctx, "get", ops.EncodeKeyArgs(k))
		if err != nil {
			t.Fatalf("get %d after restart: %v", i, err)
		}
		if !bytes.Equal(got, wantV) {
			t.Fatalf("get %d after restart = %q, want %q", i, got, wantV)
		}
	}
}

// TestSmartClientWarmRestart writes 100 keys to a 3-node cluster with
// Durable=true (mmap-backed cache), stops all 3 nodes, restarts them,
// and verifies the keys are readable from mmap-persisted state without
// replaying the Raft log.
//
// This is the canonical mmap durability test: it proves that
// cache.shard's pages.dat survives a full cluster restart and that
// FSM.Apply's applied-index guard correctly skips log entries that are
// already reflected in the persisted mmap pages.
//
// Linux-only because cache.DataDir (mmap) is Linux-only.
func TestSmartClientWarmRestart(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mmap is Linux-only")
	}

	const n = 3

	makeReg := func() *ops.Registry {
		r := ops.NewRegistry()
		if err := ops.RegisterBuiltins(r); err != nil {
			t.Fatal(err)
		}
		return r
	}

	// Pre-bind Raft + server ports for all 3 nodes. Same listener-juggling
	// pattern as newTestCluster: hold all 6 open until each constructor
	// is about to claim its addr, preventing port reuse by ephemeral dials.
	raftAddrs := make([]string, n)
	serverAddrs := make([]string, n)
	raftLn := make([]net.Listener, n)
	serverLn := make([]net.Listener, n)

	prebind := func() {
		cleanupLn := func() {
			for _, l := range raftLn {
				if l != nil {
					_ = l.Close()
				}
			}
			for _, l := range serverLn {
				if l != nil {
					_ = l.Close()
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
			serverLn[i] = ln
			serverAddrs[i] = ln.Addr().String()
		}
	}
	prebind()

	peers := make([]Peer, n)
	for i := 0; i < n; i++ {
		peers[i] = Peer{
			NodeID:     fmt.Sprintf("n%d", i+1),
			RaftAddr:   raftAddrs[i],
			ServerAddr: serverAddrs[i],
		}
	}

	dataDirs := make([]string, n)
	for i := 0; i < n; i++ {
		dataDirs[i] = t.TempDir()
	}

	// spinUp constructs 3 nodes + 3 servers bound to the pre-bound addrs
	// (raftLn/serverLn hold the ports open until each constructor claims
	// them). Returns nodes and servers; callers own Close.
	spinUp := func(bootstrap bool) ([]*Node, []*server.Server) {
		nodes := make([]*Node, n)
		servers := make([]*server.Server, n)

		// Ensure any listeners we never hand off are released on panic/fatal.
		defer func() {
			for _, l := range raftLn {
				if l != nil {
					_ = l.Close()
				}
			}
			for _, l := range serverLn {
				if l != nil {
					_ = l.Close()
				}
			}
		}()

		for i := 0; i < n; i++ {
			reg := makeReg()
			cc := cache.DefaultConfig()
			cc.NumShards = 1
			cc.Durable = true
			cc.MsyncIntervalMs = 50
			// Do NOT set cc.DataDir — cluster/node.go auto-populates it
			// from shard.Config.DataDir + "/cache".

			cfg := Config{
				NodeID:    peers[i].NodeID,
				DataDir:   dataDirs[i],
				NumShards: 1,
				Bootstrap: bootstrap,
				RaftAddr:  raftAddrs[i],
				Peers:     peers,
				ShardCfg: shard.Config{
					NodeID: "ignored", DataDir: "ignored",
					Cache:           cc,
					Ops:             reg,
					Bootstrap:       bootstrap,
					RaftHeartbeatMs: 200,
					RaftElectionMs:  1000,
					NoSync:          true,
				},
				Ops: reg,
			}

			// Close-and-relisten: close the pre-bound Raft listener so
			// mux.New can bind the same addr. Remaining listeners hold
			// their ports open during this window.
			_ = raftLn[i].Close()
			raftLn[i] = nil
			node, err := New(cfg)
			if err != nil {
				for j := 0; j < i; j++ {
					if servers[j] != nil {
						_ = servers[j].Close()
					}
					if nodes[j] != nil {
						_ = nodes[j].Close()
					}
				}
				t.Fatalf("spinUp node %d (bootstrap=%v): %v", i, bootstrap, err)
			}
			nodes[i] = node

			_ = serverLn[i].Close()
			serverLn[i] = nil
			srv, err := server.New(server.Config{Addr: peers[i].ServerAddr, Dispatcher: node})
			if err != nil {
				_ = node.Close()
				for j := 0; j < i; j++ {
					if servers[j] != nil {
						_ = servers[j].Close()
					}
					if nodes[j] != nil {
						_ = nodes[j].Close()
					}
				}
				t.Fatalf("spinUp server %d (bootstrap=%v): %v", i, bootstrap, err)
			}
			go srv.Serve() //nolint:errcheck // Serve returns nil on clean Close
			servers[i] = srv
		}
		return nodes, servers
	}

	closeCluster := func(nodes []*Node, servers []*server.Server, cl *client.Client) {
		if cl != nil {
			_ = cl.Close()
		}
		for _, s := range servers {
			if s != nil {
				_ = s.Close()
			}
		}
		for _, nd := range nodes {
			if nd != nil {
				_ = nd.Close()
			}
		}
	}

	ctx := context.Background()

	// ── Phase A: bootstrap, write 100 keys, flush, stop ──────────────────

	nodes, servers := spinUp(true)

	// Wait for cluster to elect leaders.
	phaseACluster := &testCluster{nodes: nodes}
	phaseADeadline := time.Now().Add(30 * time.Second)
	ready := false
	for time.Now().Before(phaseADeadline) {
		if phaseACluster.allShardsHaveLeader() {
			ready = true
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if !ready {
		closeCluster(nodes, servers, nil)
		t.Fatal("Phase A: cluster did not elect leaders within deadline")
	}

	addrs := make([]string, n)
	for i := range servers {
		addrs[i] = servers[i].Addr().String()
	}
	c, err := client.New(client.Config{
		Servers:           addrs,
		MaxConnsPerServer: 8,
		MaxNotLeaderHops:  5,
	})
	if err != nil {
		closeCluster(nodes, servers, nil)
		t.Fatalf("Phase A: client: %v", err)
	}

	for i := 0; i < 100; i++ {
		k := fmt.Appendf(nil, "k%03d", i)
		if _, err := c.Call(ctx, "put", ops.EncodePutArgs(k, []byte{1}, 0)); err != nil {
			closeCluster(nodes, servers, c)
			t.Fatalf("Phase A: put %d: %v", i, err)
		}
	}

	// Wait for all replicas' FSMs to catch up before stopping.
	for _, nd := range nodes {
		waitAllApplied(t, nd)
	}

	// Give the msync ticker time to flush pages.dat to disk.
	time.Sleep(200 * time.Millisecond)

	closeCluster(nodes, servers, c)

	// ── Phase B: restart, re-elect, wait for apply ────────────────────────

	// Re-pre-bind the same addresses so the restart spinUp can hand them
	// off to the new mux/server listeners. The OS has released the ports
	// during closeCluster; rebind them now so no other process grabs them.
	rebind := func() bool {
		for i := 0; i < n; i++ {
			ln, lerr := net.Listen("tcp", raftAddrs[i])
			if lerr != nil {
				// Release any partial binds and report.
				for j := 0; j < i; j++ {
					if raftLn[j] != nil {
						_ = raftLn[j].Close()
						raftLn[j] = nil
					}
				}
				t.Logf("Phase B: rebind raft %s: %v (skipping)", raftAddrs[i], lerr)
				return false
			}
			raftLn[i] = ln
		}
		for i := 0; i < n; i++ {
			ln, lerr := net.Listen("tcp", serverAddrs[i])
			if lerr != nil {
				for j := 0; j < n; j++ {
					if raftLn[j] != nil {
						_ = raftLn[j].Close()
						raftLn[j] = nil
					}
				}
				for j := 0; j < i; j++ {
					if serverLn[j] != nil {
						_ = serverLn[j].Close()
						serverLn[j] = nil
					}
				}
				t.Logf("Phase B: rebind server %s: %v (skipping)", serverAddrs[i], lerr)
				return false
			}
			serverLn[i] = ln
		}
		return true
	}
	if !rebind() {
		t.Skip("Phase B: could not rebind ports (transient OS port reuse); skipping")
	}

	nodes2, servers2 := spinUp(false)
	t.Cleanup(func() { closeCluster(nodes2, servers2, nil) })

	phaseBCluster := &testCluster{nodes: nodes2}
	phaseBDeadline := time.Now().Add(30 * time.Second)
	ready = false
	for time.Now().Before(phaseBDeadline) {
		if phaseBCluster.allShardsHaveLeader() {
			ready = true
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if !ready {
		t.Fatal("Phase B: restarted cluster did not re-elect leaders within deadline")
	}

	for _, nd := range nodes2 {
		waitAllApplied(t, nd)
	}

	// ── Phase C: read all 100 keys via fresh client ───────────────────────

	addrs2 := make([]string, n)
	for i := range servers2 {
		addrs2[i] = servers2[i].Addr().String()
	}
	c2, err := client.New(client.Config{
		Servers:           addrs2,
		MaxConnsPerServer: 8,
		MaxNotLeaderHops:  5,
	})
	if err != nil {
		t.Fatalf("Phase C: client: %v", err)
	}
	defer func() { _ = c2.Close() }()

	for i := 0; i < 100; i++ {
		k := fmt.Appendf(nil, "k%03d", i)
		got, gerr := c2.Call(ctx, "get", ops.EncodeKeyArgs(k))
		if gerr != nil {
			t.Fatalf("Phase C: get %d: %v", i, gerr)
		}
		if !bytes.Equal(got, []byte{1}) {
			t.Fatalf("Phase C: get %d = %v, want [1]", i, got)
		}
	}
}

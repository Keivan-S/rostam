// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"fmt"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
)

func newTestNode(t *testing.T, numShards int) *Node {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	tmpDir := t.TempDir()
	cfg := Config{
		NodeID: "node1", DataDir: tmpDir,
		NumShards: numShards,
		Bootstrap: true,
		ShardCfg: shard.Config{
			NodeID: "ignored", DataDir: "ignored",
			Cache: cc, Ops: reg,
			Bootstrap:       true,
			RaftHeartbeatMs: 50, RaftElectionMs: 100, NoSync: true,
		},
		Ops: reg,
	}
	n, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = n.Close() })
	waitAllLeaders(t, n)
	return n
}

func newTestNodeAt(t *testing.T, dir string, numShards int) *Node {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	cfg := Config{
		NodeID: "node1", DataDir: dir,
		NumShards: numShards,
		Bootstrap: true,
		ShardCfg: shard.Config{
			NodeID: "ignored", DataDir: "ignored",
			Cache: cc, Ops: reg,
			Bootstrap:       true,
			RaftHeartbeatMs: 50, RaftElectionMs: 100, NoSync: true,
		},
		Ops: reg,
	}
	n, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = n.Close() })
	waitAllLeaders(t, n)
	return n
}

// waitAllLeaders blocks until every sub-shard has elected its own leader
// (single-node bootstrap → self). Required before any write Calls.
func waitAllLeaders(t testing.TB, n *Node) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		allReady := true
		for _, s := range n.shards {
			if !s.IsLeader() {
				allReady = false
				break
			}
		}
		if allReady {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("not all %d shards became leader within deadline", len(n.shards))
}

// waitAllApplied blocks until every sub-shard's FSM has applied all
// committed log entries. Needed on restart, where IsLeader returns true
// before background log-replay has reached last_log_index.
func waitAllApplied(t testing.TB, n *Node) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		allCaughtUp := true
		for _, s := range n.shards {
			if s == nil {
				continue // shard not hosted on this node (partitioned cluster)
			}
			rs := s.Stats().Raft
			applied, errA := strconv.ParseUint(rs["applied_index"], 10, 64)
			last, errL := strconv.ParseUint(rs["last_log_index"], 10, 64)
			if errA != nil || errL != nil {
				t.Fatalf("raft stats missing/unparseable keys: applied=%q last=%q (errA=%v errL=%v)", rs["applied_index"], rs["last_log_index"], errA, errL)
			}
			if applied < last {
				allCaughtUp = false
				break
			}
		}
		if allCaughtUp {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("shards did not catch up applied_index within deadline")
}

// waitClusterApplied blocks until every node hosting a shard has APPLIED the
// entries that the shard's most advanced replica (its leader) already holds.
//
// waitAllApplied cannot express that: it compares each node's own applied_index
// against that same node's own last_log_index, so a follower that has not yet
// been SENT an entry has applied == last and passes trivially. It proves "this
// node applied what it received", never "this node received what the leader
// wrote" — a vacuous barrier for any assertion about cluster-wide visibility.
//
// The per-shard target is the MAX commit_index across the shard's hosting nodes.
//
// It used to be the max last_log_index, which is not a reachable target and made
// this helper fail with a misleading message. Two reasons, both real:
//
//   - a FOLLOWER can transiently hold a HIGHER last_log_index than the current
//     leader — uncommitted entries appended in a superseded term, which the new
//     leader TRUNCATES. Nothing ever applies them, so waiting for every node to
//     reach that index waits forever;
//   - even on the leader, last_log_index counts entries that are merely
//     APPENDED. applied_index converges to the COMMITTED frontier, so a leader
//     that has appended past its commit index sets a target its peers cannot
//     satisfy until (and unless) those entries commit.
//
// commit_index is the right frontier: monotonic, never truncated, and exactly
// the bound applied_index converges to. It is snapshotted ONCE, before waiting:
// re-reading it each round would chase a log that keeps growing under background
// traffic and could never converge. Nodes that do not host a shard are skipped —
// they have no log to catch up on.
func waitClusterApplied(t testing.TB, nodes []*Node) {
	t.Helper()

	raftIndex := func(s *shard.Store, key string) uint64 {
		t.Helper()
		raw := s.Stats().Raft[key]
		v, err := strconv.ParseUint(raw, 10, 64)
		if err != nil {
			t.Fatalf("raft stats %q unparseable: %q (%v)", key, raw, err)
		}
		return v
	}

	// Snapshot the per-shard target once.
	numShards := 0
	for _, n := range nodes {
		if n != nil && len(n.shards) > numShards {
			numShards = len(n.shards)
		}
	}
	targets := make([]uint64, numShards)
	for _, n := range nodes {
		if n == nil {
			continue
		}
		for i, s := range n.shards {
			if s == nil {
				continue
			}
			if got := raftIndex(s, "commit_index"); got > targets[i] {
				targets[i] = got
			}
		}
	}

	deadline := time.Now().Add(cpuScaled(20 * time.Second))
	for {
		lagShard, lagNode, lagAt := -1, -1, uint64(0)
		for ni, n := range nodes {
			if n == nil {
				continue
			}
			for i, s := range n.shards {
				if s == nil {
					continue // not hosted here — nothing to apply
				}
				if applied := raftIndex(s, "applied_index"); applied < targets[i] {
					lagShard, lagNode, lagAt = i, ni, applied
					break
				}
				// applied_index IS NOT ENOUGH ON ITS OWN, and this is the whole
				// reason TestThreeNodeWASMRegistration flaked under load. In
				// hashicorp/raft, processLogs ENQUEUES the batch onto fsmMutateCh
				// (buffered 128) and only THEN calls setLastApplied:
				//
				//	if len(batch) != 0 { applyBatch(batch) }   // enqueue
				//	r.setLastApplied(index)                    // advance
				//
				// So applied_index advances when an entry is HANDED to the FSM
				// goroutine, not when that goroutine has RUN it. The proposing node
				// is safe — its Apply future is responded to by the FSM goroutine
				// itself, after the apply — but every OTHER node can report a
				// caught-up applied_index while its FSM has not yet executed the
				// hook. For a WASM registration that means the route gate is still
				// shut after this barrier returned, and the next call fails with
				// ErrWASMOpNotInThisGroup.
				//
				// fsm_pending is len(fsmMutateCh), so zero means nothing is queued.
				// RESIDUAL, stated because it is not zero: the FSM goroutine may
				// still be mid-batch on entries it has already dequeued. That window
				// is one batch wide rather than 128, which is what makes this a
				// barrier instead of a coin flip.
				if pending := raftIndex(s, "fsm_pending"); pending > 0 {
					lagShard, lagNode, lagAt = i, ni, raftIndex(s, "applied_index")
					break
				}
			}
			if lagShard >= 0 {
				break
			}
		}
		if lagShard < 0 {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("node %d shard %d applied_index=%d never reached the cluster-wide target %d",
				lagNode, lagShard, lagAt, targets[lagShard])
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestNodeCallPutGetRoundtrip(t *testing.T) {
	n := newTestNode(t, 4)
	if _, err := n.Call("put", ops.EncodePutArgs([]byte("k"), []byte("v"), 0)); err != nil {
		t.Fatalf("Call put: %v", err)
	}
	res, err := n.Call("get", ops.EncodeKeyArgs([]byte("k")))
	if err != nil {
		t.Fatalf("Call get: %v", err)
	}
	if !bytes.Equal(res, []byte("v")) {
		t.Fatalf("Get = %q, want v", res)
	}
}

func TestNodeRoutingDistributes(t *testing.T) {
	n := newTestNode(t, 8)
	const total = 800
	for i := range total {
		k := fmt.Appendf(nil, "k%05d", i)
		if _, err := n.Call("put", ops.EncodePutArgs(k, []byte("v"), 0)); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}
	// Check that at least 6 of the 8 shards got entries (mostly even).
	st := n.Stats()
	occupied := 0
	for _, s := range st.PerShard {
		if s.Cache.Puts > 0 {
			occupied++
		}
	}
	if occupied < 6 {
		t.Fatalf("only %d/8 shards saw writes; distribution too skewed: %+v", occupied, st.PerShard)
	}
}

func TestNodePingShardless(t *testing.T) {
	n := newTestNode(t, 4)
	res, err := n.Call("__ping__", nil)
	if err != nil {
		t.Fatalf("Call __ping__: %v", err)
	}
	if len(res) != 0 {
		t.Fatalf("__ping__ result = %v, want empty", res)
	}
}

func TestNodeUnknownOp(t *testing.T) {
	n := newTestNode(t, 4)
	_, err := n.Call("nonexistent", nil)
	if err != ErrUnknownOp {
		t.Fatalf("unknown op: err = %v, want ErrUnknownOp", err)
	}
}

func TestNodeMalformedArgsForRoutable(t *testing.T) {
	n := newTestNode(t, 4)
	// 1-byte args for "get" — extractor returns (nil, false)
	_, err := n.Call("get", []byte{0xff})
	if err != ErrNoKeyExtractor {
		t.Fatalf("malformed args: err = %v, want ErrNoKeyExtractor", err)
	}
}

func TestNodeRestartRecovers(t *testing.T) {
	reg := ops.NewRegistry()
	_ = ops.RegisterBuiltins(reg)
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	tmpDir := t.TempDir()
	cfg := Config{
		NodeID: "node1", DataDir: tmpDir,
		NumShards: 4, Bootstrap: true,
		ShardCfg: shard.Config{
			NodeID: "ignored", DataDir: "ignored",
			Cache: cc, Ops: reg,
			Bootstrap:       true,
			RaftHeartbeatMs: 50, RaftElectionMs: 100, NoSync: false, // need durability
		},
		Ops: reg,
	}

	n, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	waitAllLeaders(t, n)
	for i := range 20 {
		k := fmt.Appendf(nil, "k%02d", i)
		v := fmt.Appendf(nil, "v%02d", i)
		if _, err := n.Call("put", ops.EncodePutArgs(k, v, 0)); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}
	_ = n.Close()

	// Re-open on the same DataDir.
	cfg.Bootstrap = false // existing state present
	cfg.ShardCfg.Bootstrap = false
	n2, err := New(cfg)
	if err != nil {
		t.Fatalf("re-open: %v", err)
	}
	defer func() { _ = n2.Close() }()
	waitAllLeaders(t, n2)
	waitAllApplied(t, n2)

	for i := range 20 {
		k := fmt.Appendf(nil, "k%02d", i)
		wantV := fmt.Appendf(nil, "v%02d", i)
		got, err := n2.Call("get", ops.EncodeKeyArgs(k))
		if err != nil {
			t.Fatalf("Get %d after restart: %v", i, err)
		}
		if !bytes.Equal(got, wantV) {
			t.Fatalf("Get %d after restart = %q, want %q", i, got, wantV)
		}
	}
}

func TestNodeConcurrentCalls(t *testing.T) {
	n := newTestNode(t, 16)
	const goroutines = 16
	const iters = 50
	var wg sync.WaitGroup
	for g := range goroutines {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := range iters {
				k := fmt.Appendf(nil, "g%02d-k%05d", id, i)
				if _, err := n.Call("put", ops.EncodePutArgs(k, []byte{1}, 0)); err != nil {
					t.Errorf("put g=%d i=%d: %v", id, i, err)
					return
				}
			}
		}(g)
	}
	wg.Wait()
}

// TestNodeSinglePeerMultiNodeMode exercises the multi-node construction
// path with Peers=[self], RaftAddr="" (auto-bind to 127.0.0.1:0).
// This validates the full multi-node wiring (mux + meta-Raft + per-shard
// StreamLayer) without needing multiple processes.
func TestNodeSinglePeerMultiNodeMode(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	cfg := Config{
		NodeID:    "node1",
		DataDir:   t.TempDir(),
		NumShards: 4,
		Bootstrap: true,
		ShardCfg: shard.Config{
			NodeID: "ignored", DataDir: "ignored",
			Cache: cc, Ops: reg,
			Bootstrap:       true,
			RaftHeartbeatMs: 50, RaftElectionMs: 100, NoSync: true,
		},
		Ops:      reg,
		Peers:    []Peer{{NodeID: "node1", RaftAddr: "", ServerAddr: "127.0.0.1:7001"}},
		RaftAddr: "",
	}
	n, err := New(cfg)
	if err != nil {
		t.Fatalf("New (multi-node single-peer): %v", err)
	}
	t.Cleanup(func() { _ = n.Close() })
	if n.mux == nil || n.meta == nil {
		t.Fatal("expected mux + meta to be populated in multi-node mode")
	}
	waitAllLeaders(t, n)

	// Put-then-get roundtrip through the multi-node path.
	if _, err := n.Call("put", ops.EncodePutArgs([]byte("k"), []byte("v"), 0)); err != nil {
		t.Fatalf("Call put: %v", err)
	}
	res, err := n.Call("get", ops.EncodeKeyArgs([]byte("k")))
	if err != nil {
		t.Fatalf("Call get: %v", err)
	}
	if !bytes.Equal(res, []byte("v")) {
		t.Fatalf("Get = %q, want v", res)
	}

	// Meta-Raft state should reflect the membership we applied.
	st := n.meta.FSM.State()
	if st.NumShards != 4 || len(st.Members) != 1 || st.Members[0].NodeID != "node1" {
		t.Errorf("meta state = %+v", st)
	}
}

// newTestNodeMultiSingle builds a Node using the multi-node code path
// (cfg.Peers != nil) with a single self-peer and RaftAddr="" (auto-bind).
// This is the helper for *_MultiNodeSingle test variants that verify the
// multi-node path is behaviourally equivalent to the single-node path.
func newTestNodeMultiSingle(t *testing.T, numShards int) *Node {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	tmpDir := t.TempDir()
	cfg := Config{
		NodeID:    "node1",
		DataDir:   tmpDir,
		NumShards: numShards,
		Bootstrap: true,
		RaftAddr:  "", // auto-bind via mux
		Peers: []Peer{
			{NodeID: "node1", RaftAddr: "", ServerAddr: "127.0.0.1:1"},
		},
		ShardCfg: shard.Config{
			NodeID: "ignored", DataDir: "ignored",
			Cache: cc, Ops: reg,
			Bootstrap:       true,
			RaftHeartbeatMs: 50, RaftElectionMs: 100, NoSync: true,
		},
		Ops: reg,
	}
	n, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = n.Close() })
	waitAllLeaders(t, n)
	return n
}

func TestNodeCallPutGetRoundtrip_MultiNodeSingle(t *testing.T) {
	n := newTestNodeMultiSingle(t, 4)
	if _, err := n.Call("put", ops.EncodePutArgs([]byte("k"), []byte("v"), 0)); err != nil {
		t.Fatalf("Call put: %v", err)
	}
	res, err := n.Call("get", ops.EncodeKeyArgs([]byte("k")))
	if err != nil {
		t.Fatalf("Call get: %v", err)
	}
	if !bytes.Equal(res, []byte("v")) {
		t.Fatalf("Get = %q, want v", res)
	}
}

func TestNodeRoutingDistributes_MultiNodeSingle(t *testing.T) {
	n := newTestNodeMultiSingle(t, 8)
	const total = 800
	for i := range total {
		k := fmt.Appendf(nil, "k%05d", i)
		if _, err := n.Call("put", ops.EncodePutArgs(k, []byte("v"), 0)); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}
	st := n.Stats()
	occupied := 0
	for _, s := range st.PerShard {
		if s.Cache.Puts > 0 {
			occupied++
		}
	}
	if occupied < 6 {
		t.Fatalf("only %d/8 shards saw writes; distribution too skewed: %+v", occupied, st.PerShard)
	}
}

func TestNodePingShardless_MultiNodeSingle(t *testing.T) {
	n := newTestNodeMultiSingle(t, 4)
	res, err := n.Call("__ping__", nil)
	if err != nil {
		t.Fatalf("Call __ping__: %v", err)
	}
	if len(res) != 0 {
		t.Fatalf("__ping__ result = %v, want empty", res)
	}
}

func TestNodeUnknownOp_MultiNodeSingle(t *testing.T) {
	n := newTestNodeMultiSingle(t, 4)
	_, err := n.Call("nonexistent", nil)
	if err != ErrUnknownOp {
		t.Fatalf("unknown op: err = %v, want ErrUnknownOp", err)
	}
}

func TestNodeMalformedArgsForRoutable_MultiNodeSingle(t *testing.T) {
	n := newTestNodeMultiSingle(t, 4)
	// 1-byte args for "get" — extractor returns (nil, false)
	_, err := n.Call("get", []byte{0xff})
	if err != ErrNoKeyExtractor {
		t.Fatalf("malformed args: err = %v, want ErrNoKeyExtractor", err)
	}
}

func TestNodeConcurrentCalls_MultiNodeSingle(t *testing.T) {
	n := newTestNodeMultiSingle(t, 16)
	const goroutines = 16
	const iters = 50
	var wg sync.WaitGroup
	for g := range goroutines {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := range iters {
				k := fmt.Appendf(nil, "g%02d-k%05d", id, i)
				if _, err := n.Call("put", ops.EncodePutArgs(k, []byte{1}, 0)); err != nil {
					t.Errorf("put g=%d i=%d: %v", id, i, err)
					return
				}
			}
		}(g)
	}
	wg.Wait()
}

// TestNodeRestartRecovers_MultiNodeSingle is skipped: the auto-bind RaftAddr
// changes across close→reopen cycles, making peer address reconciliation
// non-trivial in unit-test scope. See TestThreeNodeRestart for the canonical
// multi-node restart coverage.
func TestNodeRestartRecovers_MultiNodeSingle(t *testing.T) {
	t.Skip("complex auto-bind across restart — see TestThreeNodeRestart for the multi-node restart path")
}

func TestNodeTopologyMultiNode(t *testing.T) {
	n := newTestNodeMultiSingle(t, 4)
	// waitAllLeaders already called by newTestNodeMultiSingle; Leaders
	// entries should reflect the elected self-peer.
	top, err := n.Topology()
	if err != nil {
		t.Fatalf("Topology: %v", err)
	}
	if top.NumShards != 4 {
		t.Errorf("NumShards = %d, want 4", top.NumShards)
	}
	if len(top.Members) != 1 {
		t.Errorf("Members len = %d, want 1", len(top.Members))
	}
	if len(top.Leaders) != 4 {
		t.Errorf("Leaders len = %d, want 4", len(top.Leaders))
	}
	for i, addr := range top.Leaders {
		if addr == "" {
			t.Errorf("Leaders[%d] empty; expected self ServerAddr", i)
		}
	}
}

func TestNodeTopologySingleNode(t *testing.T) {
	n := newTestNode(t, 4) // single-node path (Peers=nil)
	top, err := n.Topology()
	if err != nil {
		t.Fatalf("Topology in single-node mode: %v", err)
	}
	if top.NumShards != 4 {
		t.Errorf("NumShards = %d, want 4", top.NumShards)
	}
	// Single-node has no meta-Raft membership; Members is empty.
	if len(top.Members) != 0 {
		t.Errorf("Members = %v, want empty in single-node mode", top.Members)
	}
	// Leaders is populated from the per-shard Raft groups.
	if len(top.Leaders) != 4 {
		t.Fatalf("len(Leaders) = %d, want 4", len(top.Leaders))
	}
}

func TestNodeTopologyOpRegistered(t *testing.T) {
	n := newTestNodeMultiSingle(t, 4)
	// The op is registered on the cfg.Ops registry. Calling it via
	// n.Call dispatches shardless to shard 0 which forwards to the handler.
	result, err := n.Call("__topology__", nil)
	if err != nil {
		t.Fatalf("Call __topology__: %v", err)
	}
	top, err := ops.DecodeTopology(result)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if top.NumShards != 4 {
		t.Errorf("got NumShards=%d, want 4", top.NumShards)
	}
}

func TestNodeNumShards256Constructs(t *testing.T) {
	if testing.Short() {
		t.Skip("256-shard construction is slow under -short")
	}
	start := time.Now()
	n := newTestNode(t, 256)
	elapsed := time.Since(start)
	if budget := cpuScaled(10 * time.Second); elapsed > budget {
		t.Errorf("256-shard construction took %v, expected <%v", elapsed, budget)
	}
	_ = n
}

// waitWASMRouteGateOpen blocks until every node has opened its WASM route gate
// for opName on EVERY shard group that node hosts.
//
// This is the only NON-PROXY barrier available for "the registration has landed
// everywhere", and the distinction is not academic — it is what made
// TestThreeNodeWASMRegistration flake for as long as it did.
//
// Every index-based barrier infers FSM progress from a counter that raft moves
// for its own reasons. applied_index advances at ENQUEUE onto fsmMutateCh, not
// at apply, so a node reports caught-up while its FSM has not yet run the
// __register_wasm__ hook that opens the gate. Adding fsm_pending==0 removes the
// queued entries from that window but not the batch already dequeued.
//
// checkWASMRouteGate reads Stats().WASMGate.ProvenGroups. The hook that fills it
// is precisely the one whose completion the index barriers are guessing at. So
// this asks the gate itself, and a pass means the next Call cannot be refused
// with ErrWASMOpNotInThisGroup — not that it probably will not be.
//
// A genuine propagation defect still fails here, on the deadline, naming the
// node and the groups that never opened. Waiting for a precondition is not the
// same as tolerating its absence.
func waitWASMRouteGateOpen(t testing.TB, nodes []*Node, opName string) {
	t.Helper()

	deadline := time.Now().Add(cpuScaled(20 * time.Second))
	for {
		lagNode, missing := -1, []int(nil)
		for ni, n := range nodes {
			if n == nil {
				continue
			}
			proven := make(map[int]struct{})
			for _, g := range n.Stats().WASMGate.ProvenGroups[opName] {
				proven[g] = struct{}{}
			}
			for i, s := range n.shards {
				if s == nil {
					continue // not hosted here — the gate never consults it
				}
				if _, ok := proven[i]; !ok {
					missing = append(missing, i)
				}
			}
			if len(missing) > 0 {
				lagNode = ni
				break
			}
		}
		if lagNode < 0 {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("node %d never opened the WASM route gate for %q on hosted groups %v; "+
				"an invocation routed to any of them would be refused with ErrWASMOpNotInThisGroup",
				lagNode, opName, missing)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

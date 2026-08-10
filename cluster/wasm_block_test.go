// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
	"github.com/rostamlabs/rostam/wasm"
)

// newBlockTestNode builds a single-node cluster with numShards groups and a FAST
// apply-retry cadence, so a classRetry block is entered and re-run within
// milliseconds rather than on the production schedule.
//
// It is single-node ON PURPOSE. A node with no peers has an EMPTY fetch source
// list, so a fetch can never succeed — which is precisely the "inject a fetch
// source that never resolves" the block gates need, obtained structurally instead
// of by injecting a fake. The block therefore lasts until the test releases it,
// and the test must bound every wait on it explicitly.
func newBlockTestNode(t *testing.T, dir string, numShards int) *Node {
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
			ApplyRetryInterval: 2 * time.Millisecond,
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

// blockGroupOnMissingModule drives a marker naming a module this node does not
// hold into ONE group's log, then starts an invocation of it in that group.
//
// It returns the fingerprint the marker names and a channel carrying the
// invocation's eventual result. The invocation does NOT complete: its apply
// resolves a binding to a module that is not resident, which is classRetry.
func blockGroupOnMissingModule(t *testing.T, n *Node, group int, opName string, module []byte) (fp [32]byte, done chan error) {
	t.Helper()
	fp = ops.WASMBlobFingerprint(module)
	marker := ops.EncodeWASMRegistration(ops.WASMRegistration{
		Name:       opName,
		Kind:       ops.OpReadWrite,
		Blob:       fp,
		ExportName: "apply",
		// put.wasm writes args-minus-the-"std"-prefix as BOTH the key and the
		// value, so "the blocked entry mutated nothing" is a check on real
		// committed state rather than on a stub that writes nothing whatever it is
		// handed.
	})
	// The shard-scoped wrapper puts the marker in exactly ONE group's log, which
	// is what makes this a group-LOCAL condition and therefore what the coupling
	// gate below is able to be about.
	//
	// The handler is invoked DIRECTLY rather than through Node.Call because
	// registerAdminOps runs only on the multi-node path; Node.Call would answer
	// ErrUnknownOp on this single-node fixture. It is the same function Node.Call
	// dispatches to, so nothing about the path under test is skipped.
	if _, err := n.handleRegisterWASMShard(encodeShardScopedWASM(group, marker)); err != nil {
		t.Fatalf("drive marker into group %d: %v", group, err)
	}
	if _, _, _, ok := n.cfg.Ops.Lookup(opName); !ok {
		t.Fatalf("op %q must be REGISTERED even though its module is absent: withholding it would turn a fetchable condition into a classFatal ErrOpNotRegistered halt", opName)
	}
	done = make(chan error, 1)
	go func() {
		_, err := n.getShard(group).Call(opName, stdArgs([]byte("blocked-key")))
		done <- err
	}()
	return fp, done
}

// waitForBlock waits, with an explicit bound, until Stats reports a block on
// group. A test that waits on a block must never wait without a bound.
func waitForBlock(t *testing.T, n *Node, group int) WASMBlockedApply {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for {
		for _, b := range n.Stats().WASMBlock.Blocked {
			if b.Group == group {
				return b
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("no block reported for shard group %d within 10s; a wait that is not visible in Stats is the failure mode the whole observability half of this design exists to prevent", group)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// TestBlockedGroupDoesNotCoupleToItsNeighbours is the coupling gate, and it is
// the test that decides whether an unbounded wait is defensible at all.
//
// A block is group-LOCAL by construction (one group's log, one entry), and the
// entire argument for retrying instead of halting rests on it staying that way.
// If a blocked group could stall another group's applies, or any group's
// snapshot, or Node.Stats, then the "process-global halt for a group-local
// condition" objection to classFatal would apply to classRetry too — and the
// halt would be the better of two bad answers, because at least it is visible.
//
// THE SNAPSHOT ASSERTION IS THE SHARP ONE. cluster's snapshotWASMState and
// restoreWASMState take wasmApplyMu, and applyWASMRegistration holds it for its
// whole compile-write-install. If the block were entered while holding that
// mutex — or if the retry hook fetched inline under it, or returned still holding
// it — then EVERY group's snapshot on this node would stall behind one group's
// missing module, converting exactly the group-local condition into the
// node-global one.
//
// FALSIFICATION (performed, message recorded in the commit): making
// Node.onShardApplyRetry take wasmApplyMu and hold it across the prefetch makes
// the "group 3 could not snapshot" assertion fail on its 5s bound.
func TestBlockedGroupDoesNotCoupleToItsNeighbours(t *testing.T) {
	n := newBlockTestNode(t, t.TempDir(), 8)
	module := readTestWASM(t, "../wasm/testdata/put.wasm")

	_, blockedCall := blockGroupOnMissingModule(t, n, 7, "blocked_udf", module)
	waitForBlock(t, n, 7)

	// 1. A HEALTHY NEIGHBOUR STILL APPLIES.
	applied := make(chan error, 1)
	go func() {
		_, err := n.getShard(3).Call("put", ops.EncodePutArgs([]byte("k3"), []byte("v3"), 0))
		applied <- err
	}()
	select {
	case err := <-applied:
		if err != nil {
			t.Fatalf("group 3 apply while group 7 is blocked: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("group 3 could not apply a write while group 7 was blocked: a block must be group-LOCAL, or the whole case for retrying instead of halting collapses")
	}

	// 2. A HEALTHY NEIGHBOUR STILL SNAPSHOTS — the wasmApplyMu property.
	snapped := make(chan []byte, 1)
	go func() { snapped <- n.snapshotWASMState(3) }()
	select {
	case <-snapped:
	case <-time.After(5 * time.Second):
		t.Fatal("group 3 could not snapshot while group 7 was blocked: the block is holding wasmApplyMu, so every group's snapshot on this node is stalled behind one group's missing module")
	}

	// 3. Node.Stats STILL ANSWERS. It is read from a lock-free published
	//    snapshot precisely so "can I still see the block" never depends on the
	//    block.
	statsDone := make(chan Stats, 1)
	go func() { statsDone <- n.Stats() }()
	select {
	case st := <-statsDone:
		if st.WASMBlock.LongestBlock <= 0 {
			t.Fatal("Stats reports no LongestBlock while a group is blocked; that gauge is what an operator alerts on, because a blocked group cannot compact its Raft log")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Node.Stats did not return while a group was blocked")
	}

	// 4. And the blocked group is still blocked — i.e. none of the above
	//    accidentally released it, which would make every assertion vacuous.
	select {
	case err := <-blockedCall:
		t.Fatalf("the blocked invocation completed on its own (%v); the rest of this test proved nothing", err)
	default:
	}
}

// TestBlockedApplyCompletesWhenTheBlobIsPushedByHand pins the operator escape
// hatch, which is the runbook entry this whole design owes an operator.
//
// A blocked group is a group that has stopped applying AND stopped compacting
// its Raft log, and the recorded remedy for every other WASM disk problem
// (wasmRecoveryAdvice) is "wipe this node's data dir and rejoin" — expensive,
// slow, and unqualified data loss if this node is the last healthy replica of any
// group it hosts. __wasm_blob_put__ is a single admin call that moves one file
// and unblocks the group immediately, with no restart and no failover.
//
// It also pins the invariant's other half: while blocked the entry mutated
// NOTHING, and once released it applies EXACTLY ONCE.
func TestBlockedApplyCompletesWhenTheBlobIsPushedByHand(t *testing.T) {
	n := newBlockTestNode(t, t.TempDir(), 8)
	module := readTestWASM(t, "../wasm/testdata/put.wasm")

	fp, blockedCall := blockGroupOnMissingModule(t, n, 7, "blocked_udf", module)
	rec := waitForBlock(t, n, 7)
	if rec.Fingerprint != ops.WASMBlobHex(fp) {
		t.Fatalf("block reports fingerprint %q, want %q: the fingerprint IS the __wasm_blob_put__ argument, so a wrong one makes the escape hatch unusable", rec.Fingerprint, ops.WASMBlobHex(fp))
	}
	if rec.Op != "blocked_udf" {
		t.Fatalf("block reports op %q, want blocked_udf", rec.Op)
	}

	// Nothing was mutated: incr.wasm writes its key, and the key must be absent.
	if _, err := n.getShard(7).Get([]byte("blocked-key")); err == nil {
		t.Fatal("the blocked entry mutated state; a blocked apply never reached its handler and must leave nothing behind")
	}

	// THE ESCAPE HATCH, exactly as an operator would use it: the same admin op,
	// with the fingerprint Stats reported and the module those bytes hash to.
	if _, err := n.handleWASMBlobPut(encodeWASMBlobPut(ops.WASMBlobHex(fp), module)); err != nil {
		t.Fatalf("__wasm_blob_put__: %v", err)
	}

	select {
	case err := <-blockedCall:
		if err != nil {
			t.Fatalf("the invocation failed after the blob was pushed: %v", err)
		}
	case <-time.After(15 * time.Second):
		t.Fatal("pushing the blob by hand did not unblock the group within 15s; __wasm_blob_put__ is the documented remedy, so it has to work without a restart")
	}
	if _, err := n.getShard(7).Get([]byte("blocked-key")); err != nil {
		t.Fatalf("the entry did not apply after the block cleared: %v", err)
	}

	// The block record must be RETIRED, not left behind. A phantom block is worse
	// than no reporting at all: it is a permanent false alarm on the gauge an
	// operator is told to alert on.
	deadline := time.Now().Add(5 * time.Second)
	for {
		st := n.Stats().WASMBlock
		if len(st.Blocked) == 0 {
			if st.Total == 0 {
				t.Fatal("Total must still count the block that HAPPENED; a cleared block that leaves no trace makes a recurring block invisible")
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("a cleared block is still reported after 5s: %+v", st.Blocked)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// TestFetchIsDeduplicatedPerFingerprint pins the dedup key.
//
// One registration is broadcast to EVERY shard group, so it produces NumShards
// markers naming ONE blob, and several groups can be blocked on it at once. The
// thing being fetched is a file addressed by its content, so keying the dedup on
// the blocked (group, op) pair instead would issue 64 identical fetches of the
// same file against the same peers — 64 dial storms for one missing module.
func TestFetchIsDeduplicatedPerFingerprint(t *testing.T) {
	n := newBlockTestNode(t, t.TempDir(), 4)
	module := readTestWASM(t, "../wasm/testdata/put.wasm")
	fp := ops.WASMBlobFingerprint(module)

	// The direct path: many callers, one fingerprint.
	for range 32 {
		n.prefetchWASMBlob(fp)
	}
	if got := n.wasmFetchStarts.Load(); got != 1 {
		t.Fatalf("32 prefetches of one fingerprint started %d fetches, want exactly 1", got)
	}

	// And through the REAL seam: several groups blocking on markers that name the
	// same module. Each group's apply goroutine calls onShardApplyRetry
	// independently, on every retry, so this is where a per-(group, op) key would
	// show up.
	before := n.wasmFetchStarts.Load()
	var calls []chan error
	for _, g := range []int{0, 1, 2, 3} {
		_, done := blockGroupOnMissingModule(t, n, g, "dedup_udf_"+string(rune('a'+g)), module)
		calls = append(calls, done)
	}
	for _, g := range []int{0, 1, 2, 3} {
		waitForBlock(t, n, g)
	}
	if got := n.wasmFetchStarts.Load(); got != before {
		t.Fatalf("four groups blocked on ONE fingerprint started %d additional fetches, want 0 (the first prefetch above is still in flight and must absorb them)", got-before)
	}

	// Release everything so the node closes promptly.
	if _, err := n.handleWASMBlobPut(encodeWASMBlobPut(ops.WASMBlobHex(fp), module)); err != nil {
		t.Fatalf("__wasm_blob_put__: %v", err)
	}
	for i, done := range calls {
		select {
		case <-done:
		case <-time.After(15 * time.Second):
			t.Fatalf("group %d did not unblock within 15s", i)
		}
	}
}

// TestRestoreAndConstructionNeverBlock is the "restore must never block" gate.
//
// restoreWASMState runs INSIDE shard.New, on the FSM goroutine, during node
// construction — before the node has peers, a server, or anything to fetch from.
// A restore that waited for module bytes would deadlock construction itself, and
// would do so on the one path (InstallSnapshot catch-up) a replica joining a
// cluster depends on. The rule is therefore absolute: a restore installs MARKERS
// and kicks a prefetch; the block, if there is one, happens later, at the first
// invocation.
//
// The same applies to the disk-reload path, which is the restart-time face of the
// same state: a sidecar naming a blob this node does not hold.
//
// FALSIFICATION (performed, message recorded in the commit): making
// materializeWASMBlob's failure fatal in reloadWASMModulesFromDisk makes the
// construction half fail with "New refused to start".
func TestRestoreAndConstructionNeverBlock(t *testing.T) {
	module := readTestWASM(t, "../wasm/testdata/put.wasm")
	fp := ops.WASMBlobFingerprint(module)

	t.Run("snapshot restore naming an unavailable module", func(t *testing.T) {
		n := newBlockTestNode(t, t.TempDir(), 2)
		// A snapshot section carrying a marker for a module NO peer can serve —
		// this node has no peers at all, so the fetch can never succeed.
		st := newWASMState()
		st.installed["ghost_udf"] = installedWASM{
			reg: ops.WASMRegistration{
				Name: "ghost_udf", Kind: ops.OpReadWrite,
				Blob: fp, ExportName: "apply",
			},
			replicated: true,
			groups: map[int]wasmGroupBinding{1: newWASMGroupBinding(ops.WASMRegistration{
				Name: "ghost_udf", Kind: ops.OpReadWrite, Blob: fp, ExportName: "apply",
			})},
		}
		blob := wasmSnapshotBlob(st, 1)

		done := make(chan error, 1)
		go func() { done <- n.restoreWASMState(1, blob) }()
		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("restoreWASMState: %v", err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("restoreWASMState did not return within 5s: it runs inside shard.New, so a restore that waits for module bytes deadlocks node construction — and it does so on the very path a replica joining by InstallSnapshot depends on")
		}
		// The marker installed and the group is bound, even though the module is
		// not resident. That is the point: the route gate opens, the op is
		// lookup-able, and only an actual invocation blocks.
		if _, ok := n.wasmState.installed["ghost_udf"]; !ok {
			t.Fatal("the restore installed nothing; a marker whose blob is absent must still install")
		}
		if n.wasmRT.HasModule(wasm.ModuleIDForBlob(fp, "apply", 0)) {
			t.Fatal("the module must NOT be resident: nothing supplied its bytes")
		}
	})

	t.Run("construction over a sidecar whose blob is absent", func(t *testing.T) {
		dir := t.TempDir()
		// Write a replicated sidecar naming a blob that is not on disk — exactly
		// what a node that applied a marker and was restarted before its fetch
		// finished has.
		if err := writeWASMSidecar(dir, ops.WASMRegistration{
			Name: "ghost_udf", Kind: ops.OpReadWrite, Blob: fp, ExportName: "apply", Epoch: 1,
		}, map[int]wasmGroupBinding{0: newWASMGroupBinding(ops.WASMRegistration{
			Name: "ghost_udf", Kind: ops.OpReadWrite, Blob: fp, ExportName: "apply", Epoch: 1,
		})}); err != nil {
			t.Fatalf("writeWASMSidecar: %v", err)
		}
		if _, err := os.Stat(filepath.Join(dir, "wasm", "ghost_udf.json")); err != nil {
			t.Fatalf("sidecar not written: %v", err)
		}

		built := make(chan *Node, 1)
		go func() { built <- newBlockTestNode(t, dir, 2) }()
		var n *Node
		select {
		case n = <-built:
		case <-time.After(30 * time.Second):
			t.Fatal("node construction did not complete within 30s over a sidecar whose blob is absent; that state is ORDINARY now (a marker names its module), and refusing to start on it takes down every group on an otherwise-healthy node")
		}
		// It SERVES: the op is registered, and unrelated traffic is unaffected.
		if _, _, _, ok := n.cfg.Ops.Lookup("ghost_udf"); !ok {
			t.Fatal("the op must be registered after the reload: withholding it recreates the classFatal ErrOpNotRegistered halt the reload exists to prevent")
		}
		if _, err := n.Call("put", ops.EncodePutArgs([]byte("ok"), []byte("v"), 0)); err != nil {
			t.Fatalf("the node does not serve other ops after starting with a missing blob: %v", err)
		}
	})
}

// TestPushDurabilityFloorRefusesWhenTheMajorityIsUnreachable is the floor gate.
//
// WITHOUT A FLOOR the registration succeeds with the module on exactly ONE node.
// That was harmless while the marker carried the bytes — every replica could
// re-derive them from the entry — and it is the blocking condition now: if that
// node dies, no node in the cluster can serve the blob, every group's fetch fails
// forever, and every group that reaches an invocation blocks permanently with no
// source. The registration would have been accepted, replicated and made
// permanent while being impossible to execute.
//
// FALSIFICATION (performed, message recorded in the commit): deleting the floor
// check in pushWASMBlob makes this fail with "the push succeeded with the module
// on exactly one node".
func TestPushDurabilityFloorRefusesWhenTheMajorityIsUnreachable(t *testing.T) {
	module := readTestWASM(t, "../wasm/testdata/put.wasm")
	r := ops.WASMRegistration{
		Name: "floor_udf", Kind: ops.OpReadWrite,
		Blob: ops.WASMBlobFingerprint(module), ExportName: "apply",
	}

	t.Run("three members, two unreachable", func(t *testing.T) {
		// 127.0.0.1:1 and :2 are addresses nothing listens on, so both legs fail
		// to dial: this node is the only holder, 1 of 3, short of the majority.
		n := &Node{cfg: Config{
			NodeID: "n1", DataDir: t.TempDir(),
			Peers: []Peer{
				{NodeID: "n1", ServerAddr: "127.0.0.1:0"},
				{NodeID: "n2", ServerAddr: "127.0.0.1:1"},
				{NodeID: "n3", ServerAddr: "127.0.0.1:2"},
			},
		}}
		_, err := n.pushWASMBlob(r, module)
		if err == nil {
			t.Fatal("the push succeeded with the module on exactly one node: a marker names its module, so a version no majority holds becomes unfetchable the moment that node dies, and every shard group blocks forever with no source")
		}
		if !strings.Contains(err.Error(), "short of the majority") {
			t.Fatalf("refusal must name the floor so an operator knows the cluster was too degraded rather than the payload bad; got: %v", err)
		}
		if !strings.Contains(err.Error(), ops.WASMRegistrationRefusedMsg) {
			t.Fatalf("refusal must carry %q so server.clientFacingErr keeps it unredacted across the RPC boundary; got: %v", ops.WASMRegistrationRefusedMsg, err)
		}
		if got := n.wasmBlobFloorFailures.Load(); got != 1 {
			t.Fatalf("wasmBlobFloorFailures = %d, want 1", got)
		}
	})

	t.Run("single member is its own majority", func(t *testing.T) {
		// A one-member cluster has no targets at all, and self is a strict
		// majority of {self}. Refusing here would make single-node deployments
		// unable to register anything.
		n := &Node{cfg: Config{NodeID: "n1", DataDir: t.TempDir()}}
		if _, err := n.pushWASMBlob(r, module); err != nil {
			t.Fatalf("a single-node cluster must satisfy its own floor: %v", err)
		}
	})
}

// TestPushDurabilityFloorIgnoresAcksFromNonMembers gates the rule the two floor
// subtests above structurally cannot reach.
//
// Both of those build a Node with n.meta == nil, so wasmMemberSet takes the
// cfg.Peers fallback and the AUTHORITATIVE-Members arm — the one the isMember
// check in pushWASMBlob lives on — is never executed. Deleting that check leaves
// them green while a node the cluster has REMOVED can satisfy a floor stated over
// live members: it is still a push target (deliberately — the union exists so a
// stale name is never silently skipped), it still acks, and counting that ack is
// counting a holder that is not in the set the fetcher searches.
//
// The arithmetic is chosen so the check is the only thing that decides the
// verdict: members = {n1, gone3, gone4} (3), self is the one real holder (1), and
// 1*2 <= 3 refuses. Count the removed node's ack and holders is 2, 2*2 > 3, and
// the registration is accepted with the module on two nodes, one of which is not
// a member.
//
// It needs a REAL peer, because the whole point is an ack that actually happened.
//
// FALSIFICATION (performed, message recorded in the commit): deleting the
// `if _, isMember := members[m.nodeID]; isMember` guard in pushWASMBlob makes
// this fail at "the push succeeded by counting an ack from a node that is not a
// member".
func TestPushDurabilityFloorIgnoresAcksFromNonMembers(t *testing.T) {
	module := readTestWASM(t, "../wasm/testdata/put.wasm")
	tc := newTestCluster(t, 2, 1)
	n1, n2Addr := tc.nodes[0], tc.peers[1].ServerAddr

	// A synthetic membership in which n2 — which is real, listening, and WILL ack
	// — has been removed, alongside two members that exist only in the table.
	f := NewMetaFSM()
	data, err := encodeLogEntry(LogEntry{
		Op: OpSetMembers,
		Members: []Peer{
			{NodeID: "n1", ServerAddr: tc.peers[0].ServerAddr},
			{NodeID: "gone3", ServerAddr: ""},
			{NodeID: "gone4", ServerAddr: ""},
		},
		NumShards: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := f.Apply(&hraft.Log{Data: data}); got != nil {
		t.Fatalf("apply OpSetMembers: %v", got)
	}

	realMeta, realPeers := n1.meta, n1.cfg.Peers
	// cfg.Peers is trimmed to the single removed node so that (a) it is still a
	// push target through the union, and (b) metaReadBarrier's no-meta-peers arm
	// applies — this synthetic MetaRaft has no Raft to barrier against.
	n1.meta = &MetaRaft{FSM: f}
	n1.cfg.Peers = []Peer{{NodeID: "n2", ServerAddr: n2Addr}}
	defer func() { n1.meta, n1.cfg.Peers = realMeta, realPeers }()

	r := ops.WASMRegistration{
		Name: "nonmember_udf", Kind: ops.OpReadWrite,
		Blob: ops.WASMBlobFingerprint(module), ExportName: "apply",
	}
	_, err = n1.pushWASMBlob(r, module)
	if err == nil {
		t.Fatal("the push succeeded by counting an ack from a node that is not a member: a removed node holding the bytes " +
			"is not a holder the floor may count — it is not in the member set the fetcher searches, so a majority of live " +
			"members can contain no holder at all")
	}
	if !strings.Contains(err.Error(), "short of the majority") {
		t.Fatalf("the refusal must name the floor; got: %v", err)
	}
	// And it must be the FLOOR that refused, not the push failing for some other
	// reason — otherwise this test would pass on a broken n2 rather than on the rule.
	if !strings.Contains(err.Error(), "reached only 1 of 3") {
		t.Fatalf("the refusal must report 1 holder of 3 members (self only); got: %v", err)
	}
}

// TestABlobAlreadyOnDiskClearsTheBlockWithNoPeer is the local-disk fetch leg.
//
// A block is a RUNTIME residency condition, not a disk one: resolveModuleForInvoke
// asks the runtime because it runs on an apply goroutine and must never do I/O. So
// a blob that reached disk but was never compiled in blocks exactly like an absent
// one, and that state is reachable by design — applyWASMRegistration treats a
// materializeWASMBlob failure as a residency condition and continues, and a reload
// can bind a group to a version it did not instantiate.
//
// This node is single-node, so its fetch source list is EMPTY: no peer can ever
// serve it. If the bytes on its own disk are not a source, nothing short of an
// operator can clear this block — on a node that already holds the exact file it
// is waiting for.
//
// FALSIFICATION (performed, message recorded in the commit): removing the local
// leg from fetchWASMBlobOnce makes this fail at "the block did not clear within
// 20s on a node whose data dir holds the exact module it is waiting for".
func TestABlobAlreadyOnDiskClearsTheBlockWithNoPeer(t *testing.T) {
	module := readTestWASM(t, "../wasm/testdata/put.wasm")
	n := newBlockTestNode(t, t.TempDir(), 1)
	_, done := blockGroupOnMissingModule(t, n, 0, "disk_udf", module)
	waitForBlock(t, n, 0)

	// The bytes land on disk WITHOUT going through any arrival path — exactly what
	// a transient materialize failure or a skipped reload leaves behind.
	if _, err := writeWASMBlob(n.cfg.DataDir, module); err != nil {
		t.Fatalf("writeWASMBlob: %v", err)
	}

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("the blocked invocation failed after its module reached this node's disk: %v", err)
		}
	case <-time.After(20 * time.Second):
		t.Fatal("the block did not clear within 20s on a node whose data dir holds the exact module it is waiting for: " +
			"with no peer able to serve it, the local store is the only source there is")
	}
}

// TestAnArrivingBlobDoesNotStallEveryGroupsSnapshot is the mirror of the coupling
// gate, aimed at the code that CLEARS a block rather than the block itself.
//
// installArrivedWASMBlob runs off the apply path, so holding wasmApplyMu across
// its work deadlocks nothing — which is exactly why it is easy to miss. What it
// does do is stall snapshotWASMState and restoreWASMState, which take the same
// mutex, for as long as reading and compiling a module takes. A multi-megabyte
// wasm.Compile is tens to hundreds of milliseconds, and it would be paid by EVERY
// group on the node, for a condition belonging to one arriving blob. That is the
// group-local-becomes-node-global escalation this whole stage exists to avoid.
//
// The runtime's module-table write lock stands in for the compile: it is the one
// part of materializeWASMBlob a test can hold open deterministically.
//
// FALSIFICATION (performed, message recorded in the commit): restoring the
// function-wide `defer n.wasmApplyMu.Unlock()` makes this fail at
// "snapshotWASMState did not complete within 5s while a blob was being installed".
func TestAnArrivingBlobDoesNotStallEveryGroupsSnapshot(t *testing.T) {
	module := readTestWASM(t, "../wasm/testdata/put.wasm")
	n := newBlockTestNode(t, t.TempDir(), 1)
	fp, done := blockGroupOnMissingModule(t, n, 0, "stall_udf", module)
	waitForBlock(t, n, 0)

	// Freeze the runtime's module table BEFORE the bytes exist, so the installer
	// cannot slip past it.
	release := n.wasmRT.HoldModuleTableForTest()
	var releaseOnce sync.Once
	releaseTable := func() { releaseOnce.Do(release) }
	defer releaseTable()

	if _, err := writeWASMBlob(n.cfg.DataDir, module); err != nil {
		t.Fatalf("writeWASMBlob: %v", err)
	}
	installing := make(chan struct{})
	go func() { defer close(installing); n.installArrivedWASMBlob(fp) }()
	// Let the installer reach the runtime. Under a function-wide lock it is holding
	// wasmApplyMu from its very first statement, so any nonzero delay is enough for
	// the falsification; this is only about not asserting before it has started.
	time.Sleep(100 * time.Millisecond)

	snapped := make(chan struct{})
	go func() { defer close(snapped); n.snapshotWASMState(0) }()
	select {
	case <-snapped:
	case <-time.After(5 * time.Second):
		releaseTable()
		t.Fatal("snapshotWASMState did not complete within 5s while a blob was being installed: an ARRIVING blob is " +
			"holding wasmApplyMu across its read and compile, so every group on this node has stopped being able to " +
			"snapshot — and therefore to compact its Raft log — for the duration")
	}

	releaseTable()
	select {
	case <-installing:
	case <-time.After(10 * time.Second):
		t.Fatal("installArrivedWASMBlob did not finish within 10s of the module table being released")
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("the blocked invocation failed after its module was installed: %v", err)
		}
	case <-time.After(20 * time.Second):
		t.Fatal("the block did not clear within 20s of the module being installed")
	}
}

// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/wasm"
)

// routableWASMPayload builds a __register_wasm__ payload for a ROUTABLE op —
// the only shape in which any of this matters, since a shardless WASM op is
// invoked from group 0 alongside its own registration.
//
// It is the CLIENT-EDGE frame, so it carries the marker AND the module
// (ops.EncodeWASMRegistrationRequest): Node.Call's __register_wasm__ intercept
// pushes the bytes to the cluster and only then broadcasts the bare marker into
// every group's log. A test that hands Node.Call a bare marker is exercising a
// frame no client ever sends.
func routableWASMPayload(t *testing.T, name string, epoch uint64) []byte {
	t.Helper()
	module := readIncrWASM(t)
	return ops.EncodeWASMRegistrationRequest(ops.WASMRegistration{
		Name:       name,
		Kind:       ops.OpReadWrite,
		Blob:       ops.WASMBlobFingerprint(module),
		ExportName: "apply",
		Epoch:      epoch,
	}, module)
}

// provenGroups reads the shard groups this node believes carry name's
// registration, straight out of the authoritative state (not the published
// snapshot, so a test failure points at the bookkeeping rather than at
// publication).
func provenGroups(t *testing.T, n *Node, name string) map[int]wasmGroupBinding {
	t.Helper()
	n.wasmApplyMu.Lock()
	defer n.wasmApplyMu.Unlock()
	return n.wasmState.installed[name].groups
}

// detachGroup hides shard group idx from this node's routing and returns a
// function that puts it back. It stands in for the two situations that leave a
// node hosting a group whose log has no registration:
//
//   - during the detached window, a broadcast leg for idx fails and the other
//     groups still accept — a PARTIAL broadcast, which needs no concurrency and
//     no injected fault to occur in production (an election, a slow peer, a
//     timeout);
//   - after reattachment, the node hosts and LEADS a group whose log never
//     received the registration — which is also the state AddShardOwner leaves a
//     node in, and the state in which a proposal into that group would be a
//     shard-wide outage.
//
// It mutates the same slot AddShardOwner/RemoveShardOwner mutate, under the same
// lock, so the gate cannot tell the difference: checkWASMRouteGate asks getShard.
func detachGroup(t *testing.T, n *Node, idx int) func() {
	t.Helper()
	n.shardMu.Lock()
	s := n.shards[idx]
	n.shards[idx] = nil
	n.shardMu.Unlock()
	if s == nil {
		t.Fatalf("precondition: node does not host shard group %d", idx)
	}
	return func() {
		n.shardMu.Lock()
		n.shards[idx] = s
		n.shardMu.Unlock()
	}
}

// TestPartialWASMBroadcastDoesNotPoisonTheStarvedGroup is the regression gate
// for the SECOND failure this branch created, the one that needs no race at all.
//
// broadcastWASMRegistration deliberately attempts every group even after one
// fails, and the groups that accepted KEEP the registration. So after a partial
// failure the op is live and routable on any node that applied any group's copy,
// while some group has no registration in its log at all. Once
// shard.classifyApplyErr treats ErrOpNotRegistered as classFatal, an
// invocation routed into that group's log HALTS every replica of the group — a
// shard-wide outage produced by an ordinary partial registration.
//
// Under the route gate the same partial failure costs a client-visible error on
// the keys that route to the starved group, and nothing else: the entry never
// enters that group's log, and every other group keeps serving.
func TestPartialWASMBroadcastDoesNotPoisonTheStarvedGroup(t *testing.T) {
	const numShards, starved = 4, 2
	n := newTestNode(t, numShards)
	waitAllApplied(t, n)

	reattach := detachGroup(t, n, starved)
	_, err := n.Call(wasmRegisterOpName, routableWASMPayload(t, "wasm_incr", 1))
	reattach()
	if err == nil {
		t.Fatal("a broadcast that could not reach one group must report the failure")
	}
	if !strings.Contains(err.Error(), "shard 2") {
		t.Errorf("broadcast error does not name the starved group: %v", err)
	}

	// The op IS registered. That is not an oversight, it is required: the groups
	// that DID accept the entry have it in their logs, and their replicas must be
	// able to look the op up to apply it. Withholding the registry entry is the
	// design that turns a lagging replica into a halt.
	if _, _, _, ok := n.cfg.Ops.Lookup("wasm_incr"); !ok {
		t.Fatal("the op must be in the registry after a partial broadcast: the groups that accepted the entry have to apply it")
	}

	proven := provenGroups(t, n, "wasm_incr")
	if _, ok := proven[starved]; ok {
		t.Errorf("group %d is recorded as carrying the registration it never received (proven=%v)", starved, sortedGroups(proven))
	}
	for _, g := range []int{0, 1, 3} {
		if _, ok := proven[g]; !ok {
			t.Errorf("group %d accepted the registration but is not recorded (proven=%v)", g, sortedGroups(proven))
		}
	}

	// The starved group answers with a client-visible, retryable error instead of
	// swallowing an entry that would halt its replicas.
	_, err = n.Call("wasm_incr", keyForShard(t, starved, numShards))
	if !errors.Is(err, ErrWASMOpNotInThisGroup) {
		t.Fatalf("invocation routed to the starved group: got %v, want ErrWASMOpNotInThisGroup", err)
	}
	// It stays client-facing through the redaction classifier (server.clientFacingErr
	// and httpapi.statusForError both key off this substring), so the caller learns
	// it should retry rather than seeing a generic internal error.
	if !strings.Contains(err.Error(), "op not registered") {
		t.Errorf("gate error would be redacted to a generic internal error: %q", err.Error())
	}

	// And the partial failure is confined to the one group: every other group
	// serves the op normally.
	for _, g := range []int{0, 1, 3} {
		if _, err := n.Call("wasm_incr", keyForShard(t, g, numShards)); err != nil {
			t.Errorf("group %d must keep serving after another group was starved: %v", g, err)
		}
	}
}

// TestWASMInvocationIsNotProposedIntoAGroupWithoutTheRegistration is the
// ORDERING gate, asserted where ordering actually lives: the group's Raft log.
//
// The invariant is that an invocation entry for op X may enter group g's log only
// above a registration for X in the SAME log. Replicas apply one log in index
// order, so that is exactly what guarantees every replica of g has X registered
// before it applies an invocation of it — and therefore that a classFatal
// ErrOpNotRegistered means genuine divergence and nothing else.
//
// The node here LEADS the starved group, so nothing but the gate stands between
// the Call and an append. With the gate removed the assertion below fails on
// last_log_index: the entry lands, and on a real replica set it halts every
// replica of that group.
func TestWASMInvocationIsNotProposedIntoAGroupWithoutTheRegistration(t *testing.T) {
	const numShards, starved = 4, 2
	n := newTestNode(t, numShards)
	waitAllApplied(t, n)

	reattach := detachGroup(t, n, starved)
	if _, err := n.Call(wasmRegisterOpName, routableWASMPayload(t, "wasm_incr", 1)); err == nil {
		t.Fatal("precondition: the broadcast leg for the detached group must fail")
	}
	reattach()

	if !n.shardIsLeader(starved) {
		t.Fatalf("precondition: this node must LEAD group %d, or the propose would be refused for an unrelated reason", starved)
	}
	before := lastLogIndexes(t, n)[starved]

	// Errorf, not Fatalf: the log-index assertion below is the one that matters
	// and it must run even when the error type is wrong.
	_, err := n.Call("wasm_incr", keyForShard(t, starved, numShards))
	if !errors.Is(err, ErrWASMOpNotInThisGroup) {
		t.Errorf("Call into a group whose log lacks the registration: got %v, want ErrWASMOpNotInThisGroup", err)
	}

	if got := lastLogIndexes(t, n)[starved]; got != before {
		t.Fatalf("an invocation entered group %d's Raft log without a registration below it (last_log_index %d -> %d); every replica of that group that applies it and has not learned the op elsewhere halts under classFatal",
			starved, before, got)
	}
}

// TestWASMRouteGateOpensForEveryGroupOnASuccessfulBroadcast is the anti-wedge
// gate. A gate that never opens is as much an outage as the halt it replaces, so
// the normal path — every group accepts — must leave every hosted group proven
// and every key invocable, with no retry loop and no waiting.
func TestWASMRouteGateOpensForEveryGroupOnASuccessfulBroadcast(t *testing.T) {
	const numShards = 4
	n := newTestNode(t, numShards)
	waitAllApplied(t, n)

	if _, err := n.Call(wasmRegisterOpName, routableWASMPayload(t, "wasm_incr", 1)); err != nil {
		t.Fatalf("register: %v", err)
	}

	proven := provenGroups(t, n, "wasm_incr")
	if len(proven) != numShards {
		t.Fatalf("proven groups = %v, want all %d: the gate is wedged shut for the rest", sortedGroups(proven), numShards)
	}
	for i := 0; i < numShards; i++ {
		if _, err := n.Call("wasm_incr", keyForShard(t, i, numShards)); err != nil {
			t.Errorf("key routing to group %d: %v", i, err)
		}
	}
}

// TestWASMOpIsRegisteredAfterTheFirstGroupApplies pins the half of the design
// that is easy to get backwards, and that a plausible-sounding alternative gets
// backwards: the ops REGISTRY must not be gated.
//
// The alternative — install the op into cfg.Ops only once the node has applied
// the registration on every group it hosts — is unsafe, because the registry is
// what the FSM looks an op up in to APPLY a committed entry. A node hosting many
// groups can be caught up on group 5 (whose log already holds the registration,
// put there by a node that hosts fewer groups and therefore "activated" earlier)
// while still behind on group 63. Its group-5 FSM then meets an invocation, finds
// no handler, and halts under classFatal — with the registration already applied
// in its own state.
//
// So: one group applying is enough to make the op APPLICABLE everywhere on this
// node. What stays per-group is only the right to PROPOSE.
func TestWASMOpIsRegisteredAfterTheFirstGroupApplies(t *testing.T) {
	const numShards = 4
	n := newTestNode(t, numShards)
	waitAllApplied(t, n)

	// Exactly the state a node is in after one group's entry has applied and the
	// rest are still in flight.
	n.wasmApplyMu.Lock()
	err := applyWASMRegistration(n.cfg.DataDir, n.wasmRT, n.cfg.Ops, n.wasmState,
		ops.WASMRegistration{
			Name: "wasm_incr", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(readIncrWASM(t)),
			ExportName: "apply", Epoch: 1,
		}, 0, nil)
	n.publishWASMGateLocked()
	n.wasmApplyMu.Unlock()
	if err != nil {
		t.Fatalf("applyWASMRegistration: %v", err)
	}

	if _, _, _, ok := n.cfg.Ops.Lookup("wasm_incr"); !ok {
		t.Fatal("the op is not in the registry after one group applied it: this node would halt (classFatal ErrOpNotRegistered) on the next invocation replicated into a group that DOES have the registration")
	}
	// ...while the groups that have not applied it are still closed to proposals.
	for _, g := range []int{1, 2, 3} {
		if _, err := n.Call("wasm_incr", keyForShard(t, g, numShards)); !errors.Is(err, ErrWASMOpNotInThisGroup) {
			t.Errorf("group %d has not applied the registration but accepted an invocation: %v", g, err)
		}
	}
}

// TestWASMRouteGateIgnoresGroupsThisNodeDoesNotHost pins the other direction: a
// node that does not host a group cannot propose into its log, so gating it there
// would refuse traffic for a group that is perfectly healthy while protecting
// nothing. The unhosted call must fall through to forwarding, where the OWNER —
// which does hold the evidence — applies the same check.
//
// This is also the ZERO-GROUP node's behavior in full: such a node hosts nothing,
// so nothing is ever gated on it and every Call is forwarded to an owner.
func TestWASMRouteGateIgnoresGroupsThisNodeDoesNotHost(t *testing.T) {
	const numShards = 4
	n := newTestNode(t, numShards)
	waitAllApplied(t, n)
	if _, err := n.Call(wasmRegisterOpName, routableWASMPayload(t, "wasm_incr", 1)); err != nil {
		t.Fatalf("register: %v", err)
	}

	// Forget that group 2 was ever proven, then stop hosting it: the node now has
	// no evidence about group 2 AND cannot propose into it.
	n.wasmApplyMu.Lock()
	in := n.wasmState.installed["wasm_incr"]
	in.groups = map[int]wasmGroupBinding{0: in.groups[0], 1: in.groups[1], 3: in.groups[3]}
	n.wasmState.installed["wasm_incr"] = in
	n.publishWASMGateLocked()
	n.wasmApplyMu.Unlock()

	if err := n.checkWASMRouteGate("wasm_incr", ops.OpReadWrite, 2); !errors.Is(err, ErrWASMOpNotInThisGroup) {
		t.Fatalf("precondition: a HOSTED, unproven group must be gated, got %v", err)
	}
	reattach := detachGroup(t, n, 2)
	defer reattach()
	if err := n.checkWASMRouteGate("wasm_incr", ops.OpReadWrite, 2); err != nil {
		t.Fatalf("a group this node does not host must not be gated locally — it is forwarded, and the owner applies its own gate: %v", err)
	}

	// A node hosting NO group is gated on nothing at all.
	var restore []func()
	for i := 0; i < numShards; i++ {
		if n.getShard(i) != nil {
			restore = append(restore, detachGroup(t, n, i))
		}
	}
	for _, f := range restore {
		defer f()
	}
	for i := 0; i < numShards; i++ {
		if err := n.checkWASMRouteGate("wasm_incr", ops.OpReadWrite, i); err != nil {
			t.Errorf("a node hosting zero groups must gate nothing (group %d): %v", i, err)
		}
	}
}

// TestWASMRouteGateDoesNotGateReadOnlyOps pins the scope of the gate. The
// invariant is about what enters a Raft log; shard.Store.Call serves an
// OpReadOnly op from local state without proposing anything, so gating it would
// deny reads that cannot possibly halt a replica.
func TestWASMRouteGateDoesNotGateReadOnlyOps(t *testing.T) {
	n := newTestNode(t, 4)
	waitAllApplied(t, n)
	n.wasmApplyMu.Lock()
	n.wasmState.installed["ro_op"] = installedWASM{replicated: true, groups: nil}
	n.publishWASMGateLocked()
	n.wasmApplyMu.Unlock()

	if err := n.checkWASMRouteGate("ro_op", ops.OpReadOnly, 1); err != nil {
		t.Errorf("a read-only op proposes nothing and must not be gated: %v", err)
	}
	if err := n.checkWASMRouteGate("ro_op", ops.OpReadWrite, 1); !errors.Is(err, ErrWASMOpNotInThisGroup) {
		t.Errorf("precondition: the same op as OpReadWrite must be gated, got %v", err)
	}
}

// TestWASMProvenGroupsSurviveARestart is the deadlock gate for the restart path.
//
// In durable mode fsm.Apply SKIPS every entry at or below the recorded applied
// index, so the __register_wasm__ entries that proved a node's groups are never
// replayed after a restart. If the proof is not persisted, a restarted node
// refuses to propose the op into ANY group forever — an outage the operator
// cannot clear by waiting, which is precisely the "gate wedged shut" failure.
//
// The proof therefore rides in the sidecar next to the module, and the group set
// is rewritten as it grows (most groups take applyWASMRegistration's
// already-installed path, which installs nothing).
func TestWASMProvenGroupsSurviveARestart(t *testing.T) {
	dir := t.TempDir()
	rt, err := wasm.NewRuntime()
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	t.Cleanup(func() { _ = rt.Close() })
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	st := newWASMState()
	seedWASMBlob(t, dir, readIncrWASM(t))
	r := ops.WASMRegistration{
		Name: "wasm_incr", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(readIncrWASM(t)),
		ExportName: "apply", Epoch: 1,
	}
	// Group 0 installs; groups 2 and 3 only add evidence.
	for _, g := range []int{0, 2, 3} {
		if err := applyWASMRegistration(dir, rt, reg, st, r, g, nil); err != nil {
			t.Fatalf("applyWASMRegistration(group %d): %v", g, err)
		}
	}

	// A fresh process: new runtime, new registry, new state, same data dir.
	rt2, err := wasm.NewRuntime()
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	t.Cleanup(func() { _ = rt2.Close() })
	reg2 := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg2); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	st2 := newWASMState()
	if err := reloadWASMModulesFromDisk(dir, rt2, reg2, st2, nil); err != nil {
		t.Fatalf("reloadWASMModulesFromDisk: %v", err)
	}

	in, ok := st2.installed["wasm_incr"]
	if !ok {
		t.Fatal("the module did not reload from disk")
	}
	if got, want := sortedGroups(in.groups), []int{0, 2, 3}; len(got) != len(want) {
		t.Fatalf("proven groups after restart = %v, want %v: the node would refuse to propose the op into any group, forever", got, want)
	} else {
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("proven groups after restart = %v, want %v", got, want)
			}
		}
	}
	// The op is registered on reload regardless of the proof, because the FSM
	// needs it to APPLY entries replayed out of any group's log.
	if _, _, _, ok := reg2.Lookup("wasm_incr"); !ok {
		t.Error("a reloaded module must be registered: withholding it would halt the replay of any group whose log holds an invocation")
	}
	if _, err := os.Stat(filepath.Join(dir, "wasm", "wasm_incr.json")); err != nil {
		t.Errorf("sidecar missing: %v", err)
	}
}

// TestWASMRouteGateAfterAddShardOwner drives the PLACEMENT case through the real
// rebalancing primitive rather than a stand-in.
//
// A node can gain a shard group long after it has been serving a WASM op:
// AddShardOwner creates the store in join mode and the group catches up
// afterwards. Two things must hold at that moment, and they pull in opposite
// directions:
//
//   - the node must NOT propose invocations into the newly gained group until it
//     knows that group's log carries the registration. It has applied none of
//     that group's log, so it knows nothing;
//   - the op must NOT be removed from the registry to achieve that. Every group
//     the node ALREADY hosts is still replicating invocations to it, and a
//     missing registry entry turns each of those into a classFatal halt. The
//     "deactivate on placement change" reflex is the same trap as gating the
//     registry, reached from the other side.
//
// The gate resolves this by asking getShard at CALL time and never touching the
// registry, so gaining a group is self-limiting and self-healing: refused until
// the joining store catches up (by snapshot, which carries the per-group flag, or
// by log replay, which applies the entry), then served.
func TestWASMRouteGateAfterAddShardOwner(t *testing.T) {
	const numShards, gained = 3, 2
	n := newTestNodeMultiSingle(t, numShards)
	waitAllApplied(t, n)

	// Stop hosting the group, so the broadcast cannot reach it and the node ends
	// up serving the op with that group starved — the state a partial broadcast
	// leaves behind.
	if err := n.RemoveShardOwner(gained); err != nil {
		t.Fatalf("RemoveShardOwner: %v", err)
	}
	if _, err := n.Call(wasmRegisterOpName, routableWASMPayload(t, "wasm_incr", 1)); err == nil {
		t.Fatal("precondition: the broadcast leg for the unhosted group must fail")
	}
	if _, ok := provenGroups(t, n, "wasm_incr")[gained]; ok {
		t.Fatalf("group %d was never reached but is recorded as carrying the registration", gained)
	}
	if _, err := n.Call("wasm_incr", keyForShard(t, 0, numShards)); err != nil {
		t.Fatalf("precondition: the op must be serving on a proven group: %v", err)
	}

	// Now gain the group.
	if err := n.AddShardOwner(gained); err != nil {
		t.Fatalf("AddShardOwner: %v", err)
	}
	if n.getShard(gained) == nil {
		t.Fatalf("precondition: the node must host group %d after AddShardOwner", gained)
	}

	if _, err := n.Call("wasm_incr", keyForShard(t, gained, numShards)); !errors.Is(err, ErrWASMOpNotInThisGroup) {
		t.Errorf("invocation into a group gained after registration: got %v, want ErrWASMOpNotInThisGroup", err)
	}
	// The op survived the placement change in the registry...
	if _, _, _, ok := n.cfg.Ops.Lookup("wasm_incr"); !ok {
		t.Fatal("gaining a group deregistered the op: every group this node already hosts would now halt on the next invocation replicated into it")
	}
	// ...and the groups that were already proven keep serving.
	for _, g := range []int{0, 1} {
		if _, err := n.Call("wasm_incr", keyForShard(t, g, numShards)); err != nil {
			t.Errorf("group %d stopped serving after an unrelated group was gained: %v", g, err)
		}
	}
}

// TestWASMSnapshotMarksOnlyTheSnapshottedGroup pins the exactness the
// AddShardOwner-joined path depends on in BOTH directions.
//
// A joining replica catches up by InstallSnapshot, so the snapshot is where it
// learns everything. It must carry EVERY registration — a module it cannot look
// up is a classFatal halt on the first invocation left in the log it is about to
// replay — but it must claim group membership for only the registrations the
// snapshotted group's log actually carries. Inflating that claim would hand the
// joiner permission to propose invocations into a group that never received the
// registration, which is the failure the gate exists to prevent.
func TestWASMSnapshotMarksOnlyTheSnapshottedGroup(t *testing.T) {
	everywhere := ops.WASMRegistration{Name: "everywhere", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint([]byte{1}), Epoch: 1}
	group0only := ops.WASMRegistration{Name: "group0only", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint([]byte{2}), Epoch: 1}
	st := newWASMState()
	st.installed["everywhere"] = installedWASM{
		reg:        everywhere,
		replicated: true,
		groups:     testBindings(everywhere, 0, 7),
	}
	st.installed["group0only"] = installedWASM{
		reg:        group0only,
		replicated: true,
		groups:     testBindings(group0only, 0),
	}

	secs, err := decodeWASMSnapshotBlob(wasmSnapshotBlob(st, 7))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(secs.installs) != 2 {
		t.Fatalf("group 7's snapshot carries %d registrations to install, want both: a joining replica that cannot look an op up halts on it", len(secs.installs))
	}
	bound := map[string]bool{}
	for _, r := range secs.bindings {
		bound[r.Name] = true
	}
	if !bound["everywhere"] {
		t.Error("group 7's log carries \"everywhere\" but the snapshot did not bind it: a joining replica would refuse to serve it forever, and any committed invocation of it left in group 7's log would halt the replica with ErrWASMNoGroupBinding")
	}
	if bound["group0only"] {
		t.Error("group 7's log does NOT carry \"group0only\", but the snapshot bound it: a joining replica would propose invocations into a group with no registration in its log, and would execute entries with a version that group's log never named")
	}
}

// TestWASMGateIsObservable pins the gate's diagnostics.
//
// The gate deliberately trades a shard-wide halt for a client-visible retryable
// error, which makes that error's VISIBILITY load-bearing rather than cosmetic:
// a (op, group) pair whose registration never arrives is refused forever, and
// with no counter and no way to ask which pairs are closed, the operator sees
// only a client that "sometimes" fails on some keys. Stats() has to answer both
// questions — how often the gate has refused, and exactly which groups it
// considers proven for each gated op.
func TestWASMGateIsObservable(t *testing.T) {
	const numShards, starved = 4, 2
	n := newTestNode(t, numShards)
	waitAllApplied(t, n)

	if got := n.Stats().WASMGate; got.Refusals != 0 || len(got.ProvenGroups) != 0 {
		t.Fatalf("precondition: a cluster with no dynamic WASM must report an empty gate, got %+v", got)
	}

	reattach := detachGroup(t, n, starved)
	_, _ = n.Call(wasmRegisterOpName, routableWASMPayload(t, "wasm_incr", 1))
	reattach()

	// The starved group is hosted again but its log never got the registration.
	if err := n.checkWASMRouteGate("wasm_incr", ops.OpReadWrite, starved); !errors.Is(err, ErrWASMOpNotInThisGroup) {
		t.Fatalf("precondition: group %d must be gated, got %v", starved, err)
	}

	st := n.Stats().WASMGate
	if st.Refusals != 1 {
		t.Errorf("Refusals = %d, want 1: a wedged gate is invisible without a counter", st.Refusals)
	}
	proven, ok := st.ProvenGroups["wasm_incr"]
	if !ok {
		t.Fatalf("ProvenGroups has no entry for the gated op: %+v", st)
	}
	for _, g := range proven {
		if g == starved {
			t.Fatalf("ProvenGroups reports group %d as proven; it never received the registration (%v)", starved, proven)
		}
	}
	if want := []int{0, 1, 3}; !sameInts(proven, want) {
		t.Errorf("ProvenGroups[wasm_incr] = %v, want %v — this map is how an operator finds the wedged (op, group) pair", proven, want)
	}

	// Refusals accumulate, so a stuck client shows up as a rising counter.
	_ = n.checkWASMRouteGate("wasm_incr", ops.OpReadWrite, starved)
	if got := n.Stats().WASMGate.Refusals; got != 2 {
		t.Errorf("Refusals = %d after a second refusal, want 2", got)
	}
}

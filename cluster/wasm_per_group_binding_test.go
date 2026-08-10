// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/wasm"
)

// publish mirrors Node.publishWASMGateLocked for a bare wasmReplica: it hands the
// replica's per-group binding table to its WASM runtime, which is what the apply
// path resolves versions from. It runs the SAME builder the node uses, so a test
// cannot pass against a snapshot shape production never produces.
func (r *wasmReplica) publish() {
	r.rt.PublishGroupBindings(wasmBindingSnapshot(r.st))
}

// invokeInGroup runs the registered op exactly as a committed entry in shard
// group g would: through the ops.Registry handler, with a TxContext whose
// dispatcher reports group g. It returns the cache the handler wrote to, so two
// replicas' resulting STATE can be compared rather than their intentions.
func (r *wasmReplica) invokeInGroup(t *testing.T, name string, group int, args []byte) (*cache.Cache, error) {
	t.Helper()
	c, err := cache.New(cache.DefaultConfig())
	if err != nil {
		t.Fatalf("cache.New: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	tx := ops.NewTxContext(c)
	tx.SetShardIndex(group)
	fn, _, _, ok := r.reg.Lookup(name)
	if !ok {
		t.Fatalf("op %q is not registered on this replica", name)
	}
	_, err = fn(tx, args)
	return c, err
}

// TestTwoReplicasOfOneGroupRunOneWASMVersion is the end-to-end gate for the defect
// this whole stage exists to fix, reproduced at its smallest.
//
// THE SETUP IS THE DEFECT'S EXACT SHAPE. Two replicas of shard group 1. Replica
// A also hosts group 0; replica B hosts group 1 alone — a perfectly ordinary
// placement. Op "udf" is registered at v1 and reaches both groups. An update to
// v2 is then broadcast and reaches GROUP 0 only (a partial broadcast: one
// election, one slow peer, one timeout — no race and no injected fault needed).
//
// Replica A applies that update, because it hosts group 0. Replica B never sees
// it. Both replicas' local states are entirely self-consistent and entirely
// correct. Now an invocation committed in GROUP 1 — a log both of them replicate
// — is applied by both.
//
// Under the node-wide binding this used to resolve to "whatever version this node
// last installed": A would execute v2 and B v1. Both applies SUCCEED. There is no
// error to classify, no halt, and no metric — the two replicas simply write
// different state and stay that way. Per-group binding makes the answer a
// function of GROUP 1's log prefix instead, which both replicas have applied
// identically, so both execute v1.
//
// The two versions are chosen so the difference is OBSERVABLE IN STATE rather
// than in a returned ModuleID: v1 is put.wasm, which writes its args as a cache
// entry, and v2 is incr.wasm, which writes nothing. A replica running the wrong
// version leaves a different cache.
func TestTwoReplicasOfOneGroupRunOneWASMVersion(t *testing.T) {
	const name = "udf"
	writes := readTestWASM(t, "../wasm/testdata/put.wasm") // v1: writes args as a key
	noop := readTestWASM(t, "../wasm/testdata/incr.wasm")  // v2: writes nothing
	key := []byte("k")
	args := stdArgs(key) // every WASM op reads [keyLen u16][key]; the module skips the prefix

	v1 := ops.WASMRegistration{
		Name: name, Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(writes),
		ExportName: "apply", Epoch: 1,
	}
	v2 := v1
	v2.Blob = ops.WASMBlobFingerprint(noop)
	v2.Epoch = 2

	// Replica A hosts groups 0 and 1. Replica B hosts group 1 only.
	// Both replicas HOLD both modules: the push delivers the bytes to every member
	// regardless of which groups it hosts, and the whole point of the test is that
	// they nevertheless execute different versions per group.
	a := newWASMReplica(t)
	a.hold(t, writes)
	a.hold(t, noop)
	a.applyFrom(t, 0, v1)
	a.applyFrom(t, 1, v1)
	b := newWASMReplica(t)
	b.hold(t, writes)
	b.hold(t, noop)
	b.applyFrom(t, 1, v1)

	// The update reaches GROUP 0 only. B does not host group 0, so it never sees
	// the entry at all.
	a.applyFrom(t, 0, v2)

	a.publish()
	b.publish()

	// PRECONDITION: the two replicas genuinely disagree node-wide. Without this the
	// test could pass because nothing ever differed.
	if aFP, bFP := ops.WASMRegistrationFingerprint(a.installed(t, name)), ops.WASMRegistrationFingerprint(b.installed(t, name)); aFP == bFP {
		t.Fatal("precondition: replica A must have installed the update node-wide and replica B must not have")
	}

	// The same committed entry, in the group they share.
	aCache, aErr := a.invokeInGroup(t, name, 1, args)
	bCache, bErr := b.invokeInGroup(t, name, 1, args)
	if aErr != nil || bErr != nil {
		t.Fatalf("applying the committed entry failed (A: %v, B: %v)", aErr, bErr)
	}

	aVal, aGet := aCache.Get(key)
	bVal, bGet := bCache.Get(key)
	if (aGet == nil) != (bGet == nil) || string(aVal) != string(bVal) {
		t.Fatalf("THE TWO REPLICAS OF GROUP 1 DIVERGED on one committed entry: A wrote %q (err %v), B wrote %q (err %v). Both applies succeeded, so nothing would ever surface this — the version was resolved from node-wide state instead of from group 1's log prefix",
			aVal, aGet, bVal, bGet)
	}
	// And it must be GROUP 1's version that ran, not merely the same one twice.
	if aGet != nil {
		t.Errorf("group 1's log committed v1 (which writes), but the entry executed a version that wrote nothing: %v", aGet)
	}

	// Group 0, on the replica that hosts it, must meanwhile be on v2 — otherwise
	// the update did not take anywhere and the test proves only that nothing moved.
	zeroCache, err := a.invokeInGroup(t, name, 0, key)
	if err != nil {
		t.Fatalf("invoking in group 0 on replica A: %v", err)
	}
	if _, err := zeroCache.Get(key); err == nil {
		t.Error("group 0 still executed v1 after applying the update from its own log: per-group binding must move a group forward as well as hold others back")
	}
}

// TestPerGroupWASMContractRefusalIsIdenticalOnEveryReplica pins the apply-time
// refusal that per-group state made possible.
//
// cluster/wasm_gate.go used to argue that a state-dependent rejection at apply
// time is ITSELF a divergence source — the replica that already holds the op
// rejects the entry while one that does not accepts it, and the two then run
// different modules under one name. That argument is correct for NODE-WIDE state.
// It stops holding for PER-GROUP state: a refusal keyed on group g's own binding
// is a pure function of g's log prefix, and every replica of g has applied that
// same prefix, so all of them reach the same verdict on the same entry.
//
// This test builds two replicas of group 0 with DIFFERENT other placements and
// different node-wide installs, feeds both the same contract-changing entry from
// group 0, and requires the verdict AND the surviving binding to match.
func TestPerGroupWASMContractRefusalIsIdenticalOnEveryReplica(t *testing.T) {
	const name = "udf"
	incr := readIncrWASM(t)

	v1 := ops.WASMRegistration{
		Name: name, Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr),
		ExportName: "apply", Epoch: 1,
	}
	// A CONTRACT change: same module, different Kind. It decides whether the
	// invocation is replicated at all, so it is frozen at first registration.
	//
	// KIND, NOT THE KEY EXTRACTOR, and that is forced rather than chosen:
	// WASMRegistration has no extractor field to differ in
	// (ops.WASMKeyExtractorHandle). Kind is the whole of the contract
	// checkWASMGroupContract still compares.
	contractChange := v1
	contractChange.Kind = ops.OpReadOnly
	contractChange.Epoch = 5

	// A hosts groups 0 and 3 and has already seen an unrelated bytes update on
	// group 3, so its node-wide install differs from B's.
	bytesUpdate := v1
	bytesUpdate.Blob = ops.WASMBlobFingerprint(readDelWASM(t))
	bytesUpdate.Epoch = 2

	a := newWASMReplica(t)
	a.applyFrom(t, 0, v1)
	a.applyFrom(t, 3, v1)
	a.applyFrom(t, 3, bytesUpdate)

	b := newWASMReplica(t)
	b.applyFrom(t, 0, v1)

	if aFP, bFP := ops.WASMRegistrationFingerprint(a.installed(t, name)), ops.WASMRegistrationFingerprint(b.installed(t, name)); aFP == bFP {
		t.Fatal("precondition: the two replicas must hold different node-wide installs, or the refusal is not being tested against differing state")
	}

	aErr := applyWASMRegistration(a.dir, a.rt, a.reg, a.st, contractChange, 0, nil)
	bErr := applyWASMRegistration(b.dir, b.rt, b.reg, b.st, contractChange, 0, nil)

	for who, err := range map[string]error{"A": aErr, "B": bErr} {
		if !errors.Is(err, ErrWASMUpdateUnsupported) {
			t.Errorf("replica %s: got %v, want ErrWASMUpdateUnsupported — a contract change must be refused against the group's own binding", who, err)
		}
		if err != nil && !strings.Contains(err.Error(), ops.WASMUpdateUnsupportedMsg) {
			t.Errorf("replica %s: refusal would be redacted to a generic internal error: %q", who, err.Error())
		}
	}
	if (aErr == nil) != (bErr == nil) {
		t.Fatalf("THE TWO REPLICAS OF GROUP 0 DISAGREED about one committed entry (A: %v, B: %v). One would skip what the other applies — the divergence a node-wide apply-time rejection produces, and the exact reason this refusal has to key on the group", aErr, bErr)
	}

	// The surviving binding for group 0 must be identical, and must still be the
	// contract group 0's log established.
	aBind, aOK := a.st.installed[name].groups[0]
	bBind, bOK := b.st.installed[name].groups[0]
	if !aOK || !bOK {
		t.Fatalf("a refused contract change destroyed group 0's binding (A ok=%v, B ok=%v): the next committed invocation would halt the node", aOK, bOK)
	}
	if aBind.id != bBind.id {
		t.Errorf("group 0's binding diverged across the refusal: A=%s B=%s", aBind.id, bBind.id)
	}
	if aBind.reg.Kind != ops.OpReadWrite {
		t.Errorf("group 0's frozen contract moved to Kind %d: the refusal did not hold", uint8(aBind.reg.Kind))
	}
}

// TestWASMSnapshotCarriesPerGroupBindingsExactly pins the snapshot round trip in
// BOTH directions, because both directions are bugs.
//
// TOO NARROW: a binding the snapshotted group had and the snapshot dropped
// leaves the restoring replica with a committed invocation in that group's log
// and no version to resolve — ops.ErrWASMNoGroupBinding, which is classFatal.
// TOO WIDE: a binding the snapshotted group did NOT have lets the restorer
// propose invocations into a group whose log never carried the registration, and
// then execute them with a version that log never named.
//
// The interesting case is the one where the group's bound version DIFFERS from
// the snapshotting node's node-wide install, which is exactly what the old
// one-record-plus-a-boolean format could not express.
func TestWASMSnapshotCarriesPerGroupBindingsExactly(t *testing.T) {
	const name = "udf"
	v1 := ops.WASMRegistration{
		Name: name, Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(readTestWASM(t, "../wasm/testdata/put.wasm")),
		ExportName: "apply", Epoch: 1,
	}
	v2 := v1
	v2.Blob = ops.WASMBlobFingerprint(readIncrWASM(t))
	v2.Epoch = 2
	unrelated := ops.WASMRegistration{
		Name: "other", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(readDelWASM(t)),
		ExportName: "apply", Epoch: 1,
	}

	// The snapshotting node: group 1 is still on v1, group 0 has the update, and
	// "other" is bound only in group 0.
	src := newWASMReplica(t)
	src.applyFrom(t, 0, v1)
	src.applyFrom(t, 1, v1)
	src.applyFrom(t, 0, v2)
	src.applyFrom(t, 0, unrelated)

	if got := src.st.installed[name].groups[1].id; got != wasmModuleIDOf(v1) {
		t.Fatalf("precondition: group 1 must still be bound to v1, got %s", got)
	}
	if got := ops.WASMRegistrationFingerprint(src.installed(t, name)); got != ops.WASMRegistrationFingerprint(v2) {
		t.Fatal("precondition: the node-wide install must be v2, so the group-1 binding genuinely differs from it")
	}

	// Take group 1's snapshot and restore it into a pristine replica.
	blob := wasmSnapshotBlob(src.st, 1)
	if blob == nil {
		t.Fatal("group 1's snapshot carries nothing")
	}
	dst := newWASMReplica(t)
	secs, err := decodeWASMSnapshotBlob(blob)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	for _, r := range secs.installs {
		if err := applyWASMRegistration(dst.dir, dst.rt, dst.reg, dst.st, r, ops.NoShardIndex, nil); err != nil {
			t.Fatalf("restore install %q: %v", r.Name, err)
		}
	}
	for _, r := range secs.bindings {
		if err := applyWASMRegistration(dst.dir, dst.rt, dst.reg, dst.st, r, 1, nil); err != nil {
			t.Fatalf("restore binding %q: %v", r.Name, err)
		}
	}

	// EXACT: group 1 is bound to "udf" at v1 and to nothing else.
	got := dst.st.installed[name].groups
	if len(got) != 1 {
		t.Fatalf("the restored replica binds %v for %q in group 1's snapshot, want exactly {1}", sortedGroups(got), name)
	}
	if got[1].id != wasmModuleIDOf(v1) {
		t.Errorf("group 1 restored to %s, want the version ITS log committed (v1 %s). Restoring the node-wide install instead is how a snapshot-joined replica ends up executing entries with a version its group never named",
			got[1].id, wasmModuleIDOf(v1))
	}
	if other := dst.st.installed["other"].groups; len(other) != 0 {
		t.Errorf("group 1's snapshot bound %q in %v; that op is bound only in group 0 on the source, and claiming otherwise lets the restorer propose into a group with no registration in its log", "other", sortedGroups(other))
	}

	// SUPERSET, in the other direction: every registration must still be
	// INSTALLED, or the restorer cannot look the op up at all and halts under
	// classFatal ErrOpNotRegistered.
	for _, want := range []string{name, "other"} {
		if _, _, _, ok := dst.reg.Lookup(want); !ok {
			t.Errorf("op %q is missing from the restored registry: the replica halts on the first committed invocation of it", want)
		}
	}
	// ...and the node-wide install must be the source's, not group 1's binding.
	if got := ops.WASMRegistrationFingerprint(dst.installed(t, name)); got != ops.WASMRegistrationFingerprint(v2) {
		t.Error("the restored node-wide install is not the source's: the two nodes would route or classify the op differently")
	}
}

// TestWASMSnapshotOntoAnExistingBindingIsNotRefused gates the one catch-up route
// on which the per-group CONTRACT CHECK actually fires, and which nothing
// previously covered.
//
// THE SHAPE. Every other snapshot path installs into a table with no prior entry
// for the snapshotted group, so checkWASMGroupContract is never consulted: the
// installs section carries ops.NoShardIndex and binds nothing, and a
// freshly-joined replica has no binding for the group to compare against. But a
// follower that has applied part of group 1's log and THEN falls far enough
// behind to be caught up by InstallSnapshot — the ordinary lagging-follower case —
// already holds a group-1 binding when the snapshot's bindings section arrives.
// That record goes through applyWASMRegistration with fromGroup=1 like any other
// registration, hits `prev, had := cur.groups[1]`, and is CHECKED.
//
// WHY IT MUST PASS, and it is a lemma rather than an accident. A group's contract
// is pinned by the FIRST registration of that name in the group's log
// (installedWASM.groups). This replica's binding is a fold over one prefix of
// group 1's log and the snapshot's binding is a fold over a longer prefix of THE
// SAME log; every non-empty prefix of a log shares its first element, so the two
// contracts are equal and the check passes. A restore may therefore move a
// group's binding FORWARD, and can never be refused.
//
// WHAT A FAILURE MEANS. The restore returns an error, the caller
// (installWASMSnapshotBlobLocked) aborts the whole restore, and the follower is
// left holding a STALE binding for group 1 while its peers execute the newer one
// — the exact silent divergence per-group binding exists to remove, reintroduced
// by the check meant to prevent it.
func TestWASMSnapshotOntoAnExistingBindingIsNotRefused(t *testing.T) {
	const name = "udf"
	v1 := ops.WASMRegistration{
		Name: name, Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(readTestWASM(t, "../wasm/testdata/put.wasm")),
		ExportName: "apply", Epoch: 1,
	}
	v2 := v1
	v2.Blob = ops.WASMBlobFingerprint(readIncrWASM(t))
	v2.Epoch = 2

	// The leader: group 1's log carries v1 then v2, so its group-1 binding is v2.
	src := newWASMReplica(t)
	src.applyFrom(t, 1, v1)
	src.applyFrom(t, 1, v2)

	// The lagging follower: it has applied group 1's log only as far as v1, and it
	// hosts group 4 as well with traffic the leader never saw, so its node-wide
	// state genuinely differs. Without this the restore could pass for the trivial
	// reason that nothing differed.
	other := v1
	other.Blob = ops.WASMBlobFingerprint(readDelWASM(t))
	other.Epoch = 7
	dst := newWASMReplica(t)
	// The follower HOLDS v2's bytes: the push delivered them at registration time,
	// which is why the restore below can make the restored binding resident.
	dst.hold(t, readIncrWASM(t))
	dst.applyFrom(t, 1, v1)
	dst.applyFrom(t, 4, other)
	if _, had := dst.st.installed[name].groups[1]; !had {
		t.Fatal("precondition: the follower must already hold a group-1 binding, or the contract check under test never fires")
	}

	// Group 1's snapshot, restored in exactly the order installWASMSnapshotBlobLocked
	// applies it: installs with no provenance, then bindings attributed to group 1.
	secs, err := decodeWASMSnapshotBlob(wasmSnapshotBlob(src.st, 1))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(secs.bindings) == 0 {
		t.Fatal("precondition: group 1's snapshot carries no binding, so nothing reaches the contract check")
	}
	for _, r := range secs.installs {
		if err := applyWASMRegistration(dst.dir, dst.rt, dst.reg, dst.st, r, ops.NoShardIndex, nil); err != nil {
			t.Fatalf("restore install %q: %v", r.Name, err)
		}
	}
	for _, r := range secs.bindings {
		if err := applyWASMRegistration(dst.dir, dst.rt, dst.reg, dst.st, r, 1, nil); err != nil {
			t.Fatalf("A SNAPSHOT WAS REFUSED INTO A GROUP THIS REPLICA IS ALREADY BOUND IN: %v.\ninstallWASMSnapshotBlobLocked aborts the whole restore on this error, so the follower keeps its STALE group-1 binding and executes committed entries with a version its peers have moved off — the divergence per-group binding exists to remove, caused by the check meant to prevent it", err)
		}
	}

	if got, want := dst.st.installed[name].groups[1].id, wasmModuleIDOf(v2); got != want {
		t.Errorf("group 1 restored to %s, want the version the snapshot carried (v2 %s): a restore must be able to move a group's binding FORWARD", got, want)
	}
	if !dst.rt.HasModule(wasmModuleIDOf(v2)) {
		t.Error("the restored binding's module is not in the runtime: the binding resolves to nothing and every group-1 entry fails")
	}

	// NEGATIVE CONTROL. The pass above must be because the contracts AGREE, not
	// because the check is dead on this path. Feed the same replica a group-1
	// record whose contract differs and require it to be refused.
	contractChange := v2
	contractChange.Kind = ops.OpReadOnly
	contractChange.Epoch = 9
	if err := applyWASMRegistration(dst.dir, dst.rt, dst.reg, dst.st, contractChange, 1, nil); !errors.Is(err, ErrWASMUpdateUnsupported) {
		t.Fatalf("the per-group contract check does not fire on the restore path at all (got %v): the assertion above proves nothing", err)
	}
}

// TestRestartReconstructsPerGroupWASMBindings is the durability gate.
//
// The binding table is NOT re-derivable after a restart: in durable mode
// fsm.Apply skips every entry at or below the persisted applied index, so the
// __register_wasm__ entries that established the bindings are never applied
// again. If the sidecar does not carry them, a restarted node comes up with the
// op registered and every group unbound — the route gate shut everywhere, and any
// committed invocation still to replay out of a group's log halting the node
// under classFatal ops.ErrWASMNoGroupBinding.
//
// It also pins the idempotent-retry path across that restart, which is the
// partial-broadcast repair: after a restart the comparison target is
// reconstructed field by field from the sidecar, so a re-sent identical
// registration must still compare equal and still be allowed.
func TestRestartReconstructsPerGroupWASMBindings(t *testing.T) {
	const name = "udf"
	dir := t.TempDir()
	v1 := ops.WASMRegistration{
		Name: name, Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(readTestWASM(t, "../wasm/testdata/put.wasm")),
		ExportName: "apply", Epoch: 1,
	}
	v2 := v1
	v2.Blob = ops.WASMBlobFingerprint(readIncrWASM(t))
	v2.Epoch = 2

	// Process 1: groups 0 and 2 take the update; group 1 is starved and stays on
	// v1. That mixed state is the one a naive sidecar (a bare group list plus one
	// version) cannot represent.
	rt1, reg1, st1, err := restartWASM(t, dir, nil)
	if err != nil {
		t.Fatalf("first start: %v", err)
	}
	// Both versions' bytes reached this node's blob store at their registrations'
	// push. The restart assertions below are about whether the BINDINGS survive,
	// and a blob the node never held would make "not resident" the reason instead.
	seedWASMBlob(t, dir, readTestWASM(t, "../wasm/testdata/put.wasm"))
	seedWASMBlob(t, dir, readIncrWASM(t))
	for _, g := range []int{0, 1, 2} {
		if err := applyWASMRegistration(dir, rt1, reg1, st1, v1, g, nil); err != nil {
			t.Fatalf("apply v1 to group %d: %v", g, err)
		}
	}
	for _, g := range []int{0, 2} {
		if err := applyWASMRegistration(dir, rt1, reg1, st1, v2, g, nil); err != nil {
			t.Fatalf("apply v2 to group %d: %v", g, err)
		}
	}

	// Process 2: a fresh runtime, registry and state over the same data dir.
	rt2, reg2, st2, err := restartWASM(t, dir, nil)
	if err != nil {
		t.Fatalf("restart: %v", err)
	}
	in, ok := st2.installed[name]
	if !ok {
		t.Fatal("the op is not installed at all after the restart")
	}
	if got, want := sortedGroups(in.groups), []int{0, 1, 2}; !sameInts(got, want) {
		t.Fatalf("bound groups after restart = %v, want %v: the route gate is shut and any committed invocation left in those logs halts this node", got, want)
	}
	wantV1, wantV2 := wasmModuleIDOf(v1), wasmModuleIDOf(v2)
	for g, want := range map[int]wasm.ModuleID{0: wantV2, 1: wantV1, 2: wantV2} {
		if got := in.groups[g].id; got != want {
			t.Errorf("group %d restored to %s, want %s: the restart lost which version that group's log committed", g, got, want)
		}
	}
	// The starved group's version must be RUNNABLE, not merely recorded — its blob
	// has to have survived even though it is not the node-wide install.
	if !rt2.HasModule(wantV1) {
		t.Error("group 1's bound module was not loaded into the runtime on restart: the recorded binding resolves to nothing and every entry in group 1's log fails")
	}
	if !rt2.HasModule(wantV2) {
		t.Error("the node-wide module was not loaded into the runtime on restart")
	}

	// The idempotent retry across the restart: the comparison target is now
	// reconstructed from the sidecar, so a re-sent identical registration has to
	// still be recognised as identical. It is the partial-broadcast repair path.
	if err := applyWASMRegistration(dir, rt2, reg2, st2, v2, 0, nil); err != nil {
		t.Errorf("re-applying the identical registration after a restart must be allowed — it is the only way a starved group ever gets the entry: %v", err)
	}
	if err := applyWASMRegistration(dir, rt2, reg2, st2, v1, 1, nil); err != nil {
		t.Errorf("re-applying group 1's own identical registration after a restart must be allowed: %v", err)
	}
	// ...and neither retry moved anything.
	for g, want := range map[int]wasm.ModuleID{0: wantV2, 1: wantV1, 2: wantV2} {
		if got := st2.installed[name].groups[g].id; got != want {
			t.Errorf("group %d moved to %s under an identical re-registration, want %s", g, got, want)
		}
	}
}

// TestWASMSidecarFromAnOlderFormatIsRefused pins the on-disk break.
//
// A version-1 sidecar records the route gate's evidence as a bare group SET, with
// no version attached. Reading one silently would leave every group UNBOUND while
// the op stayed registered: unroutable everywhere, and a classFatal
// ops.ErrWASMNoGroupBinding halt on any committed invocation still to replay.
// Absence of the bindings key cannot be used as the signal instead — a replicated
// op legitimately has an empty binding set when its only source was a snapshot's
// install section — so the format version is what makes the break loud.
func TestWASMSidecarFromAnOlderFormatIsRefused(t *testing.T) {
	dir := t.TempDir()
	rt, reg, st, err := restartWASM(t, dir, nil)
	if err != nil {
		t.Fatalf("first start: %v", err)
	}
	r := ops.WASMRegistration{
		Name: "udf", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(readIncrWASM(t)),
		ExportName: "apply", Epoch: 1,
	}
	seedWASMBlob(t, dir, readIncrWASM(t))
	if err := applyWASMRegistration(dir, rt, reg, st, r, 0, nil); err != nil {
		t.Fatalf("apply: %v", err)
	}

	// Rewrite the sidecar in the version-1 shape: no "version", and the old
	// "groups" array in place of "bindings".
	path := filepath.Join(dir, "wasm", "udf.json")
	old := `{"kind":1,"export_name":"apply","key_extractor_handle":"raw","max_fuel":0,"epoch":1,"source":"replicated","groups":[0],"blob":"` +
		sidecarBlobOf(t, path) + `"}`
	if err := os.WriteFile(path, []byte(old), 0o600); err != nil {
		t.Fatalf("write legacy sidecar: %v", err)
	}

	_, _, _, err = restartWASM(t, dir, nil)
	if err == nil {
		t.Fatal("a version-1 sidecar loaded silently: every group comes up UNBOUND, the op is unroutable, and the first committed invocation halts the node")
	}
	if !strings.Contains(err.Error(), "format version") {
		t.Errorf("the refusal does not identify the format break: %v", err)
	}
}

// sidecarBlobOf reads the blob fingerprint out of an existing sidecar, so the
// legacy file above names a real blob rather than an invented one.
//
// It is no longer needed to keep an EARLIER refusal from firing — a sidecar whose
// blob is absent is now an ordinary state the reload logs and fetches through, not
// a start failure — but it still matters: the format-version check is the one this
// test measures, and naming a plausible blob keeps the legacy file a faithful copy
// of what an older build actually wrote.
func sidecarBlobOf(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path) //nolint:gosec // test-controlled path
	if err != nil {
		t.Fatalf("read sidecar: %v", err)
	}
	const key = `"blob":"`
	i := strings.Index(string(b), key)
	if i < 0 {
		t.Fatalf("sidecar has no blob field: %s", b)
	}
	rest := string(b)[i+len(key):]
	j := strings.Index(rest, `"`)
	if j < 0 {
		t.Fatalf("sidecar blob field is unterminated: %s", b)
	}
	return rest[:j]
}

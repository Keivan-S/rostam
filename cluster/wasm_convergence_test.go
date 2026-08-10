// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"os"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/wasm"
)

// wasmReplica is one node's WASM-facing state: its data dir, runtime, op
// registry and installed-registration record. Two of them stand in for two
// replicas that received the same registrations in DIFFERENT orders.
type wasmReplica struct {
	dir string
	rt  *wasm.Runtime
	reg *ops.Registry
	st  *wasmState
}

func newWASMReplica(t *testing.T) *wasmReplica {
	t.Helper()
	rt, err := wasm.NewRuntime()
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	t.Cleanup(func() { _ = rt.Close() })
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	return &wasmReplica{dir: t.TempDir(), rt: rt, reg: reg, st: newWASMState()}
}

func (r *wasmReplica) apply(t *testing.T, reg ops.WASMRegistration) {
	t.Helper()
	r.applyFrom(t, 0, reg)
}

// hold puts a module's bytes in this replica's blob store, which is what the
// pre-registration push (cluster.storeWASMBlobVerified) or a background fetch
// does on a real node.
//
// It is separate from apply because the two are separate on a real node now: a
// marker names its module and does not carry it, so applying a registration
// READS the blob and leaves the module NOT RESIDENT when it is absent. A test
// that only inspects bindings, sidecars or the registry needs no hold; a test
// that wants to EXECUTE the module, or that asserts rt.HasModule, does.
func (r *wasmReplica) hold(t *testing.T, module []byte) {
	t.Helper()
	if _, err := writeWASMBlob(r.dir, module); err != nil {
		t.Fatalf("writeWASMBlob: %v", err)
	}
}

// applyFrom is apply with an explicit route-gate provenance: the shard group
// whose log carried the entry.
func (r *wasmReplica) applyFrom(t *testing.T, group int, reg ops.WASMRegistration) {
	t.Helper()
	if err := applyWASMRegistration(r.dir, r.rt, r.reg, r.st, reg, group, nil); err != nil {
		t.Fatalf("applyWASMRegistration(%q epoch=%d): %v", reg.Name, reg.Epoch, err)
	}
}

// installed returns the registration this replica currently holds for name.
func (r *wasmReplica) installed(t *testing.T, name string) ops.WASMRegistration {
	t.Helper()
	in, ok := r.st.installed[name]
	if !ok {
		t.Fatalf("no registration installed for %q", name)
	}
	return in.reg
}

func readTestWASM(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("%s not readable (%v); skipping", path, err)
	}
	return b
}

// TestConflictingWASMRegistrationsConverge is the regression gate for the
// divergence that per-group broadcasting introduced.
//
// Registrations used to be totally ordered because they all landed in shard 0's
// log alone. Broadcasting one into EVERY shard group replaced that single order
// with N independent ones, so two registrations of the SAME op name carrying
// DIFFERENT module bytes can be applied in opposite orders by two different
// groups — and therefore by two different replicas. Under a "last arrival wins"
// rule (which is what rt.AddModule's replace-on-fingerprint-mismatch gives you)
// the two replicas end up EXECUTING DIFFERENT CODE for the same op name, with no
// error and no metric.
//
// The fix is an order-independent maximum: a registration installs only if its
// (Epoch, fingerprint) pair strictly exceeds the installed one's. This test
// applies the same two registrations in opposite orders and requires the two
// replicas to agree on every field, not merely to converge on the bytes.
//
// THE NODE-WIDE FOLD MUST STAY UNCONDITIONAL, which is what the contract-
// differing pair below pins. Per-group version binding added an apply-time
// refusal of a registration that changes a group's Kind or key extractor — and
// that refusal is deliberately scoped to the BINDING. If it aborted the whole
// apply, the node-wide fold would become a function of which shard groups a node
// happens to host: with A,B ordered one way in group 0's log and the other way in
// group 7's, a node hosting only group 0 would keep A while a node hosting only
// group 7 kept B, and the two would then ROUTE the op's invocations to different
// shard groups forever. The refusal is REPORTED (as an error the caller sees) but
// the maximum still installs.
func TestConflictingWASMRegistrationsConverge(t *testing.T) {
	incr := readTestWASM(t, "../wasm/testdata/incr.wasm")
	put := readTestWASM(t, "../wasm/testdata/put.wasm")

	regA := ops.WASMRegistration{
		Name: "wasm_conflict", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr),
		ExportName: "apply", Epoch: 1,
	}
	regB := ops.WASMRegistration{
		Name: "wasm_conflict", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(put),
		ExportName: "apply", Epoch: 2,
	}

	// The two registrations must be genuinely distinct, or the test is vacuous.
	if regA.Blob == regB.Blob {
		t.Fatal("precondition: the two modules must differ")
	}

	// applyContractRace applies the two in the given order into ONE group and
	// tolerates the per-group contract refusal on the second, which is expected:
	// the group keeps the binding it established first. What must NOT vary is the
	// node-wide install.
	applyContractRace := func(r *wasmReplica, regs ...ops.WASMRegistration) {
		t.Helper()
		for _, reg := range regs {
			err := applyWASMRegistration(r.dir, r.rt, r.reg, r.st, reg, 0, nil)
			if err != nil && !errors.Is(err, ErrWASMUpdateUnsupported) {
				t.Fatalf("applyWASMRegistration(%q epoch=%d): %v", reg.Name, reg.Epoch, err)
			}
		}
	}

	first := newWASMReplica(t)
	applyContractRace(first, regA, regB)

	second := newWASMReplica(t)
	applyContractRace(second, regB, regA)

	gotFirst := first.installed(t, "wasm_conflict")
	gotSecond := second.installed(t, "wasm_conflict")

	if ops.WASMRegistrationFingerprint(gotFirst) != ops.WASMRegistrationFingerprint(gotSecond) {
		t.Fatalf("replicas diverged: A→B installed blob %s (kind=%v epoch=%d), B→A installed blob %s (kind=%v epoch=%d)",
			ops.WASMBlobHex(gotFirst.Blob), gotFirst.Kind, gotFirst.Epoch,
			ops.WASMBlobHex(gotSecond.Blob), gotSecond.Kind, gotSecond.Epoch)
	}
	// And the winner must be the higher epoch, not just "the same one".
	if gotFirst.Epoch != regB.Epoch || gotFirst.Blob != regB.Blob {
		t.Errorf("converged on epoch %d / blob %s, want the strictly-higher epoch %d / blob %s",
			gotFirst.Epoch, ops.WASMBlobHex(gotFirst.Blob), regB.Epoch, ops.WASMBlobHex(regB.Blob))
	}

	// The ops registry — not just the runtime — must agree too. Kind and the key
	// extractor live there, and they decide whether the op bypasses Raft and
	// which shard group its invocations land in, so a disagreement here is a
	// routing divergence even when the bytes match.
	_, kindFirst, keFirst, okFirst := first.reg.Lookup("wasm_conflict")
	_, kindSecond, keSecond, okSecond := second.reg.Lookup("wasm_conflict")
	if !okFirst || !okSecond {
		t.Fatalf("op missing from a registry (A→B: %v, B→A: %v)", okFirst, okSecond)
	}
	if kindFirst != kindSecond {
		t.Errorf("registry kind diverged: %v vs %v", kindFirst, kindSecond)
	}
	if (keFirst == nil) != (keSecond == nil) {
		t.Errorf("registry routability diverged: A→B routable=%v, B→A routable=%v", keFirst != nil, keSecond != nil)
	}
}

// TestWASMRegistrationEpochTieBreak pins the tiebreak itself: two registrations
// sharing an Epoch still have to resolve to the SAME winner on every replica,
// so the comparison falls through to the content fingerprint rather than to
// arrival order.
func TestWASMRegistrationEpochTieBreak(t *testing.T) {
	incr := readTestWASM(t, "../wasm/testdata/incr.wasm")
	put := readTestWASM(t, "../wasm/testdata/put.wasm")

	a := ops.WASMRegistration{Name: "tie", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr), ExportName: "apply", Epoch: 7}
	b := ops.WASMRegistration{Name: "tie", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(put), ExportName: "apply", Epoch: 7}

	// Exactly one direction must be "newer" — an antisymmetric total order.
	ab := ops.WASMRegistrationNewer(a, b)
	ba := ops.WASMRegistrationNewer(b, a)
	if ab == ba {
		t.Fatalf("tiebreak is not a strict order: newer(a,b)=%v newer(b,a)=%v", ab, ba)
	}
	// And nothing is newer than itself, which is what makes a repeated apply a
	// no-op (the broadcast runs the hook once per hosted group).
	if ops.WASMRegistrationNewer(a, a) {
		t.Error("a registration must not supersede itself; repeated applies would reinstall on every group")
	}
	// A strictly higher epoch wins regardless of which way the fingerprint falls.
	older := a
	newer := b
	newer.Epoch = a.Epoch + 1
	if !ops.WASMRegistrationNewer(newer, older) {
		t.Error("a strictly higher epoch must win")
	}
	if ops.WASMRegistrationNewer(older, newer) {
		t.Error("a strictly lower epoch must lose")
	}
}

// TestConfigWASMModuleNeverBeatsReplicated pins the asymmetry between the two
// install sources. A config module is node-LOCAL: if it could win the
// (Epoch, fingerprint) comparison, a node that happens to have it configured
// would install different code than a node that does not — divergence produced
// by the rule meant to prevent it. A replicated registration must therefore
// override a config module unconditionally, even at Epoch 0.
func TestConfigWASMModuleNeverBeatsReplicated(t *testing.T) {
	incr := readTestWASM(t, "../wasm/testdata/incr.wasm")
	put := readTestWASM(t, "../wasm/testdata/put.wasm")

	r := newWASMReplica(t)
	if _, err := loadWASMModules(r.dir, []WASMModuleConfig{{
		Name: "wasm_src", Kind: ops.OpReadWrite, Bytes: put, ExportName: "apply",
	}}, r.rt, r.st); err != nil {
		t.Fatalf("loadWASMModules: %v", err)
	}
	if in, ok := r.st.installed["wasm_src"]; !ok || in.replicated {
		t.Fatalf("precondition: config module must be recorded as non-replicated (ok=%v replicated=%v)", ok, in.replicated)
	}

	// Epoch 0 — the lowest possible — must still take over.
	r.apply(t, ops.WASMRegistration{
		Name: "wasm_src", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr), ExportName: "apply", Epoch: 0,
	})
	got := r.installed(t, "wasm_src")
	if got.Blob != ops.WASMBlobFingerprint(incr) {
		t.Fatal("a replicated registration must override a config module regardless of epoch")
	}
}

// TestWASMRegistrationCannotShadowExistingOp pins the guard on the install path.
// A winning registration REPLACES the ops.Registry entry — it has to, since it
// may change Kind or the key extractor — so without a guard an (admin-gated)
// registration named "get" would take over the builtin on every replica that
// applied it. The refusal is derived purely from the entry, so every replica
// refuses identically and no divergence follows.
func TestWASMRegistrationCannotShadowExistingOp(t *testing.T) {
	incr := readTestWASM(t, "../wasm/testdata/incr.wasm")
	r := newWASMReplica(t)

	if _, _, _, ok := r.reg.Lookup("get"); !ok {
		t.Fatal("precondition: the builtin \"get\" must be registered")
	}
	err := applyWASMRegistration(r.dir, r.rt, r.reg, r.st, ops.WASMRegistration{
		Name: "get", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr), ExportName: "apply", Epoch: 99,
	}, 0, nil)
	if err == nil {
		t.Fatal("a WASM registration must not be able to take over an existing op name")
	}
	// And the builtin is untouched: still registered, still shardless-or-routable
	// exactly as it was, and not recorded as ours.
	if _, ok := r.st.installed["get"]; ok {
		t.Error("the refused registration was still recorded as installed")
	}
}

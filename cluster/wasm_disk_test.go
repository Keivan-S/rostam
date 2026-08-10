// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/wasm"
)

// restartWASM replays node construction's WASM sequence against an existing data
// dir with a fresh runtime, registry and state — exactly the order
// newSingleNode / newMultiNode use, which is the whole point: the config path
// runs BEFORE the disk reload, so anything it writes is what the reload sees.
func restartWASM(t *testing.T, dir string, cfg []WASMModuleConfig) (*wasm.Runtime, *ops.Registry, *wasmState, error) {
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
	st := newWASMState()

	if len(cfg) > 0 {
		loaded, err := loadWASMModules(dir, cfg, rt, st)
		if err != nil {
			return rt, reg, st, err
		}
		for _, lm := range loaded {
			c := lm.Registration
			ke := ops.WASMKeyExtractor()
			if err := wasm.RegisterModule(reg, rt, c.Name, lm.ModuleID, c.Kind, ke); err != nil && !errors.Is(err, ops.ErrDuplicateOp) {
				return rt, reg, st, err
			}
		}
	}
	return rt, reg, st, reloadWASMModulesFromDisk(dir, rt, reg, st, nil)
}

// seedWASMBlob puts a module's bytes at their content address, which is what the
// pre-registration push (storeWASMBlobVerified) does on a real node. A marker
// carries no bytes, so applyWASMRegistration READS the blob instead of writing
// it: without this the module simply is not resident and every residency
// assertion below would be measuring the seeding rather than the code under test.
func seedWASMBlob(t *testing.T, dir string, b []byte) {
	t.Helper()
	if _, err := writeWASMBlob(dir, b); err != nil {
		t.Fatalf("writeWASMBlob: %v", err)
	}
}

// blobPathFor is where a module's bytes live under the content-addressed layout:
// <dataDir>/wasm/blobs/<sha256 of the bytes>.wasm. There is no per-op-name module
// file any more, so this is what "the module reached this node's disk" means.
func blobPathFor(t *testing.T, dataDir string, b []byte) string {
	t.Helper()
	sum := ops.WASMBlobFingerprint(b)
	p, err := wasmBlobPath(dataDir, hex.EncodeToString(sum[:]))
	if err != nil {
		t.Fatalf("wasmBlobPath: %v", err)
	}
	return p
}

// wasmDirFiles lists the basenames under <dir>/wasm, so a test can assert that a
// rejected registration left NOTHING behind (including temp files).
func wasmDirFiles(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(filepath.Join(dir, "wasm"))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		t.Fatalf("readdir: %v", err)
	}
	out := make([]string, 0, len(entries))
	for _, e := range entries {
		out = append(out, e.Name())
	}
	return out
}

// TestWASMRestartConfigModuleNeverClobbersReplicated is the CRITICAL-3 gate.
//
// applyWASMRegistration documents that a replicated registration always
// overrides a node-local config module, and at APPLY time it does — that is what
// TestConfigWASMModuleNeverBeatsReplicated covers. The RESTART path used to
// reverse it, silently and permanently, and nothing tested that direction.
//
// loadWASMModules runs before reloadWASMModulesFromDisk, and loadOneModule used
// to overwrite <name>.wasm and <name>.json unconditionally, stamping
// Source:"config" with no Epoch and no Groups. The reload then SKIPPED the name
// (it was already in st) and never looked at the state it had just destroyed.
// The node was left running config bytes while its peers ran replicated bytes;
// the op dropped out of the route-gate snapshot entirely, because
// publishWASMGateLocked gates only replicated installs, so the gate stopped
// guarding the very op it was armed for; the op vanished from every snapshot the
// node took; and the durable Epoch and proven-group set were erased, neither of
// which a durable restart can re-derive (fsm.Apply skips every entry at or below
// the persisted applied index, so the __register_wasm__ entries never replay).
func TestWASMRestartConfigModuleNeverClobbersReplicated(t *testing.T) {
	const name = "wasm_shared"
	incr := readTestWASM(t, "../wasm/testdata/incr.wasm")
	// The config module differs from the replicated one in BYTES and in KIND.
	// Kind is the contract half a registration is still free to vary — there is no
	// key-extractor field any more (ops.WASMKeyExtractorHandle), so it can no
	// longer be the field that shows a config module's contract taking over.
	// readonly.wasm is the only testdata
	// module that imports nothing state-mutating, so it is the only one that can
	// legally be declared OpReadOnly at config load (wasm.ValidateModuleKind).
	ro := readTestWASM(t, "../wasm/testdata/readonly.wasm")

	dir := t.TempDir()

	// Process 1: a replicated registration lands, proving groups 0, 2 and 3.
	rt1, reg1, st1, err := restartWASM(t, dir, nil)
	if err != nil {
		t.Fatalf("first start: %v", err)
	}
	seedWASMBlob(t, dir, incr)
	replicated := ops.WASMRegistration{
		Name: name, Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr),
		ExportName: "apply", Epoch: 7,
	}
	for _, g := range []int{0, 2, 3} {
		if err := applyWASMRegistration(dir, rt1, reg1, st1, replicated, g, nil); err != nil {
			t.Fatalf("applyWASMRegistration(group %d): %v", g, err)
		}
	}

	// Process 2: the operator ALSO has a config module under the same name,
	// carrying different bytes. The replicated one must survive intact.
	_, reg2, st2, err := restartWASM(t, dir, []WASMModuleConfig{{
		Name: name, Kind: ops.OpReadOnly, Bytes: ro,
		ExportName: "apply",
	}})
	if err != nil {
		t.Fatalf("restart with a colliding config module: %v", err)
	}

	in, ok := st2.installed[name]
	if !ok {
		t.Fatal("the module is not installed at all after the restart")
	}
	if !in.replicated {
		t.Fatal("the install is recorded as config: the op drops out of the route-gate snapshot (publishWASMGateLocked gates only replicated installs), so the gate is OFF for it and the node proposes into groups with no evidence")
	}
	if in.reg.Blob != ops.WASMBlobFingerprint(incr) {
		t.Errorf("the node runs config bytes while its peers run replicated bytes — divergence (got blob %s, want the replicated %s)",
			ops.WASMBlobHex(in.reg.Blob), ops.WASMBlobHex(ops.WASMBlobFingerprint(incr)))
	}
	if in.reg.Epoch != 7 {
		t.Errorf("Epoch after restart = %d, want 7: it is erased, not re-derivable, and the convergence rule restarts from scratch", in.reg.Epoch)
	}
	if in.reg.Kind != ops.OpReadWrite {
		t.Errorf("Kind after restart = %d, want %d (read-write): the config module's contract took over, and an OpReadOnly op is served WITHOUT being proposed — this node would answer invocations its peers replicate",
			uint8(in.reg.Kind), uint8(ops.OpReadWrite))
	}
	if got, want := sortedGroups(in.groups), []int{0, 2, 3}; !sameInts(got, want) {
		t.Errorf("proven groups after restart = %v, want %v: the route-gate evidence is gone and cannot be re-derived", got, want)
	}

	// The registry must carry the REPLICATED module's kind and routability, not
	// the config module's — a config-registered OpReadOnly entry would serve the
	// op locally instead of replicating it.
	if _, _, ke, ok := reg2.Lookup(name); !ok || ke == nil {
		t.Errorf("registry entry after restart: ok=%v routable=%v, want a routable replicated op", ok, ke != nil)
	}

	// And it is still carried in the shard snapshot for a group it is proven in.
	blob := wasmSnapshotBlob(st2, 2)
	if blob == nil {
		t.Fatal("the op is missing from group 2's snapshot: a snapshot-joined replica would never receive it and would halt on the first invocation")
	}
	secs, err := decodeWASMSnapshotBlob(blob)
	if err != nil {
		t.Fatalf("decodeWASMSnapshotBlob: %v", err)
	}
	if len(secs.installs) != 1 || secs.installs[0].Name != name {
		t.Fatalf("snapshot install records = %+v, want one for %q", secs.installs, name)
	}
	if len(secs.bindings) != 1 || secs.bindings[0].Name != name || secs.bindings[0].Blob != ops.WASMBlobFingerprint(incr) {
		t.Fatalf("snapshot binding records = %+v, want group 2 bound to the replicated module", secs.bindings)
	}
}

// TestWASMReadOnlyWritesStateRegistrationLeavesNoFiles is the HIGH-4 gate, and
// THIN MARKERS MOVED THE GUARD IT WATCHES, so read the contract below before
// reading the assertions.
//
// WHAT IT USED TO PIN. A registration that cannot be registered must be a clean
// no-op. The old order wrote the .wasm and the sidecar, THEN compiled, THEN
// swapped the runtime module, and only THEN ran the OpReadOnly/writes-state
// guard — so a module declared read-only that imports cache_put had its files
// written on EVERY node and only then failed. The next restart's
// reloadWASMModulesFromDisk re-read them, hit the same non-ErrDuplicateOp failure
// and returned it, which fails node construction: no node in the cluster could
// start until someone deleted the file by hand. Both Kind and Bytes were
// wire-controlled, so a client could do this.
//
// WHY THE APPLY-TIME HALF OF THAT IS NOW THE WRONG ASSERTION. A marker names its
// module and does not carry it, so applyWASMRegistration has no imports to judge
// the declared Kind against on a node that has not fetched the blob — and it must
// not judge them even when it HAS, because a refusal that consulted the bytes
// would make the install depend on RESIDENCY: a node holding the module would
// refuse the registration a node without it installed, and the second would then
// propose invocations the first halts on with the classFatal
// shard.ErrOpNotRegistered. So the pairing verdict moved to
// wasm.Runtime.resolveModuleForInvoke, which asks it once per INVOCATION on every
// node and therefore reaches one verdict everywhere. See materializeWASMBlob.
//
// WHAT THIS TEST PINS NOW, which is the same safety property relocated:
//  1. the marker installs — no refusal, because there is nothing to refuse on;
//  2. the node still STARTS on the next restart, which was always the real
//     cluster-scale hazard and is now guaranteed by the reload never failing on a
//     module rather than by nothing having been written;
//  3. the guard still fires, at the moment the op is INVOKED with its bytes
//     resident — so the read-only-module-that-writes is still unable to execute;
//  4. the CONFIG path is UNCHANGED: config carries the bytes inline, so
//     loadOneModule still compiles and runs wasm.ValidateModuleKind before it
//     writes anything, and a refused config module still leaves no files.
func TestWASMReadOnlyWritesStateRegistrationLeavesNoFiles(t *testing.T) {
	const name = "wasm_ro_writer"
	put := readTestWASM(t, "../wasm/testdata/put.wasm") // imports cache_put

	dir := t.TempDir()
	rt, reg, st, err := restartWASM(t, dir, nil)
	if err != nil {
		t.Fatalf("first start: %v", err)
	}

	bad := ops.WASMRegistration{
		Name: name, Kind: ops.OpReadOnly, Blob: ops.WASMBlobFingerprint(put),
		ExportName: "apply", Epoch: 1,
	}
	// (1) The blob is deliberately NOT seeded: this is the ordinary state of a node
	// the push could not reach. The marker must install anyway.
	if err := applyWASMRegistration(dir, rt, reg, st, bad, 0, nil); err != nil {
		t.Fatalf("the marker must install even though its module is absent and its declared Kind is a lie: %v", err)
	}
	if _, ok := st.installed[name]; !ok {
		t.Error("the marker was not recorded as installed")
	}
	if _, _, _, ok := reg.Lookup(name); !ok {
		t.Error("the marker did not reach the ops registry: a committed invocation would halt this node with ErrOpNotRegistered")
	}
	id := wasm.ModuleIDForBlob(bad.Blob, bad.ExportName, bad.MaxFuel)
	if rt.HasModule(id) {
		t.Error("the module is resident although its blob was never written")
	}

	// (2) The node still starts on the next restart — with the op registered, not
	// with it silently gone.
	_, reg2, st2, err := restartWASM(t, dir, nil)
	if err != nil {
		t.Fatalf("restart with a sidecar whose blob is absent: %v (this is the cluster-wide start failure the reload must no longer produce)", err)
	}
	if _, ok := st2.installed[name]; !ok {
		t.Error("restart dropped the op: it comes up clean and halts on the first committed invocation instead")
	}
	if _, _, _, ok := reg2.Lookup(name); !ok {
		t.Error("restart did not re-register the op")
	}

	// (3) THE GUARD, AT ITS NEW LOCATION. Make the bytes resident on the runtime
	// the op was registered against, then run the handler exactly as a committed
	// entry would. wasm.Runtime.resolveModuleForInvoke is where the marker's Kind
	// and the module's imports are both in hand for the first time.
	seedWASMBlob(t, dir, put)
	if err := materializeWASMBlob(dir, rt, bad.Blob, bad.ExportName, bad.MaxFuel); err != nil {
		t.Fatalf("materializeWASMBlob: %v", err)
	}
	fn, _, _, ok := reg.Lookup(name)
	if !ok {
		t.Fatal("op vanished from the registry")
	}
	if _, err := fn(ops.NewTxContext(nil), []byte("k")); err == nil {
		t.Fatal("a read-only module that imports a state-mutating host function executed: an OpReadOnly op is served without being proposed, so its write is never logged and diverges this replica silently")
	} else if !strings.Contains(err.Error(), "read-only") {
		t.Errorf("the invocation failed for some other reason than the guard: %v", err)
	}

	// (4) Same guard, same requirement, on the CONFIG path — unchanged, because
	// config carries the bytes inline and so can still judge them before writing.
	cfgDir := t.TempDir()
	if _, _, _, err := restartWASM(t, cfgDir, []WASMModuleConfig{{
		Name: name, Kind: ops.OpReadOnly, Bytes: put, ExportName: "apply",
	}}); err == nil {
		t.Fatal("a config module must fail the same guard")
	}
	if files := wasmDirFiles(t, cfgDir); len(files) != 0 {
		t.Fatalf("the refused CONFIG module left files behind: %v", files)
	}
}

// TestWASMRefusedRegistrationDoesNotDisplaceTheRuntimeModule is HIGH-4's consequence
// B: when the guard fired under the old order, rt.AddModule had ALREADY replaced
// the runtime module while the ops.Registry entry and wasmState still described
// the previous one — so the handler executed the NEW bytes under the OLD Kind and
// key extractor.
//
// The content-addressed runtime makes the displacement UNREPRESENTABLE rather
// than merely avoided — a slot is named by its own content, so nothing can
// overwrite another version's slot — but the assertion is kept, and strengthened
// to check the ops.Registry handler too. That is now where the binding lives: the
// handler closure captures the ModuleID, so "which module does this op run" is
// the registry entry, and the entry is what must not have moved.
//
// THE REFUSAL DRIVING IT CHANGED WITH THIN MARKERS, and the substitution is the
// point rather than an accident. The OpReadOnly/writes-state guard was the
// apply-time refusal this test used to trigger; it needs the module's imports, so
// it moved to invocation time (see
// TestWASMReadOnlyWritesStateRegistrationLeavesNoFiles and
// wasm.Runtime.resolveModuleForInvoke). What is left at apply time is exactly the
// set of refusals that are PURE FUNCTIONS OF THE ENTRY, and this test now drives
// one of those — an out-of-range Kind byte (validateWASMRegistrationKind), which
// is also wire-controlled and also arrives on a strictly-newer registration. The
// property under test is unchanged: whatever the refusal, it must fire before any
// side effect, so the runtime, the registry entry and wasmState all still describe
// the previous registration afterwards.
func TestWASMRefusedRegistrationDoesNotDisplaceTheRuntimeModule(t *testing.T) {
	const name = "wasm_swap"
	incr := readTestWASM(t, "../wasm/testdata/incr.wasm")
	put := readTestWASM(t, "../wasm/testdata/put.wasm")

	dir := t.TempDir()
	rt, reg, st, err := restartWASM(t, dir, nil)
	if err != nil {
		t.Fatalf("first start: %v", err)
	}

	seedWASMBlob(t, dir, incr)
	good := ops.WASMRegistration{
		Name: name, Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr),
		ExportName: "apply", Epoch: 1,
	}
	if err := applyWASMRegistration(dir, rt, reg, st, good, 0, nil); err != nil {
		t.Fatalf("applyWASMRegistration: %v", err)
	}
	firstID := wasm.ModuleIDFor(incr, "apply", 0)
	if !rt.HasModule(firstID) {
		t.Fatal("precondition: the runtime must hold the first module")
	}

	// A strictly newer registration that an entry-derived guard must refuse. Its
	// blob is seeded, so nothing about the refusal depends on residency.
	seedWASMBlob(t, dir, put)
	bad := good
	bad.Blob = ops.WASMBlobFingerprint(put)
	bad.Kind = ops.OpKind(9) // outside the two values OpKind defines
	bad.Epoch = 2
	if err := applyWASMRegistration(dir, rt, reg, st, bad, 0, nil); err == nil {
		t.Fatal("the superseding registration must be refused")
	}

	if !rt.HasModule(firstID) {
		t.Error("the refused registration displaced the runtime module: the handler now runs the NEW bytes under the OLD registry kind and key extractor")
	}
	if rt.HasModule(wasm.ModuleIDFor(put, "apply", 0)) {
		t.Error("the refused module was instantiated on the runtime; the guard must fire before AddModule")
	}
	// The registry entry is where the op's module is BOUND now (the handler
	// closure captures the ModuleID), so the entry must still describe the first
	// registration in every respect.
	if _, kind, ke, ok := reg.Lookup(name); !ok || kind != ops.OpReadWrite || ke == nil {
		t.Errorf("the registry entry moved to the refused registration: ok=%v kind=%v routable=%v", ok, kind, ke != nil)
	}
	if in := st.installed[name]; in.reg.Blob != ops.WASMBlobFingerprint(incr) || in.reg.Epoch != 1 {
		t.Errorf("installed state moved to the refused registration: epoch=%d blob=%s", in.reg.Epoch, ops.WASMBlobHex(in.reg.Blob))
	}
}

// TestWASMTornSidecarDoesNotSilentlyOpenTheGate is the HIGH-5 gate.
//
// writeWASMSidecar was a bare os.WriteFile — truncate in place, no temp+rename,
// no fsync — and it runs up to k times per registration on a node hosting k
// groups, so the torn-file window is not narrow. The READ side swallowed
// unmarshal errors and fell back to the zero wasmMeta, whose every field is the
// worst available value:
//
//   - Source "" ⇒ replicated:false ⇒ the op drops out of the gate snapshot
//     (publishWASMGateLocked), so the gate returns nil for EVERY group and the
//     node proposes invocations into groups with no evidence at all. A corrupt
//     sidecar CAN falsely open the gate;
//   - Groups nil ⇒ the durable, non-re-derivable route-gate evidence is gone;
//   - Kind 0 ⇒ ops.OpReadOnly ⇒ the apply-time skew classifyApplyErr now has to
//     halt on;
//
// Refusing to load is the only safe answer: the registration is still in every
// group's Raft log, so deleting the pair and letting the node re-receive it is
// lossless, whereas guessing is permanent.
func TestWASMTornSidecarDoesNotSilentlyOpenTheGate(t *testing.T) {
	const name = "wasm_torn"
	// readonly.wasm imports no state-mutating host function, so it is the only
	// testdata module that SURVIVES the zero sidecar's Kind 0 (== OpReadOnly).
	// With a writing module the read-only guard rejects the reload and masks the
	// behaviour under test — which is whether a torn sidecar can be consumed
	// silently, not whether some later guard happens to catch one particular
	// module.
	incr := readTestWASM(t, "../wasm/testdata/readonly.wasm")

	dir := t.TempDir()
	rt, reg, st, err := restartWASM(t, dir, nil)
	if err != nil {
		t.Fatalf("first start: %v", err)
	}
	r := ops.WASMRegistration{
		Name: name, Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr),
		ExportName: "apply", Epoch: 3,
	}
	seedWASMBlob(t, dir, incr)
	for _, g := range []int{0, 1} {
		if err := applyWASMRegistration(dir, rt, reg, st, r, g, nil); err != nil {
			t.Fatalf("applyWASMRegistration(group %d): %v", g, err)
		}
	}

	// Tear the sidecar exactly as an interrupted truncate-in-place write would.
	sidecar := filepath.Join(dir, "wasm", name+".json")
	full, err := os.ReadFile(sidecar) //nolint:gosec // test-controlled path
	if err != nil {
		t.Fatalf("read sidecar: %v", err)
	}
	if err := os.WriteFile(sidecar, full[:len(full)/2], 0o600); err != nil {
		t.Fatalf("truncate sidecar: %v", err)
	}

	_, _, st2, err := restartWASM(t, dir, nil)
	if err == nil {
		in := st2.installed[name]
		t.Fatalf("a torn sidecar loaded silently: replicated=%v (gate %s), groups=%v, kind=%v",
			in.replicated, gateWord(in.replicated), sortedGroups(in.groups), in.reg.Kind)
	}
	if !strings.Contains(err.Error(), "unreadable") {
		t.Errorf("the failure does not identify the sidecar as the problem: %v", err)
	}
	if _, ok := st2.installed[name]; ok {
		t.Error("a module was installed from a sidecar that could not be read")
	}
}

func gateWord(replicated bool) string {
	if replicated {
		return "ARMED"
	}
	return "OFF for this op"
}

func sameInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestWASMSnapshotBlobRejectsAnImpossibleRecordCount pins the allocation clamp.
//
// The record count is an untrusted u32 read straight off the wire — it arrives
// inside an InstallSnapshot from a peer, or out of an object-store backup — and
// it used to size the result slice before a single record had been validated.
// Each record needs at least five header bytes, so a count that cannot fit in the
// remaining buffer is provably a lie and must be refused BEFORE the make().
func TestWASMSnapshotBlobRejectsAnImpossibleRecordCount(t *testing.T) {
	// count = 4294967295, zero record bytes, behind a valid version byte.
	blob := []byte{wasmSnapshotBlobVersion, 0xFF, 0xFF, 0xFF, 0xFF}
	if _, err := decodeWASMSnapshotBlob(blob); err == nil {
		t.Fatal("a record count that cannot fit in the buffer must be rejected before the pre-allocation")
	} else if !strings.Contains(err.Error(), "can fit") {
		t.Errorf("rejection does not identify the count as impossible: %v", err)
	}

	// The clamp must not reject a legitimate blob.
	incr := readTestWASM(t, "../wasm/testdata/incr.wasm")
	reg := ops.WASMRegistration{
		Name: "ok", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr),
		ExportName: "apply", Epoch: 1,
	}
	st := newWASMState()
	st.installed["ok"] = installedWASM{
		reg:        reg,
		replicated: true,
		groups:     testBindings(reg, 1),
	}
	good := wasmSnapshotBlob(st, 1)
	if _, err := decodeWASMSnapshotBlob(good); err != nil {
		t.Fatalf("a well-formed blob must still decode: %v", err)
	}
}

// TestWASMSnapshotBlobRejectsAnOverstatedRecordLength pins the per-record
// canonicality assertion.
//
// The blob-level trailing-bytes check is not enough on its own. Each record's
// length is an untrusted u32 and ops.DecodeWASMRegistration does not assert it
// CONSUMED its input, so a record that declares more bytes than its registration
// encodes to decodes cleanly; `off += n` then steps over the padding and the
// blob-level check still balances at the end. The padding rides into the restored
// state's provenance for free.
//
// This is defence in depth, not a live vector: the only producer is
// wasmSnapshotBlob and the only consumer path is a peer's InstallSnapshot, which is
// trusted input today.
func TestWASMSnapshotBlobRejectsAnOverstatedRecordLength(t *testing.T) {
	incr := readTestWASM(t, "../wasm/testdata/incr.wasm")
	enc := ops.EncodeWASMRegistration(ops.WASMRegistration{
		Name: "ok", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr),
		ExportName: "apply", Epoch: 1,
	})
	const pad = 16

	blob := make([]byte, 0, 1+4+4+len(enc)+pad+4)
	var lenBuf [4]byte
	blob = append(blob, wasmSnapshotBlobVersion)
	binary.BigEndian.PutUint32(lenBuf[:], 1) // one install record
	blob = append(blob, lenBuf[:]...)
	binary.BigEndian.PutUint32(lenBuf[:], uint32(len(enc)+pad))
	blob = append(blob, lenBuf[:]...)
	const bodyOff = 1 + 4 + 4
	blob = append(blob, enc...)
	blob = append(blob, bytes.Repeat([]byte{0xff}, pad)...)
	binary.BigEndian.PutUint32(lenBuf[:], 0) // no binding records
	blob = append(blob, lenBuf[:]...)

	// PRECONDITION: the record body must still DECODE, or the decode error is what
	// is being measured rather than the length assertion.
	if _, err := ops.DecodeWASMRegistration(blob[bodyOff:]); err != nil {
		t.Fatalf("precondition: the padded record must decode: %v", err)
	}

	if _, err := decodeWASMSnapshotBlob(blob); err == nil {
		t.Fatal("a record whose declared length overstates its encoding was accepted, padding and all")
	} else if !strings.Contains(err.Error(), "encodes to") {
		t.Errorf("rejection does not identify the length as overstated: %v", err)
	}
}

// TestWASMTwoOpsShareOneBlob is the dedup half of content addressing, and the direct
// consequence of addressing the blob by ops.WASMBlobFingerprint (a hash of the
// BYTES) rather than by ops.WASMRegistrationFingerprint (a hash of the whole
// registration).
//
// The same compiled module can legitimately back more than one op — here the same
// bytes registered under two NAMES, which is enough to give them DIFFERENT
// registration fingerprints. If the blob were named after the registration, the
// identical bytes would be stored twice, once per op, and the store would
// deduplicate nothing. Named after the bytes, both sidecars point at one file.
//
// The pair used to be "once routable and once shardless", differing in the key
// extractor as well as the name. Shardless is no longer a registrable shape —
// there is no extractor field (ops.WASMKeyExtractorHandle) — so the two now
// differ in Name alone, which is all this test ever needed.
func TestWASMTwoOpsShareOneBlob(t *testing.T) {
	incr := readTestWASM(t, "../wasm/testdata/incr.wasm")

	dir := t.TempDir()
	rt, reg, st, err := restartWASM(t, dir, nil)
	if err != nil {
		t.Fatalf("first start: %v", err)
	}

	routable := ops.WASMRegistration{
		Name: "wasm_shared_a", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr),
		ExportName: "apply", Epoch: 1,
	}
	second := ops.WASMRegistration{
		Name: "wasm_shared_b", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr),
		ExportName: "apply", Epoch: 1,
	}
	if ops.WASMRegistrationFingerprint(routable) == ops.WASMRegistrationFingerprint(second) {
		t.Fatal("precondition: the two registrations must have different registration fingerprints")
	}
	for _, r := range []ops.WASMRegistration{routable, second} {
		// One push per registration, exactly as a real node does: each registration
		// carries its module to this node's blob store independently, and content
		// addressing is what collapses the two into one file.
		seedWASMBlob(t, dir, incr)
		if err := applyWASMRegistration(dir, rt, reg, st, r, 0, nil); err != nil {
			t.Fatalf("applyWASMRegistration %q: %v", r.Name, err)
		}
	}

	blobs, err := os.ReadDir(filepath.Join(dir, "wasm", wasmBlobsSubdir))
	if err != nil {
		t.Fatalf("readdir blobs: %v", err)
	}
	if len(blobs) != 1 {
		names := make([]string, 0, len(blobs))
		for _, e := range blobs {
			names = append(names, e.Name())
		}
		t.Fatalf("two ops sharing one module produced %d blobs (%v), want 1", len(blobs), names)
	}
	// Both sidecars must actually name that one blob.
	want := blobPathFor(t, dir, incr)
	for _, name := range []string{routable.Name, second.Name} {
		meta, have, err := readWASMSidecar(dir, name)
		if err != nil || !have {
			t.Fatalf("read sidecar %q: have=%v err=%v", name, have, err)
		}
		got, err := wasmBlobPath(dir, meta.Blob)
		if err != nil {
			t.Fatalf("sidecar %q names an unusable blob %q: %v", name, meta.Blob, err)
		}
		if got != want {
			t.Errorf("sidecar %q points at %s, want the shared blob %s", name, got, want)
		}
	}

	// And a restart brings both ops back off that single blob.
	_, _, st2, err := restartWASM(t, dir, nil)
	if err != nil {
		t.Fatalf("restart: %v", err)
	}
	for _, name := range []string{routable.Name, second.Name} {
		in, ok := st2.installed[name]
		if !ok {
			t.Errorf("op %q did not reload from the shared blob", name)
			continue
		}
		if in.reg.Blob != ops.WASMBlobFingerprint(incr) {
			t.Errorf("op %q reloaded blob %s, want the shared module's %s", name, ops.WASMBlobHex(in.reg.Blob), ops.WASMBlobHex(ops.WASMBlobFingerprint(incr)))
		}
	}
}

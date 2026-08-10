// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/wasm"
)

// traversingNames is the set of op names that must never reach a filepath.Join.
// Every one of them resolves rather than fails when joined onto <dataDir>/wasm,
// and none of them is caught by the registry's own name rules (non-empty,
// length <= maxOpNameLen, length != 2 — see ops.validateEntry) except by accident:
// ".." happens to be two bytes long, which is why the directory-reference clause
// is checked before the length clauses in ops.ValidateWASMOpName. A path-shaped
// name must be refused with the reason it is path-shaped.
var traversingNames = []string{
	"../pwned",
	"../../../../etc/cron.d/rostam",
	"sub/mod",
	`..\pwned`,
	"a/../../b",
	"..",
	".",
}

// TestWASMRegistrationWithATraversingNameIsRefused is the CRITICAL gate.
//
// r.Name arrives off the wire and is turned into TWO filesystem paths —
// <dataDir>/wasm/<Name>.wasm and the .json sidecar beside it — with
// filepath.Join, which CLEANS a traversal instead of failing on it. A Name of
// "../../../../etc/cron.d/x" therefore wrote attacker-chosen bytes to an
// attacker-chosen path as the server user, on every replica that applied the
// entry. atomicWriteFile made it worse rather than better: its temp file is
// created in filepath.Dir(path), i.e. inside the traversed directory, so even a
// failed write landed there.
//
// The second-order damage is a cluster halt. reloadWASMModulesFromDisk globs only
// <dataDir>/wasm/*.wasm, so a traversed pair is invisible to it: after a restart
// the op is simply GONE on that node while invocation entries for it sit in every
// group's log, and each of those applies fails with shard.ErrOpNotRegistered —
// classFatal.
func TestWASMRegistrationWithATraversingNameIsRefused(t *testing.T) {
	const numShards = 2
	incr := readTestWASM(t, "../wasm/testdata/incr.wasm")

	t.Run("propose time", func(t *testing.T) {
		n := newTestNode(t, numShards)
		waitAllApplied(t, n)

		for _, name := range traversingNames {
			before := lastLogIndexes(t, n)
			_, err := n.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(ops.WASMRegistration{
				Name: name, Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr),
				ExportName: "apply", Epoch: 1,
			}, incr))
			if !errors.Is(err, ops.ErrWASMOpNameUnsafe) {
				t.Errorf("Call with name %q: got %v, want ops.ErrWASMOpNameUnsafe", name, err)
			}
			// The refusal has to survive server.clientFacingErr and
			// httpapi.statusForError, which key off this substring across the
			// stringifying Raft/RPC boundary. Redacted to "internal error" the caller
			// cannot tell a rejected name from a server fault.
			if err != nil && !strings.Contains(err.Error(), ops.WASMOpNameUnsafeMsg) {
				t.Errorf("refusal for %q would be redacted to a generic internal error: %q", name, err.Error())
			}
			// Nothing was proposed: the entry must not enter ANY group's log, or
			// every replica applies it and writes the file itself.
			for i, after := range lastLogIndexes(t, n) {
				if after != before[i] {
					t.Errorf("name %q: shard %d log grew (%d -> %d): the refused registration was still proposed", name, i, before[i], after)
				}
			}
		}

		// And nothing was written outside <dataDir>/wasm. "../pwned" resolves to
		// <dataDir>/pwned.wasm, which is the shape of the whole defect while staying
		// inside the test's own temp dir.
		for _, suffix := range []string{".wasm", ".json"} {
			escaped := filepath.Join(n.cfg.DataDir, "pwned"+suffix)
			if _, err := os.Stat(escaped); err == nil {
				t.Errorf("a wire-controlled name escaped the wasm dir: %s exists", escaped)
			}
		}
		if files := wasmDirFiles(t, n.cfg.DataDir); len(files) != 0 {
			t.Errorf("the refused registrations left files in the wasm dir: %v", files)
		}
	})

	// applyWASMRegistration is reached by a route that does NOT pass through
	// ops.RegisterWASMRegisterOp at all: installWASMSnapshotBlobLocked decodes a
	// peer's snapshot (or an object-store backup) and calls it directly. So the
	// check has to live there too, not only on the op handler.
	t.Run("apply time is identical on every replica", func(t *testing.T) {
		const replicas = 3
		msgs := make([]string, replicas)
		for i := 0; i < replicas; i++ {
			dir := t.TempDir()
			rt, reg, st, err := restartWASM(t, dir, nil)
			if err != nil {
				t.Fatalf("replica %d start: %v", i, err)
			}
			err = applyWASMRegistration(dir, rt, reg, st, ops.WASMRegistration{
				Name: "../pwned", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr),
				ExportName: "apply", Epoch: 1,
			}, 0, nil)
			if err == nil {
				t.Fatalf("replica %d applied a traversing name", i)
			}
			msgs[i] = err.Error()
			if _, statErr := os.Stat(filepath.Join(dir, "pwned.wasm")); statErr == nil {
				t.Errorf("replica %d wrote outside the wasm dir", i)
			}
		}
		// Determinism is what keeps the refusal safe at apply time: a rejection that
		// differed between replicas would itself be a divergence source.
		for i := 1; i < replicas; i++ {
			if msgs[i] != msgs[0] {
				t.Errorf("replicas refused differently:\n  %s\n  %s", msgs[0], msgs[i])
			}
		}
	})

	// The shard-scoped wrapper is the OTHER way into a group's log. It is
	// dispatched off n.adminOps, which only the MULTI-NODE constructor populates
	// (registerAdminOps is called from newMultiNode alone), so the test has to run
	// on that code path — which is also the only deployment shape where the op is
	// reachable at all.
	t.Run("shard-scoped entry point", func(t *testing.T) {
		n := newTestNodeMultiSingle(t, numShards)
		waitAllApplied(t, n)
		before := lastLogIndexes(t, n)
		payload := encodeShardScopedWASM(1, ops.EncodeWASMRegistration(ops.WASMRegistration{
			Name: "../pwned", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr),
			ExportName: "apply", Epoch: 1,
		}))
		if _, err := n.Call(opRegisterWASMShardName, payload); !errors.Is(err, ops.ErrWASMOpNameUnsafe) {
			t.Errorf("%s with a traversing name: got %v, want ops.ErrWASMOpNameUnsafe", opRegisterWASMShardName, err)
		}
		for i, after := range lastLogIndexes(t, n) {
			if after != before[i] {
				t.Errorf("shard %d log grew (%d -> %d)", i, before[i], after)
			}
		}
	})

	// Operator config joins the same two paths, so it gets the same rule — and
	// there it fails construction outright, since there is no cluster-wide truth to
	// honour and the operator is present to fix it.
	t.Run("config path", func(t *testing.T) {
		dir := t.TempDir()
		_, _, _, err := restartWASM(t, dir, []WASMModuleConfig{{
			Name: "../pwned", Kind: ops.OpReadWrite, Bytes: incr, ExportName: "apply",
		}})
		if !errors.Is(err, ops.ErrWASMOpNameUnsafe) {
			t.Fatalf("config module with a traversing name: got %v, want ops.ErrWASMOpNameUnsafe", err)
		}
		if _, statErr := os.Stat(filepath.Join(dir, "pwned.wasm")); statErr == nil {
			t.Error("a config module escaped the wasm dir")
		}
	})
}

// TestWASMModuleWithNoSidecarIsNotLoadedSilently is the HIGH gate for the MISSING
// sidecar, which the TORN-path hardening left open.
//
// reloadWASMModulesFromDisk used to glob <dataDir>/wasm/*.wasm and discard
// readWASMSidecar's `have` bool, so a module file with no .json beside it loaded
// with the ZERO wasmMeta — every field of which is the worst available value:
//
//   - Source "" ⇒ replicated:false ⇒ the op is dropped from publishWASMGateLocked
//     (the route gate is OFF for it) AND from replicatedRegs (it is omitted from
//     every snapshot this node takes, so a replica joining from one halts on the
//     first invocation);
//   - Kind 0 ⇒ ops.OpReadOnly ⇒ the errPBApplyReadOnly skew that is now classFatal;
//   - Groups nil ⇒ the durable, non-re-derivable route-gate evidence is gone.
//
// None of it surfaced: ExportName "" is not an error either (the backend defaults
// it to "apply"), so for a typical module the load SUCCEEDED.
//
// THE CONTENT-ADDRESSED LAYOUT MAKES THIS STRUCTURAL RATHER THAN CHECKED. Module
// bytes live at <dataDir>/wasm/blobs/<sha256>.wasm and the scan is over the
// SIDECARS, so a blob nothing references carries no op name, no Kind and no
// provenance — there is nothing it could be loaded as. This test plants exactly
// that state and pins that the reload ignores it and installs nothing.
func TestWASMModuleWithNoSidecarIsNotLoadedSilently(t *testing.T) {
	const name = "wasm_orphan"
	ro := readTestWASM(t, "../wasm/testdata/readonly.wasm")

	dir := t.TempDir()
	rt, reg, st, err := restartWASM(t, dir, nil)
	if err != nil {
		t.Fatalf("first start: %v", err)
	}
	r := ops.WASMRegistration{
		Name: name, Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(ro),
		ExportName: "apply", Epoch: 5,
	}
	seedWASMBlob(t, dir, ro)
	if err := applyWASMRegistration(dir, rt, reg, st, r, 0, nil); err != nil {
		t.Fatalf("applyWASMRegistration: %v", err)
	}
	// Delete the sidecar, leaving the blob: the module's bytes are still on disk
	// with nothing describing them.
	if err := os.Remove(filepath.Join(dir, "wasm", name+".json")); err != nil {
		t.Fatalf("remove sidecar: %v", err)
	}
	sum := ops.WASMBlobFingerprint(ro)
	blob := filepath.Join(dir, "wasm", wasmBlobsSubdir, hex.EncodeToString(sum[:])+".wasm")
	if _, statErr := os.Stat(blob); statErr != nil {
		t.Fatalf("precondition: the blob must survive the sidecar deletion: %v", statErr)
	}

	_, _, st2, err := restartWASM(t, dir, nil)
	if err != nil {
		t.Fatalf("restart over an unreferenced blob must succeed, got: %v", err)
	}
	if in, ok := st2.installed[name]; ok {
		t.Fatalf("a module with no sidecar loaded silently: replicated=%v, kind=%v, epoch=%d, groups=%v",
			in.replicated, in.reg.Kind, in.reg.Epoch, sortedGroups(in.groups))
	}
	if len(st2.installed) != 0 {
		t.Errorf("restart installed %d modules from a data dir with no sidecars", len(st2.installed))
	}
}

// TestWASMSidecarWithMissingOrCorruptBlobIsRefused pins the reload's answer to an
// unusable blob, and THIN MARKERS REVERSED IT for the first two of its three
// cases. Read the contract before the assertions.
//
// WHAT IT USED TO PIN. The sidecar is what the reload scan finds, so its blob MUST
// resolve and must hash to its own filename; anything else failed node
// construction. That was defensible while a registration CARRIED its module: a
// node that had applied the entry provably had the bytes, so "sidecar here, blob
// gone" could only mean local corruption, and refusing to start was a true
// statement about a broken data dir. (It was already the better half of a pair:
// the OLD <name>.wasm layout scanned *.wasm, so deleting the module left the .json
// unread, the op vanished from the registry with no complaint, and the node halted
// under classFatal on the first committed invocation of it.)
//
// WHY THAT PREMISE IS GONE. A marker NAMES its module and does not carry it, so
// applying a registration no longer implies holding the bytes. "The sidecar is
// here and the blob is not" is now an ORDINARY, EXPECTED state: this node was
// unreachable during the registration's push, or it was restarted between applying
// the marker and finishing the fetch. Refusing to start on it is the worst
// available answer — it converts a self-healing condition (fetch the blob) into an
// outage of EVERY shard group this node hosts, on a node that is otherwise
// completely healthy, at exactly the moment an operator restarted it hoping to fix
// something. And it would be unsafe as well as unhelpful: a start-time refusal
// that consults the bytes makes node availability a function of RESIDENCY, which
// differs legitimately between replicas of the same group.
//
// A corrupt blob is the same condition reached differently: the file on disk is
// not the module the marker names, so this node does not hold that module. It is
// still DETECTED — the self-verification gate is what detects it, and it is why
// bit rot or a hand-edited file can never be compiled and run — but the verdict is
// "not resident", not "refuse to start". The bytes are then re-fetched from a peer
// that has them, which repairs the corruption instead of merely reporting it.
//
// WHAT IS PINNED NOW, per case:
//   - missing blob and corrupt blob: the reload SUCCEEDS, the op IS registered
//     (withholding it would recreate the classFatal ErrOpNotRegistered halt on
//     the first entry replayed out of a group's log), the module is NOT resident,
//     and a prefetch is kicked for exactly the fingerprint that is missing;
//   - a sidecar naming NO blob at all is UNCHANGED and still refuses: that is not
//     a residency condition, it is a data dir written by a build that predates the
//     content-addressed store, and there is no fingerprint to fetch.
func TestWASMSidecarWithMissingOrCorruptBlobIsRefused(t *testing.T) {
	const name = "wasm_blobcheck"
	ro := readTestWASM(t, "../wasm/testdata/readonly.wasm")
	roFP := ops.WASMBlobFingerprint(ro)

	seed := func(t *testing.T) (string, string) {
		t.Helper()
		dir := t.TempDir()
		rt, reg, st, err := restartWASM(t, dir, nil)
		if err != nil {
			t.Fatalf("first start: %v", err)
		}
		r := ops.WASMRegistration{
			Name: name, Kind: ops.OpReadWrite, Blob: roFP,
			ExportName: "apply", Epoch: 5,
		}
		seedWASMBlob(t, dir, ro)
		if err := applyWASMRegistration(dir, rt, reg, st, r, 0, nil); err != nil {
			t.Fatalf("applyWASMRegistration: %v", err)
		}
		return dir, filepath.Join(dir, "wasm", wasmBlobsSubdir, hex.EncodeToString(roFP[:])+".wasm")
	}

	// reloadRecordingPrefetch replays node construction's reload over dir with a
	// fresh runtime, registry and state, and records every fingerprint the reload
	// asked to fetch. restartWASM cannot be used here because it passes a nil
	// prefetch, and "a prefetch was kicked" is half of the new contract.
	reloadRecordingPrefetch := func(t *testing.T, dir string) (*wasm.Runtime, *ops.Registry, *wasmState, [][32]byte, error) {
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
		var asked [][32]byte
		err = reloadWASMModulesFromDisk(dir, rt, reg, st, func(fp [32]byte) { asked = append(asked, fp) })
		return rt, reg, st, asked, err
	}

	// assertInstalledButNotResident is the whole new contract in one place: come up,
	// register the op, do not pretend to hold the module, and go and get it.
	assertInstalledButNotResident := func(t *testing.T, rt *wasm.Runtime, reg *ops.Registry, st *wasmState, asked [][32]byte, err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("the reload must SUCCEED on a blob it cannot use: refusing takes down every group this node hosts on an otherwise healthy node, and the condition is self-healing. got: %v", err)
		}
		in, ok := st.installed[name]
		if !ok {
			t.Fatal("the op was not installed: it comes up silently absent and the node halts on the first committed invocation of it")
		}
		if in.reg.Blob != roFP {
			t.Errorf("installed blob = %s, want %s", ops.WASMBlobHex(in.reg.Blob), ops.WASMBlobHex(roFP))
		}
		if _, _, _, found := reg.Lookup(name); !found {
			t.Error("the op is missing from the ops registry: an entry replayed out of a group's log halts this node with classFatal ErrOpNotRegistered")
		}
		if _, bound := in.groups[0]; !bound {
			t.Error("group 0's binding was not restored: the route gate shuts and any committed invocation halts under ErrWASMNoGroupBinding")
		}
		id := wasm.ModuleIDForBlob(roFP, "apply", 0)
		if rt.HasModule(id) {
			t.Error("the module is reported RESIDENT although its bytes are unusable: an invocation would execute something other than the module the marker names")
		}
		found := false
		for _, fp := range asked {
			if fp == roFP {
				found = true
			}
		}
		if !found {
			t.Errorf("no prefetch was kicked for %s (asked for %d fingerprints): the op is registered but its bytes never arrive, so every invocation of it blocks forever", ops.WASMBlobHex(roFP), len(asked))
		}
	}

	t.Run("missing blob", func(t *testing.T) {
		dir, blob := seed(t)
		if err := os.Remove(blob); err != nil {
			t.Fatalf("remove blob: %v", err)
		}
		rt, reg, st, asked, err := reloadRecordingPrefetch(t, dir)
		assertInstalledButNotResident(t, rt, reg, st, asked, err)
	})

	// THE SELF-VERIFICATION GATE. A blob is valid iff sha256(contents) == its own
	// filename, so bit rot, a truncated write, or a hand-edited file is caught at
	// load rather than compiled and run. The old <name>.wasm layout could check
	// nothing: the filename carried no claim about the contents. What changed is the
	// CONSEQUENCE of catching it — not a start-time refusal any more, but the same
	// "this node does not hold that module" verdict a missing file gets, repaired by
	// the fetch rather than by an operator.
	t.Run("blob contents do not hash to its filename", func(t *testing.T) {
		dir, blob := seed(t)
		other := readTestWASM(t, "../wasm/testdata/incr.wasm")
		if bytes.Equal(other, ro) {
			t.Fatal("test fixtures are identical; pick two different modules")
		}
		// A perfectly VALID wasm module, just not the one this path names — so the
		// only thing that can reject it is the hash check.
		if err := os.WriteFile(blob, other, 0o600); err != nil {
			t.Fatalf("swap the blob contents: %v", err)
		}
		rt, reg, st, asked, err := reloadRecordingPrefetch(t, dir)
		assertInstalledButNotResident(t, rt, reg, st, asked, err)
		// And the swapped-in module must NOT have been loaded under the name the
		// sidecar gives: that is the whole point of hashing the contents.
		if rt.HasModule(wasm.ModuleIDFor(other, "apply", 0)) {
			t.Error("the substituted module was instantiated: a blob's contents must be checked against its own name, not trusted from the path it was read at")
		}
	})

	// A sidecar written by a build that predates the blob store names no blob at
	// all. It must refuse rather than fall back to reading <name>.wasm.
	t.Run("sidecar predating the blob store", func(t *testing.T) {
		dir, _ := seed(t)
		raw, err := os.ReadFile(filepath.Join(dir, "wasm", name+".json"))
		if err != nil {
			t.Fatalf("read sidecar: %v", err)
		}
		var meta map[string]any
		if err := json.Unmarshal(raw, &meta); err != nil {
			t.Fatalf("unmarshal sidecar: %v", err)
		}
		delete(meta, "blob")
		stripped, err := json.Marshal(meta)
		if err != nil {
			t.Fatalf("marshal sidecar: %v", err)
		}
		if err := os.WriteFile(filepath.Join(dir, "wasm", name+".json"), stripped, 0o600); err != nil {
			t.Fatalf("write stripped sidecar: %v", err)
		}
		// The pre-blob-store layout also kept the bytes here; the refusal must not
		// be satisfied by finding them.
		if err := os.WriteFile(filepath.Join(dir, "wasm", name+".wasm"), ro, 0o600); err != nil {
			t.Fatalf("plant the legacy module file: %v", err)
		}
		_, _, _, err = restartWASM(t, dir, nil)
		if err == nil {
			t.Fatal("a Blob-less sidecar must refuse to load")
		}
		if !errors.Is(err, errWASMBlob) {
			t.Errorf("refusal is not the blob error: %v", err)
		}
		// The message is the point of the readWASMSidecar check specifically.
		// wasmBlobPath would reject the empty fingerprint anyway (it is not 32 hex
		// bytes), so without the dedicated check the operator is told the sidecar
		// names a malformed hash rather than that the data dir predates the blob
		// store — which is the one fact that tells them what to do about it.
		if !strings.Contains(err.Error(), "names no blob") {
			t.Errorf("the refusal does not name the pre-blob-store layout as the cause: %v", err)
		}
	})
}

// TestWASMSidecarWriteFailureLeavesDiskOnThePreviousRegistration is the
// producer-side half. The module bytes and the sidecar are two INDEPENDENTLY
// atomic writes with no cross-file atomicity, so the window between them decides
// what a restart sees.
//
// Under the old <name>.wasm layout that window was actively dangerous: the bytes
// had ALREADY been overwritten with the new module while the sidecar still
// described the old Kind, extractor and Epoch, so a restart loaded that pair
// happily and ran new code under the old contract. The fix then was to UNLINK the
// .wasm, which traded it for an orphan the next restart refused to start on.
//
// Content addressing removes the whole dilemma, and this test pins the stronger
// property that replaced it: after a failed sidecar write, disk still describes
// the PREVIOUS registration completely, the node restarts cleanly on it, and the
// new bytes are inert unreferenced content. Nothing is unlinked.
//
// The fault is injected by making the sidecar path a DIRECTORY, so
// atomicWriteFile's final rename fails with EISDIR — a real filesystem refusal on
// the exact syscall, not a stubbed-out writer.
func TestWASMSidecarWriteFailureLeavesDiskOnThePreviousRegistration(t *testing.T) {
	const name = "wasm_nospace"
	incr := readTestWASM(t, "../wasm/testdata/incr.wasm")

	dir := t.TempDir()
	rt, reg, st, err := restartWASM(t, dir, nil)
	if err != nil {
		t.Fatalf("first start: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "wasm", name+".json"), 0o750); err != nil {
		t.Fatalf("plant the sidecar obstruction: %v", err)
	}

	err = applyWASMRegistration(dir, rt, reg, st, ops.WASMRegistration{
		Name: name, Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr),
		ExportName: "apply", Epoch: 1,
	}, 0, nil)
	if err == nil {
		t.Fatal("precondition: the sidecar write must fail")
	}
	if _, ok := st.installed[name]; ok {
		t.Error("the failed registration was recorded as installed")
	}

	// THE APPLY PATH NO LONGER WRITES A BLOB AT ALL, which is why this assertion
	// inverted. Under inline registration bytes, applyWASMRegistration wrote the
	// blob and THEN the sidecar, and the point being pinned was that the failed
	// second write left the first behind harmlessly — content addressing means an
	// unreferenced blob cannot be mistaken for a registration.
	//
	// A marker names its module instead of carrying it, so the blob is put on disk
	// by the pre-registration push, a fetch, or __wasm_blob_put__ — never by this
	// path. The sidecar is now the ONLY artifact an apply writes, which strengthens
	// the crash argument rather than weakening it: there is no longer a PAIR of
	// writes that can be interrupted between. So what is pinned here now is that a
	// failed apply leaves the blob store exactly as it found it — empty, in this
	// case, because nothing has pushed these bytes to this node.
	sum := ops.WASMBlobFingerprint(incr)
	blobPath := filepath.Join(dir, "wasm", wasmBlobsSubdir, hex.EncodeToString(sum[:])+".wasm")
	if _, statErr := os.Stat(blobPath); statErr == nil {
		t.Fatal("the apply path wrote a blob; a thin marker carries no module, so an apply must never create one — the bytes arrive by push, fetch or __wasm_blob_put__")
	}

	// The obstruction is what a real ENOSPC would not leave behind, so clear it
	// before asserting on the restart — the property under test is that the BLOB
	// does not break startup, not that a directory named <op>.json does.
	if err := os.Remove(filepath.Join(dir, "wasm", name+".json")); err != nil {
		t.Fatalf("clear the obstruction: %v", err)
	}
	_, _, st2, err := restartWASM(t, dir, nil)
	if err != nil {
		t.Fatalf("restart after a failed sidecar write must succeed, got: %v", err)
	}
	if len(st2.installed) != 0 {
		t.Errorf("restart installed %d modules for a registration that never committed", len(st2.installed))
	}
}

// TestRegisterWASMShardCannotBypassTheGates is the HIGH gate for the
// admin-reachable bypass.
//
// Node.Call dispatches n.adminOps FIRST, before the __register_wasm__ intercept,
// and __register_wasm_shard__ is in that map. It routed straight to
// handleRegisterWASMShard → callHostedShard → propose with NO update gate and NO
// size cap. The op is not in authz's allowlist by name, so it falls through to the
// deny-by-default "admin" bucket — which means it is reachable by any
// admin-authenticated EXTERNAL client, not only by peers carrying the internal
// service token.
//
// The update-gate bypass is the worse of the two: it drives a CONTRACT-CHANGING
// registration into exactly ONE group's log, leaving one group serving the op
// read-only, or routing it by a different key, while every other group does not.
// A bytes-differing registration is no longer the divergent shape — per-group
// version binding made that safe — but Kind is read
// node-wide on the propose side, so they cannot be per-group and a single group's
// log must not be able to move them.
func TestRegisterWASMShardCannotBypassTheGates(t *testing.T) {
	const numShards, target = 4, 1
	// n.adminOps is populated by registerAdminOps, which only newMultiNode calls,
	// so __register_wasm_shard__ exists only on the multi-node code path. That is
	// also the only shape in which the bypass is reachable — and the shape every
	// real deployment of this feature has.
	n := newTestNodeMultiSingle(t, numShards)
	waitAllApplied(t, n)

	if _, ok := n.adminOps[opRegisterWASMShardName]; !ok {
		t.Fatalf("precondition: %s must be dispatchable off adminOps", opRegisterWASMShardName)
	}
	if _, err := n.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(routableWASMReg(t, "wasm_incr", 1), readIncrWASM(t))); err != nil {
		t.Fatalf("first registration: %v", err)
	}
	wantFP := ops.WASMRegistrationFingerprint(installedReg(t, n, "wasm_incr"))

	// A CONTRACT change, expressed in Kind. It has to be Kind: the key extractor
	// is pinned to one legal value cluster-wide (ops.WASMKeyExtractorHandle), so a
	// registration that differed there would be refused by
	// validateWASMKeyExtractor before the update gate ever ran, and this test
	// would be asserting the wrong refusal.
	differing := routableWASMReg(t, "wasm_incr", 2)
	differing.Kind = ops.OpReadOnly

	// THE SIZE CAP THIS ENTRY POINT ENFORCES IS THE FRAME CAP NOW. A marker names
	// its module and does not carry it, so "an oversized module" is not a shape the
	// shard-scoped leg can be handed at all — the bytes never travel this way. What
	// it must still refuse, and what the bypass used to let through, is an oversized
	// encoded FRAME headed for a Raft log. The bulk goes in ExportName because
	// nothing else bounds that field, so the frame cap is the only check that can
	// catch it.
	oversized := routableWASMReg(t, "wasm_oversized", 1)
	oversized.ExportName = strings.Repeat("x", maxWASMRegistrationFrame+1)

	for _, tc := range []struct {
		what string
		reg  ops.WASMRegistration
		want error
	}{
		{"an in-place contract change", differing, ErrWASMUpdateUnsupported},
		{"an oversized frame", oversized, nil},
	} {
		t.Run(tc.what, func(t *testing.T) {
			before := lastLogIndexes(t, n)
			_, err := n.Call(opRegisterWASMShardName, encodeShardScopedWASM(target, ops.EncodeWASMRegistration(tc.reg)))
			if err == nil {
				t.Fatalf("%s slipped through the shard-scoped entry point", tc.what)
			}
			if tc.want != nil && !errors.Is(err, tc.want) {
				t.Errorf("got %v, want %v", err, tc.want)
			}
			if tc.want == nil && !strings.Contains(err.Error(), "over the") {
				t.Errorf("refusal does not name the size cap: %v", err)
			}
			for i, after := range lastLogIndexes(t, n) {
				if after != before[i] {
					t.Errorf("shard %d log grew (%d -> %d): the refused registration reached a group's log — one group on a different version than the rest is the maximally divergent state",
						i, before[i], after)
				}
			}
		})
	}

	if got := ops.WASMRegistrationFingerprint(installedReg(t, n, "wasm_incr")); got != wantFP {
		t.Error("the live module changed under a refused shard-scoped registration")
	}

	// The legitimate use of this op — one leg of a broadcast carrying the SAME
	// registration — must still work, or the checks above have broken the fan-out.
	if _, err := n.Call(opRegisterWASMShardName, encodeShardScopedWASM(target, ops.EncodeWASMRegistration(routableWASMReg(t, "wasm_incr", 1)))); err != nil {
		t.Fatalf("an identical shard-scoped leg is a normal broadcast hop and must be allowed: %v", err)
	}
}

// TestIdenticalWASMReRegistrationIsAllowedAfterARestart is the highest-value gate
// in this file, and nothing caught what it catches.
//
// After a restart cur.reg is not the struct that was accepted — it is REBUILT
// FIELD BY FIELD from the wasmMeta sidecar (reloadWASMModulesFromDisk). Any field
// that comes back as its zero value silently changes what this node does with the
// next registration under that name, on three separate paths:
//
//   - checkWASMUpdateGate compares the incoming Kind
//     against it, so a dropped one turns every post-restart retry of the identical
//     registration into a refused contract change;
//   - the node-wide fold is a maximum under (Epoch, fingerprint), so a dropped
//     Epoch or a dropped fingerprint-covered field makes a restarted node settle on
//     a different winner than its peers;
//   - the per-group binding's ModuleID covers bytes, export name and fuel cap, so
//     a dropped one binds a group to a version it never committed.
//
// The first of those kills the partial-broadcast repair path silently: re-sending
// the same registration is the ONLY way a group the first broadcast starved ever
// receives it, and a restarted node would refuse to do it. The op stays
// permanently unroutable for every key that lands in the starved group, and the
// error blames the client for a change it never attempted.
//
// Every fingerprint-covered field is therefore set to a NON-ZERO, non-default
// value here: dropping any one of them from wasmMeta fails this test.
func TestIdenticalWASMReRegistrationIsAllowedAfterARestart(t *testing.T) {
	const numShards = 2
	dir := t.TempDir()

	reg := ops.WASMRegistration{
		Name:       "wasm_incr",
		Kind:       ops.OpReadWrite,
		Blob:       ops.WASMBlobFingerprint(readIncrWASM(t)),
		ExportName: "apply",
		MaxFuel:    5_000_000,
		Epoch:      7,
	}
	wantFP := ops.WASMRegistrationFingerprint(reg)

	n1 := newTestNodeAt(t, dir, numShards)
	waitAllApplied(t, n1)
	if _, err := n1.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(reg, readIncrWASM(t))); err != nil {
		t.Fatalf("first registration: %v", err)
	}
	if err := n1.Close(); err != nil {
		t.Fatalf("close n1: %v", err)
	}

	n2 := newTestNodeAt(t, dir, numShards)
	waitAllApplied(t, n2)

	// The direct statement of the invariant: what the sidecar reconstructed is the
	// same registration, field for field, that was accepted before the restart.
	got := installedReg(t, n2, "wasm_incr")
	if gotFP := ops.WASMRegistrationFingerprint(got); gotFP != wantFP {
		t.Errorf("the reconstructed registration does not match the accepted one (kind=%v export=%q fuel=%d): a fingerprint-covered field is missing from wasmMeta, so every retry after a restart is refused",
			got.Kind, got.ExportName, got.MaxFuel)
	}
	if got.Epoch != reg.Epoch {
		t.Errorf("Epoch after restart = %d, want %d: it orders the node-wide fold separately from the fingerprint, so a dropped Epoch makes this node settle on a different winner than its peers", got.Epoch, reg.Epoch)
	}

	// The behaviour those fields exist for.
	if _, err := n2.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(reg, readIncrWASM(t))); err != nil {
		t.Fatalf("re-sending the identical registration after a restart is the partial-broadcast repair path and must be allowed: %v", err)
	}
}

// TestIdenticalWASMReRegistrationIsAllowedOnANonOriginatingNode covers the retry
// aimed at a node OTHER than the one that first accepted the registration — the
// normal case for any client that rotates over a node list.
//
// The receiving node's install state was built by APPLYING a replicated entry,
// not by originating a Call, so this is the branch of checkWASMUpdateGate that a
// production retry actually lands on.
//
// HONEST LIMITATION: this uses a second single-node cluster and feeds it the
// registration through applyWASMRegistration — the exact function the FSM hook
// calls — rather than standing up a real multi-node cluster, which is expensive
// and flaky here. That is a faithful stand-in because the two paths are the same
// code: applyWASMRegistration records installedWASM{reg: r}, so an originating
// node and a replicating one end up with byte-identical cur.reg. The place where
// they genuinely DIVERGE is the restart path, where cur.reg is reconstructed from
// the sidecar instead — which is why the test above exists and is the stronger of
// the two.
func TestIdenticalWASMReRegistrationIsAllowedOnANonOriginatingNode(t *testing.T) {
	const numShards = 2
	reg := routableWASMReg(t, "wasm_incr", 3)

	origin := newTestNode(t, numShards)
	waitAllApplied(t, origin)
	if _, err := origin.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(reg, readIncrWASM(t))); err != nil {
		t.Fatalf("registration on the originating node: %v", err)
	}

	peer := newTestNode(t, numShards)
	waitAllApplied(t, peer)
	peer.wasmApplyMu.Lock()
	err := applyWASMRegistration(peer.cfg.DataDir, peer.wasmRT, peer.cfg.Ops, peer.wasmState, reg, 0, nil)
	peer.publishWASMGateLocked()
	peer.wasmApplyMu.Unlock()
	if err != nil {
		t.Fatalf("replicating the entry to the peer: %v", err)
	}

	if _, err := peer.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(reg, readIncrWASM(t))); err != nil {
		t.Fatalf("a retry aimed at a node that did not originate the registration must be allowed: %v", err)
	}
	// A CONTRACT-CHANGING one aimed at the same node is still refused — otherwise
	// the test above would pass just as well against a gate that had been deleted.
	// (A bytes-only update is deliberately NOT refused any more: per-group version
	// binding made it safe. See TestWASMBytesUpdateIsAcceptedEndToEnd.)
	differing := reg
	differing.Kind = ops.OpReadOnly
	differing.Epoch = reg.Epoch + 1
	if _, err := peer.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(differing, readIncrWASM(t))); !errors.Is(err, ErrWASMUpdateUnsupported) {
		t.Errorf("a contract-changing registration on the same node: got %v, want ErrWASMUpdateUnsupported", err)
	}
}

// TestConfigInstalledWASMIsNotGuardedByTheUpdateGate pins the `!cur.replicated`
// carve-out at wasm_gate.go, which had no coverage at all — so simplifying the
// gate to `if have { … }` would have passed the entire suite while breaking the
// cluster.
//
// A module from cfg.WASMModules is node-LOCAL operator state that was never
// proposed into anyone's log. applyWASMRegistration deliberately lets a replicated
// registration override it (see wasmMeta.Source: if a config module could win, a
// node that happens to have it configured would run different bytes from one that
// does not — divergence produced by the rule meant to prevent it). Guarding it
// here would break that override on exactly the subset of nodes that carry the
// config, which is worse than the update it would prevent.
func TestConfigInstalledWASMIsNotGuardedByTheUpdateGate(t *testing.T) {
	const name, numShards = "wasm_shared", 2
	put := readTestWASM(t, "../wasm/testdata/put.wasm")
	incr := readIncrWASM(t)

	n := newTestNode(t, numShards)
	waitAllApplied(t, n)

	// Install a CONFIG module through the production loader, against the live
	// node's data dir, runtime and state — the same call the constructors make.
	n.wasmApplyMu.Lock()
	loaded, err := loadWASMModules(n.cfg.DataDir, []WASMModuleConfig{{
		Name: name, Kind: ops.OpReadWrite, Bytes: put,
		ExportName: "apply",
	}}, n.wasmRT, n.wasmState)
	n.publishWASMGateLocked()
	n.wasmApplyMu.Unlock()
	if err != nil {
		t.Fatalf("loadWASMModules: %v", err)
	}
	for _, lm := range loaded {
		c := lm.Registration
		if err := wasm.RegisterModule(n.cfg.Ops, n.wasmRT, c.Name, lm.ModuleID, c.Kind, ops.WASMKeyExtractor()); err != nil {
			t.Fatalf("register config module: %v", err)
		}
	}
	if in := n.wasmState.installed[name]; in.replicated {
		t.Fatal("precondition: the config module must be recorded as non-replicated")
	}

	// A replicated registration under the SAME name with DIFFERENT content. It
	// must be accepted — the gate keys on cur.replicated, not on mere presence.
	replicated := ops.WASMRegistration{
		Name: name, Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr),
		ExportName: "apply", Epoch: 4,
	}
	if _, err := n.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(replicated, incr)); err != nil {
		t.Fatalf("a replicated registration must override a config module, not be refused as an update: %v", err)
	}

	in := n.wasmState.installed[name]
	if !in.replicated {
		t.Fatal("the install is still recorded as config: the route gate stays OFF for this op and it is omitted from every snapshot this node takes")
	}
	if in.reg.Blob != ops.WASMBlobFingerprint(incr) {
		t.Errorf("the node kept the config blob (%s) instead of the replicated one (%s): it now runs different code from every peer that lacks the config", ops.WASMBlobHex(in.reg.Blob), ops.WASMBlobHex(ops.WASMBlobFingerprint(incr)))
	}

	// And the contract refusal is back in force now that the install IS replicated
	// — so this test cannot pass by the gate simply being gone. (Only the CONTRACT
	// is refused: a bytes-only update of a replicated op is supported.)
	second := replicated
	second.Kind = ops.OpReadOnly
	second.Epoch = replicated.Epoch + 1
	if _, err := n.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(second, incr)); !errors.Is(err, ErrWASMUpdateUnsupported) {
		t.Errorf("after the override the name is replicated and a contract change must be refused: got %v", err)
	}
}

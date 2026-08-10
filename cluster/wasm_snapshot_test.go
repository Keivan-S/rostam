// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/rostamlabs/rostam/ops"
)

// TestWASMRegistrationSurvivesSnapshotInstall is the regression gate for the
// DETERMINISTIC half of the missing-module bug.
//
// A WASM registration reaches a replica as a committed log entry. A replica that
// is brought up by InstallSnapshot — every AddShardOwner-joined owner, and every
// replica that catches up after the log has been compacted past the registration
// — applies NONE of those entries: the snapshot replaced them. The shard FSM
// snapshot carried cache + vector state only, and Restore did not re-run the
// registration hook, so such a replica was permanently missing every
// dynamically-registered op. That is not a race; it happens every time.
//
// It matters more now that shard.classifyApplyErr treats ErrOpNotRegistered as
// classFatal: the replica no longer silently skips those invocations, it HALTS.
// So without this fix, a snapshot-installed replica is not merely divergent, it
// is guaranteed to stop on the first invocation its peers execute normally.
//
// The test takes a real backup snapshot from a node that HAS the registration
// and installs it into a fresh node that has never seen it, then requires the op
// to be registered, on disk, and invocable there.
func TestWASMRegistrationSurvivesSnapshotInstall(t *testing.T) {
	const numShards = 1
	wasmBytes := readIncrWASM(t)
	ctx := context.Background()

	source := newTestNode(t, numShards)
	payload := ops.EncodeWASMRegistrationRequest(ops.WASMRegistration{
		Name: "wasm_incr",
		Kind: ops.OpReadWrite,
		Blob: ops.WASMBlobFingerprint(wasmBytes),
		// Routable, so the op is dispatched through shardOf(args)'s group — the
		// configuration in which a missing registration is actually reachable.
		ExportName: "apply",
		Epoch:      1,
	}, wasmBytes)
	if _, err := source.Call(wasmRegisterOpName, payload); err != nil {
		t.Fatalf("register on the source node: %v", err)
	}

	data, appliedIndex, err := source.shards[0].BackupSnapshot(ctx)
	if err != nil {
		t.Fatalf("BackupSnapshot: %v", err)
	}
	if len(data) == 0 {
		t.Fatal("BackupSnapshot returned no data")
	}

	// A pristine node: its own data dir, its own ops registry, and no
	// registration has ever been applied to it.
	target := newTestNodeAt(t, t.TempDir(), numShards)
	if _, _, _, ok := target.cfg.Ops.Lookup("wasm_incr"); ok {
		t.Fatal("precondition: the target node must NOT already know the op")
	}

	// The snapshot carries the MARKER, not the module: a marker names its module by
	// content address, so a restored replica installs the registration and FETCHES
	// the bytes from a member that holds them. This target is a standalone node with
	// no peer to fetch from, so the push's effect is seeded by hand — which is what
	// keeps assertion 3 below a statement about the RESTORE rather than about the
	// fetcher.
	seedWASMBlob(t, target.cfg.DataDir, wasmBytes)

	if err := target.shards[0].RestoreSnapshot(ctx, data, appliedIndex); err != nil {
		t.Fatalf("RestoreSnapshot into the target node: %v", err)
	}

	// 1. The op is in the target's registry...
	_, _, ke, ok := target.cfg.Ops.Lookup("wasm_incr")
	if !ok {
		t.Fatal("the snapshot-installed replica does not have the op registered: it would fail closed (classFatal ErrOpNotRegistered) on the first invocation its peers execute")
	}
	// ...with its ROUTING intact. A registration that arrived without its key
	// extractor would route the op to shard 0 while peers route it by key.
	if ke == nil {
		t.Error("the restored op lost its key extractor: it is no longer routable, so it would be dispatched to a different group than on its peers")
	}

	// 2. The sidecar reached the target's disk, so a restart reloads the
	// registration. It is the ONLY artifact the restore writes now — the marker
	// carries no bytes — and what it must carry is the CONTENT ADDRESS the source's
	// marker named: that is the address the reload resolves and the address a fetch
	// asks a peer for, so a wrong one is a module that never arrives.
	if _, err := os.Stat(filepath.Join(target.cfg.DataDir, "wasm", "wasm_incr.json")); err != nil {
		t.Errorf("the restored registration has no metadata sidecar: %v", err)
	}
	meta, have, err := readWASMSidecar(target.cfg.DataDir, "wasm_incr")
	if err != nil || !have {
		t.Fatalf("read the restored sidecar: have=%v err=%v", have, err)
	}
	if want := ops.WASMBlobHex(ops.WASMBlobFingerprint(wasmBytes)); meta.Blob != want {
		t.Errorf("the restored sidecar names blob %s, want %s: the reload resolves a module this node will never hold", meta.Blob, want)
	}

	// 3. And it actually runs — the registry entry is wired to a live module in
	// the target's own WASM runtime, not a dangling handler.
	key := keyForShard(t, 0, numShards)
	if _, err := target.Call("wasm_incr", key); err != nil {
		t.Fatalf("invoke the restored op on the target node: %v", err)
	}
}

// TestOversizedWASMModuleRejected pins the propose-side bound.
//
// WHAT THE BOUND PROTECTS MOVED WITH THIN MARKERS, and the cap survived the move
// unchanged. It used to protect the Raft logs: the broadcast wrote the module into
// EVERY shard group's log, so an accepted module cost NumShards x its size on
// every node — and again in each group's snapshot once it compacted. A marker
// carries no module, so the logs are bounded by the marker now whatever the module
// weighs. What maxDynamicWASMBytes bounds today is the CLIENT-EDGE REQUEST and the
// blob transport it feeds (maxWASMBlobPutFrame), i.e. the bytes this node stores
// and pushes to every member. The wire alone would still allow up to the 16 MiB
// frame cap, and the refusal must still happen before anything is proposed.
func TestOversizedWASMModuleRejected(t *testing.T) {
	n := newTestNode(t, 1)
	module := make([]byte, maxDynamicWASMBytes+1)
	payload := ops.EncodeWASMRegistrationRequest(ops.WASMRegistration{
		Name:       "wasm_too_big",
		Kind:       ops.OpReadWrite,
		Blob:       ops.WASMBlobFingerprint(module),
		ExportName: "apply",
	}, module)
	if _, err := n.Call(wasmRegisterOpName, payload); err == nil {
		t.Fatal("an oversized module must be rejected before anything is proposed")
	}
	// Nothing may have been registered by the rejected attempt.
	if _, _, _, ok := n.cfg.Ops.Lookup("wasm_too_big"); ok {
		t.Error("the rejected registration still reached the ops registry")
	}
}

// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"os"
	"strconv"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/wasm"
)

// readIncrWASM loads the shared no-op test module. It ignores its args and
// returns an empty result, so it is safe to invoke with any routing key.
func readIncrWASM(t *testing.T) []byte {
	t.Helper()
	b, err := os.ReadFile("../wasm/testdata/incr.wasm")
	if err != nil {
		t.Skipf("incr.wasm not readable (%v); skipping", err)
	}
	return b
}

// lastLogIndexes snapshots every hosted shard group's raft last_log_index.
func lastLogIndexes(t *testing.T, n *Node) []uint64 {
	t.Helper()
	out := make([]uint64, len(n.shards))
	for i, s := range n.shards {
		if s == nil {
			continue
		}
		raw := s.Stats().Raft["last_log_index"]
		v, err := strconv.ParseUint(raw, 10, 64)
		if err != nil {
			t.Fatalf("shard %d: last_log_index %q unparseable: %v", i, raw, err)
		}
		out[i] = v
	}
	return out
}

// keyForShard returns a key that routes to shardIdx under numShards.
func keyForShard(t *testing.T, shardIdx, numShards int) []byte {
	t.Helper()
	for i := 0; i < 100000; i++ {
		k := []byte(fmt.Sprintf("route-%d", i))
		if shardOf(k, numShards) == shardIdx {
			return stdArgs(k)
		}
	}
	t.Fatalf("no key found routing to shard %d of %d", shardIdx, numShards)
	return nil
}

// stdArgs frames key as the invoke args of an op using the ONLY key extractor a
// WASM op may declare: [keyLen u16][key], with an empty payload.
//
// It exists because the args and the ROUTING KEY are no longer the same bytes.
// Under the retired "raw" handle they were, and every test here could pass a bare
// key straight to Call. Every WASM op now uses the one extractor
// (ops.WASMKeyExtractorHandle — see it for why there is no longer a choice), so a
// caller hands over a frame the extractor reads the key OUT of, and a test that
// passed a bare key would either be refused (fewer than 2 bytes) or route on a
// length prefix that is not the key at all.
func stdArgs(key []byte) []byte {
	out := make([]byte, 2+len(key))
	binary.BigEndian.PutUint16(out[:2], uint16(len(key))) //nolint:gosec // test keys are tiny
	copy(out[2:], key)
	return out
}

// TestRegisterWASMReachesEveryShardGroup is the regression gate for the silent
// cross-replica divergence described on broadcastWASMRegistration: a routable
// WASM op is INVOKED from shardOf(key)'s Raft log, so a group that never
// receives the registration can see an invocation for an op it does not have.
//
// What it asserts is REACH, not ordering. An earlier version of this comment
// claimed the broadcast orders the registration ahead of that group's
// invocations; it does not. The loop is sequential while the ops registry is
// node-wide, so nothing in the broadcast itself stops a node whose group 0 has
// applied from routing an invocation to group j before the loop reaches j.
//
// Ordering is established by the ROUTE GATE (checkWASMRouteGate), which is
// asserted in wasm_gate_test.go. Reach is what this test covers, and reach is
// what makes the gate OPENABLE: a group the broadcast never reaches is a group
// the gate keeps permanently shut, so every invocation routed there fails.
//
// The assertion is on the logs themselves, not on callability: the ops registry
// is node-wide, so the op would be callable on this node even if only one group
// carried the entry. Every group's last_log_index must advance.
func TestRegisterWASMReachesEveryShardGroup(t *testing.T) {
	const numShards = 4
	wasmBytes := readIncrWASM(t)
	n := newTestNode(t, numShards)

	// Settle every group's election FIRST: the leader's initial no-op entry also
	// grows last_log_index, and counting it would let the assertion below pass
	// without the registration ever arriving. After this point nothing appends to
	// a group's log except the ops this test issues.
	waitAllLeaders(t, n)
	waitAllApplied(t, n)

	before := lastLogIndexes(t, n)

	// The CLIENT EDGE carries the marker AND the module: __register_wasm__ takes
	// ops.EncodeWASMRegistrationRequest so the coordinator has bytes to push to the
	// cluster before it proposes anything. What lands in each group's log is still
	// the bare marker (ops.EncodeWASMRegistration), which is what this test counts.
	payload := ops.EncodeWASMRegistrationRequest(ops.WASMRegistration{
		Name: "wasm_incr",
		Kind: ops.OpReadWrite,
		Blob: ops.WASMBlobFingerprint(wasmBytes),
		// A WASM op is always ROUTABLE (RegisterRoutableCrossShard): its
		// invocations land in shardOf(args)'s group, not shard 0's. This is the
		// wire-controlled field that made the defect reachable by any client.
		ExportName: "apply",
	}, wasmBytes)
	if _, err := n.Call("__register_wasm__", payload); err != nil {
		t.Fatalf("Call __register_wasm__: %v", err)
	}

	after := lastLogIndexes(t, n)
	for i := range after {
		if after[i] <= before[i] {
			t.Errorf("shard %d log did not grow (last_log_index %d -> %d): the registration never reached this group's Raft log",
				i, before[i], after[i])
		}
	}

	// And the op is genuinely invocable through a key that routes AWAY from
	// shard 0 — the path that used to hit an unregistered op on a lagging peer.
	for i := 0; i < numShards; i++ {
		key := keyForShard(t, i, numShards)
		if _, err := n.Call("wasm_incr", key); err != nil {
			t.Errorf("Call wasm_incr with a key routing to shard %d: %v", i, err)
		}
	}
}

// TestApplyWASMRegistrationIsIdempotent pins the contract the broadcast relies
// on: the FSM hook now runs once per hosted shard group (plus again on any
// client retry), so repeating it must neither error nor accumulate runtime
// state. (The no-re-instantiation half of that contract is asserted directly on
// the runtime in wasm.TestAddModuleIdenticalIsNoOp.)
func TestApplyWASMRegistrationIsIdempotent(t *testing.T) {
	wasmBytes := readIncrWASM(t)
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
	r := ops.WASMRegistration{
		Name:       "wasm_incr",
		Kind:       ops.OpReadWrite,
		Blob:       ops.WASMBlobFingerprint(wasmBytes),
		ExportName: "apply",
	}

	// The marker names the module and does not carry it, so applyWASMRegistration
	// no longer writes the blob — it READS it. Seed it once, which is what the
	// pre-registration push does on a real node, or every apply below would install
	// the marker with the module NOT resident and the residency assertion would be
	// asserting the wrong thing.
	if _, err := writeWASMBlob(dir, wasmBytes); err != nil {
		t.Fatalf("writeWASMBlob: %v", err)
	}

	for i := 0; i < 5; i++ {
		if err := applyWASMRegistration(dir, rt, reg, st, r, 0, nil); err != nil {
			t.Fatalf("applyWASMRegistration call %d: %v", i+1, err)
		}
	}

	if !rt.HasModule(wasm.ModuleIDForBlob(r.Blob, r.ExportName, r.MaxFuel)) {
		t.Fatal("module missing from the runtime after repeated registration")
	}
	fn, _, ke, ok := reg.Lookup(r.Name)
	if !ok {
		t.Fatal("op missing from the registry after repeated registration")
	}
	if ke == nil {
		t.Error("op lost its key extractor across repeats (should stay routable)")
	}
	if fn == nil {
		t.Error("op has a nil handler after repeats")
	}

	// The sidecar is rewritten each time; it must still be intact, and the blob
	// seeded above must still be there and unduplicated. The module bytes live at
	// their content address and the apply path only READS them, so five repeats
	// leave exactly the one blob.
	blobFP := r.Blob
	for _, name := range []string{
		"blobs/" + hex.EncodeToString(blobFP[:]) + ".wasm",
		"wasm_incr.json",
	} {
		st, err := os.Stat(dir + "/wasm/" + name)
		if err != nil {
			t.Errorf("stat %s: %v", name, err)
			continue
		}
		if st.Size() == 0 {
			t.Errorf("%s is empty after repeated writes", name)
		}
	}
	blobs, err := os.ReadDir(dir + "/wasm/blobs")
	if err != nil {
		t.Fatalf("readdir blobs: %v", err)
	}
	if len(blobs) != 1 {
		t.Errorf("5 identical registrations left %d blobs, want 1 (content addressing must dedup, and the apply path must not add copies)", len(blobs))
	}
}

// TestShardScopedWASMWrapperRoundTrip covers the wire form of the internal
// wrapper op that carries one group's leg of the broadcast to a peer.
func TestShardScopedWASMWrapperRoundTrip(t *testing.T) {
	reg := []byte("payload-bytes")
	idx, got, err := decodeShardScopedWASM(encodeShardScopedWASM(7, reg))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if idx != 7 {
		t.Errorf("shard idx = %d, want 7", idx)
	}
	if string(got) != string(reg) {
		t.Errorf("payload = %q, want %q", got, reg)
	}
	if _, _, err := decodeShardScopedWASM([]byte{1, 2}); err == nil {
		t.Error("decode of a truncated payload should fail")
	}
}

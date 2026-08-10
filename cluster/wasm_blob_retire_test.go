// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"crypto/sha256"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
	"github.com/rostamlabs/rostam/wasm"
)

// retireWindow is the retention every deterministic test below runs with. The
// value is irrelevant to the rule — the tests drive the clock rather than the
// wall — so it is chosen only to be obviously non-zero.
const retireWindow = time.Hour

// readPutWASM loads the SECOND shared test module, so a test can express "this
// version was superseded by a different one" with two blobs that both really
// compile.
func readPutWASM(t *testing.T) []byte {
	t.Helper()
	b, err := os.ReadFile("../wasm/testdata/put.wasm")
	if err != nil {
		t.Skipf("put.wasm not readable (%v); skipping", err)
	}
	return b
}

// newRetireTestNode builds a node whose retention is set AFTER construction, so
// NO SWEEPER GOROUTINE EXISTS and every sweep in the test is one the test issued.
//
// That is not a convenience, it is what makes these tests assertions about the
// RULE rather than about a timer: the sweeper's cadence, the machine's load and
// the test's own scheduling cannot move the result, and nothing races the test
// for the bookkeeping map. The one test that must prove the goroutine is actually
// wired (TestWASMBlobRetirementSweeperIsWiredUp) builds its node the other way,
// through New, and waits.
func newRetireTestNode(t *testing.T) *Node {
	t.Helper()
	n := newTestNode(t, 1)
	n.cfg.WASMBlobRetention = retireWindow
	return n
}

// stageBlob writes b into the node's content-addressed blob store and returns
// its hex fingerprint, without registering anything. That is exactly the state a
// __wasm_blob_put__ leaves behind, and exactly what a blob whose registration has
// moved on looks like.
func stageBlob(t *testing.T, n *Node, b []byte) string {
	t.Helper()
	fp, err := writeWASMBlob(n.cfg.DataDir, b)
	if err != nil {
		t.Fatalf("writeWASMBlob: %v", err)
	}
	return fp
}

// bindWASM applies one registration of name at bytes b into group's binding,
// through the production apply path.
func bindWASM(t *testing.T, n *Node, name string, b []byte, epoch uint64, group int) {
	t.Helper()
	n.wasmApplyMu.Lock()
	err := applyWASMRegistration(n.cfg.DataDir, n.wasmRT, n.cfg.Ops, n.wasmState,
		ops.WASMRegistration{
			Name: name, Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(b),
			ExportName: "apply", Epoch: epoch,
		}, group, nil)
	n.wasmApplyMu.Unlock()
	if err != nil {
		t.Fatalf("applyWASMRegistration(%q epoch %d group %d): %v", name, epoch, group, err)
	}
}

// blobPresent reports whether the blob file for fp exists. It LSTATS, so a
// symlink planted under a blob name counts as present — which is the whole point
// in the traversal test.
func blobPresent(t *testing.T, n *Node, fp string) bool {
	t.Helper()
	p, err := wasmBlobPath(n.cfg.DataDir, fp)
	if err != nil {
		t.Fatalf("wasmBlobPath(%q): %v", fp, err)
	}
	_, err = os.Lstat(p)
	if err == nil {
		return true
	}
	if errors.Is(err, os.ErrNotExist) {
		return false
	}
	t.Fatalf("lstat %s: %v", p, err)
	return false
}

// sweepAt runs one pass and fails the test on an I/O error.
func sweepAt(t *testing.T, n *Node, at time.Time) []string {
	t.Helper()
	retired, err := n.sweepWASMBlobsAt(at)
	if err != nil {
		t.Fatalf("sweepWASMBlobsAt(%v): %v", at, err)
	}
	return retired
}

// TestWASMBlobRetirementIsOffByDefault is the gate on the DEFAULT, and it is the
// most important test in this file.
//
// Retirement is the only mechanism in the WASM subsystem that deletes anything,
// and it cannot be made safe against a lagging replica by any purely local rule
// (see wasm_blob_retire.go's header). Its whole claim to being shippable is that
// a deployment which has not asked for it is byte-identical to the one before it
// existed: nothing swept, nothing tracked, nothing removed, EVEN IF a sweep is
// somehow issued. So the assertion is deliberately stronger than "no goroutine
// runs" — it drives the sweep directly, at a clock far past any window, and
// requires it to be a total no-op.
func TestWASMBlobRetirementIsOffByDefault(t *testing.T) {
	n := newTestNode(t, 1) // NOT newRetireTestNode: cfg.WASMBlobRetention stays zero
	fp := stageBlob(t, n, []byte("an orphan nothing references"))

	// Two sweeps, a decade apart. Under any non-zero retention this blob would be
	// gone by the second one.
	for _, at := range []time.Time{time.Now(), time.Now().Add(10 * 365 * 24 * time.Hour)} {
		if retired := sweepAt(t, n, at); len(retired) != 0 {
			t.Fatalf("retirement is off but the sweep removed %v", retired)
		}
	}
	if !blobPresent(t, n, fp) {
		t.Fatal("retirement is off and a blob was deleted anyway")
	}

	st := n.Stats().WASMBlobRetire
	if st.Retention != 0 {
		t.Errorf("Stats reports Retention=%v for an unconfigured node; the default must be 0 (off)", st.Retention)
	}
	if st.Sweeps != 0 || st.Retired != 0 || st.Pending != 0 {
		t.Errorf("an off node did bookkeeping: Sweeps=%d Retired=%d Pending=%d, want 0/0/0", st.Sweeps, st.Retired, st.Pending)
	}
}

// TestABoundBlobIsNeverRetired is the floor under the rule: a version a hosted
// group's binding still names is not a candidate at ANY age.
//
// This is the case that would be catastrophic and silent. The blob is what the
// group EXECUTES; deleting it fails nothing immediately (the module is already
// instantiated) and shows up only at the next restart, when
// reloadWASMModulesFromDisk finds the sidecar naming a blob that is gone and the
// group parks — on a version the cluster considers CURRENT, which nobody is
// pushing any more.
func TestABoundBlobIsNeverRetired(t *testing.T) {
	n := newRetireTestNode(t)
	incr := readIncrWASM(t)
	fp := stageBlob(t, n, incr)
	bindWASM(t, n, "wasm_incr", incr, 1, 0)

	if retired := sweepAt(t, n, time.Now()); len(retired) != 0 {
		t.Fatalf("first sweep removed %v", retired)
	}
	if retired := sweepAt(t, n, time.Now().Add(1000*retireWindow)); len(retired) != 0 {
		t.Fatalf("a blob shard group 0's binding still names was retired: %v", retired)
	}
	if !blobPresent(t, n, fp) {
		t.Fatal("the currently bound module's blob is gone")
	}
	// And it was never even a candidate: nothing is waiting out a window.
	if p := n.Stats().WASMBlobRetire.Pending; p != 0 {
		t.Errorf("Pending=%d; a bound blob must not be tracked as unreferenced at all", p)
	}
}

// TestASupersededBlobIsKeptForItsWholeRetentionWindow pins the rule's timing on
// the case it exists for, and it asserts a DELAY rather than a deletion.
//
// The delay is the entire safety story. A superseded version is still needed by
// any replica of any group that has not yet applied past the supersession point,
// and this node cannot see those replicas — so the window is the operator's
// stated bound on how far behind one may be. Shortening it in code, or starting
// the clock at the wrong moment (from the blob's mtime, say, rather than from the
// first sweep that observed it unreferenced), silently shortens their assertion.
func TestASupersededBlobIsKeptForItsWholeRetentionWindow(t *testing.T) {
	n := newRetireTestNode(t)
	v1, v2 := readIncrWASM(t), readPutWASM(t)
	fp1, fp2 := stageBlob(t, n, v1), stageBlob(t, n, v2)
	if fp1 == fp2 {
		t.Fatal("the two test modules must differ")
	}
	bindWASM(t, n, "wasm_incr", v1, 1, 0)
	bindWASM(t, n, "wasm_incr", v2, 2, 0) // supersedes: group 0 and the node now name v2

	t0 := time.Now()
	// The clock starts here, not when the blob was written and not when the
	// superseding registration applied.
	if retired := sweepAt(t, n, t0); len(retired) != 0 {
		t.Fatalf("the sweep that first OBSERVED the blob unreferenced retired it: %v", retired)
	}
	if p := n.Stats().WASMBlobRetire.Pending; p != 1 {
		t.Errorf("Pending=%d after the first sweep, want 1 (the superseded version)", p)
	}
	// One nanosecond short of the window is still inside it.
	if retired := sweepAt(t, n, t0.Add(retireWindow-time.Nanosecond)); len(retired) != 0 {
		t.Fatalf("a superseded blob was retired BEFORE its retention window elapsed: %v", retired)
	}
	if !blobPresent(t, n, fp1) {
		t.Fatal("the superseded blob is gone inside its window")
	}

	retired := sweepAt(t, n, t0.Add(retireWindow))
	if len(retired) != 1 || retired[0] != fp1 {
		t.Fatalf("sweep past the window retired %v, want exactly [%s]", retired, fp1)
	}
	if blobPresent(t, n, fp1) {
		t.Error("the superseded blob survived a sweep that reported retiring it")
	}
	if !blobPresent(t, n, fp2) {
		t.Fatal("the CURRENT bound version was retired")
	}
	if st := n.Stats().WASMBlobRetire; st.Retired != 1 || st.Pending != 0 {
		t.Errorf("Stats after retirement: Retired=%d Pending=%d, want 1/0", st.Retired, st.Pending)
	}
}

// TestAPutOnlyOrphanIsRetiredOnlyAfterTheWindow covers the class that is easy to
// forget: a blob created by __wasm_blob_put__ with NO registration behind it.
//
// It arises three ways — an operator pre-staging bytes, a push whose registration
// then failed, and the ordinary window between the pre-registration push landing
// on a member and that member applying the marker — and the third is why the
// window is not optional for orphans either. A zero-delay orphan sweep would
// delete the push's work out from under an in-flight registration, on every
// member at once, dissolving the durability floor the push had just established.
//
// The same ONE condition handles it as handles a superseded version: an orphan is
// by construction referenced by nothing.
func TestAPutOnlyOrphanIsRetiredOnlyAfterTheWindow(t *testing.T) {
	n := newRetireTestNode(t)
	incr := readIncrWASM(t)
	fp := blobFP(incr)

	// The real op, not a hand-written file: this is the state the escape hatch and
	// the push both leave behind.
	if _, err := n.handleWASMBlobPut(encodeWASMBlobPut(fp, incr)); err != nil {
		t.Fatalf("blob put: %v", err)
	}
	if !blobPresent(t, n, fp) {
		t.Fatal("the put did not store the blob")
	}

	t0 := time.Now()
	if retired := sweepAt(t, n, t0); len(retired) != 0 {
		t.Fatalf("a freshly put blob was retired immediately: %v", retired)
	}
	if retired := sweepAt(t, n, t0.Add(retireWindow-time.Nanosecond)); len(retired) != 0 {
		t.Fatalf("a put-only orphan was retired inside its window: %v", retired)
	}
	if !blobPresent(t, n, fp) {
		t.Fatal("the orphan is gone inside its window; a push's bytes would not survive to the marker")
	}

	if retired := sweepAt(t, n, t0.Add(retireWindow)); len(retired) != 1 || retired[0] != fp {
		t.Fatalf("sweep past the window retired %v, want exactly [%s]", retired, fp)
	}
	if blobPresent(t, n, fp) {
		t.Error("the put-only orphan survived past its window")
	}
}

// TestRetirementNeverTakesABlobThisNodeIsBlockedOnOrFetching pins the two
// protections that have nothing to do with age.
//
// Both describe a node that is ALREADY in the scarce-blob state — a group parked
// on a fingerprint, or a fetch loop scouring peers for one — and taking a copy
// away at that moment is the worst available timing. Neither is implied by the
// referenced set: a fetch that has landed the file races the parked group's next
// apply retry, and in that window the blob is on disk and no binding has moved.
func TestRetirementNeverTakesABlobThisNodeIsBlockedOnOrFetching(t *testing.T) {
	n := newRetireTestNode(t)
	fp := stageBlob(t, n, []byte("bytes something is waiting for"))

	// (1) a live block naming it.
	n.recordWASMBlock(blockKey{group: 0, op: "wasm_incr"}, fp, 3, 2*time.Second, "not resident yet", nil)
	if retired := sweepAt(t, n, time.Now()); len(retired) != 0 {
		t.Fatalf("first sweep removed %v", retired)
	}
	if retired := sweepAt(t, n, time.Now().Add(1000*retireWindow)); len(retired) != 0 {
		t.Fatalf("retired a blob a shard group is BLOCKED on: %v", retired)
	}
	if !blobPresent(t, n, fp) {
		t.Fatal("the blob a group is parked on was deleted")
	}
	n.wasmBlockMu.Lock()
	delete(n.wasmBlockLive, blockKey{group: 0, op: "wasm_incr"})
	n.publishWASMBlocksLocked()
	n.wasmBlockMu.Unlock()

	// (2) an in-flight fetch for it.
	var sum [sha256.Size]byte
	copy(sum[:], mustDecodeWASMBlobFP(fp))
	n.wasmFetchMu.Lock()
	if n.wasmFetching == nil {
		n.wasmFetching = make(map[[sha256.Size]byte]*wasmBlobFetch, 1)
	}
	n.wasmFetching[sum] = &wasmBlobFetch{started: time.Now()}
	n.wasmFetchMu.Unlock()

	if retired := sweepAt(t, n, time.Now()); len(retired) != 0 {
		t.Fatalf("sweep removed %v", retired)
	}
	if retired := sweepAt(t, n, time.Now().Add(1000*retireWindow)); len(retired) != 0 {
		t.Fatalf("retired a blob with a fetch in flight for it: %v", retired)
	}

	// With both protections gone it is an ordinary orphan again, which proves the
	// two clauses above were what held it and not some unrelated skip.
	n.wasmFetchMu.Lock()
	delete(n.wasmFetching, sum)
	n.wasmFetchMu.Unlock()
	t0 := time.Now()
	sweepAt(t, n, t0)
	if retired := sweepAt(t, n, t0.Add(retireWindow)); len(retired) != 1 || retired[0] != fp {
		t.Fatalf("with nothing protecting it the sweep retired %v, want [%s]", retired, fp)
	}
}

// TestARetiredVersionStaysExecutableAndCanBeSuppliedAgain is the gate on the
// thing this whole mechanism risks: that retirement resurrects the PERMANENT
// block. It asserts the two halves that together make a retirement recoverable
// rather than terminal.
//
// FIRST, retirement is invisible to EXECUTION. Only the file is removed; the
// instantiated module stays in the runtime, and wasm.Runtime.resolveModuleForInvoke
// asks the runtime rather than the filesystem. So a group whose binding still
// names a retired version — a group on this node that has not yet applied the
// supersession, or an invocation already running — keeps working. That is what
// makes the "grace period for invocations currently executing under it" in the
// recorded design unnecessary rather than merely satisfied.
//
// SECOND, the version is re-suppliable. This node can no longer SERVE the bytes
// (which is the real cost, stated plainly: a peer's __wasm_blob_get__ now misses
// here), but the blob is content addressed, so any copy of the right bytes is THE
// right bytes — and one __wasm_blob_put__ restores both the file and this node's
// ability to serve it, with no restart. A lagging replica that blocks on a
// retired version is therefore one admin call from unblocked, which is the only
// reason the trade is offerable at all.
func TestARetiredVersionStaysExecutableAndCanBeSuppliedAgain(t *testing.T) {
	n := newRetireTestNode(t)
	v1, v2 := readIncrWASM(t), readPutWASM(t)
	fp1 := stageBlob(t, n, v1)
	stageBlob(t, n, v2)
	bindWASM(t, n, "wasm_incr", v1, 1, 0)
	id1 := wasm.ModuleIDForBlob(ops.WASMBlobFingerprint(v1), "apply", 0)
	if !n.wasmRT.HasModule(id1) {
		t.Fatal("precondition: v1 was never instantiated")
	}
	bindWASM(t, n, "wasm_incr", v2, 2, 0) // supersedes

	t0 := time.Now()
	sweepAt(t, n, t0)
	if retired := sweepAt(t, n, t0.Add(retireWindow)); len(retired) != 1 || retired[0] != fp1 {
		t.Fatalf("precondition: the superseded version was not retired (%v)", retired)
	}

	// (1) still executable.
	if !n.wasmRT.HasModule(id1) {
		t.Error("retirement evicted the runtime module; an invocation still resolving to this version would newly block")
	}
	// (2) but no longer servable, which is the cost.
	if _, err := n.handleWASMBlobGet([]byte(fp1)); err == nil {
		t.Error("a retired blob is still served by __wasm_blob_get__; the file was not actually removed")
	}

	// The escape hatch, on exactly the fingerprint that was retired.
	if _, err := n.handleWASMBlobPut(encodeWASMBlobPut(fp1, v1)); err != nil {
		t.Fatalf("a retired version could not be supplied again: %v", err)
	}
	got, err := n.handleWASMBlobGet([]byte(fp1))
	if err != nil {
		t.Fatalf("the re-supplied version is not servable: %v", err)
	}
	if string(got) != string(v1) {
		t.Fatalf("the re-supplied blob serves %d bytes, want the original %d", len(got), len(v1))
	}
}

// TestRetirementCannotBeSteeredOutsideTheBlobDirectory is the security gate, and
// it covers the highest-risk thing in this stage: a fingerprint→path join whose
// consumer is os.Remove.
//
// Three properties, each a different way the join could be abused:
//
//   - a directory entry that is not exactly `<64 lower-case hex>.wasm` is never a
//     deletion candidate at all, so nothing whose name this build cannot derive
//     can be removed;
//   - the path removed is RE-DERIVED from the validated fingerprint and must equal
//     the entry that was listed, so the two can never drift apart;
//   - os.Remove does not follow symlinks, so a link planted under a perfectly
//     valid fingerprint name removes the LINK and never its target — the only
//     remaining way to point the delete outside the directory.
func TestRetirementCannotBeSteeredOutsideTheBlobDirectory(t *testing.T) {
	// The name-level gate first, since everything else rests on it.
	for _, name := range []string{
		"../../../../etc/passwd.wasm",
		"..%2f..%2fescape.wasm",
		"../evil.wasm",
		strings.ToUpper(strings.Repeat("ab", 32)) + ".wasm", // upper-case hex
		strings.Repeat("ab", 31) + ".wasm",                  // too short
		strings.Repeat("ab", 33) + ".wasm",                  // too long
		strings.Repeat("ab", 32),                            // no .wasm suffix
		".wasm",
		"..wasm",
		"notablob.txt",
		"." + strings.Repeat("ab", 32) + ".wasm.tmp-123", // an atomicWriteFile temp
	} {
		if _, _, ok := retirableWASMBlob("/data", filepath.Join("/data", "wasm", wasmBlobsSubdir), name); ok {
			t.Errorf("retirableWASMBlob accepted %q as a deletion candidate", name)
		}
	}
	// ...and the one shape it must accept, so the negatives above are not passing
	// merely because the function rejects everything.
	valid := strings.Repeat("ab", 32) + ".wasm"
	blobs := filepath.Join("/data", "wasm", wasmBlobsSubdir)
	fp, path, ok := retirableWASMBlob("/data", blobs, valid)
	if !ok {
		t.Fatalf("retirableWASMBlob rejected the canonical name %q", valid)
	}
	if fp != strings.Repeat("ab", 32) || path != filepath.Join(blobs, valid) {
		t.Fatalf("retirableWASMBlob(%q) = %q, %q", valid, fp, path)
	}

	// Now the live directory, with a symlink under a VALID fingerprint name aimed
	// at a file the node must never touch.
	n := newRetireTestNode(t)
	blobsDir := filepath.Join(n.cfg.DataDir, "wasm", wasmBlobsSubdir)
	if err := os.MkdirAll(blobsDir, 0o750); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(t.TempDir(), "precious")
	if err := os.WriteFile(outside, []byte("must survive"), 0o600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(blobsDir, strings.Repeat("cd", 32)+".wasm")
	if err := os.Symlink(outside, link); err != nil {
		t.Skipf("symlinks unavailable here (%v)", err)
	}
	junk := filepath.Join(blobsDir, "not-a-fingerprint.txt")
	if err := os.WriteFile(junk, []byte("left by something else"), 0o600); err != nil {
		t.Fatal(err)
	}

	t0 := time.Now()
	sweepAt(t, n, t0)
	sweepAt(t, n, t0.Add(retireWindow))

	if _, err := os.Stat(outside); err != nil {
		t.Fatalf("the sweep followed a symlink out of the blob directory: %s is gone (%v)", outside, err)
	}
	if _, err := os.Lstat(link); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("the symlink itself should have been retired like any unreferenced blob name; Lstat=%v", err)
	}
	if _, err := os.Stat(junk); err != nil {
		t.Errorf("the sweep removed a file whose name it cannot derive: %v", err)
	}
}

// TestANodeHostingNoShardGroupRetiresNothing pins the guard that keeps the
// durability floor's sharpest case out of the sweeper's reach.
//
// Markers apply per shard group, so a node hosting NO group applies no marker,
// writes no sidecar, and has an empty live set — every blob it holds looks like a
// put-only orphan. They are not garbage: they are the pre-registration push's
// residue, and that node is one of the members whose ack let the marker be
// proposed at all. Retiring them would delete the floor's holdings on precisely
// the nodes that hold nothing else.
func TestANodeHostingNoShardGroupRetiresNothing(t *testing.T) {
	n := newRetireTestNode(t)
	fp := stageBlob(t, n, []byte("push residue on a group-less member"))

	// Exactly the state of a member that owns no group: the shard slots exist and
	// every one of them is empty.
	n.shardMu.Lock()
	hosted := append([]*shard.Store(nil), n.shards...)
	for i := range n.shards {
		n.shards[i] = nil
	}
	n.shardMu.Unlock()
	defer func() {
		n.shardMu.Lock()
		copy(n.shards, hosted)
		n.shardMu.Unlock()
	}()

	for _, at := range []time.Time{time.Now(), time.Now().Add(1000 * retireWindow)} {
		if retired := sweepAt(t, n, at); len(retired) != 0 {
			t.Fatalf("a node hosting no shard group retired %v; those blobs are the push's floor residue", retired)
		}
	}
	if !blobPresent(t, n, fp) {
		t.Fatal("the group-less member's blob was deleted")
	}
}

// TestWASMBlobRetirementSweeperIsWiredUp proves the background loop exists — the
// one thing the clock-driven tests above cannot show, since they never let a
// sweeper run.
//
// A knob that is threaded end to end but whose goroutine is never started is a
// failure this repo has shipped before (see TestClusterConfigFromThreadsPBKnobs
// for the PBAutoFailover case), so the wiring gets its own assertion instead of
// being assumed.
func TestWASMBlobRetirementSweeperIsWiredUp(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	n, err := New(Config{
		NodeID: "node1", DataDir: t.TempDir(),
		NumShards: 1,
		Bootstrap: true,
		ShardCfg: shard.Config{
			NodeID: "ignored", DataDir: "ignored",
			Cache: cc, Ops: reg,
			Bootstrap:       true,
			RaftHeartbeatMs: 50, RaftElectionMs: 100, NoSync: true,
		},
		Ops: reg,
		// Short enough for a test to wait out, and the sweep cadence follows it
		// (wasmBlobRetireInterval is a quarter of the window).
		WASMBlobRetention: 300 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = n.Close() })
	waitAllLeaders(t, n)

	fp := stageBlob(t, n, []byte("an orphan the background sweeper should reclaim"))

	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		if !blobPresent(t, n, fp) {
			if got := n.Stats().WASMBlobRetire; got.Retired == 0 || got.Sweeps == 0 {
				t.Fatalf("the blob is gone but Stats did not record it: %+v", got)
			}
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("the background sweeper never retired the orphan; Stats=%+v", n.Stats().WASMBlobRetire)
}

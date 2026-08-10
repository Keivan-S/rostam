// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/ops"
)

// blobFP is the hex fingerprint of b — the name the content-addressed store
// gives it, and the value the wire format carries.
func blobFP(b []byte) string {
	sum := ops.WASMBlobFingerprint(b)
	return hex.EncodeToString(sum[:])
}

// blobsDirEntries lists the basenames under <dataDir>/wasm/blobs. A missing
// directory is the empty set, which is what "nothing was written" looks like
// before the first blob ever lands.
func blobsDirEntries(t *testing.T, dataDir string) []string {
	t.Helper()
	entries, err := os.ReadDir(filepath.Join(dataDir, "wasm", wasmBlobsSubdir))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		t.Fatalf("readdir blobs: %v", err)
	}
	out := make([]string, 0, len(entries))
	for _, e := range entries {
		out = append(out, e.Name())
	}
	return out
}

// TestWASMBlobPutRefusesBytesThatDoNotHashToTheClaimedFingerprint is the gate on
// the FIRST half of the acceptance rule.
//
// The put's fingerprint and its bytes are two independent wire fields, and the
// store is addressed by ONE of them. Without this check the op degenerates into
// "write these bytes somewhere and call them X": the sender's claim and the
// content stop being tied together, and the self-verification the whole
// content-addressed layout rests on (readWASMBlob re-derives a blob's identity
// from its contents) would be checking something nobody ever asserted. Under
// The marker names a fingerprint and nothing else, so a fingerprint
// that does not mean its bytes is the only thing left to get wrong.
//
// Note WHY this cannot be waved away as "writeWASMBlob addresses by the computed
// hash anyway, so the mismatch is harmless": it is harmless to the FILESYSTEM and
// silent to the CALLER. The sender is told its blob was accepted under the
// fingerprint it named, and the node holds it under a different one — so the
// later fetch for the named fingerprint misses on a node that acked it.
func TestWASMBlobPutRefusesBytesThatDoNotHashToTheClaimedFingerprint(t *testing.T) {
	n := newTestNode(t, 1)
	incr := readIncrWASM(t)

	// A fingerprint of DIFFERENT bytes: well-formed hex, correct length, simply
	// not this module's.
	wrong := blobFP(append(append([]byte(nil), incr...), 0x00))

	_, err := n.handleWASMBlobPut(encodeWASMBlobPut(wrong, incr))
	if err == nil {
		t.Fatal("put accepted bytes that do not hash to the claimed fingerprint")
	}
	if !errors.Is(err, ErrWASMBlobRefused) {
		t.Fatalf("put refusal must be ErrWASMBlobRefused, got %T: %v", err, err)
	}
	for _, want := range []string{wrong, blobFP(incr)} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal must name both fingerprints; %q missing from %q", want, err.Error())
		}
	}
	// Nothing may be written: neither under the claimed name nor under the real
	// one. Writing under the real one would be the "harmless" outcome above, and
	// it is exactly the state that makes the ack a lie.
	if got := blobsDirEntries(t, n.cfg.DataDir); len(got) != 0 {
		t.Fatalf("a refused put wrote %v", got)
	}
}

// TestWASMBlobPutRefusesAModuleThatDoesNotCompile is the gate on the SECOND half
// of the acceptance rule, and it is the one with the real payoff.
//
// The compile in the ack is what converts "the fleet's wasmtime versions
// disagree about these bytes" from an APPLY-TIME discovery into a
// REGISTRATION-TIME refusal. Without it the put is a file copy that always
// succeeds, and a node that cannot compile the module says so for the first time
// when it applies the registration — at which point the entry is committed in
// every group's log, and with the thin marker that node cannot even re-derive the bytes
// from the entry to try again. The verdict has to be rendered here, before the
// registration is allowed to proceed.
//
// It also asserts the ORDER: a module that does not compile leaves NO file
// behind, the same guarantee loadOneModule gives config modules.
func TestWASMBlobPutRefusesAModuleThatDoesNotCompile(t *testing.T) {
	n := newTestNode(t, 1)
	// Well-formed frame, honest fingerprint, and not a WASM module.
	junk := []byte("\x00asm this is not a module")
	fp := blobFP(junk)

	_, err := n.handleWASMBlobPut(encodeWASMBlobPut(fp, junk))
	if err == nil {
		t.Fatal("put accepted a module that does not compile")
	}
	if !errors.Is(err, ErrWASMBlobRefused) {
		t.Fatalf("put refusal must be ErrWASMBlobRefused, got %T: %v", err, err)
	}
	if !strings.Contains(err.Error(), "does not compile") {
		t.Errorf("refusal must say the module does not compile: %q", err.Error())
	}
	if got := blobsDirEntries(t, n.cfg.DataDir); len(got) != 0 {
		t.Fatalf("a module that does not compile was stored anyway: %v", got)
	}
}

// TestWASMBlobRoundTripsAndRefusesAnUnknownFingerprint is the transport's whole
// reason to exist, asserted end to end on one node: bytes put under a
// fingerprint come back out under that fingerprint, and they VERIFY.
//
// The unknown-fingerprint half matters as much as the round trip. "I do not hold
// this" is the ordinary answer a fetcher must be able to act on — it means "ask
// someone else" — so it has to be an error the caller can tell apart from a
// malformed request, and it must not be a panic, a hang, or an empty success.
func TestWASMBlobRoundTripsAndRefusesAnUnknownFingerprint(t *testing.T) {
	n := newTestNode(t, 1)
	incr := readIncrWASM(t)
	fp := blobFP(incr)

	if _, err := n.handleWASMBlobPut(encodeWASMBlobPut(fp, incr)); err != nil {
		t.Fatalf("put: %v", err)
	}
	got, err := n.handleWASMBlobGet([]byte(fp))
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if blobFP(got) != fp {
		t.Fatalf("get returned %d bytes hashing to %s, not to the requested %s", len(got), blobFP(got), fp)
	}

	// A fingerprint this node does not hold: well-formed, simply absent.
	absent := hex.EncodeToString(make([]byte, sha256.Size))
	if _, err := n.handleWASMBlobGet([]byte(absent)); err == nil {
		t.Fatal("get returned something for a fingerprint this node does not hold")
	} else if !errors.Is(err, errWASMBlob) {
		t.Fatalf("an absent blob must be errWASMBlob, got %T: %v", err, err)
	}

	// And a blob whose contents have rotted under it is refused rather than
	// served: the get's answer is only ever bytes that re-derive their own name.
	path, err := wasmBlobPath(n.cfg.DataDir, fp)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(append([]byte(nil), incr...), 0x7f), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := n.handleWASMBlobGet([]byte(fp)); err == nil {
		t.Fatal("get served a blob whose contents do not hash to its own name")
	} else if !errors.Is(err, errWASMBlob) {
		t.Fatalf("a corrupt blob must be errWASMBlob, got %T: %v", err, err)
	}
}

// TestWASMBlobFingerprintTraversalIsRefused is the path-traversal gate on BOTH
// new ops.
//
// This is the same class as the op-name traversal fixed earlier — an
// admin-reachable arbitrary file write — and it is newly reachable because a
// fingerprint now arrives over the WIRE and is joined onto a filesystem path.
// filepath.Join RESOLVES "../.." rather than failing on it, so the only thing
// standing between a hostile fingerprint and an arbitrary path is wasmBlobPath's
// format check, and both ops must route through it.
//
// The uppercase case is here deliberately: it traverses nothing, and it is the
// one that would survive a hand-rolled "reject anything containing a separator"
// check. A blob store that accepts both "AB..." and "ab..." for one blob has two
// names for one content address, which is not a content address.
//
// IT ASSERTS WHICH CHECK REFUSES, not merely that something did, and that is the
// difference between a gate and a coincidence. On the PUT path a malformed
// fingerprint is ALSO caught by the content check further down (bytes cannot hash
// to a string that is not a hash), so deleting the format check would leave every
// one of these still refused — with a message about hashing. That accidental
// cover is a property of the ONE write site this handler happens to have today
// (writeWASMBlob addresses the file by the hash it computes, never by the claimed
// string), not of the op, and the next write added here would not inherit it. The
// message assertion is what stops the format check being deleted as redundant.
// On the GET path there is no such cover: wasmBlobPath is the ONLY thing between
// a wire fingerprint and an arbitrary file read.
func TestWASMBlobFingerprintTraversalIsRefused(t *testing.T) {
	n := newTestNode(t, 1)
	incr := readIncrWASM(t)
	real := blobFP(incr)

	cases := []struct {
		what string
		fp   string
		// why is a fragment of wasmBlobPath's refusal. Asserting it is what makes
		// the FORMAT check the thing under test rather than whatever else happens
		// to reject the same input — see the header note.
		why string
	}{
		// Exactly wasmBlobFPHexLen bytes, so the length check cannot be what
		// refuses these — the FORMAT check has to.
		{"a path separator", "../../../../../../../../../../../../../../../../tmp/x/" + real[54:], "not a sha256 hex fingerprint"},
		{"a parent-directory escape", strings.Repeat("../", 21) + "a", "not a sha256 hex fingerprint"},
		{"a NUL byte", "\x00" + real[1:], "not a sha256 hex fingerprint"},
		{"upper-case hex", strings.ToUpper(real), "not lower-case hex"},
	}
	for _, c := range cases {
		if len(c.fp) != wasmBlobFPHexLen {
			t.Fatalf("%s: test fingerprint is %d bytes, must be exactly %d so the length check is not what refuses it",
				c.what, len(c.fp), wasmBlobFPHexLen)
		}
		switch _, err := n.handleWASMBlobPut(encodeWASMBlobPut(c.fp, incr)); {
		case err == nil:
			t.Errorf("put accepted %s in a fingerprint", c.what)
		case !errors.Is(err, ErrWASMBlobRefused):
			t.Errorf("put refusal for %s must be ErrWASMBlobRefused, got %T: %v", c.what, err, err)
		case !strings.Contains(err.Error(), c.why):
			t.Errorf("put must refuse %s BY THE FINGERPRINT FORMAT CHECK (want %q), got: %v", c.what, c.why, err)
		}
		switch _, err := n.handleWASMBlobGet([]byte(c.fp)); {
		case err == nil:
			t.Errorf("get accepted %s in a fingerprint", c.what)
		case !errors.Is(err, errWASMBlob):
			t.Errorf("get refusal for %s must be errWASMBlob, got %T: %v", c.what, err, err)
		case !strings.Contains(err.Error(), c.why):
			t.Errorf("get must refuse %s BY THE FINGERPRINT FORMAT CHECK (want %q), got: %v", c.what, c.why, err)
		}
	}
	if got := blobsDirEntries(t, n.cfg.DataDir); len(got) != 0 {
		t.Fatalf("a traversing fingerprint wrote %v", got)
	}
	// Nothing escaped the data dir either: the whole point is that no path
	// outside <dataDir>/wasm/blobs is representable.
	if _, err := os.Stat(filepath.Join(n.cfg.DataDir, "..", "x")); err == nil {
		t.Fatal("a traversing fingerprint wrote outside the data dir")
	}
}

// TestWASMBlobPutRefusesAnOversizedOrUndecodableFrameBeforeAnyWork is the
// payload-bounds gate, and it is the mirror of checkWASMRegistrationArgs's
// "THE ENCODED SIZE CAP RUNS FIRST, BEFORE THE DECODE".
//
// __wasm_blob_put__ carries module bytes, so it needs __register_wasm__'s size
// discipline. Nothing below the client edge bounds it: server.MaxFrameSize admits
// 16 MiB. The cap has to be checked on the RAW frame, because every check that
// could bound the module — the header split, the sha256, the compile — is work
// done on a payload that may be 16 MiB of garbage, and a decoded-field cap cannot
// see a frame that has no valid header at all.
//
// The short-frame case is the other half: a frame under wasmBlobFPHexLen has no
// fingerprint to read, and slicing one out of it would panic.
func TestWASMBlobPutRefusesAnOversizedOrUndecodableFrameBeforeAnyWork(t *testing.T) {
	n := newTestNode(t, 1)

	// One byte past the cap. The body is zeros, so it is neither a real module
	// nor correctly fingerprinted — if the frame cap does not fire first, one of
	// the later checks will, and the assertion on the MESSAGE is what tells the
	// two apart.
	over := make([]byte, maxWASMBlobPutFrame+1)
	copy(over, blobFP(nil))
	_, err := n.handleWASMBlobPut(over)
	if err == nil {
		t.Fatal("put accepted a frame over the cap")
	}
	if !errors.Is(err, ErrWASMBlobRefused) {
		t.Fatalf("oversize refusal must be ErrWASMBlobRefused, got %T: %v", err, err)
	}
	if !strings.Contains(err.Error(), "over the") {
		t.Errorf("an oversized frame must be refused BY THE FRAME CAP, not by a later check: %q", err.Error())
	}

	// Undecodable: too short to carry a fingerprint at all. Every length from
	// nothing to one byte short must be refused without panicking.
	for _, size := range []int{0, 1, wasmBlobFPHexLen - 1} {
		short := make([]byte, size)
		if _, err := n.handleWASMBlobPut(short); err == nil {
			t.Errorf("put accepted a %d-byte frame", size)
		} else if !errors.Is(err, ErrWASMBlobRefused) {
			t.Errorf("short-frame refusal must be ErrWASMBlobRefused, got %T: %v", err, err)
		}
		// The get's frame bound is exact rather than a minimum: its payload IS a
		// fingerprint, so any other length is refused before any work.
		if _, err := n.handleWASMBlobGet(short); err == nil {
			t.Errorf("get accepted a %d-byte payload", size)
		} else if !errors.Is(err, ErrWASMBlobRefused) {
			t.Errorf("get short-payload refusal must be ErrWASMBlobRefused, got %T: %v", err, err)
		}
	}
	// A get payload one byte too LONG is refused too. That is what makes the
	// format canonical: there is nowhere for a padding byte to ride along.
	if _, err := n.handleWASMBlobGet([]byte(blobFP(nil) + "0")); err == nil {
		t.Error("get accepted a payload longer than a fingerprint")
	}

	if got := blobsDirEntries(t, n.cfg.DataDir); len(got) != 0 {
		t.Fatalf("a refused frame wrote %v", got)
	}
}

// TestWASMBlobGetTouchesNoApplyLock defends opWASMBlobGetName's CONTRACT
// mechanically rather than by comment: the get must complete while wasmApplyMu —
// the lock every applyWASMRegistration holds for the whole of its compile, its
// two file writes and its registry install — is held by someone else.
//
// WHY THIS IS THE PROPERTY WORTH PINNING. A node fetches a blob because it is
// trying to apply something that needs it, so with the thin marker the fetch is issued
// from inside that position. If serving a get took a lock an apply holds, then
// node A blocked in an apply waiting on a fetch from B, while B is blocked in an
// apply holding the lock that would serve it, is a deadlock that no timeout turns
// into progress. Keeping the get free of every apply-path lock removes the edge
// rather than reasoning about the cycle.
//
// WHAT IT GENUINELY CATCHES, stated exactly, because an earlier version of this
// comment claimed a list it did not cover. It holds the TWO locks an apply
// actually holds — cluster's wasmApplyMu, and the wasm.Runtime module-table write
// lock that AddModule takes from inside that apply — so it catches:
//
//   - consulting n.wasmState to find which op a blob belongs to (wasmApplyMu);
//   - short-circuiting on rt.HasModule to skip the disk read (the Runtime's
//     RWMutex, for reading, against the write lock held here). This is the real
//     deadlock edge and the reason the test was widened: holding wasmApplyMu
//     alone, a get that called HasModule passed;
//   - serving from a cache guarded by EITHER of those two mutexes.
//
// What it does NOT catch, so nobody reads more into a green run than is there: a
// cache guarded by some THIRD mutex an apply also happens to take, and a
// freshness check through Raft or the meta FSM — that one needs the get to wait
// on an apply LOOP, and holding a lock does not stall one. The route gate is not
// in either list: it is an atomic.Pointer load and cannot block anything.
func TestWASMBlobGetTouchesNoApplyLock(t *testing.T) {
	n := newTestNode(t, 1)
	incr := readIncrWASM(t)
	fp := blobFP(incr)
	if _, err := n.handleWASMBlobPut(encodeWASMBlobPut(fp, incr)); err != nil {
		t.Fatalf("put: %v", err)
	}

	// Stand in for an apply in flight, in the exact state it is in at the moment
	// it installs a module: applyWASMRegistration holds wasmApplyMu for the whole
	// call and rt.AddModule takes the runtime's module-table write lock inside it.
	// Both are held for the whole of the get below.
	n.wasmApplyMu.Lock()
	defer n.wasmApplyMu.Unlock()
	releaseRT := n.wasmRT.HoldModuleTableForTest()
	defer releaseRT()

	type result struct {
		b   []byte
		err error
	}
	done := make(chan result, 1)
	go func() {
		b, err := n.handleWASMBlobGet([]byte(fp))
		done <- result{b, err}
	}()

	select {
	case r := <-done:
		if r.err != nil {
			t.Fatalf("get under a held wasmApplyMu: %v", r.err)
		}
		if blobFP(r.b) != fp {
			t.Fatalf("get returned bytes hashing to %s, not %s", blobFP(r.b), fp)
		}
	case <-time.After(cpuScaled(10 * time.Second)):
		t.Fatal("__wasm_blob_get__ blocked on an apply-path lock (wasmApplyMu or the wasm runtime's " +
			"module table), which is exactly the dependency opWASMBlobGetName's contract forbids")
	}
}

// TestWASMRegistrationPushesBlobToEveryMember is the gate on the push phase's
// happy path: registering a module delivers its bytes to every OTHER member,
// with a compile verdict from each, before the marker is broadcast.
//
// THE ASSERTION IS ON THE PUSH, NOT ON THE BLOB'S PRESENCE, and the reason
// changed with thin markers rather than going away. It used to be that a
// blob-store assertion proved nothing: applyWASMRegistration wrote the file from
// the registration's own inline Bytes on every node, so the file would be there
// with the push deleted entirely, and only the ack COUNTER distinguished the two.
// Now the marker carries no bytes and the push IS the only thing that puts them
// anywhere — so the file's presence has become a real signal, and the last block
// below asserts exactly that by making every member SERVE it. The counter is still
// checked first because it distinguishes "every member acked" from "the bytes are
// somehow there", which is the difference the durability floor is stated over.
//
// It also pins that both ops are actually dispatchable — an authz entry for a
// name no node serves guards nothing.
func TestWASMRegistrationPushesBlobToEveryMember(t *testing.T) {
	const nodes = 3
	incr := readIncrWASM(t)
	tc := newTestCluster(t, nodes, 1)
	waitClusterApplied(t, tc.nodes)

	leader, ok := tc.findShardLeader(0)
	if !ok {
		t.Fatal("no leader for shard 0")
	}
	n := tc.nodes[leader]
	for _, op := range []string{opWASMBlobPutName, opWASMBlobGetName} {
		if _, dispatchable := n.adminOps[op]; !dispatchable {
			t.Fatalf("%s is not registered in adminOps; nothing serves it", op)
		}
	}

	before := n.Stats().WASMBlobPush
	reply, err := n.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(ops.WASMRegistration{
		Name: "wasm_blob_push", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr), ExportName: "apply",
	}, incr))
	if err != nil {
		t.Fatalf("register: %v", err)
	}
	if len(reply) != 0 {
		t.Fatalf("every member was reachable, so the skip report must be empty; got %q", reply)
	}

	got := n.Stats().WASMBlobPush
	if want := before.Acks + nodes - 1; got.Acks != want {
		t.Errorf("push acks = %d, want %d (one per other member)", got.Acks, want)
	}
	if got.Skips != before.Skips {
		t.Errorf("push skips = %d, want %d (no member was unreachable)", got.Skips, before.Skips)
	}

	// Every member can now SERVE the bytes, which is the invariant this stage
	// exists to establish: any node can obtain them from any node that has them.
	fp := blobFP(incr)
	for i, peer := range tc.nodes {
		out, err := peer.Call(opWASMBlobGetName, []byte(fp))
		if err != nil {
			t.Errorf("node %d cannot serve blob %s: %v", i, fp, err)
			continue
		}
		if blobFP(out) != fp {
			t.Errorf("node %d served %d bytes hashing to %s, not %s", i, len(out), blobFP(out), fp)
		}
	}
}

// TestWASMBlobPushToleratesAnUnreachableMember is the gate on the tolerance rule.
//
// An unreachable member has rendered NO verdict, so there is nothing to honour
// and the registration must proceed — otherwise any node being restarted stops
// the whole cluster from registering a module, and a rolling restart becomes a
// registration outage. What must NOT happen is for it to proceed SILENTLY: the
// member is missing bytes its groups will need, so it has to be named.
//
// THE TOLERANCE IS BOUNDED BY THE DURABILITY FLOOR, and the cluster size here is
// chosen to sit inside it rather than by accident. A marker carries no module, so
// a version no MAJORITY of members holds could become permanently unfetchable and
// block every shard group; pushWASMBlob therefore refuses a registration that
// cannot reach the floor (see its "THE DURABILITY FLOOR" block). Three members
// with one unreachable leaves two holders out of three — a genuine majority — so
// what is under test here is the tolerance and not the floor. Losing TWO of three
// would be refused, and that is correct, not a regression in tolerance.
//
// THE SETUP, and why it is shaped this way: only the member's CLIENT-FACING
// server is closed. Its Raft transport is a different listener and stays up, so
// the group still commits with all three replicas and the broadcast is unaffected
// — the only thing that breaks is the push's dial. One shard group means the
// registrar (which leads it) proposes locally and never forwards over a
// client-facing port either, so the closed server can affect nothing but the leg
// under test.
func TestWASMBlobPushToleratesAnUnreachableMember(t *testing.T) {
	const nodes = 3
	incr := readIncrWASM(t)
	tc := newTestCluster(t, nodes, 1)
	waitClusterApplied(t, tc.nodes)

	leader, ok := tc.findShardLeader(0)
	if !ok {
		t.Fatal("no leader for shard 0")
	}
	victim := (leader + 1) % nodes
	if err := tc.servers[victim].Close(); err != nil {
		t.Fatalf("close node %d's server: %v", victim, err)
	}
	tc.servers[victim] = nil // do not double-close in tc.Close

	n := tc.nodes[leader]
	before := n.Stats().WASMBlobPush
	reply, err := n.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(ops.WASMRegistration{
		Name: "wasm_blob_partial", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr), ExportName: "apply",
	}, incr))
	if err != nil {
		t.Fatalf("an unreachable member must not fail the registration: %v", err)
	}

	// It must be REPORTED, by node id, and it must say what the consequence is.
	report := string(reply)
	if !strings.Contains(report, tc.peers[victim].NodeID) {
		t.Fatalf("the skip report must name the unreachable member %q; got %q", tc.peers[victim].NodeID, report)
	}
	if !strings.Contains(report, "unreachable") {
		t.Errorf("the skip report must say the member was unreachable: %q", report)
	}
	// And a reachable member must NOT be reported as skipped — a report that
	// names everyone is as useless as no report.
	other := (leader + 2) % nodes
	if other != leader && strings.Contains(report, tc.peers[other].NodeID) {
		t.Errorf("reachable member %q must not appear in the skip report: %q", tc.peers[other].NodeID, report)
	}

	got := n.Stats().WASMBlobPush
	if want := before.Skips + 1; got.Skips != want {
		t.Errorf("push skips = %d, want %d", got.Skips, want)
	}
	if want := before.Acks + 1; got.Acks != want {
		t.Errorf("push acks = %d, want %d (the one reachable member)", got.Acks, want)
	}
}

// TestWASMBlobPushTargetsAreTheUnionOfMembersAndConfig pins that no member can
// be invisible to the push.
//
// The failure this closes is specific and silent. wasmBlobPushTargets used to
// PREFER meta Members and read cfg.Peers only when Members was empty, so a node
// config knows about but that meta has not yet published — a joiner whose
// OpSetMembers has not committed, a Members table lagging a config change — was
// neither pushed to nor named in the skip report. It simply did not exist as far
// as the registration was concerned, which is the exact shape
// wasmBlobPushTargets' own contract says must not happen: with the thin marker the
// marker carries no bytes, so a member the push never considered is a member
// whose groups cannot execute the op and nothing said so.
//
// The union's cost is asserted too, in the opposite direction: a node in config
// but NOT in Members is still a target. That is deliberate — a stale name in a
// skip report is diagnosable; a member silently missing every module's bytes is
// not — and pinning it here stops it being "fixed" back into invisibility.
func TestWASMBlobPushTargetsAreTheUnionOfMembersAndConfig(t *testing.T) {
	n := newTestNode(t, 1)
	n.cfg.NodeID = "n1"
	// Config knows n1 (self), n2 and n3. Meta has published only n1 and n2, and
	// carries a fresher server address for n2 than config does.
	n.cfg.Peers = []Peer{
		{NodeID: "n1", ServerAddr: "self:1"},
		{NodeID: "n2", ServerAddr: "stale:2"},
		{NodeID: "n3", ServerAddr: "n3:3"},
	}
	f := NewMetaFSM()
	data, err := encodeLogEntry(LogEntry{
		Op: OpSetMembers,
		Members: []Peer{
			{NodeID: "n1", ServerAddr: "self:1"},
			{NodeID: "n2", ServerAddr: "live:2"},
		},
		NumShards: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := f.Apply(&raft.Log{Data: data}); got != nil {
		t.Fatalf("apply OpSetMembers: %v", got)
	}
	// A synthetic meta group: wasmBlobPushTargets reads only FSM.State(), and this
	// node is single-node so nothing else touches n.meta. Detached again before
	// newTestNode's Close cleanup runs, which would otherwise shut down a MetaRaft
	// that has no Raft.
	n.meta = &MetaRaft{FSM: f}
	defer func() { n.meta = nil }()

	got := map[string]string{}
	for _, m := range n.wasmBlobPushTargets() {
		if _, dup := got[m.nodeID]; dup {
			t.Errorf("%s appears twice in the push targets: it would be pushed to, and counted, twice", m.nodeID)
		}
		got[m.nodeID] = m.serverAddr
	}
	// n3 is the whole point: config-only, and it must be a target.
	if _, ok := got["n3"]; !ok {
		t.Errorf("n3 is in cfg.Peers but not in Members, and is not a push target: it would be neither "+
			"pushed to nor named in the skip report. targets = %v", got)
	}
	// Members wins on the address where both tables know a node.
	if got["n2"] != "live:2" {
		t.Errorf("n2's address = %q, want the live Members value %q, not the frozen cfg.Peers one", got["n2"], "live:2")
	}
	// Self is never a target: pushWASMBlob stores locally before it pushes.
	if _, ok := got["n1"]; ok {
		t.Errorf("this node is its own push target: targets = %v", got)
	}
	if len(got) != 2 {
		t.Errorf("targets = %v, want exactly n2 and n3", got)
	}
}

// TestWASMBlobPeerRefusalIsNotMistakenForUnreachability pins the line the whole
// tolerance rule turns on: a peer that ANSWERED is honoured, a peer that could
// not be reached is skipped.
//
// Getting this backwards in either direction is a real failure. Classifying a
// refusal as unreachable throws away the compile verdict this stage exists to
// obtain — the registration would proceed past a node that has explicitly said it
// cannot run the module. Classifying a transport failure as a refusal makes any
// node being restarted able to fail every registration in the cluster.
func TestWASMBlobPeerRefusalIsNotMistakenForUnreachability(t *testing.T) {
	refusals := []error{
		&client.RemoteError{Op: opWASMBlobPutName, Msg: "module does not compile on this node"},
		client.ErrUnauthorized,
		client.ErrNotFound,
		// A genuine refusal whose message ALSO contains the unknown-op substring.
		// The substring is forgeable by the payload — a compile failure's text
		// quotes module-controlled names — so the mixed-version carve-out below
		// must not be triggerable by a module that arranges to fail this way. The
		// discriminator is the refusal marker, which only a real refusal carries.
		&client.RemoteError{
			Op:  opWASMBlobPutName,
			Msg: ErrWASMBlobRefused.Error() + `: module does not compile on this node: export "op not registered" is invalid`,
		},
	}
	for _, err := range refusals {
		if !wasmBlobPeerRefused(err) {
			t.Errorf("%v is a verdict FROM the peer and must fail the registration", err)
		}
		// It must survive wrapping: pushWASMBlob sees the error after the client's
		// own layers have wrapped it.
		if !wasmBlobPeerRefused(fmt.Errorf("member n2: %w", err)) {
			t.Errorf("wrapped %v must still be recognised as a peer verdict", err)
		}
	}
	unreachable := []error{
		errors.New("dial tcp 127.0.0.1:1: connect: connection refused"),
		context.DeadlineExceeded,
		io.EOF,
		ErrNoShardOwner,
		// THE MIXED-VERSION PEER, and it is the case with the largest blast radius
		// of anything in this test.
		//
		// A node on any build older than this file has no __wasm_blob_put__ in
		// adminOps and none in the ops registry, so Node.Call answers ErrUnknownOp;
		// server.clientFacingErr keeps that message unredacted and returns
		// StatusError, which client.mapCallStatus turns into a *client.RemoteError
		// — structurally identical to a compile refusal. Read as one, EVERY
		// __register_wasm__ in the cluster fails for the whole of a rolling upgrade
		// through this commit, with a message blaming a compile refusal for what is
		// version skew. Deploying this code IS that window, for every existing
		// cluster, unavoidably.
		//
		// The peer never ran the handler, so it rendered no verdict on the module.
		// There is nothing to honour: skip it and name it, exactly like a peer that
		// could not be reached at all.
		&client.RemoteError{Op: opWASMBlobPutName, Msg: ErrUnknownOp.Error()},
	}
	for _, err := range unreachable {
		if wasmBlobPeerRefused(err) {
			t.Errorf("%v is a failure to REACH the peer and must not fail the registration", err)
		}
		if wasmBlobPeerRefused(fmt.Errorf("member n2: %w", err)) {
			t.Errorf("wrapped %v must still not be treated as a peer verdict", err)
		}
	}
}

// TestWASMBlobUnknownOpMsgTracksErrUnknownOp pins the substring
// wasmBlobPeerRefused matches to the sentinel that produces it.
//
// The match has to be by substring — the refusal is raised on the PEER and
// reaches this node as a bare string, so no sentinel identity survives the RPC
// boundary — which means the coupling is invisible to the compiler. If
// ErrUnknownOp is ever reworded, the carve-out silently stops firing and every
// mixed-version peer goes back to being classified as a compile refusal, which is
// the outage this const exists to prevent. Fail here instead.
func TestWASMBlobUnknownOpMsgTracksErrUnknownOp(t *testing.T) {
	if !strings.Contains(ErrUnknownOp.Error(), wasmBlobUnknownOpMsg) {
		t.Fatalf("ErrUnknownOp is %q, which no longer contains %q: a peer that does not know %s "+
			"would be classified as having REFUSED the module, failing every registration during a "+
			"mixed-version rollout", ErrUnknownOp.Error(), wasmBlobUnknownOpMsg, opWASMBlobPutName)
	}
	// And the marker that keeps a real refusal out of the carve-out must not be a
	// substring of the unknown-op message, or the carve-out could never fire.
	if strings.Contains(ErrUnknownOp.Error(), ops.WASMRegistrationRefusedMsg) {
		t.Fatalf("ErrUnknownOp %q now carries the refusal marker %q, so a mixed-version peer is "+
			"indistinguishable from a refusal", ErrUnknownOp.Error(), ops.WASMRegistrationRefusedMsg)
	}
}

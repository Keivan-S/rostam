// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/ops"
)

// readDelWASM loads a SECOND test module. It exports the same "apply" symbol as
// incr.wasm and is equally safe to invoke, so swapping one for the other changes
// nothing but the bytes — which is exactly the "same name, different content"
// case the update gate exists to refuse.
func readDelWASM(t *testing.T) []byte {
	t.Helper()
	// FATAL, not Skip. del.wasm is checked into the repo, so "not readable" is a
	// broken checkout or a broken build, never a legitimate reason to pass. As a
	// Skipf it silently disabled the ENTIRE update-gate suite — every case in this
	// file goes through this helper — so the branch's headline invariant could have
	// stopped being enforced with a green run to show for it.
	b, err := os.ReadFile("../wasm/testdata/del.wasm")
	if err != nil {
		t.Fatalf("del.wasm must be present (it is checked in): %v", err)
	}
	return b
}

// routableWASMReg is routableWASMPayload's struct form, so a test can vary one
// field at a time.
func routableWASMReg(t *testing.T, name string, epoch uint64) ops.WASMRegistration {
	t.Helper()
	return ops.WASMRegistration{
		Name:       name,
		Kind:       ops.OpReadWrite,
		Blob:       ops.WASMBlobFingerprint(readIncrWASM(t)),
		ExportName: "apply",
		Epoch:      epoch,
	}
}

// installedReg returns the registration this node currently has installed under
// name, straight out of the authoritative state.
func installedReg(t *testing.T, n *Node, name string) ops.WASMRegistration {
	t.Helper()
	n.wasmApplyMu.Lock()
	defer n.wasmApplyMu.Unlock()
	return n.wasmState.installed[name].reg
}

// TestWASMContractChangeIsRefusedAtProposeTime is the gate for what per-group
// version binding did NOT make safe.
//
// A registration may change the module. It may not change the op's CONTRACT —
// its Kind — because that is read on the PROPOSE side, before any shard group is
// known: Kind decides whether the invocation is replicated at all. It cannot be
// resolved per group without knowing the group first, and the group cannot be
// known without resolving it, so it is frozen at first registration and a change
// is a new-op-name operation.
//
// The key extractor used to be the other half of that contract. It is now
// CONSTANT — WASMRegistration has no field for it — so there is nothing left here
// to refuse. See ops.WASMKeyExtractorHandle.
//
// Three things are asserted per case, and the second matters as much as the
// first: the call is refused, and NOTHING was proposed. A refusal that still put
// the entry into some group's log would have started the very change it claims to
// have prevented.
func TestWASMContractChangeIsRefusedAtProposeTime(t *testing.T) {
	const numShards = 4
	n := newTestNode(t, numShards)
	waitAllLeaders(t, n)
	waitAllApplied(t, n)

	if _, err := n.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(routableWASMReg(t, "wasm_incr", 1), readIncrWASM(t))); err != nil {
		t.Fatalf("first registration of a fresh name must succeed: %v", err)
	}
	wantFP := ops.WASMRegistrationFingerprint(installedReg(t, n, "wasm_incr"))

	// A different Kind decides whether an entry is replicated at all — this is
	// exactly the errPBApplyReadOnly skew shard/apply_class.go documents.
	//
	// KIND IS THE ONLY HALF OF THE CONTRACT THIS TEST CAN STILL EXERCISE, and that
	// is the point rather than a gap. The other half — the key extractor — used to
	// be a second case here; WASMRegistration has no field for it any more, so the
	// case is not merely refused but unwritable. The invariant it used to protect
	// is gated by TestWASMRegistrationsOfOneNameRouteIdentically.
	differentKind := routableWASMReg(t, "wasm_incr", 2)
	differentKind.Kind = ops.OpReadOnly

	for _, tc := range []struct {
		what string
		reg  ops.WASMRegistration
	}{
		{"different kind", differentKind},
	} {
		t.Run(tc.what, func(t *testing.T) {
			before := lastLogIndexes(t, n)

			_, err := n.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(tc.reg, readIncrWASM(t)))
			// Reported rather than fatal, so the "nothing was proposed" assertion
			// below still runs: it is the deeper of the two claims.
			if !errors.Is(err, ErrWASMUpdateUnsupported) {
				t.Errorf("re-registering a live name with %s: got %v, want ErrWASMUpdateUnsupported", tc.what, err)
			}
			if err != nil {
				// The message has to survive the redaction classifiers
				// (server.clientFacingErr and httpapi.statusForError both key off
				// this substring) or the caller sees "internal error" and never
				// learns the remedy.
				if !strings.Contains(err.Error(), ops.WASMUpdateUnsupportedMsg) {
					t.Errorf("refusal would be redacted to a generic internal error: %q", err.Error())
				}
				if !strings.Contains(err.Error(), "NEW op name") {
					t.Errorf("refusal does not state the remedy: %q", err.Error())
				}
			}

			// Nothing was proposed. This is the assertion that fails first when the
			// check is removed: the registration reaches every group's log.
			for i, after := range lastLogIndexes(t, n) {
				if after != before[i] {
					t.Errorf("shard %d log grew (last_log_index %d -> %d): the refused registration was still proposed",
						i, before[i], after)
				}
			}

			// And the live module is untouched.
			if got := ops.WASMRegistrationFingerprint(installedReg(t, n, "wasm_incr")); got != wantFP {
				t.Errorf("the installed module changed under a refused registration")
			}
		})
	}

	// The op still works: a refusal is a clean no-op, not a wedge.
	for i := 0; i < numShards; i++ {
		if _, err := n.Call("wasm_incr", keyForShard(t, i, numShards)); err != nil {
			t.Errorf("group %d must keep serving after a refused contract change: %v", i, err)
		}
	}
}

// TestWASMBytesUpdateIsAcceptedEndToEnd is the positive half of the same rule,
// and it is the operation per-group version binding exists to legalise.
//
// A second registration under a live name that changes only the MODULE must be
// accepted, proposed to every group, installed node-wide, and — the part that
// makes it safe — bound in EVERY group at the new version, so that whichever
// group an invocation lands in, every replica of that group executes it with the
// version that group's own log committed.
func TestWASMBytesUpdateIsAcceptedEndToEnd(t *testing.T) {
	const numShards = 4
	n := newTestNode(t, numShards)
	waitAllLeaders(t, n)
	waitAllApplied(t, n)

	v1 := routableWASMReg(t, "wasm_incr", 1)
	if _, err := n.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(v1, readIncrWASM(t))); err != nil {
		t.Fatalf("first registration: %v", err)
	}
	v1ID := wasmModuleIDOf(v1)
	for g := 0; g < numShards; g++ {
		if got := provenGroups(t, n, "wasm_incr")[g].id; got != v1ID {
			t.Fatalf("precondition: group %d is bound to %s, want v1 %s", g, got, v1ID)
		}
	}

	// The update: same name, same Kind, same extractor, DIFFERENT module.
	v2 := routableWASMReg(t, "wasm_incr", 2)
	v2.Blob = ops.WASMBlobFingerprint(readDelWASM(t))
	if _, err := n.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(v2, readDelWASM(t))); err != nil {
		t.Fatalf("a bytes-only update must be accepted: per-group version binding is what makes it safe, and refusing it is the behaviour that was removed: %v", err)
	}

	// Node-wide: the registry entry and the recorded install both moved.
	if got := installedReg(t, n, "wasm_incr"); got.Blob != v2.Blob || got.Epoch != 2 {
		t.Errorf("node-wide install after the update = blob %s epoch %d, want blob %s epoch 2", ops.WASMBlobHex(got.Blob), got.Epoch, ops.WASMBlobHex(v2.Blob))
	}

	// Per group: EVERY group's binding moved, because the broadcast reached them
	// all. This is the assertion that distinguishes "the update was accepted" from
	// "the update took effect where entries actually execute".
	v2ID := wasmModuleIDOf(v2)
	for g := 0; g < numShards; g++ {
		got, ok := provenGroups(t, n, "wasm_incr")[g]
		if !ok {
			t.Errorf("group %d lost its binding across the update: a committed invocation in its log would now halt the node with ErrWASMNoGroupBinding", g)
			continue
		}
		if got.id != v2ID {
			t.Errorf("group %d is still bound to %s after the update, want %s", g, got.id, v2ID)
		}
	}

	// And the op still serves on every group.
	for i := 0; i < numShards; i++ {
		if _, err := n.Call("wasm_incr", keyForShard(t, i, numShards)); err != nil {
			t.Errorf("group %d must serve the op after the update: %v", i, err)
		}
	}
}

// TestWASMEpochOnlyBumpIsAccepted pins a rule that REVERSED.
//
// An Epoch bump with identical content used to be refused, on the reasoning that
// Epoch exists to order updates so bumping it is an attempted update by
// definition — which was a sound thing to say while updates were unsupported. Now
// that they are supported it is just an update whose module happens to be
// unchanged, and it converges like any other.
func TestWASMEpochOnlyBumpIsAccepted(t *testing.T) {
	const numShards = 2
	n := newTestNode(t, numShards)
	waitAllLeaders(t, n)
	waitAllApplied(t, n)

	if _, err := n.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(routableWASMReg(t, "wasm_incr", 1), readIncrWASM(t))); err != nil {
		t.Fatalf("first registration: %v", err)
	}
	if _, err := n.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(routableWASMReg(t, "wasm_incr", 2), readIncrWASM(t))); err != nil {
		t.Fatalf("an epoch-only bump must be accepted: %v", err)
	}
	if got := installedReg(t, n, "wasm_incr").Epoch; got != 2 {
		t.Errorf("installed epoch = %d, want 2", got)
	}
}

// TestIdenticalWASMReRegistrationIsAllowedAfterAPartialBroadcast is the
// counterweight, and it is the case that actually matters in production.
//
// broadcastWASMRegistration is documented as safe to retry, and the recovery
// path for a PARTIAL broadcast is precisely re-sending the same registration:
// under the route gate a group the first broadcast starved serves nothing until
// a retry lands the entry in its log. An update check that refused every second
// registration of a live name would turn a routine partial broadcast — one
// election, one slow peer, one timeout — into a permanently unroutable op.
func TestIdenticalWASMReRegistrationIsAllowedAfterAPartialBroadcast(t *testing.T) {
	const numShards, starved = 4, 2
	n := newTestNode(t, numShards)
	waitAllLeaders(t, n)
	waitAllApplied(t, n)

	reg := routableWASMReg(t, "wasm_incr", 1)

	// A partial broadcast: every group but one accepts, and the ones that
	// accepted KEEP the registration — so the retry below is a second
	// registration of a name this node already has installed.
	reattach := detachGroup(t, n, starved)
	_, err := n.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(reg, readIncrWASM(t)))
	reattach()
	if err == nil {
		t.Fatal("precondition: a broadcast that could not reach one group must report the failure")
	}
	if _, proven := provenGroups(t, n, "wasm_incr")[starved]; proven {
		t.Fatalf("precondition: group %d must not be proven after a partial broadcast", starved)
	}
	if _, err := n.Call("wasm_incr", keyForShard(t, starved, numShards)); !errors.Is(err, ErrWASMOpNotInThisGroup) {
		t.Fatalf("precondition: the starved group must refuse invocations: got %v", err)
	}

	// The retry: the IDENTICAL struct, Epoch included, which is what a
	// well-behaved client re-sends. It must be accepted.
	if _, err := n.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(reg, readIncrWASM(t))); err != nil {
		t.Fatalf("re-sending the identical registration is the documented retry and must be allowed: %v", err)
	}

	// And it must have REPAIRED the starved group, not merely been tolerated.
	if _, proven := provenGroups(t, n, "wasm_incr")[starved]; !proven {
		t.Errorf("the retry did not reach the starved group %d (proven=%v)", starved, sortedGroups(provenGroups(t, n, "wasm_incr")))
	}
	for i := 0; i < numShards; i++ {
		if _, err := n.Call("wasm_incr", keyForShard(t, i, numShards)); err != nil {
			t.Errorf("group %d must serve the op after the retry: %v", i, err)
		}
	}
}

// TestWASMRegistrationOfANewNameIsNotRefused pins the gate's scope: it keys on
// the op NAME, so a genuinely new module registers normally even while another
// one is live. Without this a plausible-looking "reject any second
// __register_wasm__" implementation would pass every other test here.
func TestWASMRegistrationOfANewNameIsNotRefused(t *testing.T) {
	const numShards = 4
	n := newTestNode(t, numShards)
	waitAllLeaders(t, n)
	waitAllApplied(t, n)

	if _, err := n.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(routableWASMReg(t, "wasm_incr", 1), readIncrWASM(t))); err != nil {
		t.Fatalf("first registration: %v", err)
	}

	// A different name carrying DIFFERENT bytes — the shape a caller uses to ship
	// a new version of a module, which is the remedy the refusal points at.
	v2 := routableWASMReg(t, "wasm_incr_v2", 1)
	v2.Blob = ops.WASMBlobFingerprint(readDelWASM(t))
	if _, err := n.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(v2, readDelWASM(t))); err != nil {
		t.Fatalf("registering a NEW name must not be refused: %v", err)
	}

	for _, name := range []string{"wasm_incr", "wasm_incr_v2"} {
		for i := 0; i < numShards; i++ {
			if _, err := n.Call(name, keyForShard(t, i, numShards)); err != nil {
				t.Errorf("%s on group %d: %v", name, i, err)
			}
		}
	}
}

// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/wasm"
)

// TestWASMRegistrationRefusedAfterTheWritesBricksEveryNode is the HIGH gate for
// the incomplete-validation defect.
//
// Two wire-controlled fields used to be range-checked only by ops.validateEntry,
// which applyWASMRegistration does not reach until 27 lines PAST the .wasm write
// and the sidecar write:
//
//   - a Name of length 2 (the protocol-v2 version-byte collision) or over
//     maxOpNameLen — ops.ValidateWASMOpName checked neither;
//   - a Kind outside {OpReadOnly, OpReadWrite} — wasm.ValidateModuleKind
//     short-circuits on `kind != OpReadOnly`, so 2..255 sailed through it.
//
// Both writes SUCCEED for such a registration, so the sidecar-failure unlink never
// fires and the files persist. The apply then fails, classifyApplyErr calls it
// classAdvance (no halt, no metric), and the node carries on looking healthy — but
// the NEXT RESTART's reloadWASMModulesFromDisk finds the pair, compiles it, and
// re-derives the same rejection out of the registry. That error is not
// ErrDuplicateOp, so the reload returns it, finishWASMSetup propagates it, and
// NODE CONSTRUCTION FAILS. Since the entry is broadcast to every group, every node
// applies it and every node fails to start, until an operator deletes the files by
// hand.
//
// CAVEAT, so the narrative above is not read as covering all three cases: the
// over-maxOpNameLen case has a 261-byte filename component, past Linux's NAME_MAX
// of 255, so under the unfixed code its writes fail with ENAMETOOLONG and leave
// nothing behind — the persist-then-brick story is the length-2 and bad-Kind cases.
// That case remains a real gate for the refusal itself and its error identity
// (unfixed, apply returns the write error, not ops.ErrWASMOpNameUnsafe).
//
// The gate is therefore three assertions per case, and the third is the one that
// matters: refused, no files, and the node still starts.
func TestWASMRegistrationRefusedAfterTheWritesBricksEveryNode(t *testing.T) {
	incr := readTestWASM(t, "../wasm/testdata/incr.wasm")

	cases := []struct {
		what    string
		reg     ops.WASMRegistration
		files   []string // what the unfixed code left behind
		wantIs  error
		wantMsg string
	}{
		{
			what: "a two-character op name",
			reg: ops.WASMRegistration{
				Name: "ab", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr),
				ExportName: "apply", Epoch: 1,
			},
			files:   []string{"ab.wasm", "ab.json"},
			wantIs:  ops.ErrWASMOpNameUnsafe,
			wantMsg: ops.WASMOpNameUnsafeMsg,
		},
		{
			what: "an op name over maxOpNameLen",
			reg: ops.WASMRegistration{
				Name: strings.Repeat("x", 256), Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr),
				ExportName: "apply", Epoch: 1,
			},
			files:   []string{strings.Repeat("x", 256) + ".wasm", strings.Repeat("x", 256) + ".json"},
			wantIs:  ops.ErrWASMOpNameUnsafe,
			wantMsg: ops.WASMOpNameUnsafeMsg,
		},
		{
			what: "an out-of-range Kind byte",
			reg: ops.WASMRegistration{
				Name: "wasm_badkind", Kind: ops.OpKind(2), Blob: ops.WASMBlobFingerprint(incr),
				ExportName: "apply", Epoch: 1,
			},
			files:   []string{"wasm_badkind.wasm", "wasm_badkind.json"},
			wantIs:  ErrWASMRegistrationRefused,
			wantMsg: ops.WASMRegistrationRefusedMsg,
		},
	}

	for _, tc := range cases {
		t.Run(tc.what, func(t *testing.T) {
			// PRECONDITION for the "out-of-range Kind" case: the guard that already
			// ran before the writes does not cover it. If wasm.ValidateModuleKind ever
			// starts rejecting it, this test stops proving what it claims to.
			if tc.reg.Kind != ops.OpReadOnly && tc.reg.Kind != ops.OpReadWrite {
				m, err := wasm.Compile(incr)
				if err != nil {
					t.Fatalf("compile testdata: %v", err)
				}
				vErr := wasm.ValidateModuleKind(tc.reg.Name, tc.reg.Kind, m)
				_ = m.Close() //nolint:errcheck,gosec
				if vErr != nil {
					t.Fatalf("precondition: wasm.ValidateModuleKind must NOT catch Kind %d (it short-circuits on kind != OpReadOnly); got %v", tc.reg.Kind, vErr)
				}
			}

			// PROPOSE TIME: the client is refused before anything enters a log.
			t.Run("propose time", func(t *testing.T) {
				n := newTestNode(t, 2)
				waitAllApplied(t, n)
				before := lastLogIndexes(t, n)

				// The client edge takes the marker AND the module
				// (ops.EncodeWASMRegistrationRequest); only the marker is broadcast.
				_, err := n.Call(wasmRegisterOpName, ops.EncodeWASMRegistrationRequest(tc.reg, incr))
				if !errors.Is(err, tc.wantIs) {
					t.Fatalf("Call: got %v, want %v", err, tc.wantIs)
				}
				// The refusal must survive server.clientFacingErr / httpapi.statusForError,
				// which key off this substring across the stringifying Raft/RPC boundary.
				// Redacted to "internal error" the caller cannot tell its own bad payload
				// from a server fault.
				if !strings.Contains(err.Error(), tc.wantMsg) {
					t.Errorf("refusal would be redacted to a generic internal error: %q", err.Error())
				}
				for i, after := range lastLogIndexes(t, n) {
					if after != before[i] {
						t.Errorf("shard %d log grew (%d -> %d): the refused registration was proposed anyway", i, before[i], after)
					}
				}
				if files := wasmDirFiles(t, n.cfg.DataDir); len(files) != 0 {
					t.Errorf("the refused registration left files behind: %v", files)
				}
			})

			// APPLY TIME is the authoritative leg: a replica replaying the log, and a
			// snapshot restore (installWASMSnapshotBlobLocked), never run the propose
			// check at all.
			t.Run("apply time leaves no files and the node still starts", func(t *testing.T) {
				dir := t.TempDir()
				rt, reg, st, err := restartWASM(t, dir, nil)
				if err != nil {
					t.Fatalf("first start: %v", err)
				}
				if err := applyWASMRegistration(dir, rt, reg, st, tc.reg, 0, nil); err == nil {
					t.Fatal("the registration applied")
				} else if !errors.Is(err, tc.wantIs) {
					t.Errorf("apply: got %v, want %v", err, tc.wantIs)
				}
				if files := wasmDirFiles(t, dir); len(files) != 0 {
					t.Errorf("the refused registration left %v in the wasm dir; the next restart re-derives the same rejection and fails node construction", files)
				}
				for _, f := range tc.files {
					if _, statErr := os.Stat(filepath.Join(dir, "wasm", f)); statErr == nil {
						t.Errorf("%s was written before the rejection", f)
					}
				}
				if _, ok := st.installed[tc.reg.Name]; ok {
					t.Error("the refused registration was recorded as installed")
				}
				// THE POINT OF THE WHOLE TEST. With the files on disk this call is what
				// returned the error that failed node construction cluster-wide.
				if _, _, _, err := restartWASM(t, dir, nil); err != nil {
					t.Fatalf("the node no longer starts after a refused registration: %v", err)
				}
			})
		})
	}
}

// TestUndecodableWASMRegistrationIsRefusedAtProposeTime is the MEDIUM gate.
//
// Both propose-time entry points ran their checks only `if err == nil` on the
// decode and otherwise FELL THROUGH to the broadcast, deferring to the apply-time
// decode error. That deferral is not a conservative choice: the decode is a pure
// function of the bytes the caller just sent, so a frame that fails it here can
// only ever fail on every replica of every group — after being appended to all
// NumShards logs on every node and replicated, then discarded as a classAdvance
// apply error. Nothing is learned and nothing is served; only the logs grow.
func TestUndecodableWASMRegistrationIsRefusedAtProposeTime(t *testing.T) {
	// name_len says 65535 with three bytes present: DecodeWASMRegistration fails at
	// its first bounds check.
	garbage := []byte{0xff, 0xff, 0x00}
	if _, err := ops.DecodeWASMRegistration(garbage); err == nil {
		t.Fatal("precondition: the payload must not decode")
	}

	// Reporting rather than failing fast: the log-growth assertion that follows each
	// call is the one that measures the AMPLIFICATION, and it is worth seeing even
	// when the refusal itself came from the wrong place.
	assertRefused := func(t *testing.T, err error) {
		t.Helper()
		if err == nil {
			t.Fatal("an undecodable payload was accepted")
		}
		if !errors.Is(err, ErrWASMRegistrationRefused) {
			t.Errorf("got %v, want ErrWASMRegistrationRefused", err)
		}
		if !strings.Contains(err.Error(), ops.WASMRegistrationRefusedMsg) {
			t.Errorf("refusal would be redacted to a generic internal error: %q", err.Error())
		}
	}

	t.Run("broadcast entry point", func(t *testing.T) {
		n := newTestNode(t, 4)
		waitAllApplied(t, n)
		before := lastLogIndexes(t, n)
		_, err := n.Call(wasmRegisterOpName, garbage)
		assertRefused(t, err)
		for i, after := range lastLogIndexes(t, n) {
			if after != before[i] {
				t.Errorf("shard %d log grew (%d -> %d): an undecodable payload was replicated into a group's log", i, before[i], after)
			}
		}
	})

	// The shard-scoped wrapper is the OTHER way into a group's log, and it is
	// dispatched off n.adminOps — populated only by the multi-node constructor,
	// which is also the only shape where the op is reachable.
	t.Run("shard-scoped entry point", func(t *testing.T) {
		n := newTestNodeMultiSingle(t, 4)
		waitAllApplied(t, n)
		before := lastLogIndexes(t, n)
		_, err := n.Call(opRegisterWASMShardName, encodeShardScopedWASM(1, garbage))
		assertRefused(t, err)
		for i, after := range lastLogIndexes(t, n) {
			if after != before[i] {
				t.Errorf("shard %d log grew (%d -> %d)", i, before[i], after)
			}
		}
	})
}

// TestOversizedEncodedWASMRegistrationIsRefusedBeforeTheDecode is the other half
// of the MEDIUM gate, and the more expensive half.
//
// maxDynamicWASMBytes is checked against a DECODED field (r.Bytes), so it bounded
// nothing at all for a frame that does not decode — and nothing below it does:
// server.MaxFrameSize admits 16 MiB and neither shard nor raft imposes a
// propose-side entry-size limit. So 16 MiB of garbage was accepted and appended to
// every group's log: NumShards defaults to 64 and validates up to 65536, i.e.
// ~1 GiB of replicated Raft log per attempt, repeatable by any admin-authenticated
// client, all of it discarded again at apply time.
//
// The payload here is BOTH over the cap and undecodable, which is what makes the
// ORDER observable: the refusal must name the frame size, not the decode failure.
// If the decode ran first the caller would get the decode error and the cap would
// still be doing nothing.
func TestOversizedEncodedWASMRegistrationIsRefusedBeforeTheDecode(t *testing.T) {
	// THE TWO ENTRY POINTS NOW HAVE TWO DIFFERENT CAPS, because they take two
	// different frames. The client edge carries the marker AND the module, so its
	// cap is maxWASMRegistrationRequestFrame; the shard-scoped leg carries the bare
	// MARKER that goes into a log, so its cap stays maxWASMRegistrationFrame. Each
	// oversized payload has to exceed the cap of the door it is pushed through, or
	// the frame check would not be the thing being measured.
	//
	// The filler byte matters. 0xff no longer works: with the module replaced by a
	// fixed-width 32-byte content address, a run of 0xff decodes as a 65535-byte
	// name followed by fields that all fit, so the payload would be VALID and the
	// ordering unobservable. 0x00 keeps the precondition true — it declares a
	// zero-length name and then a zero-length export and handle, which is a
	// perfectly short encoding, so a multi-megabyte 0x00 run fails the canonicality
	// check rather than decoding to itself. What the assertion needs is only that
	// the payload is not accepted for a reason OTHER than its size, and the loop
	// below proves the refusal names the size.
	oversizedRequest := bytes.Repeat([]byte{0x00}, maxWASMRegistrationRequestFrame+1)
	oversizedMarker := bytes.Repeat([]byte{0x00}, maxWASMRegistrationFrame+1)
	if r, err := ops.DecodeWASMRegistration(oversizedMarker); err == nil &&
		bytes.Equal(ops.EncodeWASMRegistration(r), oversizedMarker) {
		t.Fatal("precondition: the payload must not be a valid canonical marker, or the ordering is not observable")
	}
	if _, _, err := ops.DecodeWASMRegistrationRequest(oversizedRequest); err == nil {
		t.Fatal("precondition: the request payload must not decode, or the ordering is not observable")
	}

	// names is the fragment identifying WHICH size refusal fired. "encoded
	// registration is" alone no longer pins it: the canonicality refusal opens with
	// the same words, and the request cap says "encoded registration request is".
	// The SIZE refusal is the one that also names the limit it exceeded.
	assertRefusedBySize := func(t *testing.T, err error, names string) {
		t.Helper()
		if err == nil {
			t.Fatal("an oversized payload was accepted")
		}
		if !errors.Is(err, ErrWASMRegistrationRefused) {
			t.Errorf("got %v, want ErrWASMRegistrationRefused", err)
		}
		if !strings.Contains(err.Error(), names) || !strings.Contains(err.Error(), "over the") {
			t.Errorf("the refusal does not name the FRAME size, so the size cap did not run before the decode: %q", err.Error())
		}
		if !strings.Contains(err.Error(), ops.WASMRegistrationRefusedMsg) {
			t.Errorf("refusal would be redacted to a generic internal error: %q", err.Error())
		}
	}

	t.Run("broadcast entry point", func(t *testing.T) {
		n := newTestNode(t, 2)
		waitAllApplied(t, n)
		before := lastLogIndexes(t, n)
		_, err := n.Call(wasmRegisterOpName, oversizedRequest)
		assertRefusedBySize(t, err, "encoded registration request is")
		for i, after := range lastLogIndexes(t, n) {
			if after != before[i] {
				t.Errorf("shard %d log grew (%d -> %d): an oversized payload was replicated", i, before[i], after)
			}
		}
	})

	t.Run("shard-scoped entry point", func(t *testing.T) {
		n := newTestNodeMultiSingle(t, 2)
		waitAllApplied(t, n)
		before := lastLogIndexes(t, n)
		_, err := n.Call(opRegisterWASMShardName, encodeShardScopedWASM(1, oversizedMarker))
		assertRefusedBySize(t, err, "encoded registration is")
		for i, after := range lastLogIndexes(t, n) {
			if after != before[i] {
				t.Errorf("shard %d log grew (%d -> %d)", i, before[i], after)
			}
		}
	})

	// THE CASE THE FRAME CAP BOUNDS BUT DOES NOT CLOSE, which is why
	// checkWASMRegistrationArgs also requires the frame to be CANONICAL.
	//
	// ops.DecodeWASMRegistration never asserts that it CONSUMED its input — it reads
	// the fixed tail (maxfuel, epoch) at whatever offset it has reached and returns.
	// So a frame that is a valid registration followed by trailing junk DECODES
	// CLEANLY and the decode refusal does not see it. The leg that forwards a
	// broadcast to a peer proposes the RAW args — not a re-encode — so every byte of
	// that junk would be appended to that group's log and replicated, and the
	// registration would then apply perfectly well: nothing refused, nothing logged,
	// only the log grown. And because checkWASMUpdateGate fingerprints the DECODED
	// registration, padding changes no field, so a padded re-send of an
	// already-installed module compares EQUAL, takes the idempotent-retry path, and
	// is forwarded again with fresh padding — as often as the client likes.
	//
	// IT IS DRIVEN THROUGH THE SHARD-SCOPED ENTRY POINT NOW, because that is where
	// the hazard still lives. Thin markers closed it structurally at the CLIENT
	// edge: Node.Call decodes the request and re-encodes the marker
	// (ops.EncodeWASMRegistration(r)) rather than broadcasting the bytes it was
	// handed, so no padding a client sends can reach a log by that route. The
	// shard-scoped wrapper still forwards its raw marker verbatim into one group's
	// log, and it is reachable by any admin-authenticated external client, so
	// checkWASMRegistrationArgs' canonicality refusal is what stands there and is
	// what this subtest measures.
	//
	// The frame cap bounds ONE attempt; it does not refuse the padding. The padding
	// here is therefore deliberately far UNDER that cap, so this subtest measures the
	// canonicality refusal and cannot quietly fall back on the size refusal. The
	// original preconditions are kept for the same reason: the frame must still
	// decode, and it must stay under the frame cap, or one of the other checks is
	// what is being exercised.
	t.Run("a decodable frame padded with trailing junk under the frame cap", func(t *testing.T) {
		incr := readTestWASM(t, "../wasm/testdata/incr.wasm")
		valid := ops.EncodeWASMRegistration(ops.WASMRegistration{
			Name: "wasm_padded", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr),
			ExportName: "apply", Epoch: 1,
		})
		padded := append(valid, bytes.Repeat([]byte{0xff}, 64)...) //nolint:gocritic // a new slice is intended
		if _, err := ops.DecodeWASMRegistration(padded); err != nil {
			t.Fatalf("precondition: the padded frame must still DECODE, or it is caught by the decode refusal instead: %v", err)
		}
		if len(padded) > maxWASMRegistrationFrame {
			t.Fatalf("precondition: the padded frame is %d bytes, over the %d-byte frame cap, so the size refusal catches this instead",
				len(padded), maxWASMRegistrationFrame)
		}

		n := newTestNodeMultiSingle(t, 2)
		waitAllApplied(t, n)
		before := lastLogIndexes(t, n)
		_, err := n.Call(opRegisterWASMShardName, encodeShardScopedWASM(1, padded))
		if err == nil {
			t.Fatal("a padded registration was accepted")
		}
		if !errors.Is(err, ErrWASMRegistrationRefused) {
			t.Errorf("got %v, want ErrWASMRegistrationRefused", err)
		}
		if !strings.Contains(err.Error(), "not canonical") {
			t.Errorf("the padding was refused by something other than the canonicality check: %q", err.Error())
		}
		if !strings.Contains(err.Error(), ops.WASMRegistrationRefusedMsg) {
			t.Errorf("refusal would be redacted to a generic internal error: %q", err.Error())
		}
		for i, after := range lastLogIndexes(t, n) {
			if after != before[i] {
				t.Errorf("shard %d log grew (%d -> %d): %d bytes of trailing junk were replicated into the group's log behind a registration that decodes and applies perfectly well",
					i, before[i], after, len(padded)-len(valid))
			}
		}
		if files := wasmDirFiles(t, n.cfg.DataDir); len(files) != 0 {
			t.Errorf("the padded registration installed: %v", files)
		}
	})

	// The frame cap must NOT shadow the module cap: a well-formed frame carrying an
	// over-limit module has to keep being refused by the message that names the
	// MODULE size, which is the number an operator can act on. The module now rides
	// in the client-edge REQUEST rather than in the marker, so the cap it must not
	// be shadowed by is maxWASMRegistrationRequestFrame.
	t.Run("a well-formed over-limit module still names the module size", func(t *testing.T) {
		n := newTestNode(t, 2)
		waitAllApplied(t, n)
		module := bytes.Repeat([]byte{0}, maxDynamicWASMBytes+1)
		reg := ops.WASMRegistration{
			Name: "wasm_big", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(module),
			ExportName: "apply", Epoch: 1,
		}
		enc := ops.EncodeWASMRegistrationRequest(reg, module)
		if len(enc) > maxWASMRegistrationRequestFrame {
			t.Fatalf("the envelope slack is too small: a %d-byte module encodes to %d bytes, over the %d-byte request frame cap",
				len(module), len(enc), maxWASMRegistrationRequestFrame)
		}
		_, err := n.Call(wasmRegisterOpName, enc)
		if !errors.Is(err, ErrWASMRegistrationRefused) {
			t.Fatalf("got %v, want ErrWASMRegistrationRefused", err)
		}
		if !strings.Contains(err.Error(), "module is") {
			t.Errorf("the frame cap shadowed the module cap: %q", err.Error())
		}
	})
}

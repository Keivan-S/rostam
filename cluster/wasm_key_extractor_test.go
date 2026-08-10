// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/ops"
)

// TestWASMRegistrationsOfOneNameRouteIdentically is THE invariant gate for the
// silent cross-replica divergence this change exists to close, and it is
// deliberately the cheapest test in the package: no cluster, no goroutines.
//
// THE DEFECT IT STANDS AGAINST. The key extractor COMPUTES the shard group a
// proposal routes to (Node.Call: Lookup → ke → shardOf). The node-wide contract
// for an op is a fold over the set of registrations a node RECEIVED, and
// checkWASMUpdateGate's forwarded-leg gate makes that set node-dependent ON
// PURPOSE. So two FIRST-TIME registrations of one name declaring different
// extractors left two nodes routing INV(X) to DIFFERENT shard groups: different
// replica sets applied it, every apply SUCCEEDED, and there was no error to
// classify. A differing Kind fails closed (errPBApplyReadOnly is classFatal); the
// extractor had no backstop, and none could be built.
//
// THE PROPERTY. Any two registrations of one name route identically, for every
// args value.
//
// IT IS ASSERTED AGAINST THE EXTRACTOR THE INSTALL PATH ACTUALLY PRODUCED, pulled
// back out of the ops.Registry after applyWASMRegistration ran — not against
// ops.WASMKeyExtractor called directly. That distinction is the whole value of
// the test now that WASMRegistration has no handle field: "the struct cannot
// express two extractors" is a fact about the type, and a compiler enforces it.
// What a compiler does NOT enforce is that the install path keeps deriving the
// extractor from nothing. Someone reintroducing a per-registration choice — a new
// field, a lookup keyed on ExportName, a branch on Kind — would compile fine and
// fail here.
//
// It asserts over shardOf's OUTPUT, not extractor identity. Two different
// extractors that happened to agree on every key would be harmless; what is
// unsafe is precisely a disagreement about the destination group.
func TestWASMRegistrationsOfOneNameRouteIdentically(t *testing.T) {
	const name = "udf"
	incr, del := readIncrWASM(t), readDelWASM(t)

	// Two registrations of ONE name that differ in every field they still CAN
	// differ in — the module, the export name, the fuel cap, the epoch. Under the
	// old handle field a third difference was expressible, and it was the fatal
	// one.
	regs := []ops.WASMRegistration{
		{Name: name, Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr), ExportName: "apply", Epoch: 1},
		{Name: name, Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(del), ExportName: "apply", MaxFuel: 999, Epoch: 2},
	}

	// The args table is wide enough to separate the extractors that ever existed:
	// "std" reads [keyLen u16][key], the retired "raw" returned the whole slice,
	// and an unrecognised handle returned nil (shardless ⇒ group 0). Each row is a
	// frame on which those three disagree.
	argsTable := [][]byte{
		stdArgs([]byte("a")),
		stdArgs([]byte("route-0")),
		stdArgs([]byte("route-1")),
		stdArgs([]byte("a-much-longer-key-than-the-others")),
		append(stdArgs([]byte("keyed")), []byte("payload-that-must-not-affect-routing")...),
	}

	// Drive the REAL install path for each registration and take the extractor the
	// registry ended up holding.
	extractors := make([]ops.KeyExtractor, 0, len(regs))
	for i, r := range regs {
		dir := t.TempDir()
		rt, reg, st, err := restartWASM(t, dir, nil)
		if err != nil {
			t.Fatalf("registration %d: start: %v", i, err)
		}
		seedWASMBlob(t, dir, incr)
		seedWASMBlob(t, dir, del)
		if err := applyWASMRegistration(dir, rt, reg, st, r, 0, nil); err != nil {
			t.Fatalf("registration %d: apply: %v", i, err)
		}
		_, _, ke, ok := reg.Lookup(name)
		if !ok {
			t.Fatalf("registration %d: the op is not in the registry after a successful apply", i)
		}
		if ke == nil {
			t.Fatalf("registration %d installed a NIL key extractor, which registers the op SHARDLESS: every invocation would land in group 0 regardless of its key", i)
		}
		extractors = append(extractors, ke)
	}

	// The documented constant is the anchor: whatever the install path produced
	// must agree with ops.WASMKeyExtractorHandle's extractor, or the constant
	// clients are told to encode their args for is a lie.
	extractors = append(extractors, ops.KeyExtractorByHandle(ops.WASMKeyExtractorHandle))
	labels := []string{"registration 1 (incr)", "registration 2 (del, other fuel)", "ops.WASMKeyExtractorHandle"}

	const numShards = 64
	for i, ke1 := range extractors {
		for j, ke2 := range extractors {
			for _, args := range argsTable {
				k1, ok1 := ke1(args)
				k2, ok2 := ke2(args)
				if ok1 != ok2 {
					t.Fatalf("%s and %s disagree on whether args %q yield a key at all (%v vs %v)", labels[i], labels[j], args, ok1, ok2)
				}
				if !ok1 {
					continue
				}
				if g1, g2 := shardOf(k1, numShards), shardOf(k2, numShards); g1 != g2 {
					t.Fatalf("TWO REGISTRATIONS OF ONE NAME ROUTE TO DIFFERENT SHARD GROUPS.\n  %s → key %q → group %d\n  %s → key %q → group %d\n  args %q\nBoth installed successfully, so both can be a node's node-wide contract for the same op. Two nodes would propose INV of that op into different groups, different replica sets would apply it, every apply would SUCCEED, and nothing would report an error — the silent divergence ops.WASMKeyExtractorHandle exists to make unrepresentable.",
						labels[i], k1, g1, labels[j], k2, g2, args)
				}
			}
		}
	}
}

// TestWASMSidecarFromBeforeTheKeyExtractorPinRefusesToStart pins the ON-DISK
// BREAK, which is the half of the change an operator meets rather than a client.
//
// A version-2 sidecar is not detectably wrong by SHAPE: every field is present
// and it parses cleanly. What makes it unusable is its `key_extractor_handle`,
// written when "raw" and "" were legal — a value this build has no field to hold
// and would silently ignore, so the node would come up running a routing rule its
// peers no longer use. The version byte is the only thing that turns that into a
// loud refusal.
//
// The refusal must NAME the data dir as the problem and carry the recovery
// advice, because the remedy is an operator action (wipe and rejoin, or delete
// <dataDir>/wasm for a config-only deployment) and not a retry.
func TestWASMSidecarFromBeforeTheKeyExtractorPinRefusesToStart(t *testing.T) {
	incr := readIncrWASM(t)
	dir := t.TempDir()
	rt, reg, st, err := restartWASM(t, dir, nil)
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	r := ops.WASMRegistration{
		Name: "wasm_old", Kind: ops.OpReadWrite, Blob: ops.WASMBlobFingerprint(incr),
		ExportName: "apply", Epoch: 1,
	}
	seedWASMBlob(t, dir, incr)
	if err := applyWASMRegistration(dir, rt, reg, st, r, 0, nil); err != nil {
		t.Fatalf("seed: %v", err)
	}

	// Rewrite the sidecar as the pre-pin build left it: format version 2, carrying
	// the `key_extractor_handle` key that build wrote and this one has no field
	// for. Written through a map rather than wasmMeta precisely BECAUSE wasmMeta
	// can no longer express the field — which is also why the version byte, not a
	// value check, has to be what refuses it.
	path := filepath.Join(dir, "wasm", "wasm_old.json")
	b, err := os.ReadFile(path) //nolint:gosec // test-controlled path
	if err != nil {
		t.Fatalf("read sidecar: %v", err)
	}
	var raw map[string]any
	if err := json.Unmarshal(b, &raw); err != nil {
		t.Fatalf("unmarshal sidecar: %v", err)
	}
	// Hardcoded, NOT wasmMetaVersion-1. Deriving it from the constant makes this
	// pass against any version number at all — it would prove only "N-1 is
	// refused", never that the pin actually moved the version. 2 is the version a
	// pre-pin build wrote, and that is the artefact this has to refuse. (Same
	// standard as the snapshot test below, which hardcodes its version byte for
	// the same reason.)
	raw["version"] = 2
	raw["key_extractor_handle"] = "raw"
	out, err := json.Marshal(raw)
	if err != nil {
		t.Fatalf("marshal sidecar: %v", err)
	}
	if err := os.WriteFile(path, out, 0o600); err != nil {
		t.Fatalf("write sidecar: %v", err)
	}

	_, _, st2, err := restartWASM(t, dir, nil)
	if err == nil {
		in := st2.installed["wasm_old"]
		t.Fatalf("a pre-pin data dir started clean: the node reloaded a sidecar declaring the \"raw\" extractor and would route this op to a different group than every peer (replicated=%v, groups=%v)",
			in.replicated, sortedGroups(in.groups))
	}
	if !strings.Contains(err.Error(), "format version") {
		t.Errorf("the refusal does not identify the sidecar format as the problem: %v", err)
	}
	if !strings.Contains(err.Error(), wasmRecoveryAdvice) {
		t.Errorf("the refusal does not carry the operator recovery advice: %v", err)
	}
}

// TestWASMSnapshotFromBeforeTheKeyExtractorPinIsRefused is the same on-disk break
// on the OTHER carrier, and it is the one that could not have been left alone.
//
// wasmSnapshotBlob's records are POSITIONAL bytes. Dropping
// [handle_len u16][handle] from ops.EncodeWASMRegistration means a version-3
// record read under version-4 rules takes the handle's length prefix as the first
// two bytes of MaxFuel and slides every later field — it DECODES, to nonsense,
// rather than failing. A restore would then install ops with garbage fuel caps
// and epochs, and epochs order the convergence rule.
//
// IT BUILDS A REAL PRE-PIN RECORD rather than stamping an older byte on a current
// one. Decrementing the version constant would make that trick pass against any
// version number at all, proving only "N-1 is refused" and nothing about whether
// the constant moved for THIS change. A record encoded the OLD way is the actual
// artefact a pre-pin peer serves.
//
// The first half of the test is the JUSTIFICATION and has to come first: it shows
// the current decoder consuming that record WITHOUT error and producing the wrong
// MaxFuel and Epoch. That is what makes the version byte load-bearing — if the
// old record simply failed to decode, no bump would have been needed.
func TestWASMSnapshotFromBeforeTheKeyExtractorPinIsRefused(t *testing.T) {
	// A registration as a pre-pin build encoded it: the current field order with
	// [handle_len u16][handle] still sitting between the export name and MaxFuel.
	const wantFuel, wantEpoch = 4096, 7
	var rec []byte
	putU16 := func(n int) {
		rec = append(rec, byte(n>>8), byte(n))
	}
	putU64 := func(n uint64) {
		for shift := 56; shift >= 0; shift -= 8 {
			rec = append(rec, byte(n>>uint(shift)))
		}
	}
	blob := ops.WASMBlobFingerprint(readIncrWASM(t))
	putU16(len("wasm_old"))
	rec = append(rec, "wasm_old"...)
	rec = append(rec, byte(ops.OpReadWrite))
	rec = append(rec, blob[:]...)
	putU16(len("apply"))
	rec = append(rec, "apply"...)
	putU16(len("std")) // the field this build no longer has
	rec = append(rec, "std"...)
	putU64(wantFuel)
	putU64(wantEpoch)

	// THE JUSTIFICATION. The current decoder eats that record happily and gets it
	// wrong, which is precisely the case wasmSnapshotBlobVersion's comment says
	// must move the version byte.
	got, err := ops.DecodeWASMRegistration(rec)
	if err != nil {
		t.Skipf("a pre-pin record no longer decodes at all (%v); the version byte is belt-and-braces here rather than load-bearing, and this test has nothing to prove", err)
	}
	if got.MaxFuel == wantFuel && got.Epoch == wantEpoch {
		t.Fatalf("precondition: a pre-pin record decoded CORRECTLY under the current rules (fuel=%d epoch=%d), so the record shape did not actually change and this test is vacuous",
			got.MaxFuel, got.Epoch)
	}
	t.Logf("a pre-pin record decodes silently to the wrong registration: fuel=%d (want %d), epoch=%d (want %d) — this is what the version byte has to stop",
		got.MaxFuel, wantFuel, got.Epoch, wantEpoch)

	// Now the gate: that record, in a snapshot section a pre-pin node would serve.
	var blobSection []byte
	blobSection = append(blobSection, 3) // the pre-pin wasmSnapshotBlobVersion
	appendSection := func(n int) {
		blobSection = append(blobSection, byte(n>>24), byte(n>>16), byte(n>>8), byte(n))
	}
	appendSection(1) // one install record
	appendSection(len(rec))
	blobSection = append(blobSection, rec...)
	appendSection(0) // no bindings

	if _, err := decodeWASMSnapshotBlob(blobSection); err == nil {
		t.Fatal("a pre-pin snapshot section was accepted: its records carry an extra [handle_len u16][handle] pair, so every field after the export name is read at the wrong offset — the restore would install ops with garbage fuel caps and epochs, and epochs order the convergence rule")
	} else if !strings.Contains(err.Error(), "version") {
		t.Errorf("the refusal does not identify the blob version as the problem: %v", err)
	}
}

// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"
	"strings"
	"testing"
)

func TestWASMRegistrationRoundtrip(t *testing.T) {
	orig := WASMRegistration{
		Name:       "my_udf",
		Kind:       OpReadWrite,
		Blob:       WASMBlobFingerprint([]byte{0x00, 0x61, 0x73, 0x6d}), // wasm magic
		ExportName: "apply",
		MaxFuel:    1_000_000,
	}
	encoded := EncodeWASMRegistration(orig)
	got, err := DecodeWASMRegistration(encoded)
	if err != nil {
		t.Fatalf("DecodeWASMRegistration: %v", err)
	}
	if got.Name != orig.Name {
		t.Errorf("Name = %q, want %q", got.Name, orig.Name)
	}
	if got.Kind != orig.Kind {
		t.Errorf("Kind = %v, want %v", got.Kind, orig.Kind)
	}
	if got.Blob != orig.Blob {
		t.Errorf("Blob = %x, want %x", got.Blob, orig.Blob)
	}
	if got.ExportName != orig.ExportName {
		t.Errorf("ExportName = %q, want %q", got.ExportName, orig.ExportName)
	}
	if got.MaxFuel != orig.MaxFuel {
		t.Errorf("MaxFuel = %d, want %d", got.MaxFuel, orig.MaxFuel)
	}
	// The Blob is a VALUE ARRAY, so there is nothing left to alias: the old
	// version of this check existed because Bytes was a slice that could have
	// pointed into the encoded buffer. Mutating the buffer must still leave the
	// already-decoded registration untouched.
	want := got.Blob
	encoded[5] ^= 0xFF
	if got.Blob != want {
		t.Error("the decoded registration must not alias the encoded buffer")
	}
}

func TestDecodeWASMRegistrationTruncatedRejects(t *testing.T) {
	_, err := DecodeWASMRegistration([]byte{0x01})
	if err == nil {
		t.Fatal("expected error for truncated input, got nil")
	}
}

func TestRegisterWASMRegisterOpInvokesHook(t *testing.T) {
	reg := NewRegistry()

	var capturedName string
	capturedShard := -99
	hook := func(shardIdx int, r WASMRegistration) error {
		capturedName = r.Name
		capturedShard = shardIdx
		return nil
	}

	if err := RegisterWASMRegisterOp(reg, hook); err != nil {
		t.Fatalf("RegisterWASMRegisterOp: %v", err)
	}

	h, kind, _, ok := reg.Lookup("__register_wasm__")
	if !ok {
		t.Fatal("__register_wasm__ not found in registry")
	}
	if kind != OpReadWrite {
		t.Errorf("kind = %v, want OpReadWrite", kind)
	}

	payload := EncodeWASMRegistration(WASMRegistration{
		Name:       "my_udf",
		Kind:       OpReadWrite,
		Blob:       WASMBlobFingerprint([]byte{0x00, 0x61, 0x73, 0x6d}),
		ExportName: "apply",
		MaxFuel:    500_000,
	})

	res, err := h(nil, payload)
	if err != nil {
		t.Fatalf("handler returned error: %v", err)
	}
	if res != nil {
		t.Errorf("handler result = %v, want nil", res)
	}
	if capturedName != "my_udf" {
		t.Errorf("hook captured Name = %q, want my_udf", capturedName)
	}
	// A nil TxContext has no dispatcher behind it, so there is no shard group to
	// attribute the apply to. It must report NoShardIndex, never 0: the cluster
	// layer turns this value into "group g's log carries the registration", and a
	// false claim about group 0 is exactly the evidence the route gate must not
	// be handed.
	if capturedShard != NoShardIndex {
		t.Errorf("hook captured shardIdx = %d, want NoShardIndex (%d)", capturedShard, NoShardIndex)
	}
}

func TestRegisterWASMRegisterOpRejectsEmptyName(t *testing.T) {
	reg := NewRegistry()
	hook := func(_ int, _ WASMRegistration) error { return nil }
	if err := RegisterWASMRegisterOp(reg, hook); err != nil {
		t.Fatalf("RegisterWASMRegisterOp: %v", err)
	}

	h, _, _, _ := reg.Lookup("__register_wasm__")

	payload := EncodeWASMRegistration(WASMRegistration{
		Name:    "",
		Kind:    OpReadWrite,
		Blob:    WASMBlobFingerprint([]byte{0x00, 0x61, 0x73, 0x6d}),
		MaxFuel: 1,
	})

	_, err := h(nil, payload)
	if err == nil {
		t.Fatal("expected error for empty Name, got nil")
	}
}

// TestValidateWASMOpName pins the rule that keeps a wire-controlled op name from
// becoming a wire-controlled filesystem path.
//
// Every PATH-SHAPED name below is accepted by validateEntry — which checks only
// emptiness, maxOpNameLen and the protocol-v2 length-2 collision — and every one
// of them RESOLVES rather than fails when passed to filepath.Join, which is what
// made it a write primitive rather than an error.
//
// The last two cases are validateEntry's OWN rules, which this function now
// carries as well. They are here because validateEntry is reached far too late:
// cluster.applyWASMRegistration does not call into the registry until after it has
// written <Name>.wasm and its sidecar, so a name that only validateEntry rejects
// left both files on every node and then failed node construction on the next
// restart. See ValidateWASMOpName.
func TestValidateWASMOpName(t *testing.T) {
	bad := []string{
		"",
		".",
		"..",
		"../pwned",
		"../../../../etc/cron.d/rostam",
		"sub/mod",
		`..\pwned`,
		"a/../../b",
		"/absolute",
		"trailing/",
		"nul\x00byte",
		"dots..inside",
		// A Windows volume separator. Rejected on every GOOS, because the predicate
		// that used to (accidentally) catch it — filepath.Base(name) != name — is
		// GOOS-dependent: it strips a drive prefix on Windows and nothing on Linux, so
		// a heterogeneous cluster would have refused this entry on some replicas and
		// accepted it on others, at APPLY time.
		"a:b",
		// validateEntry's rules, hoisted in front of the file writes.
		"ab",
		strings.Repeat("x", 256),
	}
	for _, name := range bad {
		if err := ValidateWASMOpName(name); err == nil {
			t.Errorf("ValidateWASMOpName(%q) = nil; it would be joined onto the data dir as a path", name)
		} else if !errors.Is(err, ErrWASMOpNameUnsafe) {
			t.Errorf("ValidateWASMOpName(%q) = %v, want ErrWASMOpNameUnsafe", name, err)
		}
	}

	// The rule must not reject anything a real deployment uses. If it did, the
	// authoritative apply-time check would refuse registrations the propose side
	// had already accepted on older nodes.
	good := []string{"my_udf", "wasm_incr", "wasm-incr", "incr.v2", "__register_wasm__", "a.b.c", "UDF_2024"}
	for _, name := range good {
		if err := ValidateWASMOpName(name); err != nil {
			t.Errorf("ValidateWASMOpName(%q) = %v, want nil", name, err)
		}
	}
}

// TestRegisterWASMRegisterOpRejectsUnsafeName proves the check is enforced at
// APPLY time, not only at propose time. This is the path a log entry replayed on
// a replica takes: the replica never ran the propose-time check, so without this
// one the name reaches a filepath.Join unvalidated.
//
// IT IS NOT THE ONLY APPLY-SIDE CHECK, and an earlier version of this comment
// claiming it was "the only path a snapshot-restored or peer-forwarded
// registration is guaranteed to pass through" was wrong in a way that argued for
// deleting a check that must not be deleted. A snapshot restore does NOT pass
// through this op at all — cluster.installWASMSnapshotBlobLocked decodes the blob
// and calls cluster.applyWASMRegistration directly — which is precisely why
// applyWASMRegistration runs ops.ValidateWASMOpName itself. The two checks are
// both load-bearing and cover disjoint routes; neither is redundant with the
// other.
func TestRegisterWASMRegisterOpRejectsUnsafeName(t *testing.T) {
	reg := NewRegistry()
	called := false
	if err := RegisterWASMRegisterOp(reg, func(_ int, _ WASMRegistration) error {
		called = true
		return nil
	}); err != nil {
		t.Fatalf("RegisterWASMRegisterOp: %v", err)
	}
	h, _, _, _ := reg.Lookup("__register_wasm__")

	payload := EncodeWASMRegistration(WASMRegistration{
		Name: "../../../../etc/cron.d/rostam",
		Kind: OpReadWrite,
		Blob: WASMBlobFingerprint([]byte{0x00, 0x61, 0x73, 0x6d}),
	})
	_, err := h(nil, payload)
	if !errors.Is(err, ErrWASMOpNameUnsafe) {
		t.Fatalf("apply of a traversing name: got %v, want ErrWASMOpNameUnsafe", err)
	}
	if called {
		t.Error("the hook ran: the name reached the code that turns it into a file path")
	}
}

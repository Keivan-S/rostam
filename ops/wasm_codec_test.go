// SPDX-License-Identifier: Apache-2.0

package ops

import "testing"

// TestWASMRegistrationRoundTrip covers every field, Epoch included: the epoch is
// the ordering key of the convergence rule, so an encoding that dropped it would
// silently reduce every registration to "epoch 0" and leave the winner decided
// by the fingerprint tiebreak alone.
func TestWASMRegistrationRoundTrip(t *testing.T) {
	want := WASMRegistration{
		Name:       "udf",
		Kind:       OpReadWrite,
		Blob:       WASMBlobFingerprint([]byte{0x00, 0x61, 0x73, 0x6d}),
		ExportName: "apply",
		MaxFuel:    12345,
		Epoch:      9,
	}
	got, err := DecodeWASMRegistration(EncodeWASMRegistration(want))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.Name != want.Name || got.Kind != want.Kind || got.ExportName != want.ExportName ||
		got.MaxFuel != want.MaxFuel || got.Epoch != want.Epoch {
		t.Fatalf("round trip = %+v, want %+v", got, want)
	}
	if got.Blob != want.Blob {
		t.Fatalf("blob = %x, want %x", got.Blob, want.Blob)
	}
}

// TestWASMRegistrationDecodeTruncated pins that a payload cut short at the new
// trailing epoch field is rejected rather than silently decoding to epoch 0 —
// which would look like a valid, lowest-priority registration.
func TestWASMRegistrationDecodeTruncated(t *testing.T) {
	full := EncodeWASMRegistration(WASMRegistration{
		Name: "udf", Kind: OpReadWrite, Blob: WASMBlobFingerprint([]byte{1, 2, 3}), ExportName: "apply", Epoch: 4,
	})
	if _, err := DecodeWASMRegistration(full[:len(full)-1]); err == nil {
		t.Fatal("a payload truncated inside the epoch field must be rejected")
	}
}

// TestWASMRegistrationFingerprintCoversRouting pins that the fingerprint covers
// the field the wasm.Runtime's own fingerprint does NOT — Kind. It decides
// whether the op bypasses Raft, so two registrations differing only there are
// genuinely different and must not compare equal (they would otherwise be
// indistinguishable to the convergence rule and to the on-disk sidecar check).
//
// THE OTHER ROUTING FIELD IS GONE. KeyExtractorHandle used to be checked here
// too, because it decided which shard group an op's invocations landed in. That
// is exactly why it could not stay a field: see WASMKeyExtractorHandle. With one
// extractor for every WASM op there is nothing left for a fingerprint to
// distinguish.
func TestWASMRegistrationFingerprintCoversRouting(t *testing.T) {
	base := WASMRegistration{Name: "udf", Kind: OpReadWrite, Blob: WASMBlobFingerprint([]byte{1, 2, 3}), ExportName: "apply"}

	readOnly := base
	readOnly.Kind = OpReadOnly
	if WASMRegistrationFingerprint(base) == WASMRegistrationFingerprint(readOnly) {
		t.Error("Kind must change the fingerprint: it decides whether the op goes through Raft at all")
	}

	// Epoch is the ORDERING key, not content: two registrations of the same
	// module at different epochs must share a fingerprint, or the tiebreak would
	// never actually be a tiebreak.
	bumped := base
	bumped.Epoch = 42
	if WASMRegistrationFingerprint(base) != WASMRegistrationFingerprint(bumped) {
		t.Error("Epoch must NOT affect the fingerprint")
	}
}

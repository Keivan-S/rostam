// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"reflect"
	"testing"
)

// TestDecodeVectorInsertArgsKeyExpiresIntoMatches confirms the Into decoder
// returns the same dense vector as the allocating public decoder (both for a
// reused dst that fits and a nil dst that falls back to a fresh allocation).
func TestDecodeVectorInsertArgsKeyExpiresIntoMatches(t *testing.T) {
	want := []float32{1.5, -2, 3.25, 4}
	args := EncodeVectorInsertArgs("docs", 7, want)

	_, _, refVec, _, _, _, _, err := DecodeVectorInsertArgs(args)
	if err != nil {
		t.Fatalf("DecodeVectorInsertArgs: %v", err)
	}

	// Reused dst with capacity >= dim → vec aliases dst, bytes still match.
	dst := make([]float32, 0, len(want))
	_, _, gotVec, _, _, _, _, _, _, _, _, err := DecodeVectorInsertArgsKeyExpiresInto(dst[:0], args)
	if err != nil {
		t.Fatalf("DecodeVectorInsertArgsKeyExpiresInto: %v", err)
	}
	if !reflect.DeepEqual(gotVec, refVec) || !reflect.DeepEqual(gotVec, want) {
		t.Fatalf("Into(dst) vec = %v, want %v", gotVec, want)
	}

	// nil dst → fresh allocation, identical bytes (byte-for-byte legacy path).
	_, _, nilVec, _, _, _, _, _, _, _, _, err := DecodeVectorInsertArgsKeyExpiresInto(nil, args)
	if err != nil {
		t.Fatalf("DecodeVectorInsertArgsKeyExpiresInto(nil): %v", err)
	}
	if !reflect.DeepEqual(nilVec, want) {
		t.Fatalf("Into(nil) vec = %v, want %v", nilVec, want)
	}
}

// TestDecodeVectorInsertArgsIntoAllocFree asserts the Into decoder with a REUSED
// dst buffer does not allocate the dense []float32 per call, while the allocating
// public decoder does — i.e. the pooled single-insert decode path drops at least
// one allocation (the dense buffer) relative to the legacy decode.
func TestDecodeVectorInsertArgsIntoAllocFree(t *testing.T) {
	vec := make([]float32, 128)
	for i := range vec {
		vec[i] = float32(i)
	}
	args := EncodeVectorInsertArgs("docs", 7, vec)

	dst := make([]float32, 0, len(vec)) // reused, cap >= dim
	into := testing.AllocsPerRun(200, func() {
		_, _, dst, _, _, _, _, _, _, _, _, _ = DecodeVectorInsertArgsKeyExpiresInto(dst[:0], args)
	})
	legacy := testing.AllocsPerRun(200, func() {
		_, _, _, _, _, _, _, _ = DecodeVectorInsertArgs(args)
	})
	t.Logf("decode allocs/op: Into(reused dst) = %.1f, legacy = %.1f", into, legacy)
	if into >= legacy {
		t.Fatalf("Into decode allocs/op = %.1f, legacy = %.1f; want Into strictly fewer (dense buffer elided)", into, legacy)
	}
	if legacy-into < 1 {
		t.Fatalf("expected the Into path to save at least the dense []float32 alloc; saved %.1f", legacy-into)
	}
}

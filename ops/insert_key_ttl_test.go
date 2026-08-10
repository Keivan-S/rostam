// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/vector"
)

// TestInsertKeyTTLByteIdenticalWhenAbsent: the per-key-TTL encoder is
// BYTE-IDENTICAL to the legacy encoder when the key TTL map is nil/empty — the
// vecFlagKeyTTL bit stays unset and NO trailing bytes are appended (zero overhead
// for the no-key_ttl path). Also covers the upsert + CAS wrappers.
func TestInsertKeyTTLByteIdenticalWhenAbsent(t *testing.T) {
	col, id := "docs", uint64(42)
	vec := []float32{1.5, -2.5, 3.5, 4.5}
	meta := vector.Metadata{"k": vector.NewInt(7)}
	sv := vector.SparseVector{Indices: []uint32{0, 3}, Values: []float32{1, 2}}

	cases := []struct {
		name      string
		base      []byte
		withKeyTL []byte
	}{
		{
			"insert-ext-vs-keyttl-nil",
			EncodeVectorInsertArgsExt(col, id, vec, 5*time.Second, meta, sv),
			EncodeVectorInsertArgsKeyTTL(col, id, vec, 5*time.Second, meta, sv, nil),
		},
		{
			"insert-ext-vs-keyttl-empty",
			EncodeVectorInsertArgsExt(col, id, vec, 5*time.Second, meta, sv),
			EncodeVectorInsertArgsKeyTTL(col, id, vec, 5*time.Second, meta, sv, map[string]int64{}),
		},
		{
			"insert-cas-vs-caskeyttl-nil",
			EncodeVectorInsertArgsCAS(col, id, vec, 0, meta, sv, 9, true),
			EncodeVectorInsertArgsCASKeyTTL(col, id, vec, 0, meta, sv, 9, true, nil),
		},
		{
			"upsert-vs-upsertkeyttl-nil",
			EncodeVectorUpsertArgs(col, id, vec, "body", 0, meta, sv),
			EncodeVectorUpsertArgsCASKeyTTL(col, id, vec, "body", 0, meta, sv, 0, false, nil),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !bytes.Equal(tc.base, tc.withKeyTL) {
				t.Fatalf("not byte-identical:\n base=%x\n keyttl=%x", tc.base, tc.withKeyTL)
			}
		})
	}
}

// TestInsertKeyTTLCodecRoundTrip: a present key TTL map round-trips through the
// encoder + DecodeVectorInsertArgsKeyTTL, and the legacy DecodeVectorInsertArgsCAS
// still decodes the rest of the record (the keyTTL block is self-delimiting and
// ignored by the older decoder).
func TestInsertKeyTTLCodecRoundTrip(t *testing.T) {
	col, id := "docs", uint64(7)
	vec := []float32{1, 2, 3}
	meta := vector.Metadata{"a": vector.NewInt(1)}
	keyTTL := map[string]int64{"a": 1000, "b": 2000}

	args := EncodeVectorInsertArgsKeyTTL(col, id, vec, time.Second, meta, vector.SparseVector{}, keyTTL)

	gotCol, gotID, gotVec, ttl, gotMeta, _, version, exp, hasExp, gotKeyTTL, err := DecodeVectorInsertArgsKeyTTL(args)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if gotCol != col || gotID != id || len(gotVec) != 3 || ttl != time.Second {
		t.Fatalf("decoded col/id/vec/ttl = %q/%d/%v/%v", gotCol, gotID, gotVec, ttl)
	}
	if version != 0 || hasExp || exp != 0 {
		t.Fatalf("unexpected version/cas: version=%d hasExp=%v exp=%d", version, hasExp, exp)
	}
	if gotMeta["a"].Int != 1 {
		t.Fatalf("decoded meta = %v", gotMeta)
	}
	if len(gotKeyTTL) != 2 || gotKeyTTL["a"] != 1000 || gotKeyTTL["b"] != 2000 {
		t.Fatalf("decoded key_ttl_ms = %v, want {a:1000,b:2000}", gotKeyTTL)
	}

	// The legacy CAS decoder still reads everything BEFORE the keyTTL block.
	lCol, lID, _, lTTL, lMeta, _, _, _, _, err := DecodeVectorInsertArgsCAS(args)
	if err != nil {
		t.Fatalf("legacy decode: %v", err)
	}
	if lCol != col || lID != id || lTTL != time.Second || lMeta["a"].Int != 1 {
		t.Fatalf("legacy decode mismatch: %q/%d/%v/%v", lCol, lID, lTTL, lMeta)
	}
}

// TestInsertKeyTTLCoexistsWithCAS: CAS (expectedVersion) and the key TTL block
// coexist on the same insert — the keyTTL block rides AFTER expectedVersion, so
// both trailers decode correctly (trailer order matches enc/dec).
func TestInsertKeyTTLCoexistsWithCAS(t *testing.T) {
	col, id := "docs", uint64(3)
	vec := []float32{1, 0}
	meta := vector.Metadata{"x": vector.NewInt(5)}
	keyTTL := map[string]int64{"x": 500}

	args := EncodeVectorInsertArgsCASKeyTTL(col, id, vec, 0, meta, vector.SparseVector{}, 11, true, keyTTL)

	_, _, _, _, gotMeta, _, _, exp, hasExp, gotKeyTTL, err := DecodeVectorInsertArgsKeyTTL(args)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !hasExp || exp != 11 {
		t.Fatalf("CAS not decoded: hasExp=%v exp=%d", hasExp, exp)
	}
	if gotMeta["x"].Int != 5 {
		t.Fatalf("meta = %v", gotMeta)
	}
	if gotKeyTTL["x"] != 500 {
		t.Fatalf("key_ttl_ms = %v, want {x:500}", gotKeyTTL)
	}
}

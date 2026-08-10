// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

func opsSV(pairs ...float32) *vector.SparseVector {
	n := len(pairs) / 2
	out := &vector.SparseVector{Indices: make([]uint32, n), Values: make([]float32, n)}
	for i := 0; i < n; i++ {
		out.Indices[i] = uint32(pairs[2*i])
		out.Values[i] = pairs[2*i+1]
	}
	return out
}

// TestMVAddDenseOnlyWireByteIdentical pins the byte-compat invariant: an MV add with
// no CAS, no keyTTL, and NO sparse encodes IDENTICALLY whether routed through the
// legacy EncodeMVAddArgs or the new sparse-aware encoder with a nil sparse.
func TestMVAddDenseOnlyWireByteIdentical(t *testing.T) {
	name := "mv"
	tokens := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}
	meta := vector.Metadata{"k": vector.NewInt(1)}

	legacy := EncodeMVAddArgs(name, 7, tokens, meta)
	viaCAS := EncodeMVAddArgsCASKeyTTL(name, 7, tokens, meta, 0, false, nil)
	viaSparse := EncodeMVAddArgsCASKeyTTLSparse(name, 7, tokens, meta, 0, false, nil, nil)
	if !bytes.Equal(legacy, viaCAS) {
		t.Fatalf("CASKeyTTL(no trailers) != legacy: %d vs %d bytes", len(viaCAS), len(legacy))
	}
	if !bytes.Equal(legacy, viaSparse) {
		t.Fatalf("CASKeyTTLSparse(nil sparse) != legacy: %d vs %d bytes", len(viaSparse), len(legacy))
	}

	// Versioned family: version 0, no keyExpires, nil sparse == legacy.
	legacyV := EncodeMVAddArgs(name, 7, tokens, meta)
	viaVerSparse := EncodeMVAddArgsVersionedKeyExpiresSparse(name, 7, tokens, meta, 0, nil, nil)
	if !bytes.Equal(legacyV, viaVerSparse) {
		t.Fatalf("VersionedKeyExpiresSparse(nil) != legacy: %d vs %d bytes", len(viaVerSparse), len(legacyV))
	}
}

// TestMVAddSparseTrailerRoundTrip checks the doc-sparse trailer round-trips through
// the CASKeyTTL family alone and combined with CAS + keyTTL.
func TestMVAddSparseTrailerRoundTrip(t *testing.T) {
	tokens := [][]float32{{1, 0, 0, 0}}
	meta := vector.Metadata{"k": vector.NewInt(1)}
	sp := opsSV(0, 1.0, 3, 2.5)

	cases := []struct {
		name        string
		exp         uint64
		hasExp      bool
		keyTTL      map[string]int64
		wantExp     uint64
		wantHasExp  bool
		wantKeyTTLN int
	}{
		{name: "sparse only", wantHasExp: false},
		{name: "sparse+cas", exp: 9, hasExp: true, wantExp: 9, wantHasExp: true},
		{name: "sparse+keyttl", keyTTL: map[string]int64{"a": 1000}, wantKeyTTLN: 1},
		{name: "sparse+cas+keyttl", exp: 4, hasExp: true, keyTTL: map[string]int64{"a": 5}, wantExp: 4, wantHasExp: true, wantKeyTTLN: 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			enc := EncodeMVAddArgsCASKeyTTLSparse("mv", 7, tokens, meta, tc.exp, tc.hasExp, tc.keyTTL, sp)
			gotName, gotID, gotTok, gotMeta, gotExp, gotHasExp, gotKT, gotSparse, err := DecodeMVAddArgsCASKeyTTLSparse(enc)
			if err != nil {
				t.Fatal(err)
			}
			if gotName != "mv" || gotID != 7 || len(gotTok) != 1 || gotMeta["k"].Int != 1 {
				t.Fatalf("base fields wrong: %s %d %+v %+v", gotName, gotID, gotTok, gotMeta)
			}
			if gotExp != tc.wantExp || gotHasExp != tc.wantHasExp {
				t.Fatalf("cas = (%d,%v), want (%d,%v)", gotExp, gotHasExp, tc.wantExp, tc.wantHasExp)
			}
			if len(gotKT) != tc.wantKeyTTLN {
				t.Fatalf("keyTTL len = %d, want %d", len(gotKT), tc.wantKeyTTLN)
			}
			if gotSparse == nil || len(gotSparse.Indices) != 2 || gotSparse.Indices[1] != 3 || gotSparse.Values[1] != 2.5 {
				t.Fatalf("sparse round-trip = %+v", gotSparse)
			}
			// The legacy decoder (no sparse) still works on the sparse-bearing wire.
			_, _, _, _, lExp, lHasExp, lKT, lerr := DecodeMVAddArgsCASKeyTTL(enc)
			if lerr != nil {
				t.Fatalf("legacy decode of sparse wire: %v", lerr)
			}
			if lExp != tc.wantExp || lHasExp != tc.wantHasExp || len(lKT) != tc.wantKeyTTLN {
				t.Fatalf("legacy decode lost trailers: cas(%d,%v) kt%d", lExp, lHasExp, len(lKT))
			}
		})
	}
}

// TestMVAddVersionedSparseTrailerRoundTrip checks the versioned/keyExpires family
// carries the doc sparse alone and with version + keyExpires.
func TestMVAddVersionedSparseTrailerRoundTrip(t *testing.T) {
	tokens := [][]float32{{1, 0, 0, 0}}
	meta := vector.Metadata{"k": vector.NewInt(2)}
	sp := opsSV(2, 4.0)

	cases := []struct {
		name    string
		version uint64
		ke      map[string]uint64
	}{
		{name: "sparse only", version: 0},
		{name: "sparse+version", version: 11},
		{name: "sparse+version+ke", version: 11, ke: map[string]uint64{"a": 123456}},
		{name: "sparse+ke (v0)", version: 0, ke: map[string]uint64{"b": 999}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			enc := EncodeMVAddArgsVersionedKeyExpiresSparse("mv", 5, tokens, meta, tc.version, tc.ke, sp)
			n, id, tok, md, v, ke, gotSparse, err := DecodeMVAddArgsVersionedKeyExpiresSparse(enc)
			if err != nil {
				t.Fatal(err)
			}
			if n != "mv" || id != 5 || len(tok) != 1 || md["k"].Int != 2 {
				t.Fatalf("base fields wrong: %s %d", n, id)
			}
			if v != tc.version {
				t.Fatalf("version = %d, want %d", v, tc.version)
			}
			if len(ke) != len(tc.ke) {
				t.Fatalf("keyExpires len = %d, want %d", len(ke), len(tc.ke))
			}
			if gotSparse == nil || len(gotSparse.Indices) != 1 || gotSparse.Indices[0] != 2 || gotSparse.Values[0] != 4.0 {
				t.Fatalf("sparse round-trip = %+v", gotSparse)
			}
			// Legacy versioned decoder still reads version + keyExpires off the sparse wire.
			_, _, _, _, lv, lke, lerr := DecodeMVAddArgsVersionedKeyExpires(enc)
			if lerr != nil {
				t.Fatalf("legacy versioned decode: %v", lerr)
			}
			if lv != tc.version || len(lke) != len(tc.ke) {
				t.Fatalf("legacy versioned decode lost data: v=%d ke=%d", lv, len(lke))
			}
		})
	}
}

// TestMVAddLegacyWireDecodesNilSparse confirms an old (pre-sparse) MV add wire
// decodes to a nil sparse on the sparse-aware decoders (backward compatibility).
func TestMVAddLegacyWireDecodesNilSparse(t *testing.T) {
	tokens := [][]float32{{1, 0, 0, 0}}
	meta := vector.Metadata{"k": vector.NewInt(1)}

	legacyCAS := EncodeMVAddArgsCAS("mv", 1, tokens, meta, 3, true)
	_, _, _, _, _, _, _, sparse, err := DecodeMVAddArgsCASKeyTTLSparse(legacyCAS)
	if err != nil {
		t.Fatal(err)
	}
	if sparse != nil {
		t.Fatalf("legacy CAS wire decoded a non-nil sparse: %+v", sparse)
	}

	legacyVer := EncodeMVAddArgsVersioned("mv", 1, tokens, meta, 9)
	_, _, _, _, _, _, sparse2, err := DecodeMVAddArgsVersionedKeyExpiresSparse(legacyVer)
	if err != nil {
		t.Fatal(err)
	}
	if sparse2 != nil {
		t.Fatalf("legacy versioned wire decoded a non-nil sparse: %+v", sparse2)
	}
}

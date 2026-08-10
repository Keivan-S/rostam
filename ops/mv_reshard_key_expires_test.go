// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// TestMVScanResultKeyExpiresRoundTrip: the MV scan codec carries the per-record
// ABSOLUTE per-key payload TTL map (KeyExpires) verbatim, alongside tokens/meta/
// version. Deadlines are absolute unix-ms, decoded byte-for-byte.
func TestMVScanResultKeyExpiresRoundTrip(t *testing.T) {
	recs := []vector.MultiScanRecord{
		{
			ID:         7,
			Tokens:     [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}},
			Metadata:   vector.Metadata{"src": vector.NewString("a")},
			Version:    3,
			KeyExpires: map[string]uint64{"temp": 1_700_000_123_456, "other": 42},
		},
		{ID: 8, Tokens: [][]float32{{0, 0, 0, 0}}, Version: 1}, // no per-key TTL
	}
	got, err := DecodeMVScanResult(EncodeMVScanResult(recs))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("decoded %d records, want 2", len(got))
	}
	if got[0].Version != 3 {
		t.Fatalf("record 0 version = %d, want 3", got[0].Version)
	}
	if len(got[0].KeyExpires) != 2 || got[0].KeyExpires["temp"] != 1_700_000_123_456 || got[0].KeyExpires["other"] != 42 {
		t.Fatalf("record 0 KeyExpires = %v, want {temp:1700000123456, other:42}", got[0].KeyExpires)
	}
	if got[1].KeyExpires != nil {
		t.Fatalf("record 1 KeyExpires = %v, want nil (no per-key TTL)", got[1].KeyExpires)
	}
}

// TestMVScanResultKeyExpiresEmptyTrailer: a record with no per-key TTL appends
// only the single present=0 trailer byte beyond the version trailer, and the
// no-key-TTL path round-trips to nil.
func TestMVScanResultKeyExpiresEmptyTrailer(t *testing.T) {
	recs := []vector.MultiScanRecord{{ID: 5, Tokens: [][]float32{{1, 2, 3, 4}}, Version: 2}}
	blob := EncodeMVScanResult(recs)
	if blob[len(blob)-1] != 0 {
		t.Fatalf("empty-keyExpires record trailer = %d, want present=0", blob[len(blob)-1])
	}
	got, err := DecodeMVScanResult(blob)
	if err != nil {
		t.Fatal(err)
	}
	if got[0].KeyExpires != nil {
		t.Fatalf("KeyExpires = %v, want nil", got[0].KeyExpires)
	}
}

// TestMVScanResultOldBlobNoTrailerDecodesNil: an OLD MV scan blob (version trailer
// but NO keyExpires present byte) decodes per-record to KeyExpires==nil, mirroring
// the dense scan codec's EOF tolerance. We synthesize an old blob by stripping the
// trailing present=0 byte from a single-record encode.
func TestMVScanResultOldBlobNoTrailerDecodesNil(t *testing.T) {
	recs := []vector.MultiScanRecord{{ID: 9, Tokens: [][]float32{{1, 2, 3, 4}}, Version: 4}}
	blob := EncodeMVScanResult(recs)
	old := blob[:len(blob)-1] // drop the keyExpires present byte → pre-change layout
	got, err := DecodeMVScanResult(old)
	if err != nil {
		t.Fatalf("decode old blob: %v", err)
	}
	if len(got) != 1 || got[0].ID != 9 || got[0].Version != 4 || got[0].KeyExpires != nil {
		t.Fatalf("old blob decode = %+v, want id=9 version=4 KeyExpires=nil", got[0])
	}
}

// TestMVReshardKeyExpiresEncoderByteIdentical: the ABSOLUTE MV reinsert encoder is
// BYTE-IDENTICAL to EncodeMVAddArgsVersioned when the keyExpires map is nil/empty
// (zero overhead for the no-key-TTL reshard path).
func TestMVReshardKeyExpiresEncoderByteIdentical(t *testing.T) {
	name, docID := "docs", uint64(42)
	tokens := [][]float32{{1.5, -2.5, 3.5, 4.5}, {0, 1, 0, 1}}
	meta := vector.Metadata{"k": vector.NewInt(7)}

	cases := []struct {
		name string
		base []byte
		with []byte
	}{
		{
			"versioned-vs-keyexpires-nil",
			EncodeMVAddArgsVersioned(name, docID, tokens, meta, 11),
			EncodeMVAddArgsVersionedKeyExpires(name, docID, tokens, meta, 11, nil),
		},
		{
			"versioned-vs-keyexpires-empty",
			EncodeMVAddArgsVersioned(name, docID, tokens, meta, 11),
			EncodeMVAddArgsVersionedKeyExpires(name, docID, tokens, meta, 11, map[string]uint64{}),
		},
		{
			"version0-keyexpires-nil-vs-add",
			EncodeMVAddArgs(name, docID, tokens, meta),
			EncodeMVAddArgsVersionedKeyExpires(name, docID, tokens, meta, 0, nil),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !bytes.Equal(tc.base, tc.with) {
				t.Fatalf("not byte-identical:\n base=%x\n with=%x", tc.base, tc.with)
			}
		})
	}
}

// TestMVReshardKeyExpiresEncoderRoundTrip: the ABSOLUTE MV reinsert encoder writes
// the keyExpires deadlines VERBATIM (absolute unix-ms), and the decoder returns
// them alongside the version.
func TestMVReshardKeyExpiresEncoderRoundTrip(t *testing.T) {
	name, docID := "docs", uint64(7)
	tokens := [][]float32{{1, 0, 0, 0}}
	ke := map[string]uint64{"temp": 1_700_000_999_000, "more": 5}
	args := EncodeMVAddArgsVersionedKeyExpires(name, docID, tokens, vector.Metadata{"m": vector.NewInt(1)}, 9, ke)

	gotName, gotID, gotTokens, _, version, keyExp, err := DecodeMVAddArgsVersionedKeyExpires(args)
	if err != nil {
		t.Fatal(err)
	}
	if gotName != name || gotID != docID || version != 9 || len(gotTokens) != 1 {
		t.Fatalf("decode mismatch name=%q id=%d version=%d tokens=%d", gotName, gotID, version, len(gotTokens))
	}
	if len(keyExp) != 2 || keyExp["temp"] != 1_700_000_999_000 || keyExp["more"] != 5 {
		t.Fatalf("decoded keyExpires = %v, want {temp:1700000999000, more:5}", keyExp)
	}
}

// TestMVReshardKeyExpiresVersion0WithMap: a keyExpires map with version==0 still
// emits the trailers (so the absolute deadlines are carried) and decodes back the
// map with version 0 (the engine then defaults a fresh add to version 1).
func TestMVReshardKeyExpiresVersion0WithMap(t *testing.T) {
	args := EncodeMVAddArgsVersionedKeyExpires("c", 3, [][]float32{{1, 2}}, nil, 0, map[string]uint64{"a": 100, "b": 200})
	_, _, _, _, version, got, err := DecodeMVAddArgsVersionedKeyExpires(args)
	if err != nil {
		t.Fatal(err)
	}
	if version != 0 {
		t.Fatalf("version = %d, want 0", version)
	}
	if len(got) != 2 || got["a"] != 100 || got["b"] != 200 {
		t.Fatalf("absolute keyExpires = %v, want {a:100,b:200}", got)
	}
}

// TestMVAddArgsVersionedKeyExpiresLegacyDecode: a legacy EncodeMVAddArgs blob (no
// trailers at all) decodes to version 0, keyExpires nil via the new decoder.
func TestMVAddArgsVersionedKeyExpiresLegacyDecode(t *testing.T) {
	args := EncodeMVAddArgs("c", 1, [][]float32{{1, 2, 3, 4}}, vector.Metadata{"x": vector.NewInt(1)})
	_, _, _, _, version, got, err := DecodeMVAddArgsVersionedKeyExpires(args)
	if err != nil {
		t.Fatal(err)
	}
	if version != 0 || got != nil {
		t.Fatalf("legacy decode version=%d keyExpires=%v, want 0/nil", version, got)
	}
}

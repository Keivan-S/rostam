// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// TestMVScanResultVersionRoundtrip confirms the MV scan codec carries the
// per-document version: a version>1 and a version-0 record both round-trip.
func TestMVScanResultVersionRoundtrip(t *testing.T) {
	in := []vector.MultiScanRecord{
		{ID: 1, Tokens: [][]float32{{1, 2, 3, 4}}, Metadata: vector.Metadata{"k": vector.NewInt(1)}, Version: 7},
		{ID: 2, Tokens: [][]float32{{5, 6, 7, 8}}, Version: 0}, // version 0 round-trips
	}
	got, err := DecodeMVScanResult(EncodeMVScanResult(in))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("count = %d, want 2", len(got))
	}
	if got[0].Version != 7 {
		t.Errorf("rec0 version = %d, want 7", got[0].Version)
	}
	if got[1].Version != 0 {
		t.Errorf("rec1 version = %d, want 0", got[1].Version)
	}
}

// TestMVScanResultRequiresVersionTrailer confirms the trailing per-record version
// u64 is REQUIRED (mirroring the dense DecodeScanVectorsResult contract): a blob
// missing it is rejected as truncated rather than silently decoded as version 0.
// MV scan results are transient (a live reshard scan, never a stored artifact), so
// the encoder always writes the trailer and there is no pre-version blob to be
// "tolerant" of — and per-record tolerance would be UNSAFE for a multi-record blob
// (a missing trailer on record N would consume record N+1's [id:u64] as N's version
// and corrupt every following record). This test pins the safe, precedent-aligned
// behavior.
func TestMVScanResultRequiresVersionTrailer(t *testing.T) {
	// Hand-build a single-record blob in the OLD (pre-version) layout:
	//   [count:u32=1][id:u64][numTokens:u32=1][dim:u32=4][tok 4×f32][metaLen:u32=0]
	// i.e. exactly EncodeMVScanResult's output MINUS the trailing [version:u64].
	var b bytes.Buffer
	w32 := func(v uint32) { var t [4]byte; binary.BigEndian.PutUint32(t[:], v); b.Write(t[:]) }
	w64 := func(v uint64) { var t [8]byte; binary.BigEndian.PutUint64(t[:], v); b.Write(t[:]) }
	wf := func(f float32) { var t [4]byte; binary.BigEndian.PutUint32(t[:], math.Float32bits(f)); b.Write(t[:]) }
	w32(1)  // count
	w64(42) // id
	w32(1)  // numTokens
	w32(4)  // dim
	wf(1)
	wf(0)
	wf(0)
	wf(0)
	w32(0) // metaLen = 0, NO trailing version

	if _, err := DecodeMVScanResult(b.Bytes()); err == nil {
		t.Fatal("blob missing the required trailing version decoded without error, want errVectorArgsTruncated")
	}
}

// TestMVAddArgsVersionedByteIdenticalAtZero confirms the versioned MV add wire is
// BYTE-IDENTICAL to the plain EncodeMVAddArgs when version==0 (the no-version path
// stays zero-overhead) and carries the verbatim version when version>0.
func TestMVAddArgsVersionedByteIdenticalAtZero(t *testing.T) {
	name := "acme/docs"
	tokens := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}
	meta := vector.Metadata{"kind": vector.NewString("a")}

	base := EncodeMVAddArgs(name, 5, tokens, meta)
	v0 := EncodeMVAddArgsVersioned(name, 5, tokens, meta, 0)
	if !bytes.Equal(base, v0) {
		t.Fatalf("version==0 wire differs from EncodeMVAddArgs:\n base=%x\n v0  =%x", base, v0)
	}

	// version>0: trailer present; base block decodes identically, version recovered.
	v7 := EncodeMVAddArgsVersioned(name, 5, tokens, meta, 7)
	if len(v7) != len(base)+9 { // +1 verPresent +8 version
		t.Fatalf("versioned wire len = %d, want %d", len(v7), len(base)+9)
	}
	gName, gID, gTok, gMeta, gVer, err := DecodeMVAddArgsVersioned(v7)
	if err != nil {
		t.Fatal(err)
	}
	if gName != name || gID != 5 || gVer != 7 {
		t.Fatalf("decoded name=%q id=%d ver=%d, want %q 5 7", gName, gID, gVer, name)
	}
	if len(gTok) != 2 || gTok[1][1] != 1 || gMeta["kind"].Str != "a" {
		t.Fatalf("decoded tokens/meta wrong: tok=%v meta=%v", gTok, gMeta)
	}

	// A legacy EncodeMVAddArgs blob decodes via the versioned decoder as version 0.
	_, _, _, _, legacyVer, err := DecodeMVAddArgsVersioned(base)
	if err != nil {
		t.Fatal(err)
	}
	if legacyVer != 0 {
		t.Fatalf("legacy blob version = %d, want 0", legacyVer)
	}
}

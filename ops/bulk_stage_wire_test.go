// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"math"
	"testing"
)

// TestBulkStageArgsHeaderGolden pins the vector_bulk_stage wire layout to a
// GOLDEN byte string.
//
// Comparing BulkStageArgsHeader against EncodeBulkStageArgs proves nothing on
// its own: the encoder is built on the header function, so the two agree by
// construction and would keep agreeing while both drifted together. The HTTP
// binary bulk transport emits this header and then streams raw rows in behind
// it, so the layout is a cross-package contract; only a literal fixes it.
func TestBulkStageArgsHeaderGolden(t *testing.T) {
	// [colLen=04]["docs"][dim=3 u32 BE][count=2 u32 BE]
	const goldenHeader = "04" + "646f6373" + "00000003" + "00000002"
	got := BulkStageArgsHeader("docs", 3, 2)
	if hex.EncodeToString(got) != goldenHeader {
		t.Fatalf("header layout changed\n got %s\nwant %s", hex.EncodeToString(got), goldenHeader)
	}

	// Full request: the header above followed by two [id u64][3 × f32] rows.
	// 1.0 = 0x3f800000, -2.0 = 0xc0000000, 0.5 = 0x3f000000, 2.0 = 0x40000000.
	const goldenArgs = goldenHeader +
		"0000000000000001" + "3f800000" + "c0000000" + "3f000000" +
		"0000000000000007" + "00000000" + "3f800000" + "40000000"
	args, err := EncodeBulkStageArgs("docs",
		[]uint64{1, 7},
		[][]float32{{1, -2, 0.5}, {0, 1, 2}})
	if err != nil {
		t.Fatal(err)
	}
	if hex.EncodeToString(args) != goldenArgs {
		t.Fatalf("bulk stage args layout changed\n got %s\nwant %s", hex.EncodeToString(args), goldenArgs)
	}
}

// TestBulkStageStreamedRowsMatchEncoder is the contract the HTTP binary bulk
// path depends on: a header from BulkStageArgsHeader followed by raw big-endian
// [id u64][dim×f32] rows — exactly what arrives on that wire — is byte-identical
// to EncodeBulkStageArgs and decodes back to the same points. If either side's
// row layout moves, the transport would silently stage garbage.
func TestBulkStageStreamedRowsMatchEncoder(t *testing.T) {
	ids := []uint64{1, 7, 1 << 40}
	vecs := [][]float32{
		{0.5, -1.25, 3},
		{0, 0, 0},
		{float32(math.Pi), -0, 1e9},
	}
	const dim = 3

	// Simulate the transport: header, then rows appended as they "arrive".
	var buf bytes.Buffer
	buf.Write(BulkStageArgsHeader("docs", dim, len(ids)))
	if buf.Len() != 1+len("docs")+4+4 {
		t.Fatalf("header len = %d, want %d", buf.Len(), 1+len("docs")+4+4)
	}
	var row [8]byte
	for i, id := range ids {
		binary.BigEndian.PutUint64(row[:], id)
		buf.Write(row[:])
		for _, f := range vecs[i] {
			binary.BigEndian.PutUint32(row[:4], math.Float32bits(f))
			buf.Write(row[:4])
		}
	}
	want, err := EncodeBulkStageArgs("docs", ids, vecs)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(buf.Bytes(), want) {
		t.Fatalf("streamed layout differs from EncodeBulkStageArgs\n got %x\nwant %x", buf.Bytes(), want)
	}
	if want := 1 + len("docs") + 4 + 4 + len(ids)*BulkStageRowLen(dim); buf.Len() != want {
		t.Fatalf("total len = %d, want %d", buf.Len(), want)
	}

	gotCol, gotIDs, gotVecs, err := DecodeBulkStageArgs(buf.Bytes())
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if gotCol != "docs" || len(gotIDs) != len(ids) || len(gotVecs) != len(vecs) {
		t.Fatalf("decoded col=%q ids=%d vecs=%d", gotCol, len(gotIDs), len(gotVecs))
	}
	for i := range ids {
		if gotIDs[i] != ids[i] {
			t.Fatalf("id[%d] = %d, want %d", i, gotIDs[i], ids[i])
		}
		for d := range vecs[i] {
			if gotVecs[i][d] != vecs[i][d] {
				t.Fatalf("vec[%d][%d] = %v, want %v", i, d, gotVecs[i][d], vecs[i][d])
			}
		}
	}

	// Zero points: header only, and still a valid (empty) decode.
	empty := BulkStageArgsHeader("docs", 0, 0)
	if _, gotIDs, _, err = DecodeBulkStageArgs(empty); err != nil || len(gotIDs) != 0 {
		t.Fatalf("empty decode: ids=%d err=%v", len(gotIDs), err)
	}
	wantEmpty, err := EncodeBulkStageArgs("docs", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(empty, wantEmpty) {
		t.Fatalf("empty layout differs: got %x want %x", empty, wantEmpty)
	}
}

// TestBulkStageArgsHeaderRejectsOversizeName pins the documented precondition:
// the name length is a SINGLE byte on this wire, so a >=256-byte name cannot be
// encoded. Truncating it (byte(len(name)) wraps 256 → 0) would silently retarget
// the write at a DIFFERENT collection, so the encoder refuses outright rather
// than emitting a corrupt header.
func TestBulkStageArgsHeaderRejectsOversizeName(t *testing.T) {
	if MaxCollectionNameWire != 255 {
		t.Fatalf("MaxCollectionNameWire = %d, want 255 (one length byte)", MaxCollectionNameWire)
	}
	long := string(bytes.Repeat([]byte("a"), MaxCollectionNameWire))
	if got := BulkStageArgsHeader(long, 1, 0); int(got[0]) != MaxCollectionNameWire {
		t.Fatalf("len byte = %d, want %d", got[0], MaxCollectionNameWire)
	}
	defer func() {
		if recover() == nil {
			t.Fatal("BulkStageArgsHeader accepted a 256-byte name; the length byte would have wrapped to 0")
		}
	}()
	BulkStageArgsHeader(long+"a", 1, 0)
}

// TestCountFitsInIsTheSharedBound guards the reason CountFitsIn is exported: the
// HTTP binary bulk transport calls it instead of restating the arithmetic. A
// hand-copied bound drifts silently, and a drifted bound fails as an OOM rather
// than as a test.
func TestCountFitsInIsTheSharedBound(t *testing.T) {
	cases := []struct {
		n, remaining, per int
		want              bool
	}{
		{0, 0, 1, true},
		{10, 100, 10, true},
		{11, 100, 10, false},
		{-1, 100, 10, false},     // negative widening (32-bit int)
		{10, -1, 10, false},      // negative remaining
		{10, 100, 0, false},      // zero floor would divide by zero
		{1 << 30, 16, 12, false}, // the hostile shape: huge count, tiny body
	}
	for _, c := range cases {
		if got := CountFitsIn(c.n, c.remaining, c.per); got != c.want {
			t.Fatalf("CountFitsIn(%d,%d,%d) = %v, want %v", c.n, c.remaining, c.per, got, c.want)
		}
	}
}

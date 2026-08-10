// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"encoding/binary"
	"errors"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// A decoder that sizes a reservation from an unvalidated wire count turns a corrupt
// or hostile frame into an out-of-memory abort — not an error a caller can reject
// the frame on, but a process death, reachable from every transport that decodes
// that frame. The tests below pin the bound at each decoder that reserves from a
// count: a hostile count must be a decode ERROR, and the exact-fit boundary must
// still decode (so the bound is not merely "reject everything").
//
// CountFitsIn carries the shared argument; these tests are what stop it from being
// quietly dropped at any one call site.

func TestCountFitsIn(t *testing.T) {
	// 2^31 as an int, computed at RUNTIME through a non-constant shift. The
	// obvious `1 << 31` is an untyped constant that does not fit a 32-bit int, so
	// it fails to COMPILE under GOARCH=386 — the very platform this case is
	// about — and so does `int(int64(1) << 31)`, which Go still folds as a
	// constant conversion. Going through a variable shift defers it to run time.
	//
	// On 64-bit this lands as an oversized positive count; on 32-bit the
	// conversion truncates it to MinInt32, which is the wire-decode negative
	// wrap itself. CountFitsIn must refuse both, which is exactly the point.
	shift := uint(31)
	oversized := int(int64(1) << shift)

	cases := []struct {
		n, remaining, minPer int
		want                 bool
	}{
		{0, 0, 9, true},           // an empty body can carry zero elements
		{1, 9, 9, true},           // exact fit
		{2, 9, 9, false},          // one byte short of two elements
		{1, 8, 9, false},          // one byte short of one element
		{-1, 100, 9, false},       // a negative count (32-bit int wrap) is never valid
		{1, -1, 9, false},         // a negative remaining is never valid
		{1, 100, 0, false},        // a zero floor would divide by zero
		{oversized, 22, 4, false}, // see above: oversized on 64-bit, negative on 32-bit
	}
	for _, c := range cases {
		if got := CountFitsIn(c.n, c.remaining, c.minPer); got != c.want {
			t.Errorf("CountFitsIn(%d, %d, %d) = %v, want %v", c.n, c.remaining, c.minPer, got, c.want)
		}
	}
}

// TestMVGetBatchResultHostileRowCount: an MV batch row costs >= 9 bytes
// ([id:u64] + a not-found [found=0:u8]).
func TestMVGetBatchResultHostileRowCount(t *testing.T) {
	for _, n := range []uint32{0xFFFFFFFF, 2} {
		body := make([]byte, 4+9)
		binary.BigEndian.PutUint32(body, n)
		binary.BigEndian.PutUint64(body[4:], 7)
		if _, err := DecodeMVGetBatchResult(body); !errors.Is(err, errVectorArgsTruncated) {
			t.Fatalf("n=%d: err = %v, want errVectorArgsTruncated", n, err)
		}
	}
	ok := make([]byte, 4+9)
	binary.BigEndian.PutUint32(ok, 1)
	binary.BigEndian.PutUint64(ok[4:], 7)
	got, err := DecodeMVGetBatchResult(ok)
	if err != nil {
		t.Fatalf("exact-fit body rejected: %v", err)
	}
	if len(got) != 1 || got[0].ID != 7 || got[0].Found {
		t.Fatalf("exact-fit body decoded %+v", got)
	}
}

// TestMVGetResultTokenCountOverflow is the regression test for the multiplication
// that WRAPPED: numTokens*dim*4 with numTokens = dim = 2^31 is exactly 0 in 64-bit
// int arithmetic, so the byte check passed and a 22-byte body reached
// make([][]float32, 2^31) — a 51.5GB reservation. Both the single-get record
// decoder and the scan-record decoder shared the shape.
func TestMVGetResultTokenCountOverflow(t *testing.T) {
	// [found=1][numTokens:u32][dim:u32] chosen so numTokens*dim*4 wraps to 0.
	body := make([]byte, 1+4+4+8)
	body[0] = 1
	binary.BigEndian.PutUint32(body[1:], 1<<31)
	binary.BigEndian.PutUint32(body[5:], 1<<31)
	if _, _, _, err := DecodeMVGetResult(body); !errors.Is(err, errVectorArgsTruncated) {
		t.Fatalf("wrapping numTokens*dim: err = %v, want errVectorArgsTruncated", err)
	}
	// Plain oversized counts (no wrap) must be rejected too.
	for _, pair := range [][2]uint32{{0xFFFFFFFF, 1}, {1, 0xFFFFFFFF}, {1 << 20, 1 << 20}} {
		b := make([]byte, 1+4+4+8)
		b[0] = 1
		binary.BigEndian.PutUint32(b[1:], pair[0])
		binary.BigEndian.PutUint32(b[5:], pair[1])
		if _, _, _, err := DecodeMVGetResult(b); !errors.Is(err, errVectorArgsTruncated) {
			t.Fatalf("numTokens=%d dim=%d: err = %v, want errVectorArgsTruncated", pair[0], pair[1], err)
		}
	}
	// A well-formed record still round-trips (the bound rejects only the impossible).
	tokens := [][]float32{{1, 2}, {3, 4}}
	enc := EncodeMVGetResult(true, tokens, vector.Metadata{}, true, true)
	found, gotTokens, _, err := DecodeMVGetResult(enc)
	if err != nil || !found || len(gotTokens) != 2 || gotTokens[1][1] != 4 {
		t.Fatalf("well-formed record: found=%v tokens=%v err=%v", found, gotTokens, err)
	}
}

// TestMVScanResultHostileCounts covers the scan-record twin of the above: the
// record count and the per-record token matrix.
func TestMVScanResultHostileCounts(t *testing.T) {
	// Hostile record count.
	body := make([]byte, 4+16)
	binary.BigEndian.PutUint32(body, 0xFFFFFFFF)
	if _, err := DecodeMVScanResult(body); !errors.Is(err, errVectorArgsTruncated) {
		t.Fatalf("record count: err = %v, want errVectorArgsTruncated", err)
	}
	// One record whose numTokens*dim*4 wraps to zero.
	wrap := make([]byte, 4+8+4+4+8)
	binary.BigEndian.PutUint32(wrap, 1)          // one record
	binary.BigEndian.PutUint64(wrap[4:], 42)     // id
	binary.BigEndian.PutUint32(wrap[12:], 1<<31) // numTokens
	binary.BigEndian.PutUint32(wrap[16:], 1<<31) // dim
	if _, err := DecodeMVScanResult(wrap); !errors.Is(err, errVectorArgsTruncated) {
		t.Fatalf("wrapping token matrix: err = %v, want errVectorArgsTruncated", err)
	}
}

// TestNamedGetResultHostileSpaceCount: a named space costs >= 6 bytes
// ([nameLen:u16] with an empty name + [dim:u32] with dim 0).
func TestNamedGetResultHostileSpaceCount(t *testing.T) {
	for _, n := range []uint32{0xFFFFFFFF, 2} {
		body := make([]byte, 1+4+6)
		body[0] = 1
		binary.BigEndian.PutUint32(body[1:], n)
		if _, _, _, _, err := DecodeNamedGetResult(body); !errors.Is(err, errVectorArgsTruncated) {
			t.Fatalf("numSpaces=%d: err = %v, want errVectorArgsTruncated", n, err)
		}
	}
	// Exact fit: one space, empty name, dim 0 — must decode, not be rejected.
	body := make([]byte, 1+4+2+4+8+1+1)
	body[0] = 1
	binary.BigEndian.PutUint32(body[1:], 1) // one space
	// [nameLen:u16]=0, [dim:u32]=0 already zeroed; ttl/metaPresent/verPresent follow.
	found, vectors, _, _, err := DecodeNamedGetResult(body)
	if err != nil {
		t.Fatalf("exact-fit body rejected: %v", err)
	}
	if !found || len(vectors) != 1 {
		t.Fatalf("exact-fit body decoded found=%v vectors=%v", found, vectors)
	}
}

// TestDecodeMatrixHostileRowCount: a matrix row costs >= 4 bytes ([dim:u32]).
func TestDecodeMatrixHostileRowCount(t *testing.T) {
	for _, rows := range []uint32{0xFFFFFFFF, 2} {
		b := make([]byte, 4+4)
		binary.BigEndian.PutUint32(b, rows)
		if _, _, err := decodeMatrix(b); !errors.Is(err, errVectorArgsTruncated) {
			t.Fatalf("rows=%d: err = %v, want errVectorArgsTruncated", rows, err)
		}
	}
	// Exact fit: one row of dim 0.
	b := make([]byte, 4+4)
	binary.BigEndian.PutUint32(b, 1)
	out, n, err := decodeMatrix(b)
	if err != nil || len(out) != 1 || n != 8 {
		t.Fatalf("exact-fit matrix: out=%v n=%d err=%v", out, n, err)
	}
}

// TestVectorDocsHostileCount / groups / scan share the block-count shape: each
// element has a fixed minimum width, so a count beyond the body is a decode error.
func TestVectorBlockDecodersHostileCounts(t *testing.T) {
	hostile := func(minPer int) []byte {
		b := make([]byte, 4+minPer)
		binary.BigEndian.PutUint32(b, 0xFFFFFFFF)
		return b
	}
	if _, err := DecodeVectorDocs(hostile(20)); !errors.Is(err, errVectorArgsTruncated) {
		t.Errorf("docs: err = %v, want errVectorArgsTruncated", err)
	}
	if _, err := DecodeScanVectorsResult(hostile(12)); !errors.Is(err, errVectorArgsTruncated) {
		t.Errorf("scan: err = %v, want errVectorArgsTruncated", err)
	}
	if _, err := DecodeGroups(hostile(4)); !errors.Is(err, errVectorArgsTruncated) {
		t.Errorf("groups: err = %v, want errVectorArgsTruncated", err)
	}
	// Zero-element blocks still decode (the bound must not reject the empty case).
	empty := make([]byte, 4)
	if _, err := DecodeVectorDocs(empty); err != nil {
		t.Errorf("empty docs block: %v", err)
	}
	if _, err := DecodeScanVectorsResult(empty); err != nil {
		t.Errorf("empty scan block: %v", err)
	}
	if _, err := DecodeGroups(empty); err != nil {
		t.Errorf("empty groups block: %v", err)
	}
}

// TestPutBatchHostileLengths covers the two decode sites in batch.go that were
// verified as reachable crashes on a 32-bit build from network-supplied args:
// DecodePutBatchArgs's entry COUNT, which reached make([]PutEntry, 0, n), and
// decodeOnePut's value LENGTH, which reached a slice bound.
//
// Both cases are asymmetric between platforms, which is exactly why they sat
// unnoticed. On 64-bit the hostile value stays positive and huge, so the
// existing size checks reject it and this test passes with or without the
// guards. On 32-bit (the linux/386 lane) `int` is 32 bits: 0xFFFFFFFF widens to
// -1, a negative length SATISFIES `len(buf) < off+n+8`, and the decode walks
// straight into a panic. So this test only has teeth where the bug lives — which
// is the point of running the suite under GOARCH=386 at all.
func TestPutBatchHostileLengths(t *testing.T) {
	t.Run("entry count", func(t *testing.T) {
		// A declared count of 2^32-1 with no entry bytes behind it.
		args := make([]byte, 4)
		binary.BigEndian.PutUint32(args, 0xFFFFFFFF)
		if _, err := DecodePutBatchArgs(args); !errors.Is(err, ErrShortArgs) {
			t.Fatalf("hostile entry count: err = %v, want ErrShortArgs", err)
		}
	})

	t.Run("value length", func(t *testing.T) {
		// One well-formed entry header declaring a 2^32-1 byte value. The body is
		// sized so that on a 32-bit build the negative-widened length PASSES the
		// `len(buf) < off+vlen+8` check — without the guard this is the panic.
		body := make([]byte, 4+2+1+4+8)
		binary.BigEndian.PutUint32(body[0:], 1)          // one entry
		binary.BigEndian.PutUint16(body[4:], 1)          // keyLen = 1
		body[6] = 'a'                                    // key
		binary.BigEndian.PutUint32(body[7:], 0xFFFFFFFF) // valLen
		if _, err := DecodePutBatchArgs(body); !errors.Is(err, ErrShortArgs) {
			t.Fatalf("hostile value length: err = %v, want ErrShortArgs", err)
		}
	})

	t.Run("exact fit still decodes", func(t *testing.T) {
		// The bound must not have become "reject everything".
		body := make([]byte, 4+2+1+4+3+8)
		binary.BigEndian.PutUint32(body[0:], 1)
		binary.BigEndian.PutUint16(body[4:], 1)
		body[6] = 'a'
		binary.BigEndian.PutUint32(body[7:], 3)
		copy(body[11:], "xyz")
		got, err := DecodePutBatchArgs(body)
		if err != nil {
			t.Fatalf("exact-fit batch rejected: %v", err)
		}
		if len(got) != 1 || string(got[0].Key) != "a" || string(got[0].Val) != "xyz" {
			t.Fatalf("exact-fit batch decoded %+v", got)
		}
	})
}

// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"reflect"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// The vector_bulk_stage_payload wire is the vector_bulk_stage wire plus a
// per-point payload section, and it is a CROSS-PACKAGE contract twice over: the
// HTTP binary transport streams both regions in behind the op header without
// re-encoding either, and the op is Raft-replicated, so a decoder anywhere in the
// cluster reads what an encoder somewhere else wrote. Both halves of that are
// pinned here.

// TestBulkStagePayloadArgsGolden pins the layout to a GOLDEN byte string.
// Comparing the encoder against the decoder proves nothing on its own — they
// would keep agreeing while both drifted — and comparing against
// EncodeBulkStageArgs proves only the prefix. Only a literal fixes the suffix.
func TestBulkStagePayloadArgsGolden(t *testing.T) {
	// The prefix is byte-for-byte a vector_bulk_stage request; the payload section
	// follows it. Point 1 carries {"n":{"kind":"int","int":5}}; point 7 carries
	// nothing, which is a bare zero length and no bytes.
	const goldenRows = "04" + "646f6373" + "00000002" + "00000002" +
		"0000000000000001" + "3f800000" + "c0000000" +
		"0000000000000007" + "00000000" + "3f800000"
	metaJSON := `{"n":{"kind":"int","int":5}}`
	golden := goldenRows +
		hexU32(len(metaJSON)) + hex.EncodeToString([]byte(metaJSON)) +
		"00000000"

	args, err := EncodeBulkStagePayloadArgs("docs",
		[]uint64{1, 7},
		[][]float32{{1, -2}, {0, 1}},
		[]vector.Metadata{{"n": vector.NewInt(5)}, nil})
	if err != nil {
		t.Fatal(err)
	}
	if hex.EncodeToString(args) != golden {
		t.Fatalf("bulk stage payload args layout changed\n got %s\nwant %s",
			hex.EncodeToString(args), golden)
	}

	// The ROW REGION must remain byte-identical to the vectors-only op's, which is
	// what lets the HTTP transport emit one header for either and lets a reader of
	// one wire understand the other's prefix.
	rowsOnly, err := EncodeBulkStageArgs("docs", []uint64{1, 7}, [][]float32{{1, -2}, {0, 1}})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(args, rowsOnly) {
		t.Fatal("the payload-bearing wire is no longer the vectors-only wire plus a suffix")
	}
}

func hexU32(n int) string {
	var b [4]byte
	binary.BigEndian.PutUint32(b[:], uint32(n)) //nolint:gosec // test literal
	return hex.EncodeToString(b[:])
}

// TestBulkStagePayloadRoundTrip covers the shapes a real load produces: every
// point with a payload, none, and a mix — plus an EMPTY batch, which the
// authorizer builds as its representative args.
func TestBulkStagePayloadRoundTrip(t *testing.T) {
	cases := []struct {
		name  string
		ids   []uint64
		vecs  [][]float32
		metas []vector.Metadata
	}{
		{"empty", nil, nil, nil},
		{
			"all-payloads",
			[]uint64{0, 1, 1 << 40},
			[][]float32{{1, 2}, {3, 4}, {5, 6}},
			[]vector.Metadata{
				{"id": vector.NewInt(0)},
				{"s": vector.NewString("x"), "f": vector.NewFloat(1.5)},
				{"tags": vector.NewStrings([]string{"a", "b"})},
			},
		},
		{
			"mixed",
			[]uint64{9, 10, 11},
			[][]float32{{1, 2}, {3, 4}, {5, 6}},
			[]vector.Metadata{nil, {"b": vector.NewBool(true)}, nil},
		},
		{
			// An all-nil payload column still round-trips: the section is present
			// (three zero lengths) and decodes to three absent payloads.
			"no-payloads",
			[]uint64{1, 2, 3},
			[][]float32{{1, 2}, {3, 4}, {5, 6}},
			[]vector.Metadata{nil, nil, nil},
		},
	}
	for _, tc := range cases {

		t.Run(tc.name, func(t *testing.T) {
			args, err := EncodeBulkStagePayloadArgs("c", tc.ids, tc.vecs, tc.metas)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			col, ids, vecs, metas, err := DecodeBulkStagePayloadArgs(args)
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if col != "c" {
				t.Fatalf("collection %q", col)
			}
			if len(ids) != len(tc.ids) || len(vecs) != len(tc.vecs) || len(metas) != len(tc.ids) {
				t.Fatalf("lengths: ids %d vecs %d metas %d, want %d",
					len(ids), len(vecs), len(metas), len(tc.ids))
			}
			for i := range tc.ids {
				if ids[i] != tc.ids[i] {
					t.Fatalf("id %d: got %d want %d", i, ids[i], tc.ids[i])
				}
				if !reflect.DeepEqual(vecs[i], tc.vecs[i]) {
					t.Fatalf("vec %d: got %v want %v", i, vecs[i], tc.vecs[i])
				}
				if len(tc.metas[i]) == 0 {
					if metas[i] != nil {
						t.Fatalf("payload %d: got %#v, want absent", i, metas[i])
					}
					continue
				}
				if !reflect.DeepEqual(metas[i], tc.metas[i]) {
					t.Fatalf("payload %d: got %#v want %#v", i, metas[i], tc.metas[i])
				}
			}
		})
	}
}

// TestBulkStagePayloadEncodeRejectsMisalignment: a payload column of the wrong
// length is the one encoder-side mistake that pairs payloads with the wrong
// points, and it must be an error rather than a truncation.
func TestBulkStagePayloadEncodeRejectsMisalignment(t *testing.T) {
	_, err := EncodeBulkStagePayloadArgs("c",
		[]uint64{1, 2},
		[][]float32{{1}, {2}},
		[]vector.Metadata{{"a": vector.NewInt(1)}})
	if err == nil {
		t.Fatal("encoder accepted 2 ids with 1 payload")
	}
	// Raggedness is still rejected by the shared row encoder, not re-implemented.
	_, err = EncodeBulkStagePayloadArgs("c",
		[]uint64{1, 2},
		[][]float32{{1, 1}, {2}},
		[]vector.Metadata{nil, nil})
	if err == nil {
		t.Fatal("encoder accepted a ragged batch")
	}
}

// TestBulkStagePayloadDecodeBounds is the bounds-discipline gate: every
// count/length on this wire is validated against what REMAINS in the buffer
// before it sizes a reservation or a slice. A hostile body must produce an error,
// never a multi-GB allocation, never an out-of-bounds read, and never a decode
// that silently pairs a payload with a different point.
func TestBulkStagePayloadDecodeBounds(t *testing.T) {
	good, err := EncodeBulkStagePayloadArgs("c",
		[]uint64{1, 2},
		[][]float32{{1, 1}, {2, 2}},
		[]vector.Metadata{{"a": vector.NewInt(1)}, {"b": vector.NewInt(2)}})
	if err != nil {
		t.Fatal(err)
	}

	// Where the payload section begins: header (1+1+4+4) + 2 rows of (8 + 2*4).
	const secStart = 1 + 1 + 4 + 4 + 2*(8+2*4)

	corrupt := func(mutate func([]byte) []byte) []byte {
		b := append([]byte(nil), good...)
		return mutate(b)
	}

	cases := []struct {
		name string
		args []byte
	}{
		{
			// A count that claims more points than the body can hold: the ROW bound
			// must reject it before the per-point slices are made.
			"count-exceeds-body",
			corrupt(func(b []byte) []byte {
				binary.BigEndian.PutUint32(b[6:], 1<<30) // count word
				return b
			}),
		},
		{
			// The rows fit but the payload section is absent entirely. 4*count bytes
			// are the floor, so this must be caught before make([]Metadata, count).
			"payload-section-missing",
			good[:secStart],
		},
		{
			// A payload length larger than the bytes behind it.
			"payload-length-overruns",
			corrupt(func(b []byte) []byte {
				binary.BigEndian.PutUint32(b[secStart:], 1<<30)
				return b
			}),
		},
		{
			// The maximum u32, which on a 32-bit build widens NEGATIVE and would sail
			// past a naive `>` bound.
			"payload-length-max-u32",
			corrupt(func(b []byte) []byte {
				binary.BigEndian.PutUint32(b[secStart:], ^uint32(0))
				return b
			}),
		},
		{
			// The first payload swallows the second's length prefix: the body is
			// self-consistent in total size but the framing is not, and accepting it
			// would stage point 2 against nothing.
			"payload-truncated-mid-section",
			good[:len(good)-3],
		},
		{
			// TRUNCATION AT A BLOB BOUNDARY: k complete blobs and then nothing, with
			// k < count. This is the gap between the two cases either side of it —
			// "truncated-mid-section" cuts INSIDE a blob (caught by blobLen >
			// remaining) and "payload-section-missing" removes the section entirely
			// (caught by the up-front CountFitsIn) — and it is the one shape neither
			// catches, because the section is >= 4*count while still ending early.
			//
			// It panicked: the loop read each 4-byte prefix without proving 4 bytes
			// remained, and index-out-of-range inside a raft FSM apply is a
			// cluster-wide crash loop, not a 500 (see DecodeBulkStagePayloadArgs).
			// The blob is "{  }" — a VALID 4-byte metadata object — specifically so
			// the non-object guard cannot short-circuit the case and the length check
			// is what has to catch it.
			"payload-blob-boundary-short-count",
			func() []byte {
				b := append([]byte(nil), good[:secStart]...)
				var l [4]byte
				binary.BigEndian.PutUint32(l[:], 4)
				b = append(b, l[:]...)
				return append(b, []byte("{  }")...) // 8 bytes: CountFitsIn(2,8,4) passes exactly
			}(),
		},
		{
			// JSON null: well-framed, unmarshals into a map with NO error, and leaves
			// it nil — so it would otherwise be accepted as a payload that is neither
			// absent nor an object. Zero length is the one spelling of "no payload".
			"payload-json-null",
			func() []byte {
				b := append([]byte(nil), good[:secStart]...)
				var l [4]byte
				binary.BigEndian.PutUint32(l[:], 4)
				b = append(b, l[:]...)
				b = append(b, []byte("null")...)
				b = append(b, 0, 0, 0, 0) // second point: no payload
				return b
			}(),
		},
		{
			// Valid JSON that is not an object, the other half of the same guard.
			"payload-json-scalar",
			func() []byte {
				b := append([]byte(nil), good[:secStart]...)
				var l [4]byte
				binary.BigEndian.PutUint32(l[:], 3)
				b = append(b, l[:]...)
				b = append(b, []byte("123")...)
				b = append(b, 0, 0, 0, 0)
				return b
			}(),
		},
		{
			// Bytes after the last payload: the sender framed the request differently
			// than it was read.
			"trailing-bytes",
			append(append([]byte(nil), good...), 0, 0, 0),
		},
		{
			// A payload that is not a JSON object.
			"payload-not-json",
			corrupt(func(b []byte) []byte {
				// The first payload is `{"a":{"kind":"int","int":1}}`; clobber its
				// opening brace, keeping the length intact.
				b[secStart+4] = '!'
				return b
			}),
		},
		{"empty-args", nil},
		{"header-only", good[:6]},
	}

	for _, tc := range cases {

		t.Run(tc.name, func(t *testing.T) {
			_, ids, _, metas, err := DecodeBulkStagePayloadArgs(tc.args)
			if err == nil {
				t.Fatalf("decoder accepted a malformed body: %d ids, %d payloads", len(ids), len(metas))
			}
		})
	}

	// The unmutated body must still decode, or the cases above would all be
	// passing for the wrong reason.
	if _, _, _, _, err := DecodeBulkStagePayloadArgs(good); err != nil {
		t.Fatalf("the reference body no longer decodes: %v", err)
	}
}

// TestBulkStagePayloadStreamedSectionMatchesEncoder is the contract the HTTP
// binary transport depends on: an op header, then raw rows, then the RVB1 payload
// section copied VERBATIM off the socket, is byte-identical to what the encoder
// produces. If either side's framing moves, the transport would silently stage
// payloads against the wrong points.
func TestBulkStagePayloadStreamedSectionMatchesEncoder(t *testing.T) {
	ids := []uint64{1, 7, 1 << 40}
	vecs := [][]float32{{0.5, -1.25}, {0, 0}, {3, 4}}
	metas := []vector.Metadata{{"a": vector.NewInt(1)}, nil, {"b": vector.NewString("z")}}
	const dim = 2

	var buf bytes.Buffer
	buf.Write(BulkStageArgsHeader("docs", dim, len(ids)))
	for i, id := range ids {
		var row [8]byte
		binary.BigEndian.PutUint64(row[:], id)
		buf.Write(row[:])
		for _, f := range vecs[i] {
			var w [4]byte
			binary.BigEndian.PutUint32(w[:], math.Float32bits(f))
			buf.Write(w[:])
		}
	}
	for _, m := range metas {
		var blob []byte
		if len(m) > 0 {
			b, err := json.Marshal(m)
			if err != nil {
				t.Fatal(err)
			}
			blob = b
		}
		var l [4]byte
		binary.BigEndian.PutUint32(l[:], uint32(len(blob))) //nolint:gosec // test fixture
		buf.Write(l[:])
		buf.Write(blob)
	}

	want, err := EncodeBulkStagePayloadArgs("docs", ids, vecs, metas)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(buf.Bytes(), want) {
		t.Fatalf("streamed framing differs from the encoder\n got %s\nwant %s",
			hex.EncodeToString(buf.Bytes()), hex.EncodeToString(want))
	}
}

// SPDX-License-Identifier: Apache-2.0
package wire

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/rostamlabs/rostam/vtypes"
)

// oracleVectorSearchArgsNoFilter hand-builds the no-filter vector_search wire
// bytes INDEPENDENTLY of the production encoder, so a layout error in
// AppendVectorSearchArgsExt is caught (a self-comparison against Encode*, which
// now delegates to Append*, would move both sides together and hide it).
//
//	[flags:u8=0][colLen:u8][col][k:u32][dim:u32][query f32 BE...]
func oracleVectorSearchArgsNoFilter(collection string, k int, query []float32) []byte {
	buf := make([]byte, 0, 2+len(collection)+8+4*len(query))
	buf = append(buf, 0)                     // flags = 0 (no filter)
	buf = append(buf, byte(len(collection))) // colLen u8
	buf = append(buf, collection...)         // col
	buf = binary.BigEndian.AppendUint32(buf, uint32(k))
	buf = binary.BigEndian.AppendUint32(buf, uint32(len(query)))
	for _, f := range query {
		buf = binary.BigEndian.AppendUint32(buf, math.Float32bits(f))
	}
	return buf
}

// TestAppendVectorSearchArgsByteIdentical guards the append-into-dst encoder
// against (a) an INDEPENDENT hand-built oracle for the no-filter layout and
// (b) the allocating Encode* form for the filtered path, across a nil dst and a
// pre-grown (reused-cap) dst. The golden oracle only covers dst=nil no-filter;
// this additionally pins the reused-cap branch VectorSearchInto's pool exercises.
func TestAppendVectorSearchArgsByteIdentical(t *testing.T) {
	query := []float32{1.5, -2.25, 3, 4, 5, 6, 7, 8}

	// Independent oracle check: the no-filter layout must match a hand-built
	// encoding, not just the (delegating) Encode* form.
	oracle := oracleVectorSearchArgsNoFilter("posts", 10, query)
	if got := AppendVectorSearchArgsExt(nil, "posts", 10, query, vtypes.Filter{}); !bytes.Equal(got, oracle) {
		t.Fatalf("no-filter layout diverged from independent oracle:\n got %x\nwant %x", got, oracle)
	}

	cases := []struct {
		name   string
		filter vtypes.Filter
	}{
		{"no_filter", vtypes.Filter{}},
		{"filtered", vtypes.Filter{Op: vtypes.FilterEq, Field: "genre", Value: vtypes.NewString("scifi")}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			want := EncodeVectorSearchArgsExt("posts", 10, query, tc.filter)

			// dst=nil must reproduce the legacy bytes exactly.
			if got := AppendVectorSearchArgsExt(nil, "posts", 10, query, tc.filter); !bytes.Equal(got, want) {
				t.Fatalf("dst=nil mismatch:\n got %x\nwant %x", got, want)
			}

			// A pre-grown, dirty dst (reused-cap branch) must yield identical
			// bytes and length — no stale data beyond the written region.
			dirty := make([]byte, 0, len(want)+64)
			dirty = dirty[:cap(dirty)]
			for i := range dirty {
				dirty[i] = 0xEE
			}
			got := AppendVectorSearchArgsExt(dirty[:0], "posts", 10, query, tc.filter)
			if !bytes.Equal(got, want) {
				t.Fatalf("reused-cap mismatch:\n got %x\nwant %x", got, want)
			}
			if len(got) != len(want) {
				t.Fatalf("reused-cap length = %d, want %d", len(got), len(want))
			}
		})
	}
}

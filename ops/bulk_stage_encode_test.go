// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// TestEncodeBulkStageArgsRejectsRagged pins the encoder's core precondition: the
// wire carries ONE dim for the whole batch, so every vector must have it.
//
// The middle-short case is the reason this cannot be checked anywhere else.
// Encoding it used to succeed and produce a buffer that decoded back to
// ids [1 2 4674736414298996736] with mangled vectors and err=nil — a point
// stored under an id nobody sent. DecodeBulkStageArgs materializes
// make([]float32, dim) per row, so by the time anything downstream looks, every
// vector is uniform: the raggedness has already been converted into corruption.
func TestEncodeBulkStageArgsRejectsRagged(t *testing.T) {
	cases := []struct {
		name string
		ids  []uint64
		vecs [][]float32
	}{
		{"later longer", []uint64{1, 2}, [][]float32{{1, 1, 1, 1}, {9, 9, 9, 9, 9, 9}}},
		{"later shorter", []uint64{1, 2}, [][]float32{{1, 1, 1, 1}, {9, 9}}},
		{"middle short", []uint64{1, 2, 3}, [][]float32{{1, 1, 1, 1}, {9, 9}, {7, 7, 7, 7}}},
		{"first short", []uint64{1, 2}, [][]float32{{1}, {9, 9, 9, 9}}},
		{"nil vector", []uint64{1, 2}, [][]float32{{1, 1}, nil}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			args, err := EncodeBulkStageArgs("docs", c.ids, c.vecs)
			if err == nil {
				t.Fatalf("ragged batch encoded without error; decodes to a corrupt frame")
			}
			if !errors.Is(err, vector.ErrDimMismatch) {
				t.Fatalf("err = %v, want ErrDimMismatch (so transports map it to 400)", err)
			}
			if args != nil {
				t.Fatal("a rejected batch returned a buffer")
			}
		})
	}
}

// TestEncodeBulkStageArgsRejectsLengthMismatch: ids and vecs must correspond.
func TestEncodeBulkStageArgsRejectsLengthMismatch(t *testing.T) {
	if _, err := EncodeBulkStageArgs("docs", []uint64{1, 2, 3}, [][]float32{{1}}); err == nil {
		t.Fatal("3 ids with 1 vector encoded without error")
	}
	// The old code read vecs[i] for every id and would have panicked here.
	if _, err := EncodeBulkStageArgs("docs", []uint64{1}, [][]float32{{1}, {2}}); err == nil {
		t.Fatal("1 id with 2 vectors encoded without error")
	}
}

// TestEncodeBulkStageArgsUniformRoundTrips guards against over-rejection: a
// uniform batch — including an empty one and a zero-dim one — still encodes and
// decodes unchanged. A uniform batch whose dim is wrong for some COLLECTION is
// not this function's business; it has no collection to compare against.
func TestEncodeBulkStageArgsUniformRoundTrips(t *testing.T) {
	ids := []uint64{1, 7, 1 << 40}
	vecs := [][]float32{{0.5, -1.25, 3}, {0, 0, 0}, {1, 2, 3}}
	args, err := EncodeBulkStageArgs("docs", ids, vecs)
	if err != nil {
		t.Fatalf("uniform batch rejected: %v", err)
	}
	col, gotIDs, gotVecs, err := DecodeBulkStageArgs(args)
	if err != nil || col != "docs" {
		t.Fatalf("decode: col=%q err=%v", col, err)
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

	// Empty batch: used as the representative args for authorization, so it must
	// never fail to encode.
	if _, err := EncodeBulkStageArgs("docs", nil, nil); err != nil {
		t.Fatalf("empty batch rejected: %v", err)
	}
	// Uniformly zero-dim is degenerate but well-formed.
	if _, err := EncodeBulkStageArgs("docs", []uint64{1}, [][]float32{{}}); err != nil {
		t.Fatalf("zero-dim batch rejected: %v", err)
	}
}

// TestEncodeBulkStageArgsBoundsTheProduct covers the amplifier directly at the
// encoder: dim and the point count used to be independent, so a caller could ask
// for a buffer neither number alone justified. Ragged rejection removes the
// attack (every vector must really hold dim floats), and the overflow guard
// catches what remains.
func TestEncodeBulkStageArgsBoundsTheProduct(t *testing.T) {
	// The attack shape: one long vector, many vector-less points. It is now a
	// length mismatch on the very first short vector rather than a 1.6 GB buffer.
	long := make([]float32, 20000)
	vecs := make([][]float32, 20000)
	ids := make([]uint64, 20000)
	vecs[0] = long
	for i := 1; i < len(vecs); i++ {
		vecs[i] = nil // no vector supplied for this point
		ids[i] = uint64(i)
	}
	args, err := EncodeBulkStageArgs("docs", ids, vecs)
	if err == nil {
		t.Fatalf("the amplification shape encoded successfully into %d bytes", len(args))
	}
	if !errors.Is(err, vector.ErrDimMismatch) {
		t.Fatalf("err = %v, want ErrDimMismatch", err)
	}
}

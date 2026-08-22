// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"reflect"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// TestNamedHybridArgsRoundTrip checks EncodeNamedHybridArgs / DecodeNamedHybridArgs
// preserve every field (both space names, dense + sparse queries, fusion opts,
// filter, rc/opa).
func TestNamedHybridArgsRoundTrip(t *testing.T) {
	denseQ := []float32{1, 2, 3, 4}
	sparseQ := vector.SparseVector{Indices: []uint32{2, 5}, Values: []float32{2, 1}}
	opts := vector.HybridOpts{
		Method: vector.FusionWeighted, Alpha: 0.25, RRFK: 17, DenseK: 33, SparseK: 44,
		Filter: vector.Filter{Op: vector.FilterEq, Field: "kind", Value: vector.NewString("a")},
	}
	args := EncodeNamedHybridArgs("docs", "title", denseQ, "terms", sparseQ, 9, opts, 2, 1, 0)

	col, ds, dq, ss, sq, k, gotOpts, rc, opa, _, err := DecodeNamedHybridArgs(args)
	if err != nil {
		t.Fatal(err)
	}
	if col != "docs" || ds != "title" || ss != "terms" || k != 9 || rc != 2 || opa != 1 {
		t.Fatalf("scalars: col=%q ds=%q ss=%q k=%d rc=%d opa=%d", col, ds, ss, k, rc, opa)
	}
	if !reflect.DeepEqual(dq, denseQ) {
		t.Fatalf("denseQ = %v, want %v", dq, denseQ)
	}
	if !reflect.DeepEqual(sq, sparseQ) {
		t.Fatalf("sparseQ = %v, want %v", sq, sparseQ)
	}
	if gotOpts.Method != opts.Method || gotOpts.Alpha != opts.Alpha || gotOpts.RRFK != opts.RRFK ||
		gotOpts.DenseK != opts.DenseK || gotOpts.SparseK != opts.SparseK {
		t.Fatalf("opts = %+v, want %+v", gotOpts, opts)
	}
	if !reflect.DeepEqual(gotOpts.Filter, opts.Filter) {
		t.Fatalf("filter = %+v, want %+v", gotOpts.Filter, opts.Filter)
	}
}

// TestNamedHybridArgsNoOptsTrailer checks rc==0 && opa==0 omits the opts trailer
// (byte-identical-trailer property) yet still round-trips, and the dense-only /
// sparse-only degradation queries decode correctly.
func TestNamedHybridArgsNoOptsTrailer(t *testing.T) {
	denseQ := []float32{1, 0, 0, 0}
	// rc==0 && opa==0: no trailer. Two encodes with the same payload are identical.
	a1 := EncodeNamedHybridArgs("c", "title", denseQ, "terms", vector.SparseVector{}, 5, vector.HybridOpts{}, 0, 0, 0)
	a2 := EncodeNamedHybridArgs("c", "title", denseQ, "terms", vector.SparseVector{}, 5, vector.HybridOpts{}, 0, 0, 0)
	if !bytes.Equal(a1, a2) {
		t.Fatal("encode not deterministic")
	}
	// The no-rc encoding must NOT carry the opts flag bit.
	if a1[0]&NamedHybridFlagOpts != 0 {
		t.Fatal("opts flag set despite rc==0 && opa==0")
	}
	col, ds, dq, ss, sq, k, _, rc, opa, _, err := DecodeNamedHybridArgs(a1)
	if err != nil {
		t.Fatal(err)
	}
	if col != "c" || ds != "title" || ss != "terms" || k != 5 || rc != 0 || opa != 0 {
		t.Fatalf("scalars: col=%q ds=%q ss=%q k=%d rc=%d opa=%d", col, ds, ss, k, rc, opa)
	}
	if !reflect.DeepEqual(dq, denseQ) {
		t.Fatalf("denseQ = %v", dq)
	}
	if !sq.IsZero() {
		t.Fatalf("sparseQ should be empty, got %v", sq)
	}

	// Sparse-only: empty dense query (dim 0) decodes to a nil dense slice.
	sparseQ := vector.SparseVector{Indices: []uint32{1}, Values: []float32{1}}
	b := EncodeNamedHybridArgs("c", "title", nil, "terms", sparseQ, 5, vector.HybridOpts{}, 0, 0, 0)
	_, _, dq2, _, sq2, _, _, _, _, _, err := DecodeNamedHybridArgs(b)
	if err != nil {
		t.Fatal(err)
	}
	if len(dq2) != 0 {
		t.Fatalf("dense should be empty, got %v", dq2)
	}
	if !reflect.DeepEqual(sq2, sparseQ) {
		t.Fatalf("sparseQ = %v, want %v", sq2, sparseQ)
	}
}

// TestNamedHybridReadConsistencyOf checks a Linearizable named hybrid arms the
// barrier (ReadConsistencyOf reads the rc byte for both the search + lanes ops).
func TestNamedHybridReadConsistencyOf(t *testing.T) {
	args := EncodeNamedHybridArgs("c", "title", []float32{1, 0, 0, 0}, "terms",
		vector.SparseVector{Indices: []uint32{1}, Values: []float32{1}}, 5, vector.HybridOpts{}, 2, 0, 0)
	for _, op := range []string{"vector_named_hybrid_search", "vector_named_hybrid_lanes"} {
		rc, ok := ReadConsistencyOf(op, args)
		if !ok || rc != 2 {
			t.Fatalf("%s: rc=%d ok=%v, want rc=2 ok=true", op, rc, ok)
		}
	}
}

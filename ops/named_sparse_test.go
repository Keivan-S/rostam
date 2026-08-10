// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"reflect"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/vector"
)

// TestNamedInsertSparseAllDenseByteIdentical proves the per-space sparse framing is
// byte-identical to the pre-sparse encoder when NO sparse values are present, for
// every trailer combination (base, +keyTTL, +CAS, +both).
func TestNamedInsertSparseAllDenseByteIdentical(t *testing.T) {
	// ONE dense space so the base encoder's map iteration is deterministic (the
	// byte-identity claim is about the sparse block adding NOTHING when empty, not
	// about multi-space map ordering, which is non-deterministic in both encoders).
	vecs := map[string][]float32{"title": {1, 2, 3, 4}}
	meta := vector.Metadata{"a": vector.NewInt(1)}
	ttl := 2 * time.Second
	keyTTL := map[string]int64{"a": 1000}

	cases := []struct {
		name       string
		legacy     []byte
		withSparse []byte
	}{
		{
			"base",
			EncodeNamedInsertArgs("c", 7, vecs, meta, ttl),
			EncodeNamedInsertArgsSparseCASKeyTTL("c", 7, vecs, nil, meta, ttl, 0, false, nil),
		},
		{
			"keyttl",
			EncodeNamedInsertArgsKeyTTL("c", 7, vecs, meta, ttl, keyTTL),
			EncodeNamedInsertArgsSparseCASKeyTTL("c", 7, vecs, nil, meta, ttl, 0, false, keyTTL),
		},
		{
			"cas",
			EncodeNamedInsertArgsCAS("c", 7, vecs, meta, ttl, 5, true),
			EncodeNamedInsertArgsSparseCASKeyTTL("c", 7, vecs, nil, meta, ttl, 5, true, nil),
		},
		{
			"cas+keyttl",
			EncodeNamedInsertArgsCASKeyTTL("c", 7, vecs, meta, ttl, 5, true, keyTTL),
			EncodeNamedInsertArgsSparseCASKeyTTL("c", 7, vecs, nil, meta, ttl, 5, true, keyTTL),
		},
		{
			"empty-sparse-map",
			EncodeNamedInsertArgs("c", 7, vecs, meta, ttl),
			EncodeNamedInsertArgsSparseCASKeyTTL("c", 7, vecs, map[string]*vector.SparseVector{}, meta, ttl, 0, false, nil),
		},
	}
	for _, tc := range cases {
		if !bytes.Equal(tc.legacy, tc.withSparse) {
			t.Fatalf("%s: all-dense not byte-identical\n legacy=%v\n sparse=%v", tc.name, tc.legacy, tc.withSparse)
		}
	}
}

// TestNamedInsertSparseRoundtrip round-trips a mixed dense+sparse insert through the
// full sparse codec (with keyTTL + CAS trailers) and checks every field decodes.
func TestNamedInsertSparseRoundtrip(t *testing.T) {
	vecs := map[string][]float32{"title": {1, 2, 3, 4}}
	sparse := map[string]*vector.SparseVector{
		"terms": {Indices: []uint32{0, 3, 9}, Values: []float32{1.5, 2.5, 3.5}},
	}
	meta := vector.Metadata{"k": vector.NewString("v")}
	keyTTL := map[string]int64{"k": 500}
	args := EncodeNamedInsertArgsSparseCASKeyTTL("c", 11, vecs, sparse, meta, 3*time.Second, 9, true, keyTTL)

	col, id, gotVecs, gotSparse, gotMeta, gotTTL, exp, hasExp, gotKeyTTL, err := DecodeNamedInsertArgsSparseKeyTTL(args)
	if err != nil {
		t.Fatal(err)
	}
	if col != "c" || id != 11 || gotTTL != 3*time.Second {
		t.Fatalf("scalars: col=%q id=%d ttl=%v", col, id, gotTTL)
	}
	if !hasExp || exp != 9 {
		t.Fatalf("cas: exp=%d has=%v, want exp=9 has=true", exp, hasExp)
	}
	if !reflect.DeepEqual(gotVecs, vecs) {
		t.Fatalf("dense = %v, want %v", gotVecs, vecs)
	}
	if !reflect.DeepEqual(gotSparse, sparse) {
		t.Fatalf("sparse = %v, want %v", gotSparse, sparse)
	}
	if !reflect.DeepEqual(gotMeta, meta) {
		t.Fatalf("meta = %v, want %v", gotMeta, meta)
	}
	if !reflect.DeepEqual(gotKeyTTL, keyTTL) {
		t.Fatalf("keyTTL = %v, want %v", gotKeyTTL, keyTTL)
	}
}

// TestNamedInsertSparseLegacyDecode proves a legacy (dense-only) blob decodes with
// nil sparseVectors through the new sparse decoder (back-compat).
func TestNamedInsertSparseLegacyDecode(t *testing.T) {
	legacy := EncodeNamedInsertArgs("c", 1, map[string][]float32{"title": {1, 2, 3, 4}}, nil, 0)
	_, _, _, gotSparse, _, _, _, hasExp, _, err := DecodeNamedInsertArgsSparseKeyTTL(legacy)
	if err != nil {
		t.Fatal(err)
	}
	if gotSparse != nil {
		t.Fatalf("legacy decode: sparse = %v, want nil", gotSparse)
	}
	if hasExp {
		t.Fatalf("legacy decode: hasExpected true")
	}
}

// TestNamedSparseSearchArgsRoundtrip round-trips the sparse-search codec (with the
// rc/opa opts trailer) and the byte-identical-when-zero-rc property.
func TestNamedSparseSearchArgsRoundtrip(t *testing.T) {
	q := vector.SparseVector{Indices: []uint32{2, 5}, Values: []float32{1, 2}}
	filter := vector.Filter{Op: vector.FilterEq, Field: "k", Value: vector.NewString("v")}

	col, space, gotQ, k, gotF, rc, opa, _, err := DecodeNamedSparseSearchArgsOpts(
		EncodeNamedSparseSearchArgsOpts("c", "terms", q, 7, filter, 2, 1, 0))
	if err != nil {
		t.Fatal(err)
	}
	if col != "c" || space != "terms" || k != 7 || rc != 2 || opa != 1 {
		t.Fatalf("scalars: col=%q space=%q k=%d rc=%d opa=%d", col, space, k, rc, opa)
	}
	if !reflect.DeepEqual(gotQ, q) {
		t.Fatalf("query = %v, want %v", gotQ, q)
	}
	if !reflect.DeepEqual(gotF, filter) {
		t.Fatalf("filter = %v, want %v", gotF, filter)
	}
	// rc==0 && opa==0 ⇒ byte-identical to the no-opts base encoder.
	base := EncodeNamedSparseSearchArgs("c", "terms", q, 7, filter)
	withZero := EncodeNamedSparseSearchArgsOpts("c", "terms", q, 7, filter, 0, 0, 0)
	if !bytes.Equal(base, withZero) {
		t.Fatalf("zero-rc opts not byte-identical to base")
	}
}

// TestNamedSparseSearchArgsTruncated checks every prefix-truncation of a full
// sparse-search arg buffer is fail-loud.
func TestNamedSparseSearchArgsTruncated(t *testing.T) {
	full := EncodeNamedSparseSearchArgs("c", "terms",
		vector.SparseVector{Indices: []uint32{1, 2}, Values: []float32{3, 4}}, 5,
		vector.Filter{Op: vector.FilterEq, Field: "k", Value: vector.NewInt(1)})
	for chop := 0; chop < len(full); chop++ {
		if _, _, _, _, _, err := DecodeNamedSparseSearchArgs(full[:chop]); err == nil {
			t.Fatalf("chop %d: expected truncation error", chop)
		}
	}
}

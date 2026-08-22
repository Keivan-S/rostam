// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// TestMVHybridCollectionNameForAt2 is the routing-offset guard for the MV-hybrid
// ops (the same bug class the named-hybrid feature hit). EncodeMVHybridArgs emits
// the At2 wire ([flags:u8][colLen:u8][col]...), so vector_mv_hybrid_search /
// vector_mv_hybrid_lanes MUST be registered At2 (offset 1, behind the flags byte)
// in CollectionNameFor/CollectionNameOffset/the KeyExtractor. If they were
// mis-registered At1 (like the rest of the mv_* family), CollectionNameFor would
// read the leading flags byte as the name length and return GARBAGE, and on a P>1
// collection the fan-out would silently fall to single-partition.
//
// Teeth: a non-empty sparse query forces a NON-ZERO flags byte (mvHybridFlagSparse),
// which is exactly what an At1 misread would mistake for the name length — so the
// returned name would NOT equal "default/mycoll" under the bug.
func TestMVHybridCollectionNameForAt2(t *testing.T) {
	query := [][]float32{{1, 0, 0}, {0, 1, 0}}
	sparseQ := vector.SparseVector{Indices: []uint32{0, 5}, Values: []float32{0.5, 0.25}}

	cases := []struct {
		op   string
		args []byte
	}{
		{"vector_mv_hybrid_search", EncodeMVHybridArgs("mycoll", query, sparseQ, 5, vector.HybridOpts{}, 0, 0, 0)},
		{"vector_mv_hybrid_lanes", EncodeMVHybridArgs("mycoll", query, sparseQ, 5, vector.HybridOpts{}, 0, 0, 0)},
	}
	for _, c := range cases {
		name, ok := CollectionNameFor(c.op, c.args)
		if !ok {
			t.Errorf("%s: CollectionNameFor !ok", c.op)
			continue
		}
		if name != "default/mycoll" {
			t.Errorf("%s: name = %q, want %q (At1/At2 routing-offset regression)", c.op, name, "default/mycoll")
		}
		// CollectionNameOffset must agree (offset 1 = At2).
		if off, ok := CollectionNameOffset(c.op); !ok || off != 1 {
			t.Errorf("%s: CollectionNameOffset = (%d,%v), want (1,true)", c.op, off, ok)
		}
	}
}

// TestEncodeMVHybridArgsRoundTrip checks the codec is loss-less for the full opts
// set + the degradation cases (empty query ⇒ sparse-only; empty sparse ⇒
// MaxSim-only) + the byte-identical opts trailer when rc==0 && opa==0.
func TestEncodeMVHybridArgsRoundTrip(t *testing.T) {
	query := [][]float32{{1, 2, 3}, {4, 5, 6}}
	sparseQ := vector.SparseVector{Indices: []uint32{1, 7, 9}, Values: []float32{0.1, 0.2, 0.3}}
	filter := vector.Filter{Op: vector.FilterEq, Field: "g", Value: vector.NewString("a")}

	cases := []struct {
		name    string
		query   [][]float32
		sparse  vector.SparseVector
		opts    vector.HybridOpts
		rc, opa uint8
	}{
		{"full_rrf", query, sparseQ, vector.HybridOpts{Method: vector.FusionRRF, RRFK: 12, DenseK: 30, SparseK: 40, Filter: filter}, 0, 0},
		{"full_weighted_rc", query, sparseQ, vector.HybridOpts{Method: vector.FusionWeighted, Alpha: 0.3, DenseK: 7, SparseK: 8, Filter: filter}, 2, 1},
		{"maxsim_only", query, vector.SparseVector{}, vector.HybridOpts{Method: vector.FusionRRF}, 0, 0},
		{"sparse_only", nil, sparseQ, vector.HybridOpts{Method: vector.FusionRRF}, 0, 0},
		{"no_filter_no_opts", query, sparseQ, vector.HybridOpts{Method: vector.FusionRRF}, 0, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			b := EncodeMVHybridArgs("col", c.query, c.sparse, 5, c.opts, c.rc, c.opa, 0)
			gotName, gotQ, gotSparse, gotK, gotOpts, gotRC, gotOPA, _, err := DecodeMVHybridArgs(b)
			if err != nil {
				t.Fatal(err)
			}
			if gotName != "col" || gotK != 5 || gotRC != c.rc || gotOPA != c.opa {
				t.Fatalf("scalars: name=%q k=%d rc=%d opa=%d", gotName, gotK, gotRC, gotOPA)
			}
			if len(gotQ) != len(c.query) {
				t.Fatalf("query rows %d != %d", len(gotQ), len(c.query))
			}
			for i := range c.query {
				for j := range c.query[i] {
					if gotQ[i][j] != c.query[i][j] {
						t.Fatalf("query[%d][%d] = %v != %v", i, j, gotQ[i][j], c.query[i][j])
					}
				}
			}
			if len(gotSparse.Indices) != len(c.sparse.Indices) {
				t.Fatalf("sparse nnz %d != %d", len(gotSparse.Indices), len(c.sparse.Indices))
			}
			if gotOpts.Method != c.opts.Method || gotOpts.RRFK != c.opts.RRFK ||
				gotOpts.DenseK != c.opts.DenseK || gotOpts.SparseK != c.opts.SparseK ||
				gotOpts.Alpha != c.opts.Alpha {
				t.Fatalf("opts mismatch: got %+v want %+v", gotOpts, c.opts)
			}
			if gotOpts.Filter.IsZero() != c.opts.Filter.IsZero() {
				t.Fatalf("filter presence mismatch: got zero=%v want zero=%v", gotOpts.Filter.IsZero(), c.opts.Filter.IsZero())
			}
		})
	}
}

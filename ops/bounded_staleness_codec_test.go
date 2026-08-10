// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// boundedCodecCase pairs an op name with (a) its rc==0 bounded-vs-legacy
// byte-identity check and (b) its bound round-trip via ReadStalenessOf at rc==3.
type boundedCodecCase struct {
	op string
	// legacy builds the op's args with rc==0 && opa==0 && bound==0 via the LEGACY
	// (pre-bounded) encoder OR the Opts encoder with bound 0 — both must be equal.
	legacy func() []byte
	zero   func() []byte // Opts encoder, rc==0, opa==0, bound==0
	// bounded builds the op's args with rc==BoundedStaleness and the given bound.
	bounded func(bound uint64) []byte
}

// TestBoundedStalenessCodecRoundTrip is the per-op codec guard: for EVERY
// read op that carries rc, it proves (a) the rc==0 path is BYTE-IDENTICAL to the
// legacy encoder (zero added cost / wire-unchanged default), and (b) a rc==3 read's
// staleness bound round-trips through ReadStalenessOf (op-aware, no blind offset).
func TestBoundedStalenessCodecRoundTrip(t *testing.T) {
	q := []float32{1, 2, 3, 4}
	sp := vector.SparseVector{Indices: []uint32{0, 2}, Values: []float32{0.5, 0.25}}
	mtx := [][]float32{{1, 0}, {0, 1}}

	cases := []boundedCodecCase{
		{
			op:     "vector_search",
			legacy: func() []byte { return EncodeVectorSearchArgsExt("c", 5, q, vector.Filter{}) },
			zero:   func() []byte { return EncodeVectorSearchArgsOpts("c", 5, q, vector.Filter{}, 0, 0, 0) },
			bounded: func(b uint64) []byte {
				return EncodeVectorSearchArgsOpts("c", 5, q, vector.Filter{}, ConsistencyBoundedStaleness, 0, b)
			},
		},
		{
			op:     "vector_hybrid_search",
			legacy: func() []byte { return EncodeHybridSearchArgs("c", q, 5, sp, vector.HybridOpts{}) },
			zero:   func() []byte { return EncodeHybridSearchArgsOpts("c", q, 5, sp, vector.HybridOpts{}, 0, 0, 0) },
			bounded: func(b uint64) []byte {
				return EncodeHybridSearchArgsOpts("c", q, 5, sp, vector.HybridOpts{}, ConsistencyBoundedStaleness, 0, b)
			},
		},
		{
			op:     "vector_search_groups",
			legacy: func() []byte { return EncodeGroupSearchArgs("c", 5, q, vector.GroupOpts{GroupBy: "g"}) },
			zero:   func() []byte { return EncodeGroupSearchArgsOpts("c", 5, q, vector.GroupOpts{GroupBy: "g"}, 0, 0, 0) },
			bounded: func(b uint64) []byte {
				return EncodeGroupSearchArgsOpts("c", 5, q, vector.GroupOpts{GroupBy: "g"}, ConsistencyBoundedStaleness, 0, b)
			},
		},
		{
			op:     "vector_scroll",
			legacy: func() []byte { return EncodeScrollArgs("c", vector.Filter{}, 10) },
			zero:   func() []byte { return EncodeScrollArgsCursorBounded("c", vector.Filter{}, 10, 0, 0, 0, false, 0) },
			bounded: func(b uint64) []byte {
				return EncodeScrollArgsCursorBounded("c", vector.Filter{}, 10, ConsistencyBoundedStaleness, 0, 0, false, b)
			},
		},
		{
			op:     "vector_mv_search",
			legacy: func() []byte { return EncodeMVSearchArgs("c", mtx, 5, 3) },
			zero:   func() []byte { return EncodeMVSearchArgsOpts("c", mtx, 5, 3, 0, 0, 0) },
			bounded: func(b uint64) []byte {
				return EncodeMVSearchArgsOpts("c", mtx, 5, 3, ConsistencyBoundedStaleness, 0, b)
			},
		},
		{
			op:     "vector_named_search",
			legacy: func() []byte { return EncodeNamedSearchArgs("c", "v", q, 5, vector.Filter{}) },
			zero:   func() []byte { return EncodeNamedSearchArgsOpts("c", "v", q, 5, vector.Filter{}, 0, 0, 0) },
			bounded: func(b uint64) []byte {
				return EncodeNamedSearchArgsOpts("c", "v", q, 5, vector.Filter{}, ConsistencyBoundedStaleness, 0, b)
			},
		},
		{
			op:     "vector_named_sparse_search",
			legacy: func() []byte { return EncodeNamedSparseSearchArgs("c", "s", sp, 5, vector.Filter{}) },
			zero:   func() []byte { return EncodeNamedSparseSearchArgsOpts("c", "s", sp, 5, vector.Filter{}, 0, 0, 0) },
			bounded: func(b uint64) []byte {
				return EncodeNamedSparseSearchArgsOpts("c", "s", sp, 5, vector.Filter{}, ConsistencyBoundedStaleness, 0, b)
			},
		},
		{
			op:     "vector_named_hybrid_search",
			legacy: func() []byte { return EncodeNamedHybridArgs("c", "d", q, "s", sp, 5, vector.HybridOpts{}, 0, 0, 0) },
			zero:   func() []byte { return EncodeNamedHybridArgs("c", "d", q, "s", sp, 5, vector.HybridOpts{}, 0, 0, 0) },
			bounded: func(b uint64) []byte {
				return EncodeNamedHybridArgs("c", "d", q, "s", sp, 5, vector.HybridOpts{}, ConsistencyBoundedStaleness, 0, b)
			},
		},
		{
			op:     "vector_mv_hybrid_search",
			legacy: func() []byte { return EncodeMVHybridArgs("c", mtx, sp, 5, vector.HybridOpts{}, 0, 0, 0) },
			zero:   func() []byte { return EncodeMVHybridArgs("c", mtx, sp, 5, vector.HybridOpts{}, 0, 0, 0) },
			bounded: func(b uint64) []byte {
				return EncodeMVHybridArgs("c", mtx, sp, 5, vector.HybridOpts{}, ConsistencyBoundedStaleness, 0, b)
			},
		},
		{
			op:     "vector_named_scroll",
			legacy: func() []byte { return EncodeNamedScrollArgs("c", vector.Filter{}, 10) },
			zero:   func() []byte { return EncodeNamedScrollArgsOptsBounded("c", vector.Filter{}, 10, 0, false, 0, 0, 0) },
			bounded: func(b uint64) []byte {
				return EncodeNamedScrollArgsOptsBounded("c", vector.Filter{}, 10, 0, false, ConsistencyBoundedStaleness, 0, b)
			},
		},
		{
			op:     "vector_mv_scroll",
			legacy: func() []byte { return EncodeMVScrollArgs("c", vector.Filter{}, 10) },
			zero:   func() []byte { return EncodeMVScrollArgsOptsBounded("c", vector.Filter{}, 10, 0, 0, 0, false, 0) },
			bounded: func(b uint64) []byte {
				return EncodeMVScrollArgsOptsBounded("c", vector.Filter{}, 10, ConsistencyBoundedStaleness, 0, 0, false, b)
			},
		},
		{
			op:      "vector_get",
			legacy:  func() []byte { return EncodeVectorGetArgs("c", 7, 0) },
			zero:    func() []byte { return EncodeVectorGetArgsOpts("c", 7, 0, 0, 0, 0) },
			bounded: func(b uint64) []byte { return EncodeVectorGetArgsOpts("c", 7, 0, ConsistencyBoundedStaleness, 0, b) },
		},
		{
			op:      "vector_get_config",
			legacy:  func() []byte { return EncodeGetConfigArgs("c") },
			zero:    func() []byte { return EncodeGetConfigArgsOpts("c", 0, 0, 0) },
			bounded: func(b uint64) []byte { return EncodeGetConfigArgsOpts("c", ConsistencyBoundedStaleness, 0, b) },
		},
		{
			op:      "vector_named_get_config",
			legacy:  func() []byte { return EncodeNamedNameArgs("c") },
			zero:    func() []byte { return EncodeNamedNameArgsOpts("c", 0, 0, 0) },
			bounded: func(b uint64) []byte { return EncodeNamedNameArgsOpts("c", ConsistencyBoundedStaleness, 0, b) },
		},
		{
			op:      "vector_mv_get_config",
			legacy:  func() []byte { return EncodeMVGetConfigArgs("c") },
			zero:    func() []byte { return EncodeMVGetConfigArgsOpts("c", 0, 0, 0) },
			bounded: func(b uint64) []byte { return EncodeMVGetConfigArgsOpts("c", ConsistencyBoundedStaleness, 0, b) },
		},
	}

	bounds := []uint64{0, 1, 42, 1 << 32, ^uint64(0)}
	for _, tc := range cases {
		t.Run(tc.op, func(t *testing.T) {
			// (a) rc==0 byte-identical: legacy == Opts(rc=0,opa=0,bound=0).
			if !bytes.Equal(tc.legacy(), tc.zero()) {
				t.Fatalf("%s rc==0 NOT byte-identical:\n legacy=%x\n zero  =%x", tc.op, tc.legacy(), tc.zero())
			}
			// rc-classification still works at rc==0.
			if rc, ok := ReadConsistencyOf(tc.op, tc.zero()); !ok || rc != 0 {
				t.Fatalf("%s rc==0: ReadConsistencyOf=(%d,%v), want (0,true)", tc.op, rc, ok)
			}
			// At rc==0 there is no bound.
			if _, ok := ReadStalenessOf(tc.op, tc.zero()); ok {
				t.Fatalf("%s rc==0: ReadStalenessOf ok=true, want false", tc.op)
			}
			// (b) bound round-trips at rc==BoundedStaleness, op-aware.
			for _, b := range bounds {
				args := tc.bounded(b)
				if rc, ok := ReadConsistencyOf(tc.op, args); !ok || rc != ConsistencyBoundedStaleness {
					t.Fatalf("%s bound=%d: ReadConsistencyOf=(%d,%v), want (3,true)", tc.op, b, rc, ok)
				}
				got, ok := ReadStalenessOf(tc.op, args)
				if !ok || got != b {
					t.Fatalf("%s bound=%d: ReadStalenessOf=(%d,%v), want (%d,true)", tc.op, b, got, ok, b)
				}
			}
		})
	}
}

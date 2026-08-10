// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// TestReadConsistencyOfPerOp encodes each consistency-carrying read op with
// rc=0/1/2 and asserts the op-aware peek recovers the exact byte. Each sub-case
// uses the op's REAL encoder, so a layout drift in any encoder fails here.
func TestReadConsistencyOfPerOp(t *testing.T) {
	query := []float32{1, 2, 3}
	mvQuery := [][]float32{{1, 2}, {3, 4}}
	for _, rc := range []uint8{ConsistencyAnyReplica, ConsistencyLeaderOnly, ConsistencyLinearizable} {

		cases := []struct {
			op   string
			args []byte
		}{
			{"vector_search", EncodeVectorSearchArgsOpts("docs", 5, query, vector.Filter{}, rc, 0, 0)},
			{"vector_search_docs", EncodeVectorSearchArgsOpts("docs", 5, query, vector.Filter{}, rc, 0, 0)},
			{"vector_hybrid_search", EncodeHybridSearchArgsOpts("docs", query, 5, vector.SparseVector{}, vector.HybridOpts{}, rc, 0, 0)},
			// vector_hybrid_lanes / vector_group_candidates are the per-partition shard
			// ops the hybrid/group fan-outs scatter (same wire as their public names);
			// the barrier gate keys off these, so they MUST peek the rc byte.
			{"vector_hybrid_lanes", EncodeHybridSearchArgsOpts("docs", query, 5, vector.SparseVector{}, vector.HybridOpts{}, rc, 0, 0)},
			{"vector_search_groups", EncodeGroupSearchArgsOpts("docs", 5, query, vector.GroupOpts{}, rc, 0, 0)},
			{"vector_group_candidates", EncodeGroupSearchArgsOpts("docs", 5, query, vector.GroupOpts{}, rc, 0, 0)},
			{"vector_scroll", EncodeScrollArgsCursorBounded("docs", vector.Filter{}, 10, rc, 0, 0, false, 0)},
			{"vector_mv_search", EncodeMVSearchArgsOpts("docs", mvQuery, 5, 3, rc, 0, 0)},
			// Named search/_docs share one wire; named scroll carries rc in its
			// marker-bitfield trailer (no cursor here).
			{"vector_named_search", EncodeNamedSearchArgsOpts("docs", "title", query, 5, vector.Filter{}, rc, 0, 0)},
			{"vector_named_search_docs", EncodeNamedSearchArgsOpts("docs", "title", query, 5, vector.Filter{}, rc, 0, 0)},
			{"vector_named_scroll", EncodeNamedScrollArgsOptsBounded("docs", vector.Filter{}, 10, 0, false, rc, 0, 0)},
			// MV scroll carries rc in its marker-bitfield trailer (mirrors named scroll).
			{"vector_mv_scroll", EncodeMVScrollArgsOptsBounded("docs", vector.Filter{}, 10, rc, 0, 0, false, 0)},
		}
		for _, c := range cases {
			got, ok := ReadConsistencyOf(c.op, c.args)
			if !ok {
				t.Errorf("ReadConsistencyOf(%s, rc=%d): ok=false, want true", c.op, rc)
				continue
			}
			if got != rc {
				t.Errorf("ReadConsistencyOf(%s) = %d, want %d", c.op, got, rc)
			}
		}
	}
}

// TestReadConsistencyOfScrollWithCursor confirms the rc byte is recovered even
// when the scroll trailer also carries an afterID cursor (the trailer layout
// the fan-out actually emits on a resumed scroll).
func TestReadConsistencyOfScrollWithCursor(t *testing.T) {
	args := EncodeScrollArgsCursorBounded("docs", vector.Filter{}, 10, ConsistencyLinearizable, 1, 42, true, 0)
	got, ok := ReadConsistencyOf("vector_scroll", args)
	if !ok || got != ConsistencyLinearizable {
		t.Fatalf("ReadConsistencyOf(vector_scroll, cursor) = (%d,%v), want (2,true)", got, ok)
	}
}

// TestReadConsistencyOfLegacyArgs verifies backward-compat: args encoded by the
// legacy (no-opts-trailer) encoders peek as AnyReplica (0,true) — they are still
// consistency-carrying read ops, just defaulted — never panicking, never
// mis-reading a tail byte as a non-zero rc.
func TestReadConsistencyOfLegacyArgs(t *testing.T) {
	cases := []struct {
		op   string
		args []byte
	}{
		{"vector_search", EncodeVectorSearchArgs("docs", 5, []float32{1, 2, 3})},
		{"vector_scroll", EncodeScrollArgsCursorBounded("docs", vector.Filter{}, 10, 0, 0, 0, false, 0)},
		{"vector_mv_search", EncodeMVSearchArgsOpts("docs", [][]float32{{1, 2}}, 5, 3, 0, 0, 0)},
		{"vector_named_search", EncodeNamedSearchArgs("docs", "title", []float32{1, 2, 3}, 5, vector.Filter{})},
		{"vector_named_search_docs", EncodeNamedSearchArgs("docs", "title", []float32{1, 2, 3}, 5, vector.Filter{})},
		{"vector_named_scroll", EncodeNamedScrollArgsCursor("docs", vector.Filter{}, 10, 0, false)},
		{"vector_mv_scroll", EncodeMVScrollArgs("docs", vector.Filter{}, 10)},
	}
	for _, c := range cases {
		got, ok := ReadConsistencyOf(c.op, c.args)
		if !ok {
			t.Errorf("ReadConsistencyOf(%s, legacy) ok=false, want true", c.op)
		}
		if got != ConsistencyAnyReplica {
			t.Errorf("ReadConsistencyOf(%s, legacy) = %d, want 0 (AnyReplica)", c.op, got)
		}
	}
}

// TestReadConsistencyOfNonConsistencyOps asserts ops that carry NO
// read_consistency byte at this wire layer return (0,false): writes, admin reads,
// and unknown ops. (The named search/scroll family now carries the byte via its
// opts trailer and is covered by the per-op / legacy tables above.)
func TestReadConsistencyOfNonConsistencyOps(t *testing.T) {
	cases := []struct {
		op   string
		args []byte
	}{
		{"get", EncodeKeyArgs([]byte("k"))},
		{"put", EncodePutArgs([]byte("k"), []byte("v"), 0)},
		{"vector_insert", EncodeVectorInsertArgs("docs", 1, []float32{1, 2, 3})},
		{"totally_unknown_op", []byte{1, 2, 3}},
	}
	for _, c := range cases {
		if got, ok := ReadConsistencyOf(c.op, c.args); ok {
			t.Errorf("ReadConsistencyOf(%s) = (%d,true), want (_,false)", c.op, got)
		}
	}
}

// TestReadConsistencyOfAdversarialInputs feeds short/empty/garbage args to every
// covered op and asserts the peek NEVER panics and NEVER reports a stray
// Linearizable (which would gate a barrier on garbage). Short/garbage args yield
// (0,false): a decode error ⇒ no barrier; the handler fails the read moments later.
func TestReadConsistencyOfAdversarialInputs(t *testing.T) {
	ops := []string{
		"vector_search", "vector_search_docs", "vector_hybrid_search",
		"vector_search_groups", "vector_scroll", "vector_mv_search",
		"vector_mv_scroll", "vector_named_search", "vector_named_scroll", "get", "put",
	}
	inputs := [][]byte{
		nil,
		{},
		{0},
		{1},
		{255},
		{0, 0},
		{1, 2, 3, 4, 5},
		{2, 'a', 'b', 0xFF, 0xFF, 0xFF, 0xFF},
		{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF},
	}
	for _, op := range ops {
		for _, in := range inputs {
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("ReadConsistencyOf(%s, %v) panicked: %v", op, in, r)
					}
				}()
				rc, ok := ReadConsistencyOf(op, in)
				if ok && rc == ConsistencyLinearizable {
					t.Errorf("ReadConsistencyOf(%s, %v) classified garbage as Linearizable", op, in)
				}
			}()
		}
	}
}

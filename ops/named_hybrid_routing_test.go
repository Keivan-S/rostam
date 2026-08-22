// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// TestNamedHybridCollectionNameForAt2 is the regression guard for the routing bug
// fixed in 273d9b2: the named-hybrid ops emit the At2 wire
// ([flags:u8][colLen:u8][col]...) like dense hybrid, but were registered At1
// (col-first) in CollectionNameFor/CollectionNameOffset/the KeyExtractor. With the
// At1 mapping, CollectionNameFor read the leading flags byte as the name length,
// producing a GARBAGE name; on a P>1 collection the fan-out dispatcher then failed
// to recognise the collection and fell through to single-partition results.
//
// This asserts CollectionNameFor returns the EXACT canonical name for both
// named-hybrid ops (offset 1, behind the flags byte). vector_named_sparse_search
// is included for contrast: it is a genuine At1 op (col-first, no flags byte) and
// must ALSO resolve correctly — confirming the test pins the per-op layout, not a
// blanket "everything is At2".
//
// Teeth: the encoder writes a non-zero flags byte (namedHybridFlagSparse is set
// because the sparse query is non-empty). Under the buggy At1 mapping the u8 at
// offset 0 (the flags byte) would be misread as the name length, so the returned
// name would NOT equal "default/mycoll" — this test would fail. With the At2 fix it
// passes.
func TestNamedHybridCollectionNameForAt2(t *testing.T) {
	denseQ := []float32{1, 2, 3, 4}
	// A non-empty sparse query forces a non-zero flags byte (namedHybridFlagSparse),
	// which is exactly what the At1 misread would mistake for the name length.
	sparseQ := vector.SparseVector{Indices: []uint32{0, 5}, Values: []float32{0.5, 0.25}}

	cases := []struct {
		op   string
		args []byte
	}{
		// At2 layout ([flags:u8][colLen:u8][col]...): the two ops the fix moved.
		{"vector_named_hybrid_search", EncodeNamedHybridArgs("mycoll", "dense", denseQ, "sparse", sparseQ, 5, vector.HybridOpts{}, 0, 0, 0)},
		{"vector_named_hybrid_lanes", EncodeNamedHybridArgs("mycoll", "dense", denseQ, "sparse", sparseQ, 5, vector.HybridOpts{}, 0, 0, 0)},
		// At1 layout (col-first, no flags byte) — contrast/coverage: a genuinely
		// col-first named op must still resolve to its canonical name.
		{"vector_named_sparse_search", EncodeNamedSparseSearchArgs("mycoll", "sparse", sparseQ, 5, vector.Filter{})},
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
	}
}

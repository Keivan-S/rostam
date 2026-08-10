// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// TestEmbeddedMVAddSparseWireSucceeds drives the full MV add wire with an OPTIONAL
// doc-level sparse vector through the embedded op path (encode trailer → handler →
// engine) and confirms the add succeeds and the document is dense-searchable. The
// deep sparse-storage + persistence assertions live at the vector/ + ops/ layers;
// this pins that the WriteOpts.Sparse field threads end-to-end without error and the
// MaxSim lane is unaffected.
func TestEmbeddedMVAddSparseWireSucceeds(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	const coll = "mv"
	if err := s.VectorMVCreateCollection(ctx, coll, MultiVectorConfig{Dim: 4}); err != nil {
		t.Fatalf("create: %v", err)
	}

	tokens := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}
	sp := &vector.SparseVector{Indices: []uint32{0, 3}, Values: []float32{1.5, 2.0}}
	if err := s.VectorMVAdd(ctx, coll, 1, tokens, VectorMetadata{"id": vector.NewInt(1)}, WriteOpts{Sparse: sp}); err != nil {
		t.Fatalf("mv add with sparse: %v", err)
	}
	// A dense-only add still works alongside (no sparse trailer).
	if err := s.VectorMVAdd(ctx, coll, 2, tokens, VectorMetadata{"id": vector.NewInt(2)}); err != nil {
		t.Fatalf("dense-only mv add: %v", err)
	}

	// MaxSim search is untouched: doc 1 retrievable by its dense tokens.
	res, _, err := s.VectorMVSearch(ctx, coll, [][]float32{{1, 0, 0, 0}}, 2, MultiSearchOpts{CandidatesPerToken: 50})
	if err != nil {
		t.Fatalf("mv search: %v", err)
	}
	if len(res) == 0 {
		t.Fatal("mv search returned no results after sparse-bearing add")
	}

	// A malformed sparse vector (unsorted indices) is rejected at the engine.
	bad := &vector.SparseVector{Indices: []uint32{3, 0}, Values: []float32{1, 2}}
	if err := s.VectorMVAdd(ctx, coll, 3, tokens, nil, WriteOpts{Sparse: bad}); err == nil {
		t.Fatal("expected error for unsorted doc sparse")
	}
}

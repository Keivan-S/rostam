// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/vector"
)

// mvSparseLaneHit reports whether the per-doc sparse lane returns docID for a
// sparse-only MV hybrid query (empty MaxSim query ⇒ sparse-only). The sparse lane
// only knows about a doc if its per-doc sparse vector survived storage — so this
// doubles as a "the sparse vector is still present" probe.
func mvSparseLaneHit(t *testing.T, s rostam.Store, coll string, q vector.SparseVector, docID uint64) bool {
	t.Helper()
	res, err := s.VectorMVHybridSearch(context.Background(), coll, nil, q, 10, rostam.MVHybridOpts{})
	if err != nil {
		t.Fatalf("VectorMVHybridSearch %s: %v", coll, err)
	}
	for _, r := range res {
		if r.ID == docID {
			return true
		}
	}
	return false
}

// TestMVOnlineReshardPreservesSparse is the doc-level-sparse preservation gate for
// the ONLINE MV reshard path (VectorMVReshard → mvReshardCopyPass →
// vector_mv_add_if_absent with the sparse trailer → MultiAddIfAbsentVersionSparse).
// Before the fix the MV scan codec carried no Sparse field, so the scan→reinsert
// copy SILENTLY DROPPED the per-doc sparse vector and the sparse lane went dark for
// the resharded doc. revert-fails-it: reverting the scan-populate or the copy-pass
// r.Sparse makes the post-reshard sparse-lane probe miss the doc.
func TestMVOnlineReshardPreservesSparse(t *testing.T) {
	defer rostam.SetReshardDrainGrace(20 * time.Millisecond)()

	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	must(t, s.VectorMVCreateCollection(ctx, "mvsp", rostam.MultiVectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: 2}))

	sv := vector.SparseVector{Indices: []uint32{2, 5, 11}, Values: []float32{0.9, 0.4, 0.7}}
	// doc 1 carries token vectors + a doc-level sparse vector; doc 2 is dense-only.
	must(t, s.VectorMVAdd(ctx, "mvsp", 1, [][]float32{{1, 0, 0, 0}}, vector.Metadata{"k": vector.NewString("a")},
		rostam.WriteOpts{Sparse: &sv}))
	must(t, s.VectorMVAdd(ctx, "mvsp", 2, [][]float32{{0, 1, 0, 0}}, vector.Metadata{"k": vector.NewString("b")}, rostam.WriteOpts{}))

	if !mvSparseLaneHit(t, s, "mvsp", sv, 1) {
		t.Fatal("pre-reshard: sparse lane missed doc 1 (setup wrong)")
	}

	// ONLINE MV reshard 2 -> 4: the copy pass must carry doc 1's sparse vector.
	must(t, s.VectorMVReshard(ctx, "mvsp", 4))

	if !mvSparseLaneHit(t, s, "mvsp", sv, 1) {
		t.Fatal("post-reshard: sparse lane missed doc 1 — the reshard copy dropped the doc sparse vector")
	}
	// doc 2 (dense-only) still retrievable, unaffected.
	found, _, _, err := s.VectorMVGet(ctx, "mvsp", 2, false, true)
	must(t, err)
	if !found {
		t.Fatal("post-reshard: dense-only doc 2 lost")
	}
}

// TestMVOfflineResplitPreservesSparse mirrors the online gate for the OFFLINE MV
// path (VectorMVResplit → vector_mv_add_versioned with the sparse trailer →
// MultiRestoreAddSparse → restoreAdd verbatim).
func TestMVOfflineResplitPreservesSparse(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	must(t, s.VectorMVCreateCollection(ctx, "mvspr", rostam.MultiVectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: 2}))

	sv := vector.SparseVector{Indices: []uint32{1, 7, 13}, Values: []float32{0.5, 0.8, 0.3}}
	must(t, s.VectorMVAdd(ctx, "mvspr", 1, [][]float32{{1, 0, 0, 0}}, vector.Metadata{"k": vector.NewString("a")},
		rostam.WriteOpts{Sparse: &sv}))

	if !mvSparseLaneHit(t, s, "mvspr", sv, 1) {
		t.Fatal("pre-resplit: sparse lane missed doc 1 (setup wrong)")
	}

	// OFFLINE MV resplit 2 -> 4.
	must(t, s.VectorMVResplit(ctx, "mvspr", 4))

	if !mvSparseLaneHit(t, s, "mvspr", sv, 1) {
		t.Fatal("post-resplit: sparse lane missed doc 1 — the resplit copy dropped the doc sparse vector")
	}
}

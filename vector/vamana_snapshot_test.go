// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"testing"
)

// nonDefaultVamanaCfg is a Vamana config with R/L/alpha ALL set away from the
// package defaults (defaultVamanaR=64, defaultVamanaL=100, defaultVamanaAlpha=1.2).
// A snapshot/restore path that drops the geometry would re-presize the level-0 slab
// at the wrong stride (effectiveM0⇒2*M instead of VamanaR) and corrupt every
// neighbor list — these tests are designed to catch exactly that.
func nonDefaultVamanaCfg(dim int, seed int64) Config {
	return Config{
		Dim: dim, Seed: seed, Metric: Cosine, IndexType: IndexVamana,
		VamanaR: 40, VamanaL: 90, VamanaAlpha: 1.4,
	}
}

// TestVamanaSnapshotRestoreIdenticalGraph is the single-node round-trip: build an
// IndexVamana graph with NON-DEFAULT VamanaR, Snapshot it, Restore into a fresh
// newVamana index, and assert the restored graph is BIT-IDENTICAL (same medoid
// entry point, same per-slot neighbor lists) and returns identical search results.
//
// If snapshot/restore dropped IndexType/VamanaR, the restore would re-presize the
// slab with stride 2*M (=32 here, R is 40) instead of VamanaR, scrambling the
// neighbor lists — this test would then fail on the neighbor-list comparison.
func TestVamanaSnapshotRestoreIdenticalGraph(t *testing.T) {
	const (
		n    = 1_500
		dim  = 48
		nq   = 30
		k    = 10
		seed = 11
	)
	corpus, queries, gt := vamanaClusteredCorpus(t, n, dim, nq, 24, k, seed, 0.15)

	src, err := newVamana(nonDefaultVamanaCfg(dim, seed))
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	if err := src.BuildConcurrent(ids, corpus, 0); err != nil {
		t.Fatalf("build: %v", err)
	}

	var blob bytes.Buffer
	if err := src.Snapshot(&blob); err != nil {
		t.Fatalf("snapshot: %v", err)
	}

	// Restore into a fresh Vamana index (mirrors the real path: newIndex→newVamana
	// then Collection.Restore). The restored index must reconstruct the SAME graph.
	dst, err := newVamana(nonDefaultVamanaCfg(dim, seed))
	if err != nil {
		t.Fatal(err)
	}
	if err := dst.Restore(bytes.NewReader(blob.Bytes())); err != nil {
		t.Fatalf("restore: %v", err)
	}

	// The restored index must keep the Vamana geometry: single-layer (mL=0),
	// slab stride m0=VamanaR, pruneAlpha=VamanaAlpha, vamana flag, and the cfg
	// params (so a subsequent re-snapshot/re-restore stays correct).
	if dst.mL != 0 {
		t.Fatalf("restored mL = %v, want 0 (single-layer Vamana)", dst.mL)
	}
	if dst.m0 != 40 {
		t.Fatalf("restored m0 = %d, want 40 (VamanaR)", dst.m0)
	}
	if dst.pruneAlpha != 1.4 {
		t.Fatalf("restored pruneAlpha = %v, want 1.4 (VamanaAlpha)", dst.pruneAlpha)
	}
	if !dst.vamana {
		t.Fatal("restored index lost the vamana flag")
	}
	if dst.cfg.IndexType != IndexVamana || dst.cfg.VamanaR != 40 || dst.cfg.VamanaL != 90 || dst.cfg.VamanaAlpha != 1.4 {
		t.Fatalf("restored cfg dropped Vamana params: type=%d R=%d L=%d alpha=%v",
			dst.cfg.IndexType, dst.cfg.VamanaR, dst.cfg.VamanaL, dst.cfg.VamanaAlpha)
	}

	// Same medoid entry point.
	if dst.entryPoint != src.entryPoint {
		t.Fatalf("entry point differs after restore: %d vs %d", dst.entryPoint, src.entryPoint)
	}

	// Bit-identical per-slot neighbor lists.
	for slot := 0; slot < n; slot++ {
		sa, sb := src.nodes[slot], dst.nodes[slot]
		if sa == nil || sb == nil {
			t.Fatalf("slot %d: nil node (src=%v dst=%v)", slot, sa == nil, sb == nil)
		}
		la := src.nbrsAt(sa, 0)
		lb := dst.nbrsAt(sb, 0)
		if len(la) != len(lb) {
			t.Fatalf("slot %d: neighbor count %d vs %d", slot, len(la), len(lb))
		}
		for i := range la {
			if la[i] != lb[i] {
				t.Fatalf("slot %d: neighbor %d differs: %d vs %d", slot, i, la[i], lb[i])
			}
		}
	}

	// Identical search results (id + order) for every query.
	for qi, q := range queries {
		ra, err := src.Search(q, k)
		if err != nil {
			t.Fatalf("src search: %v", err)
		}
		rb, err := dst.Search(q, k)
		if err != nil {
			t.Fatalf("dst search: %v", err)
		}
		if len(ra) != len(rb) {
			t.Fatalf("query %d: result count %d vs %d", qi, len(ra), len(rb))
		}
		for i := range ra {
			if ra[i].ID != rb[i].ID {
				t.Fatalf("query %d result %d: id %d vs %d", qi, i, ra[i].ID, rb[i].ID)
			}
		}
	}
	_ = gt // ground truth not needed; the bit-identical assertion is stronger than recall.
}

// TestVamanaStoreSnapshotRaftRoundTrip is the cluster-facing round-trip: an
// IndexVamana collection with NON-DEFAULT R/L/alpha is captured via SnapshotAll
// (the Raft FSM snapshot) and rebuilt via RestoreAll on a fresh store (simulating
// Raft InstallSnapshot / new-node catch-up). RestoreAll calls
// NewCollection(name, sc.toConfig())→newVamana, so the restored collection's
// geometry is whatever snapColCfg carried.
//
// This test would FAIL if snapColCfg dropped VamanaR/VamanaL/VamanaAlpha: toConfig
// would yield VamanaR=0 ⇒ effectiveVamanaR=defaultVamanaR=64 (≠ the built 40), the
// restored slab stride m0 would be 64 not 40, and the neighbor lists would diverge
// from the source — the per-slot comparison below catches it.
func TestVamanaStoreSnapshotRaftRoundTrip(t *testing.T) {
	const (
		n    = 1_200
		dim  = 32
		nq   = 25
		k    = 10
		seed = 23
	)
	corpus, queries, _ := vamanaClusteredCorpus(t, n, dim, nq, 16, k, seed, 0.15)

	src, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	cfg := nonDefaultVamanaCfg(dim, seed)
	if err := src.CreateCollection("vam", cfg); err != nil {
		t.Fatal(err)
	}
	for i, v := range corpus {
		if err := src.Insert("vam", uint64(i+1), v, 0, nil, nil); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	srcCol, ok := src.Get("vam")
	if !ok {
		t.Fatal("source collection missing")
	}
	srcIdx := srcCol.idx.(*hnsw)

	var blob bytes.Buffer
	if err := src.SnapshotAll(&blob); err != nil {
		t.Fatalf("SnapshotAll: %v", err)
	}
	// Reference search results from the source.
	refResults := make([][]Result, nq)
	for qi, q := range queries {
		res, err := srcCol.Search(q, k)
		if err != nil {
			t.Fatalf("src search: %v", err)
		}
		refResults[qi] = res
	}
	_ = src.Close()

	// Restore onto a fresh store (Raft InstallSnapshot / new-node catch-up).
	dst, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = dst.Close() }()
	if err := dst.RestoreAll(bytes.NewReader(blob.Bytes())); err != nil {
		t.Fatalf("RestoreAll: %v", err)
	}

	dstCol, ok := dst.Get("vam")
	if !ok {
		t.Fatal("collection missing after RestoreAll")
	}

	// The restored config must carry the Vamana geometry (the snapColCfg fix). If
	// dropped, R/L/alpha would be zero here.
	if dstCol.cfg.IndexType != IndexVamana {
		t.Fatalf("restored IndexType = %d, want IndexVamana", dstCol.cfg.IndexType)
	}
	if dstCol.cfg.VamanaR != 40 || dstCol.cfg.VamanaL != 90 || dstCol.cfg.VamanaAlpha != 1.4 {
		t.Fatalf("restored cfg dropped Vamana params: R=%d L=%d alpha=%v",
			dstCol.cfg.VamanaR, dstCol.cfg.VamanaL, dstCol.cfg.VamanaAlpha)
	}

	dstIdx := dstCol.idx.(*hnsw)
	if dstIdx.m0 != 40 {
		t.Fatalf("restored slab stride m0 = %d, want 40 (VamanaR) — snapColCfg dropped geometry", dstIdx.m0)
	}
	if dstIdx.mL != 0 || !dstIdx.vamana {
		t.Fatalf("restored index not single-layer Vamana: mL=%v vamana=%v", dstIdx.mL, dstIdx.vamana)
	}
	if dstIdx.entryPoint != srcIdx.entryPoint {
		t.Fatalf("entry point differs after RestoreAll: %d vs %d", dstIdx.entryPoint, srcIdx.entryPoint)
	}

	// Bit-identical neighbor lists: the strongest proof the geometry rebuilt
	// correctly through the cluster path.
	for slot := 0; slot < n; slot++ {
		sa, sb := srcIdx.nodes[slot], dstIdx.nodes[slot]
		if sa == nil || sb == nil {
			t.Fatalf("slot %d: nil node (src=%v dst=%v)", slot, sa == nil, sb == nil)
		}
		la := srcIdx.nbrsAt(sa, 0)
		lb := dstIdx.nbrsAt(sb, 0)
		if len(la) != len(lb) {
			t.Fatalf("slot %d: neighbor count %d vs %d (wrong slab stride?)", slot, len(la), len(lb))
		}
		for i := range la {
			if la[i] != lb[i] {
				t.Fatalf("slot %d: neighbor %d differs: %d vs %d", slot, i, la[i], lb[i])
			}
		}
	}

	// Search works and matches the source.
	for qi, q := range queries {
		got, err := dstCol.Search(q, k)
		if err != nil {
			t.Fatalf("dst search: %v", err)
		}
		want := refResults[qi]
		if len(got) != len(want) {
			t.Fatalf("query %d: result count %d vs %d", qi, len(got), len(want))
		}
		for i := range got {
			if got[i].ID != want[i].ID {
				t.Fatalf("query %d result %d: id %d vs %d", qi, i, got[i].ID, want[i].ID)
			}
		}
	}
}

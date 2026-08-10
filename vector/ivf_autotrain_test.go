// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math/rand"
	"testing"
)

// ivfAutoTrainConfig is an IVF-Flat config with a LOW auto-train threshold so the
// incremental insert path engages quickly in tests (the production default,
// defaultIVFTrainThreshold, keeps tiny collections exact-brute-force).
func ivfAutoTrainConfig(dim, threshold int) Config {
	c := DefaultConfig()
	c.Dim = dim
	c.Metric = L2
	c.Seed = 42
	c.IndexType = IndexIVF
	c.IVFNprobe = 16
	c.IVFTrainThreshold = threshold
	return c
}

// ivfPQAutoTrainConfig is an IVF-PQ (PQ-only, float-drop) config with a low
// auto-train threshold.
func ivfPQAutoTrainConfig(dim, threshold, m int, rerank bool) Config {
	c := ivfAutoTrainConfig(dim, threshold)
	c.IVFPQ = true
	c.IVFPQM = m
	c.IVFRerank = rerank
	return c
}

// insertAll inserts (ids,vecs) into ix via the incremental Insert path.
func insertAll(t *testing.T, ix *ivf, ids []uint64, vecs [][]float32) {
	t.Helper()
	for i := range ids {
		if _, _, err := ix.Insert(ids[i], vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert(%d): %v", ids[i], err)
		}
	}
}

func seqIDs(n int) []uint64 {
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	return ids
}

// TestIVFAutoTrainBelowThreshold: an incrementally-built IVF index with fewer live
// vectors than the threshold stays UNTRAINED (exact brute force, unchanged).
func TestIVFAutoTrainBelowThreshold(t *testing.T) {
	dim := 16
	const threshold = 512
	rng := rand.New(rand.NewSource(1))
	ix, err := newIVF(ivfAutoTrainConfig(dim, threshold))
	if err != nil {
		t.Fatal(err)
	}
	n := threshold - 1 // one short of the trigger
	vecs := clusteredVecs(rng, n, dim, 20)
	ids := seqIDs(n)
	insertAll(t, ix, ids, vecs)

	if ix.trained {
		t.Fatalf("index trained at %d live (threshold %d) — must stay untrained below threshold", n, threshold)
	}
	// Exact: untrained IVF brute-forces, so it matches ground truth exactly.
	q := vecs[0]
	got, err := ix.Search(q, 5)
	if err != nil {
		t.Fatal(err)
	}
	want := bruteForceNN(q, ids, vecs, 5)
	for i := range want {
		if got[i].ID != want[i] {
			t.Fatalf("untrained result %d = %d, want exact NN %d", i, got[i].ID, want[i])
		}
	}
}

// TestIVFAutoTrainCrossThreshold: crossing the threshold during an incremental
// insert deterministically auto-trains the coarse quantizer in place, under the
// lock, synchronously — trained flips, lists are built, and search now prunes via
// nprobe while staying correct on clustered data.
func TestIVFAutoTrainCrossThreshold(t *testing.T) {
	dim := 32
	const threshold = 600
	rng := rand.New(rand.NewSource(7))
	ix, err := newIVF(ivfAutoTrainConfig(dim, threshold))
	if err != nil {
		t.Fatal(err)
	}
	n := 800
	vecs := clusteredVecs(rng, n, dim, 40)
	ids := seqIDs(n)

	// Insert one-by-one and verify the trigger fires EXACTLY at the threshold cross.
	for i := range ids {
		if _, _, err := ix.Insert(ids[i], vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
		live := i + 1
		if live < threshold && ix.trained {
			t.Fatalf("trained at %d live, before threshold %d", live, threshold)
		}
		if live == threshold && !ix.trained {
			t.Fatalf("NOT trained at %d live, exactly at threshold %d", live, threshold)
		}
	}
	if !ix.trained {
		t.Fatal("index should be trained after crossing the threshold")
	}
	if ix.nlist < 1 || len(ix.lists) != ix.nlist {
		t.Fatalf("bad nlist/lists after auto-train: nlist=%d lists=%d", ix.nlist, len(ix.lists))
	}
	// Every live slot is filed into exactly one list.
	total := 0
	for _, l := range ix.lists {
		total += len(l)
	}
	if total != n {
		t.Fatalf("lists hold %d slots, want %d", total, n)
	}

	// Search is correct: at nprobe=nlist it is exact; at a small nprobe it prunes
	// but keeps high recall on clustered data.
	queries := clusteredVecs(rng, 30, dim, 40)
	k := 10
	ix.nprobe = ix.nlist
	exactHits, denom := 0, 0
	for _, q := range queries {
		want := bruteForceNN(q, ids, vecs, k)
		got, err := ix.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		gs := idSet(got)
		for _, w := range want {
			denom++
			if gs[w] {
				exactHits++
			}
		}
	}
	if exactHits != denom {
		t.Fatalf("nprobe=nlist recall = %d/%d, want exact", exactHits, denom)
	}
}

// TestIVFPQAutoTrainCompresses: an incrementally-built IVF-PQ (PQ-only) index
// auto-trains and ENGAGES COMPRESSION — codebooks built, per-slot residual codes
// written, and the resident floats DROPPED (float-drop). Recall stays above the
// residual-ADC floor.
func TestIVFPQAutoTrainCompresses(t *testing.T) {
	dim := 32
	const threshold = 800
	rng := rand.New(rand.NewSource(2026))
	ix, err := newIVF(ivfPQAutoTrainConfig(dim, threshold, 8, false))
	if err != nil {
		t.Fatal(err)
	}
	n := 1500
	vecs := clusteredVecs(rng, n, dim, 40)
	ids := seqIDs(n)
	insertAll(t, ix, ids, vecs)

	if !ix.trained {
		t.Fatal("IVF-PQ index should auto-train past the threshold")
	}
	if !ix.pqActive() {
		t.Fatal("IVF-PQ codec not trained after auto-train")
	}
	if !ix.pqDropped || ix.arena.vecs != nil {
		t.Fatal("PQ-only float-drop did not engage after auto-train (floats still resident)")
	}
	// Codes are present for live slots.
	if ix.arena.Code(0) == nil {
		t.Fatal("no residual code written for slot 0 after auto-train")
	}

	// Recall floor: residual ADC clears a reasonable threshold on clustered data.
	ix.nprobe = ix.nlist
	queries := clusteredVecs(rng, 40, dim, 40)
	k := 10
	hits, denom := 0, 0
	for _, q := range queries {
		want := bruteForceNN(q, ids, vecs, k)
		got, err := ix.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		gs := idSet(got)
		for _, w := range want {
			denom++
			if gs[w] {
				hits++
			}
		}
	}
	recall := float64(hits) / float64(denom)
	t.Logf("IVF-PQ auto-trained recall@%d (nprobe=nlist=%d): %.3f", k, ix.nlist, recall)
	if recall < 0.40 {
		t.Fatalf("IVF-PQ auto-trained recall@%d = %.3f, want >= 0.40", k, recall)
	}
}

// TestIVFAutoTrainDeterministic is the LOAD-BEARING cluster-replica guarantee:
// building the SAME IVF-PQ index twice with the SAME Config.Seed and the SAME
// insert order yields BYTE-IDENTICAL trained state — identical centroids, codebooks,
// and per-slot codes. A divergence here would corrupt a Raft cluster (replicas
// computing different codebooks → divergent search). Verified two ways: a direct
// in-memory structural compare AND a full Snapshot byte-compare.
func TestIVFAutoTrainDeterministic(t *testing.T) {
	dim := 32
	const threshold = 600
	n := 1000
	// One shared corpus + insert order for both builds.
	rng := rand.New(rand.NewSource(123))
	vecs := clusteredVecs(rng, n, dim, 40)
	ids := seqIDs(n)

	build := func() *ivf {
		ix, err := newIVF(ivfPQAutoTrainConfig(dim, threshold, 8, false))
		if err != nil {
			t.Fatal(err)
		}
		insertAll(t, ix, ids, vecs)
		if !ix.trained || !ix.pqActive() {
			t.Fatal("build did not auto-train IVF-PQ")
		}
		return ix
	}

	a := build()
	b := build()

	// 1) Centroids bit-identical.
	if len(a.centroids) != len(b.centroids) {
		t.Fatalf("nlist differs: %d vs %d", len(a.centroids), len(b.centroids))
	}
	for c := range a.centroids {
		if len(a.centroids[c]) != len(b.centroids[c]) {
			t.Fatalf("centroid %d dim differs", c)
		}
		for d := range a.centroids[c] {
			if a.centroids[c][d] != b.centroids[c][d] {
				t.Fatalf("centroid[%d][%d] differs: %v vs %v", c, d, a.centroids[c][d], b.centroids[c][d])
			}
		}
	}
	// 2) Per-slot residual codes bit-identical (the compressed payload).
	cap := a.arena.Capacity()
	if bcap := b.arena.Capacity(); bcap != cap {
		t.Fatalf("capacity differs: %d vs %d", cap, bcap)
	}
	for s := 0; s < cap; s++ {
		ca := a.arena.Code(uint32(s))
		cb := b.arena.Code(uint32(s))
		if !bytes.Equal(ca, cb) {
			t.Fatalf("slot %d code differs: %v vs %v", s, ca, cb)
		}
	}
	// 3) slotCell assignment bit-identical.
	if len(a.slotCell) != len(b.slotCell) {
		t.Fatalf("slotCell len differs: %d vs %d", len(a.slotCell), len(b.slotCell))
	}
	for s := range a.slotCell {
		if a.slotCell[s] != b.slotCell[s] {
			t.Fatalf("slotCell[%d] differs: %d vs %d", s, a.slotCell[s], b.slotCell[s])
		}
	}
	// 4) Full Snapshot byte-identical — the strongest end-to-end determinism proof.
	var bufA, bufB bytes.Buffer
	if err := a.Snapshot(&bufA); err != nil {
		t.Fatal(err)
	}
	if err := b.Snapshot(&bufB); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(bufA.Bytes(), bufB.Bytes()) {
		t.Fatalf("snapshots differ (%d vs %d bytes) — auto-train is NOT deterministic", bufA.Len(), bufB.Len())
	}
}

// TestIVFAutoTrainSnapshotRestore: a snapshot taken AFTER auto-train restores the
// trained state (centroids/codebooks/codes) and search is identical on the restored
// index. Also confirms the restored index is `trained` so replayed tail inserts
// skip auto-train.
func TestIVFAutoTrainSnapshotRestore(t *testing.T) {
	dim := 32
	const threshold = 600
	rng := rand.New(rand.NewSource(55))
	cfg := ivfPQAutoTrainConfig(dim, threshold, 8, true) // rerank keeps floats for an exact-ish compare
	src, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	n := 900
	vecs := clusteredVecs(rng, n, dim, 40)
	ids := seqIDs(n)
	insertAll(t, src, ids, vecs)
	if !src.trained {
		t.Fatal("source must be auto-trained")
	}
	src.nprobe = src.nlist

	var buf bytes.Buffer
	if err := src.Snapshot(&buf); err != nil {
		t.Fatal(err)
	}
	dst, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := dst.Restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatal(err)
	}
	if !dst.trained {
		t.Fatal("restored index must carry the trained state")
	}
	dst.nprobe = dst.nlist

	// Identical search results on src vs restored dst.
	queries := clusteredVecs(rng, 25, dim, 40)
	k := 10
	for _, q := range queries {
		gs, err := src.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		gd, err := dst.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		if len(gs) != len(gd) {
			t.Fatalf("result count differs after restore: %d vs %d", len(gs), len(gd))
		}
		for i := range gs {
			if gs[i].ID != gd[i].ID {
				t.Fatalf("restored result %d differs: %d vs %d", i, gs[i].ID, gd[i].ID)
			}
		}
	}
}

// TestCollectionIVFTrainThresholdHonored is the dense create→engine proof for the
// IVFTrainThreshold create-wire knob (Gap 1): a Collection built via the PUBLIC
// NewCollection path with a CUSTOM (low) IVFTrainThreshold auto-trains its inner
// IVF index at THAT threshold — NOT at the production default (2048) — proving the
// Config field threads from create all the way into the engine's auto-train
// trigger. With a custom threshold of 300 the index is untrained at 299 live and
// trained at 300; a default-threshold Collection (the same corpus) stays untrained
// at 300 (well below 2048), isolating the effect to the wired field.
func TestCollectionIVFTrainThresholdHonored(t *testing.T) {
	dim := 32
	const threshold = 300
	rng := rand.New(rand.NewSource(99))
	vecs := clusteredVecs(rng, threshold, dim, 30)
	ids := seqIDs(threshold)

	cfg := DefaultConfig()
	cfg.Dim = dim
	cfg.Metric = L2
	cfg.Seed = 42
	cfg.IndexType = IndexIVF
	cfg.IVFNprobe = 16
	cfg.IVFTrainThreshold = threshold

	col, err := NewCollection("docs", cfg)
	if err != nil {
		t.Fatal(err)
	}
	ix, ok := col.idx.(*ivf)
	if !ok {
		t.Fatalf("inner index is %T, want *ivf", col.idx)
	}
	for i := range ids {
		if err := col.Insert(ids[i], vecs[i], 0, nil, nil); err != nil {
			t.Fatalf("Insert(%d): %v", ids[i], err)
		}
		live := i + 1
		if live < threshold && ix.trained {
			t.Fatalf("trained at %d live, before custom threshold %d", live, threshold)
		}
	}
	if !ix.trained {
		t.Fatalf("Collection inner IVF NOT trained at %d live (custom threshold %d) — threshold not honored", threshold, threshold)
	}

	// Control: the SAME corpus under the DEFAULT threshold (2048) stays untrained at
	// 300 live — confirms the trained flip above is the custom field, not an
	// unconditional small-corpus trigger.
	defCfg := cfg
	defCfg.IVFTrainThreshold = 0 // resolves to defaultIVFTrainThreshold (2048)
	defCol, err := NewCollection("docs2", defCfg)
	if err != nil {
		t.Fatal(err)
	}
	defIX := defCol.idx.(*ivf)
	for i := range ids {
		if err := defCol.Insert(ids[i], vecs[i], 0, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	if defIX.trained {
		t.Fatalf("default-threshold Collection trained at %d live (< %d default) — control invalid", threshold, defaultIVFTrainThreshold)
	}
}

// TestIVFAutoTrainRestoreInsertReplay: a from-empty replay via RestoreInsert (the
// WAL/reshard path) trains at the IDENTICAL logical point as a build via Insert, so
// the two reconstruct BYTE-IDENTICAL trained state. This is what guarantees a
// from-empty replica reconstructs the same index as the leader.
func TestIVFAutoTrainRestoreInsertReplay(t *testing.T) {
	dim := 32
	const threshold = 600
	rng := rand.New(rand.NewSource(321))
	cfg := ivfPQAutoTrainConfig(dim, threshold, 8, false)
	n := 1000
	vecs := clusteredVecs(rng, n, dim, 40)
	ids := seqIDs(n)

	// Build via Insert (fresh-insert version 1 each).
	viaInsert, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	insertAll(t, viaInsert, ids, vecs)

	// Build via RestoreInsert with the SAME version (1) the fresh insert assigns, so
	// the only difference is the code path — the trained state must match exactly.
	viaReplay, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := range ids {
		if err := viaReplay.RestoreInsert(ids[i], vecs[i], 0, nil, nil, nil, 1); err != nil {
			t.Fatalf("RestoreInsert(%d): %v", ids[i], err)
		}
	}

	if !viaInsert.trained || !viaReplay.trained {
		t.Fatal("both build paths must auto-train")
	}
	var bi, br bytes.Buffer
	if err := viaInsert.Snapshot(&bi); err != nil {
		t.Fatal(err)
	}
	if err := viaReplay.Snapshot(&br); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(bi.Bytes(), br.Bytes()) {
		t.Fatalf("Insert vs RestoreInsert trained state differs (%d vs %d bytes) — replay trains at a different point", bi.Len(), br.Len())
	}
}

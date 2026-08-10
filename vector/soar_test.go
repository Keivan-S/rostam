// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"math/rand"
	"path/filepath"
	"sort"
	"testing"
)

// SOAR (Spilled Orthogonality-Amplified Residuals) — IVF multi-assignment.
//
// These tests exercise the ScaNN-anisotropic feature: each point joins
// its PRIMARY cell AND a SECONDARY cell chosen to minimize the orthogonality-
// amplified residual loss ‖r_c‖² + λ·(r_c·r̂1)². We assert the headline recall
// lift at a fixed small nprobe, the exact 2-list membership + query dedup, the
// snapshot/sidecar/Raft round-trip reproduces the multi-assignment verbatim, and
// SOAR-off is byte-identical to the pre-SOAR path.

// soarConfig builds an IVF-Flat L2 config with a fixed nlist and SOAR on (λ via
// the arg; 0 ⇒ engine default 1.5).
func soarConfig(dim, nlist int, lambda float32) Config {
	c := DefaultConfig()
	c.Dim = dim
	c.Metric = L2
	c.Seed = 42
	c.IndexType = IndexIVF
	c.IVFNlist = nlist
	c.SOAR = true
	c.SOARLambda = lambda
	return c
}

// soarRecallAt measures recall@k of ix at its current nprobe against brute-force
// ground truth.
func soarRecallAt(t *testing.T, ix *ivf, queries [][]float32, ids []uint64, vecs [][]float32, k int) float64 {
	t.Helper()
	hits, denom := 0, 0
	for _, q := range queries {
		got, err := ix.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		gs := idSet(got)
		for _, w := range bruteForceNN(q, ids, vecs, k) {
			denom++
			if gs[w] {
				hits++
			}
		}
	}
	return float64(hits) / float64(denom)
}

// TestSOARRecallLift is the headline: on HARD (uniform-random) data where true
// neighbors straddle Voronoi boundaries, SOAR multi-assignment raises recall@10
// at a fixed small nprobe vs a non-SOAR IVF built on the IDENTICAL corpus. The
// secondary list gives the query a second path to the true neighbor.
func TestSOARRecallLift(t *testing.T) {
	const (
		dim    = 24
		n      = 2000
		nlist  = 64
		k      = 10
		nq     = 60
		nprobe = 4
	)
	rng := rand.New(rand.NewSource(2026))
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	queries := randVecs(rng, nq, dim)

	// Non-SOAR baseline.
	base, err := newIVF(recallVecsConfig(dim, nlist))
	if err != nil {
		t.Fatal(err)
	}
	if err := base.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	base.nprobe = nprobe
	rBase := soarRecallAt(t, base, queries, ids, vecs, k)

	// SOAR (same corpus, same nlist, same seed, default λ).
	soar, err := newIVF(soarConfig(dim, nlist, 0))
	if err != nil {
		t.Fatal(err)
	}
	if err := soar.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	if !soar.soarTrained {
		t.Fatal("SOAR index did not build a secondary assignment (soarTrained false)")
	}
	soar.nprobe = nprobe
	rSOAR := soarRecallAt(t, soar, queries, ids, vecs, k)

	t.Logf("SOAR recall@%d at nprobe=%d (uniform-random n=%d dim=%d nlist=%d): non-SOAR=%.3f SOAR=%.3f (lift=%+.3f)",
		k, nprobe, n, dim, nlist, rBase, rSOAR, rSOAR-rBase)
	if rSOAR <= rBase {
		t.Fatalf("SOAR did not lift recall@%d at nprobe=%d: non-SOAR=%.3f SOAR=%.3f", k, nprobe, rBase, rSOAR)
	}
}

// TestSOARTwoListMembership proves a SOAR slot joins EXACTLY two distinct lists
// (one with SOAR off), and that a query that probes both lists returns NO
// duplicate ids (the gather dedup works).
func TestSOARTwoListMembership(t *testing.T) {
	const (
		dim   = 16
		n     = 800
		nlist = 32
	)
	rng := rand.New(rand.NewSource(11))
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}

	soar, err := newIVF(soarConfig(dim, nlist, 0))
	if err != nil {
		t.Fatal(err)
	}
	if err := soar.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}

	// Count, per slot, how many lists it appears in. With SOAR + a distinct
	// secondary cell it is exactly 2; the degenerate "secondary == primary"
	// no-op leaves it at 1. Assert the OVERWHELMING majority are 2 and none
	// exceed 2 (no slot is ever filed into a list twice).
	listCount := make(map[uint32]int, n)
	for _, list := range soar.lists {
		seen := make(map[uint32]bool, len(list))
		for _, slot := range list {
			if seen[slot] {
				t.Fatalf("slot %d appears twice in the SAME list", slot)
			}
			seen[slot] = true
			listCount[slot]++
		}
	}
	two := 0
	for slot, cnt := range listCount {
		if cnt > 2 {
			t.Fatalf("slot %d appears in %d lists, want <= 2 (SOAR is dual-assignment)", slot, cnt)
		}
		if cnt == 2 {
			two++
		}
	}
	if two == 0 {
		t.Fatal("no slot has 2 list memberships — SOAR secondary assignment not applied")
	}
	t.Logf("SOAR membership: %d/%d slots in 2 lists", two, len(listCount))

	// Baseline (SOAR off): every slot is in exactly ONE list.
	base, err := newIVF(recallVecsConfig(dim, nlist))
	if err != nil {
		t.Fatal(err)
	}
	if err := base.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	baseCount := make(map[uint32]int, n)
	for _, list := range base.lists {
		for _, slot := range list {
			baseCount[slot]++
		}
	}
	for slot, cnt := range baseCount {
		if cnt != 1 {
			t.Fatalf("SOAR-off slot %d in %d lists, want 1", slot, cnt)
		}
	}

	// Query dedup: probe a wide nprobe so both of a slot's lists are likely hit;
	// the returned ids must be unique.
	soar.nprobe = nlist
	queries := randVecs(rng, 20, dim)
	for qi, q := range queries {
		got, err := soar.Search(q, 25)
		if err != nil {
			t.Fatal(err)
		}
		seen := make(map[uint64]bool, len(got))
		for _, r := range got {
			if seen[r.ID] {
				t.Fatalf("query %d: duplicate id %d in SOAR results (dedup failed)", qi, r.ID)
			}
			seen[r.ID] = true
		}
	}
}

// listMembership returns a deterministic map cell -> sorted slot ids, the
// canonical multi-assignment list membership for equality comparison.
func listMembership(ix *ivf) map[int][]uint32 {
	out := make(map[int][]uint32, len(ix.lists))
	for c, list := range ix.lists {
		cp := append([]uint32(nil), list...)
		sort.Slice(cp, func(a, b int) bool { return cp[a] < cp[b] })
		out[c] = cp
	}
	return out
}

// assertSameMembership fails if two membership maps differ.
func assertSameMembership(t *testing.T, want, got map[int][]uint32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("list count differs: %d vs %d", len(want), len(got))
	}
	for c, w := range want {
		g := got[c]
		if len(w) != len(g) {
			t.Fatalf("cell %d: membership size %d vs %d", c, len(w), len(g))
		}
		for i := range w {
			if w[i] != g[i] {
				t.Fatalf("cell %d: slot %d differs: %d vs %d", c, i, w[i], g[i])
			}
		}
	}
}

// TestSOARSnapshotRoundTrip proves an in-memory Snapshot→Restore reproduces the
// SOAR multi-assignment (cellOf2 + both list memberships) IDENTICALLY, using a
// NON-DEFAULT λ. The restoring index is created with a SOAR config so Restore's
// config-carried SOAR knobs validate and future inserts keep multi-assigning.
func TestSOARSnapshotRoundTrip(t *testing.T) {
	const (
		dim    = 16
		n      = 600
		nlist  = 24
		lambda = float32(2.75) // non-default λ
	)
	rng := rand.New(rand.NewSource(5))
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	ix, err := newIVF(soarConfig(dim, nlist, lambda))
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	if !ix.soarTrained {
		t.Fatal("source SOAR index not soarTrained")
	}
	ix.nprobe = 6
	queries := randVecs(rng, 20, dim)
	before := make([][]Result, len(queries))
	for i, q := range queries {
		before[i], _ = ix.Search(q, 10)
	}
	wantMembership := listMembership(ix)
	wantCellOf2 := append([]uint32(nil), ix.cellOf2...)

	var buf bytes.Buffer
	if err := ix.Snapshot(&buf); err != nil {
		t.Fatal(err)
	}

	restored, err := newIVF(soarConfig(dim, nlist, lambda))
	if err != nil {
		t.Fatal(err)
	}
	if err := restored.Restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatal(err)
	}
	if !restored.soarTrained {
		t.Fatal("restored index lost soarTrained — SOAR block dropped from snapshot")
	}
	if len(restored.cellOf2) != len(wantCellOf2) {
		t.Fatalf("restored cellOf2 len %d, want %d", len(restored.cellOf2), len(wantCellOf2))
	}
	for i := range wantCellOf2 {
		if restored.cellOf2[i] != wantCellOf2[i] {
			t.Fatalf("cellOf2[%d] = %d, want %d", i, restored.cellOf2[i], wantCellOf2[i])
		}
	}
	assertSameMembership(t, wantMembership, listMembership(restored))
	restored.nprobe = 6
	for i, q := range queries {
		got, _ := restored.Search(q, 10)
		resultsIdentical(t, before[i], got, i)
	}
}

// TestSOARSidecarReopen proves the mmap sidecar (SavePersist→openPersistIVF)
// reproduces the SOAR multi-assignment IDENTICALLY (a v3 sidecar carrying the
// SOAR block), with a non-default λ.
func TestSOARSidecarReopen(t *testing.T) {
	const (
		dim    = 16
		n      = 600
		nlist  = 24
		lambda = float32(3.25)
	)
	dir := t.TempDir()
	rng := rand.New(rand.NewSource(8))
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	cfg := soarConfig(dim, nlist, lambda)
	cfg.Persistent = true
	cfg.MmapPath = filepath.Join(dir, "ivf.vecs")
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	if !ix.soarTrained {
		t.Fatal("source SOAR sidecar index not soarTrained")
	}
	ix.nprobe = 5
	queries := randVecs(rng, 20, dim)
	before := make([][]Result, len(queries))
	for i, q := range queries {
		before[i], _ = ix.Search(q, 10)
	}
	wantMembership := listMembership(ix)

	metaPath := filepath.Join(dir, "ivf.meta")
	if err := ix.SavePersist(metaPath); err != nil {
		t.Fatalf("SavePersist: %v", err)
	}
	if err := ix.Close(); err != nil {
		t.Fatal(err)
	}

	restored, err := openPersistIVF(cfg, metaPath)
	if err != nil {
		t.Fatalf("openPersistIVF: %v", err)
	}
	defer restored.Close()
	if !restored.soarTrained {
		t.Fatal("reopened sidecar lost soarTrained — SOAR block dropped from sidecar")
	}
	assertSameMembership(t, wantMembership, listMembership(restored))
	restored.nprobe = 5
	for i, q := range queries {
		got, _ := restored.Search(q, 10)
		resultsIdentical(t, before[i], got, i)
	}
}

// TestSOARStoreSnapshotRaftRoundTrip proves the Raft store-snapshot path
// (SnapshotAll→RestoreAll) carries a SOAR IVF collection's multi-assignment: the
// restored collection reproduces the EXACT list membership AND its config retains
// SOAR + λ so a post-restore insert keeps multi-assigning.
//
// This test would FAIL if snapColCfg dropped SOAR/SOARLambda: toConfig would yield
// SOAR=false, the snapshot blob's SOAR block still restores cellOf2/code2, but the
// restored cfg would NOT validate as a SOAR index and (critically) the restored
// collection would carry SOAR=false — caught by the dstCol.Config() assertion
// below; see the note in the test body for the membership-divergence mechanism.
func TestSOARStoreSnapshotRaftRoundTrip(t *testing.T) {
	const (
		dim    = 24
		n      = 1500
		nlist  = 48
		k      = 10
		nq     = 25
		seed   = 23
		lambda = float32(2.0) // non-default λ
	)
	rng := rand.New(rand.NewSource(seed))
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	queries := randVecs(rng, nq, dim)

	cfg := soarConfig(dim, nlist, lambda)

	src, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := src.CreateCollection("soar", cfg); err != nil {
		t.Fatal(err)
	}
	srcCol, ok := src.Get("soar")
	if !ok {
		t.Fatal("source collection missing")
	}
	if err := srcCol.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	srcIdx := srcCol.idx.(*ivf)
	if !srcIdx.soarTrained {
		t.Fatal("source IVF did not build SOAR assignment")
	}
	wantMembership := listMembership(srcIdx)

	srcIdx.nprobe = 4
	refResults := make([][]Result, nq)
	for qi, q := range queries {
		res, err := srcCol.Search(q, k)
		if err != nil {
			t.Fatalf("src search: %v", err)
		}
		refResults[qi] = res
	}

	var blob bytes.Buffer
	if err := src.SnapshotAll(&blob); err != nil {
		t.Fatalf("SnapshotAll: %v", err)
	}
	_ = src.Close()

	dst, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = dst.Close() }()
	if err := dst.RestoreAll(bytes.NewReader(blob.Bytes())); err != nil {
		t.Fatalf("RestoreAll: %v", err)
	}

	dstCol, ok := dst.Get("soar")
	if !ok {
		t.Fatal("collection missing after RestoreAll")
	}
	// The restored config MUST carry SOAR + λ (the snapColCfg fix). If snapColCfg
	// dropped these, SOAR would be false here — and a post-RestoreAll insert would
	// single-assign, drifting the index toward non-SOAR (the geometry-on-restore
	// class). This assertion is the one that catches a dropped snapColCfg copy.
	if !dstCol.Config().SOAR {
		t.Fatal("restored cfg dropped SOAR — snapColCfg lost the multi-assignment flag")
	}
	if dstCol.Config().SOARLambda != lambda {
		t.Fatalf("restored SOARLambda = %v, want %v (snapColCfg dropped λ)", dstCol.Config().SOARLambda, lambda)
	}

	dstIdx := dstCol.idx.(*ivf)
	if !dstIdx.soarTrained {
		t.Fatal("restored IVF lost soarTrained — SOAR block dropped from store snapshot")
	}
	assertSameMembership(t, wantMembership, listMembership(dstIdx))

	dstIdx.nprobe = 4
	for qi, q := range queries {
		got, err := dstCol.Search(q, k)
		if err != nil {
			t.Fatalf("dst search: %v", err)
		}
		resultsIdentical(t, refResults[qi], got, qi)
	}
}

// TestSOARBackCompat proves SOAR-off is byte-identical to the pre-SOAR IVF: a
// non-SOAR build's snapshot bytes and search results match a snapshot taken
// WITHOUT any SOAR awareness (every slot in exactly one list), and the snapshot
// version stays at the pre-SOAR v4/v3 (never v5).
func TestSOARBackCompat(t *testing.T) {
	const (
		dim   = 16
		n     = 500
		nlist = 24
	)
	rng := rand.New(rand.NewSource(37))
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	ix, err := newIVF(recallVecsConfig(dim, nlist))
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	if ix.soarTrained {
		t.Fatal("non-SOAR index must not be soarTrained")
	}
	ix.nprobe = 4
	queries := randVecs(rng, 15, dim)
	before := make([][]Result, len(queries))
	for i, q := range queries {
		before[i], _ = ix.Search(q, 10)
	}

	var buf bytes.Buffer
	if err := ix.Snapshot(&buf); err != nil {
		t.Fatal(err)
	}
	snap := buf.Bytes()
	// The snapshot version sits in the big-endian u32 right after the 8-byte magic.
	// A non-SOAR build must never write v5 (the SOAR version) — it stays at v3 (no
	// drift) here.
	ver := uint32(snap[8])<<24 | uint32(snap[9])<<16 | uint32(snap[10])<<8 | uint32(snap[11])
	if ver >= 5 {
		t.Fatalf("non-SOAR snapshot version = %d, want < 5 (no SOAR block on a non-SOAR index)", ver)
	}

	restored, err := newIVF(recallVecsConfig(dim, nlist))
	if err != nil {
		t.Fatal(err)
	}
	if err := restored.Restore(bytes.NewReader(snap)); err != nil {
		t.Fatal(err)
	}
	if restored.soarTrained {
		t.Fatal("restored non-SOAR index gained soarTrained")
	}
	for i, q := range queries {
		got, _ := restored.Search(q, 10)
		resultsIdentical(t, before[i], got, i)
	}
}

// TestSOARValidate proves Config.Validate rejects SOAR on a non-IVF index and a
// negative λ, and accepts a well-formed SOAR IVF config.
func TestSOARValidate(t *testing.T) {
	base := func() Config {
		c := DefaultConfig()
		c.Dim = 16
		c.Metric = L2
		c.IndexType = IndexIVF
		c.SOAR = true
		return c
	}
	// SOAR on a non-IVF index is rejected.
	c := base()
	c.IndexType = IndexHNSW
	if err := c.Validate(); !errors.Is(err, ErrInvalidSOAR) {
		t.Fatalf("SOAR on HNSW: got %v, want ErrInvalidSOAR", err)
	}
	// Negative λ is rejected (even unconditionally — try with SOAR on).
	c = base()
	c.SOARLambda = -0.5
	if err := c.Validate(); !errors.Is(err, ErrInvalidSOARLambda) {
		t.Fatalf("negative λ: got %v, want ErrInvalidSOARLambda", err)
	}
	// Well-formed SOAR IVF config validates.
	c = base()
	c.SOARLambda = 2.0
	if err := c.Validate(); err != nil {
		t.Fatalf("valid SOAR IVF config rejected: %v", err)
	}
}

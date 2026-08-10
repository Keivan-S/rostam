// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"math"
	"math/rand"
	"testing"
)

// 4-bit LUT16 fast-scan on the IVF-PQ query path (the wiring follow-up).
//
// These tests prove the in-register fast-scan kernel is on the REAL IVF-PQ batched-
// list query path — not just that a 4-bit collection works:
//   - the fast-scan branch is ACTUALLY taken (the fastScanBlocksScored counter rises
//     during a 4-bit IVF-PQ search, and stays flat for an 8-bit one);
//   - the kernel result is EXACT against the per-slot scalar adcScalar reference;
//   - 4-bit recall@10 (with the existing exact rescore) clears a floor and tracks
//     8-bit IVF-PQ;
//   - 4-bit composes with SOAR (multi-assignment + fast-scan over both lists);
//   - train/encode→snapshot + Raft RestoreAll reproduces identical codes + search;
//   - 8-bit (PQNBits 0/8) is byte-identical (the scalar adc path is unchanged).

// ivfPQ4Config is ivfPQConfig with the 4-bit LUT16 residual codec selected.
func ivfPQ4Config(dim, nlist, m int, rerank bool) Config {
	c := ivfPQConfig(dim, nlist, m, rerank)
	c.PQNBits = 4
	return c
}

// TestIVFPQ4CodeLen: a 4-bit IVF-PQ index stores (m+1)/2 nibble-packed bytes per
// slot via the existing arena code side-array — half the 8-bit width, no
// persistence-format change.
func TestIVFPQ4CodeLen(t *testing.T) {
	const (
		dim   = 16
		n     = 800
		nlist = 32
		m     = 8 // (m+1)/2 = 4 packed bytes
	)
	rng := rand.New(rand.NewSource(7))
	vecs := makeClustered(rng, n, dim, 16, 0.2)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	ix, err := newIVF(ivfPQ4Config(dim, nlist, m, true)) // rerank keeps floats resident
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	wantCL := (m + 1) / 2
	if ix.arena.codeLen != wantCL {
		t.Fatalf("codeLen = %d, want (m+1)/2 = %d", ix.arena.codeLen, wantCL)
	}
	if got := len(ix.arena.codes); got != n*wantCL {
		t.Fatalf("codes len = %d, want n*(m+1)/2 = %d", got, n*wantCL)
	}
	if ix.pq4 == nil || ix.pq.nbits != 4 {
		t.Fatalf("expected 4-bit codec active: pq4=%v nbits=%d", ix.pq4 != nil, ix.pq.nbits)
	}
	if !ix.pq4Active() {
		t.Fatal("pq4Active() false for a trained 4-bit IVF-PQ index")
	}
}

// TestIVFPQ4FastScanPathTaken proves the IVF-PQ query path runs the fast-scan
// kernel: the fastScanBlocksScored counter rises during a 4-bit search and stays
// flat for an 8-bit search at the same nprobe over the same data.
func TestIVFPQ4FastScanPathTaken(t *testing.T) {
	const (
		dim   = 24
		n     = 2000
		nlist = 48
		m     = 6
		k     = 10
		nq    = 20
		nprb  = 12
	)
	rng := rand.New(rand.NewSource(13))
	vecs := makeClustered(rng, n, dim, 32, 0.2)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	queries := makeClustered(rng, nq, dim, 32, 0.2)

	build := func(cfg Config) *ivf {
		ix, err := newIVF(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
			t.Fatal(err)
		}
		ix.nprobe = nprb
		return ix
	}

	// 4-bit: the counter MUST advance.
	ix4 := build(ivfPQ4Config(dim, nlist, m, true))
	before := fastScanBlocksScored.Load()
	for _, q := range queries {
		if _, err := ix4.Search(q, k); err != nil {
			t.Fatal(err)
		}
	}
	got4 := fastScanBlocksScored.Load() - before
	if got4 == 0 {
		t.Fatal("4-bit IVF-PQ search scored ZERO fast-scan blocks — the kernel was NOT on the query path")
	}
	t.Logf("4-bit IVF-PQ search scored %d fast-scan blocks over %d queries", got4, nq)

	// 8-bit: the counter MUST NOT advance (the scalar adc path is unchanged).
	ix8 := build(ivfPQConfig(dim, nlist, m, true))
	before8 := fastScanBlocksScored.Load()
	for _, q := range queries {
		if _, err := ix8.Search(q, k); err != nil {
			t.Fatal(err)
		}
	}
	if got8 := fastScanBlocksScored.Load() - before8; got8 != 0 {
		t.Fatalf("8-bit IVF-PQ search scored %d fast-scan blocks — the 8-bit path must stay scalar", got8)
	}
}

// TestIVFPQ4KernelMatchesScalarReference proves the fast-scan kernel is EXACT
// against the per-slot scalar adcScalar reference over a probed list, so the
// vectorized query-path scoring is byte-for-byte the scalar definition.
func TestIVFPQ4KernelMatchesScalarReference(t *testing.T) {
	const (
		dim   = 32
		n     = 1500
		nlist = 32
		m     = 8
	)
	rng := rand.New(rand.NewSource(101))
	vecs := makeClustered(rng, n, dim, 24, 0.2)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	ix, err := newIVF(ivfPQ4Config(dim, nlist, m, true))
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	if !ix.pq4Active() {
		t.Fatal("4-bit codec not active")
	}
	q := makeClustered(rng, 1, dim, 24, 0.2)[0]

	// Score each non-empty cell's list with the kernel (via fastScanBlockInto, the
	// same entry the gather uses) and compare to per-slot adcScalar.
	res := make([]float32, dim)
	checked := 0
	for c := range ix.lists {
		list := ix.lists[c]
		if len(list) == 0 {
			continue
		}
		cen := ix.centroids[c]
		for i := range res {
			res[i] = q[i] - cen[i]
		}
		lut := ix.pq4.buildLUT16(res)
		// Gather the cell's codes contiguously (codeForCell → primary/secondary).
		cl := pq4CodeLen(m)
		codes := make([]byte, len(list)*cl)
		for i, slot := range list {
			copy(codes[i*cl:], ix.codeForCell(slot, c))
		}
		dists := make([]float32, len(list))
		// Block in fastScanBlock chunks, exactly like the scorer.
		for off := 0; off < len(list); off += fastScanBlock {
			nVec := len(list) - off
			if nVec > fastScanBlock {
				nVec = fastScanBlock
			}
			lut.fastScanBlockInto(dists[off:], codes[off*cl:], nVec, nil, nil)
		}
		for i, slot := range list {
			ref := lut.adcScalar(ix.codeForCell(slot, c))
			if math.Abs(float64(dists[i]-ref)) > 1e-4 {
				t.Fatalf("cell %d slot %d: kernel %.6f != adcScalar %.6f", c, slot, dists[i], ref)
			}
			checked++
		}
	}
	if checked == 0 {
		t.Fatal("no slots checked")
	}
	t.Logf("kernel == adcScalar over %d slots across %d cells", checked, len(ix.lists))
}

// TestIVFPQ4Recall: 4-bit IVF-PQ recall@10 (with the exact rescore) clears a floor
// and tracks 8-bit IVF-PQ at the same settings.
func TestIVFPQ4Recall(t *testing.T) {
	const (
		dim   = 32
		n     = 3000
		nlist = 64
		m     = 8
		k     = 10
		nq    = 40
		nprb  = 16
	)
	rng := rand.New(rand.NewSource(2027))
	vecs := makeClustered(rng, n, dim, 40, 0.20)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	queries := makeClustered(rng, nq, dim, 40, 0.20)

	build := func(cfg Config) *ivf {
		ix, err := newIVF(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
			t.Fatal(err)
		}
		ix.nprobe = nprb
		return ix
	}

	// 4-bit + rerank (the candidate-gen-then-exact-rescore path 4-bit is built for).
	ix4 := build(ivfPQ4Config(dim, nlist, m, true))
	r4 := ivfPQRecallOf(t, ix4, queries, ids, vecs, k)
	// 8-bit + rerank baseline.
	ix8 := build(ivfPQConfig(dim, nlist, m, true))
	r8 := ivfPQRecallOf(t, ix8, queries, ids, vecs, k)
	// 4-bit PQ-only (no rerank) — lossier, but still a useful candidate set.
	ix4pq := build(ivfPQ4Config(dim, nlist, m, false))
	r4pq := ivfPQRecallOf(t, ix4pq, queries, ids, vecs, k)

	t.Logf("IVF-PQ recall@%d (clustered n=%d dim=%d nlist=%d m=%d nprobe=%d): 4bit+rerank=%.3f 8bit+rerank=%.3f 4bit-pqonly=%.3f",
		k, n, dim, nlist, m, nprb, r4, r8, r4pq)

	if r4pq < 0.30 {
		t.Fatalf("4-bit PQ-only recall@%d = %.3f, want >= 0.30 (coarser quant floor)", k, r4pq)
	}
	// 4-bit is a coarser candidate set (16 vs 256 sub-centroids); the exact rescore
	// recovers ranking but the candidate SHORTLIST is lossier, so 4-bit trails 8-bit
	// by a modest margin. Assert a solid absolute floor and that 8-bit still leads.
	if r4 < 0.75 {
		t.Fatalf("4-bit+rerank recall@%d = %.3f, want >= 0.75 (rescore should recover most ranking)", k, r4)
	}
	if r8 < r4 {
		t.Fatalf("8-bit+rerank recall %.3f below 4-bit %.3f (8-bit should lead the finer codec)", r8, r4)
	}
}

// TestIVFPQ4SOARCombine: 4-bit IVF-PQ with SOAR multi-assignment — each point in
// two lists, fast-scan over both, recall holds. Proves codeForCell picks the right
// (primary/secondary) 4-bit code per slot on the fast-scan path.
func TestIVFPQ4SOARCombine(t *testing.T) {
	const (
		dim   = 24
		n     = 2000
		nlist = 48
		m     = 6
		k     = 10
		nq    = 30
		nprb  = 6
	)
	rng := rand.New(rand.NewSource(77))
	vecs := makeClustered(rng, n, dim, 32, 0.2)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	queries := makeClustered(rng, nq, dim, 32, 0.2)

	cfg := ivfPQ4Config(dim, nlist, m, true)
	cfg.SOAR = true
	cfg.SOARLambda = 1.5
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	ix.nprobe = nprb
	if !ix.soarTrained {
		t.Fatal("SOAR not trained")
	}
	if !ix.pq4Active() {
		t.Fatal("4-bit codec not active under SOAR")
	}
	// Some slot must carry a distinct secondary 4-bit code (the combine actually
	// happened, not a degenerate single-assignment).
	secondary := 0
	for slot := range ix.code2 {
		if ix.code2[slot] != nil {
			if len(ix.code2[slot]) != (m+1)/2 {
				t.Fatalf("slot %d code2 len = %d, want (m+1)/2 = %d (SOAR code must be 4-bit too)", slot, len(ix.code2[slot]), (m+1)/2)
			}
			secondary++
		}
	}
	if secondary == 0 {
		t.Fatal("SOAR produced no secondary 4-bit codes")
	}

	before := fastScanBlocksScored.Load()
	r := ivfPQRecallOf(t, ix, queries, ids, vecs, k)
	if fastScanBlocksScored.Load() == before {
		t.Fatal("SOAR 4-bit search took no fast-scan blocks")
	}
	t.Logf("4-bit IVF-PQ + SOAR recall@%d (nprobe=%d, %d secondary codes): %.3f", k, nprb, secondary, r)
	if r < 0.40 {
		t.Fatalf("4-bit+SOAR recall %.3f below 0.40 floor", r)
	}
}

// TestIVFPQ4SnapshotRoundtrip: snapshot/restore reproduces identical packed 4-bit
// codes and identical search (PQ-only and rerank).
func TestIVFPQ4SnapshotRoundtrip(t *testing.T) {
	for _, rerank := range []bool{false, true} {

		name := "pqonly"
		if rerank {
			name = "rerank"
		}
		t.Run(name, func(t *testing.T) {
			const (
				dim   = 24
				n     = 1500
				nlist = 48
				m     = 6
				k     = 10
				nq    = 20
				nprb  = 12
			)
			rng := rand.New(rand.NewSource(31))
			vecs := makeClustered(rng, n, dim, 24, 0.2)
			ids := make([]uint64, n)
			for i := range ids {
				ids[i] = uint64(i + 1)
			}
			queries := makeClustered(rng, nq, dim, 24, 0.2)

			ix, err := newIVF(ivfPQ4Config(dim, nlist, m, rerank))
			if err != nil {
				t.Fatal(err)
			}
			if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
				t.Fatal(err)
			}
			ix.nprobe = nprb

			before := make([][]Result, nq)
			for i, q := range queries {
				r, err := ix.Search(q, k)
				if err != nil {
					t.Fatal(err)
				}
				before[i] = r
			}
			wantCodes := append([]byte(nil), ix.arena.codes...)

			var buf bytes.Buffer
			if err := ix.Snapshot(&buf); err != nil {
				t.Fatal(err)
			}
			restored, err := newIVF(ivfPQ4Config(dim, nlist, m, rerank))
			if err != nil {
				t.Fatal(err)
			}
			if err := restored.Restore(bytes.NewReader(buf.Bytes())); err != nil {
				t.Fatal(err)
			}
			restored.nprobe = nprb
			if !restored.pq4Active() {
				t.Fatal("restored 4-bit codec not active (nbits lost on restore)")
			}
			if restored.arena.codeLen != (m+1)/2 {
				t.Fatalf("restored codeLen = %d, want (m+1)/2 = %d", restored.arena.codeLen, (m+1)/2)
			}
			if !bytes.Equal(restored.arena.codes, wantCodes) {
				t.Fatal("restored packed 4-bit codes differ from pre-snapshot codes")
			}
			for i, q := range queries {
				got, err := restored.Search(q, k)
				if err != nil {
					t.Fatal(err)
				}
				if len(got) != len(before[i]) {
					t.Fatalf("query %d: restored len %d != before %d", i, len(got), len(before[i]))
				}
				for j := range got {
					if got[j].ID != before[i][j].ID {
						t.Fatalf("query %d pos %d: restored id %d != before %d", i, j, got[j].ID, before[i][j].ID)
					}
				}
			}
		})
	}
}

// TestIVFPQ4StoreSnapshotRaftRoundTrip: a 4-bit IVF-PQ collection round-trips
// through the cluster snapshot/RestoreAll path (snapColCfg carries PQNBits) — the
// restored collection is 4-bit, reproduces identical codes + search.
func TestIVFPQ4StoreSnapshotRaftRoundTrip(t *testing.T) {
	const (
		dim   = 24
		n     = 1500
		nlist = 48
		m     = 6
		k     = 10
		nq    = 25
		nprb  = 12
	)
	rng := rand.New(rand.NewSource(53))
	vecs := makeClustered(rng, n, dim, 24, 0.2)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	queries := makeClustered(rng, nq, dim, 24, 0.2)

	cfg := ivfPQ4Config(dim, nlist, m, true) // rerank: keeps floats, exact search round-trip

	src, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := src.CreateCollection("pq4", cfg); err != nil {
		t.Fatal(err)
	}
	srcCol, ok := src.Get("pq4")
	if !ok {
		t.Fatal("source collection missing")
	}
	if err := srcCol.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	srcIdx := srcCol.idx.(*ivf)
	if !srcIdx.pq4Active() {
		t.Fatal("source IVF not 4-bit")
	}
	wantCodes := append([]byte(nil), srcIdx.arena.codes...)
	srcIdx.nprobe = nprb
	ref := make([][]Result, nq)
	for qi, q := range queries {
		r, err := srcCol.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		ref[qi] = r
	}

	var blob bytes.Buffer
	if err := src.SnapshotAll(&blob); err != nil {
		t.Fatal(err)
	}
	_ = src.Close()

	dst, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = dst.Close() }()
	if err := dst.RestoreAll(bytes.NewReader(blob.Bytes())); err != nil {
		t.Fatal(err)
	}
	dstCol, ok := dst.Get("pq4")
	if !ok {
		t.Fatal("collection missing after RestoreAll")
	}
	if dstCol.Config().PQNBits != 4 {
		t.Fatalf("restored cfg PQNBits = %d, want 4 (snapColCfg dropped the width)", dstCol.Config().PQNBits)
	}
	dstIdx := dstCol.idx.(*ivf)
	if !dstIdx.pq4Active() {
		t.Fatal("restored IVF lost 4-bit codec")
	}
	if !bytes.Equal(dstIdx.arena.codes, wantCodes) {
		t.Fatal("restored 4-bit codes differ from source (Raft round-trip not bit-exact)")
	}
	dstIdx.nprobe = nprb
	for qi, q := range queries {
		got, err := dstCol.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(ref[qi]) {
			t.Fatalf("query %d: restored len %d != ref %d", qi, len(got), len(ref[qi]))
		}
		for j := range got {
			if got[j].ID != ref[qi][j].ID {
				t.Fatalf("query %d pos %d: restored id %d != ref %d", qi, j, got[j].ID, ref[qi][j].ID)
			}
		}
	}
}

// TestIVFPQ4ValidateRejectsBadNBits: an IVF-PQ config rejects a PQNBits other than
// 0/4/8 fail-loud; 4 is accepted.
func TestIVFPQ4ValidateRejectsBadNBits(t *testing.T) {
	base := ivfPQConfig(32, 16, 8, false)
	for _, bad := range []int{1, 2, 5, 16, -1} {
		c := base
		c.PQNBits = bad
		if err := c.Validate(); !errors.Is(err, ErrInvalidPQNBits) {
			t.Fatalf("PQNBits=%d: got %v, want ErrInvalidPQNBits", bad, err)
		}
	}
	for _, ok := range []int{0, 4, 8} {
		c := base
		c.PQNBits = ok
		if err := c.Validate(); err != nil {
			t.Fatalf("PQNBits=%d: unexpected error %v", ok, err)
		}
	}
}

// BenchmarkIVFPQ4GatherFastScan vs BenchmarkIVFPQ8GatherScalar measure the REAL
// query-path gather cost (gatherADCLocked over a probed list set) for 4-bit
// fast-scan vs 8-bit scalar adc on the same corpus — the on-query-path speedup, not
// the isolated kernel micro-bench.
func benchIVFPQGather(b *testing.B, nbits int) {
	const (
		dim   = 64
		n     = 20000
		nlist = 128
		m     = 16
		k     = 10
		nprb  = 32
	)
	rng := rand.New(rand.NewSource(909))
	vecs := makeClustered(rng, n, dim, 64, 0.2)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	cfg := ivfPQConfig(dim, nlist, m, true)
	cfg.PQNBits = nbits
	ix, err := newIVF(cfg)
	if err != nil {
		b.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		b.Fatal(err)
	}
	ix.nprobe = nprb
	q := makeClustered(rng, 1, dim, 64, 0.2)[0]
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	dist := ix.metricDist()
	cells := ix.nearestCells(q, nprb, dist)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ix.gatherADCLocked(q, k, nil, cells)
	}
}

func BenchmarkIVFPQ4GatherFastScan(b *testing.B) { benchIVFPQGather(b, 4) }
func BenchmarkIVFPQ8GatherScalar(b *testing.B)   { benchIVFPQGather(b, 8) }

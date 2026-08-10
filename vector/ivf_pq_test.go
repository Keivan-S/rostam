// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math/rand"
	"testing"
)

// IVF-PQ (residual product quantization) tests: build + ADC search recall on
// clustered data, the IVFRerank shortlist→exact path, the PQ-only memory model
// (codes-only, floats dropped), and snapshot round-trip + back-compat. Modest N
// throughout (memory-safety: these run in a 10g-capped container).

// ivfPQConfig builds an IVF-PQ L2 config with a fixed nlist for determinism.
func ivfPQConfig(dim, nlist, m int, rerank bool) Config {
	c := DefaultConfig()
	c.Dim = dim
	c.Metric = L2
	c.Seed = 42
	c.IndexType = IndexIVF
	c.IVFNlist = nlist
	c.IVFPQ = true
	c.IVFPQM = m
	c.IVFRerank = rerank
	return c
}

// recallOf measures recall@k of ix against brute-force ground truth.
func ivfPQRecallOf(t *testing.T, ix *ivf, queries [][]float32, ids []uint64, vecs [][]float32, k int) float64 {
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

// TestIVFPQRecall: residual-PQ-only ADC recall on clustered data clears a
// reasonable PQ threshold, and IVFRerank lifts it close to exact (IVF-Flat).
func TestIVFPQRecall(t *testing.T) {
	const (
		dim   = 32
		n     = 3000
		nlist = 64
		m     = 8 // dsub = 4
		k     = 10
		nq    = 40
		nprb  = 16
	)
	rng := rand.New(rand.NewSource(2026))
	vecs := makeClustered(rng, n, dim, 40, 0.20)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	queries := makeClustered(rng, nq, dim, 40, 0.20)

	// PQ-only.
	pqOnly, err := newIVF(ivfPQConfig(dim, nlist, m, false))
	if err != nil {
		t.Fatal(err)
	}
	if err := pqOnly.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	pqOnly.nprobe = nprb
	if !pqOnly.pqActive() {
		t.Fatal("PQ-only: pq codec not trained after build")
	}
	if !pqOnly.pqDropped || pqOnly.arena.vecs != nil {
		t.Fatal("PQ-only: resident floats not dropped after build")
	}
	rPQ := ivfPQRecallOf(t, pqOnly, queries, ids, vecs, k)

	// IVFRerank (keeps floats, exact rescore of the ADC shortlist).
	rerank, err := newIVF(ivfPQConfig(dim, nlist, m, true))
	if err != nil {
		t.Fatal(err)
	}
	if err := rerank.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	rerank.nprobe = nprb
	if rerank.pqDropped || rerank.arena.vecs == nil {
		t.Fatal("IVFRerank: floats must stay resident")
	}
	rRerank := ivfPQRecallOf(t, rerank, queries, ids, vecs, k)

	// IVF-Flat baseline (same nprobe) for the rerank-≈-flat comparison.
	flatCfg := recallVecsConfig(dim, nlist)
	flatCfg.IndexType = IndexIVF
	flat, err := newIVF(flatCfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := flat.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	flat.nprobe = nprb
	rFlat := ivfPQRecallOf(t, flat, queries, ids, vecs, k)

	t.Logf("IVF-PQ recall@%d (clustered n=%d dim=%d nlist=%d m=%d nprobe=%d): PQ-only=%.3f rerank=%.3f flat=%.3f",
		k, n, dim, nlist, m, nprb, rPQ, rRerank, rFlat)

	if rPQ < 0.40 {
		t.Fatalf("PQ-only recall@%d = %.3f, want >= 0.40 (residual ADC threshold)", k, rPQ)
	}
	// Rerank beats PQ-only and lands close to exact-Flat.
	if rRerank < rPQ {
		t.Fatalf("IVFRerank recall %.3f < PQ-only %.3f (rerank must not hurt)", rRerank, rPQ)
	}
	if rRerank < rFlat-0.05 {
		t.Fatalf("IVFRerank recall %.3f far below IVF-Flat %.3f (want near-exact)", rRerank, rFlat)
	}
}

// TestIVFPQMemory: PQ-only stores exactly M code bytes/vector and no resident
// floats; IVFRerank keeps the floats.
func TestIVFPQMemory(t *testing.T) {
	const (
		dim   = 16
		n     = 800
		nlist = 32
		m     = 4
	)
	rng := rand.New(rand.NewSource(5))
	vecs := makeClustered(rng, n, dim, 16, 0.2)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}

	ix, err := newIVF(ivfPQConfig(dim, nlist, m, false))
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	if ix.arena.codeLen != m {
		t.Fatalf("codeLen = %d, want m = %d", ix.arena.codeLen, m)
	}
	if got := len(ix.arena.codes); got != n*m {
		t.Fatalf("codes len = %d, want n*m = %d", got, n*m)
	}
	if ix.arena.vecs != nil {
		t.Fatalf("PQ-only: arena.vecs not dropped (len %d)", len(ix.arena.vecs))
	}
	// A subsequent point Get reconstructs (approximate) and does not panic.
	if _, _, _, _, _, ok := ix.Get(ids[0]); !ok {
		t.Fatal("Get after float-drop failed")
	}
}

// TestIVFPQSnapshotRoundtrip: snapshot/restore restores codebooks + codes so
// search is identical post-restore (PQ-only and IVFRerank).
func TestIVFPQSnapshotRoundtrip(t *testing.T) {
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
			rng := rand.New(rand.NewSource(11))
			vecs := makeClustered(rng, n, dim, 24, 0.2)
			ids := make([]uint64, n)
			for i := range ids {
				ids[i] = uint64(i + 1)
			}
			queries := makeClustered(rng, nq, dim, 24, 0.2)

			ix, err := newIVF(ivfPQConfig(dim, nlist, m, rerank))
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

			var buf bytes.Buffer
			if err := ix.Snapshot(&buf); err != nil {
				t.Fatal(err)
			}

			restored, err := newIVF(ivfPQConfig(dim, nlist, m, rerank))
			if err != nil {
				t.Fatal(err)
			}
			if err := restored.Restore(bytes.NewReader(buf.Bytes())); err != nil {
				t.Fatal(err)
			}
			restored.nprobe = nprb
			if !restored.pqActive() {
				t.Fatal("restored: pq codec missing")
			}
			if restored.pqDropped != ix.pqDropped {
				t.Fatalf("restored pqDropped = %v, want %v", restored.pqDropped, ix.pqDropped)
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

// TestIVFFlatSnapshotRestoresAsFlat: an IVF-Flat (non-PQ) snapshot restores with
// no PQ codec (pq == nil), proving the v3 trailer's back-compat default.
func TestIVFFlatSnapshotRestoresAsFlat(t *testing.T) {
	const (
		dim   = 16
		n     = 500
		nlist = 24
		k     = 10
		nq    = 15
	)
	rng := rand.New(rand.NewSource(3))
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	cfg := recallVecsConfig(dim, nlist)
	cfg.IndexType = IndexIVF
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	ix.nprobe = nlist

	var buf bytes.Buffer
	if err := ix.Snapshot(&buf); err != nil {
		t.Fatal(err)
	}
	restored, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := restored.Restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatal(err)
	}
	if restored.pqActive() {
		t.Fatal("IVF-Flat snapshot restored with a PQ codec (should be nil)")
	}
	if restored.arena.vecs == nil {
		t.Fatal("IVF-Flat restore dropped floats")
	}
	queries := randVecs(rng, nq, dim)
	restored.nprobe = nlist
	for _, q := range queries {
		a, _ := ix.Search(q, k)
		b, _ := restored.Search(q, k)
		if len(a) != len(b) {
			t.Fatalf("flat restore search len mismatch %d vs %d", len(a), len(b))
		}
		for i := range a {
			if a[i].ID != b[i].ID {
				t.Fatalf("flat restore search id mismatch at %d: %d vs %d", i, a[i].ID, b[i].ID)
			}
		}
	}
}

// TestIVFPQConfigValidation exercises the IVFPQ/IVFPQM/IVFRerank validation.
func TestIVFPQConfigValidation(t *testing.T) {
	base := func() Config {
		c := DefaultConfig()
		c.Dim = 32
		c.Metric = L2
		c.IndexType = IndexIVF
		return c
	}
	// PQ on a non-IVF index is rejected.
	c := base()
	c.IndexType = IndexHNSW
	c.IVFPQ = true
	if err := c.Validate(); err != ErrInvalidIVFPQ {
		t.Fatalf("IVFPQ on HNSW: err = %v, want ErrInvalidIVFPQ", err)
	}
	// IVFRerank also requires IVF.
	c = base()
	c.IndexType = IndexHNSW
	c.IVFRerank = true
	if err := c.Validate(); err != ErrInvalidIVFPQ {
		t.Fatalf("IVFRerank on HNSW: err = %v, want ErrInvalidIVFPQ", err)
	}
	// Indivisible m rejected.
	c = base()
	c.IVFPQ = true
	c.IVFPQM = 7 // 32 % 7 != 0
	if err := c.Validate(); err != ErrInvalidIVFPQM {
		t.Fatalf("indivisible m: err = %v, want ErrInvalidIVFPQM", err)
	}
	// Negative m rejected.
	c = base()
	c.IVFPQM = -1
	if err := c.Validate(); err != ErrInvalidIVFPQM {
		t.Fatalf("negative m: err = %v, want ErrInvalidIVFPQM", err)
	}
	// m == 0 (default) + IVFPQ valid.
	c = base()
	c.IVFPQ = true
	if err := c.Validate(); err != nil {
		t.Fatalf("IVFPQ m=0 default: unexpected err %v", err)
	}
	// Divisible m valid.
	c = base()
	c.IVFPQ = true
	c.IVFPQM = 8
	if err := c.Validate(); err != nil {
		t.Fatalf("IVFPQ m=8: unexpected err %v", err)
	}
}

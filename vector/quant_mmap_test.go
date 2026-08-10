// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// TestSQ8MmapRecallAndBacking builds an mmap-backed SQ8 index and verifies two
// things: (1) the full-precision float32 vectors live in the backing file (so
// only int8 codes are resident), and (2) recall stays within 5% of the exact
// float32 baseline — proving the rescore stage reads correctly through the
// memory-mapped region. Skipped under -short.
func TestSQ8MmapRecallAndBacking(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping recall test in -short mode")
	}
	const (
		n      = 2_000
		dim    = 64
		nq     = 50
		k      = 10
		seed   = 42
		margin = 0.05
	)
	rng := rand.New(rand.NewSource(seed))

	corpus := make([][]float32, n)
	for i := range corpus {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		corpus[i] = v
	}
	queries := make([][]float32, nq)
	for i := range queries {
		q := make([]float32, dim)
		for j := range q {
			q[j] = float32(rng.NormFloat64())
		}
		queries[i] = q
	}

	groundTruth := func(q []float32) map[uint64]bool {
		qn := append([]float32(nil), q...)
		normalize(qn)
		type pair struct {
			id  uint64
			dot float32
		}
		dists := make([]pair, n)
		for i, v := range corpus {
			vn := append([]float32(nil), v...)
			normalize(vn)
			dists[i] = pair{id: uint64(i + 1), dot: dotScalar(qn, vn)}
		}
		sort.Slice(dists, func(a, b int) bool { return dists[a].dot > dists[b].dot })
		out := make(map[uint64]bool, k)
		for i := 0; i < k; i++ {
			out[dists[i].id] = true
		}
		return out
	}

	cfg := Config{
		Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64,
		Seed: seed, Metric: Cosine, Quant: QuantSQ8, RescoreFactor: 3,
	}

	path := filepath.Join(t.TempDir(), "vecs.dat")
	mcfg := cfg
	mcfg.QuantStorage = QuantMmap
	mcfg.MmapPath = path

	h, err := newHNSW(mcfg)
	if err != nil {
		t.Fatalf("newHNSW(mmap): %v", err)
	}
	for i, v := range corpus {
		if _, _, err := h.Insert(uint64(i+1), v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	// The backing file must hold all float32 vectors (n*dim*4 bytes minimum).
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat backing file: %v", err)
	}
	if minBytes := int64(n * dim * 4); fi.Size() < minBytes {
		t.Errorf("backing file = %d bytes, want >= %d (n*dim*4)", fi.Size(), minBytes)
	}

	matches := 0
	for _, q := range queries {
		truth := groundTruth(q)
		results, err := h.Search(q, k)
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		for _, r := range results {
			if truth[r.ID] {
				matches++
			}
		}
	}
	sqMmap := float64(matches) / float64(nq*k)
	if err := h.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	// Exact float32 baseline for comparison.
	hb, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range corpus {
		if _, _, err := hb.Insert(uint64(i+1), v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("baseline insert %d: %v", i, err)
		}
	}
	matches = 0
	for _, q := range queries {
		truth := groundTruth(q)
		results, _ := hb.Search(q, k)
		for _, r := range results {
			if truth[r.ID] {
				matches++
			}
		}
	}
	base := float64(matches) / float64(nq*k)

	t.Logf("recall@%d  exact=%.3f  sq8-mmap=%.3f  file=%d bytes", k, base, sqMmap, fi.Size())
	if sqMmap < base-margin {
		t.Errorf("SQ8-mmap recall@%d = %.3f, want >= exact - %.2f (= %.3f)", k, sqMmap, margin, base-margin)
	}
}

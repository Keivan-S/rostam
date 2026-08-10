// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"os"
	"sort"
	"testing"
	"time"
)

// TestRecallCurve measures recall@10 across an ef sweep on a clustered cosine
// corpus that mimics GloVe's anisotropic structure (the dataset where Rostam's
// high-ef recall-per-work trails Qdrant). Opt-in via ROSTAM_RECALL_CURVE=1.
//
// It reports two curves — serial Insert vs BuildConcurrent — at identical
// params, so a build-path graph-quality deficit (the bulk path the benchmark
// uses) shows up as a gap between them. Run:
//
//	ROSTAM_RECALL_CURVE=1 go test ./vector/ -run TestRecallCurve -v -timeout 20m
func TestRecallCurve(t *testing.T) {
	if !envOn("ROSTAM_RECALL_CURVE") {
		t.Skip("set ROSTAM_RECALL_CURVE=1 to run")
	}
	const (
		n        = 50_000
		dim      = 100
		nClust   = 200
		nq       = 1000
		k        = 10
		seed     = 7
		efC      = 200
		m        = 32
		clustStd = 0.18
	)
	rng := rand.New(rand.NewSource(seed))
	corpus := makeClustered(rng, n, dim, nClust, clustStd)
	queries := makeClustered(rng, nq, dim, nClust, clustStd)

	truth := bruteTruthCosine(corpus, queries, k)
	efs := []int{64, 128, 256, 512}

	build := func(extend bool, extMax int) (*hnsw, time.Duration) {
		h, err := newHNSW(Config{Dim: dim, M: m, EfConstruction: efC, EfSearch: 64, Seed: seed, Metric: Cosine, ExtendCandidates: extend, ExtendCandidatesMax: extMax})
		if err != nil {
			t.Fatal(err)
		}
		ids := make([]uint64, n)
		for i := range ids {
			ids[i] = uint64(i + 1)
		}
		start := time.Now()
		if err := h.BuildConcurrent(ids, corpus, 8); err != nil {
			t.Fatal(err)
		}
		return h, time.Since(start)
	}

	for _, mode := range []struct {
		name   string
		extend bool
		extMax int
	}{
		{"baseline (no extend)", false, 0},
		{"extend cap=400", true, 400},
		{"extend cap=800", true, 800},
		{"extend unbounded", true, 0},
	} {
		h, dur := build(mode.extend, mode.extMax)
		t.Logf("=== %s build === (%.1fs)", mode.name, dur.Seconds())
		for _, ef := range efs {
			h.cfg.EfSearch = ef
			rec := measureRecall(h, queries, truth, k)
			t.Logf("  ef=%-4d recall@%d=%.4f", ef, k, rec)
		}
	}
}

// makeClustered builds n unit vectors drawn from nClust random cluster centers
// with Gaussian jitter — a stand-in for real embedding anisotropy.
func makeClustered(rng *rand.Rand, n, dim, nClust int, std float64) [][]float32 {
	centers := make([][]float32, nClust)
	for c := range centers {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		centers[c] = v
	}
	out := make([][]float32, n)
	for i := range out {
		c := centers[rng.Intn(nClust)]
		v := make([]float32, dim)
		for j := range v {
			v[j] = c[j] + float32(rng.NormFloat64())*float32(std)
		}
		out[i] = v
	}
	return out
}

func bruteTruthCosine(corpus, queries [][]float32, k int) []map[uint64]bool {
	// Normalize corpus copies for exact cosine ranking.
	norm := make([][]float32, len(corpus))
	for i, v := range corpus {
		c := append([]float32(nil), v...)
		normalize(c)
		norm[i] = c
	}
	truth := make([]map[uint64]bool, len(queries))
	for qi, q := range queries {
		qn := append([]float32(nil), q...)
		normalize(qn)
		type pair struct {
			id uint64
			d  float32
		}
		ds := make([]pair, len(norm))
		for i, v := range norm {
			ds[i] = pair{uint64(i + 1), 1 - dotScalar(qn, v)}
		}
		sort.Slice(ds, func(a, b int) bool { return ds[a].d < ds[b].d })
		set := make(map[uint64]bool, k)
		for i := 0; i < k && i < len(ds); i++ {
			set[ds[i].id] = true
		}
		truth[qi] = set
	}
	return truth
}

func measureRecall(h *hnsw, queries [][]float32, truth []map[uint64]bool, k int) float64 {
	var match int
	for qi, q := range queries {
		res, err := h.Search(q, k)
		if err != nil {
			panic(err)
		}
		for _, r := range res {
			if truth[qi][r.ID] {
				match++
			}
		}
	}
	return float64(match) / float64(len(queries)*k)
}

func envOn(key string) bool { return os.Getenv(key) == "1" }

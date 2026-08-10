// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestGloveProbe reproduces Rostam's recall-per-ef on the real glove-100-angular
// corpus in-process (no docker/network) and A/Bs search-side levers against the
// production search path, to investigate why Qdrant extracts more recall per ef
// at the high end. Opt-in: ROSTAM_GLOVE_PROBE=1 with extracted arrays at
// $GLOVE_DIR (default /root/bench/glove): train.fvecs, test.fvecs,
// neighbors.ivecs (0-based indices into train).
//
//	ROSTAM_GLOVE_PROBE=1 go test ./vector/ -run TestGloveProbe -v -timeout 60m
func TestGloveProbe(t *testing.T) {
	if os.Getenv("ROSTAM_GLOVE_PROBE") != "1" {
		t.Skip("set ROSTAM_GLOVE_PROBE=1 to run")
	}
	dir := os.Getenv("GLOVE_DIR")
	if dir == "" {
		dir = "/root/bench/glove"
	}
	train, err := readFvecs(filepath.Join(dir, "train.fvecs"))
	if err != nil {
		t.Fatal(err)
	}
	test, err := readFvecs(filepath.Join(dir, "test.fvecs"))
	if err != nil {
		t.Fatal(err)
	}
	gt, err := readIvecs(filepath.Join(dir, "neighbors.ivecs"))
	if err != nil {
		t.Fatal(err)
	}
	// glove-100-angular's ground truth is 100-wide and the benchmark leaves
	// `top` unset, so it measures recall@100 (k=100). Match that here, else the
	// in-process numbers are recall@10 and not comparable to the benchmark.
	k := 100
	if ks := os.Getenv("GLOVE_K"); ks == "10" {
		k = 10
	}
	dim := len(train[0])
	t.Logf("train=%d test=%d dim=%d k=%d", len(train), len(test), dim, k)

	// Ground truth: id = trainIndex+1, so neighbor index j → id j+1.
	truth := make([]map[uint64]bool, len(test))
	for i := range test {
		m := make(map[uint64]bool, k)
		for j := 0; j < k && j < len(gt[i]); j++ {
			m[uint64(gt[i][j])+1] = true
		}
		truth[i] = m
	}

	ids := make([]uint64, len(train))
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	efs := []int{128, 256, 512}

	// Construction-quality sweep: M (degree) and EfConstruction (build breadth)
	// are the remaining levers for recall@100 now that the diversity heuristic
	// and search beam are ruled out. Each entry builds a fresh graph and sweeps
	// the query ef. Qdrant reference (M=32, efC=200): ef128=0.853, ef256=0.914.
	builds := []struct {
		m, efc int
		l0full bool
	}{
		{32, 200, false}, // baseline (matches Qdrant's params)
		{32, 200, true},  // + level-0 full degree (2M forward) — the Qdrant m0 convention
		{32, 400, true},
	}
	for _, b := range builds {
		h, err := newHNSW(Config{Dim: dim, M: b.m, EfConstruction: b.efc, EfSearch: 64, Seed: 42, Metric: Cosine, Level0FullDegree: b.l0full})
		if err != nil {
			t.Fatal(err)
		}
		if err := h.BuildConcurrent(ids, train, 16); err != nil {
			t.Fatal(err)
		}
		t.Logf("=== build M=%d efC=%d level0full=%v (maxLevel=%d) ===", b.m, b.efc, b.l0full, h.maxLevel)
		for _, ef := range efs {
			h.cfg.EfSearch = ef
			t.Logf("  ef=%-4d recall@%d=%.4f", ef, k, gloveRecall(h, test, truth, k, gloveBaselineSearch))
		}
		_ = h.Close()
	}
}

// TestQBuildSymNavGloveAB measures, on the real glove corpus with SQ8 +
// QuantizedBuild + L0full, whether symmetric VNNI navigation (buildSymNav) costs
// recall@100 vs the current asymmetric float-query navigation — the one risk of
// the symmetric-nav build lever. It builds both ways and reports build time +
// recall at several ef. Recall must hold (Qdrant navigates symmetrically at
// 0.997); build time is a secondary signal here (100-dim, so the high-dim build
// speedup is measured by TestHighDimSymNavAB instead). Opt-in:
// ROSTAM_GLOVE_PROBE=1 with arrays at $GLOVE_DIR.
//
//	ROSTAM_GLOVE_PROBE=1 go test ./vector/ -run TestQBuildSymNavGloveAB -v -timeout 60m
func TestQBuildSymNavGloveAB(t *testing.T) {
	if os.Getenv("ROSTAM_GLOVE_PROBE") != "1" {
		t.Skip("set ROSTAM_GLOVE_PROBE=1 to run")
	}
	dir := os.Getenv("GLOVE_DIR")
	if dir == "" {
		dir = "/root/bench/glove"
	}
	train, err := readFvecs(filepath.Join(dir, "train.fvecs"))
	if err != nil {
		t.Fatal(err)
	}
	test, err := readFvecs(filepath.Join(dir, "test.fvecs"))
	if err != nil {
		t.Fatal(err)
	}
	gt, err := readIvecs(filepath.Join(dir, "neighbors.ivecs"))
	if err != nil {
		t.Fatal(err)
	}
	k := 100
	dim := len(train[0])
	truth := make([]map[uint64]bool, len(test))
	for i := range test {
		m := make(map[uint64]bool, k)
		for j := 0; j < k && j < len(gt[i]); j++ {
			m[uint64(gt[i][j])+1] = true
		}
		truth[i] = m
	}
	ids := make([]uint64, len(train))
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	efs := []int{128, 256, 512}

	saved := buildSymNav
	defer func() { buildSymNav = saved }()

	run := func(label string, sym bool) {
		buildSymNav = sym
		h, err := newHNSW(Config{Dim: dim, M: 32, EfConstruction: 200, EfSearch: 64, Seed: 42, Metric: Cosine, Level0FullDegree: true, Quant: QuantSQ8, RescoreFactor: 3, QuantizedBuild: true})
		if err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		if err := h.BuildConcurrent(ids, train, 16); err != nil {
			t.Fatal(err)
		}
		bt := time.Since(t0)
		t.Logf("=== %s (build=%.1fs) ===", label, bt.Seconds())
		for _, ef := range efs {
			h.cfg.EfSearch = ef
			t.Logf("  ef=%-4d recall@%d=%.4f", ef, k, gloveRecall(h, test, truth, k, gloveBaselineSearch))
		}
		_ = h.Close()
	}

	run("asym-nav (current)", false)
	run("sym-nav (VNNI)", true)
}

type gloveSearchFn func(h *hnsw, q []float32, k, ef int) []Result

func gloveRecall(h *hnsw, test [][]float32, truth []map[uint64]bool, k int, fn gloveSearchFn) float64 {
	// Use the ef currently set in cfg as the level-0 beam.
	ef := h.cfg.EfSearch
	var match int
	for qi, q := range test {
		for _, r := range fn(h, q, k, ef) {
			if truth[qi][r.ID] {
				match++
			}
		}
	}
	return float64(match) / float64(len(test)*k)
}

func gloveBaselineSearch(h *hnsw, q []float32, k, ef int) []Result {
	res, _ := h.Search(q, k)
	return res
}

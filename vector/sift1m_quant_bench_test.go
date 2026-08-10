// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"testing"
	"time"
)

// TestSIFT1MQuantBench measures whether SQ8 quantization (4× less memory traffic
// per candidate) speeds up search in the memory-bound regime, and at what recall
// cost. SQ8 is Cosine-scope, and SIFT's bundled ground truth is L2, so this
// normalizes the vectors, uses Cosine, and recomputes cosine ground truth by
// brute force over a query subset. Compares QuantNone vs QuantSQ8 at matched
// params. Opt-in: ROSTAM_SIFT1M=1 with the dataset at /tmp/rostam-sift1m/sift/.
//
//	ROSTAM_SIFT1M=1 go test ./vector/ -run TestSIFT1MQuantBench -v -timeout 40m
func TestSIFT1MQuantBench(t *testing.T) {
	if os.Getenv("ROSTAM_SIFT1M") != "1" {
		t.Skip("set ROSTAM_SIFT1M=1 with dataset at /tmp/rostam-sift1m/sift/ to run")
	}
	dir := filepath.Join(os.TempDir(), "rostam-sift1m", "sift")
	corpus, err := readFvecs(filepath.Join(dir, "sift_base.fvecs"))
	if err != nil {
		t.Fatal(err)
	}
	allQueries, err := readFvecs(filepath.Join(dir, "sift_query.fvecs"))
	if err != nil {
		t.Fatal(err)
	}
	const (
		k    = 10
		nq   = 1000 // query subset, so brute-force cosine GT stays ~seconds
		dim  = 128
		seed = 42
	)
	queries := allQueries[:nq]

	// Normalize in place: cosine over normalized vectors is a dot product, and
	// BuildConcurrent re-normalizes on insert (a no-op once normalized).
	for _, v := range corpus {
		normalize(v)
	}
	for _, v := range queries {
		normalize(v)
	}

	// Brute-force cosine ground truth (top-k by dot) for the query subset.
	// Parallelized: each query is independent and writes a disjoint gt[qi].
	workers := runtime.GOMAXPROCS(0)
	fmt.Fprintf(os.Stderr, "[sq8] computing cosine GT for %d queries over %d vecs (%d workers)...\n", nq, len(corpus), workers)
	t0 := time.Now()
	gt := make([]map[uint64]bool, nq)
	work := make(chan int, nq)
	for qi := range queries {
		work <- qi
	}
	close(work)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			type pair struct {
				id  uint64
				dot float32
			}
			ds := make([]pair, len(corpus)) // reused across this worker's queries
			for qi := range work {
				q := queries[qi]
				for i, v := range corpus {
					ds[i] = pair{uint64(i + 1), dotProduct(q, v)}
				}
				sort.Slice(ds, func(a, b int) bool { return ds[a].dot > ds[b].dot })
				m := make(map[uint64]bool, k)
				for i := 0; i < k; i++ {
					m[ds[i].id] = true
				}
				gt[qi] = m // disjoint index per query — no lock needed
			}
		}()
	}
	wg.Wait()
	fmt.Fprintf(os.Stderr, "[sq8] GT done in %v\n", time.Since(t0).Round(time.Second))

	ids := make([]uint64, len(corpus))
	for i := range ids {
		ids[i] = uint64(i + 1)
	}

	run := func(mode QuantMode, label string) {
		cfg := Config{Dim: dim, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed, Quant: mode, RescoreFactor: 3}
		h, err := newHNSW(cfg)
		if err != nil {
			t.Fatal(err)
		}
		bt := time.Now()
		if err := h.BuildConcurrent(ids, corpus, runtime.GOMAXPROCS(0)); err != nil {
			t.Fatalf("%s build: %v", label, err)
		}
		fmt.Fprintf(os.Stderr, "[sq8] %s built in %v\n", label, time.Since(bt).Round(time.Millisecond))
		dst := make([]Result, 0, k)
		for _, ef := range []int{16, 32, 64, 128} {
			h.cfg.EfSearch = ef
			var matches int
			var elapsed time.Duration
			for qi, q := range queries {
				s := time.Now()
				res, _ := h.SearchInto(dst[:0], q, k, Filter{})
				elapsed += time.Since(s)
				for _, r := range res {
					if gt[qi][r.ID] {
						matches++
					}
				}
			}
			recall := float64(matches) / float64(nq*k)
			qps := float64(nq) / elapsed.Seconds()
			fmt.Fprintf(os.Stderr, "[sq8] %-12s ef=%-4d recall@10=%.4f QPS=%.0f\n", label, ef, recall, qps)
		}
		_ = h.Close()
	}

	fmt.Fprintf(os.Stderr, "[sq8] === Cosine SIFT-1M: QuantNone vs QuantSQ8 (M=16 efC=200) ===\n")
	run(QuantNone, "none(f32)")
	run(QuantSQ8, "sq8")
}

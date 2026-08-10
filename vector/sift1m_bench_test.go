// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// TestSIFT1MBench measures Rostam's recall@10 vs. single-threaded QPS on SIFT-1M
// across a sweep of EfSearch, plus build time — the same protocol used for the
// hnswlib and Qdrant harnesses under bench/sift1m/, so the three are
// comparable. Opt-in: ROSTAM_SIFT1M=1 with the dataset at
// /tmp/rostam-sift1m/sift/ (see TestSIFT1M for fetch instructions).
//
//	ROSTAM_SIFT1M=1 go test ./vector/ -run TestSIFT1MBench -v -timeout 30m
func TestSIFT1MBench(t *testing.T) {
	if os.Getenv("ROSTAM_SIFT1M") != "1" {
		t.Skip("set ROSTAM_SIFT1M=1 with dataset at /tmp/rostam-sift1m/sift/ to run")
	}
	dir := filepath.Join(os.TempDir(), "rostam-sift1m", "sift")
	corpus, err := readFvecs(filepath.Join(dir, "sift_base.fvecs"))
	if err != nil {
		t.Fatal(err)
	}
	queries, err := readFvecs(filepath.Join(dir, "sift_query.fvecs"))
	if err != nil {
		t.Fatal(err)
	}
	gt, err := readIvecs(filepath.Join(dir, "sift_groundtruth.ivecs"))
	if err != nil {
		t.Fatal(err)
	}

	const k = 10
	h, err := newHNSW(Config{Dim: len(corpus[0]), M: 16, EfConstruction: 200, EfSearch: 64, Seed: 42, Metric: L2})
	if err != nil {
		t.Fatal(err)
	}

	// Concurrent bulk build across all cores. Set ROSTAM_SIFT1M_SERIAL=1 to
	// build single-threaded instead (for an apples-to-apples build-speed
	// comparison with the single-thread hnswlib harness).
	workers := runtime.GOMAXPROCS(0)
	ids := make([]uint64, len(corpus))
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	fmt.Fprintf(os.Stderr, "[sift] loaded corpus=%d queries=%d dim=%d; building (workers=%d)...\n",
		len(corpus), len(queries), len(corpus[0]), workers)
	t0 := time.Now()
	if os.Getenv("ROSTAM_SIFT1M_SERIAL") == "1" {
		workers = 1
		for i, v := range corpus {
			if _, _, err := h.Insert(ids[i], v, 0, nil, nil, nil, CASCond{}); err != nil {
				t.Fatalf("insert %d: %v", i, err)
			}
		}
	} else if err := h.BuildConcurrent(ids, corpus, workers); err != nil {
		t.Fatalf("BuildConcurrent: %v", err)
	}
	build := time.Since(t0)
	fmt.Fprintf(os.Stderr, "[sift] build done in %v (%.0f vec/s, workers=%d)\n",
		build.Round(time.Millisecond), float64(len(corpus))/build.Seconds(), workers)

	t.Logf("Rostam SIFT-1M: corpus=%d dim=%d  build=%v (%.0f vec/s)  [M=16 efC=200, single-thread build]",
		len(corpus), len(corpus[0]), build.Round(time.Millisecond), float64(len(corpus))/build.Seconds())
	t.Logf("%-8s %-12s %-12s", "efSearch", "recall@10", "QPS(1-thread)")

	// Build the truth sets once.
	truth := make([]map[uint64]bool, len(queries))
	for qi := range queries {
		m := make(map[uint64]bool, k)
		for _, id := range gt[qi][:k] {
			m[uint64(id)+1] = true // Rostam ids are 1-based; SIFT gt is 0-based
		}
		truth[qi] = m
	}

	dst := make([]Result, 0, k)
	for _, ef := range []int{16, 32, 64, 128, 256, 512} {
		h.cfg.EfSearch = ef // reuse the same graph; EfSearch only affects query width
		var matches int
		var elapsed time.Duration
		for qi, q := range queries {
			s := time.Now()
			res, _ := h.SearchInto(dst[:0], q, k, Filter{})
			elapsed += time.Since(s)
			for _, r := range res {
				if truth[qi][r.ID] {
					matches++
				}
			}
		}
		recall := float64(matches) / float64(len(queries)*k)
		qps := float64(len(queries)) / elapsed.Seconds()
		fmt.Fprintf(os.Stderr, "[sift] efSearch=%-4d recall@10=%.4f QPS=%.0f\n", ef, recall, qps)
		t.Logf("%-8d %-12.4f %-12.0f", ef, recall, qps)
	}
}

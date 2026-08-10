// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestSIFTLatencyQPS measures Rostam's in-process query latency (p50/p99, single
// thread) and saturated throughput (all cores busy) on SIFT-1M, at matched
// efSearch points. This is the in-process / embedded side of the latency+QPS
// comparison with Qdrant (bench/sift1m/qdrant_latency_qps.py): Rostam here pays
// NO network round-trip — it is a library call — which is the point of the
// embedded deployment, but means it is not directly comparable to a networked
// service except on raw per-query work. Opt-in: ROSTAM_SIFT1M=1.
//
//	ROSTAM_SIFT1M=1 go test ./vector/ -run TestSIFTLatencyQPS -v -timeout 30m
func TestSIFTLatencyQPS(t *testing.T) {
	if os.Getenv("ROSTAM_SIFT1M") != "1" {
		t.Skip("set ROSTAM_SIFT1M=1 with dataset at /tmp/rostam-sift1m/sift/ to run")
	}
	dir := filepath.Join(os.TempDir(), "rostam-sift1m", "sift")
	if d := os.Getenv("ROSTAM_SIFT_DIR"); d != "" {
		dir = d
	}
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
	ids := make([]uint64, len(corpus))
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	if err := h.BuildConcurrent(ids, corpus, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatalf("BuildConcurrent: %v", err)
	}

	truth := make([]map[uint64]bool, len(queries))
	for qi := range queries {
		m := make(map[uint64]bool, k)
		for _, id := range gt[qi][:k] {
			m[uint64(id)+1] = true // Rostam ids are 1-based; SIFT gt is 0-based
		}
		truth[qi] = m
	}

	cores := runtime.GOMAXPROCS(0)
	fmt.Fprintf(os.Stderr, "[lat] Rostam IN-PROCESS (no network), %d cores, k=%d\n", cores, k)
	fmt.Fprintf(os.Stderr, "%-6s %-9s %-10s %-10s %-10s %-12s\n",
		"ef", "recall", "p50(us)", "p99(us)", "mean(us)", "satQPS")

	for _, ef := range []int{16, 64, 128} {
		h.cfg.EfSearch = ef // reuse the same graph; EfSearch only affects query width

		// --- single-thread latency: time every query once ---
		lat := make([]time.Duration, len(queries))
		dst := make([]Result, 0, k)
		var matches int
		for qi, q := range queries {
			s := time.Now()
			res, _ := h.SearchInto(dst[:0], q, k, Filter{})
			lat[qi] = time.Since(s)
			for _, r := range res {
				if truth[qi][r.ID] {
					matches++
				}
			}
		}
		recall := float64(matches) / float64(len(queries)*k)
		sort.Slice(lat, func(a, b int) bool { return lat[a] < lat[b] })
		p50 := lat[len(lat)*50/100]
		p99 := lat[len(lat)*99/100]
		var sum time.Duration
		for _, d := range lat {
			sum += d
		}
		mean := sum / time.Duration(len(lat))

		// --- saturated throughput: all cores busy for ~3s ---
		var count int64
		var wg sync.WaitGroup
		deadline := time.Now().Add(3 * time.Second)
		start := time.Now()
		for w := 0; w < cores; w++ {
			wg.Add(1)
			go func(seed int) {
				defer wg.Done()
				d := make([]Result, 0, k)
				i := seed
				var local int64
				for time.Now().Before(deadline) {
					for j := 0; j < 64; j++ { // batch before re-checking the clock
						_, _ = h.SearchInto(d[:0], queries[i%len(queries)], k, Filter{})
						i++
						local++
					}
				}
				atomic.AddInt64(&count, local)
			}(w * 977)
		}
		wg.Wait()
		qps := float64(count) / time.Since(start).Seconds()

		fmt.Fprintf(os.Stderr, "%-6d %-9.4f %-10.1f %-10.1f %-10.1f %-12.0f\n",
			ef, recall, float64(p50.Microseconds()), float64(p99.Microseconds()), float64(mean.Microseconds()), qps)
		t.Logf("ef=%d recall=%.4f p50=%dus p99=%dus mean=%dus satQPS=%.0f",
			ef, recall, p50.Microseconds(), p99.Microseconds(), mean.Microseconds(), qps)
	}
}

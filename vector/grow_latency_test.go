// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// THE MEASUREMENT: what a grow boundary costs a concurrent query.
//
// This is the A/B behind reserve.go. It builds an index whose slabs are sized
// EXACTLY to n slots (BuildConcurrent presizes to the element count), starts a
// searcher sampling per-query latency, then inserts one more vector — which is
// the insert that must grow both slabs. On the legacy path that is a full
// copy/remap of the whole slab under the write lock and the searcher stalls for
// its entire duration; on the reservation path it is one syscall against the new
// tail.
//
// Opt-in, because a meaningful measurement needs a multi-gigabyte slab:
//
//	ROSTAM_GROW_LATENCY=1 go test ./vector -run TestGrowStallLatency -v -timeout 60m
//
// Tunable via ROSTAM_GROW_N (default 500000) and ROSTAM_GROW_DIM (default 768).
// Run it on a quiet, dedicated box: the number being reported is a worst-case
// tail, which is exactly the statistic a noisy laptop fabricates.

func growLatencyEnvInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return def
}

// latencyStats is the per-query latency distribution a searcher observed while
// the writer crossed the boundary.
type latencyStats struct {
	n               int
	maxD, p999, p50 time.Duration
	growD           time.Duration // wall time of the insert that grew the slabs
}

func (s latencyStats) String() string {
	return fmt.Sprintf("queries=%d p50=%s p99.9=%s MAX=%s | growing-insert=%s",
		s.n, s.p50.Round(time.Microsecond), s.p999.Round(time.Microsecond),
		s.maxD.Round(time.Microsecond), s.growD.Round(time.Microsecond))
}

func summarize(samples []time.Duration, grow time.Duration) latencyStats {
	if len(samples) == 0 {
		return latencyStats{growD: grow}
	}
	sorted := append([]time.Duration(nil), samples...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	idx := func(q float64) time.Duration {
		i := int(q * float64(len(sorted)-1))
		return sorted[i]
	}
	return latencyStats{
		n:     len(sorted),
		p50:   idx(0.50),
		p999:  idx(0.999),
		maxD:  sorted[len(sorted)-1],
		growD: grow,
	}
}

// measureGrowStall builds an index presized to exactly n vectors, then times a
// single boundary-crossing insert against a live searcher.
func measureGrowStall(t *testing.T, cfg Config, n int, reserved bool) latencyStats {
	t.Helper()
	if reserved {
		// The production threshold, stated explicitly so the two arms differ in
		// exactly one knob. ROSTAM_GROW_THRESHOLD lowers it for a quick smoke run
		// at a slab size that would otherwise stay below it.
		withSmallReservations(t, int64(growLatencyEnvInt("ROSTAM_GROW_THRESHOLD", 32<<20)), 64<<20, 64)
	} else {
		withSmallReservations(t, math.MaxInt64, 64<<20, 64)
	}

	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = h.Close() }()

	rng := rand.New(rand.NewSource(int64(n)))
	ids := make([]uint64, n)
	vecs := make([][]float32, n)
	for i := range vecs {
		ids[i] = uint64(i)
		v := make([]float32, cfg.Dim)
		for d := range v {
			v[d] = rng.Float32()
		}
		vecs[i] = v
	}
	buildStart := time.Now()
	if err := h.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	t.Logf("built %d x %dd in %s (vec cap=%d floats, level0 cap=%d u32, codes cap=%d, reserved vec=%v graph=%v codes=%v)",
		n, cfg.Dim, time.Since(buildStart).Round(time.Millisecond),
		cap(h.arena.vecs), cap(h.level0), cap(h.arena.codes),
		h.arena.vecsRes != nil, h.graphRes != nil, h.arena.codesRes != nil)
	// Release the source vectors — a second copy of the whole dataset — and settle
	// the heap before measuring, so a GC cycle triggered by the build does not land
	// in the sample window and get mistaken for the grow.
	query := append([]float32(nil), vecs[0]...)
	clear(vecs) // drop the per-vector backing arrays; `vecs` itself dies at return
	runtime.GC()

	// The searcher. It samples continuously; the interesting sample is whichever
	// query overlapped the grow.
	var stop atomic.Bool
	var wg sync.WaitGroup
	samplesCh := make(chan []time.Duration, 1)
	wg.Add(1)
	go func() {
		defer wg.Done()
		qrng := rand.New(rand.NewSource(4242))
		q := append([]float32(nil), query...)
		var dst []Result
		samples := make([]time.Duration, 0, 1<<16)
		for !stop.Load() {
			for d := range q {
				q[d] = qrng.Float32()
			}
			start := time.Now()
			var serr error
			dst, serr = h.SearchInto(dst[:0], q, 10, Filter{})
			samples = append(samples, time.Since(start))
			if serr != nil {
				t.Errorf("search: %v", serr)
				break
			}
		}
		samplesCh <- samples
	}()

	// Let the searcher settle so its steady-state percentiles are real.
	time.Sleep(2 * time.Second)

	extra := make([]float32, cfg.Dim)
	for d := range extra {
		extra[d] = rng.Float32()
	}
	beforeCap := cap(h.arena.vecs)
	growStart := time.Now()
	if _, _, err := h.Insert(uint64(n), extra, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatal(err)
	}
	grow := time.Since(growStart)
	if cap(h.arena.vecs) == beforeCap {
		t.Fatalf("the boundary insert did not grow the vector slab (cap stayed %d) — nothing was measured", beforeCap)
	}

	// A short tail so the stalled queries that were queued behind the grow are
	// all sampled.
	time.Sleep(500 * time.Millisecond)
	stop.Store(true)
	wg.Wait()
	return summarize(<-samplesCh, grow)
}

func TestGrowStallLatency(t *testing.T) {
	if os.Getenv("ROSTAM_GROW_LATENCY") == "" {
		t.Skip("set ROSTAM_GROW_LATENCY=1 to run the grow-boundary latency A/B (needs multi-GB RAM)")
	}
	n := growLatencyEnvInt("ROSTAM_GROW_N", 500_000)
	dim := growLatencyEnvInt("ROSTAM_GROW_DIM", 768)

	base := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 40, EfSearch: 64, Seed: 11}
	mmapCfg := func() Config {
		c := base
		dir := t.TempDir()
		c.Quant = QuantSQ8
		c.QuantStorage = QuantMmap
		c.MmapPath = filepath.Join(dir, "vecs.dat")
		c.GraphMmapPath = filepath.Join(dir, "graph.dat")
		return c
	}

	arms := []struct {
		name     string
		cfg      func() Config
		reserved bool
	}{
		{"heap/legacy", func() Config { return base }, false},
		{"heap/reserved", func() Config { return base }, true},
		{"mmap/legacy", mmapCfg, false},
		{"mmap/reserved", mmapCfg, true},
	}
	results := map[string]latencyStats{}
	for _, arm := range arms {
		t.Run(arm.name, func(t *testing.T) {
			s := measureGrowStall(t, arm.cfg(), n, arm.reserved)
			results[arm.name] = s
			t.Logf("%-14s %s", arm.name, s)
		})
	}
	t.Log("=== grow-boundary latency, worst query around the boundary ===")
	for _, arm := range arms {
		t.Logf("%-14s %s", arm.name, results[arm.name])
	}
}

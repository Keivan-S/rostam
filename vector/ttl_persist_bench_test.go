// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// Wave-8 benchmarks for TTL sweep (the periodic O(n) expiry scan) and the mmap
// instant-restart pair (SavePersist / OpenPersist). The matrix flags the sweep as
// an O(n) arena scan and instant-restart as a headline latency feature; neither
// was benchmarked. See BENCHMARKS.md.

// BenchmarkTTLSweep measures the point-TTL sweep's O(n) arena scan. sweepOnce
// unconditionally scans every live slot (arena.idMap) checking each deadline, so
// the scan cost is independent of how many points expire. The index is built ONCE
// with far-future TTLs (every slot carries a deadline, none expired), then swept
// repeatedly: each sweep does the full O(n) scan and removes nothing, so it is
// idempotent and needs no per-iteration rebuild. (Removal/tombstone cost is on
// top, proportional to the expired count.)
func BenchmarkTTLSweep(b *testing.B) {
	const n, dim = 20_000, 128
	corpus := makeCorpus(n, dim, 42)
	h, err := newHNSW(Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		b.Fatal(err)
	}
	defer h.Close()
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }
	for id := 1; id <= n; id++ {
		if _, _, err := h.Insert(uint64(id), corpus[id-1], time.Hour, nil, nil, nil, CASCond{}); err != nil {
			b.Fatal(err)
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = h.sweepOnce() // full O(n) scan, removes 0 (idempotent)
	}
}

// BenchmarkKeyTTLSweep measures the per-key-TTL scan: every point carries a
// far-future per-key deadline, so the sweep's per-key pass examines each slot's
// deadline map every scan (none expired → no drop). Built once, swept repeatedly
// (idempotent), so only the scan is measured — distinct from the point-level pass.
func BenchmarkKeyTTLSweep(b *testing.B) {
	const n, dim = 20_000, 128
	corpus := makeCorpus(n, dim, 42)
	h, err := newHNSW(Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		b.Fatal(err)
	}
	defer h.Close()
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }
	for id := 1; id <= n; id++ {
		meta := Metadata{"temp": NewString("hot")}
		keyTTL := map[string]int64{"temp": 3_600_000} // far-future per-key deadline
		if _, _, err := h.Insert(uint64(id), corpus[id-1], 0, meta, nil, keyTTL, CASCond{}); err != nil {
			b.Fatal(err)
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = h.sweepOnce() // full O(n) scan + per-key deadline check, removes 0
	}
}

// buildPersistableBench builds an mmap+graph-mmap SQ8 index (the only config that
// supports SavePersist / openPersist) over n normalized vectors in dir.
func buildPersistableBench(b *testing.B, dir string, n, dim int) (*hnsw, Config) {
	b.Helper()
	corpus := makeCorpus(n, dim, 7)
	for _, v := range corpus {
		normalize(v)
	}
	cfg := Config{
		Dim: dim, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1,
		Quant: QuantSQ8, QuantStorage: QuantMmap, RescoreFactor: 3,
		MmapPath:      filepath.Join(dir, "vecs.dat"),
		GraphMmapPath: filepath.Join(dir, "graph.dat"),
	}
	h, err := newHNSW(cfg)
	if err != nil {
		b.Fatal(err)
	}
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	if err := h.BuildConcurrent(ids, corpus, runtime.GOMAXPROCS(0)); err != nil {
		b.Fatal(err)
	}
	return h, cfg
}

// BenchmarkSavePersist measures writing the persist sidecar (per-slot ids, node
// levels, level-0 lengths, entry point) + flushing the mmap — the checkpoint cost.
func BenchmarkSavePersist(b *testing.B) {
	const n, dim = 20_000, 128
	dir := b.TempDir()
	h, _ := buildPersistableBench(b, dir, n, dim)
	defer h.Close()
	metaPath := filepath.Join(dir, "meta.bin")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := h.SavePersist(metaPath); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkOpenPersist measures the instant-restart path: re-map the vec + graph
// mmap files and the sidecar with NO graph rebuild / no insert loop. This is the
// headline restart-latency claim — it should be far below a from-scratch build.
func BenchmarkOpenPersist(b *testing.B) {
	const n, dim = 20_000, 128
	dir := b.TempDir()
	h, cfg := buildPersistableBench(b, dir, n, dim)
	metaPath := filepath.Join(dir, "meta.bin")
	if err := h.SavePersist(metaPath); err != nil {
		b.Fatal(err)
	}
	_ = h.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h2, err := openPersist(cfg, metaPath)
		if err != nil {
			b.Fatal(err)
		}
		_ = h2.Close()
	}
}

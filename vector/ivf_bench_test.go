// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"runtime"
	"strconv"
	"testing"
)

// Wave-1 benchmarks for the IVF family (IVF-Flat + IVF-PQ search, insert, build,
// and k-means training). Before this file the IVF index — a whole index type with
// k-means training — had ZERO benchmark coverage. See BENCHMARKS.md.
//
// Corpus sizes are kept moderate so the suite finishes in reasonable wall-clock
// while still being large enough to exercise real cell pruning (nlist defaults to
// ~4*sqrt(N), so 100k -> ~1265 cells). Run a single bench with, e.g.:
//   go test ./vector/ -run '^$' -bench BenchmarkIVFFlatSearch -benchmem

// buildTrainedIVF returns an IVF index trained over n dim-vectors via
// BuildConcurrent (which trains immediately on the supplied set). pqm>0 selects
// IVF-PQ with that many sub-quantizers; pqm==0 selects IVF-Flat.
func buildTrainedIVF(b *testing.B, n, dim, nlist, pqm int) *ivf {
	b.Helper()
	cfg := DefaultConfig()
	cfg.Dim = dim
	cfg.Metric = L2
	cfg.Seed = 42
	cfg.IndexType = IndexIVF
	cfg.IVFNlist = nlist
	if pqm > 0 {
		cfg.IVFPQ = true
		cfg.IVFPQM = pqm
		cfg.IVFRerank = true
	}
	ix, err := newIVF(cfg)
	if err != nil {
		b.Fatal(err)
	}
	corpus := makeCorpus(n, dim, 42)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	if err := ix.BuildConcurrent(ids, corpus, 0); err != nil {
		b.Fatal(err)
	}
	return ix
}

// BenchmarkIVFFlatSearch sweeps nprobe to expose the cell-fan-out vs latency curve
// of trained IVF-Flat search. nprobe is mutated on the built index (ix.nprobe is
// what Search reads), so the 100k corpus is built once.
func BenchmarkIVFFlatSearch(b *testing.B) {
	const dim, n, k = 128, 100_000, 10
	ix := buildTrainedIVF(b, n, dim, 0, 0) // nlist=0 -> ~4*sqrt(N) cells
	defer ix.Close()
	queries := makeCorpus(1_000, dim, 99)
	for _, np := range []int{1, 4, 8, 16, 32, 64} {
		ix.nprobe = np
		b.Run("nprobe="+strconv.Itoa(np), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := ix.Search(queries[i%len(queries)], k); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkIVFPQSearch sweeps nprobe for IVF-PQ (ADC LUT scoring + rerank). The
// ADC path is the suspected hotspot: per-cell LUT build + over-collect & float32
// rescore. Compare nprobe scaling against BenchmarkIVFFlatSearch at equal nprobe.
func BenchmarkIVFPQSearch(b *testing.B) {
	const dim, n, k, pqm = 128, 100_000, 10, 16
	ix := buildTrainedIVF(b, n, dim, 0, pqm)
	defer ix.Close()
	queries := makeCorpus(1_000, dim, 99)
	for _, np := range []int{1, 4, 8, 16, 32, 64} {
		ix.nprobe = np
		b.Run("nprobe="+strconv.Itoa(np), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := ix.Search(queries[i%len(queries)], k); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkIVFInsert measures per-point insert into an already-trained index
// (coarse-quantizer assignment + list append), separately for IVF-Flat and IVF-PQ
// (which additionally residual-encodes the vector). Ids continue past the corpus.
func BenchmarkIVFInsert(b *testing.B) {
	const dim, n = 128, 50_000
	cases := []struct {
		name string
		pqm  int
	}{{"flat", 0}, {"pq", 16}}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			ix := buildTrainedIVF(b, n, dim, 0, tc.pqm)
			defer ix.Close()
			extra := makeCorpus(b.N, dim, 7)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, _, err := ix.Insert(uint64(n+i+1), extra[i], 0, nil, nil, nil, CASCond{}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkIVFBuild measures end-to-end BuildConcurrent (k-means train + assign +
// list population), swept over worker counts to expose parallel-build scaling.
// Each iteration builds a fresh index, so n is kept modest.
func BenchmarkIVFBuild(b *testing.B) {
	const dim, n, pqm = 128, 20_000, 0
	corpus := makeCorpus(n, dim, 42)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	for _, w := range []int{1, 4, runtime.NumCPU()} {
		b.Run("workers="+strconv.Itoa(w), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				cfg := DefaultConfig()
				cfg.Dim = dim
				cfg.Metric = L2
				cfg.Seed = 42
				cfg.IndexType = IndexIVF
				ix, err := newIVF(cfg)
				if err != nil {
					b.Fatal(err)
				}
				if err := ix.BuildConcurrent(ids, corpus, w); err != nil {
					b.Fatal(err)
				}
				ix.Close()
			}
		})
	}
}

// BenchmarkIVFSearchFilteredWith exercises the external-metadata gather path
// (gatherLockedWith / metaOf != nil) used by the named & multi-vector families,
// where the predicate is evaluated against a caller-supplied metadata provider.
// Sweeps nprobe for both IVF-Flat and IVF-PQ; the fix makes allocs/op constant in
// nprobe (pooled visited set + per-query LUT) just like the plain gather paths.
func BenchmarkIVFSearchFilteredWith(b *testing.B) {
	const dim, n, k = 128, 100_000, 10
	// Pre-built metadata so the provider itself allocates nothing per call (real
	// providers read a shared payload store); otherwise the closure's per-candidate
	// map alloc would dominate and mask the gather path being measured.
	hitMeta := Metadata{"bucket": NewString("hit")}
	missMeta := Metadata{"bucket": NewString("miss")}
	metaOf := func(id uint64) Metadata {
		if id%2 == 0 {
			return hitMeta
		}
		return missMeta
	}
	filter := Filter{Op: FilterEq, Field: "bucket", Value: NewString("hit")}
	cases := []struct {
		name string
		pqm  int
	}{{"flat", 0}, {"pq", 16}}
	for _, tc := range cases {
		ix := buildTrainedIVF(b, n, dim, 0, tc.pqm)
		queries := makeCorpus(1_000, dim, 99)
		for _, np := range []int{8, 32, 64} {
			ix.nprobe = np
			b.Run(tc.name+"/nprobe="+strconv.Itoa(np), func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if _, err := ix.SearchFilteredWith(nil, queries[i%len(queries)], k, filter, metaOf); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
		ix.Close()
	}
}

// BenchmarkIVFFlatSearchBatchedAB is the A/B for the batched IVF-Flat cell scan
// (ivf.gatherFlatBatched) against the per-pair loop it replaced, over the SAME
// trained index — the batchedExpand knob is flipped between arms rather than the
// index rebuilt, because two builds would be two different k-means partitions and
// the delta would be mostly training nondeterminism, not scan cost.
//
// The two arms are INTERLEAVED per nprobe (batched, then per-pair, adjacent
// sub-benchmarks) so a thermal or frequency drift over the run perturbs both arms
// alike; take medians across -count runs. The arms are bit-identical in output
// (TestIVFBatchedCellScanMatchesPerPair), so any difference here is throughput.
//
//	go test ./vector -run '^$' -bench BenchmarkIVFFlatSearchBatchedAB -count=7
func BenchmarkIVFFlatSearchBatchedAB(b *testing.B) {
	const dim, n, k = 128, 100_000, 10
	ix := buildTrainedIVF(b, n, dim, 0, 0)
	defer ix.Close()
	queries := makeCorpus(1_000, dim, 99)
	defer func(prev bool) { batchedExpand = prev }(batchedExpand)
	for _, np := range []int{1, 8, 32, 64} {
		ix.nprobe = np
		for _, arm := range []struct {
			name    string
			batched bool
		}{{"batched", true}, {"perpair", false}} {
			b.Run("nprobe="+strconv.Itoa(np)+"/"+arm.name, func(b *testing.B) {
				batchedExpand = arm.batched
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if _, err := ix.Search(queries[i%len(queries)], k); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// BenchmarkKMeans isolates the k-means coarse-quantizer training cost (the heart
// of IVF build), swept over worker counts to verify the parallel path actually
// scales. k=256 cells over a 50k corpus is a representative training load.
func BenchmarkKMeans(b *testing.B) {
	const dim, n, k = 128, 50_000, 256
	corpus := makeCorpus(n, dim, 42)
	for _, w := range []int{1, 4, runtime.NumCPU()} {
		b.Run("workers="+strconv.Itoa(w), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if c := kmeans(corpus, k, 42, L2, w); len(c) == 0 {
					b.Fatal("kmeans returned no centroids")
				}
			}
		})
	}
}

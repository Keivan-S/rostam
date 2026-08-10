// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
)

// The VDBBench filter shape, reproduced locally.
//
// VectorDBBench's "filter" case runs a RANGE predicate (id >= N) over a
// Cohere-1M corpus (768d) with k=100 at ef=300, where N is chosen so the
// predicate passes ~99% of the corpus. That case costs Rostam 2.39x throughput
// versus the unfiltered run while Milvus answers it for roughly free, and the
// reason is structural: at 99% pass NEITHER fast path applies — filter-first
// declines (990k candidates is far past the crossover) and the m4 bitset gate
// declines too (posting mass dwarfs the traversal it would accelerate) — so
// every traversed candidate pays admitVerdictOf -> liveMeta -> a string-keyed
// map fetch -> the compiled predicate.
//
// This file is the measurement, not the fix: one index, one query set, the same
// k and ef, swept across pass rates so the 99% number can be read against 50%
// and 1% (where m4 already wins) and against the unfiltered baseline (the 1.0x
// Milvus is close to).
//
// Corpus size is a knob because the effect is size-dependent in opposite
// directions for the two costs — posting mass scales with N, traversal does not
// — so a 10k-point run would flatter the gate and hide the real shape. Default
// is 100k (a ~2 minute build at 768d); -rangebench.n raises it.

// rangeBenchCorpusN is the live-point count of the range benchmark corpus.
// Kept modest by default so `go test -bench` stays runnable; the shape of the
// result (filtered/unfiltered ratio) is what carries over to 1M, not the
// absolute QPS.
var rangeBenchCorpusN = 100_000

// rangeBenchIndex builds the benchmark corpus: `n` 768-dimensional points, each
// carrying exactly the VDBBench payload — {"id": <ordinal>} — so a NumGE filter
// over "id" has a known, exact pass rate.
//
// Built ONCE per (n, dim) and shared by every arm via sync.Once, because a
// 768d HNSW build dominates the wall clock of the whole benchmark and rebuilding
// it per arm would make the arms incomparable (different graphs).
func rangeBenchIndex(tb testing.TB, n, dim int) *hnsw {
	tb.Helper()
	key := fmt.Sprintf("%d/%d", n, dim)
	rangeBenchMu.Lock()
	defer rangeBenchMu.Unlock()
	if h := rangeBenchCache[key]; h != nil {
		return h
	}
	corpus := makeCorpus(n, dim, 42)
	h, err := newHNSW(Config{
		Dim: dim, M: 16, EfConstruction: 200, EfSearch: 300, Seed: 1, Metric: L2,
	})
	if err != nil {
		tb.Fatalf("newHNSW: %v", err)
	}
	for i, v := range corpus {
		if _, _, err := h.Insert(uint64(i+1), v, 0, Metadata{"id": NewInt(int64(i))}, nil, nil, CASCond{}); err != nil { //nolint:gosec // bounded loop index
			tb.Fatalf("insert: %v", err)
		}
	}
	if rangeBenchCache == nil {
		rangeBenchCache = map[string]*hnsw{}
	}
	rangeBenchCache[key] = h
	return h
}

var (
	rangeBenchMu    sync.Mutex
	rangeBenchCache map[string]*hnsw
)

// rangeFilterFor returns the NumGE filter over "id" that passes `pass` of an
// n-point corpus whose ids are 0..n-1. pass=0.99 is the VDBBench shape.
func rangeFilterFor(n int, pass float64) Filter {
	bound := int64(float64(n) * (1 - pass))
	return Filter{Op: FilterGte, Field: "id", Value: NewInt(bound)}
}

// BenchmarkRangeFilterVDB is THE benchmark: unfiltered versus a NumGE range
// predicate at three pass rates, same index, same queries, k=100, ef=300.
//
// Read it as ratios against the "unfiltered" arm — that ratio is the number
// VectorDBBench reports as the filter penalty, and the one Milvus keeps near 1.
func BenchmarkRangeFilterVDB(b *testing.B) {
	const dim, k = 768, 100
	n := rangeBenchCorpusN
	h := rangeBenchIndex(b, n, dim)
	queries := makeCorpus(200, dim, 99)

	arms := []struct {
		name string
		f    Filter
	}{
		{"unfiltered", Filter{}},
		{"pass=0.99", rangeFilterFor(n, 0.99)},
		{"pass=0.50", rangeFilterFor(n, 0.50)},
		{"pass=0.01", rangeFilterFor(n, 0.01)},
	}
	for _, arm := range arms {
		b.Run(arm.name, func(b *testing.B) {
			dst := make([]Result, 0, k)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				var err error
				dst, err = h.SearchInto(dst[:0], queries[i%len(queries)], k, arm.f)
				if err != nil {
					b.Fatalf("search: %v", err)
				}
			}
			b.StopTimer()
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
		})
	}
}

// BenchmarkRangeFilterVDBParallel is BenchmarkRangeFilterVDB from every core,
// which is how VectorDBBench drives QPS. The serial form measures the
// per-candidate cost; this one also exposes whatever the admission path does to
// shared cache lines.
func BenchmarkRangeFilterVDBParallel(b *testing.B) {
	const dim, k = 768, 100
	n := rangeBenchCorpusN
	h := rangeBenchIndex(b, n, dim)
	queries := makeCorpus(200, dim, 99)

	arms := []struct {
		name string
		f    Filter
	}{
		{"unfiltered", Filter{}},
		{"pass=0.99", rangeFilterFor(n, 0.99)},
		{"pass=0.50", rangeFilterFor(n, 0.50)},
		{"pass=0.01", rangeFilterFor(n, 0.01)},
	}
	for _, arm := range arms {
		b.Run(arm.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			var seed atomic.Int64
			b.RunParallel(func(pb *testing.PB) {
				dst := make([]Result, 0, k)
				j := int(seed.Add(17))
				for pb.Next() {
					dst, _ = h.SearchInto(dst[:0], queries[j%len(queries)], k, arm.f)
					j++
				}
			})
			b.StopTimer()
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "qps")
		})
	}
}

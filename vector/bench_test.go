// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"strconv"
	"testing"
	"time"
)

func makeCorpus(n, dim int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([][]float32, n)
	for i := range out {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		out[i] = v
	}
	return out
}

func BenchmarkHNSWInsert(b *testing.B) {
	for _, dim := range []int{64, 128, 768} {
		b.Run("dim="+strconv.Itoa(dim), func(b *testing.B) {
			corpus := makeCorpus(b.N, dim, 42)
			h, _ := newHNSW(Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: L2})
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _, _ = h.Insert(uint64(i+1), corpus[i], 0, nil, nil, nil, CASCond{})
			}
		})
	}
}

func BenchmarkHNSWSearch(b *testing.B) {
	const dim = 128
	corpus := makeCorpus(10_000, dim, 42)
	queries := makeCorpus(1_000, dim, 99)
	h, _ := newHNSW(Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: L2})
	for i, v := range corpus {
		_, _, _ = h.Insert(uint64(i+1), v, 0, nil, nil, nil, CASCond{})
	}
	for _, k := range []int{1, 10, 100} {
		b.Run("k="+strconv.Itoa(k), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = h.Search(queries[i%len(queries)], k)
			}
		})
	}
}

// BenchmarkHNSWSearchInto measures the zero-allocation search path: a reused
// dst buffer + the L2 metric (no query normalization). Steady state should
// report 0 allocs/op.
func BenchmarkHNSWSearchInto(b *testing.B) {
	const dim = 128
	corpus := makeCorpus(10_000, dim, 42)
	queries := makeCorpus(1_000, dim, 99)
	h, _ := newHNSW(Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: L2})
	for i, v := range corpus {
		_, _, _ = h.Insert(uint64(i+1), v, 0, nil, nil, nil, CASCond{})
	}
	for _, k := range []int{1, 10, 100} {
		b.Run("k="+strconv.Itoa(k), func(b *testing.B) {
			dst := make([]Result, 0, k)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				dst, _ = h.SearchInto(dst[:0], queries[i%len(queries)], k, Filter{})
			}
		})
	}
}

func BenchmarkDotScalar(b *testing.B) {
	for _, dim := range []int{64, 128, 768, 1536} {
		b.Run("dim="+strconv.Itoa(dim), func(b *testing.B) {
			a := makeCorpus(2, dim, 1)
			b.ResetTimer()
			var sum float32
			for i := 0; i < b.N; i++ {
				sum = dotScalar(a[0], a[1])
			}
			_ = sum
		})
	}
}

func BenchmarkL2SquaredScalar(b *testing.B) {
	for _, dim := range []int{64, 128, 768, 1536} {
		b.Run("dim="+strconv.Itoa(dim), func(b *testing.B) {
			a := makeCorpus(2, dim, 1)
			b.ResetTimer()
			var sum float32
			for i := 0; i < b.N; i++ {
				sum = l2SquaredScalar(a[0], a[1])
			}
			_ = sum
		})
	}
}

// BenchmarkInsertWithTTL measures the per-insert cost when each vector carries
// a TTL deadline. Compare against BenchmarkInsertNoTTL to verify the TTL
// bookkeeping (one extra expires-slice write + clock call) stays under 5 %.
func BenchmarkInsertWithTTL(b *testing.B) {
	const dim = 128
	corpus := makeCorpus(b.N, dim, 42)
	h, _ := newHNSW(Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: L2})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = h.Insert(uint64(i+1), corpus[i], time.Hour, nil, nil, nil, CASCond{})
	}
}

// BenchmarkInsertNoTTL is the baseline for BenchmarkInsertWithTTL: same workload,
// ttl=0 so the expires-slice write is skipped.
func BenchmarkInsertNoTTL(b *testing.B) {
	const dim = 128
	corpus := makeCorpus(b.N, dim, 42)
	h, _ := newHNSW(Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: L2})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = h.Insert(uint64(i+1), corpus[i], 0, nil, nil, nil, CASCond{})
	}
}

// BenchmarkInsertWithQuotaCheck measures per-insert cost with all three
// quotas configured (and far from their caps). Token bucket Take + count
// check + byte estimate together should add well under 1 µs.
func BenchmarkInsertWithQuotaCheck(b *testing.B) {
	const dim = 128
	corpus := makeCorpus(b.N, dim, 42)
	h, _ := newHNSW(Config{
		Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: L2,
		MaxVectors:          1_000_000,
		MaxBytes:            1 << 30,
		MaxInsertsPerSecond: 1_000_000,
	})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = h.Insert(uint64(i+1), corpus[i], 0, nil, nil, nil, CASCond{})
	}
}

// BenchmarkInsertWithMetadata measures per-insert cost when each vector
// carries a 4-key metadata map. Compare against BenchmarkInsertNoTTL —
// metadata adds one map allocation + a SetMetadata call. Target: < 10%
// overhead since the graph-connect cost dominates.
func BenchmarkInsertWithMetadata(b *testing.B) {
	const dim = 128
	corpus := makeCorpus(b.N, dim, 42)
	h, _ := newHNSW(Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: L2})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		meta := Metadata{
			"tenant": NewString("acme"),
			"score":  NewInt(int64(i)),
			"active": NewBool(true),
			"tags":   NewStrings([]string{"a", "b"}),
		}
		_, _, _ = h.Insert(uint64(i+1), corpus[i], 0, meta, nil, nil, CASCond{})
	}
}

// makeSparseCorpus builds n sparse vectors of nnz nonzeros each, drawn from a
// vocabulary of `vocab` dims. Indices are sorted+unique per vector.
func makeSparseCorpus(n, nnz, vocab int, seed int64) []SparseVector {
	rng := rand.New(rand.NewSource(seed))
	out := make([]SparseVector, n)
	for i := range out {
		seen := make(map[uint32]bool, nnz)
		idx := make([]uint32, 0, nnz)
		for len(idx) < nnz {
			d := uint32(rng.Intn(vocab))
			if !seen[d] {
				seen[d] = true
				idx = append(idx, d)
			}
		}
		// sort ascending
		for a := 1; a < len(idx); a++ {
			for b := a; b > 0 && idx[b-1] > idx[b]; b-- {
				idx[b-1], idx[b] = idx[b], idx[b-1]
			}
		}
		vals := make([]float32, nnz)
		for j := range vals {
			vals[j] = float32(rng.Float64())
		}
		out[i] = SparseVector{Indices: idx, Values: vals}
	}
	return out
}

// BenchmarkSearchFiltered measures filtered KNN over a 10k corpus where 50% of
// vectors match the predicate. Compare against BenchmarkHNSWSearch/k=10 to see
// the metadata-predicate + dynamic-ef-widening overhead.
func BenchmarkSearchFiltered(b *testing.B) {
	const dim, n = 128, 10_000
	corpus := makeCorpus(n, dim, 42)
	queries := makeCorpus(1_000, dim, 99)
	h, _ := newHNSW(Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: L2})
	rng := rand.New(rand.NewSource(7))
	for i, v := range corpus {
		bucket := "hit"
		if rng.Float64() < 0.5 {
			bucket = "miss"
		}
		_, _, _ = h.Insert(uint64(i+1), v, 0, Metadata{"bucket": NewString(bucket)}, nil, nil, CASCond{})
	}
	filter := Filter{Op: FilterEq, Field: "bucket", Value: NewString("hit")}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = h.SearchFiltered(queries[i%len(queries)], 10, filter)
	}
}

// BenchmarkSearchFilteredParallel is BenchmarkSearchFiltered run from every
// core at once. The serial benchmark cannot see cross-core contention at all,
// so it could not price the per-rejected-candidate atomic the admission gate
// used to do: one shared cache line, bounced between all 20 hardware threads,
// several times per graph edge on a selective filter.
//
// The filter is deliberately selective enough (~10% match, below the planner's
// filter-first crossover for this corpus so the GRAPH path is exercised, not
// brute force) that most traversed candidates are rejected — which is exactly
// the regime where the counter was hottest.
func BenchmarkSearchFilteredParallel(b *testing.B) {
	const dim, n = 128, 10_000
	corpus := makeCorpus(n, dim, 42)
	queries := makeCorpus(1_000, dim, 99)
	// FilterFirstThreshold 1 keeps the query planner off the brute-force path so
	// the benchmark measures filtered GRAPH traversal (where the admission gate
	// runs per candidate), not an index-narrowed rescan.
	h, _ := newHNSW(Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: L2,
		FilterFirstThreshold: 1})
	rng := rand.New(rand.NewSource(7))
	for i, v := range corpus {
		bucket := "miss"
		if rng.Float64() < 0.1 {
			bucket = "hit"
		}
		_, _, _ = h.Insert(uint64(i+1), v, 0, Metadata{"bucket": NewString(bucket)}, nil, nil, CASCond{})
	}
	filter := Filter{Op: FilterEq, Field: "bucket", Value: NewString("hit")}
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		dst := make([]Result, 0, 10)
		for pb.Next() {
			dst, _ = h.SearchInto(dst[:0], queries[i%len(queries)], 10, filter)
			i++
		}
	})
}

// BenchmarkHybridSearch measures fused dense+sparse search over a 10k corpus
// with ~30-nonzero sparse vectors (SPLADE-scale vocab). Compare against
// BenchmarkHNSWSearch/k=10 for the sparse-lane + fusion overhead.
func BenchmarkHybridSearch(b *testing.B) {
	const dim, n, nnz, vocab = 128, 10_000, 30, 30_000
	corpus := makeCorpus(n, dim, 42)
	sparseCorpus := makeSparseCorpus(n, nnz, vocab, 7)
	queries := makeCorpus(1_000, dim, 99)
	sparseQueries := makeSparseCorpus(1_000, nnz, vocab, 11)
	h, _ := newHNSW(Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: L2})
	for i, v := range corpus {
		sv := sparseCorpus[i]
		_, _, _ = h.Insert(uint64(i+1), v, 0, nil, &sv, nil, CASCond{})
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = h.HybridSearch(queries[i%len(queries)], sparseQueries[i%len(sparseQueries)], 10, HybridOpts{})
	}
}

// BenchmarkSparseSearch measures the pure sparse inverted-index lane (no dense).
func BenchmarkSparseSearch(b *testing.B) {
	const dim, n, nnz, vocab = 128, 10_000, 30, 30_000
	corpus := makeCorpus(n, dim, 42)
	sparseCorpus := makeSparseCorpus(n, nnz, vocab, 7)
	sparseQueries := makeSparseCorpus(1_000, nnz, vocab, 11)
	h, _ := newHNSW(Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: L2})
	for i, v := range corpus {
		sv := sparseCorpus[i]
		_, _, _ = h.Insert(uint64(i+1), v, 0, nil, &sv, nil, CASCond{})
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = h.HybridSearch(nil, sparseQueries[i%len(sparseQueries)], 10, HybridOpts{})
	}
}

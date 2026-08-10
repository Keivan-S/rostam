// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"strconv"
	"sync"
	"testing"
)

// Wave-2 benchmarks for the high-level query surface: the unified Query API
// (fusion / rerank / nested fusion), Recommend, Discover, SearchGroups, SearchMMR,
// and SearchDocs. Before this file none of these end-to-end paths were benchmarked
// (only low-level HNSW Search + HybridSearch were). See BENCHMARKS.md.
//
// All benchmarks share ONE collection built once (querySurfaceCollection): the
// corpus carries dense vectors, ~30-nnz sparse vectors (SPLADE-scale), and a
// "doc_id" grouping field, so every query family has the data it needs. The
// collection is read-only across these benchmarks, which run sequentially.

const (
	qsDim    = 128
	qsN      = 30_000
	qsGroups = 3_000 // ~10 members per doc_id group
	qsNNZ    = 30
	qsVocab  = 30_000
)

var (
	qsOnce    sync.Once
	qsColl    *Collection
	qsQueries [][]float32
	qsSparseQ []SparseVector
)

func querySurfaceCollection(tb testing.TB) (*Collection, [][]float32, []SparseVector) {
	tb.Helper()
	qsOnce.Do(func() {
		c, err := NewCollection("qsbench", Config{
			Dim: qsDim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1,
		})
		if err != nil {
			panic(err)
		}
		corpus := makeCorpus(qsN, qsDim, 42)
		sparse := makeSparseCorpus(qsN, qsNNZ, qsVocab, 7)
		for i := 0; i < qsN; i++ {
			sv := sparse[i]
			meta := Metadata{"doc_id": NewInt(int64(i % qsGroups))}
			if err := c.Insert(uint64(i+1), corpus[i], 0, meta, &sv); err != nil {
				panic(err)
			}
		}
		qsColl = c
		qsQueries = makeCorpus(1_000, qsDim, 99)
		qsSparseQ = makeSparseCorpus(1_000, qsNNZ, qsVocab, 11)
	})
	return qsColl, qsQueries, qsSparseQ
}

// BenchmarkQueryFusion measures the unified Query API in FUSION mode: a dense + a
// sparse prefetch lane fused by RRF (the Qdrant-parity hybrid path).
func BenchmarkQueryFusion(b *testing.B) {
	c, queries, sparseQ := querySurfaceCollection(b)
	const k = 10
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spec := QuerySpec{
			Mode: ModeFusion,
			Prefetch: srcsBench(
				QueryLeaf{Kind: LeafDense, Dense: queries[i%len(queries)], K: k},
				QueryLeaf{Kind: LeafSparse, Sparse: sparseQ[i%len(sparseQ)], K: k, ScoreDesc: true},
			),
			Method: FusionRRF,
			K:      k,
		}
		if _, err := c.Query(spec); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkQueryRerank measures RERANK mode: collect a dense + sparse candidate
// union, then re-score it against the dense root leaf.
func BenchmarkQueryRerank(b *testing.B) {
	c, queries, sparseQ := querySurfaceCollection(b)
	const k = 10
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := queries[i%len(queries)]
		spec := QuerySpec{
			Mode: ModeRerank,
			Root: QueryLeaf{Kind: LeafDense, Dense: q, K: k},
			Prefetch: srcsBench(
				QueryLeaf{Kind: LeafDense, Dense: q, K: 4 * k},
				QueryLeaf{Kind: LeafSparse, Sparse: sparseQ[i%len(sparseQ)], K: 4 * k, ScoreDesc: true},
			),
			K: k,
		}
		if _, err := c.Query(spec); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkQueryNestedFusion measures the recursion path: a parent FUSION whose
// prefetch is [a dense leaf, a NESTED 2-lane dense+sparse FUSION sub-spec].
func BenchmarkQueryNestedFusion(b *testing.B) {
	c, queries, sparseQ := querySurfaceCollection(b)
	const k = 10
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := queries[i%len(queries)]
		sub := QuerySpec{
			Mode: ModeFusion,
			Prefetch: srcsBench(
				QueryLeaf{Kind: LeafDense, Dense: q, K: k},
				QueryLeaf{Kind: LeafSparse, Sparse: sparseQ[i%len(sparseQ)], K: k, ScoreDesc: true},
			),
			Method: FusionRRF,
			K:      k,
		}
		parent := QuerySpec{
			Mode: ModeFusion,
			Prefetch: []QuerySource{
				LeafSource(QueryLeaf{Kind: LeafDense, Dense: q, K: k}),
				{Spec: &sub},
			},
			Method: FusionRRF,
			K:      k,
		}
		if _, err := c.Query(parent); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkRecommend measures the example-synthesis path: derive mean(positive) -
// mean(negative) and search. Example ids are excluded from the results.
func BenchmarkRecommend(b *testing.B) {
	c, _, _ := querySurfaceCollection(b)
	const k = 10
	opts := RecommendOpts{Positive: []uint64{1, 2, 3}, Negative: []uint64{4, 5}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := c.Recommend(k, opts); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkDiscover measures context-pair discovery scoring around an anchor.
func BenchmarkDiscover(b *testing.B) {
	c, queries, _ := querySurfaceCollection(b)
	const k = 10
	opts := DiscoverOpts{
		Target:  queries[0],
		Context: []ContextPair{{Positive: 1, Negative: 2}, {Positive: 3, Negative: 4}},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := c.Discover(k, opts); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSearchGroups measures group-by-field collapse over a candidate pool.
func BenchmarkSearchGroups(b *testing.B) {
	c, queries, _ := querySurfaceCollection(b)
	const k = 10
	opts := GroupOpts{GroupBy: "doc_id", GroupSize: 3}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := c.SearchGroups(queries[i%len(queries)], k, opts); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSearchMMR sweeps Lambda (relevance/diversity tradeoff) for the MMR
// re-ranking path: over-collect FetchK candidates, then greedily diversify.
func BenchmarkSearchMMR(b *testing.B) {
	c, queries, _ := querySurfaceCollection(b)
	const k = 10
	for _, lambda := range []float64{0.3, 0.7} {
		b.Run("lambda="+strconv.FormatFloat(lambda, 'f', 1, 64), func(b *testing.B) {
			opts := MMROpts{Lambda: lambda, FetchK: 4 * k}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := c.SearchMMR(queries[i%len(queries)], k, opts); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkSearchDocs measures search + full Document assembly (vec + metadata +
// ttl + sparse) versus the bare Search path.
func BenchmarkSearchDocs(b *testing.B) {
	c, queries, _ := querySurfaceCollection(b)
	const k = 10
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := c.SearchDocs(queries[i%len(queries)], k, Filter{}); err != nil {
			b.Fatal(err)
		}
	}
}

// srcsBench wraps leaves as leaf QuerySources (the bench-local analogue of the
// test helper srcs).
func srcsBench(leaves ...QueryLeaf) []QuerySource {
	out := make([]QuerySource, len(leaves))
	for i, l := range leaves {
		out[i] = LeafSource(l)
	}
	return out
}

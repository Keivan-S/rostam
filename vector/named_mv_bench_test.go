// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"strconv"
	"sync"
	"testing"
)

// Wave-3 benchmarks for the named-vector and multi-vector (late-interaction)
// families — two whole subsystems with no prior benchmark coverage. Each family
// shares ONE index built once. See BENCHMARKS.md.

// ---- Named vectors ----

const (
	nmN   = 30_000
	nmDim = 128
	nmNNZ = 30
	nmVoc = 30_000
)

var (
	nmOnce   sync.Once
	nmColl   *NamedCollection
	nmDenseQ [][]float32
	nmSparQ  []SparseVector
)

func namedBenchCollection(tb testing.TB) (*NamedCollection, [][]float32, []SparseVector) {
	tb.Helper()
	nmOnce.Do(func() {
		cfg := map[string]NamedVectorParams{
			"dense": {Dim: nmDim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1},
			"terms": {Sparse: true},
		}
		nc, err := NewNamedCollection("bench/named", cfg)
		if err != nil {
			panic(err)
		}
		corpus := makeCorpus(nmN, nmDim, 42)
		sparse := makeSparseCorpus(nmN, nmNNZ, nmVoc, 7)
		for i := 0; i < nmN; i++ {
			sv := sparse[i]
			err := nc.InsertSparse(uint64(i+1),
				map[string][]float32{"dense": corpus[i]},
				map[string]*SparseVector{"terms": &sv}, nil, 0)
			if err != nil {
				panic(err)
			}
		}
		nmColl = nc
		nmDenseQ = makeCorpus(1_000, nmDim, 99)
		nmSparQ = makeSparseCorpus(1_000, nmNNZ, nmVoc, 11)
	})
	return nmColl, nmDenseQ, nmSparQ
}

// BenchmarkNamedSearch measures per-space dense kNN on a named collection.
func BenchmarkNamedSearch(b *testing.B) {
	nc, dq, _ := namedBenchCollection(b)
	const k = 10
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := nc.SearchNamed("dense", dq[i%len(dq)], k, Filter{}); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkNamedSparseSearch measures the per-space sparse inverted-index lane.
func BenchmarkNamedSparseSearch(b *testing.B) {
	nc, _, sq := namedBenchCollection(b)
	const k = 10
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := sq[i%len(sq)]
		if _, err := nc.SearchNamedSparse("terms", &s, k, Filter{}); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkNamedHybrid measures cross-space dense+sparse fusion (RRF).
func BenchmarkNamedHybrid(b *testing.B) {
	nc, dq, sq := namedBenchCollection(b)
	const k = 10
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := sq[i%len(sq)]
		if _, err := nc.NamedHybrid("dense", dq[i%len(dq)], "terms", &s, k, HybridOpts{Method: FusionRRF}); err != nil {
			b.Fatal(err)
		}
	}
}

// ---- Multi-vector (late interaction / ColBERT-style MaxSim) ----

const (
	mvN      = 5_000 // documents
	mvDim    = 128
	mvDocTok = 32 // tokens per document (ColBERT-passage scale)
	mvQryTok = 16 // tokens per query
	mvNNZ    = 30
	mvVoc    = 30_000
)

var (
	mvOnce  sync.Once
	mvIdx   *MultiVectorIndex
	mvQ     [][][]float32  // pre-built query token matrices
	mvSparQ []SparseVector // per-query doc-level sparse (for MV hybrid)
)

func mvBenchIndex(tb testing.TB) (*MultiVectorIndex, [][][]float32, []SparseVector) {
	tb.Helper()
	mvOnce.Do(func() {
		m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: mvDim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
		if err != nil {
			panic(err)
		}
		docSparse := makeSparseCorpus(mvN, mvNNZ, mvVoc, 7) // doc-level sparse for MVHybrid
		for i := 0; i < mvN; i++ {
			tokens := makeCorpus(mvDocTok, mvDim, int64(1000+i)) // distinct token set per doc
			sv := docSparse[i]
			if _, err := m.AddCASKeyTTLSparse(uint64(i+1), tokens, nil, nil, &sv, CASCond{}); err != nil {
				panic(err)
			}
		}
		mvIdx = m
		const nq = 200
		mvQ = make([][][]float32, nq)
		for i := range mvQ {
			mvQ[i] = makeCorpus(mvQryTok, mvDim, int64(50_000+i))
		}
		mvSparQ = makeSparseCorpus(nq, mvNNZ, mvVoc, 11)
	})
	return mvIdx, mvQ, mvSparQ
}

// BenchmarkMVSearch measures late-interaction MaxSim search, sweeping the
// first-stage candidate width (CandidatesPerToken).
func BenchmarkMVSearch(b *testing.B) {
	m, queries, _ := mvBenchIndex(b)
	const k = 10
	for _, cpt := range []int{50, 200} {
		b.Run("cpt="+strconv.Itoa(cpt), func(b *testing.B) {
			opts := MultiSearchOpts{CandidatesPerToken: cpt}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := m.Search(queries[i%len(queries)], k, opts); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkMVQueryRerank measures the MV Query-API RERANK path (maxSimRerankLocked):
// a candidate union from an MV + sparse prefetch, re-scored by the MV root leaf.
// Exercises the same view-based MaxSim rerank as MVSearch over an explicit cand set.
func BenchmarkMVQueryRerank(b *testing.B) {
	m, queries, sq := mvBenchIndex(b)
	const k = 10
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := queries[i%len(queries)]
		s := sq[i%len(sq)]
		spec := QuerySpec{
			Mode: ModeRerank,
			K:    k,
			Root: QueryLeaf{Kind: LeafMVMaxSim, Tokens: q, ScoreDesc: true},
			Prefetch: []QuerySource{
				LeafSource(QueryLeaf{Kind: LeafMVMaxSim, Tokens: q, ScoreDesc: true}),
				LeafSource(QueryLeaf{Kind: LeafSparse, Sparse: s, ScoreDesc: true}),
			},
		}
		if _, err := m.Query(spec); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkMVHybrid measures the MV MaxSim lane fused with a doc-level sparse lane.
func BenchmarkMVHybrid(b *testing.B) {
	m, queries, sq := mvBenchIndex(b)
	const k = 10
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := sq[i%len(sq)]
		if _, err := m.MVHybrid(queries[i%len(queries)], &s, k, HybridOpts{Method: FusionRRF}); err != nil {
			b.Fatal(err)
		}
	}
}

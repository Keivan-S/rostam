// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"context"
	"math/rand"
	"testing"

	"github.com/rostamlabs/rostam/grpcapi/grpcsvc"
	"github.com/rostamlabs/rostam/sdk/pb"
)

const (
	hybridN     = 5_000
	hybridDim   = 128
	hybridTerms = 8    // nonzero sparse terms per point/query
	hybridVocab = 1000 // sparse index space
)

// sparseVec builds hybridTerms deterministic (index, value) pairs over hybridVocab.
// Indices are strictly ascending and unique (one per evenly-sized bucket), as the
// engine requires.
func sparseVec(seed int64) (idx []uint32, val []float32) {
	r := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic benchmark data
	step := hybridVocab / hybridTerms
	idx = make([]uint32, hybridTerms)
	val = make([]float32, hybridTerms)
	for i := range idx {
		idx[i] = uint32(i*step + r.Intn(step)) // bucket i ⇒ strictly ascending, unique
		val[i] = r.Float32()
	}
	return idx, val
}

// seedHybridCollection creates a dense collection and upserts hybridN points, each
// carrying BOTH a dense vector and a sparse vector, so hybrid (dense+sparse RRF)
// and fusion queries exercise both lanes end-to-end.
func seedHybridCollection(b *testing.B, cl grpcsvc.VectorServiceClient, name string) [][]float32 {
	b.Helper()
	ctx := context.Background()
	if _, err := cl.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name:   name,
		Config: &pb.Config{Dim: hybridDim, Metric: "l2", M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1},
	}); err != nil {
		b.Fatal(err)
	}
	corpus := wireCorpus(hybridN, hybridDim, 42)
	for i, v := range corpus {
		si, sv := sparseVec(int64(i + 1))
		if _, err := cl.Upsert(ctx, &pb.UpsertRequest{
			Collection: name, Id: uint64(i + 1), Vector: v,
			SparseIndices: si, SparseValues: sv,
		}); err != nil {
			b.Fatal(err)
		}
	}
	return corpus
}

// BenchmarkWireHybridSearch measures dense+sparse RRF fusion over gRPC: both lanes
// run, then reciprocal-rank fusion merges the two ranked lists.
func BenchmarkWireHybridSearch(b *testing.B) {
	cl := newWireBench(b)
	corpus := seedHybridCollection(b, cl, "docs")
	ctx := context.Background()
	si, sv := sparseVec(99)
	b.ResetTimer()
	for i := 0; b.Loop(); i++ {
		if _, err := cl.HybridSearch(ctx, &pb.HybridRequest{
			Collection: "docs", K: 10, Method: "rrf",
			Dense: corpus[i%len(corpus)], SparseIndices: si, SparseValues: sv,
			DenseK: 50, SparseK: 50,
		}); err != nil {
			b.Fatal(err)
		}
	}
}

// NOTE: VectorQuery in FUSION mode is intentionally NOT benchmarked here. A
// multi-lane fusion query's op returns the UNFUSED per-lane results (result mode
// 0); the actual reciprocal-rank/score fusion into a flat top-k is done by the
// fan-out COORDINATOR layer, which sits above the op dispatcher and is absent from
// this bufconn + realDispatcher harness (the same coordinator boundary that blocks
// the write-consistency barrier in Wave 10). HybridSearch fuses inside its op so it
// runs end-to-end here; the QuerySpec request-marshal translation that fusion would
// exercise is already covered by the RERANK benchmark below (same handler path).

// BenchmarkWireVectorQueryRerank measures the Query API in RERANK mode: a dense
// root reranks a wider dense prefetch (single lane + a rerank pass).
func BenchmarkWireVectorQueryRerank(b *testing.B) {
	cl := newWireBench(b)
	corpus := seedHybridCollection(b, cl, "docs")
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; b.Loop(); i++ {
		q := corpus[i%len(corpus)]
		if _, err := cl.VectorQuery(ctx, &pb.VectorQueryRequest{
			Collection: "docs",
			Spec: &pb.QuerySpec{
				Mode: pb.QueryMode_QUERY_MODE_RERANK, K: 10,
				Root:     &pb.QueryLeaf{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{Dense: q, K: 10}}},
				Prefetch: []*pb.QueryLeaf{{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{Dense: q, K: 100}}}},
			},
		}); err != nil {
			b.Fatal(err)
		}
	}
}

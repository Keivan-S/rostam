// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/rostamlabs/rostam/grpcapi/grpcsvc"
	"github.com/rostamlabs/rostam/sdk/pb"
)

// mvTokens builds t token vectors of dim floats (a ColBERT-style token matrix).
func mvTokens(t, dim int, seed int64) []*pb.TokenVector {
	r := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic benchmark data
	out := make([]*pb.TokenVector, t)
	for i := range out {
		v := make([]float32, dim)
		for j := range v {
			v[j] = r.Float32()
		}
		out[i] = &pb.TokenVector{Values: v}
	}
	return out
}

const (
	mvDim       = 128
	mvDocs      = 500
	mvTokPerDoc = 16
)

// seedMVCollection creates an MV (late-interaction) collection and adds mvDocs
// documents, each with mvTokPerDoc tokens, over the wire.
func seedMVCollection(b *testing.B, cl grpcsvc.VectorServiceClient, name string) {
	b.Helper()
	ctx := context.Background()
	if _, err := cl.MVCreateCollection(ctx, &pb.MVCreateRequest{
		Name:   name,
		Config: &pb.MVConfig{Dim: mvDim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1},
	}); err != nil {
		b.Fatal(err)
	}
	for i := 0; i < mvDocs; i++ {
		if _, err := cl.MVAdd(ctx, &pb.MVAddRequest{
			Name: name, Id: uint64(i + 1), Tokens: mvTokens(mvTokPerDoc, mvDim, int64(i+1)),
		}); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkWireMVAdd measures MVAdd latency over gRPC as the per-doc token count
// grows — the token-matrix serialization throughput (T tokens × dim floats
// marshaled per request, then T inner-graph inserts). Index insert cost rides
// along; the sweep isolates how serialization scales with the matrix size.
func BenchmarkWireMVAdd(b *testing.B) {
	for _, toks := range []int{8, 32, 128} {
		b.Run(fmt.Sprintf("tokens=%d", toks), func(b *testing.B) {
			cl := newWireBench(b)
			ctx := context.Background()
			if _, err := cl.MVCreateCollection(ctx, &pb.MVCreateRequest{
				Name:   "mv",
				Config: &pb.MVConfig{Dim: mvDim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1},
			}); err != nil {
				b.Fatal(err)
			}
			tokens := mvTokens(toks, mvDim, 7)
			id := uint64(1)
			b.ResetTimer()
			for b.Loop() {
				if _, err := cl.MVAdd(ctx, &pb.MVAddRequest{Name: "mv", Id: id, Tokens: tokens}); err != nil {
					b.Fatal(err)
				}
				id++
			}
		})
	}
}

// BenchmarkWireMVSearch measures MaxSim search over gRPC: the multi-token query is
// marshaled, scored per token against the inner HNSW, and the top-k merged.
func BenchmarkWireMVSearch(b *testing.B) {
	cl := newWireBench(b)
	seedMVCollection(b, cl, "mv")
	ctx := context.Background()
	query := mvTokens(8, mvDim, 99)
	b.ResetTimer()
	for b.Loop() {
		if _, err := cl.MVSearch(ctx, &pb.MVSearchRequest{Name: "mv", Query: query, K: 10, CandidatesPerToken: 10}); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkWireMVGetBatch measures MV point-lookup over gRPC: each hit returns a
// full token matrix (mvTokPerDoc × dim floats), so this is the heaviest MV read
// marshal path. Batch size is swept.
func BenchmarkWireMVGetBatch(b *testing.B) {
	cl := newWireBench(b)
	seedMVCollection(b, cl, "mv")
	ctx := context.Background()
	for _, batch := range []int{1, 10, 50} {
		ids := make([]uint64, batch)
		for i := range ids {
			ids[i] = uint64(i + 1)
		}
		b.Run(fmt.Sprintf("batch=%d", batch), func(b *testing.B) {
			for b.Loop() {
				if _, err := cl.MVGetBatch(ctx, &pb.MVGetBatchRequest{Collection: "mv", Ids: ids, WithVector: true}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

const namedBenchConfig = `{"title":{"dim":128},"image":{"dim":128,"metric":2}}`

// seedNamedCollection creates a named (per-point multi-space) collection and
// upserts n points with both named vectors over the wire.
func seedNamedCollection(b *testing.B, cl grpcsvc.VectorServiceClient, name string, n int) {
	b.Helper()
	ctx := context.Background()
	if _, err := cl.NamedCreate(ctx, &pb.NamedCreateRequest{Name: name, ConfigJson: namedBenchConfig}); err != nil {
		b.Fatal(err)
	}
	for i := 0; i < n; i++ {
		v := wireCorpus(2, mvDim, int64(i+1))
		if _, err := cl.NamedUpsert(ctx, &pb.NamedUpsertRequest{
			Name: name, Id: uint64(i + 1),
			Vectors: map[string]*pb.NamedVectorList{"title": {Values: v[0]}, "image": {Values: v[1]}},
		}); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkWireNamedSearch measures a single-space named search over gRPC.
func BenchmarkWireNamedSearch(b *testing.B) {
	cl := newWireBench(b)
	seedNamedCollection(b, cl, "docs", 5_000)
	ctx := context.Background()
	q := wireCorpus(1, mvDim, 99)[0]
	b.ResetTimer()
	for b.Loop() {
		if _, err := cl.NamedSearch(ctx, &pb.NamedSearchRequest{Name: "docs", VectorName: "title", Query: q, K: 10}); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkWireNamedGetBatch measures named point-lookup over gRPC, returning all
// named vectors per hit. Batch size is swept.
func BenchmarkWireNamedGetBatch(b *testing.B) {
	cl := newWireBench(b)
	seedNamedCollection(b, cl, "docs", 5_000)
	ctx := context.Background()
	for _, batch := range []int{1, 10, 100} {
		ids := make([]uint64, batch)
		for i := range ids {
			ids[i] = uint64(i + 1)
		}
		b.Run(fmt.Sprintf("batch=%d", batch), func(b *testing.B) {
			for b.Loop() {
				if _, err := cl.NamedGetBatch(ctx, &pb.NamedGetBatchRequest{Collection: "docs", Ids: ids, WithVector: true}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

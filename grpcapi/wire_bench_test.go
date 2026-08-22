// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"context"
	"fmt"
	"math/rand"
	"net"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/grpcapi/grpcsvc"
	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// wireCorpus generates n deterministic dim-dimensional vectors.
func wireCorpus(n, dim int, seed int64) [][]float32 {
	r := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic benchmark data
	out := make([][]float32, n)
	for i := range out {
		v := make([]float32, dim)
		for j := range v {
			v[j] = r.Float32()
		}
		out[i] = v
	}
	return out
}

// newWireBench spins up a real-engine gRPC server over an in-memory bufconn pipe
// and returns a connected client. This measures the full wire path —
// proto marshal → transport → server handler (proto→op-args→engine→result→proto)
// → marshal → client unmarshal — i.e. everything the embedded-engine benchmarks
// in Part A do NOT cover. The backing store is real (RegisterBuiltins + a vector
// CollectionStore), so handler translation runs against live data.
func newWireBench(b *testing.B) grpcsvc.VectorServiceClient {
	b.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		b.Fatal(err)
	}
	c, _ := cache.New(cache.DefaultConfig())
	vstore, err := vector.OpenCollectionStore(b.TempDir())
	if err != nil {
		b.Fatal(err)
	}
	disp := &realDispatcher{reg: reg, tx: ops.NewTxContextWithVectors(c, vstore)}

	lis := bufconn.Listen(1 << 20)
	gs := grpc.NewServer()
	grpcsvc.RegisterVectorServiceServer(gs, NewServer(disp, nil))
	go func() { _ = gs.Serve(lis) }()

	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return lis.DialContext(ctx)
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() {
		_ = conn.Close()
		gs.Stop()
		_ = vstore.Close()
		c.Close()
	})
	return grpcsvc.NewVectorServiceClient(conn)
}

// seedCollection creates a dense HNSW collection and upserts n vectors over the
// wire, returning the corpus. Done outside the timed region by callers.
func seedCollection(b *testing.B, cl grpcsvc.VectorServiceClient, name string, n, dim int) [][]float32 {
	b.Helper()
	ctx := context.Background()
	if _, err := cl.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name:   name,
		Config: &pb.Config{Dim: int32(dim), Metric: "l2", M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1},
	}); err != nil {
		b.Fatal(err)
	}
	corpus := wireCorpus(n, dim, 42)
	for i, v := range corpus {
		if _, err := cl.Upsert(ctx, &pb.UpsertRequest{Collection: name, Id: uint64(i + 1), Vector: v}); err != nil {
			b.Fatal(err)
		}
	}
	return corpus
}

const (
	wireN   = 5_000
	wireDim = 128
)

// BenchmarkWireSearch measures end-to-end Search latency over gRPC (query marshal
// + handler + top-k result marshal). Compare against the embedded HNSW search to
// isolate the wire + translation overhead.
func BenchmarkWireSearch(b *testing.B) {
	cl := newWireBench(b)
	corpus := seedCollection(b, cl, "docs", wireN, wireDim)
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; b.Loop(); i++ {
		_, err := cl.Search(ctx, &pb.SearchRequest{Collection: "docs", Query: corpus[i%len(corpus)], K: 10})
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkWireSearchDocs measures Search that returns full documents (metadata +
// vector marshal per hit), the heavier-payload read path.
func BenchmarkWireSearchDocs(b *testing.B) {
	cl := newWireBench(b)
	corpus := seedCollection(b, cl, "docs", wireN, wireDim)
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; b.Loop(); i++ {
		_, err := cl.SearchDocs(ctx, &pb.SearchRequest{Collection: "docs", Query: corpus[i%len(corpus)], K: 10})
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkWireUpsert measures single-vector write latency over gRPC (vector
// marshal + vector_insert dispatch).
func BenchmarkWireUpsert(b *testing.B) {
	cl := newWireBench(b)
	_ = seedCollection(b, cl, "docs", wireN, wireDim)
	ctx := context.Background()
	v := wireCorpus(1, wireDim, 7)[0]
	id := uint64(wireN + 1)
	b.ResetTimer()
	for b.Loop() {
		if _, err := cl.Upsert(ctx, &pb.UpsertRequest{Collection: "docs", Id: id, Vector: v}); err != nil {
			b.Fatal(err)
		}
		id++
	}
}

// BenchmarkWireGetBatch measures point-lookup throughput over gRPC for batches of
// IDs (scatter + per-id result marshal). Batch size is swept.
func BenchmarkWireGetBatch(b *testing.B) {
	cl := newWireBench(b)
	_ = seedCollection(b, cl, "docs", wireN, wireDim)
	ctx := context.Background()
	for _, batch := range []int{1, 10, 100} {
		ids := make([]uint64, batch)
		for i := range ids {
			ids[i] = uint64(i + 1)
		}
		b.Run(fmt.Sprintf("batch=%d", batch), func(b *testing.B) {
			for b.Loop() {
				if _, err := cl.GetBatch(ctx, &pb.GetBatchRequest{Collection: "docs", Ids: ids, WithVector: true}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

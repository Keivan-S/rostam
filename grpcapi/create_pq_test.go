// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"context"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// newRealServerWithDisp builds a real-engine Server (like newRealServer) but also
// returns the backing dispatcher so a test can drive ops that have no dedicated
// RPC (vector_bulk_stage / vector_bulk_build) directly through the engine.
func newRealServerWithDisp(t *testing.T) (*Server, *realDispatcher) {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	c, _ := cache.New(cache.DefaultConfig())
	vstore, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = vstore.Close(); c.Close() })
	disp := &realDispatcher{reg: reg, tx: ops.NewTxContextWithVectors(c, vstore)}
	return NewServer(disp, nil), disp
}

// TestToConfigQuantPQ proves parseQuant accepts "pq" and the QuantPQM field
// rides through toConfig; an unknown quant is still rejected.
func TestToConfigQuantPQ(t *testing.T) {
	cfg, err := toConfig(&pb.Config{Dim: 8, Metric: "cosine", M: 16, Quant: "pq", QuantPqM: 8})
	if err != nil {
		t.Fatalf("toConfig pq: %v", err)
	}
	if cfg.Quant != vector.QuantPQ {
		t.Fatalf("Quant = %v, want QuantPQ", cfg.Quant)
	}
	if cfg.QuantPQM != 8 {
		t.Fatalf("QuantPQM = %d, want 8", cfg.QuantPQM)
	}
	if _, err := toConfig(&pb.Config{Dim: 8, Metric: "cosine", Quant: "bogus"}); err == nil {
		t.Fatalf("toConfig with quant=bogus: expected error, got nil")
	}
}

// TestGRPCCreatePQHNSW drives the full PQ-HNSW create wire over gRPC: quant="pq"
// + quant_pq_m creates a dense HNSW index that trains PQ at bulk-build and serves
// ADC-navigated search. The working index proves the engine received QuantPQ.
func TestGRPCCreatePQHNSW(t *testing.T) {
	s, disp := newRealServerWithDisp(t)
	ctx := context.Background()

	if _, err := s.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name:   "pq",
		Config: &pb.Config{Dim: 8, Metric: "l2", M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Quant: "pq", QuantPqM: 8},
	}); err != nil {
		t.Fatalf("CreateCollection pq: %v", err)
	}

	ids := make([]uint64, 40)
	vecs := make([][]float32, 40)
	for i := 0; i < 40; i++ {
		ids[i] = uint64(i + 1)
		v := make([]float32, 8)
		v[0] = float32(i + 1)
		vecs[i] = v
	}
	stageArgs, err := ops.EncodeBulkStageArgs("pq", ids, vecs)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := disp.Call("vector_bulk_stage", stageArgs); err != nil {
		t.Fatalf("bulk stage: %v", err)
	}
	if _, err := disp.Call("vector_bulk_build", ops.EncodeBulkBuildArgs("pq", 4)); err != nil {
		t.Fatalf("bulk build: %v", err)
	}

	res, err := s.Search(ctx, &pb.SearchRequest{Collection: "pq", Query: []float32{1, 0, 0, 0, 0, 0, 0, 0}, K: 5})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res.GetResults()) != 5 {
		t.Fatalf("PQ search = %d results, want 5", len(res.GetResults()))
	}
	if res.GetResults()[0].GetId() != 1 {
		t.Fatalf("nearest = id %d, want 1", res.GetResults()[0].GetId())
	}
}

// TestGRPCCreatePQHNSWFailLoud covers the two engine validation gates on the
// create wire: quant="pq" on an IVF index, and dim not divisible by quant_pq_m.
// Both must surface as InvalidArgument (not Internal).
func TestGRPCCreatePQHNSWFailLoud(t *testing.T) {
	s := newRealServer(t)
	ctx := context.Background()

	_, err := s.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name:   "pqivf",
		Config: &pb.Config{Dim: 8, Metric: "l2", M: 8, Quant: "pq", QuantPqM: 8, IndexType: "ivf"},
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("pq on IVF: code = %v, want InvalidArgument (%v)", status.Code(err), err)
	}

	_, err = s.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name:   "pqbad",
		Config: &pb.Config{Dim: 8, Metric: "l2", M: 8, Quant: "pq", QuantPqM: 5},
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("indivisible m: code = %v, want InvalidArgument (%v)", status.Code(err), err)
	}
}

// TestToConfigOPQ proves the opq proto field rides through toConfig onto the engine
// Config (default false).
func TestToConfigOPQ(t *testing.T) {
	cfg, err := toConfig(&pb.Config{Dim: 8, Metric: "cosine", M: 16, Quant: "pq", QuantPqM: 8, Opq: true})
	if err != nil {
		t.Fatalf("toConfig opq: %v", err)
	}
	if !cfg.OPQ {
		t.Fatal("cfg.OPQ = false, want true (opq proto field not threaded)")
	}
	def, err := toConfig(&pb.Config{Dim: 8, Metric: "cosine", M: 16, Quant: "pq", QuantPqM: 8})
	if err != nil {
		t.Fatal(err)
	}
	if def.OPQ {
		t.Fatal("cfg.OPQ should default false when opq unset")
	}
}

// TestToConfigIVFTrainThreshold proves the dense ivf_train_threshold proto field
// rides through toConfig onto the engine Config (Gap 1: closing the dense
// create-codec asymmetry). Default 0 when unset.
func TestToConfigIVFTrainThreshold(t *testing.T) {
	cfg, err := toConfig(&pb.Config{Dim: 8, Metric: "l2", M: 16, IndexType: "ivf", IvfNlist: 4, IvfNprobe: 2, IvfTrainThreshold: 1500})
	if err != nil {
		t.Fatalf("toConfig ivf_train_threshold: %v", err)
	}
	if cfg.IVFTrainThreshold != 1500 {
		t.Fatalf("cfg.IVFTrainThreshold = %d, want 1500 (proto field not threaded)", cfg.IVFTrainThreshold)
	}
	def, err := toConfig(&pb.Config{Dim: 8, Metric: "l2", M: 16, IndexType: "ivf"})
	if err != nil {
		t.Fatal(err)
	}
	if def.IVFTrainThreshold != 0 {
		t.Fatalf("cfg.IVFTrainThreshold = %d, want 0 default when unset", def.IVFTrainThreshold)
	}
}

// TestGRPCCreatePQHNSWOPQ drives a full create with opq=true over gRPC: the
// PQ-HNSW index trains rotated codebooks at bulk-build and serves search.
func TestGRPCCreatePQHNSWOPQ(t *testing.T) {
	s, disp := newRealServerWithDisp(t)
	ctx := context.Background()

	if _, err := s.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name:   "pqopq",
		Config: &pb.Config{Dim: 8, Metric: "l2", M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Quant: "pq", QuantPqM: 8, Opq: true},
	}); err != nil {
		t.Fatalf("CreateCollection pq+opq: %v", err)
	}

	ids := make([]uint64, 40)
	vecs := make([][]float32, 40)
	for i := 0; i < 40; i++ {
		ids[i] = uint64(i + 1)
		v := make([]float32, 8)
		v[0] = float32(i + 1)
		vecs[i] = v
	}
	stageArgs, err := ops.EncodeBulkStageArgs("pqopq", ids, vecs)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := disp.Call("vector_bulk_stage", stageArgs); err != nil {
		t.Fatalf("bulk stage: %v", err)
	}
	if _, err := disp.Call("vector_bulk_build", ops.EncodeBulkBuildArgs("pqopq", 4)); err != nil {
		t.Fatalf("bulk build: %v", err)
	}
	res, err := s.Search(ctx, &pb.SearchRequest{Collection: "pqopq", Query: []float32{1, 0, 0, 0, 0, 0, 0, 0}, K: 5})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res.GetResults()) != 5 {
		t.Fatalf("PQ+OPQ search = %d results, want 5", len(res.GetResults()))
	}
}

// TestGRPCCreateOPQFailLoud: opq=true WITHOUT a PQ mode (no quant=="pq", no ivf_pq)
// must surface as InvalidArgument (the cfg.Validate OPQ gate), not Internal.
func TestGRPCCreateOPQFailLoud(t *testing.T) {
	s := newRealServer(t)
	ctx := context.Background()
	_, err := s.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name:   "opqnopq",
		Config: &pb.Config{Dim: 8, Metric: "l2", M: 8, Opq: true},
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("opq without PQ: code = %v, want InvalidArgument (%v)", status.Code(err), err)
	}
}

// TestToConfigPQDropVecs proves the pq_drop_vecs proto field rides through
// toConfig onto the engine Config (default false).
func TestToConfigPQDropVecs(t *testing.T) {
	cfg, err := toConfig(&pb.Config{Dim: 8, Metric: "cosine", M: 16, Quant: "pq", QuantPqM: 8, PqDropVecs: true})
	if err != nil {
		t.Fatalf("toConfig pq_drop_vecs: %v", err)
	}
	if !cfg.PQDropVecs {
		t.Fatal("cfg.PQDropVecs = false, want true (pq_drop_vecs proto field not threaded)")
	}
	def, err := toConfig(&pb.Config{Dim: 8, Metric: "cosine", M: 16, Quant: "pq", QuantPqM: 8})
	if err != nil {
		t.Fatal(err)
	}
	if def.PQDropVecs {
		t.Fatal("cfg.PQDropVecs should default false when pq_drop_vecs unset")
	}
}

// TestGRPCCreatePQDropVecsFailLoud: pq_drop_vecs=true WITHOUT quant=="pq" must
// surface as InvalidArgument (the cfg.Validate PQDropVecs gate), not Internal.
func TestGRPCCreatePQDropVecsFailLoud(t *testing.T) {
	s := newRealServer(t)
	ctx := context.Background()
	_, err := s.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name:   "dropnopq",
		Config: &pb.Config{Dim: 8, Metric: "l2", M: 8, PqDropVecs: true},
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("pq_drop_vecs without PQ: code = %v, want InvalidArgument (%v)", status.Code(err), err)
	}
}

// TestToConfigQuantSQPRQ proves parseQuant accepts "sq"/"prq" and the sq_bits /
// prq_layers proto fields ride through toConfig onto the engine Config.
func TestToConfigQuantSQPRQ(t *testing.T) {
	sq, err := toConfig(&pb.Config{Dim: 8, Metric: "l2", M: 16, Quant: "sq", SqBits: 6})
	if err != nil {
		t.Fatalf("toConfig sq: %v", err)
	}
	if sq.Quant != vector.QuantSQ || sq.SQBits != 6 {
		t.Fatalf("Quant/SQBits = %v/%d, want QuantSQ/6", sq.Quant, sq.SQBits)
	}
	prq, err := toConfig(&pb.Config{Dim: 8, Metric: "l2", M: 16, Quant: "prq", QuantPqM: 8, PrqLayers: 3})
	if err != nil {
		t.Fatalf("toConfig prq: %v", err)
	}
	if prq.Quant != vector.QuantPRQ || prq.PRQLayers != 3 {
		t.Fatalf("Quant/PRQLayers = %v/%d, want QuantPRQ/3", prq.Quant, prq.PRQLayers)
	}
	// Defaults: sq_bits / prq_layers absent => 0 (engine resolves to 8 / 2).
	def, err := toConfig(&pb.Config{Dim: 8, Metric: "l2", M: 16, Quant: "sq"})
	if err != nil {
		t.Fatal(err)
	}
	if def.SQBits != 0 || def.PRQLayers != 0 {
		t.Fatalf("defaults SQBits/PRQLayers = %d/%d, want 0/0", def.SQBits, def.PRQLayers)
	}
}

// TestGRPCCreateSQHNSW drives the full trained-SQ create wire over gRPC:
// quant="sq" + sq_bits creates a dense HNSW index that trains the scalar
// quantizer at bulk-build and serves rescored search. The working index proves
// the engine received QuantSQ with the bit-depth.
func TestGRPCCreateSQHNSW(t *testing.T) {
	s, disp := newRealServerWithDisp(t)
	ctx := context.Background()

	if _, err := s.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name:   "sq",
		Config: &pb.Config{Dim: 8, Metric: "l2", M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Quant: "sq", SqBits: 6},
	}); err != nil {
		t.Fatalf("CreateCollection sq: %v", err)
	}

	ids := make([]uint64, 40)
	vecs := make([][]float32, 40)
	for i := 0; i < 40; i++ {
		ids[i] = uint64(i + 1)
		v := make([]float32, 8)
		v[0] = float32(i + 1)
		vecs[i] = v
	}
	stageArgs, err := ops.EncodeBulkStageArgs("sq", ids, vecs)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := disp.Call("vector_bulk_stage", stageArgs); err != nil {
		t.Fatalf("bulk stage: %v", err)
	}
	if _, err := disp.Call("vector_bulk_build", ops.EncodeBulkBuildArgs("sq", 4)); err != nil {
		t.Fatalf("bulk build: %v", err)
	}
	res, err := s.Search(ctx, &pb.SearchRequest{Collection: "sq", Query: []float32{1, 0, 0, 0, 0, 0, 0, 0}, K: 5})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res.GetResults()) != 5 {
		t.Fatalf("SQ search = %d results, want 5", len(res.GetResults()))
	}
	if res.GetResults()[0].GetId() != 1 {
		t.Fatalf("nearest = id %d, want 1", res.GetResults()[0].GetId())
	}
}

// TestGRPCCreatePRQHNSW drives the full product-residual-quant create wire over
// gRPC: quant="prq" + prq_layers (+ quant_pq_m) creates a dense HNSW index that
// trains the PRQ layers at bulk-build and serves ADC-navigated search. The
// working index proves the engine received QuantPRQ with the layer count.
func TestGRPCCreatePRQHNSW(t *testing.T) {
	s, disp := newRealServerWithDisp(t)
	ctx := context.Background()

	if _, err := s.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name:   "prq",
		Config: &pb.Config{Dim: 8, Metric: "l2", M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Quant: "prq", QuantPqM: 8, PrqLayers: 2},
	}); err != nil {
		t.Fatalf("CreateCollection prq: %v", err)
	}

	ids := make([]uint64, 40)
	vecs := make([][]float32, 40)
	for i := 0; i < 40; i++ {
		ids[i] = uint64(i + 1)
		v := make([]float32, 8)
		v[0] = float32(i + 1)
		vecs[i] = v
	}
	stageArgs, err := ops.EncodeBulkStageArgs("prq", ids, vecs)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := disp.Call("vector_bulk_stage", stageArgs); err != nil {
		t.Fatalf("bulk stage: %v", err)
	}
	if _, err := disp.Call("vector_bulk_build", ops.EncodeBulkBuildArgs("prq", 4)); err != nil {
		t.Fatalf("bulk build: %v", err)
	}
	res, err := s.Search(ctx, &pb.SearchRequest{Collection: "prq", Query: []float32{1, 0, 0, 0, 0, 0, 0, 0}, K: 5})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res.GetResults()) != 5 {
		t.Fatalf("PRQ search = %d results, want 5", len(res.GetResults()))
	}
	if res.GetResults()[0].GetId() != 1 {
		t.Fatalf("nearest = id %d, want 1", res.GetResults()[0].GetId())
	}
}

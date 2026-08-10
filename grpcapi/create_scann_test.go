// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/ops"
)

// TestToConfigScann proves the anisotropic_eta / soar / soar_lambda / pq_nbits
// proto fields ride through toConfig onto the engine Config; all default
// zero/false when unset (byte-compatible with the pre-ScaNN wire).
func TestToConfigScann(t *testing.T) {
	cfg, err := toConfig(&pb.Config{
		Dim: 8, Metric: "dot", M: 16, IndexType: "ivf", IvfNlist: 16, IvfNprobe: 4,
		AnisotropicEta: 4, Soar: true, SoarLambda: 2, PqNbits: 4,
	})
	if err != nil {
		t.Fatalf("toConfig scann: %v", err)
	}
	if cfg.AnisotropicEta != 4 || !cfg.SOAR || cfg.SOARLambda != 2 || cfg.PQNBits != 4 {
		t.Fatalf("ScaNN params not threaded: eta=%v soar=%v lambda=%v nbits=%d",
			cfg.AnisotropicEta, cfg.SOAR, cfg.SOARLambda, cfg.PQNBits)
	}

	def, err := toConfig(&pb.Config{Dim: 8, Metric: "dot", M: 16})
	if err != nil {
		t.Fatalf("toConfig scann defaults: %v", err)
	}
	if def.AnisotropicEta != 0 || def.SOAR || def.SOARLambda != 0 || def.PQNBits != 0 {
		t.Fatalf("ScaNN params should default zero when unset: eta=%v soar=%v lambda=%v nbits=%d",
			def.AnisotropicEta, def.SOAR, def.SOARLambda, def.PQNBits)
	}
}

// pqHNSWBulkSearch creates a PQ-HNSW collection over gRPC with the given Config,
// bulk-builds 40 vectors, and asserts a search round-trip returns the nearest hit.
// The working index proves Config (incl. the ScaNN knob) reached the engine.
func pqHNSWBulkSearch(t *testing.T, name string, cfg *pb.Config) {
	t.Helper()
	s, disp := newRealServerWithDisp(t)
	ctx := context.Background()

	if _, err := s.CreateCollection(ctx, &pb.CreateCollectionRequest{Name: name, Config: cfg}); err != nil {
		t.Fatalf("CreateCollection %s: %v", name, err)
	}

	ids := make([]uint64, 40)
	vecs := make([][]float32, 40)
	for i := 0; i < 40; i++ {
		ids[i] = uint64(i + 1)
		v := make([]float32, 8)
		v[0] = float32(i + 1)
		vecs[i] = v
	}
	stageArgs, err := ops.EncodeBulkStageArgs(name, ids, vecs)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := disp.Call("vector_bulk_stage", stageArgs); err != nil {
		t.Fatalf("bulk stage: %v", err)
	}
	if _, err := disp.Call("vector_bulk_build", ops.EncodeBulkBuildArgs(name, 4)); err != nil {
		t.Fatalf("bulk build: %v", err)
	}

	res, err := s.Search(ctx, &pb.SearchRequest{Collection: name, Query: []float32{1, 0, 0, 0, 0, 0, 0, 0}, K: 5})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res.GetResults()) != 5 {
		t.Fatalf("%s search = %d results, want 5", name, len(res.GetResults()))
	}
	if res.GetResults()[0].GetId() != 1 {
		t.Fatalf("%s nearest = id %d, want 1", name, res.GetResults()[0].GetId())
	}
}

// TestGRPCCreateAnisotropicPQ drives a full anisotropic PQ-HNSW create over gRPC
// (quant="pq" + anisotropic_eta): the engine trains anisotropic codebooks at
// bulk-build and serves ADC-navigated search. The working index proves the
// AnisotropicEta knob reached the engine via the create wire.
func TestGRPCCreateAnisotropicPQ(t *testing.T) {
	// L2 so id 1 (v0=1) is the deterministic nearest to query [1,0,...]; for a
	// DotProduct metric the highest-magnitude vector would be "nearest" instead.
	// AnisotropicEta rides through regardless of metric (it only shapes training).
	pqHNSWBulkSearch(t, "aniso", &pb.Config{
		Dim: 8, Metric: "l2", M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1,
		Quant: "pq", QuantPqM: 8, AnisotropicEta: 4,
	})
}

// TestGRPCCreate4BitPQ drives a full 4-bit PQ-HNSW create over gRPC (quant="pq" +
// pq_nbits=4): the engine trains LUT16 codebooks (ceil(m/2) code bytes) and serves
// search. The working index proves PQNBits=4 reached the engine via the wire.
func TestGRPCCreate4BitPQ(t *testing.T) {
	pqHNSWBulkSearch(t, "pq4", &pb.Config{
		Dim: 8, Metric: "l2", M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1,
		Quant: "pq", QuantPqM: 8, PqNbits: 4,
	})
}

// TestGRPCCreateSOARIVF drives a full SOAR-IVF create over gRPC (index_type="ivf"
// + soar): the engine builds the multi-assignment IVF (low ivf_train_threshold so
// it trains on the incremental upserts) and serves search. The working index +
// nearest hit proves SOAR/SOARLambda reached the engine via the create wire.
func TestGRPCCreateSOARIVF(t *testing.T) {
	s := newRealServer(t)
	ctx := context.Background()

	if _, err := s.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name: "soar",
		Config: &pb.Config{
			Dim: 8, Metric: "l2", M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1,
			IndexType: "ivf", IvfNlist: 8, IvfNprobe: 8, IvfTrainThreshold: 32,
			Soar: true, SoarLambda: 2,
		},
	}); err != nil {
		t.Fatalf("CreateCollection soar: %v", err)
	}

	for i := 0; i < 64; i++ {
		v := make([]float32, 8)
		v[0] = float32(i + 1)
		v[1] = float32((i % 7) + 1)
		if _, err := s.Upsert(ctx, &pb.UpsertRequest{Collection: "soar", Id: uint64(i + 1), Vector: v}); err != nil {
			t.Fatalf("upsert %d: %v", i, err)
		}
	}

	res, err := s.Search(ctx, &pb.SearchRequest{Collection: "soar", Query: []float32{1, 1, 0, 0, 0, 0, 0, 0}, K: 5})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res.GetResults()) == 0 {
		t.Fatalf("SOAR-IVF search returned no results")
	}
	// The exact nearest neighbor (id 1: v0=1,v1=1) must be reachable through the
	// multi-assignment lists at the probed nprobe.
	found := false
	for _, r := range res.GetResults() {
		if r.GetId() == 1 {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("SOAR-IVF search missed the nearest id 1: %+v", res.GetResults())
	}
}

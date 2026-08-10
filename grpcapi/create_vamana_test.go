// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/vector"
)

// TestToConfigIndexVamana proves parseIndexType accepts "vamana" and the
// VamanaR/VamanaL/VamanaAlpha proto fields ride through toConfig onto the engine
// Config; defaults are zero when unset (the engine fills its own R/L/alpha
// defaults).
func TestToConfigIndexVamana(t *testing.T) {
	cfg, err := toConfig(&pb.Config{
		Dim: 8, Metric: "l2", M: 16,
		IndexType: "vamana", VamanaR: 96, VamanaL: 150, VamanaAlpha: 1.4,
	})
	if err != nil {
		t.Fatalf("toConfig vamana: %v", err)
	}
	if cfg.IndexType != vector.IndexVamana {
		t.Fatalf("IndexType = %v, want IndexVamana", cfg.IndexType)
	}
	if cfg.VamanaR != 96 || cfg.VamanaL != 150 || cfg.VamanaAlpha != 1.4 {
		t.Fatalf("Vamana params not threaded: R=%d L=%d alpha=%v", cfg.VamanaR, cfg.VamanaL, cfg.VamanaAlpha)
	}

	def, err := toConfig(&pb.Config{Dim: 8, Metric: "l2", M: 16, IndexType: "vamana"})
	if err != nil {
		t.Fatalf("toConfig vamana defaults: %v", err)
	}
	if def.IndexType != vector.IndexVamana {
		t.Fatalf("IndexType = %v, want IndexVamana", def.IndexType)
	}
	if def.VamanaR != 0 || def.VamanaL != 0 || def.VamanaAlpha != 0 {
		t.Fatalf("Vamana params should default zero when unset: R=%d L=%d alpha=%v", def.VamanaR, def.VamanaL, def.VamanaAlpha)
	}
}

// TestGRPCCreateVamana drives the full Vamana create wire over gRPC with a
// non-default R: the engine receives IndexVamana + R/L/alpha, builds a single-
// layer graph via incremental inserts, and serves a search round-trip. The
// working index + nearest hit proves Config reached the engine intact.
func TestGRPCCreateVamana(t *testing.T) {
	s := newRealServer(t)
	ctx := context.Background()

	if _, err := s.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name: "vam",
		Config: &pb.Config{
			Dim: 8, Metric: "l2", M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1,
			IndexType: "vamana", VamanaR: 48, VamanaL: 80, VamanaAlpha: 1.3,
		},
	}); err != nil {
		t.Fatalf("CreateCollection vamana: %v", err)
	}

	for i := 0; i < 40; i++ {
		v := make([]float32, 8)
		v[0] = float32(i + 1)
		if _, err := s.Upsert(ctx, &pb.UpsertRequest{
			Collection: "vam", Id: uint64(i + 1), Vector: v,
		}); err != nil {
			t.Fatalf("upsert %d: %v", i, err)
		}
	}

	res, err := s.Search(ctx, &pb.SearchRequest{Collection: "vam", Query: []float32{1, 0, 0, 0, 0, 0, 0, 0}, K: 5})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res.GetResults()) != 5 {
		t.Fatalf("Vamana search = %d results, want 5", len(res.GetResults()))
	}
	if res.GetResults()[0].GetId() != 1 {
		t.Fatalf("nearest = id %d, want 1", res.GetResults()[0].GetId())
	}
}

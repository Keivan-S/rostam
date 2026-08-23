// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/sdk/pb"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// stubDispatcher records the last op dispatched so tests can assert that a
// rejected request never reaches the backing store.
type stubDispatcher struct {
	called    bool
	callCount int
	lastOp    string
	lastArg   []byte
}

func (s *stubDispatcher) Call(name string, args []byte) ([]byte, error) {
	s.called = true
	s.callCount++
	s.lastOp = name
	s.lastArg = args
	return nil, nil
}

func (s *stubDispatcher) LeaderAddr() string { return "" }

// TestGetRejectsBadConsistency proves the get / named-get / mv-get / named-get-config
// handlers fail loud (InvalidArgument) for an out-of-range read_consistency BEFORE
// dispatch — a bad rc never reaches the store.
func TestGetRejectsBadConsistency(t *testing.T) {
	const badRC = 4 // enum tops out at 3 (bounded-staleness)
	cases := []struct {
		name string
		call func(*Server) error
	}{
		{"Get", func(s *Server) error {
			_, err := s.Get(context.Background(), &pb.GetRequest{Collection: "c", Id: 1, ReadConsistency: badRC})
			return err
		}},
		{"NamedGet", func(s *Server) error {
			_, err := s.NamedGet(context.Background(), &pb.NamedGetRequest{Collection: "c", Id: 1, ReadConsistency: badRC})
			return err
		}},
		{"MVGet", func(s *Server) error {
			_, err := s.MVGet(context.Background(), &pb.MVGetRequest{Collection: "c", Id: 1, ReadConsistency: badRC})
			return err
		}},
		{"NamedGetConfig", func(s *Server) error {
			_, err := s.NamedGetConfig(context.Background(), &pb.NamedGetConfigRequest{Name: "c", ReadConsistency: badRC})
			return err
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			disp := &stubDispatcher{}
			s := NewServer(disp, nil)
			err := tc.call(s)
			if status.Code(err) != codes.InvalidArgument {
				t.Fatalf("code = %v, want InvalidArgument", status.Code(err))
			}
			if disp.called {
				t.Fatalf("dispatcher must not be called for a bad read_consistency")
			}
		})
	}
}

// TestGetThreadsConsistencyIntoArgs proves a valid Linearizable rc rides into the
// dispatched get args (so the fan-out coordinator / shard can arm the barrier).
func TestGetThreadsConsistencyIntoArgs(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	_, _ = s.Get(context.Background(), &pb.GetRequest{Collection: "c", Id: 9, ReadConsistency: 2})
	if disp.lastOp != "vector_get" {
		t.Fatalf("op = %q, want vector_get", disp.lastOp)
	}
	_, _, _, rc, _, _, err := ops.DecodeVectorGetArgsOpts(disp.lastArg)
	if err != nil {
		t.Fatalf("decode dispatched args: %v", err)
	}
	if rc != ops.ConsistencyLinearizable {
		t.Fatalf("dispatched rc = %d, want Linearizable", rc)
	}
}

func TestToConfigPartitions(t *testing.T) {
	cfg, err := toConfig(&pb.Config{Dim: 8, Metric: "cosine", Partitions: 4})
	if err != nil {
		t.Fatalf("toConfig: %v", err)
	}
	if cfg.Partitions != 4 {
		t.Fatalf("Partitions = %d, want 4", cfg.Partitions)
	}
	if _, err := toConfig(&pb.Config{Dim: 8, Metric: "cosine", Partitions: -1}); err == nil {
		t.Fatalf("toConfig with Partitions=-1: expected error, got nil")
	}
}

func TestCreateCollectionRejectsNegativePartitions(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	_, err := s.CreateCollection(context.Background(), &pb.CreateCollectionRequest{
		Name:   "ok",
		Config: &pb.Config{Dim: 8, Metric: "cosine", Partitions: -1},
	})
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("code = %v, want InvalidArgument", status.Code(err))
	}
	if disp.called {
		t.Fatalf("dispatcher must not be called for negative partitions")
	}
}

func TestCreateCollectionRejectsReservedName(t *testing.T) {
	for _, name := range []string{"bad#name", "bad@name"} {
		disp := &stubDispatcher{}
		s := NewServer(disp, nil)
		_, err := s.CreateCollection(context.Background(), &pb.CreateCollectionRequest{
			Name:   name,
			Config: &pb.Config{Dim: 8},
		})
		if err == nil {
			t.Fatalf("%q: expected error, got nil", name)
		}
		if status.Code(err) != codes.InvalidArgument {
			t.Fatalf("%q: code = %v, want InvalidArgument", name, status.Code(err))
		}
		if disp.called {
			t.Fatalf("%q: dispatcher must not be called for a rejected name", name)
		}
	}
}

// TestMVCreateCollectionPartitions verifies the gRPC MVConfig.partitions maps
// through to the dispatched MV create args (decoded back into the engine cfg).
func TestMVCreateCollectionPartitions(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	if _, err := s.MVCreateCollection(context.Background(), &pb.MVCreateRequest{
		Name:   "mv",
		Config: &pb.MVConfig{Dim: 8, Partitions: 4},
	}); err != nil {
		t.Fatalf("MVCreateCollection: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_mv_create_collection" {
		t.Fatalf("dispatch op = %q, want vector_mv_create_collection", disp.lastOp)
	}
	_, cfg, err := ops.DecodeMVCreateArgs(disp.lastArg)
	if err != nil {
		t.Fatalf("DecodeMVCreateArgs: %v", err)
	}
	if cfg.Partitions != 4 {
		t.Fatalf("Partitions = %d, want 4", cfg.Partitions)
	}
}

// TestMVCreateCollectionIVF verifies the gRPC MVConfig IVF knobs map through to
// the dispatched MV create args (decoded back into the engine config).
func TestMVCreateCollectionIVF(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	if _, err := s.MVCreateCollection(context.Background(), &pb.MVCreateRequest{
		Name: "mv",
		Config: &pb.MVConfig{
			Dim: 32, IndexType: "ivf", IvfNlist: 64, IvfNprobe: 12,
			IvfPq: true, IvfPqM: 8, IvfRerank: true, Opq: true, IvfTrainThreshold: 1000,
		},
	}); err != nil {
		t.Fatalf("MVCreateCollection: %v", err)
	}
	_, cfg, err := ops.DecodeMVCreateArgs(disp.lastArg)
	if err != nil {
		t.Fatalf("DecodeMVCreateArgs: %v", err)
	}
	if cfg.IndexType != vector.IndexIVF || cfg.IVFNlist != 64 || cfg.IVFNprobe != 12 ||
		!cfg.IVFPQ || cfg.IVFPQM != 8 || !cfg.IVFRerank || !cfg.OPQ || cfg.IVFTrainThreshold != 1000 {
		t.Fatalf("IVF fields not carried through gRPC: %+v", cfg)
	}
}

// TestMVCreateCollectionPQDropVecs verifies the gRPC MVConfig.pq_drop_vecs flag
// maps through to the dispatched MV create args (decoded back into the engine
// config), so the HNSW-PQ float-drop is honored for the MV inner index.
func TestMVCreateCollectionPQDropVecs(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	if _, err := s.MVCreateCollection(context.Background(), &pb.MVCreateRequest{
		Name: "mv",
		Config: &pb.MVConfig{
			Dim: 32, Quant: "pq", IvfTrainThreshold: 500, PqDropVecs: true,
		},
	}); err != nil {
		t.Fatalf("MVCreateCollection: %v", err)
	}
	_, cfg, err := ops.DecodeMVCreateArgs(disp.lastArg)
	if err != nil {
		t.Fatalf("DecodeMVCreateArgs: %v", err)
	}
	if !cfg.PQDropVecs || cfg.Quant != vector.QuantPQ || cfg.IVFTrainThreshold != 500 {
		t.Fatalf("PQDropVecs fields not carried through gRPC: %+v", cfg)
	}
}

// TestMVCreateCollectionRejectsBadIndexType proves a bad index_type fails loud at
// the gRPC edge (InvalidArgument), without dispatching.
func TestMVCreateCollectionRejectsBadIndexType(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	_, err := s.MVCreateCollection(context.Background(), &pb.MVCreateRequest{
		Name:   "mv",
		Config: &pb.MVConfig{Dim: 8, IndexType: "bogus"},
	})
	if err == nil || status.Code(err) != codes.InvalidArgument {
		t.Fatalf("bad index_type: err=%v code=%v, want InvalidArgument", err, status.Code(err))
	}
	if disp.called {
		t.Fatalf("dispatcher must not be called for a bad index_type")
	}
}

func TestMVCreateCollectionRejectsNegativePartitions(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	_, err := s.MVCreateCollection(context.Background(), &pb.MVCreateRequest{
		Name:   "mv",
		Config: &pb.MVConfig{Dim: 8, Partitions: -1},
	})
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("code = %v, want InvalidArgument", status.Code(err))
	}
	if disp.called {
		t.Fatalf("dispatcher must not be called for negative partitions")
	}
}

func TestMVCreateCollectionRejectsReservedName(t *testing.T) {
	for _, name := range []string{"bad#name", "bad@name"} {
		disp := &stubDispatcher{}
		s := NewServer(disp, nil)
		_, err := s.MVCreateCollection(context.Background(), &pb.MVCreateRequest{
			Name:   name,
			Config: &pb.MVConfig{Dim: 8},
		})
		if err == nil {
			t.Fatalf("%q: expected error, got nil", name)
		}
		if status.Code(err) != codes.InvalidArgument {
			t.Fatalf("%q: code = %v, want InvalidArgument", name, status.Code(err))
		}
		if disp.called {
			t.Fatalf("%q: dispatcher must not be called for a rejected name", name)
		}
	}
}

// countingDispatcher returns a fixed body so cleanup-count decoding can be
// exercised, while still recording the dispatched op + args.
type countingDispatcher struct {
	stubDispatcher
	body []byte
}

func (c *countingDispatcher) Call(name string, args []byte) ([]byte, error) {
	c.called = true
	c.lastOp = name
	c.lastArg = args
	return c.body, nil
}

func TestResplitDispatchesOp(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	resp, err := s.Resplit(context.Background(), &pb.ResplitRequest{Name: "docs", NewPartitions: 8})
	if err != nil {
		t.Fatalf("Resplit: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_resplit" {
		t.Fatalf("dispatch op = %q, want vector_resplit", disp.lastOp)
	}
	coll, newP, err := ops.DecodeResplitArgs(disp.lastArg)
	if err != nil {
		t.Fatalf("DecodeResplitArgs: %v", err)
	}
	if coll != "docs" || newP != 8 {
		t.Fatalf("decoded args = (%q, %d), want (docs, 8)", coll, newP)
	}
	if resp.GetName() != "docs" || resp.GetNewPartitions() != 8 {
		t.Fatalf("resp = (%q, %d), want (docs, 8)", resp.GetName(), resp.GetNewPartitions())
	}
}

func TestResplitRejectsNegative(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	_, err := s.Resplit(context.Background(), &pb.ResplitRequest{Name: "docs", NewPartitions: -1})
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("code = %v, want InvalidArgument", status.Code(err))
	}
	if disp.called {
		t.Fatalf("dispatcher must not be called for negative new_partitions")
	}
}

func TestResplitCleanupDecodesCount(t *testing.T) {
	disp := &countingDispatcher{body: ops.EncodeResplitCleanupResult(3)}
	s := NewServer(disp, nil)
	resp, err := s.ResplitCleanup(context.Background(), &pb.ResplitCleanupRequest{Name: "docs"})
	if err != nil {
		t.Fatalf("ResplitCleanup: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_resplit_cleanup" {
		t.Fatalf("dispatch op = %q, want vector_resplit_cleanup", disp.lastOp)
	}
	if resp.GetDropped() != 3 {
		t.Fatalf("Dropped = %d, want 3", resp.GetDropped())
	}
	if resp.GetName() != "docs" {
		t.Fatalf("Name = %q, want docs", resp.GetName())
	}
}

func TestMVResplitDispatchesOp(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	resp, err := s.MVResplit(context.Background(), &pb.ResplitRequest{Name: "mv", NewPartitions: 8})
	if err != nil {
		t.Fatalf("MVResplit: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_mv_resplit" {
		t.Fatalf("dispatch op = %q, want vector_mv_resplit", disp.lastOp)
	}
	coll, newP, err := ops.DecodeResplitArgs(disp.lastArg)
	if err != nil {
		t.Fatalf("DecodeResplitArgs: %v", err)
	}
	if coll != "mv" || newP != 8 {
		t.Fatalf("decoded args = (%q, %d), want (mv, 8)", coll, newP)
	}
	if resp.GetName() != "mv" || resp.GetNewPartitions() != 8 {
		t.Fatalf("resp = (%q, %d), want (mv, 8)", resp.GetName(), resp.GetNewPartitions())
	}
}

func TestMVResplitRejectsNegative(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	_, err := s.MVResplit(context.Background(), &pb.ResplitRequest{Name: "mv", NewPartitions: -1})
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("code = %v, want InvalidArgument", status.Code(err))
	}
	if disp.called {
		t.Fatalf("dispatcher must not be called for negative new_partitions")
	}
}

func TestMVResplitCleanupDecodesCount(t *testing.T) {
	disp := &countingDispatcher{body: ops.EncodeResplitCleanupResult(2)}
	s := NewServer(disp, nil)
	resp, err := s.MVResplitCleanup(context.Background(), &pb.ResplitCleanupRequest{Name: "mv"})
	if err != nil {
		t.Fatalf("MVResplitCleanup: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_mv_resplit_cleanup" {
		t.Fatalf("dispatch op = %q, want vector_mv_resplit_cleanup", disp.lastOp)
	}
	if resp.GetDropped() != 2 {
		t.Fatalf("Dropped = %d, want 2", resp.GetDropped())
	}
}

func TestGRPCReshard(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	resp, err := s.Reshard(context.Background(), &pb.ReshardRequest{Name: "docs", NewPartitions: 8})
	if err != nil {
		t.Fatalf("Reshard: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_reshard" {
		t.Fatalf("dispatch op = %q, want vector_reshard", disp.lastOp)
	}
	coll, newP, err := ops.DecodeReshardArgs(disp.lastArg)
	if err != nil {
		t.Fatalf("DecodeReshardArgs: %v", err)
	}
	if coll != "docs" || newP != 8 {
		t.Fatalf("decoded args = (%q, %d), want (docs, 8)", coll, newP)
	}
	if resp.GetName() != "docs" || resp.GetNewPartitions() != 8 {
		t.Fatalf("resp = (%q, %d), want (docs, 8)", resp.GetName(), resp.GetNewPartitions())
	}
}

func TestGRPCReshardAbort(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	resp, err := s.ReshardAbort(context.Background(), &pb.ReshardAbortRequest{Name: "docs"})
	if err != nil {
		t.Fatalf("ReshardAbort: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_reshard_abort" {
		t.Fatalf("dispatch op = %q, want vector_reshard_abort", disp.lastOp)
	}
	coll, err := ops.DecodeReshardAbortArgs(disp.lastArg)
	if err != nil {
		t.Fatalf("DecodeReshardAbortArgs: %v", err)
	}
	if coll != "docs" {
		t.Fatalf("decoded args = %q, want docs", coll)
	}
	if resp.GetName() != "docs" {
		t.Fatalf("resp = %q, want docs", resp.GetName())
	}
}

func TestGRPCMVReshard(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	resp, err := s.MVReshard(context.Background(), &pb.ReshardRequest{Name: "mv", NewPartitions: 4})
	if err != nil {
		t.Fatalf("MVReshard: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_mv_reshard" {
		t.Fatalf("dispatch op = %q, want vector_mv_reshard", disp.lastOp)
	}
	coll, newP, err := ops.DecodeReshardArgs(disp.lastArg)
	if err != nil {
		t.Fatalf("DecodeReshardArgs: %v", err)
	}
	if coll != "mv" || newP != 4 {
		t.Fatalf("decoded args = (%q, %d), want (mv, 4)", coll, newP)
	}
	if resp.GetName() != "mv" || resp.GetNewPartitions() != 4 {
		t.Fatalf("resp = (%q, %d), want (mv, 4)", resp.GetName(), resp.GetNewPartitions())
	}
}

func TestGRPCMVReshardAbort(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	resp, err := s.MVReshardAbort(context.Background(), &pb.ReshardAbortRequest{Name: "mv"})
	if err != nil {
		t.Fatalf("MVReshardAbort: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_mv_reshard_abort" {
		t.Fatalf("dispatch op = %q, want vector_mv_reshard_abort", disp.lastOp)
	}
	coll, err := ops.DecodeReshardAbortArgs(disp.lastArg)
	if err != nil {
		t.Fatalf("DecodeReshardAbortArgs: %v", err)
	}
	if coll != "mv" {
		t.Fatalf("decoded args = %q, want mv", coll)
	}
	if resp.GetName() != "mv" {
		t.Fatalf("resp = %q, want mv", resp.GetName())
	}
}

// TestSearchSurfacesDegraded asserts the Search handler decodes the degraded
// trailer and surfaces Degraded + Missing (widened to uint32) on the response.
func TestSearchSurfacesDegraded(t *testing.T) {
	body := ops.EncodeVectorSearchResultsDegraded(
		[]vector.Result{{ID: 1, Distance: 0.5}}, true, []uint16{2})
	disp := &countingDispatcher{body: body}
	s := NewServer(disp, nil)
	resp, err := s.Search(context.Background(), &pb.SearchRequest{Collection: "docs", K: 1})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_search" {
		t.Fatalf("dispatch op = %q, want vector_search", disp.lastOp)
	}
	if !resp.GetDegraded() {
		t.Fatalf("Degraded = false, want true")
	}
	if got := resp.GetMissing(); len(got) != 1 || got[0] != 2 {
		t.Fatalf("Missing = %v, want [2]", got)
	}
	if len(resp.GetResults()) != 1 || resp.GetResults()[0].GetId() != 1 {
		t.Fatalf("Results = %v, want one result id=1", resp.GetResults())
	}
}

// TestSearchNonDegraded asserts a legacy (no-trailer) body decodes with
// Degraded=false and an empty Missing slice.
func TestSearchNonDegraded(t *testing.T) {
	body := ops.EncodeVectorSearchResults([]vector.Result{{ID: 7, Distance: 0.1}})
	disp := &countingDispatcher{body: body}
	s := NewServer(disp, nil)
	resp, err := s.Search(context.Background(), &pb.SearchRequest{Collection: "docs", K: 1})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if resp.GetDegraded() {
		t.Fatalf("Degraded = true, want false")
	}
	if len(resp.GetMissing()) != 0 {
		t.Fatalf("Missing = %v, want empty", resp.GetMissing())
	}
}

// TestMVSearchSurfacesDegraded covers a second response type (MVSearch) through
// its degraded encoder.
func TestMVSearchSurfacesDegraded(t *testing.T) {
	body := ops.EncodeMVResultsDegraded(
		[]vector.MultiResult{{ID: 3, Score: 1.5}}, true, []uint16{1, 4})
	disp := &countingDispatcher{body: body}
	s := NewServer(disp, nil)
	resp, err := s.MVSearch(context.Background(), &pb.MVSearchRequest{Name: "mv", K: 1})
	if err != nil {
		t.Fatalf("MVSearch: %v", err)
	}
	if !resp.GetDegraded() {
		t.Fatalf("Degraded = false, want true")
	}
	if got := resp.GetMissing(); len(got) != 2 || got[0] != 1 || got[1] != 4 {
		t.Fatalf("Missing = %v, want [1 4]", got)
	}
}

// TestMVSearchNonDegraded asserts a legacy (no-trailer) MVSearch body decodes
// with Degraded=false and an empty Missing slice.
func TestMVSearchNonDegraded(t *testing.T) {
	body := ops.EncodeMVResults([]vector.MultiResult{{ID: 7, Score: 0.1}})
	disp := &countingDispatcher{body: body}
	s := NewServer(disp, nil)
	resp, err := s.MVSearch(context.Background(), &pb.MVSearchRequest{Name: "mv", K: 1})
	if err != nil {
		t.Fatalf("MVSearch: %v", err)
	}
	if resp.GetDegraded() {
		t.Fatalf("Degraded = true, want false")
	}
	if len(resp.GetMissing()) != 0 {
		t.Fatalf("Missing = %v, want empty", resp.GetMissing())
	}
}

// ---- consistency request fields (read_consistency / on_partition_unavailable) ----

// TestSearchConsistencyReachesArgs proves that a gRPC Search request carrying
// read_consistency=1 / on_partition_unavailable=1 dispatches *Opts-encoded args
// whose decoded rc/opa are 1/1.
func TestSearchConsistencyReachesArgs(t *testing.T) {
	disp := &countingDispatcher{body: ops.EncodeVectorSearchResults(nil)}
	s := NewServer(disp, nil)
	_, err := s.Search(context.Background(), &pb.SearchRequest{
		Collection: "docs", K: 5, Query: []float32{1, 2},
		ReadConsistency: 1, OnPartitionUnavailable: 1,
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_search" {
		t.Fatalf("dispatch op = %q, want vector_search", disp.lastOp)
	}
	_, _, _, _, rc, opa, _, derr := ops.DecodeVectorSearchArgsOpts(disp.lastArg)
	if derr != nil {
		t.Fatalf("DecodeVectorSearchArgsOpts: %v", derr)
	}
	if rc != 1 || opa != 1 {
		t.Fatalf("decoded rc/opa = %d/%d, want 1/1", rc, opa)
	}
}

// TestSearchRejectsOutOfRangeConsistency proves an out-of-range read_consistency
// (4, above BoundedStaleness=3) is rejected with InvalidArgument before dispatch.
func TestSearchRejectsOutOfRangeConsistency(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	_, err := s.Search(context.Background(), &pb.SearchRequest{
		Collection: "docs", K: 5, Query: []float32{1, 2}, ReadConsistency: 4,
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("code = %v, want InvalidArgument", status.Code(err))
	}
	if disp.called {
		t.Fatalf("dispatcher must not be called for out-of-range read_consistency")
	}
}

// TestHybridSearchConsistencyReachesArgs proves Hybrid threads rc/opa into the
// *Opts-encoded dispatch args.
func TestHybridSearchConsistencyReachesArgs(t *testing.T) {
	disp := &countingDispatcher{body: ops.EncodeHybridResults(nil)}
	s := NewServer(disp, nil)
	_, err := s.HybridSearch(context.Background(), &pb.HybridRequest{
		Collection: "docs", K: 5, Dense: []float32{1, 2}, Method: "rrf",
		ReadConsistency: 1, OnPartitionUnavailable: 1,
	})
	if err != nil {
		t.Fatalf("HybridSearch: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_hybrid_search" {
		t.Fatalf("dispatch op = %q, want vector_hybrid_search", disp.lastOp)
	}
	_, _, _, _, _, rc, opa, _, derr := ops.DecodeHybridSearchArgsOpts(disp.lastArg)
	if derr != nil {
		t.Fatalf("DecodeHybridSearchArgsOpts: %v", derr)
	}
	if rc != 1 || opa != 1 {
		t.Fatalf("decoded rc/opa = %d/%d, want 1/1", rc, opa)
	}
}

// TestHybridSearchRejectsOutOfRangeConsistency proves Hybrid rejects an
// out-of-range on_partition_unavailable before dispatch.
func TestHybridSearchRejectsOutOfRangeConsistency(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	_, err := s.HybridSearch(context.Background(), &pb.HybridRequest{
		Collection: "docs", K: 5, Dense: []float32{1, 2}, Method: "rrf",
		OnPartitionUnavailable: 2,
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("code = %v, want InvalidArgument", status.Code(err))
	}
	if disp.called {
		t.Fatalf("dispatcher must not be called for out-of-range on_partition_unavailable")
	}
}

// TestMVSearchConsistencyReachesArgs proves MVSearch threads rc/opa into the
// *Opts-encoded dispatch args.
func TestMVSearchConsistencyReachesArgs(t *testing.T) {
	disp := &countingDispatcher{body: ops.EncodeMVResults(nil)}
	s := NewServer(disp, nil)
	_, err := s.MVSearch(context.Background(), &pb.MVSearchRequest{
		Name: "mv", K: 1, ReadConsistency: 1, OnPartitionUnavailable: 1,
	})
	if err != nil {
		t.Fatalf("MVSearch: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_mv_search" {
		t.Fatalf("dispatch op = %q, want vector_mv_search", disp.lastOp)
	}
	_, _, _, _, rc, opa, _, derr := ops.DecodeMVSearchArgsOpts(disp.lastArg)
	if derr != nil {
		t.Fatalf("DecodeMVSearchArgsOpts: %v", derr)
	}
	if rc != 1 || opa != 1 {
		t.Fatalf("decoded rc/opa = %d/%d, want 1/1", rc, opa)
	}
}

// TestMVSearchFilterReachesArgs proves MVSearch parses filter_json at the edge
// and threads the compiled filter into the *OptsFilter-encoded dispatch args
// (mirrors the dense Search path).
func TestMVSearchFilterReachesArgs(t *testing.T) {
	disp := &countingDispatcher{body: ops.EncodeMVResults(nil)}
	s := NewServer(disp, nil)
	_, err := s.MVSearch(context.Background(), &pb.MVSearchRequest{
		Name: "mv", K: 1,
		FilterJson: `{"op":"eq","field":"lang","value":{"kind":"string","str":"en"}}`,
	})
	if err != nil {
		t.Fatalf("MVSearch: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_mv_search" {
		t.Fatalf("dispatch op = %q, want vector_mv_search", disp.lastOp)
	}
	_, _, _, _, _, _, filter, _, derr := ops.DecodeMVSearchArgsOptsFilter(disp.lastArg)
	if derr != nil {
		t.Fatalf("DecodeMVSearchArgsOptsFilter: %v", derr)
	}
	if filter.Op != vector.FilterEq || filter.Field != "lang" {
		t.Fatalf("decoded filter = %+v, want eq on lang", filter)
	}
}

// TestMVSearchNoFilterUnchanged proves a filter-less MVSearch still dispatches
// and decodes a zero (match-all) filter — the no-filter path is unchanged.
func TestMVSearchNoFilterUnchanged(t *testing.T) {
	disp := &countingDispatcher{body: ops.EncodeMVResults(nil)}
	s := NewServer(disp, nil)
	if _, err := s.MVSearch(context.Background(), &pb.MVSearchRequest{Name: "mv", K: 1}); err != nil {
		t.Fatalf("MVSearch: %v", err)
	}
	_, _, _, _, _, _, filter, _, derr := ops.DecodeMVSearchArgsOptsFilter(disp.lastArg)
	if derr != nil {
		t.Fatalf("DecodeMVSearchArgsOptsFilter: %v", derr)
	}
	if !filter.IsZero() {
		t.Fatalf("decoded filter = %+v, want zero (match-all) filter", filter)
	}
}

// TestMVSearchInvalidFilter asserts a syntactically-valid filter_json that fails
// to Compile (bad regex / bad RFC3339) yields codes.InvalidArgument and never
// reaches the dispatcher — fail-loud at the edge, like dense Search.
func TestMVSearchInvalidFilter(t *testing.T) {
	cases := []struct {
		name   string
		filter string
	}{
		{"bad_regex", `{"op":"regex","field":"sku","value":{"kind":"string","str":"*invalid"}}`},
		{"bad_datetime", `{"op":"dt_gte","field":"created_at","value":{"kind":"string","str":"not-a-date"}}`},
		{"bad_json", `{not json`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			disp := &stubDispatcher{}
			s := NewServer(disp, nil)
			_, err := s.MVSearch(context.Background(), &pb.MVSearchRequest{
				Name: "mv", K: 1, FilterJson: c.filter,
			})
			if status.Code(err) != codes.InvalidArgument {
				t.Fatalf("code = %v, want InvalidArgument (%v)", status.Code(err), err)
			}
			if disp.called {
				t.Fatalf("dispatcher must not be called for an invalid filter")
			}
		})
	}
}

// TestMVSearchRejectsOutOfRangeConsistency proves MVSearch rejects an
// out-of-range read_consistency (3, above Linearizable=2) before dispatch.
func TestMVSearchRejectsOutOfRangeConsistency(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	_, err := s.MVSearch(context.Background(), &pb.MVSearchRequest{
		Name: "mv", K: 1, ReadConsistency: 4,
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("code = %v, want InvalidArgument", status.Code(err))
	}
	if disp.called {
		t.Fatalf("dispatcher must not be called for out-of-range read_consistency")
	}
}

// TestGRPCLinearizableAcceptedAllFamilies proves read_consistency=2
// (Linearizable) is ACCEPTED and threads to the engine as rc=2 for every gRPC
// read family, and that read_consistency=3 (above Linearizable) is rejected with
// InvalidArgument before dispatch. The dispatcher is a stub (no shard / no
// readIndex barrier), so this isolates the wire: the level survives the gRPC edge
// unchanged — neither dropped nor clamped per family.
func TestGRPCLinearizableAcceptedAllFamilies(t *testing.T) {
	searchRC := func(b []byte) uint8 {
		_, _, _, _, rc, _, _, _ := ops.DecodeVectorSearchArgsOpts(b)
		return rc
	}
	cases := []struct {
		name     string
		op       string
		body     []byte
		call2    func(s *Server) error // dispatch with rc=2 (must succeed)
		call3    func(s *Server) error // dispatch with rc=3 (must be rejected)
		decodeRC func([]byte) uint8
	}{
		{
			name: "search", op: "vector_search", body: ops.EncodeVectorSearchResults(nil),
			call2: func(s *Server) error {
				_, e := s.Search(context.Background(), &pb.SearchRequest{Collection: "docs", K: 1, Query: []float32{1, 2}, ReadConsistency: 2})
				return e
			},
			call3: func(s *Server) error {
				_, e := s.Search(context.Background(), &pb.SearchRequest{Collection: "docs", K: 1, Query: []float32{1, 2}, ReadConsistency: 4})
				return e
			},
			decodeRC: searchRC,
		},
		{
			name: "search_docs", op: "vector_search_docs", body: ops.EncodeVectorDocs(nil),
			call2: func(s *Server) error {
				_, e := s.SearchDocs(context.Background(), &pb.SearchRequest{Collection: "docs", K: 1, Query: []float32{1, 2}, ReadConsistency: 2})
				return e
			},
			call3: func(s *Server) error {
				_, e := s.SearchDocs(context.Background(), &pb.SearchRequest{Collection: "docs", K: 1, Query: []float32{1, 2}, ReadConsistency: 4})
				return e
			},
			decodeRC: searchRC,
		},
		{
			name: "hybrid", op: "vector_hybrid_search", body: ops.EncodeHybridResults(nil),
			call2: func(s *Server) error {
				_, e := s.HybridSearch(context.Background(), &pb.HybridRequest{Collection: "docs", K: 1, Dense: []float32{1, 2}, Method: "rrf", ReadConsistency: 2})
				return e
			},
			call3: func(s *Server) error {
				_, e := s.HybridSearch(context.Background(), &pb.HybridRequest{Collection: "docs", K: 1, Dense: []float32{1, 2}, Method: "rrf", ReadConsistency: 4})
				return e
			},
			decodeRC: func(b []byte) uint8 {
				_, _, _, _, _, rc, _, _, _ := ops.DecodeHybridSearchArgsOpts(b)
				return rc
			},
		},
		{
			name: "groups", op: "vector_search_groups", body: ops.EncodeGroupsDegraded(nil, false, nil),
			call2: func(s *Server) error {
				_, e := s.SearchGroups(context.Background(), &pb.SearchGroupsRequest{Collection: "docs", K: 1, Query: []float32{1, 2}, GroupBy: "x", ReadConsistency: 2})
				return e
			},
			call3: func(s *Server) error {
				_, e := s.SearchGroups(context.Background(), &pb.SearchGroupsRequest{Collection: "docs", K: 1, Query: []float32{1, 2}, GroupBy: "x", ReadConsistency: 4})
				return e
			},
			decodeRC: func(b []byte) uint8 {
				_, _, _, _, rc, _, _, _ := ops.DecodeGroupSearchArgsOpts(b)
				return rc
			},
		},
		{
			name: "scroll", op: "vector_scroll", body: ops.EncodeScrollResult(nil, false, nil, ""),
			call2: func(s *Server) error {
				_, e := s.Scroll(context.Background(), &pb.ScrollRequest{Collection: "docs", Limit: 5, ReadConsistency: 2})
				return e
			},
			call3: func(s *Server) error {
				_, e := s.Scroll(context.Background(), &pb.ScrollRequest{Collection: "docs", Limit: 5, ReadConsistency: 4})
				return e
			},
			decodeRC: func(b []byte) uint8 {
				_, _, _, rc, _, _ := ops.DecodeScrollArgsOpts(b)
				return rc
			},
		},
		{
			name: "mv_search", op: "vector_mv_search", body: ops.EncodeMVResults(nil),
			call2: func(s *Server) error {
				_, e := s.MVSearch(context.Background(), &pb.MVSearchRequest{Name: "mv", K: 1, ReadConsistency: 2})
				return e
			},
			call3: func(s *Server) error {
				_, e := s.MVSearch(context.Background(), &pb.MVSearchRequest{Name: "mv", K: 1, ReadConsistency: 4})
				return e
			},
			decodeRC: func(b []byte) uint8 {
				_, _, _, _, rc, _, _, _ := ops.DecodeMVSearchArgsOpts(b)
				return rc
			},
		},
		{
			name: "mv_scroll", op: "vector_mv_scroll", body: ops.EncodeScrollResult(nil, false, nil, ""),
			call2: func(s *Server) error {
				_, e := s.MVScroll(context.Background(), &pb.MVScrollRequest{Collection: "mv", Limit: 5, ReadConsistency: 2})
				return e
			},
			call3: func(s *Server) error {
				_, e := s.MVScroll(context.Background(), &pb.MVScrollRequest{Collection: "mv", Limit: 5, ReadConsistency: 4})
				return e
			},
			decodeRC: func(b []byte) uint8 {
				_, _, _, rc, _, _, _, _, _ := ops.DecodeMVScrollArgsOpts(b)
				return rc
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name+"/linearizable_accepted", func(t *testing.T) {
			disp := &countingDispatcher{body: tc.body}
			s := NewServer(disp, nil)
			if err := tc.call2(s); err != nil {
				t.Fatalf("%s rc=2: %v", tc.name, err)
			}
			if !disp.called || disp.lastOp != tc.op {
				t.Fatalf("%s dispatch op = %q, want %s", tc.name, disp.lastOp, tc.op)
			}
			if got := tc.decodeRC(disp.lastArg); got != 2 {
				t.Fatalf("%s threaded rc=%d, want 2 (Linearizable) — level dropped/clamped at edge", tc.name, got)
			}
		})
		t.Run(tc.name+"/above_linearizable_rejected", func(t *testing.T) {
			disp := &stubDispatcher{}
			s := NewServer(disp, nil)
			err := tc.call3(s)
			if status.Code(err) != codes.InvalidArgument {
				t.Fatalf("%s rc=3 code = %v, want InvalidArgument", tc.name, status.Code(err))
			}
			if disp.called {
				t.Fatalf("%s dispatched on rc=3, want no dispatch", tc.name)
			}
		})
	}
}

// grpcRichFilterJSON is a filter_json string exercising the four representative
// rich operators (match / regex / is_empty / dt_gte). Valid → Compile succeeds.
const grpcRichFilterJSON = `{"op":"and","and":[` +
	`{"op":"match","field":"title","value":{"kind":"string","str":"quick brown"}},` +
	`{"op":"regex","field":"sku","value":{"kind":"string","str":"^A[0-9]+$"}},` +
	`{"op":"is_empty","field":"deleted_at"},` +
	`{"op":"dt_gte","field":"created_at","value":{"kind":"string","str":"2024-01-02T15:04:05Z"}}` +
	`]}`

// TestGRPCSearchRichFilter proves a filter_json using the rich ops parses at the
// gRPC edge and the dispatched vector_search op carries the encoded filter.
func TestGRPCSearchRichFilter(t *testing.T) {
	disp := &countingDispatcher{body: ops.EncodeVectorSearchResults(nil)}
	s := NewServer(disp, nil)
	_, err := s.Search(context.Background(), &pb.SearchRequest{
		Collection: "docs", K: 3, FilterJson: grpcRichFilterJSON,
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_search" {
		t.Fatalf("dispatch op = %q, want vector_search", disp.lastOp)
	}
	_, _, _, filter, decErr := ops.DecodeVectorSearchArgs(disp.lastArg)
	if decErr != nil {
		t.Fatalf("decode search args: %v", decErr)
	}
	if filter.Op != vector.FilterAnd || len(filter.And) != 4 {
		t.Fatalf("decoded filter = %+v, want And with 4 children", filter)
	}
	wantOps := []vector.FilterOp{vector.FilterMatch, vector.FilterRegex, vector.FilterIsEmpty, vector.FilterDtGte}
	for i, op := range wantOps {
		if filter.And[i].Op != op {
			t.Fatalf("child %d op = %v, want %v", i, filter.And[i].Op, op)
		}
	}
}

// TestGRPCInvalidFilter asserts a syntactically-valid filter_json that fails to
// Compile (bad regex / bad RFC3339, plus the geo cases) yields
// codes.InvalidArgument and — for the delete path — never reaches the dispatcher
// (no over-broad delete on a bad geo filter).
func TestGRPCInvalidFilter(t *testing.T) {
	cases := []struct {
		name   string
		filter string
	}{
		{"bad_regex", `{"op":"regex","field":"sku","value":{"kind":"string","str":"*invalid"}}`},
		{"bad_datetime", `{"op":"dt_gte","field":"created_at","value":{"kind":"string","str":"not-a-date"}}`},
		{"geo_nil", `{"op":"geo_radius","field":"loc"}`},
		{"geo_bad_polygon", `{"op":"geo_polygon","field":"loc","geo":{"polygon":[1,2,3,4]}}`},
		{"geo_out_of_range", `{"op":"geo_radius","field":"loc","geo":{"center_lat":95,"center_lon":0,"radius_m":1000}}`},
		// JSON null on a float64 → 0 (NaN is unexpressible in valid JSON; covered
		// at the unit level). Zero radius is rejected by the same !(>0) guard.
		{"geo_zero_radius", `{"op":"geo_radius","field":"loc","geo":{"center_lat":48,"center_lon":2,"radius_m":null}}`},
		{"geo_inverted_box", `{"op":"geo_bounding_box","field":"loc","geo":{"min_lat":49,"min_lon":3,"max_lat":48,"max_lon":2}}`},
	}
	for _, c := range cases {
		t.Run("search/"+c.name, func(t *testing.T) {
			disp := &stubDispatcher{}
			s := NewServer(disp, nil)
			_, err := s.Search(context.Background(), &pb.SearchRequest{
				Collection: "docs", K: 1, FilterJson: c.filter,
			})
			if status.Code(err) != codes.InvalidArgument {
				t.Fatalf("code = %v, want InvalidArgument (%v)", status.Code(err), err)
			}
			if disp.called {
				t.Fatalf("dispatcher must not be called for an invalid filter")
			}
		})
		t.Run("delete_by_filter/"+c.name, func(t *testing.T) {
			disp := &stubDispatcher{}
			s := NewServer(disp, nil)
			_, err := s.DeleteByFilter(context.Background(), &pb.DeleteByFilterRequest{
				Collection: "docs", FilterJson: c.filter,
			})
			if status.Code(err) != codes.InvalidArgument {
				t.Fatalf("code = %v, want InvalidArgument (%v)", status.Code(err), err)
			}
			if disp.called {
				t.Fatalf("dispatcher must not be called for an invalid delete filter (no over-broad delete)")
			}
		})
		t.Run("search_docs/"+c.name, func(t *testing.T) {
			disp := &stubDispatcher{}
			s := NewServer(disp, nil)
			_, err := s.SearchDocs(context.Background(), &pb.SearchRequest{
				Collection: "docs", K: 1, FilterJson: c.filter,
			})
			if status.Code(err) != codes.InvalidArgument {
				t.Fatalf("code = %v, want InvalidArgument (%v)", status.Code(err), err)
			}
			if disp.called {
				t.Fatalf("dispatcher must not be called for an invalid filter")
			}
		})
		t.Run("search_groups/"+c.name, func(t *testing.T) {
			disp := &stubDispatcher{}
			s := NewServer(disp, nil)
			_, err := s.SearchGroups(context.Background(), &pb.SearchGroupsRequest{
				Collection: "docs", K: 1, GroupBy: "x", FilterJson: c.filter,
			})
			if status.Code(err) != codes.InvalidArgument {
				t.Fatalf("code = %v, want InvalidArgument (%v)", status.Code(err), err)
			}
			if disp.called {
				t.Fatalf("dispatcher must not be called for an invalid filter")
			}
		})
		t.Run("hybrid/"+c.name, func(t *testing.T) {
			disp := &stubDispatcher{}
			s := NewServer(disp, nil)
			_, err := s.HybridSearch(context.Background(), &pb.HybridRequest{
				Collection: "docs", K: 1, Dense: []float32{1, 2}, FilterJson: c.filter,
			})
			if status.Code(err) != codes.InvalidArgument {
				t.Fatalf("code = %v, want InvalidArgument (%v)", status.Code(err), err)
			}
			if disp.called {
				t.Fatalf("dispatcher must not be called for an invalid filter")
			}
		})
	}
}

// grpcGeoFilterJSON is a filter_json exercising all three geo operators each
// with a GeoCondition; valid → Compile succeeds and the encoded filter reaches
// the dispatcher untouched (geo rides as JSON, no byte-format change).
const grpcGeoFilterJSON = `{"op":"and","and":[` +
	`{"op":"geo_radius","field":"loc","geo":{"center_lat":48.8566,"center_lon":2.3522,"radius_m":5000}},` +
	`{"op":"geo_bounding_box","field":"loc","geo":{"min_lat":48,"min_lon":2,"max_lat":49,"max_lon":3}},` +
	`{"op":"geo_polygon","field":"loc","geo":{"polygon":[48,2,49,2,49,3,48,3]}}` +
	`]}`

// TestGRPCGeoFilter proves a geo filter_json parses at the gRPC edge and the
// dispatched search / delete-by-filter op carries the encoded filter with its
// GeoConditions intact.
func TestGRPCGeoFilter(t *testing.T) {
	t.Run("search", func(t *testing.T) {
		disp := &countingDispatcher{body: ops.EncodeVectorSearchResults(nil)}
		s := NewServer(disp, nil)
		if _, err := s.Search(context.Background(), &pb.SearchRequest{
			Collection: "docs", K: 3, FilterJson: grpcGeoFilterJSON,
		}); err != nil {
			t.Fatalf("Search: %v", err)
		}
		if !disp.called || disp.lastOp != "vector_search" {
			t.Fatalf("dispatch op = %q, want vector_search", disp.lastOp)
		}
		_, _, _, filter, decErr := ops.DecodeVectorSearchArgs(disp.lastArg)
		if decErr != nil {
			t.Fatalf("decode search args: %v", decErr)
		}
		assertGRPCGeoFilter(t, filter)
	})
	t.Run("delete_by_filter", func(t *testing.T) {
		disp := &countingDispatcher{body: []byte{0, 0, 0, 0}} // delete count = 0
		s := NewServer(disp, nil)
		if _, err := s.DeleteByFilter(context.Background(), &pb.DeleteByFilterRequest{
			Collection: "docs", FilterJson: grpcGeoFilterJSON,
		}); err != nil {
			t.Fatalf("DeleteByFilter: %v", err)
		}
		if !disp.called || disp.lastOp != "vector_delete_by_filter" {
			t.Fatalf("dispatch op = %q, want vector_delete_by_filter", disp.lastOp)
		}
		_, filter, decErr := ops.DecodeDeleteByFilterArgs(disp.lastArg)
		if decErr != nil {
			t.Fatalf("decode delete args: %v", decErr)
		}
		assertGRPCGeoFilter(t, filter)
	})
}

// assertGRPCGeoFilter checks the decoded filter has the three geo-op leaves with
// their GeoConditions intact and Compiles.
func assertGRPCGeoFilter(t *testing.T, f vector.Filter) {
	t.Helper()
	if f.Op != vector.FilterAnd || len(f.And) != 3 {
		t.Fatalf("decoded filter = %+v, want And with 3 children", f)
	}
	wantOps := []vector.FilterOp{vector.FilterGeoRadius, vector.FilterGeoBox, vector.FilterGeoPolygon}
	for i, op := range wantOps {
		if f.And[i].Op != op {
			t.Fatalf("child %d op = %v, want %v", i, f.And[i].Op, op)
		}
		if f.And[i].Geo == nil {
			t.Fatalf("child %d (%v) lost its Geo condition", i, op)
		}
	}
	if r := f.And[0].Geo; r.CenterLat != 48.8566 || r.CenterLon != 2.3522 || r.RadiusM != 5000 {
		t.Fatalf("geo_radius condition = %+v, want center (48.8566,2.3522) r=5000", r)
	}
	if _, err := vector.CompileFilter(f); err != nil {
		t.Fatalf("decoded geo filter does not compile: %v", err)
	}
}

// TestGRPCUpsertGeoMetadata proves a ValueGeo metadata_json value round-trips
// through the gRPC edge: the inserted point's metadata reaches the dispatcher
// carrying a {"kind":"geo",...} value with lat/lon intact (no byte-format change).
func TestGRPCUpsertGeoMetadata(t *testing.T) {
	disp := &countingDispatcher{}
	s := NewServer(disp, nil)
	_, err := s.Upsert(context.Background(), &pb.UpsertRequest{
		Collection:   "docs",
		Id:           7,
		Vector:       []float32{1, 2, 3},
		MetadataJson: `{"loc":{"kind":"geo","lat":48.8566,"lon":2.3522}}`,
	})
	if err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_insert" {
		t.Fatalf("dispatch op = %q, want vector_insert", disp.lastOp)
	}
	_, id, _, _, meta, _, _, decErr := ops.DecodeVectorInsertArgs(disp.lastArg)
	if decErr != nil {
		t.Fatalf("decode insert args: %v", decErr)
	}
	if id != 7 {
		t.Fatalf("id = %d, want 7", id)
	}
	loc, ok := meta["loc"]
	if !ok {
		t.Fatalf("metadata missing 'loc' geo field: %+v", meta)
	}
	if loc.Kind != vector.ValueGeo || loc.Lat != 48.8566 || loc.Lon != 2.3522 {
		t.Fatalf("decoded geo metadata = %+v, want kind=geo lat=48.8566 lon=2.3522", loc)
	}
}

// ---- named-vector (Qdrant-style per-point multi-vector-space) gRPC tests ----

// realDispatcher runs ops over a real registry + vector store (mirroring how
// directStore.Call dispatches) so the named-vector RPCs can be exercised
// end-to-end through the engine, not just at the stub edge.
type realDispatcher struct {
	reg *ops.Registry
	tx  *ops.TxContext
}

func (d *realDispatcher) Call(name string, args []byte) ([]byte, error) {
	h, _, _, ok := d.reg.Lookup(name)
	if !ok {
		return nil, status.Errorf(codes.Internal, "op %q not registered", name)
	}
	return h(d.tx, args)
}

func (d *realDispatcher) LeaderAddr() string { return "" }

func newRealServer(t *testing.T) *Server {
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
	return NewServer(&realDispatcher{reg: reg, tx: ops.NewTxContextWithVectors(c, vstore)}, nil)
}

const namedConfigJSON = `{"title":{"dim":4},"image":{"dim":3,"metric":2}}`

func nvl(vals ...float32) *pb.NamedVectorList { return &pb.NamedVectorList{Values: vals} }

// TestGRPCNamedLifecycle drives create/get_config/upsert/search(+filter)/
// search_docs/scroll/delete/drop end-to-end over the new RPCs against a real
// engine.
func TestGRPCNamedLifecycle(t *testing.T) {
	s := newRealServer(t)
	ctx := context.Background()

	if _, err := s.NamedCreate(ctx, &pb.NamedCreateRequest{Name: "docs", ConfigJson: namedConfigJSON}); err != nil {
		t.Fatalf("NamedCreate: %v", err)
	}

	cfgResp, err := s.NamedGetConfig(ctx, &pb.NamedGetConfigRequest{Name: "docs"})
	if err != nil {
		t.Fatalf("NamedGetConfig: %v", err)
	}
	var cfg map[string]vector.NamedVectorParams
	if err := json.Unmarshal([]byte(cfgResp.GetConfigJson()), &cfg); err != nil {
		t.Fatalf("config json: %v", err)
	}
	if cfg["title"].Dim != 4 || cfg["image"].Metric != vector.DotProduct {
		t.Fatalf("config = %+v", cfg)
	}

	upserts := []*pb.NamedUpsertRequest{
		{Name: "docs", Id: 1, Vectors: map[string]*pb.NamedVectorList{"title": nvl(1, 0, 0, 0), "image": nvl(1, 0, 0)}, MetadataJson: `{"lang":{"kind":"string","str":"en"}}`},
		{Name: "docs", Id: 2, Vectors: map[string]*pb.NamedVectorList{"title": nvl(0, 1, 0, 0), "image": nvl(0, 1, 0)}, MetadataJson: `{"lang":{"kind":"string","str":"fr"}}`},
		{Name: "docs", Id: 3, Vectors: map[string]*pb.NamedVectorList{"title": nvl(1, 1, 0, 0)}, MetadataJson: `{"lang":{"kind":"string","str":"en"}}`},
	}
	for _, u := range upserts {
		if _, err := s.NamedUpsert(ctx, u); err != nil {
			t.Fatalf("NamedUpsert id=%d: %v", u.GetId(), err)
		}
	}

	sres, err := s.NamedSearch(ctx, &pb.NamedSearchRequest{Name: "docs", VectorName: "title", Query: []float32{1, 0, 0, 0}, K: 3})
	if err != nil {
		t.Fatalf("NamedSearch: %v", err)
	}
	if len(sres.GetResults()) == 0 || sres.GetResults()[0].GetId() != 1 {
		t.Fatalf("title search = %+v", sres.GetResults())
	}

	// Filtered (lang=en) excludes fr point 2.
	fres, err := s.NamedSearch(ctx, &pb.NamedSearchRequest{
		Name: "docs", VectorName: "title", Query: []float32{1, 0, 0, 0}, K: 3,
		FilterJson: `{"op":"eq","field":"lang","value":{"kind":"string","str":"en"}}`,
	})
	if err != nil {
		t.Fatalf("NamedSearch filtered: %v", err)
	}
	for _, r := range fres.GetResults() {
		if r.GetId() == 2 {
			t.Fatalf("filtered search returned fr point 2: %+v", fres.GetResults())
		}
	}

	dres, err := s.NamedSearchDocs(ctx, &pb.NamedSearchRequest{Name: "docs", VectorName: "image", Query: []float32{1, 0, 0}, K: 2})
	if err != nil {
		t.Fatalf("NamedSearchDocs: %v", err)
	}
	if len(dres.GetDocuments()) == 0 || dres.GetDocuments()[0].GetMetadataJson() == "" {
		t.Fatalf("search_docs lost payload: %+v", dres.GetDocuments())
	}

	scr, err := s.NamedScroll(ctx, &pb.NamedScrollRequest{Name: "docs", Limit: 10})
	if err != nil {
		t.Fatalf("NamedScroll: %v", err)
	}
	if len(scr.GetDocuments()) != 3 {
		t.Fatalf("scroll = %d docs, want 3", len(scr.GetDocuments()))
	}

	del, err := s.NamedDelete(ctx, &pb.NamedDeleteRequest{Name: "docs", Id: 2})
	if err != nil || !del.GetDeleted() {
		t.Fatalf("NamedDelete = %v, %v", del.GetDeleted(), err)
	}
	scr, _ = s.NamedScroll(ctx, &pb.NamedScrollRequest{Name: "docs", Limit: 10})
	for _, d := range scr.GetDocuments() {
		if d.GetId() == 2 {
			t.Fatalf("scroll still returns deleted point 2")
		}
	}

	if _, err := s.NamedDrop(ctx, &pb.NamedDropRequest{Name: "docs"}); err != nil {
		t.Fatalf("NamedDrop: %v", err)
	}
}

// TestGRPCNamedSearchDispatchesArgs asserts the search edge encodes
// vector_name/query/k/filter into the dispatched vector_named_search op.
func TestGRPCNamedSearchDispatchesArgs(t *testing.T) {
	disp := &countingDispatcher{body: ops.EncodeVectorSearchResults(nil)}
	s := NewServer(disp, nil)
	_, err := s.NamedSearch(context.Background(), &pb.NamedSearchRequest{
		Name: "docs", VectorName: "title", Query: []float32{1, 2, 3, 4}, K: 7,
		FilterJson: `{"op":"eq","field":"lang","value":{"kind":"string","str":"en"}}`,
	})
	if err != nil {
		t.Fatalf("NamedSearch: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_named_search" {
		t.Fatalf("op = %q, want vector_named_search", disp.lastOp)
	}
	col, vecName, q, k, f, err := ops.DecodeNamedSearchArgs(disp.lastArg)
	if err != nil {
		t.Fatalf("decode args: %v", err)
	}
	if col != "docs" || vecName != "title" || k != 7 || len(q) != 4 {
		t.Fatalf("decoded = (%q,%q,k=%d,q=%v)", col, vecName, k, q)
	}
	if f.Op != vector.FilterEq || f.Field != "lang" {
		t.Fatalf("decoded filter = %+v, want eq/lang", f)
	}
}

// TestGRPCNamedUnknownVectorName: an unconfigured space → InvalidArgument.
func TestGRPCNamedUnknownVectorName(t *testing.T) {
	s := newRealServer(t)
	ctx := context.Background()
	if _, err := s.NamedCreate(ctx, &pb.NamedCreateRequest{Name: "docs", ConfigJson: namedConfigJSON}); err != nil {
		t.Fatalf("NamedCreate: %v", err)
	}
	_, err := s.NamedUpsert(ctx, &pb.NamedUpsertRequest{Name: "docs", Id: 1, Vectors: map[string]*pb.NamedVectorList{"nope": nvl(1, 0, 0, 0)}})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("insert unknown space code = %v, want InvalidArgument", status.Code(err))
	}
	_, err = s.NamedSearch(ctx, &pb.NamedSearchRequest{Name: "docs", VectorName: "nope", Query: []float32{1, 0, 0, 0}, K: 3})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("search unknown space code = %v, want InvalidArgument", status.Code(err))
	}
}

// TestGRPCNamedDimMismatch: a wrong-length vector → InvalidArgument.
func TestGRPCNamedDimMismatch(t *testing.T) {
	s := newRealServer(t)
	ctx := context.Background()
	if _, err := s.NamedCreate(ctx, &pb.NamedCreateRequest{Name: "docs", ConfigJson: namedConfigJSON}); err != nil {
		t.Fatalf("NamedCreate: %v", err)
	}
	_, err := s.NamedUpsert(ctx, &pb.NamedUpsertRequest{Name: "docs", Id: 1, Vectors: map[string]*pb.NamedVectorList{"title": nvl(1, 0, 0)}})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("dim mismatch code = %v, want InvalidArgument", status.Code(err))
	}
}

// TestGRPCNamedEmptyConfig: create with no spaces → InvalidArgument.
func TestGRPCNamedEmptyConfig(t *testing.T) {
	s := newRealServer(t)
	_, err := s.NamedCreate(context.Background(), &pb.NamedCreateRequest{Name: "docs", ConfigJson: `{}`})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("empty config code = %v, want InvalidArgument", status.Code(err))
	}
	// A reserved or empty PER-SPACE name is a client error → InvalidArgument, not Internal.
	for _, cfg := range []string{`{"ti#tle":{"dim":4}}`, `{"":{"dim":4}}`} {
		_, err := s.NamedCreate(context.Background(), &pb.NamedCreateRequest{Name: "docs", ConfigJson: cfg})
		if status.Code(err) != codes.InvalidArgument {
			t.Fatalf("bad space name %q code = %v, want InvalidArgument", cfg, status.Code(err))
		}
	}
}

// TestGRPCNamedReservedName: a reserved-char name is rejected at the edge,
// dispatcher untouched.
func TestGRPCNamedReservedName(t *testing.T) {
	for _, name := range []string{"bad#name", "bad@name"} {
		disp := &stubDispatcher{}
		s := NewServer(disp, nil)
		_, err := s.NamedCreate(context.Background(), &pb.NamedCreateRequest{Name: name, ConfigJson: namedConfigJSON})
		if status.Code(err) != codes.InvalidArgument {
			t.Fatalf("%q: code = %v, want InvalidArgument", name, status.Code(err))
		}
		if disp.called {
			t.Fatalf("%q: dispatcher must not be called for a rejected name", name)
		}
	}
}

// TestGRPCNamedInvalidFilterReturnsInvalidArgument: a JSON filter that fails to
// Compile is InvalidArgument BEFORE dispatch on every named filter endpoint
// (esp. scroll — never traverse with a broken predicate); dispatcher untouched.
// Mirrors TestGRPCInvalidFilter's comprehensive bad-filter matrix (bad regex,
// bad RFC3339, + the five geo failures) across all three named filter RPCs.
func TestGRPCNamedInvalidFilterReturnsInvalidArgument(t *testing.T) {
	cases := []struct {
		name   string
		filter string
	}{
		{"bad_regex", `{"op":"regex","field":"sku","value":{"kind":"string","str":"*invalid"}}`},
		{"bad_datetime", `{"op":"dt_gte","field":"created_at","value":{"kind":"string","str":"not-a-date"}}`},
		{"geo_nil", `{"op":"geo_radius","field":"loc"}`},
		{"geo_bad_polygon", `{"op":"geo_polygon","field":"loc","geo":{"polygon":[1,2,3,4]}}`},
		{"geo_out_of_range", `{"op":"geo_radius","field":"loc","geo":{"center_lat":95,"center_lon":0,"radius_m":1000}}`},
		{"geo_zero_radius", `{"op":"geo_radius","field":"loc","geo":{"center_lat":48,"center_lon":2,"radius_m":null}}`},
		{"geo_inverted_box", `{"op":"geo_bounding_box","field":"loc","geo":{"min_lat":49,"min_lon":3,"max_lat":48,"max_lon":2}}`},
	}
	for _, c := range cases {
		t.Run("search/"+c.name, func(t *testing.T) {
			disp := &stubDispatcher{}
			s := NewServer(disp, nil)
			_, err := s.NamedSearch(context.Background(), &pb.NamedSearchRequest{Name: "docs", VectorName: "title", Query: []float32{1, 0, 0, 0}, K: 1, FilterJson: c.filter})
			if status.Code(err) != codes.InvalidArgument || disp.called {
				t.Fatalf("search bad filter: code=%v called=%v, want InvalidArgument + no dispatch", status.Code(err), disp.called)
			}
		})
		t.Run("search_docs/"+c.name, func(t *testing.T) {
			disp := &stubDispatcher{}
			s := NewServer(disp, nil)
			_, err := s.NamedSearchDocs(context.Background(), &pb.NamedSearchRequest{Name: "docs", VectorName: "title", Query: []float32{1, 0, 0, 0}, K: 1, FilterJson: c.filter})
			if status.Code(err) != codes.InvalidArgument || disp.called {
				t.Fatalf("search_docs bad filter: code=%v called=%v", status.Code(err), disp.called)
			}
		})
		t.Run("scroll/"+c.name, func(t *testing.T) {
			disp := &stubDispatcher{}
			s := NewServer(disp, nil)
			_, err := s.NamedScroll(context.Background(), &pb.NamedScrollRequest{Name: "docs", Limit: 5, FilterJson: c.filter})
			if status.Code(err) != codes.InvalidArgument || disp.called {
				t.Fatalf("scroll bad filter: code=%v called=%v", status.Code(err), disp.called)
			}
		})
	}
}

// ---- get-by-id + in-place payload mutation gRPC tests ----

// mustJSON marshals a metadata map to its tagged-Value JSON for a payload_json
// field. Used by the get/payload gRPC tests.
func mustJSON(t *testing.T, m vector.Metadata) string {
	t.Helper()
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	return string(b)
}

func strVal(s string) vector.Value { return vector.Value{Kind: vector.ValueString, Str: s} }

// TestGRPCDenseGetPayload drives Get + the four payload RPCs end-to-end on a
// dense collection against a real engine: get reflects vec+payload+ttl, the
// projection flags omit a field, each payload op then reflects in a re-get, an
// absent id is NotFound, and a bad payload_json is InvalidArgument.
func TestGRPCDenseGetPayload(t *testing.T) {
	s := newRealServer(t)
	ctx := context.Background()

	if _, err := s.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name: "docs", Config: &pb.Config{Dim: 3, Metric: "l2", M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1},
	}); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	if _, err := s.Upsert(ctx, &pb.UpsertRequest{
		Collection: "docs", Id: 1, Vector: []float32{1, 2, 3}, TtlMs: 600000,
		MetadataJson: mustJSON(t, vector.Metadata{"lang": strVal("en"), "n": {Kind: vector.ValueInt, Int: 5}}),
	}); err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	// Get: both projections on (the zero request defaults to both).
	g, err := s.Get(ctx, &pb.GetRequest{Collection: "docs", Id: 1})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !g.GetFound() || len(g.GetVector()) != 3 || g.GetVector()[0] != 1 {
		t.Fatalf("Get vector = %+v", g.GetVector())
	}
	if g.GetPayloadJson() == "" || g.GetTtlMs() <= 0 {
		t.Fatalf("Get payload/ttl = %q / %d", g.GetPayloadJson(), g.GetTtlMs())
	}

	// with_payload=false omits the payload; with_vector=false omits the vector.
	gv, _ := s.Get(ctx, &pb.GetRequest{Collection: "docs", Id: 1, WithVector: true})
	if len(gv.GetVector()) != 3 || gv.GetPayloadJson() != "" {
		t.Fatalf("with_vector-only get = vec %d payload %q", len(gv.GetVector()), gv.GetPayloadJson())
	}
	gp, _ := s.Get(ctx, &pb.GetRequest{Collection: "docs", Id: 1, WithPayload: true})
	if len(gp.GetVector()) != 0 || gp.GetPayloadJson() == "" {
		t.Fatalf("with_payload-only get = vec %d payload %q", len(gp.GetVector()), gp.GetPayloadJson())
	}

	// Absent id → NotFound (the found=0 flag, not an op error).
	if _, err := s.Get(ctx, &pb.GetRequest{Collection: "docs", Id: 999}); status.Code(err) != codes.NotFound {
		t.Fatalf("Get absent = %v, want NotFound", status.Code(err))
	}

	// SetPayload merges: add "city", keep "lang".
	if _, err := s.SetPayload(ctx, &pb.SetPayloadRequest{Collection: "docs", Id: 1, PayloadJson: mustJSON(t, vector.Metadata{"city": strVal("nyc")})}); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}
	g, _ = s.Get(ctx, &pb.GetRequest{Collection: "docs", Id: 1})
	var merged vector.Metadata
	_ = json.Unmarshal([]byte(g.GetPayloadJson()), &merged)
	if merged["lang"].Str != "en" || merged["city"].Str != "nyc" {
		t.Fatalf("after merge payload = %+v", merged)
	}

	// OverwritePayload replaces the whole payload.
	if _, err := s.OverwritePayload(ctx, &pb.SetPayloadRequest{Collection: "docs", Id: 1, PayloadJson: mustJSON(t, vector.Metadata{"only": strVal("v")})}); err != nil {
		t.Fatalf("OverwritePayload: %v", err)
	}
	g, _ = s.Get(ctx, &pb.GetRequest{Collection: "docs", Id: 1})
	var over vector.Metadata
	_ = json.Unmarshal([]byte(g.GetPayloadJson()), &over)
	if _, ok := over["lang"]; ok || over["only"].Str != "v" {
		t.Fatalf("after overwrite payload = %+v", over)
	}

	// DeletePayloadKeys removes "only" → empty.
	if _, err := s.DeletePayloadKeys(ctx, &pb.DeletePayloadKeysRequest{Collection: "docs", Id: 1, Keys: []string{"only"}}); err != nil {
		t.Fatalf("DeletePayloadKeys: %v", err)
	}
	g, _ = s.Get(ctx, &pb.GetRequest{Collection: "docs", Id: 1})
	if g.GetPayloadJson() != "" {
		t.Fatalf("after delete-keys payload = %q, want empty", g.GetPayloadJson())
	}

	// ClearPayload on a re-populated point empties it.
	_, _ = s.SetPayload(ctx, &pb.SetPayloadRequest{Collection: "docs", Id: 1, PayloadJson: mustJSON(t, vector.Metadata{"x": strVal("y")})})
	if _, err := s.ClearPayload(ctx, &pb.ClearPayloadRequest{Collection: "docs", Id: 1}); err != nil {
		t.Fatalf("ClearPayload: %v", err)
	}
	g, _ = s.Get(ctx, &pb.GetRequest{Collection: "docs", Id: 1})
	if g.GetPayloadJson() != "" {
		t.Fatalf("after clear payload = %q, want empty", g.GetPayloadJson())
	}

	// Payload mutation of an absent point → NotFound.
	if _, err := s.SetPayload(ctx, &pb.SetPayloadRequest{Collection: "docs", Id: 999, PayloadJson: "{}"}); status.Code(err) != codes.NotFound {
		t.Fatalf("SetPayload absent = %v, want NotFound", status.Code(err))
	}
	if _, err := s.ClearPayload(ctx, &pb.ClearPayloadRequest{Collection: "docs", Id: 999}); status.Code(err) != codes.NotFound {
		t.Fatalf("ClearPayload absent = %v, want NotFound", status.Code(err))
	}

	// Bad payload JSON → InvalidArgument, never reaching the engine.
	if _, err := s.SetPayload(ctx, &pb.SetPayloadRequest{Collection: "docs", Id: 1, PayloadJson: "{not json"}); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("SetPayload bad json = %v, want InvalidArgument", status.Code(err))
	}
	if _, err := s.OverwritePayload(ctx, &pb.SetPayloadRequest{Collection: "docs", Id: 1, PayloadJson: "{not json"}); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("OverwritePayload bad json = %v, want InvalidArgument", status.Code(err))
	}
}

// TestGRPCSetPayloadKeyTTL drives the dense SetPayload/OverwritePayload RPCs with
// the per-key TTL map: the TTL'd key is dropped from a later Get while permanent
// keys + the point survive.
func TestGRPCSetPayloadKeyTTL(t *testing.T) {
	s := newRealServer(t)
	ctx := context.Background()

	if _, err := s.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name: "docs", Config: &pb.Config{Dim: 3, Metric: "l2", M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1},
	}); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	if _, err := s.Upsert(ctx, &pb.UpsertRequest{
		Collection: "docs", Id: 1, Vector: []float32{1, 2, 3},
		MetadataJson: mustJSON(t, vector.Metadata{"keep": {Kind: vector.ValueInt, Int: 1}}),
	}); err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	// SetPayload with a per-key TTL: temp (1ms) + perm (no TTL).
	if _, err := s.SetPayload(ctx, &pb.SetPayloadRequest{
		Collection: "docs", Id: 1,
		PayloadJson: mustJSON(t, vector.Metadata{"temp": strVal("t"), "perm": strVal("p")}),
		KeyTtlMs:    map[string]int64{"temp": 1},
	}); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}

	time.Sleep(15 * time.Millisecond)
	g, err := s.Get(ctx, &pb.GetRequest{Collection: "docs", Id: 1, WithPayload: true})
	if err != nil || !g.GetFound() {
		t.Fatalf("Get after expiry: found=%v err=%v", g.GetFound(), err)
	}
	var got vector.Metadata
	if e := json.Unmarshal([]byte(g.GetPayloadJson()), &got); e != nil {
		t.Fatalf("payload json: %v", e)
	}
	if _, ok := got["temp"]; ok {
		t.Errorf("expired key temp still present: %+v", got)
	}
	if got["keep"].Int != 1 || got["perm"].Str != "p" {
		t.Errorf("non-TTL keys dropped: %+v", got)
	}
}

// TestGRPCNamedGetPayload drives NamedGet + the four named payload RPCs.
func TestGRPCNamedGetPayload(t *testing.T) {
	s := newRealServer(t)
	ctx := context.Background()

	if _, err := s.NamedCreate(ctx, &pb.NamedCreateRequest{Name: "docs", ConfigJson: namedConfigJSON}); err != nil {
		t.Fatalf("NamedCreate: %v", err)
	}
	if _, err := s.NamedUpsert(ctx, &pb.NamedUpsertRequest{
		Name: "docs", Id: 1,
		Vectors:      map[string]*pb.NamedVectorList{"title": nvl(1, 0, 0, 0), "image": nvl(1, 0, 0)},
		MetadataJson: mustJSON(t, vector.Metadata{"lang": strVal("en")}),
	}); err != nil {
		t.Fatalf("NamedUpsert: %v", err)
	}

	g, err := s.NamedGet(ctx, &pb.NamedGetRequest{Collection: "docs", Id: 1})
	if err != nil {
		t.Fatalf("NamedGet: %v", err)
	}
	if !g.GetFound() || len(g.GetVectors()) != 2 || len(g.GetVectors()["title"].GetValues()) != 4 || g.GetPayloadJson() == "" {
		t.Fatalf("NamedGet = found=%v vectors=%+v payload=%q", g.GetFound(), g.GetVectors(), g.GetPayloadJson())
	}

	// with_vector=false omits the named vectors.
	gp, _ := s.NamedGet(ctx, &pb.NamedGetRequest{Collection: "docs", Id: 1, WithPayload: true})
	if len(gp.GetVectors()) != 0 || gp.GetPayloadJson() == "" {
		t.Fatalf("with_payload-only named get = vectors %d payload %q", len(gp.GetVectors()), gp.GetPayloadJson())
	}

	if _, err := s.NamedGet(ctx, &pb.NamedGetRequest{Collection: "docs", Id: 999}); status.Code(err) != codes.NotFound {
		t.Fatalf("NamedGet absent = %v, want NotFound", status.Code(err))
	}

	// merge → overwrite → delete-keys → clear, re-getting each time.
	_, _ = s.NamedSetPayload(ctx, &pb.SetPayloadRequest{Collection: "docs", Id: 1, PayloadJson: mustJSON(t, vector.Metadata{"city": strVal("nyc")})})
	g, _ = s.NamedGet(ctx, &pb.NamedGetRequest{Collection: "docs", Id: 1})
	var merged vector.Metadata
	_ = json.Unmarshal([]byte(g.GetPayloadJson()), &merged)
	if merged["lang"].Str != "en" || merged["city"].Str != "nyc" {
		t.Fatalf("named merge = %+v", merged)
	}
	_, _ = s.NamedOverwritePayload(ctx, &pb.SetPayloadRequest{Collection: "docs", Id: 1, PayloadJson: mustJSON(t, vector.Metadata{"only": strVal("v")})})
	_, _ = s.NamedDeletePayloadKeys(ctx, &pb.DeletePayloadKeysRequest{Collection: "docs", Id: 1, Keys: []string{"only"}})
	g, _ = s.NamedGet(ctx, &pb.NamedGetRequest{Collection: "docs", Id: 1})
	if g.GetPayloadJson() != "" {
		t.Fatalf("named after delete-keys = %q, want empty", g.GetPayloadJson())
	}
	_, _ = s.NamedSetPayload(ctx, &pb.SetPayloadRequest{Collection: "docs", Id: 1, PayloadJson: mustJSON(t, vector.Metadata{"x": strVal("y")})})
	_, _ = s.NamedClearPayload(ctx, &pb.ClearPayloadRequest{Collection: "docs", Id: 1})
	g, _ = s.NamedGet(ctx, &pb.NamedGetRequest{Collection: "docs", Id: 1})
	if g.GetPayloadJson() != "" {
		t.Fatalf("named after clear = %q, want empty", g.GetPayloadJson())
	}

	if _, err := s.NamedSetPayload(ctx, &pb.SetPayloadRequest{Collection: "docs", Id: 999, PayloadJson: "{}"}); status.Code(err) != codes.NotFound {
		t.Fatalf("NamedSetPayload absent = %v, want NotFound", status.Code(err))
	}
	if _, err := s.NamedSetPayload(ctx, &pb.SetPayloadRequest{Collection: "docs", Id: 1, PayloadJson: "{bad"}); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("NamedSetPayload bad json = %v, want InvalidArgument", status.Code(err))
	}
}

// TestGRPCMVGetPayload drives MVGet + the four MV payload RPCs.
func TestGRPCMVGetPayload(t *testing.T) {
	s := newRealServer(t)
	ctx := context.Background()

	if _, err := s.MVCreateCollection(ctx, &pb.MVCreateRequest{Name: "docs", Config: &pb.MVConfig{Dim: 3, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1}}); err != nil {
		t.Fatalf("MVCreateCollection: %v", err)
	}
	tok := func(vals ...float32) *pb.TokenVector { return &pb.TokenVector{Values: vals} }
	if _, err := s.MVAdd(ctx, &pb.MVAddRequest{
		Name: "docs", Id: 1, Tokens: []*pb.TokenVector{tok(1, 0, 0), tok(0, 1, 0)},
		MetadataJson: mustJSON(t, vector.Metadata{"lang": strVal("en")}),
	}); err != nil {
		t.Fatalf("MVAdd: %v", err)
	}

	g, err := s.MVGet(ctx, &pb.MVGetRequest{Collection: "docs", Id: 1})
	if err != nil {
		t.Fatalf("MVGet: %v", err)
	}
	if !g.GetFound() || len(g.GetTokens()) != 2 || len(g.GetTokens()[0].GetValues()) != 3 || g.GetPayloadJson() == "" {
		t.Fatalf("MVGet = found=%v tokens=%+v payload=%q", g.GetFound(), g.GetTokens(), g.GetPayloadJson())
	}

	gp, _ := s.MVGet(ctx, &pb.MVGetRequest{Collection: "docs", Id: 1, WithPayload: true})
	if len(gp.GetTokens()) != 0 || gp.GetPayloadJson() == "" {
		t.Fatalf("with_payload-only mv get = tokens %d payload %q", len(gp.GetTokens()), gp.GetPayloadJson())
	}

	if _, err := s.MVGet(ctx, &pb.MVGetRequest{Collection: "docs", Id: 999}); status.Code(err) != codes.NotFound {
		t.Fatalf("MVGet absent = %v, want NotFound", status.Code(err))
	}

	_, _ = s.MVSetPayload(ctx, &pb.SetPayloadRequest{Collection: "docs", Id: 1, PayloadJson: mustJSON(t, vector.Metadata{"city": strVal("nyc")})})
	g, _ = s.MVGet(ctx, &pb.MVGetRequest{Collection: "docs", Id: 1})
	var merged vector.Metadata
	_ = json.Unmarshal([]byte(g.GetPayloadJson()), &merged)
	if merged["lang"].Str != "en" || merged["city"].Str != "nyc" {
		t.Fatalf("mv merge = %+v", merged)
	}
	_, _ = s.MVOverwritePayload(ctx, &pb.SetPayloadRequest{Collection: "docs", Id: 1, PayloadJson: mustJSON(t, vector.Metadata{"only": strVal("v")})})
	_, _ = s.MVDeletePayloadKeys(ctx, &pb.DeletePayloadKeysRequest{Collection: "docs", Id: 1, Keys: []string{"only"}})
	g, _ = s.MVGet(ctx, &pb.MVGetRequest{Collection: "docs", Id: 1})
	if g.GetPayloadJson() != "" {
		t.Fatalf("mv after delete-keys = %q, want empty", g.GetPayloadJson())
	}
	_, _ = s.MVSetPayload(ctx, &pb.SetPayloadRequest{Collection: "docs", Id: 1, PayloadJson: mustJSON(t, vector.Metadata{"x": strVal("y")})})
	_, _ = s.MVClearPayload(ctx, &pb.ClearPayloadRequest{Collection: "docs", Id: 1})
	g, _ = s.MVGet(ctx, &pb.MVGetRequest{Collection: "docs", Id: 1})
	if g.GetPayloadJson() != "" {
		t.Fatalf("mv after clear = %q, want empty", g.GetPayloadJson())
	}

	if _, err := s.MVSetPayload(ctx, &pb.SetPayloadRequest{Collection: "docs", Id: 999, PayloadJson: "{}"}); status.Code(err) != codes.NotFound {
		t.Fatalf("MVSetPayload absent = %v, want NotFound", status.Code(err))
	}
	if _, err := s.MVSetPayload(ctx, &pb.SetPayloadRequest{Collection: "docs", Id: 1, PayloadJson: "{bad"}); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("MVSetPayload bad json = %v, want InvalidArgument", status.Code(err))
	}
}

// --- collection aliases ---

// errAliasDispatcher returns a fixed error so the alias error->status mapping can
// be exercised without the engine.
type errAliasDispatcher struct{ err error }

func (d *errAliasDispatcher) Call(string, []byte) ([]byte, error) { return nil, d.err }
func (d *errAliasDispatcher) LeaderAddr() string                  { return "" }

func TestGRPCCreateAlias(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	resp, err := s.CreateAlias(context.Background(), &pb.CreateAliasRequest{Alias: "prod", Collection: "docs"})
	if err != nil {
		t.Fatalf("CreateAlias: %v", err)
	}
	if !disp.called || disp.lastOp != "alias_batch" {
		t.Fatalf("dispatch op = %q, want alias_batch", disp.lastOp)
	}
	actions, err := ops.DecodeAliasBatchArgs(disp.lastArg)
	if err != nil {
		t.Fatalf("DecodeAliasBatchArgs: %v", err)
	}
	if len(actions) != 1 || actions[0].Alias != "prod" || actions[0].Canonical != "docs" || actions[0].Delete {
		t.Fatalf("actions = %+v, want [{prod docs false}]", actions)
	}
	if resp.GetAlias() != "prod" || resp.GetCollection() != "docs" {
		t.Fatalf("resp = (%q, %q), want (prod, docs)", resp.GetAlias(), resp.GetCollection())
	}
}

func TestGRPCCreateAliasRejectsEmpty(t *testing.T) {
	for _, req := range []*pb.CreateAliasRequest{
		{Alias: "", Collection: "docs"},
		{Alias: "prod", Collection: ""},
	} {
		disp := &stubDispatcher{}
		s := NewServer(disp, nil)
		_, err := s.CreateAlias(context.Background(), req)
		if status.Code(err) != codes.InvalidArgument {
			t.Fatalf("req %+v: code = %v, want InvalidArgument", req, status.Code(err))
		}
		if disp.called {
			t.Fatalf("req %+v: dispatcher must not be called", req)
		}
	}
}

func TestGRPCDeleteAlias(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	resp, err := s.DeleteAlias(context.Background(), &pb.DeleteAliasRequest{Alias: "prod"})
	if err != nil {
		t.Fatalf("DeleteAlias: %v", err)
	}
	if !disp.called || disp.lastOp != "alias_batch" {
		t.Fatalf("dispatch op = %q, want alias_batch", disp.lastOp)
	}
	actions, err := ops.DecodeAliasBatchArgs(disp.lastArg)
	if err != nil {
		t.Fatalf("DecodeAliasBatchArgs: %v", err)
	}
	if len(actions) != 1 || actions[0].Alias != "prod" || !actions[0].Delete {
		t.Fatalf("actions = %+v, want [{prod  true}]", actions)
	}
	if resp.GetAlias() != "prod" {
		t.Fatalf("resp = %q, want prod", resp.GetAlias())
	}
}

func TestGRPCListAliases(t *testing.T) {
	disp := &countingDispatcher{body: ops.EncodeAliasListResult([]ops.AliasEntry{
		{Alias: "prod", Collection: "docs"},
		{Alias: "stage", Collection: "docs"},
	})}
	s := NewServer(disp, nil)
	resp, err := s.ListAliases(context.Background(), &pb.ListAliasesRequest{Collection: "docs"})
	if err != nil {
		t.Fatalf("ListAliases: %v", err)
	}
	if !disp.called || disp.lastOp != "alias_list" {
		t.Fatalf("dispatch op = %q, want alias_list", disp.lastOp)
	}
	coll, err := ops.DecodeAliasListArgs(disp.lastArg)
	if err != nil || coll != "docs" {
		t.Fatalf("decode list args = (%q, %v), want (docs, nil)", coll, err)
	}
	if len(resp.GetAliases()) != 2 {
		t.Fatalf("aliases = %+v, want 2 entries", resp.GetAliases())
	}
}

// TestGRPCAliasBatchSwap proves the atomic-swap request builds ONE alias_batch
// carrying [{delete prod},{create prod->docs2}].
func TestGRPCAliasBatchSwap(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	resp, err := s.AliasBatch(context.Background(), &pb.AliasBatchRequest{Actions: []*pb.AliasAction{
		{Alias: "prod", Delete: true},
		{Alias: "prod", Collection: "docs2"},
	}})
	if err != nil {
		t.Fatalf("AliasBatch: %v", err)
	}
	if !disp.called || disp.lastOp != "alias_batch" {
		t.Fatalf("dispatch op = %q, want a single alias_batch", disp.lastOp)
	}
	if disp.callCount != 1 {
		t.Fatalf("dispatch count = %d, want exactly 1 (swap must be one atomic op)", disp.callCount)
	}
	actions, err := ops.DecodeAliasBatchArgs(disp.lastArg)
	if err != nil {
		t.Fatalf("DecodeAliasBatchArgs: %v", err)
	}
	if len(actions) != 2 {
		t.Fatalf("actions = %+v, want 2 (atomic swap)", actions)
	}
	if actions[0].Alias != "prod" || !actions[0].Delete {
		t.Fatalf("action[0] = %+v, want {prod delete}", actions[0])
	}
	if actions[1].Alias != "prod" || actions[1].Canonical != "docs2" || actions[1].Delete {
		t.Fatalf("action[1] = %+v, want {prod->docs2 create}", actions[1])
	}
	if resp.GetApplied() != 2 {
		t.Fatalf("applied = %d, want 2", resp.GetApplied())
	}
}

func TestGRPCAliasBatchRejectsBadAction(t *testing.T) {
	for _, req := range []*pb.AliasBatchRequest{
		{Actions: []*pb.AliasAction{{Alias: "", Collection: "c"}}},
		{Actions: []*pb.AliasAction{{Alias: "a"}}}, // create with empty collection
	} {
		disp := &stubDispatcher{}
		s := NewServer(disp, nil)
		_, err := s.AliasBatch(context.Background(), req)
		if status.Code(err) != codes.InvalidArgument {
			t.Fatalf("req %+v: code = %v, want InvalidArgument", req, status.Code(err))
		}
		if disp.called {
			t.Fatalf("req %+v: dispatcher must not be called", req)
		}
	}
}

// TestGRPCAliasValidationMapsToInvalidArgument confirms the four alias-validation
// sentinels (all carrying the "rostam: alias " prefix) map to InvalidArgument.
func TestGRPCAliasValidationMapsToInvalidArgument(t *testing.T) {
	for _, msg := range []string{
		"alias \"prod\" → \"missing\": rostam: alias target collection does not exist",
		"alias \"docs\": rostam: alias name shadows an existing collection",
		"alias \"prod\" → \"a\": rostam: alias target is itself an alias",
		"alias \"bad#x\": rostam: alias name must not contain reserved characters '#' or '@'",
	} {
		disp := &errAliasDispatcher{err: fmt.Errorf("%s", msg)}
		s := NewServer(disp, nil)
		_, err := s.CreateAlias(context.Background(), &pb.CreateAliasRequest{Alias: "prod", Collection: "docs"})
		if status.Code(err) != codes.InvalidArgument {
			t.Fatalf("validation error %q: code = %v, want InvalidArgument", msg, status.Code(err))
		}
	}
}

// TestGRPCErrorWriteConsistency proves the write-consistency barrier-miss error
// (carried as a "cluster: write " message prefix, constructed directly like the
// alias-mapping tests) maps to codes.FailedPrecondition — the durable-but-under-
// replicated outcome — and not to Unavailable (the "not leader" bucket) or the
// Internal default.
func TestGRPCErrorWriteConsistency(t *testing.T) {
	err := fmt.Errorf("cluster: write committed at quorum but consistency factor 3 not met (2 replicas applied at index 42)")
	if got := status.Code(grpcError(err)); got != codes.FailedPrecondition {
		t.Fatalf("grpcError(write-consistency) = %v, want FailedPrecondition", got)
	}
	// A generic error still maps to Internal (no accidental collision).
	if got := status.Code(grpcError(fmt.Errorf("boom"))); got != codes.Internal {
		t.Fatalf("grpcError(generic) = %v, want Internal", got)
	}
}

// ---- named read consistency (read_consistency / on_partition_unavailable) ----

// TestNamedSearchConsistencyReachesArgs proves NamedSearch threads rc/opa into
// the *Opts-encoded dispatch args (Linearizable=2 / Fail=1).
func TestNamedSearchConsistencyReachesArgs(t *testing.T) {
	disp := &countingDispatcher{body: ops.EncodeVectorSearchResults(nil)}
	s := NewServer(disp, nil)
	_, err := s.NamedSearch(context.Background(), &pb.NamedSearchRequest{
		Name: "col", VectorName: "v", K: 1, Query: []float32{1, 2},
		ReadConsistency: 2, OnPartitionUnavailable: 1,
	})
	if err != nil {
		t.Fatalf("NamedSearch: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_named_search" {
		t.Fatalf("dispatch op = %q, want vector_named_search", disp.lastOp)
	}
	_, _, _, _, _, rc, opa, _, derr := ops.DecodeNamedSearchArgsOpts(disp.lastArg)
	if derr != nil {
		t.Fatalf("DecodeNamedSearchArgsOpts: %v", derr)
	}
	if rc != 2 || opa != 1 {
		t.Fatalf("decoded rc/opa = %d/%d, want 2/1", rc, opa)
	}
}

// TestNamedSearchDocsConsistencyReachesArgs proves NamedSearchDocs threads rc/opa
// into the *Opts-encoded dispatch args.
func TestNamedSearchDocsConsistencyReachesArgs(t *testing.T) {
	disp := &countingDispatcher{body: ops.EncodeVectorDocs(nil)}
	s := NewServer(disp, nil)
	_, err := s.NamedSearchDocs(context.Background(), &pb.NamedSearchRequest{
		Name: "col", VectorName: "v", K: 1, Query: []float32{1, 2},
		ReadConsistency: 2, OnPartitionUnavailable: 1,
	})
	if err != nil {
		t.Fatalf("NamedSearchDocs: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_named_search_docs" {
		t.Fatalf("dispatch op = %q, want vector_named_search_docs", disp.lastOp)
	}
	_, _, _, _, _, rc, opa, _, derr := ops.DecodeNamedSearchArgsOpts(disp.lastArg)
	if derr != nil {
		t.Fatalf("DecodeNamedSearchArgsOpts: %v", derr)
	}
	if rc != 2 || opa != 1 {
		t.Fatalf("decoded rc/opa = %d/%d, want 2/1", rc, opa)
	}
}

// TestNamedScrollConsistencyReachesArgs proves NamedScroll threads rc/opa into the
// *Opts-encoded dispatch args.
func TestNamedScrollConsistencyReachesArgs(t *testing.T) {
	disp := &countingDispatcher{body: ops.EncodeScrollResult(nil, false, nil, "")}
	s := NewServer(disp, nil)
	_, err := s.NamedScroll(context.Background(), &pb.NamedScrollRequest{
		Name: "col", Limit: 10,
		ReadConsistency: 2, OnPartitionUnavailable: 1,
	})
	if err != nil {
		t.Fatalf("NamedScroll: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_named_scroll" {
		t.Fatalf("dispatch op = %q, want vector_named_scroll", disp.lastOp)
	}
	_, _, _, _, _, rc, opa, _, derr := ops.DecodeNamedScrollArgsOpts(disp.lastArg)
	if derr != nil {
		t.Fatalf("DecodeNamedScrollArgsOpts: %v", derr)
	}
	if rc != 2 || opa != 1 {
		t.Fatalf("decoded rc/opa = %d/%d, want 2/1", rc, opa)
	}
}

// TestGRPCMVScrollPagesWithCursor proves MVScroll dispatches vector_mv_scroll
// with the cursor + rc/opa threaded into the args, and decodes the dispatcher's
// server-authoritative result (documents + next_cursor + degraded/missing) back
// into the response. Mirrors the dense Scroll handler over the MV op family.
func TestGRPCMVScrollPagesWithCursor(t *testing.T) {
	docs := []vector.Document{{ID: 7, Content: "a"}, {ID: 9, Content: "b"}}
	disp := &countingDispatcher{body: ops.EncodeScrollResult(docs, true, []uint16{3}, ops.EncodeScrollCursor(9))}
	s := NewServer(disp, nil)
	resp, err := s.MVScroll(context.Background(), &pb.MVScrollRequest{
		Collection: "mv", Limit: 2, Cursor: ops.EncodeScrollCursor(5),
		ReadConsistency: 2, OnPartitionUnavailable: 1,
	})
	if err != nil {
		t.Fatalf("MVScroll: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_mv_scroll" {
		t.Fatalf("dispatch op = %q, want vector_mv_scroll", disp.lastOp)
	}
	// rc/opa + cursor reach the dispatched args.
	_, _, _, rc, opa, afterID, hasAfter, _, derr := ops.DecodeMVScrollArgsOpts(disp.lastArg)
	if derr != nil {
		t.Fatalf("DecodeMVScrollArgsOpts: %v", derr)
	}
	if rc != 2 || opa != 1 {
		t.Fatalf("decoded rc/opa = %d/%d, want 2/1", rc, opa)
	}
	if !hasAfter || afterID != 5 {
		t.Fatalf("decoded cursor afterID = %d (has=%v), want 5/true", afterID, hasAfter)
	}
	// Response carries the server-authoritative next_cursor + docs + fan-out health.
	if resp.GetNextCursor() != ops.EncodeScrollCursor(9) {
		t.Fatalf("NextCursor = %q, want cursor(9)", resp.GetNextCursor())
	}
	if len(resp.GetDocuments()) != 2 || resp.GetDocuments()[0].GetId() != 7 || resp.GetDocuments()[1].GetId() != 9 {
		t.Fatalf("documents = %+v, want ids [7 9]", resp.GetDocuments())
	}
	if !resp.GetDegraded() || len(resp.GetMissing()) != 1 || resp.GetMissing()[0] != 3 {
		t.Fatalf("degraded/missing = %v/%v, want true/[3]", resp.GetDegraded(), resp.GetMissing())
	}
}

// TestGRPCMVScrollRejectsBadConsistency proves an out-of-range read_consistency
// (3, above Linearizable=2) is rejected with InvalidArgument BEFORE dispatch.
func TestGRPCMVScrollRejectsBadConsistency(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	_, err := s.MVScroll(context.Background(), &pb.MVScrollRequest{Collection: "mv", Limit: 5, ReadConsistency: 4})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("code = %v, want InvalidArgument", status.Code(err))
	}
	if disp.called {
		t.Fatalf("dispatcher must not be called for a rejected read_consistency")
	}
}

// TestGRPCMVScrollRejectsBadCursor proves a malformed cursor is rejected with
// InvalidArgument BEFORE dispatch (client error, never reaches the store).
func TestGRPCMVScrollRejectsBadCursor(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	_, err := s.MVScroll(context.Background(), &pb.MVScrollRequest{Collection: "mv", Limit: 5, Cursor: "not-a-valid-cursor!!"})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("code = %v, want InvalidArgument", status.Code(err))
	}
	if disp.called {
		t.Fatalf("dispatcher must not be called for a malformed cursor")
	}
}

// TestNamedSearchNoConsistencyUnchanged proves a no-rc NamedSearch dispatches the
// byte-identical legacy (no-trailer) args (decodes cleanly via the plain decoder).
func TestNamedSearchNoConsistencyUnchanged(t *testing.T) {
	disp := &countingDispatcher{body: ops.EncodeVectorSearchResults(nil)}
	s := NewServer(disp, nil)
	_, err := s.NamedSearch(context.Background(), &pb.NamedSearchRequest{
		Name: "col", VectorName: "v", K: 1, Query: []float32{1, 2},
	})
	if err != nil {
		t.Fatalf("NamedSearch: %v", err)
	}
	want := ops.EncodeNamedSearchArgs("col", "v", []float32{1, 2}, 1, vector.Filter{})
	if !bytes.Equal(disp.lastArg, want) {
		t.Fatalf("no-rc named search args differ from legacy encoding")
	}
}

// TestNamedReadsRejectOutOfRangeConsistency proves an out-of-range read_consistency
// (3, above Linearizable=2) is rejected with InvalidArgument before dispatch on
// every named read family.
func TestNamedReadsRejectOutOfRangeConsistency(t *testing.T) {
	cases := []struct {
		name string
		call func(s *Server) error
	}{
		{"NamedSearch rc=3", func(s *Server) error {
			_, e := s.NamedSearch(context.Background(), &pb.NamedSearchRequest{Name: "col", VectorName: "v", K: 1, Query: []float32{1, 2}, ReadConsistency: 4})
			return e
		}},
		{"NamedSearchDocs rc=3", func(s *Server) error {
			_, e := s.NamedSearchDocs(context.Background(), &pb.NamedSearchRequest{Name: "col", VectorName: "v", K: 1, Query: []float32{1, 2}, ReadConsistency: 4})
			return e
		}},
		{"NamedScroll rc=3", func(s *Server) error {
			_, e := s.NamedScroll(context.Background(), &pb.NamedScrollRequest{Name: "col", Limit: 10, ReadConsistency: 4})
			return e
		}},
		{"NamedSearch opa=2", func(s *Server) error {
			_, e := s.NamedSearch(context.Background(), &pb.NamedSearchRequest{Name: "col", VectorName: "v", K: 1, Query: []float32{1, 2}, OnPartitionUnavailable: 2})
			return e
		}},
	}
	for _, tc := range cases {
		disp := &stubDispatcher{}
		s := NewServer(disp, nil)
		err := tc.call(s)
		if status.Code(err) != codes.InvalidArgument {
			t.Fatalf("%s: code = %v, want InvalidArgument", tc.name, status.Code(err))
		}
		if disp.called {
			t.Fatalf("%s: dispatcher must not be called for out-of-range consistency", tc.name)
		}
	}
}

// TestGRPCGetBatch drives the GetBatch RPC against a recording dispatcher whose
// reply encodes a mixed found/not-found result. It asserts (1) the edge dispatches
// vector_get_batch with the requested ids+flags, (2) a partial miss is returned as
// the missing list — NOT a NotFound error (the property that distinguishes batch
// from single Get), and (3) found rows become BatchGetPoints carrying their id +
// projection. Empty ids → empty response (no points, no missing).
func TestGRPCGetBatch(t *testing.T) {
	// Reply: id 1 found (vec+payload), id 2 missing, id 3 found.
	reply := ops.EncodeVectorGetBatchResult([]ops.GetBatchRow{
		{ID: 1, Found: true, Vec: []float32{1, 2}, Meta: vector.Metadata{"lang": strVal("en")}, TTLMs: 5000},
		{ID: 2, Found: false},
		{ID: 3, Found: true, Vec: []float32{3, 4}},
	})
	disp := &countingDispatcher{body: reply}
	s := NewServer(disp, nil)

	resp, err := s.GetBatch(context.Background(), &pb.GetBatchRequest{
		Collection: "docs", Ids: []uint64{1, 2, 3}, WithVector: true, WithPayload: true,
	})
	if err != nil {
		t.Fatalf("GetBatch: %v (partial miss must NOT be an error)", err)
	}
	if status.Code(err) == codes.NotFound {
		t.Fatalf("GetBatch returned NotFound on partial miss — batch must return the missing list")
	}
	if disp.lastOp != "vector_get_batch" {
		t.Fatalf("dispatch op = %q, want vector_get_batch", disp.lastOp)
	}
	// Verify the encoded args carry the requested ids + flags.
	coll, ids, flags, derr := ops.DecodeVectorGetBatchArgs(disp.lastArg)
	if derr != nil {
		t.Fatalf("decode dispatched args: %v", derr)
	}
	if coll != "docs" || len(ids) != 3 || ids[0] != 1 || ids[1] != 2 || ids[2] != 3 {
		t.Fatalf("dispatched coll=%q ids=%v, want docs [1 2 3]", coll, ids)
	}
	if flags&ops.GetFlagWithVector == 0 || flags&ops.GetFlagWithPayload == 0 {
		t.Fatalf("flags=%#x, want both projections on", flags)
	}
	// Found points: ids 1 and 3.
	if len(resp.GetPoints()) != 2 {
		t.Fatalf("points=%d, want 2", len(resp.GetPoints()))
	}
	if resp.GetPoints()[0].GetId() != 1 || resp.GetPoints()[1].GetId() != 3 {
		t.Fatalf("point ids = %d,%d, want 1,3", resp.GetPoints()[0].GetId(), resp.GetPoints()[1].GetId())
	}
	if got := resp.GetPoints()[0].GetVector(); len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("point 1 vector = %v, want [1 2]", got)
	}
	if resp.GetPoints()[0].GetTtlMs() != 5000 {
		t.Fatalf("point 1 ttl_ms = %d, want 5000", resp.GetPoints()[0].GetTtlMs())
	}
	if resp.GetPoints()[0].GetPayloadJson() == "" {
		t.Fatalf("point 1 payload_json empty, want lang=en")
	}
	// Missing: id 2.
	if len(resp.GetMissing()) != 1 || resp.GetMissing()[0] != 2 {
		t.Fatalf("missing = %v, want [2]", resp.GetMissing())
	}
}

// TestGRPCGetBatchEmpty asserts an empty id list yields an empty response (no
// points, no missing) — never an error.
func TestGRPCGetBatchEmpty(t *testing.T) {
	disp := &countingDispatcher{body: ops.EncodeVectorGetBatchResult(nil)}
	s := NewServer(disp, nil)
	resp, err := s.GetBatch(context.Background(), &pb.GetBatchRequest{Collection: "docs", Ids: nil})
	if err != nil {
		t.Fatalf("GetBatch empty: %v", err)
	}
	if len(resp.GetPoints()) != 0 || len(resp.GetMissing()) != 0 {
		t.Fatalf("empty batch: points=%d missing=%d, want 0/0", len(resp.GetPoints()), len(resp.GetMissing()))
	}
}

// TestGRPCNamedGetBatch drives the NamedGetBatch RPC against a recording
// dispatcher whose reply encodes a mixed found/not-found named result. Asserts
// (1) it dispatches vector_named_get_batch with the requested ids+flags (args
// reuse the dense batch codec), (2) a partial miss is the missing list — NOT a
// NotFound error, and (3) found rows become NamedBatchGetPoints carrying their id
// + per-space vectors map + payload + ttl. The named clone of TestGRPCGetBatch.
func TestGRPCNamedGetBatch(t *testing.T) {
	reply := ops.EncodeNamedGetBatchResult([]ops.NamedGetBatchRow{
		{ID: 1, Found: true, Vectors: map[string][]float32{"title": {1, 2}, "image": {3, 4}}, Meta: vector.Metadata{"lang": strVal("en")}, TTLMs: 5000},
		{ID: 2, Found: false},
		{ID: 3, Found: true, Vectors: map[string][]float32{"title": {5, 6}}},
	})
	disp := &countingDispatcher{body: reply}
	s := NewServer(disp, nil)

	resp, err := s.NamedGetBatch(context.Background(), &pb.NamedGetBatchRequest{
		Collection: "docs", Ids: []uint64{1, 2, 3}, WithVector: true, WithPayload: true,
	})
	if err != nil {
		t.Fatalf("NamedGetBatch: %v (partial miss must NOT be an error)", err)
	}
	if status.Code(err) == codes.NotFound {
		t.Fatalf("NamedGetBatch returned NotFound on partial miss")
	}
	if disp.lastOp != "vector_named_get_batch" {
		t.Fatalf("dispatch op = %q, want vector_named_get_batch", disp.lastOp)
	}
	coll, ids, flags, derr := ops.DecodeVectorGetBatchArgs(disp.lastArg)
	if derr != nil {
		t.Fatalf("decode dispatched args: %v", derr)
	}
	if coll != "docs" || len(ids) != 3 || ids[0] != 1 || ids[2] != 3 {
		t.Fatalf("dispatched coll=%q ids=%v, want docs [1 2 3]", coll, ids)
	}
	if flags&ops.GetFlagWithVector == 0 || flags&ops.GetFlagWithPayload == 0 {
		t.Fatalf("flags=%#x, want both projections on", flags)
	}
	if len(resp.GetPoints()) != 2 {
		t.Fatalf("points=%d, want 2", len(resp.GetPoints()))
	}
	if resp.GetPoints()[0].GetId() != 1 || resp.GetPoints()[1].GetId() != 3 {
		t.Fatalf("point ids = %d,%d, want 1,3", resp.GetPoints()[0].GetId(), resp.GetPoints()[1].GetId())
	}
	if title := resp.GetPoints()[0].GetVectors()["title"].GetValues(); len(title) != 2 || title[0] != 1 || title[1] != 2 {
		t.Fatalf("point 1 title vector = %v, want [1 2]", title)
	}
	if resp.GetPoints()[0].GetTtlMs() != 5000 || resp.GetPoints()[0].GetPayloadJson() == "" {
		t.Fatalf("point 1 ttl/payload = %d/%q", resp.GetPoints()[0].GetTtlMs(), resp.GetPoints()[0].GetPayloadJson())
	}
	if len(resp.GetMissing()) != 1 || resp.GetMissing()[0] != 2 {
		t.Fatalf("missing = %v, want [2]", resp.GetMissing())
	}

	// empty ids -> empty response, no error.
	disp2 := &countingDispatcher{body: ops.EncodeNamedGetBatchResult(nil)}
	s2 := NewServer(disp2, nil)
	r2, err := s2.NamedGetBatch(context.Background(), &pb.NamedGetBatchRequest{Collection: "docs", Ids: nil})
	if err != nil {
		t.Fatalf("NamedGetBatch empty: %v", err)
	}
	if len(r2.GetPoints()) != 0 || len(r2.GetMissing()) != 0 {
		t.Fatalf("empty: points=%d missing=%d, want 0/0", len(r2.GetPoints()), len(r2.GetMissing()))
	}
}

// TestGRPCMVGetBatch drives the MVGetBatch RPC against a recording dispatcher
// whose reply encodes a mixed found/not-found MV result. Asserts (1) it
// dispatches vector_mv_get_batch with the requested ids+flags (args reuse the
// dense batch codec), (2) a partial miss is the missing list — NOT a NotFound
// error, and (3) found rows become MVBatchGetPoints carrying their id + token
// matrix + payload (MV has NO ttl). The MV clone of TestGRPCNamedGetBatch.
func TestGRPCMVGetBatch(t *testing.T) {
	reply := ops.EncodeMVGetBatchResult([]ops.MVGetBatchRow{
		{ID: 1, Found: true, Tokens: [][]float32{{1, 2}, {3, 4}}, Meta: vector.Metadata{"lang": strVal("en")}},
		{ID: 2, Found: false},
		{ID: 3, Found: true, Tokens: [][]float32{{5, 6}}},
	})
	disp := &countingDispatcher{body: reply}
	s := NewServer(disp, nil)

	resp, err := s.MVGetBatch(context.Background(), &pb.MVGetBatchRequest{
		Collection: "docs", Ids: []uint64{1, 2, 3}, WithVector: true, WithPayload: true,
	})
	if err != nil {
		t.Fatalf("MVGetBatch: %v (partial miss must NOT be an error)", err)
	}
	if status.Code(err) == codes.NotFound {
		t.Fatalf("MVGetBatch returned NotFound on partial miss")
	}
	if disp.lastOp != "vector_mv_get_batch" {
		t.Fatalf("dispatch op = %q, want vector_mv_get_batch", disp.lastOp)
	}
	coll, ids, flags, derr := ops.DecodeVectorGetBatchArgs(disp.lastArg)
	if derr != nil {
		t.Fatalf("decode dispatched args: %v", derr)
	}
	if coll != "docs" || len(ids) != 3 || ids[0] != 1 || ids[2] != 3 {
		t.Fatalf("dispatched coll=%q ids=%v, want docs [1 2 3]", coll, ids)
	}
	if flags&ops.GetFlagWithVector == 0 || flags&ops.GetFlagWithPayload == 0 {
		t.Fatalf("flags=%#x, want both projections on", flags)
	}
	if len(resp.GetPoints()) != 2 {
		t.Fatalf("points=%d, want 2", len(resp.GetPoints()))
	}
	if resp.GetPoints()[0].GetId() != 1 || resp.GetPoints()[1].GetId() != 3 {
		t.Fatalf("point ids = %d,%d, want 1,3", resp.GetPoints()[0].GetId(), resp.GetPoints()[1].GetId())
	}
	if toks := resp.GetPoints()[0].GetTokens(); len(toks) != 2 || len(toks[0].GetValues()) != 2 || toks[0].GetValues()[0] != 1 {
		t.Fatalf("point 1 tokens = %v, want [[1 2] [3 4]]", toks)
	}
	if resp.GetPoints()[0].GetPayloadJson() == "" {
		t.Fatalf("point 1 payload = %q, want non-empty", resp.GetPoints()[0].GetPayloadJson())
	}
	if len(resp.GetMissing()) != 1 || resp.GetMissing()[0] != 2 {
		t.Fatalf("missing = %v, want [2]", resp.GetMissing())
	}

	// empty ids -> empty response, no error.
	disp2 := &countingDispatcher{body: ops.EncodeMVGetBatchResult(nil)}
	s2 := NewServer(disp2, nil)
	r2, err := s2.MVGetBatch(context.Background(), &pb.MVGetBatchRequest{Collection: "docs", Ids: nil})
	if err != nil {
		t.Fatalf("MVGetBatch empty: %v", err)
	}
	if len(r2.GetPoints()) != 0 || len(r2.GetMissing()) != 0 {
		t.Fatalf("empty: points=%d missing=%d, want 0/0", len(r2.GetPoints()), len(r2.GetMissing()))
	}
}

// ---- VectorQuery (unified Query API) ----

// queryResultBody builds a coordinator-shaped flat fused query result (the wire
// the fan-out dispatcher / single-shard merge produces) with an optional
// degraded/missing trailer, for the countingDispatcher to return.
func queryResultBody(res []vector.Result, degraded bool, missing []uint16) []byte {
	return ops.EncodeQueryResultFusedDegraded(res, degraded, missing)
}

// TestVectorQueryFusionRoundTrip proves a FUSION query validates, dispatches the
// vector_query op with the marshaled QuerySpec (a 2-lane fusion), and decodes the
// coordinator's flat fused top-k + degraded/missing into a SearchResponse.
func TestVectorQueryFusionRoundTrip(t *testing.T) {
	want := []vector.Result{{ID: 7, Distance: 0.1, Score: 0.9}, {ID: 3, Distance: 0.2, Score: 0.5}}
	disp := &countingDispatcher{body: queryResultBody(want, true, []uint16{2})}
	s := NewServer(disp, nil)
	resp, err := s.VectorQuery(context.Background(), &pb.VectorQueryRequest{
		Collection: "docs",
		Spec: &pb.QuerySpec{
			Mode:         pb.QueryMode_QUERY_MODE_FUSION,
			FusionMethod: "dbsf",
			Alpha:        0.5,
			K:            2,
			Prefetch: []*pb.QueryLeaf{
				{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{Dense: []float32{1, 2}, K: 10}}},
				{Leaf: &pb.QueryLeaf_Sparse{Sparse: &pb.SparseLeaf{Indices: []uint32{0, 5}, Values: []float32{0.3, 0.7}, K: 10}}},
			},
		},
		ReadConsistency: 1, OnPartitionUnavailable: 1, MaxStaleness: 0,
	})
	if err != nil {
		t.Fatalf("VectorQuery: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_query" {
		t.Fatalf("dispatch op = %q, want vector_query", disp.lastOp)
	}
	// The dispatched args carry collection + rc/opa and a spec blob that decodes
	// back to the engine spec (dbsf, 2 prefetch lanes).
	coll, _, spec, rc, opa, _, derr := ops.DecodeQuerySpecArgs(disp.lastArg)
	if derr != nil {
		t.Fatalf("DecodeQuerySpecArgs: %v", derr)
	}
	if coll != "docs" || rc != 1 || opa != 1 {
		t.Fatalf("decoded coll/rc/opa = %q/%d/%d, want docs/1/1", coll, rc, opa)
	}
	if spec.Mode != vector.ModeFusion || spec.Method != vector.FusionDBSF || len(spec.Prefetch) != 2 {
		t.Fatalf("decoded spec = %+v, want fusion/dbsf/2-prefetch", spec)
	}
	if len(resp.GetResults()) != 2 || resp.GetResults()[0].GetId() != 7 {
		t.Fatalf("results = %v, want [7,3]", resp.GetResults())
	}
	if !resp.GetDegraded() || len(resp.GetMissing()) != 1 || resp.GetMissing()[0] != 2 {
		t.Fatalf("degraded/missing = %v/%v, want true/[2]", resp.GetDegraded(), resp.GetMissing())
	}
}

// TestVectorQueryRerankRoundTrip proves a RERANK query (root dense over a dense
// prefetch) round-trips and decodes the reranked top-k.
func TestVectorQueryRerankRoundTrip(t *testing.T) {
	want := []vector.Result{{ID: 1, Distance: 0.05, Score: 0.95}}
	disp := &countingDispatcher{body: queryResultBody(want, false, nil)}
	s := NewServer(disp, nil)
	resp, err := s.VectorQuery(context.Background(), &pb.VectorQueryRequest{
		Collection: "docs",
		Spec: &pb.QuerySpec{
			Mode: pb.QueryMode_QUERY_MODE_RERANK,
			K:    1,
			Root: &pb.QueryLeaf{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{Dense: []float32{1, 2}, K: 1}}},
			Prefetch: []*pb.QueryLeaf{
				{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{Dense: []float32{1, 2}, K: 50}}},
			},
		},
	})
	if err != nil {
		t.Fatalf("VectorQuery rerank: %v", err)
	}
	_, _, spec, _, _, _, derr := ops.DecodeQuerySpecArgs(disp.lastArg)
	if derr != nil {
		t.Fatalf("DecodeQuerySpecArgs: %v", derr)
	}
	if spec.Mode != vector.ModeRerank || spec.Root.Kind != vector.LeafDense {
		t.Fatalf("decoded spec = %+v, want rerank/dense-root", spec)
	}
	if len(resp.GetResults()) != 1 || resp.GetResults()[0].GetId() != 1 {
		t.Fatalf("results = %v, want [1]", resp.GetResults())
	}
	if resp.GetDegraded() {
		t.Fatalf("degraded = true, want false")
	}
}

// TestVectorQueryRejectsOutOfRangeConsistency proves an out-of-range
// read_consistency is rejected with InvalidArgument before dispatch.
func TestVectorQueryRejectsOutOfRangeConsistency(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	_, err := s.VectorQuery(context.Background(), &pb.VectorQueryRequest{
		Collection: "docs", ReadConsistency: 4,
		Spec: &pb.QuerySpec{Mode: pb.QueryMode_QUERY_MODE_FUSION, Prefetch: []*pb.QueryLeaf{
			{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{Dense: []float32{1}}}},
		}},
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("code = %v, want InvalidArgument", status.Code(err))
	}
	if disp.called {
		t.Fatalf("dispatcher must not be called for out-of-range read_consistency")
	}
}

// TestVectorQueryRejectsUnknownFusionMethod proves a bad fusion method fails
// loud at the edge (InvalidArgument), never silently degrading to RRF.
func TestVectorQueryRejectsUnknownFusionMethod(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	_, err := s.VectorQuery(context.Background(), &pb.VectorQueryRequest{
		Collection: "docs",
		Spec: &pb.QuerySpec{Mode: pb.QueryMode_QUERY_MODE_FUSION, FusionMethod: "bogus", Prefetch: []*pb.QueryLeaf{
			{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{Dense: []float32{1}}}},
		}},
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("code = %v, want InvalidArgument", status.Code(err))
	}
	if disp.called {
		t.Fatalf("dispatcher must not be called for an unknown fusion method")
	}
}

// TestVectorQueryRejectsBadLeafFilter proves a malformed per-leaf filter JSON
// fails loud at the edge (InvalidArgument), never a silent no-filter.
func TestVectorQueryRejectsBadLeafFilter(t *testing.T) {
	disp := &stubDispatcher{}
	s := NewServer(disp, nil)
	_, err := s.VectorQuery(context.Background(), &pb.VectorQueryRequest{
		Collection: "docs",
		Spec: &pb.QuerySpec{Mode: pb.QueryMode_QUERY_MODE_FUSION, Prefetch: []*pb.QueryLeaf{
			{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{Dense: []float32{1}, FilterJson: "{not json"}}},
		}},
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("code = %v, want InvalidArgument", status.Code(err))
	}
	if disp.called {
		t.Fatalf("dispatcher must not be called for a bad leaf filter")
	}
}

// namedDenseLeaf / namedSparseLeaf build the named-family oneof arms a
// NamedVectorQuery carries: each leaf targets a named vector space.
func namedDenseLeaf(space string, dense []float32, k int32) *pb.QueryLeaf {
	return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_NamedDense{NamedDense: &pb.NamedDenseLeaf{Space: space, Dense: dense, K: k}}}
}

func namedSparseLeaf(space string, idx []uint32, val []float32, k int32) *pb.QueryLeaf {
	return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_NamedSparse{NamedSparse: &pb.NamedSparseLeaf{Space: space, Indices: idx, Values: val, K: k}}}
}

// TestNamedVectorQueryFusionRoundTrip proves a multi-space FUSION named query
// validates at the edge, dispatches the vector_named_query op with the marshaled
// QuerySpec (3 lanes across 3 named spaces, each carrying its space), and decodes
// the coordinator's flat fused top-k + degraded/missing into a SearchResponse.
func TestNamedVectorQueryFusionRoundTrip(t *testing.T) {
	want := []vector.Result{{ID: 7, Distance: 0.1, Score: 0.9}, {ID: 3, Distance: 0.2, Score: 0.5}}
	disp := &countingDispatcher{body: queryResultBody(want, true, []uint16{2})}
	s := NewServer(disp, nil)
	resp, err := s.NamedVectorQuery(context.Background(), &pb.NamedVectorQueryRequest{
		Collection: "docs",
		Spec: &pb.QuerySpec{
			Mode:         pb.QueryMode_QUERY_MODE_FUSION,
			FusionMethod: "dbsf",
			Alpha:        0.5,
			K:            2,
			Prefetch: []*pb.QueryLeaf{
				namedDenseLeaf("title", []float32{1, 2}, 10),
				namedDenseLeaf("body", []float32{3, 4}, 10),
				namedSparseLeaf("kw", []uint32{0, 5}, []float32{0.3, 0.7}, 10),
			},
		},
		ReadConsistency: 1, OnPartitionUnavailable: 1, MaxStaleness: 0,
	})
	if err != nil {
		t.Fatalf("NamedVectorQuery: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_named_query" {
		t.Fatalf("dispatch op = %q, want vector_named_query", disp.lastOp)
	}
	coll, _, spec, rc, opa, _, derr := ops.DecodeQuerySpecArgs(disp.lastArg)
	if derr != nil {
		t.Fatalf("DecodeQuerySpecArgs: %v", derr)
	}
	if coll != "docs" || rc != 1 || opa != 1 {
		t.Fatalf("decoded coll/rc/opa = %q/%d/%d, want docs/1/1", coll, rc, opa)
	}
	if spec.Mode != vector.ModeFusion || spec.Method != vector.FusionDBSF || len(spec.Prefetch) != 3 {
		t.Fatalf("decoded spec = %+v, want fusion/dbsf/3-prefetch", spec)
	}
	// Each leaf must round-trip its named space (anti-silent-drop).
	if spec.Prefetch[0].Leaf.Space != "title" || spec.Prefetch[1].Leaf.Space != "body" || spec.Prefetch[2].Leaf.Space != "kw" {
		t.Fatalf("leaf spaces = %q/%q/%q, want title/body/kw", spec.Prefetch[0].Leaf.Space, spec.Prefetch[1].Leaf.Space, spec.Prefetch[2].Leaf.Space)
	}
	if spec.Prefetch[2].Leaf.Kind != vector.LeafSparse {
		t.Fatalf("leaf[2] kind = %v, want sparse", spec.Prefetch[2].Leaf.Kind)
	}
	if len(resp.GetResults()) != 2 || resp.GetResults()[0].GetId() != 7 {
		t.Fatalf("results = %v, want [7,3]", resp.GetResults())
	}
	if !resp.GetDegraded() || len(resp.GetMissing()) != 1 || resp.GetMissing()[0] != 2 {
		t.Fatalf("degraded/missing = %v/%v, want true/[2]", resp.GetDegraded(), resp.GetMissing())
	}
}

// TestNamedVectorQueryRerankRoundTrip proves a RERANK named query (root over a
// named space, prefetch over 2 named spaces) round-trips and decodes the reranked
// top-k, with the root's space riding into the dispatched spec.
func TestNamedVectorQueryRerankRoundTrip(t *testing.T) {
	want := []vector.Result{{ID: 1, Distance: 0.05, Score: 0.95}}
	disp := &countingDispatcher{body: queryResultBody(want, false, nil)}
	s := NewServer(disp, nil)
	resp, err := s.NamedVectorQuery(context.Background(), &pb.NamedVectorQueryRequest{
		Collection: "docs",
		Spec: &pb.QuerySpec{
			Mode: pb.QueryMode_QUERY_MODE_RERANK,
			K:    1,
			Root: namedDenseLeaf("title", []float32{1, 2}, 1),
			Prefetch: []*pb.QueryLeaf{
				namedDenseLeaf("title", []float32{1, 2}, 50),
				namedDenseLeaf("body", []float32{3, 4}, 50),
			},
		},
	})
	if err != nil {
		t.Fatalf("NamedVectorQuery rerank: %v", err)
	}
	_, _, spec, _, _, _, derr := ops.DecodeQuerySpecArgs(disp.lastArg)
	if derr != nil {
		t.Fatalf("DecodeQuerySpecArgs: %v", derr)
	}
	if spec.Mode != vector.ModeRerank || spec.Root.Kind != vector.LeafDense || spec.Root.Space != "title" {
		t.Fatalf("decoded spec = %+v, want rerank/dense-root/space=title", spec)
	}
	if len(resp.GetResults()) != 1 || resp.GetResults()[0].GetId() != 1 {
		t.Fatalf("results = %v, want [1]", resp.GetResults())
	}
	if resp.GetDegraded() {
		t.Fatalf("degraded = true, want false")
	}
}

// TestNamedVectorQueryRejectsMissingSpace proves a named leaf without a space
// fails loud at the edge (InvalidArgument) BEFORE dispatch — a Space-less leaf is
// never silently routed to a default space.
func TestNamedVectorQueryRejectsMissingSpace(t *testing.T) {
	cases := []struct {
		name string
		spec *pb.QuerySpec
	}{
		{"prefetch-named-no-space", &pb.QuerySpec{Mode: pb.QueryMode_QUERY_MODE_FUSION, Prefetch: []*pb.QueryLeaf{
			namedDenseLeaf("", []float32{1, 2}, 10),
		}}},
		{"prefetch-dense-arm-not-named", &pb.QuerySpec{Mode: pb.QueryMode_QUERY_MODE_FUSION, Prefetch: []*pb.QueryLeaf{
			{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{Dense: []float32{1, 2}, K: 10}}},
		}}},
		{"rerank-root-no-space", &pb.QuerySpec{Mode: pb.QueryMode_QUERY_MODE_RERANK, K: 1,
			Root:     namedDenseLeaf("", []float32{1, 2}, 1),
			Prefetch: []*pb.QueryLeaf{namedDenseLeaf("title", []float32{1, 2}, 50)},
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			disp := &stubDispatcher{}
			s := NewServer(disp, nil)
			_, err := s.NamedVectorQuery(context.Background(), &pb.NamedVectorQueryRequest{Collection: "docs", Spec: tc.spec})
			if status.Code(err) != codes.InvalidArgument {
				t.Fatalf("code = %v, want InvalidArgument", status.Code(err))
			}
			if disp.called {
				t.Fatalf("dispatcher must not be called for a missing-space named leaf")
			}
		})
	}
}

// TestNamedVectorQueryRejectsBadEdges proves the other fail-loud edges (out-of-
// range rc, unknown fusion method) reject with InvalidArgument before dispatch.
func TestNamedVectorQueryRejectsBadEdges(t *testing.T) {
	cases := []struct {
		name string
		req  *pb.NamedVectorQueryRequest
	}{
		{"bad-rc", &pb.NamedVectorQueryRequest{Collection: "docs", ReadConsistency: 4,
			Spec: &pb.QuerySpec{Mode: pb.QueryMode_QUERY_MODE_FUSION, Prefetch: []*pb.QueryLeaf{
				namedDenseLeaf("title", []float32{1, 2}, 10),
			}}}},
		{"unknown-method", &pb.NamedVectorQueryRequest{Collection: "docs",
			Spec: &pb.QuerySpec{Mode: pb.QueryMode_QUERY_MODE_FUSION, FusionMethod: "bogus", Prefetch: []*pb.QueryLeaf{
				namedDenseLeaf("title", []float32{1, 2}, 10),
			}}}},
		{"bad-leaf-filter", &pb.NamedVectorQueryRequest{Collection: "docs",
			Spec: &pb.QuerySpec{Mode: pb.QueryMode_QUERY_MODE_FUSION, Prefetch: []*pb.QueryLeaf{
				{Leaf: &pb.QueryLeaf_NamedDense{NamedDense: &pb.NamedDenseLeaf{Space: "title", Dense: []float32{1, 2}, FilterJson: "{not json"}}},
			}}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			disp := &stubDispatcher{}
			s := NewServer(disp, nil)
			_, err := s.NamedVectorQuery(context.Background(), tc.req)
			if status.Code(err) != codes.InvalidArgument {
				t.Fatalf("code = %v, want InvalidArgument", status.Code(err))
			}
			if disp.called {
				t.Fatalf("dispatcher must not be called for a bad edge")
			}
		})
	}
}

// mvMaxSimLeaf builds the MV MaxSim oneof arm (the token query matrix); mvSparseLeaf
// builds the MV doc-level sparse field (the dense-family SparseLeaf arm — no space).
func mvMaxSimLeaf(tokens [][]float32, k int32) *pb.QueryLeaf {
	q := make([]*pb.TokenVector, len(tokens))
	for i, tok := range tokens {
		q[i] = &pb.TokenVector{Values: tok}
	}
	return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_MvMaxsim{MvMaxsim: &pb.MVMaxSimLeaf{Query: q, K: k}}}
}

func mvSparseLeaf(idx []uint32, val []float32, k int32) *pb.QueryLeaf {
	return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_Sparse{Sparse: &pb.SparseLeaf{Indices: idx, Values: val, K: k}}}
}

// TestMVVectorQueryFusionRoundTrip proves a multi-lane MV FUSION query (a MaxSim
// lane + a doc-sparse lane) validates at the edge, dispatches the vector_mv_query op
// with the marshaled QuerySpec (the mv_maxsim token matrix + the sparse field both
// score-descending), and decodes the coordinator's flat fused top-k +
// degraded/missing into a SearchResponse.
func TestMVVectorQueryFusionRoundTrip(t *testing.T) {
	want := []vector.Result{{ID: 7, Distance: 0, Score: 0.9}, {ID: 3, Distance: 0, Score: 0.5}}
	disp := &countingDispatcher{body: queryResultBody(want, true, []uint16{2})}
	s := NewServer(disp, nil)
	resp, err := s.MVVectorQuery(context.Background(), &pb.MVVectorQueryRequest{
		Collection: "mv",
		Spec: &pb.QuerySpec{
			Mode:         pb.QueryMode_QUERY_MODE_FUSION,
			FusionMethod: "dbsf",
			Alpha:        0.5,
			K:            2,
			Prefetch: []*pb.QueryLeaf{
				mvMaxSimLeaf([][]float32{{1, 0, 0}, {0, 1, 0}}, 10),
				mvSparseLeaf([]uint32{0, 5}, []float32{0.3, 0.7}, 10),
			},
		},
		ReadConsistency: 1, OnPartitionUnavailable: 1, MaxStaleness: 0,
	})
	if err != nil {
		t.Fatalf("MVVectorQuery: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_mv_query" {
		t.Fatalf("dispatch op = %q, want vector_mv_query", disp.lastOp)
	}
	coll, _, spec, rc, opa, _, derr := ops.DecodeQuerySpecArgs(disp.lastArg)
	if derr != nil {
		t.Fatalf("DecodeQuerySpecArgs: %v", derr)
	}
	if coll != "mv" || rc != 1 || opa != 1 {
		t.Fatalf("decoded coll/rc/opa = %q/%d/%d, want mv/1/1", coll, rc, opa)
	}
	if spec.Mode != vector.ModeFusion || spec.Method != vector.FusionDBSF || len(spec.Prefetch) != 2 {
		t.Fatalf("decoded spec = %+v, want fusion/dbsf/2-prefetch", spec)
	}
	// MaxSim leaf carries the token matrix score-desc; the doc-sparse leaf is sparse
	// score-desc. Both MV lanes are score-descending (orientation-aware fuse).
	if spec.Prefetch[0].Leaf.Kind != vector.LeafMVMaxSim || len(spec.Prefetch[0].Leaf.Tokens) != 2 || !spec.Prefetch[0].Leaf.ScoreDesc {
		t.Fatalf("leaf[0] = %+v, want mv-maxsim 2-token score-desc", spec.Prefetch[0])
	}
	if spec.Prefetch[1].Leaf.Kind != vector.LeafSparse || !spec.Prefetch[1].Leaf.ScoreDesc {
		t.Fatalf("leaf[1] = %+v, want sparse score-desc", spec.Prefetch[1])
	}
	if len(resp.GetResults()) != 2 || resp.GetResults()[0].GetId() != 7 {
		t.Fatalf("results = %v, want [7,3]", resp.GetResults())
	}
	if !resp.GetDegraded() || len(resp.GetMissing()) != 1 || resp.GetMissing()[0] != 2 {
		t.Fatalf("degraded/missing = %v/%v, want true/[2]", resp.GetDegraded(), resp.GetMissing())
	}
}

// TestMVVectorQueryRerankRoundTrip proves a RERANK MV query (a MaxSim root over the
// candidate union, prefetch over a MaxSim + sparse lane) round-trips and decodes the
// reranked top-k, with the MaxSim root riding into the dispatched spec.
func TestMVVectorQueryRerankRoundTrip(t *testing.T) {
	want := []vector.Result{{ID: 1, Distance: 0, Score: 0.95}}
	disp := &countingDispatcher{body: queryResultBody(want, false, nil)}
	s := NewServer(disp, nil)
	resp, err := s.MVVectorQuery(context.Background(), &pb.MVVectorQueryRequest{
		Collection: "mv",
		Spec: &pb.QuerySpec{
			Mode: pb.QueryMode_QUERY_MODE_RERANK,
			K:    1,
			Root: mvMaxSimLeaf([][]float32{{0, 1, 0}}, 1),
			Prefetch: []*pb.QueryLeaf{
				mvMaxSimLeaf([][]float32{{1, 0, 0}}, 50),
				mvSparseLeaf([]uint32{0, 2}, []float32{1, 1}, 50),
			},
		},
	})
	if err != nil {
		t.Fatalf("MVVectorQuery rerank: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_mv_query" {
		t.Fatalf("dispatch op = %q, want vector_mv_query", disp.lastOp)
	}
	_, _, spec, _, _, _, derr := ops.DecodeQuerySpecArgs(disp.lastArg)
	if derr != nil {
		t.Fatalf("DecodeQuerySpecArgs: %v", derr)
	}
	if spec.Mode != vector.ModeRerank || spec.Root.Kind != vector.LeafMVMaxSim || len(spec.Root.Tokens) != 1 {
		t.Fatalf("decoded spec = %+v, want rerank/mv-maxsim-root", spec)
	}
	if len(resp.GetResults()) != 1 || resp.GetResults()[0].GetId() != 1 {
		t.Fatalf("results = %v, want [1]", resp.GetResults())
	}
	if resp.GetDegraded() {
		t.Fatalf("degraded = true, want false")
	}
}

// TestMVVectorQueryRejectsBadEdges proves the fail-loud edges (out-of-range rc,
// unknown fusion method, bad per-leaf filter JSON) reject with InvalidArgument
// BEFORE dispatch — never a silent degrade-to-RRF / no-filter.
func TestMVVectorQueryRejectsBadEdges(t *testing.T) {
	cases := []struct {
		name string
		req  *pb.MVVectorQueryRequest
	}{
		{"bad-rc", &pb.MVVectorQueryRequest{Collection: "mv", ReadConsistency: 4,
			Spec: &pb.QuerySpec{Mode: pb.QueryMode_QUERY_MODE_FUSION, Prefetch: []*pb.QueryLeaf{
				mvMaxSimLeaf([][]float32{{1, 0, 0}}, 10),
			}}}},
		{"unknown-method", &pb.MVVectorQueryRequest{Collection: "mv",
			Spec: &pb.QuerySpec{Mode: pb.QueryMode_QUERY_MODE_FUSION, FusionMethod: "bogus", Prefetch: []*pb.QueryLeaf{
				mvMaxSimLeaf([][]float32{{1, 0, 0}}, 10),
			}}}},
		{"bad-leaf-filter", &pb.MVVectorQueryRequest{Collection: "mv",
			Spec: &pb.QuerySpec{Mode: pb.QueryMode_QUERY_MODE_FUSION, Prefetch: []*pb.QueryLeaf{
				{Leaf: &pb.QueryLeaf_MvMaxsim{MvMaxsim: &pb.MVMaxSimLeaf{Query: []*pb.TokenVector{{Values: []float32{1, 0, 0}}}, FilterJson: "{not json"}}},
			}}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			disp := &stubDispatcher{}
			s := NewServer(disp, nil)
			_, err := s.MVVectorQuery(context.Background(), tc.req)
			if status.Code(err) != codes.InvalidArgument {
				t.Fatalf("code = %v, want InvalidArgument", status.Code(err))
			}
			if disp.called {
				t.Fatalf("dispatcher must not be called for a bad edge")
			}
		})
	}
}

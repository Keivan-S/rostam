// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// TestStorePersistentVectorsClusterCycle proves the shard.Config.PersistentVectors
// plumbing end-to-end: vector ops applied through Raft land in mmap-backed
// (off-heap) collections, and after a restart the on-disk files are wiped and
// repopulated by Raft log replay (Raft is the durability authority), with the
// data still searchable.
func TestStorePersistentVectorsClusterCycle(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mmap only on linux")
	}
	dir := t.TempDir()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	cfg := DefaultConfig(dir, "node1", reg)
	cfg.Bootstrap = true
	cfg.RaftHeartbeatMs = 50
	cfg.RaftElectionMs = 100
	cfg.NoSync = true
	cfg.PersistentVectors = true

	colCfg := vector.Config{Dim: 3, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Quant: vector.QuantSQ8, RescoreFactor: 4}

	s1, err := New(cfg)
	if err != nil {
		t.Fatalf("first New: %v", err)
	}
	waitLeader(t, s1)
	if _, err := s1.Call("vector_create_collection", ops.EncodeCreateCollectionArgs("docs", colCfg)); err != nil {
		t.Fatalf("create collection: %v", err)
	}
	for i := 1; i <= 12; i++ {
		args := ops.EncodeVectorUpsertArgs("docs", uint64(i), []float32{float32(i), 0, 0}, "chunk", 0, nil, vector.SparseVector{})
		if _, err := s1.Call("vector_upsert", args); err != nil {
			t.Fatalf("upsert %d: %v", i, err)
		}
	}

	// The collection is mmap-backed (off-heap) because PersistentVectors=true.
	c, ok := s1.vectors.Get("docs")
	if !ok || c.Config().MmapPath == "" {
		t.Fatalf("collection not mmap-backed (ok=%v cfg=%+v)", ok, c.Config())
	}
	if !strings.Contains(filepath.Base(c.Config().MmapPath), ".g") {
		t.Fatalf("expected generation-suffixed mmap path, got %q", c.Config().MmapPath)
	}
	if err := s1.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}

	// Reopen: vector data files are wiped at open, then Raft log replay
	// repopulates the mmap-backed collection. Data must still be searchable.
	cfg.Bootstrap = false
	s2, err := New(cfg)
	if err != nil {
		t.Fatalf("second New: %v", err)
	}
	defer func() { _ = s2.Close() }()
	waitLeader(t, s2)
	// Winning the election does NOT mean the log has finished replaying into the
	// FSM: leadership is granted on log INDEX, while the upserts this search has to
	// see only exist once each entry has been APPLIED to the vector store. Searching
	// straight after waitLeader raced replay and returned a short result set (0 hits,
	// or a partial 4) whenever the machine was loaded enough to slow the FSM.
	// Barrier commits a no-op and waits for it to be applied, so every entry ahead of
	// it — the collection create and all 12 upserts — is applied before it returns.
	if err := s2.raft.Barrier(10 * time.Second); err != nil {
		t.Fatalf("barrier waiting for log replay: %v", err)
	}

	res, err := s2.Call("vector_search", ops.EncodeVectorSearchArgs("docs", 5, []float32{1, 0, 0}))
	if err != nil {
		t.Fatalf("post-restart search: %v", err)
	}
	hits, err := ops.DecodeVectorSearchResults(res)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 5 {
		t.Fatalf("post-restart search returned %d hits, want 5", len(hits))
	}
	// Still mmap-backed after the restart.
	if c2, ok := s2.vectors.Get("docs"); !ok || c2.Config().MmapPath == "" {
		t.Fatalf("collection not mmap-backed after restart (ok=%v)", ok)
	}
}

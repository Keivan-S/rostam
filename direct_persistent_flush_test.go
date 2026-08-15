// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// openDirectOn opens a Direct store on dir without registering a cleanup close:
// these tests close and reopen the same directory, so the close is theirs to
// sequence. The directory must come from t.TempDir() BEFORE the store exists —
// t.Cleanup is LIFO, so registering the dir first is what makes it outlive the
// store rather than being removed under it.
func openDirectOn(t *testing.T, dir string) Store {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	s, err := NewDirect(DirectConfig{
		DataDir: dir,
		Ops:     reg,
		Cache:   CacheConfig{NumShardsPerNode: 1},
	})
	if err != nil {
		t.Fatalf("NewDirect(%q): %v", dir, err)
	}
	return s
}

// persistentTestConfig is a minimal Persistent collection config. Persistent
// requires a quantizer (mmap-backed vector storage holds codes), so SQ8 rides
// along; the vectors themselves stay full-precision in the mmap file.
func persistentTestConfig() vector.Config {
	cfg := vector.DefaultConfig()
	cfg.Dim, cfg.Metric = 8, vector.Cosine
	cfg.Persistent = true
	cfg.Quant = vector.QuantSQ8
	return cfg
}

// TestDirectPersistentCollectionSurvivesClose is what Persistent means for an
// embedded caller: the contents come back after a clean Close.
//
// The mmap files always survived, but the sidecar that makes them readable
// (per-slot ids, node levels, edges, entry point) is only written by a Flush,
// and nothing above the vector package ever called one — so before Close
// flushed them, a Persistent collection reopened EMPTY and the whole feature
// was silently inert from the Store API.
func TestDirectPersistentCollectionSurvivesClose(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	vec := []float32{1, 0, 0, 0, 0, 0, 0, 0}

	s := openDirectOn(t, dir)
	if err := s.CreateCollection(ctx, "docs", persistentTestConfig()); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	if err := s.VectorUpsert(ctx, "docs", 42, vec, "hello", VectorInsertOpts{}); err != nil {
		t.Fatalf("VectorUpsert: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	s2 := openDirectOn(t, dir)
	defer func() { _ = s2.Close() }()
	points, missing, err := s2.VectorGetBatch(ctx, "docs", []uint64{42}, false, true)
	if err != nil {
		t.Fatalf("VectorGetBatch after reopen: %v", err)
	}
	if len(points) != 1 || points[0].ID != 42 {
		t.Fatalf("after reopen: points = %+v, missing = %v", points, missing)
	}
}

// TestDirectPersistentSurvivesCloseAfterDelete covers the same path with a
// tombstone in the index: SavePersist records freed and tombstoned slots, so a
// collection that has had a delete must still flush rather than failing Close
// and taking the surviving points down with it.
func TestDirectPersistentSurvivesCloseAfterDelete(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	s := openDirectOn(t, dir)
	if err := s.CreateCollection(ctx, "docs", persistentTestConfig()); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	for id := uint64(1); id <= 3; id++ {
		vec := make([]float32, 8)
		vec[id] = 1
		if err := s.VectorUpsert(ctx, "docs", id, vec, "point", VectorInsertOpts{}); err != nil {
			t.Fatalf("VectorUpsert %d: %v", id, err)
		}
	}
	if _, err := s.VectorDelete(ctx, "docs", 2); err != nil {
		t.Fatalf("VectorDelete: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close after delete: %v", err)
	}

	s2 := openDirectOn(t, dir)
	defer func() { _ = s2.Close() }()
	points, missing, err := s2.VectorGetBatch(ctx, "docs", []uint64{1, 2, 3}, false, true)
	if err != nil {
		t.Fatalf("VectorGetBatch after reopen: %v", err)
	}
	if len(points) != 2 {
		t.Fatalf("want the 2 surviving points, got %+v (missing %v)", points, missing)
	}
	if len(missing) != 1 || missing[0] != 2 {
		t.Fatalf("the deleted id should still be gone: missing = %v", missing)
	}
}

// TestDirectNonPersistentCollectionNotFlushed pins the other half of the
// contract: Close flushes ONLY Persistent collections. A plain collection's
// snapshot is an explicit caller choice, and writing one at every shutdown
// would make Close's cost scale with data nobody asked to persist.
func TestDirectNonPersistentCollectionNotFlushed(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	cfg := vector.DefaultConfig()
	cfg.Dim, cfg.Metric = 8, vector.Cosine

	s := openDirectOn(t, dir)
	if err := s.CreateCollection(ctx, "scratch", cfg); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	if err := s.VectorUpsert(ctx, "scratch", 7, []float32{1, 0, 0, 0, 0, 0, 0, 0}, "hi", VectorInsertOpts{}); err != nil {
		t.Fatalf("VectorUpsert: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	s2 := openDirectOn(t, dir)
	defer func() { _ = s2.Close() }()
	points, _, err := s2.VectorGetBatch(ctx, "scratch", []uint64{7}, false, true)
	if err != nil {
		t.Fatalf("VectorGetBatch after reopen: %v", err)
	}
	if len(points) != 0 {
		t.Fatalf("a non-persistent collection should not have been flushed at Close, got %+v", points)
	}
}

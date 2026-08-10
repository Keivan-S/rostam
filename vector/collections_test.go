// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"path/filepath"
	"testing"
)

func TestCollectionStoreCreateAndGet(t *testing.T) {
	dir := t.TempDir()
	store, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	cfg := Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2}
	if err := store.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	c, ok := store.Get("docs")
	if !ok {
		t.Fatal("Get(docs) returned !ok")
	}
	// Bare names get canonicalized under "default/".
	if c.Name() != "default/docs" {
		t.Errorf("Name = %q, want default/docs", c.Name())
	}
	// Duplicate create returns an error.
	if err := store.CreateCollection("docs", cfg); err == nil {
		t.Error("duplicate CreateCollection should error")
	}
}

func TestCollectionStorePersistsAcrossRestart(t *testing.T) {
	dir := t.TempDir()

	// Create + populate + flush.
	{
		store, _ := OpenCollectionStore(dir)
		cfg := Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2}
		_ = store.CreateCollection("docs", cfg)
		c, _ := store.Get("docs")
		_ = c.Insert(1, []float32{1, 0}, 0, nil, nil)
		_ = c.Insert(2, []float32{2, 0}, 0, nil, nil)
		if err := store.Flush("docs"); err != nil {
			t.Fatal(err)
		}
		store.Close()
	}

	// Reopen, verify state.
	{
		store, err := OpenCollectionStore(dir)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		c, ok := store.Get("docs")
		if !ok {
			t.Fatal("reopened store missing docs collection")
		}
		results, _ := c.Search([]float32{1, 0}, 1)
		if len(results) != 1 || results[0].ID != 1 {
			t.Errorf("post-restart search = %+v, want id 1", results)
		}
	}
}

func TestCollectionStoreDrop(t *testing.T) {
	dir := t.TempDir()
	store, _ := OpenCollectionStore(dir)
	defer store.Close()
	cfg := Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2}
	_ = store.CreateCollection("a", cfg)
	if err := store.DropCollection("a"); err != nil {
		t.Fatal(err)
	}
	if _, ok := store.Get("a"); ok {
		t.Error("dropped collection still present")
	}
	if _, err := filepath.Glob(filepath.Join(dir, "vectors", "a.*")); err != nil {
		t.Fatal(err)
	}
	// Drop of unknown collection is a no-op.
	if err := store.DropCollection("nonexistent"); err != nil {
		t.Errorf("DropCollection(unknown) = %v, want nil", err)
	}
}

func TestCollectionStoreMaybeReclaimBelowThreshold(t *testing.T) {
	dir := t.TempDir()
	store, _ := OpenCollectionStore(dir)
	defer store.Close()
	cfg := Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2}
	_ = store.CreateCollection("docs", cfg)
	c, _ := store.Get("docs")
	for i := uint64(1); i <= 10; i++ {
		_ = c.Insert(i, []float32{float32(i), 0}, 0, nil, nil)
	}
	_ = c.Delete(1) // ratio = 1/10 = 0.1

	count, ran := store.MaybeReclaim("docs", 0.2)
	if ran {
		t.Errorf("ran=true at ratio 0.1 below threshold 0.2; count=%d", count)
	}
}

func TestCollectionStoreMaybeReclaimAboveThreshold(t *testing.T) {
	dir := t.TempDir()
	store, _ := OpenCollectionStore(dir)
	defer store.Close()
	cfg := Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2}
	_ = store.CreateCollection("docs", cfg)
	c, _ := store.Get("docs")
	for i := uint64(1); i <= 10; i++ {
		_ = c.Insert(i, []float32{float32(i), 0}, 0, nil, nil)
	}
	for _, id := range []uint64{1, 2, 3} {
		_ = c.Delete(id)
	}

	count, ran := store.MaybeReclaim("docs", 0.2)
	if !ran {
		t.Fatal("ran=false at ratio 0.3 above threshold 0.2")
	}
	if count != 3 {
		t.Errorf("count=%d, want 3", count)
	}
	if c.TombstoneRatio() != 0 {
		t.Errorf("post-reclaim ratio=%v, want 0", c.TombstoneRatio())
	}
}

func TestCollectionStoreMaybeReclaimUnknownCollection(t *testing.T) {
	dir := t.TempDir()
	store, _ := OpenCollectionStore(dir)
	defer store.Close()
	count, ran := store.MaybeReclaim("nonexistent", 0.1)
	if ran || count != 0 {
		t.Errorf("MaybeReclaim(nonexistent) returned (%d, %v)", count, ran)
	}
}

func TestCollectionStoreMetadataPersists(t *testing.T) {
	dir := t.TempDir()

	// Create, insert with metadata, flush.
	{
		store, _ := OpenCollectionStore(dir)
		cfg := Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: L2}
		if err := store.CreateCollection("docs", cfg); err != nil {
			t.Fatal(err)
		}
		if err := store.Insert("docs", 1, []float32{1, 0, 0, 0}, 0, Metadata{"tenant": NewString("acme")}, nil); err != nil {
			t.Fatal(err)
		}
		if err := store.Insert("docs", 2, []float32{0, 1, 0, 0}, 0, Metadata{"tenant": NewString("globex")}, nil); err != nil {
			t.Fatal(err)
		}
		if err := store.Flush("docs"); err != nil {
			t.Fatal(err)
		}
		store.Close()
	}

	// Reopen, filtered search must reflect persisted metadata.
	{
		store, err := OpenCollectionStore(dir)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		res, err := store.SearchFiltered("docs", []float32{1, 0, 0, 0}, 10, Filter{Op: FilterEq, Field: "tenant", Value: NewString("acme")})
		if err != nil {
			t.Fatal(err)
		}
		if len(res) != 1 || res[0].ID != 1 {
			t.Errorf("post-restart filtered search = %+v, want [id=1]", res)
		}
	}
}

// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"testing"
)

// TestCollectionStoreNamedLifecycle exercises the store path: create a named
// collection, insert points (one omitting a space), search each space, filtered
// search, delete, scroll, and drop — all via CollectionStore methods.
func TestCollectionStoreNamedLifecycle(t *testing.T) {
	cs, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer cs.Close()

	cfg := map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: Cosine},
		"image": {Dim: 3, Metric: DotProduct},
	}
	if err := cs.CreateNamed("docs", cfg); err != nil {
		t.Fatalf("create named: %v", err)
	}
	if _, ok := cs.GetNamed("docs"); !ok {
		t.Fatal("named collection missing after create")
	}

	// Insert via the store. Point 3 omits the image space.
	must := func(id uint64, vectors map[string][]float32, kind string) {
		t.Helper()
		if err := cs.NamedInsert("docs", id, vectors, Metadata{"kind": NewString(kind)}, 0); err != nil {
			t.Fatalf("named insert %d: %v", id, err)
		}
	}
	must(1, map[string][]float32{"title": {1, 0, 0, 0}, "image": {1, 0, 0}}, "a")
	must(2, map[string][]float32{"title": {0, 1, 0, 0}, "image": {0, 1, 0}}, "b")
	must(3, map[string][]float32{"title": {0, 0, 1, 0}}, "a")

	// Search title space.
	tr, err := cs.NamedSearch("docs", "title", []float32{0, 0, 1, 0}, 3, Filter{})
	if err != nil {
		t.Fatalf("named search title: %v", err)
	}
	if len(tr) == 0 || tr[0].ID != 3 {
		t.Fatalf("title top = %v, want id 3", resultIDs(tr))
	}

	// Filtered search (shared payload).
	fr, err := cs.NamedSearch("docs", "title", []float32{1, 0, 0, 0}, 10,
		Filter{Op: FilterEq, Field: "kind", Value: NewString("a")})
	if err != nil {
		t.Fatalf("named filtered search: %v", err)
	}
	if len(fr) != 2 {
		t.Fatalf("filtered (kind=a) returned %d, want 2 (ids 1,3)", len(fr))
	}

	// SearchDocs carries shared payload.
	docs, err := cs.NamedSearchDocs("docs", "title", []float32{1, 0, 0, 0}, 3, Filter{})
	if err != nil {
		t.Fatalf("named search docs: %v", err)
	}
	for _, d := range docs {
		if d.Metadata == nil || d.Metadata["kind"].Str == "" {
			t.Errorf("search doc %d missing shared payload", d.ID)
		}
	}

	// Delete id 1 → gone from both spaces.
	existed, err := cs.NamedDelete("docs", 1)
	if err != nil || !existed {
		t.Fatalf("named delete: existed=%v err=%v", existed, err)
	}
	ir, _ := cs.NamedSearch("docs", "image", []float32{1, 0, 0}, 10, Filter{})
	for _, r := range ir {
		if r.ID == 1 {
			t.Error("deleted id 1 still in image space via store")
		}
	}

	// Scroll live points + payload.
	scroll, err := cs.NamedScroll("docs", Filter{}, 0)
	if err != nil {
		t.Fatalf("named scroll: %v", err)
	}
	if len(scroll) != 2 {
		t.Fatalf("scroll = %d live points, want 2 after delete", len(scroll))
	}
	if docIDsOf(scroll)[1] {
		t.Error("deleted id 1 still in scroll")
	}

	// Config introspection.
	gotCfg, err := cs.NamedConfig("docs")
	if err != nil {
		t.Fatalf("named config: %v", err)
	}
	if len(gotCfg) != 2 || gotCfg["title"].Dim != 4 || gotCfg["image"].Dim != 3 {
		t.Fatalf("named config = %+v, want 2 spaces dim 4/3", gotCfg)
	}

	// Drop removes it.
	if err := cs.DropNamed("docs"); err != nil {
		t.Fatalf("drop named: %v", err)
	}
	if _, ok := cs.GetNamed("docs"); ok {
		t.Error("named collection still present after drop")
	}
}

// TestCollectionStoreCreateCollectionRoutesNamed verifies CreateCollection with
// Config.NamedVectors set routes to the named family.
func TestCollectionStoreCreateCollectionRoutesNamed(t *testing.T) {
	cs, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer cs.Close()
	cfg := Config{NamedVectors: map[string]NamedVectorParams{"title": {Dim: 4, Metric: Cosine}}}
	if err := cs.CreateCollection("nv", cfg); err != nil {
		t.Fatalf("create via CreateCollection(NamedVectors): %v", err)
	}
	if _, ok := cs.GetNamed("nv"); !ok {
		t.Fatal("named collection not registered via CreateCollection")
	}
	// The dense registry must NOT have it (routed to named family only).
	if _, ok := cs.Get("nv"); ok {
		t.Error("named collection leaked into the dense collections map")
	}
}

// TestCollectionStoreNamedCrossFamilyDuplicate verifies a name is exclusive
// across the dense / multi-vector / named families.
func TestCollectionStoreNamedCrossFamilyDuplicate(t *testing.T) {
	cs, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer cs.Close()

	// dense "x" then named "x" → ErrCollectionExists.
	if err := cs.CreateCollection("x", Config{Dim: 4, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64}); err != nil {
		t.Fatalf("create dense x: %v", err)
	}
	if err := cs.CreateNamed("x", map[string]NamedVectorParams{"title": {Dim: 4}}); !errors.Is(err, ErrCollectionExists) {
		t.Errorf("named x over dense x err = %v, want ErrCollectionExists", err)
	}

	// named "y" then dense "y" → ErrCollectionExists.
	if err := cs.CreateNamed("y", map[string]NamedVectorParams{"title": {Dim: 4}}); err != nil {
		t.Fatalf("create named y: %v", err)
	}
	if err := cs.CreateCollection("y", Config{Dim: 4, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64}); !errors.Is(err, ErrCollectionExists) {
		t.Errorf("dense y over named y err = %v, want ErrCollectionExists", err)
	}
	// MV "y" over named "y" → ErrCollectionExists.
	if err := cs.CreateMultiVector("y", MultiVectorConfig{Dim: 4}); !errors.Is(err, ErrCollectionExists) {
		t.Errorf("mv y over named y err = %v, want ErrCollectionExists", err)
	}
}

// TestCollectionStoreNamedConfigValidation verifies create-time validation at
// the store edge (empty config / reserved-char name).
func TestCollectionStoreNamedConfigValidation(t *testing.T) {
	cs, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer cs.Close()
	if err := cs.CreateNamed("a", map[string]NamedVectorParams{}); !errors.Is(err, ErrEmptyNamedVectors) {
		t.Errorf("empty config err = %v, want ErrEmptyNamedVectors", err)
	}
	if err := cs.CreateNamed("b", map[string]NamedVectorParams{"bad@name": {Dim: 4}}); !errors.Is(err, ErrReservedVectorName) {
		t.Errorf("reserved name err = %v, want ErrReservedVectorName", err)
	}
}

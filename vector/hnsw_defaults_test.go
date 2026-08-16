// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"testing"
)

// The struct comment on Config.M has always promised "graph degree (default 16)"
// and DefaultConfig()/DefaultConfig-equivalents use 16/200/64, but the HNSW path
// used to validate those fields and reject zero — so an embedded caller who set
// only Dim and Metric got ErrInvalidM, while the identical create over HTTP or
// the Python client succeeded (both fill the same defaults before the engine
// sees the config). These tests pin the engine to the documented behaviour so
// the three entry points agree.

func TestNewCollectionDefaultsHNSWParams(t *testing.T) {
	// Only Dim and Metric — exactly the config the quickstart's embedded example
	// used, which used to fail on first copy-paste.
	col, err := NewCollection("docs", Config{Dim: 8, Metric: Cosine})
	if err != nil {
		t.Fatalf("NewCollection with zero M/Ef: got %v, want it to default them", err)
	}
	defer col.Close()

	got := col.cfg
	if got.M != 16 {
		t.Errorf("M = %d, want the documented default 16", got.M)
	}
	if got.EfConstruction != 200 {
		t.Errorf("EfConstruction = %d, want 200", got.EfConstruction)
	}
	if got.EfSearch != 64 {
		t.Errorf("EfSearch = %d, want 64", got.EfSearch)
	}
}

func TestNewCollectionKeepsExplicitHNSWParams(t *testing.T) {
	// Explicit values must survive untouched — defaulting only fills zeros.
	col, err := NewCollection("docs", Config{
		Dim: 8, Metric: L2, M: 32, EfConstruction: 400, EfSearch: 128,
	})
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	defer col.Close()
	if col.cfg.M != 32 || col.cfg.EfConstruction != 400 || col.cfg.EfSearch != 128 {
		t.Errorf("explicit params were overwritten: got M=%d efc=%d efs=%d",
			col.cfg.M, col.cfg.EfConstruction, col.cfg.EfSearch)
	}
}

func TestNewCollectionInsertSearchAfterDefaulting(t *testing.T) {
	// A defaulted collection must be fully usable, not merely constructible.
	col, err := NewCollection("docs", Config{Dim: 4, Metric: Cosine})
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	defer col.Close()
	if err := col.Insert(1, []float32{0.1, 0.2, 0.3, 0.4}, 0, nil, nil); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	hits, err := col.Search([]float32{0.1, 0.2, 0.3, 0.4}, 3)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 1 || hits[0].ID != 1 {
		t.Fatalf("Search returned %v, want the one inserted point", hits)
	}
}

// IVF is intentionally NOT defaulted here: it validates M through the same path
// and requires the caller to configure it. This test locks that boundary so a
// future change to the HNSW default does not silently start defaulting IVF too.
func TestNewCollectionDoesNotDefaultIVFParams(t *testing.T) {
	_, err := NewCollection("docs", Config{Dim: 8, Metric: Cosine, IndexType: IndexIVF})
	if !errors.Is(err, ErrInvalidM) {
		t.Fatalf("IVF with zero M: got %v, want ErrInvalidM (IVF is not in scope of the HNSW default)", err)
	}
}

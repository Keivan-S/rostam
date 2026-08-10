// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"strings"
	"testing"
)

// TestWritePrometheusAll renders the whole store and asserts that every live
// (dense) collection contributes its own labeled series: the gauge families and
// counters appear once per collection, each tagged with the collection's full
// tenant/name.
func TestWritePrometheusAll(t *testing.T) {
	store, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	cfg := Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2}
	if err := store.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	if err := store.CreateCollection("notes", cfg); err != nil {
		t.Fatal(err)
	}

	// Populate both so size > 0 and a search bumps the search-ops counter.
	for _, name := range []string{"docs", "notes"} {
		c, ok := store.Acquire(name)
		if !ok {
			t.Fatalf("Acquire(%q) returned !ok", name)
		}
		_ = c.Insert(1, []float32{1, 0}, 0, nil, nil)
		_ = c.Insert(2, []float32{2, 0}, 0, nil, nil)
		_, _ = c.Search([]float32{1, 0}, 1)
		c.Release()
	}

	var buf bytes.Buffer
	if err := store.WritePrometheusAll(&buf); err != nil {
		t.Fatalf("WritePrometheusAll: %v", err)
	}
	out := buf.String()

	// Both collections appear, under their canonical tenant/name.
	for _, want := range []string{
		`collection="default/docs"`,
		`collection="default/notes"`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing label %q\n%s", want, out)
		}
	}

	// Expected metric families (one HELP line each, shared across collections).
	for _, fam := range []string{
		"rostam_vector_size",
		"rostam_vector_search_ops_total",
		"rostam_vector_insert_ops_total",
	} {
		if !strings.Contains(out, fam) {
			t.Errorf("output missing metric family %q\n%s", fam, out)
		}
	}

	// The size gauge must carry each collection's label with a non-empty value.
	for _, line := range []string{
		`rostam_vector_size{collection="default/docs"}`,
		`rostam_vector_size{collection="default/notes"}`,
	} {
		if !strings.Contains(out, line) {
			t.Errorf("output missing series %q\n%s", line, out)
		}
	}
}

// TestWritePrometheusAllEmpty renders a store with no collections: valid (empty)
// output and no error.
func TestWritePrometheusAllEmpty(t *testing.T) {
	store, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	var buf bytes.Buffer
	if err := store.WritePrometheusAll(&buf); err != nil {
		t.Fatalf("WritePrometheusAll: %v", err)
	}
	if buf.Len() != 0 {
		t.Errorf("empty store produced %d bytes:\n%s", buf.Len(), buf.String())
	}
}

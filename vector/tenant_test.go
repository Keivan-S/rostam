// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"testing"
)

func TestSplitTenant(t *testing.T) {
	cases := []struct {
		in   string
		t, c string
		err  bool
	}{
		{"docs", "default", "docs", false},
		{"acme/docs", "acme", "docs", false},
		{"globex/embeddings", "globex", "embeddings", false},
		{"", "", "", true},
		{"a/b/c", "", "", true},
		{"acme/", "", "", true},
		{"/docs", "", "", true},
		{"/", "", "", true},
	}
	for _, tc := range cases {
		tenant, collection, err := splitTenant(tc.in)
		if tc.err {
			if err == nil {
				t.Errorf("splitTenant(%q) err = nil, want error", tc.in)
			}
			continue
		}
		if err != nil {
			t.Errorf("splitTenant(%q) err = %v, want nil", tc.in, err)
			continue
		}
		if tenant != tc.t || collection != tc.c {
			t.Errorf("splitTenant(%q) = (%q, %q), want (%q, %q)",
				tc.in, tenant, collection, tc.t, tc.c)
		}
	}
}

func TestCollectionStoreTenantIsolation(t *testing.T) {
	dir := t.TempDir()
	store, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	cfg := Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2}
	for _, name := range []string{"acme/docs", "globex/docs"} {
		if err := store.CreateCollection(name, cfg); err != nil {
			t.Fatalf("CreateCollection(%q): %v", name, err)
		}
	}

	if err := store.Insert("acme/docs", 1, []float32{1, 0}, 0, nil, nil); err != nil {
		t.Fatalf("Insert acme: %v", err)
	}
	if err := store.Insert("globex/docs", 99, []float32{1, 0}, 0, nil, nil); err != nil {
		t.Fatalf("Insert globex: %v", err)
	}

	acme, _ := store.Get("acme/docs")
	globex, _ := store.Get("globex/docs")

	res, _ := acme.Search([]float32{1, 0}, 10)
	if len(res) != 1 || res[0].ID != 1 {
		t.Errorf("acme/docs search = %+v, want [id=1]", res)
	}
	res, _ = globex.Search([]float32{1, 0}, 10)
	if len(res) != 1 || res[0].ID != 99 {
		t.Errorf("globex/docs search = %+v, want [id=99]", res)
	}
}

func TestCollectionStoreBareNameMapsToDefault(t *testing.T) {
	dir := t.TempDir()
	store, _ := OpenCollectionStore(dir)
	defer store.Close()

	cfg := Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2}
	if err := store.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}

	// Bare name and explicit "default/" prefix should resolve to the same collection.
	c1, ok1 := store.Get("docs")
	c2, ok2 := store.Get("default/docs")
	if !ok1 || !ok2 {
		t.Fatal("Get returned !ok for one of the equivalent names")
	}
	if c1 != c2 {
		t.Error("Get(docs) and Get(default/docs) returned different *Collection pointers")
	}
}

func TestCanonicalName(t *testing.T) {
	cases := []struct {
		in  string
		out string
		err bool
	}{
		{"docs", "default/docs", false},
		{"acme/docs", "acme/docs", false},
		{"", "", true},
		{"a/b/c", "", true},
	}
	for _, tc := range cases {
		got, err := canonicalName(tc.in)
		if tc.err {
			if err == nil {
				t.Errorf("canonicalName(%q) err = nil, want error", tc.in)
			}
			continue
		}
		if err != nil {
			t.Errorf("canonicalName(%q) err = %v, want nil", tc.in, err)
			continue
		}
		if got != tc.out {
			t.Errorf("canonicalName(%q) = %q, want %q", tc.in, got, tc.out)
		}
	}
}

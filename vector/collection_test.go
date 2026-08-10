// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"reflect"
	"testing"
)

func TestNewCollectionRequiresValidConfig(t *testing.T) {
	if _, err := NewCollection("test", Config{Dim: 0}); err == nil {
		t.Fatal("NewCollection with invalid config should error")
	}
}

func TestNewCollectionInsertSearchRoundtrip(t *testing.T) {
	c, err := NewCollection("test", Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Insert(1, []float32{1, 0}, 0, nil, nil); err != nil {
		t.Fatal(err)
	}
	results, err := c.Search([]float32{1, 0}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].ID != 1 {
		t.Fatalf("Search returned %+v, want id 1", results)
	}
	if c.Name() != "test" {
		t.Errorf("Name = %q, want test", c.Name())
	}
}

func TestCollectionExposesConfig(t *testing.T) {
	cfg := Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 42, Metric: Cosine}
	c, _ := NewCollection("vectors", cfg)
	got := c.Config()
	if !reflect.DeepEqual(got, cfg) {
		t.Errorf("Config = %+v, want %+v", got, cfg)
	}
}

// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"testing"
)

// Regression coverage for the zero-id sentinel bug: point id 0 was stored, live,
// and returned by Get, but EVERY search path dropped it, because result assembly
// used `arena.ID(slot) == 0` as "this slot holds no point". The arena's id slice
// is zero-valued for a reserved-but-unwritten slot, so a genuinely stored id 0
// was indistinguishable from an empty slot — silent, permanent data loss on the
// read path only.
//
// The strongest assertion available is used throughout: querying with a point's
// OWN vector must return that point at RANK 0.

func idZeroVec(dim int, seed uint64) []float32 {
	r := rand.New(rand.NewSource(int64(seed) + 1)) //nolint:gosec // deterministic test fixture
	v := make([]float32, dim)
	for i := range v {
		v[i] = r.Float32()
	}
	return v
}

func idZeroCfg(dim int) Config {
	return Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1}
}

// requireRankZero asserts that searching with id's own vector puts id first.
func requireRankZero(t *testing.T, res []Result, id uint64, what string) {
	t.Helper()
	for i, r := range res {
		if r.ID == id {
			if i != 0 {
				t.Errorf("%s: id %d returned at rank %d, want rank 0 (results %v)", what, id, i, resultIDs(res))
			}
			return
		}
	}
	t.Errorf("%s: id %d MISSING from search results %v", what, id, resultIDs(res))
}

// TestIDZeroSearchable is the core gate: id 0 must be reachable by dense search
// in every arrangement of ids and slots.
func TestIDZeroSearchable(t *testing.T) {
	const dim = 16

	cases := []struct {
		name string
		// insert order; the arena assigns slots sequentially, so index in this
		// slice == the slot the id lands on.
		order []uint64
		// the ids whose own-vector query must return them at rank 0.
		probe []uint64
	}{
		// id 0 alone: a single-point index whose only point is id 0.
		{name: "id0 alone", order: []uint64{0}, probe: []uint64{0}},
		// id 0 among others, id 0 first (id 0 also lands on slot 0).
		{name: "id0 first among many", order: idRange(0, 64), probe: []uint64{0, 1, 63}},
		// id 0 among others, inserted last (id 0 lands on the LAST slot).
		{name: "id0 last among many", order: append(idRange(1, 64), 0), probe: []uint64{0, 1}},
		// id 0 at a NON-ZERO slot: decoys occupy slots 0..2, id 0 lands on slot 3.
		// This is the experiment that separates "id 0 is skipped" from "slot 0 is
		// skipped" — pre-fix this still failed, proving the bug keyed off the ID.
		{name: "id0 at nonzero slot", order: append([]uint64{901, 902, 903}, idRange(0, 64)...), probe: []uint64{0, 901}},
		// A NON-ZERO id at slot 0: the counter-experiment. This passed pre-fix,
		// confirming slot 0 was never the problem.
		{name: "nonzero id at slot0", order: idRange(1, 64), probe: []uint64{1, 64}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, err := NewCollection("idzero_"+tc.name, idZeroCfg(dim))
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()

			for _, id := range tc.order {
				if err := c.Insert(id, idZeroVec(dim, id), 0, nil, nil); err != nil {
					t.Fatalf("insert id %d: %v", id, err)
				}
			}

			// Every inserted id must be reachable, not just the probed ones.
			for _, id := range tc.order {
				res, err := c.Search(idZeroVec(dim, id), 10)
				if err != nil {
					t.Fatalf("search for id %d: %v", id, err)
				}
				requireRankZero(t, res, id, "Search")
			}

			// The probe list is the strong, explicitly-named subset.
			for _, id := range tc.probe {
				res, err := c.Search(idZeroVec(dim, id), 10)
				if err != nil {
					t.Fatal(err)
				}
				requireRankZero(t, res, id, "Search(probe)")
			}

			// Storage-side invariants must hold for id 0 too: Get returns it and
			// the live count includes it.
			for _, id := range tc.order {
				if _, _, _, _, _, ok := c.Get(id); !ok {
					t.Errorf("Get(%d) reports absent", id)
				}
			}
			if got := c.Stats().Size; got != len(tc.order) {
				t.Errorf("Stats().Size = %d, want %d", got, len(tc.order))
			}
		})
	}
}

func idRange(lo, n uint64) []uint64 {
	out := make([]uint64, 0, n)
	for i := uint64(0); i < n; i++ {
		out = append(out, lo+i)
	}
	return out
}

// TestIDZeroSearchableFiltered covers the filtered lanes: the filter-first exact
// path (filterFirstKNN) and the widening graph path, both of which assembled
// results through the same sentinel.
// Each lane gets its OWN subtest and its OWN collection. They used to share one
// function, where the first failure's t.Fatalf aborted everything after it — on
// unfixed code the filter-first assertion fired and the matchingIDs assertions
// below it never ran at all, so the two matchingIDs call sites were not actually
// pinned by a red test. Keep them independent.
func TestIDZeroSearchableFiltered(t *testing.T) {
	const dim = 16

	// seed fills c with ids 901, 902, 0..63; id 0 carries a marker no other point
	// has, so a filter on it selects exactly one point.
	seed := func(t *testing.T, c *Collection) {
		t.Helper()
		for _, id := range append([]uint64{901, 902}, idRange(0, 64)...) {
			meta := Metadata{"grp": NewString("a")}
			if id%2 == 1 {
				meta = Metadata{"grp": NewString("b")}
			}
			if id == 0 {
				meta["only"] = NewString("zero")
			}
			if err := c.Insert(id, idZeroVec(dim, id), 0, meta, nil); err != nil {
				t.Fatalf("insert id %d: %v", id, err)
			}
		}
	}
	onlyZero := Filter{Op: FilterEq, Field: "only", Value: NewString("zero")}

	newColl := func(t *testing.T, name string, cfg Config) *Collection {
		t.Helper()
		c, err := NewCollection(name, cfg)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = c.Close() })
		seed(t, c)
		return c
	}

	ivfCfg := func() Config {
		cfg := idZeroCfg(dim)
		cfg.IndexType = IndexIVF
		cfg.IVFNlist = 4
		cfg.IVFNprobe = 4
		cfg.IVFTrainThreshold = 32
		return cfg
	}

	// The wide filtered lane: a group filter that id 0 satisfies along with many
	// others, so the search widens through the graph rather than going exact.
	t.Run("wide filtered lane", func(t *testing.T) {
		c := newColl(t, "idzero_filtered_wide", idZeroCfg(dim))
		res, err := c.SearchFiltered(idZeroVec(dim, 0), 10, Filter{Op: FilterEq, Field: "grp", Value: NewString("a")})
		if err != nil {
			t.Fatal(err)
		}
		requireRankZero(t, res, 0, "SearchFiltered")
	})

	// The filter-first exact lane (filterFirstKNN): a filter narrow enough that
	// the payload index answers it directly and the candidate set is just id 0.
	t.Run("filter-first lane", func(t *testing.T) {
		c := newColl(t, "idzero_filtered_narrow", idZeroCfg(dim))
		res, err := c.SearchFiltered(idZeroVec(dim, 0), 10, onlyZero)
		if err != nil {
			t.Fatal(err)
		}
		if len(res) != 1 || res[0].ID != 0 {
			t.Fatalf("filter-first lane: got %v, want exactly [0]", resultIDs(res))
		}
	})

	// matchingIDs (reached via DeleteByFilter) on BOTH index families. Its
	// payload-index fast path carried the sentinel while its idMap fallback did
	// not, so the same filter selected different id sets depending on which lane
	// ran — the fast path silently under-deleted. hnsw and ivf have separate
	// copies of this function (rag.go and ivf.go), so both are asserted.
	for _, tc := range []struct {
		name string
		cfg  Config
	}{
		{"hnsw", idZeroCfg(dim)},
		{"ivf", ivfCfg()},
	} {
		t.Run("matchingIDs/"+tc.name, func(t *testing.T) {
			c := newColl(t, "idzero_matchingids_"+tc.name, tc.cfg)
			n, err := c.DeleteByFilter(onlyZero)
			if err != nil {
				t.Fatal(err)
			}
			if n != 1 {
				t.Errorf("DeleteByFilter(only id 0) removed %d points, want 1", n)
			}
			if _, _, _, _, _, ok := c.Get(0); ok {
				t.Error("id 0 still present after DeleteByFilter matched it")
			}
		})
	}
}

// TestIDZeroSearchableHybrid covers the sparse lane and full-text lane, which
// assemble results through their own copies of the sentinel.
func TestIDZeroSearchableHybrid(t *testing.T) {
	const dim = 8
	cfg := idZeroCfg(dim)
	cfg.FullText = &FullTextConfig{}
	c, err := NewCollection("idzero_hybrid", cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	// id 0 owns a unique sparse term (7) and a unique text token ("zeroterm").
	if err := c.Upsert(0, idZeroVec(dim, 0), "zeroterm alpha", 0, nil,
		&SparseVector{Indices: []uint32{7}, Values: []float32{9}}); err != nil {
		t.Fatal(err)
	}
	for id := uint64(1); id < 32; id++ {
		if err := c.Upsert(id, idZeroVec(dim, id), "alpha beta", 0, nil,
			&SparseVector{Indices: []uint32{1}, Values: []float32{0.5}}); err != nil {
			t.Fatal(err)
		}
	}

	// Sparse lane: a query on term 7 must surface id 0.
	res, err := c.HybridSearch(idZeroVec(dim, 0), SparseVector{Indices: []uint32{7}, Values: []float32{1}}, 10, HybridOpts{})
	if err != nil {
		t.Fatal(err)
	}
	requireRankZero(t, res, 0, "HybridSearch")

	// Full-text lane: "zeroterm" appears only in id 0's content.
	docs, err := c.SearchText("zeroterm", 10, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 1 || docs[0].ID != 0 {
		ids := make([]uint64, 0, len(docs))
		for _, d := range docs {
			ids = append(ids, d.ID)
		}
		t.Fatalf("SearchText(zeroterm): got %v, want exactly [0]", ids)
	}
}

// TestIDZeroSearchableIVF checks the sibling IVF index, which carries its own
// copy of the same result-assembly sentinel.
func TestIDZeroSearchableIVF(t *testing.T) {
	const dim = 16
	cfg := idZeroCfg(dim)
	cfg.IndexType = IndexIVF
	cfg.IVFNlist = 8
	cfg.IVFNprobe = 8
	cfg.IVFTrainThreshold = 64

	for _, tc := range []struct {
		name  string
		order []uint64
	}{
		{"id0 at slot0", idRange(0, 256)},
		{"id0 at nonzero slot", append([]uint64{901, 902, 903}, idRange(0, 256)...)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, err := NewCollection("idzero_ivf_"+tc.name, cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			for _, id := range tc.order {
				if err := c.Insert(id, idZeroVec(dim, id), 0, nil, nil); err != nil {
					t.Fatal(err)
				}
			}
			res, err := c.Search(idZeroVec(dim, 0), 10)
			if err != nil {
				t.Fatal(err)
			}
			requireRankZero(t, res, 0, "IVF Search")
			if _, _, _, _, _, ok := c.Get(0); !ok {
				t.Error("IVF: Get(0) reports absent")
			}
			if got := c.Stats().Size; got != len(tc.order) {
				t.Errorf("IVF: Stats().Size = %d, want %d", got, len(tc.order))
			}
		})
	}
}

// TestIDZeroSearchableVamana covers the Vamana index, which shares the hnsw
// result-assembly code.
func TestIDZeroSearchableVamana(t *testing.T) {
	const dim = 16
	cfg := idZeroCfg(dim)
	cfg.IndexType = IndexVamana
	c, err := NewCollection("idzero_vamana", cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	for _, id := range idRange(0, 128) {
		if err := c.Insert(id, idZeroVec(dim, id), 0, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	res, err := c.Search(idZeroVec(dim, 0), 10)
	if err != nil {
		t.Fatal(err)
	}
	requireRankZero(t, res, 0, "Vamana Search")
}

// TestIDZeroSearchableNamed covers the named-vector family, whose per-space
// sub-indexes reuse the same result assembly.
func TestIDZeroSearchableNamed(t *testing.T) {
	nc, err := NewNamedCollection("default/idzero_named", map[string]NamedVectorParams{
		"title": {Dim: 8, Metric: L2},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer nc.Close()

	for _, id := range idRange(0, 64) {
		if err := nc.Insert(id, map[string][]float32{"title": idZeroVec(8, id)}, nil, 0); err != nil {
			t.Fatal(err)
		}
	}
	res, err := nc.SearchNamed("title", idZeroVec(8, 0), 10, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	requireRankZero(t, res, 0, "SearchNamed")
}

// TestIDZeroSearchableMultiVector covers the multi-vector family.
func TestIDZeroSearchableMultiVector(t *testing.T) {
	const dim = 8
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	for _, id := range idRange(0, 64) {
		toks := [][]float32{idZeroVec(dim, id), idZeroVec(dim, id+1000)}
		if err := m.Add(id, toks, nil); err != nil {
			t.Fatal(err)
		}
	}
	got, err := m.Search([][]float32{idZeroVec(dim, 0)}, 5, MultiSearchOpts{CandidatesPerToken: 200})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, r := range got {
		if r.ID == 0 {
			found = true
			break
		}
	}
	if !found {
		ids := make([]uint64, 0, len(got))
		for _, r := range got {
			ids = append(ids, r.ID)
		}
		t.Fatalf("MultiVector Search: id 0 missing from %v", ids)
	}
}

// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"testing"
)

// TestWALPayloadCrashReplay is the landmine-#2 test: a dense WAL-mode collection's
// in-place payload mutation must survive a crash (reopen from checkpoint + WAL
// tail) via the walSetPayload record — AND a filtered search on the restored
// payload must work (proving replay reindexed the payload index). It exercises
// each of the four ops, taking the LAST mutation as the durable state.
//
// As in TestReindexCorrectness, the filtered-search probe must take the
// FILTER-FIRST path to be meaningful: we SEED ~10 OTHER points carrying each
// probed field-value (b==2, c==3, a==1) so the posting list exists after restore
// and candidates returns ok=true. A positive probe then checks id=1 is AMONG the
// filter-first candidates (it would be missing if replay failed to reindex); a
// negative probe checks id=1 is NOT among them (a stale entry would surface).
func TestWALPayloadCrashReplay(t *testing.T) {
	cases := []struct {
		name     string
		mutate   func(c *Collection) error
		wantMeta Metadata // expected restored payload
		probe    *Filter  // filtered search that must return id=1 after restore
		probeNeg *Filter  // filtered search that must return NOTHING after restore
	}{
		{
			name:     "merge",
			mutate:   func(c *Collection) error { return c.SetPayload(1, Metadata{"b": NewInt(2)}, nil) },
			wantMeta: Metadata{"a": NewInt(1), "b": NewInt(2)},
			probe:    &Filter{Op: FilterEq, Field: "b", Value: NewInt(2)},
		},
		{
			name:     "overwrite",
			mutate:   func(c *Collection) error { return c.OverwritePayload(1, Metadata{"c": NewInt(3)}, nil) },
			wantMeta: Metadata{"c": NewInt(3)},
			probe:    &Filter{Op: FilterEq, Field: "c", Value: NewInt(3)},
			probeNeg: &Filter{Op: FilterEq, Field: "a", Value: NewInt(1)},
		},
		{
			name:     "delete-keys",
			mutate:   func(c *Collection) error { return c.DeletePayloadKeys(1, []string{"a"}) },
			wantMeta: nil,
			probeNeg: &Filter{Op: FilterEq, Field: "a", Value: NewInt(1)},
		},
		{
			name:     "clear",
			mutate:   func(c *Collection) error { return c.ClearPayload(1) },
			wantMeta: nil,
			probeNeg: &Filter{Op: FilterEq, Field: "a", Value: NewInt(1)},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			cs, err := OpenCollectionStore(dir)
			if err != nil {
				t.Fatal(err)
			}
			cfg := walCfg()
			cfg.Dim = 4
			if err := cs.CreateCollection("docs", cfg); err != nil {
				t.Fatal(err)
			}
			// Insert id=1 with payload {a:1}, plus filler so search has a graph.
			// id=1's vector is closest to the query.
			if err := cs.Insert("docs", 1, []float32{1, 0, 0, 0}, 0, Metadata{"a": NewInt(1)}, nil); err != nil {
				t.Fatal(err)
			}
			for i := uint64(2); i <= 10; i++ {
				if err := cs.Insert("docs", i, []float32{0, float32(i), 0, 0}, 0, nil, nil); err != nil {
					t.Fatal(err)
				}
			}
			// SEED ~10 OTHER points per probed value (b==2, c==3, a==1) so those
			// posting lists exist after restore → candidates returns ok=true and the
			// probe takes the FILTER-FIRST (index) path. Without these seeders the
			// probed value would live on id=1 alone, candidates would return ok=false,
			// and the probe would fall to graph traversal — passing even if replay
			// failed to reindex (defeating the landmine guard). Their vectors are far
			// from the query so they never out-rank id=1. ids 100..129.
			seedID := uint64(100)
			for _, kv := range []struct {
				field string
				val   Value
			}{
				{"b", NewInt(2)},
				{"c", NewInt(3)},
				{"a", NewInt(1)},
			} {
				for j := 0; j < 10; j++ {
					if err := cs.Insert("docs", seedID, []float32{0, 0, float32(seedID), 0}, 0, Metadata{kv.field: kv.val}, nil); err != nil {
						t.Fatal(err)
					}
					seedID++
				}
			}
			c, ok := cs.Get("docs")
			if !ok {
				t.Fatal("collection missing")
			}
			if err := tc.mutate(c); err != nil {
				t.Fatalf("mutate: %v", err)
			}
			// Deliberately DO NOT Flush — only the WAL holds the payload mutation.
			_ = cs.Close()

			// Reopen: openPersist(checkpoint) + WAL replay (incl. walSetPayload).
			cs2, err := OpenCollectionStore(dir)
			if err != nil {
				t.Fatalf("reopen: %v", err)
			}
			defer func() { _ = cs2.Close() }()
			c2, ok := cs2.Get("docs")
			if !ok {
				t.Fatal("collection missing after reopen")
			}
			_, meta, _, _, _, ok := c2.Get(1)
			if !ok {
				t.Fatal("Get(1) ok=false after restore")
			}
			if !metaEqual(meta, tc.wantMeta) {
				t.Fatalf("restored payload = %v, want %v (walSetPayload not replayed)", meta, tc.wantMeta)
			}

			q := []float32{1, 0, 0, 0}
			// Confirm the probe takes the filter-first (index) path: the seeders make
			// candidates(probe) ok=true. We reach into the underlying hnsw to assert it.
			h2 := c2.idx.(*hnsw)
			if tc.probe != nil {
				if _, ok := h2.payloadIdx.candidates(*tc.probe, h2.filterFirstThreshold()); !ok {
					t.Fatalf("probe %v: candidates ok=false (graph fallback) — probe would not exercise filter-first", tc.probe)
				}
				res, err := c2.SearchFiltered(q, 5, *tc.probe)
				if err != nil {
					t.Fatalf("probe search: %v", err)
				}
				if !containsID(res, 1) {
					t.Fatalf("probe %v after restore = %v, want to contain id=1 (reindex not rebuilt on replay)", tc.probe, resultIDs(res))
				}
			}
			if tc.probeNeg != nil {
				if _, ok := h2.payloadIdx.candidates(*tc.probeNeg, h2.filterFirstThreshold()); !ok {
					t.Fatalf("probeNeg %v: candidates ok=false (graph fallback) — probe would not exercise filter-first", tc.probeNeg)
				}
				res, err := c2.SearchFiltered(q, 5, *tc.probeNeg)
				if err != nil {
					t.Fatalf("probeNeg search: %v", err)
				}
				if containsID(res, 1) {
					t.Fatalf("probeNeg %v after restore = %v, want NOT to contain id=1 (stale index entry surfaced)", tc.probeNeg, resultIDs(res))
				}
			}
		})
	}
}

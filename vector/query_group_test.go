// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"testing"
)

// newGroupQueryCorpus builds a Collection of 6 chunks across 3 documents laid out
// on the x-axis so distance from the origin strictly increases with id
// (id1<id2<…<id6); doc i owns ids {2i-1, 2i}. Mirrors buildGroupIndex (the groups
// oracle corpus) but on a *Collection so the Query API entry point can group it.
func newGroupQueryCorpus(t *testing.T) *Collection {
	t.Helper()
	c, err := NewCollection("gq", Config{Dim: 3, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1})
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	for i := 1; i <= 6; i++ {
		doc := int64((i + 1) / 2) // 1,1,2,2,3,3
		vec := []float32{float32(i), 0, 0}
		if err := c.Insert(uint64(i), vec, 0, Metadata{"doc": NewInt(doc)}, nil); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	return c
}

// denseLeaf builds a flat dense prefetch leaf carrying the query vector.
func denseLeaf(vec []float32) QueryLeaf {
	return QueryLeaf{Kind: LeafDense, Dense: vec, ScoreDesc: false}
}

// TestQueryGroupedEqualsSearchGroups is the INDEPENDENT-oracle proof: a grouped
// dense query (FUSION single dense lane, and RERANK with a dense root) MUST equal
// the standalone (*Collection).SearchGroups over the SAME root vector with the SAME
// k/GroupSize. SearchGroups is computed independently (it runs its own dense KNN +
// GroupDocuments), so this is a real cross-check, not GroupDocuments-vs-itself.
func TestQueryGroupedEqualsSearchGroups(t *testing.T) {
	query := []float32{0, 0, 0}
	const groupsK = 3
	const groupSize = 2

	for _, tc := range []struct {
		name string
		spec QuerySpec
	}{
		{
			name: "fusion-single-dense-lane",
			spec: QuerySpec{
				Mode:      ModeFusion,
				Prefetch:  []QuerySource{LeafSource(denseLeaf(query))},
				K:         groupsK,
				GroupBy:   "doc",
				GroupSize: groupSize,
			},
		},
		{
			name: "rerank-dense-root",
			spec: QuerySpec{
				Mode:      ModeRerank,
				Root:      denseLeaf(query),
				Prefetch:  []QuerySource{LeafSource(denseLeaf(query))},
				K:         groupsK,
				GroupBy:   "doc",
				GroupSize: groupSize,
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := newGroupQueryCorpus(t)

			qr, err := c.Query(tc.spec)
			if err != nil {
				t.Fatalf("grouped Query: %v", err)
			}
			if qr.Groups == nil {
				t.Fatal("grouped Query returned nil Groups")
			}

			want, err := c.SearchGroups(query, groupsK, GroupOpts{GroupBy: "doc", GroupSize: groupSize})
			if err != nil {
				t.Fatalf("SearchGroups oracle: %v", err)
			}

			if !sameGroups(want, qr.Groups) {
				t.Fatalf("grouped Query != SearchGroups oracle\nwant: %+v\ngot:  %+v", want, qr.Groups)
			}
		})
	}
}

// TestQueryGroupedTopKByBestMember confirms the top-K groups are ranked by their
// best (nearest) member and group_size is respected. doc1 (ids 1,2) is nearest,
// then doc2 (3,4), then doc3 (5,6).
func TestQueryGroupedTopKByBestMember(t *testing.T) {
	c := newGroupQueryCorpus(t)
	query := []float32{0, 0, 0}

	qr, err := c.Query(QuerySpec{
		Mode:      ModeFusion,
		Prefetch:  []QuerySource{LeafSource(denseLeaf(query))},
		K:         2, // top-2 groups
		GroupBy:   "doc",
		GroupSize: 2,
	})
	if err != nil {
		t.Fatalf("grouped Query: %v", err)
	}
	if len(qr.Groups) != 2 {
		t.Fatalf("want 2 groups, got %d: %+v", len(qr.Groups), qr.Groups)
	}
	// Groups ranked by best member: doc1 then doc2.
	if k, _ := groupKeyString(qr.Groups[0].Key); k != "i1" {
		t.Fatalf("group0 key = %q, want doc 1", k)
	}
	if k, _ := groupKeyString(qr.Groups[1].Key); k != "i2" {
		t.Fatalf("group1 key = %q, want doc 2", k)
	}
	// group_size=2 respected; doc1 hits are ids 1,2 best-first.
	if len(qr.Groups[0].Hits) != 2 || qr.Groups[0].Hits[0].ID != 1 || qr.Groups[0].Hits[1].ID != 2 {
		t.Fatalf("group0 hits = %+v, want ids [1 2]", qr.Groups[0].Hits)
	}
}

// TestQueryGroupedGroupSizeRespected confirms group_size caps the per-group hits.
func TestQueryGroupedGroupSizeRespected(t *testing.T) {
	c := newGroupQueryCorpus(t)
	query := []float32{0, 0, 0}

	qr, err := c.Query(QuerySpec{
		Mode:      ModeFusion,
		Prefetch:  []QuerySource{LeafSource(denseLeaf(query))},
		K:         3,
		GroupBy:   "doc",
		GroupSize: 1, // one representative chunk per doc
	})
	if err != nil {
		t.Fatalf("grouped Query: %v", err)
	}
	for i, g := range qr.Groups {
		if len(g.Hits) != 1 {
			t.Fatalf("group %d has %d hits, want 1 (group_size cap)", i, len(g.Hits))
		}
	}
}

// TestQueryNoGroupByUnchanged confirms a query WITHOUT group_by takes the flat path
// and returns NO groups — the flat Fused/Lanes path is byte/behaviour-identical to a
// non-grouped query (the #1 invariant: group fields empty ⇒ existing path).
func TestQueryNoGroupByUnchanged(t *testing.T) {
	c := newGroupQueryCorpus(t)
	query := []float32{0, 0, 0}
	spec := QuerySpec{
		Mode:     ModeFusion,
		Prefetch: []QuerySource{LeafSource(denseLeaf(query))},
		K:        3,
		// GroupBy empty
	}
	qr, err := c.Query(spec)
	if err != nil {
		t.Fatalf("flat Query: %v", err)
	}
	if qr.Groups != nil {
		t.Fatalf("flat query returned Groups (want nil): %+v", qr.Groups)
	}
	if len(qr.Fused) == 0 {
		t.Fatal("flat query returned empty Fused")
	}
}

// TestNamedGroupedQueryFailLoud / TestMVGroupedQueryFailLoud confirm a grouped query
// on the named / MV families is rejected fail-loud (dense-only v1).
func TestNamedGroupedQueryFailLoud(t *testing.T) {
	nc := newNamedQueryCorpus(t) // 4-dim "title" / "image" dense spaces + "terms" sparse
	spec := QuerySpec{
		Mode:      ModeFusion,
		Prefetch:  []QuerySource{LeafSource(QueryLeaf{Kind: LeafDense, Dense: []float32{1, 0, 0, 0}, Space: "title"})},
		K:         3,
		GroupBy:   "k",
		GroupSize: 1,
	}
	if _, err := nc.Query(spec); !errors.Is(err, ErrQueryGroupNotDense) {
		t.Fatalf("named grouped query err = %v, want ErrQueryGroupNotDense", err)
	}
}

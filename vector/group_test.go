// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"testing"
)

func TestGroupDocumentsMatchesSearchGroups(t *testing.T) {
	h := buildGroupIndex(t)
	query := []float32{0, 0, 0}
	opts := GroupOpts{GroupBy: "doc", GroupSize: 2}

	want, err := h.SearchGroups(query, 3, opts)
	if err != nil {
		t.Fatal(err)
	}
	cands, err := h.GroupCandidates(query, opts)
	if err != nil {
		t.Fatal(err)
	}
	got := GroupDocuments(cands, opts, 3)
	if !sameGroups(want, got) {
		t.Fatalf("GroupDocuments(GroupCandidates) != SearchGroups\nwant: %+v\ngot:  %+v", want, got)
	}
}

func sameGroups(a, b []Group) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		aks, aok := groupKeyString(a[i].Key)
		bks, bok := groupKeyString(b[i].Key)
		if aok != bok || aks != bks {
			return false
		}
		if len(a[i].Hits) != len(b[i].Hits) {
			return false
		}
		for j := range a[i].Hits {
			if a[i].Hits[j].ID != b[i].Hits[j].ID {
				return false
			}
		}
	}
	return true
}

// buildGroupIndex inserts 6 chunks across 3 documents (doc 1/2/3, two chunks
// each) laid out on the x-axis so distance from the origin strictly increases
// with id: id1<id2<id3<id4<id5<id6. doc i owns ids {2i-1, 2i}.
func buildGroupIndex(t *testing.T) *hnsw {
	t.Helper()
	h, err := newHNSW(Config{Dim: 3, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 6; i++ {
		doc := int64((i + 1) / 2) // 1,1,2,2,3,3
		vec := []float32{float32(i), 0, 0}
		meta := withContent(Metadata{"doc": NewInt(doc)}, "chunk")
		if _, _, err := h.Insert(uint64(i), vec, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	return h
}

func TestSearchGroupsTopKByBestMember(t *testing.T) {
	h := buildGroupIndex(t)
	q := []float32{0, 0, 0}

	groups, err := h.SearchGroups(q, 2, GroupOpts{GroupBy: "doc", GroupSize: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 2 {
		t.Fatalf("got %d groups, want 2", len(groups))
	}
	// Ranked by best member: doc 1 (best id1) then doc 2 (best id3).
	if groups[0].Key.Int != 1 || groups[1].Key.Int != 2 {
		t.Errorf("group keys = %d,%d, want 1,2", groups[0].Key.Int, groups[1].Key.Int)
	}
	if len(groups[0].Hits) != 1 || groups[0].Hits[0].ID != 1 {
		t.Errorf("group0 hits = %+v, want [id1]", groups[0].Hits)
	}
	if len(groups[1].Hits) != 1 || groups[1].Hits[0].ID != 3 {
		t.Errorf("group1 hits = %+v, want [id3]", groups[1].Hits)
	}
	// Content rides along (RAG path), stripped reserved field absent.
	if groups[0].Hits[0].Content != "chunk" {
		t.Errorf("hit content = %q, want chunk", groups[0].Hits[0].Content)
	}
	if _, leaked := groups[0].Hits[0].Metadata[contentField]; leaked {
		t.Error("reserved content field leaked into returned metadata")
	}
}

func TestSearchGroupsGroupSizeFillsBestFirst(t *testing.T) {
	h := buildGroupIndex(t)
	groups, err := h.SearchGroups([]float32{0, 0, 0}, 2, GroupOpts{GroupBy: "doc", GroupSize: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 2 {
		t.Fatalf("got %d groups, want 2", len(groups))
	}
	want := [][]uint64{{1, 2}, {3, 4}}
	for gi, g := range groups {
		if len(g.Hits) != 2 {
			t.Fatalf("group %d has %d hits, want 2", gi, len(g.Hits))
		}
		for hi, h := range g.Hits {
			if h.ID != want[gi][hi] {
				t.Errorf("group %d hit %d = id%d, want id%d", gi, hi, h.ID, want[gi][hi])
			}
		}
	}
}

// A hit lacking the group field is skipped, even when it is the nearest vector.
func TestSearchGroupsSkipsMissingField(t *testing.T) {
	h := buildGroupIndex(t)
	// id99 is the closest vector to the origin but has no "doc" field.
	if _, _, err := h.Insert(99, []float32{0.5, 0, 0}, 0, Metadata{"other": NewInt(7)}, nil, nil, CASCond{}); err != nil {
		t.Fatal(err)
	}
	groups, err := h.SearchGroups([]float32{0, 0, 0}, 2, GroupOpts{GroupBy: "doc", GroupSize: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 2 || groups[0].Key.Int != 1 || groups[1].Key.Int != 2 {
		t.Fatalf("groups = %+v, want doc 1,2 (id99 skipped)", groups)
	}
	for _, g := range groups {
		for _, hit := range g.Hits {
			if hit.ID == 99 {
				t.Error("ungrouped id99 leaked into results")
			}
		}
	}
}

func TestSearchGroupsFilterApplies(t *testing.T) {
	h := buildGroupIndex(t)
	// Restrict to doc >= 2: doc 1 must drop out; best two groups become 2,3.
	f := Filter{Op: FilterGte, Field: "doc", Value: NewInt(2)}
	groups, err := h.SearchGroups([]float32{0, 0, 0}, 3, GroupOpts{GroupBy: "doc", GroupSize: 1, Filter: f})
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 2 || groups[0].Key.Int != 2 || groups[1].Key.Int != 3 {
		t.Fatalf("groups = %+v, want doc 2,3", groups)
	}
}

// A deleted hit never surfaces; its group falls back to the next-best chunk.
func TestSearchGroupsExcludesDeleted(t *testing.T) {
	h := buildGroupIndex(t)
	h.Delete(1, CASCond{}) // doc 1's best chunk
	groups, err := h.SearchGroups([]float32{0, 0, 0}, 1, GroupOpts{GroupBy: "doc", GroupSize: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 1 || groups[0].Key.Int != 1 || groups[0].Hits[0].ID != 2 {
		t.Fatalf("groups = %+v, want doc 1 represented by id2", groups)
	}
}

func TestSearchGroupsValidation(t *testing.T) {
	h := buildGroupIndex(t)
	if _, err := h.SearchGroups([]float32{0, 0, 0}, 2, GroupOpts{}); !errors.Is(err, ErrEmptyGroupBy) {
		t.Errorf("empty GroupBy = %v, want ErrEmptyGroupBy", err)
	}
	if _, err := h.SearchGroups([]float32{0, 0}, 2, GroupOpts{GroupBy: "doc"}); !errors.Is(err, ErrDimMismatch) {
		t.Errorf("wrong dim = %v, want ErrDimMismatch", err)
	}
	if g, err := h.SearchGroups([]float32{0, 0, 0}, 0, GroupOpts{GroupBy: "doc"}); err != nil || g != nil {
		t.Errorf("k=0 = (%v,%v), want (nil,nil)", g, err)
	}
}

func TestGroupKeyString(t *testing.T) {
	// Kind-tagged so an int and the string of the same digits never collide.
	si, _ := groupKeyString(NewString("1"))
	ii, _ := groupKeyString(NewInt(1))
	if si == ii {
		t.Errorf("string and int keys collide: %q == %q", si, ii)
	}
	if _, ok := groupKeyString(NewInts([]int64{1, 2})); ok {
		t.Error("list value should not be groupable")
	}
	if _, ok := groupKeyString(Value{}); ok {
		t.Error("none value should not be groupable")
	}
}

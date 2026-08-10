// SPDX-License-Identifier: Apache-2.0

package cluster

import "testing"

func TestMergeTopKAscending(t *testing.T) {
	// Three partitions, each already sorted ascending by Dist.
	parts := [][]Scored{
		{{ID: 1, Dist: 0.1}, {ID: 2, Dist: 0.4}},
		{{ID: 3, Dist: 0.2}, {ID: 4, Dist: 0.5}},
		{{ID: 5, Dist: 0.3}},
	}
	got := MergeTopK(parts, 3, func(a, b Scored) bool { return a.Dist < b.Dist })
	wantIDs := []uint64{1, 3, 5}
	if len(got) != 3 {
		t.Fatalf("len=%d, want 3", len(got))
	}
	for i, w := range wantIDs {
		if got[i].ID != w {
			t.Errorf("pos %d ID=%d, want %d", i, got[i].ID, w)
		}
	}
}

func TestMergeTopKAllFromOnePartition(t *testing.T) {
	parts := [][]Scored{
		{{ID: 1, Dist: 0.1}, {ID: 2, Dist: 0.2}, {ID: 3, Dist: 0.3}},
		{{ID: 9, Dist: 0.9}},
	}
	got := MergeTopK(parts, 3, func(a, b Scored) bool { return a.Dist < b.Dist })
	if len(got) != 3 || got[0].ID != 1 || got[2].ID != 3 {
		t.Fatalf("got %+v, want IDs 1,2,3", got)
	}
}

func TestMergeTopKEmptyParts(t *testing.T) {
	if got := MergeTopK(nil, 5, func(a, b Scored) bool { return a.Dist < b.Dist }); len(got) != 0 {
		t.Fatalf("nil parts -> %d results, want 0", len(got))
	}
}

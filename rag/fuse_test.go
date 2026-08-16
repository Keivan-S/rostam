// SPDX-License-Identifier: Apache-2.0

package rag

import "testing"

func h(src string, idx int, score float32) Hit {
	return Hit{Content: src, Source: src, Index: idx, Score: score}
}

func keysOf(hits []Hit) []string {
	out := make([]string, len(hits))
	for i, x := range hits {
		out[i] = x.Source
	}
	return out
}

func TestFuseRRFRewardsAgreement(t *testing.T) {
	// "b" is rank 2 in each lane; "a" tops dense; "c" tops bm25. RRF should rank
	// the doc that appears well in BOTH ("b") at or above lane-only leaders.
	dense := []Hit{h("a", 0, 0), h("b", 1, 0), h("x", 2, 0)}
	bm25 := []Hit{h("c", 0, 0), h("b", 1, 0), h("y", 2, 0)}
	got := fuse(dense, bm25, 3, -1)
	if len(got) != 3 {
		t.Fatalf("want 3, got %d: %v", len(got), keysOf(got))
	}
	if got[0].Source != "b" {
		t.Fatalf("RRF should rank the agreed doc 'b' first, got %v", keysOf(got))
	}
}

func TestFuseWeightedAlphaExtremes(t *testing.T) {
	dense := []Hit{h("d0", 0, 0), h("d1", 1, 0)}
	bm25 := []Hit{h("t0", 0, 0), h("t1", 1, 0)}
	if got := fuse(dense, bm25, 1, 1.0); got[0].Source != "d0" {
		t.Fatalf("alpha=1 should be dense-led, got %v", keysOf(got))
	}
	if got := fuse(dense, bm25, 1, 0.0); got[0].Source != "t0" {
		t.Fatalf("alpha=0 should be bm25-led, got %v", keysOf(got))
	}
}

func TestFuseDedupsSameChunk(t *testing.T) {
	dense := []Hit{h("a", 0, 0), h("a", 0, 0)} // same source#index appears twice across lanes
	bm25 := []Hit{h("a", 0, 0)}
	got := fuse(dense, bm25, 5, -1)
	if len(got) != 1 {
		t.Fatalf("same chunk must fuse to one entry, got %d: %v", len(got), keysOf(got))
	}
}

func TestFuseTopKAndDeterminism(t *testing.T) {
	dense := []Hit{h("a", 0, 0), h("b", 1, 0), h("c", 2, 0)}
	bm25 := []Hit{h("c", 0, 0), h("b", 1, 0), h("a", 2, 0)}
	a := fuse(dense, bm25, 2, -1)
	b := fuse(dense, bm25, 2, -1)
	if len(a) != 2 {
		t.Fatalf("want top-2, got %d", len(a))
	}
	if keysOf(a)[0] != keysOf(b)[0] || keysOf(a)[1] != keysOf(b)[1] {
		t.Fatalf("fuse must be deterministic: %v vs %v", keysOf(a), keysOf(b))
	}
}

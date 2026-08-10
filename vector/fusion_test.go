// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"testing"
)

func approxEq(a, b float32) bool {
	return math.Abs(float64(a-b)) < 1e-5
}

func TestFuseRRF(t *testing.T) {
	// dense ranking: id 1 (rank0), id 2 (rank1), id 3 (rank2)
	dense := []Result{{ID: 1, Distance: 0.1}, {ID: 2, Distance: 0.2}, {ID: 3, Distance: 0.3}}
	// sparse ranking: id 3 (rank0), id 1 (rank1), id 4 (rank2)
	sparse := []Result{{ID: 3, Score: 9}, {ID: 1, Score: 5}, {ID: 4, Score: 1}}

	rrfK := 60
	got := fuseRRF(dense, sparse, 10, rrfK)

	// Expected RRF scores (1-based rank):
	// id1: 1/61 (dense r0) + 1/62 (sparse r1)
	// id2: 1/62 (dense r1)
	// id3: 1/63 (dense r2) + 1/61 (sparse r0)
	// id4: 1/63 (sparse r2)
	want := map[uint64]float32{
		1: 1.0/61 + 1.0/62,
		2: 1.0 / 62,
		3: 1.0/63 + 1.0/61,
		4: 1.0 / 63,
	}
	scoreByID := map[uint64]float32{}
	for _, r := range got {
		scoreByID[r.ID] = r.Score
	}
	for id, w := range want {
		if !approxEq(scoreByID[id], w) {
			t.Errorf("id %d RRF score = %v, want %v", id, scoreByID[id], w)
		}
	}
	// id1 and id3 are the top two (both appear in both lanes near the top).
	// id1: 0.0164+0.0161=0.0325; id3: 0.0159+0.0164=0.0323 → id1 first.
	if got[0].ID != 1 || got[1].ID != 3 {
		t.Errorf("RRF order top2 = [%d,%d], want [1,3]", got[0].ID, got[1].ID)
	}
	// dense distance is carried for docs in the dense lane; 0 for sparse-only id4.
	if scoreByID[4] == 0 {
		t.Error("id4 (sparse-only) should have a nonzero fused score")
	}
	for _, r := range got {
		if r.ID == 4 && r.Distance != 0 {
			t.Errorf("sparse-only id4 Distance = %v, want 0", r.Distance)
		}
		if r.ID == 1 && r.Distance != 0.1 {
			t.Errorf("id1 Distance = %v, want 0.1 (from dense lane)", r.Distance)
		}
	}
}

func TestFuseRRFTopKTruncation(t *testing.T) {
	dense := []Result{{ID: 1, Distance: 0.1}, {ID: 2, Distance: 0.2}, {ID: 3, Distance: 0.3}}
	got := fuseRRF(dense, nil, 2, 60)
	if len(got) != 2 {
		t.Fatalf("len = %d, want 2 (k truncation)", len(got))
	}
	if got[0].ID != 1 || got[1].ID != 2 {
		t.Errorf("order = [%d,%d], want [1,2]", got[0].ID, got[1].ID)
	}
}

func TestFuseWeighted(t *testing.T) {
	// dense distances 0.0, 0.5, 1.0 → inverted-normalized 1.0, 0.5, 0.0
	dense := []Result{{ID: 1, Distance: 0.0}, {ID: 2, Distance: 0.5}, {ID: 3, Distance: 1.0}}
	// sparse scores 10, 5, 0 → normalized 1.0, 0.5, 0.0
	sparse := []Result{{ID: 3, Score: 10}, {ID: 2, Score: 5}, {ID: 1, Score: 0}}

	got := fuseWeighted(dense, sparse, 10, 0.5)
	score := map[uint64]float32{}
	for _, r := range got {
		score[r.ID] = r.Score
	}
	// id1: 0.5*1.0 + 0.5*0.0 = 0.5
	// id2: 0.5*0.5 + 0.5*0.5 = 0.5
	// id3: 0.5*0.0 + 0.5*1.0 = 0.5
	for id := uint64(1); id <= 3; id++ {
		if !approxEq(score[id], 0.5) {
			t.Errorf("id %d weighted score = %v, want 0.5", id, score[id])
		}
	}
}

func TestFuseWeightedAlphaExtremes(t *testing.T) {
	dense := []Result{{ID: 1, Distance: 0.0}, {ID: 2, Distance: 1.0}}
	sparse := []Result{{ID: 2, Score: 10}, {ID: 1, Score: 0}}

	// alpha=1 → pure dense: id1 (dist 0 → norm 1) beats id2.
	d := fuseWeighted(dense, sparse, 10, 1.0)
	if d[0].ID != 1 {
		t.Errorf("alpha=1 top = %d, want 1 (pure dense)", d[0].ID)
	}
	// alpha=0 → pure sparse: id2 (score 10 → norm 1) beats id1.
	s := fuseWeighted(dense, sparse, 10, 0.0)
	if s[0].ID != 2 {
		t.Errorf("alpha=0 top = %d, want 2 (pure sparse)", s[0].ID)
	}
}

func TestNormalizeDenseSingle(t *testing.T) {
	// Single result (zero span) → relevance 1.0.
	n := normalizeDense([]Result{{ID: 1, Distance: 0.7}})
	if n[1] != 1.0 {
		t.Errorf("single dense norm = %v, want 1.0", n[1])
	}
}

// sameFused reports whether two fused result slices are equivalent: same length,
// same ID set, and each ID's Score matches within 1e-6. Tie-break order is NOT
// required to match (map-based comparison).
func sameFused(a, b []Result) bool {
	if len(a) != len(b) {
		return false
	}
	ma := map[uint64]float32{}
	for _, r := range a {
		ma[r.ID] = r.Score
	}
	for _, r := range b {
		s, ok := ma[r.ID]
		if !ok {
			return false
		}
		d := s - r.Score
		if d < 0 {
			d = -d
		}
		if d > 1e-6 {
			return false
		}
	}
	return true
}

func TestFuseMatchesInternalFusion(t *testing.T) {
	dense := []Result{{ID: 1, Distance: 0.1}, {ID: 2, Distance: 0.2}}
	sparse := []Result{{ID: 2, Score: 0.9}, {ID: 3, Score: 0.4}}
	if !sameFused(fuseRRF(dense, sparse, 3, 0), Fuse(dense, sparse, FusionRRF, 0, 0, 3)) {
		t.Fatal("Fuse RRF != fuseRRF")
	}
	if !sameFused(fuseWeighted(dense, sparse, 3, 0.5), Fuse(dense, sparse, FusionWeighted, 0, 0, 3)) {
		t.Fatal("Fuse Weighted(alpha=0->0.5) != fuseWeighted(0.5)")
	}
	if !sameFused(fuseWeighted(dense, sparse, 3, 0.7), Fuse(dense, sparse, FusionWeighted, 0.7, 0, 3)) {
		t.Fatal("Fuse Weighted(0.7) mismatch")
	}
}

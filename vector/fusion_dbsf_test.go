// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"testing"
)

// dbsf3Sigma mirrors the production 3-sigma direct normalization (score-oriented,
// no inversion) so the test can assert exact expected values independently of the
// implementation's internal helpers.
func dbsf3Sigma(vals []float32, x float32) float32 {
	n := float64(len(vals))
	var sum float64
	for _, v := range vals {
		sum += float64(v)
	}
	mean := sum / n
	var sq float64
	for _, v := range vals {
		d := float64(v) - mean
		sq += d * d
	}
	std := math.Sqrt(sq / n)
	if std == 0 {
		return 1.0
	}
	lo := mean - 3*std
	norm := (float64(x) - lo) / (6 * std)
	if norm < 0 {
		norm = 0
	}
	if norm > 1 {
		norm = 1
	}
	return float32(norm)
}

func TestLaneStats(t *testing.T) {
	// n==0
	if m, s := laneStats(nil); m != 0 || s != 0 {
		t.Errorf("laneStats(nil) = (%v,%v), want (0,0)", m, s)
	}
	// n==1: mean = value, std = 0
	if m, s := laneStats([]float32{7}); m != 7 || s != 0 {
		t.Errorf("laneStats([7]) = (%v,%v), want (7,0)", m, s)
	}
	// Known distribution [10,5,0]: mean 5, population var = 50/3, std = sqrt(50/3).
	m, s := laneStats([]float32{10, 5, 0})
	if !approxEq(m, 5) {
		t.Errorf("mean = %v, want 5", m)
	}
	wantStd := float32(math.Sqrt(50.0 / 3.0))
	if !approxEq(s, wantStd) {
		t.Errorf("population std = %v, want %v", s, wantStd)
	}
}

// TestDBSFNormalize3SigmaMath asserts the exact 3-sigma normalized values for a
// hand-constructed lane with known mu/sigma, for both the sparse (direct) and the
// dense (inverted) orientation.
func TestDBSFNormalize3SigmaMath(t *testing.T) {
	// Sparse scores [10,5,0]: mean 5, std sqrt(50/3). Direct normalization is
	// symmetric: 10 and 0 are equidistant from the mean, so they normalize to
	// values symmetric about 0.5; the mean (5) normalizes to exactly 0.5.
	sparse := []Result{{ID: 1, Score: 10}, {ID: 2, Score: 5}, {ID: 3, Score: 0}}
	sn := dbsfNormalizeSparse(sparse)
	if !approxEq(sn[1], dbsf3Sigma([]float32{10, 5, 0}, 10)) {
		t.Errorf("sparse norm id1 = %v, want %v", sn[1], dbsf3Sigma([]float32{10, 5, 0}, 10))
	}
	if !approxEq(sn[2], 0.5) {
		t.Errorf("sparse norm id2 (mean) = %v, want 0.5", sn[2])
	}
	if !approxEq(sn[3], dbsf3Sigma([]float32{10, 5, 0}, 0)) {
		t.Errorf("sparse norm id3 = %v, want %v", sn[3], dbsf3Sigma([]float32{10, 5, 0}, 0))
	}
	// Symmetry: id1 + id3 should sum to 1.0 (mirror about 0.5).
	if !approxEq(sn[1]+sn[3], 1.0) {
		t.Errorf("sparse norm id1+id3 = %v, want 1.0 (symmetry)", sn[1]+sn[3])
	}
	// Higher score -> higher relevance.
	if !(sn[1] > sn[2] && sn[2] > sn[3]) {
		t.Errorf("sparse norm not monotone: %v %v %v", sn[1], sn[2], sn[3])
	}

	// Dense distances [0,5,10]: same distribution but INVERTED — smaller distance
	// -> higher relevance. id with distance 0 should get the highest norm.
	dense := []Result{{ID: 1, Distance: 0}, {ID: 2, Distance: 5}, {ID: 3, Distance: 10}}
	dn := dbsfNormalizeDense(dense)
	// distance 0 inverted == direct-norm of the MAX-equivalent: 1 - dbsf3Sigma(d=0).
	if !approxEq(dn[1], 1-dbsf3Sigma([]float32{0, 5, 10}, 0)) {
		t.Errorf("dense norm id1 = %v, want %v", dn[1], 1-dbsf3Sigma([]float32{0, 5, 10}, 0))
	}
	if !approxEq(dn[2], 0.5) {
		t.Errorf("dense norm id2 (mean dist) = %v, want 0.5", dn[2])
	}
	if !(dn[1] > dn[2] && dn[2] > dn[3]) {
		t.Errorf("dense norm not inverted-monotone: %v %v %v", dn[1], dn[2], dn[3])
	}
}

// TestFuseDBSFBlendedRanking verifies the blended fused scores against a fully
// hand-computed expectation (3-sigma per lane, alpha blend).
func TestFuseDBSFBlendedRanking(t *testing.T) {
	dense := []Result{{ID: 1, Distance: 0}, {ID: 2, Distance: 5}, {ID: 3, Distance: 10}}
	sparse := []Result{{ID: 3, Score: 10}, {ID: 2, Score: 5}, {ID: 1, Score: 0}}
	alpha := 0.5

	got := fuseDBSF(dense, sparse, 10, alpha)

	// Hand-compute: dense distances {0,5,10} inverted; sparse scores {10,5,0} direct.
	dvals := []float32{0, 5, 10}
	svals := []float32{10, 5, 0}
	dNorm := map[uint64]float32{
		1: 1 - dbsf3Sigma(dvals, 0),
		2: 1 - dbsf3Sigma(dvals, 5),
		3: 1 - dbsf3Sigma(dvals, 10),
	}
	sNorm := map[uint64]float32{
		1: dbsf3Sigma(svals, 0),
		2: dbsf3Sigma(svals, 5),
		3: dbsf3Sigma(svals, 10),
	}
	want := map[uint64]float32{}
	for id := uint64(1); id <= 3; id++ {
		want[id] = float32(alpha)*dNorm[id] + float32(1-alpha)*sNorm[id]
	}
	gotScore := map[uint64]float32{}
	for _, r := range got {
		gotScore[r.ID] = r.Score
	}
	for id := uint64(1); id <= 3; id++ {
		if !approxEq(gotScore[id], want[id]) {
			t.Errorf("fuseDBSF id%d score = %v, want %v", id, gotScore[id], want[id])
		}
	}
	// id2 is mid on both lanes (0.5+0.5)*0.5 = 0.5. id1 strong dense + weak sparse,
	// id3 weak dense + strong sparse — symmetric, so all three tie at 0.5, and the
	// tie-break is lower id first.
	if got[0].ID != 1 || got[1].ID != 2 || got[2].ID != 3 {
		t.Errorf("symmetric DBSF order = [%d %d %d], want [1 2 3] (lower-id tie-break)", got[0].ID, got[1].ID, got[2].ID)
	}
	// Dense Distance preserved.
	for _, r := range got {
		if r.ID == 1 && r.Distance != 0 {
			t.Errorf("id1 distance = %v, want 0", r.Distance)
		}
		if r.ID == 3 && r.Distance != 10 {
			t.Errorf("id3 distance = %v, want 10", r.Distance)
		}
	}
}

// TestDBSFAlphaExtremes mirrors the Weighted alpha-extreme test: alpha clamps and
// pure-lane behaviour hold.
func TestDBSFAlphaExtremes(t *testing.T) {
	dense := []Result{{ID: 1, Distance: 0.0}, {ID: 2, Distance: 1.0}}
	sparse := []Result{{ID: 2, Score: 10}, {ID: 1, Score: 0}}

	// alpha=1 -> pure dense: id1 (dist 0) wins.
	d := fuseDBSF(dense, sparse, 10, 1.0)
	if d[0].ID != 1 {
		t.Errorf("alpha=1 top = %d, want 1 (pure dense)", d[0].ID)
	}
	// alpha=0 -> pure sparse: id2 (score 10) wins.
	s := fuseDBSF(dense, sparse, 10, 0.0)
	if s[0].ID != 2 {
		t.Errorf("alpha=0 top = %d, want 2 (pure sparse)", s[0].ID)
	}
}

// TestDBSFDegenerateNoNaN: an all-equal lane and a single-result lane must map to
// 1.0 (matching the min-max all-equal convention) with no NaN / div-by-zero.
func TestDBSFDegenerateNoNaN(t *testing.T) {
	// All-equal distances and all-equal scores.
	dense := []Result{{ID: 1, Distance: 0.3}, {ID: 2, Distance: 0.3}, {ID: 3, Distance: 0.3}}
	sparse := []Result{{ID: 1, Score: 2}, {ID: 2, Score: 2}, {ID: 3, Score: 2}}

	dn := dbsfNormalizeDense(dense)
	for id, v := range dn {
		if v != 1.0 {
			t.Errorf("all-equal dense norm id%d = %v, want 1.0", id, v)
		}
	}
	sn := dbsfNormalizeSparse(sparse)
	for id, v := range sn {
		if v != 1.0 {
			t.Errorf("all-equal sparse norm id%d = %v, want 1.0", id, v)
		}
	}

	// Single result lanes -> 1.0.
	if v := dbsfNormalizeDense([]Result{{ID: 9, Distance: 0.7}}); v[9] != 1.0 {
		t.Errorf("single dense norm = %v, want 1.0", v[9])
	}
	if v := dbsfNormalizeSparse([]Result{{ID: 9, Score: 0.7}}); v[9] != 1.0 {
		t.Errorf("single sparse norm = %v, want 1.0", v[9])
	}

	// fuseDBSF on flat lanes: scores must be finite, no NaN.
	got := fuseDBSF(dense, sparse, 10, 0.5)
	for _, r := range got {
		if math.IsNaN(float64(r.Score)) || math.IsInf(float64(r.Score), 0) {
			t.Errorf("fuseDBSF flat lane produced non-finite score for id%d: %v", r.ID, r.Score)
		}
	}

	// On a flat lane DBSF must equal Weighted (both map every id to 1.0).
	wg := fuseWeighted(dense, sparse, 10, 0.5)
	if !sameFused(got, wg) {
		t.Errorf("DBSF on flat lanes != Weighted on flat lanes\n dbsf=%+v\n wtd=%+v", got, wg)
	}
}

// TestDBSFDiffersFromRRFAndWeighted constructs an outlier distribution where one
// huge sparse score would, under min-max, squash all the other scores toward 0;
// 3-sigma handles the outlier differently, producing a distinct ranking from both
// RRF and Weighted.
func TestDBSFDiffersFromRRFAndWeighted(t *testing.T) {
	// Dense distances (asc by Distance): ids 1-3 are close, ids 4-5 are far.
	dense := []Result{
		{ID: 1, Distance: 0.10},
		{ID: 2, Distance: 0.20},
		{ID: 3, Distance: 0.30},
		{ID: 4, Distance: 0.80},
		{ID: 5, Distance: 0.85},
	}
	// Sparse scores (desc by Score): id5 is a massive outlier (200), ids 1-4 are a
	// cluster (8,7,6,5). Under min-max the outlier dominates the span and squashes
	// the cluster toward 0; under 3-sigma the outlier inflates sigma but the cluster
	// keeps meaningful separation, and the outlier saturates to 1.0. With alpha=0.5
	// the resulting top-k order is distinct from BOTH min-max Weighted and RRF:
	//   RRF      = [1 2 5 3 4]
	//   Weighted = [1 5 2 3 4]
	//   DBSF     = [5 1 2 3 4]
	sparse := []Result{
		{ID: 5, Score: 200},
		{ID: 1, Score: 8},
		{ID: 2, Score: 7},
		{ID: 3, Score: 6},
		{ID: 4, Score: 5},
	}
	k := 5
	alpha := 0.5

	dbsf := fuseDBSF(dense, sparse, k, alpha)
	wtd := fuseWeighted(dense, sparse, k, alpha)
	rrf := fuseRRF(dense, sparse, k, 60)

	idsOfR := func(rs []Result) []uint64 {
		out := make([]uint64, len(rs))
		for i, r := range rs {
			out[i] = r.ID
		}
		return out
	}
	dbsfIDs := idsOfR(dbsf)
	wtdIDs := idsOfR(wtd)
	rrfIDs := idsOfR(rrf)

	eqOrder := func(a, b []uint64) bool {
		if len(a) != len(b) {
			return false
		}
		for i := range a {
			if a[i] != b[i] {
				return false
			}
		}
		return true
	}

	if eqOrder(dbsfIDs, wtdIDs) {
		t.Errorf("DBSF ranking should differ from Weighted on outlier distribution; both = %v", dbsfIDs)
	}
	if eqOrder(dbsfIDs, rrfIDs) {
		t.Errorf("DBSF ranking should differ from RRF on outlier distribution; both = %v", dbsfIDs)
	}
	// Exact hand-verified orders (independently computed; see comment on the input).
	if !eqOrder(dbsfIDs, []uint64{5, 1, 2, 3, 4}) {
		t.Errorf("DBSF order = %v, want [5 1 2 3 4]", dbsfIDs)
	}
	if !eqOrder(wtdIDs, []uint64{1, 5, 2, 3, 4}) {
		t.Errorf("Weighted order = %v, want [1 5 2 3 4]", wtdIDs)
	}
	if !eqOrder(rrfIDs, []uint64{1, 2, 5, 3, 4}) {
		t.Errorf("RRF order = %v, want [1 2 5 3 4]", rrfIDs)
	}
	t.Logf("DBSF=%v Weighted=%v RRF=%v", dbsfIDs, wtdIDs, rrfIDs)
}

// TestFuseDBSFRouting: the Fuse / FuseScoreLanes routers dispatch FusionDBSF to the
// DBSF fuse functions (alpha==0 -> 0.5 default like Weighted).
func TestFuseDBSFRouting(t *testing.T) {
	dense := []Result{{ID: 1, Distance: 0.1}, {ID: 2, Distance: 0.2}, {ID: 3, Distance: 0.3}}
	sparse := []Result{{ID: 2, Score: 0.9}, {ID: 3, Score: 0.4}, {ID: 1, Score: 0.1}}

	if !sameFused(fuseDBSF(dense, sparse, 3, 0.5), Fuse(dense, sparse, FusionDBSF, 0, 0, 3)) {
		t.Fatal("Fuse(FusionDBSF, alpha=0) != fuseDBSF(0.5)")
	}
	if !sameFused(fuseDBSF(dense, sparse, 3, 0.7), Fuse(dense, sparse, FusionDBSF, 0.7, 0, 3)) {
		t.Fatal("Fuse(FusionDBSF, 0.7) mismatch")
	}
	// Score-lanes: both lanes are score-desc (no inversion).
	first := []Result{{ID: 1, Score: 0.9}, {ID: 2, Score: 0.5}, {ID: 3, Score: 0.1}}
	if !sameFused(fuseDBSFScoreLanes(first, sparse, 3, 0.5), FuseScoreLanes(first, sparse, FusionDBSF, 0, 0, 3)) {
		t.Fatal("FuseScoreLanes(FusionDBSF, alpha=0) != fuseDBSFScoreLanes(0.5)")
	}
}

// TestFuseDBSFScoreLanesNotInverted: in the score-lanes variant the first lane is
// NOT inverted — a higher first-lane score maps to a higher relevance (unlike the
// distance-oriented fuseDBSF).
func TestFuseDBSFScoreLanesNotInverted(t *testing.T) {
	// first lane is a SCORE lane (higher = better). With pure-first weight (alpha=1),
	// the highest first-lane score must rank first.
	first := []Result{{ID: 10, Score: 9}, {ID: 20, Score: 5}, {ID: 30, Score: 1}}
	got := fuseDBSFScoreLanes(first, nil, 10, 1.0)
	if got[0].ID != 10 {
		t.Errorf("score-lanes alpha=1 top = %d, want 10 (highest score, not inverted)", got[0].ID)
	}
	// Distance preserved (0 for MaxSim).
	for _, r := range got {
		if r.Distance != 0 {
			t.Errorf("score-lanes id%d distance = %v, want 0", r.ID, r.Distance)
		}
	}
}

// TestDBSFEndToEndDenseHybrid: hnsw.HybridSearch with FusionDBSF runs end-to-end
// (the direct fuse-call path in HybridSearch) and produces a sane top-k with no
// panic; the sparse-strong outlier doc is surfaced.
func TestDBSFEndToEndDenseHybrid(t *testing.T) {
	h := newHybridCorpus(t)
	dense := []float32{0, 0, 0, 0}
	sparse := SparseVector{Indices: []uint32{42}, Values: []float32{5.0}}

	got, err := h.HybridSearch(dense, sparse, 5, HybridOpts{Method: FusionDBSF, Alpha: 0.5})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 || len(got) > 5 {
		t.Fatalf("DBSF dense hybrid len = %d, want 1..5", len(got))
	}
	for _, r := range got {
		if math.IsNaN(float64(r.Score)) {
			t.Errorf("DBSF dense hybrid NaN score for id%d", r.ID)
		}
	}
	// alpha small -> sparse-heavy -> doc 100 (strong sparse term) ranks first.
	sparseHeavy, err := h.HybridSearch(dense, sparse, 5, HybridOpts{Method: FusionDBSF, Alpha: 0.001})
	if err != nil {
		t.Fatal(err)
	}
	if len(sparseHeavy) == 0 || sparseHeavy[0].ID != 100 {
		t.Errorf("DBSF sparse-heavy top = %+v, want doc 100 first", sparseHeavy)
	}
}

// TestDBSFEndToEndNamedHybrid: NamedHybrid with FusionDBSF equals the hand-fused
// ground truth (same lanes -> Fuse with FusionDBSF).
func TestDBSFEndToEndNamedHybrid(t *testing.T) {
	nc, denseQ, sparseQ := namedHybridFixture(t)
	k := 5
	opts := HybridOpts{Method: FusionDBSF, Alpha: 0.3}

	dense, err := nc.SearchNamed("title", denseQ, namedHybridK(opts.DenseK, k), Filter{})
	if err != nil {
		t.Fatal(err)
	}
	sparse, err := nc.SearchNamedSparse("terms", sparseQ, namedHybridK(opts.SparseK, k), Filter{})
	if err != nil {
		t.Fatal(err)
	}
	want := Fuse(dense, sparse, opts.Method, opts.Alpha, opts.RRFK, k)

	got, err := nc.NamedHybrid("title", denseQ, "terms", sparseQ, k, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !eqResults(got, want) {
		t.Fatalf("NamedHybrid DBSF = %+v\nwant (hand-fused) %+v", got, want)
	}
}

// TestDBSFEndToEndMVHybrid: MVHybrid with FusionDBSF equals the hand-fused ground
// truth via the score-lanes path (FuseScoreLanes with FusionDBSF).
func TestDBSFEndToEndMVHybrid(t *testing.T) {
	m, _ := mvHybridFixture(t)
	query := [][]float32{{1, 0, 0}, {0, 1, 0}}
	sparseQ := mvSV(0, 1.0, 2, 1.0, 5, 1.0)
	k := 5
	opts := HybridOpts{Method: FusionDBSF, Alpha: 0.3}

	want := handFuse(t, m, query, sparseQ, k, opts)
	got, err := m.MVHybrid(query, sparseQ, k, opts)
	if err != nil {
		t.Fatal(err)
	}
	resultsEqual(t, got, want, "dbsf")
}

// TestRRFWeightedUnchangedByDBSF is a regression guard: adding DBSF must not change
// RRF or Weighted fusion results.
func TestRRFWeightedUnchangedByDBSF(t *testing.T) {
	dense := []Result{{ID: 1, Distance: 0.0}, {ID: 2, Distance: 0.5}, {ID: 3, Distance: 1.0}}
	sparse := []Result{{ID: 3, Score: 10}, {ID: 2, Score: 5}, {ID: 1, Score: 0}}

	// Weighted alpha=0.5 -> all 0.5 (same as the original TestFuseWeighted oracle).
	wg := fuseWeighted(dense, sparse, 10, 0.5)
	for _, r := range wg {
		if !approxEq(r.Score, 0.5) {
			t.Errorf("Weighted id%d = %v, want 0.5 (unchanged)", r.ID, r.Score)
		}
	}
	// RRF oracle for this input (1-based ranks).
	rrf := fuseRRF(dense, sparse, 10, 60)
	score := map[uint64]float32{}
	for _, r := range rrf {
		score[r.ID] = r.Score
	}
	want := map[uint64]float32{
		1: 1.0/61 + 1.0/63, // dense r0, sparse r2
		2: 1.0/62 + 1.0/62, // dense r1, sparse r1
		3: 1.0/63 + 1.0/61, // dense r2, sparse r0
	}
	for id, w := range want {
		if !approxEq(score[id], w) {
			t.Errorf("RRF id%d = %v, want %v (unchanged)", id, score[id], w)
		}
	}
}

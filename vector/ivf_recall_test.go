// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// IVF-Flat RECALL RIGOR.
//
// TestIVFTrainedRecall (ivf_test.go) proves recall on CLUSTERED data, where
// well-separated blobs make a few probes capture every true neighbor (recall
// trivially saturates). That is too easy to prove IVF is a real APPROXIMATE
// index. The tests here use HARD data — uniform-random vectors with no cluster
// structure — so a query's true top-k genuinely spans many Voronoi cells and
// nprobe becomes a real recall/speed knob. We assert:
//   - recall@10 strictly increases from nprobe=1 to nprobe=nlist (the knob is
//     real, monotonic-ish),
//   - recall@10 at nprobe=nlist == 1.0 (probing every cell == exact),
//   - recall@10 at a fractional nprobe (nlist/8) clears a useful threshold,
//   - every IVF result is a member of the exact candidate set and the returned
//     prefix is correctly ranked by distance.
//
// nprobe is swept on a SINGLE trained index via the settable ix.nprobe field
// (read by gatherLocked per search, clamped to [1, nlist]) — no need to retrain
// N indices; the lists/centroids are fixed, only the probe width changes.

// recallVecsConfig builds an L2 config with a FIXED nlist so the recall sweep is
// deterministic and independent of the auto-nlist heuristic.
func recallVecsConfig(dim, nlist int) Config {
	c := DefaultConfig()
	c.Dim = dim
	c.Metric = L2
	c.Seed = 42
	c.IVFNlist = nlist
	return c
}

// recallAt measures recall@k of a trained IVF at its current nprobe against
// brute-force ground truth, and asserts every returned id is a real exact
// candidate ranked ascending by distance. truthIDs[i] is the ground-truth top-k
// id-set for queries[i]; exactIDs is the set of all live ids (every result must
// belong to it). dist is the metric used for the rank-monotonicity check.
func recallAt(t *testing.T, ix *ivf, queries [][]float32, truth []map[uint64]bool,
	exactIDs map[uint64]bool, k int) float64 {
	t.Helper()
	hits, denom := 0, 0
	for qi, q := range queries {
		got, err := ix.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		// Every IVF result must be a real live id (no bogus ids) and distances
		// must be non-decreasing (correctly ranked among the returned set).
		var prev float32 = -math.MaxFloat32
		for _, r := range got {
			if !exactIDs[r.ID] {
				t.Fatalf("query %d: IVF returned id %d that is not a live candidate", qi, r.ID)
			}
			if r.Distance < prev {
				t.Fatalf("query %d: IVF results not ascending by distance (%g after %g)", qi, r.Distance, prev)
			}
			prev = r.Distance
		}
		for w := range truth[qi] {
			denom++
			if idSet(got)[w] {
				hits++
			}
		}
	}
	return float64(hits) / float64(denom)
}

// TestIVFRecallNprobeKnob proves nprobe is a real recall/speed knob on HARD
// (uniform-random) data: recall rises monotonic-ish with nprobe, hits 1.0 at
// nprobe=nlist (exact), and is already useful at a small fraction of the cells.
func TestIVFRecallNprobeKnob(t *testing.T) {
	const (
		dim   = 24
		n     = 2000
		nlist = 64
		k     = 10
		nq    = 40
	)
	rng := rand.New(rand.NewSource(2026))
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)
	exactIDs := make(map[uint64]bool, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
		exactIDs[ids[i]] = true
	}

	ix, err := newIVF(recallVecsConfig(dim, nlist))
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	if ix.nlist != nlist {
		t.Fatalf("nlist = %d, want %d (fixed via cfg)", ix.nlist, nlist)
	}

	// Ground truth: brute-force exact top-k per query.
	queries := randVecs(rng, nq, dim)
	truth := make([]map[uint64]bool, nq)
	for qi, q := range queries {
		want := bruteForceNN(q, ids, vecs, k)
		set := make(map[uint64]bool, k)
		for _, w := range want {
			set[w] = true
		}
		truth[qi] = set
	}

	// Sweep nprobe = 1, nlist/8, nlist/4, nlist.
	ix.nprobe = 1
	rLow := recallAt(t, ix, queries, truth, exactIDs, k)

	ix.nprobe = nlist / 8 // 8 cells out of 64
	rEighth := recallAt(t, ix, queries, truth, exactIDs, k)

	ix.nprobe = nlist / 4 // 16 cells out of 64
	rQuarter := recallAt(t, ix, queries, truth, exactIDs, k)

	ix.nprobe = nlist
	rFull := recallAt(t, ix, queries, truth, exactIDs, k)

	t.Logf("IVF recall@%d (uniform-random, n=%d dim=%d nlist=%d seed=2026): "+
		"nprobe=1 -> %.3f | nprobe=nlist/8=%d -> %.3f | nprobe=nlist/4=%d -> %.3f | nprobe=nlist=%d -> %.3f",
		k, n, dim, nlist, rLow, nlist/8, rEighth, nlist/4, rQuarter, nlist, rFull)

	// The knob is real: more cells => more recall, monotonic across the sweep.
	if !(rLow < rEighth && rEighth < rQuarter && rQuarter <= rFull) {
		t.Fatalf("nprobe is not a monotonic recall knob: low=%.3f eighth=%.3f quarter=%.3f full=%.3f",
			rLow, rEighth, rQuarter, rFull)
	}
	// On hard data a single cell must miss a meaningful fraction (else the data
	// is not actually approximate and the test proves nothing).
	if rLow >= 0.95 {
		t.Fatalf("nprobe=1 recall %.3f too high — data is not genuinely approximate", rLow)
	}
	// Probing every cell == exact.
	if rFull < 0.999 {
		t.Fatalf("nprobe=nlist recall %.3f, want ~1.0 (exact) — IVF MISSES candidates at full probe (engine bug)", rFull)
	}
	// Useful at a fraction of the cells. NOTE: on uniform-random data (no cluster
	// structure) a query's true top-10 is spread across many cells, so the floor
	// here is MUCH lower than the clustered-data test's (TestIVFTrainedRecall hits
	// ~1.0 at nlist/4) — that is the whole point: this data is genuinely hard.
	// Tolerances were calibrated to the observed seed=2026 numbers (~0.58 at 1/8,
	// ~0.74 at 1/4) and floored conservatively below them so they are robust to
	// minor kernel/seed drift; they are recall/speed-tradeoff floors, not tight
	// targets. The load-bearing assertions are the monotonic rise + exact-at-full.
	if rEighth < 0.45 {
		t.Fatalf("nprobe=nlist/8 recall %.3f, want >= 0.45 (useful at 1/8 of cells)", rEighth)
	}
	if rQuarter < 0.65 {
		t.Fatalf("nprobe=nlist/4 recall %.3f, want >= 0.65 (useful at 1/4 of cells)", rQuarter)
	}
}

// TestIVFRecallFilterFirst proves a payload filter composes correctly with the
// IVF probe search: at nprobe=nlist the filtered search returns EXACTLY the true
// top-k among matching docs (no missed matches, no non-matching leaks), compared
// to brute-force ground truth computed over the matching subset only.
func TestIVFRecallFilterFirst(t *testing.T) {
	const (
		dim   = 20
		n     = 1500
		nlist = 48
		k     = 10
		nq    = 30
	)
	rng := rand.New(rand.NewSource(99))
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}

	ix, err := newIVF(recallVecsConfig(dim, nlist))
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	// Attach a "bucket" payload AFTER build (BuildConcurrent is vectors-only),
	// selecting ~1/3 of docs as the filter target.
	matchIDs := make([]uint64, 0, n/3)
	matchVecs := make([][]float32, 0, n/3)
	for i := 0; i < n; i++ {
		bucket := "miss"
		if i%3 == 0 {
			bucket = "hit"
			matchIDs = append(matchIDs, ids[i])
			matchVecs = append(matchVecs, vecs[i])
		}
		if _, _, _, err := ix.SetPayload(ids[i], Metadata{"bucket": NewString(bucket)}, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	ix.nprobe = nlist // probe every cell => exact among matches

	filter := Filter{Op: FilterEq, Field: "bucket", Value: NewString("hit")}
	queries := randVecs(rng, nq, dim)
	exactMatches := 0
	for qi, q := range queries {
		got, err := ix.SearchFiltered(q, k, filter)
		if err != nil {
			t.Fatal(err)
		}
		// Every returned doc must be a "hit" (filter never leaks a non-match).
		for _, r := range got {
			if (r.ID-1)%3 != 0 {
				t.Fatalf("query %d: filtered result id %d is not a 'hit'", qi, r.ID)
			}
		}
		// At nprobe=nlist the filtered result must EQUAL the exact top-k among
		// matches, in order.
		want := bruteForceNN(q, matchIDs, matchVecs, k)
		if len(got) != len(want) {
			t.Fatalf("query %d: filtered returned %d, want %d (among matches)", qi, len(got), len(want))
		}
		for i := range want {
			if got[i].ID != want[i] {
				t.Fatalf("query %d pos %d: filtered id %d != exact match-NN %d", qi, i, got[i].ID, want[i])
			}
			exactMatches++
		}
	}
	t.Logf("IVF filter-first (n=%d, matches=%d, nprobe=nlist=%d): %d/%d exact match-NN positions over %d queries",
		n, len(matchIDs), nlist, exactMatches, nq*k, nq)
	if exactMatches != nq*k {
		t.Fatalf("filter-first not exact-among-matches at nprobe=nlist: %d/%d", exactMatches, nq*k)
	}
}

// TestIVFRecallCosine confirms metric correctness: with the Cosine metric, the
// IVF at nprobe=nlist reproduces the exact cosine top-k (vectors normalized,
// distance = 1 - dot, matching the engine's stored-normalized representation).
func TestIVFRecallCosine(t *testing.T) {
	const (
		dim   = 24
		n     = 1500
		nlist = 48
		k     = 10
		nq    = 25
	)
	rng := rand.New(rand.NewSource(7))
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}

	cfg := recallVecsConfig(dim, nlist)
	cfg.Metric = Cosine
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	ix.nprobe = nlist // exact

	// Cosine ground truth: normalize both sides, rank by 1 - dot (== the engine's
	// stored-normalized cosine distance).
	cosineNN := func(q []float32, k int) []uint64 {
		qn := append([]float32(nil), q...)
		normalize(qn)
		type sd struct {
			id uint64
			d  float32
		}
		all := make([]sd, n)
		for i := range vecs {
			vn := append([]float32(nil), vecs[i]...)
			normalize(vn)
			all[i] = sd{ids[i], 1.0 - dotScalar(qn, vn)}
		}
		sort.Slice(all, func(a, b int) bool { return all[a].d < all[b].d })
		out := make([]uint64, 0, k)
		for i := 0; i < k && i < len(all); i++ {
			out = append(out, all[i].id)
		}
		return out
	}

	queries := randVecs(rng, nq, dim)
	hits, denom := 0, 0
	for _, q := range queries {
		got, err := ix.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		gs := idSet(got)
		for _, w := range cosineNN(q, k) {
			denom++
			if gs[w] {
				hits++
			}
		}
	}
	recall := float64(hits) / float64(denom)
	t.Logf("IVF cosine recall@%d at nprobe=nlist=%d (n=%d dim=%d): %.3f", k, nlist, n, dim, recall)
	if recall < 0.999 {
		t.Fatalf("cosine nprobe=nlist recall %.3f, want ~1.0 (exact) — metric/engine bug", recall)
	}
}

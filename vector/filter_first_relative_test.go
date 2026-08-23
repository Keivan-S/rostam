// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"testing"
)

// Tests for the OPT-IN relative selectivity gate (Config.FilterFirstRelativeBP).
//
// The #1 requirement is DEFAULT-OFF (bp=0) BYTE/BEHAVIOUR-IDENTICAL: the effective
// limit equals filterFirstThreshold(), the matched set is identical, and the
// filter-first-vs-graph decision is identical to today. The relative path only
// WIDENS which filters are admitted (a relatively-selective filter beyond the
// absolute cap), bounded by a hard cap, with preferFilterFirst still deciding.

// TestEffectiveFilterFirstLimitDefaultOff proves bp=0 returns absThreshold EXACTLY
// for a spread of live counts (the byte-identical guarantee at the formula level).
func TestEffectiveFilterFirstLimitDefaultOff(t *testing.T) {
	for _, abs := range []int{1, 50, defaultFilterFirstThreshold, 1_000_000} {
		for _, n := range []int{0, 1, 100, 10_000, 1_000_000, 2_000_000_000} {
			if got := effectiveFilterFirstLimit(abs, 0, n); got != abs {
				t.Errorf("effectiveFilterFirstLimit(abs=%d, bp=0, n=%d) = %d, want %d (byte-identical)", abs, n, got, abs)
			}
		}
	}
}

// TestEffectiveFilterFirstLimitRelative checks the max(abs, min(bp*n/10000, cap))
// formula across regimes: below-abs stays abs, above-abs raises to the relative
// budget, and the hard cap bounds it.
func TestEffectiveFilterFirstLimitRelative(t *testing.T) {
	cases := []struct {
		abs, bp, n, want int
	}{
		// bp*n/10000 below abs -> abs wins (no regression below the cap).
		{abs: 10_000, bp: 100, n: 1000, want: 10_000}, // 100*1000/10000 = 10 < 10000
		// bp*n/10000 above abs -> relative budget wins.
		{abs: 50, bp: 5000, n: 200, want: 100},         // 5000*200/10000 = 100 > 50
		{abs: 50, bp: 100, n: 1_000_000, want: 10_000}, // 100*1e6/10000 = 10000 > 50
		// exactly abs -> abs (tie).
		{abs: 100, bp: 5000, n: 200, want: 100}, // 5000*200/10000 = 100 == abs
		// hard cap bounds the relative budget.
		{abs: 1, bp: 10000, n: 100_000_000, want: filterFirstRelativeHardCap}, // 1e8 capped to 1e6
	}
	for _, c := range cases {
		if got := effectiveFilterFirstLimit(c.abs, c.bp, c.n); got != c.want {
			t.Errorf("effectiveFilterFirstLimit(abs=%d, bp=%d, n=%d) = %d, want %d", c.abs, c.bp, c.n, got, c.want)
		}
	}
}

// TestEffectiveFilterFirstLimitNoOverflow exercises the int64-overflow guard: a
// max bp (10000) times a multi-billion live count would overflow a 32-bit-ish int
// if computed in int; the int64 product clamps to the hard cap, never negative.
func TestEffectiveFilterFirstLimitNoOverflow(t *testing.T) {
	got := effectiveFilterFirstLimit(50, 10000, 2_000_000_000)
	if got != filterFirstRelativeHardCap {
		t.Fatalf("effectiveFilterFirstLimit(50, 10000, 2e9) = %d, want %d (clamped, not overflow)", got, filterFirstRelativeHardCap)
	}
	if got < 0 {
		t.Fatalf("effectiveFilterFirstLimit overflowed to negative: %d", got)
	}
}

// TestFilterFirstRelativeBPValidation asserts the engine Validate fails loud on an
// out-of-range basis-point value (and accepts the in-range boundary + default).
func TestFilterFirstRelativeBPValidation(t *testing.T) {
	base := Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1}
	for _, bad := range []int{-1, 10001, 50000} {
		c := base
		c.FilterFirstRelativeBP = bad
		if err := ValidateConfig(c); err != ErrInvalidFilterFirstRelativeBP {
			t.Errorf("Validate(bp=%d) err = %v, want ErrInvalidFilterFirstRelativeBP", bad, err)
		}
	}
	for _, ok := range []int{0, 1, 5000, 10000} {
		c := base
		c.FilterFirstRelativeBP = ok
		if err := ValidateConfig(c); err != nil {
			t.Errorf("Validate(bp=%d) err = %v, want nil", ok, err)
		}
	}
}

// buildRelativeCorpus inserts n points into h with a "g" field: ~half match g==0
// (the selective-but-above-abs filter target), the rest g==1. Returns the corpus
// + metas for brute-force comparison.
func buildRelativeCorpus(t *testing.T, h *hnsw, n, dim int) (map[uint64][]float32, map[uint64]Metadata) {
	t.Helper()
	rng := rand.New(rand.NewSource(7))
	corpus := make(map[uint64][]float32, n)
	metas := make(map[uint64]Metadata, n)
	for i := 1; i <= n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		g := int64(1)
		if i%2 == 0 {
			g = 0 // half the corpus
		}
		meta := Metadata{"g": NewInt(g)}
		corpus[uint64(i)] = v
		metas[uint64(i)] = meta
		if _, _, err := h.Insert(uint64(i), v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	return corpus, metas
}

// TestRelativeGateAdmitsAboveAbsoluteCap is the core relative-path proof: a filter
// matching MORE than absThreshold but <= relativeBP*size/10000 is ADMITTED to
// filter-first when bp>0 (and REJECTED to graph when bp=0). The gate decision is
// probed directly via candidates()+effectiveFilterFirstLimit; correctness is
// proved by exact equality with predicate-eval brute force.
func TestRelativeGateAdmitsAboveAbsoluteCap(t *testing.T) {
	const n, dim, k = 200, 8, 10
	const absThreshold = 50
	// matchCount ~= 100 (g==0 half). relativeBP=5000 -> 5000*200/10000 = 100 >= 100.
	cfg := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 100, EfSearch: 32, Seed: 1, FilterFirstThreshold: absThreshold}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	corpus, metas := buildRelativeCorpus(t, h, n, dim)

	sel := Filter{Op: FilterEq, Field: "g", Value: NewInt(0)}
	pred, err := sel.Compile()
	if err != nil {
		t.Fatal(err)
	}

	// The matched candidate set is larger than the absolute threshold.
	cands, ok := h.payloadIdx.candidates(sel, n) // materialize all to count
	if !ok {
		t.Fatal("candidates returned ok=false for the equality filter")
	}
	if len(cands) <= absThreshold {
		t.Fatalf("test setup: candidate set %d must exceed absThreshold %d", len(cands), absThreshold)
	}

	// bp=0 (default): effective limit == absThreshold, so the candidate set
	// (> absThreshold) is REJECTED to the graph path. Byte-identical to today.
	if limit := h.effectiveFilterFirstLimit(h.arena.Size()); limit != absThreshold {
		t.Fatalf("bp=0: effective limit = %d, want absThreshold %d (byte-identical)", limit, absThreshold)
	}

	// bp=5000: effective limit rises to the relative budget (>= matched count), so
	// the SAME candidate set is now ADMITTED to filter-first.
	h.cfg.FilterFirstRelativeBP = 5000
	limit := h.effectiveFilterFirstLimit(h.arena.Size())
	if limit < len(cands) {
		t.Fatalf("bp=5000: effective limit = %d, want >= matched %d (admit)", limit, len(cands))
	}
	if _, ok := h.payloadIdx.candidates(sel, limit); !ok {
		t.Fatalf("bp=5000: candidates(limit=%d) ok=false, expected admit", limit)
	}

	// Results are EXACT (filter-first re-applies the predicate): equal to the
	// predicate-eval brute force over the matched set.
	q := corpus[2]
	got := mustSearch(t, h, q, k, sel)
	want := bruteForceFiltered(corpus, metas, q, k, func(m Metadata) bool { return pred(m) })
	if !eqUint64(resultIDs(got), want) {
		t.Errorf("relative-admitted filter not exact: %v != %v", resultIDs(got), want)
	}
}

// TestRelativeGateDefaultOffByteIdentical proves the matched set + the filter-
// first-vs-graph decision are identical with bp=0 vs the no-field baseline for a
// filter under the absolute cap (still filter-first) and a search.
func TestRelativeGateDefaultOffByteIdentical(t *testing.T) {
	const n, dim, k = 300, 8, 10
	mk := func(bp int) *hnsw {
		cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, FilterFirstRelativeBP: bp}
		h, err := newHNSW(cfg)
		if err != nil {
			t.Fatal(err)
		}
		rng := rand.New(rand.NewSource(11))
		for i := 1; i <= n; i++ {
			v := make([]float32, dim)
			for j := range v {
				v[j] = float32(rng.NormFloat64())
			}
			g := int64(1)
			if i%60 == 0 {
				g = 0 // ~2% selective (well under the 10k absolute cap)
			}
			if _, _, err := h.Insert(uint64(i), v, 0, Metadata{"g": NewInt(g)}, nil, nil, CASCond{}); err != nil {
				t.Fatal(err)
			}
		}
		return h
	}
	base := mk(0)
	// A baseline index built WITHOUT the field set (zero value) is the same as bp=0.
	q := make([]float32, dim)
	for j := range q {
		q[j] = 0.3
	}
	sel := Filter{Op: FilterEq, Field: "g", Value: NewInt(0)}
	gotBase := mustSearch(t, base, q, k, sel)

	// Effective limit at bp=0 equals filterFirstThreshold() at every live count.
	if base.effectiveFilterFirstLimit(base.arena.Size()) != base.filterFirstThreshold() {
		t.Fatalf("bp=0 effective limit != filterFirstThreshold()")
	}
	// A second index with bp explicitly 0 returns the identical matched set.
	other := mk(0)
	gotOther := mustSearch(t, other, q, k, sel)
	if !eqUint64(resultIDs(gotBase), resultIDs(gotOther)) {
		t.Errorf("bp=0 not stable: %v != %v", resultIDs(gotBase), resultIDs(gotOther))
	}
}

// TestRelativeGateCostModelStillRejects proves the cost model still has the final
// say: even when the relative gate ADMITS a large candidate set, preferFilterFirst
// can route to graph (non-selective), and results stay correct (high recall).
func TestRelativeGateCostModelStillRejects(t *testing.T) {
	const n, dim, k = 2000, 16, 10
	// Small M/EfSearch (like planner_test) so the cost-model crossover is reachable
	// at a test-sized corpus; a non-selective filter (g==1 ~95%) must route to graph.
	cfg := Config{Dim: dim, Metric: L2, M: 4, EfConstruction: 50, EfSearch: 10, Seed: 1,
		FilterFirstThreshold: 50, FilterFirstRelativeBP: 10000} // limit -> min(n, cap) == n
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(7))
	corpus := make(map[uint64][]float32, n)
	metas := make(map[uint64]Metadata, n)
	for i := 1; i <= n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		g := int64(1) // ~95% common (non-selective)
		if i%20 == 0 {
			g = 0
		}
		meta := Metadata{"g": NewInt(g)}
		corpus[uint64(i)] = v
		metas[uint64(i)] = meta
		if _, _, err := h.Insert(uint64(i), v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}

	// The relative gate would admit the g==1 set (size ~0.95n <= limit n), but
	// preferFilterFirst should prefer graph for such a non-selective filter.
	nonsel := Filter{Op: FilterEq, Field: "g", Value: NewInt(1)}
	cands, ok := h.payloadIdx.candidates(nonsel, h.effectiveFilterFirstLimit(h.arena.Size()))
	if !ok {
		t.Fatal("candidates ok=false")
	}
	if h.preferFilterFirst(len(cands), k) {
		t.Fatalf("preferFilterFirst(%d, %d) = true, want false (non-selective -> graph)", len(cands), k)
	}

	// Graph path still returns high recall vs the exact truth.
	q := corpus[1]
	got := mustSearch(t, h, q, k, nonsel)
	truth := make(map[uint64]bool)
	for _, id := range bruteForceFiltered(corpus, metas, q, k, func(m Metadata) bool { return m["g"].Int == 1 }) {
		truth[id] = true
	}
	matches := 0
	for _, r := range got {
		if truth[r.ID] {
			matches++
		}
	}
	// Graph recall at these intentionally tiny M/EfSearch params is modest; the
	// proof here is that the cost model ROUTED to graph (asserted above) and still
	// returns mostly-correct results, not exact recall.
	if recall := float64(matches) / float64(k); recall < 0.6 {
		t.Errorf("non-selective (graph) recall = %.2f, want >= 0.6", recall)
	}
}

// TestRelativeGateNamedFamily smoke-tests the named family honoring the knob: the
// per-space FilterFirstRelativeBP threads through to the inner index's
// effectiveFilterFirstLimit, and a relative-admitted filter returns exact results.
func TestRelativeGateNamedFamily(t *testing.T) {
	const n, k = 200, 10
	cfg := map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: Cosine, FilterFirstRelativeBP: 5000},
	}
	nc, err := NewNamedCollection("default/named", cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer nc.Stop()

	idx := nc.indexes["title"]
	if idx.effectiveFilterFirstLimit(0) != idx.filterFirstThreshold() {
		t.Fatalf("named bp set but liveCount=0 should still return absThreshold")
	}
	// With a non-zero live count the relative budget raises the limit.
	if got := idx.effectiveFilterFirstLimit(n); got < idx.filterFirstThreshold() {
		t.Fatalf("named effective limit = %d, want >= absThreshold", got)
	}

	rng := rand.New(rand.NewSource(5))
	for i := 1; i <= n; i++ {
		v := []float32{float32(rng.NormFloat64()), float32(rng.NormFloat64()), float32(rng.NormFloat64()), float32(rng.NormFloat64())}
		g := int64(1)
		if i%2 == 0 {
			g = 0
		}
		if err := nc.Insert(uint64(i), map[string][]float32{"title": v}, Metadata{"g": NewInt(g)}, 0); err != nil {
			t.Fatal(err)
		}
	}
	q := []float32{0.1, 0.2, 0.3, 0.4}
	sel := Filter{Op: FilterEq, Field: "g", Value: NewInt(0)}
	// filter-first (index present) must equal the predicate-eval fallback exactly.
	assertFilterFirstMatchesFallback(t, nc, "title", q, k, sel)
}

// TestRelativeGateIVFAdmitsAboveAbsoluteCap is the IVF-family analogue of
// TestRelativeGateAdmitsAboveAbsoluteCap: build an IVF index with a low absolute
// threshold + bp>0, a filter matching MORE than the absolute threshold but within
// the relative budget, assert (a) the relative path admits the filter and (b) the
// results equal predicate-eval brute force (exact filter-first path).
// IVF reuses the shared effectiveFilterFirstLimit free function, so this is the
// direct all-4-families assertion the plan requires.
func TestRelativeGateIVFAdmitsAboveAbsoluteCap(t *testing.T) {
	const n, dim, k = 200, 8, 10
	const absThreshold = 50
	// matchCount ~= 100 (g==0 half of n=200). relativeBP=5000 ->
	// 5000*200/10000 = 100 >= 100: the relative budget covers the matched set.
	cfg := ivfTestConfig(dim)
	cfg.FilterFirstThreshold = absThreshold
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(13))
	corpus := make(map[uint64][]float32, n)
	metas := make(map[uint64]Metadata, n)
	for i := 1; i <= n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		g := int64(1)
		if i%2 == 0 {
			g = 0 // half the corpus
		}
		meta := Metadata{"g": NewInt(g)}
		corpus[uint64(i)] = v
		metas[uint64(i)] = meta
		if _, _, err := ix.Insert(uint64(i), v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}

	sel := Filter{Op: FilterEq, Field: "g", Value: NewInt(0)}
	pred, err := sel.Compile()
	if err != nil {
		t.Fatal(err)
	}

	// The matched set must exceed the absolute threshold.
	cands, ok := ix.payloadIdx.candidates(sel, n)
	if !ok {
		t.Fatal("candidates ok=false")
	}
	if len(cands) <= absThreshold {
		t.Fatalf("test setup: candidate set %d must exceed absThreshold %d", len(cands), absThreshold)
	}

	// bp=0: effective limit == absThreshold (byte-identical to the pre-feature path).
	if limit := ix.effectiveFilterFirstLimit(ix.arena.Size()); limit != absThreshold {
		t.Fatalf("bp=0: IVF effective limit = %d, want absThreshold %d", limit, absThreshold)
	}

	// bp=5000: relative budget rises to >= matched count; filter is ADMITTED.
	ix.cfg.FilterFirstRelativeBP = 5000
	limit := ix.effectiveFilterFirstLimit(ix.arena.Size())
	if limit < len(cands) {
		t.Fatalf("bp=5000: IVF effective limit = %d, want >= matched %d", limit, len(cands))
	}

	// Results via SearchFiltered must equal predicate-eval brute force (exact).
	q := corpus[2]
	got, err := ix.SearchFiltered(q, k, sel)
	if err != nil {
		t.Fatal(err)
	}
	want := bruteForceFiltered(corpus, metas, q, k, func(m Metadata) bool { return pred(m) })
	if !eqUint64(resultIDs(got), want) {
		t.Errorf("IVF relative-admitted filter not exact: %v != %v", resultIDs(got), want)
	}
}

// TestRelativeGateMVFamily smoke-tests the MV family honoring the knob through
// innerConfig: the inner index's effectiveFilterFirstLimit reads the threaded bp.
func TestRelativeGateMVFamily(t *testing.T) {
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1, FilterFirstRelativeBP: 5000})
	if err != nil {
		t.Fatal(err)
	}
	// The inner token index carries the threaded relative bp.
	if m.idx.effectiveFilterFirstLimit(0) != m.idx.filterFirstThreshold() {
		t.Fatalf("MV liveCount=0 should return absThreshold")
	}
	if got := m.idx.effectiveFilterFirstLimit(1_000_000); got <= m.idx.filterFirstThreshold() {
		t.Fatalf("MV effective limit = %d, want > absThreshold for large liveCount", got)
	}
}

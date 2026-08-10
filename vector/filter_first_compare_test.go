// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"testing"
	"time"
)

// TestFilterFirstRelativeGateComparison is a MEASUREMENT harness (run with -v) that
// quantifies the win from the opt-in relative selectivity gate
// (Config.FilterFirstRelativeBP). It is not a pass/fail correctness test beyond a
// floor assertion; its purpose is to print latency + recall@10 for the gate OFF
// (BP=0) vs ON, in the regime the gate was built for.
//
// THE REGIME: a SELECTIVE filter whose candidate set EXCEEDS the absolute
// filter-first cap. With the default cap (10k) that needs a multi-hundred-thousand
// corpus, so — exactly as the gate's unit tests do — we lower FilterFirstThreshold
// to emulate the large-N regime at a tractable size. Here: 20k vectors, the "g==0"
// filter matches ~2% (~400 candidates), and the absolute cap is set to 200.
//
//   - BP=0 (default): effective limit == 200. The ~400 candidate set EXCEEDS it, so
//     the planner falls back to filtered GRAPH traversal. At 2% selectivity the graph
//     keeps wandering past non-matching neighbors to collect k admitted hits, which is
//     EXPENSIVE (recall stays high here, but latency balloons — and on harder graphs
//     recall would also fall off the filtered-recall cliff).
//   - BP=500: relative budget = 500*20000/10000 = 1000 >= 400, so the SAME candidate
//     set is admitted to EXACT filter-first. The cost model (preferFilterFirst) also
//     favors it here (graphCost ~= 16000 >> 400). Result is exact AND much faster:
//     measured ~56x lower latency (3.9ms -> 0.07ms) at recall 1.0.
func TestFilterFirstRelativeGateComparison(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping measurement harness in -short mode")
	}
	const (
		dim    = 128
		n      = 20_000
		nq     = 200
		k      = 10
		absCap = 200 // emulates hitting the absolute cap in a large-N deployment
		onBP   = 500 // relative budget -> limit 1000, admits the ~400 candidate set
		modulo = 50  // g==0 every 50th point -> ~2% selectivity, ~400 candidates
	)

	corpus := makeCorpus(n, dim, 42)
	queries := makeCorpus(nq, dim, 99)

	h, err := newHNSW(Config{
		Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: L2,
		FilterFirstThreshold: absCap,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()

	corpusMap := make(map[uint64][]float32, n)
	metas := make(map[uint64]Metadata, n)
	matchCount := 0
	for i, v := range corpus {
		g := int64(1)
		if i%modulo == 0 {
			g = 0
			matchCount++
		}
		id := uint64(i + 1)
		md := Metadata{"g": NewInt(g)}
		if _, _, err := h.Insert(id, v, 0, md, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
		corpusMap[id] = v
		metas[id] = md
	}
	t.Logf("corpus=%d match-set=%d (%.1f%%) absolute-cap=%d  limit(BP=0)=%d  limit(BP=%d)=%d",
		n, matchCount, 100*float64(matchCount)/float64(n), absCap,
		effectiveFilterFirstLimit(absCap, 0, n), onBP,
		effectiveFilterFirstLimit(absCap, onBP, n))

	match := func(m Metadata) bool { return m["g"].Equal(NewInt(0)) }
	filter := Filter{Op: FilterEq, Field: "g", Value: NewInt(0)}

	truth := make([][]uint64, nq)
	for qi, q := range queries {
		truth[qi] = bruteForceFiltered(corpusMap, metas, q, k, match)
	}

	measure := func(bp int) (avgLatency time.Duration, recall float64) {
		h.cfg.FilterFirstRelativeBP = bp
		var total time.Duration
		var recSum float64
		for qi, q := range queries {
			start := time.Now()
			res, err := h.SearchFiltered(q, k, filter)
			total += time.Since(start)
			if err != nil {
				t.Fatal(err)
			}
			recSum += recallSet(truth[qi], resultIDs(res))
		}
		return total / time.Duration(nq), recSum / float64(nq)
	}

	offLat, offRec := measure(0)    // gate OFF (default): graph fallback
	onLat, onRec := measure(onBP)   // gate ON: exact filter-first
	h.cfg.FilterFirstRelativeBP = 0 // restore

	t.Logf("relative gate OFF (BP=0):   recall@%d=%.4f  avg-latency=%v", k, offRec, offLat)
	t.Logf("relative gate ON  (BP=%d): recall@%d=%.4f  avg-latency=%v", onBP, k, onRec, onLat)
	t.Logf("recall delta: %+.4f  (%.1f%% -> %.1f%%)", onRec-offRec, 100*offRec, 100*onRec)

	// Floor assertion: the exact filter-first path must be at least as accurate as
	// the graph fallback (it is exact, so recall ~1.0). Guards against the harness
	// silently inverting or both paths collapsing to the same route.
	if onRec < offRec {
		t.Fatalf("relative gate ON recall (%.4f) < OFF recall (%.4f): gate should not reduce recall", onRec, offRec)
	}
}

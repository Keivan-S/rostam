// SPDX-License-Identifier: Apache-2.0

// Command filtered-recall-cliff demonstrates what a selective metadata filter
// costs a vector search, and how Rostam's filter-first planner sidesteps it.
//
// It compares three strategies on identical data as the filter tightens:
//
//   - naive post-filter: retrieve the nearest by distance, then filter in app
//     code. Recall falls off a cliff — the nearest results rarely match a
//     selective filter.
//   - filter-aware graph: Rostam's SearchFiltered graph path keeps recall high
//     (the traversal respects the filter) but latency explodes as it explores
//     ever more of the graph.
//   - filter-first: Rostam's planner ranks the indexed matches directly —
//     exact recall AND low latency.
//
// The two Rostam strategies are the same engine on the same data; only
// Config.FilterFirstThreshold differs (1 = always graph; large = always
// filter-first). Run:
//
//	go run ./examples/filtered-recall-cliff
package main

import (
	"fmt"
	"math/rand"
	"sort"
	"time"

	"github.com/rostamlabs/rostam/vector"
)

const (
	n    = 20_000 // corpus size
	dim  = 32     // vector dimensionality
	nq   = 200    // queries averaged per selectivity
	k    = 10     // top-k
	seed = 1

	// poolNaive is how many nearest-by-distance results the naive post-filter
	// retrieves before filtering — a charitable 10× overfetch over k.
	poolNaive = 100
)

// selectivity tiers: each vector gets an independent bucket value in [0, g);
// filtering "g_<g> == 0" matches ~1/g of the corpus.
var groups = []int{2, 5, 10, 20, 50, 100, 200, 500, 1000}

func main() {
	rng := rand.New(rand.NewSource(seed))

	// Build the corpus: random vectors, each tagged with one bucket per tier.
	vecs := make([][]float32, n)
	buckets := make([][]int64, n) // buckets[i][t] = vector i's value for tier t
	for i := range vecs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		vecs[i] = v
		b := make([]int64, len(groups))
		for t, g := range groups {
			b[t] = int64(rng.Intn(g))
		}
		buckets[i] = b
	}
	queries := make([][]float32, nq)
	for i := range queries {
		q := make([]float32, dim)
		for j := range q {
			q[j] = float32(rng.NormFloat64())
		}
		queries[i] = q
	}

	meta := func(i int) vector.Metadata {
		m := make(vector.Metadata, len(groups))
		for t, g := range groups {
			m[fmt.Sprintf("g_%d", g)] = vector.NewInt(buckets[i][t])
		}
		return m
	}

	// Two Rostam collections, identical data, differing only in the planner
	// threshold: `graph` always takes the filter-aware graph path, `first`
	// always takes the filter-first path.
	build := func(threshold int) *vector.Collection {
		col, err := vector.NewCollection("demo", vector.Config{
			Dim: dim, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64,
			Seed: seed, FilterFirstThreshold: threshold,
		})
		if err != nil {
			panic(err)
		}
		for i := range vecs {
			if err := col.Insert(uint64(i+1), vecs[i], 0, meta(i), nil); err != nil {
				panic(err)
			}
		}
		return col
	}
	graph := build(1)       // candidate set > 1 ⇒ filter-aware graph path
	first := build(1 << 30) // always filter-first

	// The three strategies compared:
	//   naive post-filter — retrieve the top `poolNaive` by distance only
	//     (unfiltered), then keep the ones matching the filter. This is what you
	//     get filtering in application code, or from an index whose traversal
	//     ignores the filter. Recall collapses as the filter tightens.
	//   filter-aware graph — Rostam's SearchFiltered graph path: the traversal
	//     itself respects the filter, so recall holds — but it explores far more
	//     of the graph, so latency explodes.
	//   filter-first — Rostam's planner pulls matches from the payload index and
	//     ranks them exactly: exact recall AND low latency.

	// Brute-force exact top-k among the vectors matching tier t (value 0).
	truth := func(q []float32, t int) map[uint64]bool {
		type cd struct {
			id uint64
			d  float32
		}
		var cs []cd
		for i, v := range vecs {
			if buckets[i][t] != 0 {
				continue
			}
			var d float32
			for j := range v {
				e := v[j] - q[j]
				d += e * e
			}
			cs = append(cs, cd{uint64(i + 1), d})
		}
		sort.Slice(cs, func(a, b int) bool { return cs[a].d < cs[b].d })
		out := make(map[uint64]bool, k)
		for i := 0; i < len(cs) && i < k; i++ {
			out[cs[i].id] = true
		}
		return out
	}

	recallAndLatency := func(col *vector.Collection, t int) (float64, time.Duration) {
		g := groups[t]
		filter := vector.Filter{Op: vector.FilterEq, Field: fmt.Sprintf("g_%d", g), Value: vector.NewInt(0)}
		var matched, total int
		start := time.Now()
		for _, q := range queries {
			tr := truth(q, t)
			if len(tr) == 0 {
				continue
			}
			res, err := col.SearchFiltered(q, k, filter)
			if err != nil {
				panic(err)
			}
			for _, r := range res {
				if tr[r.ID] {
					matched++
				}
			}
			total += len(tr) // denominator caps at the number of true matches
		}
		// timing excludes brute-force truth: re-run searches only.
		st := time.Now()
		for _, q := range queries {
			_, _ = col.SearchFiltered(q, k, filter)
		}
		lat := time.Since(st) / time.Duration(len(queries))
		_ = start
		if total == 0 {
			return 0, lat
		}
		return float64(matched) / float64(total), lat
	}

	// naiveRecall models post-filtering: retrieve poolNaive nearest by distance
	// (unfiltered), then keep those matching the filter and take the top k.
	naiveRecall := func(col *vector.Collection, t int) float64 {
		var matched, total int
		for _, q := range queries {
			tr := truth(q, t)
			if len(tr) == 0 {
				continue
			}
			res, err := col.Search(q, poolNaive)
			if err != nil {
				panic(err)
			}
			kept := 0
			for _, r := range res {
				if buckets[r.ID-1][t] != 0 { // post-filter in "app code"
					continue
				}
				if tr[r.ID] {
					matched++
				}
				if kept++; kept == k {
					break
				}
			}
			total += len(tr)
		}
		if total == 0 {
			return 0
		}
		return float64(matched) / float64(total)
	}

	type row struct {
		g          int
		matches    int
		nr, wr, fr float64       // recall: naive, filter-aware graph, filter-first
		wl, fl     time.Duration // latency: filter-aware graph, filter-first
	}
	rows := make([]row, len(groups))
	for t, g := range groups {
		nr := naiveRecall(graph, t)
		wr, wl := recallAndLatency(graph, t)
		fr, fl := recallAndLatency(first, t)
		rows[t] = row{g, n / g, nr, wr, fr, wl, fl}
	}
	_ = graph.Close()
	_ = first.Close()

	fmt.Println("Filtered vector search: the cost of a selective metadata filter")
	fmt.Printf("corpus=%d  dim=%d  k=%d  queries=%d  (L2, M=16, efSearch=64)\n", n, dim, k, nq)

	fmt.Println("\n[1] RECALL@10 — naive post-filter falls off a cliff; filter-first is exact")
	fmt.Printf("%-13s %-8s | %-14s %-15s %-12s\n", "selectivity", "matches", "naive post-filt", "filter-aware grf", "filter-first")
	fmt.Println("---------------------------------------------------------------------")
	for _, r := range rows {
		fmt.Printf("%-13s %-8d | %-14.3f %-15.3f %-12.3f\n",
			fmt.Sprintf("1/%d (%.1f%%)", r.g, 100.0/float64(r.g)), r.matches, r.nr, r.wr, r.fr)
	}

	fmt.Println("\n[2] LATENCY — filter-aware graph holds recall but explodes in latency;")
	fmt.Println("    filter-first gets cheaper as the filter tightens")
	fmt.Printf("%-13s %-8s | %-15s %-14s %-10s\n", "selectivity", "matches", "filter-aware grf", "filter-first", "speedup")
	fmt.Println("---------------------------------------------------------------------")
	for _, r := range rows {
		speedup := float64(r.wl) / float64(r.fl)
		fmt.Printf("%-13s %-8d | %-15s %-14s %-9.0f×\n",
			fmt.Sprintf("1/%d (%.1f%%)", r.g, 100.0/float64(r.g)), r.matches,
			r.wl.Round(time.Microsecond).String(), r.fl.Round(time.Microsecond).String(), speedup)
	}

	fmt.Println("\nTakeaway: on selective filters you otherwise trade away either recall")
	fmt.Println("(naive post-filtering) or latency (filter-aware graph traversal). Rostam's")
	fmt.Println("filter-first planner gives both — exact results AND low latency — by ranking")
	fmt.Println("the indexed matches directly.")
}

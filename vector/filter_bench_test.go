// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"strconv"
	"sync"
	"testing"
)

// Wave-5 benchmarks for filtering & payload: compiled-predicate evaluation across
// every operator family (equality / range / contains / match / geo / compound)
// and end-to-end filtered search. Before this file only plain equality
// SearchFiltered was benchmarked. See BENCHMARKS.md.

const (
	flN   = 20_000
	flDim = 128
)

var (
	flOnce  sync.Once
	flIdx   *hnsw
	flQuery []float32
	flMetas []Metadata // a representative metadata corpus for predicate-eval benchmarks
)

func filterBenchIndex(tb testing.TB) (*hnsw, []float32, []Metadata) {
	tb.Helper()
	flOnce.Do(func() {
		h, err := newHNSW(Config{Dim: flDim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
		if err != nil {
			panic(err)
		}
		corpus := makeCorpus(flN, flDim, 42)
		metas := make([]Metadata, flN)
		for i := 0; i < flN; i++ {
			m := flMetaFor(i)
			metas[i] = m
			if _, _, err := h.Insert(uint64(i+1), corpus[i], 0, m, nil, nil, CASCond{}); err != nil {
				panic(err)
			}
		}
		flIdx = h
		flMetas = metas
		flQuery = makeCorpus(1, flDim, 99)[0]
	})
	return flIdx, flQuery, flMetas
}

// flMetaFor builds a deterministic rich metadata record exercising every operator:
// a categorical (bucket), a numeric (priority), a datetime (ts, unix-ms), a string
// list (tags), a text field (title), and a geo point (loc).
func flMetaFor(i int) Metadata {
	return Metadata{
		"bucket":   NewString([]string{"hit", "miss"}[i%2]),
		"priority": NewInt(int64(i % 100)),
		// int64 arithmetic: the epoch-ms literal does not fit a 32-bit int, so an
		// untyped-constant expression here fails to compile under GOARCH=386.
		"ts":    NewInt(int64(1_700_000_000_000) + int64(i)*60_000), // ~1-min spacing
		"tags":  NewStrings([]string{"prod", "team-" + strconv.Itoa(i%8)}),
		"title": NewString("the quick brown fox number " + strconv.Itoa(i%500)),
		"loc":   NewGeo(37.0+float64(i%90)*0.1, -122.0+float64(i%90)*0.1),
	}
}

// flFilters is the operator matrix shared by the eval + search benchmarks.
func flFilters() []struct {
	name string
	f    Filter
} {
	return []struct {
		name string
		f    Filter
	}{
		{"eq", Filter{Op: FilterEq, Field: "bucket", Value: NewString("hit")}},
		{"range", Filter{Op: FilterGte, Field: "priority", Value: NewInt(50)}},
		{"datetime", Filter{Op: FilterDtGte, Field: "ts", Value: NewString("2023-11-15T00:00:00Z")}},
		{"contains", Filter{Op: FilterContains, Field: "tags", Value: NewString("prod")}},
		{"match", Filter{Op: FilterMatch, Field: "title", Value: NewString("quick fox")}},
		{"geo_radius", Filter{Op: FilterGeoRadius, Field: "loc", Geo: &GeoCondition{CenterLat: 37.5, CenterLon: -121.5, RadiusM: 500_000}}},
		{"compound", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "bucket", Value: NewString("hit")},
			{Op: FilterOr, Or: []Filter{
				{Op: FilterGte, Field: "priority", Value: NewInt(50)},
				{Op: FilterContains, Field: "tags", Value: NewString("team-3")},
			}},
			{Op: FilterNot, Not: &Filter{Op: FilterEq, Field: "priority", Value: NewInt(0)}},
		}}},
	}
}

// BenchmarkPredicateEval isolates the compiled-predicate CPU: it compiles each
// filter once, then evaluates it across the whole metadata corpus. This measures
// the per-record predicate cost (the inner loop of every filtered search) without
// the ANN traversal.
func BenchmarkPredicateEval(b *testing.B) {
	_, _, metas := filterBenchIndex(b)
	for _, tc := range flFilters() {
		pred, err := tc.f.Compile()
		if err != nil {
			b.Fatalf("%s compile: %v", tc.name, err)
		}
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			var matched int
			for i := 0; i < b.N; i++ {
				if pred(metas[i%len(metas)]) {
					matched++
				}
			}
			_ = matched
		})
	}
}

// BenchmarkFilterCompile measures Filter.Compile (predicate-tree construction),
// run per-query on every filtered search.
func BenchmarkFilterCompile(b *testing.B) {
	for _, tc := range flFilters() {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := tc.f.Compile(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkSearchFilteredByOp measures end-to-end filtered kNN per operator family
// (compile + ANN traversal with the per-candidate predicate gate).
func BenchmarkSearchFilteredByOp(b *testing.B) {
	h, q, _ := filterBenchIndex(b)
	const k = 10
	for _, tc := range flFilters() {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := h.SearchFiltered(q, k, tc.f); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

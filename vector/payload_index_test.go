// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math/rand"
	"sort"
	"testing"
)

// bruteForceFiltered returns the k ids closest to q (L2) among corpus entries
// whose metadata satisfies match, as an exact reference for the filter-first
// search path.
func bruteForceFiltered(corpus map[uint64][]float32, metas map[uint64]Metadata, q []float32, k int, match func(Metadata) bool) []uint64 {
	type cd struct {
		id uint64
		d  float32
	}
	var cands []cd
	for id, v := range corpus {
		if !match(metas[id]) {
			continue
		}
		cands = append(cands, cd{id, l2SquaredScalar(q, v)})
	}
	sort.Slice(cands, func(a, b int) bool {
		if cands[a].d != cands[b].d {
			return cands[a].d < cands[b].d
		}
		return cands[a].id < cands[b].id
	})
	out := make([]uint64, 0, k)
	for i := 0; i < len(cands) && i < k; i++ {
		out = append(out, cands[i].id)
	}
	return out
}

func resultIDs(rs []Result) []uint64 {
	out := make([]uint64, len(rs))
	for i, r := range rs {
		out[i] = r.ID
	}
	return out
}

func sortedIDs(ids []uint64) []uint64 {
	out := append([]uint64(nil), ids...)
	sort.Slice(out, func(a, b int) bool { return out[a] < out[b] })
	return out
}

func eqUint64(a, b []uint64) bool {
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

// TestFilterFirstExact builds a corpus with an indexed "cat" field and verifies
// that an equality-filtered search (which the planner answers via the payload
// index + brute force) returns exactly the brute-force ground truth — including
// for an And of an indexed equality and a non-indexed range (the index narrows,
// the predicate re-check stays exact).
func TestFilterFirstExact(t *testing.T) {
	const (
		n   = 600
		dim = 8
		k   = 10
	)
	rng := rand.New(rand.NewSource(11))
	h, err := newHNSW(Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	corpus := make(map[uint64][]float32, n)
	metas := make(map[uint64]Metadata, n)
	for i := 1; i <= n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		id := uint64(i)
		meta := Metadata{
			"cat":   NewInt(int64(i % 10)),
			"score": NewFloat(rng.Float64()),
		}
		corpus[id] = v
		metas[id] = meta
		if _, _, err := h.Insert(id, v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	q := make([]float32, dim)
	for j := range q {
		q[j] = float32(rng.NormFloat64())
	}

	// Pure indexed equality.
	eq := Filter{Op: FilterEq, Field: "cat", Value: NewInt(3)}
	got, err := h.SearchFiltered(q, k, eq)
	if err != nil {
		t.Fatal(err)
	}
	want := bruteForceFiltered(corpus, metas, q, k, func(m Metadata) bool { return m["cat"].Int == 3 })
	if !eqUint64(resultIDs(got), want) {
		t.Errorf("eq filter: filter-first ids %v != brute-force %v", resultIDs(got), want)
	}

	// And(indexed eq, non-indexed range): index narrows by cat, predicate
	// re-check enforces the range — must still be exact.
	andF := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterEq, Field: "cat", Value: NewInt(3)},
		{Op: FilterGte, Field: "score", Value: NewFloat(0.5)},
	}}
	got2, err := h.SearchFiltered(q, k, andF)
	if err != nil {
		t.Fatal(err)
	}
	want2 := bruteForceFiltered(corpus, metas, q, k, func(m Metadata) bool {
		return m["cat"].Int == 3 && m["score"].Flt >= 0.5
	})
	if !eqUint64(resultIDs(got2), want2) {
		t.Errorf("and(eq,range): filter-first ids %v != brute-force %v", resultIDs(got2), want2)
	}
}

// TestPayloadIndexReuseAfterReclaim verifies the payload index stays correct
// across delete -> reclaim -> reinsert (slot reuse): a query for a value only
// present on reused slots must return exactly those, and no stale deleted ids
// may surface.
func TestPayloadIndexReuseAfterReclaim(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	vec := func(i int) []float32 { return []float32{float32(i), 0, 0, 0} }
	for i := 1; i <= 20; i++ {
		meta := Metadata{"cat": NewInt(int64(i % 2))} // cat 0 or 1
		if _, _, err := h.Insert(uint64(i), vec(i), 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	// Delete ids 1..10, then physically reclaim their slots.
	for i := 1; i <= 10; i++ {
		h.Delete(uint64(i), CASCond{})
	}
	h.Reclaim()

	// Reinsert ids 21..30 (reusing freed slots) with a brand-new cat value 5.
	for i := 21; i <= 30; i++ {
		if _, _, err := h.Insert(uint64(i), vec(i), 0, Metadata{"cat": NewInt(5)}, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}

	// cat==5 must return only the reused-slot ids 21..30.
	got, err := h.SearchFiltered([]float32{25, 0, 0, 0}, 10, Filter{Op: FilterEq, Field: "cat", Value: NewInt(5)})
	if err != nil {
		t.Fatal(err)
	}
	ids := sortedIDs(resultIDs(got))
	want := []uint64{21, 22, 23, 24, 25, 26, 27, 28, 29, 30}
	if !eqUint64(ids, want) {
		t.Errorf("cat==5 after reuse = %v, want %v", ids, want)
	}

	// No deleted id (1..10) may appear for cat==0 or cat==1.
	for _, val := range []int64{0, 1} {
		res, err := h.SearchFiltered([]float32{1, 0, 0, 0}, 20, Filter{Op: FilterEq, Field: "cat", Value: NewInt(val)})
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range res {
			if r.ID >= 1 && r.ID <= 10 {
				t.Errorf("cat==%d returned deleted id %d", val, r.ID)
			}
		}
	}
}

// TestPayloadIndexSnapshotRoundtrip verifies the payload index is rebuilt on
// Restore, so equality-filtered search works against a restored index.
func TestPayloadIndexSnapshotRoundtrip(t *testing.T) {
	src, err := newHNSW(Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 30; i++ {
		meta := Metadata{"cat": NewInt(int64(i % 3))}
		if _, _, err := src.Insert(uint64(i), []float32{float32(i), 0, 0, 0}, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	var buf bytes.Buffer
	if err := src.Snapshot(&buf); err != nil {
		t.Fatal(err)
	}
	dst, err := newHNSW(Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := dst.Restore(&buf); err != nil {
		t.Fatal(err)
	}
	got, err := dst.SearchFiltered([]float32{1, 0, 0, 0}, 20, Filter{Op: FilterEq, Field: "cat", Value: NewInt(1)})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range got {
		if r.ID%3 != 1 {
			t.Errorf("restored cat==1 search returned id %d (cat %d)", r.ID, r.ID%3)
		}
	}
	if len(got) == 0 {
		t.Error("restored cat==1 search returned no results")
	}
}

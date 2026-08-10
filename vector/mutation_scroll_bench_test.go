// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"strconv"
	"sync"
	"testing"
)

// Wave-6 benchmarks for payload mutations (set / overwrite / delete-keys / clear,
// plain and CAS) and scroll/order-by pagination. None of these were benchmarked
// before. See BENCHMARKS.md.

const (
	msN   = 20_000
	msDim = 128
)

var (
	msOnce     sync.Once
	msColl     *Collection
	msReadOnce sync.Once
	msReadColl *Collection
)

func buildMutationCollection(name string) *Collection {
	c, err := NewCollection(name, Config{Dim: msDim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		panic(err)
	}
	corpus := makeCorpus(msN, msDim, 42)
	for i := 0; i < msN; i++ {
		meta := Metadata{
			"priority": NewInt(int64(i % 1000)),
			"title":    NewString("doc " + strconv.Itoa(i)),
			"bucket":   NewString([]string{"a", "b", "c"}[i%3]),
		}
		if err := c.Insert(uint64(i+1), corpus[i], 0, meta, nil); err != nil {
			panic(err)
		}
	}
	return c
}

// mutationBenchCollection is the (mutable) collection the payload-mutation
// benchmarks write to.
func mutationBenchCollection(tb testing.TB) *Collection {
	tb.Helper()
	msOnce.Do(func() { msColl = buildMutationCollection("msbench") })
	return msColl
}

// scrollBenchCollection is a SEPARATE pristine collection for the read-only
// scroll/order-by benchmarks — isolated so the mutation benchmarks (which delete
// the "title" field, etc.) cannot corrupt the fields the scroll sorts on.
func scrollBenchCollection(tb testing.TB) *Collection {
	tb.Helper()
	msReadOnce.Do(func() { msReadColl = buildMutationCollection("msbench-read") })
	return msReadColl
}

// BenchmarkSetPayload measures a merge-patch payload update (re-index of the
// changed field). Cycles ids so the working set spans the corpus.
func BenchmarkSetPayload(b *testing.B) {
	c := mutationBenchCollection(b)
	patch := Metadata{"priority": NewInt(7)}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := c.SetPayload(uint64(i%msN+1), patch, nil); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSetPayloadCAS measures the same merge-patch with an optimistic version
// precondition (the read-version + check + apply path).
func BenchmarkSetPayloadCAS(b *testing.B) {
	c := mutationBenchCollection(b)
	patch := Metadata{"priority": NewInt(9)}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := uint64(i%msN + 1)
		// Read the current version so the CAS precondition is satisfied.
		_, _, _, _, ver, ok := c.idx.Get(id)
		if !ok {
			b.Fatalf("missing id %d", id)
		}
		if _, err := c.SetPayloadCAS(id, patch, nil, CASCond{Expected: ver, Has: true}); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkOverwritePayload measures a full payload replace (drop old fields from
// the indexes, index the new set).
func BenchmarkOverwritePayload(b *testing.B) {
	c := mutationBenchCollection(b)
	meta := Metadata{"priority": NewInt(3), "bucket": NewString("a")}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := c.OverwritePayload(uint64(i%msN+1), meta, nil); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkDeletePayloadKeys measures selective key removal (de-index those keys).
func BenchmarkDeletePayloadKeys(b *testing.B) {
	c := mutationBenchCollection(b)
	keys := []string{"title"}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := c.DeletePayloadKeys(uint64(i%msN+1), keys); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkScrollDocsPage measures cursor pagination by id (one page per op).
func BenchmarkScrollDocsPage(b *testing.B) {
	c := scrollBenchCollection(b)
	const pageSize = 100
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, _, err := c.ScrollDocsPage(Filter{}, 0, false, pageSize); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkScrollDocsPageOrder measures order-by pagination (first page) for the
// numeric and string key paths. The first call after a mutation rebuilds the
// order snapshot; steady state hits the cached snapshot.
func BenchmarkScrollDocsPageOrder(b *testing.B) {
	c := scrollBenchCollection(b)
	const pageSize = 100
	cases := []struct {
		name  string
		order *OrderBy
	}{
		{"numeric", &OrderBy{Key: "priority", Kind: OrderNumeric}},
		{"string", &OrderBy{Key: "title", Kind: OrderString}},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, _, _, err := c.ScrollDocsPageOrder(Filter{}, tc.order, 0, 0, false, pageSize); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"strconv"
	"testing"
)

// BenchmarkQuantizedSearch compares end-to-end search latency across the three
// quantization modes at a fixed corpus/dim. SQ8 and BQ1 navigate on cheaper
// codes but over-collect (RescoreFactor) and rescore on float32, so this
// captures the real per-query tradeoff, not just the kernel cost.
func BenchmarkQuantizedSearch(b *testing.B) {
	const (
		dim = 128
		n   = 10_000
		k   = 10
	)
	corpus := makeCorpus(n, dim, 42)
	queries := makeCorpus(1_000, dim, 99)

	modes := []struct {
		name    string
		mode    QuantMode
		rescore int
	}{
		{"none", QuantNone, 0},
		{"sq8", QuantSQ8, 3},
		{"bq1", QuantBQ1, 32},
	}
	for _, m := range modes {
		h, err := newHNSW(Config{
			Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64,
			Seed: 1, Metric: Cosine, Quant: m.mode, RescoreFactor: m.rescore,
		})
		if err != nil {
			b.Fatal(err)
		}
		for i, v := range corpus {
			if _, _, err := h.Insert(uint64(i+1), v, 0, nil, nil, nil, CASCond{}); err != nil {
				b.Fatal(err)
			}
		}
		b.Run(m.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = h.Search(queries[i%len(queries)], k)
			}
		})
		_ = h.Close()
	}
}

// BenchmarkSQ8Encode measures the int8 scalar-quantization encode (clamp +
// round, one component per dim) at representative dims — the per-insert cost for
// an SQ8 collection.
func BenchmarkSQ8Encode(b *testing.B) {
	for _, dim := range []int{128, 768, 1536} {
		q := newSQ8(dim)
		v := makeCorpus(1, dim, 7)[0]
		dst := make([]byte, q.CodeLen())
		b.Run("dim="+strconv.Itoa(dim), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				q.Encode(dst, v)
			}
		})
	}
}

// BenchmarkBQ1Encode measures the sign-bit packing encode (zero + set-bit per
// positive component) — the per-insert cost for a BQ1 collection.
func BenchmarkBQ1Encode(b *testing.B) {
	for _, dim := range []int{128, 768, 1536} {
		q := newBQ1(dim)
		v := makeCorpus(1, dim, 7)[0]
		dst := make([]byte, q.CodeLen())
		b.Run("dim="+strconv.Itoa(dim), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				q.Encode(dst, v)
			}
		})
	}
}

// BenchmarkPQEncode measures the per-subspace nearest-sub-centroid search that
// the insert path runs for every PQ-coded vector — m subspaces × 256 centroids ×
// dsub-length L2. Compares plain PQ against OPQ, whose encode additionally rotates
// the input (Rx) before the subspace split.
func BenchmarkPQEncode(b *testing.B) {
	const dim, m = 128, 16
	vecs := makeCorpus(2_000, dim, 42)
	for _, tc := range []struct {
		name string
		opq  bool
	}{
		{"pq", false},
		{"opq", true},
	} {
		p, err := trainPQ(vecs, m, dim, 7, L2, 1, tc.opq, 0, 1, 8)
		if err != nil {
			b.Fatal(err)
		}
		v := vecs[0]
		dst := make([]byte, p.CodeLen())
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				p.encodeInto(dst, v)
			}
		})
	}
}

// BenchmarkPQQueryLUT measures the per-cell ADC LUT build the IVF gather runs
// once per probed cell (queryLUTInto on the residual). Compares plain PQ against
// OPQ, whose LUT build additionally rotates the query (Rq) — that rotation must
// stay allocation-free or it costs one alloc per probed cell per query.
func BenchmarkPQQueryLUT(b *testing.B) {
	const dim, m = 128, 16
	vecs := makeCorpus(2_000, dim, 42)
	for _, tc := range []struct {
		name string
		opq  bool
	}{
		{"pq", false},
		{"opq", true},
	} {
		p, err := trainPQ(vecs, m, dim, 7, L2, 1, tc.opq, 0, 1, 8)
		if err != nil {
			b.Fatal(err)
		}
		q := vecs[0]
		lut := make([]float32, p.lutLen())
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				p.queryLUTInto(lut, q)
			}
		})
	}
}

// BenchmarkBQ1Hamming measures the Hamming-distance kernel (uint64-chunked
// popcount) at representative code lengths.
func BenchmarkBQ1Hamming(b *testing.B) {
	for _, dim := range []int{128, 768, 1536} {
		q := newBQ1(dim)
		corpus := makeCorpus(2, dim, 7)
		a := make([]byte, q.CodeLen())
		c := make([]byte, q.CodeLen())
		q.Encode(a, corpus[0])
		q.Encode(c, corpus[1])
		b.Run("dim="+strconv.Itoa(dim), func(b *testing.B) {
			var s int
			for i := 0; i < b.N; i++ {
				s = bq1Hamming(a, c)
			}
			_ = s
		})
	}
}

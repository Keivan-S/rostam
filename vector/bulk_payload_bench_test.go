// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"testing"
)

// THE A/B THIS CHANGE EXISTS TO WIN.
//
// A payload-bearing load had exactly one route — one indexed insert per point,
// because the staging wire carried ids and vectors and nothing else. These two
// benchmarks load the SAME corpus with the SAME payloads the two ways and report
// vec/s, so the claim is measured rather than asserted.
//
// Both are index-level on purpose. The wire was never the bottleneck here: the
// binary framing bought the vector-only bulk path ~13%, because that path is
// index-bound. What the inline route pays is the serialized HNSW link phase, one
// point at a time, and that is what the staged build parallelizes.
//
// Run:
//
//	go test ./vector -run '^$' -bench 'PayloadLoad' -benchtime 1x
//	ROSTAM_BENCH_N=100000 ROSTAM_BENCH_DIM=768 go test ./vector -run '^$' \
//	    -bench 'PayloadLoad' -benchtime 1x -timeout 60m
//
// -benchtime 1x matters: one iteration is a whole corpus load, and the framework
// would otherwise repeat it looking for a stable timing.

func benchEnvInt(key string, def int) int {
	if s := os.Getenv(key); s != "" {
		if v, err := strconv.Atoi(s); err == nil && v > 0 {
			return v
		}
	}
	return def
}

// benchPayloadCorpus builds the load: random unit-ish vectors and the one-scalar
// payload the filtered benchmark case uses ({"id": n}).
func benchPayloadCorpus(n, dim int) ([]uint64, [][]float32, []Metadata) {
	r := rand.New(rand.NewSource(7)) //nolint:gosec // deterministic fixture
	ids := make([]uint64, n)
	vecs := make([][]float32, n)
	metas := make([]Metadata, n)
	flat := make([]float32, n*dim) // one allocation, as the real load path has
	for i := 0; i < n; i++ {
		ids[i] = uint64(i)
		v := flat[i*dim : (i+1)*dim : (i+1)*dim]
		for d := range v {
			v[d] = r.Float32()*2 - 1
		}
		vecs[i] = v
		metas[i] = Metadata{"id": NewInt(int64(i))}
	}
	return ids, vecs, metas
}

func benchPayloadConfig(dim int) Config {
	// The VectorDBBench defaults, so the numbers are comparable with the 1M runs.
	return Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1}
}

// BenchmarkPayloadLoadInline is the OLD route: one indexed insert per point,
// payload and all — what /points/batch does, and what a filter case was forced
// onto.
func BenchmarkPayloadLoadInline(b *testing.B) {
	n := benchEnvInt("ROSTAM_BENCH_N", 20000)
	dim := benchEnvInt("ROSTAM_BENCH_DIM", 768)
	ids, vecs, metas := benchPayloadCorpus(n, dim)
	cfg := benchPayloadConfig(dim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		coll, err := NewCollection("bench_inline", cfg)
		if err != nil {
			b.Fatal(err)
		}
		for j := range ids {
			if err := coll.Insert(ids[j], vecs[j], 0, metas[j], nil); err != nil {
				b.Fatal(err)
			}
		}
		b.StopTimer()
		if got := coll.Stats().Size; got != n {
			b.Fatalf("loaded %d of %d points", got, n)
		}
		_ = coll.Close()
		b.StartTimer()
	}
	b.ReportMetric(float64(n*b.N)/b.Elapsed().Seconds(), "vec/s")
}

// BenchmarkPayloadLoadBulkStaged is the NEW route: stage the same points WITH
// their payloads, then one concurrent build that applies both.
func BenchmarkPayloadLoadBulkStaged(b *testing.B) { benchBulkStaged(b, true) }

// BenchmarkVectorOnlyLoadBulkStaged is the CONTROL: the identical staged build
// with the payload column absent. It is the number the payload-bearing bulk load
// has to be compared against to say what carrying payloads actually costs on this
// path — as opposed to what it cost on the route it replaces.
func BenchmarkVectorOnlyLoadBulkStaged(b *testing.B) { benchBulkStaged(b, false) }

func benchBulkStaged(b *testing.B, withPayloads bool) {
	n := benchEnvInt("ROSTAM_BENCH_N", 20000)
	dim := benchEnvInt("ROSTAM_BENCH_DIM", 768)
	ids, vecs, metas := benchPayloadCorpus(n, dim)
	if !withPayloads {
		metas = nil
	}
	cfg := benchPayloadConfig(dim)
	// The reference chunk size the HTTP bulk route delivers, so the staging call
	// count is realistic rather than one giant call.
	const chunk = 10000

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		coll, err := NewCollection("bench_bulk", cfg)
		if err != nil {
			b.Fatal(err)
		}
		for start := 0; start < n; start += chunk {
			end := start + chunk
			if end > n {
				end = n
			}
			var chunkMetas []Metadata
			if metas != nil {
				chunkMetas = metas[start:end]
			}
			if err := coll.StageBulkPayloads(ids[start:end], vecs[start:end], chunkMetas); err != nil {
				b.Fatal(err)
			}
		}
		if err := coll.BuildStaged(runtime.GOMAXPROCS(0)); err != nil {
			b.Fatal(err)
		}
		b.StopTimer()
		if got := coll.Stats().Size; got != n {
			b.Fatalf("loaded %d of %d points", got, n)
		}
		_ = coll.Close()
		b.StartTimer()
	}
	b.ReportMetric(float64(n*b.N)/b.Elapsed().Seconds(), "vec/s")
}

// SPDX-License-Identifier: Apache-2.0
//go:build amd64

package vector

import (
	"math/rand"
	"os"
	"testing"
	"time"
)

// TestHighDimBuildAVX512 measures the build and search wall-time of a high-dim
// (1536) SQ8+L0full index with the AVX-512 kernels ON vs forced OFF, to quantify
// the dimension-gated AVX-512 win on the high-dim build (which is distance-bound)
// and on search. Opt-in: ROSTAM_HIGHDIM=1; requires an AVX-512 CPU.
//
//	ROSTAM_HIGHDIM=1 go test ./vector/ -run TestHighDimBuildAVX512 -v -timeout 30m
func TestHighDimBuildAVX512(t *testing.T) {
	if os.Getenv("ROSTAM_HIGHDIM") != "1" {
		t.Skip("set ROSTAM_HIGHDIM=1 to run")
	}
	if !avx512Enabled {
		t.Skip("AVX-512 not available on this CPU")
	}
	const (
		n   = 200_000
		dim = 1536
		nq  = 2000
		k   = 100
	)
	rng := rand.New(rand.NewSource(1))
	ids := make([]uint64, n)
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		normalize(v)
		vecs[i] = v
		ids[i] = uint64(i + 1)
	}
	queries := make([][]float32, nq)
	for i := range queries {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		normalize(v)
		queries[i] = v
	}

	run := func(label string, quantizedBuild bool) {
		h, err := newHNSW(Config{Dim: dim, M: 32, EfConstruction: 200, EfSearch: 256, Seed: 42, Metric: Cosine, Level0FullDegree: true, Quant: QuantSQ8, RescoreFactor: 3, QuantizedBuild: quantizedBuild})
		if err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		if err := h.BuildConcurrent(ids, vecs, 16); err != nil {
			t.Fatal(err)
		}
		buildT := time.Since(t0)
		t1 := time.Now()
		for _, q := range queries {
			if _, err := h.Search(q, k); err != nil {
				t.Fatal(err)
			}
		}
		searchUS := float64(time.Since(t1).Microseconds()) / float64(nq)
		t.Logf("%-18s build=%6.1fs  search=%7.1f us/query", label, buildT.Seconds(), searchUS)
		_ = h.Close()
	}

	run("float-build", false)    // navigate+select on exact float32 (default)
	run("quantized-build", true) // navigate+select on int8 codes (the lever)
}

// TestHighDimCodeKernelAB measures the high-dim (1536) quantized-build wall-time
// with the symmetric code kernel forced to AVX-512-VNNI vs AVX2, back-to-back in
// one process so any shared-box contention (e.g. a co-tenant hammering memory
// bandwidth) affects both arms equally. The build navigates + selects neighbors
// on int8 codes (QuantizedBuild), so sq8CodeDot is on the hot path. Opt-in:
// ROSTAM_HIGHDIM=1; requires an AVX-512-VNNI CPU.
//
//	ROSTAM_HIGHDIM=1 go test ./vector/ -run TestHighDimCodeKernelAB -v -timeout 30m
func TestHighDimCodeKernelAB(t *testing.T) {
	if os.Getenv("ROSTAM_HIGHDIM") != "1" {
		t.Skip("set ROSTAM_HIGHDIM=1 to run")
	}
	if !avx512VNNIEnabled {
		t.Skip("AVX-512-VNNI not available on this CPU")
	}
	const (
		n   = 200_000
		dim = 1536
	)
	rng := rand.New(rand.NewSource(3))
	ids := make([]uint64, n)
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		normalize(v)
		vecs[i] = v
		ids[i] = uint64(i + 1)
	}

	saved := sq8CodeDot
	defer func() { sq8CodeDot = saved }()

	build := func(label string, kernel func(a, b []byte) int32) time.Duration {
		sq8CodeDot = kernel
		h, err := newHNSW(Config{Dim: dim, M: 32, EfConstruction: 200, EfSearch: 64, Seed: 42, Metric: Cosine, Level0FullDegree: true, Quant: QuantSQ8, RescoreFactor: 3, QuantizedBuild: true})
		if err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		if err := h.BuildConcurrent(ids, vecs, 16); err != nil {
			t.Fatal(err)
		}
		d := time.Since(t0)
		_ = h.Close()
		t.Logf("%-22s build=%6.1fs", label, d.Seconds())
		return d
	}

	// Interleave to average out contention drift: AVX2, VNNI, VNNI, AVX2.
	a1 := build("avx2-code", sq8CodeDotAVX2)
	v1 := build("vnni-code", sq8CodeDotVNNI)
	v2 := build("vnni-code", sq8CodeDotVNNI)
	a2 := build("avx2-code", sq8CodeDotAVX2)
	avg := func(a, b time.Duration) float64 { return (a.Seconds() + b.Seconds()) / 2 }
	avx2Avg, vnniAvg := avg(a1, a2), avg(v1, v2)
	t.Logf("AVG avx2-code=%6.1fs  vnni-code=%6.1fs  speedup=%.2fx", avx2Avg, vnniAvg, avx2Avg/vnniAvg)
}

// TestHighDimSymNavAB measures the high-dim (1536) quantized-build wall-time with
// symmetric VNNI navigation ON vs OFF, back-to-back in one process (shared-box
// contention hits both arms equally). A CPU profile showed navigation
// (searchLayer) was ~38% of the build and ran on the slower AVX2 asymmetric
// kernel; buildSymNav routes it through the symmetric VNNI kernel too. This
// measures only build time at the target dim — recall is checked separately on
// real data (TestQBuildSymNavGloveAB), since random Gaussian recall is
// uninformative. Opt-in: ROSTAM_HIGHDIM=1; requires AVX-512-VNNI.
//
//	ROSTAM_HIGHDIM=1 go test ./vector/ -run TestHighDimSymNavAB -v -timeout 30m
func TestHighDimSymNavAB(t *testing.T) {
	if os.Getenv("ROSTAM_HIGHDIM") != "1" {
		t.Skip("set ROSTAM_HIGHDIM=1 to run")
	}
	if !avx512VNNIEnabled {
		t.Skip("AVX-512-VNNI not available on this CPU")
	}
	const (
		n   = 200_000
		dim = 1536
	)
	rng := rand.New(rand.NewSource(5))
	ids := make([]uint64, n)
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		normalize(v)
		vecs[i] = v
		ids[i] = uint64(i + 1)
	}

	saved := buildSymNav
	defer func() { buildSymNav = saved }()

	build := func(label string, sym bool) time.Duration {
		buildSymNav = sym
		h, err := newHNSW(Config{Dim: dim, M: 32, EfConstruction: 200, EfSearch: 64, Seed: 42, Metric: Cosine, Level0FullDegree: true, Quant: QuantSQ8, RescoreFactor: 3, QuantizedBuild: true})
		if err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		if err := h.BuildConcurrent(ids, vecs, 16); err != nil {
			t.Fatal(err)
		}
		d := time.Since(t0)
		_ = h.Close()
		t.Logf("%-22s build=%6.1fs", label, d.Seconds())
		return d
	}

	// Interleave: asym, sym, sym, asym.
	a1 := build("asym-nav", false)
	s1 := build("sym-nav", true)
	s2 := build("sym-nav", true)
	a2 := build("asym-nav", false)
	avg := func(a, b time.Duration) float64 { return (a.Seconds() + b.Seconds()) / 2 }
	asymAvg, symAvg := avg(a1, a2), avg(s1, s2)
	t.Logf("AVG asym-nav=%6.1fs  sym-nav=%6.1fs  speedup=%.2fx", asymAvg, symAvg, asymAvg/symAvg)
}

// BenchmarkHighDimBuild does a single 1536-dim SQ8+L0full build, for CPU
// profiling the high-dim build path. Set ROSTAM_QBUILD=1 to profile the
// quantized build (graph built on int8 codes — the path the VNNI code kernel and
// any code-locality work touch); unset profiles the exact-float32 build.
//
//	ROSTAM_HIGHDIM=1 ROSTAM_QBUILD=1 go test ./vector/ -run x -bench BenchmarkHighDimBuild \
//	  -benchtime=1x -cpuprofile=/src/cpu.prof
func BenchmarkHighDimBuild(b *testing.B) {
	if os.Getenv("ROSTAM_HIGHDIM") != "1" {
		b.Skip("set ROSTAM_HIGHDIM=1 to run")
	}
	const n, dim = 200_000, 1536
	qbuild := os.Getenv("ROSTAM_QBUILD") == "1"
	rng := rand.New(rand.NewSource(2))
	ids := make([]uint64, n)
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		normalize(v)
		vecs[i] = v
		ids[i] = uint64(i + 1)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h, err := newHNSW(Config{Dim: dim, M: 32, EfConstruction: 200, EfSearch: 64, Seed: 42, Metric: Cosine, Level0FullDegree: true, Quant: QuantSQ8, RescoreFactor: 3, QuantizedBuild: qbuild})
		if err != nil {
			b.Fatal(err)
		}
		if err := h.BuildConcurrent(ids, vecs, 16); err != nil {
			b.Fatal(err)
		}
		_ = h.Close()
	}
}

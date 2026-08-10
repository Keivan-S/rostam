// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
	"testing"
)

// TestSearchProfile builds a representative-scale index, then captures a CPU
// profile of ONLY the query loop (the build is excluded — a plain -cpuprofile
// benchmark is swamped by the one-time build). Opt-in:
//
//	ROSTAM_PROFILE=1 go test ./vector/ -run TestSearchProfile -v -timeout 10m
//
// Profile is written to /tmp/search.prof; analyze with `go tool pprof`.
func TestSearchProfile(t *testing.T) {
	if os.Getenv("ROSTAM_PROFILE") != "1" {
		t.Skip("set ROSTAM_PROFILE=1 to run")
	}
	const (
		n    = 300_000
		dim  = 128
		k    = 10
		ef   = 128
		iter = 300_000
	)
	rng := rand.New(rand.NewSource(1))
	ids := make([]uint64, n)
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		ids[i] = uint64(i + 1)
		vecs[i] = v
	}
	queries := make([][]float32, 2000)
	for i := range queries {
		q := make([]float32, dim)
		for j := range q {
			q[j] = float32(rng.NormFloat64())
		}
		queries[i] = q
	}

	h, err := newHNSW(Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: ef, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := h.BuildConcurrent(ids, vecs, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatal(err)
	}

	f, err := os.Create("/tmp/search.prof")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	dst := make([]Result, 0, k)
	if err := pprof.StartCPUProfile(f); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < iter; i++ {
		dst, _ = h.SearchInto(dst[:0], queries[i%len(queries)], k, Filter{})
	}
	pprof.StopCPUProfile()
	t.Logf("profiled %d searches (n=%d dim=%d ef=%d) -> /tmp/search.prof", iter, n, dim, ef)
}

// SPDX-License-Identifier: Apache-2.0
//go:build arm64

package vector

import (
	"math"
	"math/rand"
	"strconv"
	"testing"
)

// These tests validate the NEON kernels against the scalar reference. Run them
// on an amd64 host via QEMU:
//
//	CGO_ENABLED=0 GOARCH=arm64 go test -exec=qemu-aarch64-static ./vector/

func TestDotNEONMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range []int{0, 1, 3, 4, 5, 7, 8, 15, 16, 17, 31, 64, 127, 128, 129, 768, 1536} {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
			b[i] = float32(rng.NormFloat64())
		}
		got := dotNEONSlice(a, b)
		want := dotScalar(a, b)
		tol := 1e-4 * (1 + float32(math.Abs(float64(want))))
		if float32(math.Abs(float64(got-want))) > tol {
			t.Errorf("n=%d: dotNEON=%v dotScalar=%v (diff %v > tol %v)", n, got, want, got-want, tol)
		}
	}
}

// BenchmarkDotNEON and BenchmarkL2NEON compare the NEON kernels against the
// scalar reference. They produce meaningful numbers only on native arm64
// hardware — under QEMU the emulated timing is not representative.
func BenchmarkDotNEON(b *testing.B) {
	for _, dim := range []int{128, 768, 1536} {
		c := makeCorpus(2, dim, 7)
		b.Run("scalar/dim="+strconv.Itoa(dim), func(b *testing.B) {
			var s float32
			for i := 0; i < b.N; i++ {
				s = dotScalar(c[0], c[1])
			}
			_ = s
		})
		b.Run("neon/dim="+strconv.Itoa(dim), func(b *testing.B) {
			var s float32
			for i := 0; i < b.N; i++ {
				s = dotNEONSlice(c[0], c[1])
			}
			_ = s
		})
	}
}

func BenchmarkL2NEON(b *testing.B) {
	for _, dim := range []int{128, 768, 1536} {
		c := makeCorpus(2, dim, 7)
		b.Run("scalar/dim="+strconv.Itoa(dim), func(b *testing.B) {
			var s float32
			for i := 0; i < b.N; i++ {
				s = l2SquaredScalar(c[0], c[1])
			}
			_ = s
		})
		b.Run("neon/dim="+strconv.Itoa(dim), func(b *testing.B) {
			var s float32
			for i := 0; i < b.N; i++ {
				s = l2SquaredNEONSlice(c[0], c[1])
			}
			_ = s
		})
	}
}

func TestL2SquaredNEONMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, n := range []int{0, 1, 3, 4, 5, 7, 8, 15, 16, 17, 31, 64, 127, 128, 129, 768, 1536} {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
			b[i] = float32(rng.NormFloat64())
		}
		got := l2SquaredNEONSlice(a, b)
		want := l2SquaredScalar(a, b)
		tol := 1e-4 * (1 + float32(math.Abs(float64(want))))
		if float32(math.Abs(float64(got-want))) > tol {
			t.Errorf("n=%d: l2NEON=%v l2Scalar=%v (diff %v > tol %v)", n, got, want, got-want, tol)
		}
	}
}

// SPDX-License-Identifier: Apache-2.0
//go:build amd64

package vector

import (
	"math"
	"math/rand"
	"testing"
)

func TestDotAVX2MatchesScalar(t *testing.T) {
	if !avx2Enabled {
		t.Skip("AVX2 not available on this CPU")
	}
	rng := rand.New(rand.NewSource(1))
	// Lengths spanning: empty, < 8, exactly 8, 8+tail, and realistic dims.
	for _, n := range []int{0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 64, 127, 128, 129, 768, 1536} {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
			b[i] = float32(rng.NormFloat64())
		}
		got := dotAVX2(a, b)
		want := dotScalar(a, b)
		// SIMD sums lanes in a different order, so allow a small relative error.
		tol := 1e-4 * (1 + float32(math.Abs(float64(want))))
		if float32(math.Abs(float64(got-want))) > tol {
			t.Errorf("n=%d: dotAVX2=%v dotScalar=%v (diff %v > tol %v)", n, got, want, got-want, tol)
		}
	}
}

func TestL2SquaredAVX2MatchesScalar(t *testing.T) {
	if !avx2Enabled {
		t.Skip("AVX2 not available on this CPU")
	}
	rng := rand.New(rand.NewSource(2))
	for _, n := range []int{0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 64, 127, 128, 129, 768, 1536} {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
			b[i] = float32(rng.NormFloat64())
		}
		got := l2SquaredAVX2Slice(a, b)
		want := l2SquaredScalar(a, b)
		tol := 1e-4 * (1 + float32(math.Abs(float64(want))))
		if float32(math.Abs(float64(got-want))) > tol {
			t.Errorf("n=%d: l2AVX2=%v l2Scalar=%v (diff %v > tol %v)", n, got, want, got-want, tol)
		}
	}
}

func TestSQ8DotAVX2MatchesScalar(t *testing.T) {
	if !avx2Enabled {
		t.Skip("AVX2 not available on this CPU")
	}
	rng := rand.New(rand.NewSource(3))
	for _, n := range []int{0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 64, 127, 128, 129, 768, 1536} {
		query := make([]float32, n)
		code := make([]byte, n)
		for i := range query {
			query[i] = float32(rng.NormFloat64())
			code[i] = byte(int8(rng.Intn(255) - 127)) //nolint:gosec // test: arbitrary int8 code values
		}
		got := sq8DotAVX2(query, code)
		want := sq8DotScalar(query, code)
		// SIMD reduces lanes in a different order, so allow a small relative error.
		tol := 1e-3 * (1 + float32(math.Abs(float64(want))))
		if float32(math.Abs(float64(got-want))) > tol {
			t.Errorf("n=%d: sq8DotAVX2=%v sq8DotScalar=%v (diff %v > tol %v)", n, got, want, got-want, tol)
		}
	}
}

func TestDotAVX512MatchesScalar(t *testing.T) {
	if !avx512Enabled {
		t.Skip("AVX-512 not available on this CPU")
	}
	rng := rand.New(rand.NewSource(11))
	for _, n := range []int{0, 1, 3, 7, 8, 15, 16, 17, 31, 33, 63, 64, 65, 100, 127, 128, 129, 768, 1536} {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
			b[i] = float32(rng.NormFloat64())
		}
		got := dotAVX512(a, b)
		want := dotScalar(a, b)
		tol := 1e-4 * (1 + float32(math.Abs(float64(want))))
		if float32(math.Abs(float64(got-want))) > tol {
			t.Errorf("n=%d: dotAVX512=%v dotScalar=%v (diff %v > tol %v)", n, got, want, got-want, tol)
		}
	}
}

func TestL2SquaredAVX512MatchesScalar(t *testing.T) {
	if !avx512Enabled {
		t.Skip("AVX-512 not available on this CPU")
	}
	rng := rand.New(rand.NewSource(12))
	for _, n := range []int{0, 1, 3, 7, 8, 15, 16, 17, 31, 33, 63, 64, 65, 100, 127, 128, 129, 768, 1536} {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
			b[i] = float32(rng.NormFloat64())
		}
		got := l2SquaredAVX512Slice(a, b)
		want := l2SquaredScalar(a, b)
		tol := 1e-4 * (1 + float32(math.Abs(float64(want))))
		if float32(math.Abs(float64(got-want))) > tol {
			t.Errorf("n=%d: l2AVX512=%v l2Scalar=%v (diff %v > tol %v)", n, got, want, got-want, tol)
		}
	}
}

func BenchmarkDotAVX512(b *testing.B) {
	if !avx512Enabled {
		b.Skip("AVX-512 not available on this CPU")
	}
	for _, dim := range []int{64, 100, 128, 768, 1536} {
		corpus := makeCorpus(2, dim, 1)
		b.Run("dim="+itoa(dim), func(b *testing.B) {
			var sum float32
			for i := 0; i < b.N; i++ {
				sum = dotAVX512(corpus[0], corpus[1])
			}
			_ = sum
		})
	}
}

func BenchmarkL2SquaredAVX512(b *testing.B) {
	if !avx512Enabled {
		b.Skip("AVX-512 not available on this CPU")
	}
	for _, dim := range []int{64, 100, 128, 768, 1536} {
		corpus := makeCorpus(2, dim, 1)
		b.Run("dim="+itoa(dim), func(b *testing.B) {
			var sum float32
			for i := 0; i < b.N; i++ {
				sum = l2SquaredAVX512Slice(corpus[0], corpus[1])
			}
			_ = sum
		})
	}
}

func TestSQ8CodeDotAVX2MatchesScalar(t *testing.T) {
	if !avx2Enabled {
		t.Skip("AVX2 not available on this CPU")
	}
	rng := rand.New(rand.NewSource(4))
	for _, n := range []int{0, 1, 3, 7, 8, 15, 16, 17, 31, 64, 100, 127, 128, 129, 768, 1536} {
		a := make([]byte, n)
		b := make([]byte, n)
		for i := range a {
			a[i] = byte(int8(rng.Intn(255) - 127)) //nolint:gosec // arbitrary int8 codes
			b[i] = byte(int8(rng.Intn(255) - 127)) //nolint:gosec
		}
		got := sq8CodeDotAVX2(a, b)
		want := sq8CodeDotScalar(a, b)
		if got != want {
			t.Errorf("n=%d: sq8CodeDotAVX2=%d sq8CodeDotScalar=%d", n, got, want)
		}
	}
}

func TestSQ8CodeDotVNNIMatchesScalar(t *testing.T) {
	if !avx512VNNIEnabled {
		t.Skip("AVX-512-VNNI not available on this CPU")
	}
	rng := rand.New(rand.NewSource(7))
	for _, n := range []int{0, 1, 3, 7, 8, 15, 16, 17, 31, 63, 64, 65, 100, 127, 128, 129, 256, 768, 1536} {
		a := make([]byte, n)
		b := make([]byte, n)
		for i := range a {
			a[i] = byte(int8(rng.Intn(256) - 128)) //nolint:gosec // arbitrary int8 codes incl. -128
			b[i] = byte(int8(rng.Intn(256) - 128)) //nolint:gosec
		}
		got := sq8CodeDotVNNI(a, b)
		want := sq8CodeDotScalar(a, b)
		if got != want {
			t.Errorf("n=%d: sq8CodeDotVNNI=%d sq8CodeDotScalar=%d", n, got, want)
		}
	}
}

func BenchmarkSQ8CodeDotVNNI(b *testing.B) {
	if !avx512VNNIEnabled {
		b.Skip("AVX-512-VNNI not available on this CPU")
	}
	rng := rand.New(rand.NewSource(11))
	for _, dim := range []int{128, 768, 1536} {
		x := make([]byte, dim)
		y := make([]byte, dim)
		for i := range x {
			x[i] = byte(int8(rng.Intn(256) - 128)) //nolint:gosec
			y[i] = byte(int8(rng.Intn(256) - 128)) //nolint:gosec
		}
		b.Run("vnni/dim="+itoa(dim), func(b *testing.B) {
			var s int32
			for i := 0; i < b.N; i++ {
				s = sq8CodeDotVNNI(x, y)
			}
			_ = s
		})
		b.Run("avx2/dim="+itoa(dim), func(b *testing.B) {
			var s int32
			for i := 0; i < b.N; i++ {
				s = sq8CodeDotAVX2(x, y)
			}
			_ = s
		})
	}
}

func TestDetectAVX2DoesNotPanic(t *testing.T) {
	// Just exercise the CPUID/XGETBV path; the result is hardware-dependent.
	_ = detectAVX2()
	t.Logf("avx2Enabled = %v", avx2Enabled)
}

func BenchmarkDotAVX2(b *testing.B) {
	if !avx2Enabled {
		b.Skip("AVX2 not available on this CPU")
	}
	for _, dim := range []int{64, 128, 768, 1536} {
		corpus := makeCorpus(2, dim, 1)
		b.Run("dim="+itoa(dim), func(b *testing.B) {
			var sum float32
			for i := 0; i < b.N; i++ {
				sum = dotAVX2(corpus[0], corpus[1])
			}
			_ = sum
		})
	}
}

func BenchmarkL2SquaredAVX2(b *testing.B) {
	if !avx2Enabled {
		b.Skip("AVX2 not available on this CPU")
	}
	for _, dim := range []int{64, 128, 768, 1536} {
		corpus := makeCorpus(2, dim, 1)
		b.Run("dim="+itoa(dim), func(b *testing.B) {
			var sum float32
			for i := 0; i < b.N; i++ {
				sum = l2SquaredAVX2Slice(corpus[0], corpus[1])
			}
			_ = sum
		})
	}
}

func BenchmarkSQ8Dot(b *testing.B) {
	for _, dim := range []int{128, 768, 1536} {
		corpus := makeCorpus(1, dim, 7)
		query := append([]float32(nil), corpus[0]...)
		normalize(query)
		code := make([]byte, dim)
		newSQ8(dim).Encode(code, query)
		b.Run("scalar/dim="+itoa(dim), func(b *testing.B) {
			var s float32
			for i := 0; i < b.N; i++ {
				s = sq8DotScalar(query, code)
			}
			_ = s
		})
		b.Run("avx2/dim="+itoa(dim), func(b *testing.B) {
			if !avx2Enabled {
				b.Skip("AVX2 not available on this CPU")
			}
			var s float32
			for i := 0; i < b.N; i++ {
				s = sq8DotAVX2(query, code)
			}
			_ = s
		})
	}
}

// itoa avoids importing strconv just for the sub-benchmark label.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

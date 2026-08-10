// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"testing"
)

const epsilon = 1e-6

func almostEqual(a, b float32) bool {
	return math.Abs(float64(a-b)) < epsilon
}

func TestDotScalar(t *testing.T) {
	tests := []struct {
		a, b []float32
		want float32
	}{
		{[]float32{1, 0, 0}, []float32{1, 0, 0}, 1.0},
		{[]float32{1, 0, 0}, []float32{0, 1, 0}, 0.0},
		{[]float32{1, 2, 3}, []float32{4, 5, 6}, 32.0},
		{[]float32{}, []float32{}, 0.0},
	}
	for _, tc := range tests {
		got := dotScalar(tc.a, tc.b)
		if !almostEqual(got, tc.want) {
			t.Errorf("dotScalar(%v, %v) = %v, want %v", tc.a, tc.b, got, tc.want)
		}
	}
}

func TestL2SquaredScalar(t *testing.T) {
	tests := []struct {
		a, b []float32
		want float32
	}{
		{[]float32{0, 0, 0}, []float32{0, 0, 0}, 0.0},
		{[]float32{1, 0, 0}, []float32{0, 0, 0}, 1.0},
		{[]float32{3, 4}, []float32{0, 0}, 25.0},
		{[]float32{1, 2, 3}, []float32{4, 5, 6}, 27.0},
	}
	for _, tc := range tests {
		got := l2SquaredScalar(tc.a, tc.b)
		if !almostEqual(got, tc.want) {
			t.Errorf("l2SquaredScalar(%v, %v) = %v, want %v", tc.a, tc.b, got, tc.want)
		}
	}
}

func TestNormalize(t *testing.T) {
	v := []float32{3, 4}
	normalize(v)
	if !almostEqual(v[0], 0.6) || !almostEqual(v[1], 0.8) {
		t.Errorf("normalize({3,4}) = %v, want {0.6, 0.8}", v)
	}
	z := []float32{0, 0, 0}
	normalize(z)
	for _, x := range z {
		if x != 0 {
			t.Errorf("normalize zero vector should stay zero, got %v", z)
		}
	}
}

func TestPickDistCosine(t *testing.T) {
	f := pickDist(Cosine)
	a := []float32{1, 0, 0}
	b := []float32{1, 0, 0}
	if !almostEqual(f(a, b), 0.0) {
		t.Errorf("cosine distance same vector = %v, want 0", f(a, b))
	}
	c := []float32{0, 1, 0}
	if !almostEqual(f(a, c), 1.0) {
		t.Errorf("cosine distance orthogonal = %v, want 1", f(a, c))
	}
}

func TestPickDistL2(t *testing.T) {
	f := pickDist(L2)
	a := []float32{0, 0}
	b := []float32{3, 4}
	if !almostEqual(f(a, b), 25.0) {
		t.Errorf("L2 squared = %v, want 25", f(a, b))
	}
}

func TestPickDistDotProduct(t *testing.T) {
	f := pickDist(DotProduct)
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	if !almostEqual(f(a, b), -32.0) {
		t.Errorf("DotProduct distance = %v, want -32", f(a, b))
	}
}

func TestPickDistUnknownPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for unknown metric")
		}
	}()
	pickDist(Metric(99))
}

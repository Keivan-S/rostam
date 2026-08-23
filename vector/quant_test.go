// SPDX-License-Identifier: Apache-2.0

package vector

import "testing"

// TestSQ8Encode checks that the scalar int8 quantizer maps normalized
// components through a symmetric 1/127 scale, rounds to the nearest int8, and
// clamps values outside [-1, 1]. CodeLen is one byte per dimension.
func TestSQ8Encode(t *testing.T) {
	q := newSQ8(3)
	if got := q.CodeLen(); got != 3 {
		t.Fatalf("CodeLen() = %d, want 3", got)
	}
	tests := []struct {
		vec  []float32
		want []int8
	}{
		{[]float32{1, 0, -1}, []int8{127, 0, -127}},
		{[]float32{0.5, -0.5, 0}, []int8{64, -64, 0}}, // round(63.5) -> 64
		{[]float32{2, -2, 0}, []int8{127, -127, 0}},   // clamp out-of-range
	}
	for _, tc := range tests {
		code := make([]byte, q.CodeLen())
		q.Encode(code, tc.vec)
		for i, w := range tc.want {
			if int8(code[i]) != w {
				t.Errorf("Encode(%v)[%d] = %d, want %d", tc.vec, i, int8(code[i]), w)
			}
		}
	}
}

// TestBQ1Encode checks that binary quantization packs one sign bit per
// dimension (bit set when the component is positive), LSB-first within each
// byte, and that CodeLen is ceil(dim/8).
func TestBQ1Encode(t *testing.T) {
	if got := newBQ1(8).CodeLen(); got != 1 {
		t.Errorf("CodeLen(8) = %d, want 1", got)
	}
	if got := newBQ1(10).CodeLen(); got != 2 {
		t.Errorf("CodeLen(10) = %d, want 2", got)
	}
	tests := []struct {
		dim  int
		vec  []float32
		want []byte
	}{
		{8, []float32{1, 1, 1, 1, 1, 1, 1, 1}, []byte{0xFF}},
		{8, []float32{-1, -1, -1, -1, -1, -1, -1, -1}, []byte{0x00}},
		{8, []float32{1, -1, 1, -1, 1, -1, 1, -1}, []byte{0x55}}, // bits 0,2,4,6
		{10, []float32{1, 0, 0, 0, 0, 0, 0, 0, 1, 1}, []byte{0x01, 0x03}},
	}
	for _, tc := range tests {
		q := newBQ1(tc.dim)
		code := make([]byte, q.CodeLen())
		q.Encode(code, tc.vec)
		for i, w := range tc.want {
			if code[i] != w {
				t.Errorf("Encode(%v) byte[%d] = %#x, want %#x", tc.vec, i, code[i], w)
			}
		}
	}
}

// TestBQ1DistanceHamming checks that Distance returns the Hamming distance —
// the count of differing sign bits between query and code.
func TestBQ1DistanceHamming(t *testing.T) {
	q := newBQ1(8)
	query := []float32{1, 1, 1, 1, 1, 1, 1, 1}
	qc := q.PrepareQuery(query)
	tests := []struct {
		vec  []float32
		want float32
	}{
		{[]float32{1, 1, 1, 1, 1, 1, 1, 1}, 0},         // identical signs
		{[]float32{-1, -1, -1, -1, -1, -1, -1, -1}, 8}, // all opposite
		{[]float32{1, -1, 1, -1, 1, -1, 1, -1}, 4},     // half differ
	}
	for _, tc := range tests {
		code := make([]byte, q.CodeLen())
		q.Encode(code, tc.vec)
		if got := q.Distance(qc, code); got != tc.want {
			t.Errorf("Distance(query, %v) = %v, want %v", tc.vec, got, tc.want)
		}
	}
}

// TestConfigValidateQuant checks the quantization-related validation rules:
// QuantMmap requires both a quantizer and a backing path, and unknown enum
// values are rejected.
func TestConfigValidateQuant(t *testing.T) {
	base := Config{Dim: 8, M: 16, EfConstruction: 200, EfSearch: 64}
	withFn := func(mut func(*Config)) Config {
		c := base
		mut(&c)
		return c
	}
	tests := []struct {
		name    string
		cfg     Config
		wantErr bool
	}{
		{"mmap without quantizer", withFn(func(c *Config) { c.QuantStorage = QuantMmap; c.MmapPath = "/tmp/x" }), true},
		{"mmap without path", withFn(func(c *Config) { c.Quant = QuantSQ8; c.QuantStorage = QuantMmap }), true},
		{"mmap with quantizer and path", withFn(func(c *Config) { c.Quant = QuantSQ8; c.QuantStorage = QuantMmap; c.MmapPath = "/tmp/x" }), false},
		{"sq8 in-ram", withFn(func(c *Config) { c.Quant = QuantSQ8 }), false},
		{"unknown quant mode", withFn(func(c *Config) { c.Quant = QuantMode(99) }), true},
		{"unknown storage mode", withFn(func(c *Config) { c.Quant = QuantSQ8; c.QuantStorage = QuantStore(99); c.MmapPath = "/tmp/x" }), true},
	}
	for _, tc := range tests {
		err := ValidateConfig(tc.cfg)
		if (err != nil) != tc.wantErr {
			t.Errorf("%s: Validate() err = %v, wantErr = %v", tc.name, err, tc.wantErr)
		}
	}
}

// TestSQ8DistanceApproximatesCosine checks that asymmetric distance — float32
// query against an int8 code — tracks the exact cosine distance (1 - dot over
// normalized vectors) within quantization tolerance.
func TestSQ8DistanceApproximatesCosine(t *testing.T) {
	const dim = 8
	const tol = 0.02
	exact := pickDist(Cosine)

	pairs := [][2][]float32{
		{{1, 1, 0, 0, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0, 0}},  // identical -> ~0
		{{1, 1, 0, 0, 0, 0, 0, 0}, {1, -1, 0, 0, 0, 0, 0, 0}}, // orthogonal -> ~1
		{{1, 2, 3, 4, 5, 6, 7, 8}, {8, 7, 6, 5, 4, 3, 2, 1}},  // arbitrary
		{{3, 1, 4, 1, 5, 9, 2, 6}, {2, 7, 1, 8, 2, 8, 1, 8}},  // arbitrary
	}

	q := newSQ8(dim)
	for _, p := range pairs {
		query := append([]float32(nil), p[0]...)
		vec := append([]float32(nil), p[1]...)
		normalize(query)
		normalize(vec)

		code := make([]byte, q.CodeLen())
		q.Encode(code, vec)

		got := q.Distance(q.PrepareQuery(query), code)
		want := exact(query, vec)
		if d := got - want; d > tol || d < -tol {
			t.Errorf("Distance approx = %v, exact = %v (diff %v > tol %v)", got, want, d, tol)
		}
	}
}

// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// Differential tests for the batched "one query vs N slots" kernels.
//
// TWO ORACLES, AND WHY THE TOLERANCES DIFFER.
//
// The primary oracle is the PER-PAIR kernel — the code the batched path
// replaces, on the same machine with the same CPU features. Against it the
// assertion is EXACT BIT EQUALITY, not a tolerance. That is achievable (and
// therefore demanded) because the batched kernels reproduce the per-pair
// accumulator layout and phase order deliberately; see distance_ny.go. It is
// also the assertion that actually matters for this change: it proves the search
// path's numbers did not move, so any result-set difference in the rest of the
// suite is attributable to expandBatched's admission ordering alone and not to
// arithmetic. A tolerance here would have let a genuine summation regression
// hide inside "well, floats".
//
// The secondary oracle is an independent float64 accumulation (nyOracle),
// written from the definition rather than derived from either kernel. It catches
// the failure mode exact-equality-to-pair cannot: both kernels computing the
// same WRONG thing (a mis-indexed slab, a dropped tail element, a wrong metric
// transform). Against float64 the comparison must be a tolerance, and the
// tolerance is stated relative to Σ|term| rather than to the result:
//
//	|got - oracle| <= relTol * Σ|term|
//
// Σ|term| is the condition number of the summation. Using it is what makes the
// bound meaningful for the DOT kernel on adversarial input, where the terms
// cancel and the result can be arbitrarily close to zero — there, "relative
// error of the result" is unbounded for any correct float32 implementation, so
// asserting on it would produce a test that fails on correct code. For the L2
// kernel every term is non-negative, so Σ|term| IS the result and the bound
// degenerates to exactly the relative error < 1e-5 the task asks for.
//
// relTol is 1e-5. float32 has ~1.19e-7 unit roundoff, and the kernels sum up to
// 1024 terms through four accumulators, so the realistic error is ~sqrt(n)*eps ≈
// 4e-6 at the largest dimension tested and far below that elsewhere. 1e-5 sits
// just above the worst realistic case and orders of magnitude below any genuine
// bug (a dropped tail element or a transposed index moves the result by a
// fraction of Σ|term|, not by a few ulp).
//
// The bound also carries an ABSOLUTE floor of dim * SmallestNonzeroFloat32.
// That term is not slack — it is float32's underflow granularity, and without it
// the denormal regime fails on correct code: with inputs at 1e-40 the individual
// products land near 1e-80, roughly thirty-five orders of magnitude below the
// smallest subnormal float32 (1.4e-45), so every one of them flushes to zero and
// the kernel correctly returns exactly 0 while the float64 oracle reports 1e-78.
// The error there IS the whole value, so no relative bound can accommodate it;
// the absolute float32 underflow floor is the only correct characterization. At
// 1e-42 for the widest vector it is negligible against every non-underflowing
// case, so it does not weaken the check anywhere it matters.
const nyRelTol = 1e-5

// nyAbsFloor is the absolute underflow allowance for a dim-term float32 sum.
func nyAbsFloor(dim int) float64 { return float64(dim) * math.SmallestNonzeroFloat32 }

// nyDims is the dimension sweep. It covers every AVX2 loop-shape boundary
// (1..16 exercises the pure scalar tail and the first 8-wide iteration; 63/64/65
// and 127/128 straddle the 32-float main-loop stride) and both AVX-512
// boundaries (768 and 1024, the latter also being avx512MinDim, where the
// per-pair path itself switches width).
var nyDims = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 63, 64, 65, 127, 128, 768, 1024}

// nyCounts is the candidate-count sweep. Odd values exercise nyDispatch's
// per-pair tail (the assembly handles only even blocks), and 0/1 the degenerate
// entries.
var nyCounts = []int{0, 1, 2, 3, 5, 8, 17}

// nyExtraDims is appended to nyDims by the width-agreement test. It is empty
// here and populated per-architecture, because the dimension where the dispatch
// switches kernel width is an amd64 concept (avx512MinDim) that must not leak
// into a file every architecture compiles.
var nyExtraDims []int

// nyOracle computes one distance in float64 straight from the definition,
// returning the value and Σ|term| (the summation's condition number). It shares
// no code with either kernel.
func nyOracle(q, v []float32, l2 bool) (val, scale float64) {
	for i := range q {
		var t float64
		if l2 {
			d := float64(q[i]) - float64(v[i])
			t = d * d
		} else {
			t = float64(q[i]) * float64(v[i])
		}
		val += t
		scale += math.Abs(t)
	}
	return val, scale
}

// nyGen builds one adversarial or random value in [-1,1)-ish scaled to the named
// regime. The regimes are chosen to stress the parts of a float32 kernel that
// differ from a float64 oracle: subnormals (where flush-to-zero behaviour would
// show), magnitudes near the top of float32's range (where an accumulator
// ordering change can overflow), and sign-alternating input (where the dot
// product cancels catastrophically).
func nyGen(rng *rand.Rand, regime string, i int) float32 {
	u := float32(rng.Float64()*2 - 1)
	switch regime {
	case "random":
		return u
	case "denormal":
		// ~1e-40: below float32's smallest normal (1.18e-38), so the INPUTS are
		// genuine subnormals. Their products underflow to zero entirely, which
		// is itself worth pinning — the kernel must return 0 rather than a NaN
		// or a trap, and must agree with the per-pair kernel about doing so.
		return u * 1e-40
	case "subnormal-product":
		// ~1e-20, chosen so the PRODUCTS (~1e-40) land inside the representable
		// subnormal range rather than vanishing: this is the regime that
		// actually exercises subnormal arithmetic in the FMA path, including any
		// flush-to-zero difference between the wide loop and the scalar tail.
		return u * 1e-20
	case "large":
		// Squares reach ~1e30 and sums over 1024 dims ~1e33, comfortably inside
		// float32's 3.4e38 ceiling but far enough up to catch a bogus rescale.
		return u * 1e15
	case "alternating":
		// Near-total cancellation for the dot kernel.
		if i%2 == 0 {
			return u * 1e6
		}
		return -u * 1e6
	case "mixed":
		// Wildly different exponents in one vector: the classic case where
		// summation order changes the answer.
		switch i % 4 {
		case 0:
			return u * 1e12
		case 1:
			return u * 1e-12
		case 2:
			return u
		default:
			return -u * 1e8
		}
	}
	panic("unknown regime")
}

var nyRegimes = []string{"random", "denormal", "subnormal-product", "large", "alternating", "mixed"}

// nyFixture builds a query, a flat slab of nSlots vectors, and a slot list that
// is deliberately NOT in slab order (so a kernel that ignored the slot list and
// walked the slab sequentially would fail).
func nyFixture(rng *rand.Rand, regime string, dim, nSlots int) (q, base []float32, slots []uint32) {
	q = make([]float32, dim)
	for i := range q {
		q[i] = nyGen(rng, regime, i)
	}
	// The slab holds more records than the query asks for, so out-of-order and
	// repeated slot references are both meaningful.
	nRecords := nSlots + 3
	base = make([]float32, nRecords*dim)
	for i := range base {
		base[i] = nyGen(rng, regime, i)
	}
	slots = make([]uint32, nSlots)
	for i := range slots {
		slots[i] = uint32((i*7 + 3) % nRecords)
	}
	return q, base, slots
}

// checkNy runs one batched kernel against both oracles for one configuration.
func checkNy(t *testing.T, name string, ny nyFunc, pair distFunc, l2 bool, regime string, dim, nSlots int, seed int64) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	q, base, slots := nyFixture(rng, regime, dim, nSlots)

	out := make([]float32, nSlots)
	// Poison the output so a kernel that skips an element is caught rather than
	// silently agreeing with a zero-initialized buffer.
	for i := range out {
		out[i] = float32(math.NaN())
	}
	ny(q, base, dim, slots, out)

	for i, s := range slots {
		v := base[int(s)*dim : int(s)*dim+dim]

		// Primary: exact equality with the per-pair kernel this replaces.
		want := pair(q, v)
		if math.Float32bits(out[i]) != math.Float32bits(want) {
			t.Errorf("%s/%s dim=%d n=%d [%d]: batched=%v (bits %#x) pair=%v (bits %#x) — "+
				"the batched kernels must be BIT-IDENTICAL to the per-pair kernel; a "+
				"difference here means the accumulator layout or phase order drifted",
				name, regime, dim, nSlots, i, out[i], math.Float32bits(out[i]), want, math.Float32bits(want))
			continue
		}

		// Secondary: independent float64 definition, bounded by the summation's
		// condition number (see the file comment).
		oracle, scale := nyOracle(q, v, l2)
		if math.IsInf(oracle, 0) || math.IsNaN(oracle) || math.IsInf(float64(out[i]), 0) {
			continue // out of float32's range; nothing meaningful to compare
		}
		tol := nyRelTol*scale + nyAbsFloor(dim)
		if diff := math.Abs(float64(out[i]) - oracle); diff > tol {
			t.Errorf("%s/%s dim=%d n=%d [%d]: batched=%v oracle=%v diff=%g > %g "+
				"(relTol*Σ|term| + float32 underflow floor, Σ|term|=%g)",
				name, regime, dim, nSlots, i, out[i], oracle, diff, tol, scale)
		}
	}
}

// TestNyKernelsDifferential sweeps the ACTIVE batched kernels (whatever this
// CPU selected) across every dimension, candidate count and value regime.
func TestNyKernelsDifferential(t *testing.T) {
	var seed int64
	for _, dim := range nyDims {
		for _, n := range nyCounts {
			for _, regime := range nyRegimes {
				seed++
				checkNy(t, "l2SquaredNy", l2SquaredNy, l2Squared, true, regime, dim, n, seed)
				checkNy(t, "dotNy", dotNy, dotProduct, false, regime, dim, n, seed)
			}
		}
	}
}

// TestNyPortableDifferential pins the portable drivers directly. On amd64 the
// active kernels are assembly, so without this the fallback used by other
// architectures — and by nyDispatch when a slot is out of range — would never be
// exercised here.
func TestNyPortableDifferential(t *testing.T) {
	var seed int64
	for _, dim := range nyDims {
		for _, n := range nyCounts {
			for _, regime := range nyRegimes {
				seed++
				checkNy(t, "l2SquaredNyPortable", l2SquaredNyPortable, l2Squared, true, regime, dim, n, seed)
				checkNy(t, "dotNyPortable", dotNyPortable, dotProduct, false, regime, dim, n, seed)
			}
		}
	}
}

// TestNyRepeatedAndReversedSlots checks the slot list is honoured element by
// element: repeats must produce identical results and a reversed list must
// produce reversed results. A kernel that walked the slab sequentially, or that
// advanced its slot cursor by the wrong stride, passes the ordered test and
// fails this one.
func TestNyRepeatedAndReversedSlots(t *testing.T) {
	for _, dim := range []int{1, 7, 8, 33, 128, 1024} {
		rng := rand.New(rand.NewSource(int64(dim)))
		q := make([]float32, dim)
		for i := range q {
			q[i] = float32(rng.NormFloat64())
		}
		const nRecords = 9
		base := make([]float32, nRecords*dim)
		for i := range base {
			base[i] = float32(rng.NormFloat64())
		}

		fwd := []uint32{0, 3, 3, 8, 1, 8, 0}
		rev := make([]uint32, len(fwd))
		for i := range fwd {
			rev[i] = fwd[len(fwd)-1-i]
		}
		a := make([]float32, len(fwd))
		b := make([]float32, len(rev))
		l2SquaredNy(q, base, dim, fwd, a)
		l2SquaredNy(q, base, dim, rev, b)
		for i := range fwd {
			if a[i] != b[len(fwd)-1-i] {
				t.Errorf("dim=%d: reversing the slot list changed result %d: %v vs %v",
					dim, i, a[i], b[len(fwd)-1-i])
			}
		}
		// slots[1] and slots[2] are both slot 3; slots[0] and slots[6] both 0.
		if a[1] != a[2] || a[0] != a[6] {
			t.Errorf("dim=%d: repeated slots gave different results: %v %v / %v %v",
				dim, a[1], a[2], a[0], a[6])
		}
	}
}

// TestNyOutOfRangeSlotFallsBack pins nyDispatch's guard. The assembly kernels
// index the slab with no bounds check of their own, so a slot past the end must
// be routed to the portable driver, whose slicing panics at the offending slot
// exactly as arena.Vec did before batching. Silently scoring out-of-slab memory
// would be the worst outcome.
func TestNyOutOfRangeSlotFallsBack(t *testing.T) {
	const dim, nRecords = 16, 4
	q := make([]float32, dim)
	base := make([]float32, nRecords*dim)
	out := make([]float32, 2)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected a panic for an out-of-slab slot, got none — the batched " +
				"path must not read past the arena")
		}
	}()
	l2SquaredNy(q, base, dim, []uint32{0, nRecords}, out)
}

// TestPickNyMatchesPickDistWidth asserts the batched dispatcher makes the SAME
// width choice as the per-pair dispatcher at every dimension. That agreement is
// what keeps the two bit-identical: if pickDistDim upgraded to AVX-512 at some
// dimension while pickNy did not (or vice versa), the two paths would sum in
// different orders and the exact-equality assertion above would be the only
// thing standing between that and a silent recall change.
func TestPickNyMatchesPickDistWidth(t *testing.T) {
	dims := append(append([]int(nil), nyDims...), nyExtraDims...)
	for _, dim := range dims {
		rng := rand.New(rand.NewSource(int64(dim)))
		q, base, slots := nyFixture(rng, "random", dim, 6)

		// L2 and DotProduct are compared exactly: pickNy returns the RAW kernel
		// (the metric transform lives in batchExact), and DotProduct's transform
		// is a negation, which float32 performs exactly — so it can be undone
		// without losing bits. Cosine's 1-d cannot (1-(1-d) != d for small d),
		// which is why Cosine is checked by kernel identity instead, below.
		for _, m := range []Metric{L2, DotProduct} {
			ny := pickNy(m, dim)
			if ny == nil {
				t.Errorf("pickNy(%v, %d) = nil, want a kernel for every supported metric", m, dim)
				continue
			}
			out := make([]float32, len(slots))
			ny(q, base, dim, slots, out)
			pair := pickDistDim(m, dim)
			for i, s := range slots {
				want := pair(q, base[int(s)*dim:int(s)*dim+dim])
				if m == DotProduct {
					want = -want
				}
				if math.Float32bits(out[i]) != math.Float32bits(want) {
					t.Errorf("metric=%v dim=%d [%d]: batched raw=%v pair raw=%v — "+
						"pickNy and pickDistDim disagree on kernel width", m, dim, i, out[i], want)
					break
				}
			}
		}

		// Cosine and DotProduct share the dot kernel, so they must select the
		// same width for the same dimension.
		cos, dp := pickNy(Cosine, dim), pickNy(DotProduct, dim)
		if cos == nil || dp == nil {
			t.Errorf("dim=%d: pickNy returned nil for Cosine or DotProduct", dim)
			continue
		}
		a, b := make([]float32, len(slots)), make([]float32, len(slots))
		cos(q, base, dim, slots, a)
		dp(q, base, dim, slots, b)
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				t.Errorf("dim=%d [%d]: Cosine and DotProduct raw kernels differ (%v vs %v); "+
					"both must be the dot kernel at the same width", dim, i, a[i], b[i])
				break
			}
		}
	}
}

// TestBatchExactMatchesPerSlotScorer is the end-to-end pin one level up: for
// every metric, the batched scorer hnsw.batchExact builds must agree EXACTLY
// with the per-slot scorer exactScorer builds over the same live index. This is
// what covers the metric transforms (1-d for Cosine, -d for DotProduct) and the
// arena wiring, neither of which the raw-kernel tests above touch.
func TestBatchExactMatchesPerSlotScorer(t *testing.T) {
	for _, m := range []Metric{L2, Cosine, DotProduct} {
		for _, dim := range []int{3, 8, 64, 128} {
			name := fmt.Sprintf("%v/dim=%d", m, dim)
			t.Run(name, func(t *testing.T) {
				const n = 200
				h, err := newHNSW(Config{Dim: dim, Metric: m, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 3})
				if err != nil {
					t.Fatal(err)
				}
				defer h.Close()
				rng := rand.New(rand.NewSource(int64(dim)))
				var q []float32
				for i := 0; i < n; i++ {
					v := make([]float32, dim)
					for j := range v {
						v[j] = float32(rng.NormFloat64())
					}
					if q == nil {
						q = append([]float32(nil), v...)
					}
					if _, _, err := h.Insert(uint64(i+1), v, 0, nil, nil, nil, CASCond{}); err != nil {
						t.Fatal(err)
					}
				}
				// Cosine normalizes on insert, so score the normalized form the
				// search path would actually use.
				if m == Cosine {
					normalize(q)
				}

				sc := h.exactScorer(q)
				if !sc.batch.ok() {
					t.Fatalf("exactScorer for %v dim=%d has no batched kernel; the "+
						"unquantized float path is exactly where it must be present", m, dim)
				}
				slots := make([]uint32, 0, n)
				for s := 0; s < n; s++ {
					slots = append(slots, uint32(s))
				}
				out := make([]float32, len(slots))
				sc.batch.score(slots, out)
				for i, s := range slots {
					want := sc.score(s)
					if math.Float32bits(out[i]) != math.Float32bits(want) {
						t.Fatalf("slot %d: batch=%v per-slot=%v — batchExact and "+
							"exactScorer must agree bit for bit", s, out[i], want)
					}
				}
			})
		}
	}
}

// TestBatchExactDeclinesWhenUnusable pins the nil-returning guards. Each of
// these is a case where the batched kernel's (base, dim) contract cannot be
// honoured, and returning nil — leaving the caller on the per-pair path — is the
// only safe answer.
func TestBatchExactDeclinesWhenUnusable(t *testing.T) {
	const dim = 16
	h, err := newHNSW(Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	v := make([]float32, dim)
	if _, _, err := h.Insert(1, v, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatal(err)
	}

	if h.batchExact(make([]float32, dim+1)).ok() {
		t.Error("batchExact accepted a query whose length differs from the arena stride")
	}
	if h.batchExact(make([]float32, dim-1)).ok() {
		t.Error("batchExact accepted a short query")
	}

	saved := h.arena.vecsDropped
	h.arena.vecsDropped = true
	if h.batchExact(v).ok() {
		t.Error("batchExact returned a kernel for an arena whose float slab was dropped")
	}
	h.arena.vecsDropped = saved
}

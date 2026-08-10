// SPDX-License-Identifier: Apache-2.0
//go:build amd64

package vector

import (
	"math/rand"
	"testing"
)

// The active-kernel sweep in distance_ny_test.go only reaches the AVX-512
// batched kernels at dim >= avx512MinDim, because that is where the dispatcher
// selects them. These tests drive both assembly widths DIRECTLY across the whole
// dimension sweep, so every loop shape in each .s file — the wide main loop, the
// narrower follow-on loop, the scalar tail, and the paths where one or two of
// those are skipped entirely — is exercised at both widths rather than only at
// the dimensions the dispatcher happens to route there.

// The width-agreement test must straddle avx512MinDim, the dimension where the
// per-pair dispatch switches from YMM to ZMM — the one place pickNy and
// pickDistDim could disagree and silently give up bit-identity.
func init() { nyExtraDims = []int{avx512MinDim - 1, avx512MinDim, avx512MinDim + 1} }

func TestNyAVX2KernelsDifferential(t *testing.T) {
	if !avx2Enabled {
		t.Skip("no AVX2 on this CPU")
	}
	l2 := nyDispatch(l2SquaredNyAVX2, l2SquaredAVX2, l2SquaredNyPortable)
	dot := nyDispatch(dotNyAVX2, dotProductAVX2, dotNyPortable)
	var seed int64
	for _, dim := range nyDims {
		for _, n := range nyCounts {
			for _, regime := range nyRegimes {
				seed++
				// The AVX2 batched kernel must match the AVX2 per-pair kernel,
				// regardless of which width the dispatcher would have chosen for
				// this dimension.
				checkNy(t, "l2SquaredNyAVX2", l2, l2SquaredAVX2Slice, true, regime, dim, n, seed)
				checkNy(t, "dotNyAVX2", dot, dotAVX2, false, regime, dim, n, seed)
			}
		}
	}
}

func TestNyAVX512KernelsDifferential(t *testing.T) {
	if !avx512Enabled {
		t.Skip("no AVX-512 on this CPU")
	}
	l2 := nyDispatch(l2SquaredNyAVX512, l2SquaredAVX512, l2SquaredNyPortable)
	dot := nyDispatch(dotNyAVX512, dotProductAVX512, dotNyPortable)
	var seed int64
	for _, dim := range nyDims {
		for _, n := range nyCounts {
			for _, regime := range nyRegimes {
				seed++
				checkNy(t, "l2SquaredNyAVX512", l2, l2SquaredAVX512Slice, true, regime, dim, n, seed)
				checkNy(t, "dotNyAVX512", dot, dotAVX512, false, regime, dim, n, seed)
			}
		}
	}
}

// TestNyAssemblyLeavesNeighboringOutputUntouched pins the store side of the
// block loop. The kernels write two results per iteration through a cursor they
// advance themselves; an off-by-one there would corrupt the element after the
// block instead of failing any value comparison.
func TestNyAssemblyLeavesNeighboringOutputUntouched(t *testing.T) {
	if !avx2Enabled {
		t.Skip("no AVX2 on this CPU")
	}
	const dim = 40
	rng := rand.New(rand.NewSource(11))
	q, base, slots := nyFixture(rng, "random", dim, 5)

	const guard = 3
	out := make([]float32, len(slots)+guard)
	for i := range out {
		out[i] = -12345
	}
	l2SquaredNyAVX2(&q[0], &base[0], dim, &slots[0], len(slots)&^1, &out[0])

	// The kernel was asked for an even prefix only; everything from there on
	// must be untouched.
	for i := len(slots) &^ 1; i < len(out); i++ {
		if out[i] != -12345 {
			t.Errorf("out[%d] = %v, want the untouched guard value: the block loop "+
				"wrote past the %d results it was asked for", i, out[i], len(slots)&^1)
		}
	}
}

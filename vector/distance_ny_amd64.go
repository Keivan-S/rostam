// SPDX-License-Identifier: Apache-2.0
//go:build amd64

package vector

// l2SquaredNyAVX2 computes the squared L2 distance from q to each of the first
// n arena slots, writing n float32 results to out. Implemented in
// distance_ny_amd64.s.
//
// CONTRACT, all of which the Go wrapper below establishes:
//   - n must be EVEN. The kernel interleaves two candidates per block and simply
//     returns when fewer than two remain; it does not handle an odd tail.
//   - every slots[i] must satisfy slots[i]*dim+dim <= len(base).
//   - dim >= 0, len(q) == dim, and out must have room for n results.
func l2SquaredNyAVX2(q, base *float32, dim int, slots *uint32, n int, out *float32)

// dotNyAVX2 is the dot-product counterpart of l2SquaredNyAVX2. Same contract.
func dotNyAVX2(q, base *float32, dim int, slots *uint32, n int, out *float32)

// l2SquaredNyAVX512 and dotNyAVX512 are the 16-lane (ZMM) counterparts, used at
// dim >= avx512MinDim on CPUs where the per-pair path would also have chosen
// AVX-512. Same contract. Implemented in distance_ny_avx512_amd64.s.
func l2SquaredNyAVX512(q, base *float32, dim int, slots *uint32, n int, out *float32)
func dotNyAVX512(q, base *float32, dim int, slots *uint32, n int, out *float32)

// nyDispatch adapts a pointer-based batched kernel to the nyFunc signature: it
// validates the assembly's preconditions, runs the even prefix through the
// kernel, and finishes an odd trailing candidate on the matching per-pair
// kernel.
//
// Routing the odd candidate to pair rather than teaching the assembly a
// one-candidate path keeps a whole second accumulator layout out of the .s file
// — and since pair IS the kernel the block path is bit-identical to, the tail
// cannot introduce a discrepancy of its own.
//
// If any slot is out of range the whole block falls back to the portable driver,
// whose slab slicing panics at the offending slot exactly as arena.Vec did
// before batching.
func nyDispatch(
	kernel func(q, base *float32, dim int, slots *uint32, n int, out *float32),
	pair func(a, b *float32, n int) float32,
	portable nyFunc,
) nyFunc {
	return func(q, base []float32, dim int, slots []uint32, out []float32) {
		n := len(slots)
		if n == 0 {
			return
		}
		if dim <= 0 {
			// A zero-dimension slab has no records to address; every distance is
			// the empty sum. Guarding here keeps the assembly free of the case.
			for i := range slots {
				out[i] = 0
			}
			return
		}
		if !nyBoundsOK(slots, base, dim) {
			portable(q, base, dim, slots, out)
			return
		}
		if n2 := n &^ 1; n2 > 0 {
			kernel(&q[0], &base[0], dim, &slots[0], n2, &out[0])
		}
		if n&1 == 1 {
			off := int(slots[n-1]) * dim
			out[n-1] = pair(&q[0], &base[off], dim)
		}
	}
}

func init() {
	if avx2Enabled {
		l2SquaredNy = nyDispatch(l2SquaredNyAVX2, l2SquaredAVX2, l2SquaredNyPortable)
		dotNy = nyDispatch(dotNyAVX2, dotProductAVX2, dotNyPortable)
	}
	// The batched path follows the per-pair path's width choice exactly (same
	// avx512MinDim threshold, same metric mapping), because bit-identity between
	// the two is what lets the search-path change be validated as a pure
	// admission-order effect rather than a numeric one.
	if avx512Enabled {
		l2Ny := nyDispatch(l2SquaredNyAVX512, l2SquaredAVX512, l2SquaredNyPortable)
		dNy := nyDispatch(dotNyAVX512, dotProductAVX512, dotNyPortable)
		avx512NyDim = func(m Metric, dim int) nyFunc {
			if dim < avx512MinDim {
				return nil
			}
			switch m {
			case L2:
				return l2Ny
			case Cosine, DotProduct:
				return dNy
			}
			return nil
		}
	}
}

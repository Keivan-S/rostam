// SPDX-License-Identifier: Apache-2.0

package vector

import "unsafe"

// Batched "one query vs N arena slots" distance kernels — the fvec_L2sqr_ny
// shape borrowed from faiss.
//
// WHY THIS EXISTS. The per-pair path scores a neighbor like this: the graph loop
// calls a scorer closure, the closure calls arena.Vec (a bounds check plus a
// three-word slice header materialized on the stack), the slice wrapper
// dereferences &v[0] and re-derives len, and only then does the assembly kernel
// run. Profiling put that plumbing at ~18% ON TOP OF the kernel across the ~5000
// neighbor scorings a single query performs, and the kernel additionally re-loads
// the query vector from memory on every one of those calls.
//
// Batching amortizes all of it: one call, one bounds check, and — because the
// kernels below interleave two candidates against the SAME query registers — one
// query load per two candidate loads instead of one per one.
//
// BIT-IDENTITY IS A DESIGN CONSTRAINT, NOT AN ACCIDENT. Every kernel here
// reproduces the accumulator layout and phase order of its per-pair counterpart
// EXACTLY: same number of accumulators, same chunk-to-accumulator assignment,
// same combine tree, same 8-wide (or 16-wide) follow-on loop, same scalar tail.
// Floating-point addition is not associative, so any other grouping would give
// answers that differ in the last bits — and then every result-set difference in
// the test suite would be ambiguous between "a tie broke the other way" and "the
// kernel is wrong". Holding the summation order fixed removes that ambiguity
// entirely: the batched search path produces the same distances AND the same
// admissions as the per-pair path, bit for bit, so any result-set difference is
// a bug rather than a judgement call. It is also what lets the differential
// tests assert EXACT equality against the per-pair kernel rather than a
// tolerance. (expandBatched explains why hoisting the distances out of the loop
// does NOT staleness the admission gate, which is the delta one would otherwise
// expect from batching.)
//
// That constraint is also why the interleave factor is two rather than four:
// four candidates times four accumulators would need every YMM register with
// none left for the query, forcing a different (fewer-accumulator) grouping and
// giving up bit-identity for roughly another 12% off the load count.

// nyFunc computes the distance from a fixed query to each of N arena slots in a
// single call, writing len(slots) results into out.
//
// base is the arena's flat vector slab, so slot s occupies
// base[s*dim : (s+1)*dim]. Callers MUST ensure len(q) == dim, len(out) >=
// len(slots), and that every slot is in range — the assembly implementations
// index the slab with no bounds check of their own (see nyBoundsOK).
// Every implementation also OWNS ITS PREFETCHING, warming candidates
// nyPrefetchAhead positions in front of the one it is scoring. Making that part
// of the contract is what lets expandBatched hand over a whole neighbor list in
// one call: prefetch depth stops being tied to call granularity, so the list is
// amortized over a single call while the warming window stays bounded and
// overlaps the scoring of earlier candidates. A caller-side burst before the
// call cannot do that — it issues far more misses than the core can track, with
// no work to hide them behind, which measured 3-9% SLOWER than the per-pair path
// on SIFT-1M.
type nyFunc func(q, base []float32, dim int, slots []uint32, out []float32)

// nyPrefetchAhead is how many candidates ahead a batched kernel warms. It
// matches prefetchDistance, the depth the per-pair path was already tuned to.
const nyPrefetchAhead = 4

// l2SquaredNy and dotNy are the active batched kernels, mirroring the l2Squared
// and dotProduct per-pair vars. They default to the portable drivers and are
// replaced by SIMD implementations in the amd64 init().
var (
	l2SquaredNy nyFunc = l2SquaredNyPortable
	dotNy       nyFunc = dotNyPortable
)

// l2SquaredNyPortable is the portable batched L2 driver. It still delegates each
// candidate to the ACTIVE per-pair kernel (l2Squared, which arm64 and non-AVX2
// amd64 have already pointed at their best available implementation), so it is
// bit-identical to the per-pair path by construction on every platform. What it
// removes is the scorer closure and the arena.Vec slice-header construction; the
// call itself remains.
//
// It is also the oracle-adjacent reference the assembly kernels are tested
// against — see distance_ny_test.go, which additionally compares both against an
// independent naive implementation.
func l2SquaredNyPortable(q, base []float32, dim int, slots []uint32, out []float32) {
	lines := (dim*4 + 63) / 64
	for i, s := range slots {
		nyWarm(base, dim, lines, slots, i+nyPrefetchAhead)
		off := int(s) * dim
		out[i] = l2Squared(q, base[off:off+dim:off+dim])
	}
}

// nyWarm issues a whole-record prefetch for slots[j], if that index exists and
// addresses a complete record. The in-range test is not redundant with
// nyBoundsOK: this driver is also the FALLBACK nyDispatch uses precisely when a
// slot is out of range, and a prefetch computed from a bad slot must not be the
// thing that faults — the scoring line below should be, so the panic names the
// offending slot the way arena.Vec used to.
func nyWarm(base []float32, dim, lines int, slots []uint32, j int) {
	if j >= len(slots) {
		return
	}
	off := int(slots[j]) * dim
	if off < 0 || off+dim > len(base) {
		return
	}
	prefetchRange(unsafe.Pointer(&base[off]), lines)
}

// dotNyPortable is the dot-product counterpart of l2SquaredNyPortable. Cosine
// and DotProduct both reduce to this kernel plus a cheap per-element transform
// applied by the caller (see hnsw.batchExact), exactly as pickDist wraps
// dotProduct for the per-pair path.
func dotNyPortable(q, base []float32, dim int, slots []uint32, out []float32) {
	lines := (dim*4 + 63) / 64
	for i, s := range slots {
		nyWarm(base, dim, lines, slots, i+nyPrefetchAhead)
		off := int(s) * dim
		out[i] = dotProduct(q, base[off:off+dim:off+dim])
	}
}

// avx512NyDim is the batched counterpart of avx512DistDim: on AVX-512-capable
// amd64 the package init sets it to a function returning the 16-lane ZMM batched
// kernel for dim >= avx512MinDim, and nil below that so the caller falls back to
// the default. It stays nil elsewhere, keeping this file platform-neutral.
//
// Sharing avx512MinDim with the per-pair path is deliberate: whichever width the
// per-pair kernel would have used for this dimension, the batched kernel uses
// too, which is what keeps the two bit-identical at every dimension.
var avx512NyDim func(m Metric, dim int) nyFunc

// pickNy returns the batched kernel for metric m at dimension dim, mirroring
// pickDistDim's dispatch. Cosine and DotProduct share the dot kernel — the
// metric's transform (1-d and -d respectively) is applied by the caller over the
// finished output block, which keeps it out of the inner loop. Returns nil for
// an unknown metric rather than panicking: an absent batched kernel simply
// leaves the caller on the per-pair path.
//
// Called once per scorer construction, not per distance.
func pickNy(m Metric, dim int) nyFunc {
	if avx512NyDim != nil {
		if f := avx512NyDim(m, dim); f != nil {
			return f
		}
	}
	switch m {
	case L2:
		return l2SquaredNy
	case Cosine, DotProduct:
		return dotNy
	}
	return nil
}

// nyBoundsOK reports whether every slot addresses a complete dim-element record
// inside base.
//
// This is the ONE place the assembly kernels' precondition is enforced. The
// per-pair path got this for free: arena.Vec slices the slab, so a corrupt
// adjacency entry produced an index-out-of-range panic at the exact offending
// slot. The batched kernels index raw memory, where the same corruption would
// silently score garbage or fault far from its cause — so the check is
// reinstated here rather than dropped as "can't happen".
//
// The test is written against the last valid START offset rather than against a
// record count, specifically to keep an integer DIVISION off the path: this runs
// once per kernel call, and len(base)/dim would put a ~20-40 cycle divide in
// front of a chunk whose whole point is to be cheap. A multiply and a compare
// per slot is a few cycles against a kernel doing far more per candidate.
func nyBoundsOK(slots []uint32, base []float32, dim int) bool {
	limit := len(base) - dim
	for _, s := range slots {
		if int(s)*dim > limit {
			return false
		}
	}
	return true
}

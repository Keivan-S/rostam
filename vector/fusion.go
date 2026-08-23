// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"slices"
)

// FusionMethod and its FusionRRF / FusionWeighted / FusionDBSF constants now live
// in the engine-free vtypes leaf package and are re-exported from
// vtypes_aliases.go.

// defaultRRFK is the standard RRF constant (Cormack et al. 2009).
const defaultRRFK = 60

// fuseRRF combines two ranked lists by reciprocal rank fusion. dense is sorted
// ascending by Distance (rank 0 = nearest); sparse is sorted descending by
// Score (rank 0 = most relevant). A document present in only one lane still
// scores. Returns the top-k by fused Score, descending, ties broken by lower
// ID. Each result carries its dense Distance (0 if sparse-only) and the fused
// Score.
func fuseRRF(dense, sparse []Result, k, rrfK int) []Result {
	if rrfK <= 0 {
		rrfK = defaultRRFK
	}
	// Accumulate into one slice indexed by a map[id]index — avoids allocating a
	// *Result per unique id (previously the dominant hybrid-path allocation).
	acc := make([]Result, 0, len(dense)+len(sparse))
	idx := make(map[uint64]int, len(dense)+len(sparse))
	get := func(id uint64) int {
		if i, ok := idx[id]; ok {
			return i
		}
		i := len(acc)
		acc = append(acc, Result{ID: id})
		idx[id] = i
		return i
	}
	for rank, r := range dense {
		i := get(r.ID)
		acc[i].Score += 1.0 / float32(rrfK+rank+1)
		acc[i].Distance = r.Distance
	}
	for rank, r := range sparse {
		i := get(r.ID)
		acc[i].Score += 1.0 / float32(rrfK+rank+1)
	}
	return rankFused(acc, k)
}

// fuseWeighted combines two ranked lists by a linear blend of min-max
// normalized lane relevances. Dense distances are inverted (smaller distance →
// higher relevance) before normalization. A document missing from a lane
// contributes 0 for that lane. alpha is the dense weight in [0,1].
func fuseWeighted(dense, sparse []Result, k int, alpha float64) []Result {
	if alpha < 0 {
		alpha = 0
	} else if alpha > 1 {
		alpha = 1
	}
	denseNorm := normalizeDense(dense)
	sparseNorm := normalizeSparse(sparse)

	acc := make([]Result, 0, len(dense)+len(sparse))
	idx := make(map[uint64]int, len(dense)+len(sparse))
	get := func(id uint64) int {
		if i, ok := idx[id]; ok {
			return i
		}
		i := len(acc)
		acc = append(acc, Result{ID: id})
		idx[id] = i
		return i
	}
	for _, r := range dense {
		i := get(r.ID)
		acc[i].Distance = r.Distance
		acc[i].Score += float32(alpha) * denseNorm[r.ID]
	}
	for _, r := range sparse {
		i := get(r.ID)
		acc[i].Score += float32(1-alpha) * sparseNorm[r.ID]
	}
	return rankFused(acc, k)
}

// normalizeDense maps each dense result to a [0,1] relevance via inverted
// min-max over distances (smaller distance → higher relevance). A single
// result (or all-equal distances) maps to 1.0.
func normalizeDense(dense []Result) map[uint64]float32 {
	out := make(map[uint64]float32, len(dense))
	if len(dense) == 0 {
		return out
	}
	min, max := dense[0].Distance, dense[0].Distance
	for _, r := range dense {
		if r.Distance < min {
			min = r.Distance
		}
		if r.Distance > max {
			max = r.Distance
		}
	}
	span := max - min
	for _, r := range dense {
		if span == 0 {
			out[r.ID] = 1.0
		} else {
			out[r.ID] = (max - r.Distance) / span
		}
	}
	return out
}

// normalizeSparse maps each sparse result to a [0,1] relevance via min-max over
// scores (larger score → higher relevance).
func normalizeSparse(sparse []Result) map[uint64]float32 {
	out := make(map[uint64]float32, len(sparse))
	if len(sparse) == 0 {
		return out
	}
	min, max := sparse[0].Score, sparse[0].Score
	for _, r := range sparse {
		if r.Score < min {
			min = r.Score
		}
		if r.Score > max {
			max = r.Score
		}
	}
	span := max - min
	for _, r := range sparse {
		if span == 0 {
			out[r.ID] = 1.0
		} else {
			out[r.ID] = (r.Score - min) / span
		}
	}
	return out
}

// laneStats computes the mean and POPULATION standard deviation (divide by n,
// not n-1) of vals, iterating in slice order for determinism. n==0 returns
// (0,0); n==1 returns (vals[0], 0).
func laneStats(vals []float32) (mean, std float32) {
	n := len(vals)
	if n == 0 {
		return 0, 0
	}
	var sum float64
	for _, v := range vals {
		sum += float64(v)
	}
	m := sum / float64(n)
	if n == 1 {
		return float32(m), 0
	}
	var sq float64
	for _, v := range vals {
		d := float64(v) - m
		sq += d * d
	}
	variance := sq / float64(n) // population variance
	return float32(m), float32(math.Sqrt(variance))
}

// clamp01 clamps x to [0,1].
func clamp01(x float32) float32 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

// dbsfNormalizeDense maps each dense result to a [0,1] relevance via 3-sigma
// distribution normalization over the DISTANCES, then INVERTS it (smaller
// distance -> higher relevance), mirroring normalizeDense's orientation. The
// 3-sigma bounds are [mu-3*sigma, mu+3*sigma] over a 6*sigma span. When sigma==0
// (all-equal distances) or n<=1, every id maps to 1.0 (matching normalizeDense's
// all-equal convention).
func dbsfNormalizeDense(dense []Result) map[uint64]float32 {
	out := make(map[uint64]float32, len(dense))
	if len(dense) == 0 {
		return out
	}
	vals := make([]float32, len(dense))
	for i, r := range dense {
		vals[i] = r.Distance
	}
	mean, std := laneStats(vals)
	for _, r := range dense {
		if std == 0 {
			out[r.ID] = 1.0
			continue
		}
		lo := mean - 3*std
		norm := clamp01((r.Distance - lo) / (6 * std))
		out[r.ID] = 1 - norm // invert: smaller distance -> higher relevance
	}
	return out
}

// dbsfNormalizeSparse maps each sparse result to a [0,1] relevance via 3-sigma
// distribution normalization over the SCORES, directly (higher score -> higher
// relevance), mirroring normalizeSparse's orientation. When sigma==0 (all-equal
// scores) or n<=1, every id maps to 1.0.
func dbsfNormalizeSparse(sparse []Result) map[uint64]float32 {
	out := make(map[uint64]float32, len(sparse))
	if len(sparse) == 0 {
		return out
	}
	vals := make([]float32, len(sparse))
	for i, r := range sparse {
		vals[i] = r.Score
	}
	mean, std := laneStats(vals)
	for _, r := range sparse {
		if std == 0 {
			out[r.ID] = 1.0
			continue
		}
		lo := mean - 3*std
		out[r.ID] = clamp01((r.Score - lo) / (6 * std))
	}
	return out
}

// fuseDBSF combines two ranked lists by a linear blend of 3-sigma
// distribution-normalized lane relevances. It mirrors fuseWeighted exactly
// (alpha clamp, blend, top-k desc, lower-id tie-break) but substitutes the
// 3-sigma normalizers for the min-max ones. Dense distances are inverted
// (smaller distance -> higher relevance) before blending. alpha is the dense
// weight in [0,1].
func fuseDBSF(dense, sparse []Result, k int, alpha float64) []Result {
	if alpha < 0 {
		alpha = 0
	} else if alpha > 1 {
		alpha = 1
	}
	denseNorm := dbsfNormalizeDense(dense)
	sparseNorm := dbsfNormalizeSparse(sparse)

	acc := make([]Result, 0, len(dense)+len(sparse))
	idx := make(map[uint64]int, len(dense)+len(sparse))
	get := func(id uint64) int {
		if i, ok := idx[id]; ok {
			return i
		}
		i := len(acc)
		acc = append(acc, Result{ID: id})
		idx[id] = i
		return i
	}
	for _, r := range dense {
		i := get(r.ID)
		acc[i].Distance = r.Distance
		acc[i].Score += float32(alpha) * denseNorm[r.ID]
	}
	for _, r := range sparse {
		i := get(r.ID)
		acc[i].Score += float32(1-alpha) * sparseNorm[r.ID]
	}
	return rankFused(acc, k)
}

// fuseDBSFScoreLanes blends TWO descending-SCORE lanes (both higher = better)
// via 3-sigma distribution normalization. Like fuseWeightedScoreLanes it does
// NOT invert the first lane: both lanes are normalized directly by their Score
// (dbsfNormalizeSparse), so a higher input maps to a higher relevance. alpha
// weights the first (MaxSim) lane. firstLane's per-result Distance is preserved.
func fuseDBSFScoreLanes(firstLane, sparse []Result, k int, alpha float64) []Result {
	if alpha < 0 {
		alpha = 0
	} else if alpha > 1 {
		alpha = 1
	}
	firstNorm := dbsfNormalizeSparse(firstLane) // score-oriented, NOT inverted
	sparseNorm := dbsfNormalizeSparse(sparse)

	acc := make([]Result, 0, len(firstLane)+len(sparse))
	idx := make(map[uint64]int, len(firstLane)+len(sparse))
	get := func(id uint64) int {
		if i, ok := idx[id]; ok {
			return i
		}
		i := len(acc)
		acc = append(acc, Result{ID: id})
		idx[id] = i
		return i
	}
	for _, r := range firstLane {
		i := get(r.ID)
		acc[i].Distance = r.Distance // preserved (MaxSim leaves 0)
		acc[i].Score += float32(alpha) * firstNorm[r.ID]
	}
	for _, r := range sparse {
		i := get(r.ID)
		acc[i].Score += float32(1-alpha) * sparseNorm[r.ID]
	}
	return rankFused(acc, k)
}

// Fuse combines dense and sparse lanes into the top-k, mirroring the fusion
// branch of HybridSearch exactly (including the alpha==0 -> 0.5 default).
// Exported so a cross-partition coordinator can re-fuse globally-merged lanes
// (exact hybrid fan-out). dense is the dense lane (asc by Distance), sparse the
// sparse lane (desc by Score). rrfK==0 lets fuseRRF apply its own defaultRRFK
// (same as opts.RRFK==0 in HybridSearch).
func Fuse(dense, sparse []Result, method FusionMethod, alpha float64, rrfK, k int) []Result {
	switch method {
	case FusionWeighted:
		if alpha == 0 {
			alpha = 0.5
		}
		return fuseWeighted(dense, sparse, k, alpha)
	case FusionDBSF:
		if alpha == 0 {
			alpha = 0.5
		}
		return fuseDBSF(dense, sparse, k, alpha)
	}
	return fuseRRF(dense, sparse, k, rrfK)
}

// FuseScoreLanes combines TWO descending-SCORE lanes into the top-k. It is the
// score-oriented sibling of Fuse for the case where the "dense" lane is itself a
// DESCENDING relevance SCORE (higher = better) rather than an ascending distance
// — the MV MaxSim lane. RRF is rank-only, so it is identical to Fuse (both lanes
// are already rank-ordered: dense by Score desc, sparse by Score desc). The
// difference is the WEIGHTED path: Fuse's fuseWeighted INVERTS the dense lane via
// normalizeDense (it assumes dense.Distance is an ascending distance), which would
// INVERT a MaxSim score lane (ranking the WORST doc highest). FuseScoreLanes
// instead normalizes BOTH lanes as scores (normalizeSparse on each), so a higher
// MaxSim score maps to a higher relevance. firstLane carries Score desc; sparse
// carries Score desc. The fused Score is the blended relevance; firstLane's
// per-result Distance is preserved (MaxSim leaves Distance 0).
func FuseScoreLanes(firstLane, sparse []Result, method FusionMethod, alpha float64, rrfK, k int) []Result {
	if method == FusionWeighted {
		if alpha == 0 {
			alpha = 0.5
		}
		return fuseWeightedScoreLanes(firstLane, sparse, k, alpha)
	}
	if method == FusionDBSF {
		if alpha == 0 {
			alpha = 0.5
		}
		return fuseDBSFScoreLanes(firstLane, sparse, k, alpha)
	}
	// RRF is rank-only; fuseRRF reads each lane in slice order (dense by rank,
	// sparse by rank) and never touches Distance for ranking, so a score-desc
	// "dense" lane fuses correctly with no change.
	return fuseRRF(firstLane, sparse, k, rrfK)
}

// fuseWeightedScoreLanes blends TWO min-max-normalized SCORE lanes (both higher =
// better). Unlike fuseWeighted it does NOT invert the first lane's distance: the
// first lane (MaxSim) is normalized by its Score (normalizeSparse), exactly like
// the sparse lane, so both lanes contribute a [0,1] relevance where higher input =
// higher relevance. alpha weights the first (MaxSim) lane.
func fuseWeightedScoreLanes(firstLane, sparse []Result, k int, alpha float64) []Result {
	if alpha < 0 {
		alpha = 0
	} else if alpha > 1 {
		alpha = 1
	}
	firstNorm := normalizeSparse(firstLane) // score-oriented (higher = better), NOT inverted
	sparseNorm := normalizeSparse(sparse)

	acc := make([]Result, 0, len(firstLane)+len(sparse))
	idx := make(map[uint64]int, len(firstLane)+len(sparse))
	get := func(id uint64) int {
		if i, ok := idx[id]; ok {
			return i
		}
		i := len(acc)
		acc = append(acc, Result{ID: id})
		idx[id] = i
		return i
	}
	for _, r := range firstLane {
		i := get(r.ID)
		acc[i].Distance = r.Distance // preserved (MaxSim leaves 0)
		acc[i].Score += float32(alpha) * firstNorm[r.ID]
	}
	for _, r := range sparse {
		i := get(r.ID)
		acc[i].Score += float32(1-alpha) * sparseNorm[r.ID]
	}
	return rankFused(acc, k)
}

// rankFused sorts the accumulated results in place by Score descending (ties →
// lower ID) and truncates to k. Uses slices.SortFunc (generic, no reflection)
// to avoid the per-call allocations sort.Slice's reflect.Swapper incurs.
func rankFused(acc []Result, k int) []Result {
	slices.SortFunc(acc, func(a, b Result) int {
		switch {
		case a.Score > b.Score:
			return -1
		case a.Score < b.Score:
			return 1
		case a.ID < b.ID:
			return -1
		case a.ID > b.ID:
			return 1
		default:
			return 0
		}
	})
	if k > 0 && len(acc) > k {
		acc = acc[:k]
	}
	return acc
}

// SPDX-License-Identifier: Apache-2.0

package rag

import (
	"sort"
	"strconv"
)

func fuseKey(h Hit) string { return h.Source + "#" + strconv.Itoa(h.Index) }

// fuse merges a dense-ranked and a BM25-ranked hit list into the top-k fused
// hits. alpha < 0 selects RRF (equal-weight 1/(60+rank+1) per lane); 0 <=
// alpha <= 1 selects a weighted rank-blend (dense weight = alpha). Chunk
// identity for dedup is source#index. Output order is deterministic: ties
// break by first-seen order across dense then bm25. Each returned Hit's Score
// is set to its fusion score (RRF or weighted), NOT the original lane score.
func fuse(dense, bm25 []Hit, k int, alpha float64) []Hit {
	type acc struct {
		hit   Hit
		score float64
	}
	byKey := map[string]*acc{}
	order := []string{} // first-seen order, for deterministic tie-breaks

	add := func(hits []Hit, weight float64, rrf bool) {
		for rank, ht := range hits {
			key := fuseKey(ht)
			a, ok := byKey[key]
			if !ok {
				a = &acc{hit: ht}
				byKey[key] = a
				order = append(order, key)
			}
			var contrib float64
			if rrf {
				contrib = 1.0 / float64(60+rank+1)
			} else {
				contrib = weight * (1.0 / float64(rank+1))
			}
			a.score += contrib
		}
	}

	if alpha < 0 { // RRF: equal-weight rank fusion
		add(dense, 0, true)
		add(bm25, 0, true)
	} else { // weighted rank-blend
		add(dense, alpha, false)
		add(bm25, 1-alpha, false)
	}

	fused := make([]*acc, 0, len(order))
	pos := map[string]int{}
	for i, key := range order {
		pos[key] = i
		fused = append(fused, byKey[key])
	}
	sort.SliceStable(fused, func(i, j int) bool {
		if fused[i].score != fused[j].score {
			return fused[i].score > fused[j].score
		}
		return pos[fuseKey(fused[i].hit)] < pos[fuseKey(fused[j].hit)]
	})
	if k > 0 && len(fused) > k {
		fused = fused[:k]
	}
	out := make([]Hit, len(fused))
	for i, a := range fused {
		out[i] = a.hit
		out[i].Score = float32(a.score)
	}
	return out
}

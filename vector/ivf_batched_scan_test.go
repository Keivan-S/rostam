// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"math"
	"testing"
)

// The IVF-Flat cell scan hands each probed cell's admitted members to the same
// batched kernels the HNSW traversal uses (see ivf.gatherFlatBatched). Those
// kernels are bit-identical to the per-pair ones BY CONSTRUCTION — same
// accumulator layout, same combine tree — and the batched cell scan visits the
// same slots in the same order, so the two gathers must produce byte-identical
// []slotDist, sort included. That is a strictly stronger claim than the
// "same top-k" a recall test would make, and it is the claim these tests assert:
// anything weaker would leave a real kernel bug indistinguishable from a tie
// breaking the other way.

// buildTrainedIVFT builds a trained IVF-Flat index over a deterministic corpus.
// The test analogue of ivf_bench_test.go's buildTrainedIVF; it uses the
// incremental Insert + auto-train path (IVFTrainThreshold well below n) so the
// lists carry the same reused-slot/dedup shape a live index has.
func buildTrainedIVFT(t *testing.T, m Metric, dim, n, nprobe int, seed int64) (*ivf, [][]float32) {
	t.Helper()
	ids, vecs := siftLikeCorpus(n, dim, seed)

	cfg := DefaultConfig()
	cfg.Dim = dim
	cfg.Metric = m
	cfg.Seed = seed
	cfg.IndexType = IndexIVF
	cfg.IVFNprobe = nprobe
	cfg.IVFTrainThreshold = n / 4
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	insertAll(t, ix, ids, vecs)
	if !ix.trained || len(ix.centroids) == 0 {
		t.Fatal("fixture: index did not train; the cell-scan path is never reached")
	}
	return ix, vecs
}

// gatherBothWays runs one query through gatherLocked (or gatherLockedWith, when
// metaOf != nil) with the batched cell scan ON and then OFF, returning both
// results. The knob is read inside the gather, so flipping it between the two
// calls over the SAME index is what isolates the scan from every other source of
// difference (training, assignment, probe order).
func gatherBothWays(t *testing.T, ix *ivf, q []float32, metaOf metaProvider) (on, off []slotDist) {
	t.Helper()
	defer func(prev bool) { batchedExpand = prev }(batchedExpand)

	ix.mu.RLock()
	defer ix.mu.RUnlock()

	batchedExpand = true
	if !ix.batchExact(q).ok() {
		t.Fatal("fixture: batchedExpand=true must yield a batched kernel for an " +
			"unquantized IVF-Flat arena; the batched scan is never exercised")
	}
	on = append(on, ix.gatherLockedWith(q, 10, nil, metaOf)...)

	batchedExpand = false
	if ix.batchExact(q).ok() {
		t.Fatal("batchedExpand=false still produced a batched kernel; the knob does not work")
	}
	off = append(off, ix.gatherLockedWith(q, 10, nil, metaOf)...)
	return on, off
}

// requireIdenticalGathers asserts the two gathers agree on length, slot order,
// and the exact bits of every distance.
func requireIdenticalGathers(t *testing.T, qi int, on, off []slotDist) {
	t.Helper()
	if len(on) != len(off) {
		t.Fatalf("query %d: batched gather returned %d candidates, per-pair returned %d",
			qi, len(on), len(off))
	}
	for i := range on {
		if on[i].slot != off[i].slot {
			t.Fatalf("query %d [%d]: batched gather ordered slot %d where per-pair had %d",
				qi, i, on[i].slot, off[i].slot)
		}
		if math.Float32bits(on[i].dist) != math.Float32bits(off[i].dist) {
			t.Fatalf("query %d [%d] slot %d: batched=%v per-pair=%v — the batched cell "+
				"scan must be bit-identical to the per-pair scan",
				qi, i, on[i].slot, on[i].dist, off[i].dist)
		}
	}
	if len(on) == 0 {
		t.Fatalf("query %d: both gathers returned nothing; the fixture probes no candidates", qi)
	}
}

// TestIVFBatchedCellScanMatchesPerPair is the differential bit-equality pin for
// the IVF-Flat cell scan, per metric. Cosine and DotProduct additionally cover
// the metric transform batchKernel.score applies over the finished block (1-d
// and -d), which the L2 arm does not touch.
func TestIVFBatchedCellScanMatchesPerPair(t *testing.T) {
	for _, m := range []Metric{L2, Cosine, DotProduct} {
		for _, dim := range []int{8, 64, 128} {
			t.Run(fmt.Sprintf("%v/dim=%d", m, dim), func(t *testing.T) {
				const (
					n      = 4000
					nprobe = 16
					seed   = 11
				)
				ix, _ := buildTrainedIVFT(t, m, dim, n, nprobe, seed)
				defer ix.Close()
				_, queries := siftLikeCorpus(25, dim, 3)

				for qi, raw := range queries {
					q := raw
					// Cosine normalizes on insert; score the normalized form the
					// search path would actually hand the gather.
					if m == Cosine {
						q = append([]float32(nil), raw...)
						normalize(q)
					}
					on, off := gatherBothWays(t, ix, q, nil)
					requireIdenticalGathers(t, qi, on, off)
				}
			})
		}
	}
}

// TestIVFBatchedCellScanMatchesPerPairWithMeta covers the SECOND batched call
// site: gatherLockedWith's external-metadata arm, whose admit gate is
// admitsWith(metaOf) rather than admits. It is a separate loop in the source, so
// a divergence there would not be caught by the test above. The predicate is
// non-nil and selective so admission actually rejects members — a gate that
// accepts everything would not distinguish "the admitted set is preserved" from
// "the whole list is scored".
func TestIVFBatchedCellScanMatchesPerPairWithMeta(t *testing.T) {
	const (
		dim    = 64
		n      = 4000
		nprobe = 16
		seed   = 13
	)
	ix, _ := buildTrainedIVFT(t, L2, dim, n, nprobe, seed)
	defer ix.Close()
	_, queries := siftLikeCorpus(25, dim, 5)

	// Half the ids carry the payload the predicate wants.
	metaOf := func(id uint64) Metadata {
		if id%2 == 0 {
			return Metadata{"kind": NewString("even")}
		}
		return Metadata{"kind": NewString("odd")}
	}
	pred, err := CompileFilter(Filter{Op: FilterEq, Field: "kind", Value: NewString("even")})
	if err != nil {
		t.Fatal(err)
	}

	defer func(prev bool) { batchedExpand = prev }(batchedExpand)
	for qi, q := range queries {
		ix.mu.RLock()
		batchedExpand = true
		if !ix.batchExact(q).ok() {
			ix.mu.RUnlock()
			t.Fatal("fixture: the external-metadata arm must reach the batched cell scan")
		}
		on := append([]slotDist(nil), ix.gatherLockedWith(q, 10, pred, metaOf)...)
		batchedExpand = false
		off := append([]slotDist(nil), ix.gatherLockedWith(q, 10, pred, metaOf)...)
		ix.mu.RUnlock()

		requireIdenticalGathers(t, qi, on, off)
		for _, sd := range on {
			if id := ix.slotID(sd.slot); id%2 != 0 {
				t.Fatalf("query %d: slot %d (id %d) was admitted despite the predicate", qi, sd.slot, id)
			}
		}
	}
}

// TestIVFBatchExactDeclinesWhenUnusable pins ivf.batchExact's decline rules.
// Each is a case where the kernel's single (base, dim) contract cannot be
// honoured, and declining — leaving the gather on the per-pair path — is the
// only safe answer.
func TestIVFBatchExactDeclinesWhenUnusable(t *testing.T) {
	const dim = 16
	cfg := DefaultConfig()
	cfg.Dim = dim
	cfg.Metric = L2
	cfg.IndexType = IndexIVF
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer ix.Close()
	v := make([]float32, dim)
	if _, _, err := ix.Insert(1, v, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatal(err)
	}

	if !ix.batchExact(v).ok() {
		t.Fatal("fixture: a well-formed IVF-Flat arena must yield a batched kernel")
	}
	if ix.batchExact(make([]float32, dim+1)).ok() {
		t.Error("batchExact accepted a query longer than the arena stride")
	}
	if ix.batchExact(make([]float32, dim-1)).ok() {
		t.Error("batchExact accepted a short query")
	}

	saved := ix.arena.vecsDropped
	ix.arena.vecsDropped = true
	if ix.batchExact(v).ok() {
		t.Error("batchExact returned a kernel for an arena whose float slab was dropped " +
			"(the IVF-PQ PQ-only state); the flat scan would read reclaimed storage")
	}
	ix.arena.vecsDropped = saved
}

// TestIVFQuantizedGathersStayPerPair pins the SCOPE of the change: the batched
// float kernel must not reach the IVF-PQ gathers. Their candidates are scored
// from arena.Code through the residual codec, which has no batched float form —
// routing them through batchExact would score the wrong array. Asserted by
// flipping the knob and requiring the ADC results to be unchanged, which they
// are only because nothing on that path consults it.
func TestIVFQuantizedGathersStayPerPair(t *testing.T) {
	const (
		dim    = 64
		n      = 4000
		nprobe = 16
		seed   = 17
		pqm    = 8
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	cfg := DefaultConfig()
	cfg.Dim = dim
	cfg.Metric = L2
	cfg.Seed = seed
	cfg.IndexType = IndexIVF
	cfg.IVFNprobe = nprobe
	cfg.IVFPQ = true
	cfg.IVFPQM = pqm
	// Rerank keeps the float slab resident, so batchExact would HAND OUT a kernel
	// if the ADC gather asked for one. That is what makes the assertion below
	// meaningful rather than vacuous: PQ-only drops the floats and would decline
	// for an unrelated reason (see TestIVFBatchExactDeclinesWhenUnusable).
	cfg.IVFRerank = true
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer ix.Close()
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	if !ix.pqActive() {
		t.Fatal("fixture: IVF-PQ did not train; the ADC gather is never reached")
	}

	_, queries := siftLikeCorpus(25, dim, 19)
	defer func(prev bool) { batchedExpand = prev }(batchedExpand)
	for qi, q := range queries {
		ix.mu.RLock()
		batchedExpand = true
		// The floats are still resident here, so a batched kernel IS available —
		// which is what keeps this assertion from being vacuous. The ADC gather
		// must ignore it anyway.
		if !ix.batchExact(q).ok() {
			ix.mu.RUnlock()
			t.Fatal("fixture: a kernel must be available, otherwise this test proves nothing")
		}
		on := append([]slotDist(nil), ix.gatherLocked(q, 10, nil)...)
		batchedExpand = false
		off := append([]slotDist(nil), ix.gatherLocked(q, 10, nil)...)
		ix.mu.RUnlock()
		requireIdenticalGathers(t, qi, on, off)
	}
}

// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// TestNamedHybridFanOutReportsDegraded proves finding 023 for the named family: a
// P>1 named hybrid search with one unreachable partition under the default Partial
// mode must report FanMeta{Degraded:true, Missing:...} instead of silently returning
// an incomplete top-k. Before the fix, namedHybridFanOut discarded cluster.FanResult
// and VectorNamedHybridSearch had NO FanMeta return, so there was no channel for the
// signal. The new VectorNamedHybridSearchExt threads it through. Fail mode still errors.
func TestNamedHybridFanOutReportsDegraded(t *testing.T) {
	const n, k = 80, 10
	ctx := context.Background()
	denseQ := []float32{1.2, 0.6, 0.3, 0.4}
	sparseQ := vector.SparseVector{Indices: []uint32{1, 14, 26}, Values: []float32{2, 3, 1}}

	const P = 4
	s := seedNamedHybridCollection(t, "nhd", P, n)
	emb := s.(*embedded)

	// Sanity: healthy search is not degraded.
	if _, meta, err := emb.VectorNamedHybridSearchExt(ctx, "nhd", "title", denseQ, "terms", sparseQ, k,
		NamedHybridOpts{Method: FusionRRF}); err != nil || meta.Degraded {
		t.Fatalf("healthy named hybrid: err=%v degraded=%v", err, meta.Degraded)
	}

	// Make partition 1 unreachable by dropping its physical collection (gen 0).
	if _, err := emb.Call(ctx, "vector_named_drop_collection",
		ops.EncodeNamedNameArgs(string(ops.PartitionKeyGen("nhd", 0, 1)))); err != nil {
		t.Fatalf("drop partition 1: %v", err)
	}

	// Fail mode → error.
	if _, _, err := emb.VectorNamedHybridSearchExt(ctx, "nhd", "title", denseQ, "terms", sparseQ, k,
		NamedHybridOpts{Method: FusionRRF, OnPartitionUnavailable: 1}); err == nil {
		t.Fatal("expected error with OnPartitionUnavailable=Fail and an unreachable partition")
	}

	// Partial mode (default) → degraded + partial (no error).
	res, meta, err := emb.VectorNamedHybridSearchExt(ctx, "nhd", "title", denseQ, "terms", sparseQ, k,
		NamedHybridOpts{Method: FusionRRF})
	if err != nil {
		t.Fatalf("partial-mode named hybrid errored: %v", err)
	}
	if !meta.Degraded {
		t.Fatal("expected Degraded=true after a partition was made unreachable")
	}
	if len(meta.Missing) == 0 {
		t.Fatal("expected non-empty Missing after a partition was made unreachable")
	}
	if len(res) == 0 {
		t.Fatal("expected partial results from the reachable partitions")
	}
}

// TestMVHybridFanOutReportsDegraded is the MV-family mirror of
// TestNamedHybridFanOutReportsDegraded: a P>1 MV hybrid search with one unreachable
// partition under the default Partial mode reports FanMeta{Degraded, Missing} via the
// new VectorMVHybridSearchExt. Keeps the three hybrid families' degradation contract
// consistent (finding 023).
func TestMVHybridFanOutReportsDegraded(t *testing.T) {
	const n, k = 80, 10
	ctx := context.Background()
	query := [][]float32{{1.2, 0.6, 0.3, 0.4}, {0.2, 0.9, 0.1, 0.5}}
	sparseQ := vector.SparseVector{Indices: []uint32{1, 14, 26}, Values: []float32{2, 3, 1}}

	const P = 4
	s := seedMVHybridCollection(t, "mhd_deg", P, n)
	emb := s.(*embedded)

	// Sanity: healthy search is not degraded.
	if _, meta, err := emb.VectorMVHybridSearchExt(ctx, "mhd_deg", query, sparseQ, k,
		MVHybridOpts{Method: FusionRRF}); err != nil || meta.Degraded {
		t.Fatalf("healthy mv hybrid: err=%v degraded=%v", err, meta.Degraded)
	}

	// Make partition 1 unreachable by dropping its physical collection (gen 0).
	if _, err := emb.Call(ctx, "vector_mv_drop_collection",
		ops.EncodeMVDeleteArgs(string(ops.PartitionKeyGen("mhd_deg", 0, 1)), 0)); err != nil {
		t.Fatalf("drop partition 1: %v", err)
	}

	// Fail mode → error.
	if _, _, err := emb.VectorMVHybridSearchExt(ctx, "mhd_deg", query, sparseQ, k,
		MVHybridOpts{Method: FusionRRF, OnPartitionUnavailable: 1}); err == nil {
		t.Fatal("expected error with OnPartitionUnavailable=Fail and an unreachable partition")
	}

	// Partial mode (default) → degraded + partial (no error).
	res, meta, err := emb.VectorMVHybridSearchExt(ctx, "mhd_deg", query, sparseQ, k,
		MVHybridOpts{Method: FusionRRF})
	if err != nil {
		t.Fatalf("partial-mode mv hybrid errored: %v", err)
	}
	if !meta.Degraded {
		t.Fatal("expected Degraded=true after a partition was made unreachable")
	}
	if len(meta.Missing) == 0 {
		t.Fatal("expected non-empty Missing after a partition was made unreachable")
	}
	if len(res) == 0 {
		t.Fatal("expected partial results from the reachable partitions")
	}
}

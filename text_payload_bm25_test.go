// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// scoreOf returns the BM25 score VectorSearchText assigns to id for query q on
// collection coll (0 if the doc is not in the result set).
func scoreOf(t *testing.T, s Store, coll, q string, id uint64) float32 {
	t.Helper()
	docs, _, err := s.VectorSearchText(context.Background(), coll, q, 50, VectorSearchOpts{})
	if err != nil {
		t.Fatalf("SearchText %q: %v", q, err)
	}
	for _, d := range docs {
		if d.ID == id {
			return d.Score
		}
	}
	return 0
}

// matchesText reports whether VectorSearchText for q returns id at all.
func matchesText(t *testing.T, s Store, coll, q string, id uint64) bool {
	t.Helper()
	return scoreOf(t, s, coll, q, id) != 0
}

// TestPayloadMutationsMaintainBM25 is the B1 regression: the four payload-mutation
// ops (SetPayload / OverwritePayload / DeletePayloadKeys / ClearPayload) must
// re-maintain the BM25 index when they change or drop the reserved $content, or
// stale postings + skewed corpus stats (n/df/avgdl) corrupt scoring for ALL docs
// until a Reclaim(). Each sub-case compares against a fresh reference collection.
func TestPayloadMutationsMaintainBM25(t *testing.T) {
	ctx := context.Background()

	// (a) OverwritePayload that DROPS $content: the doc must leave the BM25 corpus
	// cleanly — it no longer matches its old terms AND the global stats reflect its
	// removal, so an UNRELATED doc's score for an unrelated term equals the score it
	// has in a reference corpus that never contained the dropped doc.
	t.Run("overwrite_drops_content", func(t *testing.T) {
		s := newSingleEmbedded(t)
		waitLeaderEmbedded(t, s)
		// Live corpus: docs 1 (fox) + 2 (dog) + 3 (dog).
		seedSubset(t, s, "ft", 1, []uint64{1, 2, 3})
		// Reference corpus WITHOUT doc 1 (as if it had never carried content).
		ref := newSingleEmbedded(t)
		waitLeaderEmbedded(t, ref)
		seedSubset(t, ref, "ref", 1, []uint64{2, 3})

		// Drop doc 1's content (overwrite with a payload that has NO $content).
		if _, err := s.VectorOverwritePayload(ctx, "ft", 1, vector.Metadata{"k": vector.NewInt(1)}, nil); err != nil {
			t.Fatalf("overwrite drop content: %v", err)
		}

		// Doc 1 no longer matches its old term "fox".
		if matchesText(t, s, "ft", "fox", 1) {
			t.Fatal("doc 1 still matches 'fox' after its $content was dropped")
		}
		// An unrelated doc's score for "dog" now matches the reference corpus that
		// never contained doc 1 — i.e. n/df/avgdl were corrected by the removal.
		got := scoreOf(t, s, "ft", "dog", 2)
		want := scoreOf(t, ref, "ref", "dog", 2)
		if got != want {
			t.Fatalf("doc 2 'dog' score after drop = %v, want %v (corpus stats not corrected)", got, want)
		}
	})

	// (b) OverwritePayload / SetPayload that SETS $content to NEW text: SearchText
	// finds the doc on the new terms, not the old.
	t.Run("set_new_content", func(t *testing.T) {
		s := newSingleEmbedded(t)
		waitLeaderEmbedded(t, s)
		seedSubset(t, s, "ft", 1, []uint64{1, 2, 3})

		// OverwritePayload: replace doc 1's content with brand-new vocabulary.
		newText := "telescope astronomy nebula"
		if _, err := s.VectorOverwritePayload(ctx, "ft", 1, vector.WithContent(nil, newText), nil); err != nil {
			t.Fatalf("overwrite new content: %v", err)
		}
		if matchesText(t, s, "ft", "fox", 1) {
			t.Fatal("doc 1 still matches old term 'fox' after content replaced")
		}
		if !matchesText(t, s, "ft", "telescope", 1) {
			t.Fatal("doc 1 does not match new term 'telescope' after content replaced")
		}

		// SetPayload patch that INCLUDES $content overwrites just the content key.
		if _, err := s.VectorSetPayload(ctx, "ft", 1, vector.WithContent(nil, "microscope biology cell"), nil); err != nil {
			t.Fatalf("set new content: %v", err)
		}
		if matchesText(t, s, "ft", "telescope", 1) {
			t.Fatal("doc 1 still matches 'telescope' after SetPayload replaced $content")
		}
		if !matchesText(t, s, "ft", "microscope", 1) {
			t.Fatal("doc 1 does not match 'microscope' after SetPayload set $content")
		}
	})

	// (c) ClearPayload / DeletePayloadKeys including $content: the doc drops from the
	// BM25 corpus (no longer matches, stats corrected vs a reference without it).
	t.Run("clear_and_delete_drop_content", func(t *testing.T) {
		for _, mode := range []string{"clear", "deletekeys"} {
			t.Run(mode, func(t *testing.T) {
				s := newSingleEmbedded(t)
				waitLeaderEmbedded(t, s)
				seedSubset(t, s, "ft", 1, []uint64{1, 2, 3})
				ref := newSingleEmbedded(t)
				waitLeaderEmbedded(t, ref)
				seedSubset(t, ref, "ref", 1, []uint64{2, 3})

				var err error
				if mode == "clear" {
					_, err = s.VectorClearPayload(ctx, "ft", 1)
				} else {
					// $content is the reserved content key — deleting it drops the doc's text.
					_, err = s.VectorDeletePayloadKeys(ctx, "ft", 1, []string{"$content"})
				}
				if err != nil {
					t.Fatalf("%s: %v", mode, err)
				}

				if matchesText(t, s, "ft", "fox", 1) {
					t.Fatalf("%s: doc 1 still matches 'fox' after content removed", mode)
				}
				got := scoreOf(t, s, "ft", "dog", 2)
				want := scoreOf(t, ref, "ref", "dog", 2)
				if got != want {
					t.Fatalf("%s: doc 2 'dog' score = %v, want %v (stats not corrected)", mode, got, want)
				}
			})
		}
	})

	// (d) SetPayload patch that does NOT touch $content: BM25 results + scores are
	// byte-identical to before the patch (no churn, no stat drift).
	t.Run("set_unrelated_no_churn", func(t *testing.T) {
		s := newSingleEmbedded(t)
		waitLeaderEmbedded(t, s)
		seedSubset(t, s, "ft", 1, []uint64{1, 2, 3})

		// Baseline: every doc's score for the common term "dog".
		base := map[uint64]float32{}
		for _, id := range []uint64{1, 2, 3} {
			base[id] = scoreOf(t, s, "ft", "dog", id)
		}

		// Patch an unrelated metadata key (does NOT include $content).
		if _, err := s.VectorSetPayload(ctx, "ft", 1, vector.Metadata{"tag": vector.NewInt(7)}, nil); err != nil {
			t.Fatalf("set unrelated: %v", err)
		}

		for _, id := range []uint64{1, 2, 3} {
			if got := scoreOf(t, s, "ft", "dog", id); got != base[id] {
				t.Fatalf("doc %d 'dog' score drifted after a $content-untouching patch: %v != %v", id, got, base[id])
			}
		}
		// And the content is still searchable (old terms still match).
		if !matchesText(t, s, "ft", "fox", 1) {
			t.Fatal("doc 1 lost its 'fox' match after an unrelated SetPayload patch")
		}
	})
}

// TestHybridTextFanOutStopwordSingleShard is the M2 regression: a non-empty query
// that analyzes to ZERO terms (all-stopword "the the") must take the SAME single-
// lane-degradation path under partitioning as on a single node. On a single-shard
// placement (every touched id in one partition) P-many must equal P1 (pure-dense
// degradation), not Fuse(dense, empty).
func TestHybridTextFanOutStopwordSingleShard(t *testing.T) {
	const P = 4
	part, ids := allInOnePartition(P)
	if len(ids) < 2 {
		t.Fatalf("single-partition corpus too small (part %d has %d ids)", part, len(ids))
	}
	ctx := context.Background()

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedSubset(t, s1, "ft1", 1, ids)

	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedSubset(t, sP, "ftP", P, ids)

	for _, id := range ids {
		if vp := ops.PartitionOf(id, P); vp != part {
			t.Fatalf("id %d not in partition %d (got %d)", id, part, vp)
		}
	}

	// "the the" analyzes to zero terms (all stopwords) — no text lane. With a dense
	// vector present, both single-node HybridText and the fan-out must degrade to
	// PURE DENSE (identical ids + scores on a single-shard placement).
	dense := denseFor(ids[0])
	opts := VectorHybridOpts{Method: FusionRRF}
	r1, _, err := s1.VectorHybridText(ctx, "ft1", dense, "the the", 10, opts)
	if err != nil {
		t.Fatalf("P1 HybridText stopword: %v", err)
	}
	rP, fm, err := sP.VectorHybridText(ctx, "ftP", dense, "the the", 10, opts)
	if err != nil {
		t.Fatalf("P%d HybridText stopword: %v", P, err)
	}
	if fm.Degraded {
		t.Fatal("P-many HybridText degraded unexpectedly")
	}
	if !equalIDs(resultIDs(rP), resultIDs(r1)) {
		t.Fatalf("stopword hybrid P-many != P1:\n P%d=%v\n P1=%v", P, resultIDs(rP), resultIDs(r1))
	}
	// Pure-dense degradation: scores match too on a single-shard placement.
	for i := range r1 {
		if rP[i].Score != r1[i].Score {
			t.Fatalf("stopword hybrid score mismatch at %d: P-many=%v P1=%v", i, rP[i].Score, r1[i].Score)
		}
	}
}

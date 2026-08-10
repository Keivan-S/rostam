// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// seedMV creates a partitioned multi-vector collection and adds ids in RANDOM
// order (so a stable global ascending scroll proves the merge, not add order),
// each tagged even/odd for the filter test. Mirrors seedDense for the MV family.
func seedMV(t *testing.T, s Store, coll string, P int, ids []uint64) {
	t.Helper()
	ctx := context.Background()
	if err := s.VectorMVCreateCollection(ctx, coll, MultiVectorConfig{Dim: 4, Partitions: P}); err != nil {
		t.Fatalf("VectorMVCreateCollection: %v", err)
	}
	for _, id := range ids {
		md := VectorMetadata{"even": vector.NewBool(id%2 == 0)}
		if err := s.VectorMVAdd(ctx, coll, id, [][]float32{mvTokenAt(int(id))}, md); err != nil {
			t.Fatalf("VectorMVAdd %d: %v", id, err)
		}
	}
}

// pageAllMV pages an MV collection start-to-exhaustion via the cursor, returning
// the concatenated id sequence (page order). It asserts per-page invariants
// (ascending within a page, no cross-page descent, next_cursor empty IFF the last
// page is short) so a merge/cursor bug fails loud here. Mirrors pageAllDense.
func pageAllMV(t *testing.T, s Store, coll string, filter VectorFilter, limit int) (ids []uint64, pages int) {
	t.Helper()
	ctx := context.Background()
	cursor := ""
	var last uint64
	have := false
	for {
		docs, _, next, err := s.VectorMVScroll(ctx, coll, filter, limit, cursor)
		if err != nil {
			t.Fatalf("VectorMVScroll page %d: %v", pages, err)
		}
		pages++
		for i, d := range docs {
			if i > 0 && d.ID <= docs[i-1].ID {
				t.Fatalf("MV page %d not strictly ascending at %d: %d <= %d", pages, i, d.ID, docs[i-1].ID)
			}
			if have && d.ID <= last {
				t.Fatalf("MV page %d id %d not > previous page's last %d (gap/dup/order bug)", pages, d.ID, last)
			}
			ids = append(ids, d.ID)
			last = d.ID
			have = true
		}
		// Exhaustion rule: a full page (len==limit) must carry a next_cursor; a
		// short/empty page must end pagination.
		if len(docs) == limit {
			if next == "" {
				t.Fatalf("MV page %d full (len=%d) but next_cursor empty", pages, limit)
			}
		} else if next != "" {
			t.Fatalf("MV page %d short (len=%d<%d) but next_cursor=%q (not exhausted)", pages, len(docs), limit, next)
		}
		if next == "" {
			return ids, pages
		}
		cursor = next
		if pages > limit*1000+100 { // runaway guard
			t.Fatalf("MV pagination did not terminate after %d pages", pages)
		}
	}
}

// TestEmbeddedMVScrollCursorDeepPaginationPartitioned: a P=4 MV collection seeded
// with N distinct ids in random order, paged with limit L from cursor="" to
// exhaustion. Every id appears EXACTLY once, globally ascending across page
// boundaries (gap-free + dup-free vs the ground-truth set); page count ≈
// ceil(N/L). Single global cursor pages the whole partitioned collection.
func TestEmbeddedMVScrollCursorDeepPaginationPartitioned(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)

	const (
		coll = "mvdeep"
		P    = 4
		N    = 250
		L    = 30
	)
	ids := shuffledIDs(N, 73)
	seedMV(t, s, coll, P, ids)

	want := map[uint64]bool{}
	for _, id := range ids {
		want[id] = true
	}

	got, pages := pageAllMV(t, s, coll, VectorFilter{}, L)
	assertExactlyOnceAscending(t, got, want)

	wantPages := (N + L - 1) / L // ceil(N/L)
	if pages != wantPages && pages != wantPages+1 {
		t.Fatalf("page count = %d, want %d or %d (ceil(N/L))", pages, wantPages, wantPages+1)
	}
}

// TestEmbeddedMVScrollCursorSingleGlobalCursorResume: a page's next_cursor resumes
// correctly on the next call across partitions — the second page begins strictly
// after the first page's last id, and the two pages together are gap-free.
func TestEmbeddedMVScrollCursorSingleGlobalCursorResume(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	const (
		coll = "mvresume"
		P    = 4
		N    = 100
		L    = 17
	)
	ids := shuffledIDs(N, 11)
	seedMV(t, s, coll, P, ids)

	page1, _, next1, err := s.VectorMVScroll(ctx, coll, VectorFilter{}, L, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(page1) != L {
		t.Fatalf("page1 len = %d, want %d", len(page1), L)
	}
	if next1 == "" {
		t.Fatalf("page1 full but next_cursor empty")
	}
	// page1 is the smallest L ids ascending (1..L since ids are 1..N).
	for i := 0; i < L; i++ {
		if page1[i].ID != uint64(i+1) {
			t.Fatalf("page1[%d].ID = %d, want %d (smallest-id ascending)", i, page1[i].ID, i+1)
		}
	}

	page2, _, _, err := s.VectorMVScroll(ctx, coll, VectorFilter{}, L, next1)
	if err != nil {
		t.Fatal(err)
	}
	if len(page2) == 0 {
		t.Fatalf("page2 empty after resume")
	}
	// Resume is strictly after page1's last id (the single global cursor crosses
	// partitions): page2 begins at L+1, no gap, no dup.
	if page2[0].ID != uint64(L+1) {
		t.Fatalf("page2[0].ID = %d, want %d (resume strictly after page1 last %d)", page2[0].ID, L+1, page1[len(page1)-1].ID)
	}
}

// TestEmbeddedMVScrollCursorFilterDeepPagination: filter + cursor returns only
// matching ids, ascending, exactly once across pages (filter applied during the
// partitioned scroll).
func TestEmbeddedMVScrollCursorFilterDeepPagination(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)

	const (
		coll = "mvfiltered"
		P    = 4
		N    = 200
		L    = 13
	)
	ids := shuffledIDs(N, 97)
	seedMV(t, s, coll, P, ids)

	evenFilter := VectorFilter{Op: vector.FilterEq, Field: "even", Value: vector.NewBool(true)}
	want := map[uint64]bool{}
	for _, id := range ids {
		if id%2 == 0 {
			want[id] = true
		}
	}

	got, _ := pageAllMV(t, s, coll, evenFilter, L)
	assertExactlyOnceAscending(t, got, want)
}

// TestEmbeddedMVScrollCursorUnpartitioned: same deep-pagination correctness on a
// single (P=1) MV collection — the unpartitioned embedded callReadLeader path.
func TestEmbeddedMVScrollCursorUnpartitioned(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)

	const (
		coll = "mvsingle"
		P    = 1
		N    = 180
		L    = 23
	)
	ids := shuffledIDs(N, 19)
	seedMV(t, s, coll, P, ids)

	want := map[uint64]bool{}
	for _, id := range ids {
		want[id] = true
	}
	got, _ := pageAllMV(t, s, coll, VectorFilter{}, L)
	assertExactlyOnceAscending(t, got, want)
}

// TestEmbeddedMVScrollNoCursorUnchanged: a no-cursor scroll with limit=L returns
// the SMALLEST-id L ids ascending (the documented deterministic-ascending
// behavior) — proves the no-rc/no-cursor path is the default.
func TestEmbeddedMVScrollNoCursorUnchanged(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	const (
		coll = "mvnocursor"
		P    = 4
		N    = 100
		L    = 17
	)
	seedMV(t, s, coll, P, shuffledIDs(N, 7))

	docs, _, _, err := s.VectorMVScroll(ctx, coll, VectorFilter{}, L, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != L {
		t.Fatalf("no-cursor limit=%d returned %d docs", L, len(docs))
	}
	for i := 0; i < L; i++ {
		if docs[i].ID != uint64(i+1) {
			t.Fatalf("no-cursor page[%d].ID = %d, want %d (smallest-id ascending)", i, docs[i].ID, i+1)
		}
	}
}

// TestEmbeddedMVScrollBadCursor: a malformed cursor token fails loud up front
// (before any dispatch), on both the partitioned and unpartitioned paths.
func TestEmbeddedMVScrollBadCursor(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	for _, P := range []int{1, 4} {
		coll := "mvbad"
		if P > 1 {
			coll = "mvbadp"
		}
		seedMV(t, s, coll, P, []uint64{1, 2, 3})
		if _, _, _, err := s.VectorMVScroll(ctx, coll, VectorFilter{}, 10, "!!!not-a-cursor!!!"); err == nil {
			t.Fatalf("P=%d: bad cursor accepted, want error", P)
		}
	}
}

// TestMVScrollFanOutRcRidesEveryArg is the anti-silent-drop gate: the per-partition
// arg that mvScrollFanOut emits (built by the SAME EncodeMVScrollArgsOpts call the
// fan-out Encode closure uses) must carry the Linearizable rc, so ReadConsistencyOf
// arms each shard's data barrier. If rc were dropped from the per-partition encode,
// ReadConsistencyOf would report AnyReplica and this test fails (RED-reproduces the
// silent-degrade hole). Mirrors the cursor re-encode the fan-out performs.
func TestMVScrollFanOutRcRidesEveryArg(t *testing.T) {
	filter := VectorFilter{Op: vector.FilterEq, Field: "even", Value: vector.NewBool(true)}
	const (
		limit   = 25
		afterID = uint64(7)
	)
	// The exact per-partition encode line from mvScrollFanOut (rc/opa + afterID ride
	// every arg). Linearizable rc must survive the encode → decode round-trip.
	arg := ops.EncodeMVScrollArgsOpts("phys#2", filter, limit,
		ops.ConsistencyLinearizable, 0 /*opa*/, afterID, true /*hasAfter*/)

	rc, ok := ops.ReadConsistencyOf("vector_mv_scroll", arg)
	if !ok {
		t.Fatalf("ReadConsistencyOf(vector_mv_scroll) not covered — Linearizable scroll would NOT arm the shard barrier")
	}
	if rc != ops.ConsistencyLinearizable {
		t.Fatalf("per-partition arg rc = %d, want Linearizable(%d) — rc dropped (silent-degrade)", rc, ops.ConsistencyLinearizable)
	}

	// The afterID + filter must also round-trip in the per-partition arg (the global
	// cursor + filter ride every partition).
	col, gotFilter, gotLimit, gotRC, gotOpa, gotAfter, gotHas, _, err := ops.DecodeMVScrollArgsOpts(arg)
	if err != nil {
		t.Fatalf("DecodeMVScrollArgsOpts: %v", err)
	}
	if col != "phys#2" || gotLimit != limit || gotRC != ops.ConsistencyLinearizable || gotOpa != 0 ||
		gotAfter != afterID || !gotHas || gotFilter.Field != "even" {
		t.Fatalf("per-partition arg round-trip mismatch: col=%q limit=%d rc=%d opa=%d after=%d has=%v field=%q",
			col, gotLimit, gotRC, gotOpa, gotAfter, gotHas, gotFilter.Field)
	}
}

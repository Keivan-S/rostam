// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"math/rand"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// pageAllDense pages a dense collection start-to-exhaustion via the cursor,
// returning the concatenated id sequence (in page order) and the page count. It
// asserts per-page invariants (ascending within a page, no cross-page descent,
// next_cursor empty IFF the last page is short) so a merge bug fails loud here.
func pageAllDense(t *testing.T, s Store, coll string, filter VectorFilter, limit int) (ids []uint64, pages int) {
	t.Helper()
	ctx := context.Background()
	cursor := ""
	var last uint64
	have := false
	for {
		docs, _, next, err := s.VectorScroll(ctx, coll, filter, limit, VectorScrollOpts{Cursor: cursor})
		if err != nil {
			t.Fatalf("VectorScroll page %d: %v", pages, err)
		}
		pages++
		for i, d := range docs {
			if i > 0 && d.ID <= docs[i-1].ID {
				t.Fatalf("page %d not strictly ascending at %d: %d <= %d", pages, i, d.ID, docs[i-1].ID)
			}
			if have && d.ID <= last {
				t.Fatalf("page %d id %d not > previous page's last %d (gap/dup/order bug)", pages, d.ID, last)
			}
			ids = append(ids, d.ID)
			last = d.ID
			have = true
		}
		// Exhaustion rule: a full page (len==limit) must carry a next_cursor; a
		// short/empty page must end pagination.
		if len(docs) == limit {
			if next == "" {
				t.Fatalf("page %d full (len=%d) but next_cursor empty", pages, limit)
			}
		} else if next != "" {
			t.Fatalf("page %d short (len=%d<%d) but next_cursor=%q (not exhausted)", pages, len(docs), limit, next)
		}
		if next == "" {
			return ids, pages
		}
		cursor = next
		if pages > limit*1000+100 { // runaway guard
			t.Fatalf("pagination did not terminate after %d pages", pages)
		}
	}
}

// assertExactlyOnceAscending asserts the paged id sequence is globally strictly
// ascending and equals want exactly once each (no gaps, no dups).
func assertExactlyOnceAscending(t *testing.T, got []uint64, want map[uint64]bool) {
	t.Helper()
	for i := 1; i < len(got); i++ {
		if got[i] <= got[i-1] {
			t.Fatalf("not globally ascending at %d: %d <= %d", i, got[i], got[i-1])
		}
	}
	seen := make(map[uint64]int, len(got))
	for _, id := range got {
		seen[id]++
	}
	for id := range want {
		if seen[id] != 1 {
			t.Fatalf("id %d appeared %d times across pages, want exactly 1", id, seen[id])
		}
	}
	for id, n := range seen {
		if !want[id] {
			t.Fatalf("unexpected id %d (×%d) not in the want set", id, n)
		}
	}
	if len(got) != len(want) {
		t.Fatalf("total paged %d ids, want %d", len(got), len(want))
	}
}

// seedDense creates a partitioned dense collection and inserts ids in RANDOM
// order (so a stable global ascending result proves the merge, not insert order),
// each tagged with even/odd for the filter test.
func seedDense(t *testing.T, s Store, coll string, P int, ids []uint64) {
	t.Helper()
	ctx := context.Background()
	if err := s.CreateCollection(ctx, coll, VectorConfig{
		Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2, Partitions: P,
	}); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	for _, id := range ids {
		v := []float32{float32(id), 0, 0, 0}
		md := VectorMetadata{"even": vector.NewBool(id%2 == 0)}
		if err := s.VectorInsertExt(ctx, coll, id, v, VectorInsertOpts{Metadata: md}); err != nil {
			t.Fatalf("VectorInsertExt %d: %v", id, err)
		}
	}
}

func shuffledIDs(n int, seed int64) []uint64 {
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1) // distinct, tie-free, starting at 1
	}
	r := rand.New(rand.NewSource(seed))
	r.Shuffle(len(ids), func(i, j int) { ids[i], ids[j] = ids[j], ids[i] })
	return ids
}

// TestEmbeddedScrollCursorDeepPaginationPartitioned is the core correctness test:
// a P=4 collection seeded with N distinct ids in random order, paged with limit L
// from cursor="" to exhaustion. Every id must appear EXACTLY once, globally
// ascending across page boundaries; page count ≈ ceil(N/L).
func TestEmbeddedScrollCursorDeepPaginationPartitioned(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)

	const (
		coll = "deep"
		P    = 4
		N    = 250
		L    = 30
	)
	ids := shuffledIDs(N, 42)
	seedDense(t, s, coll, P, ids)

	want := map[uint64]bool{}
	for _, id := range ids {
		want[id] = true
	}

	got, pages := pageAllDense(t, s, coll, VectorFilter{}, L)
	assertExactlyOnceAscending(t, got, want)

	wantPages := (N + L - 1) / L // ceil(N/L)
	// A final short/empty page may add one; accept wantPages or wantPages+1.
	if pages != wantPages && pages != wantPages+1 {
		t.Fatalf("page count = %d, want %d or %d (ceil(N/L))", pages, wantPages, wantPages+1)
	}
}

// TestEmbeddedScrollCursorNoCursorSmallestL: a no-cursor scroll with limit=L
// returns the SMALLEST-id L ids ascending (the documented deterministic-ascending
// behavior).
func TestEmbeddedScrollCursorNoCursorSmallestL(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	const (
		coll = "smallest"
		P    = 4
		N    = 100
		L    = 17
	)
	seedDense(t, s, coll, P, shuffledIDs(N, 7))

	docs, _, _, err := s.VectorScroll(ctx, coll, VectorFilter{}, L, VectorScrollOpts{})
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

// TestEmbeddedScrollCursorFilterDeepPagination: filter + cursor returns only
// matching ids, ascending, exactly once across pages.
func TestEmbeddedScrollCursorFilterDeepPagination(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)

	const (
		coll = "filtered"
		P    = 4
		N    = 200
		L    = 13
	)
	ids := shuffledIDs(N, 99)
	seedDense(t, s, coll, P, ids)

	evenFilter := VectorFilter{Op: vector.FilterEq, Field: "even", Value: vector.NewBool(true)}
	want := map[uint64]bool{}
	for _, id := range ids {
		if id%2 == 0 {
			want[id] = true
		}
	}

	got, _ := pageAllDense(t, s, coll, evenFilter, L)
	assertExactlyOnceAscending(t, got, want)
}

// TestEmbeddedScrollCursorDeleteMidPagination: deleting a not-yet-paged id
// mid-scroll removes it from later pages with no gap; inserting a new id beyond
// the current cursor surfaces it in a later page.
func TestEmbeddedScrollCursorDeleteAndInsertMidPagination(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	const (
		coll = "mutate"
		P    = 4
		N    = 120
		L    = 20
	)
	ids := shuffledIDs(N, 5)
	seedDense(t, s, coll, P, ids)

	// Page once to advance the cursor partway, then mutate ids beyond the cursor.
	docs, _, next, err := s.VectorScroll(ctx, coll, VectorFilter{}, L, VectorScrollOpts{})
	if err != nil {
		t.Fatal(err)
	}
	cursorMax := docs[len(docs)-1].ID // largest id seen so far (< N comfortably)
	if cursorMax >= N-2 {
		t.Fatalf("first page advanced too far (cursorMax=%d, N=%d); cannot test mid-pagination", cursorMax, N)
	}

	// Pick a not-yet-paged id to DELETE (the smallest id strictly beyond the
	// cursor; ids 1..N all exist, so cursorMax+1 is guaranteed live and unpaged).
	delID := cursorMax + 1
	if _, err := s.VectorDelete(ctx, coll, delID); err != nil {
		t.Fatalf("delete %d: %v", delID, err)
	}
	newID := uint64(N + 50) // strictly greater than every existing id and the cursor
	if err := s.VectorInsertExt(ctx, coll, newID, []float32{float32(newID), 0, 0, 0},
		VectorInsertOpts{Metadata: VectorMetadata{"even": vector.NewBool(newID%2 == 0)}}); err != nil {
		t.Fatalf("insert %d: %v", newID, err)
	}

	// Continue paging from the cursor to exhaustion; collect the remaining ids.
	cursor := next
	rest := []uint64{}
	for cursor != "" {
		pg, _, nxt, err := s.VectorScroll(ctx, coll, VectorFilter{}, L, VectorScrollOpts{Cursor: cursor})
		if err != nil {
			t.Fatalf("scroll: %v", err)
		}
		for _, d := range pg {
			rest = append(rest, d.ID)
		}
		cursor = nxt
	}

	restSet := map[uint64]bool{}
	for _, id := range rest {
		restSet[id] = true
	}
	if restSet[delID] {
		t.Fatalf("deleted id %d still appeared in a later page", delID)
	}
	if !restSet[newID] {
		t.Fatalf("inserted id %d (> cursor) did not appear in a later page", newID)
	}
	// No gap: the remaining ids must be exactly the live ids > cursorMax (original
	// ids in (cursorMax, N] minus delID, plus newID), each once, ascending.
	wantRest := map[uint64]bool{}
	for _, id := range ids {
		if id > cursorMax && id != delID {
			wantRest[id] = true
		}
	}
	wantRest[newID] = true
	assertExactlyOnceAscending(t, append([]uint64{}, rest...), wantRest)
}

// TestEmbeddedScrollCursorExhaustion: the final page returns < L (or empty) and
// next_cursor=="". Uses N a multiple of L to also exercise the empty-final-page case.
func TestEmbeddedScrollCursorExhaustion(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)

	const (
		coll = "exhaust"
		P    = 4
		L    = 25
		N    = 100 // exact multiple of L => a trailing empty page is allowed
	)
	seedDense(t, s, coll, P, shuffledIDs(N, 3))

	got, _ := pageAllDense(t, s, coll, VectorFilter{}, L)
	want := map[uint64]bool{}
	for id := uint64(1); id <= N; id++ {
		want[id] = true
	}
	assertExactlyOnceAscending(t, got, want)
}

// TestEmbeddedScrollCursorUnpartitioned: same deep-pagination correctness on a
// single (P=1) collection — the unpartitioned embedded path.
func TestEmbeddedScrollCursorUnpartitioned(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)

	const (
		coll = "single"
		P    = 1
		N    = 180
		L    = 23
	)
	ids := shuffledIDs(N, 17)
	seedDense(t, s, coll, P, ids)

	want := map[uint64]bool{}
	for _, id := range ids {
		want[id] = true
	}
	got, _ := pageAllDense(t, s, coll, VectorFilter{}, L)
	assertExactlyOnceAscending(t, got, want)
}

// pageAllNamed is pageAllDense for the named family.
func pageAllNamed(t *testing.T, s Store, coll string, filter VectorFilter, limit int) (ids []uint64) {
	t.Helper()
	ctx := context.Background()
	cursor := ""
	var last uint64
	have := false
	pages := 0
	for {
		docs, next, err := s.VectorNamedScroll(ctx, coll, filter, limit, cursor)
		if err != nil {
			t.Fatalf("VectorNamedScroll page %d: %v", pages, err)
		}
		pages++
		for i, d := range docs {
			if i > 0 && d.ID <= docs[i-1].ID {
				t.Fatalf("named page %d not ascending at %d", pages, i)
			}
			if have && d.ID <= last {
				t.Fatalf("named page %d id %d not > previous last %d", pages, d.ID, last)
			}
			ids = append(ids, d.ID)
			last = d.ID
			have = true
		}
		if len(docs) == limit {
			if next == "" {
				t.Fatalf("named page %d full but next_cursor empty", pages)
			}
		} else if next != "" {
			t.Fatalf("named page %d short but next_cursor=%q", pages, next)
		}
		if next == "" {
			return ids
		}
		cursor = next
		if pages > limit*1000+100 {
			t.Fatalf("named pagination did not terminate")
		}
	}
}

// TestEmbeddedNamedScrollCursorDeepPaginationPartitioned: the named family's
// partitioned deep-pagination exactly-once test.
func TestEmbeddedNamedScrollCursorDeepPaginationPartitioned(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	const (
		coll = "ndeep"
		P    = 4
		N    = 250
		L    = 30
	)
	cfg := map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: vector.Cosine},
	}
	if err := s.VectorNamedCreateCollection(ctx, coll, cfg, P); err != nil {
		t.Fatalf("VectorNamedCreateCollection: %v", err)
	}
	ids := shuffledIDs(N, 71)
	want := map[uint64]bool{}
	for _, id := range ids {
		if err := s.VectorNamedInsert(ctx, coll, id,
			map[string][]float32{"title": {float32(id), 0, 0, 0}},
			vector.Metadata{"even": vector.NewBool(id%2 == 0)}, 0); err != nil {
			t.Fatalf("VectorNamedInsert %d: %v", id, err)
		}
		want[id] = true
	}

	got := pageAllNamed(t, s, coll, VectorFilter{}, L)
	assertExactlyOnceAscending(t, got, want)
}

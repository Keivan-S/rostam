// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"sort"
	"sync"
	"testing"
	"time"
)

// newScrollHNSW builds a small L2 index for scroll-cursor tests.
func newScrollHNSW(t *testing.T) *hnsw {
	t.Helper()
	h, err := newHNSW(Config{Dim: 4, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 7, Metric: L2})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	return h
}

// insertScroll inserts id with a deterministic vector and the given metadata.
func insertScroll(t *testing.T, h *hnsw, id uint64, meta Metadata) {
	t.Helper()
	v := []float32{float32(id), float32(id % 7), float32(id % 3), 1}
	if _, _, err := h.Insert(id, v, 0, meta, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert %d: %v", id, err)
	}
}

func docIDs(docs []Document) []uint64 {
	out := make([]uint64, len(docs))
	for i, d := range docs {
		out[i] = d.ID
	}
	return out
}

func isAscending(ids []uint64) bool {
	for i := 1; i < len(ids); i++ {
		if ids[i] <= ids[i-1] {
			return false
		}
	}
	return true
}

// TestScrollAscendingOrder: ids inserted in random order scroll back ascending.
func TestScrollAscendingOrder(t *testing.T) {
	h := newScrollHNSW(t)
	ids := []uint64{50, 3, 17, 0, 999, 42, 7, 1, 256, 128}
	for _, id := range ids {
		insertScroll(t, h, id, nil)
	}
	docs, err := h.scrollDocs(Filter{}, 0)
	if err != nil {
		t.Fatalf("scrollDocs: %v", err)
	}
	got := docIDs(docs)
	if len(got) != len(ids) {
		t.Fatalf("scroll returned %d docs, want %d", len(got), len(ids))
	}
	if !isAscending(got) {
		t.Fatalf("scroll order not ascending: %v", got)
	}
	want := append([]uint64(nil), ids...)
	sort.Slice(want, func(i, j int) bool { return want[i] < want[j] })
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("scroll[%d] = %d, want %d (full %v)", i, got[i], want[i], got)
		}
	}
}

// TestScrollIncludesIDZero: id 0 is a valid scroll id (only ANN excludes it).
func TestScrollIncludesIDZero(t *testing.T) {
	h := newScrollHNSW(t)
	for _, id := range []uint64{0, 1, 2} {
		insertScroll(t, h, id, nil)
	}
	docs, _, _ := h.scrollPage(Filter{}, nil, nil, nil, 0, 0, false, 0)
	if len(docs) != 3 || docs[0].ID != 0 {
		t.Fatalf("first page = %v, want [0 1 2] with id 0 first", docIDs(docs))
	}
	// With afterID=0, hasAfter=true: id 0 excluded.
	docs2, _, _ := h.scrollPage(Filter{}, nil, nil, nil, 0, 0, true, 0)
	got := docIDs(docs2)
	if len(got) != 2 || got[0] != 1 {
		t.Fatalf("after id 0 = %v, want [1 2] (id 0 excluded)", got)
	}
}

// TestScrollSnapshotReuse: scrolling twice with no write between rebuilds the
// snapshot exactly once; an insert/delete bumps the version (rebuild) but a
// payload update does NOT.
func TestScrollSnapshotReuse(t *testing.T) {
	h := newScrollHNSW(t)
	for _, id := range []uint64{1, 2, 3} {
		insertScroll(t, h, id, Metadata{"k": NewString("v")})
	}

	if _, err := h.scrollDocs(Filter{}, 0); err != nil {
		t.Fatalf("scroll 1: %v", err)
	}
	rebuilds1 := h.scrollRebuilds
	if _, err := h.scrollDocs(Filter{}, 0); err != nil {
		t.Fatalf("scroll 2: %v", err)
	}
	if h.scrollRebuilds != rebuilds1 {
		t.Fatalf("snapshot rebuilt on a no-write second scroll: rebuilds %d -> %d", rebuilds1, h.scrollRebuilds)
	}
	verAfterScroll := h.idSetVersion

	// Payload update must NOT bump the id-set version (so the snapshot stays valid).
	if _, _, _, err := h.SetPayload(1, Metadata{"k": NewString("v2")}, nil, CASCond{}); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}
	if _, _, _, err := h.OverwritePayload(2, Metadata{"x": NewString("y")}, nil, CASCond{}); err != nil {
		t.Fatalf("OverwritePayload: %v", err)
	}
	if _, _, _, err := h.DeletePayloadKeys(3, []string{"k"}, CASCond{}); err != nil {
		t.Fatalf("DeletePayloadKeys: %v", err)
	}
	if _, _, _, err := h.ClearPayload(1, CASCond{}); err != nil {
		t.Fatalf("ClearPayload: %v", err)
	}
	if h.idSetVersion != verAfterScroll {
		t.Fatalf("payload updates bumped idSetVersion: %d -> %d (must NOT bump)", verAfterScroll, h.idSetVersion)
	}
	rebuildsBeforeScroll := h.scrollRebuilds
	if _, err := h.scrollDocs(Filter{}, 0); err != nil {
		t.Fatalf("scroll after payload: %v", err)
	}
	if h.scrollRebuilds != rebuildsBeforeScroll {
		t.Fatalf("snapshot rebuilt after a payload-only update (should reuse): %d -> %d", rebuildsBeforeScroll, h.scrollRebuilds)
	}

	// Insert a NEW id ⇒ version bumps ⇒ snapshot rebuilds ⇒ scroll includes it.
	insertScroll(t, h, 4, nil)
	if h.idSetVersion == verAfterScroll {
		t.Fatalf("insert of a new id did not bump idSetVersion")
	}
	docs, _, _ := h.scrollPage(Filter{}, nil, nil, nil, 0, 0, false, 0)
	got := docIDs(docs)
	if len(got) != 4 || got[3] != 4 || !isAscending(got) {
		t.Fatalf("after insert id 4: %v, want [1 2 3 4]", got)
	}

	// Delete an id ⇒ version bumps ⇒ gone.
	verBeforeDelete := h.idSetVersion
	if ok, _ := h.Delete(2, CASCond{}); !ok {
		t.Fatalf("Delete(2) returned false")
	}
	if h.idSetVersion == verBeforeDelete {
		t.Fatalf("Delete did not bump idSetVersion")
	}
	docs2, _, _ := h.scrollPage(Filter{}, nil, nil, nil, 0, 0, false, 0)
	got2 := docIDs(docs2)
	if len(got2) != 3 {
		t.Fatalf("after delete id 2: %v, want 3 docs", got2)
	}
	for _, id := range got2 {
		if id == 2 {
			t.Fatalf("deleted id 2 still present: %v", got2)
		}
	}
}

// TestScrollPageCursorPagination: paging with limit=L through N ids yields every
// id exactly once, ascending, no gaps/dups, exhaustion reports hasMore=false.
func TestScrollPageCursorPagination(t *testing.T) {
	h := newScrollHNSW(t)
	const n = 100
	ids := make([]uint64, n)
	for i := 0; i < n; i++ {
		ids[i] = uint64(i) * 3 // sparse, includes 0
	}
	shuffled := append([]uint64(nil), ids...)
	rng := rand.New(rand.NewSource(1))
	rng.Shuffle(len(shuffled), func(i, j int) { shuffled[i], shuffled[j] = shuffled[j], shuffled[i] })
	for _, id := range shuffled {
		insertScroll(t, h, id, nil)
	}

	const limit = 7
	var collected []uint64
	var afterID uint64
	hasAfter := false
	pages := 0
	for {
		pages++
		if pages > n+5 {
			t.Fatalf("pagination did not terminate")
		}
		docs, nextAfter, hasMore := h.scrollPage(Filter{}, nil, nil, nil, afterID, 0, hasAfter, limit)
		got := docIDs(docs)
		if !isAscending(got) {
			t.Fatalf("page not ascending: %v", got)
		}
		collected = append(collected, got...)
		if !hasMore {
			break
		}
		if len(got) != limit {
			t.Fatalf("hasMore=true but page len %d != limit %d", len(got), limit)
		}
		afterID = nextAfter
		hasAfter = true
	}
	if len(collected) != n {
		t.Fatalf("collected %d ids across pages, want %d", len(collected), n)
	}
	if !isAscending(collected) {
		t.Fatalf("collected ids not globally ascending: %v", collected)
	}
	for i := range ids {
		if collected[i] != ids[i] {
			t.Fatalf("collected[%d] = %d, want %d", i, collected[i], ids[i])
		}
	}
}

// TestScrollPageExhaustion: a page that walks to the end without filling limit
// reports hasMore=false.
func TestScrollPageExhaustion(t *testing.T) {
	h := newScrollHNSW(t)
	for _, id := range []uint64{1, 2, 3} {
		insertScroll(t, h, id, nil)
	}
	docs, nextAfter, hasMore := h.scrollPage(Filter{}, nil, nil, nil, 0, 0, false, 10)
	if len(docs) != 3 {
		t.Fatalf("got %d docs, want 3", len(docs))
	}
	if hasMore {
		t.Fatalf("hasMore=true on an under-limit page (exhausted), want false")
	}
	if nextAfter != 3 {
		t.Fatalf("nextAfter = %d, want 3 (last collected)", nextAfter)
	}
}

// TestScrollPageFilterAndCursor: a selective filter combined with a cursor walks
// forward correctly, returning only matching ids ascending.
func TestScrollPageFilterAndCursor(t *testing.T) {
	h := newScrollHNSW(t)
	const n = 60
	var want []uint64
	for i := uint64(0); i < n; i++ {
		kind := "odd"
		if i%2 == 0 {
			kind = "even"
			want = append(want, i)
		}
		insertScroll(t, h, i, Metadata{"kind": NewString(kind)})
	}
	evenFilter := Filter{Op: FilterEq, Field: "kind", Value: NewString("even")}
	pred, err := evenFilter.Compile()
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	const limit = 5
	var collected []uint64
	var afterID uint64
	hasAfter := false
	for {
		docs, nextAfter, hasMore := h.scrollPage(evenFilter, pred, nil, nil, afterID, 0, hasAfter, limit)
		collected = append(collected, docIDs(docs)...)
		if !hasMore {
			break
		}
		afterID = nextAfter
		hasAfter = true
	}
	if len(collected) != len(want) {
		t.Fatalf("filtered+cursor collected %d, want %d", len(collected), len(want))
	}
	for i := range want {
		if collected[i] != want[i] {
			t.Fatalf("filtered[%d] = %d, want %d", i, collected[i], want[i])
		}
	}
}

// TestScrollPageDeleteBetweenPages: deleting a not-yet-paged id between pages
// makes it absent with no gap/dup.
func TestScrollPageDeleteBetweenPages(t *testing.T) {
	h := newScrollHNSW(t)
	const n = 20
	for i := uint64(0); i < n; i++ {
		insertScroll(t, h, i, nil)
	}
	const limit = 5
	// Page 1: ids 0..4.
	docs, nextAfter, hasMore := h.scrollPage(Filter{}, nil, nil, nil, 0, 0, false, limit)
	if !hasMore || docIDs(docs)[0] != 0 {
		t.Fatalf("page 1 unexpected: %v hasMore=%v", docIDs(docs), hasMore)
	}
	collected := docIDs(docs)
	// Delete an id NOT yet paged (id 12, > nextAfter=4).
	if ok, _ := h.Delete(12, CASCond{}); !ok {
		t.Fatalf("Delete(12) false")
	}
	afterID := nextAfter
	for {
		docs, nextAfter, hasMore = h.scrollPage(Filter{}, nil, nil, nil, afterID, 0, true, limit)
		collected = append(collected, docIDs(docs)...)
		if !hasMore {
			break
		}
		afterID = nextAfter
	}
	// 12 must be absent; all others present exactly once.
	seen := map[uint64]int{}
	for _, id := range collected {
		seen[id]++
	}
	if seen[12] != 0 {
		t.Fatalf("deleted id 12 appeared %d times", seen[12])
	}
	for i := uint64(0); i < n; i++ {
		if i == 12 {
			continue
		}
		if seen[i] != 1 {
			t.Fatalf("id %d seen %d times, want 1", i, seen[i])
		}
	}
	if !isAscending(collected) {
		t.Fatalf("collected not ascending: %v", collected)
	}
}

// TestScrollPageRaceConcurrentInsert: concurrent inserts + scrollPage must be
// race-free and panic-free (run under -race).
func TestScrollPageRaceConcurrentInsert(t *testing.T) {
	h := newScrollHNSW(t)
	for i := uint64(0); i < 50; i++ {
		insertScroll(t, h, i, nil)
	}
	var wg sync.WaitGroup
	stop := make(chan struct{})

	// Writers: insert new ids and delete some.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := uint64(50); i < 250; i++ {
			select {
			case <-stop:
				return
			default:
			}
			v := []float32{float32(i), 0, 0, 1}
			_, _, _ = h.Insert(i, v, 0, nil, nil, nil, CASCond{})
			if i%5 == 0 {
				h.Delete(i-50, CASCond{})
			}
		}
	}()

	// Readers: page through repeatedly.
	for r := 0; r < 4; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 200; j++ {
				var afterID uint64
				hasAfter := false
				for {
					docs, nextAfter, hasMore := h.scrollPage(Filter{}, nil, nil, nil, afterID, 0, hasAfter, 8)
					if !isAscending(docIDs(docs)) {
						t.Errorf("concurrent page not ascending: %v", docIDs(docs))
						return
					}
					if !hasMore {
						break
					}
					afterID = nextAfter
					hasAfter = true
				}
			}
		}()
	}
	time.Sleep(50 * time.Millisecond)
	close(stop)
	wg.Wait()
}

// TestNamedScrollAscending: a named collection scrolls ascending.
func TestNamedScrollAscending(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("NewNamedCollection: %v", err)
	}
	ids := []uint64{30, 5, 0, 17, 2, 99}
	for _, id := range ids {
		if err := nc.Insert(id, map[string][]float32{
			"title": {1, 0, 0, 0},
			"image": {0, 1, 0},
		}, Metadata{"g": NewString("x")}, 0); err != nil {
			t.Fatalf("named Insert %d: %v", id, err)
		}
	}
	docs, err := nc.ScrollDocs(Filter{}, 0)
	if err != nil {
		t.Fatalf("named ScrollDocs: %v", err)
	}
	got := docIDs(docs)
	if len(got) != len(ids) || !isAscending(got) {
		t.Fatalf("named scroll = %v, want ascending of %v", got, ids)
	}
	want := append([]uint64(nil), ids...)
	sort.Slice(want, func(i, j int) bool { return want[i] < want[j] })
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("named scroll[%d] = %d, want %d", i, got[i], want[i])
		}
	}

	// Named cursor seek: afterID excludes <= afterID.
	docs2, _, _ := nc.scrollPage(Filter{}, nil, nil, 5, 0, true, 0)
	got2 := docIDs(docs2)
	for _, id := range got2 {
		if id <= 5 {
			t.Fatalf("named after id 5 contains %d (<=5): %v", id, got2)
		}
	}
	if !isAscending(got2) {
		t.Fatalf("named cursor page not ascending: %v", got2)
	}
}

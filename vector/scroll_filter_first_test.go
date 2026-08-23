// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"sort"
	"sync"
	"testing"
)

// Scroll filter-first narrowing oracle tests (all 4 families).
//
// The load-bearing invariant: a scroll driven through the index-accelerated
// filter-first path (a selective Eq / range / In filter) must yield the EXACT SAME
// full-pagination result (ids, order, page boundaries, exhaustion, cursors) as the
// SAME logical scroll forced through the predicate-eval walk (an equivalent
// NON-accelerable filter, e.g. Ne / Not / regex, for which candidates() returns
// ok=false ⇒ the full-snapshot fallback). Filter-first only changes HOW candidates
// are discovered (the payload-index superset), never WHICH docs match (the existing
// per-candidate predicate recheck still runs), so the two pages are identical.

// pageRecord captures one page's emitted ids + its (nextAfter, hasMore) cursor, so a
// full-pagination comparison checks not just the concatenated ids but the page
// boundaries and cursor sequence too.
type pageRecord struct {
	ids       []uint64
	nextAfter uint64
	hasMore   bool
}

// scrollFn paginates one filtered scroll. Returns the per-page records.
type scrollFn func(filter Filter, afterID uint64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool)

func paginateAll(scroll scrollFn, filter Filter, limit int) []pageRecord {
	var pages []pageRecord
	var afterID uint64
	hasAfter := false
	for {
		docs, nextAfter, hasMore := scroll(filter, afterID, hasAfter, limit)
		pages = append(pages, pageRecord{ids: docIDs(docs), nextAfter: nextAfter, hasMore: hasMore})
		if !hasMore {
			break
		}
		afterID = nextAfter
		hasAfter = true
	}
	return pages
}

func samePages(t *testing.T, label string, a, b []pageRecord) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("%s: page count %d != %d\n filter-first=%v\n pred-eval=%v", label, len(a), len(b), a, b)
	}
	for i := range a {
		if a[i].hasMore != b[i].hasMore || a[i].nextAfter != b[i].nextAfter {
			t.Fatalf("%s: page %d cursor mismatch: filter-first (next=%d more=%v) != pred-eval (next=%d more=%v)",
				label, i, a[i].nextAfter, a[i].hasMore, b[i].nextAfter, b[i].hasMore)
		}
		if len(a[i].ids) != len(b[i].ids) {
			t.Fatalf("%s: page %d size %d != %d: ff=%v pe=%v", label, i, len(a[i].ids), len(b[i].ids), a[i].ids, b[i].ids)
		}
		for j := range a[i].ids {
			if a[i].ids[j] != b[i].ids[j] {
				t.Fatalf("%s: page %d id[%d] = %d != %d\n ff=%v\n pe=%v", label, i, j, a[i].ids[j], b[i].ids[j], a[i].ids, b[i].ids)
			}
		}
	}
}

// concatPages flattens all page ids into one slice (the full scroll result).
func concatPages(pages []pageRecord) []uint64 {
	var out []uint64
	for _, p := range pages {
		out = append(out, p.ids...)
	}
	return out
}

// --- per-family scrollFn adapters ---

func hnswScrollFn(h *hnsw) scrollFn {
	return func(filter Filter, afterID uint64, hasAfter bool, limit int) ([]Document, uint64, bool) {
		pred, err := CompileFilter(filter)
		if err != nil {
			panic(err)
		}
		return h.scrollPage(filter, pred, nil, nil, afterID, 0, hasAfter, limit)
	}
}

func ivfScrollFn(ix *ivf) scrollFn {
	return func(filter Filter, afterID uint64, hasAfter bool, limit int) ([]Document, uint64, bool) {
		pred, err := CompileFilter(filter)
		if err != nil {
			panic(err)
		}
		return ix.scrollPage(filter, pred, nil, nil, afterID, 0, hasAfter, limit)
	}
}

func namedScrollFn(nc *NamedCollection) scrollFn {
	return func(filter Filter, afterID uint64, hasAfter bool, limit int) ([]Document, uint64, bool) {
		docs, na, hm, err := nc.ScrollDocsPage(filter, afterID, hasAfter, limit)
		if err != nil {
			panic(err)
		}
		return docs, na, hm
	}
}

func mvScrollFn(m *MultiVectorIndex) scrollFn {
	return func(filter Filter, afterID uint64, hasAfter bool, limit int) ([]Document, uint64, bool) {
		docs, na, hm, err := m.ScrollDocsPage(filter, afterID, hasAfter, limit)
		if err != nil {
			panic(err)
		}
		return docs, na, hm
	}
}

// scrollDataset: a deterministic id set with a "kind" (even/odd), a numeric "n", and
// a "bucket" In-able field. Ids are inserted in scrambled order to prove the scroll
// (and the filter-first candidate sort) re-establish ascending id order.
type scrollPoint struct {
	id     uint64
	kind   string
	n      int64
	bucket int64
}

func scrollDataset(n uint64) []scrollPoint {
	pts := make([]scrollPoint, 0, n)
	for i := uint64(0); i < n; i++ {
		kind := "odd"
		if i%2 == 0 {
			kind = "even"
		}
		pts = append(pts, scrollPoint{id: i, kind: kind, n: int64(i), bucket: int64(i % 5)})
	}
	// scramble insertion order deterministically
	for i := range pts {
		j := (i*7 + 3) % len(pts)
		pts[i], pts[j] = pts[j], pts[i]
	}
	return pts
}

func ptMeta(p scrollPoint) Metadata {
	return Metadata{
		"kind":   NewString(p.kind),
		"n":      NewInt(p.n),
		"bucket": NewInt(p.bucket),
	}
}

// acceleratedFilter (Eq kind=even) and equivalentFallback (Ne kind=odd) select the
// SAME docs; the first narrows via the payload index (filter-first), the second is
// non-accelerable so candidates() returns ok=false (predicate-eval fallback).
var acceleratedEq = Filter{Op: FilterEq, Field: "kind", Value: NewString("even")}
var fallbackNe = Filter{Op: FilterNe, Field: "kind", Value: NewString("odd")}

// acceleratedRange (n >= 40) via the numeric posting list, and its fallback (Not(n < 40)).
var acceleratedRange = Filter{Op: FilterGte, Field: "n", Value: NewInt(40)}
var fallbackRange = Filter{Op: FilterNot, Not: &Filter{Op: FilterLt, Field: "n", Value: NewInt(40)}}

// acceleratedIn (bucket in {1,3}) and its fallback (Or(bucket==1, bucket==3) — Or is
// non-accelerable so it falls back).
var acceleratedIn = Filter{Op: FilterIn, Field: "bucket", Value: NewInts([]int64{1, 3})}
var fallbackIn = Filter{Op: FilterOr, Or: []Filter{
	{Op: FilterEq, Field: "bucket", Value: NewInt(1)},
	{Op: FilterEq, Field: "bucket", Value: NewInt(3)},
}}

// oracleFamily runs the full filter-first == predicate-eval oracle for one family
// across the three accelerable filter shapes (eq / range / In) at several page sizes,
// plus the no-filter byte-identical check and the gap/dup-free check.
func oracleFamily(t *testing.T, name string, scroll scrollFn, total uint64) {
	t.Helper()
	cases := []struct {
		label   string
		accel   Filter
		fallbk  Filter
		matches func(scrollPoint) bool
	}{
		{"eq", acceleratedEq, fallbackNe, func(p scrollPoint) bool { return p.kind == "even" }},
		{"range", acceleratedRange, fallbackRange, func(p scrollPoint) bool { return p.n >= 40 }},
		{"in", acceleratedIn, fallbackIn, func(p scrollPoint) bool { return p.bucket == 1 || p.bucket == 3 }},
	}
	for _, c := range cases {
		for _, limit := range []int{1, 3, 7, 0} {
			ff := paginateAll(scroll, c.accel, limit)
			pe := paginateAll(scroll, c.fallbk, limit)
			samePages(t, fmt.Sprintf("%s/%s/limit=%d", name, c.label, limit), ff, pe)

			// Independently verify the result == the expected matching set, ascending,
			// gap-free + dup-free (every match exactly once).
			got := concatPages(ff)
			var want []uint64
			for i := uint64(0); i < total; i++ {
				if c.matches(scrollPoint{id: i, kind: kindOf(i), n: int64(i), bucket: int64(i % 5)}) {
					want = append(want, i)
				}
			}
			if !isAscending(got) && len(got) > 1 {
				t.Fatalf("%s/%s/limit=%d: result not ascending: %v", name, c.label, limit, got)
			}
			if len(got) != len(want) {
				t.Fatalf("%s/%s/limit=%d: got %d ids, want %d (gap/dup): got=%v want=%v", name, c.label, limit, len(got), len(want), got, want)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s/%s/limit=%d: id[%d]=%d want %d", name, c.label, limit, i, got[i], want[i])
				}
			}
		}
	}
	// No-filter scroll: filter-first never engages (zero filter → pred nil); just
	// confirm it returns every id ascending.
	all := concatPages(paginateAll(scroll, Filter{}, 4))
	if len(all) != int(total) || !isAscending(all) {
		t.Fatalf("%s/no-filter: got %d ascending=%v, want %d", name, len(all), isAscending(all), total)
	}
}

func kindOf(i uint64) string {
	if i%2 == 0 {
		return "even"
	}
	return "odd"
}

// TestScrollFilterFirstGate white-box-asserts the gate: an accelerable+selective
// filter engages filter-first (candidate superset returned, far smaller than N) while
// non-accelerable filters decline (fall back to the full predicate-eval walk). Proves
// filter-first touches ~len(cand) not ~N for a selective filter, and that the fallback
// is genuinely taken (not a silent always-on path).
func TestScrollFilterFirstGate(t *testing.T) {
	const total = 200
	h := newScrollHNSW(t)
	for _, p := range scrollDataset(total) {
		insertScroll(t, h, p.id, ptMeta(p))
	}
	pred, _ := CompileFilter(acceleratedEq)
	h.mu.RLock()
	cands, ok := h.filterFirstScrollCandsLocked(acceleratedEq, pred, nil)
	h.mu.RUnlock()
	if !ok {
		t.Fatalf("accelerable Eq filter should engage filter-first")
	}
	if len(cands) >= total {
		t.Fatalf("filter-first candidate set %d not narrowed below N=%d", len(cands), total)
	}
	if len(cands) != total/2 { // exactly the even docs
		t.Fatalf("filter-first candidate superset = %d, want %d (even docs)", len(cands), total/2)
	}
	if !isAscending(cands) {
		t.Fatalf("filter-first candidate set must be id-ascending: %v", cands)
	}
	// Non-accelerable filters decline.
	for _, f := range []Filter{fallbackNe, fallbackRange, fallbackIn,
		{Op: FilterRegex, Field: "kind", Value: NewString("^even$")}} {
		fp, _ := CompileFilter(f)
		h.mu.RLock()
		_, ok := h.filterFirstScrollCandsLocked(f, fp, nil)
		h.mu.RUnlock()
		if ok {
			t.Fatalf("non-accelerable filter %v should decline filter-first", f.Op)
		}
	}
	// The external-provider path always declines (empty sub-arena index).
	h.mu.RLock()
	_, okProv := h.filterFirstScrollCandsLocked(acceleratedEq, pred, func(uint64) Metadata { return nil })
	h.mu.RUnlock()
	if okProv {
		t.Fatalf("provider path (metaOf != nil) must decline filter-first")
	}
}

func TestScrollFilterFirstOracleHNSW(t *testing.T) {
	const total = 60
	h := newScrollHNSW(t)
	for _, p := range scrollDataset(total) {
		insertScroll(t, h, p.id, ptMeta(p))
	}
	oracleFamily(t, "hnsw", hnswScrollFn(h), total)
}

func TestScrollFilterFirstOracleIVF(t *testing.T) {
	const total = 60
	ix, err := newIVF(ivfTestConfig(4))
	if err != nil {
		t.Fatalf("newIVF: %v", err)
	}
	for _, p := range scrollDataset(total) {
		v := []float32{float32(p.id), float32(p.id % 7), float32(p.id % 3), 1}
		if _, _, err := ix.Insert(p.id, v, 0, ptMeta(p), nil, nil, CASCond{}); err != nil {
			t.Fatalf("ivf Insert %d: %v", p.id, err)
		}
	}
	oracleFamily(t, "ivf", ivfScrollFn(ix), total)
}

func TestScrollFilterFirstOracleNamed(t *testing.T) {
	const total = 60
	nc, err := NewNamedCollection("c", map[string]NamedVectorParams{
		"v": {Dim: 4, M: 16, EfConstruction: 200, EfSearch: 64, Metric: L2},
	})
	if err != nil {
		t.Fatalf("newNamedCollection: %v", err)
	}
	for _, p := range scrollDataset(total) {
		v := []float32{float32(p.id), float32(p.id % 7), float32(p.id % 3), 1}
		if err := nc.Insert(p.id, map[string][]float32{"v": v}, ptMeta(p), 0); err != nil {
			t.Fatalf("named Insert %d: %v", p.id, err)
		}
	}
	oracleFamily(t, "named", namedScrollFn(nc), total)
}

func TestScrollFilterFirstOracleMV(t *testing.T) {
	const total = 60
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("NewMultiVectorIndex: %v", err)
	}
	for _, p := range scrollDataset(total) {
		toks := [][]float32{{float32(p.id), float32(p.id % 7), float32(p.id % 3), 1}}
		if err := m.Add(p.id, toks, ptMeta(p)); err != nil {
			t.Fatalf("mv Add %d: %v", p.id, err)
		}
	}
	oracleFamily(t, "mv", mvScrollFn(m), total)
}

// TestScrollFilterFirstFallbackRegex: a regex filter (non-accelerable) falls back to
// the predicate-eval walk and is still correct (matches the regex-selected docs,
// ascending, gap/dup-free). Proves the fallback path is engaged + correct.
func TestScrollFilterFirstFallbackRegex(t *testing.T) {
	const total = 40
	h := newScrollHNSW(t)
	for _, p := range scrollDataset(total) {
		insertScroll(t, h, p.id, ptMeta(p))
	}
	// regex on kind matching "even"
	regex := Filter{Op: FilterRegex, Field: "kind", Value: NewString("^even$")}
	scroll := hnswScrollFn(h)
	got := concatPages(paginateAll(scroll, regex, 5))
	var want []uint64
	for i := uint64(0); i < total; i++ {
		if i%2 == 0 {
			want = append(want, i)
		}
	}
	if len(got) != len(want) || !isAscending(got) {
		t.Fatalf("regex fallback: got %v (ascending=%v), want %v", got, isAscending(got), want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("regex fallback id[%d]=%d want %d", i, got[i], want[i])
		}
	}
}

// TestScrollFilterFirstConcurrentMutation: a full paginated filter-first scroll is
// stable (gap-free + dup-free over the ids that exist for the whole scroll) under a
// concurrent insert/delete stream. The snapshot/cursor semantics (unchanged by
// filter-first) guarantee an id live for the entire scroll appears exactly once.
func TestScrollFilterFirstConcurrentMutation(t *testing.T) {
	const base = 200
	h := newScrollHNSW(t)
	// Even ids [0, base) are STABLE for the whole scroll (never mutated). Odd ids are
	// the churn set. The filter selects even (stable) docs, so every stable even id
	// must appear exactly once regardless of concurrent odd churn.
	for i := uint64(0); i < base; i++ {
		insertScroll(t, h, i, ptMeta(scrollPoint{id: i, kind: kindOf(i), n: int64(i), bucket: int64(i % 5)}))
	}
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		id := uint64(base)
		for {
			select {
			case <-stop:
				return
			default:
			}
			// churn ODD ids only (id is odd) so the even result set is untouched
			oddID := id*2 + 1
			insertScroll(t, h, oddID, ptMeta(scrollPoint{id: oddID, kind: "odd", n: int64(oddID), bucket: int64(oddID % 5)}))
			_, _ = h.Delete(oddID, CASCond{})
			id++
		}
	}()

	scroll := hnswScrollFn(h)
	got := concatPages(paginateAll(scroll, acceleratedEq, 4))
	close(stop)
	wg.Wait()

	seen := map[uint64]bool{}
	for _, id := range got {
		if seen[id] {
			t.Fatalf("duplicate id %d across pages: %v", id, got)
		}
		seen[id] = true
	}
	for i := uint64(0); i < base; i += 2 {
		if !seen[i] {
			t.Fatalf("stable even id %d missing from concurrent scroll (gap)", i)
		}
	}
}

// TestScrollFilterFirstOrderByResult: an order_by scroll WITH a (selective, accelerable)
// filter now narrows the ORDER-ROW build to the candidate superset (order_by filter-first)
// and returns the matching docs in value-sorted order — identical to the result computed
// directly. (The dedicated order-by filter-first oracle lives in order_filter_first_test.go;
// this is the focused dense-family result check, kept here next to the id-path oracle.)
func TestScrollFilterFirstOrderByResult(t *testing.T) {
	const total = 40
	h := newScrollHNSW(t)
	for _, p := range scrollDataset(total) {
		insertScroll(t, h, p.id, ptMeta(p))
	}
	order := &OrderBy{Key: "n", Desc: false}
	filter := acceleratedEq // engages order_by filter-first (narrows the order rows)
	pred, err := CompileFilter(filter)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	var got []uint64
	var afterID uint64
	var afterKey float64
	hasAfter := false
	for {
		docs, nextAfter, hasMore := h.scrollPage(filter, pred, nil, order, afterID, afterKey, hasAfter, 3)
		for _, d := range docs {
			got = append(got, d.ID)
			if v, ok := d.Metadata["n"]; ok {
				afterKey = float64(v.Int)
			}
		}
		if !hasMore {
			break
		}
		afterID = nextAfter
		hasAfter = true
	}
	// even ids ascending by n == even ids ascending by id
	var want []uint64
	for i := uint64(0); i < total; i += 2 {
		want = append(want, i)
	}
	sort.Slice(want, func(i, j int) bool { return want[i] < want[j] })
	if len(got) != len(want) {
		t.Fatalf("order_by filtered scroll got %d, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("order_by[%d]=%d want %d", i, got[i], want[i])
		}
	}
}

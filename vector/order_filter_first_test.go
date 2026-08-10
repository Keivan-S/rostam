// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"sort"
	"sync"
	"testing"
)

// order_by scroll filter-first oracle tests (all 4 families, single + multi key).
//
// The load-bearing invariant (the order_by analogue of scroll_filter_first_test.go):
// an order_by scroll driven through the index-accelerated filter-first ORDER-ROW build
// (a selective Eq / range / In filter narrows the value-sorted row set to candidates())
// must yield the EXACT SAME full-pagination DOC SEQUENCE, value-order, page sizes, and
// resume cursor as the SAME logical order_by scroll forced through the predicate-eval
// order build (an equivalent NON-accelerable filter, e.g. Ne / Not / Or, for which
// candidates() returns ok=false ⇒ the full-snapshot order build). Filter-first only
// changes WHICH docs the order rows are built FROM (the candidates superset ∩ live);
// the per-row field-presence EXCLUDE + the per-row predicate recheck still run, so the
// matched set, the value-order, and the (value,…,id) seek are identical.
//
// NOTE on hasMore: the engine's order_by hasMore is byte-identical to predicate-eval on
// every DOC-BEARING page; it can differ ONLY by a single TRAILING EMPTY page (the full
// build's i+1<len(N) is true when trailing rows exist that the filter rejects, whereas
// the narrowed build has no such trailing rows). That trailing empty page carries zero
// docs and is invisible on the wire (the leaf discards hasMore — ops/builtin.go; the
// coordinator derives next_cursor from len(docs)==limit — embedded.go). The oracle below
// therefore compares the OBSERVABLE surface (per-page ids + page sizes + nextAfter) and
// terminates pagination on the production rule (a short/empty page ends the scroll), so
// it asserts exactly the contract production depends on.

// orderScrollFn paginates one filtered+ordered scroll page (single or multi key). The
// engine resumes via afterID + the per-kind resume that rides the OrderBy (ResumeStr /
// ResumeKeys) which the caller stamps onto `order` before each page.
type orderScrollFn func(filter Filter, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) (docs []Document, nextAfter uint64, hasMore bool)

// orderPageRecord captures one order_by page's emitted ids + nextAfter (the observable
// page surface). hasMore is captured for the doc-bearing-page assertion but pagination
// terminates on page fullness (the production rule), not on hasMore.
type orderPageRecord struct {
	ids       []uint64
	nextAfter uint64
	hasMore   bool
}

// paginateOrder pages a single-key order_by scroll to exhaustion, rebuilding the v2/v3
// resume (afterKey for numeric/datetime, order.ResumeStr for string) from the last doc's
// order field each page. Terminates on a short page (len(docs) < limit) — the production
// termination rule — so a divergent trailing-empty hasMore never desynchronizes the two
// runs. limit must be > 0.
func paginateOrder(scroll orderScrollFn, filter Filter, base OrderBy, metas map[uint64]Metadata, limit int) []orderPageRecord {
	var pages []orderPageRecord
	var afterID uint64
	var afterKey float64
	hasAfter := false
	str := base.Kind == OrderString
	for {
		order := base
		if hasAfter && str {
			// String resume rides order.ResumeStr (set below from the prior page's last
			// doc); numeric/datetime resume rides the afterKey param.
			order.ResumeStr = base.ResumeStr
			order.HasResumeStr = true
		}
		docs, nextAfter, hasMore := scroll(filter, &order, afterID, afterKey, hasAfter, limit)
		pages = append(pages, orderPageRecord{ids: docIDs(docs), nextAfter: nextAfter, hasMore: hasMore})
		if len(docs) < limit { // short page ⇒ exhausted (production rule)
			break
		}
		afterID = nextAfter
		last := docs[len(docs)-1]
		m := metas[last.ID]
		if str {
			sk, _ := OrderStringKey(m, base.Key)
			base.ResumeStr = sk
		} else {
			k, _ := OrderKey(m, base.Key, base.IsDatetime)
			afterKey = k
		}
		hasAfter = true
	}
	return pages
}

// paginateOrderMulti pages a multi-key order_by scroll to exhaustion, rebuilding the v4
// resume TUPLE (ResumeKeys) from the last doc each page. Same production termination rule.
func paginateOrderMulti(scroll orderScrollFn, filter Filter, primary OrderBy, tail []OrderBy, metas map[uint64]Metadata, limit int) []orderPageRecord {
	keys := append([]OrderBy{withoutTail(primary)}, tail...)
	var pages []orderPageRecord
	var afterID uint64
	var resume []OrderVal
	hasAfter := false
	for {
		order := primary
		order.Tail = tail
		if hasAfter {
			order.ResumeKeys = resume
			order.HasResumeKeys = true
		}
		docs, nextAfter, hasMore := scroll(filter, &order, afterID, 0, hasAfter, limit)
		pages = append(pages, orderPageRecord{ids: docIDs(docs), nextAfter: nextAfter, hasMore: hasMore})
		if len(docs) < limit {
			break
		}
		afterID = nextAfter
		last := docs[len(docs)-1]
		vals, ok := orderTupleKeys(metas[last.ID], keys)
		if !ok {
			panic("last doc missing an order key")
		}
		resume = vals
		hasAfter = true
	}
	return pages
}

func sameOrderPages(t *testing.T, label string, ff, pe []orderPageRecord) {
	t.Helper()
	// Compare the OBSERVABLE surface: the concatenated doc sequence + per-page sizes +
	// nextAfter on every page that the production rule visits. (Both runs terminate on a
	// short page, so they visit the same pages.)
	if len(ff) != len(pe) {
		t.Fatalf("%s: page count %d != %d\n ff=%v\n pe=%v", label, len(ff), len(pe), ff, pe)
	}
	for i := range ff {
		if ff[i].nextAfter != pe[i].nextAfter {
			t.Fatalf("%s: page %d nextAfter %d != %d", label, i, ff[i].nextAfter, pe[i].nextAfter)
		}
		if len(ff[i].ids) != len(pe[i].ids) {
			t.Fatalf("%s: page %d size %d != %d\n ff=%v\n pe=%v", label, i, len(ff[i].ids), len(pe[i].ids), ff[i].ids, pe[i].ids)
		}
		for j := range ff[i].ids {
			if ff[i].ids[j] != pe[i].ids[j] {
				t.Fatalf("%s: page %d id[%d] = %d != %d\n ff=%v\n pe=%v", label, i, j, ff[i].ids[j], pe[i].ids[j], ff[i].ids, pe[i].ids)
			}
		}
		// hasMore is deliberately NOT asserted equal here: it is the ONE field that may
		// legitimately differ between filter-first and predicate-eval order builds, and
		// ONLY on a FULL page (len==limit) that is the terminal matching page — the full
		// build's i+1<len(N) is true when the snapshot has trailing rows the filter
		// rejects, while the narrowed build (candidates ∩ matches) has no such trailing
		// rows ⇒ false. That divergence yields at most one extra TRAILING EMPTY page in
		// the predicate-eval run, which carries zero docs and is invisible on the wire
		// (the leaf discards hasMore — ops/builtin.go; the coordinator derives the next
		// cursor from len(docs)==limit — embedded.go scrollNextCursorOrder). Both runs
		// here terminate on the production rule (a short page), so they visit the same
		// doc-bearing pages with identical ids + nextAfter — the full observable contract.
	}
}

func concatOrderPages(pages []orderPageRecord) []uint64 {
	var out []uint64
	for _, p := range pages {
		out = append(out, p.ids...)
	}
	return out
}

// --- per-family orderScrollFn adapters (filter passed THROUGH so filter-first engages) ---

func hnswOrderScrollFn(h *hnsw) orderScrollFn {
	return func(filter Filter, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) ([]Document, uint64, bool) {
		pred, err := filter.Compile()
		if err != nil {
			panic(err)
		}
		return h.scrollPage(filter, pred, nil, order, afterID, afterKey, hasAfter, limit)
	}
}

func ivfOrderScrollFn(ix *ivf) orderScrollFn {
	return func(filter Filter, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) ([]Document, uint64, bool) {
		pred, err := filter.Compile()
		if err != nil {
			panic(err)
		}
		return ix.scrollPage(filter, pred, nil, order, afterID, afterKey, hasAfter, limit)
	}
}

func namedOrderScrollFn(nc *NamedCollection) orderScrollFn {
	return func(filter Filter, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) ([]Document, uint64, bool) {
		pred, err := filter.Compile()
		if err != nil {
			panic(err)
		}
		return nc.scrollPage(filter, pred, order, afterID, afterKey, hasAfter, limit)
	}
}

func mvOrderScrollFn(m *MultiVectorIndex) orderScrollFn {
	return func(filter Filter, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) ([]Document, uint64, bool) {
		pred, err := filter.Compile()
		if err != nil {
			panic(err)
		}
		return m.scrollPage(filter, pred, order, afterID, afterKey, hasAfter, limit)
	}
}

// orderScrollEngine bundles a family's order scrollFn with the metadata map (for cursor
// rebuild) so the oracle runs identically across all four families.
type orderScrollEngine struct {
	name   string
	scroll orderScrollFn
	metas  map[uint64]Metadata
}

// buildOrderScrollEngines inserts the shared scrollDataset into all four families and
// returns an orderScrollEngine per family. The dataset's "kind"/"n"/"bucket" fields drive
// the accelerable filters; an "o" numeric + "os" string + "o2" numeric give the order keys.
func buildOrderScrollEngines(t *testing.T, total uint64) []orderScrollEngine {
	t.Helper()
	metas := map[uint64]Metadata{}
	pts := scrollDataset(total)
	for _, p := range pts {
		m := ptMeta(p)
		// Order keys, chosen to DIFFER from id-order so the value-sort is exercised:
		// "o" descends with id (so asc-by-o == desc-by-id), "os" a zero-padded string,
		// "o2" a coarse bucket (heavy ties → id tiebreak + multi-key secondary).
		m["o"] = NewInt(int64(total) - int64(p.id))
		m["os"] = NewString(fmt.Sprintf("v%04d", (p.id*7)%total))
		m["o2"] = NewInt(int64(p.id % 6))
		metas[p.id] = m
	}

	h := newScrollHNSW(t)
	for _, p := range pts {
		insertScroll(t, h, p.id, metas[p.id])
	}

	ix, err := newIVF(ivfTestConfig(4))
	if err != nil {
		t.Fatalf("newIVF: %v", err)
	}
	nc, err := NewNamedCollection("c", map[string]NamedVectorParams{
		"v": {Dim: 4, M: 16, EfConstruction: 200, EfSearch: 64, Metric: L2},
	})
	if err != nil {
		t.Fatalf("newNamedCollection: %v", err)
	}
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("NewMultiVectorIndex: %v", err)
	}
	for _, p := range pts {
		v := []float32{float32(p.id), float32(p.id % 7), float32(p.id % 3), 1}
		if _, _, err := ix.Insert(p.id, v, 0, metas[p.id], nil, nil, CASCond{}); err != nil {
			t.Fatalf("ivf Insert %d: %v", p.id, err)
		}
		if err := nc.Insert(p.id, map[string][]float32{"v": v}, metas[p.id], 0); err != nil {
			t.Fatalf("named Insert %d: %v", p.id, err)
		}
		if err := m.Add(p.id, [][]float32{v}, metas[p.id]); err != nil {
			t.Fatalf("mv Add %d: %v", p.id, err)
		}
	}
	return []orderScrollEngine{
		{"hnsw", hnswOrderScrollFn(h), metas},
		{"ivf", ivfOrderScrollFn(ix), metas},
		{"named", namedOrderScrollFn(nc), metas},
		{"mv", mvOrderScrollFn(m), metas},
	}
}

// orderFilterCases pairs each accelerable filter with a behaviourally-equivalent
// NON-accelerable filter (forces the predicate-eval order build) and the Go-level
// matcher (for the independent gap/dup-free check).
var orderFilterCases = []struct {
	label   string
	accel   Filter
	fallbk  Filter
	matches func(id uint64) bool
}{
	{"eq", acceleratedEq, fallbackNe, func(id uint64) bool { return id%2 == 0 }},
	{"range", acceleratedRange, fallbackRange, func(id uint64) bool { return int64(id) >= 40 }},
	{"in", acceleratedIn, fallbackIn, func(id uint64) bool { return id%5 == 1 || id%5 == 3 }},
}

// TestOrderFilterFirstOracleSingleKey: for EACH family, BOTH a numeric and a string
// single-key order, across the three accelerable filter shapes at several page sizes —
// the filter-first order page == the forced-fallback (predicate-eval) order page (ids,
// value-order, page sizes, cursor), AND the concatenated result == the expected matching
// set in the exact value-order, gap-free + dup-free.
func TestOrderFilterFirstOracleSingleKey(t *testing.T) {
	const total = 60
	engines := buildOrderScrollEngines(t, total)
	orders := []OrderBy{
		{Key: "o", Kind: OrderNumeric, Desc: false},
		{Key: "o", Kind: OrderNumeric, Desc: true},
		{Key: "os", Kind: OrderString, Desc: false},
		{Key: "os", Kind: OrderString, Desc: true},
	}
	for _, e := range engines {
		for _, order := range orders {
			for _, c := range orderFilterCases {
				for _, limit := range []int{1, 3, 7} {
					label := fmt.Sprintf("%s/%s/%s/desc=%v/limit=%d", e.name, c.label, order.Key, order.Desc, limit)
					ff := paginateOrder(e.scroll, c.accel, order, e.metas, limit)
					pe := paginateOrder(e.scroll, c.fallbk, order, e.metas, limit)
					sameOrderPages(t, label, ff, pe)

					// Independent ground truth: the matching ids sorted by the order key.
					got := concatOrderPages(ff)
					want := wantSingleKeyOrder(e.metas, order, total, c.matches)
					assertSameIDs(t, label, got, want)
				}
			}
		}
	}
}

// TestOrderFilterFirstOracleMultiKey: same oracle for a MULTI-KEY order (primary numeric
// desc, secondary string asc, tertiary numeric desc — mixed kinds + directions), proving
// the candidate narrowing composes with the v4 tuple cursor unchanged.
func TestOrderFilterFirstOracleMultiKey(t *testing.T) {
	const total = 60
	engines := buildOrderScrollEngines(t, total)
	primary := OrderBy{Key: "o2", Kind: OrderNumeric, Desc: true}
	tail := []OrderBy{
		{Key: "os", Kind: OrderString, Desc: false},
		{Key: "o", Kind: OrderNumeric, Desc: true},
	}
	keys := append([]OrderBy{primary}, tail...)
	for _, e := range engines {
		for _, c := range orderFilterCases {
			for _, limit := range []int{1, 2, 5} {
				label := fmt.Sprintf("%s/%s/multikey/limit=%d", e.name, c.label, limit)
				ff := paginateOrderMulti(e.scroll, c.accel, primary, tail, e.metas, limit)
				pe := paginateOrderMulti(e.scroll, c.fallbk, primary, tail, e.metas, limit)
				sameOrderPages(t, label, ff, pe)

				got := concatOrderPages(ff)
				want := wantMultiKeyOrder(e.metas, keys, total, c.matches)
				assertSameIDs(t, label, got, want)
			}
		}
	}
}

// TestOrderFilterFirstFallbackNonAccelerable: a regex filter (non-accelerable) on an
// order_by scroll falls back to the predicate-eval order build and is still correct
// (the matching docs in value-order, gap/dup-free). Proves the fallback path is engaged.
func TestOrderFilterFirstFallbackNonAccelerable(t *testing.T) {
	const total = 40
	engines := buildOrderScrollEngines(t, total)
	regex := Filter{Op: FilterRegex, Field: "kind", Value: NewString("^even$")}
	order := OrderBy{Key: "o", Kind: OrderNumeric, Desc: false}
	for _, e := range engines {
		got := concatOrderPages(paginateOrder(e.scroll, regex, order, e.metas, 5))
		want := wantSingleKeyOrder(e.metas, order, total, func(id uint64) bool { return id%2 == 0 })
		assertSameIDs(t, e.name+"/regex-fallback", got, want)
	}
}

// TestOrderFilterFirstNoFilterUnchanged: an order_by scroll with NO filter never engages
// filter-first (pred nil) and returns every doc in value-order — byte-identical to the
// pre-change path. Proves the no-filter order snapshot path is untouched.
func TestOrderFilterFirstNoFilterUnchanged(t *testing.T) {
	const total = 50
	engines := buildOrderScrollEngines(t, total)
	for _, e := range engines {
		for _, order := range []OrderBy{
			{Key: "o", Kind: OrderNumeric, Desc: false},
			{Key: "os", Kind: OrderString, Desc: true},
		} {
			got := concatOrderPages(paginateOrder(e.scroll, Filter{}, order, e.metas, 6))
			want := wantSingleKeyOrder(e.metas, order, total, func(uint64) bool { return true })
			assertSameIDs(t, fmt.Sprintf("%s/no-filter/%s/desc=%v", e.name, order.Key, order.Desc), got, want)
		}
	}
}

// TestOrderFilterFirstHNSWGate white-box-asserts the order narrowing gate: an accelerable
// selective filter builds the order rows over the candidate superset (far fewer than N),
// while a non-accelerable filter declines (nil,false ⇒ full snapshot). Mirrors the
// id-path TestScrollFilterFirstGate, for the order-row build.
func TestOrderFilterFirstHNSWGate(t *testing.T) {
	const total = 200
	metas := map[uint64]Metadata{}
	h := newScrollHNSW(t)
	for _, p := range scrollDataset(total) {
		m := ptMeta(p)
		m["o"] = NewInt(int64(total) - int64(p.id))
		metas[p.id] = m
		insertScroll(t, h, p.id, m)
	}
	order := &OrderBy{Key: "o", Kind: OrderNumeric, Desc: false}
	pred, _ := acceleratedEq.Compile()
	h.mu.RLock()
	rows, ok := h.filterFirstOrderRowsLocked(acceleratedEq, pred, nil, order)
	h.mu.RUnlock()
	if !ok {
		t.Fatalf("accelerable Eq filter should engage order filter-first")
	}
	if len(rows) != total/2 { // exactly the even docs carry "o"
		t.Fatalf("narrowed order rows = %d, want %d (even docs)", len(rows), total/2)
	}
	if len(rows) >= total {
		t.Fatalf("order rows %d not narrowed below N=%d", len(rows), total)
	}
	// rows must be value-sorted (o asc == id desc here).
	for i := 1; i < len(rows); i++ {
		if OrderLess(rows[i].Key, rows[i].ID, rows[i-1].Key, rows[i-1].ID, false) {
			t.Fatalf("narrowed order rows not value-sorted at %d", i)
		}
	}
	// Non-accelerable filters decline.
	for _, f := range []Filter{fallbackNe, fallbackRange, fallbackIn,
		{Op: FilterRegex, Field: "kind", Value: NewString("^even$")}} {
		fp, _ := f.Compile()
		h.mu.RLock()
		_, dok := h.filterFirstOrderRowsLocked(f, fp, nil, order)
		h.mu.RUnlock()
		if dok {
			t.Fatalf("non-accelerable filter %v should decline order filter-first", f.Op)
		}
	}
	// Provider path declines (empty sub-arena index).
	h.mu.RLock()
	_, okProv := h.filterFirstOrderRowsLocked(acceleratedEq, pred, func(uint64) Metadata { return nil }, order)
	h.mu.RUnlock()
	if okProv {
		t.Fatalf("provider path (metaOf != nil) must decline order filter-first")
	}
}

// TestOrderFilterFirstConcurrentMutation: a full paginated filter-first ORDER scroll is
// stable (gap-free + dup-free over the ids live for the whole scroll) under a concurrent
// insert/delete stream of NON-matching (odd) docs. The even (matching) docs are stable,
// so each appears exactly once in value-order.
func TestOrderFilterFirstConcurrentMutation(t *testing.T) {
	const base = 200
	h := newScrollHNSW(t)
	metas := map[uint64]Metadata{}
	for i := uint64(0); i < base; i++ {
		m := ptMeta(scrollPoint{id: i, kind: kindOf(i), n: int64(i), bucket: int64(i % 5)})
		m["o"] = NewInt(int64(base) - int64(i))
		metas[i] = m
		insertScroll(t, h, i, m)
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
			oddID := id*2 + 1 // odd ⇒ never in the even result set
			m := ptMeta(scrollPoint{id: oddID, kind: "odd", n: int64(oddID), bucket: int64(oddID % 5)})
			m["o"] = NewInt(int64(oddID))
			insertScroll(t, h, oddID, m)
			_, _ = h.Delete(oddID, CASCond{})
			id++
		}
	}()

	order := OrderBy{Key: "o", Kind: OrderNumeric, Desc: false}
	got := concatOrderPages(paginateOrder(hnswOrderScrollFn(h), acceleratedEq, order, metas, 4))
	close(stop)
	wg.Wait()

	seen := map[uint64]bool{}
	for _, id := range got {
		if id%2 != 0 {
			t.Fatalf("non-matching odd id %d emitted by even-filtered scroll", id)
		}
		if seen[id] {
			t.Fatalf("duplicate id %d across order pages", id)
		}
		seen[id] = true
	}
	for i := uint64(0); i < base; i += 2 {
		if !seen[i] {
			t.Fatalf("stable even id %d missing from concurrent order scroll (gap)", i)
		}
	}
	// Value-order check: o asc over even ids == id descending.
	for i := 1; i < len(got); i++ {
		if metas[got[i]]["o"].Int < metas[got[i-1]]["o"].Int {
			t.Fatalf("order scroll not value-sorted at %d: %v", i, got)
		}
	}
}

// --- ground-truth helpers ---

// wantSingleKeyOrder returns the matching ids sorted by the single order key (value, id),
// EXCLUDING ids missing the order field (the EXCLUDE policy).
func wantSingleKeyOrder(metas map[uint64]Metadata, order OrderBy, total uint64, matches func(uint64) bool) []uint64 {
	type row struct {
		id  uint64
		num float64
		str string
	}
	str := order.Kind == OrderString
	var rows []row
	for id := uint64(0); id < total; id++ {
		if !matches(id) {
			continue
		}
		m := metas[id]
		if str {
			sk, ok := OrderStringKey(m, order.Key)
			if !ok {
				continue
			}
			rows = append(rows, row{id: id, str: sk})
		} else {
			k, ok := OrderKey(m, order.Key, order.IsDatetime)
			if !ok {
				continue
			}
			rows = append(rows, row{id: id, num: k})
		}
	}
	sort.Slice(rows, func(i, j int) bool {
		if str {
			return OrderLessStr(rows[i].str, rows[i].id, rows[j].str, rows[j].id, order.Desc)
		}
		return OrderLess(rows[i].num, rows[i].id, rows[j].num, rows[j].id, order.Desc)
	})
	out := make([]uint64, len(rows))
	for i, r := range rows {
		out[i] = r.id
	}
	return out
}

// wantMultiKeyOrder returns the matching ids sorted by the tuple (k1,…,kN, id), EXCLUDING
// ids missing ANY order key.
func wantMultiKeyOrder(metas map[uint64]Metadata, keys []OrderBy, total uint64, matches func(uint64) bool) []uint64 {
	var rows []OrderedID
	for id := uint64(0); id < total; id++ {
		if !matches(id) {
			continue
		}
		vals, ok := orderTupleKeys(metas[id], keys)
		if !ok {
			continue
		}
		rows = append(rows, OrderedID{ID: id, Keys: vals})
	}
	SortOrderedIDsTuple(rows, keys)
	out := make([]uint64, len(rows))
	for i, r := range rows {
		out[i] = r.ID
	}
	return out
}

func assertSameIDs(t *testing.T, label string, got, want []uint64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d ids, want %d\n got=%v\n want=%v", label, len(got), len(want), got, want)
	}
	seen := map[uint64]bool{}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s: id[%d]=%d want %d\n got=%v\n want=%v", label, i, got[i], want[i], got, want)
		}
		if seen[got[i]] {
			t.Fatalf("%s: duplicate id %d", label, got[i])
		}
		seen[got[i]] = true
	}
}

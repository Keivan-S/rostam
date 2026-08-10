// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"sort"
	"testing"
)

// --- tuple-comparator unit tests (kind+desc per key, then id) ---

func TestOrderLessTupleCascade(t *testing.T) {
	keys := []OrderBy{
		{Key: "a", Kind: OrderNumeric, Desc: true}, // primary: num desc
		{Key: "b", Kind: OrderString, Desc: false}, // secondary: str asc
	}
	mk := func(num float64, str string, id uint64) OrderedID {
		return OrderedID{ID: id, Keys: []OrderVal{{Num: num, Kind: OrderNumeric}, {Str: str, Kind: OrderString}}}
	}
	rows := []OrderedID{
		mk(1, "z", 100),
		mk(2, "b", 1),
		mk(2, "a", 9),
		mk(2, "a", 3), // ties with above on (2,"a") -> id asc: 3 before 9
		mk(3, "q", 50),
	}
	SortOrderedIDsTuple(rows, keys)
	gotIDs := make([]uint64, len(rows))
	for i, r := range rows {
		gotIDs[i] = r.ID
	}
	// primary num DESC: 3,2,2,2,1 ; among the three num==2: str ASC a,a,b -> (a,3),(a,9),(b,1)
	want := []uint64{50, 3, 9, 1, 100}
	for i := range want {
		if gotIDs[i] != want[i] {
			t.Fatalf("tuple sort = %v, want %v", gotIDs, want)
		}
	}
}

func TestOrderAfterTupleLowerBound(t *testing.T) {
	keys := []OrderBy{{Kind: OrderNumeric, Desc: false}, {Kind: OrderString, Desc: false}}
	mk := func(num float64, str string, id uint64) OrderedID {
		return OrderedID{ID: id, Keys: []OrderVal{{Num: num, Kind: OrderNumeric}, {Str: str, Kind: OrderString}}}
	}
	rows := []OrderedID{mk(1, "a", 1), mk(1, "a", 4), mk(1, "b", 2), mk(2, "a", 8), mk(2, "a", 9)}
	SortOrderedIDsTuple(rows, keys)
	// cursor at (1,"a",4): next is (1,"b",2) at index 2.
	cur := mk(1, "a", 4)
	idx := sort.Search(len(rows), func(i int) bool { return OrderAfterTuple(cur, rows[i], keys) })
	if idx != 2 || rows[idx].ID != 2 {
		t.Fatalf("lower bound idx=%d id=%d, want idx=2 id=2", idx, rows[idx].ID)
	}
	// cursor at last (2,"a",9): no row after.
	idx = sort.Search(len(rows), func(i int) bool { return OrderAfterTuple(mk(2, "a", 9), rows[i], keys) })
	if idx != len(rows) {
		t.Fatalf("past-last idx=%d, want %d", idx, len(rows))
	}
}

// TestOrderLessTupleSingleKeyMatchesSingle: a 1-element tuple comparator produces the
// IDENTICAL order to the single-key OrderLess/OrderLessStr (the byte-identical anchor).
func TestOrderLessTupleSingleKeyMatchesSingle(t *testing.T) {
	for _, desc := range []bool{false, true} {
		// numeric
		numKeys := []OrderBy{{Kind: OrderNumeric, Desc: desc}}
		nrows := []OrderedID{{Key: 3, ID: 10}, {Key: 1, ID: 5}, {Key: 3, ID: 2}, {Key: 1, ID: 9}, {Key: 2, ID: 7}}
		single := append([]OrderedID(nil), nrows...)
		SortOrderedIDs(single, desc)
		tuple := make([]OrderedID, len(nrows))
		for i, r := range nrows {
			tuple[i] = OrderedID{ID: r.ID, Keys: []OrderVal{{Num: r.Key, Kind: OrderNumeric}}}
		}
		SortOrderedIDsTuple(tuple, numKeys)
		for i := range single {
			if single[i].ID != tuple[i].ID {
				t.Fatalf("numeric desc=%v: tuple order %v != single %v at %d", desc, tuple[i].ID, single[i].ID, i)
			}
		}
		// string
		strKeys := []OrderBy{{Kind: OrderString, Desc: desc}}
		srows := []OrderedID{{StrKey: "c", ID: 10}, {StrKey: "a", ID: 5}, {StrKey: "c", ID: 2}, {StrKey: "a", ID: 9}, {StrKey: "b", ID: 7}}
		ssingle := append([]OrderedID(nil), srows...)
		SortOrderedIDsStr(ssingle, desc)
		stuple := make([]OrderedID, len(srows))
		for i, r := range srows {
			stuple[i] = OrderedID{ID: r.ID, Keys: []OrderVal{{Str: r.StrKey, Kind: OrderString}}}
		}
		SortOrderedIDsTuple(stuple, strKeys)
		for i := range ssingle {
			if ssingle[i].ID != stuple[i].ID {
				t.Fatalf("string desc=%v: tuple order %v != single %v at %d", desc, stuple[i].ID, ssingle[i].ID, i)
			}
		}
	}
}

// --- engine multi-key full-pagination tests (drive OrderBy.Tail directly) ---

// multiKeyVal extracts the OrderVal tuple a row would have from a Metadata, for the
// reference sort. nil if any key missing.
func refTuple(meta Metadata, keys []OrderBy) ([]OrderVal, bool) {
	return orderTupleKeys(meta, keys)
}

// bruteOrderTuple is the ground-truth (k1,…,kN, id) order over (id -> meta), excluding
// ids missing any key (the EXCLUDE policy applied per key).
func bruteOrderTuple(metas map[uint64]Metadata, keys []OrderBy) []uint64 {
	type row struct {
		id   uint64
		vals []OrderVal
	}
	var rows []row
	for id, m := range metas {
		if vals, ok := refTuple(m, keys); ok {
			rows = append(rows, row{id, vals})
		}
	}
	ords := make([]OrderedID, len(rows))
	for i, r := range rows {
		ords[i] = OrderedID{ID: r.id, Keys: r.vals}
	}
	SortOrderedIDsTuple(ords, keys)
	out := make([]uint64, len(ords))
	for i, o := range ords {
		out[i] = o.ID
	}
	return out
}

// multiKeyScrollFn is the scrollPage closure shared by the dense/ivf and named/MV
// harnesses (both expose the same shape).
type multiKeyScrollFn func(pred Predicate, order *OrderBy, afterID uint64, afterKey float64, hasAfter bool, limit int) ([]Document, uint64, bool)

// scrollAllMultiKey pages a multi-key order_by scroll, rebuilding the v4 resume TUPLE
// from the last doc's metadata each page (the engine resumes via ResumeKeys/afterID).
func scrollAllMultiKey(scroll multiKeyScrollFn, primary OrderBy, tail []OrderBy, metas map[uint64]Metadata, limit int) []uint64 {
	keys := append([]OrderBy{withoutTail(primary)}, tail...)
	var collected []uint64
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
		docs, nextAfter, hasMore := scroll(nil, &order, afterID, 0, hasAfter, limit)
		for _, d := range docs {
			collected = append(collected, d.ID)
		}
		if !hasMore {
			break
		}
		afterID = nextAfter
		last := docs[len(docs)-1]
		vals, ok := orderTupleKeys(metaFor(metas, last.ID), keys)
		if !ok {
			panic("last doc missing an order key")
		}
		resume = vals
		hasAfter = true
	}
	return collected
}

func withoutTail(o OrderBy) OrderBy { o.Tail = nil; return o }

func metaFor(metas map[uint64]Metadata, id uint64) Metadata { return metas[id] }

func TestMultiKeyOrderFullPagination(t *testing.T) {
	eachOrderEngine(t, func(t *testing.T, e *orderEngine) {
		// 30 points: num key "p" with many ties (so the secondary str key "n" matters),
		// plus a tertiary numeric "r". Some ties cascade to the id tiebreak.
		metas := map[uint64]Metadata{}
		for id := uint64(1); id <= 30; id++ {
			metas[id] = Metadata{
				"p": NewInt(int64(id % 4)),               // 4 distinct values -> heavy ties
				"n": NewString(fmt.Sprintf("k%d", id%3)), // 3 distinct strings
				"r": NewFloat(float64((id * 7) % 5)),     // 5 distinct
			}
			e.insert(t, id, metas[id])
		}
		// order: p desc, n asc, r desc  (mixed kinds + mixed directions)
		primary := OrderBy{Key: "p", Kind: OrderNumeric, Desc: true}
		tail := []OrderBy{
			{Key: "n", Kind: OrderString, Desc: false},
			{Key: "r", Kind: OrderNumeric, Desc: true},
		}
		keys := append([]OrderBy{primary}, tail...)
		want := bruteOrderTuple(metas, keys)
		for _, limit := range []int{1, 2, 3, 7, 30, 100} {
			got := scrollAllMultiKey(e.scrollPage, primary, tail, metas, limit)
			if len(got) != len(want) {
				t.Fatalf("limit=%d len(got)=%d, want %d", limit, len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("limit=%d composite order mismatch at %d: got %d want %d\ngot=%v\nwant=%v", limit, i, got[i], want[i], got, want)
				}
			}
			// gap-free + dup-free: a set of all 30 ids.
			seen := map[uint64]bool{}
			for _, id := range got {
				if seen[id] {
					t.Fatalf("limit=%d duplicate id %d", limit, id)
				}
				seen[id] = true
			}
			if len(seen) != 30 {
				t.Fatalf("limit=%d collected %d unique, want 30", limit, len(seen))
			}
		}
	})
}

// TestMultiKeyTieCascade: ties on key[0] broken by key[1], then by id — proven against
// a hand-built expectation.
func TestMultiKeyTieCascade(t *testing.T) {
	eachOrderEngine(t, func(t *testing.T, e *orderEngine) {
		// All share p=5. Secondary n distinguishes; some share n -> id breaks.
		data := []struct {
			id uint64
			n  string
		}{
			{10, "b"}, {3, "a"}, {7, "a"}, {1, "b"}, {5, "a"},
		}
		metas := map[uint64]Metadata{}
		for _, d := range data {
			metas[d.id] = Metadata{"p": NewInt(5), "n": NewString(d.n)}
			e.insert(t, d.id, metas[d.id])
		}
		primary := OrderBy{Key: "p", Kind: OrderNumeric, Desc: false}
		tail := []OrderBy{{Key: "n", Kind: OrderString, Desc: false}}
		got := scrollAllMultiKey(e.scrollPage, primary, tail, metas, 2)
		// n asc: a {3,5,7 by id}, then b {1,10 by id}
		want := []uint64{3, 5, 7, 1, 10}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("tie cascade got %v, want %v", got, want)
			}
		}
	})
}

// TestMultiKeyMixedKindsStrPrimary: [str asc, num desc] primary string.
func TestMultiKeyMixedKindsStrPrimary(t *testing.T) {
	eachOrderEngine(t, func(t *testing.T, e *orderEngine) {
		metas := map[uint64]Metadata{}
		for id := uint64(1); id <= 12; id++ {
			metas[id] = Metadata{"s": NewString(fmt.Sprintf("g%d", id%3)), "v": NewInt(int64(id % 4))}
			e.insert(t, id, metas[id])
		}
		primary := OrderBy{Key: "s", Kind: OrderString, Desc: false}
		tail := []OrderBy{{Key: "v", Kind: OrderNumeric, Desc: true}}
		keys := append([]OrderBy{primary}, tail...)
		want := bruteOrderTuple(metas, keys)
		got := scrollAllMultiKey(e.scrollPage, primary, tail, metas, 3)
		if len(got) != len(want) {
			t.Fatalf("len got %d want %d", len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("mixed-kind str-primary mismatch at %d: got %v want %v", i, got, want)
			}
		}
	})
}

// TestMultiKeyExcludesMissingKey: a point missing ANY order key is excluded (per-key
// EXCLUDE policy).
func TestMultiKeyExcludesMissingKey(t *testing.T) {
	eachOrderEngine(t, func(t *testing.T, e *orderEngine) {
		e.insert(t, 1, Metadata{"p": NewInt(1), "n": NewString("a")})
		e.insert(t, 2, Metadata{"p": NewInt(2)})                      // missing n -> excluded
		e.insert(t, 3, Metadata{"p": NewInt(3), "n": NewString("b")}) // present
		e.insert(t, 4, Metadata{"n": NewString("c")})                 // missing p -> excluded
		primary := OrderBy{Key: "p", Kind: OrderNumeric, Desc: false}
		tail := []OrderBy{{Key: "n", Kind: OrderString, Desc: false}}
		metas := map[uint64]Metadata{
			1: {"p": NewInt(1), "n": NewString("a")},
			3: {"p": NewInt(3), "n": NewString("b")},
		}
		got := scrollAllMultiKey(e.scrollPage, primary, tail, metas, 10)
		want := []uint64{1, 3}
		if len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
			t.Fatalf("missing-key exclude got %v, want %v", got, want)
		}
	})
}

// TestMultiKeyOrderNamedMV: the named + MV families produce the same composite
// (k1,…,kN, id) order across full pagination as the reference tuple sort.
func TestMultiKeyOrderNamedMV(t *testing.T) {
	eachNamedMVEngine(t, func(t *testing.T, e *namedMVOrderEngine) {
		metas := map[uint64]Metadata{}
		for id := uint64(1); id <= 24; id++ {
			metas[id] = Metadata{
				"p": NewInt(int64(id % 3)),
				"n": NewString(fmt.Sprintf("k%d", id%2)),
				"r": NewFloat(float64((id * 3) % 4)),
			}
			e.insert(t, id, metas[id])
		}
		primary := OrderBy{Key: "p", Kind: OrderNumeric, Desc: false}
		tail := []OrderBy{
			{Key: "n", Kind: OrderString, Desc: true},
			{Key: "r", Kind: OrderNumeric, Desc: false},
		}
		keys := append([]OrderBy{primary}, tail...)
		want := bruteOrderTuple(metas, keys)
		for _, limit := range []int{1, 3, 5, 24} {
			got := scrollAllMultiKey(e.scrollPage, primary, tail, metas, limit)
			if len(got) != len(want) {
				t.Fatalf("%s limit=%d len got %d want %d", e.name, limit, len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s limit=%d composite mismatch at %d: got %v want %v", e.name, limit, i, got, want)
				}
			}
		}
	})
}

// TestSingleKeyStillEmitsV2V3Cursor: a SINGLE-key order (empty Tail) is NOT multi-key —
// the snapshot/cache key/seek use the single-key fast path, NOT the tuple path. This is
// the byte-identical anchor: isMultiKey is false and the cache key is the legacy
// {field, desc} (so the wire layer emits a v2/v3 cursor, never v4).
func TestSingleKeyStillEmitsV2V3Cursor(t *testing.T) {
	num := &OrderBy{Key: "p", Kind: OrderNumeric, Desc: false}
	str := &OrderBy{Key: "s", Kind: OrderString, Desc: true}
	if isMultiKey(num) || isMultiKey(str) {
		t.Fatalf("single-key order reported multi-key")
	}
	if got := orderSnapCacheKey(num); got != (orderCacheKey{"p", false}) {
		t.Fatalf("single-key numeric cache key = %v, want {p false}", got)
	}
	if got := orderSnapCacheKey(str); got != (orderCacheKey{"s", true}) {
		t.Fatalf("single-key string cache key = %v, want {s true}", got)
	}
	// A multi-key order IS multi-key and gets a distinct (non-legacy) cache key.
	multi := &OrderBy{Key: "p", Kind: OrderNumeric, Tail: []OrderBy{{Key: "s", Kind: OrderString}}}
	if !isMultiKey(multi) {
		t.Fatalf("multi-key order not reported multi-key")
	}
	if orderSnapCacheKey(multi) == (orderCacheKey{"p", false}) {
		t.Fatalf("multi-key cache key collided with single-key {p false}")
	}
}

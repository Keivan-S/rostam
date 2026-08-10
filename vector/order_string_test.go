// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"sort"
	"testing"
)

// --- unit tests: string key extraction + comparators -----------------------

func TestOrderStringKey(t *testing.T) {
	meta := Metadata{
		"name":  NewString("bob"),
		"empty": NewString(""),
		"i":     NewInt(42),
		"b":     NewBool(true),
		"ss":    NewStrings([]string{"a", "b"}),
	}
	cases := []struct {
		key  string
		want string
		ok   bool
	}{
		{"name", "bob", true},
		{"empty", "", true}, // empty STRING is a valid scalar key (present)
		{"i", "", false},    // int ⇒ exclude (non-string)
		{"b", "", false},    // bool ⇒ exclude
		{"ss", "", false},   // string list ⇒ exclude (not a scalar string)
		{"absent", "", false},
	}
	for _, c := range cases {
		got, ok := OrderStringKey(meta, c.key)
		if ok != c.ok || (ok && got != c.want) {
			t.Errorf("OrderStringKey(%q) = (%q, %v), want (%q, %v)", c.key, got, ok, c.want, c.ok)
		}
	}
	// nil meta ⇒ excluded.
	if _, ok := OrderStringKey(nil, "name"); ok {
		t.Fatalf("OrderStringKey(nil) ok=true, want false")
	}
}

func TestOrderLessStr(t *testing.T) {
	// ASC: lexicographic by key, id tiebreak ascending.
	if !OrderLessStr("apple", 100, "banana", 1, false) {
		t.Error("asc: apple < banana regardless of id")
	}
	if OrderLessStr("banana", 1, "apple", 100, false) {
		t.Error("asc: banana not < apple")
	}
	if !OrderLessStr("x", 1, "x", 2, false) {
		t.Error("asc tie: id 1 < id 2")
	}
	if OrderLessStr("x", 1, "x", 1, false) {
		t.Error("equal (str,id) not less")
	}
	// DESC: reverse on key only, id tiebreak STILL ascending.
	if !OrderLessStr("banana", 100, "apple", 1, true) {
		t.Error("desc: banana < apple (larger first)")
	}
	if !OrderLessStr("x", 1, "x", 2, true) {
		t.Error("desc tie: id 1 < id 2 (id tiebreak stays ascending)")
	}
	// Empty string sorts before any non-empty (asc).
	if !OrderLessStr("", 5, "a", 1, false) {
		t.Error("asc: empty < 'a'")
	}
}

func TestOrderAfterStrAsLowerBound(t *testing.T) {
	rows := []OrderedID{
		{StrKey: "a", ID: 1}, {StrKey: "a", ID: 4},
		{StrKey: "b", ID: 2}, {StrKey: "b", ID: 8}, {StrKey: "c", ID: 5},
	}
	SortOrderedIDsStr(rows, false)
	// Cursor at ("b", 2): next row should be ("b", 8) at index 3.
	idx := sort.Search(len(rows), func(i int) bool {
		return OrderAfterStr("b", 2, rows[i].StrKey, rows[i].ID, false)
	})
	if idx != 3 || rows[idx].StrKey != "b" || rows[idx].ID != 8 {
		t.Fatalf("lower bound idx=%d row=%v, want idx=3 row={b 8}", idx, rows[idx])
	}
	// Cursor at the last row ("c",5): no row after ⇒ idx == len.
	idx = sort.Search(len(rows), func(i int) bool {
		return OrderAfterStr("c", 5, rows[i].StrKey, rows[i].ID, false)
	})
	if idx != len(rows) {
		t.Fatalf("past-last idx=%d, want %d", idx, len(rows))
	}
}

// --- engine pagination tests (dense + ivf) ---------------------------------

// bruteOrderStr is the ground-truth (string, id) order for the direction.
func bruteOrderStr(vals map[uint64]string, desc bool) []uint64 {
	type kv struct {
		id  uint64
		val string
	}
	rows := make([]kv, 0, len(vals))
	for id, v := range vals {
		rows = append(rows, kv{id, v})
	}
	sort.Slice(rows, func(i, j int) bool {
		return OrderLessStr(rows[i].val, rows[i].id, rows[j].val, rows[j].id, desc)
	})
	out := make([]uint64, len(rows))
	for i, r := range rows {
		out[i] = r.id
	}
	return out
}

// scrollAllOrderedStr pages through a STRING order_by scroll, threading the resume
// STRING key (order.ResumeStr) from the last doc's order field between pages — the
// v3 cursor resume in miniature.
func scrollAllOrderedStr(e *orderEngine, field string, desc bool, limit int) []uint64 {
	var collected []uint64
	var afterID uint64
	hasAfter := false
	resume := ""
	for {
		order := &OrderBy{Key: field, Desc: desc, Kind: OrderString, ResumeStr: resume, HasResumeStr: hasAfter}
		docs, nextAfter, hasMore := e.scrollPage(nil, order, afterID, 0, hasAfter, limit)
		for _, d := range docs {
			collected = append(collected, d.ID)
		}
		if !hasMore {
			break
		}
		afterID = nextAfter
		last := docs[len(docs)-1]
		sk, ok := OrderStringKey(last.Metadata, field)
		if !ok {
			break
		}
		resume = sk
		hasAfter = true
	}
	return collected
}

func TestStringOrderPaginationMatchesBruteForce(t *testing.T) {
	words := []string{"delta", "alpha", "charlie", "bravo", "echo", "alpha", "foxtrot", "bravo", "golf", "alpha"}
	eachOrderEngine(t, func(t *testing.T, e *orderEngine) {
		vals := map[uint64]string{}
		for i, w := range words {
			id := uint64(i + 1)
			vals[id] = w
			e.insert(t, id, Metadata{"name": NewString(w)})
		}
		for _, desc := range []bool{false, true} {
			want := bruteOrderStr(vals, desc)
			for _, limit := range []int{0, 1, 2, 3, 100} {
				got := scrollAllOrderedStr(e, "name", desc, limit)
				if len(got) != len(want) {
					t.Fatalf("desc=%v limit=%d: got %d ids, want %d (%v vs %v)", desc, limit, len(got), len(want), got, want)
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("desc=%v limit=%d: order[%d]=%d want %d (got %v want %v)", desc, limit, i, got[i], want[i], got, want)
					}
				}
				// gap-free + dup-free.
				seen := map[uint64]bool{}
				for _, id := range got {
					if seen[id] {
						t.Fatalf("desc=%v limit=%d: duplicate id %d across pages", desc, limit, id)
					}
					seen[id] = true
				}
			}
		}
	})
}

// TestStringOrderDuplicateTiebreak: equal string values ⇒ deterministic id-ascending
// tiebreak in BOTH directions (the documented (string,id) total order).
func TestStringOrderDuplicateTiebreak(t *testing.T) {
	eachOrderEngine(t, func(t *testing.T, e *orderEngine) {
		// All the same string ⇒ pure id-ascending order regardless of direction.
		ids := []uint64{9, 3, 7, 1, 5}
		for _, id := range ids {
			e.insert(t, id, Metadata{"name": NewString("same")})
		}
		for _, desc := range []bool{false, true} {
			got := scrollAllOrderedStr(e, "name", desc, 2)
			want := []uint64{1, 3, 5, 7, 9} // id ascending in both directions
			if len(got) != len(want) {
				t.Fatalf("desc=%v: got %v, want %v", desc, got, want)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("desc=%v: dup tiebreak %v, want %v", desc, got, want)
				}
			}
		}
	})
}

// scrollAllOrderedStrNamedMV is scrollAllOrderedStr for the named/MV harness.
func scrollAllOrderedStrNamedMV(e *namedMVOrderEngine, field string, desc bool, limit int) []uint64 {
	var collected []uint64
	var afterID uint64
	hasAfter := false
	resume := ""
	for {
		order := &OrderBy{Key: field, Desc: desc, Kind: OrderString, ResumeStr: resume, HasResumeStr: hasAfter}
		docs, nextAfter, hasMore := e.scrollPage(nil, order, afterID, 0, hasAfter, limit)
		for _, d := range docs {
			collected = append(collected, d.ID)
		}
		if !hasMore {
			break
		}
		afterID = nextAfter
		last := docs[len(docs)-1]
		sk, ok := OrderStringKey(last.Metadata, field)
		if !ok {
			break
		}
		resume = sk
		hasAfter = true
	}
	return collected
}

// TestStringOrderPaginationNamedMV mirrors TestStringOrderPaginationMatchesBruteForce
// for the named + MV families.
func TestStringOrderPaginationNamedMV(t *testing.T) {
	words := []string{"delta", "alpha", "charlie", "bravo", "echo", "alpha", "foxtrot", "bravo", "golf", "alpha"}
	eachNamedMVEngine(t, func(t *testing.T, e *namedMVOrderEngine) {
		vals := map[uint64]string{}
		for i, w := range words {
			id := uint64(i + 1)
			vals[id] = w
			e.insert(t, id, Metadata{"name": NewString(w)})
		}
		for _, desc := range []bool{false, true} {
			want := bruteOrderStr(vals, desc)
			for _, limit := range []int{0, 1, 3, 100} {
				got := scrollAllOrderedStrNamedMV(e, "name", desc, limit)
				if len(got) != len(want) {
					t.Fatalf("desc=%v limit=%d: got %d ids, want %d (%v vs %v)", desc, limit, len(got), len(want), got, want)
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("desc=%v limit=%d: order[%d]=%d want %d (got %v want %v)", desc, limit, i, got[i], want[i], got, want)
					}
				}
				seen := map[uint64]bool{}
				for _, id := range got {
					if seen[id] {
						t.Fatalf("desc=%v limit=%d: duplicate id %d across pages", desc, limit, id)
					}
					seen[id] = true
				}
			}
		}
	})
}

// TestStringOrderMissingExcluded: points whose order field is absent or non-string
// are EXCLUDED (the numeric missing-value policy), never sorted to an end.
func TestStringOrderMissingExcluded(t *testing.T) {
	eachOrderEngine(t, func(t *testing.T, e *orderEngine) {
		e.insert(t, 1, Metadata{"name": NewString("b")})
		e.insert(t, 2, Metadata{"name": NewString("a")})
		e.insert(t, 3, Metadata{"other": NewString("x")}) // missing order field
		e.insert(t, 4, Metadata{"name": NewInt(5)})       // non-string ⇒ excluded
		e.insert(t, 5, Metadata{"name": NewString("c")})
		got := scrollAllOrderedStr(e, "name", false, 2)
		want := []uint64{2, 1, 5} // a,b,c — ids 3 and 4 excluded
		if len(got) != len(want) {
			t.Fatalf("got %v, want %v", got, want)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("missing-excluded order %v, want %v", got, want)
			}
		}
	})
}

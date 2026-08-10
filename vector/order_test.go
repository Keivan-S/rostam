// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"sort"
	"testing"
)

func TestOrderKeyNumeric(t *testing.T) {
	meta := Metadata{
		"i":   NewInt(42),
		"f":   NewFloat(3.5),
		"neg": NewInt(-7),
		"s":   NewString("nope"),
		"b":   NewBool(true),
		"is":  NewInts([]int64{1, 2}),
	}
	cases := []struct {
		key     string
		wantKey float64
		wantOK  bool
	}{
		{"i", 42, true},
		{"f", 3.5, true},
		{"neg", -7, true},
		{"s", 0, false},   // non-numeric string ⇒ exclude
		{"b", 0, false},   // bool ⇒ exclude
		{"is", 0, false},  // int list ⇒ exclude
		{"abs", 0, false}, // missing ⇒ exclude
	}
	for _, c := range cases {
		got, ok := OrderKey(meta, c.key, false)
		if ok != c.wantOK || (ok && got != c.wantKey) {
			t.Errorf("OrderKey(%q) = (%v, %v), want (%v, %v)", c.key, got, ok, c.wantKey, c.wantOK)
		}
	}
}

func TestOrderKeyDatetime(t *testing.T) {
	// Datetime is stored as int unix-ms; OrderKey returns it as float64 verbatim.
	const ms int64 = 1_718_000_000_123
	meta := Metadata{"ts": NewInt(ms)}
	got, ok := OrderKey(meta, "ts", true)
	if !ok {
		t.Fatalf("OrderKey datetime ok=false, want true")
	}
	if got != float64(ms) {
		t.Fatalf("OrderKey datetime = %v, want %v (int-ms as float64)", got, float64(ms))
	}
	// Missing datetime field still excluded.
	if _, ok := OrderKey(meta, "absent", true); ok {
		t.Fatalf("OrderKey missing datetime ok=true, want false")
	}
}

func TestOrderKeyNilMeta(t *testing.T) {
	if _, ok := OrderKey(nil, "k", false); ok {
		t.Fatalf("OrderKey(nil) ok=true, want false")
	}
}

func TestOrderLessAsc(t *testing.T) {
	// Distinct keys: smaller key first.
	if !OrderLess(1, 100, 2, 1, false) {
		t.Error("asc: 1 should be < 2 regardless of id")
	}
	if OrderLess(2, 1, 1, 100, false) {
		t.Error("asc: 2 should not be < 1")
	}
	// Value tie: smaller id first.
	if !OrderLess(5, 1, 5, 2, false) {
		t.Error("asc tie: id 1 should be < id 2")
	}
	if OrderLess(5, 2, 5, 1, false) {
		t.Error("asc tie: id 2 should not be < id 1")
	}
	// Equal (value,id) ⇒ not less.
	if OrderLess(5, 1, 5, 1, false) {
		t.Error("equal (value,id) should not be less")
	}
}

func TestOrderLessDesc(t *testing.T) {
	// Distinct keys: larger key first in desc.
	if !OrderLess(2, 100, 1, 1, true) {
		t.Error("desc: 2 should be < 1 (larger key first)")
	}
	if OrderLess(1, 1, 2, 100, true) {
		t.Error("desc: 1 should not be < 2")
	}
	// Value tie: id STILL ascending (the documented tiebreak choice).
	if !OrderLess(5, 1, 5, 2, true) {
		t.Error("desc tie: id 1 should be < id 2 (id tiebreak stays ascending)")
	}
	if OrderLess(5, 2, 5, 1, true) {
		t.Error("desc tie: id 2 should not be < id 1")
	}
}

func TestOrderLessSortsDeterministically(t *testing.T) {
	rows := []OrderedID{
		{Key: 3, ID: 10},
		{Key: 1, ID: 5},
		{Key: 3, ID: 2},
		{Key: 1, ID: 9},
		{Key: 2, ID: 7},
	}
	// ASC: by key, ties by id ascending.
	asc := append([]OrderedID(nil), rows...)
	SortOrderedIDs(asc, false)
	wantAsc := []OrderedID{{Key: 1, ID: 5}, {Key: 1, ID: 9}, {Key: 2, ID: 7}, {Key: 3, ID: 2}, {Key: 3, ID: 10}}
	if !eqOrdered(asc, wantAsc) {
		t.Errorf("asc sort = %v, want %v", asc, wantAsc)
	}
	// DESC: by key descending, ties by id ASCENDING.
	desc := append([]OrderedID(nil), rows...)
	SortOrderedIDs(desc, true)
	wantDesc := []OrderedID{{Key: 3, ID: 2}, {Key: 3, ID: 10}, {Key: 2, ID: 7}, {Key: 1, ID: 5}, {Key: 1, ID: 9}}
	if !eqOrdered(desc, wantDesc) {
		t.Errorf("desc sort = %v, want %v", desc, wantDesc)
	}
}

func TestOrderAfterBoundary(t *testing.T) {
	// OrderAfter == "strictly after the cursor" == OrderLess(cursor, candidate).
	// ASC cursor at (5, 10).
	cases := []struct {
		name      string
		curK      float64
		curID     uint64
		k         float64
		id        uint64
		desc      bool
		wantAfter bool
	}{
		{"asc same point not after", 5, 10, 5, 10, false, false},
		{"asc same value bigger id after", 5, 10, 5, 11, false, true},
		{"asc same value smaller id not after", 5, 10, 5, 9, false, false},
		{"asc bigger value after", 5, 10, 6, 0, false, true},
		{"asc smaller value not after", 5, 10, 4, 999, false, false},
		{"desc same point not after", 5, 10, 5, 10, true, false},
		{"desc same value bigger id after", 5, 10, 5, 11, true, true},
		{"desc smaller value after", 5, 10, 4, 0, true, true},
		{"desc bigger value not after", 5, 10, 6, 0, true, false},
	}
	for _, c := range cases {
		got := OrderAfter(c.curK, c.curID, c.k, c.id, c.desc)
		if got != c.wantAfter {
			t.Errorf("%s: OrderAfter = %v, want %v", c.name, got, c.wantAfter)
		}
	}
}

// TestOrderAfterAsLowerBound proves sort.Search over an OrderLess-sorted slice with
// OrderAfter finds exactly the first row strictly after the cursor (the next page's
// first row), at boundaries including value ties.
func TestOrderAfterAsLowerBound(t *testing.T) {
	rows := []OrderedID{{Key: 1, ID: 1}, {Key: 1, ID: 4}, {Key: 2, ID: 2}, {Key: 2, ID: 8}, {Key: 3, ID: 5}}
	SortOrderedIDs(rows, false) // already sorted, but be explicit
	// Cursor at (2, 2): next row should be (2, 8) at index 3.
	curK, curID := 2.0, uint64(2)
	idx := sort.Search(len(rows), func(i int) bool {
		return OrderAfter(curK, curID, rows[i].Key, rows[i].ID, false)
	})
	if idx != 3 || rows[idx].Key != 2 || rows[idx].ID != 8 {
		t.Fatalf("lower bound idx=%d row=%v, want idx=3 row={2 8}", idx, rows[idx])
	}
	// Cursor at the last row (3,5): no row after ⇒ idx == len.
	idx = sort.Search(len(rows), func(i int) bool {
		return OrderAfter(3, 5, rows[i].Key, rows[i].ID, false)
	})
	if idx != len(rows) {
		t.Fatalf("past-last lower bound idx=%d, want %d", idx, len(rows))
	}
}

func TestOrderKeyHashDetectsChange(t *testing.T) {
	if OrderKeyHash("price") == OrderKeyHash("created_at") {
		t.Error("distinct keys hashed equal (want very likely different)")
	}
	if OrderKeyHash("price") != OrderKeyHash("price") { //nolint:staticcheck // intentional: same call twice asserts determinism
		t.Error("same key hashed differently (must be deterministic)")
	}
}

func TestOrderLessSpecialValues(t *testing.T) {
	// Negative, zero, large values order correctly under OrderLess.
	rows := []OrderedID{
		{Key: math.MaxFloat64, ID: 1},
		{Key: -1e308, ID: 2},
		{Key: 0, ID: 3},
		{Key: math.Copysign(0, -1), ID: 4},
	}
	SortOrderedIDs(rows, false)
	if rows[0].Key != -1e308 {
		t.Errorf("asc smallest = %v, want -1e308", rows[0].Key)
	}
	if rows[len(rows)-1].Key != math.MaxFloat64 {
		t.Errorf("asc largest = %v, want MaxFloat64", rows[len(rows)-1].Key)
	}
}

func eqOrdered(a, b []OrderedID) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		// Single-key rows leave Keys nil; compare the scalar fields only (OrderedID is
		// no longer == comparable now that it carries the multi-key Keys slice).
		if a[i].Key != b[i].Key || a[i].ID != b[i].ID || a[i].StrKey != b[i].StrKey {
			return false
		}
	}
	return true
}

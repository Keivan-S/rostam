// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"math/rand"
	"reflect"
	"sort"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// orderRow is one inserted point's (id, orderKey) — the independent ground truth
// the paged order_by sequence is checked against.
type orderRow struct {
	id  uint64
	key float64
}

// seedDenseOrder creates a P-partition dense collection and inserts each id with a
// numeric "rank" payload field = key. Inserts happen in RANDOM order so a correct
// global (value,id) page sequence proves the per-call sort + merge, not insert order.
// Some ids (when missingEvery>0) are inserted WITHOUT the rank field to exercise the
// missing-field EXCLUDE policy.
func seedDenseOrder(t *testing.T, s Store, coll string, P int, rows []orderRow, missingEvery int) {
	t.Helper()
	ctx := context.Background()
	if err := s.CreateCollection(ctx, coll, VectorConfig{
		Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2, Partitions: P,
	}); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	order := append([]orderRow(nil), rows...)
	rand.New(rand.NewSource(99)).Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
	for n, rw := range order {
		v := []float32{float32(rw.id), 0, 0, 0}
		md := VectorMetadata{}
		if missingEvery <= 0 || n%missingEvery != 0 {
			md["rank"] = vector.NewFloat(rw.key)
		}
		if err := s.VectorInsertExt(ctx, coll, rw.id, v, VectorInsertOpts{Metadata: md}); err != nil {
			t.Fatalf("VectorInsertExt %d: %v", rw.id, err)
		}
	}
}

// groundTruthOrder sorts rows by the (value,id) total order for direction desc — the
// INDEPENDENT expected page sequence (no engine code involved).
func groundTruthOrder(rows []orderRow, desc bool) []uint64 {
	cp := append([]orderRow(nil), rows...)
	sort.Slice(cp, func(i, j int) bool {
		return vector.OrderLess(cp[i].key, cp[i].id, cp[j].key, cp[j].id, desc)
	})
	out := make([]uint64, len(cp))
	for i, r := range cp {
		out[i] = r.id
	}
	return out
}

// pageAllDenseOrder pages a dense order_by scroll to exhaustion, returning the
// concatenated id sequence. It asserts the v2 exhaustion rule (full page ⇒ cursor).
func pageAllDenseOrder(t *testing.T, s Store, coll string, ob *vector.OrderBy, limit int) []uint64 {
	t.Helper()
	ctx := context.Background()
	cursor := ""
	var ids []uint64
	for page := 0; ; page++ {
		docs, _, next, err := s.VectorScroll(ctx, coll, VectorFilter{}, limit, VectorScrollOpts{Cursor: cursor, OrderBy: ob})
		if err != nil {
			t.Fatalf("order_by scroll page %d: %v", page, err)
		}
		for _, d := range docs {
			ids = append(ids, d.ID)
		}
		if len(docs) == limit {
			if next == "" {
				t.Fatalf("page %d full (len=%d) but next_cursor empty", page, limit)
			}
		} else if next != "" {
			t.Fatalf("page %d short (len=%d) but next_cursor=%q", page, len(docs), next)
		}
		if next == "" {
			return ids
		}
		cursor = next
		if page > len(ids)+1000 {
			t.Fatalf("pagination did not terminate")
		}
	}
}

// TestDenseOrderByAscDescPaged: paged order_by ASC and DESC over a numeric field, on
// P=4, equals the INDEPENDENT (value,id) ground truth exactly (globally ordered,
// gap-free, dup-free).
func TestDenseOrderByAscDescPaged(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	const (
		coll = "ob_asc_desc"
		P    = 4
		N    = 200
		L    = 23
	)
	// Distinct ids; keys with deliberate VALUE TIES (key = id/3) so the id tiebreak
	// is exercised across partitions.
	rows := make([]orderRow, N)
	for i := 0; i < N; i++ {
		rows[i] = orderRow{id: uint64(i + 1), key: float64((i + 1) / 3)}
	}
	seedDenseOrder(t, s, coll, P, rows, 0)

	for _, desc := range []bool{false, true} {
		ob := &vector.OrderBy{Key: "rank", Desc: desc}
		got := pageAllDenseOrder(t, s, coll, ob, L)
		want := groundTruthOrder(rows, desc)
		if len(got) != len(want) {
			t.Fatalf("desc=%v: paged %d ids, want %d", desc, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("desc=%v: paged[%d]=%d, want %d (full got=%v)", desc, i, got[i], want[i], got)
			}
		}
	}
}

// TestDenseOrderByPartitionInvariance: the P=4 page sequence must EQUAL the P=1
// sequence (partition count is invisible to the ordered result).
func TestDenseOrderByPartitionInvariance(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	const (
		N = 150
		L = 19
	)
	rows := make([]orderRow, N)
	for i := 0; i < N; i++ {
		rows[i] = orderRow{id: uint64(i + 1), key: float64((i * 7) % 50)} // many ties
	}
	seedDenseOrder(t, s, "ob_p1", 1, rows, 0)
	seedDenseOrder(t, s, "ob_p4", 4, rows, 0)

	ob := &vector.OrderBy{Key: "rank"}
	p1 := pageAllDenseOrder(t, s, "ob_p1", ob, L)
	p4 := pageAllDenseOrder(t, s, "ob_p4", ob, L)
	if len(p1) != len(p4) {
		t.Fatalf("P1 paged %d, P4 paged %d", len(p1), len(p4))
	}
	for i := range p1 {
		if p1[i] != p4[i] {
			t.Fatalf("partition variance at %d: P1=%d P4=%d", i, p1[i], p4[i])
		}
	}
	// And both equal the ground truth.
	want := groundTruthOrder(rows, false)
	for i := range want {
		if p4[i] != want[i] {
			t.Fatalf("P4[%d]=%d want %d", i, p4[i], want[i])
		}
	}
}

// TestDenseOrderByMissingFieldExcluded: ids inserted WITHOUT the order field never
// appear in an order_by scroll.
func TestDenseOrderByMissingFieldExcluded(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	const (
		coll = "ob_missing"
		P    = 4
		N    = 120
		L    = 13
	)
	rows := make([]orderRow, N)
	for i := 0; i < N; i++ {
		rows[i] = orderRow{id: uint64(i + 1), key: float64(N - i)}
	}
	// Every 5th inserted point omits "rank".
	seedDenseOrder(t, s, coll, P, rows, 5)

	ob := &vector.OrderBy{Key: "rank"}
	got := pageAllDenseOrder(t, s, coll, ob, L)
	// The returned set must be a strict subset of all ids; none may be missing-field.
	// Reconstruct which ids carried the field by re-deriving the same skip pattern is
	// fragile (seed shuffles), so instead assert the count is < N and the sequence is
	// globally (value,id)-ordered over whatever survived.
	if len(got) >= N {
		t.Fatalf("missing-field points were NOT excluded: paged %d >= N=%d", len(got), N)
	}
	if len(got) == 0 {
		t.Fatalf("excluded everything")
	}
	// Whatever survived must be in ascending key order (verify via Get of the key).
	prevKey, prevID, have := 0.0, uint64(0), false
	ctx := context.Background()
	for _, id := range got {
		_, _, md, _, _, err := s.VectorGet(ctx, coll, id, false, true)
		if err != nil {
			t.Fatalf("get %d: %v", id, err)
		}
		k, ok := vector.OrderKey(md, "rank", false)
		if !ok {
			t.Fatalf("paged id %d has NO rank field (missing-field leaked into order_by)", id)
		}
		if have && !vector.OrderLess(prevKey, prevID, k, id, false) {
			t.Fatalf("not globally ordered at id %d: (%v,%d) !< (%v,%d)", id, prevKey, prevID, k, id)
		}
		prevKey, prevID, have = k, id, true
	}
}

// TestDenseOrderByDatetime: a datetime field (stored as unix-ms int) orders
// chronologically.
func TestDenseOrderByDatetime(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	const coll = "ob_dt"
	if err := s.CreateCollection(ctx, coll, VectorConfig{
		Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2, Partitions: 4,
	}); err != nil {
		t.Fatal(err)
	}
	// ms timestamps, inserted in scrambled order.
	base := int64(1_700_000_000_000)
	type pt struct {
		id uint64
		ms int64
	}
	pts := []pt{{3, base + 3000}, {1, base + 1000}, {5, base + 5000}, {2, base + 2000}, {4, base + 4000}}
	for _, p := range pts {
		if err := s.VectorInsertExt(ctx, coll, p.id, []float32{float32(p.id), 0, 0, 0},
			VectorInsertOpts{Metadata: VectorMetadata{"ts": vector.NewInt(p.ms)}}); err != nil {
			t.Fatal(err)
		}
	}
	ob := &vector.OrderBy{Key: "ts", IsDatetime: true}
	got := pageAllDenseOrder(t, s, coll, ob, 2)
	want := []uint64{1, 2, 3, 4, 5} // chronological
	if len(got) != len(want) {
		t.Fatalf("paged %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("datetime order paged %v, want %v", got, want)
		}
	}
}

// TestDenseOrderByStartFrom: start_from skips earlier values (inclusive value bound).
func TestDenseOrderByStartFrom(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	const (
		coll = "ob_start"
		P    = 4
		N    = 100
	)
	rows := make([]orderRow, N)
	for i := 0; i < N; i++ {
		rows[i] = orderRow{id: uint64(i + 1), key: float64(i + 1)} // key == id, no ties
	}
	seedDenseOrder(t, s, coll, P, rows, 0)

	ob := &vector.OrderBy{Key: "rank", StartFrom: 40, HasStart: true}
	got := pageAllDenseOrder(t, s, coll, ob, 16)
	// Inclusive: first id has key 40.
	if got[0] != 40 {
		t.Fatalf("start_from=40 first id = %d, want 40 (inclusive)", got[0])
	}
	if len(got) != N-39 { // keys 40..100
		t.Fatalf("start_from=40 paged %d ids, want %d", len(got), N-39)
	}
	for i, id := range got {
		if id != uint64(40+i) {
			t.Fatalf("start_from paged[%d]=%d want %d", i, id, 40+i)
		}
	}
}

// TestDenseOrderByCursorMismatchRejected: a cursor issued for one direction/key must
// be rejected loud when the request's order_by changes (direction flip, key change,
// dropping order_by, or a v1 cursor on an order_by request).
func TestDenseOrderByCursorMismatchRejected(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	const coll = "ob_mismatch"
	rows := make([]orderRow, 60)
	for i := range rows {
		rows[i] = orderRow{id: uint64(i + 1), key: float64(i + 1)}
	}
	seedDenseOrder(t, s, coll, 4, rows, 0)

	// Get a valid ASC cursor from page 1.
	ascOB := &vector.OrderBy{Key: "rank"}
	_, _, next, err := s.VectorScroll(ctx, coll, VectorFilter{}, 10, VectorScrollOpts{OrderBy: ascOB})
	if err != nil || next == "" {
		t.Fatalf("page 1: err=%v next=%q", err, next)
	}

	// Flip direction → mismatch.
	if _, _, _, err := s.VectorScroll(ctx, coll, VectorFilter{}, 10,
		VectorScrollOpts{Cursor: next, OrderBy: &vector.OrderBy{Key: "rank", Desc: true}}); err == nil {
		t.Fatal("direction flip mid-pagination was NOT rejected")
	}
	// Change key → mismatch.
	if _, _, _, err := s.VectorScroll(ctx, coll, VectorFilter{}, 10,
		VectorScrollOpts{Cursor: next, OrderBy: &vector.OrderBy{Key: "other"}}); err == nil {
		t.Fatal("order_by key change mid-pagination was NOT rejected")
	}
	// Drop order_by entirely (v2 cursor on a no-order_by request) → mismatch.
	if _, _, _, err := s.VectorScroll(ctx, coll, VectorFilter{}, 10,
		VectorScrollOpts{Cursor: next}); err == nil {
		t.Fatal("dropping order_by mid-pagination (v2 cursor, no order_by) was NOT rejected")
	}
	// A v1 (id) cursor on an order_by request → mismatch.
	v1 := ops.EncodeScrollCursor(5)
	if _, _, _, err := s.VectorScroll(ctx, coll, VectorFilter{}, 10,
		VectorScrollOpts{Cursor: v1, OrderBy: ascOB}); err == nil {
		t.Fatal("v1 cursor on an order_by request was NOT rejected")
	}
}

// TestDenseOrderByMergeNotByID is the TEETH test: it FAILS if the fan-out merge (or
// the engine sort) orders by id instead of (value,id). The order field is the REVERSE
// of id (key = -id), so a correct order_by ASC scroll returns ids DESCENDING; an
// id-sorted merge would return them ascending and the very first page would be wrong.
func TestDenseOrderByMergeNotByID(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	const (
		coll = "ob_teeth"
		P    = 4
		N    = 80
		L    = 11
	)
	rows := make([]orderRow, N)
	for i := 0; i < N; i++ {
		rows[i] = orderRow{id: uint64(i + 1), key: -float64(i + 1)} // value order REVERSES id order
	}
	seedDenseOrder(t, s, coll, P, rows, 0)

	ob := &vector.OrderBy{Key: "rank"} // ASC by key ⇒ DESC by id
	got := pageAllDenseOrder(t, s, coll, ob, L)
	want := groundTruthOrder(rows, false) // N, N-1, ..., 1
	if len(got) != N {
		t.Fatalf("paged %d, want %d", len(got), N)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("merge appears to sort by id, not (value,id): paged[%d]=%d want %d", i, got[i], want[i])
		}
	}
	// Sanity: the result is strictly DESCENDING in id (the opposite of an id-sort).
	for i := 1; i < len(got); i++ {
		if got[i] >= got[i-1] {
			t.Fatalf("expected descending ids (value=-id, asc), got %d >= %d at %d", got[i], got[i-1], i)
		}
	}
}

// TestScrollArgsOrderByteIdenticalWhenAbsent proves the additive order_by codec block
// is BYTE-IDENTICAL to the legacy cursor codec when order==nil — the no-order_by wire
// is zero-overhead. Covers the no-opts/no-cursor, opts-only, and opts+cursor forms.
func TestScrollArgsOrderByteIdenticalWhenAbsent(t *testing.T) {
	filter := vector.Filter{}
	cases := []struct {
		rc, opa  uint8
		afterID  uint64
		hasAfter bool
	}{
		{0, 0, 0, false}, // no opts, no cursor (legacy plain)
		{1, 0, 0, false}, // opts only
		{1, 2, 0, false}, // opts only
		{0, 0, 7, true},  // cursor only
		{1, 1, 99, true}, // opts + cursor
	}
	for _, c := range cases {
		legacy := ops.EncodeScrollArgsCursorBounded("col", filter, 25, c.rc, c.opa, c.afterID, c.hasAfter, 0)
		withNil := ops.EncodeScrollArgsOrderBounded("col", filter, 25, c.rc, c.opa, c.afterID, c.hasAfter, nil, 0)
		if len(legacy) != len(withNil) {
			t.Fatalf("case %+v: len %d != %d (order block leaked when absent)", c, len(withNil), len(legacy))
		}
		for i := range legacy {
			if legacy[i] != withNil[i] {
				t.Fatalf("case %+v: byte %d differs (%d != %d) — NOT byte-identical at order==nil", c, i, withNil[i], legacy[i])
			}
		}
	}
}

// TestScrollArgsOrderRoundTrip proves the order block survives encode→decode with all
// fields, and that a present order block decodes to a non-nil *ScrollOrder.
func TestScrollArgsOrderRoundTrip(t *testing.T) {
	order := &ops.ScrollOrder{
		Key: "rank", Desc: true, IsDatetime: true,
		StartFrom: 12.5, HasStart: true,
		ResumeKey: 99.5, HasResume: true,
	}
	args := ops.EncodeScrollArgsOrderBounded("col", vector.Filter{}, 7, 1, 2, 42, true, order, 0)
	col, _, limit, rc, opa, afterID, hasAfter, got, err := ops.DecodeScrollArgsOrder(args)
	if err != nil {
		t.Fatal(err)
	}
	if col != "col" || limit != 7 || rc != 1 || opa != 2 || afterID != 42 || !hasAfter {
		t.Fatalf("base/trailer round-trip mismatch: col=%q limit=%d rc=%d opa=%d afterID=%d hasAfter=%v", col, limit, rc, opa, afterID, hasAfter)
	}
	if got == nil {
		t.Fatal("order block decoded as nil")
	}
	if !reflect.DeepEqual(got, order) {
		t.Fatalf("order round-trip: got %+v want %+v", *got, *order)
	}
	// A nil order block decodes back to nil.
	_, _, _, _, _, _, _, gotNil, err := ops.DecodeScrollArgsOrder(ops.EncodeScrollArgsOrderBounded("c", vector.Filter{}, 1, 0, 0, 0, false, nil, 0))
	if err != nil {
		t.Fatal(err)
	}
	if gotNil != nil {
		t.Fatalf("nil order block decoded as %+v, want nil", gotNil)
	}
}

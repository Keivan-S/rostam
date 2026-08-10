// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"math/rand"
	"reflect"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// This file mirrors embedded_order_by_scroll_test.go (the DENSE order_by scroll
// tests) for the NAMED + MV families: paged order_by ASC/DESC equals an independent
// (value,id) ground truth (globally ordered, gap-free, dup-free), partition-count
// invariance, missing-field EXCLUDE, datetime, start_from, the codec byte-identical
// at nil, and the cursor direction/key/v1 mismatch rejection. It reuses orderRow /
// groundTruthOrder from embedded_order_by_scroll_test.go.

// ----- named helpers -----

// seedNamedOrder creates a P-partition named collection and inserts each id with a
// numeric "rank" shared-payload field = key, in RANDOM order (so a correct global
// (value,id) page sequence proves the per-call sort + merge, not insert order). When
// missingEvery>0, every missingEvery-th inserted point omits "rank" (missing-field
// EXCLUDE policy).
func seedNamedOrder(t *testing.T, s Store, coll string, P int, rows []orderRow, missingEvery int) {
	t.Helper()
	ctx := context.Background()
	cfg := map[string]NamedVectorParams{"title": {Dim: 4, Metric: vector.L2}}
	if err := s.VectorNamedCreateCollection(ctx, coll, cfg, P); err != nil {
		t.Fatalf("VectorNamedCreateCollection %q (P=%d): %v", coll, P, err)
	}
	order := append([]orderRow(nil), rows...)
	rand.New(rand.NewSource(99)).Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
	for n, rw := range order {
		vecs := map[string][]float32{"title": {float32(rw.id), 0, 0, 0}}
		md := VectorMetadata{}
		if missingEvery <= 0 || n%missingEvery != 0 {
			md["rank"] = vector.NewFloat(rw.key)
		}
		if err := s.VectorNamedInsert(ctx, coll, rw.id, vecs, md, 0); err != nil {
			t.Fatalf("VectorNamedInsert %d: %v", rw.id, err)
		}
	}
}

// pageAllNamedOrder pages a named order_by scroll to exhaustion, returning the
// concatenated id sequence. Asserts the v2 exhaustion rule (full page ⇒ cursor).
func pageAllNamedOrder(t *testing.T, s Store, coll string, ob *vector.OrderBy, limit int) []uint64 {
	t.Helper()
	ctx := context.Background()
	cursor := ""
	var ids []uint64
	for page := 0; ; page++ {
		docs, next, err := s.VectorNamedScrollExt(ctx, coll, VectorFilter{}, limit, cursor, NamedScrollOpts{OrderBy: ob})
		if err != nil {
			t.Fatalf("named order_by scroll page %d: %v", page, err)
		}
		for _, d := range docs {
			ids = append(ids, d.ID)
		}
		if len(docs) == limit {
			if next == "" {
				t.Fatalf("named page %d full (len=%d) but next_cursor empty", page, limit)
			}
		} else if next != "" {
			t.Fatalf("named page %d short (len=%d) but next_cursor=%q", page, len(docs), next)
		}
		if next == "" {
			return ids
		}
		cursor = next
		if page > len(ids)+1000 {
			t.Fatalf("named pagination did not terminate")
		}
	}
}

// ----- MV helpers -----

// seedMVOrder creates a P-partition MV collection and inserts each id with a numeric
// "rank" payload field = key, in RANDOM order. missingEvery>0 omits "rank" on every
// missingEvery-th inserted doc.
func seedMVOrder(t *testing.T, s Store, coll string, P int, rows []orderRow, missingEvery int) {
	t.Helper()
	ctx := context.Background()
	if err := s.VectorMVCreateCollection(ctx, coll, MultiVectorConfig{Dim: 4, Partitions: P}); err != nil {
		t.Fatalf("VectorMVCreateCollection %q (P=%d): %v", coll, P, err)
	}
	order := append([]orderRow(nil), rows...)
	rand.New(rand.NewSource(99)).Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
	for n, rw := range order {
		md := VectorMetadata{}
		if missingEvery <= 0 || n%missingEvery != 0 {
			md["rank"] = vector.NewFloat(rw.key)
		}
		if err := s.VectorMVAdd(ctx, coll, rw.id, [][]float32{mvTokenAt(int(rw.id))}, md); err != nil {
			t.Fatalf("VectorMVAdd %d: %v", rw.id, err)
		}
	}
}

// pageAllMVOrder pages an MV order_by scroll to exhaustion, returning the concatenated
// id sequence. Asserts the v2 exhaustion rule.
func pageAllMVOrder(t *testing.T, s Store, coll string, ob *vector.OrderBy, limit int) []uint64 {
	t.Helper()
	ctx := context.Background()
	cursor := ""
	var ids []uint64
	for page := 0; ; page++ {
		docs, _, next, err := s.VectorMVScrollExt(ctx, coll, VectorFilter{}, limit, cursor, MVScrollOpts{OrderBy: ob})
		if err != nil {
			t.Fatalf("MV order_by scroll page %d: %v", page, err)
		}
		for _, d := range docs {
			ids = append(ids, d.ID)
		}
		if len(docs) == limit {
			if next == "" {
				t.Fatalf("MV page %d full (len=%d) but next_cursor empty", page, limit)
			}
		} else if next != "" {
			t.Fatalf("MV page %d short (len=%d) but next_cursor=%q", page, len(docs), next)
		}
		if next == "" {
			return ids
		}
		cursor = next
		if page > len(ids)+1000 {
			t.Fatalf("MV pagination did not terminate")
		}
	}
}

func makeOrderRows(n int, keyFn func(i int) float64) []orderRow {
	rows := make([]orderRow, n)
	for i := 0; i < n; i++ {
		rows[i] = orderRow{id: uint64(i + 1), key: keyFn(i)}
	}
	return rows
}

// ===== NAMED tests =====

// TestNamedOrderByAscDescPaged: paged order_by ASC and DESC over a numeric shared-payload
// field, on P=4, equals the INDEPENDENT (value,id) ground truth exactly.
func TestNamedOrderByAscDescPaged(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	const (
		coll = "nob_asc_desc"
		P    = 4
		N    = 200
		L    = 23
	)
	rows := makeOrderRows(N, func(i int) float64 { return float64((i + 1) / 3) }) // value ties
	seedNamedOrder(t, s, coll, P, rows, 0)

	for _, desc := range []bool{false, true} {
		ob := &vector.OrderBy{Key: "rank", Desc: desc}
		got := pageAllNamedOrder(t, s, coll, ob, L)
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

// TestNamedOrderByPartitionInvariance: the P=4 page sequence equals the P=1 sequence.
func TestNamedOrderByPartitionInvariance(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	const (
		N = 150
		L = 19
	)
	rows := makeOrderRows(N, func(i int) float64 { return float64((i * 7) % 50) }) // many ties
	seedNamedOrder(t, s, "nob_p1", 1, rows, 0)
	seedNamedOrder(t, s, "nob_p4", 4, rows, 0)

	ob := &vector.OrderBy{Key: "rank"}
	p1 := pageAllNamedOrder(t, s, "nob_p1", ob, L)
	p4 := pageAllNamedOrder(t, s, "nob_p4", ob, L)
	if len(p1) != len(p4) {
		t.Fatalf("P1 paged %d, P4 paged %d", len(p1), len(p4))
	}
	for i := range p1 {
		if p1[i] != p4[i] {
			t.Fatalf("named partition variance at %d: P1=%d P4=%d", i, p1[i], p4[i])
		}
	}
	want := groundTruthOrder(rows, false)
	for i := range want {
		if p4[i] != want[i] {
			t.Fatalf("named P4[%d]=%d want %d", i, p4[i], want[i])
		}
	}
}

// TestNamedOrderByMissingFieldExcluded: ids inserted WITHOUT the order field never
// appear in an order_by scroll, and survivors are globally ordered.
func TestNamedOrderByMissingFieldExcluded(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	const (
		coll = "nob_missing"
		P    = 4
		N    = 120
		L    = 13
	)
	rows := makeOrderRows(N, func(i int) float64 { return float64(N - i) })
	seedNamedOrder(t, s, coll, P, rows, 5)

	ob := &vector.OrderBy{Key: "rank"}
	got := pageAllNamedOrder(t, s, coll, ob, L)
	if len(got) >= N {
		t.Fatalf("named missing-field points were NOT excluded: paged %d >= N=%d", len(got), N)
	}
	if len(got) == 0 {
		t.Fatalf("excluded everything")
	}
	prevKey, prevID, have := 0.0, uint64(0), false
	ctx := context.Background()
	for _, id := range got {
		_, _, md, _, err := s.VectorNamedGet(ctx, coll, id, false, true)
		if err != nil {
			t.Fatalf("named get %d: %v", id, err)
		}
		k, ok := vector.OrderKey(md, "rank", false)
		if !ok {
			t.Fatalf("named paged id %d has NO rank field (missing-field leaked)", id)
		}
		if have && !vector.OrderLess(prevKey, prevID, k, id, false) {
			t.Fatalf("named not globally ordered at id %d: (%v,%d) !< (%v,%d)", id, prevKey, prevID, k, id)
		}
		prevKey, prevID, have = k, id, true
	}
}

// TestNamedOrderByDatetime: a datetime field (unix-ms int) orders chronologically.
func TestNamedOrderByDatetime(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	const coll = "nob_dt"
	cfg := map[string]NamedVectorParams{"title": {Dim: 4, Metric: vector.L2}}
	if err := s.VectorNamedCreateCollection(ctx, coll, cfg, 4); err != nil {
		t.Fatal(err)
	}
	base := int64(1_700_000_000_000)
	type pt struct {
		id uint64
		ms int64
	}
	pts := []pt{{3, base + 3000}, {1, base + 1000}, {5, base + 5000}, {2, base + 2000}, {4, base + 4000}}
	for _, p := range pts {
		vecs := map[string][]float32{"title": {float32(p.id), 0, 0, 0}}
		if err := s.VectorNamedInsert(ctx, coll, p.id, vecs, VectorMetadata{"ts": vector.NewInt(p.ms)}, 0); err != nil {
			t.Fatal(err)
		}
	}
	ob := &vector.OrderBy{Key: "ts", IsDatetime: true}
	got := pageAllNamedOrder(t, s, coll, ob, 2)
	want := []uint64{1, 2, 3, 4, 5}
	if len(got) != len(want) {
		t.Fatalf("named datetime paged %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("named datetime order paged %v, want %v", got, want)
		}
	}
}

// TestNamedOrderByStartFrom: start_from skips earlier values (inclusive value bound).
func TestNamedOrderByStartFrom(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	const (
		coll = "nob_start"
		P    = 4
		N    = 100
	)
	rows := makeOrderRows(N, func(i int) float64 { return float64(i + 1) }) // key == id
	seedNamedOrder(t, s, coll, P, rows, 0)

	ob := &vector.OrderBy{Key: "rank", StartFrom: 40, HasStart: true}
	got := pageAllNamedOrder(t, s, coll, ob, 16)
	if got[0] != 40 {
		t.Fatalf("named start_from=40 first id = %d, want 40 (inclusive)", got[0])
	}
	if len(got) != N-39 {
		t.Fatalf("named start_from=40 paged %d ids, want %d", len(got), N-39)
	}
	for i, id := range got {
		if id != uint64(40+i) {
			t.Fatalf("named start_from paged[%d]=%d want %d", i, id, 40+i)
		}
	}
}

// TestNamedOrderByMergeNotByID is the named TEETH test: it FAILS if the merge/engine
// sorts by id instead of (value,id). key = -id ⇒ a correct ASC order_by returns ids
// DESCENDING; an id-sorted merge returns them ascending.
func TestNamedOrderByMergeNotByID(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	const (
		coll = "nob_teeth"
		P    = 4
		N    = 80
		L    = 11
	)
	rows := makeOrderRows(N, func(i int) float64 { return -float64(i + 1) })
	seedNamedOrder(t, s, coll, P, rows, 0)

	ob := &vector.OrderBy{Key: "rank"}
	got := pageAllNamedOrder(t, s, coll, ob, L)
	want := groundTruthOrder(rows, false)
	if len(got) != N {
		t.Fatalf("named teeth paged %d, want %d", len(got), N)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("named merge appears to sort by id: paged[%d]=%d want %d", i, got[i], want[i])
		}
	}
	for i := 1; i < len(got); i++ {
		if got[i] >= got[i-1] {
			t.Fatalf("named expected descending ids, got %d >= %d at %d", got[i], got[i-1], i)
		}
	}
}

// TestNamedOrderByCursorMismatchRejected: a cursor issued for one direction/key is
// rejected when the request's order_by changes.
func TestNamedOrderByCursorMismatchRejected(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	const coll = "nob_mismatch"
	rows := makeOrderRows(60, func(i int) float64 { return float64(i + 1) })
	seedNamedOrder(t, s, coll, 4, rows, 0)

	ascOB := &vector.OrderBy{Key: "rank"}
	_, next, err := s.VectorNamedScrollExt(ctx, coll, VectorFilter{}, 10, "", NamedScrollOpts{OrderBy: ascOB})
	if err != nil || next == "" {
		t.Fatalf("named page 1: err=%v next=%q", err, next)
	}
	if _, _, err := s.VectorNamedScrollExt(ctx, coll, VectorFilter{}, 10, next,
		NamedScrollOpts{OrderBy: &vector.OrderBy{Key: "rank", Desc: true}}); err == nil {
		t.Fatal("named direction flip mid-pagination was NOT rejected")
	}
	if _, _, err := s.VectorNamedScrollExt(ctx, coll, VectorFilter{}, 10, next,
		NamedScrollOpts{OrderBy: &vector.OrderBy{Key: "other"}}); err == nil {
		t.Fatal("named order_by key change mid-pagination was NOT rejected")
	}
	if _, _, err := s.VectorNamedScrollExt(ctx, coll, VectorFilter{}, 10, next,
		NamedScrollOpts{}); err == nil {
		t.Fatal("named dropping order_by mid-pagination (v2 cursor) was NOT rejected")
	}
	v1 := ops.EncodeScrollCursor(5)
	if _, _, err := s.VectorNamedScrollExt(ctx, coll, VectorFilter{}, 10, v1,
		NamedScrollOpts{OrderBy: ascOB}); err == nil {
		t.Fatal("named v1 cursor on an order_by request was NOT rejected")
	}
}

// TestNamedOrderByNoOrderUnchanged: a no-order_by named scroll still pages id-ascending
// (the existing path is unaffected by the order_by wiring).
func TestNamedOrderByNoOrderUnchanged(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	const (
		coll = "nob_plain"
		P    = 4
		N    = 60
		L    = 7
	)
	rows := makeOrderRows(N, func(i int) float64 { return float64(N - i) })
	seedNamedOrder(t, s, coll, P, rows, 0)
	ctx := context.Background()
	cursor := ""
	var got []uint64
	for {
		docs, next, err := s.VectorNamedScrollExt(ctx, coll, VectorFilter{}, L, cursor, NamedScrollOpts{})
		if err != nil {
			t.Fatalf("plain named scroll: %v", err)
		}
		for _, d := range docs {
			got = append(got, d.ID)
		}
		if next == "" {
			break
		}
		cursor = next
	}
	if len(got) != N {
		t.Fatalf("plain named scroll paged %d, want %d", len(got), N)
	}
	for i := 1; i < len(got); i++ {
		if got[i] <= got[i-1] {
			t.Fatalf("plain named scroll not id-ascending at %d: %d <= %d", i, got[i], got[i-1])
		}
	}
}

// ===== MV tests =====

func TestMVOrderByAscDescPaged(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	const (
		coll = "mob_asc_desc"
		P    = 4
		N    = 200
		L    = 23
	)
	rows := makeOrderRows(N, func(i int) float64 { return float64((i + 1) / 3) })
	seedMVOrder(t, s, coll, P, rows, 0)

	for _, desc := range []bool{false, true} {
		ob := &vector.OrderBy{Key: "rank", Desc: desc}
		got := pageAllMVOrder(t, s, coll, ob, L)
		want := groundTruthOrder(rows, desc)
		if len(got) != len(want) {
			t.Fatalf("MV desc=%v: paged %d ids, want %d", desc, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("MV desc=%v: paged[%d]=%d, want %d", desc, i, got[i], want[i])
			}
		}
	}
}

func TestMVOrderByPartitionInvariance(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	const (
		N = 150
		L = 19
	)
	rows := makeOrderRows(N, func(i int) float64 { return float64((i * 7) % 50) })
	seedMVOrder(t, s, "mob_p1", 1, rows, 0)
	seedMVOrder(t, s, "mob_p4", 4, rows, 0)

	ob := &vector.OrderBy{Key: "rank"}
	p1 := pageAllMVOrder(t, s, "mob_p1", ob, L)
	p4 := pageAllMVOrder(t, s, "mob_p4", ob, L)
	if len(p1) != len(p4) {
		t.Fatalf("MV P1 paged %d, P4 paged %d", len(p1), len(p4))
	}
	for i := range p1 {
		if p1[i] != p4[i] {
			t.Fatalf("MV partition variance at %d: P1=%d P4=%d", i, p1[i], p4[i])
		}
	}
	want := groundTruthOrder(rows, false)
	for i := range want {
		if p4[i] != want[i] {
			t.Fatalf("MV P4[%d]=%d want %d", i, p4[i], want[i])
		}
	}
}

func TestMVOrderByMissingFieldExcluded(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	const (
		coll = "mob_missing"
		P    = 4
		N    = 120
		L    = 13
	)
	rows := makeOrderRows(N, func(i int) float64 { return float64(N - i) })
	seedMVOrder(t, s, coll, P, rows, 5)

	ob := &vector.OrderBy{Key: "rank"}
	got := pageAllMVOrder(t, s, coll, ob, L)
	if len(got) >= N {
		t.Fatalf("MV missing-field points were NOT excluded: paged %d >= N=%d", len(got), N)
	}
	if len(got) == 0 {
		t.Fatalf("excluded everything")
	}
	// Survivors must be globally (value,id)-ordered. Read the order key back from each
	// scrolled doc's metadata (the order field travels in Metadata).
	ctx := context.Background()
	docs, _, _, err := s.VectorMVScrollExt(ctx, coll, VectorFilter{}, N, "", MVScrollOpts{OrderBy: ob})
	if err != nil {
		t.Fatalf("MV full scroll: %v", err)
	}
	prevKey, prevID, have := 0.0, uint64(0), false
	for _, d := range docs {
		k, ok := vector.OrderKey(d.Metadata, "rank", false)
		if !ok {
			t.Fatalf("MV paged id %d has NO rank field (missing-field leaked)", d.ID)
		}
		if have && !vector.OrderLess(prevKey, prevID, k, d.ID, false) {
			t.Fatalf("MV not globally ordered at id %d", d.ID)
		}
		prevKey, prevID, have = k, d.ID, true
	}
}

func TestMVOrderByDatetime(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	const coll = "mob_dt"
	if err := s.VectorMVCreateCollection(ctx, coll, MultiVectorConfig{Dim: 4, Partitions: 4}); err != nil {
		t.Fatal(err)
	}
	base := int64(1_700_000_000_000)
	type pt struct {
		id uint64
		ms int64
	}
	pts := []pt{{3, base + 3000}, {1, base + 1000}, {5, base + 5000}, {2, base + 2000}, {4, base + 4000}}
	for _, p := range pts {
		if err := s.VectorMVAdd(ctx, coll, p.id, [][]float32{mvTokenAt(int(p.id))}, VectorMetadata{"ts": vector.NewInt(p.ms)}); err != nil {
			t.Fatal(err)
		}
	}
	ob := &vector.OrderBy{Key: "ts", IsDatetime: true}
	got := pageAllMVOrder(t, s, coll, ob, 2)
	want := []uint64{1, 2, 3, 4, 5}
	if len(got) != len(want) {
		t.Fatalf("MV datetime paged %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("MV datetime order paged %v, want %v", got, want)
		}
	}
}

func TestMVOrderByStartFrom(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	const (
		coll = "mob_start"
		P    = 4
		N    = 100
	)
	rows := makeOrderRows(N, func(i int) float64 { return float64(i + 1) })
	seedMVOrder(t, s, coll, P, rows, 0)

	ob := &vector.OrderBy{Key: "rank", StartFrom: 40, HasStart: true}
	got := pageAllMVOrder(t, s, coll, ob, 16)
	if got[0] != 40 {
		t.Fatalf("MV start_from=40 first id = %d, want 40", got[0])
	}
	if len(got) != N-39 {
		t.Fatalf("MV start_from=40 paged %d ids, want %d", len(got), N-39)
	}
	for i, id := range got {
		if id != uint64(40+i) {
			t.Fatalf("MV start_from paged[%d]=%d want %d", i, id, 40+i)
		}
	}
}

func TestMVOrderByMergeNotByID(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	const (
		coll = "mob_teeth"
		P    = 4
		N    = 80
		L    = 11
	)
	rows := makeOrderRows(N, func(i int) float64 { return -float64(i + 1) })
	seedMVOrder(t, s, coll, P, rows, 0)

	ob := &vector.OrderBy{Key: "rank"}
	got := pageAllMVOrder(t, s, coll, ob, L)
	want := groundTruthOrder(rows, false)
	if len(got) != N {
		t.Fatalf("MV teeth paged %d, want %d", len(got), N)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("MV merge appears to sort by id: paged[%d]=%d want %d", i, got[i], want[i])
		}
	}
	for i := 1; i < len(got); i++ {
		if got[i] >= got[i-1] {
			t.Fatalf("MV expected descending ids, got %d >= %d at %d", got[i], got[i-1], i)
		}
	}
}

func TestMVOrderByCursorMismatchRejected(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	const coll = "mob_mismatch"
	rows := makeOrderRows(60, func(i int) float64 { return float64(i + 1) })
	seedMVOrder(t, s, coll, 4, rows, 0)

	ascOB := &vector.OrderBy{Key: "rank"}
	_, _, next, err := s.VectorMVScrollExt(ctx, coll, VectorFilter{}, 10, "", MVScrollOpts{OrderBy: ascOB})
	if err != nil || next == "" {
		t.Fatalf("MV page 1: err=%v next=%q", err, next)
	}
	if _, _, _, err := s.VectorMVScrollExt(ctx, coll, VectorFilter{}, 10, next,
		MVScrollOpts{OrderBy: &vector.OrderBy{Key: "rank", Desc: true}}); err == nil {
		t.Fatal("MV direction flip mid-pagination was NOT rejected")
	}
	if _, _, _, err := s.VectorMVScrollExt(ctx, coll, VectorFilter{}, 10, next,
		MVScrollOpts{OrderBy: &vector.OrderBy{Key: "other"}}); err == nil {
		t.Fatal("MV order_by key change mid-pagination was NOT rejected")
	}
	if _, _, _, err := s.VectorMVScrollExt(ctx, coll, VectorFilter{}, 10, next,
		MVScrollOpts{}); err == nil {
		t.Fatal("MV dropping order_by mid-pagination (v2 cursor) was NOT rejected")
	}
	v1 := ops.EncodeScrollCursor(5)
	if _, _, _, err := s.VectorMVScrollExt(ctx, coll, VectorFilter{}, 10, v1,
		MVScrollOpts{OrderBy: ascOB}); err == nil {
		t.Fatal("MV v1 cursor on an order_by request was NOT rejected")
	}
}

// ===== codec byte-identical-at-nil + round-trip (BOTH families) =====

// TestNamedScrollArgsOrderByteIdenticalWhenAbsent proves EncodeNamedScrollArgsOrder
// with order==nil is BYTE-IDENTICAL to EncodeNamedScrollArgsOpts (zero-overhead wire).
func TestNamedScrollArgsOrderByteIdenticalWhenAbsent(t *testing.T) {
	filter := vector.Filter{}
	cases := []struct {
		rc, opa  uint8
		afterID  uint64
		hasAfter bool
	}{
		{0, 0, 0, false},
		{1, 0, 0, false},
		{1, 2, 0, false},
		{0, 0, 7, true},
		{1, 1, 99, true},
	}
	for _, c := range cases {
		legacy := ops.EncodeNamedScrollArgsOptsBounded("col", filter, 25, c.afterID, c.hasAfter, c.rc, c.opa, 0)
		withNil := ops.EncodeNamedScrollArgsOrderBounded("col", filter, 25, c.afterID, c.hasAfter, c.rc, c.opa, nil, 0)
		if len(legacy) != len(withNil) {
			t.Fatalf("named case %+v: len %d != %d (order block leaked when absent)", c, len(withNil), len(legacy))
		}
		for i := range legacy {
			if legacy[i] != withNil[i] {
				t.Fatalf("named case %+v: byte %d differs — NOT byte-identical at order==nil", c, i)
			}
		}
	}
}

// TestMVScrollArgsOrderByteIdenticalWhenAbsent: the MV mirror.
func TestMVScrollArgsOrderByteIdenticalWhenAbsent(t *testing.T) {
	filter := vector.Filter{}
	cases := []struct {
		rc, opa  uint8
		afterID  uint64
		hasAfter bool
	}{
		{0, 0, 0, false},
		{1, 0, 0, false},
		{1, 2, 0, false},
		{0, 0, 7, true},
		{1, 1, 99, true},
	}
	for _, c := range cases {
		legacy := ops.EncodeMVScrollArgsOptsBounded("col", filter, 25, c.rc, c.opa, c.afterID, c.hasAfter, 0)
		withNil := ops.EncodeMVScrollArgsOrderBounded("col", filter, 25, c.rc, c.opa, c.afterID, c.hasAfter, nil, 0)
		if len(legacy) != len(withNil) {
			t.Fatalf("MV case %+v: len %d != %d (order block leaked when absent)", c, len(withNil), len(legacy))
		}
		for i := range legacy {
			if legacy[i] != withNil[i] {
				t.Fatalf("MV case %+v: byte %d differs — NOT byte-identical at order==nil", c, i)
			}
		}
	}
}

// TestNamedScrollArgsOrderRoundTrip / TestMVScrollArgsOrderRoundTrip prove the order
// block survives encode→decode with all fields for both families.
func TestNamedScrollArgsOrderRoundTrip(t *testing.T) {
	order := &ops.ScrollOrder{Key: "rank", Desc: true, IsDatetime: true, StartFrom: 12.5, HasStart: true, ResumeKey: 99.5, HasResume: true}
	args := ops.EncodeNamedScrollArgsOrderBounded("col", vector.Filter{}, 7, 42, true, 1, 2, order, 0)
	col, _, limit, afterID, hasAfter, rc, opa, got, err := ops.DecodeNamedScrollArgsOrder(args)
	if err != nil {
		t.Fatal(err)
	}
	if col != "col" || limit != 7 || rc != 1 || opa != 2 || afterID != 42 || !hasAfter {
		t.Fatalf("named base/trailer round-trip mismatch: col=%q limit=%d rc=%d opa=%d afterID=%d hasAfter=%v", col, limit, rc, opa, afterID, hasAfter)
	}
	if got == nil || !reflect.DeepEqual(got, order) {
		t.Fatalf("named order round-trip: got %+v want %+v", got, *order)
	}
	_, _, _, _, _, _, _, gotNil, err := ops.DecodeNamedScrollArgsOrder(ops.EncodeNamedScrollArgsOrderBounded("c", vector.Filter{}, 1, 0, false, 0, 0, nil, 0))
	if err != nil {
		t.Fatal(err)
	}
	if gotNil != nil {
		t.Fatalf("named nil order block decoded as %+v, want nil", gotNil)
	}
}

func TestMVScrollArgsOrderRoundTrip(t *testing.T) {
	order := &ops.ScrollOrder{Key: "rank", Desc: true, IsDatetime: true, StartFrom: 12.5, HasStart: true, ResumeKey: 99.5, HasResume: true}
	args := ops.EncodeMVScrollArgsOrderBounded("col", vector.Filter{}, 7, 1, 2, 42, true, order, 0)
	col, _, limit, rc, opa, afterID, hasAfter, got, err := ops.DecodeMVScrollArgsOrder(args)
	if err != nil {
		t.Fatal(err)
	}
	if col != "col" || limit != 7 || rc != 1 || opa != 2 || afterID != 42 || !hasAfter {
		t.Fatalf("MV base/trailer round-trip mismatch: col=%q limit=%d rc=%d opa=%d afterID=%d hasAfter=%v", col, limit, rc, opa, afterID, hasAfter)
	}
	if got == nil || !reflect.DeepEqual(got, order) {
		t.Fatalf("MV order round-trip: got %+v want %+v", got, *order)
	}
	_, _, _, _, _, _, _, gotNil, err := ops.DecodeMVScrollArgsOrder(ops.EncodeMVScrollArgsOrderBounded("c", vector.Filter{}, 1, 0, 0, 0, false, nil, 0))
	if err != nil {
		t.Fatal(err)
	}
	if gotNil != nil {
		t.Fatalf("MV nil order block decoded as %+v, want nil", gotNil)
	}
}

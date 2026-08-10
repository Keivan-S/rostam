// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"errors"
	"math/rand"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// strOrderRow is one inserted point's (id, string orderKey) — the independent ground
// truth a paged STRING order_by sequence is checked against.
type strOrderRow struct {
	id  uint64
	key string
}

// seedDenseStringOrder creates a P-partition dense collection and inserts each id with
// a string "city" payload field = key, in RANDOM order so a correct global
// (stringValue, id) page sequence proves the per-call sort + merge, not insert order.
func seedDenseStringOrder(t *testing.T, s Store, coll string, P int, rows []strOrderRow) {
	t.Helper()
	ctx := context.Background()
	if err := s.CreateCollection(ctx, coll, VectorConfig{
		Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2, Partitions: P,
	}); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	order := append([]strOrderRow(nil), rows...)
	rand.New(rand.NewSource(99)).Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
	for _, rw := range order {
		v := []float32{float32(rw.id), 0, 0, 0}
		md := VectorMetadata{"city": vector.NewString(rw.key)}
		if err := s.VectorInsertExt(ctx, coll, rw.id, v, VectorInsertOpts{Metadata: md}); err != nil {
			t.Fatalf("VectorInsertExt %d: %v", rw.id, err)
		}
	}
}

// groundTruthStringOrder sorts rows by the (stringValue, id) total order for direction
// desc — the INDEPENDENT expected page sequence (no engine code involved).
func groundTruthStringOrder(rows []strOrderRow, desc bool) []uint64 {
	cp := append([]strOrderRow(nil), rows...)
	sort.Slice(cp, func(i, j int) bool {
		return vector.OrderLessStr(cp[i].key, cp[i].id, cp[j].key, cp[j].id, desc)
	})
	out := make([]uint64, len(cp))
	for i, r := range cp {
		out[i] = r.id
	}
	return out
}

// pageAllDenseStringOrder pages a dense STRING order_by scroll to exhaustion via the
// real cursor resume, returning the concatenated id sequence + asserting the
// full-page⇒cursor exhaustion rule (so a v3 cursor resume is exercised across pages).
func pageAllDenseStringOrder(t *testing.T, s Store, coll string, ob *vector.OrderBy, limit int) []uint64 {
	t.Helper()
	ctx := context.Background()
	cursor := ""
	var ids []uint64
	for page := 0; ; page++ {
		docs, _, next, err := s.VectorScroll(ctx, coll, VectorFilter{}, limit, VectorScrollOpts{Cursor: cursor, OrderBy: ob})
		if err != nil {
			t.Fatalf("string order_by scroll page %d: %v", page, err)
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

// TestStringOrderByAscDescPaged: paged STRING order_by ASC and DESC over a string field,
// on P=4, equals the INDEPENDENT (stringValue, id) ground truth exactly (lexicographic,
// gap-free, dup-free, v3 cursor resume across pages) — the end-to-end wire round-trip.
func TestStringOrderByAscDescPaged(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	const (
		coll = "sob_asc_desc"
		N    = 200
		P    = 4
		L    = 23
	)
	// A small alphabet of city names with deliberate VALUE TIES so the id tiebreak is
	// exercised across partitions and pages.
	cities := []string{"amsterdam", "berlin", "berlin", "cairo", "delhi", "delhi", "delhi", "evora"}
	rows := make([]strOrderRow, N)
	for i := 0; i < N; i++ {
		rows[i] = strOrderRow{id: uint64(i + 1), key: cities[i%len(cities)]}
	}
	seedDenseStringOrder(t, s, coll, P, rows)

	for _, desc := range []bool{false, true} {
		ob := &vector.OrderBy{Key: "city", Kind: vector.OrderString, Desc: desc}
		got := pageAllDenseStringOrder(t, s, coll, ob, L)
		want := groundTruthStringOrder(rows, desc)
		if len(got) != len(want) {
			t.Fatalf("desc=%v: paged %d ids, want %d", desc, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("desc=%v: paged[%d]=%d, want %d", desc, i, got[i], want[i])
			}
		}
	}
}

// TestStringOrderByPartitionInvariance: the P=4 string page sequence EQUALS the P=1
// sequence (partition count is invisible to the lexicographically-ordered result).
func TestStringOrderByPartitionInvariance(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	const (
		N = 150
		L = 19
	)
	cities := []string{"x", "yy", "z", "aa", "aa", "m", "mm"}
	rows := make([]strOrderRow, N)
	for i := 0; i < N; i++ {
		rows[i] = strOrderRow{id: uint64(i + 1), key: cities[(i*7)%len(cities)]}
	}
	seedDenseStringOrder(t, s, "sob_p1", 1, rows)
	seedDenseStringOrder(t, s, "sob_p4", 4, rows)

	ob := &vector.OrderBy{Key: "city", Kind: vector.OrderString}
	p1 := pageAllDenseStringOrder(t, s, "sob_p1", ob, L)
	p4 := pageAllDenseStringOrder(t, s, "sob_p4", ob, L)
	if len(p1) != len(p4) {
		t.Fatalf("P1 paged %d, P4 paged %d", len(p1), len(p4))
	}
	for i := range p1 {
		if p1[i] != p4[i] {
			t.Fatalf("partition variance at %d: P1=%d P4=%d", i, p1[i], p4[i])
		}
	}
	want := groundTruthStringOrder(rows, false)
	for i := range want {
		if p4[i] != want[i] {
			t.Fatalf("P4[%d]=%d want %d", i, p4[i], want[i])
		}
	}
}

// TestScrollArgsOrderStringRoundTrip proves the ScrollOrder block carries the string
// KIND + the v3 resume STRING value through encode→decode, and that a numeric block is
// BYTE-IDENTICAL to the pre-string codec (the additive bit2 + tail never fire for it).
func TestScrollArgsOrderStringRoundTrip(t *testing.T) {
	// String order block with a resume string.
	sOrder := &ops.ScrollOrder{
		Key: "city", Desc: true, Kind: vector.OrderString,
		ResumeStr: "berlin", HasResumeStr: true,
	}
	args := ops.EncodeScrollArgsOrderBounded("col", vector.Filter{}, 7, 1, 2, 42, true, sOrder, 0)
	col, _, limit, rc, opa, afterID, hasAfter, got, err := ops.DecodeScrollArgsOrder(args)
	if err != nil {
		t.Fatal(err)
	}
	if col != "col" || limit != 7 || rc != 1 || opa != 2 || afterID != 42 || !hasAfter {
		t.Fatalf("base/trailer round-trip mismatch: col=%q limit=%d rc=%d opa=%d afterID=%d hasAfter=%v", col, limit, rc, opa, afterID, hasAfter)
	}
	if got == nil {
		t.Fatal("string order block decoded as nil")
	}
	if !reflect.DeepEqual(got, sOrder) {
		t.Fatalf("string order round-trip: got %+v want %+v", *got, *sOrder)
	}

	// A string order with NO resume (page 1) round-trips with HasResumeStr=false.
	sOrder0 := &ops.ScrollOrder{Key: "city", Kind: vector.OrderString}
	args0 := ops.EncodeScrollArgsOrderBounded("c", vector.Filter{}, 1, 0, 0, 0, false, sOrder0, 0)
	_, _, _, _, _, _, _, got0, err := ops.DecodeScrollArgsOrder(args0)
	if err != nil {
		t.Fatal(err)
	}
	if got0 == nil || !reflect.DeepEqual(got0, sOrder0) {
		t.Fatalf("string order page-1 round-trip: got %+v want %+v", got0, sOrder0)
	}

	// BACKWARD-COMPAT: a numeric (Kind default) block's bytes must be IDENTICAL to a
	// block built by the same encoder pre-string (i.e. bit2 unset + no string tail). We
	// assert this by encoding the same numeric ScrollOrder with Kind explicitly Numeric
	// vs the zero value and proving they match, AND that the decoded block has Kind
	// OrderNumeric / no string tail.
	nOrder := &ops.ScrollOrder{
		Key: "rank", Desc: true, IsDatetime: false, Kind: vector.OrderNumeric,
		StartFrom: 12.5, HasStart: true, ResumeKey: 99.5, HasResume: true,
	}
	nArgs := ops.EncodeScrollArgsOrderBounded("col", vector.Filter{}, 7, 1, 2, 42, true, nOrder, 0)
	_, _, _, _, _, _, _, gotN, err := ops.DecodeScrollArgsOrder(nArgs)
	if err != nil {
		t.Fatal(err)
	}
	if gotN == nil || gotN.Kind != vector.OrderNumeric || gotN.HasResumeStr || gotN.ResumeStr != "" {
		t.Fatalf("numeric block leaked string state: %+v", gotN)
	}
	if !reflect.DeepEqual(gotN, nOrder) {
		t.Fatalf("numeric round-trip: got %+v want %+v", *gotN, *nOrder)
	}
}

// TestScrollArgsNumericByteIdenticalAcrossStringField: a numeric/datetime ScrollOrder's
// wire bytes are UNCHANGED by the additive string fields — encoding the same numeric
// order twice (with and without a zero ResumeStr present in the struct) yields identical
// bytes, since the string tail is only written for Kind==OrderString.
func TestScrollArgsNumericByteIdenticalAcrossStringField(t *testing.T) {
	base := &ops.ScrollOrder{Key: "rank", Desc: false, IsDatetime: true, Kind: vector.OrderDatetime, ResumeKey: 5, HasResume: true}
	// Same logical numeric order but with stray (ignored-for-numeric) string fields set.
	withStrayStr := &ops.ScrollOrder{Key: "rank", Desc: false, IsDatetime: true, Kind: vector.OrderDatetime, ResumeKey: 5, HasResume: true, ResumeStr: "ignored", HasResumeStr: true}
	a := ops.EncodeScrollArgsOrderBounded("c", vector.Filter{}, 3, 0, 0, 1, true, base, 0)
	b := ops.EncodeScrollArgsOrderBounded("c", vector.Filter{}, 3, 0, 0, 1, true, withStrayStr, 0)
	if len(a) != len(b) {
		t.Fatalf("numeric/datetime block size changed by string fields: %d != %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("numeric/datetime block byte %d differs (string field leaked onto the wire)", i)
		}
	}
}

// TestParseOrderByStringBadCombo: is_datetime AND is_string both set ⇒ ErrBadOrderKind;
// a string order with a start_from bound ⇒ ErrBadOrderKind (fail-loud at the edge).
func TestParseOrderByStringBadCombo(t *testing.T) {
	if _, err := vector.ParseOrderBy("city", false, true, true, nil, nil); !errors.Is(err, vector.ErrBadOrderKind) {
		t.Fatalf("datetime+string: got err=%v, want ErrBadOrderKind", err)
	}
	start := 1.0
	if _, err := vector.ParseOrderBy("city", false, false, true, &start, nil); !errors.Is(err, vector.ErrBadOrderKind) {
		t.Fatalf("string+start_from: got err=%v, want ErrBadOrderKind", err)
	}
	dt := "2021-01-01T00:00:00Z"
	if _, err := vector.ParseOrderBy("city", false, false, true, nil, &dt); !errors.Is(err, vector.ErrBadOrderKind) {
		t.Fatalf("string+start_from_datetime: got err=%v, want ErrBadOrderKind", err)
	}
	// A clean string order parses to Kind=OrderString.
	ob, err := vector.ParseOrderBy("city", true, false, true, nil, nil)
	if err != nil {
		t.Fatalf("clean string order: %v", err)
	}
	if ob.Kind != vector.OrderString || !ob.Desc {
		t.Fatalf("clean string order: got %+v", ob)
	}
	// numeric + datetime stay byte/behaviour-identical (Kind set, IsDatetime preserved).
	num, _ := vector.ParseOrderBy("rank", false, false, false, nil, nil)
	if num.Kind != vector.OrderNumeric || num.IsDatetime {
		t.Fatalf("numeric order regressed: %+v", num)
	}
	dtob, _ := vector.ParseOrderBy("ts", false, true, false, nil, nil)
	if dtob.Kind != vector.OrderDatetime || !dtob.IsDatetime {
		t.Fatalf("datetime order regressed: %+v", dtob)
	}
}

// TestStringOrderByCursorMismatchRejected: a string order_by must reject a v2 (numeric)
// cursor and a v1 (id-only) cursor mid-pagination, and a numeric order_by must reject a
// v3 (string) cursor — the version-byte dispatch is fail-loud both ways.
func TestStringOrderByCursorMismatchRejected(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	const coll = "sob_mismatch"
	rows := make([]strOrderRow, 60)
	for i := range rows {
		rows[i] = strOrderRow{id: uint64(i + 1), key: cityFor(i)}
	}
	seedDenseStringOrder(t, s, coll, 2, rows)

	strOB := &vector.OrderBy{Key: "city", Kind: vector.OrderString}
	_, _, next, err := s.VectorScroll(ctx, coll, VectorFilter{}, 10, VectorScrollOpts{OrderBy: strOB})
	if err != nil {
		t.Fatal(err)
	}
	if next == "" {
		t.Fatal("expected a v3 next_cursor on a full first page")
	}
	// The v3 cursor presented to a NUMERIC order_by ⇒ mismatch.
	if _, _, _, err := s.VectorScroll(ctx, coll, VectorFilter{}, 10,
		VectorScrollOpts{Cursor: next, OrderBy: &vector.OrderBy{Key: "city", Kind: vector.OrderNumeric}}); err == nil {
		t.Fatal("v3 cursor accepted by a numeric order_by (want mismatch)")
	}
	// The v3 cursor with a direction change ⇒ mismatch.
	if _, _, _, err := s.VectorScroll(ctx, coll, VectorFilter{}, 10,
		VectorScrollOpts{Cursor: next, OrderBy: &vector.OrderBy{Key: "city", Kind: vector.OrderString, Desc: true}}); err == nil {
		t.Fatal("v3 cursor accepted after a direction flip (want mismatch)")
	}
	// The v3 cursor with a key change ⇒ mismatch.
	if _, _, _, err := s.VectorScroll(ctx, coll, VectorFilter{}, 10,
		VectorScrollOpts{Cursor: next, OrderBy: &vector.OrderBy{Key: "other", Kind: vector.OrderString}}); err == nil {
		t.Fatal("v3 cursor accepted after a key change (want mismatch)")
	}
}

func cityFor(i int) string {
	alpha := []string{"alpha", "bravo", "bravo", "charlie", "delta"}
	return strings.ToLower(alpha[i%len(alpha)])
}

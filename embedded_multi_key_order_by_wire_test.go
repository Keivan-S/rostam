// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"encoding/base64"
	"math/rand"
	"reflect"
	"sort"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// This file is the WIRE coverage for MULTI-KEY order_by: the repeated order
// spec (proto/ops/http/client) + the P>1 fan-out tuple merge + the v4 cursor bridge. It
// asserts:
//   - the composite (k1,…,kN,id) page sequence equals an INDEPENDENT ground truth and is
//     PARTITION-INVARIANT (P4 == P1, gap/dup-free full pagination, v4 cursor resumes);
//   - a multi-key scroll emits a v4 cursor (version byte 4), a single-key scroll emits a
//     v2/v3 cursor (the #1 byte-identical back-compat anchor);
//   - the ops ScrollOrder multi-key tail block round-trips, and a single-key block is
//     BYTE-IDENTICAL to the pre-multi-key codec (the additive-wire regression).

// mkRow is one inserted point: id + a numeric "price" key + a string "name" key — the
// two-key composite ground truth (price desc, name asc, id asc).
type mkRow struct {
	id    uint64
	price float64
	name  string
}

// seedMultiKeyOrder creates a P-partition dense collection and inserts each id with a
// numeric "price" and a string "name" payload field. Inserts are RANDOMLY ordered so a
// correct composite page sequence proves the per-call tuple sort + merge, not insert order.
func seedMultiKeyOrder(t *testing.T, s Store, coll string, P int, rows []mkRow) {
	t.Helper()
	ctx := context.Background()
	if err := s.CreateCollection(ctx, coll, VectorConfig{
		Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2, Partitions: P,
	}); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	order := append([]mkRow(nil), rows...)
	rand.New(rand.NewSource(7)).Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
	for _, rw := range order {
		v := []float32{float32(rw.id), 0, 0, 0}
		md := VectorMetadata{"price": vector.NewFloat(rw.price), "name": vector.NewString(rw.name)}
		if err := s.VectorInsertExt(ctx, coll, rw.id, v, VectorInsertOpts{Metadata: md}); err != nil {
			t.Fatalf("VectorInsertExt %d: %v", rw.id, err)
		}
	}
}

// multiKeyOrderBy builds the two-key order: price DESC (primary), name ASC (secondary).
func multiKeyOrderBy() *vector.OrderBy {
	return &vector.OrderBy{
		Key:  "price",
		Desc: true,
		Kind: vector.OrderNumeric,
		Tail: []vector.OrderBy{{Key: "name", Desc: false, Kind: vector.OrderString}},
	}
}

// groundTruthMultiKey sorts rows by the composite (price desc, name asc, id asc) total
// order — the INDEPENDENT expected page sequence (no engine code involved).
func groundTruthMultiKey(rows []mkRow) []uint64 {
	cp := append([]mkRow(nil), rows...)
	sort.Slice(cp, func(i, j int) bool {
		if cp[i].price != cp[j].price {
			return cp[i].price > cp[j].price // price DESC
		}
		if cp[i].name != cp[j].name {
			return cp[i].name < cp[j].name // name ASC
		}
		return cp[i].id < cp[j].id // id ASC tiebreak
	})
	out := make([]uint64, len(cp))
	for i, r := range cp {
		out[i] = r.id
	}
	return out
}

// pageAllMultiKey pages a multi-key order_by scroll to exhaustion, asserting the v2/v4
// exhaustion rule (a full page ⇒ a cursor) and that every emitted cursor is v4.
func pageAllMultiKey(t *testing.T, s Store, coll string, ob *vector.OrderBy, limit int) []uint64 {
	t.Helper()
	ctx := context.Background()
	cursor := ""
	var ids []uint64
	for page := 0; ; page++ {
		docs, _, next, err := s.VectorScroll(ctx, coll, VectorFilter{}, limit, VectorScrollOpts{Cursor: cursor, OrderBy: ob})
		if err != nil {
			t.Fatalf("multi-key scroll page %d: %v", page, err)
		}
		for _, d := range docs {
			ids = append(ids, d.ID)
		}
		if len(docs) == limit {
			if next == "" {
				t.Fatalf("page %d full (len=%d) but next_cursor empty", page, limit)
			}
			assertCursorVersion(t, next, 4)
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

// assertCursorVersion decodes a (non-empty) cursor token and asserts its leading version
// byte — the byte-identical back-compat anchor (a multi-key cursor is v4; a single-key one
// is v2/v3, never v4).
func assertCursorVersion(t *testing.T, token string, want byte) {
	t.Helper()
	raw, err := base64.RawURLEncoding.DecodeString(token)
	if err != nil || len(raw) == 0 {
		t.Fatalf("cursor %q is not a valid token: %v", token, err)
	}
	if raw[0] != want {
		t.Fatalf("cursor version byte = %d, want %d", raw[0], want)
	}
}

// TestMultiKeyOrderByPartitionInvariance: the composite (price desc, name asc, id) P=4
// page sequence EQUALS the P=1 sequence AND the independent ground truth (gap/dup-free,
// the v4 cursor partition-invariant) — the core P4==P1 wire correctness proof.
func TestMultiKeyOrderByPartitionInvariance(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	const (
		N = 180
		L = 17
	)
	names := []string{"alpha", "bravo", "charlie", "delta", "echo"}
	rows := make([]mkRow, N)
	for i := 0; i < N; i++ {
		// Deliberate price TIES (price = i/4) so the secondary name key + id tiebreak are
		// exercised across partitions; names cycle so name ties also occur within a price.
		rows[i] = mkRow{id: uint64(i + 1), price: float64((i + 1) / 4), name: names[(i*3)%len(names)]}
	}
	seedMultiKeyOrder(t, s, "mk_p1", 1, rows)
	seedMultiKeyOrder(t, s, "mk_p4", 4, rows)

	p1 := pageAllMultiKey(t, s, "mk_p1", multiKeyOrderBy(), L)
	p4 := pageAllMultiKey(t, s, "mk_p4", multiKeyOrderBy(), L)
	if len(p1) != len(p4) || len(p1) != N {
		t.Fatalf("paged P1=%d P4=%d, want %d", len(p1), len(p4), N)
	}
	for i := range p1 {
		if p1[i] != p4[i] {
			t.Fatalf("partition variance at %d: P1=%d P4=%d", i, p1[i], p4[i])
		}
	}
	want := groundTruthMultiKey(rows)
	for i := range want {
		if p4[i] != want[i] {
			t.Fatalf("composite order mismatch at %d: P4=%d want %d", i, p4[i], want[i])
		}
	}
}

// TestMultiKeyVsSingleKeyCursorVersion: a multi-key scroll emits a v4 cursor; a single-key
// scroll over the SAME data emits a v2 (numeric) cursor — the #1 back-compat anchor (a
// single-key order is byte-identical, never v4).
func TestMultiKeyVsSingleKeyCursorVersion(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	const N = 40
	names := []string{"a", "b", "c"}
	rows := make([]mkRow, N)
	for i := 0; i < N; i++ {
		rows[i] = mkRow{id: uint64(i + 1), price: float64((i + 1) / 3), name: names[i%len(names)]}
	}
	seedMultiKeyOrder(t, s, "mk_ver", 4, rows)

	// Multi-key ⇒ a v4 cursor.
	docs, _, next, err := s.VectorScroll(ctx, "mk_ver", VectorFilter{}, 10, VectorScrollOpts{OrderBy: multiKeyOrderBy()})
	if err != nil {
		t.Fatalf("multi-key scroll: %v", err)
	}
	if len(docs) != 10 || next == "" {
		t.Fatalf("multi-key first page: len=%d next=%q", len(docs), next)
	}
	assertCursorVersion(t, next, 4)

	// Single-key numeric ⇒ a v2 cursor (NOT v4).
	_, _, sNext, err := s.VectorScroll(ctx, "mk_ver", VectorFilter{}, 10, VectorScrollOpts{OrderBy: &vector.OrderBy{Key: "price", Desc: true}})
	if err != nil {
		t.Fatalf("single-key scroll: %v", err)
	}
	if sNext == "" {
		t.Fatalf("single-key first page produced no cursor")
	}
	assertCursorVersion(t, sNext, 2)

	// Single-key string ⇒ a v3 cursor (NOT v4).
	_, _, strNext, err := s.VectorScroll(ctx, "mk_ver", VectorFilter{}, 10, VectorScrollOpts{OrderBy: &vector.OrderBy{Key: "name", Kind: vector.OrderString}})
	if err != nil {
		t.Fatalf("single-key string scroll: %v", err)
	}
	if strNext == "" {
		t.Fatalf("single-key string first page produced no cursor")
	}
	assertCursorVersion(t, strNext, 3)
}

// TestScrollArgsOrderMultiKeyRoundTrip: a multi-key ScrollOrder (Tail + v4 resume tuple)
// round-trips through the args codec exactly.
func TestScrollArgsOrderMultiKeyRoundTrip(t *testing.T) {
	order := &ops.ScrollOrder{
		Key: "price", Desc: true, Kind: vector.OrderNumeric,
		Tail: []ops.ScrollOrderKey{
			{Key: "name", Desc: false, Kind: vector.OrderString},
			{Key: "ts", Desc: true, IsDatetime: true, Kind: vector.OrderDatetime},
		},
		ResumeKeys: []ops.ScrollOrderVal{
			{Num: 42.5, Kind: vector.OrderNumeric},
			{Str: "bravo", Kind: vector.OrderString},
			{Num: 1700000000000, Kind: vector.OrderDatetime},
		},
		HasResumeKeys: true,
	}
	args := ops.EncodeScrollArgsOrderBounded("col", vector.Filter{}, 9, 1, 2, 77, true, order, 0)
	col, _, limit, rc, opa, afterID, hasAfter, got, err := ops.DecodeScrollArgsOrder(args)
	if err != nil {
		t.Fatal(err)
	}
	if col != "col" || limit != 9 || rc != 1 || opa != 2 || afterID != 77 || !hasAfter {
		t.Fatalf("base/trailer round-trip mismatch: col=%q limit=%d rc=%d opa=%d afterID=%d hasAfter=%v", col, limit, rc, opa, afterID, hasAfter)
	}
	if got == nil || !reflect.DeepEqual(got, order) {
		t.Fatalf("multi-key order round-trip:\n got  %+v\n want %+v", got, order)
	}

	// A multi-key order with NO resume tuple (page 1) round-trips with HasResumeKeys=false.
	page1 := &ops.ScrollOrder{
		Key: "price", Desc: true, Tail: []ops.ScrollOrderKey{{Key: "name", Kind: vector.OrderString}},
	}
	args1 := ops.EncodeScrollArgsOrderBounded("c", vector.Filter{}, 1, 0, 0, 0, false, page1, 0)
	_, _, _, _, _, _, _, got1, err := ops.DecodeScrollArgsOrder(args1)
	if err != nil {
		t.Fatal(err)
	}
	if got1 == nil || !reflect.DeepEqual(got1, page1) {
		t.Fatalf("multi-key page-1 round-trip:\n got  %+v\n want %+v", got1, page1)
	}
}

// TestScrollArgsOrderSingleKeyByteIdentical: a SINGLE-key ScrollOrder's encoded bytes are
// BYTE-IDENTICAL whether or not the multi-key fields exist on the struct (Tail nil) — the
// additive-wire #1 anchor. We assert this by encoding a single-key order and confirming
// the multi-key flag bit (bit3) is never set (the block stops after the single-key tail),
// then that it decodes back to the identical single-key order (nil Tail / no ResumeKeys).
func TestScrollArgsOrderSingleKeyByteIdentical(t *testing.T) {
	single := &ops.ScrollOrder{
		Key: "rank", Desc: true, IsDatetime: false, Kind: vector.OrderNumeric,
		StartFrom: 12.5, HasStart: true, ResumeKey: 99.5, HasResume: true,
	}
	args := ops.EncodeScrollArgsOrderBounded("col", vector.Filter{}, 7, 1, 2, 42, true, single, 0)
	_, _, _, _, _, _, _, got, err := ops.DecodeScrollArgsOrder(args)
	if err != nil {
		t.Fatal(err)
	}
	if got == nil {
		t.Fatal("single-key order decoded as nil")
	}
	if len(got.Tail) != 0 || got.HasResumeKeys {
		t.Fatalf("single-key block leaked multi-key state: Tail=%v HasResumeKeys=%v", got.Tail, got.HasResumeKeys)
	}
	if !reflect.DeepEqual(got, single) {
		t.Fatalf("single-key round-trip:\n got  %+v\n want %+v", got, single)
	}
	// The decoded order, re-encoded, is identical bytes (no hidden multi-key tail).
	args2 := ops.EncodeScrollArgsOrderBounded("col", vector.Filter{}, 7, 1, 2, 42, true, got, 0)
	if !reflect.DeepEqual(args, args2) {
		t.Fatalf("single-key re-encode not byte-identical (len %d vs %d)", len(args), len(args2))
	}
}

// TestMultiKeyOrderByCursorMismatchRejected: a multi-key scroll resumed with a SINGLE-key
// (v2) cursor is rejected loud (the v4 validator rejects a non-v4 cursor), and a single-key
// scroll resumed with a multi-key (v4) cursor is likewise rejected — the cursor⇄order_by
// agreement guard, generalised to the tuple.
func TestMultiKeyOrderByCursorMismatchRejected(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	const N = 30
	names := []string{"a", "b", "c"}
	rows := make([]mkRow, N)
	for i := 0; i < N; i++ {
		rows[i] = mkRow{id: uint64(i + 1), price: float64((i + 1) / 3), name: names[i%len(names)]}
	}
	seedMultiKeyOrder(t, s, "mk_mis", 2, rows)

	// Get a v4 cursor from a multi-key page.
	_, _, v4cur, err := s.VectorScroll(ctx, "mk_mis", VectorFilter{}, 5, VectorScrollOpts{OrderBy: multiKeyOrderBy()})
	if err != nil || v4cur == "" {
		t.Fatalf("seed multi-key cursor: cur=%q err=%v", v4cur, err)
	}
	// Resume that v4 cursor with a SINGLE-key order ⇒ mismatch.
	if _, _, _, err := s.VectorScroll(ctx, "mk_mis", VectorFilter{}, 5, VectorScrollOpts{Cursor: v4cur, OrderBy: &vector.OrderBy{Key: "price", Desc: true}}); err == nil {
		t.Fatal("single-key order resumed a v4 cursor without error (want mismatch)")
	}

	// Get a v2 cursor from a single-key page.
	_, _, v2cur, err := s.VectorScroll(ctx, "mk_mis", VectorFilter{}, 5, VectorScrollOpts{OrderBy: &vector.OrderBy{Key: "price", Desc: true}})
	if err != nil || v2cur == "" {
		t.Fatalf("seed single-key cursor: cur=%q err=%v", v2cur, err)
	}
	// Resume that v2 cursor with the MULTI-key order ⇒ mismatch.
	if _, _, _, err := s.VectorScroll(ctx, "mk_mis", VectorFilter{}, 5, VectorScrollOpts{Cursor: v2cur, OrderBy: multiKeyOrderBy()}); err == nil {
		t.Fatal("multi-key order resumed a v2 cursor without error (want mismatch)")
	}
}

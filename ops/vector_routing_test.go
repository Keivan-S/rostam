// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"sort"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// TestVectorRouteKeyConsistency checks that every op for a given collection
// yields the same routing key (so all of a collection's ops land on one shard),
// regardless of the op's arg layout, and that the key is canonicalized.
func TestVectorRouteKeyConsistency(t *testing.T) {
	q := []float32{1, 2, 3, 4}
	cases := []struct {
		name string
		args []byte
		ke   KeyExtractor
	}{
		{"create", EncodeCreateCollectionArgs("docs", vector.Config{Dim: 4}), vectorKeyColAt1},
		{"drop", EncodeDropCollectionArgs("docs"), vectorKeyColAt1},
		{"delete", EncodeVectorDeleteArgs("docs", 7), vectorKeyColAt1},
		{"delete_by_filter", EncodeDeleteByFilterArgs("docs", vector.Filter{Op: vector.FilterEq, Field: "x", Value: vector.NewInt(1)}), vectorKeyColAt1},
		{"scroll", EncodeScrollArgs("docs", vector.Filter{}, 0), vectorKeyColAt1},
		{"search_groups", EncodeGroupSearchArgs("docs", 5, q, vector.GroupOpts{GroupBy: "g"}), vectorKeyColAt1},
		{"insert", EncodeVectorInsertArgs("docs", 1, q), vectorKeyColAt2},
		{"search", EncodeVectorSearchArgs("docs", 5, q), vectorKeyColAt2},
		{"hybrid", EncodeHybridSearchArgs("docs", q, 5, vector.SparseVector{}, vector.HybridOpts{}), vectorKeyColAt2},
		{"mv_create", EncodeMVCreateArgs("docs", vector.MultiVectorConfig{Dim: 4}), vectorKeyColAt1},
		{"mv_add", EncodeMVAddArgs("docs", 1, [][]float32{q}, nil), vectorKeyColAt1},
		{"mv_search", EncodeMVSearchArgs("docs", [][]float32{q}, 5, 0), vectorKeyColAt1},
	}
	for _, c := range cases {
		key, ok := c.ke(c.args)
		if !ok {
			t.Errorf("%s: extractor returned !ok", c.name)
			continue
		}
		if string(key) != "default/docs" {
			t.Errorf("%s: route key = %q, want %q", c.name, key, "default/docs")
		}
	}
}

// TestCollectionNameFor checks the cheap op->extractor mapping the fan-out
// dispatcher uses to peek the collection name before a full decode: each of the
// six intercepted read ops must resolve to its canonical collection name (proving
// the At1 vs At2 layout is mapped correctly), and non-vector ops report !ok.
func TestCollectionNameFor(t *testing.T) {
	q := []float32{1, 2, 3, 4}
	cases := []struct {
		op   string
		args []byte
	}{
		// At2 layout.
		{"vector_search", EncodeVectorSearchArgs("docs", 5, q)},
		{"vector_search_docs", EncodeVectorSearchArgsExt("docs", 5, q, vector.Filter{})},
		{"vector_hybrid_search", EncodeHybridSearchArgs("docs", q, 5, vector.SparseVector{}, vector.HybridOpts{})},
		// At1 layout.
		{"vector_search_groups", EncodeGroupSearchArgs("docs", 5, q, vector.GroupOpts{GroupBy: "g"})},
		{"vector_scroll", EncodeScrollArgs("docs", vector.Filter{}, 0)},
		{"vector_delete_by_filter", EncodeDeleteByFilterArgs("docs", vector.Filter{Op: vector.FilterEq, Field: "x", Value: vector.NewInt(1)})},
		// At1 layout — multi-vector ops the fan-out dispatcher intercepts. Note
		// vector_mv_create_collection is intentionally absent (it full-decodes to
		// gate on cfg.Partitions), as is vector_mv_get_config (single-collection
		// probe, always passthrough).
		{"vector_mv_add", EncodeMVAddArgs("docs", 1, [][]float32{q}, nil)},
		{"vector_mv_delete", EncodeMVDeleteArgs("docs", 1)},
		{"vector_mv_search", EncodeMVSearchArgs("docs", [][]float32{q}, 5, 0)},
		{"vector_mv_drop_collection", EncodeMVDeleteArgs("docs", 0)},
		// Online-reshard copy primitives: collection-routed point ops that must
		// resolve to a partition on a partitioned collection (At2 for the dense
		// insert-shaped op, At1 for the delete-shaped exists + MV variants).
		{"vector_insert_if_absent", EncodeVectorInsertArgs("docs", 1, q)},
		{"vector_exists", EncodeExistsArgs("docs", 1)},
		{"vector_mv_add_if_absent", EncodeMVAddArgs("docs", 1, [][]float32{q}, nil)},
		{"vector_mv_exists", EncodeMVExistsArgs("docs", 1)},
	}
	for _, c := range cases {
		name, ok := CollectionNameFor(c.op, c.args)
		if !ok {
			t.Errorf("%s: CollectionNameFor !ok", c.op)
			continue
		}
		if name != "default/docs" {
			t.Errorf("%s: name = %q, want default/docs", c.op, name)
		}
	}
	// Non-vector / non-intercepted op: report !ok so the caller passes through.
	if _, ok := CollectionNameFor("get", []byte{0, 1, 'a'}); ok {
		t.Error("CollectionNameFor for unknown op should be !ok")
	}
	// Truncated args for a known op: !ok (cannot peek).
	if _, ok := CollectionNameFor("vector_search", []byte{0}); ok {
		t.Error("CollectionNameFor on truncated args should be !ok")
	}
}

func TestVectorRouteKeyTenantQualified(t *testing.T) {
	// A tenant-qualified name routes by its canonical form unchanged; a bare name
	// gets the default tenant — and the two must differ so they can co-route only
	// when they refer to the same canonical collection.
	bare, _ := vectorKeyColAt1(EncodeDropCollectionArgs("docs"))
	qual, _ := vectorKeyColAt1(EncodeDropCollectionArgs("acme/docs"))
	if string(bare) != "default/docs" {
		t.Errorf("bare = %q", bare)
	}
	if string(qual) != "acme/docs" {
		t.Errorf("qualified = %q", qual)
	}
	// "docs" and "default/docs" must hash to the same shard key.
	d2, _ := vectorKeyColAt1(EncodeDropCollectionArgs("default/docs"))
	if string(bare) != string(d2) {
		t.Errorf("bare %q != default-qualified %q", bare, d2)
	}
}

func TestVectorRouteKeyRejectsEmpty(t *testing.T) {
	if _, ok := vectorKeyColAt1([]byte{0}); ok {
		t.Error("empty name should not yield a routing key")
	}
	if _, ok := vectorKeyColAt2([]byte{0, 0}); ok {
		t.Error("empty name should not yield a routing key")
	}
}

func TestPartitionKeyAndOf(t *testing.T) {
	// PartitionKey canonicalizes the collection (bare -> default/) then appends #p.
	if got := string(PartitionKey("docs", 3)); got != "default/docs#3" {
		t.Errorf("PartitionKey bare = %q, want default/docs#3", got)
	}
	if got := string(PartitionKey("tenantA/docs", 0)); got != "tenantA/docs#0" {
		t.Errorf("PartitionKey qualified = %q", got)
	}
	// PartitionOf is deterministic and in range.
	for _, id := range []uint64{0, 1, 42, 1 << 40} {
		p := PartitionOf(id, 8)
		if p < 0 || p >= 8 {
			t.Errorf("PartitionOf(%d,8)=%d out of range", id, p)
		}
		if PartitionOf(id, 8) != p {
			t.Errorf("PartitionOf not deterministic")
		}
	}
	// P<=1 always maps to partition 0.
	if PartitionOf(99, 1) != 0 || PartitionOf(99, 0) != 0 {
		t.Error("P<=1 must map to partition 0")
	}
}

// TestPartitionGenOf verifies the inverse of PartitionKeyGen's generation
// encoding: gen 0 from the legacy "canonical#p" form, gen g from "canonical@g#p",
// and 0 for malformed names (the conservative legacy default).
func TestPartitionGenOf(t *testing.T) {
	// Round-trip: PartitionGenOf(PartitionKeyGen(coll,g,p)) == g for a range of g.
	for _, g := range []uint32{0, 1, 2, 7, 42, 1000, 4294967295} {
		phys := string(PartitionKeyGen("default/docs", g, 3))
		if got := PartitionGenOf(phys); got != g {
			t.Errorf("PartitionGenOf(%q)=%d, want %d", phys, got, g)
		}
	}
	// Gen 0 is the legacy form with no '@'.
	if got := PartitionGenOf("default/docs#0"); got != 0 {
		t.Errorf("legacy form gen = %d, want 0", got)
	}
	// Explicit gen g from "@g#p".
	if got := PartitionGenOf("default/docs@5#2"); got != 5 {
		t.Errorf("@5#2 gen = %d, want 5", got)
	}
	// Tenant-qualified names (a '/' before the '#') still parse correctly.
	if got := PartitionGenOf("tenantA/docs@9#1"); got != 9 {
		t.Errorf("tenant @9 gen = %d, want 9", got)
	}
	// Malformed names ⇒ 0 (the conservative legacy default).
	malformed := []string{
		"",                                    // empty
		"default/docs",                        // no '#'
		"default/docs@",                       // '@' but no digits/'#'
		"default/docs@#2",                     // '@' immediately followed by '#' (empty gen)
		"default/docs@x#2",                    // non-numeric gen
		"default/docs#2@3",                    // '#' before '@' (wrong order)
		"default/docs@99999999999999999999#2", // overflows uint32
	}
	for _, m := range malformed {
		if got := PartitionGenOf(m); got != 0 {
			t.Errorf("PartitionGenOf(%q)=%d, want 0 (malformed ⇒ legacy gen 0)", m, got)
		}
	}
}

func TestPartitionOfGolden(t *testing.T) {
	// Pinned outputs guard against accidental changes to the hash that would
	// silently re-shuffle every collection's data across partitions.
	const P = 16
	want := map[uint64]int{
		0:                    15,
		1:                    1,
		42:                   5,
		1000000:              7,
		18446744073709551615: 0, // max uint64
	}
	for id, w := range want {
		if got := PartitionOf(id, P); got != w {
			t.Errorf("PartitionOf(%d,%d)=%d, want %d", id, P, got, w)
		}
	}
}

func TestPartitionDistributionEven(t *testing.T) {
	const P, N = 8, 100000
	counts := make([]int, P)
	for id := uint64(0); id < N; id++ {
		counts[PartitionOf(id, P)]++
	}
	// Each partition should hold ~N/P; allow +/-20%.
	lo, hi := N/P*8/10, N/P*12/10
	for p, c := range counts {
		if c < lo || c > hi {
			t.Errorf("partition %d count %d outside [%d,%d]", p, c, lo, hi)
		}
	}
}

func TestCanonicalName(t *testing.T) {
	if CanonicalName("docs") != "default/docs" {
		t.Errorf("bare = %q, want default/docs", CanonicalName("docs"))
	}
	if CanonicalName("t/docs") != "t/docs" {
		t.Errorf("qualified = %q, want t/docs", CanonicalName("t/docs"))
	}
}

func TestPartitionKeyGen(t *testing.T) {
	// gen 0 is byte-identical to PartitionKey (backward compat).
	if got := string(PartitionKeyGen("docs", 0, 3)); got != "default/docs#3" {
		t.Errorf("gen0 = %q, want default/docs#3", got)
	}
	if string(PartitionKeyGen("docs", 0, 3)) != string(PartitionKey("docs", 3)) {
		t.Error("gen0 must equal PartitionKey")
	}
	// gen >= 1 is a distinct, non-colliding name.
	if got := string(PartitionKeyGen("docs", 2, 3)); got != "default/docs@2#3" {
		t.Errorf("gen2 = %q, want default/docs@2#3", got)
	}
	// qualified name preserved.
	if got := string(PartitionKeyGen("t/docs", 1, 0)); got != "t/docs@1#0" {
		t.Errorf("qualified gen1 = %q, want t/docs@1#0", got)
	}
	// Empty/invalid collection -> nil (matches PartitionKey contract).
	if PartitionKeyGen("", 1, 0) != nil {
		t.Error("empty collection should yield nil")
	}
}

// TestRouteKeyIntoMatchesKeyExtractor is the DEPTH half of the fast path's
// invariant: for one op of each layout, RouteKeyInto(layout) returns the SAME
// routing key bytes as the registered KeyExtractor over a bare/qualified/oversized
// name and over EVERY truncated prefix of the args (where both must report "no
// key"). TestEveryBuiltinLayoutMatchesItsExtractor is the BREADTH half — every
// registered op, so a mis-tagged table row cannot hide behind a sampled op.
//
// A divergence in either direction would send the same collection to two different
// shards depending on which path a caller took.
func TestRouteKeyIntoMatchesKeyExtractor(t *testing.T) {
	r := NewRegistry()
	if err := RegisterBuiltins(r); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	names := []string{"docs", "default/docs", "tenant7/docs", strings.Repeat("x", 200), ""}
	for _, op := range []string{"vector_get", "vector_search"} {
		_, ke, layout, ok := r.LookupRouting(op)
		if !ok {
			t.Fatalf("%s not registered", op)
		}
		if layout == RouteLayoutNone {
			t.Fatalf("%s has no RouteLayout", op)
		}
		off := 0
		if layout == RouteLayoutColAt2 {
			off = 1
		}
		for _, name := range names {
			full := make([]byte, off)
			full = append(full, byte(len(name)))
			full = append(full, name...)
			full = append(full, make([]byte, 8)...)
			// Every prefix, so truncation is covered as thoroughly as the full args.
			for cut := 0; cut <= len(full); cut++ {
				args := full[:cut]
				want, found := ke(args)
				var buf [128]byte
				got := RouteKeyInto(layout, args, buf[:0])
				if !found {
					if got != nil {
						t.Fatalf("%s %q cut=%d: KeyExtractor says no key, RouteKeyInto returned %q", op, name, cut, got)
					}
					continue
				}
				if !bytes.Equal(want, got) {
					t.Fatalf("%s %q cut=%d: KeyExtractor=%q RouteKeyInto=%q", op, name, cut, want, got)
				}
			}
		}
	}
}

// TestEveryBuiltinLayoutMatchesItsExtractor is the BREADTH half of the fast path's
// invariant (see TestRouteKeyIntoMatchesKeyExtractor for the depth half): EVERY
// registered built-in routable op must agree with its own RouteLayout, on both wire
// shapes and on the bare / qualified / physical-partition name forms.
//
// This is the test that would catch the failure mode the sampled test cannot: an op
// whose table row pairs the At1 extractor with the At2 layout (or vice versa) still
// routes fine for names the two layouts happen to read identically, but sends that
// ONE op's traffic to a different shard than every other op on the same collection
// — a silent split-brain for that collection. Iterating the registry means a newly
// added op is covered the moment it is registered, with no test to remember to
// update.
//
// KV built-ins (get/put/del/expire/incr/put_batch) route by raw key bytes, not by a
// collection name, so they carry no layout and keep the allocating KeyExtractor;
// they are skipped rather than asserted.
func TestEveryBuiltinLayoutMatchesItsExtractor(t *testing.T) {
	r := NewRegistry()
	if err := RegisterBuiltins(r); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	r.mu.RLock()
	names := make([]string, 0, len(r.m))
	for n := range r.m {
		names = append(names, n)
	}
	r.mu.RUnlock()
	sort.Strings(names) // deterministic failure order

	// Both wire shapes for each name form: At1 ([colLen][col]...) and At2
	// ([flags][colLen][col]...). An op is asserted against BOTH, so a layout that
	// reads the name from the wrong offset shows up as a differing key.
	var shapes [][]byte
	for _, col := range []string{"docs", "default/docs", "default/docs#3", "default/docs@2#1"} {
		at1 := append([]byte{byte(len(col))}, col...)
		at2 := append([]byte{0x01, byte(len(col))}, col...)
		shapes = append(shapes,
			append(at1, make([]byte, 16)...),
			append(at2, make([]byte, 16)...))
	}

	checked := 0
	for _, name := range names {
		_, ke, layout, ok := r.LookupRouting(name)
		if !ok || ke == nil || layout == RouteLayoutNone {
			continue // shardless, or a KV op that routes by raw key
		}
		checked++
		for _, args := range shapes {
			want, found := ke(args)
			var buf [128]byte
			got := RouteKeyInto(layout, args, buf[:0])
			if !found {
				if got != nil {
					t.Errorf("%s: KeyExtractor says no key, RouteKeyInto returned %q (args %q)", name, got, args)
				}
				continue
			}
			if !bytes.Equal(want, got) {
				t.Errorf("%s: KeyExtractor=%q RouteKeyInto=%q (args %q)", name, want, got, args)
			}
		}
	}
	// Coverage floor. Without it this test degrades silently: a refactor that drops
	// the layout from most of the table (or from all of it) leaves every surviving
	// op still in agreement, so the loop passes having verified almost nothing while
	// the ops it stopped covering quietly fall back to the allocating path. The
	// number only ever goes UP as routable ops are added — if this fires because you
	// added ops, raise it; if it fires because you removed layouts, that is the bug
	// it is here to catch.
	const wantAtLeast = 69
	if checked < wantAtLeast {
		t.Fatalf("only %d built-in routable ops carry a RouteLayout, want >= %d — "+
			"layouts were dropped and those ops silently lost the allocation-free routing path",
			checked, wantAtLeast)
	}
	t.Logf("verified %d built-in routable ops", checked)
}

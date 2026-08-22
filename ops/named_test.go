// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"encoding/binary"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/vector"
)

// ---- codec round-trips ----

func TestNamedCreateArgsRoundtrip(t *testing.T) {
	cfg := map[string]vector.NamedVectorParams{
		"title": {Dim: 4, Metric: vector.Cosine, M: 8, EfConstruction: 100},
		"image": {Dim: 8, Metric: vector.DotProduct},
	}
	col, got, partitions, err := DecodeNamedCreateArgs(EncodeNamedCreateArgs("acme/docs", cfg, 4))
	if err != nil {
		t.Fatal(err)
	}
	if col != "acme/docs" {
		t.Errorf("col = %q", col)
	}
	if partitions != 4 {
		t.Errorf("partitions = %d, want 4", partitions)
	}
	if !reflect.DeepEqual(got, cfg) {
		t.Errorf("cfg = %+v, want %+v", got, cfg)
	}
}

// TestNamedCreateArgsIVFRoundtrip proves an IVF / IVF-PQ per-space config
// round-trips through the named create codec into the right NamedVectorParams.
func TestNamedCreateArgsIVFRoundtrip(t *testing.T) {
	cfg := map[string]vector.NamedVectorParams{
		"dense": {Dim: 32, Metric: vector.Cosine, M: 16, EfConstruction: 200},
		"ivf": {
			Dim: 32, Metric: vector.Cosine,
			IndexType: vector.IndexIVF, IVFNlist: 64, IVFNprobe: 12,
			IVFPQ: true, IVFPQM: 8, IVFRerank: true, OPQ: true, IVFTrainThreshold: 1000,
		},
	}
	col, got, partitions, err := DecodeNamedCreateArgs(EncodeNamedCreateArgs("acme/named", cfg, 4))
	if err != nil {
		t.Fatal(err)
	}
	if col != "acme/named" || partitions != 4 || !reflect.DeepEqual(got, cfg) {
		t.Errorf("IVF roundtrip: col=%q partitions=%d cfg=%+v want %+v", col, partitions, got, cfg)
	}
	sp := got["ivf"]
	if sp.IndexType != vector.IndexIVF || sp.IVFNlist != 64 || sp.IVFNprobe != 12 ||
		!sp.IVFPQ || sp.IVFPQM != 8 || !sp.IVFRerank || !sp.OPQ || sp.IVFTrainThreshold != 1000 {
		t.Errorf("IVF space fields not carried: %+v", sp)
	}
}

// TestNamedCreateArgsIVFByteIdentical proves an all-HNSW named config encodes
// byte-identically to the pre-IVF encoder: NamedVectorParams' IVF fields are all
// omitempty, so a config that leaves them zero serializes with no IVF JSON keys.
func TestNamedCreateArgsIVFByteIdentical(t *testing.T) {
	hnsw := map[string]vector.NamedVectorParams{
		"title": {Dim: 4, Metric: vector.Cosine, M: 8, EfConstruction: 100},
		"image": {Dim: 8, Metric: vector.DotProduct},
	}
	wire := EncodeNamedCreateArgs("docs", hnsw, 0)
	// The cfgJSON portion must contain no IVF keys for an all-HNSW config.
	for _, key := range []string{"index_type", "ivf_nlist", "ivf_nprobe", "ivf_pq", "ivf_rerank", "opq", "ivf_train_threshold"} {
		if bytes.Contains(wire, []byte(key)) {
			t.Fatalf("HNSW named wire unexpectedly contains IVF key %q", key)
		}
	}
}

// TestNamedCreateArgsLegacyNoPartitions proves the trailing partitions word is
// optional: a payload that ends right after cfgJSON (the pre-fan-out encoding)
// decodes with partitions=0 (single-partition), so old encoders interoperate.
func TestNamedCreateArgsLegacyNoPartitions(t *testing.T) {
	cfg := map[string]vector.NamedVectorParams{"t": {Dim: 4}}
	full := EncodeNamedCreateArgs("docs", cfg, 0)
	// Strip the trailing 4-byte partitions word to simulate a legacy encoding.
	legacy := full[:len(full)-4]
	col, got, partitions, err := DecodeNamedCreateArgs(legacy)
	if err != nil {
		t.Fatal(err)
	}
	if col != "docs" || partitions != 0 || !reflect.DeepEqual(got, cfg) {
		t.Errorf("legacy decode: col=%q partitions=%d cfg=%+v", col, partitions, got)
	}
}

func TestNamedCreateArgsTruncated(t *testing.T) {
	full := EncodeNamedCreateArgs("docs", map[string]vector.NamedVectorParams{"t": {Dim: 4}}, 2)
	// Chop within the col+cfg prefix (before the optional partitions word) must
	// error; the partitions word itself is optional so stop before it.
	prefix := len(full) - 4
	for chop := 1; chop < prefix; chop++ {
		if _, _, _, err := DecodeNamedCreateArgs(full[:chop]); err == nil {
			t.Errorf("chop to %d bytes: expected error, got nil", chop)
		}
	}
}

func TestNamedInsertArgsRoundtrip(t *testing.T) {
	vecs := map[string][]float32{
		"title": {1, 0, 0, 0},
		"image": {0, 1, 0, 0, 0, 0, 0, 0},
	}
	meta := vector.Metadata{"lang": vector.NewString("en"), "n": vector.NewInt(7)}
	ttl := 90 * time.Second
	col, id, got, gotMeta, gotTTL, err := DecodeNamedInsertArgs(EncodeNamedInsertArgs("c", 42, vecs, meta, ttl))
	if err != nil {
		t.Fatal(err)
	}
	if col != "c" || id != 42 {
		t.Errorf("col=%q id=%d", col, id)
	}
	if !reflect.DeepEqual(got, vecs) {
		t.Errorf("vectors = %+v, want %+v", got, vecs)
	}
	if !reflect.DeepEqual(gotMeta, meta) {
		t.Errorf("meta = %+v, want %+v", gotMeta, meta)
	}
	if gotTTL != ttl {
		t.Errorf("ttl = %v, want %v", gotTTL, ttl)
	}
}

func TestNamedInsertArgsNoPayloadNoTTL(t *testing.T) {
	vecs := map[string][]float32{"title": {1, 2, 3, 4}}
	col, id, got, gotMeta, gotTTL, err := DecodeNamedInsertArgs(EncodeNamedInsertArgs("c", 1, vecs, nil, 0))
	if err != nil {
		t.Fatal(err)
	}
	if col != "c" || id != 1 || gotTTL != 0 || gotMeta != nil {
		t.Errorf("col=%q id=%d ttl=%v meta=%v", col, id, gotTTL, gotMeta)
	}
	if !reflect.DeepEqual(got, vecs) {
		t.Errorf("vectors = %+v", got)
	}
}

func TestNamedInsertArgsTruncated(t *testing.T) {
	full := EncodeNamedInsertArgs("c", 1, map[string][]float32{"t": {1, 2}}, vector.Metadata{"a": vector.NewInt(1)}, time.Second)
	for chop := 1; chop < len(full); chop++ {
		if _, _, _, _, _, err := DecodeNamedInsertArgs(full[:chop]); err == nil {
			t.Errorf("chop to %d bytes: expected error, got nil", chop)
		}
	}
}

func TestNamedSearchArgsRoundtrip(t *testing.T) {
	q := []float32{1, 0, 0, 0}
	filter := vector.Filter{Op: vector.FilterEq, Field: "lang", Value: vector.NewString("en")}
	col, name, got, k, gotFilter, err := DecodeNamedSearchArgs(EncodeNamedSearchArgs("c", "title", q, 5, filter))
	if err != nil {
		t.Fatal(err)
	}
	if col != "c" || name != "title" || k != 5 {
		t.Errorf("col=%q name=%q k=%d", col, name, k)
	}
	if !reflect.DeepEqual(got, q) {
		t.Errorf("query = %v", got)
	}
	if !reflect.DeepEqual(gotFilter, filter) {
		t.Errorf("filter = %+v, want %+v", gotFilter, filter)
	}
}

func TestNamedSearchArgsNoFilter(t *testing.T) {
	q := []float32{1, 2, 3}
	col, name, got, k, gotFilter, err := DecodeNamedSearchArgs(EncodeNamedSearchArgs("c", "img", q, 3, vector.Filter{}))
	if err != nil {
		t.Fatal(err)
	}
	if col != "c" || name != "img" || k != 3 {
		t.Errorf("col=%q name=%q k=%d", col, name, k)
	}
	if !reflect.DeepEqual(got, q) {
		t.Errorf("query = %v", got)
	}
	if !gotFilter.IsZero() {
		t.Errorf("filter = %+v, want zero", gotFilter)
	}
}

func TestNamedSearchArgsTruncated(t *testing.T) {
	full := EncodeNamedSearchArgs("c", "title", []float32{1, 2, 3, 4}, 5,
		vector.Filter{Op: vector.FilterEq, Field: "x", Value: vector.NewInt(1)})
	for chop := 1; chop < len(full); chop++ {
		if _, _, _, _, _, err := DecodeNamedSearchArgs(full[:chop]); err == nil {
			t.Errorf("chop to %d bytes: expected error, got nil", chop)
		}
	}
}

func TestNamedDeleteArgsRoundtrip(t *testing.T) {
	col, id, err := DecodeNamedDeleteArgs(EncodeNamedDeleteArgs("acme/docs", 99))
	if err != nil {
		t.Fatal(err)
	}
	if col != "acme/docs" || id != 99 {
		t.Errorf("col=%q id=%d", col, id)
	}
	for chop := 1; chop < 1+len("acme/docs")+8; chop++ {
		if _, _, err := DecodeNamedDeleteArgs(EncodeNamedDeleteArgs("acme/docs", 99)[:chop]); err == nil {
			t.Errorf("chop to %d: expected error", chop)
		}
	}
}

func TestNamedScrollArgsRoundtrip(t *testing.T) {
	filter := vector.Filter{Op: vector.FilterEq, Field: "lang", Value: vector.NewString("en")}
	col, gotFilter, limit, err := DecodeNamedScrollArgs(EncodeNamedScrollArgs("c", filter, 25))
	if err != nil {
		t.Fatal(err)
	}
	if col != "c" || limit != 25 {
		t.Errorf("col=%q limit=%d", col, limit)
	}
	if !reflect.DeepEqual(gotFilter, filter) {
		t.Errorf("filter = %+v", gotFilter)
	}
	// no-filter form
	col, gotFilter, limit, err = DecodeNamedScrollArgs(EncodeNamedScrollArgs("c", vector.Filter{}, 0))
	if err != nil {
		t.Fatal(err)
	}
	if col != "c" || limit != 0 || !gotFilter.IsZero() {
		t.Errorf("no-filter: col=%q limit=%d filter=%+v", col, limit, gotFilter)
	}
}

func TestNamedScrollArgsTruncated(t *testing.T) {
	full := EncodeNamedScrollArgs("c", vector.Filter{Op: vector.FilterEq, Field: "x", Value: vector.NewInt(1)}, 10)
	for chop := 1; chop < len(full); chop++ {
		if _, _, _, err := DecodeNamedScrollArgs(full[:chop]); err == nil {
			t.Errorf("chop to %d: expected error", chop)
		}
	}
}

// ---- rc/opa opts trailer (read-consistency) ----

// TestNamedSearchArgsOptsRoundtrip round-trips rc/opa through the named-search
// opts trailer, with and without a filter (the filter stays in the base block —
// only rc/opa ride the trailer).
func TestNamedSearchArgsOptsRoundtrip(t *testing.T) {
	q := []float32{1, 0, 0, 0}
	cases := []struct {
		name   string
		filter vector.Filter
		rc     uint8
		opa    uint8
	}{
		{"linearizable+filter", vector.Filter{Op: vector.FilterEq, Field: "lang", Value: vector.NewString("en")}, ConsistencyLinearizable, 1},
		{"leader+nofilter", vector.Filter{}, ConsistencyLeaderOnly, 0},
		{"anyreplica+opa+filter", vector.Filter{Op: vector.FilterEq, Field: "x", Value: vector.NewInt(1)}, ConsistencyAnyReplica, 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			enc := EncodeNamedSearchArgsOpts("c", "title", q, 5, tc.filter, tc.rc, tc.opa, 0)
			col, vn, got, k, gotFilter, rc, opa, _, err := DecodeNamedSearchArgsOpts(enc)
			if err != nil {
				t.Fatal(err)
			}
			if col != "c" || vn != "title" || k != 5 {
				t.Errorf("col=%q name=%q k=%d", col, vn, k)
			}
			if !reflect.DeepEqual(got, q) {
				t.Errorf("query = %v", got)
			}
			if !reflect.DeepEqual(gotFilter, tc.filter) {
				t.Errorf("filter = %+v, want %+v", gotFilter, tc.filter)
			}
			if rc != tc.rc || opa != tc.opa {
				t.Errorf("rc=%d opa=%d, want rc=%d opa=%d", rc, opa, tc.rc, tc.opa)
			}
		})
	}
}

// TestNamedSearchArgsOptsByteIdentical proves the no-rc/no-opa opts encoder is
// BYTE-IDENTICAL to the legacy EncodeNamedSearchArgs (backward-compat: old
// decoders read the same base, the trailing-bytes contract holds).
func TestNamedSearchArgsOptsByteIdentical(t *testing.T) {
	q := []float32{1, 2, 3, 4}
	for _, filter := range []vector.Filter{
		{},
		{Op: vector.FilterEq, Field: "lang", Value: vector.NewString("en")},
	} {
		legacy := EncodeNamedSearchArgs("c", "title", q, 5, filter)
		opts := EncodeNamedSearchArgsOpts("c", "title", q, 5, filter, 0, 0, 0)
		if !bytes.Equal(legacy, opts) {
			t.Errorf("no-rc/no-opa opts encode != legacy: legacy=%x opts=%x", legacy, opts)
		}
	}
	// And the legacy decoder must still read an opts-carrying arg (trailing bytes
	// ignored): the base fields decode unchanged.
	enc := EncodeNamedSearchArgsOpts("c", "title", q, 5, vector.Filter{}, ConsistencyLinearizable, 0, 0)
	col, vn, got, k, _, err := DecodeNamedSearchArgs(enc)
	if err != nil {
		t.Fatalf("legacy decode of opts-carrying arg: %v", err)
	}
	if col != "c" || vn != "title" || k != 5 || !reflect.DeepEqual(got, q) {
		t.Errorf("legacy decode mismatch: col=%q name=%q k=%d q=%v", col, vn, k, got)
	}
}

// TestNamedSearchArgsOptsMalformed asserts a present marker with a truncated
// rc/opa block fails loud (never a silent rc=0 degrade).
func TestNamedSearchArgsOptsMalformed(t *testing.T) {
	base := EncodeNamedSearchArgs("c", "title", []float32{1, 2, 3, 4}, 5, vector.Filter{})
	// marker says opts present, but the rc/opa bytes are missing/short.
	if _, _, _, _, _, _, _, _, err := DecodeNamedSearchArgsOpts(append(append([]byte{}, base...), NamedTrailerOpts)); err == nil {
		t.Error("missing rc/opa block: expected error, got nil")
	}
	if _, _, _, _, _, _, _, _, err := DecodeNamedSearchArgsOpts(append(append([]byte{}, base...), NamedTrailerOpts, 2)); err == nil {
		t.Error("short rc/opa block (1 byte): expected error, got nil")
	}
}

// TestNamedScrollArgsOptsRoundtrip round-trips cursor + rc/opa through the
// scroll opts trailer in every present/absent combination.
func TestNamedScrollArgsOptsRoundtrip(t *testing.T) {
	filter := vector.Filter{Op: vector.FilterEq, Field: "lang", Value: vector.NewString("en")}
	cases := []struct {
		name     string
		afterID  uint64
		hasAfter bool
		rc       uint8
		opa      uint8
	}{
		{"cursor+opts", 42, true, ConsistencyLinearizable, 1},
		{"cursor-only", 7, true, 0, 0},
		{"opts-only", 0, false, ConsistencyLeaderOnly, 0},
		{"neither", 0, false, 0, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			enc := EncodeNamedScrollArgsOptsBounded("c", filter, 25, tc.afterID, tc.hasAfter, tc.rc, tc.opa, 0)
			col, gotF, limit, afterID, hasAfter, rc, opa, _, err := DecodeNamedScrollArgsOpts(enc)
			if err != nil {
				t.Fatal(err)
			}
			if col != "c" || limit != 25 {
				t.Errorf("col=%q limit=%d", col, limit)
			}
			if !reflect.DeepEqual(gotF, filter) {
				t.Errorf("filter = %+v", gotF)
			}
			if hasAfter != tc.hasAfter || afterID != tc.afterID {
				t.Errorf("afterID=%d hasAfter=%v, want %d/%v", afterID, hasAfter, tc.afterID, tc.hasAfter)
			}
			if rc != tc.rc || opa != tc.opa {
				t.Errorf("rc=%d opa=%d, want %d/%d", rc, opa, tc.rc, tc.opa)
			}
		})
	}
}

// TestNamedScrollArgsOptsByteIdentical proves: (1) no-cursor/no-opts opts encode
// == legacy EncodeNamedScrollArgs (the base, NOTHING appended); (2) the
// cursor-only opts encode == the legacy EncodeNamedScrollArgsCursor form.
func TestNamedScrollArgsOptsByteIdentical(t *testing.T) {
	filter := vector.Filter{Op: vector.FilterEq, Field: "x", Value: vector.NewInt(1)}
	// (1) neither cursor nor opts: byte-identical to the plain scroll base.
	base := EncodeNamedScrollArgs("c", filter, 10)
	none := EncodeNamedScrollArgsOptsBounded("c", filter, 10, 0, false, 0, 0, 0)
	if !bytes.Equal(base, none) {
		t.Errorf("no-cursor/no-opts opts encode != legacy base: base=%x none=%x", base, none)
	}
	// (2) cursor-only: byte-identical to the legacy cursor encoder.
	legacyCursor := EncodeNamedScrollArgsCursor("c", filter, 10, 99, true)
	optsCursor := EncodeNamedScrollArgsOptsBounded("c", filter, 10, 99, true, 0, 0, 0)
	if !bytes.Equal(legacyCursor, optsCursor) {
		t.Errorf("cursor-only opts encode != legacy cursor: legacy=%x opts=%x", legacyCursor, optsCursor)
	}
	// And the legacy cursor decoder reads an opts-carrying arg's cursor unchanged.
	enc := EncodeNamedScrollArgsOptsBounded("c", filter, 10, 99, true, ConsistencyLinearizable, 0, 0)
	col, _, limit, afterID, hasAfter, err := DecodeNamedScrollArgsCursor(enc)
	if err != nil {
		t.Fatalf("legacy cursor decode of opts-carrying arg: %v", err)
	}
	if col != "c" || limit != 10 || afterID != 99 || !hasAfter {
		t.Errorf("legacy cursor decode mismatch: col=%q limit=%d afterID=%d hasAfter=%v", col, limit, afterID, hasAfter)
	}
}

// TestNamedScrollArgsOptsMalformed asserts truncated cursor/opts blocks fail loud.
func TestNamedScrollArgsOptsMalformed(t *testing.T) {
	base := EncodeNamedScrollArgs("c", vector.Filter{}, 10)
	// marker says cursor present, afterID missing.
	if _, _, _, _, _, _, _, _, err := DecodeNamedScrollArgsOpts(append(append([]byte{}, base...), NamedScrollCursor)); err == nil {
		t.Error("missing afterID: expected error, got nil")
	}
	// marker says opts present, rc/opa missing.
	if _, _, _, _, _, _, _, _, err := DecodeNamedScrollArgsOpts(append(append([]byte{}, base...), NamedScrollOpts)); err == nil {
		t.Error("missing rc/opa: expected error, got nil")
	}
	// marker says both present, only afterID supplied (opts short).
	var idb [8]byte
	binary.BigEndian.PutUint64(idb[:], 5)
	trailer := append([]byte{NamedScrollCursor | NamedScrollOpts}, idb[:]...)
	if _, _, _, _, _, _, _, _, err := DecodeNamedScrollArgsOpts(append(append([]byte{}, base...), trailer...)); err == nil {
		t.Error("cursor present but opts truncated: expected error, got nil")
	}
}

// TestReadConsistencyOfNamed asserts ReadConsistencyOf arms the shard barrier for
// each named read op carrying rc, and returns (0,false) for legacy args. This is
// THE guard against the silent stale-serve regression: if a named op is dropped
// from ReadConsistencyOf, a Linearizable named read skips the shard data barrier.
func TestReadConsistencyOfNamed(t *testing.T) {
	q := []float32{1, 0, 0, 0}
	// vector_named_search / _search_docs share the named-search wire.
	searchRC := EncodeNamedSearchArgsOpts("c", "title", q, 5, vector.Filter{}, ConsistencyLinearizable, 0, 0)
	for _, op := range []string{"vector_named_search", "vector_named_search_docs"} {
		if rc, ok := ReadConsistencyOf(op, searchRC); !ok || rc != ConsistencyLinearizable {
			t.Errorf("ReadConsistencyOf(%q, rc=2) = (%d,%v), want (2,true)", op, rc, ok)
		}
		// Legacy (no trailer) args decode cleanly ⇒ (0,true): rc=0 is AnyReplica, so
		// no barrier fires (matching the dense/MV legacy contract in
		// TestReadConsistencyOfLegacyArgs). The CRITICAL property is rc != Linearizable.
		legacy := EncodeNamedSearchArgs("c", "title", q, 5, vector.Filter{})
		if rc, ok := ReadConsistencyOf(op, legacy); !ok || rc != ConsistencyAnyReplica {
			t.Errorf("ReadConsistencyOf(%q, legacy) = (%d,%v), want (0,true)", op, rc, ok)
		}
	}
	// vector_named_scroll.
	scrollRC := EncodeNamedScrollArgsOptsBounded("c", vector.Filter{}, 10, 0, false, ConsistencyLinearizable, 0, 0)
	if rc, ok := ReadConsistencyOf("vector_named_scroll", scrollRC); !ok || rc != ConsistencyLinearizable {
		t.Errorf("ReadConsistencyOf(vector_named_scroll, rc=2) = (%d,%v), want (2,true)", rc, ok)
	}
	legacyScroll := EncodeNamedScrollArgs("c", vector.Filter{}, 10)
	if rc, ok := ReadConsistencyOf("vector_named_scroll", legacyScroll); !ok || rc != ConsistencyAnyReplica {
		t.Errorf("ReadConsistencyOf(vector_named_scroll, legacy) = (%d,%v), want (0,true)", rc, ok)
	}
}

func TestNamedNameArgsRoundtrip(t *testing.T) {
	col, err := DecodeNamedNameArgs(EncodeNamedNameArgs("acme/docs"))
	if err != nil {
		t.Fatal(err)
	}
	if col != "acme/docs" {
		t.Errorf("col = %q", col)
	}
	if _, err := DecodeNamedNameArgs(nil); err == nil {
		t.Error("nil args: expected error")
	}
}

func TestNamedConfigResultRoundtrip(t *testing.T) {
	cfg := map[string]vector.NamedVectorParams{
		"title": {Dim: 4, Metric: vector.Cosine},
		"image": {Dim: 8, Metric: vector.DotProduct, EfSearch: 32},
	}
	got, err := DecodeNamedConfigResult(EncodeNamedConfigResult(cfg))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, cfg) {
		t.Errorf("cfg = %+v, want %+v", got, cfg)
	}
	if _, err := DecodeNamedConfigResult(nil); err == nil {
		t.Error("nil body: expected error")
	}
}

// ---- op-kind contract ----

// TestNamedOpKinds asserts the Raft serialization contract: create/insert/
// delete/drop are OpReadWrite; search/search_docs/scroll/get_config are OpReadOnly.
func TestNamedOpKinds(t *testing.T) {
	r := NewRegistry()
	if err := RegisterBuiltins(r); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	cases := []struct {
		op   string
		kind OpKind
	}{
		{"vector_named_create_collection", OpReadWrite},
		{"vector_named_drop_collection", OpReadWrite},
		{"vector_named_insert", OpReadWrite},
		{"vector_named_delete", OpReadWrite},
		{"vector_named_search", OpReadOnly},
		{"vector_named_search_docs", OpReadOnly},
		{"vector_named_scroll", OpReadOnly},
		{"vector_named_get_config", OpReadOnly},
	}
	for _, tc := range cases {
		_, kind, ke, ok := r.Lookup(tc.op)
		if !ok {
			t.Fatalf("op %q not registered", tc.op)
		}
		if kind != tc.kind {
			t.Fatalf("op %q kind = %d, want %d", tc.op, kind, tc.kind)
		}
		if ke == nil {
			t.Fatalf("op %q has nil KeyExtractor (must be routable)", tc.op)
		}
	}
}

// TestNamedCollectionNameFor confirms partition routing can peek the collection
// name from each named op's args (At1 layout), needed for fan-out.
func TestNamedCollectionNameFor(t *testing.T) {
	cfg := map[string]vector.NamedVectorParams{"title": {Dim: 4}}
	cases := []struct {
		op   string
		args []byte
	}{
		{"vector_named_create_collection", EncodeNamedCreateArgs("docs", cfg, 0)},
		{"vector_named_drop_collection", EncodeNamedNameArgs("docs")},
		{"vector_named_insert", EncodeNamedInsertArgs("docs", 1, map[string][]float32{"title": {1, 2, 3, 4}}, nil, 0)},
		{"vector_named_delete", EncodeNamedDeleteArgs("docs", 1)},
		{"vector_named_search", EncodeNamedSearchArgs("docs", "title", []float32{1, 2, 3, 4}, 5, vector.Filter{})},
		{"vector_named_search_docs", EncodeNamedSearchArgs("docs", "title", []float32{1, 2, 3, 4}, 5, vector.Filter{})},
		{"vector_named_scroll", EncodeNamedScrollArgs("docs", vector.Filter{}, 10)},
		{"vector_named_get_config", EncodeNamedNameArgs("docs")},
	}
	for _, tc := range cases {
		name, ok := CollectionNameFor(tc.op, tc.args)
		if !ok {
			t.Errorf("op %q: CollectionNameFor returned ok=false", tc.op)
			continue
		}
		if name != "default/docs" {
			t.Errorf("op %q: name = %q, want default/docs", tc.op, name)
		}
	}
}

// ---- dispatch through handlers on a real TxContext ----

func newNamedTx(t *testing.T) *TxContext {
	t.Helper()
	c, _ := cache.New(cache.DefaultConfig())
	t.Cleanup(func() { _ = c.Close() })
	vstore, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = vstore.Close() })
	return NewTxContextWithVectors(c, vstore)
}

// TestNamedViaDispatch drives the full op path: create a 2-space named
// collection -> insert points (map of vecs) -> search a space (+ filtered) ->
// search_docs (payload returned) -> delete -> scroll -> get_config.
func TestNamedViaDispatch(t *testing.T) {
	tx := newNamedTx(t)
	cfg := map[string]vector.NamedVectorParams{
		"title": {Dim: 4, Metric: vector.Cosine},
		"image": {Dim: 4, Metric: vector.Cosine},
	}
	if _, err := handleNamedCreate(tx, EncodeNamedCreateArgs("docs", cfg, 0)); err != nil {
		t.Fatalf("create: %v", err)
	}

	// point 1: aligned with the title query; tagged en
	if _, err := handleNamedInsert(tx, EncodeNamedInsertArgs("docs", 1,
		map[string][]float32{"title": {1, 0, 0, 0}, "image": {0, 0, 1, 0}},
		vector.Metadata{"lang": vector.NewString("en")}, 0)); err != nil {
		t.Fatalf("insert 1: %v", err)
	}
	// point 2: not aligned with the title query; tagged fr. omits image.
	if _, err := handleNamedInsert(tx, EncodeNamedInsertArgs("docs", 2,
		map[string][]float32{"title": {0, 1, 0, 0}},
		vector.Metadata{"lang": vector.NewString("fr")}, 0)); err != nil {
		t.Fatalf("insert 2: %v", err)
	}

	// search title space
	body, err := handleNamedSearch(tx, EncodeNamedSearchArgs("docs", "title", []float32{1, 0, 0, 0}, 5, vector.Filter{}))
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	res, err := DecodeVectorSearchResults(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 || res[0].ID != 1 {
		t.Fatalf("search = %+v, want point 1 first, 2 results", res)
	}

	// filtered search (lang=en) returns only point 1
	enFilter := vector.Filter{Op: vector.FilterEq, Field: "lang", Value: vector.NewString("en")}
	body, err = handleNamedSearch(tx, EncodeNamedSearchArgs("docs", "title", []float32{1, 0, 0, 0}, 5, enFilter))
	if err != nil {
		t.Fatalf("filtered search: %v", err)
	}
	res, _ = DecodeVectorSearchResults(body)
	if len(res) != 1 || res[0].ID != 1 {
		t.Fatalf("filtered search = %+v, want only point 1", res)
	}

	// search_docs returns the shared payload
	body, err = handleNamedSearchDocs(tx, EncodeNamedSearchArgs("docs", "title", []float32{1, 0, 0, 0}, 5, vector.Filter{}))
	if err != nil {
		t.Fatalf("search_docs: %v", err)
	}
	docs, err := DecodeVectorDocs(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 2 {
		t.Fatalf("search_docs = %d docs, want 2", len(docs))
	}
	var found bool
	for _, d := range docs {
		if d.ID == 1 {
			found = true
			if d.Metadata["lang"].Str != "en" {
				t.Errorf("point 1 payload = %+v, want lang=en", d.Metadata)
			}
		}
	}
	if !found {
		t.Errorf("point 1 missing from search_docs: %+v", docs)
	}

	// search the image space: only point 1 populated it
	body, _ = handleNamedSearch(tx, EncodeNamedSearchArgs("docs", "image", []float32{0, 0, 1, 0}, 5, vector.Filter{}))
	res, _ = DecodeVectorSearchResults(body)
	if len(res) != 1 || res[0].ID != 1 {
		t.Fatalf("image search = %+v, want only point 1", res)
	}

	// get_config returns the configured spaces
	body, err = handleNamedGetConfig(tx, EncodeNamedNameArgs("docs"))
	if err != nil {
		t.Fatalf("get_config: %v", err)
	}
	gotCfg, err := DecodeNamedConfigResult(body)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotCfg, cfg) {
		t.Errorf("get_config = %+v, want %+v", gotCfg, cfg)
	}

	// delete point 1 from every space + shared payload
	db, err := handleNamedDelete(tx, EncodeNamedDeleteArgs("docs", 1))
	if err != nil {
		t.Fatalf("delete: %v", err)
	}
	if len(db) == 0 || db[0] != 1 {
		t.Errorf("delete = %v, want [1]", db)
	}
	// point 1 gone from title AND image
	body, _ = handleNamedSearch(tx, EncodeNamedSearchArgs("docs", "title", []float32{1, 0, 0, 0}, 5, vector.Filter{}))
	res, _ = DecodeVectorSearchResults(body)
	if len(res) != 1 || res[0].ID != 2 {
		t.Errorf("title after delete = %+v, want only point 2", res)
	}
	body, _ = handleNamedSearch(tx, EncodeNamedSearchArgs("docs", "image", []float32{0, 0, 1, 0}, 5, vector.Filter{}))
	res, _ = DecodeVectorSearchResults(body)
	if len(res) != 0 {
		t.Errorf("image after delete = %+v, want empty", res)
	}

	// scroll returns the live point set (only point 2 now) with payload
	body, err = handleNamedScroll(tx, EncodeNamedScrollArgs("docs", vector.Filter{}, 0))
	if err != nil {
		t.Fatalf("scroll: %v", err)
	}
	docs, _ = DecodeVectorDocs(body)
	if len(docs) != 1 || docs[0].ID != 2 || docs[0].Metadata["lang"].Str != "fr" {
		t.Fatalf("scroll = %+v, want only point 2 (lang=fr)", docs)
	}

	// drop is idempotent
	if _, err := handleNamedDrop(tx, EncodeNamedNameArgs("docs")); err != nil {
		t.Fatalf("drop: %v", err)
	}
	if _, err := handleNamedDrop(tx, EncodeNamedNameArgs("docs")); err != nil {
		t.Fatalf("idempotent drop: %v", err)
	}
}

// ---- fail-loud ----

func TestNamedFailLoudBadFilter(t *testing.T) {
	tx := newNamedTx(t)
	cfg := map[string]vector.NamedVectorParams{"title": {Dim: 4}}
	if _, err := handleNamedCreate(tx, EncodeNamedCreateArgs("docs", cfg, 0)); err != nil {
		t.Fatal(err)
	}
	bad := vector.Filter{Op: vector.FilterRegex, Field: "title", Value: vector.NewString("(unclosed")}

	body, err := handleNamedSearch(tx, EncodeNamedSearchArgs("docs", "title", []float32{1, 0, 0, 0}, 5, bad))
	if err == nil {
		t.Error("search with bad filter: expected Compile error, got nil")
	}
	if body != nil {
		t.Errorf("bad filter must not return a body, got %x", body)
	}

	body, err = handleNamedScroll(tx, EncodeNamedScrollArgs("docs", bad, 0))
	if err == nil {
		t.Error("scroll with bad filter: expected Compile error, got nil")
	}
	if body != nil {
		t.Errorf("bad filter scroll must not return a body, got %x", body)
	}
}

func TestNamedFailLoudUnknownVectorName(t *testing.T) {
	tx := newNamedTx(t)
	cfg := map[string]vector.NamedVectorParams{"title": {Dim: 4}}
	if _, err := handleNamedCreate(tx, EncodeNamedCreateArgs("docs", cfg, 0)); err != nil {
		t.Fatal(err)
	}
	// insert with an unknown space name
	if _, err := handleNamedInsert(tx, EncodeNamedInsertArgs("docs", 1,
		map[string][]float32{"nope": {1, 2, 3, 4}}, nil, 0)); !errors.Is(err, vector.ErrUnknownVectorName) {
		t.Errorf("insert unknown name: err = %v, want ErrUnknownVectorName", err)
	}
	// search an unknown space name
	if _, err := handleNamedSearch(tx, EncodeNamedSearchArgs("docs", "nope", []float32{1, 2, 3, 4}, 5, vector.Filter{})); !errors.Is(err, vector.ErrUnknownVectorName) {
		t.Errorf("search unknown name: err = %v, want ErrUnknownVectorName", err)
	}
}

func TestNamedFailLoudDimMismatch(t *testing.T) {
	tx := newNamedTx(t)
	cfg := map[string]vector.NamedVectorParams{"title": {Dim: 4}}
	if _, err := handleNamedCreate(tx, EncodeNamedCreateArgs("docs", cfg, 0)); err != nil {
		t.Fatal(err)
	}
	if _, err := handleNamedInsert(tx, EncodeNamedInsertArgs("docs", 1,
		map[string][]float32{"title": {1, 2, 3}}, nil, 0)); !errors.Is(err, vector.ErrDimMismatch) {
		t.Errorf("insert dim mismatch: err = %v, want ErrDimMismatch", err)
	}
}

// ---- named batch get (vector_named_get_batch) ----

// TestNamedGetBatchResultRoundtrip encodes a mix of found/not-found rows with
// various projections and decodes them back, asserting ids + order + the
// vectors-map / meta / ttl projection survive the round-trip. The named clone of
// TestVectorGetBatchResultRoundtrip (the named row carries a vectors MAP + ttl,
// no sparse lane).
func TestNamedGetBatchResultRoundtrip(t *testing.T) {
	meta := vector.Metadata{"a": vector.NewInt(1), "tag": vector.NewString("x")}
	vecs := map[string][]float32{"title": {1, 2, 3, 4}, "image": {5, 6, 7, 8}}

	rows := []NamedGetBatchRow{
		{ID: 1, Found: true, Vectors: vecs, Meta: meta, TTLMs: 5000}, // both projections
		{ID: 2, Found: false},                            // not-found
		{ID: 3, Found: true, Meta: meta, TTLMs: 0},       // with_vector off (no map)
		{ID: 4, Found: true, Vectors: vecs, TTLMs: 1000}, // with_payload off (no meta)
		{ID: 5, Found: true},                             // bare
	}
	got, err := DecodeNamedGetBatchResult(EncodeNamedGetBatchResult(rows))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(rows) {
		t.Fatalf("rows = %d, want %d", len(got), len(rows))
	}
	for i := range rows {
		if got[i].ID != rows[i].ID || got[i].Found != rows[i].Found {
			t.Errorf("row %d: got (id=%d,found=%v), want (id=%d,found=%v)", i, got[i].ID, got[i].Found, rows[i].ID, rows[i].Found)
		}
	}
	if !reflect.DeepEqual(got[0].Vectors, vecs) || got[0].Meta["a"].Int != 1 || got[0].TTLMs != 5000 {
		t.Errorf("row 0 = %+v", got[0])
	}
	if len(got[2].Vectors) != 0 || got[2].Meta == nil {
		t.Errorf("row 2 (with_vector off): %+v", got[2])
	}
	if !reflect.DeepEqual(got[3].Vectors, vecs) || got[3].Meta != nil {
		t.Errorf("row 3 (with_payload off): %+v", got[3])
	}
	if len(got[4].Vectors) != 0 || got[4].Meta != nil || got[4].TTLMs != 0 {
		t.Errorf("row 4 (bare): %+v", got[4])
	}

	// zero rows.
	if z, err := DecodeNamedGetBatchResult(EncodeNamedGetBatchResult(nil)); err != nil || len(z) != 0 {
		t.Errorf("zero-row result: rows=%d err=%v", len(z), err)
	}
	// truncation (fail-loud).
	body := EncodeNamedGetBatchResult(rows)
	if _, err := DecodeNamedGetBatchResult(nil); !errors.Is(err, ErrVectorArgsTruncated) {
		t.Errorf("nil: err = %v, want ErrVectorArgsTruncated", err)
	}
	if _, err := DecodeNamedGetBatchResult(body[:2]); !errors.Is(err, ErrVectorArgsTruncated) {
		t.Errorf("header chop: err = %v, want ErrVectorArgsTruncated", err)
	}
	if _, err := DecodeNamedGetBatchResult(body[:10]); !errors.Is(err, ErrVectorArgsTruncated) {
		t.Errorf("mid-row chop: err = %v, want ErrVectorArgsTruncated", err)
	}
}

// TestHandleNamedGetBatch drives handleNamedGetBatch over a built named
// collection: a mixed present/absent batch preserves input order, a partial miss
// is a found=0 row (never an op error), and the projection flags gate the
// vectors-map and the payload. Args reuse EncodeVectorGetBatchArgs verbatim. The
// named clone of TestHandleVectorGetBatchDense.
func TestHandleNamedGetBatch(t *testing.T) {
	tx := newNamedTx(t)
	cfg := map[string]vector.NamedVectorParams{
		"title": {Dim: 4, Metric: vector.L2},
		"image": {Dim: 4, Metric: vector.L2},
	}
	if _, err := handleNamedCreate(tx, EncodeNamedCreateArgs("docs", cfg, 0)); err != nil {
		t.Fatal(err)
	}
	if _, err := handleNamedInsert(tx, EncodeNamedInsertArgs("docs", 1,
		map[string][]float32{"title": {1, 0, 0, 0}, "image": {0, 0, 1, 0}},
		vector.Metadata{"a": vector.NewInt(1)}, time.Hour)); err != nil {
		t.Fatal(err)
	}
	if _, err := handleNamedInsert(tx, EncodeNamedInsertArgs("docs", 2,
		map[string][]float32{"title": {0, 1, 0, 0}},
		vector.Metadata{"b": vector.NewInt(2)}, 0)); err != nil {
		t.Fatal(err)
	}

	// mixed batch: present, absent, present, absent — input order preserved.
	body, err := handleNamedGetBatch(tx, EncodeVectorGetBatchArgs("docs", []uint64{1, 99, 2, 7}, GetFlagsBoth))
	if err != nil {
		t.Fatalf("unexpected op error: %v", err)
	}
	rows, err := DecodeNamedGetBatchResult(body)
	if err != nil {
		t.Fatal(err)
	}
	wantIDs := []uint64{1, 99, 2, 7}
	wantFound := []bool{true, false, true, false}
	if len(rows) != 4 {
		t.Fatalf("rows = %d, want 4", len(rows))
	}
	for i, r := range rows {
		if r.ID != wantIDs[i] || r.Found != wantFound[i] {
			t.Errorf("row %d: got (id=%d,found=%v), want (id=%d,found=%v)", i, r.ID, r.Found, wantIDs[i], wantFound[i])
		}
	}
	if rows[0].Vectors["title"][0] != 1 || rows[0].Meta["a"].Int != 1 || rows[0].TTLMs == 0 {
		t.Errorf("row 0 found content: %+v", rows[0])
	}
	if rows[2].Vectors["title"][1] != 1 || rows[2].Meta["b"].Int != 2 {
		t.Errorf("row 2 found content: %+v", rows[2])
	}

	// projection: with_vector off -> no map, payload kept; with_payload off -> map kept, no meta.
	body, _ = handleNamedGetBatch(tx, EncodeVectorGetBatchArgs("docs", []uint64{1}, GetFlagWithPayload))
	rows, _ = DecodeNamedGetBatchResult(body)
	if !rows[0].Found || len(rows[0].Vectors) != 0 || rows[0].Meta["a"].Int != 1 {
		t.Errorf("with_vector off: %+v", rows[0])
	}
	body, _ = handleNamedGetBatch(tx, EncodeVectorGetBatchArgs("docs", []uint64{1}, GetFlagWithVector))
	rows, _ = DecodeNamedGetBatchResult(body)
	if !rows[0].Found || rows[0].Vectors["title"][0] != 1 || rows[0].Meta != nil {
		t.Errorf("with_payload off: %+v", rows[0])
	}

	// all-absent batch -> all not-found rows, NO op error.
	body, err = handleNamedGetBatch(tx, EncodeVectorGetBatchArgs("docs", []uint64{100, 200, 300}, GetFlagsBoth))
	if err != nil {
		t.Fatalf("all-absent: unexpected op error %v", err)
	}
	rows, _ = DecodeNamedGetBatchResult(body)
	if len(rows) != 3 {
		t.Fatalf("all-absent: rows = %d, want 3", len(rows))
	}
	for i, r := range rows {
		if r.Found {
			t.Errorf("all-absent row %d: found = true, want false", i)
		}
	}

	// empty batch -> zero rows, no error.
	body, err = handleNamedGetBatch(tx, EncodeVectorGetBatchArgs("docs", nil, GetFlagsBoth))
	if err != nil {
		t.Fatal(err)
	}
	if rows, _ := DecodeNamedGetBatchResult(body); len(rows) != 0 {
		t.Fatalf("empty batch: rows = %d, want 0", len(rows))
	}
}

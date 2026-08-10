// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/vector"
)

// richFilter builds a filter tree exercising every Task-1 operator (match,
// regex, is_empty, is_null, and a dt_gte+dt_lt datetime range) plus a dotted
// path and a legacy leaf, so a single round-trip asserts the whole new-op set
// survives JSON encode/decode through the wire codecs.
func richFilter() vector.Filter {
	return vector.Filter{
		Op: vector.FilterAnd,
		And: []vector.Filter{
			{Op: vector.FilterMatch, Field: "title", Value: vector.NewString("hello world")},
			{Op: vector.FilterRegex, Field: "sku", Value: vector.NewString("^A[0-9]+$")},
			{Op: vector.FilterIsEmpty, Field: "tags"},
			{Op: vector.FilterIsNull, Field: "deleted_at"},
			{Op: vector.FilterDtGte, Field: "created", Value: vector.NewString("2024-01-01T00:00:00Z")},
			{Op: vector.FilterDtLt, Field: "created", Value: vector.NewString("2024-12-31T23:59:59Z")},
			{Op: vector.FilterEq, Field: "address.city", Value: vector.NewString("NYC")}, // dotted path
		},
	}
}

// TestRichFilterSearchArgsRoundtrip asserts the rich-op filter tree survives
// EncodeVectorSearchArgsExt -> DecodeVectorSearchArgs byte-for-byte (the filter
// rides as JSON inside the existing length-prefixed blob; no byte-format change).
func TestRichFilterSearchArgsRoundtrip(t *testing.T) {
	orig := richFilter()
	args := EncodeVectorSearchArgsExt("docs", 7, []float32{0.1, 0.2, 0.3}, orig)
	_, _, _, got, err := DecodeVectorSearchArgs(args)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !reflect.DeepEqual(orig, got) {
		t.Errorf("search filter roundtrip mismatch:\n orig=%+v\n got =%+v", orig, got)
	}
	// The decoded filter must still compile (proves every op name resolved to a
	// real op and every value is well-formed).
	if _, cerr := got.Compile(); cerr != nil {
		t.Errorf("decoded filter does not compile: %v", cerr)
	}
}

// TestRichFilterScrollArgsRoundtrip mirrors the search case through the scroll codec.
func TestRichFilterScrollArgsRoundtrip(t *testing.T) {
	orig := richFilter()
	_, got, limit, err := DecodeScrollArgs(EncodeScrollArgs("acme/docs", orig, 25))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if limit != 25 {
		t.Errorf("limit = %d, want 25", limit)
	}
	if !reflect.DeepEqual(orig, got) {
		t.Errorf("scroll filter roundtrip mismatch:\n orig=%+v\n got =%+v", orig, got)
	}
	if _, cerr := got.Compile(); cerr != nil {
		t.Errorf("decoded filter does not compile: %v", cerr)
	}
}

// TestRichFilterDeleteByFilterArgsRoundtrip mirrors the search case through the
// delete-by-filter codec.
func TestRichFilterDeleteByFilterArgsRoundtrip(t *testing.T) {
	orig := richFilter()
	_, got, err := DecodeDeleteByFilterArgs(EncodeDeleteByFilterArgs("docs", orig))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !reflect.DeepEqual(orig, got) {
		t.Errorf("delete-by-filter roundtrip mismatch:\n orig=%+v\n got =%+v", orig, got)
	}
	if _, cerr := got.Compile(); cerr != nil {
		t.Errorf("decoded filter does not compile: %v", cerr)
	}
}

// TestRichFilterHybridArgsRoundtrip mirrors the search case through the hybrid codec.
func TestRichFilterHybridArgsRoundtrip(t *testing.T) {
	orig := richFilter()
	opts := vector.HybridOpts{Filter: orig}
	_, _, _, _, got, err := DecodeHybridSearchArgs(EncodeHybridSearchArgs("docs", []float32{1, 2}, 5, vector.SparseVector{}, opts))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !reflect.DeepEqual(orig, got.Filter) {
		t.Errorf("hybrid filter roundtrip mismatch:\n orig=%+v\n got =%+v", orig, got.Filter)
	}
	if _, cerr := got.Filter.Compile(); cerr != nil {
		t.Errorf("decoded filter does not compile: %v", cerr)
	}
}

// TestRichFilterGroupArgsRoundtrip mirrors the search case through the groups codec.
func TestRichFilterGroupArgsRoundtrip(t *testing.T) {
	orig := richFilter()
	opts := vector.GroupOpts{GroupBy: "doc", Filter: orig}
	_, _, _, got, err := DecodeGroupSearchArgs(EncodeGroupSearchArgs("docs", 3, []float32{1, 2, 3}, opts))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !reflect.DeepEqual(orig, got.Filter) {
		t.Errorf("group filter roundtrip mismatch:\n orig=%+v\n got =%+v", orig, got.Filter)
	}
	if _, cerr := got.Filter.Compile(); cerr != nil {
		t.Errorf("decoded filter does not compile: %v", cerr)
	}
}

// TestLegacyFilterBlobByteIdentical asserts a legacy-op-only filter (eq/and/range)
// encodes and decodes UNCHANGED — proving the new ops introduced no byte-format
// change and old blobs are still honored verbatim.
func TestLegacyFilterBlobByteIdentical(t *testing.T) {
	legacy := vector.Filter{
		Op: vector.FilterAnd,
		And: []vector.Filter{
			{Op: vector.FilterEq, Field: "tenant", Value: vector.NewString("acme")},
			{Op: vector.FilterGte, Field: "ts", Value: vector.NewInt(100)},
			{Op: vector.FilterLt, Field: "ts", Value: vector.NewInt(200)},
		},
	}
	args := EncodeVectorSearchArgsExt("docs", 5, []float32{0.1, 0.2}, legacy)

	_, _, _, got, err := DecodeVectorSearchArgs(args)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !reflect.DeepEqual(legacy, got) {
		t.Errorf("legacy filter roundtrip mismatch:\n orig=%+v\n got =%+v", legacy, got)
	}

	// Re-encoding the decoded filter must yield byte-identical args — the JSON
	// embedding is stable for the legacy ops (no reordering, no new fields).
	if re := EncodeVectorSearchArgsExt("docs", 5, []float32{0.1, 0.2}, got); !bytes.Equal(args, re) {
		t.Errorf("legacy filter re-encode not byte-identical:\n first=%x\n again=%x", args, re)
	}
}

// TestUnknownFilterOpNameDecodeError asserts that a wire blob carrying an
// unknown op NAME in its filter JSON fails loud at decode time (the
// UnmarshalText contract) rather than silently dropping the filter.
func TestUnknownFilterOpNameDecodeError(t *testing.T) {
	// Hand-craft a search args blob whose filter JSON uses a bogus op name.
	badJSON := []byte(`{"op":"totally_not_a_real_op","field":"x","value":"y"}`)
	args := encodeSearchArgsWithRawFilter(t, "docs", 3, []float32{1, 2, 3}, badJSON)

	if _, _, _, _, err := DecodeVectorSearchArgs(args); err == nil {
		t.Fatal("expected decode error for unknown op name, got nil")
	} else if !strings.Contains(err.Error(), "decode filter") {
		t.Errorf("error %q does not mention filter decode", err)
	}
}

// --- Fail-loud: invalid regex / RFC3339 surface a clean op error, no panic ---

// TestHandleSearchInvalidRegexFailsLoud asserts the vector_search handler
// returns a non-nil error (not a panic, not an empty/unfiltered result) when the
// decoded filter carries an invalid RE2 pattern.
func TestHandleSearchInvalidRegexFailsLoud(t *testing.T) {
	tx, query := newRichFilterTx(t)
	bad := vector.Filter{Op: vector.FilterRegex, Field: "title", Value: vector.NewString("(unclosed")}
	body, err := handleVectorSearch(tx, EncodeVectorSearchArgsExt("docs", 3, query, bad))
	if err == nil {
		t.Fatalf("expected error for invalid regex, got body=%x", body)
	}
	if body != nil {
		t.Errorf("invalid filter must not return a result body, got %x", body)
	}
}

// TestHandleScrollInvalidDatetimeFailsLoud asserts the vector_scroll handler
// returns a non-nil error when the decoded filter carries an invalid RFC3339.
func TestHandleScrollInvalidDatetimeFailsLoud(t *testing.T) {
	tx, _ := newRichFilterTx(t)
	bad := vector.Filter{Op: vector.FilterDtGte, Field: "created", Value: vector.NewString("not-a-date")}
	body, err := handleVectorScroll(tx, EncodeScrollArgs("docs", bad, 0))
	if err == nil {
		t.Fatalf("expected error for invalid datetime, got body=%x", body)
	}
	if body != nil {
		t.Errorf("invalid filter must not return a result body, got %x", body)
	}
}

// TestHandleDeleteByFilterInvalidFailsLoud asserts the vector_delete_by_filter
// handler returns a non-nil error for an invalid regex (a bad filter must never
// drive deletes against an unfiltered/wrong candidate set).
func TestHandleDeleteByFilterInvalidFailsLoud(t *testing.T) {
	tx, _ := newRichFilterTx(t)
	bad := vector.Filter{Op: vector.FilterRegex, Field: "title", Value: vector.NewString("[")}
	body, err := handleVectorDeleteByFilter(tx, EncodeDeleteByFilterArgs("docs", bad))
	if err == nil {
		t.Fatalf("expected error for invalid regex, got body=%x", body)
	}
	if body != nil {
		t.Errorf("invalid filter must not return a result body, got %x", body)
	}
}

// TestHandleSearchDocsInvalidDatetimeFailsLoud covers the search_docs handler
// (used by the RAG/scroll-adjacent path) with an invalid RFC3339 datetime.
func TestHandleSearchDocsInvalidDatetimeFailsLoud(t *testing.T) {
	tx, query := newRichFilterTx(t)
	bad := vector.Filter{Op: vector.FilterDtLt, Field: "created", Value: vector.NewString("2024-13-99")}
	body, err := handleVectorSearchDocs(tx, EncodeVectorSearchArgsExt("docs", 3, query, bad))
	if err == nil {
		t.Fatalf("expected error for invalid datetime, got body=%x", body)
	}
	if body != nil {
		t.Errorf("invalid filter must not return a result body, got %x", body)
	}
}

// TestHandleHybridInvalidRegexFailsLoud covers the hybrid-search handler.
func TestHandleHybridInvalidRegexFailsLoud(t *testing.T) {
	tx, query := newRichFilterTx(t)
	bad := vector.HybridOpts{Filter: vector.Filter{Op: vector.FilterRegex, Field: "title", Value: vector.NewString("*invalid")}}
	body, err := handleVectorHybridSearch(tx, EncodeHybridSearchArgs("docs", query, 3, vector.SparseVector{}, bad))
	if err == nil {
		t.Fatalf("expected error for invalid regex, got body=%x", body)
	}
	if body != nil {
		t.Errorf("invalid filter must not return a result body, got %x", body)
	}
}

// TestHandleGroupsInvalidRegexFailsLoud covers the search_groups handler.
func TestHandleGroupsInvalidRegexFailsLoud(t *testing.T) {
	tx, query := newRichFilterTx(t)
	opts := vector.GroupOpts{GroupBy: "doc", Filter: vector.Filter{Op: vector.FilterRegex, Field: "title", Value: vector.NewString("(")}}
	body, err := handleVectorSearchGroups(tx, EncodeGroupSearchArgs("docs", 2, query, opts))
	if err == nil {
		t.Fatalf("expected error for invalid regex, got body=%x", body)
	}
	if body != nil {
		t.Errorf("invalid filter must not return a result body, got %x", body)
	}
}

// TestHandleHybridLanesInvalidRegexFailsLoud covers the hybrid_lanes fan-out
// primitive (used by the partitioned hybrid-search coordinator).
func TestHandleHybridLanesInvalidRegexFailsLoud(t *testing.T) {
	tx, query := newRichFilterTx(t)
	bad := vector.HybridOpts{Filter: vector.Filter{Op: vector.FilterRegex, Field: "title", Value: vector.NewString("*invalid")}}
	body, err := handleVectorHybridLanes(tx, EncodeHybridSearchArgs("docs", query, 3, vector.SparseVector{}, bad))
	if err == nil {
		t.Fatalf("expected error for invalid regex, got body=%x", body)
	}
	if body != nil {
		t.Errorf("invalid filter must not return a result body, got %x", body)
	}
}

// TestHandleGroupCandidatesInvalidRegexFailsLoud covers the group_candidates
// fan-out primitive (used by the partitioned group-search coordinator).
func TestHandleGroupCandidatesInvalidRegexFailsLoud(t *testing.T) {
	tx, query := newRichFilterTx(t)
	opts := vector.GroupOpts{GroupBy: "doc", Filter: vector.Filter{Op: vector.FilterRegex, Field: "title", Value: vector.NewString("(")}}
	body, err := handleVectorGroupCandidates(tx, EncodeGroupSearchArgs("docs", 2, query, opts))
	if err == nil {
		t.Fatalf("expected error for invalid regex, got body=%x", body)
	}
	if body != nil {
		t.Errorf("invalid filter must not return a result body, got %x", body)
	}
}

// TestHandleSearchValidRichFilterSucceeds is the positive control: a valid rich
// filter (regex + match + datetime range) is accepted by the handler and
// returns a clean (possibly empty) result — proving the fail-loud path rejects
// only genuinely-invalid filters, not all rich filters.
func TestHandleSearchValidRichFilterSucceeds(t *testing.T) {
	tx, query := newRichFilterTx(t)
	good := vector.Filter{
		Op: vector.FilterAnd,
		And: []vector.Filter{
			{Op: vector.FilterRegex, Field: "title", Value: vector.NewString("^chunk")},
			{Op: vector.FilterDtGte, Field: "created", Value: vector.NewString("2024-01-01T00:00:00Z")},
		},
	}
	if _, err := handleVectorSearch(tx, EncodeVectorSearchArgsExt("docs", 3, query, good)); err != nil {
		t.Fatalf("valid rich filter rejected: %v", err)
	}
}

// --- helpers ---

// newRichFilterTx builds a tx with a small "docs" collection (dim 3) holding a
// few records carrying a string "title" and an int64-ms "created" datetime
// field, and returns a usable query vector.
func newRichFilterTx(t *testing.T) (*TxContext, []float32) {
	t.Helper()
	c, _ := cache.New(cache.DefaultConfig())
	t.Cleanup(func() { _ = c.Close() })
	vstore, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = vstore.Close() })
	tx := NewTxContextWithVectors(c, vstore)

	cfg := vector.Config{Dim: 3, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("docs", cfg)); err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 6; i++ {
		meta := vector.Metadata{
			"title":   vector.NewString("chunk"),
			"doc":     vector.NewInt(int64(i % 2)),
			"created": vector.NewInt(int64(1_704_067_200_000) + int64(i)*1000), // 2024-01-01T00:00:00Z + i s, in ms
		}
		args := EncodeVectorUpsertArgs("docs", uint64(i), []float32{float32(i), 0, 0}, "chunk", 0, meta, vector.SparseVector{})
		if _, err := handleVectorUpsert(tx, args); err != nil {
			t.Fatalf("upsert %d: %v", i, err)
		}
	}
	return tx, []float32{1, 0, 0}
}

// encodeSearchArgsWithRawFilter builds a vector_search args blob whose filter
// block carries an arbitrary raw JSON payload (used to inject a malformed
// filter the normal encoder would never produce). It mirrors the wire layout of
// EncodeVectorSearchArgsExt with vecFlagFilter set.
func encodeSearchArgsWithRawFilter(t *testing.T, collection string, k int, query []float32, filterJSON []byte) []byte {
	t.Helper()
	// Sanity: the injected JSON is well-formed JSON (only the op NAME is bogus),
	// so the failure we assert is the op-name resolution, not a JSON syntax error.
	if !json.Valid(filterJSON) {
		t.Fatalf("test bug: injected filter JSON is not valid JSON: %s", filterJSON)
	}
	// Hand-encode the exact vector_search wire layout with vecFlagFilter set:
	// [flags:u8][colLen:u8][col][k:u32][dim:u32][query][filterLen:u32][filterJSON]
	var buf bytes.Buffer
	var u32 [4]byte
	putU32 := func(v uint32) { binary.BigEndian.PutUint32(u32[:], v); buf.Write(u32[:]) }
	buf.WriteByte(vecFlagFilter)
	buf.WriteByte(byte(len(collection)))
	buf.WriteString(collection)
	putU32(uint32(k))
	putU32(uint32(len(query)))
	for _, f := range query {
		putU32(math.Float32bits(f))
	}
	putU32(uint32(len(filterJSON)))
	buf.Write(filterJSON)
	return buf.Bytes()
}

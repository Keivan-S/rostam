// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"reflect"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// geoFilter builds a filter tree exercising every geo operator (radius / box /
// polygon) each carrying a GeoCondition, so a single round-trip asserts the geo
// op set + the new Filter.Geo pointer survive JSON encode/decode through the
// wire codecs with NO byte-format change (geo, like the rich ops, rides as JSON
// inside the existing length-prefixed filter blob).
func geoFilter() vector.Filter {
	return vector.Filter{
		Op: vector.FilterAnd,
		And: []vector.Filter{
			{Op: vector.FilterGeoRadius, Field: "loc", Geo: &vector.GeoCondition{
				CenterLat: 48.8566, CenterLon: 2.3522, RadiusM: 5000,
			}},
			{Op: vector.FilterGeoBox, Field: "loc", Geo: &vector.GeoCondition{
				MinLat: 48.0, MinLon: 2.0, MaxLat: 49.0, MaxLon: 3.0,
			}},
			{Op: vector.FilterGeoPolygon, Field: "loc", Geo: &vector.GeoCondition{
				Polygon: []float64{48.0, 2.0, 49.0, 2.0, 49.0, 3.0, 48.0, 3.0},
			}},
		},
	}
}

// TestGeoFilterSearchArgsRoundtrip asserts the geo-op filter tree survives
// EncodeVectorSearchArgsExt -> DecodeVectorSearchArgs and the decoded filter
// compiles (proves every geo op name resolved and every GeoCondition is valid).
func TestGeoFilterSearchArgsRoundtrip(t *testing.T) {
	orig := geoFilter()
	args := EncodeVectorSearchArgsExt("docs", 7, []float32{0.1, 0.2, 0.3}, orig)
	_, _, _, got, err := DecodeVectorSearchArgs(args)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !reflect.DeepEqual(orig, got) {
		t.Errorf("search geo-filter roundtrip mismatch:\n orig=%+v\n got =%+v", orig, got)
	}
	if _, cerr := vector.CompileFilter(got); cerr != nil {
		t.Errorf("decoded geo filter does not compile: %v", cerr)
	}
}

// TestGeoFilterScrollArgsRoundtrip mirrors the search case through the scroll codec.
func TestGeoFilterScrollArgsRoundtrip(t *testing.T) {
	orig := geoFilter()
	_, got, limit, err := DecodeScrollArgs(EncodeScrollArgs("acme/docs", orig, 25))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if limit != 25 {
		t.Errorf("limit = %d, want 25", limit)
	}
	if !reflect.DeepEqual(orig, got) {
		t.Errorf("scroll geo-filter roundtrip mismatch:\n orig=%+v\n got =%+v", orig, got)
	}
	if _, cerr := vector.CompileFilter(got); cerr != nil {
		t.Errorf("decoded geo filter does not compile: %v", cerr)
	}
}

// TestGeoFilterDeleteByFilterArgsRoundtrip mirrors the search case through the
// delete-by-filter codec (the path that, with a bad filter, would otherwise
// drive an over-broad delete).
func TestGeoFilterDeleteByFilterArgsRoundtrip(t *testing.T) {
	orig := geoFilter()
	_, got, err := DecodeDeleteByFilterArgs(EncodeDeleteByFilterArgs("docs", orig))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !reflect.DeepEqual(orig, got) {
		t.Errorf("delete-by-filter geo roundtrip mismatch:\n orig=%+v\n got =%+v", orig, got)
	}
	if _, cerr := vector.CompileFilter(got); cerr != nil {
		t.Errorf("decoded geo filter does not compile: %v", cerr)
	}
}

// TestGeoFilterHybridArgsRoundtrip mirrors the search case through the hybrid codec.
func TestGeoFilterHybridArgsRoundtrip(t *testing.T) {
	orig := geoFilter()
	opts := vector.HybridOpts{Filter: orig}
	_, _, _, _, got, err := DecodeHybridSearchArgs(EncodeHybridSearchArgs("docs", []float32{1, 2}, 5, vector.SparseVector{}, opts))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !reflect.DeepEqual(orig, got.Filter) {
		t.Errorf("hybrid geo-filter roundtrip mismatch:\n orig=%+v\n got =%+v", orig, got.Filter)
	}
	if _, cerr := vector.CompileFilter(got.Filter); cerr != nil {
		t.Errorf("decoded geo filter does not compile: %v", cerr)
	}
}

// TestGeoFilterGroupArgsRoundtrip mirrors the search case through the groups codec.
func TestGeoFilterGroupArgsRoundtrip(t *testing.T) {
	orig := geoFilter()
	opts := vector.GroupOpts{GroupBy: "doc", Filter: orig}
	_, _, _, got, err := DecodeGroupSearchArgs(EncodeGroupSearchArgs("docs", 3, []float32{1, 2, 3}, opts))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !reflect.DeepEqual(orig, got.Filter) {
		t.Errorf("group geo-filter roundtrip mismatch:\n orig=%+v\n got =%+v", orig, got.Filter)
	}
	if _, cerr := vector.CompileFilter(got.Filter); cerr != nil {
		t.Errorf("decoded geo filter does not compile: %v", cerr)
	}
}

// TestGeoMetadataInsertArgsRoundtrip asserts a ValueGeo metadata field
// ({"kind":"geo","lat":…,"lon":…}) survives EncodeVectorInsertArgsExt ->
// DecodeVectorInsertArgs exactly. The metadata, like the filter, rides as JSON
// inside the existing insert-args blob — no byte-format change for the new kind.
func TestGeoMetadataInsertArgsRoundtrip(t *testing.T) {
	meta := vector.Metadata{
		"loc":   vector.NewGeo(48.8566, 2.3522),
		"label": vector.NewString("paris"),
	}
	args := EncodeVectorInsertArgsExt("docs", 42, []float32{1, 2, 3}, 0, meta, vector.SparseVector{})
	col, id, _, _, got, _, _, err := DecodeVectorInsertArgs(args)
	if err != nil {
		t.Fatalf("decode insert args: %v", err)
	}
	if col != "docs" || id != 42 {
		t.Fatalf("decoded (col,id) = (%q,%d), want (docs,42)", col, id)
	}
	gloc, ok := got["loc"]
	if !ok {
		t.Fatalf("decoded metadata missing 'loc' geo field: %+v", got)
	}
	if gloc.Kind != vector.ValueGeo || gloc.Lat != 48.8566 || gloc.Lon != 2.3522 {
		t.Fatalf("decoded geo value = %+v, want kind=geo lat=48.8566 lon=2.3522", gloc)
	}
	if !reflect.DeepEqual(meta, got) {
		t.Errorf("geo-metadata insert roundtrip mismatch:\n orig=%+v\n got =%+v", meta, got)
	}
}

// TestLegacyFilterBlobByteIdenticalWithGeoPresent re-affirms the byte-format
// invariant from the rich-filtering work AFTER adding the Geo pointer field:
// a legacy-op-only filter (no Geo) still encodes/decodes/re-encodes
// byte-identical — the additive, omitempty Geo field introduced no wire change.
func TestLegacyFilterBlobByteIdenticalWithGeoPresent(t *testing.T) {
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
	if got.Geo != nil {
		t.Errorf("legacy filter decoded a non-nil Geo: %+v", got.Geo)
	}
	if re := EncodeVectorSearchArgsExt("docs", 5, []float32{0.1, 0.2}, got); !bytes.Equal(args, re) {
		t.Errorf("legacy filter re-encode not byte-identical:\n first=%x\n again=%x", args, re)
	}
}

// --- Fail-loud: invalid geo filter surfaces a clean op error, no panic ---

// TestHandleSearchInvalidGeoFailsLoud asserts the vector_search handler returns
// a non-nil error (not a panic, not an unfiltered result) when the decoded
// filter carries an out-of-range geo radius condition.
func TestHandleSearchInvalidGeoFailsLoud(t *testing.T) {
	tx, query := newRichFilterTx(t)
	bad := vector.Filter{Op: vector.FilterGeoRadius, Field: "loc", Geo: &vector.GeoCondition{
		CenterLat: 95, CenterLon: 0, RadiusM: 1000, // latitude out of range
	}}
	body, err := handleVectorSearch(tx, EncodeVectorSearchArgsExt("docs", 3, query, bad))
	if err == nil {
		t.Fatalf("expected error for out-of-range geo, got body=%x", body)
	}
	if body != nil {
		t.Errorf("invalid geo filter must not return a result body, got %x", body)
	}
}

// TestHandleDeleteByFilterInvalidGeoFailsLoud asserts the vector_delete_by_filter
// handler returns a non-nil error for a nil-Geo geo op (a bad filter must never
// drive deletes against an unfiltered/wrong candidate set).
func TestHandleDeleteByFilterInvalidGeoFailsLoud(t *testing.T) {
	tx, _ := newRichFilterTx(t)
	bad := vector.Filter{Op: vector.FilterGeoBox, Field: "loc", Geo: nil} // nil geo condition
	body, err := handleVectorDeleteByFilter(tx, EncodeDeleteByFilterArgs("docs", bad))
	if err == nil {
		t.Fatalf("expected error for nil geo condition, got body=%x", body)
	}
	if body != nil {
		t.Errorf("invalid geo filter must not return a result body, got %x", body)
	}
}

// TestHandleScrollInvalidGeoPolygonFailsLoud asserts the vector_scroll handler
// rejects a polygon with too few vertices (bad arity).
func TestHandleScrollInvalidGeoPolygonFailsLoud(t *testing.T) {
	tx, _ := newRichFilterTx(t)
	bad := vector.Filter{Op: vector.FilterGeoPolygon, Field: "loc", Geo: &vector.GeoCondition{
		Polygon: []float64{1, 2, 3, 4}, // only 2 vertices, need >= 3
	}}
	body, err := handleVectorScroll(tx, EncodeScrollArgs("docs", bad, 0))
	if err == nil {
		t.Fatalf("expected error for bad polygon arity, got body=%x", body)
	}
	if body != nil {
		t.Errorf("invalid geo filter must not return a result body, got %x", body)
	}
}

// TestHandleSearchValidGeoFilterSucceeds is the positive control: a valid geo
// radius filter is accepted by the handler and returns a clean (possibly empty)
// result — proving the fail-loud path rejects only genuinely-invalid geo.
func TestHandleSearchValidGeoFilterSucceeds(t *testing.T) {
	tx, query := newRichFilterTx(t)
	good := vector.Filter{Op: vector.FilterGeoRadius, Field: "loc", Geo: &vector.GeoCondition{
		CenterLat: 0, CenterLon: 0, RadiusM: 100000,
	}}
	if _, err := handleVectorSearch(tx, EncodeVectorSearchArgsExt("docs", 3, query, good)); err != nil {
		t.Fatalf("valid geo filter rejected: %v", err)
	}
}

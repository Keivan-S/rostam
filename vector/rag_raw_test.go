// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"reflect"
	"testing"
)

// TestRawDocumentMirrorsDocument is the guard that makes RawDocument safe to
// substitute for Document in a JSON response: the two must expose the SAME JSON
// fields, in the same order, with the same tags. Only the metadata field's Go
// type may differ (decoded map vs raw bytes) — that difference is the entire
// point, and it is JSON-invisible because the raw bytes are the marshalling of
// that map.
//
// Without this test, adding a field to Document would silently start dropping it
// from every response the raw path renders.
func TestRawDocumentMirrorsDocument(t *testing.T) {
	assertJSONShapeMirrors(t, reflect.TypeOf(Document{}), reflect.TypeOf(RawDocument{}), "Metadata")
}

// TestRawGroupMirrorsGroup is TestRawDocumentMirrorsDocument for the grouped
// shape: same fields and tags, with Key raw instead of decoded and Hits carrying
// RawDocuments instead of Documents.
func TestRawGroupMirrorsGroup(t *testing.T) {
	assertJSONShapeMirrors(t, reflect.TypeOf(Group{}), reflect.TypeOf(RawGroup{}), "Key", "Hits")
}

// assertJSONShapeMirrors fails unless want and got declare the same fields in the
// same order with the same json tags. Fields named in typeMayDiffer are allowed
// to carry a different Go type (their JSON rendering is asserted elsewhere); every
// other field must match on type too.
func assertJSONShapeMirrors(t *testing.T, want, got reflect.Type, typeMayDiffer ...string) {
	t.Helper()
	exempt := make(map[string]bool, len(typeMayDiffer))
	for _, n := range typeMayDiffer {
		exempt[n] = true
	}
	if want.NumField() != got.NumField() {
		t.Fatalf("%s has %d fields, %s has %d — they must mirror each other",
			want, want.NumField(), got, got.NumField())
	}
	for i := 0; i < want.NumField(); i++ {
		w, g := want.Field(i), got.Field(i)
		if w.Name != g.Name {
			t.Errorf("field %d: %s.%s vs %s.%s — names and order must match", i, want, w.Name, got, g.Name)
			continue
		}
		if w.Tag.Get("json") != g.Tag.Get("json") {
			t.Errorf("field %s: json tag %q vs %q", w.Name, w.Tag.Get("json"), g.Tag.Get("json"))
		}
		if !exempt[w.Name] && w.Type != g.Type {
			t.Errorf("field %s: type %s vs %s", w.Name, w.Type, g.Type)
		}
	}
}

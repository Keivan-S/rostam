// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"encoding/json"
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// TestJSONIDAcceptsNumberAndString pins the two input forms. The engine's ids
// are full-width uint64, but a JavaScript client's JSON number is a double, so
// anything above 2^53-1 needs the decimal-string form to survive the trip.
func TestJSONIDAcceptsNumberAndString(t *testing.T) {
	const big = uint64(1)<<63 + 12345 // far above any double's exact range
	for _, tc := range []struct {
		in   string
		want uint64
		bad  bool
	}{
		{in: `7`, want: 7},
		{in: `"7"`, want: 7},
		{in: `0`, want: 0},
		{in: `"0"`, want: 0},
		{in: strconv.FormatUint(big, 10), want: big},
		{in: `"` + strconv.FormatUint(big, 10) + `"`, want: big},
		{in: `"18446744073709551615"`, want: math.MaxUint64},
		{in: `-1`, bad: true},
		{in: `"-1"`, bad: true},
		{in: `1.5`, bad: true},
		{in: `"abc"`, bad: true},
		{in: `""`, bad: true},
		{in: `"18446744073709551616"`, bad: true}, // one past uint64
		{in: `null`, bad: true},
		{in: `[1]`, bad: true},
	} {
		var got jsonID
		err := json.Unmarshal([]byte(tc.in), &got)
		if tc.bad {
			if err == nil {
				t.Fatalf("jsonID(%s) = %d, want an error", tc.in, got)
			}
			continue
		}
		if err != nil {
			t.Fatalf("jsonID(%s): %v", tc.in, err)
		}
		if uint64(got) != tc.want {
			t.Fatalf("jsonID(%s) = %d, want %d", tc.in, got, tc.want)
		}
	}
}

// TestJSONIDMarshalsAsNumber pins jsonID's own (incidental) marshal
// behavior: it has no custom MarshalJSON, so its underlying uint64 always
// writes as a plain number. That's fine because jsonID is decode-only — no
// generic tool result marshals one directly; see TestIDOutBoundary for the
// type that actually decides a result id's wire form.
func TestJSONIDMarshalsAsNumber(t *testing.T) {
	b, err := json.Marshal(map[string]jsonID{"id": 42})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(b) != `{"id":42}` {
		t.Fatalf("got %s, want a plain number", b)
	}
}

// TestIDOutBoundary pins idOut's decision at the jsSafeIDMask boundary: a
// number at or below it stays a JSON number, one above it becomes a decimal
// string. This is what keeps a big id round-trip-safe through a JavaScript
// client (see idOut's doc comment) — get it backwards and either small ids
// grow needless quotes or big ids silently go back to rounding.
func TestIDOutBoundary(t *testing.T) {
	for _, id := range []uint64{0, 1, jsSafeIDMask} {
		out := idOut(id)
		n, ok := out.(uint64)
		if !ok || n != id {
			t.Fatalf("idOut(%d) = %#v (%T), want the uint64 %d", id, out, out, id)
		}
	}
	for _, id := range []uint64{jsSafeIDMask + 1, math.MaxUint64} {
		out := idOut(id)
		s, ok := out.(string)
		if !ok || s != strconv.FormatUint(id, 10) {
			t.Fatalf("idOut(%d) = %#v (%T), want the decimal string %q", id, out, out, strconv.FormatUint(id, 10))
		}
	}

	// Round-trip through the actual wire encoding: below the boundary must
	// decode as a JSON number (float64), above it as a string.
	b, err := json.Marshal(map[string]any{"lo": idOut(jsSafeIDMask), "hi": idOut(jsSafeIDMask + 1)})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, ok := m["lo"].(float64); !ok {
		t.Fatalf("lo = %#v (%T), want a JSON number", m["lo"], m["lo"])
	}
	if _, ok := m["hi"].(string); !ok {
		t.Fatalf("hi = %#v (%T), want a JSON string", m["hi"], m["hi"])
	}
}

// TestMemoryIDIsJSSafe: a generated memory id is handed to the client and
// comes straight back to forget. A JavaScript client parses it as a double, so
// an id above 2^53-1 would round and forget would report the memory missing.
func TestMemoryIDIsJSSafe(t *testing.T) {
	for _, ns := range []string{"default", "projA", "", "a-longer-namespace-name"} {
		for i := range 2000 {
			id := memoryID(ns, "fact number "+strconv.Itoa(i))
			if id > jsSafeIDMask {
				t.Fatalf("memoryID(%q, %d) = %d exceeds 2^53-1", ns, i, id)
			}
			// The real contract: it survives a round trip through a double.
			if uint64(float64(id)) != id {
				t.Fatalf("memoryID(%q, %d) = %d does not round-trip through float64", ns, i, id)
			}
		}
	}
}

// TestMemoryIDStillDedupes: masking must not break the reason the id is
// derived from the content in the first place.
func TestMemoryIDStillDedupes(t *testing.T) {
	first := memoryID("ns", "same fact")
	again := memoryID("ns", "same fact")
	if first != again {
		t.Fatal("the same (namespace, content) must give the same id")
	}
	if memoryID("a", "fact") == memoryID("b", "fact") {
		t.Fatal("the same content in different namespaces should not collide")
	}
}

func TestJSONToMetadataScalars(t *testing.T) {
	raw := map[string]json.RawMessage{
		"s": json.RawMessage(`"x"`), "i": json.RawMessage(`2`),
		"f": json.RawMessage(`2.5`), "b": json.RawMessage(`true`),
	}
	m, err := jsonToMetadata(raw)
	if err != nil {
		t.Fatalf("jsonToMetadata: %v", err)
	}
	if m["s"].Kind != vector.ValueString || m["s"].Str != "x" {
		t.Fatalf("s: %+v", m["s"])
	}
	if m["i"].Kind != vector.ValueInt || m["i"].Int != 2 {
		t.Fatalf("i: %+v", m["i"])
	}
	if m["f"].Kind != vector.ValueFloat || m["f"].Flt != 2.5 {
		t.Fatalf("f: %+v", m["f"])
	}
	if m["b"].Kind != vector.ValueBool || !m["b"].Bool {
		t.Fatalf("b: %+v", m["b"])
	}
}

func TestJSONToMetadataArrays(t *testing.T) {
	raw := map[string]json.RawMessage{
		"strs":   json.RawMessage(`["a","b"]`),
		"ints":   json.RawMessage(`[1,2]`),
		"floats": json.RawMessage(`[1,2.5]`),
	}
	m, err := jsonToMetadata(raw)
	if err != nil {
		t.Fatalf("jsonToMetadata: %v", err)
	}
	if m["strs"].Kind != vector.ValueStrings || len(m["strs"].Strs) != 2 || m["strs"].Strs[0] != "a" || m["strs"].Strs[1] != "b" {
		t.Fatalf("strs: %+v", m["strs"])
	}
	if m["ints"].Kind != vector.ValueInts || len(m["ints"].Ints) != 2 || m["ints"].Ints[0] != 1 || m["ints"].Ints[1] != 2 {
		t.Fatalf("ints: %+v", m["ints"])
	}
	if m["floats"].Kind != vector.ValueFloats || len(m["floats"].Flts) != 2 || m["floats"].Flts[0] != 1 || m["floats"].Flts[1] != 2.5 {
		t.Fatalf("floats: %+v", m["floats"])
	}
}

func TestJSONToMetadataRejectsNested(t *testing.T) {
	_, err := jsonToMetadata(map[string]json.RawMessage{"o": json.RawMessage(`{"a":1}`)})
	if err == nil {
		t.Fatal("nested object must be rejected")
	}
	if !strings.Contains(err.Error(), "o") {
		t.Fatalf("error must name the key: %v", err)
	}
}

func TestJSONToMetadataRejectsNull(t *testing.T) {
	if _, err := jsonToMetadata(map[string]json.RawMessage{"n": json.RawMessage(`null`)}); err == nil {
		t.Fatal("null must be rejected")
	}
}

func TestJSONToMetadataRejectsMixedArray(t *testing.T) {
	if _, err := jsonToMetadata(map[string]json.RawMessage{"m": json.RawMessage(`[1,"a"]`)}); err == nil {
		t.Fatal("mixed-type array must be rejected")
	}
}

func TestJSONToMetadataRejectsNestedArray(t *testing.T) {
	if _, err := jsonToMetadata(map[string]json.RawMessage{"n": json.RawMessage(`[[1,2],[3,4]]`)}); err == nil {
		t.Fatal("nested array must be rejected")
	}
}

func TestMetadataToJSONInverse(t *testing.T) {
	m := vector.Metadata{
		"s":        vector.NewString("x"),
		"i":        vector.NewInt(2),
		"f":        vector.NewFloat(2.5),
		"b":        vector.NewBool(true),
		"strs":     vector.NewStrings([]string{"a", "b"}),
		"ints":     vector.NewInts([]int64{1, 2}),
		"floats":   vector.NewFloats([]float64{1, 2.5}),
		"$content": vector.NewString("hidden"),
	}
	out := metadataToJSON(m)
	if _, ok := out["$content"]; ok {
		t.Fatal("$content must be skipped")
	}
	if out["s"] != "x" {
		t.Fatalf("s: %+v", out["s"])
	}
	if out["i"] != int64(2) {
		t.Fatalf("i: %+v", out["i"])
	}
	if out["f"] != 2.5 {
		t.Fatalf("f: %+v", out["f"])
	}
	if out["b"] != true {
		t.Fatalf("b: %+v", out["b"])
	}
	strs, ok := out["strs"].([]string)
	if !ok || len(strs) != 2 || strs[0] != "a" || strs[1] != "b" {
		t.Fatalf("strs: %+v", out["strs"])
	}
	ints, ok := out["ints"].([]int64)
	if !ok || len(ints) != 2 || ints[0] != 1 || ints[1] != 2 {
		t.Fatalf("ints: %+v", out["ints"])
	}
	floats, ok := out["floats"].([]float64)
	if !ok || len(floats) != 2 || floats[0] != 1 || floats[1] != 2.5 {
		t.Fatalf("floats: %+v", out["floats"])
	}
}

func TestParseFilterEmpty(t *testing.T) {
	f, err := parseFilter(nil)
	if err != nil {
		t.Fatalf("parseFilter(nil): %v", err)
	}
	if !f.IsZero() {
		t.Fatalf("expected zero filter, got %+v", f)
	}
	f, err = parseFilter(json.RawMessage(``))
	if err != nil {
		t.Fatalf("parseFilter(empty): %v", err)
	}
	if !f.IsZero() {
		t.Fatalf("expected zero filter, got %+v", f)
	}
}

// TestParseFilterCompound verifies vector.Value has no custom UnmarshalJSON
// (confirmed by grep: no UnmarshalJSON/MarshalJSON method on Value in
// vector/*.go), so leaf filter values must use the tagged form
// {"kind":"...", "<field>":...} rather than a bare JSON scalar.
func TestParseFilterCompound(t *testing.T) {
	raw := json.RawMessage(`{
		"op": "and",
		"and": [
			{"op": "gte", "field": "price", "value": {"kind": "int", "int": 10}},
			{"op": "eq", "field": "ok", "value": {"kind": "bool", "bool": true}}
		]
	}`)
	f, err := parseFilter(raw)
	if err != nil {
		t.Fatalf("parseFilter: %v", err)
	}
	if f.Op != vector.FilterAnd {
		t.Fatalf("Op: %v", f.Op)
	}
	if len(f.And) != 2 {
		t.Fatalf("And: %+v", f.And)
	}
	if f.And[0].Op != vector.FilterGte || f.And[0].Field != "price" || f.And[0].Value.Int != 10 {
		t.Fatalf("And[0]: %+v", f.And[0])
	}
	if f.And[1].Op != vector.FilterEq || f.And[1].Field != "ok" || !f.And[1].Value.Bool {
		t.Fatalf("And[1]: %+v", f.And[1])
	}
}

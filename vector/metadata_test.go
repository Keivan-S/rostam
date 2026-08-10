// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestValueConstructors(t *testing.T) {
	cases := []struct {
		name string
		v    Value
		kind ValueKind
	}{
		{"string", NewString("hello"), ValueString},
		{"int", NewInt(42), ValueInt},
		{"float", NewFloat(3.14), ValueFloat},
		{"bool", NewBool(true), ValueBool},
		{"strings", NewStrings([]string{"a", "b"}), ValueStrings},
		{"ints", NewInts([]int64{1, 2, 3}), ValueInts},
		{"floats", NewFloats([]float64{1.5, 2.5}), ValueFloats},
		{"geo", NewGeo(48.8566, 2.3522), ValueGeo},
	}
	for _, c := range cases {
		if c.v.Kind != c.kind {
			t.Errorf("%s: kind = %d, want %d", c.name, c.v.Kind, c.kind)
		}
		if c.v.IsZero() {
			t.Errorf("%s: IsZero=true on non-none value", c.name)
		}
	}

	var zero Value
	if !zero.IsZero() {
		t.Error("zero Value: IsZero=false")
	}
	if zero.Kind != ValueNone {
		t.Errorf("zero Value kind = %d, want ValueNone", zero.Kind)
	}
}

func TestValueEqual(t *testing.T) {
	cases := []struct {
		a, b Value
		eq   bool
	}{
		{NewString("x"), NewString("x"), true},
		{NewString("x"), NewString("y"), false},
		{NewInt(1), NewInt(1), true},
		{NewInt(1), NewFloat(1), false},
		{NewBool(true), NewBool(true), true},
		{NewStrings([]string{"a", "b"}), NewStrings([]string{"a", "b"}), true},
		{NewStrings([]string{"a", "b"}), NewStrings([]string{"b", "a"}), false},
		{NewInts([]int64{1, 2, 3}), NewInts([]int64{1, 2, 3}), true},
		{NewFloats([]float64{1, 2}), NewFloats([]float64{1, 2, 3}), false},
		{Value{}, Value{}, true},
		// Geo: equal lat+lon -> true. CRITICAL: the pre-geo Value.Equal default
		// returned true, so without a ValueGeo case two differing geo points
		// would silently compare equal (always-match FilterEq/Ne on a geo field).
		{NewGeo(48.8566, 2.3522), NewGeo(48.8566, 2.3522), true},
		{NewGeo(48.8566, 2.3522), NewGeo(48.8566, 2.4000), false}, // differing lon
		{NewGeo(48.8566, 2.3522), NewGeo(40.0000, 2.3522), false}, // differing lat
		{NewGeo(48.8566, 2.3522), NewGeo(40.0000, 2.4000), false}, // both differ
		{NewGeo(48.8566, 2.3522), NewString("48.8566"), false},    // geo vs non-geo
	}
	for i, c := range cases {
		if got := c.a.Equal(c.b); got != c.eq {
			t.Errorf("case %d: Equal(%+v, %+v) = %v, want %v", i, c.a, c.b, got, c.eq)
		}
	}
}

func TestMetadataMapShape(t *testing.T) {
	m := Metadata{
		"tenant": NewString("acme"),
		"status": NewString("active"),
		"score":  NewInt(95),
		"tags":   NewStrings([]string{"prod", "v2"}),
	}
	if got := m["tenant"].Str; got != "acme" {
		t.Errorf("tenant = %q, want acme", got)
	}
	if got := m["score"].Int; got != 95 {
		t.Errorf("score = %d, want 95", got)
	}
	if _, present := m["missing"]; present {
		t.Error("missing key reported present")
	}
}

func TestValueJSONRoundtrip(t *testing.T) {
	cases := []Value{
		NewString("acme"),
		NewInt(95),
		NewFloat(3.14),
		NewBool(true),
		NewBool(false),
		NewStrings([]string{"a", "b", "c"}),
		NewInts([]int64{1, 2, 3}),
		NewFloats([]float64{1.5, 2.5}),
	}
	for i, v := range cases {
		data, err := json.Marshal(v)
		if err != nil {
			t.Fatalf("case %d Marshal: %v", i, err)
		}
		var got Value
		if err := json.Unmarshal(data, &got); err != nil {
			t.Fatalf("case %d Unmarshal(%s): %v", i, data, err)
		}
		if !got.Equal(v) {
			t.Errorf("case %d roundtrip: got %+v, want %+v (json=%s)", i, got, v, data)
		}
	}
}

func TestValueKindNameInJSON(t *testing.T) {
	data, err := json.Marshal(NewString("x"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"kind":"string"`) {
		t.Errorf("marshaled string value = %s, want kind:string", data)
	}
}

// TestValueGeoConstructor verifies NewGeo populates Kind and the Lat/Lon fields.
func TestValueGeoConstructor(t *testing.T) {
	v := NewGeo(48.8566, 2.3522)
	if v.Kind != ValueGeo {
		t.Errorf("kind = %d, want ValueGeo", v.Kind)
	}
	if v.Lat != 48.8566 {
		t.Errorf("Lat = %v, want 48.8566", v.Lat)
	}
	if v.Lon != 2.3522 {
		t.Errorf("Lon = %v, want 2.3522", v.Lon)
	}
	if v.IsZero() {
		t.Error("geo Value reports IsZero=true")
	}
}

// TestValueGeoJSONRoundtrip verifies the {"kind":"geo","lat":…,"lon":…} shape and
// exact float preservation through marshal+unmarshal.
func TestValueGeoJSONRoundtrip(t *testing.T) {
	v := NewGeo(48.8566, 2.3522)
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if !strings.Contains(string(data), `"kind":"geo"`) {
		t.Errorf("marshaled geo value = %s, want kind:geo", data)
	}
	if !strings.Contains(string(data), `"lat":48.8566`) {
		t.Errorf("marshaled geo value = %s, want lat:48.8566", data)
	}
	if !strings.Contains(string(data), `"lon":2.3522`) {
		t.Errorf("marshaled geo value = %s, want lon:2.3522", data)
	}
	var got Value
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal(%s): %v", data, err)
	}
	if got.Kind != ValueGeo || got.Lat != 48.8566 || got.Lon != 2.3522 {
		t.Errorf("roundtrip got %+v, want kind=geo lat=48.8566 lon=2.3522 (json=%s)", got, data)
	}
	if !got.Equal(v) {
		t.Errorf("roundtrip Equal failed: got %+v, want %+v", got, v)
	}

	// (0,0) is a VALID point (Gulf of Guinea), not an absence. omitempty on
	// lat/lon suppresses both fields in the JSON (keeping non-geo values clean),
	// producing {"kind":"geo"} — but unmarshal must still recover Lat=0, Lon=0.
	z := NewGeo(0, 0)
	zdata, err := json.Marshal(z)
	if err != nil {
		t.Fatalf("Marshal(0,0): %v", err)
	}
	var zgot Value
	if err := json.Unmarshal(zdata, &zgot); err != nil {
		t.Fatalf("Unmarshal(%s): %v", zdata, err)
	}
	if zgot.Kind != ValueGeo || zgot.Lat != 0 || zgot.Lon != 0 {
		t.Errorf("(0,0) roundtrip got %+v, want kind=geo lat=0 lon=0 (json=%s)", zgot, zdata)
	}
	if !zgot.Equal(z) {
		t.Errorf("(0,0) roundtrip Equal failed: got %+v, want %+v", zgot, z)
	}
}

// TestValueGeoUnmarshalText verifies "geo" decodes to ValueGeo and an unknown
// kind name still hard-errors (loud, not silent).
func TestValueGeoUnmarshalText(t *testing.T) {
	var k ValueKind
	if err := k.UnmarshalText([]byte("geo")); err != nil {
		t.Fatalf("UnmarshalText(geo): %v", err)
	}
	if k != ValueGeo {
		t.Errorf("UnmarshalText(geo) = %d, want ValueGeo", k)
	}
	if name, err := ValueGeo.MarshalText(); err != nil || string(name) != "geo" {
		t.Errorf("MarshalText(ValueGeo) = %q, %v; want \"geo\", nil", name, err)
	}
	var bad ValueKind
	if err := bad.UnmarshalText([]byte("definitely-not-a-kind")); err == nil {
		t.Error("UnmarshalText(unknown) succeeded; want hard error")
	}
}

// TestNumericValueGeoDeclines confirms a geo Value declines the numeric ordering
// path (so it never participates in gt/gte/lt/lte comparisons by accident).
func TestNumericValueGeoDeclines(t *testing.T) {
	f, ok := numericValue(NewGeo(48.8566, 2.3522))
	if ok || f != 0 {
		t.Errorf("numericValue(geo) = (%v, %v), want (0, false)", f, ok)
	}
}

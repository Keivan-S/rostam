// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"encoding/json"
	"math"
	"testing"
	"time"
)

func compileOrFail(t *testing.T, f Filter) Predicate {
	t.Helper()
	p, err := f.Compile()
	if err != nil {
		t.Fatalf("Compile(%+v): %v", f, err)
	}
	return p
}

func TestFilterEqNe(t *testing.T) {
	m := Metadata{"tenant": NewString("acme"), "score": NewInt(95)}

	eq := compileOrFail(t, Filter{Op: FilterEq, Field: "tenant", Value: NewString("acme")})
	if !eq(m) {
		t.Error("eq tenant=acme should match")
	}
	eqMiss := compileOrFail(t, Filter{Op: FilterEq, Field: "tenant", Value: NewString("globex")})
	if eqMiss(m) {
		t.Error("eq tenant=globex should not match")
	}
	// Missing field never matches eq.
	eqAbsent := compileOrFail(t, Filter{Op: FilterEq, Field: "missing", Value: NewString("x")})
	if eqAbsent(m) {
		t.Error("eq on missing field should not match")
	}

	ne := compileOrFail(t, Filter{Op: FilterNe, Field: "tenant", Value: NewString("globex")})
	if !ne(m) {
		t.Error("ne tenant!=globex should match")
	}
	// Strict-exists: ne on a missing field does NOT match.
	neAbsent := compileOrFail(t, Filter{Op: FilterNe, Field: "missing", Value: NewString("x")})
	if neAbsent(m) {
		t.Error("ne on missing field should not match (strict-exists)")
	}
}

func TestFilterOrdering(t *testing.T) {
	m := Metadata{"score": NewInt(50), "ratio": NewFloat(0.5), "name": NewString("m")}

	cases := []struct {
		op    FilterOp
		field string
		val   Value
		want  bool
	}{
		{FilterGt, "score", NewInt(40), true},
		{FilterGt, "score", NewInt(50), false},
		{FilterGte, "score", NewInt(50), true},
		{FilterLt, "score", NewInt(51), true},
		{FilterLte, "score", NewInt(50), true},
		{FilterGt, "score", NewFloat(49.9), true}, // int field vs float value
		{FilterLt, "ratio", NewFloat(0.6), true},
		{FilterGt, "name", NewString("a"), true}, // string ordering
		{FilterLt, "name", NewString("a"), false},
		{FilterGt, "missing", NewInt(0), false}, // missing field
		{FilterGt, "name", NewInt(5), false},    // kind mismatch
	}
	for i, c := range cases {
		p := compileOrFail(t, Filter{Op: c.op, Field: c.field, Value: c.val})
		if got := p(m); got != c.want {
			t.Errorf("case %d (%s %s %+v) = %v, want %v", i, mustOpName(c.op), c.field, c.val, got, c.want)
		}
	}
}

func TestFilterIn(t *testing.T) {
	m := Metadata{"lang": NewString("en"), "level": NewInt(2)}

	inStr := compileOrFail(t, Filter{Op: FilterIn, Field: "lang", Value: NewStrings([]string{"en", "fr"})})
	if !inStr(m) {
		t.Error("lang in [en,fr] should match")
	}
	inStrMiss := compileOrFail(t, Filter{Op: FilterIn, Field: "lang", Value: NewStrings([]string{"de", "fr"})})
	if inStrMiss(m) {
		t.Error("lang in [de,fr] should not match")
	}
	inInt := compileOrFail(t, Filter{Op: FilterIn, Field: "level", Value: NewInts([]int64{1, 2, 3})})
	if !inInt(m) {
		t.Error("level in [1,2,3] should match")
	}
}

func TestFilterContains(t *testing.T) {
	m := Metadata{"tags": NewStrings([]string{"prod", "v2"}), "perms": NewInts([]int64{4, 8})}

	c := compileOrFail(t, Filter{Op: FilterContains, Field: "tags", Value: NewString("prod")})
	if !c(m) {
		t.Error("tags contains prod should match")
	}
	cMiss := compileOrFail(t, Filter{Op: FilterContains, Field: "tags", Value: NewString("dev")})
	if cMiss(m) {
		t.Error("tags contains dev should not match")
	}
	cInt := compileOrFail(t, Filter{Op: FilterContains, Field: "perms", Value: NewInt(8)})
	if !cInt(m) {
		t.Error("perms contains 8 should match")
	}
}

func TestFilterComposition(t *testing.T) {
	m := Metadata{"tenant": NewString("acme"), "status": NewString("active"), "priority": NewInt(7)}

	// tenant=acme AND (status=active OR priority>5)
	f := Filter{
		Op: FilterAnd,
		And: []Filter{
			{Op: FilterEq, Field: "tenant", Value: NewString("acme")},
			{Op: FilterOr, Or: []Filter{
				{Op: FilterEq, Field: "status", Value: NewString("active")},
				{Op: FilterGt, Field: "priority", Value: NewInt(5)},
			}},
		},
	}
	if !compileOrFail(t, f)(m) {
		t.Error("composite filter should match")
	}

	// NOT tenant=acme → should not match
	notF := Filter{Op: FilterNot, Not: &Filter{Op: FilterEq, Field: "tenant", Value: NewString("acme")}}
	if compileOrFail(t, notF)(m) {
		t.Error("not tenant=acme should not match")
	}

	// tenant=globex AND status=active → false (first clause fails)
	f2 := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterEq, Field: "tenant", Value: NewString("globex")},
		{Op: FilterEq, Field: "status", Value: NewString("active")},
	}}
	if compileOrFail(t, f2)(m) {
		t.Error("and with failing clause should not match")
	}
}

func TestFilterZeroCompilesToNil(t *testing.T) {
	p, err := (Filter{}).Compile()
	if err != nil {
		t.Fatalf("zero filter Compile: %v", err)
	}
	if p != nil {
		t.Error("zero filter should compile to nil predicate (match all)")
	}
}

func TestFilterCompileErrors(t *testing.T) {
	cases := []Filter{
		{Op: FilterNot}, // not without child
		{Op: FilterEq},  // leaf without field
		{Op: FilterIn, Field: "x", Value: NewString("scalar")},             // in with scalar value
		{Op: FilterContains, Field: "x", Value: NewStrings([]string{"a"})}, // contains with array value
		{Op: FilterOp(250), Field: "x", Value: NewInt(1)},                  // unknown op
	}
	for i, f := range cases {
		if _, err := f.Compile(); err == nil {
			t.Errorf("case %d (%+v): Compile err = nil, want error", i, f)
		}
	}
}

func TestFilterJSONRoundtrip(t *testing.T) {
	f := Filter{
		Op: FilterAnd,
		And: []Filter{
			{Op: FilterEq, Field: "tenant", Value: NewString("acme")},
			{Op: FilterOr, Or: []Filter{
				{Op: FilterEq, Field: "status", Value: NewString("active")},
				{Op: FilterGt, Field: "priority", Value: NewInt(5)},
			}},
			{Op: FilterNot, Not: &Filter{Op: FilterContains, Field: "tags", Value: NewString("archived")}},
			{Op: FilterMatch, Field: "title", Value: NewString("quick")},
			{Op: FilterRegex, Field: "sku", Value: NewString(`^ABC`)},
			{Op: FilterIsEmpty, Field: "deleted"},
			{Op: FilterIsNull, Field: "explicit_null"},
			{Op: FilterDtGt, Field: "created", Value: NewString("2024-01-01T00:00:00Z")},
			{Op: FilterDtGte, Field: "created", Value: NewString("2024-01-01T00:00:00Z")},
			{Op: FilterDtLt, Field: "created", Value: NewString("2030-01-01T00:00:00Z")},
			{Op: FilterDtLte, Field: "created", Value: NewString("2030-01-01T00:00:00Z")},
		},
	}
	data, err := json.Marshal(f)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got Filter
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal(%s): %v", data, err)
	}

	// Compile both and check they agree on a sample metadata.
	created, _ := time.Parse(time.RFC3339, "2025-06-01T00:00:00Z")
	m := Metadata{
		"tenant":        NewString("acme"),
		"status":        NewString("active"),
		"priority":      NewInt(7),
		"tags":          NewStrings([]string{"prod"}),
		"title":         NewString("the quick brown fox"),
		"sku":           NewString("ABC-1"),
		"explicit_null": {Kind: ValueNone},
		"created":       NewInt(created.UnixMilli()),
	}
	want := compileOrFail(t, f)(m)
	gotPred := compileOrFail(t, got)(m)
	if want != gotPred {
		t.Errorf("roundtripped filter disagrees: original=%v roundtrip=%v (json=%s)", want, gotPred, data)
	}
	if !gotPred {
		t.Errorf("expected sample metadata to match; json=%s", data)
	}
}

func TestFilterMatch(t *testing.T) {
	m := Metadata{
		"title": NewString("The Quick Brown Fox"),
		"tags":  NewStrings([]string{"Go Programming", "vector search"}),
		"score": NewInt(5),
	}

	cases := []struct {
		name  string
		field string
		query string
		want  bool
	}{
		{"single token hit", "title", "quick", true},
		{"all tokens hit any order", "title", "fox brown", true},
		{"case insensitive", "title", "QUICK", true},
		{"missing token miss", "title", "quick lazy", false},
		{"substring is not a whole-token match", "title", "ox", false},
		{"prefix is not a whole-token match", "title", "quic", false},
		{"empty query matches present string", "title", "", true},
		{"punctuation tokenization", "title", "quick,brown", true},
		{"missing field", "absent", "quick", false},
		{"wrong kind (int)", "score", "5", false},
		{"strings any element hit", "tags", "vector search", true},
		{"strings any element hit second token order", "tags", "programming go", true},
		{"strings no element matches", "tags", "python", false},
		{"strings empty query matches present", "tags", "", true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := compileOrFail(t, Filter{Op: FilterMatch, Field: c.field, Value: NewString(c.query)})
			if got := p(m); got != c.want {
				t.Errorf("match field=%q query=%q = %v, want %v", c.field, c.query, got, c.want)
			}
		})
	}
}

func TestFilterRegex(t *testing.T) {
	m := Metadata{
		"sku":  NewString("ABC-1234"),
		"tags": NewStrings([]string{"alpha", "beta-7"}),
		"n":    NewInt(9),
	}

	cases := []struct {
		name    string
		field   string
		pattern string
		want    bool
	}{
		{"string hit", "sku", `^ABC-\d+$`, true},
		{"string miss", "sku", `^XYZ`, false},
		{"strings any element hit", "tags", `beta-\d`, true},
		{"strings none match", "tags", `^gamma`, false},
		{"missing field", "absent", `.*`, false},
		{"wrong kind int", "n", `9`, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := compileOrFail(t, Filter{Op: FilterRegex, Field: c.field, Value: NewString(c.pattern)})
			if got := p(m); got != c.want {
				t.Errorf("regex field=%q pattern=%q = %v, want %v", c.field, c.pattern, got, c.want)
			}
		})
	}

	// Invalid pattern -> compile error.
	if _, err := (Filter{Op: FilterRegex, Field: "sku", Value: NewString("(")}).Compile(); err == nil {
		t.Error("invalid regex should produce a Compile error")
	}
}

func TestFilterIsEmpty(t *testing.T) {
	m := Metadata{
		"present":     NewString("x"),
		"emptyStr":    NewString(""),
		"none":        {Kind: ValueNone},
		"emptyArr":    NewStrings([]string{}),
		"emptyIntArr": NewInts([]int64{}),
		"emptyFltArr": NewFloats([]float64{}),
		"nonEmptyArr": NewStrings([]string{"a"}),
		"zeroInt":     NewInt(0),
	}

	cases := []struct {
		field string
		want  bool
	}{
		{"absent", true},
		{"present", false},
		{"emptyStr", true},
		{"none", true},
		{"emptyArr", true},
		{"emptyIntArr", true},
		{"emptyFltArr", true},
		{"nonEmptyArr", false},
		{"zeroInt", false}, // a zero int is present and non-empty
	}
	for _, c := range cases {
		t.Run(c.field, func(t *testing.T) {
			p := compileOrFail(t, Filter{Op: FilterIsEmpty, Field: c.field})
			if got := p(m); got != c.want {
				t.Errorf("is_empty field=%q = %v, want %v", c.field, got, c.want)
			}
		})
	}
}

func TestFilterIsNull(t *testing.T) {
	m := Metadata{
		"present":  NewString("x"),
		"emptyStr": NewString(""),
		"none":     {Kind: ValueNone},
		"emptyArr": NewStrings([]string{}),
	}

	cases := []struct {
		field string
		want  bool
	}{
		{"absent", false}, // distinct from is_empty
		{"present", false},
		{"emptyStr", false},
		{"none", true},
		{"emptyArr", false},
	}
	for _, c := range cases {
		t.Run(c.field, func(t *testing.T) {
			p := compileOrFail(t, Filter{Op: FilterIsNull, Field: c.field})
			if got := p(m); got != c.want {
				t.Errorf("is_null field=%q = %v, want %v", c.field, got, c.want)
			}
		})
	}
}

func TestFilterDatetime(t *testing.T) {
	ms := func(s string) int64 {
		t.Helper()
		tm, err := time.Parse(time.RFC3339, s)
		if err != nil {
			t.Fatalf("parse %q: %v", s, err)
		}
		return tm.UnixMilli()
	}
	m := Metadata{
		"created": NewInt(ms("2024-06-15T12:00:00Z")),
		"name":    NewString("not-a-date"),
	}

	cases := []struct {
		name  string
		op    FilterOp
		field string
		lit   string
		want  bool
	}{
		{"dt_gt true", FilterDtGt, "created", "2024-06-15T11:00:00Z", true},
		{"dt_gt false equal", FilterDtGt, "created", "2024-06-15T12:00:00Z", false},
		{"dt_gte true equal", FilterDtGte, "created", "2024-06-15T12:00:00Z", true},
		{"dt_lt true", FilterDtLt, "created", "2024-06-15T13:00:00Z", true},
		{"dt_lt false equal", FilterDtLt, "created", "2024-06-15T12:00:00Z", false},
		{"dt_lte true equal", FilterDtLte, "created", "2024-06-15T12:00:00Z", true},
		{"missing field", FilterDtGt, "absent", "2024-06-15T11:00:00Z", false},
		{"non-numeric field", FilterDtGt, "name", "2024-06-15T11:00:00Z", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := compileOrFail(t, Filter{Op: c.op, Field: c.field, Value: NewString(c.lit)})
			if got := p(m); got != c.want {
				t.Errorf("%s field=%q lit=%q = %v, want %v", c.name, c.field, c.lit, got, c.want)
			}
		})
	}

	// Range via And: 2024-06-15T00:00:00Z <= created < 2024-06-16T00:00:00Z.
	rng := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterDtGte, Field: "created", Value: NewString("2024-06-15T00:00:00Z")},
		{Op: FilterDtLt, Field: "created", Value: NewString("2024-06-16T00:00:00Z")},
	}}
	if !compileOrFail(t, rng)(m) {
		t.Error("datetime range via And should match created in-range")
	}
	rngMiss := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterDtGte, Field: "created", Value: NewString("2024-06-16T00:00:00Z")},
		{Op: FilterDtLt, Field: "created", Value: NewString("2024-06-17T00:00:00Z")},
	}}
	if compileOrFail(t, rngMiss)(m) {
		t.Error("datetime range via And should not match out-of-range")
	}

	// Invalid RFC3339 literal -> compile error.
	for _, op := range []FilterOp{FilterDtGt, FilterDtGte, FilterDtLt, FilterDtLte} {
		if _, err := (Filter{Op: op, Field: "created", Value: NewString("nope")}).Compile(); err == nil {
			t.Errorf("invalid RFC3339 for %s should produce a Compile error", mustOpName(op))
		}
	}
}

func TestFilterDottedPath(t *testing.T) {
	m := Metadata{
		"address.city": NewString("NYC"),
		"plain":        NewString("v"),
	}

	// Exact flat dotted-key hit.
	if v, ok := lookupPath(m, "address.city"); !ok || v.Str != "NYC" {
		t.Errorf("lookupPath(address.city) = %+v,%v; want NYC,true", v, ok)
	}
	// Non-dotted key unchanged.
	if v, ok := lookupPath(m, "plain"); !ok || v.Str != "v" {
		t.Errorf("lookupPath(plain) = %+v,%v; want v,true", v, ok)
	}
	// Missing dotted key -> not found.
	if _, ok := lookupPath(m, "address.zip"); ok {
		t.Error("lookupPath(address.zip) should not be found")
	}
	// Missing plain key -> not found.
	if _, ok := lookupPath(m, "missing"); ok {
		t.Error("lookupPath(missing) should not be found")
	}
	// nil metadata -> not found.
	if _, ok := lookupPath(nil, "anything"); ok {
		t.Error("lookupPath(nil, ...) should not be found")
	}

	// A leaf op routes through lookupPath: eq on a flat dotted key works.
	eq := compileOrFail(t, Filter{Op: FilterEq, Field: "address.city", Value: NewString("NYC")})
	if !eq(m) {
		t.Error("eq on flat dotted key should match via lookupPath")
	}
}

func TestFilterGeoRadius(t *testing.T) {
	// Paris point; center is also Paris, target is London (~343 km away).
	const (
		parisLat, parisLon   = 48.8566, 2.3522
		londonLat, londonLon = 51.5074, -0.1278
	)
	m := Metadata{
		"loc":  NewGeo(londonLat, londonLon),
		"name": NewString("london"),
	}

	// 350km radius around Paris: London (~343km) is inside.
	in := compileOrFail(t, Filter{Op: FilterGeoRadius, Field: "loc", Geo: &GeoCondition{
		CenterLat: parisLat, CenterLon: parisLon, RadiusM: 350_000,
	}})
	if !in(m) {
		t.Error("london should be within 350km of paris")
	}
	// 300km radius: London (~343km) is outside.
	out := compileOrFail(t, Filter{Op: FilterGeoRadius, Field: "loc", Geo: &GeoCondition{
		CenterLat: parisLat, CenterLon: parisLon, RadiusM: 300_000,
	}})
	if out(m) {
		t.Error("london should be outside 300km of paris")
	}
	// Boundary: a tiny radius around the exact point matches itself.
	self := compileOrFail(t, Filter{Op: FilterGeoRadius, Field: "loc", Geo: &GeoCondition{
		CenterLat: londonLat, CenterLon: londonLon, RadiusM: 1,
	}})
	if !self(m) {
		t.Error("point should match a radius centered on itself")
	}
	// Missing field -> false.
	if self(Metadata{"other": NewGeo(londonLat, londonLon)}) {
		t.Error("missing geo field should not match")
	}
	// Non-geo field -> false.
	if self(Metadata{"loc": NewString("not-geo")}) {
		t.Error("non-geo field should not match")
	}
}

func TestFilterGeoBox(t *testing.T) {
	// SW (10,20) -> NE (30,40).
	cond := &GeoCondition{MinLat: 10, MinLon: 20, MaxLat: 30, MaxLon: 40}
	p := compileOrFail(t, Filter{Op: FilterGeoBox, Field: "loc", Geo: cond})

	cases := []struct {
		name     string
		lat, lon float64
		want     bool
	}{
		{"inside", 20, 30, true},
		{"outside north", 31, 30, false},
		{"outside west", 20, 19, false},
		{"on SW corner inclusive", 10, 20, true},
		{"on NE corner inclusive", 30, 40, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := p(Metadata{"loc": NewGeo(c.lat, c.lon)}); got != c.want {
				t.Errorf("box(%v,%v) = %v, want %v", c.lat, c.lon, got, c.want)
			}
		})
	}
	// Missing / non-geo -> false.
	if p(Metadata{"other": NewGeo(20, 30)}) {
		t.Error("missing field should not match box")
	}
	if p(Metadata{"loc": NewInt(5)}) {
		t.Error("non-geo field should not match box")
	}
}

func TestFilterGeoPolygon(t *testing.T) {
	// L-shaped concave polygon (see geo_test.go concavePoly for geometry). The
	// upper-left notch lat∈(4,10],lon∈[0,6) is OUTSIDE — a point there proves
	// even-odd ray-casting (a convex hull would wrongly include it).
	cond := &GeoCondition{Polygon: []float64{
		0, 0,
		0, 10,
		10, 10,
		10, 6,
		4, 6,
		4, 0,
	}}
	p := compileOrFail(t, Filter{Op: FilterGeoPolygon, Field: "loc", Geo: cond})

	cases := []struct {
		name     string
		lat, lon float64
		want     bool
	}{
		{"inside lower band", 2, 5, true},
		{"inside right tower high", 8, 8, true},
		{"in the notch (upper-left) -> outside", 8, 2, false},
		{"far outside", 50, 50, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := p(Metadata{"loc": NewGeo(c.lat, c.lon)}); got != c.want {
				t.Errorf("polygon(%v,%v) = %v, want %v", c.lat, c.lon, got, c.want)
			}
		})
	}
	// Missing / non-geo -> false.
	if p(Metadata{"other": NewGeo(2, 5)}) {
		t.Error("missing field should not match polygon")
	}
	if p(Metadata{"loc": NewString("x")}) {
		t.Error("non-geo field should not match polygon")
	}
}

func TestFilterGeoCompileErrors(t *testing.T) {
	cases := []struct {
		name string
		f    Filter
	}{
		{"radius nil geo", Filter{Op: FilterGeoRadius, Field: "loc"}},
		{"box nil geo", Filter{Op: FilterGeoBox, Field: "loc"}},
		{"polygon nil geo", Filter{Op: FilterGeoPolygon, Field: "loc"}},
		{"radius zero", Filter{Op: FilterGeoRadius, Field: "loc", Geo: &GeoCondition{CenterLat: 1, CenterLon: 1, RadiusM: 0}}},
		{"radius negative", Filter{Op: FilterGeoRadius, Field: "loc", Geo: &GeoCondition{CenterLat: 1, CenterLon: 1, RadiusM: -5}}},
		{"radius NaN", Filter{Op: FilterGeoRadius, Field: "loc", Geo: &GeoCondition{CenterLat: 1, CenterLon: 1, RadiusM: math.NaN()}}},
		{"radius +Inf", Filter{Op: FilterGeoRadius, Field: "loc", Geo: &GeoCondition{CenterLat: 1, CenterLon: 1, RadiusM: math.Inf(1)}}},
		{"radius center lat out of range", Filter{Op: FilterGeoRadius, Field: "loc", Geo: &GeoCondition{CenterLat: 91, CenterLon: 1, RadiusM: 10}}},
		{"radius center lon out of range", Filter{Op: FilterGeoRadius, Field: "loc", Geo: &GeoCondition{CenterLat: 1, CenterLon: 181, RadiusM: 10}}},
		{"box min>max lat", Filter{Op: FilterGeoBox, Field: "loc", Geo: &GeoCondition{MinLat: 30, MinLon: 20, MaxLat: 10, MaxLon: 40}}},
		{"box min>max lon", Filter{Op: FilterGeoBox, Field: "loc", Geo: &GeoCondition{MinLat: 10, MinLon: 40, MaxLat: 30, MaxLon: 20}}},
		{"box corner out of range", Filter{Op: FilterGeoBox, Field: "loc", Geo: &GeoCondition{MinLat: -91, MinLon: 20, MaxLat: 30, MaxLon: 40}}},
		{"polygon odd length", Filter{Op: FilterGeoPolygon, Field: "loc", Geo: &GeoCondition{Polygon: []float64{0, 0, 1, 1, 2}}}},
		{"polygon too few vertices", Filter{Op: FilterGeoPolygon, Field: "loc", Geo: &GeoCondition{Polygon: []float64{0, 0, 1, 1}}}},
		{"polygon vertex out of range", Filter{Op: FilterGeoPolygon, Field: "loc", Geo: &GeoCondition{Polygon: []float64{0, 0, 0, 10, 200, 10}}}},
		{"geo op missing field", Filter{Op: FilterGeoRadius, Geo: &GeoCondition{CenterLat: 1, CenterLon: 1, RadiusM: 10}}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := c.f.Compile(); err == nil {
				t.Errorf("Compile(%s) err = nil, want error", c.name)
			}
		})
	}
}

func TestFilterGeoJSONRoundtrip(t *testing.T) {
	f := Filter{
		Op: FilterOr,
		Or: []Filter{
			{Op: FilterGeoRadius, Field: "loc", Geo: &GeoCondition{CenterLat: 48.8566, CenterLon: 2.3522, RadiusM: 350_000}},
			{Op: FilterGeoBox, Field: "loc", Geo: &GeoCondition{MinLat: 10, MinLon: 20, MaxLat: 30, MaxLon: 40}},
			{Op: FilterGeoPolygon, Field: "loc", Geo: &GeoCondition{Polygon: []float64{0, 0, 0, 10, 10, 10, 10, 6, 4, 6, 4, 0}}},
		},
	}
	data, err := json.Marshal(f)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got Filter
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal(%s): %v", data, err)
	}

	// Sample metadata that matches the radius clause (London near Paris).
	m := Metadata{"loc": NewGeo(51.5074, -0.1278)}
	want := compileOrFail(t, f)(m)
	gotPred := compileOrFail(t, got)(m)
	if want != gotPred {
		t.Errorf("roundtripped geo filter disagrees: original=%v roundtrip=%v (json=%s)", want, gotPred, data)
	}
	if !gotPred {
		t.Errorf("expected sample geo metadata to match; json=%s", data)
	}

	// Verify the op names ride the wire as the spec'd strings.
	var generic struct {
		Or []struct {
			Op  string       `json:"op"`
			Geo GeoCondition `json:"geo"`
		} `json:"or"`
	}
	if err := json.Unmarshal(data, &generic); err != nil {
		t.Fatalf("Unmarshal generic: %v", err)
	}
	wantOps := []string{"geo_radius", "geo_bounding_box", "geo_polygon"}
	for i, w := range wantOps {
		if generic.Or[i].Op != w {
			t.Errorf("clause %d op = %q, want %q", i, generic.Or[i].Op, w)
		}
	}
	if generic.Or[0].Geo.RadiusM != 350_000 {
		t.Errorf("radius_m did not survive JSON: %v", generic.Or[0].Geo.RadiusM)
	}
}

func TestFilterGeoIsZero(t *testing.T) {
	// A geo leaf op is never the "match all" zero filter.
	f := Filter{Op: FilterGeoRadius, Field: "loc", Geo: &GeoCondition{CenterLat: 1, CenterLon: 1, RadiusM: 10}}
	if f.IsZero() {
		t.Error("geo leaf filter should not be IsZero (non-FilterAnd op)")
	}
	// A FilterAnd carrying a Geo condition must NOT be treated as the match-all
	// zero filter (else it would silently compile to nil and ignore the geo clause).
	withGeo := Filter{Op: FilterAnd, Geo: &GeoCondition{CenterLat: 1, CenterLon: 1, RadiusM: 10}}
	if withGeo.IsZero() {
		t.Error("FilterAnd with a Geo condition must not be IsZero")
	}
}

func TestFilterJSONShape(t *testing.T) {
	// Verify the wire shape matches the spec's Pinecone-style dialect.
	f := Filter{Op: FilterEq, Field: "tenant", Value: NewString("acme")}
	data, err := json.Marshal(f)
	if err != nil {
		t.Fatal(err)
	}
	var generic map[string]any
	if err := json.Unmarshal(data, &generic); err != nil {
		t.Fatal(err)
	}
	if generic["op"] != "eq" {
		t.Errorf("op = %v, want eq", generic["op"])
	}
	if generic["field"] != "tenant" {
		t.Errorf("field = %v, want tenant", generic["field"])
	}
	val, ok := generic["value"].(map[string]any)
	if !ok {
		t.Fatalf("value not an object: %v", generic["value"])
	}
	if val["kind"] != "string" || val["str"] != "acme" {
		t.Errorf("value = %v, want kind:string str:acme", val)
	}
}

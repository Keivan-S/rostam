// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// EQUIVALENCE IS THE WHOLE POINT OF THE RAW DECODERS.
//
// DecodeVectorDocsRaw and friends exist so a response whose only destination is
// JSON never decodes the metadata it is about to re-encode. That is only a valid
// substitution if the BYTES A CLIENT RECEIVES are unchanged, so these tests assert
// on the marshalled response bytes — not on decoded structs, which would hide
// exactly the escaping/formatting differences that could bite.
//
// The battery below deliberately includes the shapes where a raw splice could
// plausibly differ from a re-marshal: HTML characters (Go escapes <, > and & by
// default), the line/paragraph separators Go also escapes, every ValueKind, floats
// that switch to exponent notation, values that are THEMSELVES JSON text, keys
// that collide only after case folding, and the values that fail to marshal at all
// (NaN/±Inf) and so leave the wire carrying no metadata.

// rawEquivCase is one named document battery entry.
type rawEquivCase struct {
	name string
	docs []vector.Document
}

// rawEquivCases returns the document shapes both decoders must render identically.
func rawEquivCases() []rawEquivCase {
	htmlMeta := vector.Metadata{
		"tag":    vector.NewString(`<script>alert("x&y")</script>`),
		"cmp":    vector.NewString("a < b && b > c"),
		"amp":    vector.NewString("&amp;"),
		"seps":   vector.NewString("line\u2028para\u2029end"), // Go escapes both by default
		"escape": vector.NewString("quote\" backslash\\ tab\t newline\n"),
	}
	unicodeMeta := vector.Metadata{
		"emoji": vector.NewString("🎉🧪 ünïcödé"),
		"cjk":   vector.NewString("日本語のテキスト"),
		"rtl":   vector.NewString("مرحبا بالعالم"),
		"zero":  vector.NewString("a\u0000b\u200bc"), // NUL and zero-width space
	}
	// Keys that differ only by case / by a trailing space / by a dot — the map
	// ordering encoding/json applies must land the same way through both paths.
	dupishMeta := vector.Metadata{
		"Key": vector.NewInt(1), "key": vector.NewInt(2), "KEY": vector.NewInt(3),
		"key ": vector.NewInt(4), "key.sub": vector.NewInt(5), "key_sub": vector.NewInt(6),
		"": vector.NewInt(7),
	}
	// A payload whose VALUE is itself JSON text: the raw path must keep it escaped
	// as a string, never splice it as structure.
	rawJSONMeta := vector.Metadata{
		"doc":   vector.NewString(`{"nested":{"a":[1,2,3]},"b":null}`),
		"arr":   vector.NewString(`[{"x":1},{"y":2}]`),
		"empty": vector.NewString(`{}`),
	}
	allKinds := vector.Metadata{
		"none":    {},
		"string":  vector.NewString("s"),
		"int":     vector.NewInt(-9007199254740993),
		"float":   vector.NewFloat(1.0 / 3.0),
		"bool":    vector.NewBool(true),
		"strings": vector.NewStrings([]string{"a", "", "<b>"}),
		"ints":    vector.NewInts([]int64{math.MinInt64, 0, math.MaxInt64}),
		"floats":  vector.NewFloats([]float64{0, math.Copysign(0, -1), 1e-7, 1e21, math.SmallestNonzeroFloat64, math.MaxFloat64}),
		"geo":     vector.NewGeo(0, 0),
		"geo2":    vector.NewGeo(-33.8688, 151.2093),
	}
	// Slices that are non-nil but empty: omitempty drops them either way, so the
	// two paths must still agree.
	emptySlices := vector.Metadata{
		"strs": vector.NewStrings([]string{}),
		"ints": vector.NewInts([]int64{}),
		"flts": vector.NewFloats([]float64{}),
	}
	// json.Marshal REFUSES NaN/±Inf, and EncodeVectorDocs drops the metadata it
	// cannot marshal (the error is deliberately ignored there). Both decoders then
	// see a zero-length metadata window, so both must render the doc without it.
	unmarshalable := vector.Metadata{"nan": vector.NewFloat(math.NaN()), "ok": vector.NewString("kept?")}
	// Invalid UTF-8 in a Go string: json.Marshal replaces it with U+FFFD ON THE WAY
	// TO THE WIRE, so both decoders start from already-sanitized bytes. This is what
	// makes the escape-canonicalization in TestDocsRawCanonicalizesReplacementEscape
	// the only thing standing between the two paths here.
	badUTF8 := vector.Metadata{
		"raw":  vector.NewString("\xff\xfe pre\x80post"),
		"strs": vector.NewStrings([]string{"ok", "\xed\xa0\x80"}),
	}

	return []rawEquivCase{
		{"nil", nil},
		{"empty", []vector.Document{}},
		{"k1-no-metadata", []vector.Document{{ID: 1, Distance: 0.5, Score: 0.25, Content: "hello"}}},
		{"k1-nil-metadata-empty-content", []vector.Document{{ID: 0}}},
		{"k1-empty-metadata-map", []vector.Document{{ID: 2, Metadata: vector.Metadata{}}}},
		{"html-escaping", []vector.Document{{ID: 3, Content: `<a href="x">&</a>`, Metadata: htmlMeta}}},
		{"unicode", []vector.Document{{ID: 4, Content: "🎉 日本語  ", Metadata: unicodeMeta}}},
		{"duplicate-ish-keys", []vector.Document{{ID: 5, Metadata: dupishMeta}}},
		{"metadata-holding-raw-json", []vector.Document{{ID: 6, Metadata: rawJSONMeta}}},
		{"all-value-kinds", []vector.Document{{ID: 7, Metadata: allKinds}}},
		{"empty-slices", []vector.Document{{ID: 8, Metadata: emptySlices}}},
		{"unmarshalable-floats", []vector.Document{{ID: 9, Metadata: unmarshalable}}},
		{"invalid-utf8", []vector.Document{{ID: 13, Content: "bad \xff\xfe content", Metadata: badUTF8}}},
		{"float32-extremes", []vector.Document{
			{ID: 10, Distance: math.MaxFloat32, Score: math.SmallestNonzeroFloat32},
			{ID: 11, Distance: -0, Score: 1e-45},
			{ID: 12, Distance: 0.1, Score: 3.4028235e38},
		}},
		{"mixed-k10", mixedDocs(10)},
		{"mixed-k100", mixedDocs(100)},
		{"k1", mixedDocs(1)},
	}
}

// mixedDocs builds k documents cycling through the metadata shapes, so a battery
// entry covers per-document variation (some carrying metadata, some not) inside a
// single response.
func mixedDocs(k int) []vector.Document {
	docs := make([]vector.Document, k)
	for i := range docs {
		docs[i] = vector.Document{
			ID:       uint64(i),
			Distance: float32(i) / 7,
			Score:    float32(i) * 1.5,
			Content:  fmt.Sprintf("chunk %d <b>&</b> 🎉", i),
		}
		switch i % 4 {
		case 0: // no metadata at all
		case 1:
			docs[i].Metadata = vector.Metadata{"i": vector.NewInt(int64(i))}
		case 2:
			docs[i].Metadata = vector.Metadata{
				"title": vector.NewString(fmt.Sprintf("doc <%d> & more", i)),
				"tags":  vector.NewStrings([]string{"a", "b"}),
				"score": vector.NewFloat(float64(i) / 3),
				"live":  vector.NewBool(i%8 == 0),
			}
		case 3:
			docs[i].Metadata = vector.Metadata{"json": vector.NewString(`{"raw":true}`), "at": vector.NewGeo(1.5, -2.5)}
		}
	}
	return docs
}

// renderResponse marshals v the way writeJSON does (json.Encoder, HTML escaping
// on, trailing newline), so the comparison is against the exact bytes an HTTP
// client would read rather than against json.Marshal's slightly different output.
func renderResponse(t *testing.T, v any) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(v); err != nil {
		t.Fatalf("encode response: %v", err)
	}
	return buf.Bytes()
}

// TestDocsRawRendersIdentically is the primary equivalence proof for search_docs:
// for every battery shape, the response bytes built from the raw decoder must
// equal the response bytes built from the typed decoder.
func TestDocsRawRendersIdentically(t *testing.T) {
	for _, tc := range rawEquivCases() {
		t.Run(tc.name, func(t *testing.T) {
			body := EncodeVectorDocs(tc.docs)

			typed, err := DecodeVectorDocs(body)
			if err != nil {
				t.Fatalf("typed decode: %v", err)
			}
			raw, err := DecodeVectorDocsRaw(body)
			if err != nil {
				t.Fatalf("raw decode: %v", err)
			}
			if len(typed) != len(raw) {
				t.Fatalf("doc count: typed %d, raw %d", len(typed), len(raw))
			}
			want := renderResponse(t, map[string]any{"documents": typed})
			got := renderResponse(t, map[string]any{"documents": raw})
			if !bytes.Equal(want, got) {
				t.Fatalf("response bytes differ\n typed: %s\n   raw: %s", want, got)
			}
		})
	}
}

// TestDocsDegradedRawRendersIdentically covers the search_docs response as the
// HTTP layer actually shapes it — documents plus the degraded trailer — across
// the trailer's own variants.
func TestDocsDegradedRawRendersIdentically(t *testing.T) {
	trailers := []struct {
		name     string
		degraded bool
		missing  []uint16
	}{
		{"healthy", false, nil},
		{"degraded-none-named", true, nil},
		{"degraded-one", true, []uint16{3}},
		{"degraded-many", true, []uint16{0, 1, 2, 65535}},
	}
	for _, tc := range rawEquivCases() {
		for _, tr := range trailers {
			t.Run(tc.name+"/"+tr.name, func(t *testing.T) {
				body := EncodeVectorDocsDegraded(tc.docs, tr.degraded, tr.missing)

				typed, tDeg, tMiss, err := DecodeVectorDocsDegraded(body)
				if err != nil {
					t.Fatalf("typed decode: %v", err)
				}
				raw, rDeg, rMiss, err := DecodeVectorDocsDegradedRaw(body)
				if err != nil {
					t.Fatalf("raw decode: %v", err)
				}
				if tDeg != rDeg || fmt.Sprint(tMiss) != fmt.Sprint(rMiss) {
					t.Fatalf("trailer: typed (%v,%v) raw (%v,%v)", tDeg, tMiss, rDeg, rMiss)
				}
				want := renderResponse(t, map[string]any{"documents": typed, "degraded": tDeg, "missing": tMiss})
				got := renderResponse(t, map[string]any{"documents": raw, "degraded": rDeg, "missing": rMiss})
				if !bytes.Equal(want, got) {
					t.Fatalf("response bytes differ\n typed: %s\n   raw: %s", want, got)
				}
			})
		}
	}
}

// TestScrollRawRendersIdentically covers the scroll response (documents +
// trailer + next_cursor), including the legacy bodies DecodeScrollResult is
// documented to tolerate.
func TestScrollRawRendersIdentically(t *testing.T) {
	cursors := []string{"", "eyJhZnRlcl9pZCI6MTB9", strings.Repeat("c", 512)}
	for _, tc := range rawEquivCases() {
		for ci, cur := range cursors {
			t.Run(fmt.Sprintf("%s/cursor%d", tc.name, ci), func(t *testing.T) {
				for _, body := range [][]byte{
					EncodeScrollResult(tc.docs, false, nil, cur),
					EncodeScrollResult(tc.docs, true, []uint16{7}, cur),
					EncodeVectorDocs(tc.docs),                               // legacy: no trailer, no cursor
					EncodeVectorDocsDegraded(tc.docs, true, []uint16{1, 2}), // legacy: trailer, no cursor
				} {
					typed, tDeg, tMiss, tCur, err := DecodeScrollResult(body)
					if err != nil {
						t.Fatalf("typed decode: %v", err)
					}
					raw, rDeg, rMiss, rCur, err := DecodeScrollResultRaw(body)
					if err != nil {
						t.Fatalf("raw decode: %v", err)
					}
					if tDeg != rDeg || fmt.Sprint(tMiss) != fmt.Sprint(rMiss) || tCur != rCur {
						t.Fatalf("tail: typed (%v,%v,%q) raw (%v,%v,%q)", tDeg, tMiss, tCur, rDeg, rMiss, rCur)
					}
					want := renderResponse(t, map[string]any{"documents": typed, "next_cursor": tCur})
					got := renderResponse(t, map[string]any{"documents": raw, "next_cursor": rCur})
					if !bytes.Equal(want, got) {
						t.Fatalf("response bytes differ\n typed: %s\n   raw: %s", want, got)
					}
				}
			})
		}
	}
}

// TestGroupsRawRendersIdentically is the grouped counterpart: group keys carry
// every ValueKind (including the ones whose JSON needs escaping) and each group's
// hits come from the document battery.
func TestGroupsRawRendersIdentically(t *testing.T) {
	keys := []vector.Value{
		{},
		vector.NewString("plain"),
		vector.NewString(`<b>&"quoted"</b>`),
		vector.NewString("🎉 日本語"),
		vector.NewInt(-1),
		vector.NewFloat(1e21),
		vector.NewBool(false),
		vector.NewStrings([]string{"a", "<b>"}),
		vector.NewInts([]int64{1, 2}),
		vector.NewFloats([]float64{0.5}),
		vector.NewGeo(12.5, -0.5),
	}
	cases := rawEquivCases()
	groups := make([]vector.Group, 0, len(keys))
	for i, k := range keys {
		groups = append(groups, vector.Group{Key: k, Hits: cases[i%len(cases)].docs})
	}

	for _, gs := range [][]vector.Group{nil, {}, groups[:1], groups} {
		for _, deg := range []bool{false, true} {
			body := EncodeGroupsDegraded(gs, deg, []uint16{4})

			typed, tDeg, tMiss, err := DecodeGroupsDegraded(body)
			if err != nil {
				t.Fatalf("typed decode: %v", err)
			}
			raw, rDeg, rMiss, err := DecodeGroupsDegradedRaw(body)
			if err != nil {
				t.Fatalf("raw decode: %v", err)
			}
			if tDeg != rDeg || fmt.Sprint(tMiss) != fmt.Sprint(rMiss) {
				t.Fatalf("trailer: typed (%v,%v) raw (%v,%v)", tDeg, tMiss, rDeg, rMiss)
			}
			want := renderResponse(t, map[string]any{"groups": typed, "degraded": tDeg, "missing": tMiss})
			got := renderResponse(t, map[string]any{"groups": raw, "degraded": rDeg, "missing": rMiss})
			if !bytes.Equal(want, got) {
				t.Fatalf("group response bytes differ (n=%d degraded=%v)\n typed: %s\n   raw: %s", len(gs), deg, want, got)
			}
		}
	}
}

// TestVectorDocsWireUnchanged pins the ENCODER. The raw decoders are a read-side
// change only: the bytes a shard puts on the result wire — the bytes a remote peer
// receives, and the bytes Raft-adjacent paths carry — must be exactly what they
// were. A golden hex fixture catches any drift.
func TestVectorDocsWireUnchanged(t *testing.T) {
	docs := []vector.Document{
		{ID: 1, Distance: 0.5, Score: 0.25, Content: "hi <b>", Metadata: vector.Metadata{"a": vector.NewInt(7)}},
		{ID: 2},
	}
	const want = "00000002" + // count
		"0000000000000001" + "3f000000" + "3e800000" + "00000006" + "6869203c623e" +
		"0000001c" + "7b2261223a7b226b696e64223a22696e74222c22696e74223a377d7d" +
		"0000000000000002" + "00000000" + "00000000" + "00000000" + "00000000"
	if got := fmt.Sprintf("%x", EncodeVectorDocs(docs)); got != want {
		t.Fatalf("EncodeVectorDocs wire changed:\n got %s\nwant %s", got, want)
	}
}

// TestDocsRawRejectsMalformedMetadata pins the raw decoder's guard: a metadata
// window that is not well-formed JSON must be REFUSED, never spliced into a
// response. The typed decoder refuses the same body (that is the behaviour the
// raw path has to preserve); only the message differs.
func TestDocsRawRejectsMalformedMetadata(t *testing.T) {
	for _, meta := range []string{`{"a":`, `not json`, `{"a":1,}`, "\x00", `{"a":1}{"b":2}`} {
		body := encodeDocWithRawMetadata(1, []byte(meta))
		if _, err := DecodeVectorDocs(body); err == nil {
			t.Fatalf("typed decode accepted malformed metadata %q", meta)
		}
		if _, err := DecodeVectorDocsRaw(body); err == nil {
			t.Fatalf("raw decode accepted malformed metadata %q", meta)
		}
	}
}

// TestDocsRawTruncationMatchesTyped pins the FRAMING: both decoders share one
// walker, so every truncated prefix of a real body must be rejected by both (or
// accepted by both) — a raw decoder that read one byte further than the typed one
// would be a parsing divergence, not just a rendering one.
func TestDocsRawTruncationMatchesTyped(t *testing.T) {
	body := EncodeVectorDocs(mixedDocs(6))
	for n := 0; n <= len(body); n++ {
		_, terr := DecodeVectorDocs(body[:n])
		_, rerr := DecodeVectorDocsRaw(body[:n])
		if (terr == nil) != (rerr == nil) {
			t.Fatalf("prefix len %d: typed err=%v, raw err=%v", n, terr, rerr)
		}
	}
}

// encodeDocWithRawMetadata builds a one-document body carrying meta VERBATIM as
// the metadata window, so tests can plant bytes EncodeVectorDocs would never
// produce (a corrupt or hostile peer's result).
func encodeDocWithRawMetadata(id uint64, meta []byte) []byte {
	buf := make([]byte, 0, 4+8+4+4+4+4+len(meta))
	buf = binary.BigEndian.AppendUint32(buf, 1)
	buf = binary.BigEndian.AppendUint64(buf, id)
	buf = binary.BigEndian.AppendUint32(buf, 0) // distance
	buf = binary.BigEndian.AppendUint32(buf, 0) // score
	buf = binary.BigEndian.AppendUint32(buf, 0) // content len
	buf = binary.BigEndian.AppendUint32(buf, uint32(len(meta)))
	return append(buf, meta...)
}

// FuzzDocsRawRendersIdentically fuzzes the substitution over the inputs that can
// actually reach it: documents, encoded by the one encoder that writes this wire.
// It fuzzes the STRINGS in particular — content, a metadata key, a metadata value
// — because escaping is where a verbatim splice and a re-marshal can part ways,
// and arbitrary bytes there cover the invalid-UTF-8 and control-character cases
// the fixed battery only samples.
func FuzzDocsRawRendersIdentically(f *testing.F) {
	f.Add("plain content", "key", "value", int64(1), 0.5, true)
	f.Add("<b>&</b>", "<k>", `{"raw":true}`, int64(-1), 1e21, false)
	f.Add("\xff\xfe", "\xed\xa0\x80", "\xc0", int64(0), math.Inf(1), true)
	f.Add("line\u2028para", "\u0000", "\U0001F389", int64(1<<62), math.Copysign(0, -1), false)

	f.Fuzz(func(t *testing.T, content, key, val string, i int64, flt float64, b bool) {
		docs := []vector.Document{
			{ID: 1, Distance: 0.5, Score: 1, Content: content, Metadata: vector.Metadata{
				key:      vector.NewString(val),
				"i":      vector.NewInt(i),
				"f":      vector.NewFloat(flt),
				"b":      vector.NewBool(b),
				"strs":   vector.NewStrings([]string{val, key}),
				key + "": vector.NewGeo(flt, -flt),
			}},
			{ID: 2, Content: content},
		}
		body := EncodeVectorDocs(docs)

		typed, err := DecodeVectorDocs(body)
		if err != nil {
			t.Fatalf("typed decode: %v", err)
		}
		raw, err := DecodeVectorDocsRaw(body)
		if err != nil {
			t.Fatalf("raw decode: %v", err)
		}
		var wbuf, gbuf bytes.Buffer
		if err := json.NewEncoder(&wbuf).Encode(map[string]any{"documents": typed}); err != nil {
			return // a value the typed path cannot marshal is not a raw-path concern
		}
		if err := json.NewEncoder(&gbuf).Encode(map[string]any{"documents": raw}); err != nil {
			t.Fatalf("raw render failed where typed succeeded: %v", err)
		}
		if !bytes.Equal(wbuf.Bytes(), gbuf.Bytes()) {
			t.Fatalf("render differs\n typed: %s\n   raw: %s", wbuf.Bytes(), gbuf.Bytes())
		}
	})
}

// FuzzDocsRawAcceptsWhateverTypedDoes fuzzes RAW BYTES — bodies no encoder would
// produce — for the one property that must hold there: the raw decoder never
// refuses a body the typed decoder accepts. That is what lets a caller swap in the
// raw decoder without turning some previously-served response into a 500.
//
// It deliberately does NOT compare renderings. On a body outside the encoder's
// range the two legitimately differ: json.Unmarshal into a struct SILENTLY DROPS
// unknown fields and defaults missing ones, so `{"a":{"zzz":1}}` decodes to a
// zero Value and re-marshals as {"kind":"none"}, while a splice preserves it
// verbatim. See TestDocsRawKnownShapeDivergence.
func FuzzDocsRawAcceptsWhateverTypedDoes(f *testing.F) {
	for _, tc := range rawEquivCases() {
		f.Add(EncodeVectorDocs(tc.docs))
		f.Add(EncodeVectorDocsDegraded(tc.docs, true, []uint16{1}))
	}
	f.Add(encodeDocWithRawMetadata(1, []byte(`{"a":{"kind":"string","str":"<&>"}}`)))
	f.Add([]byte{})
	f.Add([]byte{0, 0, 0, 1})

	f.Fuzz(func(t *testing.T, body []byte) {
		typed, terr := DecodeVectorDocs(body)
		raw, rerr := DecodeVectorDocsRaw(body)
		if terr == nil && rerr != nil {
			t.Fatalf("typed accepted but raw rejected: %v", rerr)
		}
		if rerr != nil {
			return
		}
		if terr != nil {
			return
		}
		if len(typed) != len(raw) {
			t.Fatalf("doc count: typed %d, raw %d", len(typed), len(raw))
		}
		// Whatever the raw decoder accepts, it must render as WELL-FORMED JSON
		// wherever the typed decoder's result does — a splice must never be able to
		// corrupt a response body. (A body carrying a NaN distance fails to marshal
		// on BOTH paths; that is the wire's problem, not the splice's.)
		if _, err := json.Marshal(map[string]any{"documents": typed}); err != nil {
			return
		}
		out, err := json.Marshal(map[string]any{"documents": raw})
		if err != nil {
			t.Fatalf("raw render failed where typed succeeded: %v", err)
		}
		if !json.Valid(out) {
			t.Fatalf("raw render produced malformed JSON: %s", out)
		}
	})
}

// TestDocsRawCanonicalizesReplacementEscape pins the ONE sequence that does not
// survive an unmarshal+marshal unchanged: encoding/json writes invalid UTF-8 as
// the six-character � escape but writes a valid U+FFFD as its three raw
// bytes, so a verbatim splice of such a window would differ from the typed path's
// re-marshal. checkRawMetadataJSON round-trips those windows instead of splicing
// them; this asserts that actually happens, at the byte level, so the mechanism
// cannot be optimized away by someone who has not hit the case.
func TestDocsRawCanonicalizesReplacementEscape(t *testing.T) {
	escape := []byte(`\ufffd`) // the six-character escape, not the rune
	docs := []vector.Document{{ID: 1, Metadata: vector.Metadata{"bad": vector.NewString("\xff")}}}
	body := EncodeVectorDocs(docs)
	if !bytes.Contains(body, escape) {
		t.Fatalf("fixture no longer exercises the escape; wire = %q", body)
	}

	raw, err := DecodeVectorDocsRaw(body)
	if err != nil {
		t.Fatalf("raw decode: %v", err)
	}
	if bytes.Contains(raw[0].Metadata, escape) {
		t.Fatalf("raw metadata was spliced verbatim, not canonicalized: %s", raw[0].Metadata)
	}
	typed, err := DecodeVectorDocs(body)
	if err != nil {
		t.Fatalf("typed decode: %v", err)
	}
	want := renderResponse(t, map[string]any{"documents": typed})
	got := renderResponse(t, map[string]any{"documents": raw})
	if !bytes.Equal(want, got) {
		t.Fatalf("response bytes differ\n typed: %s\n   raw: %s", want, got)
	}
}

// TestDocsRawKnownShapeDivergence pins the deliberate gap between the two
// decoders on metadata windows OUTSIDE the encoder's range — bodies only a
// corrupt or hostile peer can send, since EncodeVectorDocs writes json.Marshal of
// a vector.Metadata and nothing else. Pinned so the gap stays a KNOWN one rather
// than a discovered one. Two flavours:
//
//   - REJECTED BY TYPED, ACCEPTED BY RAW: json.Valid is a syntax check, so JSON
//     that is well-formed but not an object-of-objects passes it and fails the
//     unmarshal.
//   - ACCEPTED BY BOTH, RENDERED DIFFERENTLY: json.Unmarshal into a struct drops
//     unknown fields and defaults missing ones, so an object-of-objects with
//     foreign keys decodes to a zero Value and re-marshals as {"kind":"none"},
//     while a splice preserves it. This is why the byte-level fuzz target asserts
//     acceptance and well-formedness rather than identical rendering.
func TestDocsRawKnownShapeDivergence(t *testing.T) {
	t.Run("typed-rejects-raw-accepts", func(t *testing.T) {
		for _, meta := range []string{`[1,2]`, `"a string"`, `42`, `{"a":1}`} {
			body := encodeDocWithRawMetadata(1, []byte(meta))
			if _, err := DecodeVectorDocs(body); err == nil {
				t.Fatalf("typed decode unexpectedly ACCEPTED %q — the divergence note is stale", meta)
			}
			if _, err := DecodeVectorDocsRaw(body); err != nil {
				t.Fatalf("raw decode rejected %q: %v — the divergence note is stale", meta, err)
			}
		}
	})
	t.Run("both-accept-render-differs", func(t *testing.T) {
		body := encodeDocWithRawMetadata(1, []byte(`{"a":{"zzz":1}}`))
		typed, err := DecodeVectorDocs(body)
		if err != nil {
			t.Fatalf("typed decode: %v — the divergence note is stale", err)
		}
		raw, err := DecodeVectorDocsRaw(body)
		if err != nil {
			t.Fatalf("raw decode: %v — the divergence note is stale", err)
		}
		if bytes.Equal(renderResponse(t, typed), renderResponse(t, raw)) {
			t.Fatal("renders now agree on an out-of-range window — the divergence note is stale")
		}
	})
}

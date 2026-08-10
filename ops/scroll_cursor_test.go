// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"encoding/base64"
	"encoding/binary"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

func TestScrollCursorRoundTrip(t *testing.T) {
	cases := []uint64{0, 1, 2, 42, 255, 256, 1 << 32, 1<<64 - 1}
	for _, want := range cases {
		tok := EncodeScrollCursor(want)
		if tok == "" {
			t.Fatalf("EncodeScrollCursor(%d) returned empty token", want)
		}
		// URL-safe, no padding.
		if strings.ContainsAny(tok, "+/=") {
			t.Fatalf("EncodeScrollCursor(%d) = %q is not URL-safe / no-pad", want, tok)
		}
		got, present, err := DecodeScrollCursor(tok)
		if err != nil {
			t.Fatalf("DecodeScrollCursor(%q): %v", tok, err)
		}
		if !present {
			t.Fatalf("DecodeScrollCursor(%q): present=false, want true (a non-empty token always has a bound)", tok)
		}
		if got != want {
			t.Fatalf("DecodeScrollCursor(%q) = %d, want %d", tok, got, want)
		}
	}
}

func TestScrollCursorZeroIsPresent(t *testing.T) {
	// lastID=0 must round-trip as present=true: "after id 0" is distinct from
	// "first page" (empty token).
	tok := EncodeScrollCursor(0)
	got, present, err := DecodeScrollCursor(tok)
	if err != nil {
		t.Fatalf("DecodeScrollCursor: %v", err)
	}
	if !present || got != 0 {
		t.Fatalf("DecodeScrollCursor(encode(0)) = (%d, %v), want (0, true)", got, present)
	}
}

func TestScrollCursorEmptyIsFirstPage(t *testing.T) {
	got, present, err := DecodeScrollCursor("")
	if err != nil {
		t.Fatalf("DecodeScrollCursor(\"\"): %v", err)
	}
	if present {
		t.Fatalf("empty token present=true, want false (first page, no lower bound)")
	}
	if got != 0 {
		t.Fatalf("empty token lastID=%d, want 0", got)
	}
}

func TestScrollCursorMalformed(t *testing.T) {
	// 8-byte payload (one short).
	eightBytes := base64.RawURLEncoding.EncodeToString([]byte{1, 0, 0, 0, 0, 0, 0, 0})
	// 10-byte payload (one long).
	tenBytes := base64.RawURLEncoding.EncodeToString([]byte{1, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	// Correct length, wrong version (2).
	badVer := base64.RawURLEncoding.EncodeToString([]byte{2, 0, 0, 0, 0, 0, 0, 0, 0})

	cases := map[string]string{
		"bad base64":    "!!!not base64!!!",
		"eight bytes":   eightBytes,
		"ten bytes":     tenBytes,
		"version 2":     badVer,
		"single byte":   base64.RawURLEncoding.EncodeToString([]byte{1}),
		"empty decoded": base64.RawURLEncoding.EncodeToString(nil) + "x", // garbage
	}
	for name, tok := range cases {
		t.Run(name, func(t *testing.T) {
			// Must not panic.
			got, present, err := DecodeScrollCursor(tok)
			if err == nil {
				t.Fatalf("DecodeScrollCursor(%q) = (%d, %v, nil), want ErrBadScrollCursor", tok, got, present)
			}
			if !errors.Is(err, ErrBadScrollCursor) {
				t.Fatalf("DecodeScrollCursor(%q) err = %v, want ErrBadScrollCursor", tok, err)
			}
		})
	}
}

func sampleScrollDocs() []vector.Document {
	return []vector.Document{
		{ID: 1, Distance: 0.5, Score: 0.9, Content: "alpha", Metadata: vector.Metadata{"k": vector.NewString("v")}},
		{ID: 2, Distance: 0.7, Content: "beta"},
		{ID: 9, Content: "gamma"},
	}
}

func eqDocs(t *testing.T, got, want []vector.Document) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("docs len = %d, want %d (%+v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i].ID != want[i].ID || got[i].Content != want[i].Content {
			t.Fatalf("docs[%d] = {%d,%q}, want {%d,%q}", i, got[i].ID, got[i].Content, want[i].ID, want[i].Content)
		}
	}
}

// TestScrollResultRoundTrip exercises every (degraded, missing, cursor) combo.
func TestScrollResultRoundTrip(t *testing.T) {
	docs := sampleScrollDocs()
	cur := EncodeScrollCursor(9)
	cases := []struct {
		name     string
		degraded bool
		missing  []uint16
		cursor   string
	}{
		{"plain+cursor", false, nil, cur},
		{"plain+no-cursor", false, nil, ""},
		{"degraded+missing+cursor", true, []uint16{1, 3}, cur},
		{"degraded+missing+no-cursor", true, []uint16{2}, ""},
		{"degraded-no-missing+cursor", true, nil, cur},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			body := EncodeScrollResult(docs, c.degraded, c.missing, c.cursor)
			got, degraded, missing, next, err := DecodeScrollResult(body)
			if err != nil {
				t.Fatalf("DecodeScrollResult: %v", err)
			}
			eqDocs(t, got, docs)
			if degraded != c.degraded {
				t.Errorf("degraded = %v, want %v", degraded, c.degraded)
			}
			if len(missing) != len(c.missing) {
				t.Errorf("missing = %v, want %v", missing, c.missing)
			}
			for i := range c.missing {
				if missing[i] != c.missing[i] {
					t.Errorf("missing[%d] = %d, want %d", i, missing[i], c.missing[i])
				}
			}
			if next != c.cursor {
				t.Errorf("next_cursor = %q, want %q", next, c.cursor)
			}
		})
	}
}

// TestScrollResultNamedShape covers the named path's call shape (no degraded/missing).
func TestScrollResultNamedShape(t *testing.T) {
	docs := sampleScrollDocs()
	cur := EncodeScrollCursor(9)
	body := EncodeScrollResult(docs, false, nil, cur)
	got, degraded, missing, next, err := DecodeScrollResult(body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	eqDocs(t, got, docs)
	if degraded || missing != nil || next != cur {
		t.Fatalf("named result = (deg=%v, miss=%v, next=%q), want (false, nil, %q)", degraded, missing, next, cur)
	}
}

// TestScrollResultOldDecoderIgnoresCursor is the CRITICAL backward-compat proof:
// an OLD DecodeVectorDocsDegraded reader parses docs + degraded + missing from an
// EncodeScrollResult body and STOPS at the missing trailer, ignoring the trailing
// cursor (it never errors on the extra bytes). Asserted with + without degraded.
func TestScrollResultOldDecoderIgnoresCursor(t *testing.T) {
	docs := sampleScrollDocs()
	cur := EncodeScrollCursor(123456789)

	// (a) non-degraded result with a cursor: the degraded trailer is emitted
	// UNCONDITIONALLY, so the old decoder reads degraded=false, missing=nil and
	// stops BEFORE the cursor tail.
	body := EncodeScrollResult(docs, false, nil, cur)
	gotDocs, degraded, missing, err := DecodeVectorDocsDegraded(body)
	if err != nil {
		t.Fatalf("old DecodeVectorDocsDegraded(non-degraded scroll result): %v", err)
	}
	eqDocs(t, gotDocs, docs)
	if degraded {
		t.Error("old decoder: degraded=true, want false (cursor bytes leaked into trailer)")
	}
	if missing != nil {
		t.Errorf("old decoder: missing=%v, want nil (cursor bytes leaked into trailer)", missing)
	}
	// And DecodeVectorDocs (docs-only) trivially ignores everything past the docs.
	plainDocs, err := DecodeVectorDocs(body)
	if err != nil {
		t.Fatalf("old DecodeVectorDocs: %v", err)
	}
	eqDocs(t, plainDocs, docs)

	// (b) degraded result with a cursor: old decoder reads the real trailer and
	// ignores the cursor that follows it.
	body = EncodeScrollResult(docs, true, []uint16{5, 7}, cur)
	gotDocs, degraded, missing, err = DecodeVectorDocsDegraded(body)
	if err != nil {
		t.Fatalf("old DecodeVectorDocsDegraded(degraded scroll result): %v", err)
	}
	eqDocs(t, gotDocs, docs)
	if !degraded {
		t.Error("old decoder: degraded=false, want true")
	}
	if len(missing) != 2 || missing[0] != 5 || missing[1] != 7 {
		t.Errorf("old decoder: missing=%v, want [5 7]", missing)
	}
}

// TestScrollResultNewDecoderToleratesOldBody is the forward-compat proof: the new
// DecodeScrollResult on an OLD body (no cursor tail) returns nextCursor="".
// Covers both the plain EncodeVectorDocs body and the EncodeVectorDocsDegraded body.
func TestScrollResultNewDecoderToleratesOldBody(t *testing.T) {
	docs := sampleScrollDocs()

	// Old plain body (EncodeVectorDocs only — no trailer at all).
	plain := EncodeVectorDocs(docs)
	gotDocs, degraded, missing, next, err := DecodeScrollResult(plain)
	if err != nil {
		t.Fatalf("DecodeScrollResult(plain old body): %v", err)
	}
	eqDocs(t, gotDocs, docs)
	if degraded || missing != nil || next != "" {
		t.Fatalf("plain old body = (deg=%v, miss=%v, next=%q), want (false, nil, \"\")", degraded, missing, next)
	}

	// Old degraded body (EncodeVectorDocsDegraded with a real trailer, no cursor).
	deg := EncodeVectorDocsDegraded(docs, true, []uint16{4})
	gotDocs, degraded, missing, next, err = DecodeScrollResult(deg)
	if err != nil {
		t.Fatalf("DecodeScrollResult(degraded old body): %v", err)
	}
	eqDocs(t, gotDocs, docs)
	if !degraded || len(missing) != 1 || missing[0] != 4 || next != "" {
		t.Fatalf("degraded old body = (deg=%v, miss=%v, next=%q), want (true, [4], \"\")", degraded, missing, next)
	}
}

// TestScrollResultByteLayout pins the exact wire layout so a regression in the
// trailer/cursor ordering is caught: [docs][deg:u8][missCount:u16][cursorLen:u32][cursor].
func TestScrollResultByteLayout(t *testing.T) {
	docs := sampleScrollDocs()
	cur := EncodeScrollCursor(42)
	body := EncodeScrollResult(docs, false, nil, cur)
	docsBody := EncodeVectorDocs(docs)

	// docs prefix is byte-identical to EncodeVectorDocs.
	if !equalBytes(body[:len(docsBody)], docsBody) {
		t.Fatal("docs prefix differs from EncodeVectorDocs")
	}
	off := len(docsBody)
	// [deg:u8] == 0, [missCount:u16] == 0.
	if body[off] != 0 {
		t.Errorf("degraded byte = %d, want 0", body[off])
	}
	off++
	if mc := binary.BigEndian.Uint16(body[off:]); mc != 0 {
		t.Errorf("missingCount = %d, want 0", mc)
	}
	off += 2
	// [cursorLen:u32] == len(cur).
	if cl := binary.BigEndian.Uint32(body[off:]); int(cl) != len(cur) {
		t.Errorf("cursorLen = %d, want %d", cl, len(cur))
	}
	off += 4
	if string(body[off:]) != cur {
		t.Errorf("cursor bytes = %q, want %q", string(body[off:]), cur)
	}
}

func equalBytes(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ---------------------------------------------------------------------------
// Cursor v2 (ordered (value, id) resume cursor).
// ---------------------------------------------------------------------------

func TestScrollCursorOrderRoundTrip(t *testing.T) {
	values := []float64{0, math.Copysign(0, -1), 1, -1, 3.14159, -2.71828, 1e308, -1e308, 1<<53 + 1}
	ids := []uint64{0, 1, 255, 1 << 32, 1<<64 - 1}
	keyHashes := []uint16{0, 1, 0xABCD, 0xFFFF}
	for _, v := range values {
		for _, id := range ids {
			for _, kh := range keyHashes {
				for _, desc := range []bool{false, true} {
					tok := EncodeScrollCursorOrder(v, id, desc, kh)
					if strings.ContainsAny(tok, "+/=") {
						t.Fatalf("v2 token %q not URL-safe/no-pad", tok)
					}
					got, err := DecodeScrollCursorTyped(tok)
					if err != nil {
						t.Fatalf("decode v2: %v", err)
					}
					if !got.Present || got.Version != 2 {
						t.Fatalf("v2 decode = present %v version %d, want true 2", got.Present, got.Version)
					}
					if got.Value != v || got.LastID != id || got.Desc != desc || got.KeyHash != kh {
						t.Fatalf("v2 round-trip = {v=%v id=%d desc=%v kh=%d}, want {v=%v id=%d desc=%v kh=%d}",
							got.Value, got.LastID, got.Desc, got.KeyHash, v, id, desc, kh)
					}
				}
			}
		}
	}
}

// TestScrollCursorTypedV1 proves the v2-aware decoder still reads v1 id-only
// tokens and that v1 encoding is byte-identical to before.
func TestScrollCursorTypedV1(t *testing.T) {
	for _, id := range []uint64{0, 7, 1<<64 - 1} {
		tok := EncodeScrollCursor(id)
		got, err := DecodeScrollCursorTyped(tok)
		if err != nil {
			t.Fatalf("typed decode v1: %v", err)
		}
		if !got.Present || got.Version != 1 || got.LastID != id {
			t.Fatalf("v1 typed = {present=%v ver=%d id=%d}, want {true 1 %d}", got.Present, got.Version, got.LastID, id)
		}
		// v2 fields are zero for a v1 cursor.
		if got.Value != 0 || got.Desc || got.KeyHash != 0 {
			t.Fatalf("v1 typed leaked v2 fields: %+v", got)
		}
	}
	// Empty token ⇒ first page.
	got, err := DecodeScrollCursorTyped("")
	if err != nil || got.Present {
		t.Fatalf("empty typed = (%+v, %v), want present=false nil", got, err)
	}
}

// TestScrollCursorV1ByteIdentical pins that adding v2 did not change the v1 wire.
func TestScrollCursorV1ByteIdentical(t *testing.T) {
	tok := EncodeScrollCursor(0x0102030405060708)
	raw, err := base64.RawURLEncoding.DecodeString(tok)
	if err != nil {
		t.Fatalf("decode b64: %v", err)
	}
	want := []byte{1, 1, 2, 3, 4, 5, 6, 7, 8}
	if !equalBytes(raw, want) {
		t.Fatalf("v1 wire = %v, want %v", raw, want)
	}
	// Legacy 3-tuple decoder still works and still rejects a v2 token loudly.
	id, present, err := DecodeScrollCursor(tok)
	if err != nil || !present || id != 0x0102030405060708 {
		t.Fatalf("legacy decode v1 = (%d, %v, %v)", id, present, err)
	}
	v2 := EncodeScrollCursorOrder(1.5, 9, true, 7)
	if _, _, err := DecodeScrollCursor(v2); !errors.Is(err, ErrBadScrollCursor) {
		t.Fatalf("legacy decoder accepted v2 token, want ErrBadScrollCursor, got %v", err)
	}
}

// TestScrollCursorV2ByteLayout pins the exact v2 wire layout.
func TestScrollCursorV2ByteLayout(t *testing.T) {
	tok := EncodeScrollCursorOrder(2.5, 0x1122334455667788, true, 0xBEEF)
	raw, err := base64.RawURLEncoding.DecodeString(tok)
	if err != nil {
		t.Fatalf("decode b64: %v", err)
	}
	if len(raw) != 20 {
		t.Fatalf("v2 len = %d, want 20", len(raw))
	}
	if raw[0] != 2 {
		t.Errorf("version byte = %d, want 2", raw[0])
	}
	if raw[1]&1 == 0 {
		t.Errorf("desc flag bit0 not set: flags=%08b", raw[1])
	}
	if kh := binary.BigEndian.Uint16(raw[2:]); kh != 0xBEEF {
		t.Errorf("keyHash = %#x, want 0xBEEF", kh)
	}
	if v := binary.BigEndian.Uint64(raw[4:]); v != 0x4004000000000000 { // float64 bits of 2.5
		t.Errorf("value bits = %#x, want bits(2.5)", v)
	}
	if id := binary.BigEndian.Uint64(raw[12:]); id != 0x1122334455667788 {
		t.Errorf("tiebreakID = %#x, want 0x1122334455667788", id)
	}
}

func TestScrollCursorTypedMalformed(t *testing.T) {
	cases := map[string]string{
		"bad base64":      "!!!",
		"v1 wrong len":    base64.RawURLEncoding.EncodeToString([]byte{1, 0, 0}),
		"v2 short":        base64.RawURLEncoding.EncodeToString(make([]byte, 19)[:19]),
		"v2 long":         base64.RawURLEncoding.EncodeToString(make([]byte, 21)),
		"unknown version": base64.RawURLEncoding.EncodeToString(append([]byte{9}, make([]byte, 19)...)),
		"empty decoded":   base64.RawURLEncoding.EncodeToString(nil) + "x",
	}
	// Make the v2-short case actually start with version byte 2.
	short := make([]byte, 19)
	short[0] = 2
	cases["v2 short"] = base64.RawURLEncoding.EncodeToString(short)
	long := make([]byte, 21)
	long[0] = 2
	cases["v2 long"] = base64.RawURLEncoding.EncodeToString(long)

	for name, tok := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := DecodeScrollCursorTyped(tok)
			if !errors.Is(err, ErrBadScrollCursor) {
				t.Fatalf("DecodeScrollCursorTyped(%q) err = %v, want ErrBadScrollCursor", tok, err)
			}
		})
	}
}

func TestValidateOrderCursor(t *testing.T) {
	const kh = uint16(0x1234)
	v2 := EncodeScrollCursorOrder(10, 5, false, kh)
	dec, err := DecodeScrollCursorTyped(v2)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	// Matching desc + keyHash ⇒ ok.
	if err := ValidateOrderCursor(dec, false, kh); err != nil {
		t.Fatalf("matching cursor rejected: %v", err)
	}
	// Wrong direction ⇒ mismatch.
	if err := ValidateOrderCursor(dec, true, kh); !errors.Is(err, ErrCursorOrderMismatch) {
		t.Fatalf("direction mismatch err = %v, want ErrCursorOrderMismatch", err)
	}
	// Wrong key hash ⇒ mismatch.
	if err := ValidateOrderCursor(dec, false, kh+1); !errors.Is(err, ErrCursorOrderMismatch) {
		t.Fatalf("keyHash mismatch err = %v, want ErrCursorOrderMismatch", err)
	}
	// A v1 cursor presented to an order_by request ⇒ mismatch.
	v1, _ := DecodeScrollCursorTyped(EncodeScrollCursor(9))
	if err := ValidateOrderCursor(v1, false, kh); !errors.Is(err, ErrCursorOrderMismatch) {
		t.Fatalf("v1-vs-order err = %v, want ErrCursorOrderMismatch", err)
	}
	// First page (absent cursor) ⇒ ok, nothing to resume.
	if err := ValidateOrderCursor(DecodedScrollCursor{Present: false}, true, 999); err != nil {
		t.Fatalf("first page validate rejected: %v", err)
	}
}

// TestScrollCursorVersionsDistinguishable proves v1 and v2 tokens never collide.
func TestScrollCursorVersionsDistinguishable(t *testing.T) {
	v1 := EncodeScrollCursor(5)
	v2 := EncodeScrollCursorOrder(0, 5, false, 0)
	if v1 == v2 {
		t.Fatal("v1 and v2 tokens identical")
	}
	d1, _ := DecodeScrollCursorTyped(v1)
	d2, _ := DecodeScrollCursorTyped(v2)
	if d1.Version != 1 || d2.Version != 2 {
		t.Fatalf("versions = %d, %d, want 1, 2", d1.Version, d2.Version)
	}
}

// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"encoding/base64"
	"encoding/binary"
	"errors"
	"strings"
	"testing"
)

// TestScrollCursorOrderStringRoundTrip: encode -> decode yields the same
// (string, id, desc, keyHash) across a spread of values incl. empty + unicode +
// embedded NUL bytes (the strLen is a byte count, not a rune count).
func TestScrollCursorOrderStringRoundTrip(t *testing.T) {
	values := []string{
		"", "a", "Z", "alpha", "Älpha", "日本語",
		"with space", "tab\tnul\x00byte", strings.Repeat("x", 1000),
	}
	ids := []uint64{0, 1, 255, 1 << 32, 1<<64 - 1}
	keyHashes := []uint16{0, 1, 0xABCD, 0xFFFF}
	for _, v := range values {
		for _, id := range ids {
			for _, kh := range keyHashes {
				for _, desc := range []bool{false, true} {
					tok, err := EncodeScrollCursorOrderString(v, id, desc, kh)
					if err != nil {
						t.Fatalf("encode v3 (%q): %v", v, err)
					}
					if strings.ContainsAny(tok, "+/=") {
						t.Fatalf("v3 token %q not URL-safe/no-pad", tok)
					}
					got, derr := DecodeScrollCursorTyped(tok)
					if derr != nil {
						t.Fatalf("decode v3: %v", derr)
					}
					if !got.Present || got.Version != 3 {
						t.Fatalf("v3 decode = present %v version %d, want true 3", got.Present, got.Version)
					}
					if got.StrValue != v || got.LastID != id || got.Desc != desc || got.KeyHash != kh {
						t.Fatalf("v3 round-trip = {s=%q id=%d desc=%v kh=%d}, want {s=%q id=%d desc=%v kh=%d}",
							got.StrValue, got.LastID, got.Desc, got.KeyHash, v, id, desc, kh)
					}
					// A v3 cursor must not leak the float Value field.
					if got.Value != 0 {
						t.Fatalf("v3 leaked float Value=%v", got.Value)
					}
				}
			}
		}
	}
}

// TestScrollCursorV3ByteLayout pins the exact v3 wire layout:
// [ver=3][flags][keyHash:u16][strLen:u16][strBytes][id:u64].
func TestScrollCursorV3ByteLayout(t *testing.T) {
	const s = "abc"
	tok, err := EncodeScrollCursorOrderString(s, 0x1122334455667788, true, 0xBEEF)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	raw, err := base64.RawURLEncoding.DecodeString(tok)
	if err != nil {
		t.Fatalf("decode b64: %v", err)
	}
	wantLen := 6 + len(s) + 8
	if len(raw) != wantLen {
		t.Fatalf("v3 len = %d, want %d", len(raw), wantLen)
	}
	if raw[0] != 3 {
		t.Errorf("version byte = %d, want 3", raw[0])
	}
	if raw[1]&1 == 0 {
		t.Errorf("desc flag bit0 not set: flags=%08b", raw[1])
	}
	if kh := binary.BigEndian.Uint16(raw[2:]); kh != 0xBEEF {
		t.Errorf("keyHash = %#x, want 0xBEEF", kh)
	}
	if sl := binary.BigEndian.Uint16(raw[4:]); int(sl) != len(s) {
		t.Errorf("strLen = %d, want %d", sl, len(s))
	}
	if string(raw[6:6+len(s)]) != s {
		t.Errorf("strBytes = %q, want %q", raw[6:6+len(s)], s)
	}
	if id := binary.BigEndian.Uint64(raw[6+len(s):]); id != 0x1122334455667788 {
		t.Errorf("tiebreakID = %#x, want 0x1122334455667788", id)
	}
}

// TestScrollCursorV2StillDecodesAfterV3: adding v3 does NOT change v1/v2 decode —
// the version-byte dispatch keeps numeric/datetime cursors byte/behaviour-identical.
func TestScrollCursorV2StillDecodesAfterV3(t *testing.T) {
	// v2 numeric cursor decodes exactly as before.
	v2 := EncodeScrollCursorOrder(2.5, 9, true, 0x1234)
	d2, err := DecodeScrollCursorTyped(v2)
	if err != nil || !d2.Present || d2.Version != 2 {
		t.Fatalf("v2 decode = (%+v, %v), want present v2", d2, err)
	}
	if d2.Value != 2.5 || d2.LastID != 9 || !d2.Desc || d2.KeyHash != 0x1234 {
		t.Fatalf("v2 fields = %+v, want {2.5 9 true 0x1234}", d2)
	}
	if d2.StrValue != "" {
		t.Fatalf("v2 leaked StrValue=%q", d2.StrValue)
	}
	// v1 id-only cursor decodes exactly as before.
	v1 := EncodeScrollCursor(7)
	d1, err := DecodeScrollCursorTyped(v1)
	if err != nil || !d1.Present || d1.Version != 1 || d1.LastID != 7 {
		t.Fatalf("v1 decode = (%+v, %v), want present v1 id=7", d1, err)
	}
}

// TestScrollCursorV3Malformed: a corrupt/truncated/oversized-strLen v3 token is
// rejected loudly (ErrBadScrollCursor), never a panic / OOB read.
func TestScrollCursorV3Malformed(t *testing.T) {
	cases := map[string][]byte{}

	// Too short: header (6) + id (8) = 14 minimum; give 13.
	short := make([]byte, 13)
	short[0] = 3
	cases["below header+id"] = short

	// strLen declares 5 but only 2 string bytes present (len disagrees with size).
	mismatch := make([]byte, 6+2+8)
	mismatch[0] = 3
	binary.BigEndian.PutUint16(mismatch[4:], 5) // claims 5 string bytes
	cases["strLen overstates"] = mismatch

	// strLen declares 1 but 3 string bytes present (len understates).
	under := make([]byte, 6+3+8)
	under[0] = 3
	binary.BigEndian.PutUint16(under[4:], 1)
	cases["strLen understates"] = under

	// strLen = 0 but trailing bytes beyond header+0+id.
	extra := make([]byte, 6+0+8+4)
	extra[0] = 3
	binary.BigEndian.PutUint16(extra[4:], 0)
	cases["trailing extra"] = extra

	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			tok := base64.RawURLEncoding.EncodeToString(raw)
			_, err := DecodeScrollCursorTyped(tok)
			if !errors.Is(err, ErrBadScrollCursor) {
				t.Fatalf("decode(%q) err = %v, want ErrBadScrollCursor", name, err)
			}
		})
	}
}

// TestScrollCursorVersionsDistinguishableV3: v1/v2/v3 tokens never collide and each
// decodes to its own version (the dispatch is unambiguous; a v2 never misparses as v3).
func TestScrollCursorVersionsDistinguishableV3(t *testing.T) {
	v1 := EncodeScrollCursor(5)
	v2 := EncodeScrollCursorOrder(0, 5, false, 0)
	v3, err := EncodeScrollCursorOrderString("", 5, false, 0)
	if err != nil {
		t.Fatalf("encode v3: %v", err)
	}
	if v1 == v2 || v1 == v3 || v2 == v3 {
		t.Fatalf("token collision: v1=%q v2=%q v3=%q", v1, v2, v3)
	}
	d1, _ := DecodeScrollCursorTyped(v1)
	d2, _ := DecodeScrollCursorTyped(v2)
	d3, _ := DecodeScrollCursorTyped(v3)
	if d1.Version != 1 || d2.Version != 2 || d3.Version != 3 {
		t.Fatalf("versions = %d, %d, %d, want 1, 2, 3", d1.Version, d2.Version, d3.Version)
	}
}

// TestValidateOrderCursorString: the v3 validator accepts a matching v3 cursor and
// rejects a v1/v2 cursor (wrong resume type), a wrong direction, and a wrong key hash.
func TestValidateOrderCursorString(t *testing.T) {
	const kh = uint16(0x1234)
	v3, err := EncodeScrollCursorOrderString("hello", 5, false, kh)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	dec, err := DecodeScrollCursorTyped(v3)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if err := ValidateOrderCursorString(dec, false, kh); err != nil {
		t.Fatalf("matching v3 cursor rejected: %v", err)
	}
	if err := ValidateOrderCursorString(dec, true, kh); !errors.Is(err, ErrCursorOrderMismatch) {
		t.Fatalf("direction mismatch err = %v, want ErrCursorOrderMismatch", err)
	}
	if err := ValidateOrderCursorString(dec, false, kh+1); !errors.Is(err, ErrCursorOrderMismatch) {
		t.Fatalf("keyHash mismatch err = %v, want ErrCursorOrderMismatch", err)
	}
	// A v2 (float) cursor presented to a string order ⇒ mismatch.
	v2dec, _ := DecodeScrollCursorTyped(EncodeScrollCursorOrder(1, 5, false, kh))
	if err := ValidateOrderCursorString(v2dec, false, kh); !errors.Is(err, ErrCursorOrderMismatch) {
		t.Fatalf("v2-vs-string err = %v, want ErrCursorOrderMismatch", err)
	}
	// First page (absent cursor) ⇒ ok.
	if err := ValidateOrderCursorString(DecodedScrollCursor{Present: false}, true, 999); err != nil {
		t.Fatalf("first page validate rejected: %v", err)
	}
}

// TestScrollCursorStringSymmetricValidate: a v3 string cursor presented to the
// numeric/datetime ValidateOrderCursor (v2 guard) is rejected loudly — the two
// resume types never silently cross.
func TestScrollCursorStringSymmetricValidate(t *testing.T) {
	v3, err := EncodeScrollCursorOrderString("x", 1, false, 7)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	dec, _ := DecodeScrollCursorTyped(v3)
	if err := ValidateOrderCursor(dec, false, 7); !errors.Is(err, ErrCursorOrderMismatch) {
		t.Fatalf("v3-vs-numeric err = %v, want ErrCursorOrderMismatch", err)
	}
}

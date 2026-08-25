// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"encoding/base64"
	"errors"
	"strings"
	"testing"
)

// TestScrollCursorOrderTupleRoundTrip: encode -> decode yields the same per-key tuple
// (mixed numeric/datetime/string kinds), id, primary desc, and keyHash across a spread
// of arities (2..maxKeys) and values incl. empty + unicode + embedded-NUL strings.
func TestScrollCursorOrderTupleRoundTrip(t *testing.T) {
	tuples := [][]OrderKeyVal{
		{{Num: 1.5, Kind: 0}, {Str: "x", Kind: ScrollCursorKindString}},
		{{Str: "alpha", Kind: ScrollCursorKindString}, {Num: -7, Kind: 0}},
		{{Num: 100, Kind: 1 /*datetime*/}, {Num: 0.25, Kind: 0}, {Str: "", Kind: ScrollCursorKindString}},
		{{Str: "日本語", Kind: ScrollCursorKindString}, {Str: "nul\x00byte", Kind: ScrollCursorKindString}},
		{{Num: 1, Kind: 0}, {Num: 2, Kind: 0}, {Num: 3, Kind: 0}, {Str: strings.Repeat("y", 500), Kind: ScrollCursorKindString}},
	}
	ids := []uint64{0, 1, 1 << 32, 1<<64 - 1}
	for _, tup := range tuples {
		for _, id := range ids {
			for _, desc := range []bool{false, true} {
				tok, err := EncodeScrollCursorOrderTuple(tup, id, desc, 0xABCD)
				if err != nil {
					t.Fatalf("encode v4 (%v): %v", tup, err)
				}
				if strings.ContainsAny(tok, "+/=") {
					t.Fatalf("v4 token %q not URL-safe/no-pad", tok)
				}
				got, derr := DecodeScrollCursorTyped(tok)
				if derr != nil {
					t.Fatalf("decode v4: %v", derr)
				}
				if !got.Present || got.Version != 4 {
					t.Fatalf("v4 decode = present %v version %d, want true 4", got.Present, got.Version)
				}
				if got.LastID != id || got.Desc != desc || got.KeyHash != 0xABCD {
					t.Fatalf("v4 header = {id=%d desc=%v kh=%d}, want {id=%d desc=%v kh=%d}",
						got.LastID, got.Desc, got.KeyHash, id, desc, 0xABCD)
				}
				if len(got.Tuple) != len(tup) {
					t.Fatalf("v4 tuple len=%d, want %d", len(got.Tuple), len(tup))
				}
				for i := range tup {
					if got.Tuple[i].Kind != tup[i].Kind ||
						got.Tuple[i].Num != tup[i].Num ||
						got.Tuple[i].Str != tup[i].Str {
						t.Fatalf("v4 tuple[%d] = %+v, want %+v", i, got.Tuple[i], tup[i])
					}
				}
				// v4 must not leak the single-key scalar fields.
				if got.Value != 0 || got.StrValue != "" {
					t.Fatalf("v4 leaked single-key fields Value=%v StrValue=%q", got.Value, got.StrValue)
				}
			}
		}
	}
}

// TestScrollCursorV4ByteLayout pins the exact v4 wire layout:
// [ver=4][flags][keyHash:u16][numKeys][per-key kind+value][id:u64].
func TestScrollCursorV4ByteLayout(t *testing.T) {
	tup := []OrderKeyVal{{Num: 0, Kind: 0}, {Str: "ab", Kind: ScrollCursorKindString}}
	tok, err := EncodeScrollCursorOrderTuple(tup, 0x1122334455667788, true, 0xBEEF)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	raw, err := base64.RawURLEncoding.DecodeString(tok)
	if err != nil {
		t.Fatalf("b64 decode: %v", err)
	}
	// header(5) + key0(1 kind + 8 float) + key1(1 kind + 2 strLen + 2 bytes) + id(8) = 27
	if len(raw) != 5+9+5+8 {
		t.Fatalf("v4 raw len=%d, want %d", len(raw), 5+9+5+8)
	}
	if raw[0] != 4 {
		t.Fatalf("version byte = %d, want 4", raw[0])
	}
	if raw[1]&ScrollCursorFlagDesc == 0 {
		t.Fatalf("desc flag not set")
	}
	if raw[4] != 2 {
		t.Fatalf("numKeys byte = %d, want 2", raw[4])
	}
	if raw[5] != 0 { // key0 kind = numeric
		t.Fatalf("key0 kind = %d, want 0", raw[5])
	}
	if raw[5+9] != ScrollCursorKindString { // key1 kind = string
		t.Fatalf("key1 kind = %d, want %d", raw[5+9], ScrollCursorKindString)
	}
}

// TestScrollCursorV4SingleKeyRejectedEncode: encoding a <2-key tuple as v4 is fail-loud
// (single-key MUST emit v2/v3, never v4).
func TestScrollCursorV4SingleKeyRejectedEncode(t *testing.T) {
	for _, n := range []int{0, 1} {
		tup := make([]OrderKeyVal, n)
		for i := range tup {
			tup[i] = OrderKeyVal{Num: 1, Kind: 0}
		}
		if _, err := EncodeScrollCursorOrderTuple(tup, 1, false, 0); !errors.Is(err, ErrBadScrollCursor) {
			t.Fatalf("encode v4 with %d keys err=%v, want ErrBadScrollCursor", n, err)
		}
	}
}

// TestScrollCursorV4TooManyKeysRejectedEncode: > ScrollCursorMaxKeys keys is fail-loud.
func TestScrollCursorV4TooManyKeysRejectedEncode(t *testing.T) {
	tup := make([]OrderKeyVal, ScrollCursorMaxKeys+1)
	for i := range tup {
		tup[i] = OrderKeyVal{Num: float64(i), Kind: 0}
	}
	if _, err := EncodeScrollCursorOrderTuple(tup, 1, false, 0); !errors.Is(err, ErrBadScrollCursor) {
		t.Fatalf("encode v4 too-many-keys err=%v, want ErrBadScrollCursor", err)
	}
}

// TestScrollCursorV4MalformedDecode: corrupt v4 tokens fail-loud (no panic):
// bad numKeys, strLen overrun, trailing slop, truncated.
func TestScrollCursorV4MalformedDecode(t *testing.T) {
	good := []OrderKeyVal{{Num: 1, Kind: 0}, {Str: "ab", Kind: ScrollCursorKindString}}
	tok, err := EncodeScrollCursorOrderTuple(good, 7, false, 0x1234)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	raw, _ := base64.RawURLEncoding.DecodeString(tok)

	mutate := func(fn func([]byte) []byte) string {
		b := append([]byte(nil), raw...)
		return base64.RawURLEncoding.EncodeToString(fn(b))
	}

	cases := map[string]string{
		"numKeys=0":           mutate(func(b []byte) []byte { b[4] = 0; return b }),
		"numKeys=1":           mutate(func(b []byte) []byte { b[4] = 1; return b }),
		"numKeys=max+1":       mutate(func(b []byte) []byte { b[4] = ScrollCursorMaxKeys + 1; return b }),
		"numKeys huge":        mutate(func(b []byte) []byte { b[4] = 0xFF; return b }),
		"strLen overrun":      mutate(func(b []byte) []byte { b[5+9+1] = 0xFF; b[5+9+2] = 0xFF; return b }), // key1 strLen u16 huge
		"truncated header":    base64.RawURLEncoding.EncodeToString(raw[:4]),
		"trailing slop":       mutate(func(b []byte) []byte { return append(b, 0x00) }),
		"truncated mid-tuple": base64.RawURLEncoding.EncodeToString(raw[:6]),
	}
	for name, bad := range cases {
		got, derr := DecodeScrollCursorTyped(bad)
		if !errors.Is(derr, ErrBadScrollCursor) {
			t.Fatalf("%s: decode err=%v want ErrBadScrollCursor (got %+v)", name, derr, got)
		}
	}
}

// TestScrollCursorVersionsDistinct: v1/v2/v3 tokens still decode correctly and never
// misparse as v4, and a v4 token never decodes as a single-key version. The version
// byte is the sole discriminator.
func TestScrollCursorVersionsDistinct(t *testing.T) {
	v1 := EncodeScrollCursor(42)
	v2 := EncodeScrollCursorOrder(3.14, 9, true, 0x11)
	v3, err := EncodeScrollCursorOrderString("k", 9, false, 0x22)
	if err != nil {
		t.Fatalf("v3 encode: %v", err)
	}
	v4, err := EncodeScrollCursorOrderTuple([]OrderKeyVal{{Num: 1, Kind: 0}, {Num: 2, Kind: 0}}, 9, false, 0x33)
	if err != nil {
		t.Fatalf("v4 encode: %v", err)
	}
	for ver, tok := range map[uint8]string{1: v1, 2: v2, 3: v3, 4: v4} {
		got, derr := DecodeScrollCursorTyped(tok)
		if derr != nil {
			t.Fatalf("v%d decode: %v", ver, derr)
		}
		if got.Version != ver {
			t.Fatalf("token decoded as version %d, want %d", got.Version, ver)
		}
	}
}

// TestValidateOrderCursorTuple: the v4 mid-pagination guard rejects a wrong direction,
// changed key hash, wrong arity, and a non-v4 cursor; accepts a matching v4 and an
// absent (first-page) cursor.
func TestValidateOrderCursorTuple(t *testing.T) {
	tok, err := EncodeScrollCursorOrderTuple([]OrderKeyVal{{Num: 1, Kind: 0}, {Str: "z", Kind: ScrollCursorKindString}}, 5, true, 0xAA)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	c, derr := DecodeScrollCursorTyped(tok)
	if derr != nil {
		t.Fatalf("decode: %v", derr)
	}
	if err := ValidateOrderCursorTuple(c, true, 0xAA, 2); err != nil {
		t.Fatalf("matching v4 should validate, got %v", err)
	}
	if err := ValidateOrderCursorTuple(DecodedScrollCursor{}, true, 0xAA, 2); err != nil {
		t.Fatalf("absent cursor should validate (first page), got %v", err)
	}
	if err := ValidateOrderCursorTuple(c, false, 0xAA, 2); !errors.Is(err, ErrCursorOrderMismatch) {
		t.Fatalf("wrong desc should mismatch, got %v", err)
	}
	if err := ValidateOrderCursorTuple(c, true, 0xBB, 2); !errors.Is(err, ErrCursorOrderMismatch) {
		t.Fatalf("changed keyHash should mismatch, got %v", err)
	}
	if err := ValidateOrderCursorTuple(c, true, 0xAA, 3); !errors.Is(err, ErrCursorOrderMismatch) {
		t.Fatalf("wrong arity should mismatch, got %v", err)
	}
	// A v2 (single-key) cursor presented to a multi-key request is a mismatch.
	v2 := EncodeScrollCursorOrder(1, 5, true, 0xAA)
	cv2, _ := DecodeScrollCursorTyped(v2)
	if err := ValidateOrderCursorTuple(cv2, true, 0xAA, 2); !errors.Is(err, ErrCursorOrderMismatch) {
		t.Fatalf("v2 cursor to multi-key request should mismatch, got %v", err)
	}
}

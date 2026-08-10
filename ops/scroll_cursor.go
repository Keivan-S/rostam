// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// Scroll cursor token. A cursor is an opaque, URL-safe string wrapping the last
// (largest) point id returned on the previous page. The next page returns ids
// strictly greater than it (id-ascending, resume-after-id). The empty string is
// the FIRST page: no lower bound, so id 0 is included (scroll/get do not exclude
// id 0 even though ANN search does).
//
// Wire layout (9 bytes, then base64-url no-pad):
//
//	[ver:u8=1][lastID:u64 big-endian]
//
// base64.RawURLEncoding (URL-safe alphabet, NO padding) keeps the token safe in
// a query string / JSON body without escaping. The version byte lets a future
// cursor format be distinguished and rejected loudly rather than silently
// misread.

// scrollCursorVersion is the current cursor wire version.
const scrollCursorVersion byte = 1

// scrollCursorLen is the decoded byte length of a v1 cursor: 1 version byte +
// 8 id bytes.
const scrollCursorLen = 9

// ErrBadScrollCursor is returned by DecodeScrollCursor for any malformed,
// truncated, or unknown-version token. Transports map it to a 400 / InvalidArgument
// (fail loud — a bad cursor is a client error, never a panic).
var ErrBadScrollCursor = errors.New("ops: malformed scroll cursor")

// EncodeScrollCursor returns the opaque resume-after-id token for lastID: the
// largest id on the page just returned. The next page asks for ids strictly
// greater than lastID. The token is base64-url (no padding) of
// [ver:1][lastID:u64 BE].
func EncodeScrollCursor(lastID uint64) string {
	var buf [scrollCursorLen]byte
	buf[0] = scrollCursorVersion
	binary.BigEndian.PutUint64(buf[1:], lastID)
	return base64.RawURLEncoding.EncodeToString(buf[:])
}

// DecodeScrollCursor parses a scroll cursor token.
//
//   - An EMPTY token means the first page: it returns (0, false, nil) — there is
//     NO lower bound, so the caller must include id 0.
//   - A non-empty token is base64-decoded, length-checked (must be exactly 9
//     bytes), and version-checked (must be 1). On success it returns
//     (lastID, true, nil); the caller scrolls ids strictly greater than lastID.
//   - A malformed token (bad base64, wrong length, or unknown version) returns
//     (0, false, ErrBadScrollCursor). It never panics.
func DecodeScrollCursor(token string) (lastID uint64, present bool, err error) {
	if token == "" {
		return 0, false, nil
	}
	raw, decErr := base64.RawURLEncoding.DecodeString(token)
	if decErr != nil {
		return 0, false, ErrBadScrollCursor
	}
	if len(raw) != scrollCursorLen {
		return 0, false, ErrBadScrollCursor
	}
	if raw[0] != scrollCursorVersion {
		return 0, false, ErrBadScrollCursor
	}
	return binary.BigEndian.Uint64(raw[1:]), true, nil
}

// ---------------------------------------------------------------------------
// Cursor v2: ordered (value, id) resume cursor for order_by scroll.
// ---------------------------------------------------------------------------
//
// A v2 token resumes an order_by scroll: the next page returns points STRICTLY
// AFTER the (value, id) position of the last point on the previous page, in the
// (value, id) total order for the request's direction (see vector.OrderLess).
//
// Wire layout (20 bytes, then base64-url no-pad):
//
//	[ver:u8=2][flags:u8][keyHash:u16 BE][value:float64 BE][tiebreakID:u64 BE]
//	 0          1         2..3            4..11             12..19
//
//   - flags bit0 = desc (the scroll direction the cursor was issued for).
//   - keyHash    = vector.OrderKeyHash(order_by key) — lets a mid-pagination
//                  order_by-key change be detected and rejected loudly.
//   - value      = math.Float64bits of the order field value of the last point.
//   - tiebreakID = the last point's id (the (value, id) total-order tiebreak).
//
// v1 (id-only) and v2 (ordered) are distinguished by the leading version byte;
// v1 stays byte-identical and the version gate rejects any unknown version loudly.

// scrollCursorVersionOrder is the v2 (ordered) cursor wire version.
const scrollCursorVersionOrder byte = 2

// scrollCursorOrderLen is the decoded byte length of a v2 cursor:
// 1 version + 1 flags + 2 keyHash + 8 value + 8 id = 20.
const scrollCursorOrderLen = 20

// ---------------------------------------------------------------------------
// Cursor v3: ordered (stringValue, id) resume cursor for a STRING order_by scroll.
// ---------------------------------------------------------------------------
//
// A v3 token resumes a STRING order_by scroll: the next page returns points
// STRICTLY AFTER the (stringValue, id) position of the last point on the previous
// page, in the (stringValue, id) total order for the direction (see
// vector.OrderLessStr). It is the string analogue of v2 — the resume KEY is a
// variable-length string instead of a float64.
//
// Wire layout (variable, then base64-url no-pad):
//
//	[ver:u8=3][flags:u8][keyHash:u16 BE][strLen:u16 BE][strBytes...][tiebreakID:u64 BE]
//	 0          1         2..3            4..5           6..6+strLen   last 8
//
//   - flags bit0 = desc (the scroll direction the cursor was issued for).
//   - keyHash    = vector.OrderKeyHash(order_by key) — same mid-pagination
//                  key-change guard as v2.
//   - strLen     = the byte length of the resume string value (bounded; an
//                  oversized strLen or a length that disagrees with the token size
//                  is rejected loudly — no panic / no OOB).
//   - strBytes   = the order field's string value of the last point.
//   - tiebreakID = the last point's id (the (stringValue, id) total-order tiebreak).
//
// The leading version byte distinguishes v1/v2/v3; v2 (and v1) stay byte-identical
// and the version gate rejects any unknown version loudly. A v2 token is a fixed 20
// bytes and a v3 token starts with 0x03, so the two never collide.

// scrollCursorVersionString is the v3 (string-ordered) cursor wire version.
const scrollCursorVersionString byte = 3

// scrollCursorStringHeaderLen is the v3 fixed-header length BEFORE strBytes:
// 1 version + 1 flags + 2 keyHash + 2 strLen = 6.
const scrollCursorStringHeaderLen = 6

// scrollCursorStringMaxLen bounds the v3 resume string's byte length. The resume
// key is an order field value; 64 KiB is far beyond any realistic keyword/string
// field and keeps the cursor token small. strLen is a u16 so the wire max is 65535;
// this cap is the same so an honest u16 always fits, while a hostile/corrupt token
// whose declared length disagrees with its actual byte count is still rejected by
// the exact-length check in the decoder.
const scrollCursorStringMaxLen = 1 << 16

// scrollCursorFlagDesc is flags bit0: set when the cursor was issued for a
// descending order_by scroll.
const scrollCursorFlagDesc byte = 1 << 0

// ---------------------------------------------------------------------------
// Cursor v4: ordered (k1, …, kN, id) resume cursor for a MULTI-KEY order_by scroll.
// ---------------------------------------------------------------------------
//
// A v4 token resumes a MULTI-KEY order_by scroll: the next page returns points
// STRICTLY AFTER the (k1, …, kN, id) tuple position of the last point on the previous
// page, in the tuple-lexicographic total order for the per-key directions (see
// vector.OrderLessTuple). It is the multi-key generalisation of v2/v3 — the resume KEY
// is a TUPLE of typed values (each numeric/datetime float64 OR a string) instead of a
// single value.
//
// IMPORTANT BACK-COMPAT: v4 is emitted ONLY for N>1 keys. A SINGLE-key order_by still
// emits a v2 (numeric/datetime) or v3 (string) cursor — byte-identical to before. The
// version byte distinguishes v1/v2/v3/v4; v2/v3 are fixed/typed and a v4 token starts
// with 0x04, so none collide.
//
// Wire layout (variable, then base64-url no-pad):
//
//	[ver:u8=4][flags:u8][keyHash:u16 BE][numKeys:u8]
//	  { per key: [kind:u8] [ value ] }   -- value is:
//	      kind==string (2):  [strLen:u16 BE][strBytes...]
//	      else (0 numeric / 1 datetime):  [float64 BE]
//	[tiebreakID:u64 BE]
//
//   - flags bit0 = desc of the PRIMARY key (the request's primary direction; the
//     per-key desc is part of the order spec, not the cursor — the cursor only carries
//     the resume VALUES; the comparator re-applies the per-key directions). Kept for the
//     mid-pagination primary-direction guard, symmetric with v2/v3.
//   - keyHash    = vector.OrderKeyHash over the joined order_by key list — detects a
//     mid-pagination order_by change (same guard as v2/v3, generalised to the tuple).
//   - numKeys    = number of keys in the tuple (>= 2 for a real v4; bounded by
//     scrollCursorMaxKeys, fail-loud otherwise).
//   - per key kind+value, then the id tiebreak.

// scrollCursorVersionTuple is the v4 (multi-key-ordered) cursor wire version.
const scrollCursorVersionTuple byte = 4

// scrollCursorTupleHeaderLen is the v4 fixed-header length BEFORE the per-key block:
// 1 version + 1 flags + 2 keyHash + 1 numKeys = 5.
const scrollCursorTupleHeaderLen = 5

// scrollCursorMaxKeys bounds the v4 tuple arity. A multi-key order with more than this
// many keys is rejected loudly (a corrupt/hostile numKeys never drives an unbounded
// allocation or read). 16 is far beyond any realistic composite sort.
const scrollCursorMaxKeys = 16

// scrollCursorKindString is the v4 per-key kind byte for a string key (mirrors
// vector.OrderString == 2). Numeric (0) / datetime (1) keys carry a float64.
const scrollCursorKindString byte = 2

// OrderKeyVal is one typed key value in a v4 cursor resume tuple: a float64 (Num) for a
// numeric/datetime key or a string (Str) for a string key, tagged by Kind (the wire
// kind byte). It is the ops-local codec analogue of vector.OrderVal so the cursor codec
// stays dependency-free (v2/v3 likewise do not import vector). The wire path
// translates between vector.OrderVal and OrderKeyVal.
type OrderKeyVal struct {
	Num  float64
	Str  string
	Kind byte
}

// ErrCursorOrderMismatch is returned by ValidateOrderCursor when a decoded v2
// cursor's direction or order-key hash disagrees with the current request's
// order_by — i.e. the order_by changed mid-pagination. Transports map it to a
// 400 / InvalidArgument (fail loud — resuming would silently mis-page).
var ErrCursorOrderMismatch = errors.New("ops: scroll cursor order_by mismatch")

// DecodedScrollCursor is the typed result of decoding a scroll cursor token. It
// distinguishes a v1 id-only cursor from a v2 ordered cursor so the embedded
// coordinator can: decode → validate the v2 desc/keyHash against the request →
// seek the (Value, LastID) lower bound.
//
//   - Present == false  ⇒ empty token (first page; no lower bound).
//   - Version == 1      ⇒ v1: only LastID is meaningful (id-ascending resume).
//   - Version == 2      ⇒ v2: Value/Desc/KeyHash + LastID are meaningful
//     (ordered (value, id) resume).
//   - Version == 3      ⇒ v3: StrValue/Desc/KeyHash + LastID are meaningful
//     (ordered (stringValue, id) resume).
//   - Version == 4      ⇒ v4: Tuple/Desc/KeyHash + LastID are meaningful
//     (ordered (k1,…,kN, id) MULTI-KEY resume).
type DecodedScrollCursor struct {
	Present  bool
	Version  uint8
	LastID   uint64        // resume-after id (all versions; v2/v3/v4 tuple tiebreak).
	Value    float64       // v2 only: the order field value to resume after.
	StrValue string        // v3 only: the order field STRING value to resume after.
	Tuple    []OrderKeyVal // v4 only: the per-key resume tuple (>= 2 keys).
	Desc     bool          // v2/v3/v4: the (primary) direction the cursor was issued for.
	KeyHash  uint16        // v2/v3/v4: hash of the order_by key(s) the cursor was issued for.
}

// EncodeScrollCursorOrder returns the opaque v2 (ordered) resume cursor for the
// last point on the page just returned: its order field value, its id, the scroll
// direction, and the order_by key hash. The next page returns points strictly
// after (value, id) in the direction's (value, id) total order. See the layout
// comment above.
func EncodeScrollCursorOrder(value float64, id uint64, desc bool, keyHash uint16) string {
	var buf [scrollCursorOrderLen]byte
	buf[0] = scrollCursorVersionOrder
	if desc {
		buf[1] |= scrollCursorFlagDesc
	}
	binary.BigEndian.PutUint16(buf[2:], keyHash)
	binary.BigEndian.PutUint64(buf[4:], math.Float64bits(value))
	binary.BigEndian.PutUint64(buf[12:], id)
	return base64.RawURLEncoding.EncodeToString(buf[:])
}

// EncodeScrollCursorOrderString returns the opaque v3 (string-ordered) resume cursor
// for the last point on the page just returned: its order field STRING value, its id,
// the scroll direction, and the order_by key hash. The next page returns points
// strictly after (stringValue, id) in the direction's total order. See the v3 layout
// comment above. The string value's byte length must fit the v3 bound
// (scrollCursorStringMaxLen) — an oversized value returns ErrBadScrollCursor (the
// encoder is fail-loud too, so a corrupt token is never produced).
func EncodeScrollCursorOrderString(value string, id uint64, desc bool, keyHash uint16) (string, error) {
	if len(value) > scrollCursorStringMaxLen {
		return "", fmt.Errorf("%w: order_by string resume value too long (%d bytes)", ErrBadScrollCursor, len(value))
	}
	buf := make([]byte, scrollCursorStringHeaderLen+len(value)+8)
	buf[0] = scrollCursorVersionString
	if desc {
		buf[1] |= scrollCursorFlagDesc
	}
	binary.BigEndian.PutUint16(buf[2:], keyHash)
	binary.BigEndian.PutUint16(buf[4:], uint16(len(value))) //nolint:gosec // bounded above by scrollCursorStringMaxLen
	copy(buf[scrollCursorStringHeaderLen:], value)
	binary.BigEndian.PutUint64(buf[scrollCursorStringHeaderLen+len(value):], id)
	return base64.RawURLEncoding.EncodeToString(buf), nil
}

// EncodeScrollCursorOrderTuple returns the opaque v4 (multi-key-ordered) resume cursor
// for the last point on the page just returned: its per-key resume TUPLE (one OrderKeyVal
// per sort key, index 0 the primary), its id, the primary scroll direction, and the
// order_by key-list hash. The next page returns points strictly after the (k1,…,kN, id)
// tuple in the tuple-lexicographic total order. See the v4 layout comment above.
//
// It is FAIL-LOUD: an empty tuple, a tuple with fewer than 2 keys (single-key MUST emit
// v2/v3, never v4), more than scrollCursorMaxKeys keys, or a string key value longer than
// scrollCursorStringMaxLen all return ErrBadScrollCursor (a corrupt token is never
// produced).
func EncodeScrollCursorOrderTuple(tuple []OrderKeyVal, id uint64, desc bool, keyHash uint16) (string, error) {
	if len(tuple) < 2 {
		return "", fmt.Errorf("%w: v4 cursor requires >= 2 keys (got %d; single-key uses v2/v3)", ErrBadScrollCursor, len(tuple))
	}
	if len(tuple) > scrollCursorMaxKeys {
		return "", fmt.Errorf("%w: v4 cursor has too many keys (%d > %d)", ErrBadScrollCursor, len(tuple), scrollCursorMaxKeys)
	}
	// Size the buffer: fixed header + per-key (1 kind byte + value) + 8-byte id.
	size := scrollCursorTupleHeaderLen + 8
	for _, kv := range tuple {
		size++ // kind byte
		if kv.Kind == scrollCursorKindString {
			if len(kv.Str) > scrollCursorStringMaxLen {
				return "", fmt.Errorf("%w: v4 string key value too long (%d bytes)", ErrBadScrollCursor, len(kv.Str))
			}
			size += 2 + len(kv.Str) // strLen + bytes
		} else {
			size += 8 // float64
		}
	}
	buf := make([]byte, size)
	buf[0] = scrollCursorVersionTuple
	if desc {
		buf[1] |= scrollCursorFlagDesc
	}
	binary.BigEndian.PutUint16(buf[2:], keyHash)
	buf[4] = byte(len(tuple)) //nolint:gosec // bounded above by scrollCursorMaxKeys
	off := scrollCursorTupleHeaderLen
	for _, kv := range tuple {
		buf[off] = kv.Kind
		off++
		if kv.Kind == scrollCursorKindString {
			binary.BigEndian.PutUint16(buf[off:], uint16(len(kv.Str))) //nolint:gosec // bounded above
			off += 2
			copy(buf[off:], kv.Str)
			off += len(kv.Str)
		} else {
			binary.BigEndian.PutUint64(buf[off:], math.Float64bits(kv.Num))
			off += 8
		}
	}
	binary.BigEndian.PutUint64(buf[off:], id)
	return base64.RawURLEncoding.EncodeToString(buf), nil
}

// decodeScrollCursorTuple parses the v4 (multi-key) body (raw[1:] onward already past
// the version byte; raw includes the version byte at index 0). It is fail-loud: a
// numKeys out of [2, scrollCursorMaxKeys], a per-key strLen that overruns the token, an
// unknown kind byte, or a trailing-byte mismatch all return ErrBadScrollCursor — no
// panic, no OOB read, no unbounded allocation.
func decodeScrollCursorTuple(raw []byte) (DecodedScrollCursor, error) {
	if len(raw) < scrollCursorTupleHeaderLen+8 {
		return DecodedScrollCursor{}, ErrBadScrollCursor
	}
	numKeys := int(raw[4])
	if numKeys < 2 || numKeys > scrollCursorMaxKeys {
		return DecodedScrollCursor{}, ErrBadScrollCursor
	}
	tuple := make([]OrderKeyVal, numKeys)
	off := scrollCursorTupleHeaderLen
	for i := 0; i < numKeys; i++ {
		if off+1 > len(raw)-8 { // need at least the kind byte + 8 trailing id bytes
			return DecodedScrollCursor{}, ErrBadScrollCursor
		}
		kind := raw[off]
		off++
		if kind == scrollCursorKindString {
			if off+2 > len(raw)-8 {
				return DecodedScrollCursor{}, ErrBadScrollCursor
			}
			strLen := int(binary.BigEndian.Uint16(raw[off:]))
			off += 2
			if strLen > scrollCursorStringMaxLen || off+strLen > len(raw)-8 {
				return DecodedScrollCursor{}, ErrBadScrollCursor
			}
			tuple[i] = OrderKeyVal{Str: string(raw[off : off+strLen]), Kind: kind}
			off += strLen
		} else {
			if off+8 > len(raw)-8 {
				return DecodedScrollCursor{}, ErrBadScrollCursor
			}
			tuple[i] = OrderKeyVal{Num: math.Float64frombits(binary.BigEndian.Uint64(raw[off:])), Kind: kind}
			off += 8
		}
	}
	// After every key, exactly the 8-byte id must remain (no trailing slop).
	if off != len(raw)-8 {
		return DecodedScrollCursor{}, ErrBadScrollCursor
	}
	return DecodedScrollCursor{
		Present: true,
		Version: scrollCursorVersionTuple,
		Desc:    raw[1]&scrollCursorFlagDesc != 0,
		KeyHash: binary.BigEndian.Uint16(raw[2:]),
		Tuple:   tuple,
		LastID:  binary.BigEndian.Uint64(raw[off:]),
	}, nil
}

// DecodeScrollCursorTyped parses ANY scroll cursor token into a typed result,
// handling both v1 (id-only) and v2 (ordered) tokens.
//
//   - An EMPTY token ⇒ (DecodedScrollCursor{Present:false}, nil): first page.
//   - A v1 token     ⇒ Version 1, Present true, LastID set.
//   - A v2 token     ⇒ Version 2, Present true, LastID/Value/Desc/KeyHash set.
//   - Anything malformed (bad base64, wrong length for its version, or an unknown
//     version byte) ⇒ ErrBadScrollCursor. It never panics.
//
// This is the v1+v2-aware decoder. The legacy DecodeScrollCursor (3-tuple) stays
// for the id-only scroll path and rejects a v2 token (unknown version) loudly.
func DecodeScrollCursorTyped(token string) (DecodedScrollCursor, error) {
	if token == "" {
		return DecodedScrollCursor{Present: false}, nil
	}
	raw, decErr := base64.RawURLEncoding.DecodeString(token)
	if decErr != nil || len(raw) == 0 {
		return DecodedScrollCursor{}, ErrBadScrollCursor
	}
	switch raw[0] {
	case scrollCursorVersion: // v1
		if len(raw) != scrollCursorLen {
			return DecodedScrollCursor{}, ErrBadScrollCursor
		}
		return DecodedScrollCursor{
			Present: true,
			Version: scrollCursorVersion,
			LastID:  binary.BigEndian.Uint64(raw[1:]),
		}, nil
	case scrollCursorVersionOrder: // v2
		if len(raw) != scrollCursorOrderLen {
			return DecodedScrollCursor{}, ErrBadScrollCursor
		}
		return DecodedScrollCursor{
			Present: true,
			Version: scrollCursorVersionOrder,
			Desc:    raw[1]&scrollCursorFlagDesc != 0,
			KeyHash: binary.BigEndian.Uint16(raw[2:]),
			Value:   math.Float64frombits(binary.BigEndian.Uint64(raw[4:])),
			LastID:  binary.BigEndian.Uint64(raw[12:]),
		}, nil
	case scrollCursorVersionString: // v3
		// Must hold at least the fixed header (incl. strLen) + the 8-byte id.
		if len(raw) < scrollCursorStringHeaderLen+8 {
			return DecodedScrollCursor{}, ErrBadScrollCursor
		}
		strLen := int(binary.BigEndian.Uint16(raw[4:]))
		if strLen > scrollCursorStringMaxLen {
			return DecodedScrollCursor{}, ErrBadScrollCursor
		}
		// The declared strLen must EXACTLY account for the token size: header +
		// strLen string bytes + 8 id bytes. A length that disagrees with the actual
		// byte count is corruption — reject loudly (no OOB slice, no panic).
		if len(raw) != scrollCursorStringHeaderLen+strLen+8 {
			return DecodedScrollCursor{}, ErrBadScrollCursor
		}
		strEnd := scrollCursorStringHeaderLen + strLen
		return DecodedScrollCursor{
			Present:  true,
			Version:  scrollCursorVersionString,
			Desc:     raw[1]&scrollCursorFlagDesc != 0,
			KeyHash:  binary.BigEndian.Uint16(raw[2:]),
			StrValue: string(raw[scrollCursorStringHeaderLen:strEnd]),
			LastID:   binary.BigEndian.Uint64(raw[strEnd:]),
		}, nil
	case scrollCursorVersionTuple: // v4 (multi-key)
		return decodeScrollCursorTuple(raw)
	default:
		return DecodedScrollCursor{}, ErrBadScrollCursor
	}
}

// ValidateOrderCursor checks a decoded v2 cursor against the current request's
// order_by (its direction desc and key hash). It is the loud-rejection guard the
// caller runs after DecodeScrollCursorTyped before seeking:
//
//   - A v1 (id-only) cursor presented to an order_by request ⇒ mismatch (the
//     cursor carries no ordered position; resuming would mis-page).
//   - A v2 cursor whose Desc or KeyHash disagrees with the request ⇒ mismatch
//     (the order_by changed mid-pagination).
//   - An absent cursor (first page) ⇒ ok (nil); there is nothing to resume.
//
// A matching v2 cursor returns nil. Transports map ErrCursorOrderMismatch to a
// 400 / InvalidArgument. The symmetric guard (a v2 cursor presented to a request
// with NO order_by) is the caller's: it should reject a v2 cursor when no order_by
// is set (the no-order_by path only understands v1).
func ValidateOrderCursor(c DecodedScrollCursor, desc bool, keyHash uint16) error {
	if !c.Present {
		return nil
	}
	if c.Version != scrollCursorVersionOrder {
		return fmt.Errorf("%w: cursor is not an ordered (v2) cursor (version %d)", ErrCursorOrderMismatch, c.Version)
	}
	if c.Desc != desc {
		return fmt.Errorf("%w: cursor direction desc=%v, request desc=%v", ErrCursorOrderMismatch, c.Desc, desc)
	}
	if c.KeyHash != keyHash {
		return fmt.Errorf("%w: cursor order_by key changed mid-pagination", ErrCursorOrderMismatch)
	}
	return nil
}

// ValidateOrderCursorString is the v3 (string-ordered) analogue of ValidateOrderCursor:
// it checks a decoded cursor against a STRING order_by request's direction + key hash.
// A non-v3 cursor (v1 id-only, or a v2 float cursor presented to a string order) is a
// mismatch — resuming would mis-page (the resume key type differs). An absent cursor
// (first page) is ok. Used by the string-order wire path. Kept separate so the
// numeric/datetime ValidateOrderCursor stays byte/behaviour-identical.
func ValidateOrderCursorString(c DecodedScrollCursor, desc bool, keyHash uint16) error {
	if !c.Present {
		return nil
	}
	if c.Version != scrollCursorVersionString {
		return fmt.Errorf("%w: cursor is not a string-ordered (v3) cursor (version %d)", ErrCursorOrderMismatch, c.Version)
	}
	if c.Desc != desc {
		return fmt.Errorf("%w: cursor direction desc=%v, request desc=%v", ErrCursorOrderMismatch, c.Desc, desc)
	}
	if c.KeyHash != keyHash {
		return fmt.Errorf("%w: cursor order_by key changed mid-pagination", ErrCursorOrderMismatch)
	}
	return nil
}

// ValidateOrderCursorTuple is the v4 (multi-key-ordered) analogue of ValidateOrderCursor:
// it checks a decoded cursor against a MULTI-KEY order_by request's primary direction +
// key-list hash + arity. A non-v4 cursor (v1 id-only, or a v2/v3 single-key cursor
// presented to a multi-key order) is a mismatch — resuming would mis-page (the resume
// shape differs). An absent cursor (first page) is ok. numKeys must equal the request's
// key count (a different arity is a mid-pagination order change). Used by the multi-key
// wire path. Kept separate so the single-key validators stay byte/behaviour-
// identical.
func ValidateOrderCursorTuple(c DecodedScrollCursor, desc bool, keyHash uint16, numKeys int) error {
	if !c.Present {
		return nil
	}
	if c.Version != scrollCursorVersionTuple {
		return fmt.Errorf("%w: cursor is not a multi-key (v4) cursor (version %d)", ErrCursorOrderMismatch, c.Version)
	}
	if c.Desc != desc {
		return fmt.Errorf("%w: cursor direction desc=%v, request desc=%v", ErrCursorOrderMismatch, c.Desc, desc)
	}
	if c.KeyHash != keyHash {
		return fmt.Errorf("%w: cursor order_by key changed mid-pagination", ErrCursorOrderMismatch)
	}
	if len(c.Tuple) != numKeys {
		return fmt.Errorf("%w: cursor key count %d != request %d (order_by changed mid-pagination)", ErrCursorOrderMismatch, len(c.Tuple), numKeys)
	}
	return nil
}

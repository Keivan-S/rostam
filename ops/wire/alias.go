// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"encoding/binary"
	"errors"
)

// errAliasArgsTruncated is returned by the alias Decode* helpers when the args
// bytes are shorter than the layout requires (fail-loud, mirroring
// ErrVectorArgsTruncated).
var errAliasArgsTruncated = errors.New("ops: alias args truncated")

// AliasAction is one mutation in an atomic alias batch on the wire. It mirrors
// the embedded-level rostam.AliasAction and cluster.AliasAction; the ops codec
// keeps the alias coordinator ops decoupled from those packages. Delete=true
// removes Alias; otherwise Alias is created/overwritten to point at Canonical.
type AliasAction struct {
	Alias     string
	Canonical string
	Delete    bool
}

// AliasEntry is one alias→collection pair in a list result.
type AliasEntry struct {
	Alias      string
	Collection string
}

// putString appends [len:u16][s] to buf and returns the grown slice. Lengths
// are bounded by collection/alias name limits well under 65535.
func putString(buf []byte, s string) []byte {
	var hdr [2]byte
	binary.BigEndian.PutUint16(hdr[:], uint16(len(s))) //nolint:gosec // name length bounded
	buf = append(buf, hdr[:]...)
	return append(buf, s...)
}

// getString reads a [len:u16][s] field at off, returning the string and the new
// offset. ok=false on truncation.
func getString(b []byte, off int) (string, int, bool) {
	if off+2 > len(b) {
		return "", off, false
	}
	n := int(binary.BigEndian.Uint16(b[off : off+2]))
	off += 2
	if off+n > len(b) {
		return "", off, false
	}
	return string(b[off : off+n]), off + n, true
}

// EncodeAliasBatchArgs serializes an atomic alias batch. Wire:
//
//	[count:u32]( [aliasLen:u16][alias][canonLen:u16][canon][delete:u8] )*
//
// All N actions apply in one meta-Raft log entry (atomic swap). For a delete the
// canonical is ignored (encoded empty).
func EncodeAliasBatchArgs(actions []AliasAction) []byte {
	buf := make([]byte, 4, 4+len(actions)*16)
	binary.BigEndian.PutUint32(buf[0:4], uint32(len(actions))) //nolint:gosec // batch size bounded
	for _, a := range actions {
		buf = putString(buf, a.Alias)
		buf = putString(buf, a.Canonical)
		var d byte
		if a.Delete {
			d = 1
		}
		buf = append(buf, d)
	}
	return buf
}

// DecodeAliasBatchArgs reads args produced by EncodeAliasBatchArgs. Length-checked
// (fail-loud) on every field.
func DecodeAliasBatchArgs(args []byte) ([]AliasAction, error) {
	if len(args) < 4 {
		return nil, errAliasArgsTruncated
	}
	n := int(binary.BigEndian.Uint32(args[0:4]))
	off := 4
	// Bound the declared count against the buffer before allocating: the minimum
	// action is [aliasLen:u16][canonLen:u16][delete:u8] = 5 bytes, so a body that
	// cannot hold n actions is truncated. This stops a tiny payload declaring a
	// huge n (up to ~4.29e9) from forcing a multi-hundred-GB capacity allocation.
	if n < 0 || n > (len(args)-4)/5 {
		return nil, errAliasArgsTruncated
	}
	out := make([]AliasAction, 0, n)
	for i := 0; i < n; i++ {
		var (
			alias, canon string
			ok           bool
		)
		alias, off, ok = getString(args, off)
		if !ok {
			return nil, errAliasArgsTruncated
		}
		canon, off, ok = getString(args, off)
		if !ok {
			return nil, errAliasArgsTruncated
		}
		if off+1 > len(args) {
			return nil, errAliasArgsTruncated
		}
		del := args[off] != 0
		off++
		out = append(out, AliasAction{Alias: alias, Canonical: canon, Delete: del})
	}
	return out, nil
}

// EncodeAliasCreateArgs is a thin convenience that lowers a single create to a
// one-action batch (Delete=false).
func EncodeAliasCreateArgs(alias, collection string) []byte {
	return EncodeAliasBatchArgs([]AliasAction{{Alias: alias, Canonical: collection}})
}

// EncodeAliasDeleteArgs is a thin convenience that lowers a single delete to a
// one-action batch (Delete=true, canonical ignored).
func EncodeAliasDeleteArgs(alias string) []byte {
	return EncodeAliasBatchArgs([]AliasAction{{Alias: alias, Delete: true}})
}

// EncodeAliasListArgs serializes a list request with an optional target-collection
// filter. Wire: [collLen:u16][coll]. An empty collection means "no filter".
func EncodeAliasListArgs(collection string) []byte {
	return putString(nil, collection)
}

// DecodeAliasListArgs reads args produced by EncodeAliasListArgs.
func DecodeAliasListArgs(args []byte) (string, error) {
	coll, _, ok := getString(args, 0)
	if !ok {
		return "", errAliasArgsTruncated
	}
	return coll, nil
}

// EncodeAliasListResult serializes a list response. Wire:
//
//	[count:u32]( [aliasLen:u16][alias][collLen:u16][coll] )*
func EncodeAliasListResult(entries []AliasEntry) []byte {
	buf := make([]byte, 4, 4+len(entries)*16)
	binary.BigEndian.PutUint32(buf[0:4], uint32(len(entries))) //nolint:gosec // entry count bounded
	for _, e := range entries {
		buf = putString(buf, e.Alias)
		buf = putString(buf, e.Collection)
	}
	return buf
}

// DecodeAliasListResult reads a result produced by EncodeAliasListResult.
// Length-checked (fail-loud).
func DecodeAliasListResult(body []byte) ([]AliasEntry, error) {
	if len(body) < 4 {
		return nil, errAliasArgsTruncated
	}
	n := int(binary.BigEndian.Uint32(body[0:4]))
	off := 4
	// Bound the declared count before allocating: the minimum entry is
	// [aliasLen:u16][collLen:u16] = 4 bytes, so a truncated/hostile body cannot
	// force an oversized capacity reservation.
	if n < 0 || n > (len(body)-4)/4 {
		return nil, errAliasArgsTruncated
	}
	out := make([]AliasEntry, 0, n)
	for i := 0; i < n; i++ {
		var (
			alias, coll string
			ok          bool
		)
		alias, off, ok = getString(body, off)
		if !ok {
			return nil, errAliasArgsTruncated
		}
		coll, off, ok = getString(body, off)
		if !ok {
			return nil, errAliasArgsTruncated
		}
		out = append(out, AliasEntry{Alias: alias, Collection: coll})
	}
	return out, nil
}

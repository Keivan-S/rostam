// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"encoding/binary"
	"errors"
)

// Online key-admin coordinator virtual-op names. Like alias_batch/alias_list
// these are intercepted at the coordinator dispatcher (NOT shard-routed, NOT in
// the ops registry) and mutate/read the node's *vector.KeyRegistry directly. The
// authz layer classifies all three as admin (adminOps), so only an admin-scoped
// key passes the normal authorize gate.
//
//	__keys_add__    register a new API key (token,tenant,scopes,cert_cn)
//	__keys_revoke__ remove an API key by token
//	__keys_list__   list registered keys REDACTED (fingerprint, never raw token)
const (
	OpKeysAdd    = "__keys_add__"
	OpKeysRevoke = "__keys_revoke__"
	OpKeysList   = "__keys_list__"
)

// errKeysArgsTruncated is returned by the keys Decode* helpers when the args
// bytes are shorter than the layout requires (fail-loud, mirroring
// errAliasArgsTruncated).
var errKeysArgsTruncated = errors.New("ops: keys args truncated")

// KeysAddArgs is the decoded wire form of an __keys_add__ request. The new key's
// raw Token travels here (a secret) — it is consumed by the dispatcher's
// registry.AddKey and never echoed back in any result.
type KeysAddArgs struct {
	Token  string
	Tenant string
	Scopes []string
	CertCN string
}

// RedactedKeyEntry is one token-free entry in an __keys_list__ result. It mirrors
// vector.RedactedKey on the wire; the ops codec keeps the keys ops decoupled from
// the vector package. It deliberately has NO token field so no list result can
// ever carry the secret.
type RedactedKeyEntry struct {
	Fingerprint string
	Tenant      string
	Scopes      []string
	CertCN      string
}

// putStrings appends [count:u16]([len:u16][s])* to buf. Scope counts/lengths are
// well under 65535.
func putStrings(buf []byte, ss []string) []byte {
	var hdr [2]byte
	binary.BigEndian.PutUint16(hdr[:], uint16(len(ss))) //nolint:gosec // scope count bounded
	buf = append(buf, hdr[:]...)
	for _, s := range ss {
		buf = putString(buf, s)
	}
	return buf
}

// getStrings reads a [count:u16]([len:u16][s])* field at off, returning the
// slice and the new offset. ok=false on truncation.
func getStrings(b []byte, off int) ([]string, int, bool) {
	if off+2 > len(b) {
		return nil, off, false
	}
	n := int(binary.BigEndian.Uint16(b[off : off+2]))
	off += 2
	// A string costs >= 2 bytes ([len:u16] with an empty string). The count is a
	// u16 so the exposure is far smaller than the u32 counts elsewhere, but an
	// unvalidated reservation is an unvalidated reservation.
	if !CountFitsIn(n, len(b)-off, 2) {
		return nil, off, false
	}
	out := make([]string, 0, n)
	for i := 0; i < n; i++ {
		var (
			s  string
			ok bool
		)
		s, off, ok = getString(b, off)
		if !ok {
			return nil, off, false
		}
		out = append(out, s)
	}
	return out, off, true
}

// EncodeKeysAddArgs serializes an __keys_add__ request. Wire:
//
//	[tokenLen:u16][token][tenantLen:u16][tenant]
//	[scopeCount:u16]([scopeLen:u16][scope])*[cnLen:u16][cn]
func EncodeKeysAddArgs(a KeysAddArgs) []byte {
	buf := putString(nil, a.Token)
	buf = putString(buf, a.Tenant)
	buf = putStrings(buf, a.Scopes)
	buf = putString(buf, a.CertCN)
	return buf
}

// DecodeKeysAddArgs reads args produced by EncodeKeysAddArgs. Length-checked
// (fail-loud) on every field.
func DecodeKeysAddArgs(args []byte) (KeysAddArgs, error) {
	var (
		a   KeysAddArgs
		off int
		ok  bool
	)
	a.Token, off, ok = getString(args, off)
	if !ok {
		return KeysAddArgs{}, errKeysArgsTruncated
	}
	a.Tenant, off, ok = getString(args, off)
	if !ok {
		return KeysAddArgs{}, errKeysArgsTruncated
	}
	a.Scopes, off, ok = getStrings(args, off)
	if !ok {
		return KeysAddArgs{}, errKeysArgsTruncated
	}
	a.CertCN, _, ok = getString(args, off)
	if !ok {
		return KeysAddArgs{}, errKeysArgsTruncated
	}
	return a, nil
}

// EncodeKeysRevokeArgs serializes an __keys_revoke__ request. Wire:
// [tokenLen:u16][token].
func EncodeKeysRevokeArgs(token string) []byte {
	return putString(nil, token)
}

// DecodeKeysRevokeArgs reads args produced by EncodeKeysRevokeArgs.
func DecodeKeysRevokeArgs(args []byte) (string, error) {
	token, _, ok := getString(args, 0)
	if !ok {
		return "", errKeysArgsTruncated
	}
	return token, nil
}

// EncodeKeysListArgs serializes an __keys_list__ request (no parameters). The
// frame is empty; provided for symmetry with the other keys codecs.
func EncodeKeysListArgs() []byte { return nil }

// EncodeKeysListResult serializes a REDACTED list response. Wire:
//
//	[count:u32]( [fpLen:u16][fp][tenantLen:u16][tenant]
//	             [scopeCount:u16]([scopeLen:u16][scope])*[cnLen:u16][cn] )*
//
// SECURITY: there is NO token field in this layout — a list result can never
// carry the raw token by construction.
func EncodeKeysListResult(entries []RedactedKeyEntry) []byte {
	buf := make([]byte, 4)
	binary.BigEndian.PutUint32(buf[0:4], uint32(len(entries))) //nolint:gosec // entry count bounded
	for _, e := range entries {
		buf = putString(buf, e.Fingerprint)
		buf = putString(buf, e.Tenant)
		buf = putStrings(buf, e.Scopes)
		buf = putString(buf, e.CertCN)
	}
	return buf
}

// DecodeKeysListResult reads a result produced by EncodeKeysListResult.
// Length-checked (fail-loud).
func DecodeKeysListResult(body []byte) ([]RedactedKeyEntry, error) {
	if len(body) < 4 {
		return nil, errKeysArgsTruncated
	}
	n := int(binary.BigEndian.Uint32(body[0:4]))
	off := 4
	// Bound the declared count before allocating: the minimum entry is
	// [fpLen:u16][tenantLen:u16][scopeCount:u16][cnLen:u16] = 8 bytes, so a
	// truncated/hostile body cannot force an oversized capacity reservation.
	if n < 0 || n > (len(body)-4)/8 {
		return nil, errKeysArgsTruncated
	}
	out := make([]RedactedKeyEntry, 0, n)
	for i := 0; i < n; i++ {
		var (
			e  RedactedKeyEntry
			ok bool
		)
		e.Fingerprint, off, ok = getString(body, off)
		if !ok {
			return nil, errKeysArgsTruncated
		}
		e.Tenant, off, ok = getString(body, off)
		if !ok {
			return nil, errKeysArgsTruncated
		}
		e.Scopes, off, ok = getStrings(body, off)
		if !ok {
			return nil, errKeysArgsTruncated
		}
		e.CertCN, off, ok = getString(body, off)
		if !ok {
			return nil, errKeysArgsTruncated
		}
		out = append(out, e)
	}
	return out, nil
}

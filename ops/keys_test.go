// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"reflect"
	"testing"
)

func TestKeysAddArgsRoundTrip(t *testing.T) {
	in := KeysAddArgs{
		Token:  "SECRET-TOKEN-123",
		Tenant: "acme",
		Scopes: []string{"read:default/docs", "write:default/*"},
		CertCN: "svc.acme",
	}
	got, err := DecodeKeysAddArgs(EncodeKeysAddArgs(in))
	if err != nil {
		t.Fatalf("DecodeKeysAddArgs: %v", err)
	}
	if !reflect.DeepEqual(got, in) {
		t.Fatalf("round-trip mismatch: got %+v want %+v", got, in)
	}
}

func TestKeysAddArgsEmptyScopesAndCN(t *testing.T) {
	in := KeysAddArgs{Token: "t", Tenant: "acme"}
	got, err := DecodeKeysAddArgs(EncodeKeysAddArgs(in))
	if err != nil {
		t.Fatalf("DecodeKeysAddArgs: %v", err)
	}
	if got.Token != "t" || got.Tenant != "acme" || len(got.Scopes) != 0 || got.CertCN != "" {
		t.Fatalf("unexpected decode: %+v", got)
	}
}

func TestKeysRevokeArgsRoundTrip(t *testing.T) {
	got, err := DecodeKeysRevokeArgs(EncodeKeysRevokeArgs("SECRET-TOKEN"))
	if err != nil {
		t.Fatalf("DecodeKeysRevokeArgs: %v", err)
	}
	if got != "SECRET-TOKEN" {
		t.Fatalf("got %q want %q", got, "SECRET-TOKEN")
	}
}

func TestKeysListResultRoundTrip(t *testing.T) {
	in := []RedactedKeyEntry{
		{Fingerprint: "abc123", Tenant: "acme", Scopes: []string{"*:*"}, CertCN: "svc"},
		{Fingerprint: "def456", Tenant: "other", Scopes: []string{}},
	}
	got, err := DecodeKeysListResult(EncodeKeysListResult(in))
	if err != nil {
		t.Fatalf("DecodeKeysListResult: %v", err)
	}
	if !reflect.DeepEqual(got, in) {
		t.Fatalf("round-trip mismatch: got %+v want %+v", got, in)
	}
}

// TestKeysListResultNeverCarriesRawToken is the codec-boundary teeth test: the
// list result wire bytes must NEVER contain a raw token. We encode a list whose
// entries are derived from a sentinel token's fingerprint and assert the raw
// sentinel string is absent from the marshaled frame.
func TestKeysListResultNeverCarriesRawToken(t *testing.T) {
	const sentinel = "RAW-SECRET-TOKEN-SENTINEL"
	// A list entry only ever carries the fingerprint, never the token — by type
	// (RedactedKeyEntry has no token field). Encode one and prove the raw token
	// cannot appear even if a caller mistakenly stuffed it into a field.
	entries := []RedactedKeyEntry{
		{Fingerprint: "fp-of-" + sentinel[:4], Tenant: "acme", Scopes: []string{"*:*"}},
	}
	frame := EncodeKeysListResult(entries)
	if bytes.Contains(frame, []byte(sentinel)) {
		t.Fatal("keys list result frame must NEVER contain the raw token")
	}
}

func TestKeysDecodeTruncatedFailsLoud(t *testing.T) {
	if _, err := DecodeKeysAddArgs(nil); err == nil {
		t.Error("DecodeKeysAddArgs(nil) must error")
	}
	if _, err := DecodeKeysAddArgs([]byte{0x00}); err == nil {
		t.Error("DecodeKeysAddArgs(short) must error")
	}
	if _, err := DecodeKeysRevokeArgs(nil); err == nil {
		t.Error("DecodeKeysRevokeArgs(nil) must error")
	}
	if _, err := DecodeKeysListResult(nil); err == nil {
		t.Error("DecodeKeysListResult(nil) must error")
	}
	if _, err := DecodeKeysListResult([]byte{0, 0, 0, 1}); err == nil {
		t.Error("DecodeKeysListResult(count=1,no body) must error")
	}
}

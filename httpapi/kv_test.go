// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"encoding/base64"
	"net/http"
	"strings"
	"testing"
)

// TestHTTPKVPutGetUTF8 covers a UTF-8 value round-trip: PUT with a "value"
// string, then GET returns found + both value_b64 (raw bytes) and value_utf8
// (the string, present because the bytes are valid UTF-8).
func TestHTTPKVPutGetUTF8(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "PUT", "/v1/kv/greeting", `{"value":"hello"}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("put = %d, want 200 (%s)", rec.Code, rec.Body)
	}

	var out kvGetResponse
	rec = do(t, h, "GET", "/v1/kv/greeting", "", &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("get = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if !out.Found {
		t.Fatalf("found = false, want true")
	}
	if out.ValueUTF8 == nil || *out.ValueUTF8 != "hello" {
		t.Fatalf("value_utf8 = %v, want hello", out.ValueUTF8)
	}
	if want := base64.StdEncoding.EncodeToString([]byte("hello")); out.ValueB64 == nil || *out.ValueB64 != want {
		t.Fatalf("value_b64 = %v, want %q", out.ValueB64, want)
	}
}

// TestHTTPKVPutGetBinary covers a binary value: PUT with value_b64 of bytes that
// are NOT valid UTF-8, then GET returns value_b64 but OMITS value_utf8.
func TestHTTPKVPutGetBinary(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	raw := []byte{0xff, 0xfe, 0x00, 0x01}
	b64 := base64.StdEncoding.EncodeToString(raw)
	rec := do(t, h, "PUT", "/v1/kv/blob", `{"value_b64":"`+b64+`"}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("put = %d, want 200 (%s)", rec.Code, rec.Body)
	}

	var out kvGetResponse
	rec = do(t, h, "GET", "/v1/kv/blob", "", &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("get = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if !out.Found || out.ValueB64 == nil || *out.ValueB64 != b64 {
		t.Fatalf("found/value_b64 = %v/%v, want true/%q", out.Found, out.ValueB64, b64)
	}
	if out.ValueUTF8 != nil {
		t.Fatalf("value_utf8 = %v, want omitted (binary bytes)", out.ValueUTF8)
	}
}

// TestHTTPKVPutGetEmpty covers an empty-but-present value: PUT with "" must
// round-trip as found=true with BOTH value_b64 and value_utf8 present (empty
// strings), not omitted as if the key were absent — an empty value is valid
// UTF-8 and distinct from a miss.
func TestHTTPKVPutGetEmpty(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "PUT", "/v1/kv/empty", `{"value":""}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("put = %d, want 200 (%s)", rec.Code, rec.Body)
	}

	var out kvGetResponse
	rec = do(t, h, "GET", "/v1/kv/empty", "", &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("get = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if !out.Found {
		t.Fatalf("found = false, want true")
	}
	if out.ValueB64 == nil || *out.ValueB64 != "" {
		t.Fatalf("value_b64 = %v, want present and empty", out.ValueB64)
	}
	if out.ValueUTF8 == nil || *out.ValueUTF8 != "" {
		t.Fatalf("value_utf8 = %v, want present and empty", out.ValueUTF8)
	}

	// A genuine miss still omits both value fields.
	var missOut kvGetResponse
	do(t, h, "GET", "/v1/kv/nope-not-there", "", &missOut)
	if missOut.Found {
		t.Fatalf("found = true, want false for a missing key")
	}
	if missOut.ValueB64 != nil || missOut.ValueUTF8 != nil {
		t.Fatalf("value_b64/value_utf8 = %v/%v, want both nil on a miss", missOut.ValueB64, missOut.ValueUTF8)
	}
}

// TestHTTPKVGetNotFound proves a key miss is {"found":false} at HTTP 200, NOT a
// 404 — a client distinguishes absent from error without status-code parsing.
func TestHTTPKVGetNotFound(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	var out kvGetResponse
	rec := do(t, h, "GET", "/v1/kv/missing", "", &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("get missing = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if out.Found {
		t.Fatalf("found = true, want false for a missing key")
	}
}

// TestHTTPKVGetWithTTL covers ?with_ttl=1: a key written WITHOUT a TTL reports
// ttl_ms = -1 (no expiry), and one written WITH a ttl_ms reports a positive
// remaining budget.
func TestHTTPKVGetWithTTL(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	// No TTL -> -1 (the ttl op's no-expiry convention).
	do(t, h, "PUT", "/v1/kv/perm", `{"value":"x"}`, nil)
	var out kvGetResponse
	do(t, h, "GET", "/v1/kv/perm?with_ttl=1", "", &out)
	if out.TTLMs == nil || *out.TTLMs != -1 {
		t.Fatalf("ttl_ms = %v, want -1 (no expiry)", out.TTLMs)
	}

	// With TTL -> a positive remaining budget.
	do(t, h, "PUT", "/v1/kv/temp", `{"value":"x","ttl_ms":60000}`, nil)
	out = kvGetResponse{}
	do(t, h, "GET", "/v1/kv/temp?with_ttl=1", "", &out)
	if out.TTLMs == nil || *out.TTLMs <= 0 {
		t.Fatalf("ttl_ms = %v, want > 0", out.TTLMs)
	}

	// Without the flag, ttl_ms is omitted.
	out = kvGetResponse{}
	do(t, h, "GET", "/v1/kv/perm", "", &out)
	if out.TTLMs != nil {
		t.Fatalf("ttl_ms = %v, want omitted without ?with_ttl", out.TTLMs)
	}
}

// TestHTTPKVPutValidation covers the both/neither value guard: a body with BOTH
// value and value_b64, and one with NEITHER, are each rejected 400.
func TestHTTPKVPutValidation(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "PUT", "/v1/kv/k", `{"value":"a","value_b64":"YQ=="}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("both value+value_b64 = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	rec = do(t, h, "PUT", "/v1/kv/k", `{}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("neither value nor value_b64 = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	// A malformed base64 value is also a 400.
	rec = do(t, h, "PUT", "/v1/kv/k", `{"value_b64":"not!base64"}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("bad value_b64 = %d, want 400 (%s)", rec.Code, rec.Body)
	}
}

// TestHTTPKVDelete covers DELETE: an existing key reports deleted=true and is
// then a miss; a second delete reports deleted=false (idempotent).
func TestHTTPKVDelete(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	do(t, h, "PUT", "/v1/kv/gone", `{"value":"x"}`, nil)

	var del struct {
		Deleted bool `json:"deleted"`
	}
	rec := do(t, h, "DELETE", "/v1/kv/gone", "", &del)
	if rec.Code != http.StatusOK || !del.Deleted {
		t.Fatalf("delete = %d deleted=%v, want 200 true (%s)", rec.Code, del.Deleted, rec.Body)
	}

	var out kvGetResponse
	do(t, h, "GET", "/v1/kv/gone", "", &out)
	if out.Found {
		t.Fatalf("key still found after delete")
	}

	del.Deleted = true
	rec = do(t, h, "DELETE", "/v1/kv/gone", "", &del)
	if rec.Code != http.StatusOK || del.Deleted {
		t.Fatalf("second delete = %d deleted=%v, want 200 false (%s)", rec.Code, del.Deleted, rec.Body)
	}
}

// TestHTTPKVKeyCap proves a key over the 512-byte cap is rejected 400 on every
// KV verb, before any dispatch.
func TestHTTPKVKeyCap(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	long := "/v1/kv/" + strings.Repeat("a", maxKVKeyLen+1)
	for _, m := range []struct {
		method, body string
	}{
		{"GET", ""},
		{"PUT", `{"value":"x"}`},
		{"DELETE", ""},
	} {
		rec := do(t, h, m.method, long, m.body, nil)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("%s over-long key = %d, want 400 (%s)", m.method, rec.Code, rec.Body)
		}
	}

	// A key exactly at the cap is accepted (boundary is inclusive).
	atCap := "/v1/kv/" + strings.Repeat("a", maxKVKeyLen)
	rec := do(t, h, "PUT", atCap, `{"value":"x"}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("at-cap key PUT = %d, want 200 (%s)", rec.Code, rec.Body)
	}
}

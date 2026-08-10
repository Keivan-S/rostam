// SPDX-License-Identifier: Apache-2.0

package server

import (
	"errors"
	"path/filepath"
	"testing"

	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

type stubDispatcher struct{}

func (stubDispatcher) Call(_ string, _ []byte) ([]byte, error) { return []byte("ok"), nil }
func (stubDispatcher) LeaderAddr() string                      { return "" }

// countingDispatcher records whether Call was reached — used to prove a denied
// request never reaches the engine.
type countingDispatcher struct{ calls int }

func (d *countingDispatcher) Call(_ string, _ []byte) ([]byte, error) {
	d.calls++
	return []byte("ok"), nil
}
func (d *countingDispatcher) LeaderAddr() string { return "" }

// TestDispatchRBACScopes exercises the granular per-collection RBAC matrix at the
// TCP dispatch seam: a read:default/docs key's vector_search docs is allowed,
// vector_insert docs is denied (StatusUnauthorized) and never reaches the engine,
// vector_search other is denied; an admin:* key's create is allowed; a v1 frame
// (no token) is denied. Args are threaded into the authorizer so the target
// collection is derived from the decoded frame, not just the op name.
func TestDispatchRBACScopes(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	keyReg, err := vector.OpenKeyRegistry(filepath.Join(t.TempDir(), "keys.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range []vector.APIKey{
		{Token: "k_read", Tenant: "t", Scopes: []string{"read:default/docs"}},
		{Token: "k_admin", Tenant: "t", Scopes: []string{"*:*"}},
	} {
		if err := keyReg.AddKey(k); err != nil {
			t.Fatalf("AddKey(%q): %v", k.Token, err)
		}
	}
	auth := authz.NewRBACAuthenticator(keyReg, reg, "")

	searchDocs := ops.EncodeVectorSearchArgs("docs", 1, []float32{1, 2, 3})
	insertDocs := ops.EncodeVectorInsertArgsExt("docs", 1, []float32{1, 2, 3}, 0, nil, vector.SparseVector{})
	searchOther := ops.EncodeVectorSearchArgs("other", 1, []float32{1, 2, 3})
	createDocs := ops.EncodeCreateCollectionArgs("docs", vector.Config{Dim: 3})

	cases := []struct {
		name        string
		frame       []byte
		wantStatus  uint8
		wantReached bool
	}{
		{"read searches docs", EncodeRequestV2("k_read", "vector_search", searchDocs), StatusOK, true},
		{"read insert docs denied", EncodeRequestV2("k_read", "vector_insert", insertDocs), StatusUnauthorized, false},
		{"read search other denied", EncodeRequestV2("k_read", "vector_search", searchOther), StatusUnauthorized, false},
		{"admin create allowed", EncodeRequestV2("k_admin", "vector_create_collection", createDocs), StatusOK, true},
		{"v1 no-token denied", EncodeRequest("vector_search", searchDocs), StatusUnauthorized, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			disp := &countingDispatcher{}
			status, _ := dispatch(disp, tc.frame, auth, "", nil)
			if status != tc.wantStatus {
				t.Errorf("status = %d, want %d", status, tc.wantStatus)
			}
			reached := disp.calls > 0
			if reached != tc.wantReached {
				t.Errorf("reached engine = %v, want %v (a denied op must NOT reach the engine)", reached, tc.wantReached)
			}
		})
	}
}

func TestDispatchV1NoAuthenticator(t *testing.T) {
	body := EncodeRequest("ping", nil)
	status, payload := dispatch(stubDispatcher{}, body, nil, "", nil)
	if status != StatusOK {
		t.Fatalf("status = %d, want StatusOK", status)
	}
	if string(payload) != "ok" {
		t.Fatalf("payload = %q, want ok", payload)
	}
}

func TestDispatchV2WithApprovingAuthenticator(t *testing.T) {
	body := EncodeRequestV2("k_a1b2", "vector_insert", []byte("args"))
	var seenToken, seenOp string
	var seenArgs []byte
	auth := func(req authz.AuthRequest) bool {
		seenToken = req.Token
		seenOp = req.Op
		seenArgs = req.Args
		return true
	}
	status, _ := dispatch(stubDispatcher{}, body, auth, "", nil)
	if status != StatusOK {
		t.Fatalf("status = %d, want StatusOK", status)
	}
	if seenToken != "k_a1b2" {
		t.Errorf("Authenticator got token = %q, want k_a1b2", seenToken)
	}
	if seenOp != "vector_insert" {
		t.Errorf("Authenticator got op = %q, want vector_insert", seenOp)
	}
	if string(seenArgs) != "args" {
		t.Errorf("Authenticator got args = %q, want args", seenArgs)
	}
}

func TestDispatchV2WithRejectingAuthenticator(t *testing.T) {
	body := EncodeRequestV2("k_bad", "vector_insert", []byte("args"))
	auth := func(authz.AuthRequest) bool { return false }
	status, _ := dispatch(stubDispatcher{}, body, auth, "", nil)
	if status != StatusUnauthorized {
		t.Fatalf("status = %d, want StatusUnauthorized", status)
	}
}

func TestDispatchV1RejectedWhenAuthenticatorRequiresToken(t *testing.T) {
	body := EncodeRequest("vector_insert", []byte("args"))
	auth := func(req authz.AuthRequest) bool {
		return req.Token != "" // require non-empty token
	}
	status, _ := dispatch(stubDispatcher{}, body, auth, "", nil)
	if status != StatusUnauthorized {
		t.Fatalf("v1 frame with token-required auth: status = %d, want StatusUnauthorized", status)
	}
}

// TestDispatchClientCNThreadedToAuthRequest proves the verified mTLS CN supplied
// by the connection loop reaches AuthRequest.ClientCN, and that a token (when
// present) still takes precedence in the authorizer's own resolution order. Here
// we only assert the plumbing: dispatch forwards clientCN verbatim.
func TestDispatchClientCNThreadedToAuthRequest(t *testing.T) {
	body := EncodeRequest("vector_search", []byte("args")) // v1 frame: no token
	var seenToken, seenCN string
	auth := func(req authz.AuthRequest) bool {
		seenToken = req.Token
		seenCN = req.ClientCN
		return req.ClientCN == "svcA" // cert-only principal
	}
	status, _ := dispatch(stubDispatcher{}, body, auth, "svcA", nil)
	if status != StatusOK {
		t.Fatalf("status = %d, want StatusOK (cert CN authorized)", status)
	}
	if seenToken != "" {
		t.Errorf("token = %q, want empty (v1 frame carries no token)", seenToken)
	}
	if seenCN != "svcA" {
		t.Errorf("ClientCN = %q, want svcA (verified CN must be threaded in)", seenCN)
	}
}

func TestDispatchV2EmptyToken(t *testing.T) {
	// A v2 frame with no token (tokenLen=0) is wire-legal; the authenticator
	// sees an empty string and decides whether to accept.
	body := EncodeRequestV2("", "vector_search", nil)
	var seenToken string
	auth := func(req authz.AuthRequest) bool {
		seenToken = req.Token
		return false
	}
	status, _ := dispatch(stubDispatcher{}, body, auth, "", nil)
	if status != StatusUnauthorized {
		t.Fatalf("status = %d, want StatusUnauthorized", status)
	}
	if seenToken != "" {
		t.Errorf("empty-token v2 frame: authenticator got %q, want empty", seenToken)
	}
}

func TestDecodeRequestV2Roundtrip(t *testing.T) {
	body := EncodeRequestV2("k_test", "vector_search", []byte("query_args"))
	token, suffix, err := DecodeRequestV2(body)
	if err != nil {
		t.Fatalf("DecodeRequestV2: %v", err)
	}
	if token != "k_test" {
		t.Errorf("token = %q, want k_test", token)
	}
	op, args, err := DecodeRequest(suffix)
	if err != nil {
		t.Fatalf("DecodeRequest on suffix: %v", err)
	}
	if op != "vector_search" {
		t.Errorf("op = %q, want vector_search", op)
	}
	if string(args) != "query_args" {
		t.Errorf("args = %q, want query_args", args)
	}
}

func TestDecodeRequestV2RejectsNonV2(t *testing.T) {
	body := EncodeRequest("foo", nil)
	if _, _, err := DecodeRequestV2(body); err == nil {
		t.Fatal("DecodeRequestV2 on v1 frame: nil err, want error")
	}
}

func TestDecodeRequestV2Truncated(t *testing.T) {
	// Truncated v2 prefix (claims a long token but body cuts off).
	body := []byte{0x02, 0x10, 'a'} // tokenLen=16, only 1 byte after
	if _, _, err := DecodeRequestV2(body); !errors.Is(err, ErrFrameTruncated) {
		t.Fatalf("err = %v, want ErrFrameTruncated", err)
	}
}

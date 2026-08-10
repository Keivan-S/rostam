// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"net/http"

	"github.com/rostamlabs/rostam/ops"
)

// Online key-admin over HTTP (add/revoke/list API keys at RUNTIME, no restart).
// These endpoints lower their JSON request into the keys coordinator virtual-ops
// (__keys_add__/__keys_revoke__/__keys_list__) and dispatch through a.call, so the
// authorize gate — which classifies the three op names as admin — denies a
// non-admin caller (401) before the registry is touched. They mirror the alias
// management transport: a thin JSON<->ops-codec bridge over a.call.
//
// SECURITY:
//   - The raw token is a SECRET. DELETE takes it in the request BODY, never the
//     path, so it is not captured in access logs / proxies. add/revoke return only
//     a success ack and NEVER echo the token.
//   - GET returns ONLY the redacted view (fingerprint + tenant + scopes +
//     cert_cn). The op result codec has no token field, so no raw token can reach
//     the JSON by construction.
//
// Error mapping (via statusForError): dup token → 409, unknown token → 404, no
// registry wired → 412, bad input → 400, non-admin → 401.

// keysAddReq is the body for POST /v1/admin/keys: register a new API key.
type keysAddReq struct {
	Token  string   `json:"token"`
	Tenant string   `json:"tenant"`
	Scopes []string `json:"scopes"`
	CertCN string   `json:"cert_cn"`
}

// keysAdd registers a new API key on the live registry. token+tenant are required;
// the raw token is consumed by the registry and never echoed in the response.
func (a *api) keysAdd(w http.ResponseWriter, r *http.Request) {
	var req keysAddReq
	if !decodeBody(w, r, &req) {
		return
	}
	if req.Token == "" || req.Tenant == "" {
		writeError(w, http.StatusBadRequest, "token and tenant are required")
		return
	}
	args := ops.EncodeKeysAddArgs(ops.KeysAddArgs{
		Token:  req.Token,
		Tenant: req.Tenant,
		Scopes: req.Scopes,
		CertCN: req.CertCN,
	})
	if _, ok := a.call(w, r, ops.OpKeysAdd, args); !ok {
		return
	}
	// No token in the response — only a success signal.
	writeJSON(w, http.StatusCreated, map[string]bool{"added": true})
}

// keysRevokeReq is the body for DELETE /v1/admin/keys: revoke a key by its raw
// token. The token rides in the BODY (never the URL/path) so the secret is not
// logged.
type keysRevokeReq struct {
	Token string `json:"token"`
}

// keysRevoke removes an API key by its raw token. After this returns the token no
// longer authenticates. An unknown token → 404 (via statusForError).
func (a *api) keysRevoke(w http.ResponseWriter, r *http.Request) {
	var req keysRevokeReq
	if !decodeBody(w, r, &req) {
		return
	}
	if req.Token == "" {
		writeError(w, http.StatusBadRequest, "token is required")
		return
	}
	if _, ok := a.call(w, r, ops.OpKeysRevoke, ops.EncodeKeysRevokeArgs(req.Token)); !ok {
		return
	}
	// No token in the response — only a success signal.
	writeJSON(w, http.StatusOK, map[string]bool{"revoked": true})
}

// redactedKeyJSON is one token-free entry in the list response. It deliberately
// has NO token field so the marshaled JSON can never carry the secret.
type redactedKeyJSON struct {
	Fingerprint string   `json:"fingerprint"`
	Tenant      string   `json:"tenant"`
	Scopes      []string `json:"scopes"`
	CertCN      string   `json:"cert_cn"`
}

// keysList returns the REDACTED registry snapshot as
// {"keys":[{"fingerprint","tenant","scopes","cert_cn"}]}. The op result codec has
// no token field (the registry replaces each raw token with its fingerprint at the
// snapshot boundary), so no raw token can reach the JSON by construction.
func (a *api) keysList(w http.ResponseWriter, r *http.Request) {
	body, ok := a.call(w, r, ops.OpKeysList, ops.EncodeKeysListArgs())
	if !ok {
		return
	}
	entries, err := ops.DecodeKeysListResult(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	out := make([]redactedKeyJSON, 0, len(entries))
	for _, e := range entries {
		out = append(out, redactedKeyJSON{
			Fingerprint: e.Fingerprint,
			Tenant:      e.Tenant,
			Scopes:      e.Scopes,
			CertCN:      e.CertCN,
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"keys": out})
}

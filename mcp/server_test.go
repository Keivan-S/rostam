// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"encoding/json"
	"testing"
)

func TestInitializeHandshake(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	res := c.rpc("initialize", map[string]any{"protocolVersion": "2025-06-18"}, false)
	var init struct {
		ProtocolVersion string `json:"protocolVersion"`
		Capabilities    struct {
			Tools *struct{} `json:"tools"`
		} `json:"capabilities"`
		ServerInfo struct {
			Name string `json:"name"`
		} `json:"serverInfo"`
	}
	if err := json.Unmarshal(res, &init); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if init.ProtocolVersion != "2025-06-18" || init.Capabilities.Tools == nil || init.ServerInfo.Name != "rostam" {
		t.Fatalf("bad initialize result: %s", res)
	}
}

func TestRequestBeforeInitializeRejected(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	e := c.rpc("tools/list", nil, true)
	var re rpcError
	if err := json.Unmarshal(e, &re); err != nil || re.Code != codeInvalidRequest {
		t.Fatalf("want codeInvalidRequest, got %s", e)
	}
}

func TestUnknownMethodAndPing(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	c.rpc("ping", nil, false)
	e := c.rpc("no/such/method", nil, true)
	var re rpcError
	if err := json.Unmarshal(e, &re); err != nil || re.Code != codeMethodNotFound {
		t.Fatalf("want codeMethodNotFound, got %s", e)
	}
}

// TestInvalidEnvelopeIsInvalidRequest drives the envelope rules through the
// real dispatch path. The two that matter most: a 1.0 request must NOT get a
// successful 2.0 answer, and a request with no method must be Invalid Request
// (-32600), not method-not-found (-32601) — the latter would tell the client
// its (nonexistent) method name was the problem.
func TestInvalidEnvelopeIsInvalidRequest(t *testing.T) {
	for _, tc := range []struct {
		name   string
		line   string
		wantID string
	}{
		{"wrong version", `{"jsonrpc":"1.0","id":1,"method":"ping"}`, "1"},
		{"missing method", `{"jsonrpc":"2.0","id":2}`, "2"},
		{"empty method", `{"jsonrpc":"2.0","id":"s","method":""}`, `"s"`},
		{"missing version", `{"id":4,"method":"ping"}`, "4"},
		{"bad id shape", `{"jsonrpc":"2.0","id":{"a":1},"method":"ping"}`, "null"},
		{"bad version, no id", `{"jsonrpc":"1.0","method":"ping"}`, "null"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := startServer(t, Config{Store: newHeapStore(t)})
			c.initialize()
			resp := c.raw(tc.line)
			if _, ok := resp["result"]; ok {
				t.Fatalf("%s got a successful result: %s", tc.line, resp["result"])
			}
			var re rpcError
			if err := json.Unmarshal(resp["error"], &re); err != nil {
				t.Fatalf("decode error object: %v", err)
			}
			if re.Code != codeInvalidRequest {
				t.Fatalf("code = %d, want %d (%s)", re.Code, codeInvalidRequest, tc.line)
			}
			if got := string(resp["id"]); got != tc.wantID {
				t.Fatalf("id = %s, want %s", got, tc.wantID)
			}
		})
	}
}

func TestUnknownToolIsInvalidParams(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	e := c.rpc("tools/call", map[string]any{"name": "nope", "arguments": map[string]any{}}, true)
	var re rpcError
	if err := json.Unmarshal(e, &re); err != nil || re.Code != codeInvalidParams {
		t.Fatalf("want codeInvalidParams, got %s", e)
	}
}

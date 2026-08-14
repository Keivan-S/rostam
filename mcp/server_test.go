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

func TestUnknownToolIsInvalidParams(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	e := c.rpc("tools/call", map[string]any{"name": "nope", "arguments": map[string]any{}}, true)
	var re rpcError
	if err := json.Unmarshal(e, &re); err != nil || re.Code != codeInvalidParams {
		t.Fatalf("want codeInvalidParams, got %s", e)
	}
}

// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"encoding/json"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/internal/buildinfo"
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
			Name    string `json:"name"`
			Version string `json:"version"`
		} `json:"serverInfo"`
		Instructions string `json:"instructions"`
	}
	if err := json.Unmarshal(res, &init); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if init.ProtocolVersion != "2025-06-18" || init.Capabilities.Tools == nil || init.ServerInfo.Name != "rostam" {
		t.Fatalf("bad initialize result: %s", res)
	}
	// The instructions string is how the server teaches an agent WHEN to use
	// the memory tools; clients inject it into the model's context. Assert it's
	// present and carries the load-bearing doctrine, not just non-empty.
	for _, want := range []string{"recall", "remember", "namespace", "secret"} {
		if !strings.Contains(strings.ToLower(init.Instructions), want) {
			t.Errorf("initialize instructions missing %q; got %q", want, init.Instructions)
		}
	}
	// The version used to be a literal "0.1.0" written once and never touched,
	// so a v0.2.0 binary introduced itself to its client as 0.1.0 and every bug
	// report filed through an MCP client named the wrong release. This assertion
	// existed in spirit -- the test checked Name and stopped -- which is why the
	// stale value survived. Comparing against buildinfo rather than a literal is
	// the point: a hardcoded expectation here would rot the same way.
	if init.ServerInfo.Version != buildinfo.Version() {
		t.Errorf("serverInfo.version = %q, want the binary's own version %q",
			init.ServerInfo.Version, buildinfo.Version())
	}
	if init.ServerInfo.Version == "" {
		t.Error("serverInfo.version is empty; clients display this")
	}
}

func TestMemoryToolDescriptionsCarryUsageNudges(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	res := c.rpc("tools/list", nil, false)
	var out struct {
		Tools []struct {
			Name        string `json:"name"`
			Description string `json:"description"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(res, &out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	desc := map[string]string{}
	for _, tl := range out.Tools {
		desc[tl.Name] = strings.ToLower(tl.Description)
	}
	// recall should nudge "call at task start / before re-reading"; remember
	// should nudge "one self-contained fact / project namespace / no secrets".
	if d, ok := desc["recall"]; !ok || !strings.Contains(d, "start") || !strings.Contains(d, "namespace") {
		t.Errorf("recall description missing usage nudge: %q", desc["recall"])
	}
	if d, ok := desc["remember"]; !ok || !strings.Contains(d, "namespace") || !strings.Contains(d, "secret") {
		t.Errorf("remember description missing usage nudge: %q", desc["remember"])
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

// TestShutdownWaitsForTheCallInFlight is the property runMcpCmd relies on to
// close the Store on a signal: after Shutdown returns, no handler is running.
// A slow tool call is held open here so Shutdown has something to wait for; if
// it returned early, the store close that follows it in the real caller would
// land under a live handler.
func TestShutdownWaitsForTheCallInFlight(t *testing.T) {
	release := make(chan struct{})
	entered := make(chan struct{})
	finished := make(chan struct{})

	s, err := NewServer(context.Background(), Config{Store: newHeapStore(t)})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	s.register(toolDef{
		Name:        "slow",
		Description: "blocks until the test lets it go",
		InputSchema: map[string]any{"type": "object"},
		Handler: func(context.Context, json.RawMessage) (any, error) {
			close(entered)
			<-release
			close(finished)
			return map[string]any{"ok": true}, nil
		},
	})

	cr, cw := io.Pipe()
	sr, sw := io.Pipe()
	served := make(chan struct{})
	go func() { defer close(served); _ = s.Serve(cr, sw); _ = sw.Close() }()
	drained := make(chan struct{})
	go func() { defer close(drained); _, _ = io.Copy(io.Discard, sr) }()

	write := func(v any) {
		b, _ := json.Marshal(v)
		if _, err := cw.Write(append(b, '\n')); err != nil {
			t.Errorf("write: %v", err)
		}
	}
	write(map[string]any{"jsonrpc": "2.0", "id": 1, "method": "initialize"})
	write(map[string]any{"jsonrpc": "2.0", "id": 2, "method": "tools/call",
		"params": map[string]any{"name": "slow", "arguments": map[string]any{}}})
	<-entered

	shutdownReturned := make(chan struct{})
	go func() { defer close(shutdownReturned); s.Shutdown() }()
	select {
	case <-shutdownReturned:
		t.Fatal("Shutdown returned while a tool call was still running")
	case <-time.After(100 * time.Millisecond):
	}

	close(release)
	<-finished
	select {
	case <-shutdownReturned:
	case <-time.After(5 * time.Second):
		t.Fatal("Shutdown did not return after the tool call finished")
	}

	// And nothing new is dispatched: the next request ends Serve instead.
	write(map[string]any{"jsonrpc": "2.0", "id": 3, "method": "ping"})
	select {
	case <-served:
	case <-time.After(5 * time.Second):
		t.Fatal("Serve kept running after Shutdown")
	}
	_ = cw.Close()
	<-drained
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

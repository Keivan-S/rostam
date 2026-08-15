// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"testing"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
)

// newHeapStore returns an ephemeral single-node engine (no disk, no raft).
func newHeapStore(t *testing.T) rostam.Store {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("ops: %v", err)
	}
	st, err := rostam.NewDirect(rostam.DirectConfig{Ops: reg})
	if err != nil {
		t.Fatalf("NewDirect: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
	return st
}

// testClient drives a Server through pipes like an MCP client would.
type testClient struct {
	t   *testing.T
	in  io.WriteCloser
	out *bufio.Scanner
	id  int
}

func startServer(t *testing.T, cfg Config) *testClient {
	t.Helper()
	c, _ := startServerStop(t, cfg)
	return c
}

// startServerStop is startServer plus an explicit stop func: it closes the
// client's write end (ending Serve) and blocks until the server goroutine has
// returned. A test that closes and reopens a data dir needs that ordering —
// the store must not be closed while a handler is still running against it.
func startServerStop(t *testing.T, cfg Config) (*testClient, func()) {
	t.Helper()
	s, err := NewServer(context.Background(), cfg)
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	cr, cw := io.Pipe() // client -> server
	sr, sw := io.Pipe() // server -> client
	done := make(chan struct{})
	go func() { defer close(done); _ = s.Serve(cr, sw); _ = sw.Close() }()
	t.Cleanup(func() { _ = cw.Close() })
	sc := bufio.NewScanner(sr)
	sc.Buffer(make([]byte, 64<<10), maxLine)
	stop := func() {
		_ = cw.Close() // io.PipeWriter.Close is idempotent
		<-done
	}
	return &testClient{t: t, in: cw, out: sc}, stop
}

// rpc sends a request and returns the raw result; fails the test on a
// JSON-RPC error unless wantErr is true (then it returns the error object).
func (c *testClient) rpc(method string, params any, wantErr bool) json.RawMessage {
	c.t.Helper()
	c.id++
	req := map[string]any{"jsonrpc": "2.0", "id": c.id, "method": method}
	if params != nil {
		req["params"] = params
	}
	b, _ := json.Marshal(req)
	if _, err := c.in.Write(append(b, '\n')); err != nil {
		c.t.Fatalf("write: %v", err)
	}
	if !c.out.Scan() {
		c.t.Fatalf("no response to %s: %v", method, c.out.Err())
	}
	var resp struct {
		Result json.RawMessage `json:"result"`
		Error  json.RawMessage `json:"error"`
	}
	if err := json.Unmarshal(c.out.Bytes(), &resp); err != nil {
		c.t.Fatalf("bad response line %q: %v", c.out.Text(), err)
	}
	if wantErr {
		if resp.Error == nil {
			c.t.Fatalf("%s: expected JSON-RPC error, got result %s", method, resp.Result)
		}
		return resp.Error
	}
	if resp.Error != nil {
		c.t.Fatalf("%s: unexpected error %s", method, resp.Error)
	}
	return resp.Result
}

// raw writes one already-encoded line and returns the whole decoded response.
// rpc builds a well-formed 2.0 envelope by construction, so an envelope-level
// test (bad version, missing method, malformed id) has to bypass it.
func (c *testClient) raw(line string) map[string]json.RawMessage {
	c.t.Helper()
	if _, err := c.in.Write([]byte(line + "\n")); err != nil {
		c.t.Fatalf("write: %v", err)
	}
	if !c.out.Scan() {
		c.t.Fatalf("no response to %q: %v", line, c.out.Err())
	}
	var resp map[string]json.RawMessage
	if err := json.Unmarshal(c.out.Bytes(), &resp); err != nil {
		c.t.Fatalf("bad response line %q: %v", c.out.Text(), err)
	}
	return resp
}

func (c *testClient) initialize() {
	c.t.Helper()
	c.rpc("initialize", map[string]any{"protocolVersion": "2025-06-18"}, false)
	b, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "method": "notifications/initialized"})
	if _, err := c.in.Write(append(b, '\n')); err != nil {
		c.t.Fatalf("write: %v", err)
	}
}

// callTool invokes tools/call and decodes the text payload into out (a
// pointer). Fails the test if the tool returned isError=true, unless
// wantToolErr — then it returns the error text and out is ignored.
func (c *testClient) callTool(name string, args any, out any, wantToolErr bool) string {
	c.t.Helper()
	res := c.rpc("tools/call", map[string]any{"name": name, "arguments": args}, false)
	var body struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		IsError bool `json:"isError"`
	}
	if err := json.Unmarshal(res, &body); err != nil {
		c.t.Fatalf("tool result decode: %v", err)
	}
	if len(body.Content) != 1 {
		c.t.Fatalf("want 1 content block, got %d", len(body.Content))
	}
	if wantToolErr {
		if !body.IsError {
			c.t.Fatalf("%s: expected tool error, got %q", name, body.Content[0].Text)
		}
		return body.Content[0].Text
	}
	if body.IsError {
		c.t.Fatalf("%s: tool error: %s", name, body.Content[0].Text)
	}
	if out != nil {
		if err := json.Unmarshal([]byte(body.Content[0].Text), out); err != nil {
			c.t.Fatalf("%s: payload decode %q: %v", name, body.Content[0].Text, err)
		}
	}
	return body.Content[0].Text
}

// toolNames returns the tools/list names in order.
func (c *testClient) toolNames() []string {
	var lst struct {
		Tools []struct {
			Name string `json:"name"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(c.rpc("tools/list", nil, false), &lst); err != nil {
		c.t.Fatalf("tools/list decode: %v", err)
	}
	names := make([]string, len(lst.Tools))
	for i, tl := range lst.Tools {
		names[i] = tl.Name
	}
	return names
}

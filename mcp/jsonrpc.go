// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"
)

// JSON-RPC 2.0 error codes used by the server. Tool-level failures do NOT use
// these — they are MCP tool results with isError=true; protocol errors are
// reserved for malformed or unroutable requests.
const (
	codeParse          = -32700
	codeInvalidRequest = -32600
	codeMethodNotFound = -32601
	codeInvalidParams  = -32602
)

// maxLine bounds a single wire message, matching the TCP protocol's
// MaxFrameSize so an MCP client cannot make the server buffer unboundedly.
const maxLine = 16 << 20

// errLine marks a line that was not valid JSON-RPC. The read loop reports it
// per-line so the stream survives one garbage message.
var errLine = errors.New("mcp: malformed message")

// request is an incoming JSON-RPC request or notification (ID nil).
type request struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

// validate checks the JSON-RPC 2.0 envelope itself, separately from routing.
// A decode that merely succeeded is not a valid request: {"jsonrpc":"1.0"} and
// an object with no "method" both unmarshal cleanly into this struct, and
// answering the first with a 2.0 result or routing the second to
// method-not-found reports the wrong thing. Both are Invalid Request.
func (r *request) validate() error {
	if r.JSONRPC != "2.0" {
		return fmt.Errorf("jsonrpc must be \"2.0\", got %q", r.JSONRPC)
	}
	if r.Method == "" {
		return errors.New("method is required")
	}
	if !validID(r.ID) {
		return errors.New("id must be a string, a number, or null")
	}
	return nil
}

// validID reports whether an id is one of the shapes JSON-RPC 2.0 allows: a
// string, a number, or null. A nil RawMessage is an absent id (a
// notification), which is valid on the wire.
func validID(id json.RawMessage) bool {
	if id == nil {
		return true
	}
	var v any
	if err := json.Unmarshal(id, &v); err != nil {
		return false
	}
	switch v.(type) {
	case string, float64, nil:
		return true
	}
	return false
}

// errorID is the id to answer an Invalid Request with: the client's own id
// when it had a usable one, JSON null when it was absent or malformed. The
// spec requires the id be echoed where it can be, so a client with several
// requests in flight can still match the failure to the one that caused it.
func errorID(id json.RawMessage) json.RawMessage {
	if id != nil && validID(id) {
		return id
	}
	return json.RawMessage("null")
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type response struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  any             `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

// conn frames newline-delimited JSON-RPC messages over a reader/writer pair.
// Writes are mutex-serialized; reads are single-consumer (the Serve loop).
type conn struct {
	sc *bufio.Scanner
	w  io.Writer
	mu sync.Mutex
}

func newConn(r io.Reader, w io.Writer) *conn {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64<<10), maxLine)
	return &conn{sc: sc, w: w}
}

// next returns the next message. Blank lines are skipped; a non-JSON line
// returns an error wrapping errLine (recoverable); end of stream is io.EOF.
func (c *conn) next() (*request, error) {
	for c.sc.Scan() {
		line := c.sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var req request
		if err := json.Unmarshal(line, &req); err != nil {
			return nil, fmt.Errorf("%w: %v", errLine, err)
		}
		return &req, nil
	}
	if err := c.sc.Err(); err != nil {
		return nil, err
	}
	return nil, io.EOF
}

func (c *conn) reply(id json.RawMessage, result any) error {
	return c.send(response{JSONRPC: "2.0", ID: id, Result: result})
}

func (c *conn) replyError(id json.RawMessage, code int, msg string) error {
	return c.send(response{JSONRPC: "2.0", ID: id, Error: &rpcError{Code: code, Message: msg}})
}

func (c *conn) send(resp response) error {
	b, err := json.Marshal(resp)
	if err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, err := c.w.Write(b); err != nil {
		return err
	}
	_, err = c.w.Write([]byte{'\n'})
	return err
}

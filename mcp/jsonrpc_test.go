// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
)

func TestConnReadsRequestsAndSkipsBlankLines(t *testing.T) {
	in := strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"ping"}` + "\n\n" +
		`{"jsonrpc":"2.0","method":"notifications/initialized"}` + "\n")
	c := newConn(in, io.Discard)

	r1, err := c.next()
	if err != nil {
		t.Fatalf("next: %v", err)
	}
	if r1.Method != "ping" || string(r1.ID) != "1" {
		t.Fatalf("got method=%q id=%s", r1.Method, r1.ID)
	}
	r2, err := c.next()
	if err != nil {
		t.Fatalf("next: %v", err)
	}
	if r2.Method != "notifications/initialized" || r2.ID != nil {
		t.Fatalf("notification parsed wrong: %+v", r2)
	}
	if _, err := c.next(); !errors.Is(err, io.EOF) {
		t.Fatalf("want EOF, got %v", err)
	}
}

func TestConnMalformedLineIsErrLineNotFatal(t *testing.T) {
	in := strings.NewReader("{not json\n" + `{"jsonrpc":"2.0","id":2,"method":"ping"}` + "\n")
	c := newConn(in, io.Discard)
	if _, err := c.next(); !errors.Is(err, errLine) {
		t.Fatalf("want errLine, got %v", err)
	}
	r, err := c.next() // the stream continues after a bad line
	if err != nil || r.Method != "ping" {
		t.Fatalf("stream did not recover: %v %+v", err, r)
	}
}

func TestConnReplyWritesOneLine(t *testing.T) {
	var out strings.Builder
	c := newConn(strings.NewReader(""), &out)
	if err := c.reply(json.RawMessage("7"), map[string]string{"ok": "yes"}); err != nil {
		t.Fatalf("reply: %v", err)
	}
	line := out.String()
	if !strings.HasSuffix(line, "\n") || strings.Count(line, "\n") != 1 {
		t.Fatalf("want exactly one newline-terminated line, got %q", line)
	}
	var resp struct {
		JSONRPC string            `json:"jsonrpc"`
		ID      json.RawMessage   `json:"id"`
		Result  map[string]string `json:"result"`
	}
	if err := json.Unmarshal([]byte(line), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if resp.JSONRPC != "2.0" || string(resp.ID) != "7" || resp.Result["ok"] != "yes" {
		t.Fatalf("bad response: %+v", resp)
	}
}

func TestConnReplyError(t *testing.T) {
	var out strings.Builder
	c := newConn(strings.NewReader(""), &out)
	if err := c.replyError(json.RawMessage("3"), codeMethodNotFound, "no such method"); err != nil {
		t.Fatalf("replyError: %v", err)
	}
	var resp struct {
		Error *rpcError `json:"error"`
	}
	if err := json.Unmarshal([]byte(out.String()), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if resp.Error == nil || resp.Error.Code != codeMethodNotFound {
		t.Fatalf("bad error: %+v", resp.Error)
	}
}

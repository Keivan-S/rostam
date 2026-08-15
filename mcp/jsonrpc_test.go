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

// TestRequestValidateEnvelope pins the envelope rules a successful decode does
// NOT give you: the version string has to be exactly "2.0", a method has to be
// present, and the id has to be one of the shapes the spec allows.
func TestRequestValidateEnvelope(t *testing.T) {
	for _, tc := range []struct {
		name string
		line string
		ok   bool
	}{
		{"good", `{"jsonrpc":"2.0","id":1,"method":"ping"}`, true},
		{"good string id", `{"jsonrpc":"2.0","id":"abc","method":"ping"}`, true},
		{"good null id", `{"jsonrpc":"2.0","id":null,"method":"ping"}`, true},
		{"notification", `{"jsonrpc":"2.0","method":"notifications/initialized"}`, true},
		{"wrong version", `{"jsonrpc":"1.0","id":1,"method":"ping"}`, false},
		{"missing version", `{"id":1,"method":"ping"}`, false},
		{"missing method", `{"jsonrpc":"2.0","id":1}`, false},
		{"empty method", `{"jsonrpc":"2.0","id":1,"method":""}`, false},
		{"object id", `{"jsonrpc":"2.0","id":{"a":1},"method":"ping"}`, false},
		{"array id", `{"jsonrpc":"2.0","id":[1],"method":"ping"}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var req request
			if err := json.Unmarshal([]byte(tc.line), &req); err != nil {
				t.Fatalf("decode: %v", err)
			}
			err := req.validate()
			if tc.ok && err != nil {
				t.Fatalf("validate(%s) = %v, want nil", tc.line, err)
			}
			if !tc.ok && err == nil {
				t.Fatalf("validate(%s) = nil, want an Invalid Request", tc.line)
			}
		})
	}
}

// TestErrorIDNormalizes covers the id echoed back on an Invalid Request:
// present and well-formed ids are preserved so a client can match the failure
// to the request; absent or malformed ones become null.
func TestErrorIDNormalizes(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{`7`, `7`},
		{`"abc"`, `"abc"`},
		{`null`, `null`},
		{`{"a":1}`, `null`},
	} {
		if got := string(errorID(json.RawMessage(tc.in))); got != tc.want {
			t.Fatalf("errorID(%s) = %s, want %s", tc.in, got, tc.want)
		}
	}
	if got := string(errorID(nil)); got != "null" {
		t.Fatalf("errorID(absent) = %s, want null", got)
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

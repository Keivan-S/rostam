// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/rlog"
	"github.com/rostamlabs/rostam/vector"
)

// tcpRawToken is a stand-in bearer secret; no TCP access line may contain it.
const tcpRawToken = "tcp-super-secret-token-DO-NOT-LOG"

// TestTCPDispatchAccessLogRedacts proves the TCP ingress path, with the access
// log enabled, emits exactly one access line per dispatched frame that (a) is
// valid JSON carrying a request-id, the op, an "ok" status, latency and bytes,
// (b) records the principal as the token FINGERPRINT, and (c) NEVER contains the
// raw bearer token carried in the v2 frame.
func TestTCPDispatchAccessLogRedacts(t *testing.T) {
	var buf bytes.Buffer
	al, err := rlog.NewTo(&buf, "json")
	if err != nil {
		t.Fatalf("rlog.NewTo: %v", err)
	}

	frame := EncodeRequestV2(tcpRawToken, "ping", nil)
	status, payload := dispatch(stubDispatcher{}, frame, nil, "", al)
	if status != StatusOK || string(payload) != "ok" {
		t.Fatalf("dispatch = %d,%q; want StatusOK,ok", status, payload)
	}

	line := strings.TrimSpace(buf.String())
	if strings.Contains(line, tcpRawToken) {
		t.Fatalf("SECRET LEAK: TCP access line contains the raw token: %s", line)
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(line), &m); err != nil {
		t.Fatalf("access line not valid JSON: %v\n%s", err, line)
	}
	if m["transport"] != "tcp" {
		t.Errorf("transport = %v, want tcp", m["transport"])
	}
	if m["op"] != "ping" {
		t.Errorf("op = %v, want ping", m["op"])
	}
	if m["status"] != "ok" {
		t.Errorf("status = %v, want ok", m["status"])
	}
	if m["principal"] != "token:"+vector.TokenFingerprint(tcpRawToken) {
		t.Errorf("principal = %v, want the token fingerprint", m["principal"])
	}
	if id, _ := m["request_id"].(string); id == "" {
		t.Error("TCP access line has empty request_id")
	}
	if _, ok := m["latency_ms"]; !ok {
		t.Error("TCP access line missing latency_ms")
	}
}

// TestTCPDispatchNoAccessLogNoOutput proves that with the access log OFF (nil),
// the dispatch path emits NO access line — the default zero-cost posture — while
// still returning the correct result.
func TestTCPDispatchNoAccessLogNoOutput(t *testing.T) {
	// Capture the process default logger so we can prove the disabled path writes
	// nothing (no access line, no token) through any slog route.
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&buf, nil)))
	defer slog.SetDefault(prev)

	frame := EncodeRequestV2(tcpRawToken, "ping", nil)
	status, payload := dispatch(stubDispatcher{}, frame, nil, "", nil)
	if status != StatusOK || string(payload) != "ok" {
		t.Fatalf("dispatch = %d,%q; want StatusOK,ok", status, payload)
	}
	if buf.Len() != 0 {
		t.Errorf("disabled dispatch emitted log output: %q", buf.String())
	}
}

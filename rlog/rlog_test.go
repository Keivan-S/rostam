// SPDX-License-Identifier: Apache-2.0

package rlog

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"
)

// TestJSONHandlerStructuredKeys proves the JSON handler emits one valid JSON
// object per log call carrying the expected structured keys — the format a
// production log aggregator consumes.
func TestJSONHandlerStructuredKeys(t *testing.T) {
	var buf bytes.Buffer
	h, err := NewHandler(&buf, "json", slog.LevelInfo)
	if err != nil {
		t.Fatalf("NewHandler: %v", err)
	}
	logger := slog.New(h)
	logger.Warn("forcing reject-writes for cross-replica determinism", "component", "shard", "shard", 3)

	line := strings.TrimSpace(buf.String())
	var rec map[string]any
	if err := json.Unmarshal([]byte(line), &rec); err != nil {
		t.Fatalf("emitted line is not valid JSON: %v\nline=%q", err, line)
	}
	for _, k := range []string{"time", "level", "msg", "component", "shard"} {
		if _, ok := rec[k]; !ok {
			t.Errorf("JSON record missing key %q: %v", k, rec)
		}
	}
	if rec["level"] != "WARN" {
		t.Errorf("level = %v, want WARN", rec["level"])
	}
	if rec["msg"] != "forcing reject-writes for cross-replica determinism" {
		t.Errorf("msg = %v", rec["msg"])
	}
	// slog encodes an integer attr as a JSON number.
	if got, ok := rec["shard"].(float64); !ok || got != 3 {
		t.Errorf("shard = %v (%T), want 3", rec["shard"], rec["shard"])
	}
}

// TestTextHandlerReadsCleanly proves the default text handler emits a leading
// timestamp, a level, the message, and key=value fields — familiar to an
// operator upgrading from the historical stderr logs.
func TestTextHandlerReadsCleanly(t *testing.T) {
	var buf bytes.Buffer
	h, err := NewHandler(&buf, "text", slog.LevelInfo)
	if err != nil {
		t.Fatalf("NewHandler: %v", err)
	}
	slog.New(h).Info("serving", "proto", "http", "addr", ":8080")
	out := buf.String()
	for _, want := range []string{"time=", "level=INFO", `msg=serving`, "proto=http", "addr=:8080"} {
		if !strings.Contains(out, want) {
			t.Errorf("text output missing %q: %s", want, out)
		}
	}
}

func TestParseLevel(t *testing.T) {
	cases := map[string]slog.Level{
		"": slog.LevelInfo, "info": slog.LevelInfo, "INFO": slog.LevelInfo,
		"debug": slog.LevelDebug, "warn": slog.LevelWarn, "warning": slog.LevelWarn,
		"error": slog.LevelError,
	}
	for in, want := range cases {
		got, err := ParseLevel(in)
		if err != nil || got != want {
			t.Errorf("ParseLevel(%q) = %v, %v; want %v", in, got, err, want)
		}
	}
	if _, err := ParseLevel("nope"); err == nil {
		t.Error("ParseLevel(\"nope\") should error")
	}
}

func TestNewHandlerRejectsBadFormat(t *testing.T) {
	var buf bytes.Buffer
	if _, err := NewHandler(&buf, "yaml", slog.LevelInfo); err == nil {
		t.Error("NewHandler with unknown format should error")
	}
}

// TestEnsureRequestID proves an id is generated when none is supplied and reused
// verbatim when the caller passes an inbound one, and that the returned context
// carries it.
func TestEnsureRequestID(t *testing.T) {
	// Generated when absent.
	ctx, id := EnsureRequestID(context.Background(), "")
	if id == "" {
		t.Fatal("EnsureRequestID generated an empty id")
	}
	if RequestID(ctx) != id {
		t.Errorf("context id = %q, want %q", RequestID(ctx), id)
	}

	// Reused when supplied.
	ctx2, id2 := EnsureRequestID(context.Background(), "client-abc")
	if id2 != "client-abc" {
		t.Errorf("inbound id = %q, want reuse of client-abc", id2)
	}
	if RequestID(ctx2) != "client-abc" {
		t.Errorf("context id = %q, want client-abc", RequestID(ctx2))
	}

	// Distinct generated ids across calls.
	_, a := EnsureRequestID(context.Background(), "")
	_, b := EnsureRequestID(context.Background(), "")
	if a == b {
		t.Errorf("generated ids collided: %q", a)
	}

	// No id in a bare context.
	if RequestID(context.Background()) != "" {
		t.Error("bare context should carry no request id")
	}
}

// TestEnsureRequestIDRejectsUnsafeInbound: an attacker-controlled X-Request-Id
// that is over-long or contains control/non-printable characters is rejected in
// favor of a fresh generated id (it is logged verbatim + echoed to the response).
func TestEnsureRequestIDRejectsUnsafeInbound(t *testing.T) {
	safe := "trace-abc123_DEF.456"
	if _, got := EnsureRequestID(context.Background(), safe); got != safe {
		t.Fatalf("safe inbound id = %q, want reused %q", got, safe)
	}
	for name, bad := range map[string]string{
		"newline":  "abc\ndef",
		"carriage": "abc\rdef",
		"tab":      "abc\tdef",
		"nonascii": "abcédef",
		"nul":      "abc\x00def",
		"too-long": strings.Repeat("x", maxInboundReqID+1),
	} {
		_, got := EnsureRequestID(context.Background(), bad)
		if got == bad {
			t.Errorf("%s: unsafe inbound id was reused verbatim (%q); want a fresh generated id", name, got)
		}
		if got == "" {
			t.Errorf("%s: got empty id, want a fresh generated one", name)
		}
	}
}

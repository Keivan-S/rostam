// SPDX-License-Identifier: Apache-2.0

// Package rlog is Rostam's small logging + request-observability layer over the
// stdlib log/slog. It owns three things, all leaf-level (no engine imports) so
// every transport and subsystem can depend on it without a cycle:
//
//   - Setup: the single point that builds the process-wide *slog.Logger from the
//     -log-format (text|json) and -log-level flags and installs it as
//     slog.Default(). Every migrated call site then logs through the package
//     slog.Info/Warn/Error functions, so there is one handler, one format, one
//     level for the whole server.
//
//   - Request IDs (reqid.go): a cheap per-request correlation id generated at each
//     transport ingress (or reused from an inbound X-Request-Id header / gRPC
//     metadata), carried in context.Context.
//
//   - The OPT-IN access log (access.go, http.go, grpc.go): one structured slog
//     line per request when -access-log is set — request-id, transport, op,
//     status, latency, a REDACTED principal (the audit token fingerprint, never
//     the raw token), and bytes. A nil *AccessLog is the default and adds zero
//     hot-path cost.
package rlog

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"
)

// ParseLevel maps a -log-level string ("debug"|"info"|"warn"|"error", any case)
// onto a slog.Level. An empty string defaults to info; an unrecognized value is
// an error so a typo in a unit file fails loud at startup rather than silently
// logging at the wrong level.
func ParseLevel(s string) (slog.Level, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "info":
		return slog.LevelInfo, nil
	case "debug":
		return slog.LevelDebug, nil
	case "warn", "warning":
		return slog.LevelWarn, nil
	case "error":
		return slog.LevelError, nil
	default:
		return 0, fmt.Errorf("rlog: unknown log level %q (want debug|info|warn|error)", s)
	}
}

// NewHandler builds a slog.Handler for the given format ("text" or "json") at
// the given minimum level, writing to w. "text" (the default) reads cleanly like
// the server's historical stderr logs — a leading timestamp, a level, the
// message, then any structured key=value fields — so an operator upgrading to
// this build sees familiar output. "json" emits one JSON object per line for
// production log aggregation. An unrecognized format is an error (fail loud).
func NewHandler(w io.Writer, format string, level slog.Level) (slog.Handler, error) {
	opts := &slog.HandlerOptions{Level: level}
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "", "text":
		return slog.NewTextHandler(w, opts), nil
	case "json":
		return slog.NewJSONHandler(w, opts), nil
	default:
		return nil, fmt.Errorf("rlog: unknown log format %q (want text|json)", format)
	}
}

// Setup builds the process default logger from format+level and installs it via
// slog.SetDefault, so every package-level slog call (slog.Info/Warn/Error, and
// the migrated call sites) routes through this one handler. It returns the
// logger for callers that want an explicit handle. Output goes to os.Stderr,
// matching the historical stdlib-log destination (stdout stays reserved for
// program data). Call once, early in main, before anything logs.
func Setup(format, level string) (*slog.Logger, error) {
	lvl, err := ParseLevel(level)
	if err != nil {
		return nil, err
	}
	h, err := NewHandler(os.Stderr, format, lvl)
	if err != nil {
		return nil, err
	}
	logger := slog.New(h)
	slog.SetDefault(logger)
	return logger, nil
}

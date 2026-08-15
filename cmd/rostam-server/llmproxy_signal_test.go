// SPDX-License-Identifier: Apache-2.0
//go:build unix

package main

import (
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// freeListenAddr picks an available loopback port for a child llm-proxy to
// bind, closing the probe listener immediately so the child can claim it.
// The gap between close and the child's own bind is a race in principle, but
// negligible in practice for a short-lived local test.
func freeListenAddr(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("finding a free port: %v", err)
	}
	addr := ln.Addr().String()
	_ = ln.Close()
	return addr
}

// startLlmProxy starts a `rostam-server llm-proxy` child process against
// dataDir and listen, capturing its stderr for diagnostics on failure.
// syncBuffer and buildServer are shared with mcp_signal_test.go (same
// package, same build tag).
func startLlmProxy(t *testing.T, bin, dataDir, listen string) (*exec.Cmd, *syncBuffer) {
	t.Helper()
	cmd := exec.Command(bin, "llm-proxy", "-data", dataDir, "-listen", listen)
	// Pin exact mode explicitly, same reasoning as startMcp: a hosted embedder
	// configured in the developer's own environment must not change the mode
	// this test runs in.
	cmd.Env = append(os.Environ(), "ROSTAM_EMBED_ENDPOINT=")
	logs := &syncBuffer{}
	cmd.Stderr = logs
	if err := cmd.Start(); err != nil {
		t.Fatalf("starting %s: %v", bin, err)
	}
	// Backstop: a child left running would hold the data dir lock and wedge
	// the rest of the test. Kill is a no-op once the process has exited.
	t.Cleanup(func() { _ = cmd.Process.Kill() })
	return cmd, logs
}

// waitForListen polls GET /stats until the proxy answers or the deadline
// passes, so the test never sends a signal before the server has bound its
// listener.
func waitForListen(t *testing.T, listen string, logs *syncBuffer) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	url := "http://" + listen + "/stats"
	for time.Now().Before(deadline) {
		resp, err := http.Get(url)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("llm-proxy never answered %s within the deadline\nserver log:\n%s", url, logs)
}

// TestLlmProxySIGTERMExitsCleanlyAndReleasesLock is the shutdown contract for
// `rostam-server llm-proxy`, mirroring TestMcpSIGTERMFlushesMemory: the
// default disposition for SIGINT/SIGTERM is to kill the process immediately,
// before store.Close and the -data lock release run. Unlike the MCP server's
// memory collections, the llm-proxy cache is created non-persistent
// regardless of -data (see doc.go's "Honest limits"), so there is nothing to
// flush across a restart — what has to hold is a clean process exit and the
// data-dir lock actually being released, which this proves by starting a
// second process against the same -data directory immediately afterward and
// requiring it to come up rather than being refused with "another process is
// using this data directory".
//
// This has to be a real subprocess: the behavior under test is what the OS
// does to a process that has not installed a handler.
func TestLlmProxySIGTERMExitsCleanlyAndReleasesLock(t *testing.T) {
	if testing.Short() {
		t.Skip("compiles the rostam-server binary and runs it as a subprocess")
	}
	bin := buildServer(t)
	dataDir := filepath.Join(t.TempDir(), "llmcache")
	listen := freeListenAddr(t)

	cmd, logs := startLlmProxy(t, bin, dataDir, listen)
	waitForListen(t, listen, logs)

	if err := cmd.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatalf("SIGTERM: %v", err)
	}
	waited := make(chan error, 1)
	go func() { waited <- cmd.Wait() }()
	select {
	case err := <-waited:
		if err != nil {
			t.Fatalf("the server should exit cleanly on SIGTERM, got %v\nserver log:\n%s", err, logs)
		}
	case <-time.After(30 * time.Second):
		t.Fatalf("the server did not exit within 30s of SIGTERM\nserver log:\n%s", logs)
	}

	// Second process over the same data directory, started right after the
	// first exited: a leaked lock would refuse this outright, and
	// waitForListen's own deadline would catch a process that never comes up.
	listen2 := freeListenAddr(t)
	cmd2, logs2 := startLlmProxy(t, bin, dataDir, listen2)
	waitForListen(t, listen2, logs2)
	_ = cmd2.Process.Kill()
	_ = cmd2.Wait()
}

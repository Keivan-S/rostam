// SPDX-License-Identifier: Apache-2.0
//go:build unix

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"testing"
	"time"
)

// syncBuffer is a mutex-protected byte buffer. os/exec copies the child's
// stderr into it from an internal goroutine that runs until Wait returns,
// while the test reads it (via String, through %s) while the child may still
// be running -- a plain strings.Builder or bytes.Buffer would race there.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// mcpProc is one `rostam-server mcp` child process driven over its stdio wire,
// the way an MCP client drives it.
type mcpProc struct {
	t    *testing.T
	cmd  *exec.Cmd
	in   io.WriteCloser
	out  *bufio.Scanner
	logs *syncBuffer
	id   int
}

// buildServer compiles the rostam-server binary once for the test. The signal
// path cannot be exercised in-process: the whole point is what the OS does to a
// real process, and Go cannot deliver a SIGTERM to itself and still observe the
// default-disposition behavior this guards against.
func buildServer(t *testing.T) string {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "rostam-server")
	build := exec.Command("go", "build", "-o", bin, ".")
	build.Dir = "."
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building rostam-server: %v\n%s", err, out)
	}
	return bin
}

func startMcp(t *testing.T, bin, dataDir string) *mcpProc {
	t.Helper()
	cmd := exec.Command(bin, "mcp", "-data", dataDir)
	// An embedder configured in the developer's own environment would change
	// the mode this test runs in (and the stored embedder identity), so the
	// child is pinned to BM25-only explicitly.
	cmd.Env = append(os.Environ(), "ROSTAM_EMBED_ENDPOINT=")
	in, err := cmd.StdinPipe()
	if err != nil {
		t.Fatalf("stdin pipe: %v", err)
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("stdout pipe: %v", err)
	}
	logs := &syncBuffer{}
	cmd.Stderr = logs
	if err := cmd.Start(); err != nil {
		t.Fatalf("starting %s: %v", bin, err)
	}
	p := &mcpProc{t: t, cmd: cmd, in: in, out: bufio.NewScanner(out), logs: logs}
	p.out.Buffer(make([]byte, 64<<10), 16<<20)
	// Backstop: a child left running would hold the data dir lock and wedge the
	// rest of the test. Kill is a no-op once the process has exited.
	t.Cleanup(func() { _ = cmd.Process.Kill() })
	return p
}

// call sends one request and returns the decoded result, failing the test if
// the server answers with a JSON-RPC error or says nothing within the deadline.
func (p *mcpProc) call(method string, params any) json.RawMessage {
	p.t.Helper()
	p.id++
	req := map[string]any{"jsonrpc": "2.0", "id": p.id, "method": method}
	if params != nil {
		req["params"] = params
	}
	b, _ := json.Marshal(req)
	if _, err := p.in.Write(append(b, '\n')); err != nil {
		p.t.Fatalf("write %s: %v\nserver log:\n%s", method, err, p.logs)
	}

	type line struct {
		text string
		ok   bool
	}
	got := make(chan line, 1)
	go func() {
		if p.out.Scan() {
			got <- line{p.out.Text(), true}
			return
		}
		got <- line{}
	}()
	select {
	case l := <-got:
		if !l.ok {
			p.t.Fatalf("no response to %s (server gone?)\nserver log:\n%s", method, p.logs)
		}
		var resp struct {
			Result json.RawMessage `json:"result"`
			Error  json.RawMessage `json:"error"`
		}
		if err := json.Unmarshal([]byte(l.text), &resp); err != nil {
			p.t.Fatalf("bad response %q: %v", l.text, err)
		}
		if resp.Error != nil {
			p.t.Fatalf("%s: %s", method, resp.Error)
		}
		return resp.Result
	case <-time.After(30 * time.Second):
		p.t.Fatalf("timed out waiting for a response to %s\nserver log:\n%s", method, p.logs)
		return nil
	}
}

// callTool invokes a tool and returns its decoded text payload.
func (p *mcpProc) callTool(name string, args any) string {
	p.t.Helper()
	res := p.call("tools/call", map[string]any{"name": name, "arguments": args})
	var body struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
		IsError bool `json:"isError"`
	}
	if err := json.Unmarshal(res, &body); err != nil {
		p.t.Fatalf("tool result decode: %v", err)
	}
	if len(body.Content) != 1 {
		p.t.Fatalf("want 1 content block, got %d", len(body.Content))
	}
	if body.IsError {
		p.t.Fatalf("%s: tool error: %s", name, body.Content[0].Text)
	}
	return body.Content[0].Text
}

func (p *mcpProc) initialize() {
	p.t.Helper()
	p.call("initialize", map[string]any{"protocolVersion": "2025-06-18"})
}

// TestMcpSIGTERMFlushesMemory is the durability contract under the shutdown
// that actually happens. An MCP client stopping its server sends SIGTERM (or
// SIGINT), whose default disposition kills the process on the spot -- before
// store.Close, which is where a persistent collection writes the sidecar that
// makes its mmap files readable again. Everything remembered since the last
// flush was silently lost on an ordinary, deliberate shutdown.
//
// This has to be a real subprocess: the behavior under test is what the OS
// does to a process that has not installed a handler.
func TestMcpSIGTERMFlushesMemory(t *testing.T) {
	if testing.Short() {
		t.Skip("compiles the rostam-server binary and runs it as a subprocess")
	}
	bin := buildServer(t)
	// Taken before the child exists, so cleanup order removes it last.
	dataDir := filepath.Join(t.TempDir(), "memory")

	const fact = "the deploy key lives in vault under ops/deploy"

	p := startMcp(t, bin, dataDir)
	p.initialize()
	p.callTool("remember", map[string]any{"content": fact})

	if err := p.cmd.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatalf("SIGTERM: %v", err)
	}
	waited := make(chan error, 1)
	go func() { waited <- p.cmd.Wait() }()
	select {
	case err := <-waited:
		if err != nil {
			t.Fatalf("the server should exit cleanly on SIGTERM, got %v\nserver log:\n%s", err, p.logs)
		}
	case <-time.After(60 * time.Second):
		t.Fatalf("the server did not exit within 60s of SIGTERM\nserver log:\n%s", p.logs)
	}

	// Second session over the same data directory. It also proves the lock was
	// released: a leaked lock would refuse this start outright.
	p2 := startMcp(t, bin, dataDir)
	p2.initialize()
	payload := p2.callTool("recall", map[string]any{"query": "deploy key vault", "k": 5})

	var rec struct {
		Hits []struct {
			Content string `json:"content"`
		} `json:"hits"`
	}
	if err := json.Unmarshal([]byte(payload), &rec); err != nil {
		t.Fatalf("recall payload %q: %v", payload, err)
	}
	if len(rec.Hits) == 0 || rec.Hits[0].Content != fact {
		t.Fatalf("the memory did not survive SIGTERM: %+v\nfirst server log:\n%s", rec.Hits, p.logs)
	}

	_ = p2.in.Close() // clean EOF shutdown for the second session
	_ = p2.cmd.Wait()
}

// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"errors"
	"net"
	"strings"
	"syscall"
	"testing"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
)

// freeTCPPort returns a loopback address nothing is listening on at the
// moment it returns. There is an unavoidable gap between releasing the
// reservation here and the server binding it, in which another process can
// steal the port — startRemoteServer below retries to absorb that.
func freeTCPPort(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserve port: %v", err)
	}
	addr := ln.Addr().String()
	if err := ln.Close(); err != nil {
		t.Fatalf("release port: %v", err)
	}
	return addr
}

// startRemoteServer boots an in-process rostam.NewServer on a free loopback
// port, retrying only when the port was taken between reservation and bind.
// Retrying ONLY that error matters: a blanket retry would mask a genuine bind
// failure and turn a real defect into a slow, confusing test.
func startRemoteServer(t *testing.T) (*rostam.Server, string) {
	t.Helper()
	const attempts = 5
	for i := range attempts {
		addr := freeTCPPort(t)
		reg := ops.NewRegistry()
		if err := ops.RegisterBuiltins(reg); err != nil {
			t.Fatalf("ops: %v", err)
		}
		srv, err := rostam.NewServer(rostam.ServerConfig{
			TCPAddr:      addr,
			DirectConfig: rostam.DirectConfig{Ops: reg},
		})
		if err == nil {
			return srv, addr
		}
		if !errors.Is(err, syscall.EADDRINUSE) {
			t.Fatalf("NewServer(%s): %v", addr, err)
		}
		t.Logf("attempt %d: %s was taken between reservation and bind, retrying", i+1, addr)
	}
	t.Fatalf("no free port survived %d attempts", attempts)
	return nil, ""
}

// newRemoteStore starts an in-process TCP rostam.Server and returns a
// rostam.NewClient talking to it, proving the mcp package works unmodified
// against the remote backend (not just the embedded heap store every other
// test in this package uses). Cleanup closes the client before the server so
// the client doesn't outlive its transport.
func newRemoteStore(t *testing.T) rostam.Store {
	t.Helper()
	srv, addr := startRemoteServer(t)

	clientReg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(clientReg); err != nil {
		t.Fatalf("ops: %v", err)
	}
	cl, err := rostam.NewClient(rostam.ClientConfig{
		Servers: []string{addr},
		Ops:     clientReg,
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	t.Cleanup(func() {
		_ = cl.Close()
		_ = srv.Close()
	})
	return cl
}

// TestRemoteBackendParity drives the same memory and generic-DB flows the
// heap-store tests use, but through rostam.NewClient against an in-process
// rostam.NewServer, to prove the mcp package is backend-agnostic.
func TestRemoteBackendParity(t *testing.T) {
	c := startServer(t, Config{Store: newRemoteStore(t)})
	c.initialize()

	// remember -> recall (BM25) -> forget -> recall miss.
	var r1 struct {
		ID uint64 `json:"id"`
	}
	c.callTool("remember", map[string]any{"content": "the deploy password is stored in vault under ops/deploy"}, &r1, false)
	c.callTool("remember", map[string]any{"content": "the coffee machine is on floor two"}, nil, false)

	var rec struct {
		Hits []struct {
			ID      uint64 `json:"id"`
			Content string `json:"content"`
		} `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "deploy password vault", "k": 1}, &rec, false)
	if len(rec.Hits) != 1 || !strings.Contains(rec.Hits[0].Content, "vault") {
		t.Fatalf("BM25 recall missed: %+v", rec.Hits)
	}
	if rec.Hits[0].ID != r1.ID {
		t.Fatalf("id mismatch: %d vs %d", rec.Hits[0].ID, r1.ID)
	}

	var fg struct {
		Deleted []uint64 `json:"deleted"`
	}
	c.callTool("forget", map[string]any{"ids": []uint64{r1.ID}}, &fg, false)
	if len(fg.Deleted) != 1 || fg.Deleted[0] != r1.ID {
		t.Fatalf("forget: %+v", fg)
	}

	var recMiss struct {
		Hits []struct {
			ID uint64 `json:"id"`
		} `json:"hits"`
	}
	c.callTool("recall", map[string]any{"query": "deploy password vault"}, &recMiss, false)
	for _, h := range recMiss.Hits {
		if h.ID == r1.ID {
			t.Fatalf("forgotten memory still recallable: %+v", recMiss.Hits)
		}
	}

	// create_collection -> upsert -> search on a user collection.
	var created struct {
		Created string `json:"created"`
	}
	c.callTool("create_collection", map[string]any{"name": "docs", "dim": 4}, &created, false)
	if created.Created != "docs" {
		t.Fatalf("create_collection: got %+v", created)
	}

	c.callTool("upsert", map[string]any{
		"collection": "docs",
		"id":         uint64(1),
		"vector":     []float32{1, 0, 0, 0},
		"content":    "red fox jumps",
	}, nil, false)
	c.callTool("upsert", map[string]any{
		"collection": "docs",
		"id":         uint64(2),
		"vector":     []float32{0, 1, 0, 0},
		"content":    "blue whale swims",
	}, nil, false)

	var textRes struct {
		Hits []struct {
			ID uint64 `json:"id"`
		} `json:"hits"`
	}
	c.callTool("search", map[string]any{"collection": "docs", "mode": "text", "query_text": "fox"}, &textRes, false)
	if len(textRes.Hits) != 1 || textRes.Hits[0].ID != 1 {
		t.Fatalf("text search: %+v", textRes.Hits)
	}

	var denseRes struct {
		Hits []struct {
			ID uint64 `json:"id"`
		} `json:"hits"`
	}
	c.callTool("search", map[string]any{"collection": "docs", "mode": "dense", "vector": []float32{0, 1, 0, 0}, "k": 1}, &denseRes, false)
	if len(denseRes.Hits) != 1 || denseRes.Hits[0].ID != 2 {
		t.Fatalf("dense search: %+v", denseRes.Hits)
	}
}

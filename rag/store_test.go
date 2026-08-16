// SPDX-License-Identifier: Apache-2.0

package rag

import (
	"context"
	"errors"
	"net"
	"strings"
	"syscall"
	"testing"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

func TestEmbeddedRetrieverRoundtripBM25(t *testing.T) {
	dir := t.TempDir() // allocate BEFORE Close cleanup is registered
	r, err := NewEmbeddedRetriever(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	ctx := context.Background()

	if err := r.EnsureCorpus(ctx, "docs", 0); err != nil { // BM25-only
		t.Fatal(err)
	}
	chunks := []StoredChunk{
		{ID: 1, Content: "the epoll transport picks its loop count from GOMAXPROCS", Source: "a.md", Index: 0},
		{ID: 2, Content: "raft shards should roughly equal core count", Source: "b.md", Index: 0},
	}
	if err := r.Upsert(ctx, "docs", chunks); err != nil {
		t.Fatal(err)
	}
	hits, err := r.Search(ctx, "docs", "epoll loop count", nil, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 || hits[0].Source != "a.md" {
		t.Fatalf("expected a.md top hit, got %+v", hits)
	}
	if hits[0].Content == "" {
		t.Fatalf("hit content should be populated: %+v", hits[0])
	}
}

func TestEmbeddedRetrieverDeleteBySource(t *testing.T) {
	dir := t.TempDir()
	r, err := NewEmbeddedRetriever(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	ctx := context.Background()
	_ = r.EnsureCorpus(ctx, "docs", 0)
	_ = r.Upsert(ctx, "docs", []StoredChunk{
		{ID: 1, Content: "alpha", Source: "a.md", Index: 0},
		{ID: 2, Content: "beta", Source: "b.md", Index: 0},
	})
	n, err := r.DeleteBySource(ctx, "docs", "a.md")
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("deleted %d, want 1", n)
	}
	hits, _ := r.Search(ctx, "docs", "alpha", nil, 5)
	for _, h := range hits {
		if h.Source == "a.md" {
			t.Fatalf("a.md should be gone: %+v", hits)
		}
	}
}

// TestEmbeddedRetrieverEnsureCorpusDimMismatch covers Finding 1 from the
// whole-branch review: a corpus's Dim is fixed at creation, so re-ingesting
// the same --corpus with a different dimension (e.g. BM25's dim=1
// placeholder, then a dense embedder) must refuse cleanly instead of
// silently succeeding and failing later, deep inside Upsert, as a raw
// vector.ErrDimMismatch.
func TestEmbeddedRetrieverEnsureCorpusDimMismatch(t *testing.T) {
	dir := t.TempDir()
	r, err := NewEmbeddedRetriever(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	ctx := context.Background()

	if err := r.EnsureCorpus(ctx, "docs", 0); err != nil { // BM25-only: dim=1 placeholder
		t.Fatal(err)
	}

	err = r.EnsureCorpus(ctx, "docs", 8) // now ask for a dense embedder's real dim
	if err == nil {
		t.Fatal("expected a dimension-mismatch error, got nil (silent success)")
	}
	if !strings.Contains(err.Error(), "dimension") {
		t.Fatalf("expected an error mentioning the dimension mismatch, got: %v", err)
	}
	if errors.Is(err, vector.ErrDimMismatch) {
		t.Fatalf("expected EnsureCorpus's own actionable error, not a raw vector.ErrDimMismatch: %v", err)
	}
}

func TestHTTPRetrieverImplementsInterface(t *testing.T) {
	var _ Retriever = (*HTTPRetriever)(nil) // compile-time interface check
}

// freeTCPPort returns a loopback address nothing is listening on at the
// moment it returns. Mirrors mcp.freeTCPPort: there's an unavoidable gap
// between releasing the reservation and the server binding it, so the caller
// retries on EADDRINUSE.
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
// TCP port (the wire protocol HTTPRetriever's client actually speaks — this
// package has no HTTP/httptest server to reuse), retrying only when the port
// was taken between reservation and bind. Mirrors mcp.startRemoteServer.
func startRemoteServer(t *testing.T) (*rostam.Server, string) {
	t.Helper()
	const attempts = 5
	for i := 0; i < attempts; i++ {
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

// TestHTTPRetrieverRoundtripBM25 runs the same assertions as
// TestEmbeddedRetrieverRoundtripBM25 but against NewHTTPRetriever talking to
// an in-process rostam.NewServer, proving behavioral parity between the two
// Retriever backends over the wire (not just the shared interface).
func TestHTTPRetrieverRoundtripBM25(t *testing.T) {
	srv, addr := startRemoteServer(t)
	defer func() { _ = srv.Close() }()

	r, err := NewHTTPRetriever(addr)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	ctx := context.Background()

	if err := r.EnsureCorpus(ctx, "docs", 0); err != nil { // BM25-only
		t.Fatal(err)
	}
	chunks := []StoredChunk{
		{ID: 1, Content: "the epoll transport picks its loop count from GOMAXPROCS", Source: "a.md", Index: 0},
		{ID: 2, Content: "raft shards should roughly equal core count", Source: "b.md", Index: 0},
	}
	if err := r.Upsert(ctx, "docs", chunks); err != nil {
		t.Fatal(err)
	}
	hits, err := r.Search(ctx, "docs", "epoll loop count", nil, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 || hits[0].Source != "a.md" {
		t.Fatalf("expected a.md top hit, got %+v", hits)
	}
	if hits[0].Content == "" {
		t.Fatalf("hit content should be populated: %+v", hits[0])
	}

	n, err := r.DeleteBySource(ctx, "docs", "a.md")
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("deleted %d, want 1", n)
	}
	hits, err = r.Search(ctx, "docs", "epoll loop count", nil, 5)
	if err != nil {
		t.Fatal(err)
	}
	for _, h := range hits {
		if h.Source == "a.md" {
			t.Fatalf("a.md should be gone: %+v", hits)
		}
	}
}

// SPDX-License-Identifier: Apache-2.0
package client

import (
	"context"
	"encoding/binary"
	"io"
	"net"
	"testing"

	"github.com/rostamlabs/rostam/ops/wire"
)

// echoOKServer accepts connections and, for every request frame it reads,
// writes one canned StatusOK response with an empty payload. It speaks just
// enough of the wire protocol to exercise the CLIENT send/recv path in
// isolation — no engine, so allocs/op are purely client-side.
func echoOKServer(tb testing.TB) (addr string, stop func()) {
	tb.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		tb.Fatal(err)
	}
	// response body: [status:1=OK][payloadLen:4=0]; outer frame len = 5.
	resp := []byte{0, 0, 0, 5, 0, 0, 0, 0, 0}
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func(c net.Conn) {
				defer c.Close()
				hdr := make([]byte, 4)
				var body []byte // reused across requests so the server goroutine's
				for {           // own allocs don't pollute the client alloc count
					if _, err := io.ReadFull(c, hdr); err != nil {
						return
					}
					n := binary.BigEndian.Uint32(hdr)
					if cap(body) < int(n) {
						body = make([]byte, n)
					}
					if _, err := io.ReadFull(c, body[:n]); err != nil {
						return
					}
					if _, err := c.Write(resp); err != nil {
						return
					}
				}
			}(conn)
		}
	}()
	return ln.Addr().String(), func() { _ = ln.Close() }
}

// newRoutedEchoClient builds a routed client whose single-shard topology points
// at a canned-OK echo server, so a keyed CallFunc exercises the full client-side
// path (pickInitialTarget routing-key extraction → doCall framing → response
// decode → fn) against a server that itself allocates nothing per request. A
// vector op is used because vector routing (RouteLayoutColAt1/At2) is the path
// that used to allocate the routing key; KV ops already subslice.
func newRoutedEchoClient(tb testing.TB) (*Client, []byte, func()) {
	tb.Helper()
	addr, stop := echoOKServer(tb)
	reg := wire.NewRegistry()
	if err := wire.RegisterRoutableBuiltins(reg); err != nil {
		stop()
		tb.Fatal(err)
	}
	c, err := NewRouted(Config{Servers: []string{addr}, Ops: reg})
	if err != nil {
		stop()
		tb.Fatal(err)
	}
	c.topology.set(wire.Topology{NumShards: 1, Leaders: []string{addr}})
	// "posts" (no slash) forces the "default/" prepend — the case that used to
	// allocate a fresh routing-key slice per call.
	args := wire.EncodeVectorSearchArgs("posts", 10, []float32{1, 2, 3, 4})
	return c, args, func() { _ = c.Close(); stop() }
}

// TestCallFuncRoutedZeroAlloc is the regression guard: a routed CallFunc
// round-trip must allocate nothing on the client. It broke before routing was
// switched from the allocating ke(args) to RouteKeyInto with a stack scratch.
func TestCallFuncRoutedZeroAlloc(t *testing.T) {
	c, args, cleanup := newRoutedEchoClient(t)
	defer cleanup()
	ctx := context.Background()
	fn := func(payload []byte) error { return nil }
	// Warm up: first call dials the conn and sizes the pool/read buffers.
	if err := c.CallFunc(ctx, "vector_search", args, fn); err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(2000, func() {
		if err := c.CallFunc(ctx, "vector_search", args, fn); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("routed CallFunc: got %v allocs/op, want 0", allocs)
	}
}

// BenchmarkCallFuncRoutedAllocs reports ns/op + allocs/op for the same path.
func BenchmarkCallFuncRoutedAllocs(b *testing.B) {
	c, args, cleanup := newRoutedEchoClient(b)
	defer cleanup()
	ctx := context.Background()
	fn := func(payload []byte) error { return nil }
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := c.CallFunc(ctx, "vector_search", args, fn); err != nil {
			b.Fatal(err)
		}
	}
}

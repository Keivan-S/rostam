// SPDX-License-Identifier: Apache-2.0
package rostam

import (
	"context"
	"encoding/binary"
	"io"
	"math"
	"net"
	"testing"
)

// cannedSearchServer answers every request with the SAME pre-built
// vector_search result frame (3 hits), reusing its read buffer, so the server
// side allocates nothing per request. This isolates the CLIENT-side allocation
// count of VectorSearchInto over a real TCP round-trip.
func cannedSearchServer(tb testing.TB, hits int) (addr string, stop func()) {
	tb.Helper()
	// payload: [count:u32][ (id:u64)(dist:f32) ]*count — matches
	// wire.EncodeVectorSearchResults / DecodeVectorSearchResultsInto.
	payload := make([]byte, 4+hits*(8+4))
	binary.BigEndian.PutUint32(payload[0:4], uint32(hits))
	off := 4
	for i := 0; i < hits; i++ {
		binary.BigEndian.PutUint64(payload[off:], uint64(i+1))
		off += 8
		binary.BigEndian.PutUint32(payload[off:], math.Float32bits(float32(i)*0.5))
		off += 4
	}
	// body: [status:1=OK][payloadLen:u32][payload]; frame: [frameLen:u32][body].
	body := make([]byte, 1+4+len(payload))
	body[0] = 0
	binary.BigEndian.PutUint32(body[1:5], uint32(len(payload)))
	copy(body[5:], payload)
	frame := make([]byte, 4+len(body))
	binary.BigEndian.PutUint32(frame[0:4], uint32(len(body)))
	copy(frame[4:], body)

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		tb.Fatal(err)
	}
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func(c net.Conn) {
				defer c.Close()
				hdr := make([]byte, 4)
				var reqBody []byte
				for {
					if _, err := io.ReadFull(c, hdr); err != nil {
						return
					}
					n := binary.BigEndian.Uint32(hdr)
					if cap(reqBody) < int(n) {
						reqBody = make([]byte, n)
					}
					if _, err := io.ReadFull(c, reqBody[:n]); err != nil {
						return
					}
					if _, err := c.Write(frame); err != nil {
						return
					}
				}
			}(conn)
		}
	}()
	return ln.Addr().String(), func() { _ = ln.Close() }
}

// TestVectorSearchIntoZeroAllocClientSide guards the end-to-end promise: with a
// reused dst, a VectorSearchInto round-trip allocates nothing on the client —
// pooled request args, no defensive payload copy, decode straight into dst. It
// broke before AppendVectorSearchArgs + the args pool eliminated the request
// buffer allocation.
func TestVectorSearchIntoZeroAllocClientSide(t *testing.T) {
	const hits = 3
	addr, stop := cannedSearchServer(t, hits)
	defer stop()

	cli, err := NewClient(ClientConfig{Servers: []string{addr}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cli.Close() }()

	ctx := context.Background()
	q := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	dst := make([]VectorResult, 0, hits)

	// Warm up: first call dials the conn and sizes pool/read/args buffers.
	dst, err = cli.VectorSearchInto(ctx, "posts", q, hits, dst[:0])
	if err != nil {
		t.Fatal(err)
	}
	if len(dst) != hits {
		t.Fatalf("got %d hits, want %d", len(dst), hits)
	}

	allocs := testing.AllocsPerRun(2000, func() {
		var e error
		dst, e = cli.VectorSearchInto(ctx, "posts", q, hits, dst[:0])
		if e != nil {
			t.Fatal(e)
		}
	})
	if allocs != 0 {
		t.Fatalf("VectorSearchInto: got %v allocs/op, want 0", allocs)
	}
}

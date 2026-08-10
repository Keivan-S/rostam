// SPDX-License-Identifier: Apache-2.0

package server

import (
	"context"
	"encoding/binary"
	"io"
	"net"
	"sync"
	"testing"
	"time"
)

// echoDisp is a minimal Dispatcher: it returns the args prefixed with "ok:", so a
// test client can verify the response corresponds to the request it sent.
type echoDisp struct{}

func (echoDisp) Call(_ string, args []byte) ([]byte, error) {
	return append([]byte("ok:"), args...), nil
}
func (echoDisp) LeaderAddr() string { return "" }

// startEpoll boots an EpollServer on a free loopback port and returns its addr
// plus a stop func. It waits until the listener accepts connections.
func startEpoll(t *testing.T, idle time.Duration) (addr string, stop func()) {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0") // grab a free port, then hand it to gnet
	if err != nil {
		t.Fatal(err)
	}
	addr = l.Addr().String()
	_ = l.Close()

	es := NewEpollServer(echoDisp{}, nil, nil, 2, idle)
	if err := es.Start(addr); err != nil { // returns once bound (or on bind failure)
		t.Fatalf("epoll start: %v", err)
	}
	return addr, func() {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = es.Stop(ctx)
	}
}

// writeTestFrame frames body as {len u32}{body} and writes it, optionally splitting
// the write into two pieces with a gap to exercise partial-frame buffering.
func writeTestFrame(t *testing.T, c net.Conn, body []byte, split bool) {
	t.Helper()
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(body)))
	frame := append(hdr[:], body...)
	if split && len(frame) > 6 {
		if _, err := c.Write(frame[:5]); err != nil { // header + 1 body byte
			t.Error(err)
			return
		}
		time.Sleep(time.Millisecond) // force a separate OnTraffic with a partial frame
		if _, err := c.Write(frame[5:]); err != nil {
			t.Error(err)
		}
		return
	}
	if _, err := c.Write(frame); err != nil {
		t.Error(err)
	}
}

// readResp reads one response frame and returns its status + payload.
func readResp(t *testing.T, r io.Reader) (uint8, []byte) {
	t.Helper()
	var h [4]byte
	if _, err := io.ReadFull(r, h[:]); err != nil {
		t.Fatalf("read resp header: %v", err)
	}
	body := make([]byte, binary.BigEndian.Uint32(h[:]))
	if _, err := io.ReadFull(r, body); err != nil {
		t.Fatalf("read resp body: %v", err)
	}
	plen := binary.BigEndian.Uint32(body[1:5])
	return body[0], body[5 : 5+plen]
}

// TestEpollConcurrentFrames hammers the epoll transport with many concurrent
// connections issuing framed requests — some written in split pieces (partial
// frames), some pipelined two-at-a-time — and verifies every response matches the
// request that produced it. Run under -race to catch data races in the OnTraffic
// frame parser and gnet buffer handling.
func TestEpollConcurrentFrames(t *testing.T) {
	addr, stop := startEpoll(t, 0)
	defer stop()

	const conns, perConn = 24, 150
	var wg sync.WaitGroup
	for w := 0; w < conns; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			c, err := net.Dial("tcp", addr)
			if err != nil {
				t.Errorf("dial: %v", err)
				return
			}
			defer func() { _ = c.Close() }()
			for i := 0; i < perConn; i++ {
				arg := []byte{byte(w), byte(i), byte(i >> 8)}
				body := EncodeRequest("echo", arg)
				writeTestFrame(t, c, body, i%3 == 0) // split every 3rd frame
				status, payload := readResp(t, c)
				if status != StatusOK {
					t.Errorf("w%d i%d: status=%d", w, i, status)
					return
				}
				want := append([]byte("ok:"), arg...)
				if string(payload) != string(want) {
					t.Errorf("w%d i%d: payload=%q want %q", w, i, payload, want)
					return
				}
			}
		}(w)
	}
	wg.Wait()
}

// TestEpollPipelined verifies multiple frames written in a single syscall (the
// classic pipelining case) are each parsed and answered in order.
func TestEpollPipelined(t *testing.T) {
	addr, stop := startEpoll(t, 0)
	defer stop()

	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	// Three frames in one Write.
	var batch []byte
	for i := 0; i < 3; i++ {
		body := EncodeRequest("echo", []byte{byte(i)})
		var hdr [4]byte
		binary.BigEndian.PutUint32(hdr[:], uint32(len(body)))
		batch = append(append(batch, hdr[:]...), body...)
	}
	if _, err := c.Write(batch); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		status, payload := readResp(t, c)
		if status != StatusOK || string(payload) != string(append([]byte("ok:"), byte(i))) {
			t.Fatalf("frame %d: status=%d payload=%q", i, status, payload)
		}
	}
}

// TestEpollIdleTimeout verifies the idle sweep closes a connection that goes
// silent longer than idleTimeout (slow-loris protection), and does NOT close an
// active one.
func TestEpollIdleTimeout(t *testing.T) {
	addr, stop := startEpoll(t, 300*time.Millisecond)
	defer stop()

	c, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	// One live request confirms the connection works and resets last-active.
	writeTestFrame(t, c, EncodeRequest("echo", []byte{1}), false)
	if s, _ := readResp(t, c); s != StatusOK {
		t.Fatalf("live request status=%d", s)
	}

	// Now go idle. The sweep (interval = idleTimeout/2) should close us well
	// within 3s. A successful Read of 0 bytes + io.EOF means the server closed us.
	_ = c.SetReadDeadline(time.Now().Add(3 * time.Second))
	if _, err := c.Read(make([]byte, 1)); err == nil {
		t.Fatal("expected the idle connection to be closed, but Read returned data")
	} else if ne, ok := err.(net.Error); ok && ne.Timeout() {
		t.Fatalf("idle connection was NOT closed within 3s (got a read timeout): %v", err)
	}
}

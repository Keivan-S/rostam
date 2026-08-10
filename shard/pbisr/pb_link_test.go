// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"bufio"
	"fmt"
	"math/rand"
	"net"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Each test below stands up a minimal in-test TCP server speaking just enough
// of the pb frame protocol: it reads request frames off an accepted conn with
// a pbFrameReader and writes response frames back with writePBTestFrame,
// exactly like the real wire format pb_link.go's readLoop/writeLoop produce
// and consume.

// writePBTestFrame serializes f (header + payload) directly to w, bypassing
// the batched writer — a minimal reference encoder for the test server side.
func writePBTestFrame(w *bufio.Writer, f *pbFrame) error {
	var hdr [pbFrameHeaderSize]byte
	writePBFrameHdr(hdr[:], f)
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	if len(f.payload) > 0 {
		if _, err := w.Write(f.payload); err != nil {
			return err
		}
	}
	return w.Flush()
}

// assertGoroutinesSettle polls runtime.NumGoroutine() until it returns to (at
// most) baseline, failing the test if it never does. Guards against a leaked
// writeLoop/readLoop goroutine after close().
func assertGoroutinesSettle(t *testing.T, baseline int) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		if runtime.NumGoroutine() <= baseline {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("goroutine count did not settle: now %d, baseline %d", runtime.NumGoroutine(), baseline)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// TestPBPeerLinkRoundTrip covers the basic single-request path: a request
// frame goes out, the (in-test) server echoes an ack response correlated by
// reqID, and roundTrip returns the decoded payload.
func TestPBPeerLinkRoundTrip(t *testing.T) {
	base := runtime.NumGoroutine()

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()

	go func() {
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		fr := pbFrameReader{r: bufio.NewReader(conn)}
		bw := bufio.NewWriter(conn)
		req, err := fr.read()
		if err != nil {
			return
		}
		ack := AckMsg{Epoch: 1, Seq: uint64(req.shard), OK: true}
		resp := pbFrame{kind: pbKindResponse, shard: req.shard, reqID: req.reqID, payload: encodeAckMsg(nil, ack)}
		_ = writePBTestFrame(bw, &resp)
	}()

	link := newPBPeerLink(ln.Addr().String(), 2*time.Second)

	msg := ReplicateMsg{Epoch: 1, Seq: 5, PrevSeq: 4, Data: []byte("op-data")}
	f := &pbFrame{kind: pbKindReplicate, shard: 7, payload: encodeReplicateMsg(nil, msg)}
	payload, err := link.roundTrip(f, 3*time.Second)
	if err != nil {
		t.Fatalf("roundTrip: %v", err)
	}
	ack, err := decodeAckMsg(payload)
	if err != nil {
		t.Fatalf("decodeAckMsg: %v", err)
	}
	if ack.Epoch != 1 || ack.Seq != 7 || !ack.OK {
		t.Fatalf("ack mismatch: got %+v", ack)
	}

	link.close()
	assertGoroutinesSettle(t, base)
}

// TestPBPeerLinkPipelines fires 100 concurrent roundTrips over the single
// shared link and asserts every one gets its own correctly correlated ack,
// even though the test server intentionally replies in a shuffled order —
// proving reqID correlation (not conn/response order) is what routes each
// response back to its waiter.
func TestPBPeerLinkPipelines(t *testing.T) {
	base := runtime.NumGoroutine()
	const total = 100

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()

	go func() {
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		fr := pbFrameReader{r: bufio.NewReader(conn)}
		reqs := make([]pbFrame, 0, total)
		for i := 0; i < total; i++ {
			req, err := fr.read()
			if err != nil {
				return
			}
			reqs = append(reqs, req)
		}
		rand.Shuffle(len(reqs), func(i, j int) { reqs[i], reqs[j] = reqs[j], reqs[i] })

		bw := bufio.NewWriter(conn)
		for _, req := range reqs {
			ack := AckMsg{Epoch: 1, Seq: uint64(req.shard), OK: true}
			resp := pbFrame{kind: pbKindResponse, shard: req.shard, reqID: req.reqID, payload: encodeAckMsg(nil, ack)}
			if err := writePBTestFrame(bw, &resp); err != nil {
				return
			}
		}
	}()

	link := newPBPeerLink(ln.Addr().String(), 2*time.Second)

	var wg sync.WaitGroup
	errCh := make(chan error, total)
	for i := 0; i < total; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			msg := ReplicateMsg{Epoch: 1, Seq: uint64(i), PrevSeq: uint64(i - 1), Data: []byte("op")}
			f := &pbFrame{kind: pbKindReplicate, shard: uint32(i), payload: encodeReplicateMsg(nil, msg)} //nolint:gosec // test values
			payload, err := link.roundTrip(f, 10*time.Second)
			if err != nil {
				errCh <- fmt.Errorf("goroutine %d: roundTrip: %w", i, err)
				return
			}
			ack, err := decodeAckMsg(payload)
			if err != nil {
				errCh <- fmt.Errorf("goroutine %d: decodeAckMsg: %w", i, err)
				return
			}
			if ack.Seq != uint64(i) { //nolint:gosec // test values
				errCh <- fmt.Errorf("goroutine %d: got ack.Seq=%d, want %d (correlation mismatch)", i, ack.Seq, i)
			}
		}(i)
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Error(err)
	}

	link.close()
	assertGoroutinesSettle(t, base)
}

// startEchoFrameServer stands up a TCP listener that, per accepted connection,
// framed-reads each request and framed-writes back a pbKindResponse carrying the
// same reqID/shard/payload. Returns the address and a stop func that closes the
// listener (and any accepted conns via their read loops ending).
func startEchoFrameServer(t *testing.T) (string, func()) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func(conn net.Conn) {
				defer conn.Close()
				fr := pbFrameReader{r: bufio.NewReader(conn)}
				bw := bufio.NewWriter(conn)
				for {
					req, err := fr.read()
					if err != nil {
						return
					}
					resp := pbFrame{kind: pbKindResponse, shard: req.shard, reqID: req.reqID, payload: req.payload}
					if err := writePBTestFrame(bw, &resp); err != nil {
						return
					}
				}
			}(conn)
		}
	}()
	return ln.Addr().String(), func() { ln.Close() }
}

// startBlackholeServer stands up a TCP listener that accepts connections and
// never replies, holding them open. Returns the address and a stop func that
// closes the listener and every accepted conn.
func startBlackholeServer(t *testing.T) (string, func()) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	var mu sync.Mutex
	var conns []net.Conn
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			mu.Lock()
			conns = append(conns, conn)
			mu.Unlock()
		}
	}()
	return ln.Addr().String(), func() {
		ln.Close()
		mu.Lock()
		for _, c := range conns {
			c.Close()
		}
		mu.Unlock()
	}
}

// TestPeerLinkSubmitAsyncDelivers checks the happy path of the async submit
// API: a framed request goes out, the echo server replies, and the per-request
// callback fires exactly once with a nil error.
func TestPeerLinkSubmitAsyncDelivers(t *testing.T) {
	srvAddr, stop := startEchoFrameServer(t)
	defer stop()

	l := newPBPeerLink(srvAddr, time.Second)
	defer l.close()

	got := make(chan pbResult, 1)
	f := &pbFrame{kind: pbKindReplicate, shard: 1, payload: []byte("hi")}
	if err := l.submitAsync(f, func(p []byte, err error) {
		got <- pbResult{payload: p, err: err}
	}); err != nil {
		t.Fatalf("submitAsync: %v", err)
	}
	select {
	case r := <-got:
		if r.err != nil {
			t.Fatalf("callback err: %v", r.err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("callback never fired")
	}
}

// TestPeerLinkFailInvokesPendingCallbacks asserts that closing the link fails
// every in-flight async submit through its callback, exactly once.
func TestPeerLinkFailInvokesPendingCallbacks(t *testing.T) {
	srvAddr, stop := startBlackholeServer(t)
	defer stop()
	l := newPBPeerLink(srvAddr, time.Second)

	var calls int32
	done := make(chan error, 1)
	f := &pbFrame{kind: pbKindReplicate, shard: 1, payload: []byte("x")}
	_ = l.submitAsync(f, func(_ []byte, err error) {
		atomic.AddInt32(&calls, 1)
		done <- err
	})
	l.close() // triggers fail(errPBLinkClosed)
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("expected error on close")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("callback never fired on fail")
	}
	time.Sleep(50 * time.Millisecond)
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("callback fired %d times, want exactly 1", got)
	}
}

// TestPeerLinkCancelReportsDeletion pins cancel's bool contract, which
// submitAsync's enqueue-failure branches rely on to avoid double-completion
// (returning an error while a concurrent fail also invokes the callback):
// cancel returns true iff it removed a live pending entry, false when the entry
// is already absent (claimed by fail/deliver, which owns the callback).
func TestPeerLinkCancelReportsDeletion(t *testing.T) {
	srvAddr, stop := startBlackholeServer(t)
	defer stop()
	l := newPBPeerLink(srvAddr, time.Second)
	defer l.close()

	// Register a live pending entry by hand, mirroring submitAsync's
	// registration (ensureConnLocked initializes the pending map).
	l.mu.Lock()
	if err := l.ensureConnLocked(); err != nil {
		l.mu.Unlock()
		t.Fatalf("ensureConnLocked: %v", err)
	}
	l.pending[42] = func([]byte, error) {}
	l.mu.Unlock()

	if !l.cancel(42) {
		t.Fatal("cancel(live entry) = false, want true")
	}
	if l.cancel(42) {
		t.Fatal("cancel(absent entry) = true, want false")
	}
}

// TestPBPeerLinkReDialsAfterFailure kills the server side of the connection
// mid-flight (after reading the request but before responding) so the
// in-flight roundTrip must fail fast (not merely time out), then proves a
// subsequent roundTrip against a freshly accepted connection on the same
// listener succeeds — i.e. the link re-dials rather than getting stuck on the
// dead generation.
func TestPBPeerLinkReDialsAfterFailure(t *testing.T) {
	base := runtime.NumGoroutine()

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()

	// Generation 1: accept, read the request, then kill the conn without
	// ever writing a response.
	go func() {
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		fr := pbFrameReader{r: bufio.NewReader(conn)}
		_, _ = fr.read()
		conn.Close()
	}()

	link := newPBPeerLink(ln.Addr().String(), 2*time.Second)
	defer link.close()

	f1 := &pbFrame{kind: pbKindReplicate, shard: 1, payload: encodeReplicateMsg(nil, ReplicateMsg{Epoch: 1, Seq: 1})}
	start := time.Now()
	if _, err := link.roundTrip(f1, 10*time.Second); err == nil {
		t.Fatal("expected roundTrip to fail after the server killed the connection mid-flight")
	}
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Fatalf("roundTrip took %v to fail — looks like it fell through to the timeout instead of the fail-fast path", elapsed)
	}

	// Generation 2: a freshly accepted connection on the SAME listener,
	// standing in for "the server restarted" — this time it replies properly.
	go func() {
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		fr := pbFrameReader{r: bufio.NewReader(conn)}
		bw := bufio.NewWriter(conn)
		req, err := fr.read()
		if err != nil {
			return
		}
		ack := AckMsg{Epoch: 2, Seq: uint64(req.shard), OK: true}
		resp := pbFrame{kind: pbKindResponse, shard: req.shard, reqID: req.reqID, payload: encodeAckMsg(nil, ack)}
		_ = writePBTestFrame(bw, &resp)
	}()

	f2 := &pbFrame{kind: pbKindReplicate, shard: 2, payload: encodeReplicateMsg(nil, ReplicateMsg{Epoch: 2, Seq: 2})}
	payload, err := link.roundTrip(f2, 5*time.Second)
	if err != nil {
		t.Fatalf("expected re-dialed roundTrip to succeed, got: %v", err)
	}
	ack, err := decodeAckMsg(payload)
	if err != nil {
		t.Fatalf("decodeAckMsg: %v", err)
	}
	if ack.Seq != 2 || !ack.OK {
		t.Fatalf("ack mismatch after re-dial: got %+v", ack)
	}

	link.close()
	assertGoroutinesSettle(t, base)
}

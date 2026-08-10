// SPDX-License-Identifier: Apache-2.0

package fabric

import (
	"bufio"
	"net"
	"sync"
	"time"
)

// peerLink is the single shared, multiplexed outbound connection to one peer.
// This node's requests for every group flow out over it and the peer's responses
// flow back, correlated by reqID (NOT by conn order, because groups interleave).
// It lazily dials, runs one batched writer goroutine and one reader goroutine per
// connection generation, and on any I/O error fails every in-flight request and
// drops the connection so the next send re-dials.
type peerLink struct {
	fabric *Fabric
	target string

	mu      sync.Mutex
	conn    net.Conn      // nil when disconnected
	sendCh  chan *frame   // per-generation; buffered
	done    chan struct{} // closed when this generation's conn fails
	pending map[uint64]chan result
}

// ensureConnLocked dials and starts the read/write goroutines if the link is not
// currently connected. Caller holds l.mu.
func (l *peerLink) ensureConnLocked() error {
	if l.conn != nil {
		return nil
	}
	if l.fabric.isClosed() {
		return errFabricClosed
	}
	conn, err := l.fabric.dialConn(l.target, dialTimeout)
	if err != nil {
		return err
	}
	// Handshake: this is the multiplexed link (not a snapshot conn).
	if _, err := conn.Write([]byte{connMux}); err != nil {
		_ = conn.Close()
		return err
	}
	l.conn = conn
	l.sendCh = make(chan *frame, sendChanBuffer)
	l.done = make(chan struct{})
	l.pending = make(map[uint64]chan result)
	go l.writeLoop(conn, l.sendCh, l.done)
	go l.readLoop(conn, l.done)
	return nil
}

// submit registers a pending waiter for fr.reqID and enqueues the frame on the
// current connection generation, dialing first if needed.
func (l *peerLink) submit(fr *frame) (chan result, error) {
	l.mu.Lock()
	if err := l.ensureConnLocked(); err != nil {
		l.mu.Unlock()
		return nil, err
	}
	ch := make(chan result, 1)
	l.pending[fr.reqID] = ch
	sendCh := l.sendCh
	done := l.done
	l.mu.Unlock()

	select {
	case sendCh <- fr:
		return ch, nil
	case <-done:
		l.cancel(fr.reqID)
		return nil, errLinkClosed
	case <-l.fabric.closed:
		l.cancel(fr.reqID)
		return nil, errFabricClosed
	}
}

// roundTrip submits fr and blocks for its response, timeout, or link failure.
func (l *peerLink) roundTrip(fr *frame, timeout time.Duration) ([]byte, error) {
	ch, err := l.submit(fr)
	if err != nil {
		return nil, err
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case res := <-ch:
		return res.payload, res.err
	case <-timer.C:
		l.cancel(fr.reqID)
		return nil, errRPCTimeout
	case <-l.fabric.closed:
		l.cancel(fr.reqID)
		return nil, errFabricClosed
	}
}

// cancel removes a pending waiter (on timeout or abandonment).
func (l *peerLink) cancel(reqID uint64) {
	l.mu.Lock()
	delete(l.pending, reqID)
	l.mu.Unlock()
}

// deliver routes a response frame to its waiter by reqID. An unknown reqID
// (already timed out, or from a prior generation) is dropped.
func (l *peerLink) deliver(fr frame) {
	l.mu.Lock()
	ch := l.pending[fr.reqID]
	if ch != nil {
		delete(l.pending, fr.reqID)
	}
	l.mu.Unlock()
	if ch != nil {
		ch <- result{payload: fr.payload}
	}
}

// fail tears down the current connection generation and fails every pending
// waiter. Idempotent per generation (guarded by conn==nil). The next submit
// re-dials a fresh generation.
func (l *peerLink) fail(err error) {
	l.mu.Lock()
	if l.conn == nil {
		l.mu.Unlock()
		return
	}
	conn := l.conn
	done := l.done
	pending := l.pending
	l.conn = nil
	l.sendCh = nil
	l.done = nil
	l.pending = nil
	l.mu.Unlock()

	_ = conn.Close()
	close(done)
	for _, ch := range pending {
		ch <- result{err: err}
	}
}

func (l *peerLink) writeLoop(conn net.Conn, sendCh <-chan *frame, done <-chan struct{}) {
	bw := bufio.NewWriterSize(conn, writeBufSize)
	if err := runFramedWriter(bw, sendCh, done, writeLinger); err != nil {
		l.fail(err)
	}
}

func (l *peerLink) readLoop(conn net.Conn, done <-chan struct{}) {
	fr := frameReader{r: bufio.NewReaderSize(conn, readBufSize)}
	for {
		msg, err := fr.read()
		if err != nil {
			l.fail(err)
			return
		}
		select {
		case <-done:
			return
		default:
		}
		l.deliver(msg)
	}
}

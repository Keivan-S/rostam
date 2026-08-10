// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"bufio"
	"crypto/tls"
	"errors"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/rostamlabs/rostam/tlsutil"
)

// pbSendChanBuffer bounds each connection generation's outbound frame queue so
// a stalled peer applies backpressure to callers (a full sendCh blocks submit,
// see roundTrip's select) instead of growing memory without bound.
const pbSendChanBuffer = 256

// pbLinkReadBufSize sizes the buffered reader wrapping each inbound conn.
const pbLinkReadBufSize = 256 << 10

var (
	errPBLinkClosed = errors.New("pbisr: peer link closed")
	errPBRPCTimeout = errors.New("pbisr: rpc timeout")
)

// pbResult is one correlated response (or failure) delivered to a waiting
// roundTrip call.
type pbResult struct {
	payload []byte
	err     error
}

// pbPeerLink is the single shared, multiplexed outbound connection to one
// peer for the batched PB transport. Every shard's replicate
// requests to this peer flow out over it, and the peer's acks flow back,
// correlated by reqID (NOT by conn order, because shards interleave on the
// shared conn). It lazily dials, runs one batched writer goroutine and one
// reader goroutine per connection generation, and on any I/O error fails
// every in-flight request and drops the connection so the next roundTrip
// re-dials.
//
// Ported from raft/fabric/link.go's peerLink, minus the connMux handshake
// byte (PB has exactly one connection kind, so there is nothing to
// disambiguate on dial).
type pbPeerLink struct {
	target      string
	dialTimeout time.Duration

	// clientTLS, when non-nil, upgrades this link's lazy dial to TLS; cnAllow pins
	// the peer's verified server-cert CN when set. Both are populated by
	// NetTransport.linkFor from the transport's config (nil ⇒ plaintext dial,
	// byte-identical to before). A link created directly (tests) leaves them nil.
	clientTLS *tls.Config
	cnAllow   map[string]bool

	mu      sync.Mutex
	conn    net.Conn // nil when disconnected
	sendCh  chan *pbFrame
	done    chan struct{} // closed when this generation's conn fails
	pending map[uint64]func([]byte, error)

	closed    chan struct{} // closed by close(); unblocks every blocked roundTrip/submit
	closeOnce sync.Once

	reqID atomic.Uint64
}

// newPBPeerLink constructs a link that lazily dials target (with dialTimeout)
// on the first roundTrip.
func newPBPeerLink(target string, dialTimeout time.Duration) *pbPeerLink {
	return &pbPeerLink{
		target:      target,
		dialTimeout: dialTimeout,
		closed:      make(chan struct{}),
	}
}

// ensureConnLocked dials and starts the writer/reader goroutines if the link
// is not currently connected. Caller holds l.mu.
func (l *pbPeerLink) ensureConnLocked() error {
	if l.conn != nil {
		return nil
	}
	select {
	case <-l.closed:
		return errPBLinkClosed
	default:
	}
	conn, err := l.dial()
	if err != nil {
		return err
	}
	l.conn = conn
	l.sendCh = make(chan *pbFrame, pbSendChanBuffer)
	l.done = make(chan struct{})
	l.pending = make(map[uint64]func([]byte, error))
	go l.writeLoop(conn, l.sendCh, l.done)
	go l.readLoop(conn, l.done)
	return nil
}

// dial opens the underlying connection, upgrading to TLS when clientTLS is
// configured. The plaintext path (nil clientTLS) is byte-identical to the
// historical net.DialTimeout. The TLS path pins the peer host as ServerName (peer
// server cert verified against its SAN — never InsecureSkipVerify) and, when an
// allowlist is set, the peer's verified server-cert CN — mutual identity to match
// the server-side gate.
func (l *pbPeerLink) dial() (net.Conn, error) {
	if l.clientTLS == nil {
		return net.DialTimeout("tcp", l.target, l.dialTimeout)
	}
	d := &net.Dialer{Timeout: l.dialTimeout}
	cfg := l.clientTLS.Clone()
	host, _, err := net.SplitHostPort(l.target)
	if err != nil {
		host = l.target
	}
	cfg.ServerName = host
	if len(l.cnAllow) > 0 {
		allow := l.cnAllow
		cfg.VerifyConnection = func(cs tls.ConnectionState) error {
			return tlsutil.PeerCNAllowed(cs, allow)
		}
	}
	return tls.DialWithDialer(d, "tcp", l.target, cfg)
}

// submitAsync assigns f.reqID from the link's counter, registers done as the
// completion callback for that reqID, and enqueues f on the current connection
// generation (dialing first if needed). done must be non-nil (a nil callback
// would linger in pending — deliver's nil-check skips its delete — and later
// panic when fail invokes it). It is fire-and-forget: on success it returns nil
// and done is invoked EXACTLY ONCE later — by deliver (success, err==nil) or by
// fail (error).
//
// On enqueue failure, the frame never reached sendCh, so it races a concurrent
// fail that may have already swapped out (and invoked) this entry. cancel
// reports whether it actually removed the registration: if it did, done
// provably will NOT fire (the frame was never sent and reqIDs are globally
// monotonic, so no response can exist), and we return the error without calling
// done — the caller owns cleanup, mirroring roundTrip's pooled-payload recycle
// contract. If cancel found the entry already gone, fail owns it and has/will
// invoke done, so we return nil: submission completed via callback, not failure.
func (l *pbPeerLink) submitAsync(f *pbFrame, done func([]byte, error)) error {
	f.reqID = l.reqID.Add(1)
	l.mu.Lock()
	if err := l.ensureConnLocked(); err != nil {
		l.mu.Unlock()
		return err
	}
	l.pending[f.reqID] = done
	sendCh := l.sendCh
	linkDone := l.done
	l.mu.Unlock()

	select {
	case sendCh <- f:
		return nil
	case <-linkDone:
		if l.cancel(f.reqID) {
			return errPBLinkClosed
		}
		return nil
	case <-l.closed:
		if l.cancel(f.reqID) {
			return errPBLinkClosed
		}
		return nil
	}
}

// trySubmitAsync is submitAsync's NON-BLOCKING, NO-DIAL variant (lever 1: the
// engine's inline fast path). It refuses — returning false, with done
// guaranteed NOT to fire — when the link is not currently connected (an inline
// caller must never eat a dial) or the send queue cannot take the frame
// without blocking. On true the frame was accepted by the current connection
// generation and done fires exactly once, same contract as submitAsync. When a
// concurrent generation failure claims the registration first, the failure
// callback IS the completion, so it reports true.
func (l *pbPeerLink) trySubmitAsync(f *pbFrame, done func([]byte, error)) bool {
	l.mu.Lock()
	if l.conn == nil {
		l.mu.Unlock()
		return false
	}
	f.reqID = l.reqID.Add(1)
	l.pending[f.reqID] = done
	sendCh := l.sendCh
	linkDone := l.done
	l.mu.Unlock()

	select {
	case sendCh <- f:
		return true
	case <-linkDone:
		// Generation died between register and send: if cancel still owned the
		// registration, done will never fire — report not-submitted; if fail
		// claimed it, done has fired (with the failure) — the submission is
		// complete either way it resolves.
		return !l.cancel(f.reqID)
	default:
		// Full queue: back out without blocking (same ownership rule as above).
		return !l.cancel(f.reqID)
	}
}

// roundTrip submits f via submitAsync with a callback that forwards the result
// into a buffered channel, then blocks for that result, a timeout, or link
// close. It is the synchronous request/response API layered on submitAsync.
func (l *pbPeerLink) roundTrip(f *pbFrame, timeout time.Duration) ([]byte, error) {
	ch := make(chan pbResult, 1)
	if err := l.submitAsync(f, func(p []byte, err error) {
		ch <- pbResult{payload: p, err: err}
	}); err != nil {
		// submitAsync failed before f ever reached sendCh, so it will never be
		// picked up by a batcher/writer — the only place that calls
		// pbPutPayload. Recycle it here or a pooled payload leaks.
		if f.pooled {
			pbPutPayload(f.payload)
		}
		return nil, err
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case res := <-ch:
		return res.payload, res.err
	case <-timer.C:
		l.cancel(f.reqID)
		return nil, errPBRPCTimeout
	case <-l.closed:
		l.cancel(f.reqID)
		return nil, errPBLinkClosed
	}
}

// cancel removes a pending waiter (on timeout or abandonment) and reports
// whether it actually deleted a live entry. A false return means the entry was
// already gone — claimed by a concurrent fail/deliver, which owns invoking the
// callback — so callers deciding completion (submitAsync) must not treat it as
// their own failure. Safe when l.pending is nil (post-fail): delete is a no-op
// and the lookup reports absent.
func (l *pbPeerLink) cancel(reqID uint64) bool {
	l.mu.Lock()
	_, ok := l.pending[reqID]
	delete(l.pending, reqID)
	l.mu.Unlock()
	return ok
}

// deliver routes a response frame to its waiter by reqID. An unknown reqID
// (already timed out/canceled, or belonging to a prior generation) is
// dropped.
func (l *pbPeerLink) deliver(f pbFrame) {
	l.mu.Lock()
	cb := l.pending[f.reqID]
	if cb != nil {
		delete(l.pending, f.reqID)
	}
	l.mu.Unlock()
	if cb != nil {
		cb(f.payload, nil)
	}
}

// fail tears down the current connection generation and fails every pending
// waiter. Idempotent per generation (guarded by conn == nil): only the first
// of {writeLoop, readLoop, close()} to observe a live conn performs the
// teardown; later callers see conn == nil and return immediately, so done is
// never closed twice and pending is never drained twice. The next submit
// re-dials a fresh generation.
func (l *pbPeerLink) fail(err error) {
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
	for _, cb := range pending {
		cb(nil, err)
	}
}

// writeLoop drains sendCh via the batched writer until done closes or a
// write fails. It writes directly to conn — NOT a bufio.Writer wrapping it —
// because runPBFramedWriter's flush uses net.Buffers.WriteTo, which (given a
// raw net.Conn) takes the writev fast path: one syscall per batch, no payload
// copy. Wrapping conn in a bufio.Writer here would silently defeat that:
// net.Buffers.WriteTo would fall back to per-slice bufio.Writer.Write calls
// that just fill its internal buffer, and since nothing ever calls Flush,
// small batches would sit unflushed until enough traffic accumulated to fill
// the buffer outright — stalling delivery instead of one writev per burst.
func (l *pbPeerLink) writeLoop(conn net.Conn, sendCh <-chan *pbFrame, done <-chan struct{}) {
	if err := runPBFramedWriter(conn, sendCh, done, pbWriteLinger); err != nil {
		l.fail(err)
	}
}

// readLoop reads frames off conn until done closes or a read fails,
// delivering each to its correlated waiter.
func (l *pbPeerLink) readLoop(conn net.Conn, done <-chan struct{}) {
	fr := pbFrameReader{r: bufio.NewReaderSize(conn, pbLinkReadBufSize)}
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

// close shuts the link down: it signals every blocked roundTrip/submit and
// tears down the current connection generation (if any), failing pending
// waiters with errPBLinkClosed. Safe to call more than once and safe to call
// even if the link never dialed.
func (l *pbPeerLink) close() {
	l.closeOnce.Do(func() { close(l.closed) })
	l.fail(errPBLinkClosed)
}

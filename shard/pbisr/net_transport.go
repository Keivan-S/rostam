// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"bufio"
	"crypto/tls"
	"net"
	"sync"
	"time"

	"github.com/rostamlabs/rostam/tlsutil"
)

// Receiver applies replicated writes on a backup and returns their acks.
// *Engine satisfies it (Engine.Receive / Engine.ReceiveGroup / Engine.CatchupInfo);
// ReceiveGroup answers one group frame with one cumulative ack, and
// CatchupInfo answers a grow handshake with the backup's applied
// high-water.
type Receiver interface {
	Receive(msg ReplicateMsg) AckMsg
	ReceiveGroup(msgs []ReplicateMsg) AckMsg
	CatchupInfo() CatchupInfoMsg
}

var _ Receiver = (*Engine)(nil)

// snapshotReceiver is the OPTIONAL Receiver capability for snapshot
// transfer. It is kept off Receiver itself so the many test receivers that
// implement only the three write-path methods keep compiling; a receiver without
// it simply nacks a snapshot chunk, which the grow reads as an aborted attempt.
type snapshotReceiver interface {
	ReceiveSnapshotChunk(c SnapshotChunk) AckMsg
}

var _ snapshotReceiver = (*Engine)(nil)

// NetTransport is one node's real inter-node PB replication transport: a TCP
// server dispatching inbound replicate requests to per-shard receivers, and a
// client side backed by one pipelined, batched pbPeerLink per peer —
// every shard's outbound replicate traffic to a given peer multiplexes over
// that peer's single shared connection.
type NetTransport struct {
	ln net.Listener

	// clientTLS, when non-nil, upgrades outbound peer-link dials to TLS. cnAllow is
	// the OPT-IN per-node identity allowlist enforced on accepted conns after the
	// handshake and pinned on the peer server cert when dialing. Both nil/empty ⇒
	// plaintext, byte-identical to before. See [NewNetTransport].
	clientTLS *tls.Config
	cnAllow   map[string]bool

	mu    sync.RWMutex
	recv  map[int]Receiver       // shard → local backup receiver
	conns map[net.Conn]struct{}  // tracked inbound connections being served
	links map[string]*pbPeerLink // peer addr → outbound pipelined link

	closeOnce sync.Once
	done      chan struct{}
}

// pbHandshakeTimeout bounds the inter-node TLS handshake on an accepted conn so a
// peer that connects but never completes the handshake cannot pin a serve
// goroutine indefinitely.
const pbHandshakeTimeout = 10 * time.Second

// NewNetTransport binds a TCP listener on addr (":0" for an ephemeral port) and
// starts the accept loop.
//
// serverTLS/clientTLS/cnAllow are the OPT-IN inter-node mTLS parameters: all
// nil/empty is the default plaintext path, byte-identical to before. When
// serverTLS is non-nil the listener is wrapped with [tls.NewListener] (mTLS via
// RequireAndVerifyClientCert) and every accepted conn's verified client-cert CN is
// checked against cnAllow BEFORE any replicate frame is applied; when clientTLS is
// non-nil the outbound peer-link dials upgrade to TLS.
func NewNetTransport(addr string, serverTLS, clientTLS *tls.Config, cnAllow map[string]bool) (*NetTransport, error) {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, err
	}
	// TLS-wrap the listener so an accepted conn completes an mTLS handshake before
	// any replicate frame reaches a receiver. nil serverTLS ⇒ ln stays the
	// plaintext listener (unchanged).
	if serverTLS != nil {
		ln = tls.NewListener(ln, serverTLS)
	}
	t := &NetTransport{
		ln:        ln,
		clientTLS: clientTLS,
		cnAllow:   cnAllow,
		recv:      make(map[int]Receiver),
		conns:     make(map[net.Conn]struct{}),
		links:     make(map[string]*pbPeerLink),
		done:      make(chan struct{}),
	}
	go t.acceptLoop()
	return t, nil
}

// Register sets the local receiver for a shard (the node's Engine for that shard).
func (t *NetTransport) Register(shard int, r Receiver) {
	t.mu.Lock()
	t.recv[shard] = r
	t.mu.Unlock()
}

// Addr returns the bound listen address.
func (t *NetTransport) Addr() string { return t.ln.Addr().String() }

// authenticatePeer completes the mTLS handshake on an accepted inter-node conn
// and enforces the per-node CN allowlist. On a plaintext listener (conn is not a
// *tls.Conn) it is a no-op passthrough. A non-nil return means the peer is NOT an
// authenticated, allowlisted cluster member; serveConn then drops the conn.
func (t *NetTransport) authenticatePeer(conn net.Conn) error {
	tc, ok := conn.(*tls.Conn)
	if !ok {
		return nil // plaintext listener: nothing to authenticate
	}
	_ = tc.SetDeadline(time.Now().Add(pbHandshakeTimeout))
	if err := tc.Handshake(); err != nil {
		return err
	}
	_ = tc.SetDeadline(time.Time{})
	// Authenticated: the peer cert chained to the cluster CA (server cfg required
	// and verified it). The allowlist adds identity pinning (empty ⇒ any CA-valid
	// peer, see tlsutil.PeerCNAllowed).
	return tlsutil.PeerCNAllowed(tc.ConnectionState(), t.cnAllow)
}

func (t *NetTransport) receiverFor(shard int) (Receiver, bool) {
	t.mu.RLock()
	r, ok := t.recv[shard]
	t.mu.RUnlock()
	return r, ok
}

func (t *NetTransport) acceptLoop() {
	for {
		conn, err := t.ln.Accept()
		if err != nil {
			return // listener closed (via Close) or fatal accept error
		}
		go t.serveConn(conn)
	}
}

// serveConn handles one inbound connection: a stream of framed replicate
// requests, each dispatched to its shard's receiver and answered with a
// correlated response frame, until the peer closes or errors. Responses are
// enqueued on a per-conn batched writer (runPBFramedWriter) so a burst of
// replies to interleaved shards coalesces into as few writev syscalls as
// possible, exactly like the outbound pbPeerLink.
func (t *NetTransport) serveConn(conn net.Conn) {
	t.mu.Lock()
	t.conns[conn] = struct{}{}
	t.mu.Unlock()

	// Trust boundary: on a TLS listener the conn is unauthenticated until the
	// handshake completes. Force it and pin the verified client-cert CN to the
	// allowlist BEFORE starting the writer or reading any replicate frame, so only
	// an authenticated, allowlisted peer can inject a replicated write. Plaintext
	// listener (nil serverTLS) ⇒ no-op, byte-identical to before. On failure untrack
	// and close without ever touching a receiver.
	if err := t.authenticatePeer(conn); err != nil {
		t.mu.Lock()
		delete(t.conns, conn)
		t.mu.Unlock()
		_ = conn.Close()
		return
	}

	sendCh := make(chan *pbFrame, pbSendChanBuffer)
	connDone := make(chan struct{})
	var closeOnce sync.Once
	closeConn := func() { closeOnce.Do(func() { _ = conn.Close() }) }

	go func() {
		_ = runPBFramedWriter(conn, sendCh, connDone, pbWriteLinger)
		closeConn() // unblock the reader if the writer died first
	}()

	defer func() {
		close(connDone)
		t.mu.Lock()
		delete(t.conns, conn)
		t.mu.Unlock()
		closeConn()
	}()

	// pooled reader: every request payload is dead once its receiver returns
	// (Receive applies synchronously, ReceiveGroup's decode copies), so it is
	// recycled right after the ack is computed below.
	fr := pbFrameReader{r: bufio.NewReaderSize(conn, pbLinkReadBufSize), pooled: true}
	for {
		req, err := fr.read()
		if err != nil {
			return // EOF / closed / bad frame
		}
		// respPayload is built per request kind: replicate/group answer with an
		// AckMsg, the catch-up handshake with its own CatchupInfoMsg payload (Stage
		// 4.1 — the handshake reports a log identity, which does not fit an ack).
		var respPayload []byte
		var ack AckMsg
		switch req.kind {
		case pbKindReplicate:
			msg, derr := decodeReplicateMsg(req.payload)
			if derr != nil {
				// Undecodable request: cannot know epoch/seq; drop the conn.
				return
			}
			if r, ok := t.receiverFor(int(req.shard)); ok {
				ack = r.Receive(msg)
			} else {
				ack = AckMsg{Epoch: msg.Epoch, Seq: msg.Seq, OK: false}
			}
		case pbKindReplicateGroup:
			msgs, derr := decodeReplicateGroup(req.payload)
			if derr != nil {
				return
			}
			if r, ok := t.receiverFor(int(req.shard)); ok {
				ack = r.ReceiveGroup(msgs)
			} else {
				// No receiver: cumulative no-credit nack (nothing applied).
				// Seq is "the last seq that applied", so no-credit is one below
				// the group's first — except at seq 0, where that subtraction
				// underflows to MaxUint64 and would read as crediting the entire
				// space. Genesis has no predecessor, so 0 is the floor.
				noCredit := uint64(0)
				if msgs[0].Seq > 0 {
					noCredit = msgs[0].Seq - 1
				}
				ack = AckMsg{Epoch: msgs[0].Epoch, Seq: noCredit, OK: false}
			}
		case pbKindCatchupReq:
			// ISR grow handshake: answer with this shard's LOG IDENTITY
			// (CatchupInfo — applied high-water plus the applied frontier and its
			// epoch). The request payload is an AckMsg carrying the growing
			// primary's epoch (informational — the primary fences on our returned
			// Epoch). No receiver ⇒ a not-OK response the grower reads as "nothing
			// to catch up from here".
			if _, derr := decodeAckMsg(req.payload); derr != nil {
				return
			}
			info := CatchupInfoMsg{OK: false}
			if r, ok := t.receiverFor(int(req.shard)); ok {
				info = r.CatchupInfo()
			}
			respPayload = encodeCatchupInfo(pbGetPayload(), info)
		case pbKindSnapshotChunk:
			// Snapshot transfer. Non-final chunks are cheap (an append into
			// the receiver's staging buffer); the FINAL chunk performs the whole
			// wipe+install synchronously on this conn goroutine, which is why the
			// sender's per-chunk timeout must accommodate a real install (see
			// pbSnapshotChunkTimeout) and why a snapshot never rides the sender path.
			chunk, derr := decodeSnapshotChunk(req.payload)
			if derr != nil {
				return
			}
			if r, ok := t.receiverFor(int(req.shard)); ok {
				if sr, ok := r.(snapshotReceiver); ok {
					ack = sr.ReceiveSnapshotChunk(chunk)
				} else {
					ack = AckMsg{Epoch: chunk.Epoch, Seq: chunk.FrontierSeq, OK: false}
				}
			} else {
				ack = AckMsg{Epoch: chunk.Epoch, Seq: chunk.FrontierSeq, OK: false}
			}
		default:
			// Unknown request kind: protocol mismatch; drop the conn.
			return
		}
		if respPayload == nil {
			respPayload = encodeAckMsg(pbGetPayload(), ack)
		}
		if req.payload != nil {
			pbPutReadBuf(req.payload) // receiver done: recycle (see reader note)
		}
		resp := &pbFrame{
			kind:    pbKindResponse,
			shard:   req.shard,
			reqID:   req.reqID,
			payload: respPayload,
			pooled:  true,
		}
		select {
		case sendCh <- resp:
		case <-connDone:
			return
		}
	}
}

// Close stops the accept loop, closes the listener, closes all inbound
// connections currently being served (unblocking their serveConn goroutines),
// and closes every outbound peer link (unblocking any in-flight roundTrip
// calls and their writer/reader goroutines).
func (t *NetTransport) Close() error {
	var err error
	t.closeOnce.Do(func() {
		close(t.done)
		err = t.ln.Close()

		t.mu.Lock()
		for conn := range t.conns {
			_ = conn.Close()
		}
		t.mu.Unlock()

		t.mu.Lock()
		for _, l := range t.links {
			l.close()
		}
		t.mu.Unlock()
	})
	return err
}

// dialTimeout bounds how long a fresh outbound dial (on a peer link's first
// use, or re-dial after a failure) may take.
const dialTimeout = 3 * time.Second

// linkFor returns the shared outbound pbPeerLink for peer, lazily creating one
// on first use. The link itself lazily dials on its first roundTrip.
func (t *NetTransport) linkFor(peer string) *pbPeerLink {
	t.mu.Lock()
	l := t.links[peer]
	if l == nil {
		l = newPBPeerLink(peer, dialTimeout)
		// Inter-node TLS for the outbound link (nil clientTLS ⇒ plaintext dial,
		// byte-identical to before). Set on the link so its lazy dial upgrades to
		// TLS with the peer host pinned as ServerName + the allowlist pinning the
		// peer's server-cert CN. Threaded here (not via newPBPeerLink) so the many
		// direct newPBPeerLink test call sites stay plaintext and unchanged.
		l.clientTLS = t.clientTLS
		l.cnAllow = t.cnAllow
		t.links[peer] = l
	}
	t.mu.Unlock()
	return l
}

// For returns a shard-scoped Transport view: its Replicate tags outbound frames
// with this shard so the peer's server dispatches to the right receiver.
func (t *NetTransport) For(shard int) Transport {
	return &shardTransport{t: t, shard: uint32(shard)} //nolint:gosec // shard ids are small, non-negative
}

// shardTransport is a NetTransport view bound to one shard; it satisfies
// pbisr.Transport.
type shardTransport struct {
	t     *NetTransport
	shard uint32
}

var _ Transport = (*shardTransport)(nil)

// Replicate submits msg to peer over the peer's shared pipelined link and
// completes asynchronously: submitAsync enqueues the frame (in submission order,
// since the engine's single per-peer sender is the only caller) and invokes the
// callback exactly once when the correlated ack arrives or the link fails. The
// callback decodes the ack payload and forwards it to done. On a submit failure
// the frame never reached the wire — its returned error IS the completion (done
// is NOT invoked; the transport guarantees exactly one of {error, callback}) — so we
// recycle the pooled payload here (mirroring roundTrip) and return the error.
func (s *shardTransport) Replicate(peer string, msg ReplicateMsg, done func(AckMsg, error)) error {
	f := &pbFrame{
		kind:    pbKindReplicate,
		shard:   s.shard,
		payload: encodeReplicateMsg(pbGetPayload(), msg),
		pooled:  true,
	}
	err := s.t.linkFor(peer).submitAsync(f, func(p []byte, cbErr error) {
		if cbErr != nil {
			done(AckMsg{}, cbErr)
			return
		}
		ack, derr := decodeAckMsg(p)
		done(ack, derr)
	})
	if err != nil {
		if f.pooled {
			pbPutPayload(f.payload)
		}
		return err
	}
	return nil
}

var _ GroupTransport = (*shardTransport)(nil)

var _ InlineTransport = (*shardTransport)(nil)

var _ CatchupTransport = (*shardTransport)(nil)

// pbCatchupTimeout bounds the grow handshake round-trip. A catch-up is a
// background control action, so a generous bound is fine; a slow/unreachable
// target simply fails the grow (retried by the driver next tick).
const pbCatchupTimeout = 5 * time.Second

// CatchupRequest sends a grow handshake to peer over the shared link and
// blocks for its CatchupInfo response (the target's log identity). The request
// payload carries the growing epoch (informational). It is synchronous — the grow
// driver runs it off the write path.
func (s *shardTransport) CatchupRequest(peer string, epoch uint64) (CatchupInfoMsg, error) {
	f := &pbFrame{
		kind:    pbKindCatchupReq,
		shard:   s.shard,
		payload: encodeAckMsg(pbGetPayload(), AckMsg{Epoch: epoch}),
		pooled:  true,
	}
	// roundTrip recycles the pooled payload on submit failure; on success the
	// writer recycles it after the wire flush. The response payload is owned here.
	resp, err := s.t.linkFor(peer).roundTrip(f, pbCatchupTimeout)
	if err != nil {
		return CatchupInfoMsg{}, err
	}
	return decodeCatchupInfo(resp)
}

var _ SnapshotTransport = (*shardTransport)(nil)

// pbSnapshotChunkTimeout bounds ONE snapshot chunk's round trip. It is far more
// generous than pbCatchupTimeout because the FINAL chunk's response is only
// produced AFTER the target has wiped and re-installed its whole FSM
// synchronously — that is disk work proportional to shard size, not a network
// round trip. A transfer that exceeds it aborts cleanly (the target's staging
// buffer is memory-only, so nothing is left half-installed).
const pbSnapshotChunkTimeout = 120 * time.Second

// SendSnapshotChunk ships one snapshot chunk to peer over the shared link and
// blocks for its ack. Synchronous by design: it is driven by the grow driver's
// own goroutine, off both engine locks, and must never enter the non-blocking
// learner channel that abandons a grow when full.
func (s *shardTransport) SendSnapshotChunk(peer string, c SnapshotChunk) (AckMsg, error) {
	f := &pbFrame{
		kind:    pbKindSnapshotChunk,
		shard:   s.shard,
		payload: encodeSnapshotChunk(pbGetPayload(), c),
		pooled:  true,
	}
	resp, err := s.t.linkFor(peer).roundTrip(f, pbSnapshotChunkTimeout)
	if err != nil {
		return AckMsg{}, err
	}
	return decodeAckMsg(resp)
}

// TryReplicate is the non-blocking inline submit (lever 1): it attempts the
// same submission as Replicate via the link's no-dial, no-block fast path.
// On refusal the pooled payload is recycled and false is returned — done will
// not fire; the caller falls back to the ordered sender path.
func (s *shardTransport) TryReplicate(peer string, msg ReplicateMsg, done func(AckMsg, error)) bool {
	f := &pbFrame{
		kind:    pbKindReplicate,
		shard:   s.shard,
		payload: encodeReplicateMsg(pbGetPayload(), msg),
		pooled:  true,
	}
	ok := s.t.linkFor(peer).trySubmitAsync(f, func(p []byte, cbErr error) {
		if cbErr != nil {
			done(AckMsg{}, cbErr)
			return
		}
		ack, derr := decodeAckMsg(p)
		done(ack, derr)
	})
	if !ok && f.pooled {
		pbPutPayload(f.payload)
	}
	return ok
}

// ReplicateGroup submits a uniform-epoch, seq-dense group of writes as ONE
// frame over the peer's shared pipelined link, answered by ONE cumulative
// ack. The payload is encoded synchronously before returning, so the
// caller may reuse msgs' backing array immediately after the call. Completion
// and error semantics are identical to Replicate.
func (s *shardTransport) ReplicateGroup(peer string, msgs []ReplicateMsg, done func(AckMsg, error)) error {
	f := &pbFrame{
		kind:    pbKindReplicateGroup,
		shard:   s.shard,
		payload: encodeReplicateGroup(pbGetPayload(), msgs),
		pooled:  true,
	}
	err := s.t.linkFor(peer).submitAsync(f, func(p []byte, cbErr error) {
		if cbErr != nil {
			done(AckMsg{}, cbErr)
			return
		}
		ack, derr := decodeAckMsg(p)
		done(ack, derr)
	})
	if err != nil {
		if f.pooled {
			pbPutPayload(f.payload)
		}
		return err
	}
	return nil
}

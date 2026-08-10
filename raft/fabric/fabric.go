// SPDX-License-Identifier: Apache-2.0

package fabric

import (
	"bufio"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/tlsutil"
)

// Connection-type handshake bytes written by the initiator right after dial,
// before any frame. connMux runs the group-agnostic multiplexed frame loop;
// connSnapshot is a one-shot InstallSnapshot exchange on its own conn so a large
// snapshot cannot head-of-line-block the shared link.
const (
	connMux      byte = 0x01
	connSnapshot byte = 0x02
)

const (
	dialTimeout    = 10 * time.Second
	rpcTimeout     = 10 * time.Second
	sendChanBuffer = 128
	respChanBuffer = 128
	inboundBuffer  = 64
	consumerBuffer = 128
	readBufSize    = 256 << 10
	writeBufSize   = 256 << 10
)

var (
	errFabricClosed = errors.New("fabric: transport closed")
	errLinkClosed   = errors.New("fabric: peer link closed")
	errRPCTimeout   = errors.New("fabric: rpc timeout")
)

// result carries a response payload (struct bytes, application-error prefix
// intact) back to a waiter, or a transport-level error (link failure/timeout).
type result struct {
	payload []byte
	err     error
}

// group holds one Raft group's inbound Consumer channel and heartbeat handler.
type group struct {
	id        uint32
	consumeCh chan hraft.RPC

	hbMu      sync.Mutex
	hbHandler func(hraft.RPC)

	closeOnce sync.Once
	closed    chan struct{}
}

// Fabric is the per-node transport: it owns one TCP listener (the node's Raft
// address) and, per peer, one shared multiplexed outbound link carrying every
// group's requests. For(groupID) yields a per-group hraft.Transport facade.
type Fabric struct {
	ln        net.Listener
	localAddr string

	// clientTLS, when non-nil, upgrades outbound peer dials (the shared mux link
	// and the one-shot snapshot conn) to TLS. cnAllow is the OPT-IN per-node
	// identity allowlist enforced on accepted conns after the handshake and pinned
	// on the peer server cert when dialing. Both nil/empty ⇒ plaintext,
	// byte-identical to the historical path. See [New].
	clientTLS *tls.Config
	cnAllow   map[string]bool

	reqID atomic.Uint64

	// droppedInbound counts request frames dropped because a group's inbound
	// queue was full (its Raft loop was backed up). Dropping is safe — Raft
	// retries the AppendEntries/vote — and it keeps one stalled group from
	// head-of-line-blocking every other group multiplexed on the same peer conn.
	droppedInbound atomic.Uint64

	mu     sync.Mutex
	links  map[string]*peerLink  // by target addr
	conns  map[net.Conn]struct{} // accepted conns, for shutdown
	groups map[uint32]*group

	closeOnce sync.Once
	closed    chan struct{}
}

// inbound is a decoded request ready to dispatch to a group's Raft loop. Decoding
// happens once, in the per-conn reader, so heartbeats can be served on a fast
// path that bypasses the per-group queue entirely.
type inbound struct {
	kind    uint8
	groupID uint32
	reqID   uint64
	rpcType uint8
	cmd     any
	isHB    bool
}

// New binds addr and starts the accept loop. groups is the full set of group IDs
// this node participates in (shard IDs plus the meta group); For panics on an
// unregistered ID, matching raft/mux.
//
// serverTLS/clientTLS/cnAllow are the OPT-IN inter-node mTLS parameters: all
// nil/empty is the default plaintext path, byte-identical to before. When
// serverTLS is non-nil the listener is wrapped with [tls.NewListener] (mTLS via
// RequireAndVerifyClientCert) and every accepted conn's verified client-cert CN
// is checked against cnAllow before any frame is served; when clientTLS is
// non-nil the outbound mux/snapshot dials upgrade to TLS.
func New(addr string, groups []uint32, serverTLS, clientTLS *tls.Config, cnAllow map[string]bool) (*Fabric, error) {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("fabric: listen %s: %w", addr, err)
	}
	// TLS-wrap the listener so accepted conns complete an mTLS handshake before we
	// read the connection-type byte or dispatch any frame. nil serverTLS ⇒ ln
	// stays the plaintext listener (unchanged).
	if serverTLS != nil {
		ln = tls.NewListener(ln, serverTLS)
	}
	f := &Fabric{
		ln:        ln,
		localAddr: ln.Addr().String(),
		clientTLS: clientTLS,
		cnAllow:   cnAllow,
		links:     make(map[string]*peerLink),
		conns:     make(map[net.Conn]struct{}),
		groups:    make(map[uint32]*group, len(groups)),
		closed:    make(chan struct{}),
	}
	for _, g := range groups {
		f.groups[g] = &group{
			id:        g,
			consumeCh: make(chan hraft.RPC, consumerBuffer),
			closed:    make(chan struct{}),
		}
	}
	go f.acceptLoop()
	return f, nil
}

// Addr returns the listener's network address.
func (f *Fabric) Addr() net.Addr { return f.ln.Addr() }

// For returns the per-group transport facade for groupID.
func (f *Fabric) For(groupID uint32) hraft.Transport {
	f.mu.Lock()
	g, ok := f.groups[groupID]
	f.mu.Unlock()
	if !ok {
		panic(fmt.Sprintf("fabric: group %d not registered", groupID))
	}
	return &groupTransport{fabric: f, g: g, timeout: rpcTimeout}
}

// Close stops the listener, tears down every outbound link and accepted conn,
// and signals all groups. Idempotent.
func (f *Fabric) Close() error {
	f.closeOnce.Do(func() {
		close(f.closed)
		_ = f.ln.Close()
		f.mu.Lock()
		links := f.links
		f.links = make(map[string]*peerLink)
		conns := f.conns
		f.conns = make(map[net.Conn]struct{})
		groups := f.groups
		f.mu.Unlock()
		for _, l := range links {
			l.fail(errFabricClosed)
		}
		for c := range conns {
			_ = c.Close()
		}
		for _, g := range groups {
			g.closeOnce.Do(func() { close(g.closed) })
		}
	})
	return nil
}

// DroppedInbound returns the number of request frames dropped because a group's
// inbound queue was full. Non-zero under sustained overload is expected and
// safe (Raft retries); a steadily climbing value flags a chronically stalled
// group. Exposed for observability and tests rather than silently swallowed.
func (f *Fabric) DroppedInbound() uint64 { return f.droppedInbound.Load() }

func (f *Fabric) isClosed() bool {
	select {
	case <-f.closed:
		return true
	default:
		return false
	}
}

func (f *Fabric) nextReqID() uint64 { return f.reqID.Add(1) }

// getLink returns the shared outbound link for target, creating an
// (undialed) entry lazily. Dialing happens under the link's own lock so distinct
// targets connect concurrently and one slow dial never blocks other peers.
func (f *Fabric) getLink(target string) *peerLink {
	f.mu.Lock()
	defer f.mu.Unlock()
	if l := f.links[target]; l != nil {
		return l
	}
	l := &peerLink{fabric: f, target: target}
	f.links[target] = l
	return l
}

// roundTrip sends fr as a request over target's shared link and waits for the
// correlated response (or timeout / link failure).
func (f *Fabric) roundTrip(target string, fr *frame, timeout time.Duration) ([]byte, error) {
	return f.getLink(target).roundTrip(fr, timeout)
}

// acceptLoop accepts inbound conns, reads the one-byte connection-type
// handshake, and dispatches mux vs snapshot handling.
func (f *Fabric) acceptLoop() {
	const baseDelay = 5 * time.Millisecond
	const maxDelay = time.Second
	var delay time.Duration
	for {
		conn, err := f.ln.Accept()
		if err != nil {
			if f.isClosed() {
				return
			}
			if delay == 0 {
				delay = baseDelay
			} else {
				delay *= 2
			}
			if delay > maxDelay {
				delay = maxDelay
			}
			select {
			case <-f.closed:
				return
			case <-time.After(delay):
				continue
			}
		}
		delay = 0
		go f.handleAccepted(conn)
	}
}

func (f *Fabric) handleAccepted(conn net.Conn) {
	// Track for shutdown; bail if already closing.
	f.mu.Lock()
	if f.isClosed() {
		f.mu.Unlock()
		_ = conn.Close()
		return
	}
	f.conns[conn] = struct{}{}
	f.mu.Unlock()
	defer f.dropConn(conn)

	// Trust boundary: on a TLS listener the conn is unauthenticated until the
	// handshake completes. Force it and pin the verified client-cert CN to the
	// allowlist BEFORE reading the connection-type byte or serving any frame, so
	// only an authenticated, allowlisted peer can drive Raft traffic. Plaintext
	// listener (nil serverTLS) ⇒ no-op, byte-identical to before.
	if err := f.authenticatePeer(conn); err != nil {
		return
	}

	var hb [1]byte
	_ = conn.SetReadDeadline(time.Now().Add(dialTimeout))
	if _, err := io.ReadFull(conn, hb[:]); err != nil {
		return
	}
	_ = conn.SetReadDeadline(time.Time{})
	switch hb[0] {
	case connMux:
		f.serveMuxConn(conn)
	case connSnapshot:
		f.serveSnapshotConn(conn)
	}
}

func (f *Fabric) dropConn(conn net.Conn) {
	f.mu.Lock()
	delete(f.conns, conn)
	f.mu.Unlock()
	_ = conn.Close()
}

// authenticatePeer completes the mTLS handshake on an accepted inter-node conn
// and enforces the per-node CN allowlist. On a plaintext listener (conn is not a
// *tls.Conn) it is a no-op passthrough. A non-nil return means the peer is NOT an
// authenticated, allowlisted cluster member; handleAccepted then drops the conn
// (via its deferred dropConn) without serving it.
func (f *Fabric) authenticatePeer(conn net.Conn) error {
	tc, ok := conn.(*tls.Conn)
	if !ok {
		return nil // plaintext listener: nothing to authenticate
	}
	_ = tc.SetDeadline(time.Now().Add(dialTimeout))
	if err := tc.Handshake(); err != nil {
		return err
	}
	_ = tc.SetDeadline(time.Time{})
	// Authenticated: the peer cert chained to the cluster CA (server cfg required
	// and verified it). The allowlist adds identity pinning (empty ⇒ any CA-valid
	// peer, see tlsutil.PeerCNAllowed).
	return tlsutil.PeerCNAllowed(tc.ConnectionState(), f.cnAllow)
}

// dialConn dials target, upgrading to TLS when clientTLS is configured. The
// plaintext path (nil clientTLS) is byte-identical to net.DialTimeout. The TLS
// path pins the peer host as ServerName and, when an allowlist is set, the peer's
// verified server-cert CN — mutual identity in both directions of the link.
func (f *Fabric) dialConn(target string, timeout time.Duration) (net.Conn, error) {
	if f.clientTLS == nil {
		return net.DialTimeout("tcp", target, timeout)
	}
	d := &net.Dialer{Timeout: timeout}
	return tls.DialWithDialer(d, "tcp", target, peerClientTLS(f.clientTLS, target, f.cnAllow))
}

// peerClientTLS builds the per-peer client *tls.Config for an outbound inter-node
// dial: base cloned, ServerName pinned to target's host (peer server cert verified
// against its SAN — never InsecureSkipVerify), and (when allow is non-empty) a
// VerifyConnection pinning the peer's verified server-cert CN. Fail-closed.
func peerClientTLS(base *tls.Config, target string, allow map[string]bool) *tls.Config {
	cfg := base.Clone()
	host, _, err := net.SplitHostPort(target)
	if err != nil {
		host = target
	}
	cfg.ServerName = host
	if len(allow) > 0 {
		cfg.VerifyConnection = func(cs tls.ConnectionState) error {
			return tlsutil.PeerCNAllowed(cs, allow)
		}
	}
	return cfg
}

// serveMuxConn reads request frames off one accepted mux conn and dispatches
// each to a per-group worker so groups process concurrently (no cross-group
// head-of-line blocking) while a single group's requests stay strictly ordered
// (matching hashicorp's one-request-at-a-time-per-conn semantics). Responses
// from all workers funnel through one batched writer.
func (f *Fabric) serveMuxConn(conn net.Conn) {
	respCh := make(chan *frame, respChanBuffer)
	connDone := make(chan struct{})
	var writerOnce sync.Once
	closeConn := func() { writerOnce.Do(func() { _ = conn.Close() }) }

	go func() {
		// Write straight to the conn: runFramedWriter batches responses via
		// net.Buffers.WriteTo (writev), so no intermediate bufio copy is needed.
		_ = runFramedWriter(conn, respCh, connDone, writeLinger)
		closeConn() // unblock the reader if the writer died first
	}()
	defer close(connDone)

	queues := make(map[uint32]chan *inbound)
	fr := frameReader{r: bufio.NewReaderSize(conn, readBufSize)}
	for {
		msg, err := fr.read()
		if err != nil {
			closeConn()
			return
		}
		if msg.isResponse() {
			continue // an initiator's mux conn only carries inbound requests
		}
		it, ok := decodeInbound(&msg)
		if !ok {
			continue // undecodable request frame: drop, Raft retries
		}
		g := f.group(it.groupID)
		if g == nil {
			continue // unregistered group: drop (also bounds worker creation)
		}
		// Heartbeat fast path: serve inline in the reader so a cheap heartbeat is
		// never queued behind an AppendEntries backlog (which could delay it past
		// the election timeout and trip a spurious election). Mirrors hashicorp's
		// out-of-band heartbeat handler.
		if it.isHB && g.heartbeatHandler() != nil {
			f.serveInbound(g, it, respCh, connDone)
			continue
		}
		q := queues[it.groupID]
		if q == nil {
			q = make(chan *inbound, inboundBuffer)
			queues[it.groupID] = q
			go f.groupWorker(g, q, respCh, connDone)
		}
		select {
		case q <- it:
		case <-connDone:
			return
		case <-f.closed:
			return
		default:
			// This group's Raft loop is backed up (queue full). Drop rather than
			// block the shared reader and stall every other group on this peer
			// conn — Raft retries the dropped AppendEntries/vote. Equivalent to
			// TCP-loss backpressure on hashicorp's per-group connections.
			f.droppedInbound.Add(1)
		}
	}
}

// group returns the registered group for id, or nil.
func (f *Fabric) group(id uint32) *group {
	f.mu.Lock()
	g := f.groups[id]
	f.mu.Unlock()
	return g
}

func (g *group) heartbeatHandler() func(hraft.RPC) {
	g.hbMu.Lock()
	fn := g.hbHandler
	g.hbMu.Unlock()
	return fn
}

// decodeInbound decodes a request frame into a dispatch-ready inbound, detecting
// the AppendEntries heartbeat case. Returns ok=false on an undecodable payload.
func decodeInbound(fr *frame) (*inbound, bool) {
	it := &inbound{kind: fr.kind, groupID: fr.groupID, reqID: fr.reqID, rpcType: fr.rpcType()}
	switch it.rpcType {
	case rpcAppendEntries:
		var req hraft.AppendEntriesRequest
		if err := decodeAppendEntriesRequest(fr.payload, &req); err != nil {
			return nil, false
		}
		it.cmd = &req
		it.isHB = isHeartbeat(&req)
	case rpcRequestVote:
		var req hraft.RequestVoteRequest
		if err := decodeRequestVoteRequest(fr.payload, &req); err != nil {
			return nil, false
		}
		it.cmd = &req
	case rpcRequestPreVote:
		var req hraft.RequestPreVoteRequest
		if err := decodeRequestPreVoteRequest(fr.payload, &req); err != nil {
			return nil, false
		}
		it.cmd = &req
	case rpcTimeoutNow:
		var req hraft.TimeoutNowRequest
		if err := decodeTimeoutNowRequest(fr.payload, &req); err != nil {
			return nil, false
		}
		it.cmd = &req
	default:
		return nil, false
	}
	return it, true
}

// groupWorker serially serves one group's non-heartbeat request stream from a
// single conn, preserving per-group AppendEntries delivery order.
func (f *Fabric) groupWorker(g *group, q <-chan *inbound, respCh chan<- *frame, connDone <-chan struct{}) {
	for {
		select {
		case it := <-q:
			f.serveInbound(g, it, respCh, connDone)
		case <-connDone:
			return
		case <-f.closed:
			return
		}
	}
}

// serveInbound dispatches one decoded request (heartbeat fast-path handler or the
// group Consumer), waits for the Raft response, and enqueues the response frame.
func (f *Fabric) serveInbound(g *group, it *inbound, respCh chan<- *frame, connDone <-chan struct{}) {
	respRPC := make(chan hraft.RPCResponse, 1)
	rpc := hraft.RPC{Command: it.cmd, RespChan: respRPC}

	dispatched := false
	if it.isHB {
		if fn := g.heartbeatHandler(); fn != nil {
			fn(rpc)
			dispatched = true
		}
	}
	if !dispatched {
		select {
		case g.consumeCh <- rpc:
		case <-connDone:
			return
		case <-g.closed:
			return
		case <-f.closed:
			return
		}
	}

	select {
	case rr := <-respRPC:
		out := &frame{
			kind:    it.kind | kindResponse,
			groupID: it.groupID,
			reqID:   it.reqID,
			payload: encodeResponseFrame(getPayload(), it.rpcType, rr),
			pooled:  true, // recycled by runFramedWriter after the respCh frame is written
		}
		select {
		case respCh <- out:
		case <-connDone:
		case <-f.closed:
		}
	case <-connDone:
	case <-f.closed:
	}
}

// serveSnapshotConn handles the one-shot InstallSnapshot exchange on a dedicated
// conn: read the request frame, hand the streamed body to Raft as rpc.Reader,
// then write the response frame.
func (f *Fabric) serveSnapshotConn(conn net.Conn) {
	r := bufio.NewReaderSize(conn, readBufSize)
	fr := frameReader{r: r}
	msg, err := fr.read()
	if err != nil || msg.rpcType() != rpcInstallSnapshot {
		return
	}
	var req hraft.InstallSnapshotRequest
	if err := decodeInstallSnapshotRequest(msg.payload, &req); err != nil {
		return
	}
	f.mu.Lock()
	g := f.groups[msg.groupID]
	f.mu.Unlock()
	if g == nil {
		return
	}
	respRPC := make(chan hraft.RPCResponse, 1)
	rpc := hraft.RPC{
		Command:  &req,
		Reader:   io.LimitReader(r, req.Size),
		RespChan: respRPC,
	}
	select {
	case g.consumeCh <- rpc:
	case <-g.closed:
		return
	case <-f.closed:
		return
	}
	select {
	case rr := <-respRPC:
		bw := bufio.NewWriterSize(conn, writeBufSize)
		out := &frame{
			kind:    msg.kind | kindResponse,
			groupID: msg.groupID,
			reqID:   msg.reqID,
			payload: encodeResponseFrame(getPayload(), rpcInstallSnapshot, rr),
		}
		if writeFrame(bw, out) == nil {
			_ = bw.Flush()
		}
		// This one-shot snapshot response uses writeFrame directly (not
		// runFramedWriter), so it can't ride the batcher's recycle — return the
		// pooled buffer inline now that writeFrame/Flush have consumed its bytes.
		// out.pooled stays false so nothing double-returns it.
		putPayload(out.payload)
	case <-f.closed:
	}
}

// isHeartbeat mirrors hashicorp's net_transport heartbeat detection: an
// AppendEntries with a term, a leader, no previous-log info, no entries, and a
// zero commit index.
func isHeartbeat(req *hraft.AppendEntriesRequest) bool {
	leaderAddr := req.Addr
	if len(leaderAddr) == 0 {
		leaderAddr = req.Leader //nolint:staticcheck // SA1019: fallback for peers that still populate the deprecated field
	}
	return req.Term != 0 && leaderAddr != nil &&
		req.PrevLogEntry == 0 && req.PrevLogTerm == 0 &&
		len(req.Entries) == 0 && req.LeaderCommitIndex == 0
}

// encodeResponseFrame encodes the Raft RPCResponse for rpcType into a payload
// with the application-error prefix, APPENDING into b (pass getPayload() to pool
// the buffer, or nil for a fresh one). On rr.Error it carries the error string;
// on success it carries the response struct.
func encodeResponseFrame(b []byte, rpcType uint8, rr hraft.RPCResponse) []byte {
	appErr := ""
	if rr.Error != nil {
		appErr = rr.Error.Error()
	}
	switch rpcType {
	case rpcAppendEntries:
		resp, _ := rr.Response.(*hraft.AppendEntriesResponse)
		if resp == nil {
			resp = &hraft.AppendEntriesResponse{}
		}
		return encodeAppendEntriesResponse(b, appErr, resp)
	case rpcRequestVote:
		resp, _ := rr.Response.(*hraft.RequestVoteResponse)
		if resp == nil {
			resp = &hraft.RequestVoteResponse{}
		}
		return encodeRequestVoteResponse(b, appErr, resp)
	case rpcRequestPreVote:
		resp, _ := rr.Response.(*hraft.RequestPreVoteResponse)
		if resp == nil {
			resp = &hraft.RequestPreVoteResponse{}
		}
		return encodeRequestPreVoteResponse(b, appErr, resp)
	case rpcTimeoutNow:
		resp, _ := rr.Response.(*hraft.TimeoutNowResponse)
		if resp == nil {
			resp = &hraft.TimeoutNowResponse{}
		}
		return encodeTimeoutNowResponse(b, appErr, resp)
	case rpcInstallSnapshot:
		resp, _ := rr.Response.(*hraft.InstallSnapshotResponse)
		if resp == nil {
			resp = &hraft.InstallSnapshotResponse{}
		}
		return encodeInstallSnapshotResponse(b, appErr, resp)
	default:
		return b
	}
}

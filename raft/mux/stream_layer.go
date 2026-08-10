// SPDX-License-Identifier: Apache-2.0

package mux

import (
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/tlsutil"
)

const acceptQueueSize = 64

// handshakeTimeout bounds the inter-node TLS handshake on an accepted conn so a
// peer that opens a socket but never completes the handshake cannot pin an
// accept goroutine indefinitely.
const handshakeTimeout = 10 * time.Second

// StreamLayer multiplexes N Raft groups over a single TCP listener.
// Use [New] to create one, then call [StreamLayer.For] to obtain a
// per-group [raft.StreamLayer] for each Raft instance.
//
// Inter-node TLS is OPT-IN and OFF by default (serverTLS/clientTLS nil ⇒
// byte-identical plaintext, the historical path): when a server *tls.Config is
// supplied the listener is TLS-wrapped (mTLS via RequireAndVerifyClientCert) and
// every accepted conn's verified client-cert CN is checked against cnAllow BEFORE
// any group frame is served; when a client *tls.Config is supplied outbound dials
// upgrade to TLS with the peer's host pinned as ServerName.
type StreamLayer struct {
	ln     net.Listener
	dialer *net.Dialer

	// clientTLS, when non-nil, upgrades outbound Dials to TLS. cnAllow is the
	// OPT-IN per-node identity allowlist enforced on accepted (server-side) conns
	// after the handshake, and additionally pinned on the peer's server cert when
	// dialing. Both nil/empty ⇒ plaintext, byte-identical to before.
	clientTLS *tls.Config
	cnAllow   map[string]bool

	groups   map[uint32]*groupChans
	mu       sync.RWMutex
	closing  chan struct{}
	closeOne sync.Once
}

type groupChans struct {
	conns chan net.Conn
}

// New creates a StreamLayer that listens on addr and routes accepted
// connections to the registered group IDs. It starts the accept loop
// immediately; call [StreamLayer.Close] to stop it.
//
// serverTLS/clientTLS/cnAllow are the OPT-IN inter-node mTLS parameters (see the
// [StreamLayer] doc): all nil/empty is the default plaintext path, byte-identical
// to before. When serverTLS is non-nil the listener is wrapped with
// [tls.NewListener] so every accepted conn is a *tls.Conn whose client cert the
// handshake verifies against serverTLS's ClientCAs; when clientTLS is non-nil
// outbound dials upgrade to TLS.
func New(addr string, groups []uint32, serverTLS, clientTLS *tls.Config, cnAllow map[string]bool) (*StreamLayer, error) {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("mux: listen %s: %w", addr, err)
	}
	// Wrap the raw TCP listener so accepted conns complete an mTLS handshake
	// before we route any group frame. nil serverTLS ⇒ ln stays the plaintext
	// listener (unchanged). tls.NewListener defers the handshake to first I/O;
	// dispatchAccepted forces it explicitly so the peer CN is known up front.
	if serverTLS != nil {
		ln = tls.NewListener(ln, serverTLS)
	}
	s := &StreamLayer{
		ln:        ln,
		dialer:    &net.Dialer{},
		clientTLS: clientTLS,
		cnAllow:   cnAllow,
		groups:    make(map[uint32]*groupChans, len(groups)),
		closing:   make(chan struct{}),
	}
	for _, g := range groups {
		s.groups[g] = &groupChans{conns: make(chan net.Conn, acceptQueueSize)}
	}
	go s.acceptLoop()
	return s, nil
}

// For returns the [raft.StreamLayer] for the given group ID.
// It panics if groupID was not registered at construction time.
func (s *StreamLayer) For(groupID uint32) raft.StreamLayer {
	s.mu.RLock()
	defer s.mu.RUnlock()
	gc, ok := s.groups[groupID]
	if !ok {
		panic(fmt.Sprintf("mux: group %d not registered", groupID))
	}
	return &groupLayer{groupID: groupID, s: s, gc: gc}
}

// Addr returns the listener's network address.
func (s *StreamLayer) Addr() net.Addr { return s.ln.Addr() }

// Close shuts down the listener and drains all per-group queues.
// It is safe to call multiple times (idempotent via sync.Once).
func (s *StreamLayer) Close() error {
	var err error
	s.closeOne.Do(func() {
		close(s.closing)
		err = s.ln.Close()
		// Do NOT close gc.conns — dispatchAccepted goroutines may be
		// mid-select on those channels; closing them would cause a panic.
		// Instead, drain any conns that already landed in the buffer.
		s.mu.Lock()
		for _, gc := range s.groups {
		drainLoop:
			for {
				select {
				case c := <-gc.conns:
					_ = c.Close()
				default:
					break drainLoop
				}
			}
		}
		s.mu.Unlock()
	})
	return err
}

func (s *StreamLayer) acceptLoop() {
	for {
		conn, err := s.ln.Accept()
		if err != nil {
			select {
			case <-s.closing:
				return
			default:
				// Transient or persistent error (e.g. EMFILE): short backoff
				// to avoid burning CPU before retrying.
				time.Sleep(5 * time.Millisecond)
				continue
			}
		}
		go s.dispatchAccepted(conn)
	}
}

func (s *StreamLayer) dispatchAccepted(c net.Conn) {
	// Trust boundary: on a TLS listener the accepted conn is unauthenticated
	// until the handshake completes. Force it here and pin the verified client-
	// cert CN to the allowlist BEFORE reading the group ID or routing any frame,
	// so a peer that is not an authenticated, allowlisted cluster member can never
	// inject a replicate/Raft frame. Plaintext listener (nil serverTLS) ⇒ the
	// assertion fails and this is a no-op, byte-identical to before.
	if err := s.authenticatePeer(c); err != nil {
		_ = c.Close()
		return
	}
	_ = c.SetReadDeadline(time.Now().Add(5 * time.Second))
	id, err := readGroupID(c)
	_ = c.SetReadDeadline(time.Time{})
	if err != nil {
		_ = c.Close()
		return
	}
	s.mu.RLock()
	gc, ok := s.groups[id]
	s.mu.RUnlock()
	if !ok {
		_ = c.Close()
		return
	}
	select {
	case gc.conns <- c:
	case <-s.closing:
		_ = c.Close()
	}
}

type groupLayer struct {
	groupID uint32
	s       *StreamLayer
	gc      *groupChans
}

// authenticatePeer completes the mTLS handshake on an accepted inter-node conn
// and enforces the per-node CN allowlist. On a plaintext listener (c is not a
// *tls.Conn) it is a no-op passthrough, keeping the historical path unchanged.
// A non-nil return means the peer is NOT an authenticated, allowlisted cluster
// member and the caller MUST close the conn without serving it.
func (s *StreamLayer) authenticatePeer(c net.Conn) error {
	tc, ok := c.(*tls.Conn)
	if !ok {
		return nil // plaintext listener: nothing to authenticate
	}
	_ = tc.SetDeadline(time.Now().Add(handshakeTimeout))
	if err := tc.Handshake(); err != nil {
		return err
	}
	_ = tc.SetDeadline(time.Time{})
	// Past this point the peer's cert chained to the cluster CA (the server cfg
	// used RequireAndVerifyClientCert): the conn is now authenticated. The
	// allowlist adds identity pinning (empty ⇒ any CA-valid peer, see PeerCNAllowed).
	return tlsutil.PeerCNAllowed(tc.ConnectionState(), s.cnAllow)
}

// dialConn dials target, upgrading to TLS when clientTLS is configured. The
// plaintext path (nil clientTLS) is byte-identical to the historical
// d.Dial("tcp", target). The TLS path clones clientTLS per peer, pins the peer's
// host as ServerName (so the peer server cert is verified against its SAN — never
// InsecureSkipVerify), and, when an allowlist is set, additionally pins the peer's
// verified server-cert CN (mirroring cluster.peerClient, so identity is enforced
// in BOTH directions of the mutually-authenticated link). tls.DialWithDialer
// completes the handshake within the dialer's timeout before returning.
func (s *StreamLayer) dialConn(d *net.Dialer, target string) (net.Conn, error) {
	if s.clientTLS == nil {
		return d.Dial("tcp", target)
	}
	return tls.DialWithDialer(d, "tcp", target, peerClientTLS(s.clientTLS, target, s.cnAllow))
}

// peerClientTLS builds the per-peer client *tls.Config for an outbound inter-node
// dial: base cloned, ServerName pinned to target's host, and (when allow is
// non-empty) a VerifyConnection that pins the peer's verified server-cert CN to
// the allowlist. Fail-closed: an unlisted CN aborts the handshake before any
// frame is written.
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

func (g *groupLayer) Dial(addr raft.ServerAddress, timeout time.Duration) (net.Conn, error) {
	d := *g.s.dialer
	d.Timeout = timeout
	c, err := g.s.dialConn(&d, string(addr))
	if err != nil {
		return nil, err
	}
	_ = c.SetWriteDeadline(time.Now().Add(timeout))
	if err := writeGroupID(c, g.groupID); err != nil {
		_ = c.Close()
		return nil, fmt.Errorf("mux: write group id: %w", err)
	}
	_ = c.SetWriteDeadline(time.Time{}) // clear for subsequent Raft writes
	return c, nil
}

func (g *groupLayer) Accept() (net.Conn, error) {
	select {
	case c := <-g.gc.conns:
		return c, nil
	case <-g.s.closing:
		return nil, errors.New("mux: stream layer closed")
	}
}

func (g *groupLayer) Close() error { return nil }

func (g *groupLayer) Addr() net.Addr { return g.s.Addr() }

var _ raft.StreamLayer = (*groupLayer)(nil)

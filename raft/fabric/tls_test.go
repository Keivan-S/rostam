// SPDX-License-Identifier: Apache-2.0

package fabric

import (
	"crypto/tls"
	"errors"
	"net"
	"os"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/tlsutil"
	"github.com/rostamlabs/rostam/tlsutil/testcerts"
)

// NOTE ON SCOPE: these tests exercise the inter-node mTLS trust boundary this
// change adds — the TLS handshake on the accepted conn, the post-handshake CN
// allowlist gate in handleAccepted, and the TLS upgrade of the outbound dial
// (dialConn). They assert at the connection level (is the authenticated conn kept
// and served, or rejected and closed) rather than driving a full AppendEntries
// application round trip: fabric's client link buffers small frames through a
// bufio writer whose net.Buffers.WriteTo path does not flush per request (a
// pre-existing property of this experimental, non-default transport, orthogonal to
// TLS), so a single-request round trip does not complete regardless of TLS. The
// security-relevant surface — who is allowed to reach serveMuxConn — is fully
// covered here. The mux (default transport) and pbisr suites cover full
// application round trips over TLS.

const tlsTestGroup uint32 = 3

// fabricTLSConfigs builds the strict-mTLS server + client configs for a node whose
// identity cert CN is nodeCN (both EKUs, loopback SANs), signed off ca — mirroring
// how cmd/rostam-server wires the inter-node replication transports.
func fabricTLSConfigs(t *testing.T, ca *testcerts.CA, nodeCN string) (server, client *tls.Config) {
	t.Helper()
	certFile, keyFile := ca.NodeCert(t, nodeCN)
	server, err := tlsutil.ServerTLS(certFile, keyFile, ca.CAFile, true)
	if err != nil {
		t.Fatalf("ServerTLS: %v", err)
	}
	client, err = tlsutil.ClientTLS(ca.CAFile, certFile, keyFile, "")
	if err != nil {
		t.Fatalf("ClientTLS: %v", err)
	}
	return server, client
}

// waitConns polls the server's tracked-conn count until it equals want or the
// deadline elapses, returning the last observed count. An accepted conn stays in
// f.conns only while handleAccepted is serving it (past the authenticate gate); a
// rejected conn is removed by the deferred dropConn.
func waitConns(f *Fabric, want int, d time.Duration) int {
	deadline := time.Now().Add(d)
	for {
		f.mu.Lock()
		n := len(f.conns)
		f.mu.Unlock()
		if n == want || time.Now().After(deadline) {
			return n
		}
		time.Sleep(2 * time.Millisecond)
	}
}

// assertKeptOpen writes the mux handshake byte and asserts the server keeps the
// conn open (a read blocks until our short deadline rather than seeing EOF) — the
// proof that the peer was authenticated and handed to serveMuxConn.
func assertKeptOpen(t *testing.T, conn net.Conn) {
	t.Helper()
	if _, err := conn.Write([]byte{connMux}); err != nil {
		t.Fatalf("write connMux: %v", err)
	}
	_ = conn.SetReadDeadline(time.Now().Add(300 * time.Millisecond))
	_, err := conn.Read(make([]byte, 1))
	if err == nil {
		t.Fatal("unexpected data from server on an idle authenticated conn")
	}
	if !errors.Is(err, os.ErrDeadlineExceeded) {
		t.Fatalf("authenticated conn should stay OPEN (read timeout), but got %v", err)
	}
}

// assertClosed writes the mux handshake byte and asserts the server closes the
// conn (a read returns a non-timeout error, i.e. EOF/reset) — the proof that the
// peer was rejected before being served.
func assertClosed(t *testing.T, conn net.Conn) {
	t.Helper()
	_, _ = conn.Write([]byte{connMux})
	_ = conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	_, err := conn.Read(make([]byte, 1))
	if err == nil {
		t.Fatal("rejected peer's conn should be closed, but a read succeeded")
	}
	if errors.Is(err, os.ErrDeadlineExceeded) {
		t.Fatalf("rejected peer's conn should be CLOSED, but the read timed out (still open): %v", err)
	}
}

// TestFabricTLSHandshakeAccepted: with matching-CA strict mTLS and an allowlisted
// client CN, the outbound TLS dial handshakes and the server accepts + serves the
// conn (authenticated), keeping it open.
func TestFabricTLSHandshakeAccepted(t *testing.T) {
	ca := testcerts.GenCA(t)
	allow := map[string]bool{"n1": true, "n2": true}
	srvS, _ := fabricTLSConfigs(t, ca, "n2")
	_, cliC := fabricTLSConfigs(t, ca, "n1")
	server, err := New("127.0.0.1:0", []uint32{tlsTestGroup}, srvS, nil, allow)
	if err != nil {
		t.Fatalf("server New: %v", err)
	}
	defer server.Close()
	client, err := New("127.0.0.1:0", []uint32{tlsTestGroup}, nil, cliC, allow)
	if err != nil {
		t.Fatalf("client New: %v", err)
	}
	defer client.Close()

	conn, err := client.dialConn(server.Addr().String(), 3*time.Second)
	if err != nil {
		t.Fatalf("TLS dialConn: %v", err)
	}
	defer conn.Close()
	if _, ok := conn.(*tls.Conn); !ok {
		t.Fatal("dialConn should have returned a *tls.Conn when clientTLS is set")
	}
	assertKeptOpen(t, conn)
	if n := waitConns(server, 1, 2*time.Second); n != 1 {
		t.Fatalf("server should be serving the authenticated conn, tracked conns=%d", n)
	}
}

// TestFabricPlaintextDialToTLSListenerFails: a plaintext dial to a TLS listener
// must be rejected at the handshake — no silent plaintext fallback.
func TestFabricPlaintextDialToTLSListenerFails(t *testing.T) {
	ca := testcerts.GenCA(t)
	srvS, _ := fabricTLSConfigs(t, ca, "n2")
	server, err := New("127.0.0.1:0", []uint32{tlsTestGroup}, srvS, nil, nil)
	if err != nil {
		t.Fatalf("server New: %v", err)
	}
	defer server.Close()

	conn, err := net.DialTimeout("tcp", server.Addr().String(), 2*time.Second)
	if err != nil {
		t.Fatalf("raw dial: %v", err)
	}
	defer conn.Close()
	// Send a full non-TLS record's worth of bytes so the server's handshake reader
	// has a complete (invalid) record header to reject immediately, rather than
	// blocking for the rest of a ClientHello that a plaintext peer never sends.
	if _, err := conn.Write(make([]byte, 64)); err != nil {
		t.Fatalf("write garbage: %v", err)
	}
	_ = conn.SetReadDeadline(time.Now().Add(3 * time.Second))
	if _, err := conn.Read(make([]byte, 1)); err == nil {
		t.Fatal("plaintext peer to a TLS listener must be rejected, but a read succeeded")
	} else if errors.Is(err, os.ErrDeadlineExceeded) {
		t.Fatalf("plaintext peer should be CLOSED at the handshake, but the read timed out: %v", err)
	}
	if n := waitConns(server, 0, 2*time.Second); n != 0 {
		t.Fatalf("plaintext peer must not remain served, tracked conns=%d", n)
	}
}

// TestFabricNonAllowlistedCNRejected: a CA-valid peer whose CN is NOT allowlisted
// completes the handshake but is dropped in handleAccepted before serveMuxConn.
func TestFabricNonAllowlistedCNRejected(t *testing.T) {
	ca := testcerts.GenCA(t)
	srvS, _ := fabricTLSConfigs(t, ca, "n2")
	_, cliC := fabricTLSConfigs(t, ca, "rogue")
	server, err := New("127.0.0.1:0", []uint32{tlsTestGroup}, srvS, nil, map[string]bool{"n1": true, "n2": true})
	if err != nil {
		t.Fatalf("server New: %v", err)
	}
	defer server.Close()
	client, err := New("127.0.0.1:0", []uint32{tlsTestGroup}, nil, cliC, nil)
	if err != nil {
		t.Fatalf("client New: %v", err)
	}
	defer client.Close()

	conn, err := client.dialConn(server.Addr().String(), 3*time.Second)
	if err != nil {
		// Acceptable: some stacks surface the peer's close during/just after the
		// handshake. The security property (not served) still holds.
		if n := waitConns(server, 0, 2*time.Second); n != 0 {
			t.Fatalf("non-allowlisted peer must not be served, conns=%d", n)
		}
		return
	}
	defer conn.Close()
	assertClosed(t, conn)
	if n := waitConns(server, 0, 2*time.Second); n != 0 {
		t.Fatalf("non-allowlisted peer must not remain served, tracked conns=%d", n)
	}
}

// TestFabricWrongCARejected: a peer signed by a DIFFERENT CA cannot complete the
// handshake against the server's cluster CA, so the outbound dial fails.
func TestFabricWrongCARejected(t *testing.T) {
	serverCA := testcerts.GenCA(t)
	otherCA := testcerts.GenCA(t)
	srvS, _ := fabricTLSConfigs(t, serverCA, "n2")
	_, cliC := fabricTLSConfigs(t, otherCA, "n1")
	server, err := New("127.0.0.1:0", []uint32{tlsTestGroup}, srvS, nil, nil)
	if err != nil {
		t.Fatalf("server New: %v", err)
	}
	defer server.Close()
	client, err := New("127.0.0.1:0", []uint32{tlsTestGroup}, nil, cliC, nil)
	if err != nil {
		t.Fatalf("client New: %v", err)
	}
	defer client.Close()

	conn, err := client.dialConn(server.Addr().String(), 2*time.Second)
	if err == nil {
		_ = conn.Close()
		t.Fatal("a peer signed by a different CA must be rejected at the handshake")
	}
}

// TestFabricNilConfigPlaintextUnchanged proves the default (nil/nil/nil) path is a
// plain-TCP transport: a plaintext dial is accepted and served, byte-identical to
// before inter-node TLS existed.
func TestFabricNilConfigPlaintextUnchanged(t *testing.T) {
	server, err := New("127.0.0.1:0", []uint32{tlsTestGroup}, nil, nil, nil)
	if err != nil {
		t.Fatalf("server New: %v", err)
	}
	defer server.Close()
	client, err := New("127.0.0.1:0", []uint32{tlsTestGroup}, nil, nil, nil)
	if err != nil {
		t.Fatalf("client New: %v", err)
	}
	defer client.Close()

	conn, err := client.dialConn(server.Addr().String(), 3*time.Second)
	if err != nil {
		t.Fatalf("plaintext dialConn: %v", err)
	}
	defer conn.Close()
	if _, ok := conn.(*tls.Conn); ok {
		t.Fatal("nil clientTLS must dial plaintext, got a *tls.Conn")
	}
	assertKeptOpen(t, conn)
	if n := waitConns(server, 1, 2*time.Second); n != 1 {
		t.Fatalf("plaintext peer should be served, tracked conns=%d", n)
	}
}

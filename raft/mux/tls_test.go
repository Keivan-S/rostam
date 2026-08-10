// SPDX-License-Identifier: Apache-2.0

package mux

import (
	"crypto/tls"
	"io"
	"net"
	"testing"
	"time"

	"github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/tlsutil"
	"github.com/rostamlabs/rostam/tlsutil/testcerts"
)

const tlsTestGroup uint32 = 7

// connResult carries an Accept outcome off the accept goroutine so tests can
// bound how long they wait — the reject cases assert Accept NEVER fires.
type connResult struct {
	Conn net.Conn
	Err  error
}

// muxTLSConfigs builds the server+client inter-node mTLS configs for a node whose
// identity cert CN is nodeCN, signed off ca. Mirrors how cmd/rostam-server wires
// the replication transports: strict mTLS server (RequireAndVerifyClientCert) +
// a client presenting the same node cert.
func muxTLSConfigs(t *testing.T, ca *testcerts.CA, nodeCN string) (server, client *tls.Config) {
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

// acceptOne runs Accept for tlsTestGroup on sl in a goroutine and reports the
// result (conn or error) on a channel so tests can bound how long they wait —
// the whole point of the reject cases is that Accept NEVER fires.
func acceptOne(sl *StreamLayer) <-chan connResult {
	ch := make(chan connResult, 1)
	go func() {
		c, err := sl.For(tlsTestGroup).Accept()
		ch <- connResult{Conn: c, Err: err}
	}()
	return ch
}

// TestMuxTLSHandshakeRoundTrip: a StreamLayer with matching-CA mTLS accepts a
// dial from a peer with an allowlisted CN, and a byte written by the dialer is
// delivered to the accepted conn — proving the TLS handshake + group-id framing +
// CN gate all pass on the happy path.
func TestMuxTLSHandshakeRoundTrip(t *testing.T) {
	ca := testcerts.GenCA(t)
	allow := map[string]bool{"n1": true, "n2": true}

	srvS, _ := muxTLSConfigs(t, ca, "n2") // server presents CN=n2
	_, cliC := muxTLSConfigs(t, ca, "n1") // client presents CN=n1
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

	accepted := acceptOne(server)
	conn, err := client.For(tlsTestGroup).Dial(raft.ServerAddress(server.Addr().String()), 3*time.Second)
	if err != nil {
		t.Fatalf("Dial over mTLS: %v", err)
	}
	defer conn.Close()

	select {
	case r := <-accepted:
		if r.Err != nil {
			t.Fatalf("Accept: %v", r.Err)
		}
		defer r.Conn.Close()
		if _, err := conn.Write([]byte("ping")); err != nil {
			t.Fatalf("client write: %v", err)
		}
		buf := make([]byte, 4)
		_ = r.Conn.SetReadDeadline(time.Now().Add(3 * time.Second))
		if _, err := io.ReadFull(r.Conn, buf); err != nil {
			t.Fatalf("server read: %v", err)
		}
		if string(buf) != "ping" {
			t.Fatalf("server read %q, want ping", buf)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Accept did not deliver the mTLS conn")
	}
}

// TestMuxPlaintextDialToTLSListenerFails: a plaintext dial to a TLS listener must
// NOT be served — the handshake fails, so Accept never delivers a conn. No silent
// fallback to plaintext.
func TestMuxPlaintextDialToTLSListenerFails(t *testing.T) {
	ca := testcerts.GenCA(t)
	srvS, _ := muxTLSConfigs(t, ca, "n2")
	server, err := New("127.0.0.1:0", []uint32{tlsTestGroup}, srvS, nil, nil)
	if err != nil {
		t.Fatalf("server New: %v", err)
	}
	defer server.Close()
	// A plaintext client (nil clientTLS) dialing the TLS listener.
	client, err := New("127.0.0.1:0", []uint32{tlsTestGroup}, nil, nil, nil)
	if err != nil {
		t.Fatalf("client New: %v", err)
	}
	defer client.Close()

	accepted := acceptOne(server)
	conn, err := client.For(tlsTestGroup).Dial(raft.ServerAddress(server.Addr().String()), 2*time.Second)
	if err == nil {
		// Some stacks let the plaintext dial+group-id write "succeed" locally; the
		// server still rejects at the handshake and never Accepts.
		_ = conn.Close()
	}
	select {
	case r := <-accepted:
		if r.Err == nil && r.Conn != nil {
			_ = r.Conn.Close()
			t.Fatal("plaintext dial to a TLS listener must NOT be accepted")
		}
	case <-time.After(750 * time.Millisecond):
		// Expected: the server rejected the plaintext peer at the handshake.
	}
}

// TestMuxNonAllowlistedCNRejected: a peer whose cert is CA-valid but whose CN is
// NOT allowlisted completes the TLS handshake, yet the server closes the conn in
// dispatchAccepted BEFORE delivering it to any group — the post-handshake identity
// gate. Accept must never fire.
func TestMuxNonAllowlistedCNRejected(t *testing.T) {
	ca := testcerts.GenCA(t)
	srvS, _ := muxTLSConfigs(t, ca, "n2")
	_, cliC := muxTLSConfigs(t, ca, "rogue") // CA-valid but CN not in the allowlist
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

	accepted := acceptOne(server)
	conn, err := client.For(tlsTestGroup).Dial(raft.ServerAddress(server.Addr().String()), 2*time.Second)
	if err == nil {
		_ = conn.Close()
	}
	select {
	case r := <-accepted:
		if r.Err == nil && r.Conn != nil {
			_ = r.Conn.Close()
			t.Fatal("a CA-valid but non-allowlisted CN must be rejected before Accept")
		}
	case <-time.After(750 * time.Millisecond):
		// Expected: rejected by the post-handshake CN gate.
	}
}

// TestMuxNilConfigPlaintextUnchanged proves the default (nil/nil/nil) path is a
// plain-TCP StreamLayer: a plaintext dial round-trips a byte, byte-identical to
// before inter-node TLS existed.
func TestMuxNilConfigPlaintextUnchanged(t *testing.T) {
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

	accepted := acceptOne(server)
	conn, err := client.For(tlsTestGroup).Dial(raft.ServerAddress(server.Addr().String()), 3*time.Second)
	if err != nil {
		t.Fatalf("plaintext Dial: %v", err)
	}
	defer conn.Close()
	select {
	case r := <-accepted:
		if r.Err != nil {
			t.Fatalf("Accept: %v", r.Err)
		}
		defer r.Conn.Close()
		if _, err := conn.Write([]byte("ok!!")); err != nil {
			t.Fatalf("write: %v", err)
		}
		buf := make([]byte, 4)
		_ = r.Conn.SetReadDeadline(time.Now().Add(3 * time.Second))
		if _, err := io.ReadFull(r.Conn, buf); err != nil {
			t.Fatalf("read: %v", err)
		}
		if string(buf) != "ok!!" {
			t.Fatalf("read %q, want ok!!", buf)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("plaintext Accept did not deliver")
	}
}

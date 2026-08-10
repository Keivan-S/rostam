// SPDX-License-Identifier: Apache-2.0

package tlsutil_test

import (
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"net"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/tlsutil"
	"github.com/rostamlabs/rostam/tlsutil/testcerts"
)

// stateWithCN builds a ConnectionState whose verified leaf carries CommonName cn,
// as PeerCNAllowed reads it (VerifiedChains[0][0].Subject.CommonName). An empty cn
// yields a state with NO verified chain (the "no cert presented" case).
func stateWithCN(cn string) tls.ConnectionState {
	if cn == "" {
		return tls.ConnectionState{}
	}
	leaf := &x509.Certificate{Subject: pkix.Name{CommonName: cn}}
	return tls.ConnectionState{VerifiedChains: [][]*x509.Certificate{{leaf}}}
}

func TestPeerCNAllowed(t *testing.T) {
	allow := map[string]bool{"n1": true, "n2": true}

	// Empty allowlist ⇒ OFF: accept any peer without inspecting the chain (even an
	// empty one — the mTLS handshake itself is the authentication).
	if err := tlsutil.PeerCNAllowed(stateWithCN(""), nil); err != nil {
		t.Errorf("empty allowlist should accept: %v", err)
	}
	// Allowlisted CN ⇒ accept.
	if err := tlsutil.PeerCNAllowed(stateWithCN("n1"), allow); err != nil {
		t.Errorf("allowlisted CN n1 should be accepted: %v", err)
	}
	// Non-allowlisted CN ⇒ fail-closed, naming the rejected CN.
	err := tlsutil.PeerCNAllowed(stateWithCN("rogue"), allow)
	if err == nil {
		t.Fatal("non-allowlisted CN must be rejected")
	}
	if !strings.Contains(err.Error(), "rogue") {
		t.Errorf("reject error should name the CN, got %v", err)
	}
	// Allowlist enabled but NO verified chain ⇒ fail-closed.
	if err := tlsutil.PeerCNAllowed(stateWithCN(""), allow); err == nil {
		t.Fatal("missing verified chain with an allowlist must be rejected")
	}
}

func TestServerTLSValid(t *testing.T) {
	ca := testcerts.GenCA(t)
	cert, key := ca.ServerCert(t, "localhost")

	// No CA, no client-cert requirement: server-auth-only TLS.
	cfg, err := tlsutil.ServerTLS(cert, key, "", false)
	if err != nil {
		t.Fatalf("ServerTLS(no ca): %v", err)
	}
	if cfg.MinVersion != tls.VersionTLS12 {
		t.Errorf("MinVersion = %x, want TLS1.2 (%x)", cfg.MinVersion, tls.VersionTLS12)
	}
	if len(cfg.Certificates) != 1 {
		t.Errorf("Certificates len = %d, want 1", len(cfg.Certificates))
	}
	if cfg.ClientAuth != tls.NoClientCert {
		t.Errorf("ClientAuth = %v, want NoClientCert with no CA", cfg.ClientAuth)
	}
}

func TestServerTLSRequireClientCert(t *testing.T) {
	ca := testcerts.GenCA(t)
	cert, key := ca.ServerCert(t, "localhost")

	cfg, err := tlsutil.ServerTLS(cert, key, ca.CAFile, true)
	if err != nil {
		t.Fatalf("ServerTLS(require mTLS): %v", err)
	}
	if cfg.ClientAuth != tls.RequireAndVerifyClientCert {
		t.Errorf("ClientAuth = %v, want RequireAndVerifyClientCert", cfg.ClientAuth)
	}
	if cfg.ClientCAs == nil {
		t.Error("ClientCAs must be set when caFile given")
	}
}

func TestServerTLSOptionalClientCert(t *testing.T) {
	ca := testcerts.GenCA(t)
	cert, key := ca.ServerCert(t, "localhost")

	cfg, err := tlsutil.ServerTLS(cert, key, ca.CAFile, false)
	if err != nil {
		t.Fatalf("ServerTLS(optional mTLS): %v", err)
	}
	if cfg.ClientAuth != tls.VerifyClientCertIfGiven {
		t.Errorf("ClientAuth = %v, want VerifyClientCertIfGiven", cfg.ClientAuth)
	}
}

func TestServerTLSMissingCertIsError(t *testing.T) {
	ca := testcerts.GenCA(t)
	_, key := ca.ServerCert(t, "localhost")

	if _, err := tlsutil.ServerTLS("", key, "", false); err == nil {
		t.Error("ServerTLS with empty certFile must error (fail-closed)")
	}
	if _, err := tlsutil.ServerTLS(filepath.Join(t.TempDir(), "nope.pem"), key, "", false); err == nil {
		t.Error("ServerTLS with nonexistent certFile must error")
	}
}

func TestServerTLSRequireClientCertWithoutCAIsError(t *testing.T) {
	ca := testcerts.GenCA(t)
	cert, key := ca.ServerCert(t, "localhost")
	if _, err := tlsutil.ServerTLS(cert, key, "", true); err == nil {
		t.Error("requireClientCert without caFile must error (nothing to verify against)")
	}
}

func TestServerTLSBadCAIsError(t *testing.T) {
	ca := testcerts.GenCA(t)
	cert, key := ca.ServerCert(t, "localhost")
	// key file is not a CA cert -> AppendCertsFromPEM fails.
	if _, err := tlsutil.ServerTLS(cert, key, key, true); err == nil {
		t.Error("ServerTLS with a non-CA PEM as caFile must error")
	}
	if _, err := tlsutil.ServerTLS(cert, key, filepath.Join(t.TempDir(), "missing-ca.pem"), true); err == nil {
		t.Error("ServerTLS with a missing caFile must error")
	}
}

func TestClientTLSValid(t *testing.T) {
	ca := testcerts.GenCA(t)

	// Server-auth-only (verify server cert against CA, no client cert).
	cfg, err := tlsutil.ClientTLS(ca.CAFile, "", "", "localhost")
	if err != nil {
		t.Fatalf("ClientTLS(server-auth): %v", err)
	}
	if cfg.RootCAs == nil {
		t.Error("RootCAs must be set when caFile given")
	}
	if cfg.ServerName != "localhost" {
		t.Errorf("ServerName = %q, want localhost", cfg.ServerName)
	}
	if cfg.InsecureSkipVerify {
		t.Error("ClientTLS must never set InsecureSkipVerify")
	}
	if len(cfg.Certificates) != 0 {
		t.Errorf("Certificates len = %d, want 0 (no client cert)", len(cfg.Certificates))
	}
}

func TestClientTLSWithClientCert(t *testing.T) {
	ca := testcerts.GenCA(t)
	cert, key := ca.ClientCert(t, "svcA")

	cfg, err := tlsutil.ClientTLS(ca.CAFile, cert, key, "localhost")
	if err != nil {
		t.Fatalf("ClientTLS(mTLS): %v", err)
	}
	if len(cfg.Certificates) != 1 {
		t.Errorf("Certificates len = %d, want 1 (client cert present)", len(cfg.Certificates))
	}
}

func TestClientTLSHalfCertPairIsError(t *testing.T) {
	ca := testcerts.GenCA(t)
	cert, key := ca.ClientCert(t, "svcA")
	if _, err := tlsutil.ClientTLS(ca.CAFile, cert, "", "localhost"); err == nil {
		t.Error("ClientTLS with cert but no key must error")
	}
	if _, err := tlsutil.ClientTLS(ca.CAFile, "", key, "localhost"); err == nil {
		t.Error("ClientTLS with key but no cert must error")
	}
}

func TestClientTLSBadCAIsError(t *testing.T) {
	if _, err := tlsutil.ClientTLS(filepath.Join(t.TempDir(), "nope.pem"), "", "", "localhost"); err == nil {
		t.Error("ClientTLS with a missing caFile must error")
	}
}

// TestServerClientHandshake is an end-to-end loopback handshake proving the
// configs interoperate: a client with the right CA completes the handshake; a
// client with a DIFFERENT CA fails it.
func TestServerClientHandshake(t *testing.T) {
	ca := testcerts.GenCA(t)
	sCert, sKey := ca.ServerCert(t, "localhost")
	serverCfg, err := tlsutil.ServerTLS(sCert, sKey, "", false)
	if err != nil {
		t.Fatal(err)
	}

	goodClient, err := tlsutil.ClientTLS(ca.CAFile, "", "", "localhost")
	if err != nil {
		t.Fatal(err)
	}
	if err := handshake(serverCfg, goodClient); err != nil {
		t.Errorf("right-CA handshake should succeed: %v", err)
	}

	otherCA := testcerts.GenCA(t)
	wrongClient, err := tlsutil.ClientTLS(otherCA.CAFile, "", "", "localhost")
	if err != nil {
		t.Fatal(err)
	}
	if err := handshake(serverCfg, wrongClient); err == nil {
		t.Error("wrong-CA handshake must FAIL")
	} else if !strings.Contains(err.Error(), "certificate") && !strings.Contains(err.Error(), "authority") {
		t.Logf("wrong-CA failed as expected: %v", err)
	}
}

// handshake runs a real loopback TLS handshake (a TCP listener on 127.0.0.1:0)
// and returns the client-side error (nil on success). A real socket (not
// net.Pipe) is used because the TLS records must flow asynchronously: net.Pipe
// is unbuffered/synchronous, so an aborted handshake can deadlock a blocked
// peer write.
func handshake(serverCfg, clientCfg *tls.Config) error {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return err
	}
	defer func() { _ = ln.Close() }()
	go func() {
		conn, aerr := ln.Accept()
		if aerr != nil {
			return
		}
		srv := tls.Server(conn, serverCfg)
		_ = srv.Handshake() // result observed via the client side
		_ = srv.Close()
	}()
	raw, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		return err
	}
	cli := tls.Client(raw, clientCfg)
	herr := cli.Handshake()
	_ = cli.Close()
	return herr
}

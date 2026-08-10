// SPDX-License-Identifier: Apache-2.0

// Package testcerts generates an in-memory ECDSA CA plus server/client leaf
// certificates for TLS/mTLS tests, writing them to PEM files in a temp dir.
//
// It is a TEST SUPPORT package (no _test.go suffix so it is importable from any
// package's tests — tlsutil, server, httpapi, grpcapi, client, and the root
// integration tests all share one cert factory). It is never linked into a
// production binary: nothing outside *_test.go imports it.
//
// Certs are short-lived, ECDSA-P256, self-contained: GenCA mints a CA, then
// ServerCert/ClientCert sign leaves off it. A second independent CA (via a
// second GenCA call) is used to prove wrong-CA handshakes fail.
package testcerts

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// CA is a generated certificate authority plus the on-disk path of its PEM cert
// (usable as the caFile for tlsutil.ServerTLS/ClientTLS).
type CA struct {
	Cert   *x509.Certificate
	Key    *ecdsa.PrivateKey
	der    []byte
	CAFile string // path to the CA cert PEM (RootCAs / ClientCAs source)
	dir    string
}

// GenCA mints a fresh self-signed ECDSA CA and writes its cert PEM under t's
// temp dir. Fails the test on any crypto/IO error.
func GenCA(t *testing.T) *CA {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("testcerts: gen CA key: %v", err)
	}
	tmpl := &x509.Certificate{
		SerialNumber:          serial(t),
		Subject:               pkix.Name{CommonName: "Rostam Test CA"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(24 * time.Hour),
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageDigitalSignature,
		BasicConstraintsValid: true,
		IsCA:                  true,
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		t.Fatalf("testcerts: create CA cert: %v", err)
	}
	cert, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatalf("testcerts: parse CA cert: %v", err)
	}
	dir := t.TempDir()
	caFile := filepath.Join(dir, "ca.pem")
	writePEM(t, caFile, "CERTIFICATE", der)
	return &CA{Cert: cert, Key: key, der: der, CAFile: caFile, dir: dir}
}

// ServerCert signs a server leaf with the given CN and the loopback SANs
// (127.0.0.1, ::1, localhost) so a client dialing 127.0.0.1 verifies the cert
// when ServerName is "localhost" or "127.0.0.1". Returns the cert+key PEM paths.
func (ca *CA) ServerCert(t *testing.T, cn string) (certFile, keyFile string) {
	t.Helper()
	return ca.leaf(t, cn, true)
}

// ClientCert signs a client leaf with the given CN (the mTLS principal the
// server's authorizer maps via LookupByCN). Returns the cert+key PEM paths.
func (ca *CA) ClientCert(t *testing.T, cn string) (certFile, keyFile string) {
	t.Helper()
	return ca.leaf(t, cn, false)
}

// NodeCert signs an inter-node leaf with the given CN carrying BOTH serverAuth
// AND clientAuth EKU plus the loopback SANs. A cluster node both serves its
// client-facing port (serverAuth, verified by a dialing peer against the SAN) and
// dials peers presenting a client cert (clientAuth, required by a peer's
// RequireAndVerifyClientCert handshake), so an inter-node identity test needs one
// cert that satisfies both roles. CN is the per-node identity matched against the
// -node-cn-allowlist. Returns the cert+key PEM paths.
func (ca *CA) NodeCert(t *testing.T, cn string) (certFile, keyFile string) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("testcerts: gen node key: %v", err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: serial(t),
		Subject:      pkix.Name{CommonName: cn},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(24 * time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
		DNSNames:     []string{"localhost"},
		IPAddresses:  []net.IP{net.IPv4(127, 0, 0, 1), net.IPv6loopback},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, ca.Cert, &key.PublicKey, ca.Key)
	if err != nil {
		t.Fatalf("testcerts: create node cert: %v", err)
	}
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		t.Fatalf("testcerts: marshal node key: %v", err)
	}
	dir := t.TempDir()
	certFile = filepath.Join(dir, "node.pem")
	keyFile = filepath.Join(dir, "node.key")
	writePEM(t, certFile, "CERTIFICATE", der)
	writePEM(t, keyFile, "EC PRIVATE KEY", keyDER)
	return certFile, keyFile
}

func (ca *CA) leaf(t *testing.T, cn string, server bool) (certFile, keyFile string) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("testcerts: gen leaf key: %v", err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: serial(t),
		Subject:      pkix.Name{CommonName: cn},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(24 * time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
	}
	if server {
		tmpl.ExtKeyUsage = []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}
		tmpl.DNSNames = []string{"localhost"}
		tmpl.IPAddresses = []net.IP{net.IPv4(127, 0, 0, 1), net.IPv6loopback}
	} else {
		tmpl.ExtKeyUsage = []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth}
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, ca.Cert, &key.PublicKey, ca.Key)
	if err != nil {
		t.Fatalf("testcerts: create leaf cert: %v", err)
	}
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		t.Fatalf("testcerts: marshal leaf key: %v", err)
	}
	dir := t.TempDir()
	certFile = filepath.Join(dir, "leaf.pem")
	keyFile = filepath.Join(dir, "leaf.key")
	writePEM(t, certFile, "CERTIFICATE", der)
	writePEM(t, keyFile, "EC PRIVATE KEY", keyDER)
	return certFile, keyFile
}

func serial(t *testing.T) *big.Int {
	t.Helper()
	n, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		t.Fatalf("testcerts: serial: %v", err)
	}
	return n
}

func writePEM(t *testing.T, path, typ string, der []byte) {
	t.Helper()
	if err := os.WriteFile(path, pem.EncodeToMemory(&pem.Block{Type: typ, Bytes: der}), 0o600); err != nil {
		t.Fatalf("testcerts: write %s: %v", path, err)
	}
}

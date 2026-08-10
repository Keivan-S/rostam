// SPDX-License-Identifier: Apache-2.0

// Package tlsutil builds *tls.Config values for Rostam's client-facing
// transports (HTTP/gRPC/TCP) and its Go client, from PEM cert/key/CA files.
//
// It is the single place TLS is assembled so every transport shares identical,
// fail-closed semantics: a missing or invalid cert/key/CA is a HARD error
// (never a silent fallback to plaintext), the minimum protocol version is
// TLS 1.2, and client-certificate verification (mTLS) is anchored ONLY to the
// configured CA pool — a client cert not chaining to that CA is rejected by
// crypto/tls during the handshake, before any application logic runs.
//
// The resulting *tls.Config is applied by the transports (server.go /
// server/server.go); nil ⇒ plaintext (unchanged). This package itself never
// reaches the network.
package tlsutil

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"os"
)

// ServerTLS builds a server-side *tls.Config from PEM files.
//
//   - certFile/keyFile: the server's leaf certificate and private key
//     (tls.LoadX509KeyPair). REQUIRED — an empty path or an unreadable/invalid
//     pair is a hard error.
//   - caFile: optional. When set, its PEM CA bundle becomes the ClientCAs pool
//     used to VERIFY client certificates (mTLS). When empty, no client-cert
//     verification is configured (server-auth-only TLS).
//   - requireClientCert: when true, ClientAuth = RequireAndVerifyClientCert — a
//     client MUST present a certificate that chains to caFile or the TLS
//     handshake FAILS (mTLS, fail-closed). requireClientCert=true REQUIRES
//     caFile (there is nothing to verify against otherwise) and errors if absent.
//
// When caFile is set but requireClientCert is false, ClientAuth =
// VerifyClientCertIfGiven: a client that presents a cert has it verified against
// caFile (so a verified CN can be extracted for principal mapping), but a client
// that presents none is still allowed (it then authenticates by bearer token or
// is denied by the authorizer). This is the deliberate "optional mTLS" mode; the
// strict mode (no token fallback at the transport) is requireClientCert=true.
//
// MinVersion is pinned to TLS 1.2. Any file error (missing/unreadable/malformed
// cert, key, or CA) returns a non-nil error and a nil config — callers MUST
// treat that as fatal and NOT start a plaintext listener.
func ServerTLS(certFile, keyFile, caFile string, requireClientCert bool) (*tls.Config, error) {
	if certFile == "" || keyFile == "" {
		return nil, fmt.Errorf("tlsutil: ServerTLS requires both certFile and keyFile (got cert=%q key=%q)", certFile, keyFile)
	}
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, fmt.Errorf("tlsutil: load server keypair: %w", err)
	}
	cfg := &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   tls.VersionTLS12,
	}

	if requireClientCert && caFile == "" {
		return nil, fmt.Errorf("tlsutil: requireClientCert set but no caFile given (nothing to verify client certs against)")
	}
	if caFile != "" {
		pool, err := loadCAPool(caFile)
		if err != nil {
			return nil, err
		}
		cfg.ClientCAs = pool
		if requireClientCert {
			// Strict mTLS: a client with no cert, or a cert not chaining to the
			// CA pool, fails the handshake. VerifiedChains is populated only on a
			// successfully verified chain, so the CN extracted downstream is always
			// CA-anchored.
			cfg.ClientAuth = tls.RequireAndVerifyClientCert
		} else {
			// Optional mTLS: a presented cert IS verified against the pool (so a
			// verified CN can drive principal mapping), but a missing cert is
			// allowed (token-or-deny fallback). A presented-but-unverifiable cert
			// still fails the handshake.
			cfg.ClientAuth = tls.VerifyClientCertIfGiven
		}
	}
	return cfg, nil
}

// ClientTLS builds a client-side *tls.Config for the Go client.
//
//   - caFile: optional PEM CA bundle that the client uses to VERIFY the SERVER's
//     certificate (RootCAs). Empty ⇒ the platform root store (rarely what a
//     private-CA deployment wants, but supported). A set-but-invalid caFile is a
//     hard error.
//   - certFile/keyFile: optional client leaf cert + key for mTLS (so the client
//     can present an identity the server verifies). BOTH must be set together;
//     setting only one is an error. Empty ⇒ no client cert (server-auth-only).
//   - serverName: the expected server certificate name (SNI + verification). Set
//     it to the cert's CN/SAN; leaving it empty means the client must reach the
//     server by a host that matches the cert, or verification fails.
//
// MinVersion is pinned to TLS 1.2. This config NEVER sets InsecureSkipVerify —
// the server cert is always verified against RootCAs (caFile or platform).
func ClientTLS(caFile, certFile, keyFile, serverName string) (*tls.Config, error) {
	cfg := &tls.Config{
		MinVersion: tls.VersionTLS12,
		ServerName: serverName,
	}
	if caFile != "" {
		pool, err := loadCAPool(caFile)
		if err != nil {
			return nil, err
		}
		cfg.RootCAs = pool
	}
	if (certFile == "") != (keyFile == "") {
		return nil, fmt.Errorf("tlsutil: ClientTLS needs both certFile and keyFile for mTLS, or neither (got cert=%q key=%q)", certFile, keyFile)
	}
	if certFile != "" {
		cert, err := tls.LoadX509KeyPair(certFile, keyFile)
		if err != nil {
			return nil, fmt.Errorf("tlsutil: load client keypair: %w", err)
		}
		cfg.Certificates = []tls.Certificate{cert}
	}
	return cfg, nil
}

// PeerCNAllowed enforces the OPT-IN per-node identity allowlist against a
// COMPLETED mTLS handshake's connection state. It is the single, shared trust
// check for the inter-node REPLICATION transports (raft/mux, raft/fabric,
// shard/pbisr), which — unlike the client-facing peerClient dial in cluster —
// cannot import that package (import cycle) but can import this leaf package.
//
// TRUST BOUNDARY: cs must come from a handshake that has ALREADY verified the
// peer's certificate chain against a CA pool — the SERVER side via
// ServerTLS(...requireClientCert=true) (RequireAndVerifyClientCert), or the
// CLIENT side via ClientTLS's RootCAs (never InsecureSkipVerify). Only then is
// cs.VerifiedChains populated, and the CN is read EXCLUSIVELY from
// VerifiedChains (never PeerCertificates, which is the unverified wire cert), so
// the identity is CA-anchored and unspoofable.
//
//   - allow empty/nil ⇒ OFF: any CA-verified peer is accepted (the mTLS
//     handshake itself is the authentication; the allowlist is an additional
//     identity pin). Returns nil without inspecting the chain.
//   - allow non-empty ⇒ the verified leaf's Subject.CommonName MUST be present,
//     else a non-nil error (fail-closed). The rejected CN is named for operator
//     diagnosis (a cert CN is not a secret); the allowlist is not logged.
func PeerCNAllowed(cs tls.ConnectionState, allow map[string]bool) error {
	if len(allow) == 0 {
		return nil
	}
	if len(cs.VerifiedChains) == 0 || len(cs.VerifiedChains[0]) == 0 {
		return errors.New("tlsutil: peer presented no verified certificate chain (per-node mTLS allowlist enabled)")
	}
	cn := cs.VerifiedChains[0][0].Subject.CommonName
	if !allow[cn] {
		return fmt.Errorf("tlsutil: peer cert CN %q not in node allowlist (per-node mTLS identity)", cn)
	}
	return nil
}

// loadCAPool reads a PEM CA bundle into a fresh x509.CertPool. A missing file or
// a file with no parseable certificate is a hard error (fail-closed: an empty
// pool would silently verify nothing).
func loadCAPool(caFile string) (*x509.CertPool, error) {
	pem, err := os.ReadFile(caFile)
	if err != nil {
		return nil, fmt.Errorf("tlsutil: read CA file %q: %w", caFile, err)
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(pem) {
		return nil, fmt.Errorf("tlsutil: CA file %q contained no valid PEM certificate", caFile)
	}
	return pool, nil
}

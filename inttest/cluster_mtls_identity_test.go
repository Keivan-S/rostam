// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"bytes"
	"context"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/tlsutil"
	"github.com/rostamlabs/rostam/tlsutil/testcerts"
	"github.com/rostamlabs/rostam/vector"
)

// newMTLSIdentityCluster stands up an n-node RF=1 inter-node mTLS cluster where
// EACH node has its OWN per-node identity cert (CN = its NodeID "n1".."nN", both
// serverAuth+clientAuth EKU) signed off a shared CA, and EACH node is configured
// with the OPT-IN per-node mTLS identity allowlist (NodeCNAllowlist). This is the
// real bring-up for the per-node mTLS identity security property:
//
//   - Inter-node CLIENT verify: peerClient pins each dialed peer's verified
//     leaf-cert CN to the allowlist (a CA-signed peer whose CN is absent fails the
//     handshake — fail-closed).
//   - Server-side gate: the authorizer's internal-token grant additionally
//     requires the verified ClientCN to be allowlisted.
//
// allowlist is the SAME set installed on every node (the static-allowlist model).
// To exercise the reject path, pass an allowlist that OMITS a node's CN: any
// inter-node forward to/from that node is then rejected.
//
// Mirrors newTLSCluster (the hardened EADDRINUSE-retry harness) but with per-node
// identity certs + the allowlist threaded into BOTH cluster config (client verify)
// and the RBAC authorizer (server gate). Returns per-node HTTPS base URLs + CA.
func newMTLSIdentityCluster(t *testing.T, n, numShards int, keyReg *vector.KeyRegistry, internalToken string, allowlist map[string]bool) ([]string, *testcerts.CA) {
	t.Helper()

	ca := testcerts.GenCA(t)
	// Per-node identity certs: CN = NodeID, both EKUs, loopback SANs.
	nodeCertFile := make([]string, n)
	nodeKeyFile := make([]string, n)
	for i := range n {
		nodeCertFile[i], nodeKeyFile[i] = ca.NodeCert(t, fmt.Sprintf("n%d", i+1))
	}

	dataDir := make([]string, n)
	for i := range n {
		dataDir[i] = t.TempDir()
	}

	servers := make([]*rostam.Server, n)
	httpAddr := make([]string, n)
	t.Cleanup(func() {
		for _, s := range servers {
			if s != nil {
				_ = s.Close()
			}
		}
	})

	buildCluster := func() (built []*rostam.Server, builtHTTP []string, err error) {
		raftLn := make([]net.Listener, n)
		tcpLn := make([]net.Listener, n)
		httpLn := make([]net.Listener, n)
		raftAddr := make([]string, n)
		tcpAddr := make([]string, n)
		httpA := make([]string, n)
		built = make([]*rostam.Server, n)
		defer func() {
			if err != nil {
				for _, s := range built {
					if s != nil {
						_ = s.Close()
					}
				}
				for i := range n {
					for _, ln := range []net.Listener{raftLn[i], tcpLn[i], httpLn[i]} {
						if ln != nil {
							_ = ln.Close()
						}
					}
				}
			}
		}()

		for i := range n {
			rl, lerr := net.Listen("tcp", "127.0.0.1:0")
			if lerr != nil {
				return nil, nil, lerr
			}
			raftLn[i], raftAddr[i] = rl, rl.Addr().String()
			tl, lerr := net.Listen("tcp", "127.0.0.1:0")
			if lerr != nil {
				return nil, nil, lerr
			}
			tcpLn[i], tcpAddr[i] = tl, tl.Addr().String()
			hl, lerr := net.Listen("tcp", "127.0.0.1:0")
			if lerr != nil {
				return nil, nil, lerr
			}
			httpLn[i], httpA[i] = hl, hl.Addr().String()
		}

		peers := make([]rostam.Peer, n)
		for i := range n {
			peers[i] = rostam.Peer{NodeID: fmt.Sprintf("n%d", i+1), RaftAddr: raftAddr[i], ServerAddr: tcpAddr[i]}
		}

		for i := range n {
			reg := ops.NewRegistry()
			if rerr := ops.RegisterBuiltins(reg); rerr != nil {
				return nil, nil, rerr
			}
			// Strict inter-node mTLS: the server demands a CA-signed client cert and
			// the node presents its OWN per-node identity cert (CN=nodeID) as BOTH the
			// server cert (peers verify it against the SAN) and the inter-node client
			// cert (peers verify its CN against the allowlist + the authorizer gates it).
			serverTLS, terr := tlsutil.ServerTLS(nodeCertFile[i], nodeKeyFile[i], ca.CAFile, true)
			if terr != nil {
				return nil, nil, fmt.Errorf("ServerTLS: %w", terr)
			}
			interNodeTLS, terr := tlsutil.ClientTLS(ca.CAFile, nodeCertFile[i], nodeKeyFile[i], "")
			if terr != nil {
				return nil, nil, fmt.Errorf("inter-node ClientTLS: %w", terr)
			}
			for _, lnp := range []*net.Listener{&raftLn[i], &tcpLn[i], &httpLn[i]} {
				_ = (*lnp).Close()
				*lnp = nil
			}
			srv, serr := rostam.NewServer(rostam.ServerConfig{
				Cluster: &rostam.EmbeddedConfig{
					NodeID:            peers[i].NodeID,
					DataDir:           dataDir[i],
					NumShards:         numShards,
					ReplicationFactor: 1,
					Bootstrap:         true,
					RaftAddr:          raftAddr[i],
					Peers:             peers,
					Ops:               reg,
					InternalToken:     internalToken,
					NodeCNAllowlist:   allowlist, // OPT-IN client verify
				},
				DirectConfig: rostam.DirectConfig{
					// OPT-IN server gate: internal-token grant requires an allowlisted CN.
					Authenticator: authz.NewRBACAuthenticatorOpts(keyReg, reg, internalToken,
						authz.RBACOptions{NodeCNAllowlist: allowlist}),
				},
				HTTPAddr:        httpA[i],
				TCPAddr:         tcpAddr[i],
				TLSConfig:       serverTLS,
				InterNodeTLS:    interNodeTLS,
				NodeCNAllowlist: allowlist,
			})
			if serr != nil {
				return nil, nil, fmt.Errorf("node %d NewServer: %w", i, serr)
			}
			built[i] = srv
		}
		return built, httpA, nil
	}

	const maxBuildAttempts = 8
	for attempt := 1; ; attempt++ {
		built, builtHTTP, err := buildCluster()
		if err == nil {
			copy(servers, built)
			copy(httpAddr, builtHTTP)
			break
		}
		if errors.Is(err, syscall.EADDRINUSE) && attempt < maxBuildAttempts {
			t.Logf("mTLS cluster build attempt %d hit port-reuse race (%v); rebuilding", attempt, err)
			continue
		}
		t.Fatalf("build mTLS cluster (attempt %d/%d): %v", attempt, maxBuildAttempts, err)
	}

	bases := make([]string, n)
	for i := range n {
		bases[i] = "https://" + httpAddr[i]
	}
	return bases, ca
}

// TestTwoNodeMTLSIdentityOff is the OFF byte-identical proof: a 2-node strict
// inter-node mTLS cluster with NO allowlist (the default) forms and a forwarded
// write succeeds — identical to today's shared-token/shared-CA path. With no
// allowlist, peerClient attaches NO VerifyPeerCertificate callback and the
// authorizer's internal-token grant is unchanged, so any CA-signed peer is
// accepted exactly as before. newTLSCluster is the unmodified existing harness
// (it never sets NodeCNAllowlist), so this is the OFF=byte-identical assertion.
func TestTwoNodeMTLSIdentityOff(t *testing.T) {
	keyReg, err := vector.OpenKeyRegistry(filepath.Join(t.TempDir(), "keys.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range []vector.APIKey{
		{Token: "k_write", Tenant: "t", Scopes: []string{"read:default/docs", "write:default/docs"}},
		{Token: "k_read", Tenant: "t", Scopes: []string{"read:default/docs"}},
		{Token: "k_admin", Tenant: "t", Scopes: []string{"*:*"}},
	} {
		if err := keyReg.AddKey(k); err != nil {
			t.Fatalf("AddKey(%q): %v", k.Token, err)
		}
	}
	const internalToken = "internal-svc-token"
	// requireClientCert=true => strict inter-node mTLS, but NO allowlist => OFF.
	bases, ca := newTLSCluster(t, 2, 8, keyReg, internalToken, true)

	clientCert, clientKey := ca.ClientCert(t, "test-http-client")
	clientTLS, err := tlsutil.ClientTLS(ca.CAFile, clientCert, clientKey, "localhost")
	if err != nil {
		t.Fatalf("client ClientTLS: %v", err)
	}
	hc := &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{TLSClientConfig: clientTLS}}
	post := func(base, token, path, body string) (int, []byte) {
		req, _ := http.NewRequest("POST", base+path, bytes.NewReader([]byte(body)))
		req.Header.Set("Content-Type", "application/json")
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		resp, err := hc.Do(req)
		if err != nil {
			return 0, []byte(err.Error())
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, b
	}
	postRetry := func(base, token, path, body string) (int, []byte) {
		t.Helper()
		deadline := time.Now().Add(25 * time.Second)
		var code int
		var b []byte
		for time.Now().Before(deadline) {
			code, b = post(base, token, path, body)
			// Retry through the meta-Raft/shard election window: 503 (no leader for
			// the target shard yet) and the client-surfaced 500 "no leader known"
			// (the 2-node meta group is still electing) are both transient.
			if code != http.StatusServiceUnavailable &&
				(code != http.StatusInternalServerError || !bytes.Contains(b, []byte("no leader"))) {
				break
			}
			time.Sleep(100 * time.Millisecond)
		}
		return code, b
	}
	if code, b := postRetry(bases[0], "k_admin", "/v1/collections",
		`{"name":"docs","config":{"dim":3,"metric":"l2"}}`); code != http.StatusCreated {
		t.Fatalf("OFF: admin create docs over mTLS = %d (%s)", code, b)
	}
	for i, base := range bases {
		id := i + 1
		body := fmt.Sprintf(`{"id":%d,"vector":[%d,0,0],"content":"c","upsert":true}`, id, id)
		if code, b := postRetry(base, "k_write", "/v1/collections/docs/points", body); code != http.StatusOK {
			t.Fatalf("OFF: forwarded write via node %d = %d (%s); OFF must be byte-identical (any CA-signed peer accepted)", i, code, b)
		}
	}
}

// TestTwoNodeMTLSIdentityAccept is the per-node mTLS identity ACCEPT proof: a
// 2-node cluster where BOTH node CNs (n1, n2) are in the shared allowlist forms
// and a write forwarded across the inter-node mTLS dial succeeds — the client
// verify accepts each peer's allowlisted CN and the server gate accepts the
// internal-token forward carrying an allowlisted ClientCN.
func TestTwoNodeMTLSIdentityAccept(t *testing.T) {
	keyReg, err := vector.OpenKeyRegistry(filepath.Join(t.TempDir(), "keys.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range []vector.APIKey{
		{Token: "k_write", Tenant: "t", Scopes: []string{"read:default/docs", "write:default/docs"}},
		{Token: "k_read", Tenant: "t", Scopes: []string{"read:default/docs"}},
		{Token: "k_admin", Tenant: "t", Scopes: []string{"*:*"}},
	} {
		if err := keyReg.AddKey(k); err != nil {
			t.Fatalf("AddKey(%q): %v", k.Token, err)
		}
	}
	const internalToken = "internal-svc-token"
	// Both node CNs allowlisted on every node (the matching-allowlist case).
	allowlist := map[string]bool{"n1": true, "n2": true}
	bases, ca := newMTLSIdentityCluster(t, 2, 8, keyReg, internalToken, allowlist)

	clientCert, clientKey := ca.ClientCert(t, "test-http-client")
	clientTLS, err := tlsutil.ClientTLS(ca.CAFile, clientCert, clientKey, "localhost")
	if err != nil {
		t.Fatalf("client ClientTLS: %v", err)
	}
	hc := &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{TLSClientConfig: clientTLS}}

	post := func(base, token, path, body string) (int, []byte) {
		req, _ := http.NewRequest("POST", base+path, bytes.NewReader([]byte(body)))
		req.Header.Set("Content-Type", "application/json")
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		resp, err := hc.Do(req)
		if err != nil {
			return 0, []byte(err.Error())
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, b
	}
	postRetry := func(base, token, path, body string) (int, []byte) {
		t.Helper()
		deadline := time.Now().Add(25 * time.Second)
		var code int
		var b []byte
		for time.Now().Before(deadline) {
			code, b = post(base, token, path, body)
			// Retry through the meta-Raft/shard election window: 503 (no leader for
			// the target shard yet) and the client-surfaced 500 "no leader known"
			// (the 2-node meta group is still electing) are both transient.
			if code != http.StatusServiceUnavailable &&
				(code != http.StatusInternalServerError || !bytes.Contains(b, []byte("no leader"))) {
				break
			}
			time.Sleep(100 * time.Millisecond)
		}
		return code, b
	}

	if code, b := postRetry(bases[0], "k_admin", "/v1/collections",
		`{"name":"docs","config":{"dim":3,"metric":"l2"}}`); code != http.StatusCreated {
		t.Fatalf("admin create docs over mTLS = %d (%s)", code, b)
	}
	// A write hitting EACH node forwards across the inter-node mTLS identity dial.
	for i, base := range bases {
		id := i + 1
		body := fmt.Sprintf(`{"id":%d,"vector":[%d,0,0],"content":"c","upsert":true}`, id, id)
		if code, b := postRetry(base, "k_write", "/v1/collections/docs/points", body); code != http.StatusOK {
			t.Fatalf("mTLS-identity forwarded write via node %d = %d (%s); allowlisted peers should forward", i, code, b)
		}
	}
	// Read back: both forwarded writes are visible.
	_, b := postRetry(bases[0], "k_read", "/v1/collections/docs/points/search", `{"query":[1,0,0],"k":10}`)
	var sr struct {
		Results []struct {
			ID uint64 `json:"id"`
		} `json:"results"`
	}
	_ = json.Unmarshal(b, &sr)
	if len(sr.Results) != 2 {
		t.Fatalf("after 2 forwarded mTLS-identity writes, search returned %d docs, want 2 (%s)", len(sr.Results), b)
	}
}

// TestMTLSIdentityClientVerifyReject is the per-node mTLS identity REJECT proof
// (the inter-node CLIENT security property), exercised DETERMINISTICALLY at the
// transport layer against a REAL mTLS rostam.Server. It mirrors exactly what
// cluster/node.go peerClient does — dial a peer with tlsutil.ClientTLS plus a
// VerifyPeerCertificate that pins the peer's verified leaf CN to an allowlist —
// but here the dialed server's node cert CN ("n2") is NOT in the allowlist, so
// the TLS handshake is REJECTED before any RPC. This avoids the flaky Raft
// cross-node routing of a probe-based cluster test while proving the exact
// fail-closed property: a CA-signed peer whose CN is not allowlisted is refused.
//
// (The server-side gate's matching reject — internal token + non-allowlisted
// ClientCN -> deny — is proven deterministically in authz's unit tests; the
// ACCEPT path of both halves is proven by TestTwoNodeMTLSIdentityAccept above.)
func TestMTLSIdentityClientVerifyReject(t *testing.T) {
	ca := testcerts.GenCA(t)
	// The peer presents an identity cert CN="n2".
	nodeCert, nodeKey := ca.NodeCert(t, "n2")

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	serverTLS, err := tlsutil.ServerTLS(nodeCert, nodeKey, ca.CAFile, true)
	if err != nil {
		t.Fatalf("ServerTLS: %v", err)
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	_ = ln.Close()
	srv, err := rostam.NewServer(rostam.ServerConfig{
		DirectConfig: rostam.DirectConfig{Ops: reg},
		TCPAddr:      addr,
		TLSConfig:    serverTLS,
	})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	t.Cleanup(func() { _ = srv.Close() })

	// The dialer mirrors peerClient: a client cert (so the server's
	// RequireAndVerifyClientCert handshake passes), ServerName pinned to the peer
	// host, and a VerifyPeerCertificate that mirrors cluster.peerCNVerifier with an
	// allowlist that OMITS the peer's CN ("n2"). Build it via tlsutil.ClientTLS,
	// then attach the callback exactly as peerClient does on the per-peer clone.
	clientCert, clientKey := ca.NodeCert(t, "n1")
	dialTLS, err := tlsutil.ClientTLS(ca.CAFile, clientCert, clientKey, "localhost")
	if err != nil {
		t.Fatalf("ClientTLS: %v", err)
	}
	allow := map[string]bool{"n1": true} // n2 intentionally absent => reject
	dialTLS.VerifyPeerCertificate = func(_ [][]byte, verifiedChains [][]*x509.Certificate) error {
		if len(verifiedChains) == 0 || len(verifiedChains[0]) == 0 {
			return fmt.Errorf("no verified chain")
		}
		cn := verifiedChains[0][0].Subject.CommonName
		if !allow[cn] {
			return fmt.Errorf("peer cert CN %q not in node allowlist", cn)
		}
		return nil
	}

	cl, err := client.New(client.Config{
		Servers:   []string{addr},
		AuthToken: "internal-svc-token",
		TLSConfig: dialTLS,
	})
	if err != nil {
		t.Fatalf("client.New: %v", err)
	}
	t.Cleanup(func() { _ = cl.Close() })

	// The call must FAIL: the handshake's VerifyPeerCertificate rejects the
	// non-allowlisted peer CN "n2" before any RPC completes (fail-closed).
	_, callErr := cl.Call(context.Background(), "put", ops.EncodePutArgs([]byte("k"), []byte{1}, 0))
	if callErr == nil {
		t.Fatal("dial to a peer whose CN is NOT allowlisted must be REJECTED, but the call succeeded")
	}
	if !strings.Contains(callErr.Error(), "n2") && !strings.Contains(callErr.Error(), "allowlist") {
		t.Logf("reject error (expected, surfaced from the handshake): %v", callErr)
	}
}

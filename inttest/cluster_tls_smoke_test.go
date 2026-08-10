// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/tlsutil"
	"github.com/rostamlabs/rostam/tlsutil/testcerts"
	"github.com/rostamlabs/rostam/vector"
)

// newTLSCluster stands up an n-node RF=1 cluster (shards partitioned across the
// nodes) where:
//
//   - Every node's client-facing transports (HTTP + TCP) are wrapped with a
//     server *tls.Config off a SHARED test CA (server leaf SANs 127.0.0.1 /
//     localhost). This is what made the pre-Step-1b cluster break: the TCP port
//     that inter-node forwarding dials is now TLS-wrapped, so a PLAINTEXT
//     inter-node dial EOFs at the peer's TLS handshake.
//   - Every node's InterNodeTLS is a CLIENT *tls.Config off the SAME CA (so the
//     inter-node forward dials each peer's TLS-wrapped port over TLS, verifying
//     the peer's server cert against the CA). This is the Step-1b production fix.
//   - Every node enforces the SAME RBAC authorizer and shares the SAME internal
//     service token (the inter-node AUTH — TLS only encrypts the transport).
//
// requireClientCert toggles strict inter-node mTLS: when true the server demands
// a CA-signed client cert at the handshake, so each node ALSO carries a node
// client cert in its InterNodeTLS (proving the inter-node dial presents a cert
// for peer mTLS); the test HTTP client is then given a client cert too. When
// false it is server-TLS-only (sufficient to prove compose — the break was at
// the handshake, before any cert exchange). Returns per-node HTTPS base URLs and
// the shared CA. Cleanup is registered with t.
func newTLSCluster(t *testing.T, n, numShards int, keyReg *vector.KeyRegistry, internalToken string, requireClientCert bool) ([]string, *testcerts.CA) {
	t.Helper()

	ca := testcerts.GenCA(t)
	// Server leaf shared by all nodes (SANs 127.0.0.1/localhost cover every
	// node's loopback address and the per-peer ServerName peerClient derives).
	sCert, sKey := ca.ServerCert(t, "rostam-node")

	// Data dirs allocated ONCE up front (not per build attempt) so their t.TempDir
	// RemoveAll cleanup is registered before the servers-Close cleanup and thus
	// runs AFTER every node is Closed (t.Cleanup LIFO). Reused across rebuild
	// attempts (each attempt fully Closes the prior node before reuse).
	dataDir := make([]string, n)
	for i := range n {
		dataDir[i] = t.TempDir()
	}

	// servers/httpAddr hold the FINAL successful cluster; the trailing cleanup
	// closes each node once.
	servers := make([]*rostam.Server, n)
	httpAddr := make([]string, n)
	t.Cleanup(func() {
		for _, s := range servers {
			if s != nil {
				_ = s.Close()
			}
		}
	})

	// buildCluster does ONE full construction attempt. Pre-binding raft/tcp/http
	// ports then closing then re-binding them inside rostam.NewServer is an inherent
	// TOCTOU: the freed ephemeral port may be reused before NewServer re-binds it,
	// yielding EADDRINUSE under load. ServerConfig/EmbeddedConfig bind from string
	// addrs and accept no live net.Listener (adding one is a production change, out
	// of scope), so on EADDRINUSE we tear the partial cluster down, pick ALL fresh
	// ports, rebuild Peers, and reconstruct — keeping Peers consistent (no node
	// ever gets a stale addr). EADDRINUSE is always a harness artifact, so retrying
	// never masks a real failure.
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
			// Per-node server TLS: wraps HTTP + TCP. With requireClientCert the TCP/HTTP
			// handshake demands a CA-signed client cert (strict mTLS).
			serverTLS, terr := tlsutil.ServerTLS(sCert, sKey, ca.CAFile, requireClientCert)
			if terr != nil {
				return nil, nil, fmt.Errorf("ServerTLS: %w", terr)
			}
			// Inter-node TLS dial config: verify the peer's server cert against the CA;
			// carry a node client cert when the peer requires one (mTLS). ServerName is
			// set per-peer by peerClient, so pass "" here.
			var nodeCert, nodeKey string
			if requireClientCert {
				nodeCert, nodeKey = ca.ClientCert(t, "rostam-node-client")
			}
			interNodeTLS, terr := tlsutil.ClientTLS(ca.CAFile, nodeCert, nodeKey, "")
			if terr != nil {
				return nil, nil, fmt.Errorf("inter-node ClientTLS: %w", terr)
			}
			// Release this node's pre-bound ports immediately before NewServer rebinds
			// them (others stay open to reserve their addresses).
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
				},
				DirectConfig: rostam.DirectConfig{
					Authenticator: authz.NewRBACAuthenticator(keyReg, reg, internalToken),
				},
				HTTPAddr:     httpA[i],
				TCPAddr:      tcpAddr[i],
				TLSConfig:    serverTLS,    // wraps client-facing transports (incl. the dialed TCP port)
				InterNodeTLS: interNodeTLS, // Step-1b: the inter-node forward dials TLS
			})
			if serr != nil {
				return nil, nil, fmt.Errorf("node %d NewServer: %w", i, serr)
			}
			built[i] = srv
		}
		return built, httpA, nil
	}

	// Retry the whole-cluster build only on EADDRINUSE (a port-reuse race — never a
	// real failure). Bounded so a genuinely unbindable environment still fails.
	const maxBuildAttempts = 8
	for attempt := 1; ; attempt++ {
		built, builtHTTP, err := buildCluster()
		if err == nil {
			copy(servers, built)
			copy(httpAddr, builtHTTP)
			break
		}
		if errors.Is(err, syscall.EADDRINUSE) && attempt < maxBuildAttempts {
			t.Logf("cluster build attempt %d hit port-reuse race (%v); rebuilding with fresh ports", attempt, err)
			continue
		}
		t.Fatalf("build cluster (attempt %d/%d): %v", attempt, maxBuildAttempts, err)
	}

	bases := make([]string, n)
	for i := range n {
		bases[i] = "https://" + httpAddr[i]
	}
	return bases, ca
}

// TestClusterTLSSmoke is the cluster-over-TLS compose proof. It is the exact cell
// that was broken before Step 1b: a multi-node cluster with client TLS on the
// client-facing transports TLS-wraps the TCP port that inter-node forwarding
// dials, so without the inter-node TLS dial a forwarded op EOFs at the peer's
// TLS handshake.
//
// It runs the server-TLS-only mode (no inter-node mTLS): server TLS alone is
// SUFFICIENT to reproduce/prove the fix because the pre-Step-1b break is at the
// TLS handshake of the wrapped peer port — it happens before any client-cert
// exchange, so requiring a client cert is orthogonal to the compose proof. The
// stronger inter-node-mTLS path is covered by TestClusterTLSSmokeMutual.
//
// A TLS HTTP client (RootCAs = test CA, ServerName matching the server cert)
// with a write:default/docs token hits a NON-leader node; the write FORWARDS to
// the docs-shard leader over the inter-node TLS dial and SUCCEEDS — proving
// client TLS + inter-node TLS compose. A read-only client's write is rejected at
// the TLS edge (scopes enforced over TLS), and the data is read back to confirm.
func TestClusterTLSSmoke(t *testing.T) {
	runClusterTLSSmoke(t, false)
}

// TestClusterTLSSmokeMutual is the strict-inter-node-mTLS variant: the server
// demands a CA-signed client cert at the handshake (RequireAndVerifyClientCert),
// so the inter-node dial MUST present the node client cert in its InterNodeTLS
// for the forward to succeed — proving the node client cert is plumbed and the
// peer verifies it. The forwarded write still succeeds, confirming inter-node
// mTLS composes too. (AUTH is still the internal token; the cert CN is irrelevant
// to authz.)
func TestClusterTLSSmokeMutual(t *testing.T) {
	runClusterTLSSmoke(t, true)
}

func runClusterTLSSmoke(t *testing.T, requireClientCert bool) {
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
	bases, ca := newTLSCluster(t, 3, 8, keyReg, internalToken, requireClientCert)

	// TLS HTTP client: verify the server cert against the test CA, ServerName
	// "localhost" (a SAN on the shared server leaf). With strict mTLS the client
	// must ALSO present a CA-signed cert at the handshake.
	var clientCert, clientKey string
	if requireClientCert {
		clientCert, clientKey = ca.ClientCert(t, "test-http-client")
	}
	clientTLS, err := tlsutil.ClientTLS(ca.CAFile, clientCert, clientKey, "localhost")
	if err != nil {
		t.Fatalf("client ClientTLS: %v", err)
	}
	hc := &http.Client{
		Timeout:   10 * time.Second,
		Transport: &http.Transport{TLSClientConfig: clientTLS},
	}

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
	// postRetry retries through the leader-election window (writes 503 until the
	// target shard's Raft group elects a leader). Gated on the 503 status from the
	// existing election path — not a fixed sleep.
	postRetry := func(base, token, path, body string, want int) (int, []byte) {
		t.Helper()
		deadline := time.Now().Add(25 * time.Second)
		var code int
		var b []byte
		for time.Now().Before(deadline) {
			code, b = post(base, token, path, body)
			if code != http.StatusServiceUnavailable {
				break
			}
			time.Sleep(100 * time.Millisecond)
		}
		return code, b
	}

	// admin (superuser) creates the docs collection from node 0 over TLS. The
	// create may itself forward to the owning shard's leader (internal token + the
	// inter-node TLS dial carry it).
	if code, b := postRetry(bases[0], "k_admin", "/v1/collections",
		`{"name":"docs","config":{"dim":3,"metric":"l2"}}`, http.StatusCreated); code != http.StatusCreated {
		t.Fatalf("admin create docs over TLS = %d (%s)", code, b)
	}

	// The compose proof: a write:default/docs client over TLS hits EVERY node
	// (including non-leaders of the docs shard); the write is forwarded to the
	// leader over the inter-node TLS dial and SUCCEEDS. Distinct ids per node so
	// each is an independent committed write.
	for i, base := range bases {
		id := i + 1
		body := fmt.Sprintf(`{"id":%d,"vector":[%d,0,0],"content":"c","upsert":true}`, id, id)
		if code, b := postRetry(base, "k_write", "/v1/collections/docs/points", body, http.StatusOK); code != http.StatusOK {
			t.Fatalf("TLS write via node %d = %d (%s) — forwarded write over the inter-node TLS dial should succeed", i, code, b)
		}
	}

	// Read back over TLS: all 3 forwarded writes are visible.
	countDocs := func() int {
		_, b := postRetry(bases[0], "k_read", "/v1/collections/docs/points/search",
			`{"query":[1,0,0],"k":10}`, http.StatusOK)
		var sr struct {
			Results []struct {
				ID uint64 `json:"id"`
			} `json:"results"`
		}
		_ = json.Unmarshal(b, &sr)
		return len(sr.Results)
	}
	if got := countDocs(); got != 3 {
		t.Fatalf("after 3 forwarded TLS writes, search returned %d docs, want 3", got)
	}

	// Scoped-deny over TLS: a read-only client's write is rejected at the TLS edge
	// (401) and never forwarded — the doc count stays 3.
	if code, b := post(bases[1], "k_read", "/v1/collections/docs/points",
		`{"id":99,"vector":[9,9,9],"content":"x","upsert":true}`); code != http.StatusUnauthorized {
		t.Errorf("read-only TLS write via non-leader = %d (%s), want 401", code, b)
	}
	if got := countDocs(); got != 3 {
		t.Errorf("after rejected read-only TLS write, search returned %d docs, want 3", got)
	}
}

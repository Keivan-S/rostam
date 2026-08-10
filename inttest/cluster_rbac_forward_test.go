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
	"github.com/rostamlabs/rostam/vector"
)

// newRBACCluster stands up an n-node RF=1 cluster (shards partitioned across the
// nodes) where EVERY node enforces the SAME RBAC authorizer (built from a shared
// KeyRegistry) and shares the SAME internal cluster service token. Each node is
// served over BOTH HTTP (so a test client can hit a SPECIFIC node's edge with a
// chosen bearer token) and TCP (so inter-node forwarding goes over the wire and
// the destination node re-authorizes the forwarded op). Returns the per-node HTTP
// base URLs. Cleanup is registered with t.
//
// The internal token is configured on every node's EmbeddedConfig, so each node's
// peerClient presents it on forwarded ops and the destination's authorizer treats
// it as superuser — WITHOUT it, a write that lands on a non-leader node would be
// forwarded token-less and the leader would reject it.
func newRBACCluster(t *testing.T, n, numShards int, keyReg *vector.KeyRegistry, internalToken string) []string {
	t.Helper()

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
				// Every node enforces the SAME authorizer (shared KeyRegistry + the same
				// ops registry + the same internal token).
				DirectConfig: rostam.DirectConfig{
					Authenticator: authz.NewRBACAuthenticator(keyReg, reg, internalToken),
				},
				HTTPAddr: httpA[i],
				TCPAddr:  tcpAddr[i],
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
		bases[i] = "http://" + httpAddr[i]
	}
	return bases
}

// TestClusterRBACForwarding proves the edge-auth + trusted-forward model end to
// end on a 3-node cluster:
//
//  1. A write:default/docs client hitting EVERY node (including non-leaders of the
//     docs shard) has its write FORWARDED to the leader and SUCCEEDS — this only
//     works because peerClient carries the internal service token and the leader's
//     authorizer treats it as superuser.
//  2. A read:default/docs-only client's write is REJECTED at the edge (401) on
//     every node and is NEVER forwarded — proven by a follow-up search showing the
//     data did not change.
func TestClusterRBACForwarding(t *testing.T) {
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
	bases := newRBACCluster(t, 3, 8, keyReg, internalToken)

	post := func(base, token, path, body string) (int, []byte) {
		req, _ := http.NewRequest("POST", base+path, bytes.NewReader([]byte(body)))
		req.Header.Set("Content-Type", "application/json")
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return 0, nil
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, b
	}
	// postRetry retries through the leader-election window (writes 503 until the
	// target shard's Raft group elects a leader).
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

	// admin (superuser) creates the docs collection from node 0. The create may
	// itself forward to the owning shard's leader (internal token carries it).
	if code, b := postRetry(bases[0], "k_admin", "/v1/collections",
		`{"name":"docs","config":{"dim":3,"metric":"l2"}}`, http.StatusCreated); code != http.StatusCreated {
		t.Fatalf("admin create docs = %d (%s)", code, b)
	}

	// 1. A write:default/docs client hitting EVERY node succeeds — the write is
	//    forwarded to the docs-shard leader carrying the internal token. We use a
	//    distinct id per node so each is an independent committed write.
	for i, base := range bases {
		id := i + 1
		body := fmt.Sprintf(`{"id":%d,"vector":[%d,0,0],"content":"c","upsert":true}`, id, id)
		if code, b := postRetry(base, "k_write", "/v1/collections/docs/points", body, http.StatusOK); code != http.StatusOK {
			t.Fatalf("write via node %d = %d (%s) — forwarded write should succeed with internal token", i, code, b)
		}
	}

	// Confirm all 3 writes are visible (search served from the leader's replica).
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
		t.Fatalf("after 3 forwarded writes, search returned %d docs, want 3", got)
	}

	// 2. A read-only client's write is rejected at the edge on EVERY node and never
	//    forwarded — the doc count stays 3 (the write never reached any engine).
	for i, base := range bases {
		body := fmt.Sprintf(`{"id":%d,"vector":[9,9,9],"content":"x","upsert":true}`, 100+i)
		if code, b := post(base, "k_read", "/v1/collections/docs/points", body); code != http.StatusUnauthorized {
			t.Errorf("read-only write via node %d = %d (%s), want 401", i, code, b)
		}
	}
	if got := countDocs(); got != 3 {
		t.Errorf("after rejected read-only writes, search returned %d docs, want 3 (rejected writes must NOT reach the engine on any node)", got)
	}

	// A read-only client also cannot touch a DIFFERENT collection (cross-collection
	// denial holds cluster-wide).
	if code, _ := post(bases[0], "k_read", "/v1/collections/other/points/search", `{"query":[1,0,0],"k":1}`); code != http.StatusUnauthorized {
		t.Errorf("read-only search other = %d, want 401", code)
	}
}

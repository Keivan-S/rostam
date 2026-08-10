// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"syscall"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
)

// TestClusterServerHTTPLeaderRedirect stands up a 3-node replicated cluster, each
// node fronted by HTTP (and TCP for inter-node forwarding), and writes to EVERY
// node's HTTP endpoint. With full replication each shard has one leader and two
// followers, so writes necessarily land on followers — which, without server-side
// leader-following, return 503. The fix forwards them to the leader, so every
// node's HTTP endpoint accepts the write.
func TestClusterServerHTTPLeaderRedirect(t *testing.T) {
	const n = 3
	const numShards = 4

	// Data dirs allocated ONCE up front (not per build attempt). The HTTP port
	// binds at :0 and is read back from srv.HTTPAddr() (no TOCTOU there); only the
	// raft + tcp ports must be known up front for Peers, so they keep the
	// pre-bind→close→re-bind pattern — which is an inherent TOCTOU.
	dataDir := make([]string, n)
	for i := range n {
		dataDir[i] = t.TempDir()
	}

	servers := make([]*rostam.Server, n)
	httpBase := make([]string, n)
	defer func() {
		for _, s := range servers {
			if s != nil {
				_ = s.Close()
			}
		}
	}()

	// buildCluster does ONE full construction attempt. Pre-binding raft/tcp ports
	// then closing then re-binding them inside rostam.NewServer is a TOCTOU: the
	// freed ephemeral port may be reused before NewServer re-binds it, yielding
	// EADDRINUSE under load. ServerConfig/EmbeddedConfig bind from string addrs and
	// accept no live net.Listener (adding one is a production change, out of
	// scope), so on EADDRINUSE we tear the partial cluster down, pick ALL fresh
	// ports, rebuild Peers, and reconstruct — keeping Peers consistent (no node
	// ever gets a stale addr). EADDRINUSE is always a harness artifact, so retrying
	// never masks a real failure.
	buildCluster := func() (built []*rostam.Server, builtHTTP []string, err error) {
		raftLn := make([]net.Listener, n)
		tcpLn := make([]net.Listener, n)
		raftAddr := make([]string, n)
		tcpAddr := make([]string, n)
		built = make([]*rostam.Server, n)
		builtHTTP = make([]string, n)
		defer func() {
			if err != nil {
				for _, s := range built {
					if s != nil {
						_ = s.Close()
					}
				}
				for i := range n {
					if raftLn[i] != nil {
						_ = raftLn[i].Close()
					}
					if tcpLn[i] != nil {
						_ = tcpLn[i].Close()
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
			// Release this node's pre-bound raft + tcp ports immediately before
			// NewServer re-binds them (others stay open to avoid port reuse).
			_ = raftLn[i].Close()
			raftLn[i] = nil
			_ = tcpLn[i].Close()
			tcpLn[i] = nil
			srv, serr := rostam.NewServer(rostam.ServerConfig{
				Cluster: &rostam.EmbeddedConfig{
					NodeID:    peers[i].NodeID,
					DataDir:   dataDir[i],
					NumShards: numShards,
					Bootstrap: true,
					RaftAddr:  raftAddr[i],
					Peers:     peers,
					Ops:       reg,
				},
				TCPAddr:  tcpAddr[i],
				HTTPAddr: "127.0.0.1:0",
			})
			if serr != nil {
				return nil, nil, fmt.Errorf("node %d NewServer: %w", i, serr)
			}
			built[i] = srv
			builtHTTP[i] = "http://" + srv.HTTPAddr()
		}
		return built, builtHTTP, nil
	}

	// Retry the whole-cluster build only on EADDRINUSE (a port-reuse race — never a
	// real failure). Bounded so a genuinely unbindable environment still fails.
	const maxBuildAttempts = 8
	for attempt := 1; ; attempt++ {
		built, builtHTTP, err := buildCluster()
		if err == nil {
			copy(servers, built)
			copy(httpBase, builtHTTP)
			break
		}
		if errors.Is(err, syscall.EADDRINUSE) && attempt < maxBuildAttempts {
			t.Logf("cluster build attempt %d hit port-reuse race (%v); rebuilding with fresh ports", attempt, err)
			continue
		}
		t.Fatalf("build cluster (attempt %d/%d): %v", attempt, maxBuildAttempts, err)
	}

	post := func(base, path, body string) (int, []byte) {
		resp, err := http.Post(base+path, "application/json", bytes.NewReader([]byte(body)))
		if err != nil {
			return 0, []byte(err.Error())
		}
		defer func() { _ = resp.Body.Close() }()
		b, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, b
	}
	// postUntil retries transient results then asserts the final status: 503/0
	// (startup election / connection race) and a 404 "unknown collection" on a
	// follower whose catalog has not yet applied the create (convergence lag —
	// only when 404 is not the wanted status). After the cluster is warm, a status
	// that never reaches `want` means a follower failed to redirect / converge —
	// the bug this test guards. cpuScaled so a starved 2-core runner gets headroom.
	postUntil := func(base, path, body string, want int) {
		t.Helper()
		deadline := time.Now().Add(cpuScaled(20 * time.Second))
		var code int
		var b []byte
		for time.Now().Before(deadline) {
			code, b = post(base, path, body)
			transient := code == http.StatusServiceUnavailable || code == 0 ||
				(code == http.StatusNotFound && want != http.StatusNotFound)
			if !transient {
				break
			}
			time.Sleep(100 * time.Millisecond)
		}
		if code != want {
			t.Fatalf("POST %s%s = %d, want %d (%s)", base, path, code, want, b)
		}
	}

	// Create a collection via node 0 (retry through the startup election window).
	postUntil(httpBase[0], "/v1/collections", `{"name":"docs","config":{"dim":3,"metric":"l2"}}`, http.StatusCreated)

	// Upsert the same point through EVERY node's HTTP endpoint. Two of the three
	// are followers for the "docs" shard; without leader-following they would 503
	// indefinitely. Idempotent upsert, so repeating across nodes is safe.
	body := `{"id":1,"vector":[1,0,0],"content":"chunk","upsert":true}`
	for i := range n {
		postUntil(httpBase[i], "/v1/collections/docs/points", body, http.StatusOK)
	}

	// And a read back through every node. postUntil rides the follower catalog-
	// convergence window (a follower that has not applied the create yet answers
	// 404 "unknown collection"), the same tolerance the upsert step above uses.
	for i := range n {
		postUntil(httpBase[i], "/v1/collections/docs/points/search", `{"query":[1,0,0],"k":1}`, http.StatusOK)
	}
}

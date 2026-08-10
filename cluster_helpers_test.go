// SPDX-License-Identifier: Apache-2.0

package rostam

// Root-package copies of the in-memory cluster test harness and small helpers that
// the inttest package now owns (they were defined in the moved
// cluster_fanout_integration_test.go / fanout_dispatcher_test.go). A handful of
// root cluster tests in embedded_test.go (and mvTokenAt in embedded_alias_resolve_test.go)
// still depend on them; per the refactor plan these stay in root until embedded_test.go
// is itself split, so a small duplicated test helper is fine. Unlike the inttest copies,
// these run IN package rostam and use the internal types directly (no rostam.* qualifier).

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// newInmemEmbeddedCluster builds an n-node in-process cluster (numShards shards,
// optional replication factor) and returns the per-node embedded Store handles.
func newInmemEmbeddedCluster(t *testing.T, n, numShards int, rf ...int) []Store {
	t.Helper()
	stores, _ := newInmemEmbeddedClusterServers(t, n, numShards, rf...)
	return stores
}

// newInmemEmbeddedClusterServers is newInmemEmbeddedCluster's full form: it also
// returns the per-node *Server handles (so a test can down a node mid-flight and
// reach its TCP address). The servers slice is index-aligned with stores.
func newInmemEmbeddedClusterServers(t *testing.T, n, numShards int, rf ...int) ([]Store, []*Server) {
	t.Helper()

	replicationFactor := 1
	if len(rf) > 0 {
		replicationFactor = rf[0]
	}

	// Data dirs are allocated ONCE, up front (not per build attempt), so their
	// t.TempDir RemoveAll cleanup is registered EARLIER than the servers-Close
	// cleanup and therefore runs LATER under t.Cleanup's LIFO order — i.e. AFTER
	// every node has been Closed. They are reused across rebuild attempts (each
	// attempt re-creates a fresh node from scratch into a clean dir; the prior
	// attempt's node is fully Closed before the dir is reused, so no live-file
	// overlap).
	dataDir := make([]string, n)
	for i := range n {
		dataDir[i] = t.TempDir()
	}

	// servers/stores are the FINAL successful cluster; the trailing cleanup closes
	// each node once.
	servers := make([]*Server, n)
	stores := make([]Store, n)
	t.Cleanup(func() {
		for _, s := range servers {
			if s != nil {
				_ = s.Close()
			}
		}
	})

	// buildCluster does ONE full construction attempt: pre-bind every node's raft
	// + tcp port (so Peers carry final addrs before any node constructs — peers
	// must know each other's addrs up front), release each node's pair just before
	// NewServer claims them, and construct all n nodes.
	//
	// Pre-binding then closing then re-binding inside NewServer is an inherent
	// TOCTOU: between Close and NewServer's re-bind the OS may hand the freed
	// ephemeral port to another process, yielding EADDRINUSE. Since Peers needs
	// the raft addr up front (bind-:0-read-back is infeasible), we make the WHOLE
	// attempt retryable: on EADDRINUSE from any node's NewServer we tear the
	// partial cluster down, pick ALL fresh ports, rebuild Peers, and reconstruct.
	// EADDRINUSE is always a harness artifact, so retrying it never masks a real
	// failure.
	buildCluster := func() (built []*Server, builtStores []Store, err error) {
		raftLn := make([]net.Listener, n)
		tcpLn := make([]net.Listener, n)
		raftAddr := make([]string, n)
		tcpAddr := make([]string, n)
		built = make([]*Server, n)
		builtStores = make([]Store, n)
		// On any failure (including EADDRINUSE), tear this attempt fully down.
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

		peers := make([]Peer, n)
		for i := range n {
			peers[i] = Peer{NodeID: fmt.Sprintf("n%d", i+1), RaftAddr: raftAddr[i], ServerAddr: tcpAddr[i]}
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
			// load-flakiness hardening: ALWAYS apply the test-tuned Raft timing,
			// not just at RF>1. The META raft group is 3 voters even at RF=1, and
			// under CPU starvation the hashicorp DefaultConfig() 1s election triggers
			// meta election storms (the 14-66s blowups). A wider election window vs
			// default means a follower starved for up to electionMs under CPU
			// oversubscription won't spuriously elect → far fewer election storms.
			// electionMs 2500 / heartbeatMs 400 keeps election ≈ 6x heartbeat;
			// it costs a bit more initial-election latency but kills the storm class.
			// Timing-only — no assertion change. NoSync (fsync off, test-speed only)
			// stays gated on RF>1 where the multi-voter Apply churn makes it matter.
			heartbeatMs, electionMs := 400, 2500
			noSync := false
			if replicationFactor > 1 {
				noSync = true
			}
			srv, serr := NewServer(ServerConfig{
				Cluster: &EmbeddedConfig{
					NodeID:            peers[i].NodeID,
					DataDir:           dataDir[i],
					NumShards:         numShards,
					ReplicationFactor: replicationFactor,
					Bootstrap:         true,
					RaftAddr:          raftAddr[i],
					Peers:             peers,
					Ops:               reg,
					RaftHeartbeatMs:   heartbeatMs,
					RaftElectionMs:    electionMs,
					NoSync:            noSync,
				},
				TCPAddr: tcpAddr[i],
			})
			if serr != nil {
				return nil, nil, fmt.Errorf("node %d NewServer: %w", i, serr)
			}
			built[i] = srv
			emb, ok := srv.store.(*embedded)
			if !ok {
				return nil, nil, fmt.Errorf("node %d: store is %T, want *embedded", i, srv.store)
			}
			builtStores[i] = emb
		}
		return built, builtStores, nil
	}

	// Retry the whole-cluster build only on EADDRINUSE (a port-reuse race in the
	// pre-bind→close→re-bind window — never a real failure). Any other error is
	// fatal immediately. Bounded so a genuinely unbindable environment still fails.
	const maxBuildAttempts = 8
	for attempt := 1; ; attempt++ {
		built, builtStores, err := buildCluster()
		if err == nil {
			copy(servers, built)
			copy(stores, builtStores)
			break
		}
		if errors.Is(err, syscall.EADDRINUSE) && attempt < maxBuildAttempts {
			t.Logf("cluster build attempt %d hit port-reuse race (%v); rebuilding with fresh ports", attempt, err)
			continue
		}
		t.Fatalf("build cluster (attempt %d/%d): %v", attempt, maxBuildAttempts, err)
	}

	if replicationFactor > 1 {
		waitClusterLeadersRF(t, stores, numShards)
		return stores, servers
	}
	waitClusterLeaders(t, stores)
	return stores, servers
}

// waitClusterLeadersRF is a coarse, write-free readiness gate for RF>1.
func waitClusterLeadersRF(t *testing.T, stores []Store, numShards int) {
	t.Helper()
	probes := numShards * 8
	keys := make([][]byte, probes)
	for i := range keys {
		keys[i] = []byte(fmt.Sprintf("__shardprobe__/%d", i))
	}
	resolved := func() (n, led int) {
		for _, k := range keys {
			if stores[0].IsLeader(k) {
				led++
			}
			if stores[0].LeaderAddr(k) != "" {
				n++
			}
		}
		return
	}
	deadline := time.Now().Add(30 * time.Second)
	prev, stable := -1, 0
	for time.Now().Before(deadline) {
		n, led := resolved()
		if n == prev {
			stable++
		} else {
			stable, prev = 0, n
		}
		if stable >= 10 && led > 0 {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Log("waitClusterLeadersRF: readiness plateau not reached within budget; proceeding (test self-gates)")
}

// waitClusterLeaders waits until node 0 can complete a keyed put for a spread of
// keys, proving every shard group it routes to has elected a leader.
func waitClusterLeaders(t *testing.T, stores []Store) {
	t.Helper()
	ctx := context.Background()
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		ready := true
		for i := 0; i < 64; i++ {
			key := []byte(fmt.Sprintf("__ready__/%d", i))
			if err := stores[0].Put(ctx, key, []byte("1"), 0); err != nil {
				ready = false
				break
			}
		}
		if ready {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatal("waitClusterLeaders: timed out waiting for cluster readiness")
}

// cpuScaled widens a test deadline when the box has few usable cores (e.g. a
// 2-core CI runner), where Raft election and reshard copy/cutover legitimately
// take longer under oversubscription. 1x on a normal dev box; kept finite so a
// genuine hang still fails loud.
// cpuScaled widens a setup/convergence deadline for CPU-constrained CI. GitHub's
// 2-vCPU runners are far slower and noisier than a developer's throttled cores —
// a 3-node RF=3 bringup that settles in seconds locally can need ~10x longer
// there — so deadlines scale with the core budget, and further under -race
// (~10x slowdown). These are upper bounds only: a healthy run returns well
// before them, so generous scaling costs wall-time only on a genuine failure.
func cpuScaled(d time.Duration) time.Duration {
	f := 1
	switch n := runtime.GOMAXPROCS(0); {
	case n <= 2:
		f = 4
	case n <= 4:
		f = 2
	}
	if raceEnabled {
		f *= 2
	}
	return d * time.Duration(f)
}

// retryUntil runs fn, retrying through the startup election window on a transient
// not-leader error, and fails the test if fn never succeeds.
func retryUntil(t *testing.T, what string, fn func() error) {
	t.Helper()
	deadline := time.Now().Add(cpuScaled(30 * time.Second))
	var err error
	for time.Now().Before(deadline) {
		if err = fn(); err == nil {
			return
		}
		if !errors.Is(err, ErrNotLeader) {
			t.Fatalf("%s: %v", what, err)
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("%s: timed out: %v", what, err)
}

// createCollectionTolerant creates coll with cfg on coord, tolerating the
// commit-but-returns-transient race that makes a bare retryUntil(create) flake
// under CPU starvation: a create can commit the collection yet still surface a
// non-not-leader transient (or a not-leader that clears only after the entry
// landed), so the NEXT retry sees "already exists" — which retryUntil fatals on
// (it retries only ErrNotLeader). Here "already exists" is treated as success
// (the physical partitions are there from the prior attempt) and the partitioned
// catalog entry is idempotently completed; not-leader is retried. It still fails
// LOUD if the coordinator catalog never converges to (P, gen 0), so a create that
// genuinely never committed is caught. Root-package mirror of inttest's
// createCollectionTolerant, with one DELIBERATE difference: this in-process copy
// matches not-leader via errors.Is(err, ErrNotLeader) (sentinel identity is
// preserved on the local handle), whereas the inttest copy string-matches because
// its errors cross a TCP/gRPC decorator that drops the sentinel. Do not "unify" them.
func createCollectionTolerant(t *testing.T, ctx context.Context, coord Store, coll string, cfg VectorConfig) {
	t.Helper()
	wantP := cfg.Partitions
	if wantP < 1 {
		wantP = 1
	}
	// A partitioned (P>1) collection registers a (P, gen 0) catalog entry we can
	// poll for convergence; a single-partition collection registers none, so its
	// convergence is simply the create returning success.
	partitioned := wantP > 1
	deadline := time.Now().Add(cpuScaled(30 * time.Second))
	created := false
	for time.Now().Before(deadline) {
		err := coord.CreateCollection(ctx, coll, cfg)
		if err == nil {
			created = true
			break
		}
		if strings.Contains(err.Error(), "already exists") {
			// A prior attempt committed but returned a transient before finishing —
			// treat as success and idempotently complete the (P, gen 0) catalog write.
			if partitioned {
				if serr := coord.(*embedded).catalog.SetPartitionsGen(coll, wantP, 0); serr != nil {
					time.Sleep(50 * time.Millisecond)
					continue
				}
			}
			created = true
			break
		}
		if !errors.Is(err, ErrNotLeader) {
			t.Fatalf("create %s: %v", coll, err)
		}
		time.Sleep(50 * time.Millisecond)
	}
	if !created {
		t.Fatalf("create %s: never succeeded within budget", coll)
	}
	// Fail loud if a partitioned catalog never converged — a create that never
	// committed is NOT silently swallowed.
	if partitioned {
		waitEmbeddedCatalogGen(t, coord.(*embedded), coll, wantP, 0, cpuScaled(15*time.Second))
	}
}

// waitEmbeddedCatalogGen waits until e's local catalog reports (wantP, wantGen)
// for collection, failing the test with the observed value on timeout.
func waitEmbeddedCatalogGen(t *testing.T, e *embedded, collection string, wantP int, wantGen uint32, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if p, gen, ok := e.catalog.PartitionsGen(collection); ok && p == wantP && gen == wantGen {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	p, gen, ok := e.catalog.PartitionsGen(collection)
	t.Fatalf("catalog %q = (p=%d, gen=%d, ok=%v), want (p=%d, gen=%d, ok=true)", collection, p, gen, ok, wantP, wantGen)
}

// mvTokenAt builds a deterministic unit token vector at index i (cos/sin sweep).
func mvTokenAt(i int) []float32 {
	theta := float64(i) * (math.Pi / 2 / 40)
	return []float32{float32(math.Cos(theta)), float32(math.Sin(theta)), 0, 0}
}

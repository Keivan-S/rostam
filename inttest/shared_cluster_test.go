// SPDX-License-Identifier: Apache-2.0

package inttest

// Flag-gated shared read-only cluster fixture.
//
// WHY THIS EXISTS
// ---------------
// inttest wall-clock is dominated by Raft election-settle latency multiplied by
// the number of SERIAL 3-node cluster bring-ups (`newInmemEmbeddedCluster`,
// electionMs=2500/heartbeatMs=400 deliberately high to avoid CPU-starvation
// election storms). The CLUSTER bring-up (meta-Raft + shard-Raft elections) is
// the expensive part; creating a collection on an already-elected cluster is
// cheap. So for LOCAL dev we can SHARE one elected cluster across multiple
// read-only tests — each test still creates its OWN uniquely-named collections
// on the shared cluster, so data isolation is preserved — and skip N bring-ups.
//
// THE FLAG
// --------
// `ROSTAM_INTTEST_SHARED_CLUSTER` controls sharing. Sharing is ON by DEFAULT
// (unset), including in CI: converted tests run against one process-lifetime
// cluster per (n, numShards, rf) key. Set the flag to "0" or "false" to force
// the opt-out — `sharedInmemEmbeddedCluster` then becomes a VERBATIM passthrough
// to `newInmemEmbeddedCluster`, so every converted test builds a FRESH per-test
// cluster with per-test `t.Cleanup` teardown. Both modes must stay green; the
// opt-out exists to isolate a suspected shared-state bug.
//
// SHAREABILITY CONTRACT (what may call sharedInmemEmbeddedCluster)
// ----------------------------------------------------------------
// A test may use the shared fixture ONLY if ALL of the following hold:
//   - It is read-only AFTER setup (search/query/get/scroll/fanout/hybrid reads).
//     Inserts are fine, but only into a collection whose name is UNIQUE to that
//     test (so two tests sharing a cluster can never collide on the same data).
//   - It performs NO cluster-destructive op: no reshard/resplit/cleanup, no
//     node-kill (servers[i].Close / newInmemEmbeddedClusterServers), no
//     partition induction, no meta-apply gating or barrier injection, and no
//     t.Parallel(). Delete-by-filter and drop-collection ARE allowed, but ONLY
//     against the test's OWN uniquely-named collection (they mutate just that
//     collection's rows/catalog entry, never another test's). CAVEAT: dropping a
//     named collection cascades through cleanupAliasesFor, which walks the shared
//     alias map — so no shared-fixture test may create an alias TARGETING another
//     shared test's collection name.
//   - It does NOT depend on a pristine/empty catalog and does NOT assert
//     cluster-wide state (leader counts, exact partition generation, exact
//     barrier/forward counts) that another test's collections would perturb.
//   - It is not a known flake.
// Tests using `newInmemEmbeddedClusterServers` (the server-handle form, for
// node-kill) are NEVER shareable. When in doubt, stay on
// `newInmemEmbeddedCluster` — a wrong conversion silently corrupts another test.
//
// COLLECTION-NAME UNIQUENESS
// --------------------------
// Every converted test names its collections with a test-specific prefix, so two
// converted tests that share the same (n, numShards, rf) cache key never read or
// write the same logical collection on the shared cluster.

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
)

// sharedClusterEnabled reports whether shared-cluster reuse is active.
// ON by default; ROSTAM_INTTEST_SHARED_CLUSTER=0 (or "false") forces every
// sharedInmemEmbeddedCluster call to fall through to newInmemEmbeddedCluster
// verbatim (a fresh per-test cluster).
func sharedClusterEnabled() bool {
	// Default ON: only the vetted read-only tests call sharedInmemEmbeddedCluster,
	// so sharing is safe by default. Override with ROSTAM_INTTEST_SHARED_CLUSTER=0
	// (or "false") to force every converted test back onto a fresh per-test cluster.
	switch os.Getenv("ROSTAM_INTTEST_SHARED_CLUSTER") {
	case "0", "false", "FALSE", "False":
		return false
	default:
		return true
	}
}

// sharedClusterEntry is a process-lifetime cached cluster keyed by its build
// parameters. servers are retained only so TestMain can Close them after all
// tests finish (they are never exposed to tests, keeping the shared path
// read-only by construction).
type sharedClusterEntry struct {
	stores  []rostam.Store
	servers []*rostam.Server
	dataDir []string
}

// sharedClusterKey identifies a cached cluster by (n, numShards, rf). Two
// requests with the same key reuse the SAME cluster.
type sharedClusterKey struct {
	n         int
	numShards int
	rf        int
}

var (
	sharedClusterMu    sync.Mutex
	sharedClusterCache = map[sharedClusterKey]*sharedClusterEntry{}
)

// sharedInmemEmbeddedCluster returns an n-node in-process cluster for a
// READ-ONLY (after setup) test. See the shareability contract at the top of this
// file before converting any test to call it.
//
// Flag OFF (CI default): a VERBATIM passthrough to newInmemEmbeddedCluster — a
// fresh cluster with per-test t.Cleanup teardown, identical to today.
//
// Flag ON (local opt-in): a process-global cache keyed by (n, numShards, rf).
// The first request for a key builds the cluster ONCE via a t-independent build
// path and registers its teardown in the package-level cache that TestMain
// drains; subsequent requests return the SAME []rostam.Store. The shared build
// registers NO t.Cleanup and uses package-lifetime temp dirs (os.MkdirTemp), so
// the cluster survives across tests and is torn down exactly once by TestMain.
func sharedInmemEmbeddedCluster(t *testing.T, n, numShards int, rf ...int) []rostam.Store {
	t.Helper()
	if !sharedClusterEnabled() {
		// CI-safety guarantee: identical to current per-test behavior.
		return newInmemEmbeddedCluster(t, n, numShards, rf...)
	}

	replicationFactor := 1
	if len(rf) > 0 {
		replicationFactor = rf[0]
	}
	key := sharedClusterKey{n: n, numShards: numShards, rf: replicationFactor}

	sharedClusterMu.Lock()
	defer sharedClusterMu.Unlock()
	if e, ok := sharedClusterCache[key]; ok {
		return e.stores
	}

	stores, servers, dataDir, err := buildSharedCluster(n, numShards, replicationFactor)
	if err != nil {
		// First caller fails loud; nothing is cached on failure.
		t.Fatalf("build shared cluster (n=%d shards=%d rf=%d): %v", n, numShards, replicationFactor, err)
	}
	sharedClusterCache[key] = &sharedClusterEntry{stores: stores, servers: servers, dataDir: dataDir}
	return stores
}

// buildSharedCluster constructs an n-node cluster WITHOUT any *testing.T
// coupling: temp dirs come from os.MkdirTemp (package-lifetime, removed by
// TestMain) and teardown is the caller's responsibility (TestMain). It mirrors
// the construction + EADDRINUSE rebuild loop + readiness gating of
// newInmemEmbeddedClusterServers, but returns errors instead of t.Fatal so the
// first caller in sharedInmemEmbeddedCluster owns the failure. newInmemEmbedded*
// is intentionally left untouched so the flag-OFF path stays byte-identical.
func buildSharedCluster(n, numShards, replicationFactor int) (stores []rostam.Store, servers []*rostam.Server, dataDir []string, err error) {
	// Package-lifetime data dirs (NOT t.TempDir): they must outlive the first
	// test and are removed by TestMain. On any failure remove what we created.
	dataDir = make([]string, 0, n)
	defer func() {
		if err != nil {
			for _, d := range dataDir {
				_ = os.RemoveAll(d)
			}
			dataDir = nil
		}
	}()
	for i := 0; i < n; i++ {
		d, derr := os.MkdirTemp("", fmt.Sprintf("rostam-shared-n%d-", i))
		if derr != nil {
			return nil, nil, nil, derr
		}
		dataDir = append(dataDir, d)
	}

	// buildOnce performs ONE full construction attempt (mirrors the inner
	// buildCluster in newInmemEmbeddedClusterServers: pre-bind every node's raft
	// + tcp port so Peers carry final addrs, release each pair just before
	// NewServer claims it, construct all n nodes). Returns the built handles.
	buildOnce := func() (built []*rostam.Server, builtStores []rostam.Store, berr error) {
		raftLn := make([]net.Listener, n)
		tcpLn := make([]net.Listener, n)
		raftAddr := make([]string, n)
		tcpAddr := make([]string, n)
		built = make([]*rostam.Server, n)
		builtStores = make([]rostam.Store, n)
		defer func() {
			if berr != nil {
				for _, s := range built {
					if s != nil {
						_ = s.Close()
					}
				}
				for i := range raftLn {
					if raftLn[i] != nil {
						_ = raftLn[i].Close()
					}
					if tcpLn[i] != nil {
						_ = tcpLn[i].Close()
					}
				}
			}
		}()

		for i := 0; i < n; i++ {
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
		for i := 0; i < n; i++ {
			peers[i] = rostam.Peer{NodeID: fmt.Sprintf("n%d", i+1), RaftAddr: raftAddr[i], ServerAddr: tcpAddr[i]}
		}

		for i := 0; i < n; i++ {
			reg := ops.NewRegistry()
			if rerr := ops.RegisterBuiltins(reg); rerr != nil {
				return nil, nil, rerr
			}
			_ = raftLn[i].Close()
			raftLn[i] = nil
			_ = tcpLn[i].Close()
			tcpLn[i] = nil
			// Same test-tuned Raft timing as newInmemEmbeddedClusterServers (see
			// its long comment): wide election window kills meta-election storms
			// under CPU oversubscription. NoSync only at RF>1.
			heartbeatMs, electionMs := testRaftHeartbeatMs, testRaftElectionMs
			noSync := false
			if replicationFactor > 1 {
				noSync = true
			}
			srv, serr := rostam.NewServer(rostam.ServerConfig{
				Cluster: &rostam.EmbeddedConfig{
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
			emb, ok := srv.Store().(*rostam.Embedded)
			if !ok {
				return nil, nil, fmt.Errorf("node %d: store is %T, want *rostam.Embedded", i, srv.Store())
			}
			builtStores[i] = emb
		}
		return built, builtStores, nil
	}

	const maxBuildAttempts = 8
	for attempt := 1; ; attempt++ {
		built, builtStores, berr := buildOnce()
		if berr == nil {
			servers = built
			stores = builtStores
			break
		}
		if errors.Is(berr, syscall.EADDRINUSE) && attempt < maxBuildAttempts {
			continue
		}
		return nil, nil, nil, fmt.Errorf("attempt %d/%d: %w", attempt, maxBuildAttempts, berr)
	}

	// Readiness gate, mirroring newInmemEmbeddedClusterServers but t-free.
	if replicationFactor > 1 {
		waitClusterLeadersRFNoT(stores, numShards)
	} else {
		if werr := waitClusterLeadersNoT(stores); werr != nil {
			// Close the just-built cluster before reporting failure.
			for _, s := range servers {
				if s != nil {
					_ = s.Close()
				}
			}
			return nil, nil, nil, werr
		}
	}
	return stores, servers, dataDir, nil
}

// waitClusterLeadersNoT is the t-free analogue of waitClusterLeaders: it waits
// until node 0 can complete a keyed put for a spread of keys (proving every shard
// group it routes to has elected a leader), returning an error on timeout instead
// of t.Fatal so buildSharedCluster's caller owns the failure.
func waitClusterLeadersNoT(stores []rostam.Store) error {
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
			return nil
		}
		time.Sleep(50 * time.Millisecond)
	}
	return fmt.Errorf("waitClusterLeaders: timed out waiting for cluster readiness")
}

// waitClusterLeadersRFNoT is the t-free analogue of waitClusterLeadersRF: a
// coarse, WRITE-FREE readiness plateau gate for RF>1. Like the t-coupled version
// it never fails — the behavioral test self-gates with generous retries — so it
// returns nothing.
func waitClusterLeadersRFNoT(stores []rostam.Store, numShards int) {
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
}

// teardownSharedClusters closes every cached cluster's servers and removes their
// data dirs. Called exactly once by TestMain after m.Run() returns, since the
// process-lifetime shared clusters cannot use a single test's t.Cleanup.
func teardownSharedClusters() {
	sharedClusterMu.Lock()
	defer sharedClusterMu.Unlock()
	for key, e := range sharedClusterCache {
		for _, s := range e.servers {
			if s != nil {
				_ = s.Close()
			}
		}
		for _, d := range e.dataDir {
			_ = os.RemoveAll(d)
		}
		delete(sharedClusterCache, key)
	}
}

// TestMain runs the suite, then drains the shared-cluster cache. When the flag
// is OFF the cache is always empty (sharedInmemEmbeddedCluster never populates
// it), so teardown is a no-op and CI behavior is unchanged.
func TestMain(m *testing.M) {
	code := m.Run()
	teardownSharedClusters()
	os.Exit(code)
}

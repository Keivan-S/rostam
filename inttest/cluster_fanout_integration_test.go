// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"errors"
	"fmt"
	"net"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/cluster"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/server"
	"github.com/rostamlabs/rostam/shard"
	"github.com/rostamlabs/rostam/vector"
)

// testRaftHeartbeatMs / testRaftElectionMs are the single source of truth for the
// test-tuned Raft timing shared by BOTH cluster builders (newInmemEmbeddedCluster's
// buildCluster and buildSharedCluster in shared_cluster_test.go). The wide election
// window (election ≈ 6x heartbeat) kills meta-election storms under CPU
// oversubscription; see the long rationale at the builder application sites. Same
// values both builders always used — centralized here so there is one knob.
const (
	testRaftHeartbeatMs = 400
	testRaftElectionMs  = 2500
)

// newInmemEmbeddedCluster stands up an n-node in-process Rostam cluster with the
// shards partitioned across the nodes (ReplicationFactor 1). Each node is a real
// *rostam.Embedded behind a TCP server, so cross-node op forwarding goes over the wire,
// and it returns the per-node embedded rostam.Store handles.
// The partition catalog is durable in the meta-Raft FSM, so any node
// can coordinate a partitioned op once its local catalog has converged — tests
// drive ops from non-creating nodes after waitEmbeddedCatalog. Cleanup is
// registered with t.
// The optional rf variadic sets the ReplicationFactor (default 1). RF>1 places a
// leader + (rf-1) follower replicas of each shard on distinct nodes, so a read
// driven on a node that hosts a shard as a FOLLOWER exercises the leader-vs-any
// replica routing (see TestClusterLeaderOnlyServedByLeader).
func newInmemEmbeddedCluster(t *testing.T, n, numShards int, rf ...int) []rostam.Store {
	t.Helper()
	stores, _ := newInmemEmbeddedClusterServers(t, n, numShards, rf...)
	return stores
}

// newInmemEmbeddedClusterServers is newInmemEmbeddedCluster's full form: it
// additionally returns the per-node *rostam.Server handles so a test can down a single
// node mid-flight (srv.Close() closes BOTH the node and its TCP listener — the
// same induction TestBarrierForShardTimeout uses to make a remote __rb_status__
// poll fail) and reach a node's TCP server address (srv.TCPAddr()) to point a
// real rostam.NewClient at it. The returned servers slice is index-aligned with stores;
// a test that closes servers[i] must NOT also touch stores[i] afterwards (the
// node is gone). Cleanup is still registered with t (double-close is guarded by
// the harness closing only non-nil servers, and *rostam.Server.Close is idempotent).
func newInmemEmbeddedClusterServers(t testing.TB, n, numShards int, rf ...int) ([]rostam.Store, []*rostam.Server) {
	t.Helper()

	replicationFactor := 1
	if len(rf) > 0 {
		replicationFactor = rf[0]
	}

	// Data dirs are allocated ONCE, up front (not per build attempt), so their
	// t.TempDir RemoveAll cleanup is registered EARLIER than the servers-Close
	// cleanup and therefore runs LATER under t.Cleanup's LIFO order — i.e. AFTER
	// every node has been Closed. Allocating a dir inline at rostam.NewServer time
	// (after the servers-Close cleanup) inverted that order, racing RemoveAll
	// against live raft files ("TempDir RemoveAll cleanup: directory not empty").
	// They are reused across rebuild attempts (each attempt re-creates a fresh
	// node from scratch into a clean dir; the prior attempt's node is fully Closed
	// before the dir is reused, so no live-file overlap).
	dataDir := make([]string, n)
	for i := range n {
		dataDir[i] = t.TempDir()
	}

	// servers/stores are the FINAL successful cluster; the trailing cleanup closes
	// each node once. It's registered after the dataDir cleanups (LIFO: runs
	// first), so nodes are torn down before their dirs are removed.
	servers := make([]*rostam.Server, n)
	stores := make([]rostam.Store, n)
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
	// rostam.NewServer claims them, and construct all n nodes. It returns the built
	// servers/stores on success.
	//
	// Pre-binding then closing then re-binding inside NewServer is an inherent
	// TOCTOU: between Close and NewServer's re-bind the OS may hand the freed
	// ephemeral port to another process, yielding EADDRINUSE. We CANNOT pass the
	// live listener through (neither ServerConfig nor EmbeddedConfig accept a
	// net.Listener — they bind from a string addr internally; adding such a hook is
	// a production change and out of scope). So instead of mirroring 1115e31's
	// bind-:0-read-back (infeasible here because Peers needs the raft addr up
	// front), we make the WHOLE attempt retryable: on EADDRINUSE from any node's
	// NewServer we tear the partial cluster down, pick ALL fresh ports, rebuild
	// Peers, and reconstruct. Rebuilding the entire cluster keeps Peers consistent
	// — no node is ever given a stale raft/server addr — and EADDRINUSE is always a
	// harness artifact, so retrying it never masks a real failure.
	buildCluster := func() (built []*rostam.Server, builtStores []rostam.Store, err error) {
		raftLn := make([]net.Listener, n)
		tcpLn := make([]net.Listener, n)
		raftAddr := make([]string, n)
		tcpAddr := make([]string, n)
		built = make([]*rostam.Server, n)
		builtStores = make([]rostam.Store, n)
		// On any failure (including EADDRINUSE), tear this attempt fully down:
		// close every node already constructed and release any listener still held.
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
			// rostam.NewServer re-binds them (others stay open to avoid port reuse).
			_ = raftLn[i].Close()
			raftLn[i] = nil
			_ = tcpLn[i].Close()
			tcpLn[i] = nil
			// load-flakiness hardening: ALWAYS apply the test-tuned Raft timing,
			// not just at RF>1. At RF=1 each shard is a single-voter group that
			// elects instantly so the shard timing barely matters — but the META
			// raft group is 3 voters even at RF=1, and under CPU starvation the
			// hashicorp DefaultConfig() 1s election triggers meta election storms
			// (the 14-66s blowups). A wider election window vs default means a
			// follower starved for up to electionMs under CPU oversubscription
			// won't spuriously elect → far fewer election storms. electionMs 2500 /
			// heartbeatMs 400 keeps election ≈ 6x heartbeat; it costs a bit more
			// initial-election latency but kills the storm class. These flow to BOTH
			// the meta and shard raft config via RaftHeartbeatMs/RaftElectionMs
			// below. Timing-only — no assertion change. NoSync (fsync off, test-speed
			// only) stays gated on RF>1 where the multi-voter Apply churn matters most.
			heartbeatMs, electionMs := testRaftHeartbeatMs, testRaftElectionMs
			noSync := false
			if replicationFactor > 1 {
				noSync = true
			}
			srv, serr := rostam.NewServer(rostam.ServerConfig{
				Cluster: &rostam.EmbeddedConfig{
					NodeID:    peers[i].NodeID,
					DataDir:   dataDir[i],
					NumShards: numShards,
					// ReplicationFactor (default 1) partitions the shards across the
					// cluster: at RF=1 each shard has exactly one owner (its sole
					// leader), so node 0's embedded Call either leads the shard it
					// owns or forwards to the single owning node. At RF>1 each shard
					// has a leader + (RF-1) follower replicas on distinct nodes, so a
					// node can host a shard it does NOT lead — the follower-serve case
					// the LeaderOnly routing test relies on.
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
		// At RF>1 the embedded write path cannot satisfy waitClusterLeaders: a
		// node that hosts a shard as a FOLLOWER returns NotLeader for a write to
		// that shard and does NOT forward it (the embedded Call has no
		// leader-following for hosted-follower shards — the same limitation the
		// RF=1 comment above calls out). So node 0's 64-key probe never fully
		// passes. Instead wait, without writing, until every shard has elected a
		// leader visible to node 0 (LeaderAddr resolves via the local replica or
		// the owners' topology), then return. Tests gate replication-applied
		// state themselves (see TestClusterLeaderOnlyServedByLeader).
		waitClusterLeadersRF(t, stores, numShards)
		return stores, servers
	}
	// Best-effort: wait for the coordinator to see a leader for its own shards.
	// With ReplicationFactor 1 a given node may not own shard 0, so this is only
	// a coarse readiness gate; ops below additionally retry through the election
	// window via retryUntil.
	waitClusterLeaders(t, stores)
	return stores, servers
}

// waitClusterLeadersRF is a coarse, WRITE-FREE readiness gate for RF>1 (the
// write-based waitClusterLeaders cannot pass — see its caller). It cannot prove
// every shard has a leader without writing: from one node, LeaderAddr only
// resolves for the shards that node HOSTS (the embedded topology path does not
// surface remote shard leaders here), so a strict "every probe key resolves"
// gate would hang at the ~hosted fraction forever. Instead we wait until the
// count of probe keys node 0 can resolve a leader for STOPS GROWING across
// several consecutive polls (raft has settled for node 0's hosted shards) AND
// node 0 leads at least one shard (raft is genuinely operational). The
// behavioral test then owns the real readiness via its own bounded
// create/insert/converge retries, which tolerate the residual election window.
func waitClusterLeadersRF(t testing.TB, stores []rostam.Store, numShards int) {
	t.Helper()
	probes := numShards * 8 // dense spread so node 0's hosted shards are all hit
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
	deadline := time.Now().Add(cpuScaled(30 * time.Second)) // cpuScaled: shard leader election is slower under 2-core CI oversubscription; finite so a genuinely stuck bringup still proceeds/logs
	prev, stable := -1, 0
	for time.Now().Before(deadline) {
		n, led := resolved()
		if n == prev {
			stable++
		} else {
			stable, prev = 0, n
		}
		// Plateaued for ~0.5s and node 0 leads something -> raft has settled
		// enough for the test's own retries to make progress.
		if stable >= 10 && led > 0 {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	// Don't fail: the behavioral test self-gates with generous retries. A coarse
	// timeout here just means we proceed and let those retries do the waiting.
	t.Log("waitClusterLeadersRF: readiness plateau not reached within budget; proceeding (test self-gates)")
}

// waitClusterLeaders waits until node 0 can complete a keyed put for a spread of
// keys, proving every shard group it routes to (own or forwarded) has elected a
// leader. We probe many distinct keys so that, with the keys hashing across all
// shards, a clean pass means the whole cluster is ready — the partitioned
// CreateCollection below is multi-step and not idempotent, so it must not run
// until every shard can accept a write.
func waitClusterLeaders(t testing.TB, stores []rostam.Store) {
	t.Helper()
	ctx := context.Background()
	deadline := time.Now().Add(cpuScaled(30 * time.Second)) // cpuScaled: shard leader election is slower under 2-core CI oversubscription; finite so a real never-ready cluster still fails loud
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
// 2-vCPU runners are far slower and noisier than a developer's throttled cores,
// and -race adds ~10x on top, so deadlines scale with the core budget and again
// under -race. Upper bounds only: a healthy run returns well before them.
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
// not-leader error. It fails the test if fn never succeeds. The partitioned
// CreateCollection is multi-step and not idempotent, so callers run it only after
// waitClusterLeaders has confirmed every shard can accept a write — by then the
// not-leader retry path here is rarely exercised, but it guards the residual race
// between the readiness probe and the create.
func retryUntil(t *testing.T, what string, fn func() error) {
	t.Helper()
	deadline := time.Now().Add(cpuScaled(30 * time.Second))
	var err error
	for time.Now().Before(deadline) {
		if err = fn(); err == nil {
			return
		}
		if !errors.Is(err, rostam.ErrNotLeader) {
			t.Fatalf("%s: %v", what, err)
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("%s: timed out: %v", what, err)
}

// ids extracts the result IDs in order.
func ids(rs []rostam.VectorResult) []uint64 {
	out := make([]uint64, len(rs))
	for i, r := range rs {
		out[i] = r.ID
	}
	return out
}

// sameIDs reports whether two result lists are equivalent as ordered-by-distance
// top-k sequences. Ties (vectors at equal distance) are order-insensitive: within
// each equal-distance run the IDs are canonicalized (sorted) before comparison.
// This is the meaningful north-star invariant — the fan-out and single-partition
// paths must agree on the same neighbors at the same distances; the relative
// order of exactly-tied neighbors is not part of the contract (HNSW within one
// partition and the cross-partition merge can break ties differently).
func sameIDs(a, b []rostam.VectorResult) bool {
	if len(a) != len(b) {
		return false
	}
	return equalCanonical(canonicalByDistance(a), canonicalByDistance(b))
}

// canonicalByDistance returns the result IDs ordered by distance, with IDs inside
// each equal-distance run sorted ascending so tie order is irrelevant.
func canonicalByDistance(rs []rostam.VectorResult) []uint64 {
	out := make([]uint64, len(rs))
	for i := 0; i < len(rs); {
		j := i
		for j < len(rs) && rs[j].Distance == rs[i].Distance {
			j++
		}
		grp := make([]uint64, 0, j-i)
		for _, r := range rs[i:j] {
			grp = append(grp, r.ID)
		}
		sort.Slice(grp, func(x, y int) bool { return grp[x] < grp[y] })
		copy(out[i:j], grp)
		i = j
	}
	return out
}

func equalCanonical(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestClusterPartitionedCreateRemoteShard is the regression guard for the
// remote partitioned-create re-expansion bug: a partitioned (P>1) logical create
// driven from node 0 whose LOGICAL-collection shard is owned by a REMOTE node
// must succeed and round-trip intact. Before the fix, the forwarded logical-create
// (carrying Partitions=P) was re-expanded by the remote owner's fanout dispatcher
// (which runs the full e.CreateCollection expansion for Partitions>1), racing the
// coordinator's own physical-create loop into "already exists" — so the create
// succeeded or failed depending on which shard the name hashed to.
//
// We pick the name deterministically: vector_create_collection is keyed by
// vectorKeyColAt1, i.e. the canonical route key ops.CanonicalName(name), and the
// embedded rostam.Store's IsLeader hashes exactly those bytes (same xxhash routing as
// Call). With ReplicationFactor 1 the shard leader is its sole owner, so the
// first candidate name where node 0 is NOT the leader of its route key is one
// whose logical-create genuinely forwards to a remote node.
func TestClusterPartitionedCreateRemoteShard(t *testing.T) {
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()

	// Deterministically find a name whose logical-collection shard is owned by a
	// NON-coordinator node (node 0 is NOT its leader), so its logical-create forwards.
	var name string
	for i := 0; i < 256; i++ {
		cand := fmt.Sprintf("remotecol%d", i)
		key := []byte(ops.CanonicalName(cand))
		if !stores[0].IsLeader(key) {
			name = cand
			break
		}
	}
	if name == "" {
		t.Fatal("could not find a collection name owned by a remote node (node 0 owns every shard?)")
	}
	t.Logf("selected remote-routed collection %q (node 0 is not its logical-shard owner)", name)

	// Create the partitioned collection FROM node 0 (the coordinator). Pre-fix this
	// fails with "already exists" because the forwarded logical-create is re-expanded.
	retryUntil(t, "create remote-routed partitioned collection", func() error {
		return stores[0].CreateCollection(ctx, name, rostam.VectorConfig{
			Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: 4,
		})
	})

	// Insert a few vectors via node 0.
	for id := uint64(1); id <= 20; id++ {
		v := []float32{float32(id), 0, 0, 0}
		retryUntil(t, fmt.Sprintf("insert %s %d", name, id), func() error {
			return stores[0].VectorInsert(ctx, name, id, v)
		})
	}

	// Search from a NON-creator node after its catalog converges to {P=4, gen=0},
	// proving the collection was created intact (logical + all physicals + catalog),
	// not half-created. Nearest to {1,0,0,0} is unambiguously id=1 (distance 0).
	e1 := stores[1].(*rostam.Embedded)
	waitEmbeddedCatalogGen(t, e1, name, 4, 0, 5*time.Second)
	res, err := stores[1].VectorSearch(ctx, name, []float32{1, 0, 0, 0}, 5)
	if err != nil {
		t.Fatalf("search from non-creator node 1: %v", err)
	}
	if len(res) == 0 || res[0].ID != 1 {
		t.Fatalf("search from node 1 top = %v, want id 1 first (collection half-created?)", ids(res))
	}
}

// TestClusterFanOutDenseFilteredMultivector is the north-star integration test:
// on a real 3-node cluster it proves that a fan-out (P>1) collection returns the
// SAME top-k as the equivalent single-partition (P=1) collection, for both plain
// dense KNN and metadata-filtered search. All ops are driven through node 0, the
// single coordinator that owns the in-process partition catalog.
func TestClusterFanOutDenseFilteredMultivector(t *testing.T) {
	// Shared-cluster eligible (read-only after setup); collection names are
	// test-unique so they cannot collide with another converted test on the
	// shared cluster.
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	store := stores[0] // single coordinator: the catalog lives on this handle.
	ctx := context.Background()
	const docs, docs1 = "fanoutdense_docs", "fanoutdense_docs1"

	const N = 600

	// Partitioned (fan-out) collection.
	retryUntil(t, "CreateCollection docs", func() error {
		return store.CreateCollection(ctx, docs, rostam.VectorConfig{
			Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: 6,
		})
	})
	for id := uint64(1); id <= N; id++ {
		v := []float32{float32(id), 0, 0, 0}
		meta := rostam.VectorMetadata{"even": vector.NewBool(id%2 == 0)}
		retryUntil(t, fmt.Sprintf("insert docs %d", id), func() error {
			return store.VectorInsertExt(ctx, docs, id, v, rostam.VectorInsertOpts{Metadata: meta})
		})
	}

	// Dense: id=1 is nearest to {1,0,0,0}.
	res, err := store.VectorSearch(ctx, docs, []float32{1, 0, 0, 0}, 5)
	if err != nil {
		t.Fatalf("dense VectorSearch: %v", err)
	}
	if len(res) == 0 || res[0].ID != 1 {
		t.Fatalf("dense top = %v, want 1 first", ids(res))
	}

	// Filtered: even-only nearest to {2,0,0,0} -> id=2. The filter rides inside
	// the per-partition args, so partition-local filtering then a global top-k is
	// exact across the fan-out (wiring; proven end-to-end below).
	f := rostam.VectorFilter{Op: vector.FilterEq, Field: "even", Value: vector.NewBool(true)}
	fr, _, err := store.VectorSearchExt(ctx, docs, []float32{2, 0, 0, 0}, 5, rostam.VectorSearchOpts{Filter: f})
	if err != nil {
		t.Fatalf("filtered VectorSearchExt: %v", err)
	}
	if len(fr) == 0 || fr[0].ID != 2 || fr[0].ID%2 != 0 {
		t.Fatalf("filtered top = %v, want even 2 first", ids(fr))
	}
	for _, r := range fr {
		if r.ID%2 != 0 {
			t.Fatalf("filtered result %d is odd; filter not applied per-partition", r.ID)
		}
	}

	// North-star equivalence: same data in a single-partition collection must
	// yield the identical top-k for both dense and filtered queries.
	retryUntil(t, "CreateCollection docs1", func() error {
		return store.CreateCollection(ctx, docs1, rostam.VectorConfig{
			Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: 1,
		})
	})
	for id := uint64(1); id <= N; id++ {
		v := []float32{float32(id), 0, 0, 0}
		meta := rostam.VectorMetadata{"even": vector.NewBool(id%2 == 0)}
		retryUntil(t, fmt.Sprintf("insert docs1 %d", id), func() error {
			return store.VectorInsertExt(ctx, docs1, id, v, rostam.VectorInsertOpts{Metadata: meta})
		})
	}

	// Dense equivalence.
	a, err := store.VectorSearch(ctx, docs, []float32{5, 0, 0, 0}, 10)
	if err != nil {
		t.Fatalf("fan-out dense search: %v", err)
	}
	b, err := store.VectorSearch(ctx, docs1, []float32{5, 0, 0, 0}, 10)
	if err != nil {
		t.Fatalf("single-partition dense search: %v", err)
	}
	if !sameIDs(a, b) {
		t.Fatalf("fan-out dense top-k %v != single-partition %v", ids(a), ids(b))
	}

	// Filtered equivalence.
	fa, _, err := store.VectorSearchExt(ctx, docs, []float32{7, 0, 0, 0}, 10, rostam.VectorSearchOpts{Filter: f})
	if err != nil {
		t.Fatalf("fan-out filtered search: %v", err)
	}
	fb, _, err := store.VectorSearchExt(ctx, docs1, []float32{7, 0, 0, 0}, 10, rostam.VectorSearchOpts{Filter: f})
	if err != nil {
		t.Fatalf("single-partition filtered search: %v", err)
	}
	if !sameIDs(fa, fb) {
		t.Fatalf("fan-out filtered top-k %v != single-partition %v", ids(fa), ids(fb))
	}
}

// TestClusterFanOutMultiCoordinator is the headline outcome of the durable
// meta-Raft partition catalog: a partitioned (P>1) collection created and
// populated through ONE node is correctly searched and written FROM other nodes
// that did NOT create it. This closes the Plan-1 "single-coordinator hazard" —
// the partition count P now lives in the meta-Raft catalog (committed cluster
// state) rather than in the creating node's in-process map, so every node can
// fan out / route point ops to the right partitions.
//
// Restart-durability of the catalog (P surviving a node restart via meta-FSM
// snapshot/log replay) is not exercised here: newInmemEmbeddedCluster has no
// stop/restart-with-same-data-dir helper. That property is covered directly by
// the meta-FSM unit test TestMetaFSMSnapshotRestoreCatalog.
func TestClusterFanOutMultiCoordinator(t *testing.T) {
	// Shared-cluster eligible (read-only after setup); collection name is
	// test-unique to avoid collision on the shared cluster.
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()
	const docs = "multicoord_docs"

	// Create + populate through node 0 (the creating coordinator).
	retryUntil(t, "CreateCollection docs", func() error {
		return stores[0].CreateCollection(ctx, docs, rostam.VectorConfig{
			Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: 6,
		})
	})
	for id := uint64(1); id <= 600; id++ {
		v := []float32{float32(id), 0, 0, 0}
		retryUntil(t, fmt.Sprintf("insert docs %d", id), func() error {
			return stores[0].VectorInsert(ctx, docs, id, v)
		})
	}

	// Nodes 1 and 2 (did NOT create it) must route searches correctly. Wait for
	// the catalog to converge on node i first — otherwise it might briefly see the
	// collection as single-partition and route to the empty logical collection,
	// returning empty/wrong results (a flake, not a real failure). The data
	// {float32(id),0,0,0} gives strictly increasing L2 distance from {1,0,0,0},
	// so the nearest is unambiguously id=1 (distance 0). No ties at position 0.
	for _, i := range []int{1, 2} {
		ei, ok := stores[i].(*rostam.Embedded)
		if !ok {
			t.Fatalf("store %d is %T, want *rostam.Embedded", i, stores[i])
		}
		waitEmbeddedCatalog(t, ei, docs, 6, 5*time.Second)

		res, err := stores[i].VectorSearch(ctx, docs, []float32{1, 0, 0, 0}, 5)
		if err != nil {
			t.Fatalf("node %d VectorSearch: %v", i, err)
		}
		if len(res) == 0 || res[0].ID != 1 {
			t.Fatalf("node %d search top = %v, want ID 1 first", i, ids(res))
		}
	}

	// A point op issued on node 1 (a non-creating node) routes to the correct
	// partition and is visible from node 2 (also non-creating). Node 1 already
	// converged in the loop above; converge node 2 explicitly before asserting.
	retryUntil(t, "insert docs 601 via node 1", func() error {
		return stores[1].VectorInsert(ctx, docs, 601, []float32{601, 0, 0, 0})
	})
	e2, ok := stores[2].(*rostam.Embedded)
	if !ok {
		t.Fatalf("store 2 is %T, want *rostam.Embedded", stores[2])
	}
	waitEmbeddedCatalog(t, e2, docs, 6, 5*time.Second)
	res, err := stores[2].VectorSearch(ctx, docs, []float32{601, 0, 0, 0}, 1)
	if err != nil {
		t.Fatalf("node 2 VectorSearch for 601: %v", err)
	}
	if len(res) == 0 || res[0].ID != 601 {
		t.Fatalf("node 2 cannot find id 601 inserted via node 1: %v", ids(res))
	}
}

// TestClusterHybridFanOutExact is the headline guarantee for hybrid fan-out: a hybrid
// search issued FROM a node that did NOT create the collection, on a partitioned
// (P>1) collection spread across a real 3-node cluster, returns results EXACTLY
// equal to the single-partition (P=1) oracle — for both RRF and Weighted fusion.
//
// It combines the durable meta-Raft partition catalog (the non-creating
// coordinator learns P from committed cluster state) with the exact hybrid
// fan-out (union per-partition dense+sparse lanes, truncate to global
// denseK/sparseK, fuse once). The query is tie-free in the dense lane (0.5 sits
// below the smallest id, so L2 distances are strictly increasing in id) so the
// fan-out-vs-oracle equality is EXACT rather than tie-order-dependent — see
// TestEmbeddedHybridFanOutSingleNodeExact.
func TestClusterHybridFanOutExact(t *testing.T) {
	// Shared-cluster eligible (read-only after setup); collection names are
	// test-unique to avoid collision on the shared cluster.
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()
	const docs, docs1 = "hybrid_docs", "hybrid_docs1"

	// Create + populate both collections through node 0 (the creating
	// coordinator that owns the in-process partition catalog at creation time).
	retryUntil(t, "CreateCollection docs (P=6)", func() error {
		return stores[0].CreateCollection(ctx, docs, rostam.VectorConfig{
			Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: 6,
		})
	})
	retryUntil(t, "CreateCollection docs1 (P=1)", func() error {
		return stores[0].CreateCollection(ctx, docs1, rostam.VectorConfig{
			Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: 1,
		})
	})
	for id := uint64(1); id <= 600; id++ {
		v := []float32{float32(id), 0, 0, 0}
		// id%11==0 selects ~54 docs; the chosen sparse query term (index 0) hits
		// exactly those, so the sparse lane genuinely contributes (non-vacuous).
		sp := rostam.VectorSparse{Indices: []uint32{uint32(id % 11)}, Values: []float32{1}}
		retryUntil(t, fmt.Sprintf("insert docs %d", id), func() error {
			return stores[0].VectorInsertExt(ctx, docs, id, v, rostam.VectorInsertOpts{Sparse: sp})
		})
		retryUntil(t, fmt.Sprintf("insert docs1 %d", id), func() error {
			return stores[0].VectorInsertExt(ctx, docs1, id, v, rostam.VectorInsertOpts{Sparse: sp})
		})
	}

	// Issue the hybrid search FROM node 1 — a node that did NOT create the
	// collection — after its local catalog has converged to P=6. Without this
	// wait node 1 might briefly route as if the collection were single-partition
	// and search the empty logical collection (a flake, not a real failure).
	e1, ok := stores[1].(*rostam.Embedded)
	if !ok {
		t.Fatalf("store 1 is %T, want *rostam.Embedded", stores[1])
	}
	waitEmbeddedCatalog(t, e1, docs, 6, 5*time.Second)

	query := []float32{0.5, 0, 0, 0}                                      // tie-free: strictly increasing L2 distance by id
	qs := rostam.VectorSparse{Indices: []uint32{0}, Values: []float32{1}} // matches ids where id%11==0 (~54 docs)
	for _, method := range []rostam.FusionMethod{rostam.FusionRRF, rostam.FusionWeighted} {
		opts := rostam.VectorHybridOpts{Sparse: qs, Method: method}
		// Fan-out (P=6) FROM node 1, the non-creating coordinator.
		got, _, err := stores[1].VectorHybridSearch(ctx, docs, query, 10, opts)
		if err != nil {
			t.Fatalf("method=%v: cluster fan-out hybrid via node 1: %v", method, err)
		}
		// Single-partition oracle via node 0.
		want, _, err := stores[0].VectorHybridSearch(ctx, docs1, query, 10, opts)
		if err != nil {
			t.Fatalf("method=%v: single-partition oracle hybrid: %v", method, err)
		}
		if !sameFusedResults(got, want) {
			t.Fatalf("method=%v: cluster hybrid fan-out %v != single-partition %v", method, ids(got), ids(want))
		}
		// Non-vacuous: both lanes must contribute, so the fused top-k must mix a
		// high-dense-rank doc (a small id, nearest to 0.5) with a high-sparse-rank
		// doc (an id where id%11==0). A dense-only result would be ids 1..10; a
		// sparse-only result would be all multiples of 11.
		gotIDs := ids(got)
		hasDense, hasSparse := false, false
		for _, id := range gotIDs {
			if id <= 10 {
				hasDense = true
			}
			if id%11 == 0 {
				hasSparse = true
			}
		}
		if !hasDense || !hasSparse {
			t.Fatalf("method=%v: fused top-k %v not non-vacuous (dense=%v sparse=%v); both lanes must contribute",
				method, gotIDs, hasDense, hasSparse)
		}
	}
}

// TestClusterGroupFanOutExact is the headline guarantee for grouped fan-out: a grouped
// search issued FROM a node that did NOT create the collection, on a partitioned
// (P>1) collection spread across a real 3-node cluster, returns group results
// EXACTLY equal to the single-partition (P=1) oracle.
//
// It combines the durable meta-Raft partition catalog (the non-creating
// coordinator learns P from committed cluster state) with the exact group fan-out
// (union per-partition candidates, truncate to the global top-fetchK by
// distance, group once). The group key id%30 spreads 30 groups across the 6
// partitions so a group's members genuinely span partitions (exercising the
// cross-partition union/re-group). The query is tie-free in the dense lane (0.5
// sits below the smallest id, so L2 distances are strictly increasing in id) so
// the fan-out-vs-oracle equality is EXACT, not tie-order-dependent — see
// TestEmbeddedGroupFanOutSingleNodeExact for the single-node analogue.
func TestClusterGroupFanOutExact(t *testing.T) {
	// Shared-cluster eligible (read-only after setup); collection names are
	// test-unique to avoid collision on the shared cluster.
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()
	const docs, docs1 = "group_docs", "group_docs1"

	// Create + populate both collections through node 0 (the creating
	// coordinator that owns the in-process partition catalog at creation time).
	retryUntil(t, "CreateCollection docs (P=6)", func() error {
		return stores[0].CreateCollection(ctx, docs, rostam.VectorConfig{
			Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: 6,
		})
	})
	retryUntil(t, "CreateCollection docs1 (P=1)", func() error {
		return stores[0].CreateCollection(ctx, docs1, rostam.VectorConfig{
			Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: 1,
		})
	})
	for id := uint64(1); id <= 600; id++ {
		v := []float32{float32(id), 0, 0, 0}
		// 30 groups (id%30) spread across the 6 partitions so each group's members
		// genuinely span partitions — exercising the cross-partition union/re-group.
		md := rostam.VectorMetadata{"doc": vector.NewInt(int64(id % 30))}
		idc := id
		retryUntil(t, fmt.Sprintf("insert docs %d", idc), func() error {
			return stores[0].VectorInsertExt(ctx, docs, idc, v, rostam.VectorInsertOpts{Metadata: md})
		})
		retryUntil(t, fmt.Sprintf("insert docs1 %d", idc), func() error {
			return stores[0].VectorInsertExt(ctx, docs1, idc, v, rostam.VectorInsertOpts{Metadata: md})
		})
	}

	// Issue the grouped search FROM node 1 — a node that did NOT create the
	// collection — after its local catalog has converged to P=6. Without this
	// wait node 1 might briefly route as if the collection were single-partition
	// and search the empty logical collection (a flake, not a real failure).
	e1, ok := stores[1].(*rostam.Embedded)
	if !ok {
		t.Fatalf("store 1 is %T, want *rostam.Embedded", stores[1])
	}
	waitEmbeddedCatalog(t, e1, docs, 6, 5*time.Second)

	query := []float32{0.5, 0, 0, 0} // tie-free: strictly increasing L2 distance by id
	opts := rostam.VectorGroupOpts{GroupBy: "doc", GroupSize: 3}
	// Fan-out (P=6) FROM node 1, the non-creating coordinator.
	got, _, err := stores[1].VectorSearchGroups(ctx, docs, query, 5, opts)
	if err != nil {
		t.Fatalf("cluster group fan-out via node 1: %v", err)
	}
	// Single-partition oracle via node 0.
	want, _, err := stores[0].VectorSearchGroups(ctx, docs1, query, 5, opts)
	if err != nil {
		t.Fatalf("single-partition oracle group search: %v", err)
	}
	if !sameGroupsResults(got, want) {
		t.Fatalf("cluster group fan-out != single-partition\n got=%+v\nwant=%+v", got, want)
	}
	// Non-vacuous: expect 5 groups, each with hits.
	if len(got) != 5 {
		t.Fatalf("expected 5 groups, got %d", len(got))
	}
	for i, g := range got {
		if len(g.Hits) == 0 {
			t.Fatalf("group %d (%+v) has no hits", i, g.Key)
		}
	}
}

// TestClusterScrollFanOut is the headline guarantee for scroll fan-out: a
// scroll issued FROM a node that did NOT create the collection, on a partitioned
// (P>1) collection spread across a real 3-node cluster, returns every document
// exactly once (completeness), with correct metadata filtering, and exact limit.
//
// It combines the durable meta-Raft partition catalog (the non-creating
// coordinator learns P from committed cluster state) with the exact scroll
// fan-out (scatter per-partition, union, cap to limit). The three
// sub-assertions are:
//   - full scroll (limit=0): all 600 docs, all distinct (no dups, no drops)
//   - filtered scroll: exactly 300 even docs
//   - limited scroll: exactly 25 distinct docs
//
// All three are issued FROM node 1, which did NOT create the collection.
func TestClusterScrollFanOut(t *testing.T) {
	// Shared-cluster eligible (read-only after setup); collection name is
	// test-unique to avoid collision on the shared cluster.
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()
	const docs = "scroll_docs"

	// Create + populate through node 0 (the creating coordinator). load-flakiness
	// hardening: tolerant create (treats a commit-but-transient "already exists" as
	// success; still fails loud if the catalog never converges) replaces the bare
	// retryUntil(create), which fataled on "already exists" under -count load.
	createCollectionTolerant(t, ctx, stores[0], docs, rostam.VectorConfig{
		Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: 6,
	})
	for id := uint64(1); id <= 600; id++ {
		v := []float32{float32(id), 0, 0, 0}
		md := rostam.VectorMetadata{"even": vector.NewBool(id%2 == 0)}
		idc := id
		retryUntil(t, "insert", func() error {
			return stores[0].VectorInsertExt(ctx, docs, idc, v, rostam.VectorInsertOpts{Metadata: md})
		})
	}

	// Wait for node 1's catalog to converge to P=6 before issuing scroll FROM it.
	// Without this wait node 1 might briefly treat the collection as single-partition
	// and scroll the empty logical name, returning 0 results (a flake, not a real bug).
	e1 := stores[1].(*rostam.Embedded)
	waitEmbeddedCatalog(t, e1, docs, 6, 5*time.Second)

	// Full scroll FROM node 1: all 600 distinct, no dups.
	all, _, _, err := stores[1].VectorScroll(ctx, docs, rostam.VectorFilter{}, 0, rostam.VectorScrollOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 600 || len(idSet(all)) != 600 {
		t.Fatalf("full scroll from node 1: %d docs (%d distinct), want 600/600", len(all), len(idSet(all)))
	}

	// Filtered: exactly 300 even docs.
	even, _, _, err := stores[1].VectorScroll(ctx, docs, rostam.VectorFilter{Op: vector.FilterEq, Field: "even", Value: vector.NewBool(true)}, 0, rostam.VectorScrollOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(even) != 300 || len(idSet(even)) != 300 {
		t.Fatalf("filtered scroll: %d docs (%d distinct), want 300/300", len(even), len(idSet(even)))
	}
	for _, d := range even {
		if d.ID%2 != 0 {
			t.Fatalf("filtered scroll returned odd id %d (filter even==true)", d.ID)
		}
	}

	// Limited: exactly 25 distinct valid docs.
	lim, _, _, err := stores[1].VectorScroll(ctx, docs, rostam.VectorFilter{}, 25, rostam.VectorScrollOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(lim) != 25 || len(idSet(lim)) != 25 {
		t.Fatalf("limited scroll: %d (%d distinct), want 25", len(lim), len(idSet(lim)))
	}
}

// waitEmbeddedCatalog polls a node's local embedded catalog until the partition
// count for collection converges to want, failing the test on timeout. This
// closes the window where a non-creating node has not yet observed the catalog
// write and would route as if the collection were single-partition.
func waitEmbeddedCatalog(t *testing.T, e *rostam.Embedded, collection string, want int, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if p, _, ok := e.Catalog().PartitionsGen(collection); ok && p == want {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	p, _, ok := e.Catalog().PartitionsGen(collection)
	t.Fatalf("catalog %q = (%d,%v), want %d", collection, p, ok, want)
}

// waitEmbeddedCatalogGen polls a node's local embedded catalog until BOTH the
// partition count and generation for collection converge to the desired values,
// failing the test on timeout. Used after a resplit to ensure a non-creating
// node has observed the catalog gen-flip before issuing searches from it.
func waitEmbeddedCatalogGen(t *testing.T, e *rostam.Embedded, collection string, wantP int, wantGen uint32, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if p, gen, ok := e.Catalog().PartitionsGen(collection); ok && p == wantP && gen == wantGen {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	p, gen, ok := e.Catalog().PartitionsGen(collection)
	t.Fatalf("catalog %q = (p=%d, gen=%d, ok=%v), want (p=%d, gen=%d, ok=true)", collection, p, gen, ok, wantP, wantGen)
}

// TestClusterSearchDocsFanOutExact is the headline guarantee for search-docs fan-out:
// a search_docs issued FROM a node that did NOT create the collection, on a
// partitioned (P>1) collection spread across a real 3-node cluster, returns
// results EXACTLY equal to the single-partition (P=1) oracle — including Content
// and Distance — for both the unfiltered and metadata-filtered cases.
//
// It combines the durable meta-Raft partition catalog (the non-creating
// coordinator learns P from committed cluster state) with the exact search_docs
// fan-out (union per-partition docs results, sort by distance,
// truncate to k). The query is tie-free (0.5 sits below the smallest id, so L2
// distances are strictly increasing in id), and each doc has a unique Content
// string ("doc-<id>") making the Content equality assertion genuinely load-bearing.
func TestClusterSearchDocsFanOutExact(t *testing.T) {
	// Shared-cluster eligible (read-only after setup); collection names are
	// test-unique to avoid collision on the shared cluster.
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()
	const docs, docs1 = "searchdocs_docs", "searchdocs_docs1"

	// Create both collections through node 0 (the creating coordinator). load-flakiness
	// hardening: tolerant create (treats a commit-but-transient "already exists" as
	// success; still fails loud if the catalog never converges) replaces the bare
	// retryUntil(create), which fataled on "already exists" under -count load.
	create := func(name string, p int) {
		createCollectionTolerant(t, ctx, stores[0], name, rostam.VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: p})
	}
	create(docs, 6)
	create(docs1, 1)

	// Populate both collections through node 0. Each doc carries distinct content
	// ("doc-<id>") so the Content equality in sameDocs is load-bearing, and metadata
	// "even" enables a genuine filter sub-case.
	for id := uint64(1); id <= 600; id++ {
		v := []float32{float32(id), 0, 0, 0}
		md := vector.Metadata{"even": vector.NewBool(id%2 == 0)}
		content := fmt.Sprintf("doc-%d", id)
		idc := id
		retryUntil(t, "upsert docs", func() error {
			return stores[0].VectorUpsert(ctx, docs, idc, v, content, rostam.VectorInsertOpts{Metadata: md})
		})
		retryUntil(t, "upsert docs1", func() error {
			return stores[0].VectorUpsert(ctx, docs1, idc, v, content, rostam.VectorInsertOpts{Metadata: md})
		})
	}

	// Issue the search_docs FROM node 1 — a node that did NOT create the collection
	// — after its local catalog has converged to P=6. Without this wait node 1 might
	// briefly route as if the collection were single-partition and search the empty
	// logical collection (a flake, not a real failure).
	e1 := stores[1].(*rostam.Embedded)
	waitEmbeddedCatalog(t, e1, docs, 6, 5*time.Second)

	// Tie-free query: 0.5 < 1 so L2 distances are strictly increasing in id.
	query := []float32{0.5, 0, 0, 0}
	for _, f := range []rostam.VectorFilter{{}, {Op: vector.FilterEq, Field: "even", Value: vector.NewBool(true)}} {
		// Fan-out (P=6) FROM node 1, the non-creating coordinator.
		got, _, err := stores[1].VectorSearchDocs(ctx, docs, query, 10, rostam.VectorSearchOpts{Filter: f})
		if err != nil {
			t.Fatal(err)
		}
		// Single-partition oracle via node 0.
		want, _, err := stores[0].VectorSearchDocs(ctx, docs1, query, 10, rostam.VectorSearchOpts{Filter: f})
		if err != nil {
			t.Fatal(err)
		}
		if !sameDocs(got, want) {
			t.Fatalf("filter=%+v: cluster search_docs fan-out != single-partition\n got=%v\nwant=%v", f, docIDs(got), docIDs(want))
		}
		if len(got) == 0 {
			t.Fatalf("filter=%+v: empty (vacuous)", f)
		}
	}
}

// TestClusterDeleteByFilterFanOut is the headline guarantee for delete-by-filter fan-out:
// delete_by_filter issued FROM a node that did NOT create the collection, on a
// partitioned (P>1) collection spread across a real 3-node cluster, deletes the
// correct number of matching docs across all partitions, and the survivors are
// observable from yet ANOTHER (third) node — exercising the durable catalog
// + deleteByFilterFanOut together.
//
// Three sub-assertions:
//   - delete FROM node 1 (non-creating): exactly 300 even docs deleted
//   - survivors FROM node 2 (third coordinator): exactly 300 distinct odd docs, none even
//   - idempotent re-delete FROM node 2: removes 0 more docs
func TestClusterDeleteByFilterFanOut(t *testing.T) {
	stores := newInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()

	// Create + populate through node 0 (the creating coordinator).
	retryUntil(t, "create docs", func() error {
		return stores[0].CreateCollection(ctx, "docs", rostam.VectorConfig{
			Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: 6,
		})
	})
	for id := uint64(1); id <= 600; id++ {
		v := []float32{float32(id), 0, 0, 0}
		md := rostam.VectorMetadata{"even": vector.NewBool(id%2 == 0)}
		idc := id
		retryUntil(t, "insert", func() error {
			return stores[0].VectorInsertExt(ctx, "docs", idc, v, rostam.VectorInsertOpts{Metadata: md})
		})
	}

	// Wait for node 1's catalog to converge to P=6 before issuing delete FROM it.
	// Without this wait node 1 might briefly treat the collection as single-partition
	// and fan out to only 1 partition, deleting far fewer than 300 (a flake, not a
	// real bug). Once the catalog converges the fan-out covers all 6 partitions.
	e1 := stores[1].(*rostam.Embedded)
	waitEmbeddedCatalog(t, e1, "docs", 6, 5*time.Second)

	// Delete even docs FROM node 1 (the non-creating coordinator).
	evenFilter := rostam.VectorFilter{Op: vector.FilterEq, Field: "even", Value: vector.NewBool(true)}
	n, err := stores[1].VectorDeleteByFilter(ctx, "docs", evenFilter)
	if err != nil {
		t.Fatal(err)
	}
	if n != 300 {
		t.Fatalf("deleted %d, want 300", n)
	}

	// Verify survivors FROM node 2 (a third coordinator — proves the delete reached
	// the physical partitions cluster-wide, not just the partitions node 1 owns).
	// Wait for node 2's catalog before scrolling from it.
	e2 := stores[2].(*rostam.Embedded)
	waitEmbeddedCatalog(t, e2, "docs", 6, 5*time.Second)

	rest, _, _, err := stores[2].VectorScroll(ctx, "docs", rostam.VectorFilter{}, 0, rostam.VectorScrollOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(rest) != 300 || len(idSet(rest)) != 300 {
		t.Fatalf("after delete: %d docs (%d distinct) remain, want 300/300", len(rest), len(idSet(rest)))
	}
	for _, d := range rest {
		if d.ID%2 == 0 {
			t.Fatalf("even id %d survived delete", d.ID)
		}
	}

	// Idempotent re-delete FROM node 2 removes nothing more.
	n2, err := stores[2].VectorDeleteByFilter(ctx, "docs", evenFilter)
	if err != nil {
		t.Fatal(err)
	}
	if n2 != 0 {
		t.Fatalf("idempotent re-delete removed %d, want 0", n2)
	}
}

// TestMVFanOutMergesDescendingScore covers the multi-vector fan-out merge
// direction in isolation: MaxSim is "higher is better", so the global merge of
// per-partition MV results must order by DESCENDING score. (MV collections do
// not yet record a partition count, so the embedded mvFanOut P>1 path is dormant
// in normal operation; this guards its merge comparator so it is correct the
// moment MV creation becomes partition-aware.) It mirrors the comparator used by
// embedded.mvFanOut.
func TestMVFanOutMergesDescendingScore(t *testing.T) {
	// Two partitions' worth of MV results, intentionally unsorted across parts.
	parts := [][]cluster.Scored{
		{{ID: 10, Score: 0.4}, {ID: 11, Score: 0.9}},
		{{ID: 20, Score: 0.7}, {ID: 21, Score: 0.1}},
	}

	wantOrder := []uint64{11, 20, 10, 21} // 0.9, 0.7, 0.4, 0.1

	// Merge via the real production primitive with the same descending-score
	// comparator that embedded.mvFanOut uses. A local sort.SliceStable copy
	// would not catch a flipped comparator inside MergeTopK itself.
	got := cluster.MergeTopK(parts, len(wantOrder), func(a, b cluster.Scored) bool { return a.Score > b.Score })

	if len(got) != len(wantOrder) {
		t.Fatalf("merged %d results, want %d", len(got), len(wantOrder))
	}
	for i, w := range wantOrder {
		if got[i].ID != w {
			t.Fatalf("merge order[%d] = %d, want %d (descending MaxSim)", i, got[i].ID, w)
		}
	}
	// Guard against accidental ascending ordering.
	for i := 1; i < len(got); i++ {
		if got[i-1].Score < got[i].Score {
			t.Fatalf("merge not descending at %d: %v < %v", i, got[i-1].Score, got[i].Score)
		}
	}
}

// TestClusterResplit is the headline guarantee of offline resplit on a
// real 3-node cluster: resplit a P=4 collection to P=8 from node 0 (the sole
// coordinator allowed to drive VectorResplit), then prove that nodes 1 and 2 —
// which did NOT initiate the resplit — correctly re-route ALL search modes to
// the new generation after the meta-Raft catalog gen-flip propagates.
//
// Five assertions cover the complete post-resplit surface:
//   - dense + content search from node 1: top-1 is id=1 with Content "doc-1"
//   - scroll completeness from node 2: all 600 distinct docs (no drops/dups)
//   - filtered fan-out from node 1: exactly 300 even docs
//   - point op via node 2 (id=601): routes by new gen, found from node 1
//
// Together these prove the gen-flip propagated through meta-Raft to every node,
// and each node's gen-aware fan-out and point-op routing use the new generation.
func TestClusterResplit(t *testing.T) {
	stores := newInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()

	// Create the collection with P=4 through node 0.
	retryUntil(t, "create", func() error {
		return stores[0].CreateCollection(ctx, "docs", rostam.VectorConfig{
			Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: 4,
		})
	})

	// Populate 600 docs (with content + even metadata) through node 0.
	for id := uint64(1); id <= 600; id++ {
		v := []float32{float32(id), 0, 0, 0}
		md := vector.Metadata{"even": vector.NewBool(id%2 == 0)}
		content := fmt.Sprintf("doc-%d", id)
		idc := id
		retryUntil(t, "upsert", func() error {
			return stores[0].VectorUpsert(ctx, "docs", idc, v, content, rostam.VectorInsertOpts{Metadata: md})
		})
	}

	// Resplit 4 -> 8 from node 0 (caller-quiesced; we have stopped writing above).
	retryUntil(t, "resplit", func() error { return stores[0].VectorResplit(ctx, "docs", 8) })

	// Wait for {P=8, gen=1} to converge on nodes 1 and 2 before verifying from them.
	for _, i := range []int{1, 2} {
		ei := stores[i].(*rostam.Embedded)
		waitEmbeddedCatalogGen(t, ei, "docs", 8, 1, 5*time.Second)
	}

	// Dense + content from node 1: nearest to {0.5,0,0,0} is id=1 with "doc-1".
	res, _, err := stores[1].VectorSearchDocs(ctx, "docs", []float32{0.5, 0, 0, 0}, 5, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || res[0].ID != 1 || res[0].Content != "doc-1" {
		t.Fatalf("post-resplit search from node 1: %+v", res)
	}

	// Scroll completeness from node 2: all 600 distinct docs.
	all, _, _, err := stores[2].VectorScroll(ctx, "docs", rostam.VectorFilter{}, 0, rostam.VectorScrollOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 600 || len(idSet(all)) != 600 {
		t.Fatalf("scroll after resplit from node 2: %d docs (%d distinct), want 600/600", len(all), len(idSet(all)))
	}

	// Filtered fan-out from node 1: exactly 300 even docs.
	even, _, err := stores[1].VectorSearchDocs(ctx, "docs", []float32{0.5, 0, 0, 0}, 600, rostam.VectorSearchOpts{
		Filter: rostam.VectorFilter{Op: vector.FilterEq, Field: "even", Value: vector.NewBool(true)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(even) != 300 {
		t.Fatalf("filtered after resplit: %d, want 300", len(even))
	}
	for _, d := range even {
		if d.ID%2 != 0 {
			t.Fatalf("filtered returned odd id %d (filter even==true)", d.ID)
		}
	}

	// Point op after resplit: insert id=601 via node 2 (routes by new gen), find from node 1.
	retryUntil(t, "post-resplit insert", func() error {
		return stores[2].VectorUpsert(ctx, "docs", 601, []float32{601, 0, 0, 0}, "doc-601",
			rostam.VectorInsertOpts{Metadata: vector.Metadata{"even": vector.NewBool(false)}})
	})
	r2, _, err := stores[1].VectorSearchDocs(ctx, "docs", []float32{601, 0, 0, 0}, 1, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(r2) == 0 || r2[0].ID != 601 {
		t.Fatalf("post-resplit insert via node 2 not found from node 1: %+v", r2)
	}
}

// TestClusterResplitCleanup is the headline guarantee of VectorResplitCleanup on a
// real 3-node cluster: orphan partitions created across shards (simulating a
// post-flip drop-old failure) are swept by VectorResplitCleanup invoked FROM a
// non-creating node, live data is untouched, and cleanup is idempotent.
//
// Four assertions:
//   - cleanup invoked from node 1 (non-creating) drops exactly 4 orphan gen-0 partitions
//   - the 4 orphans are spread across shards by normal op routing (exercising cross-shard sweep)
//   - live data (200 docs, gen-1) is intact from node 2 after cleanup
//   - second cleanup from node 1 drops 0 (idempotent)
func TestClusterResplitCleanup(t *testing.T) {
	stores := newInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()

	// Create the collection with P=4 through node 0, populate 200 docs.
	retryUntil(t, "create", func() error {
		return stores[0].CreateCollection(ctx, "docs", rostam.VectorConfig{
			Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: 4,
		})
	})
	for id := uint64(1); id <= 200; id++ {
		v := []float32{float32(id), 0, 0, 0}
		idc := id
		retryUntil(t, "insert", func() error { return stores[0].VectorInsert(ctx, "docs", idc, v) })
	}

	// Resplit 4 -> 8 from node 0; live state becomes {8, gen1}.
	retryUntil(t, "resplit", func() error { return stores[0].VectorResplit(ctx, "docs", 8) })

	// Wait for {P=8, gen=1} to converge on node 1 (the cleanup coordinator).
	e1 := stores[1].(*rostam.Embedded)
	waitEmbeddedCatalogGen(t, e1, "docs", 8, 1, 5*time.Second)

	// Simulate post-flip old-gen-0 leaks: re-create gen-0 partitions via normal op routing.
	// Each call routes by the physical name's shard key so the 4 leaks land on (possibly
	// different) shards across the cluster — exercising cross-shard cleanup.
	cfg := rostam.VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}
	ee0 := stores[0].(*rostam.Embedded)
	for p := 0; p < 4; p++ {
		phys := string(ops.PartitionKeyGen("docs", 0, p))
		pc := phys
		retryUntil(t, "leak", func() error {
			_, err := ee0.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(pc, cfg))
			return err
		})
	}

	// Cleanup from node 1 (non-creating coordinator) — reads liveGen from its
	// converged local catalog and drops the 4 gen-0 orphans wherever they live.
	dropped, err := stores[1].VectorResplitCleanup(ctx, "docs")
	if err != nil {
		t.Fatal(err)
	}
	if dropped != 4 {
		t.Fatalf("cluster cleanup dropped %d, want 4", dropped)
	}

	// Wait for node 2's catalog to converge before scrolling from it.
	e2 := stores[2].(*rostam.Embedded)
	waitEmbeddedCatalogGen(t, e2, "docs", 8, 1, 5*time.Second)

	// Live data intact from node 2: all 200 docs present, none duplicated.
	all, _, _, err := stores[2].VectorScroll(ctx, "docs", rostam.VectorFilter{}, 0, rostam.VectorScrollOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(idSet(all)) != 200 {
		t.Fatalf("after cluster cleanup: %d distinct docs, want 200", len(idSet(all)))
	}

	// Idempotent: second cleanup from node 1 drops nothing.
	d2, err := stores[1].VectorResplitCleanup(ctx, "docs")
	if err != nil {
		t.Fatal(err)
	}
	if d2 != 0 {
		t.Fatalf("second cluster cleanup dropped %d, want 0", d2)
	}
}

// TestClusterMVResplit is the multi-vector mirror of TestClusterResplit: resplit a
// partitioned (P=4) MV collection to P=8 from node 0 (the sole coordinator allowed
// to drive VectorMVResplit), then prove that NON-creating nodes 1 and 2 correctly
// re-route MaxSim late-interaction search to the new generation after the meta-Raft
// catalog gen-flip propagates.
//
// The docs use tie-free angular tokens (mvTokenAt) so the MaxSim winner for any
// chosen query is unambiguous and deterministic across the partition fan-out, and
// each doc carries metadata {"docid": NewInt(id)} so the metadata-preservation
// assertion is load-bearing.
//
// Assertions (all FROM non-creating nodes 1/2, after {P=8, gen=1} convergence):
//   - MVSearch for a chosen winner's token returns it at rank 0 (deterministic via
//     tie-free tokens) AND res[0].Metadata["docid"] == the inserted NewInt(winner)
//     (metadata preserved through resplit + fan-out; oracle = inserted value).
//   - a doc added via node 2 AFTER the resplit routes to the new generation and is
//     found from node 1 (proves new-gen routing converged cluster-wide).
func TestClusterMVResplit(t *testing.T) {
	stores := newInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()

	const (
		// "mvcoll" is arbitrary: name/shard routing no longer affects partitioned
		// create. The remote-create re-expansion bug (a remote-owned logical create
		// re-run by the owner's fanout decorator, racing into "partition already
		// exists"; see TestClusterPartitionedCreateRemoteShard) is fixed, so the
		// logical collection is created with Partitions=0 and is never re-expanded
		// regardless of which node owns its shard.
		name = "mvcoll"
		N    = 120
	)

	// Create the MV collection with P=4 through node 0 (the creating coordinator).
	retryUntil(t, "mv create", func() error {
		return stores[0].VectorMVCreateCollection(ctx, name, rostam.MultiVectorConfig{Dim: 4, Partitions: 4})
	})

	// Populate 120 tie-free docs (docIDs 1..120) through node 0, each with a unique
	// angular token and metadata {"docid": NewInt(id)}.
	for id := 1; id <= N; id++ {
		idc := uint64(id)
		md := rostam.VectorMetadata{"docid": vector.NewInt(int64(id))}
		retryUntil(t, "mv add", func() error {
			return stores[0].VectorMVAdd(ctx, name, idc, [][]float32{mvTokenAt(id)}, md)
		})
	}

	// Resplit 4 -> 8 from node 0 (caller-quiesced; writing above has stopped).
	retryUntil(t, "mv resplit", func() error { return stores[0].VectorMVResplit(ctx, name, 8) })

	// Wait for {P=8, gen=1} to converge on nodes 1 and 2 before verifying from them.
	for _, i := range []int{1, 2} {
		ei := stores[i].(*rostam.Embedded)
		waitEmbeddedCatalogGen(t, ei, name, 8, 1, 5*time.Second)
	}

	// Cross-node MV search of the resplit (P=8, gen1) collection FROM node 1 (a
	// non-creating coordinator): the winner's own token returns it at rank 0 with
	// its metadata intact. winner=17 hashes to a non-zero gen-1 partition, so a
	// correct rank-0 proves the gen-aware fan-out reached the right physical part.
	const winner = 17
	res, _, err := stores[1].VectorMVSearch(ctx, name, [][]float32{mvTokenAt(winner)}, 5, rostam.MultiSearchOpts{CandidatesPerToken: 100})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || res[0].ID != winner {
		t.Fatalf("post-resplit MV search from node 1: %+v (want rank-0 id=%d)", res, winner)
	}
	// Metadata preserved through resplit + fan-out. Oracle is the inserted value,
	// NewInt(winner), not another search.
	wantMD := vector.NewInt(int64(winner))
	gv, hasMD := res[0].Metadata["docid"]
	if !hasMD || !gv.Equal(wantMD) {
		t.Fatalf("winner metadata[docid] = %+v (present=%v), want %+v (dropped across resplit + cluster fan-out)", gv, hasMD, wantMD)
	}

	// New-gen routing converged cluster-wide: add a fresh doc via node 2 AFTER the
	// resplit (routes by the new gen-1 partition map) and find it from node 1.
	const fresh = N + 1 // 121
	freshMD := rostam.VectorMetadata{"docid": vector.NewInt(int64(fresh))}
	retryUntil(t, "mv add post-resplit", func() error {
		return stores[2].VectorMVAdd(ctx, name, uint64(fresh), [][]float32{mvTokenAt(fresh)}, freshMD)
	})
	r2, _, err := stores[1].VectorMVSearch(ctx, name, [][]float32{mvTokenAt(fresh)}, 1, rostam.MultiSearchOpts{CandidatesPerToken: 100})
	if err != nil {
		t.Fatal(err)
	}
	if len(r2) == 0 || r2[0].ID != fresh {
		t.Fatalf("post-resplit MV add via node 2 not found from node 1: %+v (want id=%d)", r2, fresh)
	}
	if gv, ok := r2[0].Metadata["docid"]; !ok || !gv.Equal(vector.NewInt(int64(fresh))) {
		t.Fatalf("post-resplit add metadata[docid] = %+v (present=%v), want %+v", gv, ok, vector.NewInt(int64(fresh)))
	}
}

// TestClusterMVResplitCleanup is the MV mirror of TestClusterResplitCleanup on a
// real 3-node cluster: orphan gen-0 MV partitions seeded across shards (simulating
// a post-flip drop-old failure) are swept by VectorMVResplitCleanup invoked FROM a
// non-creating node, the live gen-1 data is untouched and still searchable from
// another node, and a second cleanup is idempotent.
//
// The orphans are seeded via routed vector_mv_create_collection calls keyed by the
// physical gen-0 partition name, exactly mirroring how the dense cluster cleanup
// test seeds its orphans — so the 4 leaks land on (possibly different) shards
// across the cluster, exercising cross-shard MV sweep.
func TestClusterMVResplitCleanup(t *testing.T) {
	stores := newInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()

	const (
		name = "mvcln"
		N    = 120
	)

	// Create P=4 MV collection through node 0, populate 120 tie-free docs.
	retryUntil(t, "mv create", func() error {
		return stores[0].VectorMVCreateCollection(ctx, name, rostam.MultiVectorConfig{Dim: 4, Partitions: 4})
	})
	for id := 1; id <= N; id++ {
		idc := uint64(id)
		md := rostam.VectorMetadata{"docid": vector.NewInt(int64(id))}
		retryUntil(t, "mv add", func() error {
			return stores[0].VectorMVAdd(ctx, name, idc, [][]float32{mvTokenAt(id)}, md)
		})
	}

	// Resplit 4 -> 8 from node 0; live state becomes {8, gen1}, gen0 dropped.
	retryUntil(t, "mv resplit", func() error { return stores[0].VectorMVResplit(ctx, name, 8) })

	// Wait for {P=8, gen=1} to converge on node 1 (the cleanup coordinator).
	e1 := stores[1].(*rostam.Embedded)
	waitEmbeddedCatalogGen(t, e1, name, 8, 1, 5*time.Second)

	// Simulate post-flip old-gen-0 leaks: re-create gen-0 MV partitions via normal
	// op routing. Each call routes by the physical name's shard key, so the 4 leaks
	// land across the cluster's shards — exercising cross-shard MV cleanup.
	ee0 := stores[0].(*rostam.Embedded)
	physCfg := rostam.MultiVectorConfig{Dim: 4}
	for p := 0; p < 4; p++ {
		phys := string(ops.PartitionKeyGen(name, 0, p))
		retryUntil(t, "mv leak", func() error {
			_, err := ee0.Call(ctx, "vector_mv_create_collection", ops.EncodeMVCreateArgs(phys, physCfg))
			return err
		})
	}

	// Cleanup from node 1 (non-creating coordinator): reads liveGen from its
	// converged local catalog and drops the 4 gen-0 orphans wherever they live.
	dropped, err := stores[1].VectorMVResplitCleanup(ctx, name)
	if err != nil {
		t.Fatal(err)
	}
	if dropped != 4 {
		t.Fatalf("cluster MV cleanup dropped %d, want 4", dropped)
	}

	// Wait for node 2's catalog to converge, then verify live gen-1 data intact +
	// searchable from it: the winner's token still returns it at rank 0 with metadata.
	e2 := stores[2].(*rostam.Embedded)
	waitEmbeddedCatalogGen(t, e2, name, 8, 1, 5*time.Second)
	const winner = 17
	res, _, err := stores[2].VectorMVSearch(ctx, name, [][]float32{mvTokenAt(winner)}, 5, rostam.MultiSearchOpts{CandidatesPerToken: 100})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || res[0].ID != winner {
		t.Fatalf("live MV data damaged by cleanup (from node 2): %+v (want rank-0 id=%d)", res, winner)
	}
	if gv, ok := res[0].Metadata["docid"]; !ok || !gv.Equal(vector.NewInt(int64(winner))) {
		t.Fatalf("live winner metadata[docid] = %+v (present=%v), want %+v", gv, ok, vector.NewInt(int64(winner)))
	}

	// Idempotent: second cleanup from node 1 drops nothing.
	d2, err := stores[1].VectorMVResplitCleanup(ctx, name)
	if err != nil {
		t.Fatal(err)
	}
	if d2 != 0 {
		t.Fatalf("second cluster MV cleanup dropped %d, want 0", d2)
	}
}

// clusterTCPClient stands up a real binary-TCP transport entry point onto a
// 3-node cluster: it mounts server.New over node 0's fan-out decorator (exactly
// the wrap server.go's CLUSTER branch installs for TCP — rostam.NewFanoutDispatcher(emb,
// emb.Node())) and returns a rostam.NewClient connected to it. Every op the returned
// client issues therefore crosses a REAL transport codec (client -> TCP -> server
// -> decorator) into node 0's embedded backend, which coordinates the
// cross-shard/partition op across the whole cluster. Cleanup is registered with t.
//
// This is the same harness shape as TestRemoteResplitTCPClient (single node) but
// the decorator's embedded backend is node 0 of a genuine multi-node cluster, so
// a resplit driven over the wire flips the durable meta-Raft catalog cluster-wide
// — provable from the NON-creating nodes' own embedded catalogs.
func clusterTCPClient(t *testing.T, stores []rostam.Store, reg *ops.Registry) rostam.Store {
	t.Helper()
	emb0 := stores[0].(*rostam.Embedded)
	disp := rostam.NewFanoutDispatcher(emb0, emb0.Node())
	srv, err := server.New(server.Config{Addr: "127.0.0.1:0", Dispatcher: disp})
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	go func() { _ = srv.Serve() }()
	t.Cleanup(func() { _ = srv.Close() })

	client, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{srv.Addr().String()}, Ops: reg})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return client
}

// TestClusterRemoteResplit is the end-to-end headline guarantee of resplit driven
// over a REAL network transport against a 3-node cluster: a dense P=4 collection
// is resplit to P=8 by a remote Go client over TCP (the resplit op crosses a real
// codec into node 0's fan-out decorator), the meta-Raft catalog gen-flips
// cluster-wide, and a NON-creating node (node 1) sees the new generation and
// serves correct, complete, gen-routed data. Cleanup is then driven over the same
// transport: clean -> 0, seeded orphans -> exact count, idempotent -> 0.
//
// This differs from TestRemoteResplitTCPClient (single embedded node) in the
// non-negotiable property it proves: the resplit-over-transport flips the DURABLE
// cluster catalog so a node that did NOT initiate the resplit re-routes every
// search mode to the new gen — the distributed decorator-virtual-op coordinator path.
func TestClusterRemoteResplit(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	stores := newInmemEmbeddedCluster(t, 3, 8)
	client := clusterTCPClient(t, stores, reg)

	// Generous deadline: resplit is synchronous + offline (scan + re-insert of
	// every vector across the cluster) and must complete within the single call.
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	const coll = "rdocs"
	const N = 600

	// Create P=4 + populate THROUGH the remote client (over TCP into node 0's
	// decorator). Each doc carries content + even metadata so the content/filter
	// assertions below are load-bearing. retryUntil rides the startup election
	// window for the per-shard physical creates/inserts.
	retryUntil(t, "client create", func() error {
		return client.CreateCollection(ctx, coll, rostam.VectorConfig{
			Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: 4,
		})
	})
	if p, _, ok := stores[0].(*rostam.Embedded).Catalog().PartitionsGen(coll); !ok || p != 4 {
		t.Fatalf("after client create: PartitionsGen = (%d, ok=%v), want (4, true)", p, ok)
	}
	for id := uint64(1); id <= N; id++ {
		v := []float32{float32(id), 0, 0, 0}
		md := vector.Metadata{"even": vector.NewBool(id%2 == 0)}
		content := fmt.Sprintf("doc-%d", id)
		idc := id
		retryUntil(t, "client upsert", func() error {
			return client.VectorUpsert(ctx, coll, idc, v, content, rostam.VectorInsertOpts{Metadata: md})
		})
	}

	// Resplit 4 -> 8 OVER TCP (caller-quiesced; writing above has stopped). This is
	// the crux: the resplit op travels over the real transport codec into the
	// decorator, which coordinates the cluster-wide gen-flip.
	if err := client.VectorResplit(ctx, coll, 8); err != nil {
		t.Fatalf("client VectorResplit over TCP: %v", err)
	}

	// Convergence on a NON-creator node: node 1 did not initiate the resplit yet
	// must observe {P=8, gen=1} via the durable meta-Raft catalog.
	e1 := stores[1].(*rostam.Embedded)
	waitEmbeddedCatalogGen(t, e1, coll, 8, 1, 5*time.Second)

	// Data integrity from the non-creator node 1 after the gen-flip:
	// (a) dense + content search: nearest to {0.5,0,0,0} is id=1 with "doc-1".
	res, _, err := stores[1].VectorSearchDocs(ctx, coll, []float32{0.5, 0, 0, 0}, 5, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || res[0].ID != 1 || res[0].Content != "doc-1" {
		t.Fatalf("post-resplit dense search from node 1: %+v (want id=1, content doc-1)", res)
	}

	// (b) scroll completeness from node 1: every doc exactly once (no drops/dups).
	all, _, _, err := stores[1].VectorScroll(ctx, coll, rostam.VectorFilter{}, 0, rostam.VectorScrollOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != N || len(idSet(all)) != N {
		t.Fatalf("post-resplit scroll from node 1: %d docs (%d distinct), want %d/%d", len(all), len(idSet(all)), N, N)
	}

	// (c) filtered fan-out from node 1: exactly 300 even docs, none odd.
	even, _, err := stores[1].VectorSearchDocs(ctx, coll, []float32{0.5, 0, 0, 0}, N, rostam.VectorSearchOpts{
		Filter: rostam.VectorFilter{Op: vector.FilterEq, Field: "even", Value: vector.NewBool(true)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(even) != N/2 {
		t.Fatalf("post-resplit filtered from node 1: %d, want %d", len(even), N/2)
	}
	for _, d := range even {
		if d.ID%2 != 0 {
			t.Fatalf("post-resplit filtered returned odd id %d (filter even==true)", d.ID)
		}
	}

	// (d) a post-resplit insert routes by the NEW generation: insert id=601 OVER
	// TCP (the client -> node 0 decorator path routes by the converged gen-1 map)
	// and find it from non-creator node 1.
	retryUntil(t, "client post-resplit upsert", func() error {
		return client.VectorUpsert(ctx, coll, 601, []float32{601, 0, 0, 0}, "doc-601",
			rostam.VectorInsertOpts{Metadata: vector.Metadata{"even": vector.NewBool(false)}})
	})
	r2, _, err := stores[1].VectorSearchDocs(ctx, coll, []float32{601, 0, 0, 0}, 1, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(r2) == 0 || r2[0].ID != 601 {
		t.Fatalf("post-resplit insert over TCP not found from node 1: %+v", r2)
	}

	// Cleanup OVER TCP. After a clean resplit there are no orphans (VectorResplit
	// drops the old gen as its final step), so cleanup must report exactly 0.
	dropped, err := client.VectorResplitCleanup(ctx, coll)
	if err != nil {
		t.Fatalf("client VectorResplitCleanup over TCP: %v", err)
	}
	if dropped != 0 {
		t.Fatalf("cleanup after clean cluster resplit dropped %d, want 0", dropped)
	}

	// Seed orphan partitions in a NON-live generation (gen 2; live is gen 1) across
	// the cluster via normal op routing, mirroring TestClusterResplitCleanup — each
	// routed create lands on (possibly different) shards, exercising cross-shard sweep.
	const orphans = 4
	ee0 := stores[0].(*rostam.Embedded)
	physCfg := rostam.VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}
	for p := 0; p < orphans; p++ {
		phys := string(ops.PartitionKeyGen(coll, 2, p))
		pc := phys
		retryUntil(t, "seed gen-2 orphan", func() error {
			_, err := ee0.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(pc, physCfg))
			return err
		})
	}

	// Cleanup OVER TCP sweeps exactly the seeded orphans; the count flows faithfully
	// back through the remote codec.
	dropped, err = client.VectorResplitCleanup(ctx, coll)
	if err != nil {
		t.Fatalf("client VectorResplitCleanup (orphans) over TCP: %v", err)
	}
	if dropped != orphans {
		t.Fatalf("cluster cleanup over TCP dropped %d, want %d (exact seeded orphan count)", dropped, orphans)
	}

	// Idempotent: a second cleanup over TCP drops nothing.
	dropped, err = client.VectorResplitCleanup(ctx, coll)
	if err != nil {
		t.Fatalf("client VectorResplitCleanup (idempotent) over TCP: %v", err)
	}
	if dropped != 0 {
		t.Fatalf("second cluster cleanup over TCP dropped %d, want 0 (idempotent)", dropped)
	}
}

// TestClusterRemoteMVResplit is the multi-vector counterpart to
// TestClusterRemoteResplit: a partitioned (P=4) MV collection is resplit to P=8
// by a remote Go client over TCP (the MV resplit op crosses a real codec into
// node 0's fan-out decorator), the meta-Raft catalog gen-flips cluster-wide, and a
// NON-creating node (node 1) serves correct MaxSim late-interaction search WITH
// metadata over the new generation. Cleanup over the same transport: clean -> 0,
// seeded orphans -> exact count, idempotent -> 0.
//
// Tie-free angular tokens (mvTokenAt) make the MaxSim winner unambiguous across
// the partition fan-out; each doc carries {"docid": NewInt(id)} so the
// metadata-preservation assertion is load-bearing.
func TestClusterRemoteMVResplit(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	stores := newInmemEmbeddedCluster(t, 3, 8)
	client := clusterTCPClient(t, stores, reg)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	const coll = "mvrdocs"
	const N = 120

	// Create P=4 MV + populate 120 tie-free docs THROUGH the remote client (over
	// TCP into node 0's decorator). Each doc carries {"docid": NewInt(id)}.
	retryUntil(t, "client mv create", func() error {
		return client.VectorMVCreateCollection(ctx, coll, rostam.MultiVectorConfig{Dim: 4, Partitions: 4})
	})
	if p, _, ok := stores[0].(*rostam.Embedded).Catalog().PartitionsGen(coll); !ok || p != 4 {
		t.Fatalf("after client mv create: PartitionsGen = (%d, ok=%v), want (4, true)", p, ok)
	}
	for id := 1; id <= N; id++ {
		idc := uint64(id)
		md := rostam.VectorMetadata{"docid": vector.NewInt(int64(id))}
		retryUntil(t, "client mv add", func() error {
			return client.VectorMVAdd(ctx, coll, idc, [][]float32{mvTokenAt(id)}, md)
		})
	}

	// MV resplit 4 -> 8 OVER TCP — the crux: the op crosses the real transport codec
	// into the decorator, which coordinates the cluster-wide gen-flip.
	if err := client.VectorMVResplit(ctx, coll, 8); err != nil {
		t.Fatalf("client VectorMVResplit over TCP: %v", err)
	}

	// Convergence on a NON-creator node: node 1 observes {P=8, gen=1}.
	e1 := stores[1].(*rostam.Embedded)
	waitEmbeddedCatalogGen(t, e1, coll, 8, 1, 5*time.Second)

	// MV search of the resplit (P=8, gen1) collection FROM non-creator node 1: the
	// winner's own token returns it at rank 0 (deterministic via tie-free tokens)
	// with its metadata intact. winner=17 hashes to a non-zero gen-1 partition, so a
	// correct rank-0 proves the gen-aware fan-out reached the right physical part.
	const winner = 17
	res, _, err := stores[1].VectorMVSearch(ctx, coll, [][]float32{mvTokenAt(winner)}, 5, rostam.MultiSearchOpts{CandidatesPerToken: 100})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || res[0].ID != winner {
		t.Fatalf("post-resplit MV search from node 1: %+v (want rank-0 id=%d)", res, winner)
	}
	wantMD := vector.NewInt(int64(winner))
	gv, hasMD := res[0].Metadata["docid"]
	if !hasMD || !gv.Equal(wantMD) {
		t.Fatalf("winner metadata[docid] = %+v (present=%v), want %+v (dropped across resplit + cluster fan-out)", gv, hasMD, wantMD)
	}

	// New-gen routing converged cluster-wide: add a fresh doc OVER TCP AFTER the
	// resplit (routes by the new gen-1 partition map) and find it from node 1.
	const fresh = N + 1 // 121
	freshMD := rostam.VectorMetadata{"docid": vector.NewInt(int64(fresh))}
	retryUntil(t, "client mv add post-resplit", func() error {
		return client.VectorMVAdd(ctx, coll, uint64(fresh), [][]float32{mvTokenAt(fresh)}, freshMD)
	})
	r2, _, err := stores[1].VectorMVSearch(ctx, coll, [][]float32{mvTokenAt(fresh)}, 1, rostam.MultiSearchOpts{CandidatesPerToken: 100})
	if err != nil {
		t.Fatal(err)
	}
	if len(r2) == 0 || r2[0].ID != fresh {
		t.Fatalf("post-resplit MV add over TCP not found from node 1: %+v (want id=%d)", r2, fresh)
	}
	if gv, ok := r2[0].Metadata["docid"]; !ok || !gv.Equal(vector.NewInt(int64(fresh))) {
		t.Fatalf("post-resplit add metadata[docid] = %+v (present=%v), want %+v", gv, ok, vector.NewInt(int64(fresh)))
	}

	// Cleanup OVER TCP. After a clean MV resplit there are no orphans, so cleanup
	// must report exactly 0.
	dropped, err := client.VectorMVResplitCleanup(ctx, coll)
	if err != nil {
		t.Fatalf("client VectorMVResplitCleanup over TCP: %v", err)
	}
	if dropped != 0 {
		t.Fatalf("mv cleanup after clean cluster resplit dropped %d, want 0", dropped)
	}

	// Seed orphan gen-2 MV partitions across the cluster via routed creates,
	// mirroring TestClusterMVResplitCleanup.
	const orphans = 4
	ee0 := stores[0].(*rostam.Embedded)
	physCfg := rostam.MultiVectorConfig{Dim: 4}
	for p := 0; p < orphans; p++ {
		phys := string(ops.PartitionKeyGen(coll, 2, p))
		pc := phys
		retryUntil(t, "seed gen-2 mv orphan", func() error {
			_, err := ee0.Call(ctx, "vector_mv_create_collection", ops.EncodeMVCreateArgs(pc, physCfg))
			return err
		})
	}

	// Cleanup OVER TCP sweeps exactly the seeded orphans; the count flows back
	// faithfully through the remote codec.
	dropped, err = client.VectorMVResplitCleanup(ctx, coll)
	if err != nil {
		t.Fatalf("client VectorMVResplitCleanup (orphans) over TCP: %v", err)
	}
	if dropped != orphans {
		t.Fatalf("cluster MV cleanup over TCP dropped %d, want %d (exact seeded orphan count)", dropped, orphans)
	}

	// Idempotent: a second cleanup over TCP drops nothing.
	dropped, err = client.VectorMVResplitCleanup(ctx, coll)
	if err != nil {
		t.Fatalf("client VectorMVResplitCleanup (idempotent) over TCP: %v", err)
	}
	if dropped != 0 {
		t.Fatalf("second cluster MV cleanup over TCP dropped %d, want 0 (idempotent)", dropped)
	}
}

// waitEmbeddedReshardStable polls a node's local catalog until the collection's
// reshard state is cleared (Stable / absent), failing on timeout. The Phase-5
// SetReshardState(Stable) clear and the Phase-4 catalog gen-flip are SEPARATE
// meta-Raft entries: SetCollectionReshard only guarantees read-your-writes on the
// COORDINATOR (node 0); non-creator nodes apply the clear asynchronously through
// normal raft replication, slightly AFTER they observe the gen-flip. So a node can
// transiently report the new gen while the Stable clear has not yet replicated to
// it — this is a convergence wait (consensus-gated, no fixed sleep), the exact
// analogue of waitEmbeddedCatalogGen for the reshard-state map.
func waitEmbeddedReshardStable(t *testing.T, e *rostam.Embedded, collection string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if st, on := e.Catalog().ReshardState(collection); !on || st.Status == 0 {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	st, on := e.Catalog().ReshardState(collection)
	t.Fatalf("reshard state for %q not cleared within %s: %+v on=%v", collection, timeout, st, on)
}

// clusterReshardScanGen scans every physical partition of (coll,gen,P) THROUGH a
// cluster node's routed Call (node 0 forwards each physical scan to the shard's
// owner at RF=1), returning a map id -> vector. It is the cluster mirror of the
// embedded reshardScanGen oracle: it asserts each landed id is routed to the
// correct PartitionOf(id,P) and appears in exactly one gen-P partition, so a
// clobbered value, a mis-routed copy, or a duplicate is caught directly on the
// new generation rather than only through the merge fan-out. value-tagged vectors
// (tag in vec[1]) make Race A (clobber) and Race B (resurrection) observable.
func clusterReshardScanGen(t *testing.T, ee *rostam.Embedded, coll string, P int, gen uint32) map[uint64][]float32 {
	t.Helper()
	out := map[uint64][]float32{}
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen(coll, gen, p))
		body, err := ee.Call(context.Background(), "vector_scan_vectors", ops.EncodeScanVectorsArgs(phys))
		if err != nil {
			t.Fatalf("scan %s: %v", phys, err)
		}
		recs, err := ops.DecodeScanVectorsResult(body)
		if err != nil {
			t.Fatalf("decode scan %s: %v", phys, err)
		}
		for _, r := range recs {
			if want := ops.PartitionOf(r.ID, P); want != p {
				t.Fatalf("id %d found in partition %d but PartitionOf says %d (gen %d, P %d)", r.ID, p, want, gen, P)
			}
			if _, dup := out[r.ID]; dup {
				t.Fatalf("id %d present in more than one gen-%d partition", r.ID, gen)
			}
			out[r.ID] = append([]float32(nil), r.Vec...)
		}
	}
	return out
}

// clusterReshardScanGenMV is the MV mirror of clusterReshardScanGen: it scans
// every physical gen-P MV partition through a routed Call and returns docID ->
// (token matrix + "tag" metadata), asserting correct routing and no duplicates.
func clusterReshardScanGenMV(t *testing.T, ee *rostam.Embedded, coll string, P int, gen uint32) map[uint64]mvDoc {
	t.Helper()
	out := map[uint64]mvDoc{}
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen(coll, gen, p))
		body, err := ee.Call(context.Background(), "vector_mv_scan_vectors", ops.EncodeMVScanArgs(phys))
		if err != nil {
			t.Fatalf("mv scan %s: %v", phys, err)
		}
		recs, err := ops.DecodeMVScanResult(body)
		if err != nil {
			t.Fatalf("decode mv scan %s: %v", phys, err)
		}
		for _, r := range recs {
			if want := ops.PartitionOf(r.ID, P); want != p {
				t.Fatalf("doc %d found in partition %d but PartitionOf says %d (gen %d, P %d)", r.ID, p, want, gen, P)
			}
			if _, dup := out[r.ID]; dup {
				t.Fatalf("doc %d present in more than one gen-%d partition", r.ID, gen)
			}
			tag := int64(-1)
			if r.Metadata != nil {
				if v, ok := r.Metadata["tag"]; ok && v.Kind == vector.ValueInt {
					tag = v.Int
				}
			}
			toks := make([][]float32, len(r.Tokens))
			for i := range r.Tokens {
				toks[i] = append([]float32(nil), r.Tokens[i]...)
			}
			out[r.ID] = mvDoc{tokens: toks, tag: tag}
		}
	}
	return out
}

// TestClusterOnlineReshard is the headline cluster guarantee of online resharding
// (this plan): a partitioned (P=4) dense collection is resharded LIVE to P=8 by
// the coordinator (node 0) while a NON-creator node (node 2) issues concurrent
// upserts (Race A: value clobber) and deletes (Race B: delete resurrection)
// THROUGHOUT the reshard. After the reshard returns and the writers join, every
// node must converge to the new generation (waitEmbeddedCatalogGen, consensus-
// gated — NO sleeps except the orchestrator's own lowered drain grace) and serve
// the EXACT expected live set (base + inserts + upserted-values − deletes) with
// correct per-id values.
//
// Unlike the offline TestClusterResplit (caller-quiesced), this proves the
// distributed dual-write path: writes driven from node 2 during the reshard land
// in BOTH the live gen and the new gen across shards, the if-absent copy never
// clobbers a newer concurrent write, and the per-record resurrection guard keeps
// concurrently-deleted ids gone. The value-tagged-vector oracle
// (clusterReshardScanGen) catches both races directly on the new gen, and the
// per-node scroll proves the cutover flip propagated through meta-Raft to all
// three nodes.
//
// RF=1 (matching the existing resplit cluster tests): each physical partition has
// a single owner that forwards point writes, so a non-creator node's concurrent
// upsert/delete is reliably applied — exactly the routing the non-creator-writer
// scenario needs. (At RF>1 a node hosting a shard as a follower returns NotLeader
// for a write and does NOT forward it; see the newInmemEmbeddedCluster comment.)
func TestClusterOnlineReshard(t *testing.T) {
	// Lower the orchestrator drain grace (the ONLY sleep in the flow) so the test
	// is fast; restore on exit. Same pattern as the embedded money test.
	defer rostam.SetReshardDrainGrace(30 * time.Millisecond)()

	stores := newInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()
	const coll = "docs"
	const oldP, newP = 4, 8

	// Create P=4 through node 0 (the coordinator).
	retryUntil(t, "create", func() error {
		return stores[0].CreateCollection(ctx, coll, rostam.VectorConfig{
			Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: oldP,
		})
	})

	// Disjoint per-role id ranges so the expected live set is exact:
	//   base ids:   1..baseN              (some upserted, some deleted)
	//   insert ids: 10000..10000+insN-1   (added DURING the reshard)
	const baseN = 240
	const insN = 200
	const upsertN = 120 // base ids 1..upsertN get their value changed (tag 1 -> 2)
	const deleteN = 60  // base ids [upsertN+1 .. upsertN+deleteN] get deleted

	// vecOf tags the vector with a version in vec[1] so a clobbered/stale value is
	// detectable: base value tag=1, upserted value tag=2.
	vecOf := func(id uint64, tag float32) []float32 { return []float32{float32(id), tag, 0, 0} }

	for id := uint64(1); id <= baseN; id++ {
		idc, vc := id, vecOf(id, 1)
		retryUntil(t, "seed insert", func() error {
			return stores[0].VectorInsertExt(ctx, coll, idc, vc, rostam.VectorInsertOpts{})
		})
	}

	// Expected live set, computed deterministically (the writers below do EXACTLY this).
	expected := map[uint64][]float32{}
	for id := uint64(1); id <= baseN; id++ {
		expected[id] = vecOf(id, 1)
	}
	for id := uint64(1); id <= upsertN; id++ {
		expected[id] = vecOf(id, 2) // upserted value wins
	}
	for id := uint64(upsertN + 1); id <= upsertN+deleteN; id++ {
		delete(expected, id) // deleted ids gone
	}
	for i := 0; i < insN; i++ {
		id := uint64(10000 + i)
		expected[id] = vecOf(id, 1)
	}

	// Reshard 4 -> 8 from node 0 (the coordinator) in a goroutine.
	var wg sync.WaitGroup
	reshardErr := make(chan error, 1)
	wg.Add(1)
	go func() {
		defer wg.Done()
		reshardErr <- stores[0].VectorReshard(ctx, coll, newP)
	}()

	// Concurrent writers from the NON-CREATOR node 2 (writes forward to each id's
	// owning shard leader). Each role re-drives its disjoint id range to its final
	// state in a loop until the reshard returns, deterministically exercising the
	// live dual-write window across the cluster. Upsert/delete are idempotent so
	// continuous re-application is safe.
	done := make(chan struct{})
	worker := func(fn func()) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-done:
					return
				default:
					fn()
				}
			}
		}()
	}
	// Upserter: change base ids 1..upsertN to tag=2 (from non-creator node 2).
	worker(func() {
		for id := uint64(1); id <= upsertN; id++ {
			_ = stores[2].VectorUpsert(ctx, coll, id, vecOf(id, 2), "", rostam.VectorInsertOpts{})
		}
	})
	// Deleter: delete base ids [upsertN+1 .. upsertN+deleteN] (from non-creator node 2).
	worker(func() {
		for id := uint64(upsertN + 1); id <= upsertN+deleteN; id++ {
			_, _ = stores[2].VectorDelete(ctx, coll, id)
		}
	})
	// Inserter: add new ids 10000..10000+insN-1 (from non-creator node 2).
	worker(func() {
		for i := 0; i < insN; i++ {
			id := uint64(10000 + i)
			_ = stores[2].VectorUpsert(ctx, coll, id, vecOf(id, 1), "", rostam.VectorInsertOpts{})
		}
	})

	err := <-reshardErr
	close(done)
	wg.Wait()
	if err != nil {
		t.Fatalf("VectorReshard from node 0: %v", err)
	}

	// Consensus-gated convergence on EVERY node (no sleeps): all three observe the
	// flipped catalog (P=8, gen=1) AND the cleared (Stable) reshard state. The two
	// are separate meta-Raft entries that replicate to non-creator nodes
	// independently, so both must be awaited.
	for i := range stores {
		waitEmbeddedCatalogGen(t, stores[i].(*rostam.Embedded), coll, newP, 1, 10*time.Second)
		waitEmbeddedReshardStable(t, stores[i].(*rostam.Embedded), coll, 10*time.Second)
	}

	// New gen holds EXACTLY the expected live set with exact values — scanned via
	// node 0's routed Call across the cluster's shards. Catches Race A (clobber:
	// wrong tag) and a lost write (missing id).
	got := clusterReshardScanGen(t, stores[0].(*rostam.Embedded), coll, newP, 1)
	if len(got) != len(expected) {
		t.Fatalf("new gen has %d ids, want %d (lost write or resurrection)", len(got), len(expected))
	}
	for id, wantVec := range expected {
		gotVec, ok := got[id]
		if !ok {
			t.Fatalf("id %d missing from new gen (lost write or wrongly deleted)", id)
		}
		if !reflect.DeepEqual(gotVec, wantVec) {
			t.Fatalf("id %d value = %v, want %v (clobbered/stale across cluster reshard)", id, gotVec, wantVec)
		}
	}
	// Race B: deleted ids must be absent from the new gen.
	for id := uint64(upsertN + 1); id <= upsertN+deleteN; id++ {
		if _, ok := got[id]; ok {
			t.Fatalf("deleted id %d resurrected in new gen", id)
		}
	}

	// Old-gen partitions all dropped cluster-wide (scanned via node 1, a non-creator).
	e1 := stores[1].(*rostam.Embedded)
	for p := 0; p < oldP; p++ {
		phys := string(ops.PartitionKeyGen(coll, 0, p))
		if _, err := e1.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(phys)); err == nil {
			t.Fatalf("old gen-0 partition %d not dropped after reshard", p)
		}
	}

	// Every node serves the exact live set via scroll over the new-gen fan-out.
	for i := range stores {
		all, _, _, err := stores[i].VectorScroll(ctx, coll, rostam.VectorFilter{}, 0, rostam.VectorScrollOpts{})
		if err != nil {
			t.Fatalf("node %d scroll: %v", i, err)
		}
		set := idSet(all)
		if len(set) != len(expected) {
			t.Fatalf("node %d scroll: %d distinct ids, want %d", i, len(set), len(expected))
		}
		for id := range expected {
			if !set[id] {
				t.Fatalf("node %d scroll missing id %d", i, id)
			}
		}
		for id := uint64(upsertN + 1); id <= upsertN+deleteN; id++ {
			if set[id] {
				t.Fatalf("node %d scroll resurrected deleted id %d", i, id)
			}
		}
	}
}

// TestClusterMVOnlineReshard is the multi-vector mirror of TestClusterOnlineReshard:
// a partitioned (P=4) MV collection is resharded LIVE to P=8 from node 0 while a
// NON-creator node (node 2) concurrently mv-adds NEW docs, mv-adds (overwrites)
// existing docs with a changed token matrix + metadata tag (Race A), and
// mv-deletes existing docs (Race B) for the whole reshard. After convergence on
// every node, the new gen must hold EXACTLY the expected live set with the exact
// per-doc token matrix AND metadata tag preserved — proving the MV dual-write /
// if-absent copy / resurrection guard hold across the cluster and the token
// matrix + metadata thread through the distributed copy.
func TestClusterMVOnlineReshard(t *testing.T) {
	defer rostam.SetReshardDrainGrace(30 * time.Millisecond)()

	stores := newInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()
	const coll = "mvcoll"
	const oldP, newP = 4, 8

	retryUntil(t, "mv create", func() error {
		return stores[0].VectorMVCreateCollection(ctx, coll, rostam.MultiVectorConfig{Dim: 4, Partitions: oldP})
	})

	const baseN = 180
	const insN = 150
	const upsertN = 90 // base ids 1..upsertN get re-added (overwritten) tag 1 -> 2
	const deleteN = 45 // base ids [upsertN+1 .. upsertN+deleteN] get deleted

	// tokensTagged makes the first token depend on the tag so an overwrite produces
	// a detectably different token matrix; a copy-clobber would leave stale tag-1
	// tokens.
	tokensTagged := func(id uint64, tag int) [][]float32 {
		toks := mvTokensFor(id)
		toks[0] = mvTokenAt(int(id)*4 + 1000*tag)
		return toks
	}
	mdTagged := func(tag int) rostam.VectorMetadata { return rostam.VectorMetadata{"tag": vector.NewInt(int64(tag))} }

	for id := uint64(1); id <= baseN; id++ {
		idc, toks, md := id, tokensTagged(id, 1), mdTagged(1)
		retryUntil(t, "mv seed", func() error {
			return stores[0].VectorMVAdd(ctx, coll, idc, toks, md)
		})
	}

	// Expected live set (writers below do EXACTLY this).
	expected := map[uint64]mvDoc{}
	for id := uint64(1); id <= baseN; id++ {
		expected[id] = mvDoc{tokens: tokensTagged(id, 1), tag: 1}
	}
	for id := uint64(1); id <= upsertN; id++ {
		expected[id] = mvDoc{tokens: tokensTagged(id, 2), tag: 2} // overwrite wins
	}
	for id := uint64(upsertN + 1); id <= upsertN+deleteN; id++ {
		delete(expected, id)
	}
	for i := 0; i < insN; i++ {
		id := uint64(10000 + i)
		expected[id] = mvDoc{tokens: tokensTagged(id, 1), tag: 1}
	}

	var wg sync.WaitGroup
	reshardErr := make(chan error, 1)
	wg.Add(1)
	go func() {
		defer wg.Done()
		reshardErr <- stores[0].VectorMVReshard(ctx, coll, newP)
	}()

	done := make(chan struct{})
	worker := func(fn func()) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-done:
					return
				default:
					fn()
				}
			}
		}()
	}
	// Overwriter (from non-creator node 2): re-add base ids 1..upsertN tag=2.
	worker(func() {
		for id := uint64(1); id <= upsertN; id++ {
			_ = stores[2].VectorMVAdd(ctx, coll, id, tokensTagged(id, 2), mdTagged(2))
		}
	})
	// Deleter (from non-creator node 2).
	worker(func() {
		for id := uint64(upsertN + 1); id <= upsertN+deleteN; id++ {
			_, _ = stores[2].VectorMVDelete(ctx, coll, id)
		}
	})
	// Inserter (from non-creator node 2): add new ids 10000..10000+insN-1.
	worker(func() {
		for i := 0; i < insN; i++ {
			id := uint64(10000 + i)
			_ = stores[2].VectorMVAdd(ctx, coll, id, tokensTagged(id, 1), mdTagged(1))
		}
	})

	err := <-reshardErr
	close(done)
	wg.Wait()
	if err != nil {
		t.Fatalf("VectorMVReshard from node 0: %v", err)
	}

	// Consensus-gated convergence on every node + Stable state cluster-wide (both
	// the gen-flip and the Stable clear are awaited; see waitEmbeddedReshardStable).
	for i := range stores {
		waitEmbeddedCatalogGen(t, stores[i].(*rostam.Embedded), coll, newP, 1, 10*time.Second)
		waitEmbeddedReshardStable(t, stores[i].(*rostam.Embedded), coll, 10*time.Second)
	}

	// New gen holds EXACTLY the expected live set with exact token matrix + tag.
	got := clusterReshardScanGenMV(t, stores[0].(*rostam.Embedded), coll, newP, 1)
	if len(got) != len(expected) {
		t.Fatalf("new gen has %d docs, want %d", len(got), len(expected))
	}
	for id, w := range expected {
		g, ok := got[id]
		if !ok {
			t.Fatalf("doc %d missing from new gen (lost write or wrongly deleted)", id)
		}
		if !mvTokensEqual(g.tokens, w.tokens) {
			t.Fatalf("doc %d tokens = %v, want %v (clobbered/stale across cluster reshard)", id, g.tokens, w.tokens)
		}
		if g.tag != w.tag {
			t.Fatalf("doc %d metadata tag = %d, want %d (clobbered or metadata dropped)", id, g.tag, w.tag)
		}
	}
	for id := uint64(upsertN + 1); id <= upsertN+deleteN; id++ {
		if _, ok := got[id]; ok {
			t.Fatalf("deleted doc %d resurrected in new gen", id)
		}
	}

	// Old-gen MV partitions dropped cluster-wide (via non-creator node 1).
	e1 := stores[1].(*rostam.Embedded)
	for p := 0; p < oldP; p++ {
		phys := string(ops.PartitionKeyGen(coll, 0, p))
		if _, err := e1.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(phys)); err == nil {
			t.Fatalf("old gen-0 MV partition %d not dropped after reshard", p)
		}
	}

	// A surviving overwritten doc is searchable with its tag-2 metadata from a
	// non-creator node (winner uses a small id so its first-quadrant token angle is
	// tie-free). Proves the new-gen MV partitions are searchable cluster-wide and
	// the overwrite (not the stale base) won.
	const winner = 7 // <= upsertN, overwritten to tag 2
	res, _, err := stores[1].VectorMVSearch(ctx, coll, [][]float32{tokensTagged(winner, 2)[0]}, 5, rostam.MultiSearchOpts{CandidatesPerToken: 100})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || res[0].ID != winner {
		t.Fatalf("post-reshard MV search from node 1: %+v (want rank-0 id=%d)", res, winner)
	}
	if gv, ok := res[0].Metadata["tag"]; !ok || gv.Kind != vector.ValueInt || gv.Int != 2 {
		t.Fatalf("winner %d metadata tag = %+v (present=%v), want int 2 (overwrite lost or metadata dropped)", winner, gv, ok)
	}
}

// TestClusterOnlineReshardRemote drives an online dense reshard OVER A REAL TCP
// CLIENT into a 3-node cluster (client -> node 0's fanout decorator, which
// coordinates the cluster-wide reshard) WITH a concurrent writer over the same
// TCP client throughout, then asserts a NON-creator node (node 1) converges to the
// new generation and serves the correct live data. This is the cluster mirror of
// TestRemoteOnlineReshardTCPClient (single embedded) and of TestClusterRemoteResplit
// (offline, cluster) — it proves the reshard-virtual-op-over-transport flips the
// DURABLE meta-Raft catalog cluster-wide while live dual-writes from the remote
// client are preserved (no lost inserts, no resurrected deletes) across shards.
//
// It is added in addition to the embedded TestClusterOnlineReshard because it
// covers a distinct path: the reshard + concurrent writes cross a real network
// codec into the distributed coordinator, not the in-process embedded rostam.Store API.
func TestClusterOnlineReshardRemote(t *testing.T) {
	defer rostam.SetReshardDrainGrace(30 * time.Millisecond)()

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	stores := newInmemEmbeddedCluster(t, 3, 8)
	client := clusterTCPClient(t, stores, reg)

	ctx, cancel := context.WithTimeout(context.Background(), cpuScaled(2*time.Minute))
	defer cancel()
	const coll = "rdocs"
	const oldP, newP = 4, 8
	const base = 200

	vecOf := func(id uint64, tag float32) []float32 { return []float32{float32(id), tag, 0, 0} }

	// Create P=4 + seed base ids [0,base) THROUGH the TCP client (over the wire into
	// node 0's decorator). retryUntil rides the startup election window.
	retryUntil(t, "client create", func() error {
		return client.CreateCollection(ctx, coll, rostam.VectorConfig{
			Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: oldP,
		})
	})
	for id := uint64(0); id < base; id++ {
		idc := id
		retryUntil(t, "client seed", func() error {
			return client.VectorUpsert(ctx, coll, idc, vecOf(idc, 1), "", rostam.VectorInsertOpts{})
		})
	}

	// Concurrent writers over the SAME TCP client for the whole reshard:
	//   - inserter:   high disjoint id range [addedFrom,addedTo) — must all survive.
	//   - overwriter: existing seeded ids [upsFrom,upsTo) re-upserted to tag=2 (Race A:
	//                 value clobber) — survivors must hold the NEW tag, proving the
	//                 if-absent copy never clobbered a newer concurrent write over TCP.
	//   - deleter:    range [delFrom,delTo) out of the seeded set — must stay gone.
	// The three id ranges are disjoint (delete < overwrite, both inside [0,base)) so
	// the expected live set is exact.
	const addedFrom = 1000
	const addedTo = 1150
	const delFrom = 0
	const delTo = 40
	const upsFrom = 40
	const upsTo = 120

	// Expected live set, computed deterministically BEFORE launching the workers (the
	// workers below drive EXACTLY this). Base seeded tag=1; overwrite to tag=2 wins;
	// deletes removed; concurrent inserts added (tag=1).
	expected := map[uint64][]float32{}
	for id := uint64(0); id < base; id++ {
		expected[id] = vecOf(id, 1)
	}
	for id := uint64(upsFrom); id < upsTo; id++ {
		expected[id] = vecOf(id, 2) // overwritten value wins
	}
	for id := uint64(delFrom); id < delTo; id++ {
		delete(expected, id) // deleted ids gone
	}
	for id := uint64(addedFrom); id < addedTo; id++ {
		expected[id] = vecOf(id, 1)
	}

	var wg sync.WaitGroup
	done := make(chan struct{})
	writeErr := make(chan error, 3)
	worker := func(fn func() error) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-done:
					return
				default:
				}
				if err := fn(); err != nil {
					select {
					case writeErr <- err:
					default:
					}
					return
				}
			}
		}()
	}
	worker(func() error {
		for id := uint64(addedFrom); id < addedTo; id++ {
			if err := client.VectorUpsert(ctx, coll, id, vecOf(id, 1), "", rostam.VectorInsertOpts{}); err != nil {
				return fmt.Errorf("concurrent upsert %d: %w", id, err)
			}
		}
		return nil
	})
	worker(func() error {
		for id := uint64(upsFrom); id < upsTo; id++ {
			if err := client.VectorUpsert(ctx, coll, id, vecOf(id, 2), "", rostam.VectorInsertOpts{}); err != nil {
				return fmt.Errorf("concurrent overwrite %d: %w", id, err)
			}
		}
		return nil
	})
	worker(func() error {
		for id := uint64(delFrom); id < delTo; id++ {
			if _, err := client.VectorDelete(ctx, coll, id); err != nil {
				return fmt.Errorf("concurrent delete %d: %w", id, err)
			}
		}
		return nil
	})

	// Online reshard 4 -> 8 OVER TCP while the writers run.
	reshardErr := client.VectorReshard(ctx, coll, newP)
	close(done)
	wg.Wait()
	if reshardErr != nil {
		t.Fatalf("client VectorReshard over TCP: %v", reshardErr)
	}
	select {
	case werr := <-writeErr:
		t.Fatalf("concurrent writer over TCP: %v", werr)
	default:
	}

	// Convergence on a NON-creator node (node 1) via the durable meta-Raft catalog.
	// The catalog gen-flip and the Stable reshard-state clear are SEPARATE meta-Raft
	// entries; await BOTH (consensus-gated, no sleeps) on the coordinator (node 0,
	// which served the routed scan oracle) AND the non-creator serving node (node 1)
	// so the oracle scan runs against fully-converged Stable state.
	waitEmbeddedCatalogGen(t, stores[1].(*rostam.Embedded), coll, newP, 1, cpuScaled(10*time.Second))
	waitEmbeddedReshardStable(t, stores[0].(*rostam.Embedded), coll, cpuScaled(10*time.Second))
	waitEmbeddedReshardStable(t, stores[1].(*rostam.Embedded), coll, cpuScaled(10*time.Second))

	// Oracle-scan the new gen via node 0's routed Call. The new gen must hold EXACTLY
	// the expected live set with EXACT per-id values: every concurrent insert survived,
	// every concurrent delete stays gone, every seeded survivor present, and every
	// overwritten survivor carries the NEW tag (Race A: a copy clobbering a newer
	// concurrent write with a stale value would be caught here over the TCP path).
	live := clusterReshardScanGen(t, stores[0].(*rostam.Embedded), coll, newP, 1)
	if len(live) != len(expected) {
		t.Fatalf("new gen has %d ids, want %d (lost write or resurrection across cluster reshard)", len(live), len(expected))
	}
	for id, wantVec := range expected {
		gotVec, ok := live[id]
		if !ok {
			t.Fatalf("id %d missing from new gen (lost write or wrongly deleted across cluster reshard)", id)
		}
		if !reflect.DeepEqual(gotVec, wantVec) {
			t.Fatalf("id %d value = %v, want %v (clobbered/stale across cluster reshard over TCP)", id, gotVec, wantVec)
		}
	}
	for id := uint64(delFrom); id < delTo; id++ {
		if _, ok := live[id]; ok {
			t.Fatalf("concurrent delete %d resurrected across cluster reshard (present in new gen)", id)
		}
	}

	// A surviving seeded doc is searchable from the non-creator node 1 over the
	// flipped catalog (proves the new gen is searchable cluster-wide).
	res, _, err := stores[1].VectorSearchExt(ctx, coll, vecOf(uint64(delTo), 1), 1, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || res[0].ID != uint64(delTo) {
		t.Fatalf("post-reshard search from node 1: %+v (want rank-0 id=%d)", res, delTo)
	}
}

// containsInt reports whether xs contains v. Used by the degraded-search tests to
// assert a dropped partition index appears in meta.Missing without depending on
// the exact set of OTHER missing partitions (there should be none, but the
// load-bearing assertion is "the partition we dropped is reported missing").
func containsInt(xs []int, v int) bool {
	for _, x := range xs {
		if x == v {
			return true
		}
	}
	return false
}

// TestClusterSearchDegraded is the end-to-end headline guarantee of the
// degraded/partial fan-out signal on a real 3-node cluster: when a physical
// partition is genuinely unavailable CLUSTER-WIDE, a Partial-mode search issued
// FROM a node that did NOT create the collection reports Degraded=true with the
// dropped partition in Missing while still returning results from the surviving
// partitions, and a Fail-mode search errors.
//
// The partition is made unavailable by dropping its physical collection via a
// routed vector_drop_collection from node 0. Physical drops route by the shard
// key and are NOT catalog-gated, so a single drop removes that physical
// collection cluster-wide — node 1's subsequent fan-out finds it gone and
// surfaces it as a missing partition. The data {float32(id),0,0,0} is tie-free
// (strictly increasing L2 distance by id) so the baseline result set is
// deterministic.
func TestClusterSearchDegraded(t *testing.T) {
	stores := newInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()

	const (
		coll = "docs"
		P    = 4
		N    = 600
	)

	// Create + populate the partitioned collection through node 0 (the creating
	// coordinator). Tie-free vectors so the baseline top-k is deterministic.
	retryUntil(t, "create docs", func() error {
		return stores[0].CreateCollection(ctx, coll, rostam.VectorConfig{
			Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: P,
		})
	})
	for id := uint64(1); id <= N; id++ {
		v := []float32{float32(id), 0, 0, 0}
		idc := id
		retryUntil(t, "insert docs", func() error {
			return stores[0].VectorInsert(ctx, coll, idc, v)
		})
	}

	// Wait for node 1's catalog to converge to P=4 before searching FROM it, so
	// the fan-out covers all partitions (otherwise it might briefly route as
	// single-partition — a flake, not a real bug).
	e1 := stores[1].(*rostam.Embedded)
	waitEmbeddedCatalog(t, e1, coll, P, 5*time.Second)

	q := []float32{1, 0, 0, 0} // nearest is unambiguously id=1 (distance 0)

	// Baseline FROM node 1 (non-creator), Partial mode: not degraded, full results.
	base, meta, err := stores[1].VectorSearchExt(ctx, coll, q, 5, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatalf("baseline VectorSearchExt from node 1: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("baseline: meta.Degraded = true, want false (all partitions reachable)")
	}
	if meta.Missing != nil {
		t.Fatalf("baseline: meta.Missing = %v, want nil", meta.Missing)
	}
	if len(base) == 0 || base[0].ID != 1 {
		t.Fatalf("baseline top = %v, want id 1 first with full results", ids(base))
	}

	// Make physical partition `drop` genuinely unavailable cluster-wide by dropping
	// its physical collection via a routed drop from node 0. Physical drops route by
	// shard key and are not catalog-gated, so the physical collection is removed
	// everywhere — node 1's fan-out then finds it gone.
	const drop = 1
	if _, err := stores[0].(*rostam.Embedded).Call(ctx, "vector_drop_collection",
		ops.EncodeDropCollectionArgs(string(ops.PartitionKeyGen(coll, 0, drop)))); err != nil {
		t.Fatalf("drop physical partition %d: %v", drop, err)
	}

	// Partial mode FROM node 1 after the drop: degraded=true, partition `drop`
	// reported missing, and results still returned from the other 3 partitions.
	res, dmeta, err := stores[1].VectorSearchExt(ctx, coll, q, 5, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatalf("degraded Partial VectorSearchExt from node 1: %v", err)
	}
	if !dmeta.Degraded {
		t.Fatalf("after drop: meta.Degraded = false, want true (partition %d unavailable cluster-wide)", drop)
	}
	if !containsInt(dmeta.Missing, drop) {
		t.Fatalf("after drop: meta.Missing = %v, want to contain dropped partition %d", dmeta.Missing, drop)
	}
	if len(res) == 0 {
		t.Fatalf("after drop: expected partial results from the surviving partitions, got none")
	}

	// Fail mode FROM node 1: the unavailable partition errors the whole query.
	if _, _, err := stores[1].VectorSearchExt(ctx, coll, q, 5,
		rostam.VectorSearchOpts{OnPartitionUnavailable: 1}); err == nil {
		t.Fatalf("Fail mode from node 1: expected error from unavailable partition, got nil")
	}
}

// TestClusterSearchDegradedTCPClient adds the MULTI-NODE dimension to the
// degraded-over-transport proof (the TestRemoteSearchDegradedTCPClient
// covers degraded-over-wire but on a single embedded node): the degraded signal
// survives a REAL TCP round trip when the decorator's embedded backend is node 0
// of a genuine 3-node cluster and the partition is dropped cluster-wide.
//
// The remote client drives create/populate over TCP into node 0's fan-out
// decorator; a physical partition is then dropped (routed, cluster-wide); and a
// Partial search over TCP returns Degraded=true with the dropped partition in
// Missing, while a Fail search over TCP errors. This proves the degraded trailer
// flows back through the wire codec on a multi-node cluster.
func TestClusterSearchDegradedTCPClient(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	stores := newInmemEmbeddedCluster(t, 3, 8)
	client := clusterTCPClient(t, stores, reg)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	const (
		coll = "dgcl"
		P    = 4
		N    = 600
	)

	// Create P=4 + populate THROUGH the remote client (over TCP into node 0's
	// decorator). Tie-free vectors so the baseline is deterministic.
	retryUntil(t, "client create", func() error {
		return client.CreateCollection(ctx, coll, rostam.VectorConfig{
			Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: P,
		})
	})
	for id := uint64(1); id <= N; id++ {
		v := []float32{float32(id), 0, 0, 0}
		idc := id
		retryUntil(t, "client insert", func() error {
			return client.VectorInsert(ctx, coll, idc, v)
		})
	}

	// Wait for a NON-creator node's catalog to converge to P=4 (proves the
	// cluster-wide catalog state the fan-out routes by).
	e1 := stores[1].(*rostam.Embedded)
	waitEmbeddedCatalog(t, e1, coll, P, 5*time.Second)

	q := []float32{1, 0, 0, 0}

	// Baseline over TCP: not degraded, full results.
	base, meta, err := client.VectorSearchExt(ctx, coll, q, 5, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatalf("baseline VectorSearchExt over TCP: %v", err)
	}
	if meta.Degraded || meta.Missing != nil {
		t.Fatalf("baseline over TCP: meta = %+v, want not degraded / nil missing", meta)
	}
	if len(base) == 0 || base[0].ID != 1 {
		t.Fatalf("baseline over TCP top = %v, want id 1 first", ids(base))
	}

	// Drop physical partition `drop` cluster-wide via a routed drop from node 0.
	const drop = 1
	if _, err := stores[0].(*rostam.Embedded).Call(ctx, "vector_drop_collection",
		ops.EncodeDropCollectionArgs(string(ops.PartitionKeyGen(coll, 0, drop)))); err != nil {
		t.Fatalf("drop physical partition %d: %v", drop, err)
	}

	// Partial mode over TCP: degraded trailer survived the wire on a multi-node
	// cluster — degraded=true, dropped partition missing, partial results.
	res, dmeta, err := client.VectorSearchExt(ctx, coll, q, 5, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatalf("degraded Partial VectorSearchExt over TCP: %v", err)
	}
	if !dmeta.Degraded {
		t.Fatalf("after drop over TCP: meta.Degraded = false, want true (degraded trailer lost over wire?)")
	}
	if !containsInt(dmeta.Missing, drop) {
		t.Fatalf("after drop over TCP: meta.Missing = %v, want to contain dropped partition %d", dmeta.Missing, drop)
	}
	if len(res) == 0 {
		t.Fatalf("after drop over TCP: expected partial results, got none")
	}

	// Fail mode over TCP: the OnPartitionUnavailable opt flows through the request
	// codec + decorator, so the unavailable partition errors the whole query.
	if _, _, err := client.VectorSearchExt(ctx, coll, q, 5,
		rostam.VectorSearchOpts{OnPartitionUnavailable: 1}); err == nil {
		t.Fatalf("Fail mode over TCP: expected error from unavailable partition, got nil")
	}
}

// TestClusterMVSearchDegraded is the multi-vector mirror of
// TestClusterSearchDegraded: when a physical MV partition is dropped cluster-wide,
// a Partial-mode MaxSim search FROM a non-creating node reports Degraded=true with
// the dropped partition in Missing while still returning results from the
// surviving partitions.
//
// This test asserts only the Partial degraded signal: it proves the Degraded=true /
// Missing reporting surfaces from a non-creator node on a real cluster, which is the
// core proof here. MV now also supports Fail mode (rostam.MultiSearchOpts.OnPartitionUnavailable
// is plumbed through embedded.mvFanOut into cluster.FanArgs.OnUnavailable), but the
// cluster-wide Fail-mode assertion across MV/hybrid/groups/scroll lives in
// TestClusterConsistencyFailMode rather than being duplicated here.
//
// Tie-free angular tokens (mvTokenAt) make the baseline winner unambiguous. The MV
// physical partition is dropped via a routed vector_mv_drop_collection keyed by the
// physical gen-0 partition name (EncodeMVDeleteArgs(name, 0)), exactly as the MV
// cleanup tests drop physical MV partitions.
func TestClusterMVSearchDegraded(t *testing.T) {
	stores := newInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()

	const (
		name = "mvdg"
		P    = 4
		N    = 120
	)

	// Create P=4 MV collection + populate tie-free docs through node 0.
	retryUntil(t, "mv create", func() error {
		return stores[0].VectorMVCreateCollection(ctx, name, rostam.MultiVectorConfig{Dim: 4, Partitions: P})
	})
	for id := 1; id <= N; id++ {
		idc := uint64(id)
		md := rostam.VectorMetadata{"docid": vector.NewInt(int64(id))}
		retryUntil(t, "mv add", func() error {
			return stores[0].VectorMVAdd(ctx, name, idc, [][]float32{mvTokenAt(id)}, md)
		})
	}

	// Wait for node 1's catalog to converge to P=4 before searching FROM it.
	e1 := stores[1].(*rostam.Embedded)
	waitEmbeddedCatalog(t, e1, name, P, 5*time.Second)

	// Choose the winner and the partition to drop so the winner's doc does NOT live
	// in the dropped partition — then the baseline winner survives the drop and the
	// surviving-results assertion (winner still rank 0) is meaningful. The winner's
	// own token deterministically returns it at rank 0 pre-drop (tie-free tokens).
	const winner = 17
	drop := (ops.PartitionOf(winner, P) + 1) % P // any partition != winner's
	q := [][]float32{mvTokenAt(winner)}
	opts := rostam.MultiSearchOpts{CandidatesPerToken: 100}

	// Baseline FROM node 1 (non-creator), Partial mode: not degraded, winner rank 0.
	base, meta, err := stores[1].VectorMVSearch(ctx, name, q, 5, opts)
	if err != nil {
		t.Fatalf("baseline VectorMVSearch from node 1: %v", err)
	}
	if meta.Degraded || meta.Missing != nil {
		t.Fatalf("baseline MV: meta = %+v, want not degraded / nil missing", meta)
	}
	if len(base) == 0 || base[0].ID != winner {
		t.Fatalf("baseline MV top = %v, want rank-0 id=%d", res2ids(base), winner)
	}

	// Drop a physical MV partition (one the winner does NOT live in) cluster-wide via
	// a routed mv-drop from node 0.
	if _, err := stores[0].(*rostam.Embedded).Call(ctx, "vector_mv_drop_collection",
		ops.EncodeMVDeleteArgs(string(ops.PartitionKeyGen(name, 0, drop)), 0)); err != nil {
		t.Fatalf("drop physical MV partition %d: %v", drop, err)
	}

	// Partial mode FROM node 1 after the drop: degraded=true, dropped partition
	// missing, results still returned from the surviving partitions.
	got, dmeta, err := stores[1].VectorMVSearch(ctx, name, q, 5, opts)
	if err != nil {
		t.Fatalf("degraded Partial VectorMVSearch from node 1: %v", err)
	}
	if !dmeta.Degraded {
		t.Fatalf("after MV drop: meta.Degraded = false, want true (partition %d unavailable cluster-wide)", drop)
	}
	if !containsInt(dmeta.Missing, drop) {
		t.Fatalf("after MV drop: meta.Missing = %v, want to contain dropped partition %d", dmeta.Missing, drop)
	}
	if len(got) == 0 {
		t.Fatalf("after MV drop: expected partial results from surviving partitions, got none")
	}
}

// res2ids extracts rostam.MultiResult IDs in order (for failure messages).
func res2ids(rs []rostam.MultiResult) []uint64 {
	out := make([]uint64, len(rs))
	for i, r := range rs {
		out[i] = r.ID
	}
	return out
}

// TestClusterConsistencyFailMode is the end-to-end headline guarantee of the
// consistency opts wired over the wire (Plan: consistency-opts-over-wire): on a
// real 3-node cluster, with the Go client speaking the binary-TCP transport into
// node 0's fan-out decorator, the OnPartitionUnavailable=Fail knob crosses the
// wire and is honored CLUSTER-WIDE for EVERY fan-out read — including MV's new
// Fail mode — when a physical partition is genuinely unavailable. Partial mode
// (OnPartitionUnavailable=0) instead reports rostam.FanMeta.Degraded=true and returns
// degraded results.
//
// Transport coverage: search/docs/groups/hybrid/scroll are all driven over the
// Go client through real TCP (their typed methods take rostam.VectorSearchOpts /
// rostam.VectorGroupOpts / rostam.VectorHybridOpts / rostam.VectorScrollOpts, all carrying the
// consistency fields). MV (VectorMVSearch via rostam.MultiSearchOpts) is ALSO driven
// over the same TCP client. So all six ops — including MV's new Fail mode —
// prove the Fail knob over a real transport on a genuine multi-node cluster.
//
// The drop technique mirrors the degraded tests: a physical partition is removed
// cluster-wide via a routed vector_drop_collection (dense) / vector_mv_drop_collection
// (MV) from node 0; physical drops route by shard key and are not catalog-gated,
// so the partition is gone everywhere and node 0's decorator fan-out surfaces it
// as missing. Data is tie-free so baselines are deterministic.
//
// A LeaderOnly (ReadConsistency=1) Partial smoke check confirms the consistency
// field crossing the wire does not break the query (it is NOT a routing assertion
// — LeaderOnly replica selection is not behaviorally observable on this RF=1
// harness; this just proves the knob is accepted and still returns correct results).
func TestClusterConsistencyFailMode(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	stores := newInmemEmbeddedCluster(t, 3, 8)
	client := clusterTCPClient(t, stores, reg)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	const (
		dcoll  = "cfm-dense"
		mvcoll = "cfm-mv"
		P      = 4
		N      = 120
	)

	// --- Create + populate the dense collection THROUGH the client over TCP. ---
	// Each doc carries a sparse lane (so hybrid genuinely fuses) and a "doc" group
	// key (so groups genuinely collapse). Tie-free dense vectors keep baselines
	// deterministic.
	retryUntil(t, "client create dense", func() error {
		return client.CreateCollection(ctx, dcoll, rostam.VectorConfig{
			Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: P,
		})
	})
	for id := uint64(1); id <= N; id++ {
		v := []float32{float32(id), 0, 0, 0}
		sp := rostam.VectorSparse{Indices: []uint32{uint32(id % 7)}, Values: []float32{1}}
		md := rostam.VectorMetadata{"doc": vector.NewInt(int64(id % 20))}
		idc := id
		retryUntil(t, "client insert dense", func() error {
			return client.VectorInsertExt(ctx, dcoll, idc, v, rostam.VectorInsertOpts{Sparse: sp, Metadata: md})
		})
	}

	// --- Create + populate the MV collection THROUGH the client over TCP. ---
	retryUntil(t, "client mv create", func() error {
		return client.VectorMVCreateCollection(ctx, mvcoll, rostam.MultiVectorConfig{Dim: 4, Partitions: P})
	})
	for id := 1; id <= N; id++ {
		idc := uint64(id)
		md := rostam.VectorMetadata{"docid": vector.NewInt(int64(id))}
		retryUntil(t, "client mv add", func() error {
			return client.VectorMVAdd(ctx, mvcoll, idc, [][]float32{mvTokenAt(id)}, md)
		})
	}

	// Wait for a NON-creator node's catalog to converge to P=4 for both — proves
	// the cluster-wide catalog state the fan-out routes by has propagated.
	e1 := stores[1].(*rostam.Embedded)
	waitEmbeddedCatalog(t, e1, dcoll, P, 5*time.Second)
	waitEmbeddedCatalog(t, e1, mvcoll, P, 5*time.Second)

	query := []float32{0.5, 0, 0, 0}                                      // tie-free dense
	qs := rostam.VectorSparse{Indices: []uint32{3}, Values: []float32{1}} // non-vacuous sparse lane

	// MV winner / drop partition chosen so the winner does NOT live in the dropped
	// partition — the baseline winner survives the drop and partial results stay
	// non-empty. The winner's own token returns it deterministically (tie-free).
	const mvWinner = 17
	mvQuery := [][]float32{mvTokenAt(mvWinner)}
	mvDrop := (ops.PartitionOf(mvWinner, P) + 1) % P
	mvOpts := func(onUnavail uint8) rostam.MultiSearchOpts {
		return rostam.MultiSearchOpts{CandidatesPerToken: 100, OnPartitionUnavailable: onUnavail}
	}

	// ---- Baseline (no dropped partition), Partial: not degraded, full-ish ------
	// results, over the real transport for every op.
	if res, meta, err := client.VectorSearchExt(ctx, dcoll, query, 10, rostam.VectorSearchOpts{}); err != nil {
		t.Fatalf("baseline search over TCP: %v", err)
	} else if meta.Degraded || meta.Missing != nil {
		t.Fatalf("baseline search: meta = %+v, want not degraded", meta)
	} else if len(res) == 0 {
		t.Fatalf("baseline search: empty results")
	}
	if res, meta, err := client.VectorSearchDocs(ctx, dcoll, query, 10, rostam.VectorSearchOpts{}); err != nil {
		t.Fatalf("baseline docs over TCP: %v", err)
	} else if meta.Degraded || meta.Missing != nil {
		t.Fatalf("baseline docs: meta = %+v, want not degraded", meta)
	} else if len(res) == 0 {
		t.Fatalf("baseline docs: empty results")
	}
	if res, meta, err := client.VectorSearchGroups(ctx, dcoll, query, 5,
		rostam.VectorGroupOpts{GroupBy: "doc", GroupSize: 3}); err != nil {
		t.Fatalf("baseline groups over TCP: %v", err)
	} else if meta.Degraded || meta.Missing != nil {
		t.Fatalf("baseline groups: meta = %+v, want not degraded", meta)
	} else if len(res) == 0 {
		t.Fatalf("baseline groups: empty results")
	}
	if res, meta, err := client.VectorHybridSearch(ctx, dcoll, query, 10,
		rostam.VectorHybridOpts{Sparse: qs}); err != nil {
		t.Fatalf("baseline hybrid over TCP: %v", err)
	} else if meta.Degraded || meta.Missing != nil {
		t.Fatalf("baseline hybrid: meta = %+v, want not degraded", meta)
	} else if len(res) == 0 {
		t.Fatalf("baseline hybrid: empty results")
	}
	if res, meta, _, err := client.VectorScroll(ctx, dcoll, rostam.VectorFilter{}, 0, rostam.VectorScrollOpts{}); err != nil {
		t.Fatalf("baseline scroll over TCP: %v", err)
	} else if meta.Degraded || meta.Missing != nil {
		t.Fatalf("baseline scroll: meta = %+v, want not degraded", meta)
	} else if len(res) != N {
		t.Fatalf("baseline scroll: %d docs, want %d", len(res), N)
	}
	if res, meta, err := client.VectorMVSearch(ctx, mvcoll, mvQuery, 10, mvOpts(0)); err != nil {
		t.Fatalf("baseline mv over TCP: %v", err)
	} else if meta.Degraded || meta.Missing != nil {
		t.Fatalf("baseline mv: meta = %+v, want not degraded", meta)
	} else if len(res) == 0 || res[0].ID != mvWinner {
		t.Fatalf("baseline mv top = %v, want rank-0 id=%d", res2ids(res), mvWinner)
	}

	// LeaderOnly (ReadConsistency=1) Partial smoke: the consistency knob crosses
	// the wire and the query still returns correct results. Not a routing
	// assertion — RF=1 makes LeaderOnly replica selection unobservable here.
	if res, _, err := client.VectorSearchExt(ctx, dcoll, query, 10,
		rostam.VectorSearchOpts{ReadConsistency: 1}); err != nil {
		t.Fatalf("LeaderOnly search over TCP: %v", err)
	} else if len(res) == 0 {
		t.Fatalf("LeaderOnly search: empty results (knob broke the query?)")
	}

	// ---- Drop ONE physical partition cluster-wide for each collection. --------
	// Dense partition 1 via routed vector_drop_collection on the gen-0 phys name.
	const denseDrop = 1
	if _, err := stores[0].(*rostam.Embedded).Call(ctx, "vector_drop_collection",
		ops.EncodeDropCollectionArgs(string(ops.PartitionKeyGen(dcoll, 0, denseDrop)))); err != nil {
		t.Fatalf("drop dense physical partition %d: %v", denseDrop, err)
	}
	// MV partition mvDrop via routed vector_mv_drop_collection on the gen-0 phys name.
	if _, err := stores[0].(*rostam.Embedded).Call(ctx, "vector_mv_drop_collection",
		ops.EncodeMVDeleteArgs(string(ops.PartitionKeyGen(mvcoll, 0, mvDrop)), 0)); err != nil {
		t.Fatalf("drop mv physical partition %d: %v", mvDrop, err)
	}

	// ---- For EACH op over the transport: Fail -> error, Partial -> degraded. ---

	// search
	if _, _, err := client.VectorSearchExt(ctx, dcoll, query, 10,
		rostam.VectorSearchOpts{OnPartitionUnavailable: 1}); err == nil {
		t.Fatalf("search Fail over TCP: expected error from unavailable partition, got nil")
	}
	if res, meta, err := client.VectorSearchExt(ctx, dcoll, query, 10, rostam.VectorSearchOpts{}); err != nil {
		t.Fatalf("search Partial over TCP: unexpected err: %v", err)
	} else if !meta.Degraded {
		t.Fatalf("search Partial over TCP: meta.Degraded = false, want true")
	} else if !containsInt(meta.Missing, denseDrop) {
		t.Fatalf("search Partial over TCP: meta.Missing = %v, want to contain %d", meta.Missing, denseDrop)
	} else if len(res) == 0 {
		t.Fatalf("search Partial over TCP: expected partial results, got none")
	}

	// docs
	if _, _, err := client.VectorSearchDocs(ctx, dcoll, query, 10,
		rostam.VectorSearchOpts{OnPartitionUnavailable: 1}); err == nil {
		t.Fatalf("docs Fail over TCP: expected error from unavailable partition, got nil")
	}
	if res, meta, err := client.VectorSearchDocs(ctx, dcoll, query, 10, rostam.VectorSearchOpts{}); err != nil {
		t.Fatalf("docs Partial over TCP: unexpected err: %v", err)
	} else if !meta.Degraded {
		t.Fatalf("docs Partial over TCP: meta.Degraded = false, want true")
	} else if !containsInt(meta.Missing, denseDrop) {
		t.Fatalf("docs Partial over TCP: meta.Missing = %v, want to contain %d", meta.Missing, denseDrop)
	} else if len(res) == 0 {
		t.Fatalf("docs Partial over TCP: expected partial results, got none")
	}

	// groups
	if _, _, err := client.VectorSearchGroups(ctx, dcoll, query, 5,
		rostam.VectorGroupOpts{GroupBy: "doc", GroupSize: 3, OnPartitionUnavailable: 1}); err == nil {
		t.Fatalf("groups Fail over TCP: expected error from unavailable partition, got nil")
	}
	if res, meta, err := client.VectorSearchGroups(ctx, dcoll, query, 5,
		rostam.VectorGroupOpts{GroupBy: "doc", GroupSize: 3}); err != nil {
		t.Fatalf("groups Partial over TCP: unexpected err: %v", err)
	} else if !meta.Degraded {
		t.Fatalf("groups Partial over TCP: meta.Degraded = false, want true")
	} else if !containsInt(meta.Missing, denseDrop) {
		t.Fatalf("groups Partial over TCP: meta.Missing = %v, want to contain %d", meta.Missing, denseDrop)
	} else if len(res) == 0 {
		t.Fatalf("groups Partial over TCP: expected partial results, got none")
	}

	// hybrid
	if _, _, err := client.VectorHybridSearch(ctx, dcoll, query, 10,
		rostam.VectorHybridOpts{Sparse: qs, OnPartitionUnavailable: 1}); err == nil {
		t.Fatalf("hybrid Fail over TCP: expected error from unavailable partition, got nil")
	}
	if res, meta, err := client.VectorHybridSearch(ctx, dcoll, query, 10,
		rostam.VectorHybridOpts{Sparse: qs}); err != nil {
		t.Fatalf("hybrid Partial over TCP: unexpected err: %v", err)
	} else if !meta.Degraded {
		t.Fatalf("hybrid Partial over TCP: meta.Degraded = false, want true")
	} else if !containsInt(meta.Missing, denseDrop) {
		t.Fatalf("hybrid Partial over TCP: meta.Missing = %v, want to contain %d", meta.Missing, denseDrop)
	} else if len(res) == 0 {
		t.Fatalf("hybrid Partial over TCP: expected partial results, got none")
	}

	// scroll
	if _, _, _, err := client.VectorScroll(ctx, dcoll, rostam.VectorFilter{}, 0,
		rostam.VectorScrollOpts{OnPartitionUnavailable: 1}); err == nil {
		t.Fatalf("scroll Fail over TCP: expected error from unavailable partition, got nil")
	}
	if res, meta, _, err := client.VectorScroll(ctx, dcoll, rostam.VectorFilter{}, 0, rostam.VectorScrollOpts{}); err != nil {
		t.Fatalf("scroll Partial over TCP: unexpected err: %v", err)
	} else if !meta.Degraded {
		t.Fatalf("scroll Partial over TCP: meta.Degraded = false, want true")
	} else if !containsInt(meta.Missing, denseDrop) {
		t.Fatalf("scroll Partial over TCP: meta.Missing = %v, want to contain %d", meta.Missing, denseDrop)
	} else if len(res) == 0 {
		t.Fatalf("scroll Partial over TCP: expected partial results, got none")
	}

	// mv (the new Fail mode), driven over the same TCP transport.
	if _, _, err := client.VectorMVSearch(ctx, mvcoll, mvQuery, 10, mvOpts(1)); err == nil {
		t.Fatalf("mv Fail over TCP: expected error from unavailable partition, got nil (MV must honor Fail mode over the wire)")
	}
	if res, meta, err := client.VectorMVSearch(ctx, mvcoll, mvQuery, 10, mvOpts(0)); err != nil {
		t.Fatalf("mv Partial over TCP: unexpected err: %v", err)
	} else if !meta.Degraded {
		t.Fatalf("mv Partial over TCP: meta.Degraded = false, want true")
	} else if !containsInt(meta.Missing, mvDrop) {
		t.Fatalf("mv Partial over TCP: meta.Missing = %v, want to contain %d", meta.Missing, mvDrop)
	} else if len(res) == 0 {
		t.Fatalf("mv Partial over TCP: expected partial results, got none")
	}
}

// serveRecorder captures every OpReadOnly serve reported via
// shard.SetReadServedHook, recording whether the serving replica was the shard
// leader. Mutex-guarded so it is safe under the concurrent fan-out serves the
// dense scatter issues (one goroutine per partition). Mirrors the cluster-pkg
// readRecorder used by TestCallPhysicalLeaderOnlyRouting.
type serveRecorder struct {
	mu     sync.Mutex
	serves []bool // each entry: isLeader of the replica that served
}

func (r *serveRecorder) hook(isLeader bool) {
	r.mu.Lock()
	r.serves = append(r.serves, isLeader)
	r.mu.Unlock()
}

func (r *serveRecorder) reset() {
	r.mu.Lock()
	r.serves = nil
	r.mu.Unlock()
}

func (r *serveRecorder) snapshot() []bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]bool(nil), r.serves...)
}

// followerServes / leaderServes count serves by a follower (isLeader=false) and
// by a leader (isLeader=true) respectively, over the current snapshot.
func (r *serveRecorder) followerServes() int {
	n := 0
	for _, isLeader := range r.snapshot() {
		if !isLeader {
			n++
		}
	}
	return n
}

func (r *serveRecorder) leaderServes() int {
	n := 0
	for _, isLeader := range r.snapshot() {
		if isLeader {
			n++
		}
	}
	return n
}

// TestClusterLeaderOnlyServedByLeader is the end-to-end proof of the LeaderOnly
// read-consistency knob on a replicated (RF=2) cluster, exercised through the
// embedded rostam.Store's typed VectorSearchExt fan-out path.
//
// On a 3-node RF=2 cluster every physical partition shard has a leader plus one
// follower on a distinct node. We drive the READ from a COORDINATOR that hosts a
// FOLLOWER replica of at least one partition's shard — some partition p whose
// physical route key PartitionKeyGen(coll,0,p) maps to a shard the coordinator
// hosts but does NOT lead. Then:
//
//   - VectorSearchExt with rostam.VectorSearchOpts{} (AnyReplica) serves that partition
//     from the local follower, so the serve hook records at least one
//     isLeader=false serve — proving AnyReplica genuinely reads followers.
//   - VectorSearchExt with rostam.VectorSearchOpts{ReadConsistency: 1} (LeaderOnly)
//     routes every partition to its leader, so the hook records ZERO follower
//     serves — the behavioral difference that depends on the Task-1 routing fix.
//     Pre-fix, LeaderOnly would still serve the partition from the local
//     follower (an isLeader=false serve); this test is the regression guard.
//
// Both calls also assert the results are correct (non-empty, expected nearest),
// so the routing difference is proven WITHOUT sacrificing correctness.
//
// WRITER vs READER split (forced by an embedded-harness limitation, NOT a
// production change): at RF>1 the embedded write path returns NotLeader — and
// does NOT forward — for a write to a shard the driving node hosts as a FOLLOWER
// (the same "no leader-following" limitation the RF=1 harness comment calls out).
// So a partitioned CreateCollection / VectorInsert only succeeds from a node that,
// for every one of the collection's partition shards, either LEADS it or does not
// host it. Such a writer cannot simultaneously host a follower of one of those
// shards — yet that follower-hosting node is exactly the reader we need. We
// therefore split the roles: write through a node that can create+insert all
// partitions, then read from a DIFFERENT node that follows one of them. The
// partition catalog is durable in meta-Raft, so the reader learns P. (Reads
// always serve locally from any hosted replica, so the read path has no such
// limitation.)
func TestClusterLeaderOnlyServedByLeader(t *testing.T) {
	const (
		numShards = 8
		n         = 3
		rf        = 2
		P         = n // P=N partitions
		N         = 80
		k         = 5
	)
	stores := newInmemEmbeddedCluster(t, n, numShards, rf)
	ctx := context.Background()

	// Deterministically find a (writer, coll) such that the writer can create the
	// partitioned collection (it leads-or-does-not-host every partition shard) AND
	// some OTHER node hosts a follower of at least one of that collection's
	// partition shards (the reader). We iterate disposable, unique collection
	// names: a create that fails part-way (a partition shard the writer hosts as a
	// follower → NotLeader) is abandoned under its unique name, never retried, so
	// it cannot poison a later attempt. Bounded; fails loud if no pair is found.
	query := []float32{0.5, 0, 0, 0} // tie-free: 0.5 < 1 => nearest is id=1
	var (
		coll         string
		writerIdx    = -1
		readerIdx    = -1
		followerPart = -1
	)
	createCfg := rostam.VectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: P}
	deadline := time.Now().Add(60 * time.Second)
search:
	for attempt := 0; attempt < 64 && time.Now().Before(deadline); attempt++ {
		for w := 0; w < n; w++ {
			cand := fmt.Sprintf("leaderonly-w%d-a%d", w, attempt)
			// Try to create the whole partitioned collection from writer w. Any
			// error (NotLeader on a follower-hosted partition shard, or a transient
			// election) abandons this candidate name.
			if err := stores[w].CreateCollection(ctx, cand, createCfg); err != nil {
				continue
			}
			// Created. Now require a reader r != w that:
			//   - HOSTS every one of cand's P partition shards with a KNOWN leader
			//     (IsLeader OR LeaderAddr != ""), so the LeaderOnly fan-out never
			//     takes the fragile non-hosted/unknown-leader forward path (whose
			//     local leader-view can be stale or empty under the RF>1 embedded
			//     harness), AND
			//   - FOLLOWS at least one of them (IsLeader false) — the partition
			//     whose follower serve AnyReplica must observe and LeaderOnly must
			//     avoid.
			// Requiring the reader to host all partitions makes the per-partition
			// leadership-consensus gate below satisfiable (a non-hosted partition's
			// reader-view never resolves, so it could never agree with consensus).
			for r := 0; r < n; r++ {
				if r == w {
					continue
				}
				hostsAll, follows, fp := true, false, -1
				for p := 0; p < P; p++ {
					key := []byte(ops.CanonicalName(string(ops.PartitionKeyGen(cand, 0, p))))
					led := stores[r].IsLeader(key)
					known := stores[r].LeaderAddr(key) != ""
					if !led && !known {
						hostsAll = false // non-hosted (or leader not yet known)
						break
					}
					if !led {
						follows = true
						if fp < 0 {
							fp = p
						}
					}
				}
				if hostsAll && follows {
					coll, writerIdx, readerIdx, followerPart = cand, w, r, fp
					break search
				}
			}
			// Created but no all-hosting, following reader for this name; abandon it
			// and move on (a fresh name reshuffles partition→shard placement).
		}
		time.Sleep(50 * time.Millisecond)
	}
	if writerIdx < 0 {
		t.Fatalf("could not find a (writer, reader) pair: a node that creates a P=%d collection AND another node that follows one of its partition shards (RF=%d). The embedded write path's hosted-follower NotLeader gap may make some placements uncreatable; iterate numShards/n if this recurs.", P, rf)
	}
	t.Logf("writer=node%d created %q; reader=node%d hosts ALL %d partition shards (with known leaders) and FOLLOWS partition %d (IsLeader=false) — the partition whose follower-serve AnyReplica must show and LeaderOnly must avoid", writerIdx, coll, readerIdx, P, followerPart)

	// Insert N tie-free vectors through the writer. v(id)={id,0,0,0}: L2 distance
	// from {q,0,0,0} is strictly monotonic in |id-q|, so for q=0.5 (below the
	// smallest id) the nearest is unambiguously id=1 (no ties at position 0).
	// Each insert routes to a partition shard the writer just created, so it leads
	// or forwards to the leader — no hosted-follower write here.
	for id := uint64(1); id <= N; id++ {
		v := []float32{float32(id), 0, 0, 0}
		retryUntil(t, fmt.Sprintf("insert %s %d", coll, id), func() error {
			return stores[writerIdx].VectorInsert(ctx, coll, id, v)
		})
	}

	// Readiness: the reader's local catalog must converge to P, AND every
	// partition's replica the reader hosts must have caught up so a follower-served
	// read returns correct data. We gate replication-applied state by polling an
	// AnyReplica fan-out from the reader until it returns the correct nearest
	// (id=1) — an AnyReplica read serves each partition from a LOCAL replica
	// (possibly a follower) when the reader hosts the shard, so a clean pass means
	// the reader's replicas (follower included) have applied the inserts. No fixed
	// sleeps.
	er := stores[readerIdx].(*rostam.Embedded)
	waitEmbeddedCatalog(t, er, coll, P, 10*time.Second)
	convDeadline := time.Now().Add(20 * time.Second)
	for {
		res, _, err := stores[readerIdx].VectorSearchExt(ctx, coll, query, k, rostam.VectorSearchOpts{})
		if err == nil && len(res) == k && res[0].ID == 1 {
			break
		}
		if time.Now().After(convDeadline) {
			t.Fatalf("reader node%d: replication did not converge (last err=%v)", readerIdx, err)
		}
		time.Sleep(20 * time.Millisecond)
	}
	// Leadership-consensus gate. The serve assertions below are EXACT (LeaderOnly
	// must see ZERO follower serves), and the LeaderOnly fan-out resolves each
	// partition's leader from the reader's locally-tracked view, then forwards the
	// read there. OpReadOnly serves on ANY replica without a leadership check, so
	// if the reader's tracked leader is STALE (points to a node that has since
	// become a follower) the forward lands on a follower that happily serves —
	// recording a spurious isLeader=false under LeaderOnly. Eliminate that window
	// deterministically by waiting until, for EVERY partition shard:
	//   (1) exactly one store reports IsLeader==true (global single-leader
	//       consensus — no pending election / split view), AND
	//   (2) the reader's tracked leader address for the shard equals that unique
	//       leader's own server address (the reader's view AGREES with reality, so
	//       its forward targets the real leader),
	// held stable across a window of consecutive polls. No fixed sleep is the
	// gating mechanism — we poll until consensus + agreement are stable.
	partKeys := make([][]byte, P)
	for p := 0; p < P; p++ {
		partKeys[p] = []byte(ops.CanonicalName(string(ops.PartitionKeyGen(coll, 0, p))))
	}
	// uniqueLeaderAddr returns the server address of the sole store that leads
	// key, or "" if zero or more than one store currently claims leadership (a
	// pending/contested election). The leader's own LeaderAddr(key) returns its
	// server address, so it doubles as the address the reader must forward to.
	uniqueLeaderAddr := func(key []byte) string {
		addr, count := "", 0
		for i := range stores {
			if stores[i].IsLeader(key) {
				count++
				addr = stores[i].LeaderAddr(key)
			}
		}
		if count != 1 {
			return ""
		}
		return addr
	}
	// consensusReady reports whether every partition shard has a unique leader on
	// which the reader's tracked view agrees.
	consensusReady := func() bool {
		for p := 0; p < P; p++ {
			la := uniqueLeaderAddr(partKeys[p])
			if la == "" || stores[readerIdx].LeaderAddr(partKeys[p]) != la {
				return false
			}
		}
		return true
	}
	consDeadline := time.Now().Add(20 * time.Second)
	stableRounds := 0
	for {
		if consensusReady() {
			stableRounds++
		} else {
			stableRounds = 0
		}
		if stableRounds >= 10 { // ~0.5s of stable, agreed, single-leader consensus
			break
		}
		if time.Now().After(consDeadline) {
			// Report the contested placement to aid diagnosis.
			var b []byte
			for p := 0; p < P; p++ {
				b = append(b, []byte(fmt.Sprintf("p%d uniqueLeader=%q readerView=%q; ", p, uniqueLeaderAddr(partKeys[p]), stores[readerIdx].LeaderAddr(partKeys[p])))...)
			}
			t.Fatalf("reader node%d: partition-shard leadership did not reach stable consensus the reader agrees with: %s", readerIdx, string(b))
		}
		time.Sleep(50 * time.Millisecond)
	}
	// Re-confirm the reader still follows followerPart's shard after consensus.
	fkey := partKeys[followerPart]
	if stores[readerIdx].IsLeader(fkey) || stores[readerIdx].LeaderAddr(fkey) == "" {
		t.Fatalf("reader node%d no longer follows partition %d's shard (IsLeader=%v leader=%q)", readerIdx, followerPart, stores[readerIdx].IsLeader(fkey), stores[readerIdx].LeaderAddr(fkey))
	}
	coordinator := stores[readerIdx]

	rec := &serveRecorder{}
	shard.SetReadServedHook(rec.hook)
	defer shard.SetReadServedHook(nil)

	// AnyReplica: at least one partition (followerPart) is served by the
	// coordinator's local follower, so the recorder must see >=1 follower serve.
	rec.reset()
	res, meta, err := coordinator.VectorSearchExt(ctx, coll, query, k, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatalf("AnyReplica VectorSearchExt: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("AnyReplica: unexpected degraded fan-out, meta=%+v", meta)
	}
	if len(res) != k || res[0].ID != 1 {
		t.Fatalf("AnyReplica results = %v, want %d nearest with id 1 first", ids(res), k)
	}
	if fs := rec.followerServes(); fs == 0 {
		t.Fatalf("AnyReplica: expected at least one follower serve (coordinator hosts a follower of partition %d), serves=%v", followerPart, rec.snapshot())
	}

	// LeaderOnly (ReadConsistency=1): EVERY partition must be served by its
	// leader — zero follower serves. This is the behavioral difference that
	// relies on the Task-1 routing fix; pre-fix the coordinator's follower of
	// partition followerPart would still serve locally (an isLeader=false serve).
	rec.reset()
	res2, meta2, err := coordinator.VectorSearchExt(ctx, coll, query, k, rostam.VectorSearchOpts{ReadConsistency: 1})
	if err != nil {
		t.Fatalf("LeaderOnly VectorSearchExt: %v", err)
	}
	if meta2.Degraded {
		t.Fatalf("LeaderOnly: unexpected degraded fan-out, meta=%+v", meta2)
	}
	if len(res2) != k || res2[0].ID != 1 {
		t.Fatalf("LeaderOnly results = %v, want %d nearest with id 1 first", ids(res2), k)
	}
	if !sameIDs(res, res2) {
		t.Fatalf("LeaderOnly top-k %v != AnyReplica top-k %v (consistency must not change correctness)", ids(res2), ids(res))
	}
	if fs := rec.followerServes(); fs != 0 {
		t.Fatalf("LeaderOnly: %d follower serve(s) observed — LeaderOnly routing ignored (regression), serves=%v", fs, rec.snapshot())
	}
	// Sanity: LeaderOnly still served every partition (all by leaders).
	if ls := rec.leaderServes(); ls < P {
		t.Fatalf("LeaderOnly: %d leader serves, want >= %d (one per partition)", ls, P)
	}
}

// TestClusterRichFilterFanOut is the cluster-scale guarantee of rich
// payload filtering): the rich operators (match / regex / is_empty /
// datetime-range / dotted-key) fan out correctly across a partitioned (P>1)
// collection spread over a real 3-node cluster, driven FROM a node that did NOT
// create the collection — through SEARCH and DELETE-BY-FILTER.
//
// It combines the durable meta-Raft partition catalog (the non-creating
// coordinator learns P from committed cluster state via waitEmbeddedCatalog) with
// the rich-filter fan-out: routing is by id hash, so each predicate's matching
// docs are spread across ALL partitions AND nodes, and a correct fan-out must
// scatter to every physical partition. The asserted id sets are computed
// INDEPENDENTLY in-test (richGroundTruth — the same oracle the TCP test uses), so
// a dropped/misrouted partition surfaces as a missing/extra id. Readiness is
// consensus-gated (waitEmbeddedCatalog), never sleep-based.
func TestClusterRichFilterFanOut(t *testing.T) {
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()

	const coll = "rich"
	const n = 120 // > P so every partition holds several docs and predicates spread

	// Create + populate through node 0 (the creating coordinator).
	retryUntil(t, "create rich", func() error {
		return stores[0].CreateCollection(ctx, coll, rostam.VectorConfig{
			Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Partitions: 6,
		})
	})
	// Ids 1..n (id 0 skipped — the search path has a permanent `if id == 0` sentinel
	// guard, filter-independent; see richGroundTruth. SCROLL/DELETE prove all ids).
	for i := uint64(1); i <= n; i++ {
		md := richMetadata(richDocFor(i))
		idc := i
		retryUntil(t, fmt.Sprintf("insert rich %d", idc), func() error {
			return stores[0].VectorInsertExt(ctx, coll, idc, tieFreeVec(int(idc)), rostam.VectorInsertOpts{Metadata: md})
		})
	}

	matchRed, regexBerry, isEmptyTags, dtMid, cityNY := richGroundTruth(n)
	for name, set := range map[string]map[uint64]bool{
		"matchRed": matchRed, "regexBerry": regexBerry, "isEmptyTags": isEmptyTags,
		"dtMid": dtMid, "cityNY": cityNY,
	} {
		if len(set) == 0 || uint64(len(set)) == n {
			t.Fatalf("ground-truth set %q has %d/%d members (vacuous filter)", name, len(set), n)
		}
	}

	// Drive the rich-filter SEARCH from node 1 — a node that did NOT create the
	// collection — after its local catalog has converged to P=6. Without this wait
	// node 1 might briefly route as if the collection were single-partition and
	// search the empty logical collection (a flake, not a real failure).
	e1 := stores[1].(*rostam.Embedded)
	waitEmbeddedCatalog(t, e1, coll, 6, 5*time.Second)

	loMs := richBaseMs + 10*richDayMs
	hiMs := richBaseMs + 20*richDayMs
	dtRange := rostam.VectorFilter{Op: vector.FilterAnd, And: []rostam.VectorFilter{
		{Op: vector.FilterDtGte, Field: "ts", Value: vector.NewString(rfc3339(loMs))},
		{Op: vector.FilterDtLt, Field: "ts", Value: vector.NewString(rfc3339(hiMs))},
	}}
	cases := []struct {
		name   string
		filter rostam.VectorFilter
		want   map[uint64]bool
	}{
		{"match", rostam.VectorFilter{Op: vector.FilterMatch, Field: "color", Value: vector.NewString("red")}, matchRed},
		{"regex", rostam.VectorFilter{Op: vector.FilterRegex, Field: "color", Value: vector.NewString("berry$")}, regexBerry},
		{"is_empty", rostam.VectorFilter{Op: vector.FilterIsEmpty, Field: "tags"}, isEmptyTags},
		{"dt_range", dtRange, dtMid},
		{"dotted_eq", rostam.VectorFilter{Op: vector.FilterEq, Field: "address.city", Value: vector.NewString("New York")}, cityNY},
	}

	// SEARCH fan-out FROM node 1: k=n so a correct cluster-wide union returns
	// EXACTLY the matching set across all 6 partitions and 3 nodes.
	for _, c := range cases {
		res, _, err := stores[1].VectorSearchExt(ctx, coll, tieFreeQuery(), n, rostam.VectorSearchOpts{Filter: c.filter})
		if err != nil {
			t.Fatalf("cluster search %s via node 1: %v", c.name, err)
		}
		got := resultIDSet(res)
		if ok, missing, extra := sameUint64Set(got, c.want); !ok {
			t.Fatalf("cluster SEARCH %s fan-out id set wrong: missing=%v extra=%v (want %d ids; a dropped partition/node shows here)",
				c.name, missing, extra, len(c.want))
		}
	}

	// DELETE-BY-FILTER fan-out FROM node 1 (non-creator): delete the datetime-range
	// set; assert exactly those docs across ALL partitions are gone and every
	// non-matching doc survives. Verify survivors FROM node 2 (a third coordinator)
	// — proving the delete reached the physical partitions cluster-wide.
	deleted, err := stores[1].VectorDeleteByFilter(ctx, coll, dtRange)
	if err != nil {
		t.Fatalf("cluster delete-by-filter dt_range via node 1: %v", err)
	}
	if deleted != len(dtMid) {
		t.Fatalf("cluster delete-by-filter dt_range deleted %d, want %d (must hit every partition/node)", deleted, len(dtMid))
	}

	wantSurvivors := map[uint64]bool{}
	for i := uint64(1); i <= n; i++ {
		if !dtMid[i] {
			wantSurvivors[i] = true
		}
	}
	e2 := stores[2].(*rostam.Embedded)
	waitEmbeddedCatalog(t, e2, coll, 6, 5*time.Second)
	all, _, _, err := stores[2].VectorScroll(ctx, coll, rostam.VectorFilter{}, 0, rostam.VectorScrollOpts{})
	if err != nil {
		t.Fatalf("post-delete full scroll via node 2: %v", err)
	}
	gotSurvivors := docIDSet(all)
	if ok, missing, extra := sameUint64Set(gotSurvivors, wantSurvivors); !ok {
		t.Fatalf("cluster post-delete survivors wrong (from node 2): missing=%v extra=%v (want %d survivors)",
			missing, extra, len(wantSurvivors))
	}
	for id := range dtMid {
		if gotSurvivors[id] {
			t.Fatalf("cluster delete-by-filter left datetime-matched id %d alive (partition/node not reached)", id)
		}
	}
}

// TestClusterGeoFilterFanOut is the geo analogue of TestClusterRichFilterFanOut:
// it proves a geo filter (geo_radius SEARCH + a geo_box DELETE-BY-FILTER) driven
// from a NON-CREATOR node fans out correctly across every partition AND every
// node of a 3-node cluster, and that ValueGeo metadata is durable through the
// persist/replication round-trip (the returned doc's lat/lon equals the seed).
// Readiness is consensus-gated via waitEmbeddedCatalog — no sleeps. The seed,
// ground truth, and partition-spread guard are shared with the TCP test above
// (geoMetadata / geoGroundTruth / partitionSpread), so the oracle is the same
// independent in-test computation.
func TestClusterGeoFilterFanOut(t *testing.T) {
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()

	const coll = "geo"
	const n = 120 // grid cells 0..119, > P so matches spread across partitions
	const P = 6

	retryUntil(t, "create geo", func() error {
		return stores[0].CreateCollection(ctx, coll, rostam.VectorConfig{
			Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Partitions: P,
		})
	})
	// Ids 1..n (id 0 skipped on the SEARCH path — permanent sentinel guard; see
	// richGroundTruth. SCROLL/DELETE are exhaustive and prove every id).
	for i := uint64(1); i <= n; i++ {
		md := geoMetadata(i)
		idc := i
		retryUntil(t, fmt.Sprintf("insert geo %d", idc), func() error {
			return stores[0].VectorInsertExt(ctx, coll, idc, tieFreeVec(int(idc)), rostam.VectorInsertOpts{Metadata: md})
		})
	}

	radius, box, polygon := geoGroundTruth(n)
	for name, set := range map[string]map[uint64]bool{"radius": radius, "box": box, "polygon": polygon} {
		if len(set) == 0 || uint64(len(set)) == n {
			t.Fatalf("geo ground-truth set %q has %d/%d members (vacuous filter)", name, len(set), n)
		}
		if parts := partitionSpread(set, P); len(parts) < 2 {
			t.Fatalf("geo set %q lands on only %d partition(s) %v — fan-out proof would be vacuous",
				name, len(parts), parts)
		}
	}

	// Drive the geo SEARCH from node 1 (a NON-creator) after its local catalog has
	// converged to P partitions — else it might briefly route as single-partition
	// and search the empty logical collection (a flake, not a real failure).
	e1 := stores[1].(*rostam.Embedded)
	waitEmbeddedCatalog(t, e1, coll, P, 5*time.Second)

	radiusFilter := rostam.VectorFilter{Op: vector.FilterGeoRadius, Field: "loc", Geo: &vector.GeoCondition{
		CenterLat: geoCenterLat, CenterLon: geoCenterLon, RadiusM: geoRadiusM,
	}}
	boxFilter := rostam.VectorFilter{Op: vector.FilterGeoBox, Field: "loc", Geo: &vector.GeoCondition{
		MinLat: geoBoxMinLat, MinLon: geoBoxMinLon, MaxLat: geoBoxMaxLat, MaxLon: geoBoxMaxLon,
	}}

	// SEARCH fan-out FROM node 1: k=n so a correct cluster-wide union returns
	// EXACTLY the radius match set across all partitions and 3 nodes.
	res, _, err := stores[1].VectorSearchExt(ctx, coll, tieFreeQuery(), n, rostam.VectorSearchOpts{Filter: radiusFilter})
	if err != nil {
		t.Fatalf("cluster geo_radius search via node 1: %v", err)
	}
	if ok, missing, extra := sameUint64Set(resultIDSet(res), radius); !ok {
		t.Fatalf("cluster SEARCH geo_radius fan-out id set wrong: missing=%v extra=%v (want %d ids; a dropped partition/node shows here)",
			missing, extra, len(radius))
	}

	// ValueGeo durability across the cluster: SCROLL the box set from node 1 and
	// assert each returned doc's loc lat/lon equals the seed exactly (proves the
	// geo value survived persist + cross-node replication unchanged).
	boxDocs, _, _, err := stores[1].VectorScroll(ctx, coll, boxFilter, 0, rostam.VectorScrollOpts{})
	if err != nil {
		t.Fatalf("cluster geo_box scroll via node 1: %v", err)
	}
	if ok, missing, extra := sameUint64Set(docIDSet(boxDocs), box); !ok {
		t.Fatalf("cluster SCROLL geo_box fan-out id set wrong: missing=%v extra=%v", missing, extra)
	}
	for _, d := range boxDocs {
		wantLat, wantLon := geoLocFor(d.ID)
		gotLat, gotLon := docLoc(t, d)
		if gotLat != wantLat || gotLon != wantLon {
			t.Fatalf("cluster doc %d ValueGeo loc = (%v,%v), want (%v,%v) — metadata not durable across cluster",
				d.ID, gotLat, gotLon, wantLat, wantLon)
		}
	}

	// CONCAVE polygon fan-out across the cluster (even-odd ray-cast): SCROLL the
	// polygon set from node 1 before the delete and assert the exact cross-partition
	// union — proves the []float64 polygon survives the cluster wire codec + fans out.
	polyFilter := rostam.VectorFilter{Op: vector.FilterGeoPolygon, Field: "loc", Geo: &vector.GeoCondition{Polygon: geoPoly}}
	polyDocs, _, _, err := stores[1].VectorScroll(ctx, coll, polyFilter, 0, rostam.VectorScrollOpts{})
	if err != nil {
		t.Fatalf("cluster geo_polygon scroll via node 1: %v", err)
	}
	if ok, missing, extra := sameUint64Set(docIDSet(polyDocs), polygon); !ok {
		t.Fatalf("cluster SCROLL geo_polygon fan-out id set wrong: missing=%v extra=%v", missing, extra)
	}

	// DELETE-BY-FILTER fan-out FROM node 1 (non-creator): delete the geo_box set;
	// assert exactly those docs across ALL partitions/nodes are gone and every
	// non-matching doc survives. Verify survivors FROM node 2 (a third coordinator).
	deleted, err := stores[1].VectorDeleteByFilter(ctx, coll, boxFilter)
	if err != nil {
		t.Fatalf("cluster delete-by-filter geo_box via node 1: %v", err)
	}
	if deleted != len(box) {
		t.Fatalf("cluster delete-by-filter geo_box deleted %d, want %d (must hit every partition/node)", deleted, len(box))
	}

	wantSurvivors := map[uint64]bool{}
	for i := uint64(1); i <= n; i++ {
		if !box[i] {
			wantSurvivors[i] = true
		}
	}
	e2 := stores[2].(*rostam.Embedded)
	waitEmbeddedCatalog(t, e2, coll, P, 5*time.Second)
	all, _, _, err := stores[2].VectorScroll(ctx, coll, rostam.VectorFilter{}, 0, rostam.VectorScrollOpts{})
	if err != nil {
		t.Fatalf("post-delete full scroll via node 2: %v", err)
	}
	gotSurvivors := docIDSet(all)
	if ok, missing, extra := sameUint64Set(gotSurvivors, wantSurvivors); !ok {
		t.Fatalf("cluster post-delete survivors wrong (from node 2): missing=%v extra=%v (want %d survivors)",
			missing, extra, len(wantSurvivors))
	}
	for id := range box {
		if gotSurvivors[id] {
			t.Fatalf("cluster delete-by-filter left geo_box-matched id %d alive (partition/node not reached)", id)
		}
	}
}

// TestClusterNamedVectorFanOut is the named-vector counterpart of the dense/MV
// cluster fan-out tests: a PARTITIONED (P=6) named-vector collection spread
// across a real 3-node cluster, with create/insert/search/search_docs/delete/
// scroll all issued FROM a NON-creating node. It proves the durable meta-Raft
// partition catalog (a non-creating coordinator learns P from committed cluster
// state) combines with the named fan-out to return correct cluster-wide results,
// the payload is durable across replication, and the named vector NAME selects
// the right space through the network. Consensus-gated via waitEmbeddedCatalog —
// no sleeps. Ground truth is computed independently (tie-free vectors).
func TestClusterNamedVectorFanOut(t *testing.T) {
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()

	const coll = "named"
	const P = 6
	const n = 120
	const k = 10

	cfg := map[string]rostam.NamedVectorParams{
		"title": {Dim: 8, Metric: vector.Cosine, M: 8, EfConstruction: 100, EfSearch: 64},
		"image": {Dim: 8, Metric: vector.Cosine, M: 8, EfConstruction: 100, EfSearch: 64},
	}
	// Create + populate through node 0 (the creating coordinator), ids 1..n.
	retryUntil(t, "create named", func() error {
		return stores[0].VectorNamedCreateCollection(ctx, coll, cfg, P)
	})
	for i := 1; i <= n; i++ {
		lang := "en"
		if i%2 == 1 {
			lang = "fr"
		}
		vecs := map[string][]float32{"title": namedTitleVec(i), "image": namedImageVec(i, n)}
		payload := rostam.VectorMetadata{"lang": vector.NewString(lang), "n": vector.NewInt(int64(i))}
		ii := i
		retryUntil(t, "named insert", func() error {
			return stores[0].VectorNamedInsert(ctx, coll, uint64(ii), vecs, payload, 0)
		})
	}

	// Drive ALL reads/writes from node 2 (a NON-creator). Gate on its local catalog
	// converging to P before issuing fan-out ops — without this it might briefly
	// route as single-partition and hit the empty logical name (a flake, not a bug).
	e2 := stores[2].(*rostam.Embedded)
	waitEmbeddedCatalog(t, e2, coll, P, 5*time.Second)
	node2 := stores[2]

	// ---- independent ground truth ----
	wantTitle := make([]uint64, k)
	for i := 0; i < k; i++ {
		wantTitle[i] = uint64(i + 1) // title: smallest ids
	}
	wantImage := make([]uint64, k)
	for i := 0; i < k; i++ {
		wantImage[i] = uint64(n - i) // image: largest ids
	}
	assertSpread := func(label string, ids []uint64) {
		parts := map[int]bool{}
		for _, id := range ids {
			parts[ops.PartitionOf(id, P)] = true
		}
		if len(parts) < 2 {
			t.Fatalf("%s: matched ids %v span only %d partition(s); want >=2", label, ids, len(parts))
		}
	}
	assertSpread("title top-k", wantTitle)
	assertSpread("image top-k", wantImage)

	// ---- search "title" FROM node 2 (cross-partition union/top-k) ----
	gotTitle, err := node2.VectorNamedSearch(ctx, coll, "title", namedTitleQuery(), k, rostam.VectorFilter{})
	if err != nil {
		t.Fatalf("node2 VectorNamedSearch title: %v", err)
	}
	if len(gotTitle) != k {
		t.Fatalf("title search from node2 returned %d, want %d (dropped partition?)", len(gotTitle), k)
	}
	for i := range wantTitle {
		if gotTitle[i].ID != wantTitle[i] {
			t.Fatalf("title rank %d from node2: got %d, want %d\n got=%v\nwant=%v",
				i, gotTitle[i].ID, wantTitle[i], resultIDs(gotTitle), wantTitle)
		}
	}

	// ---- search "image" FROM node 2 (different ranking → name selects the space) ----
	gotImage, err := node2.VectorNamedSearch(ctx, coll, "image", namedImageQuery(), k, rostam.VectorFilter{})
	if err != nil {
		t.Fatalf("node2 VectorNamedSearch image: %v", err)
	}
	for i := range wantImage {
		if gotImage[i].ID != wantImage[i] {
			t.Fatalf("image rank %d from node2: got %d, want %d\n got=%v\nwant=%v",
				i, gotImage[i].ID, wantImage[i], resultIDs(gotImage), wantImage)
		}
	}

	// ---- filtered title search (lang=en → even ids) FROM node 2 ----
	enFilter := rostam.VectorFilter{Op: vector.FilterEq, Field: "lang", Value: vector.NewString("en")}
	gotEn, err := node2.VectorNamedSearch(ctx, coll, "title", namedTitleQuery(), k, enFilter)
	if err != nil {
		t.Fatalf("node2 filtered VectorNamedSearch: %v", err)
	}
	wantEn := make([]uint64, k)
	for i := 0; i < k; i++ {
		wantEn[i] = uint64(2 * (i + 1))
	}
	for i := range wantEn {
		if gotEn[i].ID != wantEn[i] {
			t.Fatalf("filtered rank %d from node2: got %d, want %d\n got=%v\nwant=%v",
				i, gotEn[i].ID, wantEn[i], resultIDs(gotEn), wantEn)
		}
	}

	// ---- search_docs payload durable across replication, read FROM node 2 ----
	docs, err := node2.VectorNamedSearchDocs(ctx, coll, "title", namedTitleQuery(), k, rostam.VectorFilter{})
	if err != nil {
		t.Fatalf("node2 VectorNamedSearchDocs: %v", err)
	}
	for _, d := range docs {
		wantLang := "en"
		if d.ID%2 == 1 {
			wantLang = "fr"
		}
		if got, ok := d.Metadata["lang"]; !ok || got.Str != wantLang {
			t.Fatalf("doc id %d payload lang=%+v, want %q (payload not durable across replication?)", d.ID, got, wantLang)
		}
		if nv, ok := d.Metadata["n"]; !ok || nv.Int != int64(d.ID) {
			t.Fatalf("doc id %d payload n=%+v, want %d", d.ID, nv, d.ID)
		}
	}

	// ---- scroll FROM node 2: all n distinct across partitions ----
	scrolled, _, err := node2.VectorNamedScroll(ctx, coll, rostam.VectorFilter{}, 0, "")
	if err != nil {
		t.Fatalf("node2 VectorNamedScroll: %v", err)
	}
	if len(scrolled) != n || len(idSet(scrolled)) != n {
		t.Fatalf("scroll from node2 = %d (%d distinct), want %d/%d", len(scrolled), len(idSet(scrolled)), n, n)
	}

	// ---- delete id 4 FROM node 2; gone from all spaces + scroll, cluster-wide ----
	const delID = 4
	ok, err := node2.VectorNamedDelete(ctx, coll, delID)
	if err != nil {
		t.Fatalf("node2 VectorNamedDelete: %v", err)
	}
	if !ok {
		t.Fatalf("delete id %d from node2 reported not-existed", delID)
	}
	afterTitle, err := node2.VectorNamedSearch(ctx, coll, "title", namedTitleQuery(), k, rostam.VectorFilter{})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range afterTitle {
		if r.ID == delID {
			t.Fatalf("deleted id %d still in title search", delID)
		}
	}
	afterScroll, _, err := node2.VectorNamedScroll(ctx, coll, rostam.VectorFilter{}, 0, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(afterScroll) != n-1 {
		t.Fatalf("after delete scroll from node2 = %d, want %d", len(afterScroll), n-1)
	}

	// ---- drop FROM node 2: fan-out drops every physical partition ----
	if err := node2.VectorNamedDropCollection(ctx, coll); err != nil {
		t.Fatalf("node2 VectorNamedDropCollection: %v", err)
	}
}

// TestClusterGetPayloadFanOut proves get-by-id + the four payload mutations work
// cluster-wide on a real 3-node cluster, driven from a NON-creator node. A P=6
// dense collection is created + populated through node 0; node 2 (which did NOT
// create it) then gets each point by id (route-by-id to the owning physical
// partition across nodes), mutates the payload via set/overwrite/delete-keys/
// clear, and re-gets to confirm each mutation is durable across replication and
// observable cluster-wide. A filtered search after a payload-update reflects the
// new field cross-partition (reindex held on the owning physical partition).
// Readiness is consensus-gated via waitEmbeddedCatalog (no sleeps); transient
// not-leader/forwarding errors are absorbed by retryUntil.
func TestClusterGetPayloadFanOut(t *testing.T) {
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()

	const coll = "gpc"
	const P = 6
	const N = 60

	retryUntil(t, "CreateCollection "+coll, func() error {
		return stores[0].CreateCollection(ctx, coll, rostam.VectorConfig{
			Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: P,
		})
	})
	for id := uint64(1); id <= N; id++ {
		v := []float32{float32(id), 0, 0, 0}
		md := rostam.VectorMetadata{"v": vector.NewInt(int64(id))}
		retryUntil(t, fmt.Sprintf("insert %s %d", coll, id), func() error {
			return stores[0].VectorInsertExt(ctx, coll, id, v, rostam.VectorInsertOpts{Metadata: md})
		})
	}

	// Drive get + payload ops from node 2 (a NON-creator). Converge its catalog
	// first so it never briefly routes to the empty logical collection.
	e2 := stores[2].(*rostam.Embedded)
	waitEmbeddedCatalog(t, e2, coll, P, 5*time.Second)

	// Assert the ids span >=2 partitions so route-by-id is genuinely exercised.
	parts := map[int]bool{}
	for id := uint64(1); id <= N; id++ {
		parts[ops.PartitionOf(id, P)] = true
	}
	if len(parts) < 2 {
		t.Fatalf("ids span only %d partition(s); want >=2", len(parts))
	}

	// Get every point by id from node 2 — each routes cross-node to its owning
	// physical partition and returns the correct vec + payload.
	for id := uint64(1); id <= N; id++ {
		found, vec, meta, _, _, err := stores[2].VectorGet(ctx, coll, id, true, true)
		if err != nil {
			t.Fatalf("node2 VectorGet %d: %v", id, err)
		}
		if !found {
			t.Fatalf("node2 VectorGet %d: not found (route-by-id missed owning partition)", id)
		}
		if len(vec) != 4 || vec[0] != float32(id) {
			t.Fatalf("node2 VectorGet %d: vec=%v, want [%d 0 0 0]", id, vec, id)
		}
		if meta["v"].Int != int64(id) {
			t.Fatalf("node2 VectorGet %d: payload v=%d, want %d", id, meta["v"].Int, id)
		}
	}

	// Absent id -> not-found FLAG, not an error.
	if found, _, _, _, _, err := stores[2].VectorGet(ctx, coll, 99999, true, true); err != nil || found {
		t.Fatalf("node2 VectorGet absent: found=%v err=%v, want false/nil", found, err)
	}

	// Payload ops from node 2; re-get from node 1 (the third node) to prove the
	// mutation is durable + cluster-wide, not just locally visible.
	e1 := stores[1].(*rostam.Embedded)
	waitEmbeddedCatalog(t, e1, coll, P, 5*time.Second)

	const tgt = 13
	retryUntil(t, "set payload via node2", func() error {
		applied, err := stores[2].VectorSetPayload(ctx, coll, tgt, rostam.VectorMetadata{"tag": vector.NewString("hot")}, nil)
		if err != nil {
			return err
		}
		if !applied {
			return fmt.Errorf("set payload: applied=false")
		}
		return nil
	})
	_, _, meta, _, _, err := stores[1].VectorGet(ctx, coll, tgt, false, true)
	if err != nil {
		t.Fatalf("node1 VectorGet after set: %v", err)
	}
	if meta["v"].Int != tgt || meta["tag"].Str != "hot" {
		t.Fatalf("node1 after set: %+v, want v=%d,tag=hot", meta, tgt)
	}

	// Cross-partition filtered search reflects the new field. Tag a second id on a
	// different partition so the match set spans >=2 partitions.
	var other uint64
	for id := uint64(1); id <= N; id++ {
		if id != tgt && ops.PartitionOf(id, P) != ops.PartitionOf(tgt, P) {
			other = id
			break
		}
	}
	if other == 0 {
		t.Fatal("could not find a second id on a different partition")
	}
	retryUntil(t, "set payload other via node2", func() error {
		applied, err := stores[2].VectorSetPayload(ctx, coll, other, rostam.VectorMetadata{"tag": vector.NewString("hot")}, nil)
		if err != nil {
			return err
		}
		if !applied {
			return fmt.Errorf("set payload other: applied=false")
		}
		return nil
	})
	fr, _, err := stores[1].VectorSearchExt(ctx, coll, []float32{1, 0, 0, 0}, N, rostam.VectorSearchOpts{
		Filter: rostam.VectorFilter{Op: vector.FilterEq, Field: "tag", Value: vector.NewString("hot")},
	})
	if err != nil {
		t.Fatalf("node1 filtered search post-update: %v", err)
	}
	gotMatch := map[uint64]bool{}
	for _, r := range fr {
		gotMatch[r.ID] = true
	}
	if len(fr) != 2 || !gotMatch[tgt] || !gotMatch[other] {
		t.Fatalf("filter tag=hot = %v, want exactly {%d,%d} cross-partition (reindex per-partition)", ids(fr), tgt, other)
	}

	// overwrite -> only k=1.
	retryUntil(t, "overwrite payload via node2", func() error {
		_, err := stores[2].VectorOverwritePayload(ctx, coll, tgt, rostam.VectorMetadata{"k": vector.NewInt(1)}, nil)
		return err
	})
	_, _, meta, _, _, _ = stores[1].VectorGet(ctx, coll, tgt, false, true)
	if _, ok := meta["v"]; ok || meta["k"].Int != 1 {
		t.Fatalf("node1 after overwrite: %+v, want only k=1", meta)
	}
	// delete-keys removes k.
	retryUntil(t, "delete-keys payload via node2", func() error {
		_, err := stores[2].VectorDeletePayloadKeys(ctx, coll, tgt, []string{"k"})
		return err
	})
	_, _, meta, _, _, _ = stores[1].VectorGet(ctx, coll, tgt, false, true)
	if _, ok := meta["k"]; ok {
		t.Fatalf("node1 after delete-keys: %+v, want no k", meta)
	}
	// clear -> empty.
	retryUntil(t, "clear payload via node2", func() error {
		_, err := stores[2].VectorClearPayload(ctx, coll, tgt)
		return err
	})
	_, _, meta, _, _, _ = stores[1].VectorGet(ctx, coll, tgt, false, true)
	if len(meta) != 0 {
		t.Fatalf("node1 after clear: %+v, want empty", meta)
	}
}

// waitEmbeddedAlias polls a node's local embedded catalog until the alias
// resolves to wantCanonical, failing the test on timeout. Alias targets are
// stored CANONICALLY in the catalog (SetAliases canonicalizes both alias and
// target), so we compare ResolveAlias's output against ops.CanonicalName of the
// wanted target. wantCanonical == "" means "wait until the alias is GONE"
// (ResolveAlias returns ok=false) — used for the drop-cascade assertion. This
// closes the convergence window where a non-creating node has not yet applied
// the alias batch from meta-Raft.
func waitEmbeddedAlias(t *testing.T, e *rostam.Embedded, alias, wantCanonical string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	want := ops.CanonicalName(wantCanonical)
	for time.Now().Before(deadline) {
		canon, ok := e.Catalog().ResolveAlias(alias)
		if wantCanonical == "" {
			if !ok {
				return
			}
		} else if ok && canon == want {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	canon, ok := e.Catalog().ResolveAlias(alias)
	if wantCanonical == "" {
		t.Fatalf("alias %q = (%q, ok=%v), want GONE (ok=false)", alias, canon, ok)
	}
	t.Fatalf("alias %q = (%q, ok=%v), want canonical %q", alias, canon, ok, want)
}

// TestClusterAliasSwapFanOut proves the zero-downtime alias swap end-to-end on a
// real 3-node cluster: alias create/swap/drop driven across nodes, and the
// data-plane READ THROUGH the alias performed from NON-creator nodes against
// PARTITIONED (P=6) collections. It combines the durable meta-Raft alias map
// (every node converges on the alias→target mapping) with partition fan-out (the
// resolved target is partitioned, so the read scatters across its physicals).
//
// v1 (ids 1..300) and v2 (ids 1001..1300) seed DISJOINT id ranges so the result
// ids reveal WHICH collection the alias resolved to. The swap is a single atomic
// two-action AliasBatch (delete prod→v1, create prod→v2) issued from a NON-creator
// node, proving any node can flip the alias under the single meta-FSM lock. Reads
// before/after the swap come from yet OTHER non-creator nodes. Finally, dropping
// coll_v2 cascades the prod alias away on every node.
func TestClusterAliasSwapFanOut(t *testing.T) {
	stores := newInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()

	cfg := rostam.VectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: 6}

	// Create both partitioned collections via node 0.
	for _, coll := range []string{"coll_v1", "coll_v2"} {
		c := coll
		retryUntil(t, "create "+c, func() error {
			return stores[0].CreateCollection(ctx, c, cfg)
		})
	}
	// Seed disjoint id ranges via node 0: 1..300 -> v1, 1001..1300 -> v2.
	for id := uint64(1); id <= 300; id++ {
		idc := id
		retryUntil(t, "insert coll_v1", func() error {
			return stores[0].VectorInsert(ctx, "coll_v1", idc, []float32{float32(idc), 0, 0, 0})
		})
	}
	for id := uint64(1001); id <= 1300; id++ {
		idc := id
		retryUntil(t, "insert coll_v2", func() error {
			return stores[0].VectorInsert(ctx, "coll_v2", idc, []float32{float32(idc), 0, 0, 0})
		})
	}

	// Converge the partition catalog on EVERY node before alias ops, so the
	// fan-out read on a non-creator node sees P=6 (not the empty logical name).
	for i := range stores {
		ei := stores[i].(*rostam.Embedded)
		waitEmbeddedCatalog(t, ei, "coll_v1", 6, 5*time.Second)
		waitEmbeddedCatalog(t, ei, "coll_v2", 6, 5*time.Second)
	}

	// Create alias prod -> coll_v1 from node 0.
	retryUntil(t, "create alias", func() error {
		return stores[0].CreateAlias(ctx, "prod", "coll_v1")
	})
	// Wait for the alias to converge on the non-creator nodes 1 and 2.
	waitEmbeddedAlias(t, stores[1].(*rostam.Embedded), "prod", "coll_v1", 5*time.Second)
	waitEmbeddedAlias(t, stores[2].(*rostam.Embedded), "prod", "coll_v1", 5*time.Second)

	inV1 := func(id uint64) bool { return id >= 1 && id <= 300 }
	inV2 := func(id uint64) bool { return id >= 1001 && id <= 1300 }

	// READ THROUGH the alias from NON-creator nodes 1 and 2: non-empty and ALL ids
	// in v1's range — proving alias resolution + partition fan-out work cluster-wide
	// from nodes that did NOT create the alias.
	for _, i := range []int{1, 2} {
		res, err := stores[i].VectorSearch(ctx, "prod", []float32{1, 0, 0, 0}, 5)
		if err != nil {
			t.Fatalf("node %d search via alias prod: %v", i, err)
		}
		if len(res) != 5 {
			t.Fatalf("node %d: alias search returned %d results, want 5 (alias did not resolve / fan out across v1's partitions): %v", i, len(res), ids(res))
		}
		for r := 0; r < len(res); r++ {
			if !inV1(res[r].ID) {
				t.Fatalf("node %d alias search rank %d: id %d not in v1 range 1..300: %v", i, r, res[r].ID, ids(res))
			}
		}
	}

	// ATOMIC swap from NON-creator node 1 (proving any node can flip the alias):
	// ONE AliasBatch carrying delete prod->v1 then create prod->v2, applied under a
	// single meta-FSM lock (cluster/meta_fsm.go OpSetAliasBatch).
	retryUntil(t, "atomic alias swap via node 1", func() error {
		return stores[1].AliasBatch(ctx, []rostam.AliasAction{
			{Alias: "prod", Delete: true},
			{Alias: "prod", Canonical: "coll_v2"},
		})
	})
	// Wait for the swap to converge to coll_v2 on the reading nodes.
	waitEmbeddedAlias(t, stores[2].(*rostam.Embedded), "prod", "coll_v2", 5*time.Second)
	waitEmbeddedAlias(t, stores[0].(*rostam.Embedded), "prod", "coll_v2", 5*time.Second)

	// From node 2: search via prod now returns ids in v2's range — atomic, fully v2
	// (no v1 leftovers), proving the swap presented v1-then-v2 with no mixed window.
	res, err := stores[2].VectorSearch(ctx, "prod", []float32{1001, 0, 0, 0}, 5)
	if err != nil {
		t.Fatalf("node 2 search via alias prod after swap: %v", err)
	}
	if len(res) != 5 {
		t.Fatalf("node 2: post-swap alias search returned %d results, want 5 (swap exposed an undefined/partial window): %v", len(res), ids(res))
	}
	for r := 0; r < len(res); r++ {
		if !inV2(res[r].ID) {
			t.Fatalf("node 2 post-swap rank %d: id %d not in v2 range 1001..1300 (swap left a v1+v2 mix): %v", r, res[r].ID, ids(res))
		}
	}

	// Drop-cascade: drop coll_v2 from node 0. The alias drop-cascade lives at the
	// fanout-dispatcher chokepoint (fanout_dispatcher.go fanDropCollection), which
	// is the dispatcher every real transport (TCP/HTTP/gRPC) routes through — see
	// server.go's cluster branch (disp = rostam.NewFanoutDispatcher(emb, emb.Node())). The
	// raw *rostam.Embedded.Call exposed as stores[i] is the INNER dispatcher and bypasses
	// that chokepoint (no cascade), so we drive the drop through a fanout dispatcher
	// over node 0's embedded, exactly as the wire path does. The cascade removes
	// every alias targeting the dropped collection, so prod must be GONE on all nodes.
	e0 := stores[0].(*rostam.Embedded)
	dropDisp := rostam.NewFanoutDispatcher(e0, e0.Node())
	if _, err := dropDisp.Call("vector_drop_collection", ops.EncodeDropCollectionArgs("coll_v2")); err != nil {
		t.Fatalf("drop coll_v2 via dispatcher: %v", err)
	}
	for i := range stores {
		waitEmbeddedAlias(t, stores[i].(*rostam.Embedded), "prod", "", 5*time.Second)
	}
}

// wcFindLeaderIdx returns the index of the node that currently leads the shard
// owning routeKey, polling through the residual election window. At RF=N every
// node OWNS every shard, but only the shard's Raft leader can accept a write to
// it: the embedded write path returns NotLeader for a follower-hosted shard and
// (unlike HTTP/gRPC) does NOT forward — so a WCF write must be driven from the
// leader node. Returns -1 if no leader is visible to any node within the budget.
func wcFindLeaderIdx(t *testing.T, stores []rostam.Store, routeKey []byte) int {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		for i, s := range stores {
			if s != nil && s.IsLeader(routeKey) {
				return i
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	return -1
}

// TestClusterWriteConsistencyBarrier is the end-to-end proof of the write-
// consistency barrier over the embedded rostam.Store on a 3-node RF=3 cluster (every
// node owns every shard). It drives writes through stores[i].VectorInsert with
// typed rostam.WriteOpts — the embedded path threads WriteConsistencyFactor/Wait into
// cluster.Node.BarrierForShard after the Raft commit — and verifies the barrier's
// four behaviors against INDEPENDENT ground truth: a per-owner LOCAL vector_get
// (OpReadOnly, served from each node's own hosted replica with no forwarding), so
// a point found on stores[j] proves owner j has APPLIED the write.
//
// Collection shape: a SINGLE-PARTITION (P=1, unpartitioned) collection is used so
// the target shard is deterministic — shardOf(canonical("wc"), numShards) — and
// every owner is the full RF=3 replica set of that one shard. (A partitioned
// collection would force computing the per-id partition's owners for no added
// coverage here; the TestEmbeddedWriteConsistencyPartitioned* already cover
// the partition-physical-name barrier wiring.)
//
// Writer node: at RF=3 the embedded write path does NOT forward a follower-hosted
// write (returns NotLeader, no leader-following — the documented harness limit, NOT
// a production change), so each write is driven from the node that LEADS the target
// shard (wcFindLeaderIdx). retryUntil would never promote a follower, so this is a
// leader lookup, not a not-leader retry.
//
// Downed-owner induction (cases b/c/d): mirrors cluster-pkg TestBarrierForShardTimeout
// — close a NON-leader owner's *rostam.Server, which closes BOTH its node and its TCP
// listener, so the surviving leader's __rb_status__ poll to it fails at the
// transport and degrades to a zero-value status (AppliedIndex 0 < target) → that
// owner never counts toward the factor. Closing the leader's owners is avoided so
// Raft majority (2/3) stays intact and the writes still commit.
func TestClusterWriteConsistencyBarrier(t *testing.T) {
	const (
		n         = 3
		numShards = 4
		rf        = 3
	)
	stores, servers := newInmemEmbeddedClusterServers(t, n, numShards, rf)
	ctx := context.Background()

	const coll = "wc"
	routeKey := []byte(ops.CanonicalName(coll))
	trueVal, falseVal := true, false

	// ---- create the unpartitioned collection from its shard's leader ----
	leaderIdx := wcFindLeaderIdx(t, stores, routeKey)
	if leaderIdx < 0 {
		t.Fatal("no leader elected for the target shard within budget")
	}
	cfg := rostam.VectorConfig{Dim: 4, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64}
	retryUntil(t, "create wc", func() error {
		return stores[leaderIdx].CreateCollection(ctx, coll, cfg)
	})

	// targetShard is the one Raft group all RF=3 owners replicate.
	targetShard := stores[leaderIdx].(*rostam.Embedded).Node().ShardIndexForName(coll)

	// readAllOwners returns how many of the n nodes find point id LOCALLY. A local
	// vector_get is served from the node's own hosted replica (RF=3 ⇒ every node
	// hosts the shard), so a hit proves that owner has applied the write. Closed
	// nodes (nil store) are skipped and never counted.
	readAllOwners := func(id uint64) int {
		found := 0
		for _, s := range stores {
			if s == nil {
				continue
			}
			ok, _, _, _, _, err := s.(*rostam.Embedded).VectorGet(ctx, coll, id, false, false)
			if err == nil && ok {
				found++
			}
		}
		return found
	}

	// localFound reports whether node idx has point id APPLIED in its own replica
	// (a LOCAL OpReadOnly read — no forwarding).
	localFound := func(idx int, id uint64) bool {
		if stores[idx] == nil {
			return false
		}
		ok, _, _, _, _, err := stores[idx].(*rostam.Embedded).VectorGet(ctx, coll, id, false, false)
		return err == nil && ok
	}

	// waitLiveOwnersApplied polls until ALL live (non-nil) owners have applied id,
	// or fails after d. Used for the <=majority / wait=false cases where the write
	// returns at Raft commit (leader applied) but a follower's FSM apply is
	// asynchronous — the barrier is exactly what those cases deliberately skip, so
	// follower read-visibility is EVENTUAL, gated here (not a fixed sleep).
	waitLiveOwnersApplied := func(id uint64, want int, d time.Duration) {
		t.Helper()
		deadline := time.Now().Add(d)
		for time.Now().Before(deadline) {
			if readAllOwners(id) >= want {
				return
			}
			time.Sleep(20 * time.Millisecond)
		}
		t.Fatalf("point %d applied on %d live owners, want %d within %s", id, readAllOwners(id), want, d)
	}

	// ---- (a) WCF=3 from the shard leader: on return, ALL 3 owners have applied ----
	// WCF=3 == RF blocks until every owner applied the entry, so the immediate
	// post-return LOCAL read on each of the 3 nodes MUST find the point.
	const idA = uint64(1)
	li := wcFindLeaderIdx(t, stores, routeKey)
	if li < 0 {
		t.Fatal("(a) no leader for target shard")
	}
	retryUntil(t, "(a) WCF=3 insert", func() error {
		return stores[li].VectorInsert(ctx, coll, idA, []float32{1, 0, 0, 0}, rostam.WriteOpts{WriteConsistencyFactor: 3})
	})
	if got := readAllOwners(idA); got != n {
		t.Fatalf("(a) WCF=3 returned but only %d/%d owners have point %d applied; barrier did not wait for all replicas", got, n, idA)
	}
	t.Logf("(a) WCF=3 insert visible on all %d owners immediately (target shard %d)", n, targetShard)

	// ---- down a NON-leader owner of the target shard ----
	li = wcFindLeaderIdx(t, stores, routeKey)
	if li < 0 {
		t.Fatal("no leader before downing an owner")
	}
	downIdx := (li + 1) % n
	if err := servers[downIdx].Close(); err != nil {
		t.Fatalf("closing owner %d: %v", downIdx, err)
	}
	servers[downIdx] = nil
	stores[downIdx] = nil // node gone — never touch it again

	// Re-find the leader among the two survivors (the downed node was a follower,
	// so the leader is unchanged, but re-resolve to be robust to any re-election).
	li = wcFindLeaderIdx(t, stores, routeKey)
	if li < 0 {
		t.Fatal("no leader among survivors after downing an owner")
	}

	// ---- (b) WCF=2 (== majority) succeeds without the 3rd owner ----
	// majority(RF=3)=2, so eff=2 <= maj ⇒ NO barrier: the write returns at Raft
	// commit (leader + 1 live follower = 2 = majority) without waiting for the
	// downed 3rd, and crucially WITHOUT an apply-barrier on the live follower. So
	// the contract proven here is: the call SUCCEEDS (no error/timeout) and the
	// write is immediately read-visible on the LEADER (which applied it). The live
	// follower's FSM apply is asynchronous — that read-visibility is EVENTUAL,
	// because <=majority deliberately skips the barrier — so it is gated, not
	// asserted instant. (Asserting the follower instant is precisely what WCF=3
	// would buy and what WCF=2 is documented NOT to.)
	const idB = uint64(2)
	retryUntil(t, "(b) WCF=2 insert", func() error {
		return stores[li].VectorInsert(ctx, coll, idB, []float32{0, 1, 0, 0}, rostam.WriteOpts{WriteConsistencyFactor: 2})
	})
	if !localFound(li, idB) {
		t.Fatalf("(b) WCF=2 returned but point %d not visible on the leader (node %d)", idB, li)
	}
	waitLiveOwnersApplied(idB, n-1, 5*time.Second)
	t.Log("(b) WCF=2 (==majority) succeeded at quorum without the downed 3rd owner (follower caught up eventually)")

	// ---- (c) WCF=3 with one owner down TIMES OUT, write still durable ----
	// eff=3 > maj=2 ⇒ barrier engages; the downed owner can never apply (its
	// __rb_status__ poll fails forever), so the barrier times out (5s default —
	// the embedded path has no per-call timeout override) with a *ErrWriteConsistency
	// whose message carries the "cluster: write " prefix. The write is STILL
	// committed at majority and readable on the 2 live owners.
	const idC = uint64(3)
	li = wcFindLeaderIdx(t, stores, routeKey)
	if li < 0 {
		t.Fatal("(c) no leader among survivors")
	}
	start := time.Now()
	err := stores[li].VectorInsert(ctx, coll, idC, []float32{0, 0, 1, 0}, rostam.WriteOpts{WriteConsistencyFactor: 3, Wait: &trueVal})
	el := time.Since(start)
	if err == nil {
		t.Fatal("(c) WCF=3 with one owner down: want ErrWriteConsistency, got nil")
	}
	// Match ErrWriteConsistency.Error() exactly, not the bare "cluster: write "
	// prefix (which the unrelated wasm-load errors in cluster/wasm_load.go also
	// share) — "committed at quorum" is exclusive to ErrWriteConsistency.
	if !strings.Contains(err.Error(), "cluster: write committed at quorum") {
		t.Fatalf("(c) WCF=3 timeout error = %q, want ErrWriteConsistency (committed at quorum, factor not met)", err.Error())
	}
	if el < 250*time.Millisecond {
		t.Errorf("(c) barrier returned in %s — too fast to have polled to a real timeout", el)
	}
	// Durability: the write committed at Raft majority before the barrier failed,
	// so it is present on the leader immediately and on the other live owner
	// eventually (the barrier's failure does NOT roll the write back). NOTE: with
	// the 5s default timeout, the surviving follower has had ample time to apply,
	// but gate it rather than assume.
	if !localFound(li, idC) {
		t.Fatalf("(c) after timeout, point %d not durable on the leader (node %d) — the write must persist at majority", idC, li)
	}
	waitLiveOwnersApplied(idC, n-1, 5*time.Second)
	t.Logf("(c) WCF=3 timed out (%s) with ErrWriteConsistency; write durable on %d live owners", el, n-1)

	// ---- (d) wait=false skips the barrier even with an owner down ----
	// wait=false returns at Raft majority and SKIPS the >majority barrier, so a
	// WCF=3 write with the 3rd owner down returns nil (no timeout).
	const idD = uint64(4)
	li = wcFindLeaderIdx(t, stores, routeKey)
	if li < 0 {
		t.Fatal("(d) no leader among survivors")
	}
	startD := time.Now()
	retryUntil(t, "(d) WCF=3 wait=false insert", func() error {
		return stores[li].VectorInsert(ctx, coll, idD, []float32{0, 0, 0, 1}, rostam.WriteOpts{WriteConsistencyFactor: 3, Wait: &falseVal})
	})
	if elD := time.Since(startD); elD > 2*time.Second {
		t.Errorf("(d) wait=false took %s — barrier was not skipped (should return at majority)", elD)
	}
	// wait=false skips the barrier ⇒ leader-immediate, follower-eventual (same as b).
	if !localFound(li, idD) {
		t.Fatalf("(d) wait=false returned but point %d not visible on the leader (node %d)", idD, li)
	}
	waitLiveOwnersApplied(idD, n-1, 5*time.Second)
	t.Log("(d) wait=false returned at majority without engaging the barrier (follower caught up eventually)")
}

// TestClusterScrollCursorDeepPagination is the headline cluster proof for cursor
// pagination: on a real 3-node cluster, a PARTITIONED (P=4) collection seeded
// with N distinct tie-free ids in RANDOM insert order via node 0 is deep-paged
// from a NON-creator node, and the partitioned fan-out cursor merge must return
// every id EXACTLY once, globally strictly ascending across page boundaries —
// proving the global id-ascending merge is correct cluster-wide from a node that
// did not create the data.
//
// Three sub-proofs, all driven FROM a non-creator node after its local catalog
// converges to P=4 (waitEmbeddedCatalog), so a brief single-partition mis-route
// cannot masquerade as a merge bug:
//   - (a) deep pagination: union of all pages == full id set, exactly once,
//     globally ascending, total==N.
//   - (b) filter + cursor: a selective (even-only) filter paged to exhaustion
//     returns only matching ids, exactly once, ascending.
//   - (c) deletion mid-scroll: a not-yet-paged id deleted via node 0 between
//     pages is absent from the remaining pages with no gap (the other live ids
//     still all appear exactly once). The delete's cluster-wide visibility is
//     gated by polling a full scroll on the reading node until the id is gone —
//     a consensus-gated wait, NOT a fixed sleep.
//
// The named-family partitioned deep-pagination exactly-once property is covered
// by TestEmbeddedNamedScrollCursorDeepPaginationPartitioned (embedded) and the
// HTTP/gRPC NamedScroll cursor tests (remote_scroll_cursor_integration_test.go),
// so it is not re-stood-up on the heavier 3-node harness here.
func TestClusterScrollCursorDeepPagination(t *testing.T) {
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()

	const (
		coll = "deep"
		P    = 4
		N    = 250
		L    = 30
	)

	// Create the partitioned collection through node 0 (the creating coordinator).
	retryUntil(t, "create deep", func() error {
		return stores[0].CreateCollection(ctx, coll, rostam.VectorConfig{
			Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2, Partitions: P,
		})
	})

	// Seed N distinct tie-free ids in RANDOM insert order via node 0. A stable
	// global ascending paged result then proves the merge, not insert order.
	insertIDs := shuffledIDs(N, 42) // ids 1..N permuted
	for _, id := range insertIDs {
		idc := id
		v := []float32{float32(idc), 0, 0, 0}
		md := rostam.VectorMetadata{"even": vector.NewBool(idc%2 == 0)}
		retryUntil(t, fmt.Sprintf("insert deep %d", idc), func() error {
			return stores[0].VectorInsertExt(ctx, coll, idc, v, rostam.VectorInsertOpts{Metadata: md})
		})
	}

	// Read from a NON-creator node (node 1) after its catalog converges to P=4.
	const reader = 1
	e1 := stores[reader].(*rostam.Embedded)
	waitEmbeddedCatalog(t, e1, coll, P, 5*time.Second)

	want := map[uint64]bool{}
	for _, id := range insertIDs {
		want[id] = true
	}

	// (a) Deep pagination from the non-creator node: every id exactly once,
	// globally ascending, total==N. pageAllDense also asserts per-page ascending,
	// no cross-page descent, and the exhaustion rule (full page ⇒ cursor present;
	// short page ⇒ cursor empty).
	got, pages := pageAllDense(t, stores[reader], coll, rostam.VectorFilter{}, L)
	assertExactlyOnceAscending(t, got, want)
	wantPages := (N + L - 1) / L // ceil(N/L)
	if pages != wantPages && pages != wantPages+1 {
		t.Fatalf("(a) page count = %d, want %d or %d (ceil(N/L))", pages, wantPages, wantPages+1)
	}

	// (b) Filter + cursor from the non-creator node: only even ids, exactly once,
	// ascending, to exhaustion.
	evenFilter := rostam.VectorFilter{Op: vector.FilterEq, Field: "even", Value: vector.NewBool(true)}
	wantEven := map[uint64]bool{}
	for _, id := range insertIDs {
		if id%2 == 0 {
			wantEven[id] = true
		}
	}
	gotEven, _ := pageAllDense(t, stores[reader], coll, evenFilter, 13)
	assertExactlyOnceAscending(t, gotEven, wantEven)

	// (c) Deletion mid-scroll. Page once from the non-creator node to advance the
	// cursor partway, then delete a NOT-YET-PAGED id via node 0 (the creating
	// coordinator). Gate the delete's cluster-wide visibility by polling a full
	// scroll on the reading node until the id is gone (consensus-gated; no fixed
	// sleep). Continue paging to exhaustion and assert the deleted id is absent
	// with no gap (every other live id > cursorMax still appears exactly once).
	firstPage, _, next, err := stores[reader].VectorScroll(ctx, coll, rostam.VectorFilter{}, L, rostam.VectorScrollOpts{})
	if err != nil {
		t.Fatalf("(c) first page: %v", err)
	}
	if len(firstPage) != L {
		t.Fatalf("(c) first page len = %d, want %d", len(firstPage), L)
	}
	cursorMax := firstPage[len(firstPage)-1].ID // largest id paged so far
	if cursorMax >= N-2 {
		t.Fatalf("(c) first page advanced too far (cursorMax=%d, N=%d); cannot test mid-pagination", cursorMax, N)
	}
	// ids 1..N all exist, so cursorMax+1 is a guaranteed-live, not-yet-paged id.
	delID := cursorMax + 1
	if _, err := stores[0].VectorDelete(ctx, coll, delID); err != nil {
		t.Fatalf("(c) delete %d via node 0: %v", delID, err)
	}
	// Consensus-gated wait: poll a FULL scroll on the reading node until delID is
	// absent (the delete has propagated to every partition the reader fans out to).
	// A dedicated poll loop (not retryUntil) so "still visible" is distinct from a
	// real scroll error, which fails loud immediately.
	delDeadline := time.Now().Add(30 * time.Second)
	for {
		all, _, _, err := stores[reader].VectorScroll(ctx, coll, rostam.VectorFilter{}, 0, rostam.VectorScrollOpts{})
		if err != nil {
			t.Fatalf("(c) scroll while waiting for delete %d to propagate: %v", delID, err)
		}
		if !idSet(all)[delID] {
			break
		}
		if !time.Now().Before(delDeadline) {
			t.Fatalf("(c) delete %d not visible on reader within 30s", delID)
		}
		time.Sleep(50 * time.Millisecond)
	}

	// Continue paging from the saved cursor to exhaustion; collect the remaining ids
	// with inline per-page + cross-page ascending checks (clearer failure messages).
	cursor := next
	var rest []uint64
	var prevRest uint64
	haveRest := false
	for cursor != "" {
		pg, _, nxt, err := stores[reader].VectorScroll(ctx, coll, rostam.VectorFilter{}, L, rostam.VectorScrollOpts{Cursor: cursor})
		if err != nil {
			t.Fatalf("(c) scroll page: %v", err)
		}
		for i, d := range pg {
			if i > 0 && d.ID <= pg[i-1].ID {
				t.Fatalf("(c) continuation page not ascending at %d: %d <= %d", i, d.ID, pg[i-1].ID)
			}
			if haveRest && d.ID <= prevRest {
				t.Fatalf("(c) cross-page descent: %d <= %d", d.ID, prevRest)
			}
			rest = append(rest, d.ID)
			prevRest, haveRest = d.ID, true
		}
		cursor = nxt
	}
	// The remaining ids must be exactly the live ids > cursorMax (original ids in
	// (cursorMax, N] minus delID), each exactly once, ascending — no gap.
	wantRest := map[uint64]bool{}
	for id := cursorMax + 1; id <= N; id++ {
		if id != delID {
			wantRest[id] = true
		}
	}
	if restSet := idSetU(rest); restSet[delID] {
		t.Fatalf("(c) deleted id %d still appeared in a later page", delID)
	}
	assertExactlyOnceAscending(t, rest, wantRest)
}

// idSetU is idSet for a raw id slice.
func idSetU(ids []uint64) map[uint64]bool {
	m := make(map[uint64]bool, len(ids))
	for _, id := range ids {
		m[id] = true
	}
	return m
}

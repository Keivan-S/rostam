// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"bytes"
	"errors"
	"fmt"
	"testing"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/raft"
)

// rn returns the underlying *raft.Node, for Raft-specific tests (loopback
// transport wiring, forced snapshots) that reach past the replicator seam. Only
// valid when the store runs the default Raft replicator.
func (s *Store) rn() *raft.Node { return s.raft.(*raft.Node) }

// inMemCluster builds n nodes using hashicorp/raft's InmemTransport pattern.
// Returns the stores; the first node is bootstrapped, others join.
//
// Because our raft.Node uses NewInmemTransport internally, we need a way to
// connect the in-memory transports across nodes. We do this by reaching into
// the Node.Transport (which implements hraft.Transport AND hraft.LoopbackTransport
// for InmemTransport) and calling Connect on each pair.
func inMemCluster(t *testing.T, n int) []*Store {
	t.Helper()

	stores := make([]*Store, n)
	for i := range n {
		reg := ops.NewRegistry()
		if err := ops.RegisterBuiltins(reg); err != nil {
			t.Fatal(err)
		}
		cc := cache.DefaultConfig()
		cc.NumShards = 1
		cfg := Config{
			NodeID:             fmt.Sprintf("node%d", i+1),
			DataDir:            t.TempDir(),
			Cache:              cc,
			Ops:                reg,
			Bootstrap:          i == 0,
			SnapshotIntervalMs: 0, // disable periodic snapshots in tests
			SnapshotThreshold:  10_000,
			// 20x tighter than the production default (1000ms) to keep tests
			// fast, which only holds on an unloaded box: on a 2-core CI runner
			// the leader can miss a 100ms election timeout simply by not being
			// scheduled, and followers then call a spurious election that kills
			// any in-flight commit. Scale with the core budget so the timings
			// stay aggressive locally but survive a contended runner.
			RaftHeartbeatMs: int(cpuScaled(50*time.Millisecond) / time.Millisecond),
			RaftElectionMs:  int(cpuScaled(100*time.Millisecond) / time.Millisecond),
			NoSync:          true,
		}
		s, err := New(cfg)
		if err != nil {
			t.Fatalf("node %d: %v", i, err)
		}
		t.Cleanup(func() { _ = s.Close() })
		stores[i] = s
	}

	// Connect every pair of in-memory transports.
	for i := range n {
		ti, ok := stores[i].rn().Transport.(hraft.LoopbackTransport)
		if !ok {
			t.Fatalf("node %d transport is not loopback (got %T)", i, stores[i].rn().Transport)
		}
		for j := range n {
			if i == j {
				continue
			}
			tj, ok := stores[j].rn().Transport.(hraft.LoopbackTransport)
			if !ok {
				t.Fatalf("node %d transport is not loopback", j)
			}
			ti.Connect(tj.LocalAddr(), tj)
		}
	}

	// Wait for leader on node 0.
	leader := stores[0]
	deadline := time.Now().Add(cpuScaled(3 * time.Second))
	for time.Now().Before(deadline) {
		if leader.IsLeader() {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !leader.IsLeader() {
		t.Fatalf("node 0 never became leader")
	}

	// Add the other nodes as voters.
	for i := 1; i < n; i++ {
		addr := stores[i].rn().Transport.LocalAddr()
		if err := leader.raft.AddVoter(stores[i].cfg.NodeID, string(addr), 0, cpuScaled(3*time.Second)); err != nil {
			t.Fatalf("AddVoter node %d: %v", i, err)
		}
	}

	// Wait for all followers to catch up.
	deadline = time.Now().Add(cpuScaled(3 * time.Second))
	for time.Now().Before(deadline) {
		ok := true
		for i := 1; i < n; i++ {
			if stores[i].LeaderAddr() != string(leader.rn().Transport.LocalAddr()) {
				ok = false
				break
			}
		}
		if ok {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}

	return stores
}

// putViaLeader writes through whichever node currently holds leadership,
// retrying transient churn. Losing leadership mid-commit (hraft.ErrLeadershipLost)
// or racing a just-completed election (ErrNotLeader) says nothing about the
// write path's soundness — only that this attempt did not land — so retry rather
// than fail. Any other error is a real defect and fails immediately.
func putViaLeader(t *testing.T, cluster []*Store, k, v []byte) {
	t.Helper()
	deadline := time.Now().Add(cpuScaled(5 * time.Second))
	var lastErr error
	for time.Now().Before(deadline) {
		for _, s := range cluster {
			if !s.IsLeader() {
				continue
			}
			err := s.Put(k, v, 0)
			if err == nil {
				return
			}
			if !errors.Is(err, hraft.ErrLeadershipLost) && !errors.Is(err, ErrNotLeader) {
				t.Fatalf("Put %s: %v", k, err)
			}
			lastErr = err
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("Put %s: no leader accepted the write before deadline (last error: %v)", k, lastErr)
}

func TestClusterWritesReplicate(t *testing.T) {
	cluster := inMemCluster(t, 3)

	// Write 50 keys via the leader.
	for i := range 50 {
		k := fmt.Appendf(nil, "k%03d", i)
		v := fmt.Appendf(nil, "v%03d", i)
		putViaLeader(t, cluster, k, v)
	}

	// Followers should eventually see all writes.
	for nodeIdx, s := range cluster {
		deadline := time.Now().Add(cpuScaled(2 * time.Second))
		for time.Now().Before(deadline) {
			got, _ := s.Get([]byte("k049"))
			if bytes.Equal(got, []byte("v049")) {
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
		for i := range 50 {
			k := fmt.Appendf(nil, "k%03d", i)
			want := fmt.Appendf(nil, "v%03d", i)
			got, err := s.Get(k)
			if err != nil {
				t.Fatalf("node %d Get %s: %v", nodeIdx, k, err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("node %d Get %s = %q, want %q", nodeIdx, k, got, want)
			}
		}
	}
}

func TestClusterFollowerRejectsWrite(t *testing.T) {
	cluster := inMemCluster(t, 3)
	// cluster[0] is leader; cluster[1] is a follower.
	follower := cluster[1]
	if follower.IsLeader() {
		t.Skip("test requires non-leader; cluster[1] happens to be leader")
	}
	err := follower.Put([]byte("k"), []byte("v"), 0)
	if !errors.Is(err, ErrNotLeader) {
		t.Fatalf("follower Put: err = %v, want ErrNotLeader", err)
	}
}

func TestClusterLeaderFailoverAndContinuedProgress(t *testing.T) {
	cluster := inMemCluster(t, 3)
	originalLeader := cluster[0]

	// Write something so all nodes are caught up.
	if err := originalLeader.Put([]byte("seed"), []byte("v"), 0); err != nil {
		t.Fatal(err)
	}
	time.Sleep(200 * time.Millisecond)

	// Disconnect the original leader from peers (force election).
	t1 := originalLeader.rn().Transport.(hraft.LoopbackTransport)
	for i := 1; i < 3; i++ {
		ti := cluster[i].rn().Transport.(hraft.LoopbackTransport)
		t1.Disconnect(ti.LocalAddr())
		ti.Disconnect(t1.LocalAddr())
	}

	// A new leader should emerge among cluster[1..].
	var newLeader *Store
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		for i := 1; i < 3; i++ {
			if cluster[i].IsLeader() {
				newLeader = cluster[i]
				break
			}
		}
		if newLeader != nil {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if newLeader == nil {
		t.Fatal("no new leader within 5s after partition")
	}

	// New leader can accept writes.
	if err := newLeader.Put([]byte("postfailover"), []byte("v"), 0); err != nil {
		t.Fatalf("new leader Put: %v", err)
	}
}

func TestClusterSnapshotAndRestore(t *testing.T) {
	cluster := inMemCluster(t, 3)
	leader := cluster[0]

	for i := range 100 {
		k := fmt.Appendf(nil, "k%03d", i)
		if err := leader.Put(k, []byte("v"), 0); err != nil {
			t.Fatalf("Put: %v", err)
		}
	}
	time.Sleep(200 * time.Millisecond)

	// Force a snapshot on the leader.
	if err := leader.rn().Snapshot(); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	// Add a fourth node that must catch up via the snapshot.
	reg := ops.NewRegistry()
	_ = ops.RegisterBuiltins(reg)
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	cfg := Config{
		NodeID: "node4", DataDir: t.TempDir(),
		Cache: cc, Ops: reg,
		RaftHeartbeatMs: 50, RaftElectionMs: 100, NoSync: true,
	}
	s4, err := New(cfg)
	if err != nil {
		t.Fatalf("node4 New: %v", err)
	}
	defer func() { _ = s4.Close() }()

	// Connect node4 to all others.
	t4 := s4.rn().Transport.(hraft.LoopbackTransport)
	for _, s := range cluster {
		ti := s.rn().Transport.(hraft.LoopbackTransport)
		t4.Connect(ti.LocalAddr(), ti)
		ti.Connect(t4.LocalAddr(), t4)
	}

	if err := leader.raft.AddVoter("node4", string(t4.LocalAddr()), 0, 5*time.Second); err != nil {
		t.Fatalf("AddVoter node4: %v", err)
	}

	// Wait for node4 to catch up.
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		got, err := s4.Get([]byte("k099"))
		if err == nil && bytes.Equal(got, []byte("v")) {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	got, err := s4.Get([]byte("k099"))
	if err != nil || !bytes.Equal(got, []byte("v")) {
		t.Fatalf("node4 did not catch up: got=%q err=%v", got, err)
	}

	// Sanity: node4 sees a sampling of earlier keys.
	for i := range 100 {
		if i%10 != 0 {
			continue
		}
		k := fmt.Appendf(nil, "k%03d", i)
		got, err := s4.Get(k)
		if err != nil || !bytes.Equal(got, []byte("v")) {
			t.Fatalf("node4 missing key %s: got=%q err=%v", k, got, err)
		}
	}

	// Keep raft import live.
	_ = raft.Config{}
}

// SPDX-License-Identifier: Apache-2.0

// Package raft wraps hashicorp/raft with a Rostam-friendly facade. It owns
// the purpose-built logstore (raft/logstore, replacing bbolt), file snapshot
// store, and TCP/in-memory transport wiring. Higher layers pass in a raft.FSM
// and a Config; everything else is internal.
package raft

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/raft/logstore"
)

// ErrNotLeader is returned by Apply when this node is not the Raft leader.
var ErrNotLeader = errors.New("raft: not the leader")

// ErrNoSnapshot is returned by BackupSnapshot when the shard has no stored
// snapshot to copy — a shard that has applied nothing yet (hraft reports
// ErrNothingNewToSnapshot for the forced snapshot and the snapshot store is
// empty). The caller treats it as "nothing to back up for this shard".
var ErrNoSnapshot = errors.New("raft: no snapshot available")

// Config describes how to construct a Node.
type Config struct {
	// NodeID is the local Raft server ID.
	NodeID string

	// DataDir is the directory for raft.db and snapshots.
	DataDir string

	// BindAddr is the listen address for the TCP transport. If empty,
	// an in-memory transport is used (suitable for tests). Ignored when
	// StreamLayer is non-nil.
	BindAddr string

	// AdvertiseAddr is what peers see. Defaults to BindAddr. Ignored
	// when StreamLayer is non-nil.
	AdvertiseAddr string

	// StreamLayer (optional) overrides the default transport selection.
	// When set, NewNode wraps it in raft.NewNetworkTransport. Used by
	// cluster.Node to share a single TCP listener across many Raft
	// groups via raft/mux.
	StreamLayer hraft.StreamLayer

	// Transport (optional) is a fully-built transport that takes precedence
	// over StreamLayer/BindAddr. Used by cluster.Node's fabric path to pass a
	// per-group facade over the shared multiplexed batching transport
	// (raft/fabric). When non-nil, buildTransport returns it directly.
	Transport hraft.Transport

	// Peers (optional) overrides the default single-self bootstrap voter
	// list. When non-empty AND Bootstrap is true, BootstrapCluster is
	// called with these servers.
	Peers []hraft.Server

	// Bootstrap, when true and no existing state is found, starts as a
	// single-node cluster (or a multi-voter cluster when Peers is set).
	Bootstrap bool

	HeartbeatMs        int
	ElectionMs         int
	SnapshotIntervalMs int
	SnapshotThreshold  uint64

	// NoSync skips fsync on the WAL log store: page-cache only, still on disk,
	// survives a clean restart, loses the tail on power loss. Faster.
	NoSync bool

	// LogLevel controls how loud the embedded hashicorp/raft is:
	// "TRACE"|"DEBUG"|"INFO"|"WARN"|"ERROR"|"OFF", any case. Empty means INFO.
	//
	// hashicorp's own DefaultConfig picks DEBUG, which is the wrong default for
	// a library: every election, pre-vote and configuration change is printed,
	// and `go test` merges the test binary's stdout and stderr into one stream,
	// so in any suite that builds many small clusters the raft chatter lands
	// interleaved with the results. `make bench` was unreadable for exactly this
	// reason. INFO keeps what an operator wants (leadership changes, snapshot
	// installs) and drops the per-tick noise; a benchmark or test that builds
	// clusters in a loop should pass "ERROR" or "OFF".
	LogLevel string

	// LogOutput is where those lines go. nil means os.Stderr, matching
	// hashicorp/raft's own fallback. Set it to io.Discard to silence raft
	// completely without also silencing whatever else shares stderr.
	LogOutput io.Writer

	// VolatileLog replaces the on-disk WAL with a pure in-memory log+stable store
	// (logstore.Mem): zero write() syscalls on the replication hot path.
	// Durability comes ENTIRELY from replication (a majority holds every acked
	// write in memory), so it is only sound in a replicated cluster. SAFETY
	// CONTRACT (logstore/mem.go): a node that loses its volatile store to a crash
	// MUST rejoin the cluster as a FRESH member (catch up from a leader snapshot)
	// and never resume in place — its lost StableStore (currentTerm/votedFor)
	// would otherwise let it double-vote and break Raft safety. Overrides NoSync.
	VolatileLog bool
}

// raftLogStore is the combined LogStore + StableStore raft needs, plus Close.
// logstore.WAL satisfies it (and so does logstore.Mem for the ephemeral case).
type raftLogStore interface {
	hraft.LogStore
	hraft.StableStore
	Close() error
}

// Node bundles the raft instance with its transport and stores so a single
// Shutdown call releases everything.
type Node struct {
	raft      *hraft.Raft
	Transport hraft.Transport
	logStore  raftLogStore
	snapStore hraft.SnapshotStore
}

// NewNode constructs a Node with the given FSM.
func NewNode(cfg Config, fsm hraft.FSM) (*Node, error) {
	if err := os.MkdirAll(cfg.DataDir, 0o750); err != nil { //nolint:gosec // DataDir is caller-controlled
		return nil, fmt.Errorf("raft: mkdir %s: %w", cfg.DataDir, err)
	}

	// The raft log store is our purpose-built segmented WAL (replaces bbolt: no
	// B+tree, no msgpack per entry). NoSync controls only crash durability — fsync
	// per batch (default) vs page-cache only (fast; still on disk, survives a
	// clean restart, loses the tail on power loss). Both survive a warm restart.
	// VolatileLog goes further: a pure in-memory store (no write() syscall at all;
	// durability from replication only — see Config.VolatileLog and its safety
	// contract). It takes precedence over NoSync.
	var logStore raftLogStore
	if cfg.VolatileLog {
		logStore = logstore.NewMem()
	} else {
		wal, err := logstore.OpenWAL(filepath.Join(cfg.DataDir, "raftlog"), !cfg.NoSync)
		if err != nil {
			return nil, fmt.Errorf("raft: open wal: %w", err)
		}
		logStore = wal
	}

	snapStore, err := hraft.NewFileSnapshotStore(cfg.DataDir, 2, os.Stderr)
	if err != nil {
		_ = logStore.Close() //nolint:errcheck,gosec // best-effort cleanup on error path
		return nil, fmt.Errorf("raft: snapshot store: %w", err)
	}

	transport, err := buildTransport(cfg)
	if err != nil {
		_ = logStore.Close() //nolint:errcheck,gosec // best-effort cleanup on error path
		return nil, fmt.Errorf("raft: transport: %w", err)
	}

	raftCfg := hraft.DefaultConfig()
	// Override hashicorp's DEBUG default before anything else touches the config
	// — see Config.LogLevel for why a library must not ship at DEBUG.
	raftCfg.LogLevel = cfg.LogLevel
	if raftCfg.LogLevel == "" {
		raftCfg.LogLevel = "INFO"
	}
	raftCfg.LogOutput = cfg.LogOutput
	if raftCfg.LogOutput == nil {
		raftCfg.LogOutput = os.Stderr
	}
	raftCfg.LocalID = hraft.ServerID(cfg.NodeID)
	if cfg.HeartbeatMs > 0 {
		raftCfg.HeartbeatTimeout = time.Duration(cfg.HeartbeatMs) * time.Millisecond
	}
	if cfg.ElectionMs > 0 {
		raftCfg.ElectionTimeout = time.Duration(cfg.ElectionMs) * time.Millisecond
	}
	if cfg.SnapshotIntervalMs > 0 {
		raftCfg.SnapshotInterval = time.Duration(cfg.SnapshotIntervalMs) * time.Millisecond
	}
	if cfg.SnapshotThreshold > 0 {
		raftCfg.SnapshotThreshold = cfg.SnapshotThreshold
	}
	if cfg.HeartbeatMs > 0 {
		raftCfg.LeaderLeaseTimeout = raftCfg.HeartbeatTimeout
	}
	// EXPERIMENT: buffer applyCh so raft.Apply deposits without a synchronous
	// rendezvous handoff to the single leader loop, and more entries accumulate
	// per group-commit dispatch (bigger batches, fewer context switches).
	raftCfg.BatchApplyCh = true

	r, err := hraft.NewRaft(raftCfg, fsm, logStore, logStore, snapStore, transport)
	if err != nil {
		_ = logStore.Close() //nolint:errcheck,gosec // best-effort cleanup on error path
		return nil, fmt.Errorf("raft: NewRaft: %w", err)
	}

	// bootstrapServers is retained so a group can still be FORMED after
	// construction (BootstrapGroup). Formation is a control-plane decision that
	// may not be known yet when the node is built — see cluster's shard formation
	// driver and cluster.State.ShardFormer.
	if cfg.Bootstrap {
		has, err := hraft.HasExistingState(logStore, logStore, snapStore)
		if err != nil {
			_ = r.Shutdown().Error() //nolint:errcheck,gosec // best-effort cleanup on error path
			_ = logStore.Close()     //nolint:errcheck,gosec // best-effort cleanup on error path
			return nil, fmt.Errorf("raft: HasExistingState: %w", err)
		}
		if !has {
			servers := cfg.Peers
			if len(servers) == 0 {
				servers = []hraft.Server{{
					ID:      hraft.ServerID(cfg.NodeID),
					Address: transport.LocalAddr(),
				}}
			}
			conf := hraft.Configuration{Servers: servers}
			if f := r.BootstrapCluster(conf); f.Error() != nil && !errors.Is(f.Error(), hraft.ErrCantBootstrap) {
				_ = r.Shutdown().Error() //nolint:errcheck,gosec // best-effort cleanup on error path
				_ = logStore.Close()     //nolint:errcheck,gosec // best-effort cleanup on error path
				return nil, fmt.Errorf("raft: bootstrap: %w", f.Error())
			}
		}
	}

	return &Node{
		raft:      r,
		Transport: transport,
		logStore:  logStore,
		snapStore: snapStore,
	}, nil
}

// BootstrapGroup forms this Raft group AFTER construction, with servers as the
// initial configuration. It exists because whether a node should form a group is
// not always knowable when the node is built: the decision belongs to a control
// plane that may not have published it yet (cluster.State.ShardFormer).
//
// Idempotent and safe to call repeatedly from a retry loop: it no-ops when the
// group already has persisted state, and tolerates hraft.ErrCantBootstrap (the
// race where state appeared between the check and the call). It does NOT no-op
// on "a leader already exists" — a caller must not use this to re-form a live
// group; the control plane's write-once designation is what guarantees exactly
// one former per group.
func (n *Node) BootstrapGroup(servers []hraft.Server) error {
	if len(servers) == 0 {
		return fmt.Errorf("raft: bootstrap group: empty server list")
	}
	has, err := hraft.HasExistingState(n.logStore, n.logStore, n.snapStore)
	if err != nil {
		return fmt.Errorf("raft: bootstrap group HasExistingState: %w", err)
	}
	if has {
		return nil // already formed (or already replaying a leader's log)
	}
	f := n.raft.BootstrapCluster(hraft.Configuration{Servers: servers})
	if err := f.Error(); err != nil && !errors.Is(err, hraft.ErrCantBootstrap) {
		return fmt.Errorf("raft: bootstrap group: %w", err)
	}
	return nil
}

// buildTransport selects, in order of precedence: an injected Transport, an
// injected StreamLayer (wrapped in NetworkTransport), TCP (if BindAddr set), or
// in-memory.
func buildTransport(cfg Config) (hraft.Transport, error) {
	if cfg.Transport != nil {
		return cfg.Transport, nil
	}
	if cfg.StreamLayer != nil {
		return hraft.NewNetworkTransport(cfg.StreamLayer, 3, 10*time.Second, os.Stderr), nil
	}
	if cfg.BindAddr == "" {
		_, t := hraft.NewInmemTransport(hraft.ServerAddress(cfg.NodeID))
		return t, nil
	}
	advertise := cfg.AdvertiseAddr
	if advertise == "" {
		advertise = cfg.BindAddr
	}
	tcpAddr, err := net.ResolveTCPAddr("tcp", advertise)
	if err != nil {
		return nil, fmt.Errorf("resolve advertise addr: %w", err)
	}
	return hraft.NewTCPTransport(cfg.BindAddr, tcpAddr, 3, 10*time.Second, os.Stderr)
}

// Shutdown gracefully tears down the Raft node.
func (n *Node) Shutdown() error {
	if err := n.raft.Shutdown().Error(); err != nil {
		return fmt.Errorf("raft: shutdown raft: %w", err)
	}
	if tr, ok := n.Transport.(interface{ Close() error }); ok {
		_ = tr.Close()
	}
	return n.logStore.Close()
}

// Apply submits a log entry. Returns the FSM's Response and any Raft error.
// If the node is not the leader, returns ErrNotLeader (mapped from
// hraft.ErrNotLeader).
func (n *Node) Apply(data []byte, timeout time.Duration) (any, error) {
	resp, _, err := n.ApplyIndexed(data, timeout)
	return resp, err
}

// ApplyIndexed is Apply but also returns the committed log index of the entry
// (hraft.ApplyFuture.Index(), valid once Error() reports success). The index is
// the basis for the write-consistency barrier's catch-up target. Apply is a thin
// wrapper that discards it, so existing callers are byte-for-byte unaffected.
func (n *Node) ApplyIndexed(data []byte, timeout time.Duration) (resp any, index uint64, err error) {
	f := n.raft.Apply(data, timeout)
	if err := f.Error(); err != nil {
		if errors.Is(err, hraft.ErrNotLeader) {
			return nil, 0, ErrNotLeader
		}
		return nil, 0, err
	}
	return f.Response(), f.Index(), nil
}

// IsLeader returns true if this node is the current Raft leader.
func (n *Node) IsLeader() bool {
	return n.raft.State() == hraft.Leader
}

// LeaderAddr returns the address of the current Raft leader, or "" if unknown.
func (n *Node) LeaderAddr() string {
	addr, _ := n.raft.LeaderWithID()
	return string(addr)
}

// Stats returns the underlying hashicorp/raft stats map (opaque per their docs).
func (n *Node) Stats() map[string]string {
	return n.raft.Stats()
}

// Snapshot forces a Raft snapshot now and waits for it to complete.
// Returns the future's error.
func (n *Node) Snapshot() error {
	return n.raft.Snapshot().Error()
}

// BackupSnapshot forces a fresh point-in-time FSM snapshot and returns the
// newest stored snapshot's raw FSM bytes plus its raft log index. It is the
// Raft-mode backup crux: hashicorp/raft runs fsm.Snapshot() on the FSM apply
// goroutine (never concurrently with Apply — see runFSM), so the serialized
// cache+vector blob it produces is torn-free and corresponds to a single log
// index. We then read that blob back out of the FileSnapshotStore verbatim
// (FileSnapshotStore.Open returns exactly the bytes Persist wrote), so the
// returned data is the same RSST blob restoreSnapshot consumes.
//
// A shard that has applied NOTHING yet makes the forced snapshot return
// hraft.ErrNothingNewToSnapshot AND leaves the store empty; that surfaces as
// ErrNoSnapshot so the caller can skip an empty shard rather than emit a torn
// out-of-band serialization. Any other force error is fatal (we do not fall back
// to a possibly-stale stored snapshot silently).
func (n *Node) BackupSnapshot() (data []byte, index uint64, err error) {
	if serr := n.raft.Snapshot().Error(); serr != nil && !errors.Is(serr, hraft.ErrNothingNewToSnapshot) {
		return nil, 0, fmt.Errorf("raft: force snapshot: %w", serr)
	}
	list, err := n.snapStore.List()
	if err != nil {
		return nil, 0, fmt.Errorf("raft: list snapshots: %w", err)
	}
	if len(list) == 0 {
		return nil, 0, ErrNoSnapshot
	}
	// List returns snapshots newest-first (descending index). Take the freshest.
	meta, rc, err := n.snapStore.Open(list[0].ID)
	if err != nil {
		return nil, 0, fmt.Errorf("raft: open snapshot %s: %w", list[0].ID, err)
	}
	defer func() { _ = rc.Close() }()
	data, err = io.ReadAll(rc)
	if err != nil {
		return nil, 0, fmt.Errorf("raft: read snapshot %s: %w", list[0].ID, err)
	}
	return data, meta.Index, nil
}

// Restore installs an external FSM snapshot into this (leader) node and
// replicates it to followers via the install-snapshot path — hashicorp/raft's
// disaster-recovery primitive (see (*Raft).Restore). It must be called on the
// leader of a FRESH group: the leader takes on the snapshot state and then
// commits ahead of its followers until they fault-and-install, which is exactly
// the DR-into-a-fresh-cluster shape. index is the snapshot's recorded applied
// index (used only to leave a log hole above it); the CURRENT raft configuration
// is used, not any config from the snapshot, so a same-topology restore lands on
// the freshly-bootstrapped voter set.
func (n *Node) Restore(data []byte, index uint64, timeout time.Duration) error {
	meta := &hraft.SnapshotMeta{
		Version: 1,
		Index:   index,
		Term:    1,
		Size:    int64(len(data)),
	}
	if err := n.raft.Restore(meta, bytes.NewReader(data), timeout); err != nil {
		if errors.Is(err, hraft.ErrNotLeader) || errors.Is(err, hraft.ErrLeadershipLost) {
			return ErrNotLeader
		}
		return fmt.Errorf("raft: restore snapshot: %w", err)
	}
	return nil
}

// AddVoter adds a new voter to the cluster. prevIndex of 0 means "any".
func (n *Node) AddVoter(id, addr string, prevIndex uint64, timeout time.Duration) error {
	return n.raft.AddVoter(hraft.ServerID(id), hraft.ServerAddress(addr), prevIndex, timeout).Error()
}

// RemoveServer removes a server (voter) from the cluster configuration.
// prevIndex of 0 means "any". Must be called on the leader.
func (n *Node) RemoveServer(id string, prevIndex uint64, timeout time.Duration) error {
	return n.raft.RemoveServer(hraft.ServerID(id), prevIndex, timeout).Error()
}

// LastIndex returns the last index in the stable log (committed or not).
func (n *Node) LastIndex() uint64 { return n.raft.LastIndex() }

// AppliedIndex returns the last log index applied to the FSM. Online
// rebalancing polls a joining replica's AppliedIndex against the leader's
// LastIndex to confirm catch-up before removing the old owner.
func (n *Node) AppliedIndex() uint64 { return n.raft.AppliedIndex() }

// NumServers returns the count of servers in the LATEST (possibly-uncommitted)
// Raft configuration, and ok=false if it could not be read. It is safe and cheap
// to call from any goroutine — including the FSM apply goroutine: in hashicorp/raft
// v1.7.3 GetConfiguration builds an already-resolved future from
// getLatestConfiguration(), which is a plain atomic.Value load of a config clone
// (no main-loop round trip, no lock shared with Apply). The count reflects live
// membership, so a shard that joined an RF>1 group via AddVoter reports >1 as soon
// as it has learned the configuration — used by the FSM's fail-closed apply gate
// (see shard.fsm.isReplicated). A fresh joiner that has not yet learned any
// configuration reports (0, true).
func (n *Node) NumServers() (count int, ok bool) {
	f := n.raft.GetConfiguration()
	if f.Error() != nil {
		return 0, false
	}
	return len(f.Configuration().Servers), true
}

// VerifyLeader confirms this node is STILL the leader via a quorum heartbeat
// round-trip (hraft's readIndex primitive: "prevent returning stale data from
// the FSM after the peer has lost leadership"). It blocks until the heartbeat
// resolves or the node steps down, bounded by the lease/election timeout. On a
// single-node cluster (quorum==1) it resolves immediately with no round-trip; a
// follower returns ErrNotLeader immediately. Returns ErrNotLeader (mapped from
// hraft.ErrNotLeader) if this node is not / no longer the leader; any other
// transport/raft error is propagated as-is.
func (n *Node) VerifyLeader() error {
	err := n.raft.VerifyLeader().Error()
	// Both outcomes mean "not / no longer the leader": ErrNotLeader (a heartbeat
	// reply revealed a newer term so we stepped down) and ErrLeadershipLost (the
	// verify future was drained when the leader loop exited, e.g. a partitioned
	// leader losing quorum). Map BOTH to ErrNotLeader so the readIndex barrier
	// returns NotLeaderError and the caller re-routes to the real leader (symmetric
	// with Barrier below). Returning the raw ErrLeadershipLost would fail the read
	// instead of re-routing.
	if errors.Is(err, hraft.ErrNotLeader) || errors.Is(err, hraft.ErrLeadershipLost) {
		return ErrNotLeader
	}
	return err
}

// CommitIndex returns the highest log index known to be committed (an atomic
// read of hraft's committed index). Paired with VerifyLeader it is the readIndex
// barrier target: after a confirmed-leadership VerifyLeader, capturing CommitIndex
// and waiting for the TRUE FSM-applied index to reach it serves a linearizable
// read. Distinct from AppliedIndex, which reports what hraft has DISPATCHED to the
// FSM goroutine (and lags the actually-applied state).
func (n *Node) CommitIndex() uint64 { return n.raft.CommitIndex() }

// Barrier issues a no-op LogBarrier through the Raft log and blocks until the FSM
// has applied it — which guarantees the FSM has also applied every entry committed
// before it (commands AND non-command no-op/config entries), in order. Only a
// leader can commit the barrier, so a non-leader returns ErrNotLeader. This is the
// correct catch-up primitive for a linearizable read when the FSM's command
// frontier lags the commit index because of non-command entries (a leader's
// election no-op never reaches fsm.Apply, so CommitIndex can permanently exceed the
// command frontier on an idle leader — a plain "wait fsm.AppliedIndex >= ci" would
// hang there). It costs one log entry (fsync + replicate); callers use it only on
// the slow path, after a cheap VerifyLeader + a fsm.AppliedIndex()>=ci fast path.
func (n *Node) Barrier(timeout time.Duration) error {
	err := n.raft.Barrier(timeout).Error()
	if errors.Is(err, hraft.ErrNotLeader) || errors.Is(err, hraft.ErrLeadershipLost) {
		return ErrNotLeader
	}
	return err
}

// LeadershipTransferToServer asks the current leader to hand leadership to the
// given server. Must be called on the leader. Used by online rebalancing to
// move leadership off a node before it is removed from a shard group, so the
// old leader never has to remove itself.
func (n *Node) LeadershipTransferToServer(id, addr string) error {
	return n.raft.LeadershipTransferToServer(hraft.ServerID(id), hraft.ServerAddress(addr)).Error()
}

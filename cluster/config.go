// SPDX-License-Identifier: Apache-2.0

// Package cluster provides the multi-shard fanout layer for Rostam.
package cluster

import (
	"crypto/tls"
	"errors"
	"fmt"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
	"github.com/rostamlabs/rostam/shard/pbisr"
)

// Config governs a cluster.Node.
type Config struct {
	// NodeID is the base identifier; per-shard ID becomes
	// "<NodeID>-shard-<NNNN>".
	NodeID string

	// DataDir is the base directory; per-shard subdir is
	// "<DataDir>/shard-<NNNN>".
	DataDir string

	// NumShards is the number of independent shards. Valid range
	// [1, 65536]. Default 64.
	//
	// Each shard is its own Raft group, so this is also the per-node Raft-group
	// count under full replication. The default is 64 (down from 256): on
	// commodity (<=8-core) nodes fewer groups mean less heartbeat/election
	// overhead and materially higher write throughput (measured ~1.8x at 64 vs
	// 256 on a 2-core node), while 64 still gives ample write fan-out and room to
	// scale out. Raise it (e.g. 256) for deployments on many large-core nodes
	// that genuinely need the extra shard parallelism. NumShards is fixed at
	// cluster creation (it is the key-routing modulus), so choose up front.
	NumShards int

	// Bootstrap controls whether each sub-store's Raft is bootstrapped
	// as a single-node cluster. For the single-node path, this is
	// true on first start. DefaultConfig returns false; callers must
	// set this explicitly when they need bootstrap.
	Bootstrap bool

	// ReplicationFactor is how many nodes host (replicate) each data shard.
	// 0 (default) or >= len(Peers) means full replication: every node hosts
	// every shard. A value in [1, len(Peers)) partitions shards across the
	// cluster — each node stores only the shards placed on it, and a node that
	// receives an op for a shard it does not host forwards it to an owner.
	ReplicationFactor int

	// BringupElectionConcurrency is the max number of shard Raft groups whose
	// initial (cold) leader election runs concurrently during multi-node bringup;
	// 0 = auto (runtime.GOMAXPROCS). Limits the cold-election thundering herd on
	// CPU-constrained hosts; a well-resourced host (concurrency >= shard count)
	// brings all groups up at once, unchanged. Bringup TIMING only — never affects
	// election/quorum/log semantics. Applies to the multi-node path only
	// (single-node 1-voter groups elect instantly, no herd).
	//
	// Not plumbed through EmbeddedConfig/ServerConfig: the auto-default already
	// fixes the herd; a follow-up can expose it for operator tuning if needed.
	BringupElectionConcurrency int

	// ShardCfg is the per-shard shard.Config template. New() copies it
	// per shard then overrides NodeID (derived), DataDir (derived),
	// Bootstrap (taken from Config.Bootstrap, not ShardCfg.Bootstrap),
	// and Cache.NumShards (forced to 1). All other fields (Raft tuning,
	// NoSync, snapshot interval, Ops) are inherited.
	ShardCfg shard.Config

	// Ops is the registry shared across all sub-stores.
	Ops *ops.Registry

	// Peers is the static cluster member list. The first start must
	// include this node's own Peer entry (matched by NodeID). nil means
	// single-node compatibility — no multi-node Raft setup.
	Peers []Peer

	// RaftAddr is this node's multiplexed Raft transport endpoint.
	// Required when len(Peers) > 1. Optional when len(Peers) <= 1
	// (single-peer / single-node falls through to legacy path).
	RaftAddr string

	// RaftTransport selects the inter-node Raft transport for the multi-node
	// path: "" or "mux" (default) uses the per-group NetworkTransport over the
	// raft/mux shared TCP listener; "fabric" uses the multiplexed batching
	// transport (raft/fabric) that carries every group's traffic to a peer over
	// one shared connection. Flag-gated so the default path stays byte-identical
	// until fabric is promoted.
	RaftTransport string

	// WASMModules is the list of WASM UDF modules to load at startup.
	// Each entry is persisted to <DataDir>/wasm/ and compiled. Errors
	// surface from cluster.New; Validate() does not check these.
	WASMModules []WASMModuleConfig

	// WASMBlobRetention enables WASM BLOB RETIREMENT: how long a module blob that
	// nothing on this node references — a superseded version, or a
	// __wasm_blob_put__ orphan — is kept before its file is deleted.
	//
	// ZERO (THE DEFAULT) DISABLES RETIREMENT ENTIRELY. No sweeper runs, no
	// bookkeeping is kept, and nothing is ever removed, which is byte-identical to
	// the behaviour before retirement existed. Every version any group ever bound
	// stays on disk forever, and that is the safe answer.
	//
	// ###### SETTING IT IS A STATEMENT ABOUT YOUR OPERATIONS, NOT A TUNING KNOB ##
	//
	// Read cluster/wasm_blob_retire.go before setting it. The short version: no
	// LOCAL rule can be safe against a lagging replica elsewhere, because whether
	// some other replica of some group will still apply an entry under a
	// superseded version is not a function of anything this node holds — and
	// deciding it cluster-wide is cluster-wide GC, which is deliberately not
	// built. The value you set is your assertion that no replica of any group this
	// node hosts will be further behind the supersession of a module version than
	// this. Err high: hours or days, not minutes.
	//
	// If the assertion is wrong the consequence is the ordinary blob block —
	// the replica parks, names the exact fingerprint in Stats().WASMBlock, logs it
	// with escalating severity, and unblocks the moment anyone supplies the bytes
	// with a single `__wasm_blob_put__` admin call. Loud and one-call-recoverable,
	// which is the only reason the trade is offerable at all.
	//
	// It removes the FILE only; the instantiated module stays in the runtime, so
	// an invocation executing under a retired version is unaffected.
	WASMBlobRetention time.Duration

	// InternalToken is the inter-node service credential. When non-empty, every
	// forwarding/admin client this node builds to a peer (peerClient) presents it
	// as its bearer token, and the destination node's RBAC authorizer treats it as
	// the superuser service principal so an inter-node forward passes auth even
	// though the original client's scope is collection-limited. REQUIRED for a
	// cluster running with RBAC enabled: without it, a write that lands on a
	// non-leader node is forwarded with no token and the destination denies it.
	// Empty = no token on inter-node clients (correct for nil-auth / open
	// clusters). This round inter-node traffic is plaintext, so the token is the
	// only inter-node auth until inter-node mTLS lands (documented residual gap).
	InternalToken string

	// InterNodeTLS is the TLS CLIENT config used for inter-node forwarding dials.
	// When non-nil, peerClient dials each peer's client-facing port over TLS using
	// a clone of this config with ServerName set to the per-peer host (so the peer
	// server cert is verified against this config's RootCAs / SAN). nil = plaintext
	// inter-node dial (the default; zero cost when client TLS is off).
	//
	// This exists because one ServerConfig.TLSConfig wraps ALL client-facing
	// transports (incl. the TCP server port that inter-node forwarding dials), so a
	// cluster with client TLS enabled TLS-wraps the very port the plaintext forward
	// would otherwise dial. Setting InterNodeTLS makes the inter-node DIAL use TLS
	// to reach that wrapped port. AUTH is still the internal token (see above); this
	// config only provides the encrypted transport and CA-verification of the peer
	// server cert. The node client cert it may carry (for peer mTLS handshakes) has
	// no authz meaning — it only needs to be CA-signed. NEVER InsecureSkipVerify.
	InterNodeTLS *tls.Config

	// InterNodeServerTLS is the TLS SERVER config used to wrap the inter-node
	// REPLICATION listeners — the Raft transport (raft/mux or raft/fabric) and, in
	// PB mode, the primary-backup transport (shard/pbisr). It is the SERVER-side
	// counterpart of InterNodeTLS (which only wraps the outbound forwarding DIAL):
	// InterNodeTLS protected request forwarding, but the separate replication ports
	// stayed plaintext, so anyone reaching them could read or forge every tenant's
	// replicated writes. This closes that hole.
	//
	// Build it with tlsutil.ServerTLS(nodeCert, nodeKey, clusterCA, requireClientCert=true)
	// so each listener demands and CA-verifies a peer client cert (mTLS,
	// fail-closed). Combined with NodeCNAllowlist it authenticates the peer's
	// identity: after the handshake each listener checks the verified client-cert CN
	// against the allowlist before serving any replicate/Raft frame (empty allowlist
	// ⇒ any CA-valid peer, mTLS still required). nil ⇒ plaintext replication
	// listeners (the default; BYTE-IDENTICAL to today's raft-mode/no-TLS path).
	// Threaded into the mux/fabric/pbisr constructors alongside InterNodeTLS (the
	// dial side); a node that TLS-wraps its client-facing port should set BOTH so
	// the whole cluster — forwarding AND replication — is encrypted and authenticated.
	InterNodeServerTLS *tls.Config

	// NodeCNAllowlist is the OPT-IN per-node mTLS identity allowlist: the set of
	// peer cert CommonNames this node trusts as cluster members. Empty/nil = OFF =
	// BYTE-IDENTICAL to today's shared-token/shared-CA path: peerClient attaches no
	// VerifyPeerCertificate callback (a CA-signed peer is accepted as before). When
	// non-empty, peerClient additionally pins each dialed peer's verified leaf-cert
	// CN to this set (a peer whose CN is absent fails the handshake — fail-closed),
	// and the destination node's authorizer additionally requires the internal-token
	// caller's verified ClientCN to be allowlisted (RBACOptions.NodeCNAllowlist).
	// REQUIRES inter-node client-cert TLS (InterNodeTLS presenting a node client
	// cert) — else there is no peer CN to verify (validated fail-loud at startup).
	NodeCNAllowlist map[string]bool

	// ReplicationMode selects the cluster-level data-plane replication
	// engine: "" or "raft" (default) uses per-shard Raft groups, unchanged.
	// "pb" selects primary-backup/ISR replication (shard.ReplicationModePB)
	// for every shard. Mirrors shard.Config.ReplicationMode's values for
	// consistency; per-shard mode-specific wiring (e.g. requiring Peer.PBAddr)
	// happens once the mode is known, not here.
	ReplicationMode string

	// MinISR is the minimum in-sync-replica count required by "pb" mode
	// (must be >= 1 when ReplicationMode == "pb"). Unused in "raft"/"" mode.
	MinISR int

	// PBCommitPrimary selects the commit-on-primary durability contract for
	// "pb" mode: the primary acks a write on local apply and replicates to
	// backups asynchronously. It is a DURABILITY DOWNGRADE — an acked write can
	// be lost if the primary dies before a backup received it (Aerospike's
	// commit-master posture). Default false = wait for the full ISR (no acked
	// loss while any ISR member survives). Unused in "raft"/"" mode.
	PBCommitPrimary bool

	// PBAutoFailover enables automatic primary-backup failover: each node
	// runs a primary-liveness beacon (committing OpShardLeaseRenew), and the meta
	// leader runs a failover ticker that promotes an ISR survivor when a primary
	// goes silent past the failover timeout. DEFAULT FALSE — and deliberately so:
	// with it OFF, NEITHER the beacon NOR the ticker goroutine starts, so the
	// meta-Raft log carries ZERO OpShardLeaseRenew entries and the ticker commits
	// ZERO OpSetShardEpoch bumps ⇒ the replicated state and its snapshots are
	// BYTE-IDENTICAL to the pre-Plan-4 static cluster. That is what
	// -pb-auto-failover=false buys: an explicitly STATIC cluster.
	//
	// This field's Go zero value is false, but rostam-server defaults
	// -pb-auto-failover to TRUE: both pre-default-on gates now pass (the crash-stop
	// no-acked-loss gate and the network-partition no-double-primary gate — see
	// shard/pbisr/DESIGN.md), and a PB shard without failover
	// stays DOWN on primary loss until an operator intervenes, which is the worse
	// default. A direct library embedder (cluster.Config / EmbeddedConfig) must
	// therefore opt in explicitly. Unused in "raft"/"" mode.
	PBAutoFailover bool

	// PBLeaseTTLMs, PBMetaContactStalenessMs, PBFailoverTimeoutMs, PBRenewIntervalMs
	// override the PB failover timings (all in milliseconds; 0 ⇒ the built-in
	// default). They exist mainly so the no-acked-loss failover test can shrink the
	// real (non-fake) clocks to run a kill-under-load in seconds. The HONOR RULE is
	// asserted at construction (see Validate): PBFailoverTimeout MUST exceed
	// PBLeaseTTL + PBMetaContactStaleness + PBRenewInterval (+ a one-tick margin) so a
	// meta-partitioned old primary has provably self-fenced before the leader names a
	// replacement — the renewInterval term matters because the leader's last observed
	// beacon can be a full interval old at partition time.
	PBLeaseTTLMs             int // primary-lease validity window (default 5s)
	PBMetaContactStalenessMs int // follower quorum-contact freshness bound (default 2s)
	PBFailoverTimeoutMs      int // silent-primary promotion threshold (default 10s)
	PBRenewIntervalMs        int // beacon commit interval (default 1s)

	// PBShrinkThreshold is the per-backup CONSECUTIVE replication-failure count at
	// which the shrink driver treats a backup as dead and requests its
	// removal from the ISR (never below MinISR). 0 ⇒ the built-in default. It MUST
	// be well above one RTT's worth of transient blips so a momentary hiccup never
	// shrinks a healthy ISR; the shrink driver runs only when PBAutoFailover is on.
	PBShrinkThreshold int

	// PBGrowTickMs overrides how often the grow driver evaluates each owned
	// primary for an under-replicated ISR (a placement owner missing from the ISR)
	// to catch it up and re-add it. 0 ⇒ the built-in default (pbGrowTickInterval).
	// The grow driver runs only when PBAutoFailover is on.
	PBGrowTickMs int

	// metaStreamLayerWrap, when non-nil, wraps ONLY the meta-Raft group's
	// per-group raft.StreamLayer just before its NetworkTransport is built over it
	// (mux path in node.go). nil in production ⇒ identity ⇒ BYTE-IDENTICAL to the
	// historical path: the meta transport is constructed over the unmodified
	// s.For(metaGroupID) layer, so no production caller can observe any difference.
	//
	// It is UNEXPORTED on purpose: it is settable only from within the cluster
	// package, i.e. by in-package tests. Its sole use is the no-double-primary
	// failover gate (pb_partition_test.go), which splices a partition injector over
	// exactly the meta group's transport — isolating a node's META-Raft while
	// leaving its PB data path (shard/pbisr NetTransport) and client-facing server
	// untouched — to exercise the alive-but-meta-partitioned primary window that a
	// crash-stop test cannot reach. In PB mode the mux carries no active shard-raft
	// groups, so wrapping the meta group's layer isolates exactly meta.
	metaStreamLayerWrap func(hraft.StreamLayer) hraft.StreamLayer

	// pbTransportWrap, when non-nil, wraps each PB shard's pbisr.Transport just
	// after node-ID address resolution (rebalance.go) and before it reaches the
	// engine. nil in production ⇒ identity ⇒ BYTE-IDENTICAL to the historical path.
	//
	// UNEXPORTED on purpose (settable only from in-package tests). Its sole use is
	// the shrink harness (pb_shrink_test.go), which splices a per-node drop
	// injector so replication to a chosen backup FAILS deterministically —
	// exercising the engine's per-peer failure counter and the shrink driver
	// without killing a whole node (symmetric to metaStreamLayerWrap's partition
	// injector for the failover gate).
	pbTransportWrap func(pbisr.Transport) pbisr.Transport
}

// Replication mode values for Config.ReplicationMode. Mirrors
// shard.ReplicationModeRaft / shard.ReplicationModePB.
const (
	ReplicationModeRaft = "raft"
	ReplicationModePB   = "pb"
)

// WASMModuleConfig describes a single WASM UDF module to load at startup.
// Bytes and Path are mutually exclusive: set exactly one.
type WASMModuleConfig struct {
	// Name is the op name under which the module is registered (e.g. "my_udf").
	Name string
	// Kind controls whether the op bypasses Raft (OpReadOnly) or goes through
	// it (OpReadWrite).
	Kind ops.OpKind
	// Bytes is the raw WASM binary (mutually exclusive with Path).
	Bytes []byte
	// Path is a filesystem path to the WASM binary (mutually exclusive with Bytes).
	Path string
	// ExportName is the WASM export symbol that implements the handler.
	ExportName string
	// THERE IS NO KeyExtractorHandle. Every WASM op is registered with the one
	// extractor (ops.WASMKeyExtractor), so a module's args are
	// [keyLen u16][key][payload] and the module must skip that prefix. Config is
	// node-LOCAL and enters no Raft log, so a per-node choice here would have
	// produced the routing split of ops.WASMKeyExtractorHandle with nothing
	// anywhere to catch it.
	// MaxFuel caps the WASM instruction budget (0 = use the default cap).
	MaxFuel uint64
}

// pbEffectiveTimings resolves the four PB failover timings from the config,
// substituting the package-const default for any field left 0. Centralizes the
// default logic so Validate's honor-rule assertion and the node wiring agree on
// exactly the same values.
func (c Config) pbEffectiveTimings() (leaseTTL, metaStaleness, failoverTimeout, renewInterval time.Duration) {
	leaseTTL = pbLeaseTTL
	if c.PBLeaseTTLMs > 0 {
		leaseTTL = time.Duration(c.PBLeaseTTLMs) * time.Millisecond
	}
	metaStaleness = metaContactStaleness
	if c.PBMetaContactStalenessMs > 0 {
		metaStaleness = time.Duration(c.PBMetaContactStalenessMs) * time.Millisecond
	}
	failoverTimeout = pbFailoverTimeout
	if c.PBFailoverTimeoutMs > 0 {
		failoverTimeout = time.Duration(c.PBFailoverTimeoutMs) * time.Millisecond
	}
	renewInterval = pbRenewInterval
	if c.PBRenewIntervalMs > 0 {
		renewInterval = time.Duration(c.PBRenewIntervalMs) * time.Millisecond
	}
	return leaseTTL, metaStaleness, failoverTimeout, renewInterval
}

// DefaultConfig builds a sensible Config for single-node use.
func DefaultConfig(dataDir, nodeID string, registry *ops.Registry) Config {
	return Config{
		NodeID:    nodeID,
		DataDir:   dataDir,
		NumShards: 64,
		Bootstrap: false,
		ShardCfg:  shard.DefaultConfig(dataDir, nodeID, registry),
		Ops:       registry,
	}
}

// Validate enforces invariants.
func (c Config) Validate() error {
	if c.NodeID == "" {
		return errors.New("cluster.Config: NodeID is required")
	}
	if c.DataDir == "" {
		return errors.New("cluster.Config: DataDir is required")
	}
	if c.NumShards < 1 || c.NumShards > 65536 {
		return ErrInvalidNumShards
	}
	if c.Ops == nil {
		return errors.New("cluster.Config: Ops registry is required")
	}
	switch c.ReplicationMode {
	case "", ReplicationModeRaft, ReplicationModePB:
	default:
		return fmt.Errorf("cluster.Config: unknown ReplicationMode %q (want %q or %q)",
			c.ReplicationMode, ReplicationModeRaft, ReplicationModePB)
	}
	if c.ReplicationMode == ReplicationModePB && c.MinISR < 1 {
		return fmt.Errorf("cluster.Config: ReplicationMode=pb requires MinISR >= 1 (got %d)", c.MinISR)
	}
	// HONOR RULE (no-double-primary / no-acked-loss). This guarantees a
	// meta-partitioned old primary P has provably self-fenced (its lease lapsed)
	// BEFORE the meta leader names a replacement Q.
	//
	// The bound includes a renewInterval term that the first cut omitted. The ticker
	// promotes when now − lastRenewNs > failoverTimeout, and lastRenewNs is the last
	// BEACON the leader observed for P, which at partition time τ can already be a
	// full renewInterval old. So the EARLIEST possible promotion is
	// τ + failoverTimeout − renewInterval (not τ + failoverTimeout). Meanwhile P
	// self-fences by τ + metaContactStaleness + pbLeaseTTL (it renews its lease only
	// while confirmMetaView passes). Safety (promotion-time ≥ fence-time) therefore
	// requires:
	//   failoverTimeout − renewInterval ≥ pbLeaseTTL + metaContactStaleness
	//   ⟺ failoverTimeout ≥ pbLeaseTTL + metaContactStaleness + renewInterval.
	// We assert the STRICTLY-greater form and additionally budget the up-to-one-tick
	// detection delay (pbFailoverTickInterval) as a defensive margin — it only pushes
	// promotion LATER (safer), so it is not required for safety, but keeping it in the
	// floor leaves headroom for scheduling jitter / small clock skew. Asserted here so
	// a misconfig fails LOUD at construction, whether or not PBAutoFailover is on.
	if c.ReplicationMode == ReplicationModePB {
		leaseTTL, metaStaleness, failoverTimeout, renewInterval := c.pbEffectiveTimings()
		floor := leaseTTL + metaStaleness + renewInterval + pbFailoverTickInterval
		if failoverTimeout <= floor {
			return fmt.Errorf("cluster.Config: PB failover honor rule violated: failoverTimeout (%s) must exceed pbLeaseTTL (%s) + metaContactStaleness (%s) + renewInterval (%s) + failoverTick (%s) = %s",
				failoverTimeout, leaseTTL, metaStaleness, renewInterval, pbFailoverTickInterval, floor)
		}
	}
	if c.ReplicationMode == ReplicationModePB && c.ShardCfg.PersistentVectors {
		return errors.New("cluster.Config: -persistent-vectors is not supported with -replication-mode=pb: " +
			"persistent-vectors wipes on-disk vector data at boot and relies on Raft log replay to " +
			"repopulate it, but pb mode has no Raft log to replay from, so vector data would be lost " +
			"with no recovery path")
	}
	// Multi-node validation: when Peers is set, all peers must validate
	// individually, NodeIDs must be unique, self must appear, and
	// RaftAddr is required (except in the single-peer self-only
	// auto-bind case: then the self-peer's RaftAddr may also be empty
	// and is filled in by newMultiNode after the listener is bound).
	if len(c.Peers) > 0 {
		autoBindSelf := c.RaftAddr == "" && len(c.Peers) == 1 && c.Peers[0].NodeID == c.NodeID
		if c.RaftAddr == "" && len(c.Peers) > 1 {
			return errors.New("cluster.Config: RaftAddr required when len(Peers) > 1")
		}
		seen := make(map[string]struct{}, len(c.Peers))
		selfPresent := false
		for i, p := range c.Peers {
			if autoBindSelf && p.NodeID == c.NodeID && p.RaftAddr == "" {
				// Allow empty RaftAddr on the self-peer in auto-bind mode;
				// still validate the rest of the peer fields.
				if p.NodeID == "" {
					return fmt.Errorf("cluster.Config.Peers[%d]: NodeID required", i)
				}
				if p.ServerAddr == "" {
					return fmt.Errorf("cluster.Config.Peers[%d]: ServerAddr required", i)
				}
			} else if err := p.Validate(); err != nil {
				return fmt.Errorf("cluster.Config.Peers[%d]: %w", i, err)
			}
			if _, dup := seen[p.NodeID]; dup {
				return fmt.Errorf("cluster.Config: duplicate Peer NodeID %q", p.NodeID)
			}
			seen[p.NodeID] = struct{}{}
			if p.NodeID == c.NodeID {
				selfPresent = true
			}
		}
		if !selfPresent {
			return fmt.Errorf("cluster.Config: self NodeID %q not found in Peers", c.NodeID)
		}
	}
	return nil
}

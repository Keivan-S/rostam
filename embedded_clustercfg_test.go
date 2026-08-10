// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"testing"
	"time"

	"github.com/rostamlabs/rostam/shard"
)

// TestClusterConfigFromThreadsPBKnobs guards the EmbeddedConfig → cluster.Config
// translation for the primary-backup knobs. A knob that exists on EmbeddedConfig
// but is never copied in clusterConfigFrom is silently unreachable at runtime —
// which is exactly how PBAutoFailover shipped: the field existed on
// cluster.Config, the whole failover machinery was gated on it, and yet
// nothing between the CLI and the cluster ever set it, so a `-replication-mode pb`
// server had no automatic failover AND no way to turn it on.
//
// Each knob is asserted with a NON-ZERO value, so a dropped assignment fails here
// instead of silently degrading a production cluster.
func TestClusterConfigFromThreadsPBKnobs(t *testing.T) {
	in := EmbeddedConfig{
		NodeID:          "n1",
		ReplicationMode: "pb",
		MinISR:          2,
		PBCommitPrimary: true,
		PBAutoFailover:  true,
		InternalToken:   "tok",
		// Not a PB knob, but the same class of failure and the same one-line fix:
		// a retention that never reaches cluster.Config means -wasm-blob-retention
		// is accepted by the CLI and silently does nothing.
		WASMBlobRetention: 90 * time.Minute,
	}
	got := clusterConfigFrom(in, 4, shard.Config{})

	if got.ReplicationMode != "pb" {
		t.Errorf("ReplicationMode = %q, want pb", got.ReplicationMode)
	}
	if got.MinISR != 2 {
		t.Errorf("MinISR = %d, want 2", got.MinISR)
	}
	if !got.PBCommitPrimary {
		t.Error("PBCommitPrimary was not threaded through")
	}
	if !got.PBAutoFailover {
		t.Error("PBAutoFailover was not threaded through — automatic failover " +
			"would be unreachable and a PB shard would stay DOWN on primary loss")
	}
	if got.WASMBlobRetention != 90*time.Minute {
		t.Errorf("WASMBlobRetention = %v, want 90m; -wasm-blob-retention would be accepted and do nothing", got.WASMBlobRetention)
	}
	if got.NumShards != 4 {
		t.Errorf("NumShards = %d, want the derived 4", got.NumShards)
	}
	if got.NodeID != "n1" || got.InternalToken != "tok" {
		t.Errorf("identity/token not threaded: NodeID=%q InternalToken=%q", got.NodeID, got.InternalToken)
	}

	// The zero value must stay off: -pb-auto-failover=false (and a library
	// embedder that does not opt in) must produce a byte-identical STATIC cluster.
	off := clusterConfigFrom(EmbeddedConfig{ReplicationMode: "pb"}, 1, shard.Config{})
	if off.PBAutoFailover {
		t.Error("PBAutoFailover defaulted ON for a zero-value EmbeddedConfig; the static-cluster opt-out would be impossible")
	}
	// Retirement is the one WASM mechanism that DELETES anything, and it cannot be
	// made safe against a lagging replica by any local rule — so the zero value
	// must stay off, here as well as in cluster.Config.
	if off.WASMBlobRetention != 0 {
		t.Errorf("WASMBlobRetention defaulted to %v for a zero-value EmbeddedConfig; blob deletion must be opt-in", off.WASMBlobRetention)
	}
}

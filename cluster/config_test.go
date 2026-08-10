// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
)

func validBaseConfig(t *testing.T) Config {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	return Config{
		NodeID: "node1", DataDir: t.TempDir(),
		NumShards: 4,
		Bootstrap: true,
		ShardCfg:  shard.DefaultConfig(t.TempDir(), "ignored", reg),
		Ops:       reg,
	}
}

func TestConfigValidate(t *testing.T) {
	if err := validBaseConfig(t).Validate(); err != nil {
		t.Fatalf("valid config: %v", err)
	}
}

func TestConfigValidateErrors(t *testing.T) {
	cases := []struct {
		name    string
		mut     func(*Config)
		wantErr error // optional; nil = "any error"
	}{
		{"empty NodeID", func(c *Config) { c.NodeID = "" }, nil},
		{"empty DataDir", func(c *Config) { c.DataDir = "" }, nil},
		{"NumShards zero", func(c *Config) { c.NumShards = 0 }, ErrInvalidNumShards},
		{"NumShards negative", func(c *Config) { c.NumShards = -1 }, ErrInvalidNumShards},
		{"NumShards too large", func(c *Config) { c.NumShards = 1 << 17 }, ErrInvalidNumShards},
		{"nil Ops", func(c *Config) { c.Ops = nil }, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := validBaseConfig(t)
			tc.mut(&cfg)
			err := cfg.Validate()
			if err == nil {
				t.Fatal("expected validation error, got nil")
			}
			if tc.wantErr != nil && !errors.Is(err, tc.wantErr) {
				t.Errorf("got %v, want errors.Is(%v)", err, tc.wantErr)
			}
		})
	}
}

func TestConfigValidatePeers(t *testing.T) {
	base := func() Config {
		c := validBaseConfig(t)
		c.Peers = []Peer{
			{NodeID: c.NodeID, RaftAddr: "a:1", ServerAddr: "a:2"},
			{NodeID: "n2", RaftAddr: "b:1", ServerAddr: "b:2"},
			{NodeID: "n3", RaftAddr: "c:1", ServerAddr: "c:2"},
		}
		c.RaftAddr = "a:1"
		return c
	}

	cases := []struct {
		name    string
		mut     func(*Config)
		wantErr bool
	}{
		{"happy 3-peer", func(_ *Config) {}, false},
		{"single-peer no RaftAddr is OK", func(c *Config) {
			c.Peers = []Peer{{NodeID: c.NodeID, RaftAddr: "a:1", ServerAddr: "a:2"}}
			c.RaftAddr = ""
		}, false},
		{"multi-peer no RaftAddr", func(c *Config) { c.RaftAddr = "" }, true},
		{"self missing", func(c *Config) {
			c.Peers = []Peer{
				{NodeID: "other", RaftAddr: "x:1", ServerAddr: "x:2"},
				{NodeID: "n2", RaftAddr: "b:1", ServerAddr: "b:2"},
				{NodeID: "n3", RaftAddr: "c:1", ServerAddr: "c:2"},
			}
		}, true},
		{"duplicate NodeID", func(c *Config) {
			c.Peers[1].NodeID = c.Peers[0].NodeID
		}, true},
		{"invalid peer (empty RaftAddr)", func(c *Config) {
			c.Peers[1].RaftAddr = ""
		}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := base()
			tc.mut(&cfg)
			err := cfg.Validate()
			if (err != nil) != tc.wantErr {
				t.Errorf("Validate() error = %v, wantErr = %v", err, tc.wantErr)
			}
		})
	}
}

func TestConfigReplicationModeValidation(t *testing.T) {
	cases := []struct {
		name    string
		mut     func(*Config)
		wantErr bool
	}{
		{"empty mode is valid", func(c *Config) {}, false},
		{"raft mode is valid", func(c *Config) { c.ReplicationMode = "raft" }, false},
		{"pb mode with MinISR=1 is valid", func(c *Config) {
			c.ReplicationMode = "pb"
			c.MinISR = 1
		}, false},
		{"pb mode with MinISR=2 is valid", func(c *Config) {
			c.ReplicationMode = "pb"
			c.MinISR = 2
		}, false},
		{"unknown mode is invalid", func(c *Config) { c.ReplicationMode = "bogus" }, true},
		{"pb mode with MinISR=0 is invalid", func(c *Config) {
			c.ReplicationMode = "pb"
			c.MinISR = 0
		}, true},
		{"pb default failover timings satisfy the honor rule (10s > 5+2+1+0.5=8.5s)", func(c *Config) {
			c.ReplicationMode = "pb"
			c.MinISR = 1
		}, false},
		{"pb honor rule violated: failoverTimeout below the full floor", func(c *Config) {
			c.ReplicationMode = "pb"
			c.MinISR = 1
			c.PBLeaseTTLMs = 1000
			c.PBMetaContactStalenessMs = 500
			c.PBRenewIntervalMs = 300
			c.PBFailoverTimeoutMs = 2000 // floor = 1000+500+300+500 = 2300; 2000 <= 2300 => reject
		}, true},
		{"pb honor rule: passes OLD inequality but violates the renewInterval-corrected one", func(c *Config) {
			// OLD rule was failoverTimeout > leaseTTL+staleness (1501 > 1500 ✓), but the
			// corrected floor adds renewInterval (+ tick): 1000+500+1000+500 = 3000, so
			// 1501 <= 3000 => now correctly REJECTED. This is the M1 regression guard.
			c.ReplicationMode = "pb"
			c.MinISR = 1
			c.PBLeaseTTLMs = 1000
			c.PBMetaContactStalenessMs = 500
			c.PBRenewIntervalMs = 1000
			c.PBFailoverTimeoutMs = 1501
		}, true},
		{"pb custom failover timings satisfy the corrected honor rule", func(c *Config) {
			c.ReplicationMode = "pb"
			c.MinISR = 1
			c.PBLeaseTTLMs = 1000
			c.PBMetaContactStalenessMs = 500
			c.PBRenewIntervalMs = 300
			c.PBFailoverTimeoutMs = 4000 // floor = 1000+500+300+500 = 2300; 4000 > 2300 => ok
		}, false},
		{"PersistentVectors alone (raft mode) is valid", func(c *Config) {
			c.ShardCfg.PersistentVectors = true
		}, false},
		{"pb mode alone (no PersistentVectors) is valid", func(c *Config) {
			c.ReplicationMode = "pb"
			c.MinISR = 1
		}, false},
		{"PersistentVectors with pb mode is invalid", func(c *Config) {
			c.ReplicationMode = "pb"
			c.MinISR = 1
			c.ShardCfg.PersistentVectors = true
		}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := validBaseConfig(t)
			tc.mut(&cfg)
			err := cfg.Validate()
			if (err != nil) != tc.wantErr {
				t.Errorf("Validate() error = %v, wantErr = %v", err, tc.wantErr)
			}
		})
	}
}

func TestDefaultConfigNumShards(t *testing.T) {
	reg := ops.NewRegistry()
	_ = ops.RegisterBuiltins(reg)
	cfg := DefaultConfig(t.TempDir(), "node1", reg)
	if cfg.NumShards != 64 {
		t.Errorf("default NumShards = %d, want 64", cfg.NumShards)
	}
	if err := cfg.Validate(); err != nil {
		t.Errorf("default config invalid: %v", err)
	}
	if cfg.ShardCfg.Ops != reg {
		t.Error("ShardCfg.Ops must be the same registry pointer as Ops")
	}
	if cfg.Ops != reg {
		t.Error("Ops must be the same registry pointer as the one passed in")
	}
}

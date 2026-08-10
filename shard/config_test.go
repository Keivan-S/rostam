// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
)

func TestDefaultConfigValid(t *testing.T) {
	cfg := DefaultConfig(t.TempDir(), "node1", ops.NewRegistry())
	if err := cfg.Validate(); err != nil {
		t.Fatalf("default valid: %v", err)
	}
}

func TestConfigValidationErrors(t *testing.T) {
	dir := t.TempDir()
	r := ops.NewRegistry()
	cases := []struct {
		name string
		mut  func(*Config)
	}{
		{"empty NodeID", func(c *Config) { c.NodeID = "" }},
		{"empty DataDir", func(c *Config) { c.DataDir = "" }},
		{"nil Ops", func(c *Config) { c.Ops = nil }},
		{"NumShards != 1", func(c *Config) { c.Cache.NumShards = 2 }},
		{"SnapshotInterval negative", func(c *Config) { c.SnapshotIntervalMs = -1 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultConfig(dir, "node1", r)
			tc.mut(&cfg)
			if err := cfg.Validate(); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
	// Sanity: confirm DefaultConfig sets NumShards=1.
	cfg := DefaultConfig(dir, "node1", r)
	if cfg.Cache.NumShards != 1 {
		t.Fatalf("DefaultConfig NumShards = %d, want 1", cfg.Cache.NumShards)
	}
	_ = cache.DefaultConfig() // keep import live
}

func validConfig() Config {
	reg := ops.NewRegistry()
	ops.RegisterBuiltins(reg)
	c := DefaultConfig("/tmp/x", "n1", reg)
	return c
}

func TestValidateReplicationMode(t *testing.T) {
	for _, tc := range []struct {
		mode string
		ok   bool
	}{
		{"", true}, {"raft", true}, {"pb", true}, {"bogus", false},
	} {
		c := validConfig()
		c.ReplicationMode = tc.mode
		if tc.mode == ReplicationModePB {
			c.PBControl = fakeControl{node: c.NodeID}
		}
		err := c.Validate()
		if tc.ok && err != nil {
			t.Fatalf("mode %q: unexpected error %v", tc.mode, err)
		}
		if !tc.ok && err == nil {
			t.Fatalf("mode %q: expected validation error, got nil", tc.mode)
		}
	}
}

func TestValidatePersistentVectorsPBRejected(t *testing.T) {
	// PersistentVectors alone (raft mode, the supported config) must keep working.
	c := validConfig()
	c.PersistentVectors = true
	if err := c.Validate(); err != nil {
		t.Fatalf("PersistentVectors in raft mode: unexpected error %v", err)
	}

	// pb mode alone (no PersistentVectors) must keep working.
	c = validConfig()
	c.ReplicationMode = ReplicationModePB
	c.PBControl = fakeControl{node: c.NodeID}
	if err := c.Validate(); err != nil {
		t.Fatalf("pb mode without PersistentVectors: unexpected error %v", err)
	}

	// PersistentVectors combined with pb mode must be rejected: pb mode has no
	// Raft log to repopulate the wiped vector data from.
	c = validConfig()
	c.ReplicationMode = ReplicationModePB
	c.PBControl = fakeControl{node: c.NodeID}
	c.PersistentVectors = true
	if err := c.Validate(); err == nil {
		t.Fatal("PersistentVectors with pb mode: expected validation error, got nil")
	}
}

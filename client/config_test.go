// SPDX-License-Identifier: Apache-2.0

package client

import (
	"runtime"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

func TestConfigDefaults(t *testing.T) {
	cfg := Config{Servers: []string{"127.0.0.1:7001"}}
	cfg.applyDefaults()

	if cfg.MaxConnsPerServer < 4 {
		t.Errorf("MaxConnsPerServer = %d, want >= 4", cfg.MaxConnsPerServer)
	}
	if numCPU := int32(runtime.NumCPU()); cfg.MaxConnsPerServer < numCPU && numCPU >= 4 { //nolint:gosec // NumCPU fits int32
		t.Errorf("MaxConnsPerServer = %d, want >= NumCPU (%d)", cfg.MaxConnsPerServer, numCPU)
	}
	if cfg.MinConnsPerServer != 0 {
		t.Errorf("MinConnsPerServer = %d, want 0", cfg.MinConnsPerServer)
	}
	if cfg.MaxConnLifetime != time.Hour {
		t.Errorf("MaxConnLifetime = %v, want 1h", cfg.MaxConnLifetime)
	}
	if cfg.MaxConnIdleTime != 30*time.Minute {
		t.Errorf("MaxConnIdleTime = %v, want 30m", cfg.MaxConnIdleTime)
	}
	if cfg.HealthCheckPeriod != time.Minute {
		t.Errorf("HealthCheckPeriod = %v, want 1m", cfg.HealthCheckPeriod)
	}
	if cfg.DialTimeout != 5*time.Second {
		t.Errorf("DialTimeout = %v, want 5s", cfg.DialTimeout)
	}
	if cfg.CallTimeout != 5*time.Second {
		t.Errorf("CallTimeout = %v, want 5s", cfg.CallTimeout)
	}
	if cfg.MaxNotLeaderHops != 3 {
		t.Errorf("MaxNotLeaderHops = %d, want 3", cfg.MaxNotLeaderHops)
	}
}

func TestConfigValidate(t *testing.T) {
	cases := []struct {
		name string
		cfg  Config
		ok   bool
	}{
		{"valid", Config{Servers: []string{"x:1"}}, true},
		{"no servers", Config{Servers: nil}, false},
		{"empty server", Config{Servers: []string{""}}, false},
		{"negative MaxConns", Config{Servers: []string{"x:1"}, MaxConnsPerServer: -1}, false},
		{"min > max", Config{Servers: []string{"x:1"}, MaxConnsPerServer: 2, MinConnsPerServer: 5}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.Validate()
			if tc.ok && err != nil {
				t.Fatalf("Validate: %v, want nil", err)
			}
			if !tc.ok && err == nil {
				t.Fatal("Validate: nil, want error")
			}
		})
	}
}

func TestConfigDefaultsTopologyRefreshInterval(t *testing.T) {
	cfg := Config{Servers: []string{"127.0.0.1:7001"}}
	cfg.applyDefaults()
	if cfg.TopologyRefreshInterval != 5*time.Second {
		t.Errorf("got %v, want 5s", cfg.TopologyRefreshInterval)
	}
}

func TestConfigValidateTopologyRefreshInterval(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	// Too-short interval with Ops set: error.
	cfg := Config{
		Servers:                 []string{"127.0.0.1:7001"},
		Ops:                     reg,
		TopologyRefreshInterval: 500 * time.Millisecond,
	}
	if err := cfg.Validate(); err == nil {
		t.Error("expected error for sub-1s refresh interval with Ops set")
	}
	// Same interval with Ops=nil: OK (interval ignored).
	cfg.Ops = nil
	if err := cfg.Validate(); err != nil {
		t.Errorf("Ops=nil + short interval should be OK: %v", err)
	}
	// Default interval (1s+) with Ops set: OK.
	cfg.Ops = reg
	cfg.TopologyRefreshInterval = 5 * time.Second
	if err := cfg.Validate(); err != nil {
		t.Errorf("valid config rejected: %v", err)
	}
}

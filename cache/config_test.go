// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"runtime"
	"strconv"
	"strings"
	"testing"
)

func TestDefaultConfigValid(t *testing.T) {
	cfg := DefaultConfig()
	if err := cfg.Validate(); err != nil {
		t.Fatalf("default config must validate, got: %v", err)
	}
	if cfg.NumShards != 256 {
		t.Errorf("NumShards = %d, want 256", cfg.NumShards)
	}
	if cfg.PageSize != 16<<20 {
		t.Errorf("PageSize = %d, want 16 MiB", cfg.PageSize)
	}
	if cfg.MaxMemoryPerShard != 256<<20 {
		t.Errorf("MaxMemoryPerShard = %d, want 256 MiB", cfg.MaxMemoryPerShard)
	}
	if cfg.MaxPagesPerShard() != 16 {
		t.Errorf("MaxPagesPerShard = %d, want 16", cfg.MaxPagesPerShard())
	}
}

func TestConfigValidationErrors(t *testing.T) {
	cases := []struct {
		name string
		mut  func(*Config)
	}{
		{"NumShards zero", func(c *Config) { c.NumShards = 0 }},
		{"NumShards not power of two", func(c *Config) { c.NumShards = 100 }},
		{"PageSize too small", func(c *Config) { c.PageSize = 1024 }},
		{"MaxMemoryPerShard smaller than PageSize", func(c *Config) {
			c.PageSize = 16 << 20
			c.MaxMemoryPerShard = 8 << 20
		}},
		{"AtCapPolicy invalid", func(c *Config) { c.AtCapPolicy = 99 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultConfig()
			tc.mut(&cfg)
			if err := cfg.Validate(); err == nil {
				t.Fatalf("expected validation error, got nil")
			}
		})
	}
}

func TestConfigMsyncIntervalDefault(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.MsyncIntervalMs != 100 {
		t.Errorf("default MsyncIntervalMs = %d, want 100", cfg.MsyncIntervalMs)
	}
}

func TestConfigDataDirCrossPlatform(t *testing.T) {
	cfg := DefaultConfig()
	cfg.DataDir = t.TempDir()
	err := cfg.Validate()
	// Asserted against the same constant Validate consults, so a platform
	// gaining (or losing) a file-mapping implementation cannot leave the guard
	// and this test disagreeing about what the platform can do.
	if mmapSupported {
		if err != nil {
			t.Errorf("%s maps files and should accept DataDir: %v", runtime.GOOS, err)
		}
	} else {
		if err == nil {
			t.Errorf("%s cannot map files and should reject DataDir", runtime.GOOS)
		}
	}
}

func TestConfigDurableRequiresDataDir(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Durable = true
	cfg.DataDir = ""
	if err := cfg.Validate(); err == nil {
		t.Error("Durable without DataDir should error")
	}
}

func TestConfigMlockRequiresDataDir(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Mlock = true
	cfg.DataDir = ""
	if err := cfg.Validate(); err == nil {
		t.Error("Mlock without DataDir should error")
	}
}

func TestConfigMsyncIntervalNegativeRejected(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MsyncIntervalMs = -1
	if err := cfg.Validate(); err == nil {
		t.Error("negative MsyncIntervalMs should error")
	}
}

func TestConfigRejectsMaxPagesPerShardOverUint16(t *testing.T) {
	// PageSize=1 MiB with MaxMemoryPerShard=(65536 MiB) yields 65536 pages, one
	// past the uint16 page-index ceiling makeSlabRef relies on. Without the cap
	// this validates cleanly and page index 65536 wraps to 0 on the uint16 cast.
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	// 65536 pages x the 1 MiB PageSize FLOOR is 64 GiB, which no 32-bit int can
	// hold — so on such a build this config is not merely untested but
	// unrepresentable, and the untyped constant would not even compile. Computed
	// from cfg.PageSize (a variable, hence a run-time multiply) so the file
	// builds under GOARCH=386, and skipped there because the state it asserts
	// about cannot exist.
	if strconv.IntSize < 64 {
		t.Skip("MaxPagesPerShard > 65535 needs >= 64 GiB, which does not fit a 32-bit int")
	}
	cfg.MaxMemoryPerShard = 65536 * cfg.PageSize // 65536 pages
	if cfg.MaxPagesPerShard() != 65536 {
		t.Fatalf("precondition: MaxPagesPerShard=%d, want 65536", cfg.MaxPagesPerShard())
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected validation error for MaxPagesPerShard > 65535, got nil")
	}
	if !strings.Contains(err.Error(), "MaxPagesPerShard") || !strings.Contains(err.Error(), "65535") {
		t.Errorf("error %q should name MaxPagesPerShard and the 65535 bound", err)
	}
}

func TestConfigAcceptsMaxPagesPerShardAtUint16Ceiling(t *testing.T) {
	// Regression guard: exactly 65535 pages must still validate.
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	// See the sibling test above: 65535 pages at the 1 MiB floor is ~64 GiB.
	if strconv.IntSize < 64 {
		t.Skip("65535 pages needs ~64 GiB, which does not fit a 32-bit int")
	}
	cfg.MaxMemoryPerShard = 65535 * cfg.PageSize // exactly 65535 pages
	if cfg.MaxPagesPerShard() != 65535 {
		t.Fatalf("precondition: MaxPagesPerShard=%d, want 65535", cfg.MaxPagesPerShard())
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("config at exactly 65535 pages must validate, got: %v", err)
	}
}

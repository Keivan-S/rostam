// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// TestDirectStoreRejectsPartitions pins that a Direct store REFUSES
// Partitions > 1 instead of silently ignoring it.
//
// A Direct store has no partition catalog (VectorResplit/VectorReshard say so
// explicitly), so the count can never take effect. It used to be accepted
// anyway: the create returned success, the collection was single-partition, and
// nothing told the caller. That is the failure mode the bulk-staging route
// already refuses to have — a field quietly dropped is how someone ends up
// querying data that is not laid out the way they asked for.
//
// It matters most on a single node, which is what a new user tries first.
func TestDirectStoreRejectsPartitions(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	s, err := NewDirect(DirectConfig{Ops: reg})
	if err != nil {
		t.Fatalf("NewDirect: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })

	ctx := context.Background()
	base := VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1}

	// The regression: >1 must be an explicit error, and the message must say what
	// to do instead rather than just refusing.
	cfg := base
	cfg.Partitions = 8
	err = s.CreateCollection(ctx, "parted", cfg)
	if err == nil {
		t.Fatal("Partitions=8 on a Direct store was accepted; it must be refused, " +
			"since a Direct store has no partition catalog and would silently create one partition")
	}
	if !strings.Contains(err.Error(), "clustered backend") {
		t.Fatalf("error %q does not tell the caller what to use instead", err)
	}

	// Unset and 1 are the single-partition cases and must still work — the guard
	// must not turn "no partitioning asked for" into an error.
	for name, p := range map[string]int{"unset": 0, "explicit-1": 1} {
		cfg := base
		cfg.Partitions = p
		if err := s.CreateCollection(ctx, "ok_"+name, cfg); err != nil {
			t.Fatalf("Partitions=%d (%s) must be allowed on a Direct store: %v", p, name, err)
		}
	}
}

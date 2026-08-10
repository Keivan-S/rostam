// SPDX-License-Identifier: Apache-2.0

package ops

import "testing"

// TestRegistryCrossShardFlag covers the cross-shard marking: only
// RegisterRoutableCrossShard sets it, a cross-shard op keeps its KeyExtractor
// (still routable in the cluster), and unknown ops report false.
func TestRegistryCrossShardFlag(t *testing.T) {
	r := NewRegistry()
	noop := func(_ *TxContext, _ []byte) ([]byte, error) { return nil, nil }
	std := KeyExtractorByHandle("std")

	if err := r.RegisterRoutable("plain", OpReadWrite, noop, std); err != nil {
		t.Fatalf("RegisterRoutable: %v", err)
	}
	if err := r.RegisterRoutableCrossShard("xshard", OpReadWrite, noop, std); err != nil {
		t.Fatalf("RegisterRoutableCrossShard: %v", err)
	}
	if err := r.Register("shardless", OpReadOnly, noop); err != nil {
		t.Fatalf("Register: %v", err)
	}

	if !r.CrossShard("xshard") {
		t.Error("xshard registered cross-shard, want CrossShard=true")
	}
	if r.CrossShard("plain") {
		t.Error("plain routable op must not be cross-shard")
	}
	if r.CrossShard("shardless") {
		t.Error("shardless op must not be cross-shard")
	}
	if r.CrossShard("missing") {
		t.Error("unknown op must report CrossShard=false")
	}

	// A cross-shard op stays routable — the routing key still selects the cluster
	// shard; only the single-node lock strategy changes. LookupEntry folds the
	// crossShard flag into its single registry read (finding 036), so a routing
	// caller learns the lock strategy without a second CrossShard lock+lookup.
	if _, _, ke, xs, ok := r.LookupEntry("xshard"); !ok || ke == nil || !xs {
		t.Errorf("cross-shard op via LookupEntry: ok=%v ke!=nil=%v crossShard=%v, want all true", ok, ke != nil, xs)
	}
	if _, _, ke, xs, ok := r.LookupEntry("plain"); !ok || ke == nil || xs {
		t.Errorf("plain routable op via LookupEntry: ok=%v ke!=nil=%v crossShard=%v, want crossShard=false", ok, ke != nil, xs)
	}
	if _, _, _, xs, ok := r.LookupEntry("missing"); ok || xs {
		t.Errorf("unknown op via LookupEntry: ok=%v crossShard=%v, want both false", ok, xs)
	}
	// Lookup stays the 4-return narrow accessor (unchanged for existing callers).
	if _, _, ke, ok := r.Lookup("xshard"); !ok || ke == nil {
		t.Error("cross-shard op must remain routable via Lookup (non-nil KeyExtractor)")
	}
}

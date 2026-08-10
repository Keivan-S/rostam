// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"fmt"
	"testing"
)

func TestShardOfDeterministic(t *testing.T) {
	key := []byte("user:42")
	a := shardOf(key, 256)
	b := shardOf(key, 256)
	if a != b {
		t.Fatalf("non-deterministic: %d vs %d", a, b)
	}
}

func TestShardOfBoundedByNumShards(t *testing.T) {
	for _, n := range []int{1, 16, 256, 65536} {
		for i := range 100 {
			key := fmt.Appendf(nil, "k%d", i)
			s := shardOf(key, n)
			if s < 0 || s >= n {
				t.Errorf("shardOf(_, %d) = %d, out of range", n, s)
			}
		}
	}
}

func TestShardOfDistribution(t *testing.T) {
	const n = 256
	const samples = 10_000
	counts := make([]int, n)
	for i := range samples {
		key := fmt.Appendf(nil, "key-%d", i)
		counts[shardOf(key, n)]++
	}
	// With 10k samples over 256 shards, each shard should average
	// ~39 hits. Allow ±50% per shard, and require ≥80% of shards
	// to be within ±25%.
	mean := samples / n
	hot, warm := 0, 0
	for _, c := range counts {
		if c >= mean/2 && c <= mean*3/2 {
			warm++
		}
		if c >= mean*3/4 && c <= mean*5/4 {
			hot++
		}
	}
	if hot < n*8/10 {
		t.Errorf("only %d/%d shards within ±25%% of mean (%d); distribution too skewed", hot, n, mean)
	}
	if warm < n*95/100 {
		t.Errorf("only %d/%d shards within ±50%% of mean (%d); distribution too skewed", warm, n, mean)
	}
}

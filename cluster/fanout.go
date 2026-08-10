// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"fmt"
	"sort"
	"sync"

	"github.com/rostamlabs/rostam/ops"
)

// Consistency selects how each partition is read.
type Consistency uint8

const (
	AnyReplica       Consistency = iota // load-balanced replica read (default)
	LeaderOnly                          // read from the partition's Raft leader (best-effort)
	Linearizable                        // readIndex barrier on the leader (VerifyLeader + commit-index catch-up)
	BoundedStaleness                    // any-replica read with a max-staleness bound (== ops.ConsistencyBoundedStaleness)
)

// routeToLeader reports whether a read at this consistency level must reach the
// partition's Raft leader. Only LeaderOnly and Linearizable do; AnyReplica AND
// BoundedStaleness may serve from any replica.
//
// BoundedStaleness deliberately does NOT pin to the leader (its value is 3, which
// would WRONGLY satisfy a naive `c >= LeaderOnly` test): the whole point is to
// OFFLOAD the leader's data path to a follower while still meeting a freshness
// SLO. The serving SHARD enforces the bound (shard.Store.Call peeks the
// read_consistency byte + the staleness bound from the op args) and, when the
// replica is out of bound (or its leader-frontier RTT fails closed), upgrades the
// read via a NotLeaderError so the caller transparently re-routes to the leader.
// Linearizable's freshness barrier is likewise shard-enforced; routing only has to
// deliver those two levels to the leader.
func (c Consistency) routeToLeader() bool { return c == LeaderOnly || c == Linearizable }

// OnUnavailable selects behavior when a partition has no reachable replica.
type OnUnavailable uint8

const (
	Partial OnUnavailable = iota // return reachable partitions, flag degraded (default)
	Fail                         // error the whole query
)

// FanResult carries fan-out metadata alongside the merged results.
type FanResult struct {
	Degraded bool  // true if some partitions were skipped (Partial mode)
	Missing  []int // partition indices that were unreachable (sorted)
}

// FanArgs parameterizes a fan-out search.
type FanArgs struct {
	Collection    string // logical (already canonical) collection name
	P             int    // partition count (>1)
	Generation    uint32 // generation for physical name construction; 0 = legacy coll#p
	K             int    // top-k requested
	Op            string // op name to call on each partition
	Consistency   Consistency
	OnUnavailable OnUnavailable
	// Encode builds the op args for one physical partition collection name.
	Encode func(physCol string) []byte
}

// partitionCaller issues one op against a physical partition collection. When
// leaderOnly (routeToLeader) is false the caller may read any replica; when true
// the read is pinned to the partition's Raft leader (both LeaderOnly AND
// Linearizable route there). The Linearizable freshness barrier is NOT signalled
// by this bool — it travels in the op args' read_consistency byte and is enforced
// by the serving shard. embedded supplies the real implementation (CallPhysical);
// tests inject a fake.
type partitionCaller func(physCol, op string, args []byte, leaderOnly bool) ([]byte, error)

// FanOut scatters the op to all P partitions concurrently, decodes each, and
// merges. Honors Consistency (any-replica vs leader-pinned; LeaderOnly and
// Linearizable both pin to the leader) and OnUnavailable.
//
// Each goroutine writes only results[p] (its own distinct index). The slice is
// allocated once before the goroutines start and never resized, so concurrent
// writes to separate indices are safe — no mutex required.
func FanOut[T any](a FanArgs, call partitionCaller,
	decode func([]byte) ([]T, error), merge func(parts [][]T, k int) []T) ([]T, FanResult, error) {

	type out struct {
		p   int
		res []T
		err error
	}
	results := make([]out, a.P)
	runPartition := func(p int) {
		physCol := string(ops.PartitionKeyGen(a.Collection, a.Generation, p))
		raw, err := call(physCol, a.Op, a.Encode(physCol), a.Consistency.routeToLeader())
		if err != nil {
			results[p] = out{p: p, err: err}
			return
		}
		dec, err := decode(raw)
		results[p] = out{p: p, res: dec, err: err}
	}
	var wg sync.WaitGroup
	// Partition P-1 runs on the CALLING goroutine instead of a spawned one: the
	// caller does nothing but wg.Wait() after the loop, so it may as well do the
	// last partition's work itself — that's one fewer goroutine spawned and
	// scheduled per fan-out call, with identical result handling.
	for p := 0; p < a.P-1; p++ {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			runPartition(p)
		}(p)
	}
	if a.P > 0 {
		runPartition(a.P - 1)
	}
	wg.Wait()

	var parts [][]T
	var fr FanResult
	for _, o := range results {
		if o.err != nil {
			if a.OnUnavailable == Fail {
				return nil, FanResult{}, fmt.Errorf("partition %d: %w", o.p, o.err)
			}
			fr.Degraded = true
			fr.Missing = append(fr.Missing, o.p)
			continue
		}
		parts = append(parts, o.res)
	}
	sort.Ints(fr.Missing)
	return merge(parts, a.K), fr, nil
}

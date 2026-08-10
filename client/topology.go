// SPDX-License-Identifier: Apache-2.0

package client

import (
	"sync/atomic"

	"github.com/rostamlabs/rostam/ops"
)

// topologyCache holds an atomic snapshot of the cluster topology.
// Reads are lock-free (atomic.Pointer load); writes replace the whole
// snapshot. The zero value is empty and safe to use.
type topologyCache struct {
	p atomic.Pointer[ops.Topology]
}

// get returns the current snapshot, or nil if the cache is empty.
// Callers must not mutate the returned Topology — it may be shared
// concurrently with other readers.
func (c *topologyCache) get() *ops.Topology { return c.p.Load() }

// set replaces the snapshot. The argument is captured by value; the
// caller may mutate it after the call without affecting the cache.
func (c *topologyCache) set(t ops.Topology) { c.p.Store(&t) }

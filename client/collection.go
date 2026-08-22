// SPDX-License-Identifier: Apache-2.0

package client

import (
	"fmt"
	"time"

	"github.com/rostamlabs/rostam/ops/wire"
)

// Collection is a typed, name-scoped handle for vector operations against a
// remote Rostam collection. Constructing one performs no I/O and does not
// require the collection to exist.
type Collection struct {
	c    *Client
	name string
}

// Collection returns a handle bound to the named collection.
func (c *Client) Collection(name string) *Collection { return &Collection{c: c, name: name} }

// defaultTopologyRefreshInterval mirrors Config.applyDefaults' TopologyRefreshInterval
// default. NewRouted must apply it itself (before calling New) because New validates
// before it applies defaults, and Validate rejects TopologyRefreshInterval < 1s
// whenever Ops != nil.
const defaultTopologyRefreshInterval = 5 * time.Second

// NewRouted is like New, but wires Rostam's builtin routing registry so shard-aware
// routing is on by default: each keyed op is dispatched to the shard that owns its
// key instead of round-robin. Use it for typed Collection work against a multi-node
// cluster.
//
// New itself is intentionally left with nil-Ops semantics (no self-routing): internal
// callers such as cluster/node.go's peerClient must dial a specific, freshly-resolved
// address verbatim for leader-pinned (LeaderOnly/Linearizable) reads, and rely on
// nil Ops to disable client-side routing. Do not move this wiring into New.
func NewRouted(cfg Config) (*Client, error) {
	if cfg.Ops == nil {
		reg := wire.NewRegistry()
		if err := wire.RegisterRoutableBuiltins(reg); err != nil {
			return nil, fmt.Errorf("client: wire routing registry: %w", err)
		}
		cfg.Ops = reg
	}
	if cfg.TopologyRefreshInterval == 0 {
		cfg.TopologyRefreshInterval = defaultTopologyRefreshInterval
	}
	return New(cfg)
}

// hasRoutingRegistry reports whether shard-aware routing is active on this client.
func (c *Client) hasRoutingRegistry() bool { return c.cfg.Ops != nil }

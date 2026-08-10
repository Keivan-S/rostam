// SPDX-License-Identifier: Apache-2.0

package cluster

import "errors"

// ErrUnknownOp is returned by Call when the op name is not in the
// registered Ops registry.
var ErrUnknownOp = errors.New("cluster: op not registered")

// ErrNoKeyExtractor is returned by Call when a routable op's
// KeyExtractor returns hasKey=false (malformed args).
var ErrNoKeyExtractor = errors.New("cluster: op args do not yield a routing key")

// ErrInvalidNumShards is returned by Config.Validate when NumShards
// is outside the valid range [1, 65536].
var ErrInvalidNumShards = errors.New("cluster: NumShards must be in [1, 65536]")

// ErrNotReady is returned by the __ready__ readiness probe when this node hosts
// one or more shards that currently have no usable leader (quorum lost / mid
// election / no valid PB primary lease). A load balancer should stop routing to
// a node whose readiness probe fails.
var ErrNotReady = errors.New("cluster: node not ready")

// ErrNoShardOwner is returned by Call when a shard this node does not host has
// no reachable owner to forward to.
var ErrNoShardOwner = errors.New("cluster: no reachable owner for shard")

// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"github.com/cespare/xxhash/v2"
)

// shardOf returns the shard index for key, given numShards. Uses
// xxhash for speed (~5ns) and modulo for the bucketing.
//
// numShards must be in [1, 65536] — validated upstream by Config.Validate.
func shardOf(key []byte, numShards int) int {
	return int(xxhash.Sum64(key) % uint64(numShards)) //nolint:gosec // numShards bounded
}

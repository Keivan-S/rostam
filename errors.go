// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"errors"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/shard"
)

// mapErr translates internal package errors to the public Store sentinels.
// Unknown errors are returned as-is.
func mapErr(err error) error {
	if err == nil {
		return nil
	}
	if matchesNotFound(err) {
		return ErrNotFound
	}
	if matchesNotLeader(err) {
		return ErrNotLeader
	}
	return err
}

// matchesNotFound returns true for cache.ErrNotFound or client.ErrNotFound.
func matchesNotFound(err error) bool {
	return err == cache.ErrNotFound || err == client.ErrNotFound
}

// matchesNotLeader returns true for shard.NotLeaderError (embedded path)
// or client.ErrNoLeaderKnown (networked path).
func matchesNotLeader(err error) bool {
	if err == client.ErrNoLeaderKnown {
		return true
	}
	var nle *shard.NotLeaderError
	return errors.As(err, &nle)
}

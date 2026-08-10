// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"testing"

	"github.com/rostamlabs/rostam/server"
)

// Compile-time guard: *Node satisfies server.Dispatcher. The line
// below must compile; TestDispatcherSatisfied is a no-op runtime
// hook so `go test` reports it for visibility.
var _ server.Dispatcher = (*Node)(nil)

func TestDispatcherSatisfied(t *testing.T) { _ = t }

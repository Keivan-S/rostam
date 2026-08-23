// SPDX-License-Identifier: Apache-2.0
package client

import (
	"testing"
	"time"

	"github.com/rostamlabs/rostam/sdk/wire"
)

// NewRouted wires routing. This white-box assertion checks the wiring directly
// (hasRoutingRegistry is unexported); New/NewRouted do not dial on construction,
// so a never-dialed address suffices. The engine-backed keyed round-trip that
// exercises the routed Call path lives in the root module's integration tests
// (TestNewRoutedKeyedRoundTrip) because it needs a real node stack, which the
// engine-free client module cannot import.
func TestNewRoutedWiresRoutingRegistry(t *testing.T) {
	c, err := NewRouted(Config{Servers: []string{"127.0.0.1:1"}}) // never dialed
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	if !c.hasRoutingRegistry() {
		t.Fatal("NewRouted did not wire a routing registry")
	}
}

// The load-bearing invariant: plain New must NOT auto-wire routing (nil Ops means
// "don't self-route"). This protects cluster/node.go peerClient's leader-pinned reads.
func TestNewDoesNotAutoWireRouting(t *testing.T) {
	c, err := New(Config{Servers: []string{"127.0.0.1:1"}}) // never dialed
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	if c.hasRoutingRegistry() {
		t.Fatal("New must NOT wire routing; nil Ops is load-bearing (verbatim dialing)")
	}
}

// A caller-supplied Ops registry is preserved by NewRouted (not overwritten).
func TestNewRoutedPreservesSuppliedOps(t *testing.T) {
	reg := wire.NewRegistry()
	if err := wire.RegisterRoutableBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	c, err := NewRouted(Config{
		Servers:                 []string{"127.0.0.1:1"},
		Ops:                     reg,
		TopologyRefreshInterval: time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()
	if c.cfg.Ops != reg {
		t.Fatal("NewRouted overwrote a caller-supplied Ops registry")
	}
}

// SPDX-License-Identifier: Apache-2.0
package client

import (
	"context"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops/wire"
)

// NewRouted wires routing; a keyed round-trip against a real node works.
func TestNewRoutedWiresRoutingRegistry(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()

	c, err := NewRouted(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	if !c.hasRoutingRegistry() {
		t.Fatal("NewRouted did not wire a routing registry")
	}
	ctx := context.Background()
	if _, err := c.Call(ctx, "put", wire.EncodePutArgs([]byte("k"), []byte("v"), 0)); err != nil {
		t.Fatalf("Call put via routed client: %v", err)
	}
}

// The load-bearing invariant: plain New must NOT auto-wire routing (nil Ops means
// "don't self-route"). This protects cluster/node.go peerClient's leader-pinned reads.
func TestNewDoesNotAutoWireRouting(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()

	c, err := New(Config{Servers: []string{addr}})
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

// SPDX-License-Identifier: Apache-2.0

// inttest_support.go exports helpers strictly for the sibling inttest package (the
// slow integration tests that assert internal partition routing). NOT part of the
// stable embedded API; all forwards to unexported internals, no behavior. The
// canonical package doc lives in store.go. Keep additions here.

package rostam

import (
	"io"
	"time"

	"github.com/rostamlabs/rostam/cluster"
)

// Type aliases (identical underlying type ⇒ s.(*rostam.Embedded) works with zero rename).
type (
	Embedded         = embedded
	PartitionCatalog = partitionCatalog
	FanoutDispatcher = fanoutDispatcher
	InnerDispatcher  = innerDispatcher
)

// Node exposes the unexported embedded.node field for the inttest package.
func (e *embedded) Node() *cluster.Node { return e.node }

// Catalog exposes the unexported embedded.catalog field for the inttest package.
func (e *embedded) Catalog() PartitionCatalog { return e.catalog }

// NewFanoutDispatcher forwards to the unexported newFanoutDispatcher constructor.
func NewFanoutDispatcher(e *Embedded, inner InnerDispatcher) *FanoutDispatcher {
	return newFanoutDispatcher(e, inner)
}

// Store exposes the unexported Server.store backing store for the inttest package,
// so a cluster integration test can type-assert it to *Embedded. Returns the
// io.Closer the Server holds (a *embedded for replicated/cluster servers).
func (s *Server) Store() io.Closer { return s.store }

// Partitioned exposes the unexported fanoutDispatcher.partitioned routing lookup
// (returns P, gen, ok for a collection) for the inttest fan-out tests.
func (f *FanoutDispatcher) Partitioned(coll string) (int, uint32, bool) {
	return f.partitioned(coll)
}

// WCWire forwards to the unexported wcWire write-consistency wire decision so the
// inttest package can unit-test the plain-vs-envelope branch without a transport.
func WCWire(opName string, innerArgs []byte, opts WriteOpts) (string, []byte) {
	return wcWire(opName, innerArgs, opts)
}

// SetReshardDrainGrace overrides reshardDrainGrace, returning a restore func so the
// internal read of the unexported var is unchanged after the test.
func SetReshardDrainGrace(d time.Duration) (restore func()) {
	prev := reshardDrainGrace
	reshardDrainGrace = d
	return func() { reshardDrainGrace = prev }
}

// SetReshardCutoverGateTimeout overrides reshardCutoverGateTimeout with restore.
func SetReshardCutoverGateTimeout(d time.Duration) (restore func()) {
	prev := reshardCutoverGateTimeout
	reshardCutoverGateTimeout = d
	return func() { reshardCutoverGateTimeout = prev }
}

// SetMetaReadIndexReadTimeout overrides metaReadIndexReadTimeout with restore.
func SetMetaReadIndexReadTimeout(d time.Duration) (restore func()) {
	prev := metaReadIndexReadTimeout
	metaReadIndexReadTimeout = d
	return func() { metaReadIndexReadTimeout = prev }
}

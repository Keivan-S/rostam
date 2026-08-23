// SPDX-License-Identifier: Apache-2.0

package ops

import "github.com/rostamlabs/rostam/sdk/wire"

// Topology, TopologyMember, and TopologySource are aliases onto the leaf
// (ops/wire) definitions: the wire types are the client-safe wire codec (the Go
// client decodes them directly via wire.DecodeTopology), and ops re-exports them
// so existing server-side callers (cluster.Node) keep compiling unchanged.
type (
	Topology       = wire.Topology
	TopologyMember = wire.TopologyMember
	TopologySource = wire.TopologySource
)

// EncodeTopology serializes t. See wire.EncodeTopology.
func EncodeTopology(t Topology) ([]byte, error) { return wire.EncodeTopology(t) }

// DecodeTopology parses an encoded Topology. See wire.DecodeTopology.
func DecodeTopology(b []byte) (Topology, error) { return wire.DecodeTopology(b) }

// RegisterTopology adds the __topology__ shardless op to reg. The
// source closure is invoked on every call; it must be cheap (it reads
// meta-Raft FSM state, no I/O). Returns an error if the registry
// already has __topology__ registered.
func RegisterTopology(reg *Registry, src TopologySource) error {
	return reg.Register("__topology__", OpReadOnly, func(_ *TxContext, _ []byte) ([]byte, error) {
		t, err := src()
		if err != nil {
			return nil, err
		}
		return EncodeTopology(t)
	})
}

// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"bytes"
	"encoding/gob"
	"errors"
)

// Topology is a point-in-time snapshot of cluster routing state.
//
// Members lists every node with its client-facing ServerAddr. Leaders
// is keyed by shard ID; the value is the ServerAddr of that shard's
// current leader, or "" if no leader is known. len(Leaders) == NumShards.
//
// Placement is keyed by shard ID; the value is the NodeIDs that own (host)
// that shard — the replica set. len(Placement) == NumShards (may be empty in
// legacy setups). Unlike Leaders, Placement is fully known on every node (it is
// replicated meta state), so a client can route directly to an owner even when
// the shard's leader is not yet known — the client-side sharding fast path.
// Owners that aren't leader return a NotLeader hint the client then follows.
//
// Topology is gob-encoded and returned by the __topology__ op. Clients
// cache it and refresh on a timer + on every NotLeader response.
type Topology struct {
	NumShards int
	Members   []TopologyMember
	Leaders   []string
	Placement [][]string
}

// OwnerAddr returns the client-facing ServerAddr of an owner of shardID (the
// first owner in placement order), or "" if unknown. Lets a smart client route
// directly to an owner when the leader is not yet known.
func (t Topology) OwnerAddr(shardID int) string {
	if shardID < 0 || shardID >= len(t.Placement) {
		return ""
	}
	for _, nodeID := range t.Placement[shardID] {
		for _, m := range t.Members {
			if m.NodeID == nodeID {
				return m.ServerAddr
			}
		}
	}
	return ""
}

// TopologyMember names a cluster member by NodeID + client-facing
// ServerAddr. The Raft transport addr is intentionally not exposed —
// clients don't speak Raft.
type TopologyMember struct {
	NodeID     string
	ServerAddr string
}

// EncodeTopology serializes t with gob.
func EncodeTopology(t Topology) ([]byte, error) {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(t); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// DecodeTopology parses an encoded Topology.
func DecodeTopology(b []byte) (Topology, error) {
	if len(b) == 0 {
		return Topology{}, errors.New("wire: empty topology payload")
	}
	var t Topology
	if err := gob.NewDecoder(bytes.NewReader(b)).Decode(&t); err != nil {
		return Topology{}, err
	}
	return t, nil
}

// TopologySource returns the current cluster topology. Implemented by
// *cluster.Node; injected at registration time so ops/ stays free of
// cluster-package imports.
type TopologySource func() (Topology, error)

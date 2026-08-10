// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"fmt"
)

// Peer identifies a cluster member's network endpoints.
type Peer struct {
	NodeID     string // logical ID (matches cluster.Config.NodeID on that node)
	RaftAddr   string // multiplexed Raft transport endpoint, e.g., "10.0.0.1:7102"
	ServerAddr string // client-facing TCP server endpoint, e.g., "10.0.0.1:7001"

	// PBAddr is this node's pbisr.NetTransport listen endpoint, e.g.,
	// "10.0.0.1:7200". Required only when Config.ReplicationMode is "pb";
	// left empty in "raft" mode (and validated as unused there). Its
	// presence is checked at the cluster level once the mode is known, not
	// unconditionally by Validate() below.
	PBAddr string
}

// Validate checks required fields are non-empty.
func (p Peer) Validate() error {
	if p.NodeID == "" {
		return errors.New("cluster.Peer: NodeID required")
	}
	if p.RaftAddr == "" {
		return fmt.Errorf("cluster.Peer %q: RaftAddr required", p.NodeID)
	}
	if p.ServerAddr == "" {
		return fmt.Errorf("cluster.Peer %q: ServerAddr required", p.NodeID)
	}
	return nil
}

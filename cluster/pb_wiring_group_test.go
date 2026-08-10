// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"testing"

	"github.com/rostamlabs/rostam/shard/pbisr"
)

// plainFakeTransport implements only pbisr.Transport.
type plainFakeTransport struct{}

func (plainFakeTransport) Replicate(string, pbisr.ReplicateMsg, func(pbisr.AckMsg, error)) error {
	return nil
}

// groupFakeTransport also implements the optional pbisr.GroupTransport
// capability, recording the address each group was sent to.
type groupFakeTransport struct {
	plainFakeTransport
	lastAddr string
}

func (g *groupFakeTransport) ReplicateGroup(addr string, _ []pbisr.ReplicateMsg, _ func(pbisr.AckMsg, error)) error {
	g.lastAddr = addr
	return nil
}

// TestResolvingTransportPropagatesGroupCapability pins the wiring rule:
// the node-ID-resolving wrapper advertises GroupTransport iff its base has it,
// and forwards groups with the peer node-ID rewritten to its PB dial address.
func TestResolvingTransportPropagatesGroupCapability(t *testing.T) {
	addrs := map[string]string{"n2": "10.0.0.2:7200"}

	plain := newPBResolvingTransport(plainFakeTransport{}, addrs)
	if _, ok := plain.(pbisr.GroupTransport); ok {
		t.Fatal("plain base must NOT advertise GroupTransport through the wrapper")
	}

	base := &groupFakeTransport{}
	wrapped := newPBResolvingTransport(base, addrs)
	gt, ok := wrapped.(pbisr.GroupTransport)
	if !ok {
		t.Fatal("group-capable base lost its GroupTransport capability through the wrapper")
	}
	if err := gt.ReplicateGroup("n2", nil, nil); err != nil {
		t.Fatalf("ReplicateGroup: %v", err)
	}
	if base.lastAddr != "10.0.0.2:7200" {
		t.Fatalf("group went to %q, want the resolved PB address", base.lastAddr)
	}
	if err := gt.ReplicateGroup("nX", nil, nil); err == nil {
		t.Fatal("unresolved node-ID must be a submission error, not a dial attempt")
	}
}

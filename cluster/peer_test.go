// SPDX-License-Identifier: Apache-2.0

package cluster

import "testing"

func TestPeerValidate(t *testing.T) {
	cases := []struct {
		name    string
		p       Peer
		wantErr bool
	}{
		{"happy", Peer{NodeID: "n1", RaftAddr: "a:1", ServerAddr: "a:2"}, false},
		{"empty NodeID", Peer{RaftAddr: "a:1", ServerAddr: "a:2"}, true},
		{"empty RaftAddr", Peer{NodeID: "n1", ServerAddr: "a:2"}, true},
		{"empty ServerAddr", Peer{NodeID: "n1", RaftAddr: "a:1"}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.p.Validate()
			if (err != nil) != tc.wantErr {
				t.Errorf("Validate() error = %v, wantErr = %v", err, tc.wantErr)
			}
		})
	}
}

// TestPeerPBAddrField verifies PBAddr is settable/readable on Peer, and that
// Validate() stays lenient about it (PBAddr is only required in "pb" mode,
// validated at the cluster level once the mode is known — not here).
func TestPeerPBAddrField(t *testing.T) {
	p := Peer{NodeID: "n1", RaftAddr: "a:1", ServerAddr: "a:2", PBAddr: "10.0.0.1:7200"}
	if p.PBAddr != "10.0.0.1:7200" {
		t.Fatalf("PBAddr = %q, want %q", p.PBAddr, "10.0.0.1:7200")
	}
	if err := p.Validate(); err != nil {
		t.Fatalf("Validate() with PBAddr set: %v", err)
	}

	// Raft-mode peers have no PBAddr; Validate() must not require it.
	raftOnly := Peer{NodeID: "n2", RaftAddr: "b:1", ServerAddr: "b:2"}
	if raftOnly.PBAddr != "" {
		t.Fatalf("zero-value PBAddr = %q, want empty", raftOnly.PBAddr)
	}
	if err := raftOnly.Validate(); err != nil {
		t.Fatalf("Validate() with empty PBAddr (raft mode): %v", err)
	}
}

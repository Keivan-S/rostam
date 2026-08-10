// SPDX-License-Identifier: Apache-2.0

package grpcapi

import "testing"

// TestValidConsistencyBoundedStaleness proves the gRPC edge accepts rc==3
// (bounded-staleness) and FAILS LOUD on rc==4 (out of range) — the additive,
// fail-loud range check for the new level.
func TestValidConsistencyBoundedStaleness(t *testing.T) {
	for _, rc := range []uint32{0, 1, 2, 3} {
		if err := validConsistency(rc, 0); err != nil {
			t.Fatalf("validConsistency(rc=%d) = %v, want nil (in range)", rc, err)
		}
	}
	if err := validConsistency(4, 0); err == nil {
		t.Fatal("validConsistency(rc=4) = nil, want InvalidArgument (out of range)")
	}
	// opa range is unchanged: 0/1 ok, 2 rejected.
	if err := validConsistency(3, 2); err == nil {
		t.Fatal("validConsistency(opa=2) = nil, want InvalidArgument")
	}
}

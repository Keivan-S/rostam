// SPDX-License-Identifier: Apache-2.0

package pbisr

import "testing"

func TestLeaseValidAndLastSeq(t *testing.T) {
	var clock int64 = 100
	e := New("n1", 0, nil, nil, nil, WithClock(func() int64 { return clock }))

	if e.LeaseValid() {
		t.Fatal("no lease granted yet: LeaseValid must be false")
	}
	e.GrantLease(1, 200) // epoch 1 valid until mono 200
	if !e.LeaseValid() {
		t.Fatal("fresh lease at clock<expiry: LeaseValid must be true")
	}
	clock = 250 // past expiry
	if e.LeaseValid() {
		t.Fatal("expired lease: LeaseValid must be false")
	}
	if e.LastSeq() != 0 {
		t.Fatalf("LastSeq before any write: want 0, got %d", e.LastSeq())
	}
}

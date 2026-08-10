// SPDX-License-Identifier: Apache-2.0

package rostam

import "testing"

func boolPtr(b bool) *bool { return &b }

// TestWriteOptsWCActive is the truth table for WriteOpts.wcActive: the default
// (zero) value is inactive; WCF>0 is active; an explicit wait=false is active;
// an explicit wait=true alone is NOT active.
func TestWriteOptsWCActive(t *testing.T) {
	cases := []struct {
		name string
		opts WriteOpts
		want bool
	}{
		{"zero value (default)", WriteOpts{}, false},
		{"WCF>0", WriteOpts{WriteConsistencyFactor: 1}, true},
		{"WCF>1", WriteOpts{WriteConsistencyFactor: 3}, true},
		{"wait=false explicit", WriteOpts{Wait: boolPtr(false)}, true},
		{"wait=true explicit only", WriteOpts{Wait: boolPtr(true)}, false},
		{"WCF>0 and wait=true", WriteOpts{WriteConsistencyFactor: 2, Wait: boolPtr(true)}, true},
		{"WCF>0 and wait=false", WriteOpts{WriteConsistencyFactor: 2, Wait: boolPtr(false)}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.opts.wcActive(); got != tc.want {
				t.Errorf("wcActive() = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestWriteOptsWaitValue checks the tri-state Wait resolves correctly.
func TestWriteOptsWaitValue(t *testing.T) {
	cases := []struct {
		name string
		opts WriteOpts
		want bool
	}{
		{"nil defaults true", WriteOpts{}, true},
		{"explicit true", WriteOpts{Wait: boolPtr(true)}, true},
		{"explicit false", WriteOpts{Wait: boolPtr(false)}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.opts.waitValue(); got != tc.want {
				t.Errorf("waitValue() = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestVectorInsertOptsEmbedsWriteOpts confirms the knobs are reachable through
// the embedded field on VectorInsertOpts and the zero value is inactive (no
// envelope ever built by default).
func TestVectorInsertOptsEmbedsWriteOpts(t *testing.T) {
	var o VectorInsertOpts
	if o.wcActive() {
		t.Fatal("zero VectorInsertOpts must be wc-inactive")
	}
	o.WriteConsistencyFactor = 3
	o.Wait = boolPtr(false)
	if !o.wcActive() {
		t.Fatal("VectorInsertOpts with WCF=3 must be wc-active")
	}
	if o.waitValue() {
		t.Fatal("waitValue should be false when Wait=&false")
	}
}

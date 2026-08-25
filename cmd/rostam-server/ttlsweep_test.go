// SPDX-License-Identifier: Apache-2.0

package main

import (
	"testing"
	"time"
)

func TestResolveTTLSweepMs(t *testing.T) {
	cases := []struct {
		name    string
		in      time.Duration
		want    int
		wantErr bool
	}{
		{"default-30s", 30 * time.Second, 30000, false},
		{"five-seconds", 5 * time.Second, 5000, false},
		{"one-ms", time.Millisecond, 1, false},
		{"sub-ms floored to 1", 500 * time.Microsecond, 1, false},
		{"zero disables", 0, -1, false},
		{"negative is an error", -time.Second, 0, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := resolveTTLSweepMs(tc.in)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("resolveTTLSweepMs(%v): expected error, got nil (val %d)", tc.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("resolveTTLSweepMs(%v): unexpected error: %v", tc.in, err)
			}
			if got != tc.want {
				t.Fatalf("resolveTTLSweepMs(%v) = %d, want %d", tc.in, got, tc.want)
			}
		})
	}
}

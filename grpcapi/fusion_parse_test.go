// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// TestParseFusionDBSF asserts the gRPC fusion-method parser maps "dbsf" to
// FusionDBSF, keeps the existing methods, and still fails loud on unknown input.
func TestParseFusionDBSF(t *testing.T) {
	cases := []struct {
		in      string
		want    vector.FusionMethod
		wantErr bool
	}{
		{"", vector.FusionRRF, false},
		{"rrf", vector.FusionRRF, false},
		{"weighted", vector.FusionWeighted, false},
		{"dbsf", vector.FusionDBSF, false},
		{"bogus", 0, true},
	}
	for _, tc := range cases {
		got, err := parseFusion(tc.in)
		if tc.wantErr {
			if err == nil {
				t.Errorf("parseFusion(%q) err = nil, want error", tc.in)
			}
			continue
		}
		if err != nil {
			t.Errorf("parseFusion(%q) err = %v, want nil", tc.in, err)
			continue
		}
		if got != tc.want {
			t.Errorf("parseFusion(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

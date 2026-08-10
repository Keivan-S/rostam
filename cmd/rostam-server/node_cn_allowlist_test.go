// SPDX-License-Identifier: Apache-2.0

package main

import (
	"reflect"
	"testing"
)

// TestParseCNAllowlist covers the -node-cn-allowlist CSV parse: empty/whitespace
// -> nil (OFF = byte-identical), a CSV -> the trimmed set, and dropped empties.
func TestParseCNAllowlist(t *testing.T) {
	cases := []struct {
		spec string
		want map[string]bool
		desc string
	}{
		{"", nil, "empty spec -> nil (OFF)"},
		{"   ", nil, "whitespace-only spec -> nil (OFF)"},
		{",", nil, "only separators -> nil (OFF)"},
		{" , , ", nil, "blank entries only -> nil (OFF)"},
		{"a", map[string]bool{"a": true}, "single CN"},
		{"a,b", map[string]bool{"a": true, "b": true}, "two CNs"},
		{" n1 , n2 ,n3", map[string]bool{"n1": true, "n2": true, "n3": true}, "trims whitespace around each CN"},
		{"n1,,n2,", map[string]bool{"n1": true, "n2": true}, "drops empty entries"},
		{"dup,dup", map[string]bool{"dup": true}, "dedupes"},
	}
	for _, c := range cases {
		got := parseCNAllowlist(c.spec)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("parseCNAllowlist(%q)=%v want %v — %s", c.spec, got, c.want, c.desc)
		}
		// OFF contract: the empty cases must return nil (not an empty non-nil map),
		// so every len()==0 consumer treats it as OFF crisply.
		if c.want == nil && got != nil {
			t.Errorf("parseCNAllowlist(%q) must return nil for the OFF case, got %v", c.spec, got)
		}
	}
}

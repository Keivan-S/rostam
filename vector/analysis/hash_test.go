// SPDX-License-Identifier: Apache-2.0

package analysis

import (
	"hash/fnv"
	"testing"
)

// TestTermIDMatchesStdlibFNV verifies TermID is exactly FNV-1a/32 over the
// stem bytes, so the value is reproducible by any FNV-1a implementation
// (cross-language, cross-node).
func TestTermIDMatchesStdlibFNV(t *testing.T) {
	for _, stem := range []string{"", "a", "vector", "databas", "running", "héllo", "consign"} {
		h := fnv.New32a()
		_, _ = h.Write([]byte(stem))
		want := h.Sum32()
		if got := TermID(stem); got != want {
			t.Errorf("TermID(%q) = %d, want stdlib FNV-1a %d", stem, got, want)
		}
	}
}

func TestTermIDDeterministic(t *testing.T) {
	for _, stem := range []string{"vector", "search", "rostam"} {
		a, b := TermID(stem), TermID(stem)
		if a != b {
			t.Errorf("TermID(%q) not deterministic: %d != %d", stem, a, b)
		}
	}
}

func TestTermIDDistinct(t *testing.T) {
	// Different stems should (in practice) hash differently.
	if TermID("cat") == TermID("dog") {
		t.Error("distinct stems collided unexpectedly")
	}
}

func TestTermIDMod(t *testing.T) {
	const stem = "vector"
	full := TermID(stem)

	// width 0 == no modulo == TermID.
	if got := TermIDMod(stem, 0); got != full {
		t.Errorf("TermIDMod(%q, 0) = %d, want %d", stem, got, full)
	}

	// width N folds into [0, N) and equals TermID % N.
	for _, w := range []uint32{1, 2, 16, 1024, 65536} {
		got := TermIDMod(stem, w)
		want := full % w
		if got != want {
			t.Errorf("TermIDMod(%q, %d) = %d, want %d", stem, w, got, want)
		}
		if got >= w {
			t.Errorf("TermIDMod(%q, %d) = %d out of range [0,%d)", stem, w, got, w)
		}
	}
}

func TestTermIDModDeterministic(t *testing.T) {
	for _, w := range []uint32{0, 256, 4096} {
		if TermIDMod("running", w) != TermIDMod("running", w) { //nolint:staticcheck // intentional: same call twice asserts determinism
			t.Errorf("TermIDMod not deterministic for width %d", w)
		}
	}
}

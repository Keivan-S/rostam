// SPDX-License-Identifier: Apache-2.0

package localcatalog

import (
	"regexp"
	"testing"
)

var hex64 = regexp.MustCompile(`^[0-9a-f]{64}$`)

func TestCatalogIntegrity(t *testing.T) {
	names := Names()
	if len(names) < 3 {
		t.Fatalf("want >=3 models, got %d", len(names))
	}
	seen := map[string]bool{}
	for _, n := range names {
		m, ok := Lookup(n)
		if !ok {
			t.Fatalf("Names() listed %q but Lookup failed", n)
		}
		if seen[m.Name] {
			t.Fatalf("duplicate model name %q", m.Name)
		}
		seen[m.Name] = true
		if m.Dim <= 0 {
			t.Errorf("%s: Dim=%d", m.Name, m.Dim)
		}
		if m.OnnxURL == "" || m.VocabURL == "" {
			t.Errorf("%s: empty URL", m.Name)
		}
		if !hex64.MatchString(m.OnnxSHA) || !hex64.MatchString(m.VocabSHA) {
			t.Errorf("%s: SHA not 64-hex", m.Name)
		}
		if m.License == "" {
			t.Errorf("%s: empty license", m.Name)
		}
	}
}

func TestLookupDefaults(t *testing.T) {
	for _, alias := range []string{"", "1", "default"} {
		m, ok := Lookup(alias)
		if !ok || m.Name != DefaultModel {
			t.Errorf("Lookup(%q) = (%q,%v), want default %q", alias, m.Name, ok, DefaultModel)
		}
	}
	if _, ok := Lookup("does-not-exist"); ok {
		t.Error("Lookup(unknown) should be !ok")
	}
}

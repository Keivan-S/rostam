// SPDX-License-Identifier: Apache-2.0

package analysis

import "testing"

func TestStopwordSet(t *testing.T) {
	s := NewStopwordSet([]string{"the", "a", "is"})
	if !s.Enabled() {
		t.Fatal("expected enabled set")
	}
	for _, w := range []string{"the", "a", "is"} {
		if !s.Contains(w) {
			t.Errorf("Contains(%q) = false, want true", w)
		}
	}
	for _, w := range []string{"cat", "running", "The"} {
		if s.Contains(w) {
			t.Errorf("Contains(%q) = true, want false", w)
		}
	}
}

func TestStopwordSetDisabled(t *testing.T) {
	for _, s := range []*StopwordSet{
		NewStopwordSet(nil),
		NewStopwordSet([]string{}),
		(*StopwordSet)(nil),
		{},
	} {
		if s.Enabled() {
			t.Errorf("empty/nil set should be disabled")
		}
		if s.Contains("the") {
			t.Errorf("disabled set should keep all tokens (Contains returned true)")
		}
	}
}

func TestDefaultEnglishStopwords(t *testing.T) {
	s := DefaultEnglishStopwords()
	if !s.Enabled() {
		t.Fatal("default set should be enabled")
	}
	// Spot-check well-known stopwords are present.
	for _, w := range []string{"the", "and", "is", "of", "a", "to", "in", "that"} {
		if !s.Contains(w) {
			t.Errorf("default set missing stopword %q", w)
		}
	}
	// Content words must not be filtered.
	for _, w := range []string{"vector", "database", "running", "quantum"} {
		if s.Contains(w) {
			t.Errorf("default set wrongly contains content word %q", w)
		}
	}
	if len(EnglishStopwords) < 120 {
		t.Errorf("expected the standard ~150-word English list, got %d", len(EnglishStopwords))
	}
}

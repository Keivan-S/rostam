// SPDX-License-Identifier: Apache-2.0

package analysis

import (
	"reflect"
	"sort"
	"testing"
)

// idFor mirrors the analyzer pipeline for a single already-lowercased word
// (stem → hash) so tests can assert exact term ids without hardcoding numbers.
func idFor(word string, width uint32) uint32 {
	return TermIDMod(Stem(word), width)
}

func TestEnglishAnalyzerAnalyzeSortedUnique(t *testing.T) {
	a := NewEnglishAnalyzer()
	// "running" and "runs" both stem to "run"; stopwords (the, a) dropped.
	got := a.Analyze("The running runner runs a race")
	// Expected stems: run, runner, run, race -> {run, runner, race}.
	want := []uint32{
		idFor("run", 0),
		idFor("runner", 0),
		idFor("race", 0),
	}
	sort.Slice(want, func(i, j int) bool { return want[i] < want[j] })
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Analyze = %v, want %v", got, want)
	}
	// Verify sorted + unique.
	for i := 1; i < len(got); i++ {
		if got[i] <= got[i-1] {
			t.Errorf("Analyze result not strictly increasing at %d: %v", i, got)
		}
	}
}

func TestEnglishAnalyzerEmptyAndStopwordOnly(t *testing.T) {
	a := NewEnglishAnalyzer()
	if got := a.Analyze(""); got != nil {
		t.Errorf("Analyze(\"\") = %v, want nil", got)
	}
	if got := a.Analyze("the a an of and to"); got != nil {
		t.Errorf("Analyze(all-stopwords) = %v, want nil", got)
	}
	if got := a.AnalyzeCounts("   ...!!!   "); got != nil {
		t.Errorf("AnalyzeCounts(no tokens) = %v, want nil", got)
	}
}

func TestEnglishAnalyzerAnalyzeCounts(t *testing.T) {
	a := NewEnglishAnalyzer()
	// "cat cat cats dog the the" -> cat x3 (cat,cat,cats->cat), dog x1, the dropped.
	counts := a.AnalyzeCounts("cat cat cats dog the the")
	catID := idFor("cat", 0)
	dogID := idFor("dog", 0)
	if counts[catID] != 3 {
		t.Errorf("cat tf = %d, want 3", counts[catID])
	}
	if counts[dogID] != 1 {
		t.Errorf("dog tf = %d, want 1", counts[dogID])
	}
	if _, ok := counts[idFor("the", 0)]; ok {
		t.Errorf("stopword 'the' must be excluded from counts")
	}
	if len(counts) != 2 {
		t.Errorf("expected 2 distinct terms, got %d: %v", len(counts), counts)
	}
}

func TestStopwordToggle(t *testing.T) {
	// With stopwords disabled, every token survives (after stemming).
	a := &EnglishAnalyzer{Stopwords: NewStopwordSet(nil)}
	counts := a.AnalyzeCounts("the the the cat")
	theID := idFor("the", 0)
	if counts[theID] != 3 {
		t.Errorf("with stopwords disabled, 'the' tf = %d, want 3", counts[theID])
	}
	if counts[idFor("cat", 0)] != 1 {
		t.Errorf("cat tf = %d, want 1", counts[idFor("cat", 0)])
	}

	// With stopwords enabled, "the" is gone.
	b := NewEnglishAnalyzer()
	if _, ok := b.AnalyzeCounts("the the the cat")[theID]; ok {
		t.Errorf("with stopwords enabled, 'the' must be filtered")
	}
}

func TestHashWidthFolding(t *testing.T) {
	const width = 1024
	a := &EnglishAnalyzer{Stopwords: DefaultEnglishStopwords(), HashWidth: width}
	for _, id := range a.Analyze("vector database search engine") {
		if id >= width {
			t.Errorf("term id %d exceeds hash width %d", id, width)
		}
	}
	// Folded ids equal the explicit mod of the unfolded pipeline.
	wantCat := idFor("cat", width)
	got := a.Analyze("cats")
	if len(got) != 1 || got[0] != wantCat {
		t.Errorf("folded Analyze(cats) = %v, want [%d]", got, wantCat)
	}
}

func TestCountingAnalyzerInterface(t *testing.T) {
	var ca CountingAnalyzer = NewEnglishAnalyzer()
	if ca.AnalyzeCounts("hello world") == nil {
		t.Error("CountingAnalyzer.AnalyzeCounts returned nil for non-empty text")
	}
	var _ Analyzer = ca
}

func TestByName(t *testing.T) {
	// Default name and empty name both resolve to the English analyzer.
	for _, name := range []string{"", DefaultAnalyzerName} {
		a, err := ByName(name)
		if err != nil {
			t.Fatalf("ByName(%q) error: %v", name, err)
		}
		if _, ok := a.(*EnglishAnalyzer); !ok {
			t.Errorf("ByName(%q) = %T, want *EnglishAnalyzer", name, a)
		}
	}
	if _, err := ByName("klingon"); err == nil {
		t.Error("ByName(unknown) should error")
	}
	if DefaultAnalyzerName != "english" {
		t.Errorf("DefaultAnalyzerName = %q, want \"english\"", DefaultAnalyzerName)
	}
}

func TestByNameReturnsFreshInstances(t *testing.T) {
	a, _ := ByName("english")
	b, _ := ByName("english")
	if a == b {
		t.Error("ByName should return independent instances")
	}
}

func TestRegister(t *testing.T) {
	Register("noop", func() Analyzer { return &EnglishAnalyzer{} })
	a, err := ByName("noop")
	if err != nil {
		t.Fatalf("ByName(noop) error: %v", err)
	}
	if a == nil {
		t.Fatal("ByName(noop) returned nil")
	}
	delete(registry, "noop") // keep the global registry clean for other tests.
}

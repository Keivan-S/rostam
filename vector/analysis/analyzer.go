// SPDX-License-Identifier: Apache-2.0

package analysis

import (
	"fmt"
	"sort"
	"sync"
)

// DefaultAnalyzerName is the registry name of the built-in English analyzer.
const DefaultAnalyzerName = "english"

// Analyzer turns raw text into a sorted, de-duplicated slice of term ids. This
// is the minimal surface the query path needs.
type Analyzer interface {
	// Analyze returns the sorted-unique term ids of text.
	Analyze(text string) []uint32
}

// CountingAnalyzer additionally exposes the term-frequency multiset, which the
// BM25 write path consumes (term id → frequency within the document).
type CountingAnalyzer interface {
	Analyzer
	// AnalyzeCounts returns the tf multiset: term id → frequency.
	AnalyzeCounts(text string) map[uint32]uint32
}

// EnglishAnalyzer chains tokenize → lowercase → stopword-filter → Porter2 stem
// → FNV-1a hash. It is configured by an (optional) stopword set and a hash
// width. The zero value is usable but does no stopword filtering; prefer
// NewEnglishAnalyzer.
type EnglishAnalyzer struct {
	// Stopwords filters tokens after lowercasing, before stemming. A nil or
	// disabled set keeps all tokens.
	Stopwords *StopwordSet
	// HashWidth folds term ids modulo this value (0 = full uint32 space).
	HashWidth uint32
}

// NewEnglishAnalyzer returns an EnglishAnalyzer with the default English
// stopword set and no hash-width folding (full uint32 term-id space).
func NewEnglishAnalyzer() *EnglishAnalyzer {
	return &EnglishAnalyzer{
		Stopwords: DefaultEnglishStopwords(),
		HashWidth: 0,
	}
}

// terms runs the full pipeline and returns the ordered (document-order) stream
// of term ids, with repeats preserved.
func (a *EnglishAnalyzer) terms(text string) []uint32 {
	toks := Tokenize(text)
	if len(toks) == 0 {
		return nil
	}
	out := make([]uint32, 0, len(toks))
	for _, tok := range toks {
		// Tokenize already lowercases; stopword filtering operates on the
		// lowercased surface form, before stemming.
		if a.Stopwords.Contains(tok) {
			continue
		}
		stem := Stem(tok)
		out = append(out, TermIDMod(stem, a.HashWidth))
	}
	return out
}

// Analyze returns the sorted-unique term ids of text.
func (a *EnglishAnalyzer) Analyze(text string) []uint32 {
	ids := a.terms(text)
	if len(ids) == 0 {
		return nil
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	// De-duplicate in place.
	w := 0
	for i := 0; i < len(ids); i++ {
		if i == 0 || ids[i] != ids[i-1] {
			ids[w] = ids[i]
			w++
		}
	}
	return ids[:w]
}

// AnalyzeCounts returns the tf multiset (term id → frequency). Stopwords are
// excluded; repeated terms are counted.
func (a *EnglishAnalyzer) AnalyzeCounts(text string) map[uint32]uint32 {
	ids := a.terms(text)
	if len(ids) == 0 {
		return nil
	}
	counts := make(map[uint32]uint32, len(ids))
	for _, id := range ids {
		counts[id]++
	}
	return counts
}

// Constructor builds a fresh Analyzer instance.
type Constructor func() Analyzer

// registryMu guards the registry map. Register takes the write lock and ByName
// the read lock so the global registry is safe for concurrent use (Register is
// a public OSS API surface that may run after concurrent ByName/Analyze use).
var registryMu sync.RWMutex

// registry maps an analyzer name to its constructor.
var registry = map[string]Constructor{
	DefaultAnalyzerName: func() Analyzer { return NewEnglishAnalyzer() },
}

// Register adds (or replaces) an analyzer constructor under name. It is safe for
// concurrent use with ByName.
func Register(name string, c Constructor) {
	registryMu.Lock()
	registry[name] = c
	registryMu.Unlock()
}

// ByName returns a new analyzer for the given registered name. An empty name
// resolves to the default ("english"). Unknown names return an error.
func ByName(name string) (Analyzer, error) {
	if name == "" {
		name = DefaultAnalyzerName
	}
	registryMu.RLock()
	c, ok := registry[name]
	registryMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("analysis: unknown analyzer %q", name)
	}
	return c(), nil
}

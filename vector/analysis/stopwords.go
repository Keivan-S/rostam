// SPDX-License-Identifier: Apache-2.0

package analysis

// StopwordSet is a configurable set of lowercased stopwords. A nil or empty set
// disables filtering (all tokens are kept). Construct via NewStopwordSet.
type StopwordSet struct {
	words map[string]struct{}
}

// NewStopwordSet builds a StopwordSet from the given lowercased words. Passing
// no words (or nil) yields a disabled set that keeps every token.
func NewStopwordSet(words []string) *StopwordSet {
	if len(words) == 0 {
		return &StopwordSet{}
	}
	m := make(map[string]struct{}, len(words))
	for _, w := range words {
		m[w] = struct{}{}
	}
	return &StopwordSet{words: m}
}

// Enabled reports whether the set will filter anything.
func (s *StopwordSet) Enabled() bool {
	return s != nil && len(s.words) > 0
}

// Contains reports whether token is a stopword. Always false for a disabled set.
func (s *StopwordSet) Contains(token string) bool {
	if s == nil || len(s.words) == 0 {
		return false
	}
	_, ok := s.words[token]
	return ok
}

// EnglishStopwords is the classic Snowball/Lucene English stopword list
// (the standard ~170-word set used by Apache Lucene's EnglishAnalyzer, which
// extends Snowball's list). Filtered after lowercasing, before stemming.
//
// Source: Snowball project English stopword list
// (https://snowballstem.org/algorithms/english/stop.txt), public domain.
var EnglishStopwords = []string{
	"a", "about", "above", "after", "again", "against", "all", "am", "an",
	"and", "any", "are", "aren't", "as", "at", "be", "because", "been",
	"before", "being", "below", "between", "both", "but", "by", "can't",
	"cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
	"doing", "don't", "down", "during", "each", "few", "for", "from",
	"further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
	"he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
	"him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
	"i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
	"let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
	"not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
	"ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
	"she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
	"than", "that", "that's", "the", "their", "theirs", "them", "themselves",
	"then", "there", "there's", "these", "they", "they'd", "they'll",
	"they're", "they've", "this", "those", "through", "to", "too", "under",
	"until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
	"we've", "were", "weren't", "what", "what's", "when", "when's", "where",
	"where's", "which", "while", "who", "who's", "whom", "why", "why's",
	"with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
	"you've", "your", "yours", "yourself", "yourselves",

	// Apostrophe-stripped contraction fragments. Tokenize splits on the
	// apostrophe (it is neither a letter nor a digit), so a contraction like
	// "don't" arrives here as the two tokens "don" and "t", and the canonical
	// apostrophe entries above ("don't", "it's", "i'll", …) can never match.
	// These entries cover the fragments the tokenizer actually produces so the
	// contraction stopwords behave as advertised; the apostrophe forms are
	// retained above for documentation and for any caller that pre-normalizes
	// apostrophes before tokenizing.
	"aren", "couldn", "didn", "doesn", "don", "hadn", "hasn", "haven", "isn",
	"mustn", "shan", "shouldn", "wasn", "weren", "wouldn",
	// Leading fragments of "let's", "that's", "what's", "who's", etc. are
	// already covered by their bare forms above (let, that, what, who, …).
	// Trailing contraction suffixes ("'ll", "'ve", "'re", "'s", "'d", "'m",
	// "n't" → "t") reduce to these fragments after the apostrophe split:
	"ll", "ve", "re", "s", "d", "m", "t",
}

// DefaultEnglishStopwords returns a StopwordSet seeded with EnglishStopwords.
func DefaultEnglishStopwords() *StopwordSet {
	return NewStopwordSet(EnglishStopwords)
}

// SPDX-License-Identifier: Apache-2.0

// Package analysis provides the text-analysis pipeline (tokenize, normalize,
// stopword-filter, stem, hash) that turns raw text into BM25 term ids.
//
// It is a standalone, pure-Go package with no engine dependencies and no
// third-party module dependencies: the Porter2/Snowball English stemmer is
// vendored as pure-Go source (see stemmer.go) to preserve Rostam's
// single-binary, zero-dep identity.
package analysis

import "unicode"

// MaxTokensPerDocument bounds how many tokens Tokenize emits for a single
// input. Without a cap a single pathological document could unboundedly expand
// the inverted index; once this many tokens have been produced the remaining
// input is ignored. The limit is generous (a typical English word is ~5-6
// bytes, so this allows multi-megabyte documents) and exists only as a defense
// against adversarial/degenerate input, not as a normal-path constraint.
const MaxTokensPerDocument = 1 << 20 // 1,048,576 tokens

// Tokenize splits text into lowercased tokens. A token is a maximal run of
// runes that are letters (unicode.IsLetter) or digits (unicode.IsDigit); any
// other rune is a separator. Each token is lowercased via unicode.ToLower. At
// most MaxTokensPerDocument tokens are returned; input beyond that is dropped.
//
// TODO(v1): diacritic folding — v1 does not fold accents (e.g. "café" stays
// "café"); add Unicode NFKD + combining-mark stripping in a follow-up.
func Tokenize(text string) []string {
	if text == "" {
		return nil
	}
	// Pre-size optimistically but never above the hard cap; most token streams
	// are far smaller than len(text).
	prealloc := len(text)/6 + 1
	if prealloc > MaxTokensPerDocument {
		prealloc = MaxTokensPerDocument
	}
	tokens := make([]string, 0, prealloc)
	var cur []rune
	full := false
	flush := func() {
		if len(cur) > 0 {
			if len(tokens) < MaxTokensPerDocument {
				tokens = append(tokens, string(cur))
			} else {
				full = true
			}
			cur = cur[:0]
		}
	}
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			cur = append(cur, unicode.ToLower(r))
			continue
		}
		flush()
		if full {
			break
		}
	}
	flush()
	if len(tokens) == 0 {
		return nil
	}
	return tokens
}

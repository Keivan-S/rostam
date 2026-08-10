// SPDX-License-Identifier: Apache-2.0

package analysis

import "unicode"

// Normalize lowercases a single token via unicode.ToLower, rune by rune.
//
// Tokenize already lowercases as it scans, so this is exposed mainly for
// callers that normalize pre-split tokens (e.g. query terms supplied by a
// caller) and to keep the lowercasing step an explicit, testable unit.
//
// TODO(v1): diacritic folding — see Tokenize.
func Normalize(token string) string {
	// Fast path: ASCII-only tokens that are already lowercase need no copy.
	needs := false
	for _, r := range token {
		if r != unicode.ToLower(r) {
			needs = true
			break
		}
	}
	if !needs {
		return token
	}
	out := make([]rune, 0, len(token))
	for _, r := range token {
		out = append(out, unicode.ToLower(r))
	}
	return string(out)
}

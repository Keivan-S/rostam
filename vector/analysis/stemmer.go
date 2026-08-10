// SPDX-License-Identifier: Apache-2.0

package analysis

// Porter2 (Snowball English) stemmer — vendored, pure Go.
//
// This is a from-scratch Go implementation of the Porter2 / "english" Snowball
// stemming algorithm as specified at:
//
//	https://snowballstem.org/algorithms/english/stemmer.html
//
// The Snowball algorithm and its reference materials are released by Martin
// Porter and Richard Boulton under the BSD 3-Clause license (the standard
// Snowball license, https://snowballstem.org/license.html), which permits this
// vendored, attribution-preserving reimplementation. It is reproduced here as
// pure Go source so the package adds no third-party module dependency.
//
// BSD 3-Clause License (Snowball):
//
//	Copyright (c) 2001, Dr Martin Porter
//	Copyright (c) 2004,2005, Richard Boulton
//	Copyright (c) 2013, Yoshiki Shibukawa
//	Copyright (c) 2006,2007,2009,2010,2011,2014-2019, Olly Betts
//	All rights reserved.
//
//	Redistribution and use in source and binary forms, with or without
//	modification, are permitted provided that the conditions of the BSD
//	3-Clause license are met.
//
// Fixture-verified in stemmer_test.go against canonical Snowball English
// vocabulary→stem pairs.

import "strings"

// Stem returns the Porter2 stem of an already-lowercased word. Words of two or
// fewer letters are returned unchanged (per the algorithm). Input is assumed
// lowercased ASCII-ish; non-ASCII runes are passed through untouched, which is
// fine for the English algorithm.
func Stem(word string) string {
	if len(word) <= 2 {
		return word
	}

	w := []byte(word)

	// Exceptional forms that map directly to an invariant stem (step 0 of the
	// "english" algorithm).
	if s, ok := exceptional1[string(w)]; ok {
		return s
	}

	// Remove an initial apostrophe.
	if w[0] == '\'' {
		w = w[1:]
	}
	if len(w) == 0 {
		return word
	}

	w = markVowelY(w)

	// R1/R2 are computed ONCE on the marked word, before any suffix removal,
	// exactly as the Snowball reference does. Every later step only shortens
	// the word from the right, so these left-anchored indices stay valid: a
	// region boundary already crossed by the start of a suffix stays crossed.
	r1, r2 := computeR1R2(w)

	w = step0(w)
	w = step1a(w)

	// Invariant words after step 1a.
	if exceptional2[string(w)] {
		return string(unmarkY(w))
	}

	w = step1b(w, r1)
	w = step1c(w)
	w = step2(w, r1)
	w = step3(w, r1, r2)
	w = step4(w, r2)
	w = step5(w, r1, r2)

	return string(unmarkY(w))
}

// markVowelY uppercases each 'y' that is acting as a consonant (at the start of
// the word, or after a vowel) to 'Y' so the region/vowel logic treats it
// correctly. The Snowball spec calls these the consonantal y positions.
func markVowelY(w []byte) []byte {
	for i := 0; i < len(w); i++ {
		if w[i] != 'y' {
			continue
		}
		if i == 0 {
			w[i] = 'Y'
		} else if isPlainVowel(w[i-1]) {
			w[i] = 'Y'
		}
	}
	return w
}

// isPlainVowel reports the aeiou-set membership (excludes y entirely).
func isPlainVowel(b byte) bool {
	switch b {
	case 'a', 'e', 'i', 'o', 'u':
		return true
	}
	return false
}

// unmarkY restores 'Y' markers back to 'y'.
func unmarkY(w []byte) []byte {
	for i := range w {
		if w[i] == 'Y' {
			w[i] = 'y'
		}
	}
	return w
}

// isVowelByte treats both 'y' and our 'Y' marker correctly: a marked 'Y' is a
// consonant; a plain 'y' is a vowel.
func isVowelByte(b byte) bool {
	switch b {
	case 'a', 'e', 'i', 'o', 'u', 'y':
		return true
	}
	return false
}

// computeR1R2 derives the R1 and R2 region start indices.
//
// R1 is the region after the first non-vowel following a vowel; R2 is the
// region after the first non-vowel following a vowel in R1. The "english"
// algorithm pins R1 to position 5 for words beginning gener/commun/arsen.
func computeR1R2(w []byte) (int, int) {
	s := string(w)
	var r1 int
	switch {
	case strings.HasPrefix(s, "gener"):
		r1 = 5
	case strings.HasPrefix(s, "commun"):
		r1 = 6
	case strings.HasPrefix(s, "arsen"):
		r1 = 5
	default:
		r1 = regionStart(w, 0)
	}
	r2 := regionStart(w, r1)
	if r1 > len(w) {
		r1 = len(w)
	}
	if r2 > len(w) {
		r2 = len(w)
	}
	return r1, r2
}

// regionStart returns the index after the first non-vowel that follows a vowel,
// scanning from `from`.
func regionStart(w []byte, from int) int {
	i := from
	n := len(w)
	for i < n && !isVowelByte(w[i]) {
		i++
	}
	for i < n && isVowelByte(w[i]) {
		i++
	}
	// i now points at the first non-vowel after a vowel run (or end).
	if i < n {
		i++
	}
	return i
}

// step0: remove trailing 's, 's', '.
func step0(w []byte) []byte {
	s := string(w)
	switch {
	case strings.HasSuffix(s, "'s'"):
		return w[:len(w)-3]
	case strings.HasSuffix(s, "'s"):
		return w[:len(w)-2]
	case strings.HasSuffix(s, "'"):
		return w[:len(w)-1]
	}
	return w
}

// step1a handles plural / possessive-like s endings.
func step1a(w []byte) []byte {
	s := string(w)
	switch {
	case strings.HasSuffix(s, "sses"):
		return w[:len(w)-2] // sses -> ss
	case strings.HasSuffix(s, "ied"), strings.HasSuffix(s, "ies"):
		// ied/ies -> i if preceded by more than one letter, else ie
		stem := w[:len(w)-3]
		if len(stem) > 1 {
			return append(stem, 'i')
		}
		return append(stem, 'i', 'e')
	case strings.HasSuffix(s, "us"), strings.HasSuffix(s, "ss"):
		return w
	case strings.HasSuffix(s, "s"):
		// delete final s if the preceding part contains a vowel not
		// immediately before the s.
		if len(w) >= 2 {
			for i := 0; i < len(w)-2; i++ {
				if isVowelByte(w[i]) {
					return w[:len(w)-1]
				}
			}
		}
		return w
	}
	return w
}

// step1b handles ed/edly/ing/ingly endings.
func step1b(w []byte, r1 int) []byte {
	s := string(w)
	// eed / eedly -> ee if in R1.
	for _, suf := range []string{"eedly", "eed"} {
		if strings.HasSuffix(s, suf) {
			if len(w)-len(suf) >= r1 {
				return append(w[:len(w)-len(suf)], 'e', 'e')
			}
			return w
		}
	}
	// ed / edly / ing / ingly: delete if the preceding word part contains a vowel.
	for _, suf := range []string{"ingly", "edly", "ing", "ed"} {
		if strings.HasSuffix(s, suf) {
			stem := w[:len(w)-len(suf)]
			if !containsVowel(stem) {
				return w
			}
			return step1bPost(stem, r1)
		}
	}
	return w
}

// step1bPost applies the post-deletion fix-ups from step1b (Snowball spec):
//   - if the word ends at/bl/iz, add e; else
//   - if it ends with a double preceded by something other than exactly a, e
//     or o, remove the last letter; else
//   - if it does not end with a double and is short, add e.
func step1bPost(stem []byte, r1 int) []byte {
	s := string(stem)
	switch {
	case strings.HasSuffix(s, "at"), strings.HasSuffix(s, "bl"), strings.HasSuffix(s, "iz"):
		return append(stem, 'e')
	case endsDoubleConsonant(stem):
		// Remove the last letter only if the double is preceded by something
		// other than exactly a, e or o. "Exactly a/e/o" means the entire word
		// preceding the double is that single vowel: words like add (a+dd),
		// ebb (e+bb), egg (e+gg) keep their double; abett/fitt do not.
		if len(stem) == 3 {
			switch stem[0] {
			case 'a', 'e', 'o':
				return stem
			}
		}
		return stem[:len(stem)-1]
	case isShortWord(stem, r1):
		return append(stem, 'e')
	}
	return stem
}

func containsVowel(w []byte) bool {
	for i := range w {
		if isVowelByte(w[i]) {
			return true
		}
	}
	return false
}

// endsDoubleConsonant reports a trailing doubled bb/dd/ff/gg/mm/nn/pp/rr/tt.
func endsDoubleConsonant(w []byte) bool {
	if len(w) < 2 {
		return false
	}
	a, b := w[len(w)-2], w[len(w)-1]
	if a != b {
		return false
	}
	switch a {
	case 'b', 'd', 'f', 'g', 'm', 'n', 'p', 'r', 't':
		return true
	}
	return false
}

// isShortSyllable reports whether the two bytes ending at the word form a short
// syllable: a vowel followed by a non-vowel that is not w, x, or Y, OR a vowel
// at the start of the word followed by a non-vowel.
func isShortSyllable(w []byte, i int) bool {
	// short syllable ending at position i (the syllable is w[i-1..i]).
	if i == 1 {
		// (vowel)(non-vowel) at the very start.
		return isVowelByte(w[0]) && !isVowelByte(w[1])
	}
	if i < 2 {
		return false
	}
	a, b, c := w[i-2], w[i-1], w[i]
	if isVowelByte(a) {
		return false
	}
	if !isVowelByte(b) {
		return false
	}
	if isVowelByte(c) {
		return false
	}
	switch c {
	case 'w', 'x', 'Y':
		return false
	}
	return true
}

// isShortWord reports whether w is "short" per the Snowball spec: it ends in a
// short syllable AND R1 is null (R1 is the empty region, i.e. r1 >= len(w)).
func isShortWord(w []byte, r1 int) bool {
	n := len(w)
	if n == 0 {
		return false
	}
	if r1 < n {
		return false // R1 is not null
	}
	return endsShortSyllable(w)
}

// endsShortSyllable reports whether w ends in a short syllable (cases a/b/c of
// the spec). Case (c) is the literal word "past".
func endsShortSyllable(w []byte) bool {
	if string(w) == "past" {
		return true
	}
	n := len(w)
	if n < 2 {
		return false
	}
	return isShortSyllable(w, n-1)
}

// step1c: replace suffix y or Y by i if preceded by a non-vowel and the word is
// longer than two letters.
func step1c(w []byte) []byte {
	n := len(w)
	if n < 3 {
		return w
	}
	last := w[n-1]
	if last != 'y' && last != 'Y' {
		return w
	}
	if isVowelByte(w[n-2]) {
		return w
	}
	w[n-1] = 'i'
	return w
}

// step2 suffix table (longest match wins). Each replacement is applied only if
// the suffix lies in R1.
var step2Suffixes = []struct{ suf, rep string }{
	{"ational", "ate"}, {"tional", "tion"}, {"enci", "ence"}, {"anci", "ance"},
	{"abli", "able"}, {"entli", "ent"}, {"izer", "ize"}, {"ization", "ize"},
	{"ation", "ate"}, {"ator", "ate"}, {"alism", "al"}, {"aliti", "al"},
	{"alli", "al"}, {"fulness", "ful"}, {"ousli", "ous"}, {"ousness", "ous"},
	{"iveness", "ive"}, {"iviti", "ive"}, {"biliti", "ble"}, {"bli", "ble"},
	{"ogist", "og"}, {"fulli", "ful"}, {"lessli", "less"},
}

func step2(w []byte, r1 int) []byte {
	s := string(w)
	// special: ogi -> og only if preceded by l and in R1
	if strings.HasSuffix(s, "ogi") {
		if len(w)-3 >= r1 && len(w) >= 4 && w[len(w)-4] == 'l' {
			return w[:len(w)-1]
		}
	}
	// special: li -> delete if in R1 and preceded by a valid li-ending letter.
	if strings.HasSuffix(s, "li") {
		if len(w)-2 >= r1 && len(w) >= 3 && isLiEnding(w[len(w)-3]) {
			return w[:len(w)-2]
		}
	}
	for _, e := range step2Suffixes {
		if strings.HasSuffix(s, e.suf) {
			if len(w)-len(e.suf) >= r1 {
				return append(w[:len(w)-len(e.suf)], e.rep...)
			}
			return w
		}
	}
	return w
}

func isLiEnding(b byte) bool {
	switch b {
	case 'c', 'd', 'e', 'g', 'h', 'k', 'm', 'n', 'r', 't':
		return true
	}
	return false
}

var step3Suffixes = []struct{ suf, rep string }{
	{"ational", "ate"}, {"tional", "tion"}, {"alize", "al"}, {"icate", "ic"},
	{"iciti", "ic"}, {"ical", "ic"}, {"ful", ""}, {"ness", ""},
}

func step3(w []byte, r1, r2 int) []byte {
	s := string(w)
	// ative -> delete if in R2.
	if strings.HasSuffix(s, "ative") {
		if len(w)-5 >= r2 {
			return w[:len(w)-5]
		}
		return w
	}
	for _, e := range step3Suffixes {
		if strings.HasSuffix(s, e.suf) {
			if len(w)-len(e.suf) >= r1 {
				return append(w[:len(w)-len(e.suf)], e.rep...)
			}
			return w
		}
	}
	return w
}

var step4Suffixes = []string{
	"al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
	"ment", "ent", "ism", "ate", "iti", "ous", "ive", "ize",
}

func step4(w []byte, r2 int) []byte {
	s := string(w)
	// ion -> delete if in R2 and preceded by s or t.
	if strings.HasSuffix(s, "ion") {
		if len(w)-3 >= r2 && len(w) >= 4 {
			p := w[len(w)-4]
			if p == 's' || p == 't' {
				return w[:len(w)-3]
			}
		}
		return w
	}
	for _, suf := range step4Suffixes {
		if strings.HasSuffix(s, suf) {
			if len(w)-len(suf) >= r2 {
				return w[:len(w)-len(suf)]
			}
			return w
		}
	}
	return w
}

// step5: delete a final e (if in R2, or in R1 and not preceded by a short
// syllable); delete a final l if in R2 and preceded by l.
func step5(w []byte, r1, r2 int) []byte {
	n := len(w)
	if n == 0 {
		return w
	}
	last := w[n-1]
	if last == 'e' {
		if n-1 >= r2 {
			return w[:n-1]
		}
		if n-1 >= r1 && !endsShortSyllableBefore(w, n-1) {
			return w[:n-1]
		}
		return w
	}
	if last == 'l' {
		if n-1 >= r2 && n >= 2 && w[n-2] == 'l' {
			return w[:n-1]
		}
	}
	return w
}

// endsShortSyllableBefore reports whether the word w[:cut] ends in a short
// syllable (including the literal "past" case c).
func endsShortSyllableBefore(w []byte, cut int) bool {
	if cut < 1 {
		return false
	}
	return endsShortSyllable(w[:cut])
}

// Exceptional forms (Snowball "english" algorithm, special handling).
var exceptional1 = map[string]string{
	"skis": "ski", "skies": "sky", "dying": "die", "lying": "lie",
	"tying": "tie", "idly": "idl", "gently": "gentl", "ugly": "ugli",
	"early": "earli", "only": "onli", "singly": "singl",
	"sky": "sky", "news": "news", "howe": "howe", "atlas": "atlas",
	"cosmos": "cosmos", "bias": "bias", "andes": "andes",
}

// Words that should be left invariant after step 1a.
var exceptional2 = map[string]bool{
	"inning": true, "outing": true, "canning": true, "herring": true,
	"earring": true, "proceed": true, "exceed": true, "succeed": true,
}

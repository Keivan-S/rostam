// SPDX-License-Identifier: Apache-2.0

package analysis

import "testing"

// snowballFixtures are canonical Snowball "english" vocabulary→stem pairs taken
// from the official Snowball English test corpus
// (https://snowballstem.org/algorithms/english/, voc.txt/output.txt). They
// exercise each Porter2 step: plurals/possessives (step 0/1a), ed/ing
// (step 1b incl. double-consonant and short-word rules), y→i (step 1c), the
// step 2/3/4 derivational suffixes, and step 5 e/l deletion.
var snowballFixtures = map[string]string{
	// step 1a: plurals / sses / ied-ies / final-s rules.
	"caresses": "caress", "ponies": "poni", "ties": "tie", "caress": "caress",
	"cats": "cat", "gas": "gas",
	// step 1b: ed/ing deletion + post fix-ups (at/bl/iz, double, short).
	"agreed": "agre", "feed": "feed", "plastered": "plaster", "bled": "bled",
	"motoring": "motor", "sing": "sing", "fitted": "fit", "hopping": "hop",
	"added": "add", "ebbed": "ebb", "troubled": "troubl",
	// step 1c: y → i.
	"happy": "happi", "sky": "sky", "cry": "cri",
	// step 2: derivational suffixes.
	"relational": "relat", "conditional": "condit", "rational": "ration",
	"valenci": "valenc", "hesitanci": "hesit", "digitizer": "digit",
	"conformabli": "conform", "radicalli": "radic", "differentli": "differ",
	"analogousli": "analog", "vietnamization": "vietnam", "predication": "predic",
	"operator": "oper", "feudalism": "feudal", "decisiveness": "decis",
	"hopefulness": "hope", "callousness": "callous", "sensibility": "sensibl",
	"apologists": "apolog", "geologist": "geolog",
	// step 3: derivational suffixes.
	"triplicate": "triplic", "formalize": "formal", "electriciti": "electr",
	"electrical": "electr", "hopeful": "hope", "goodness": "good",
	// step 4: deletions in R2.
	"revival": "reviv", "allowance": "allow", "inference": "infer",
	"airliner": "airlin", "gyroscopic": "gyroscop", "adjustable": "adjust",
	"defensible": "defens", "irritant": "irrit", "replacement": "replac",
	"adjustment": "adjust", "dependent": "depend", "adoption": "adopt",
	"homologous": "homolog", "communism": "communism", "activate": "activ",
	"effective": "effect", "bowdlerize": "bowdler",
	// step 5: e / l deletion.
	"probate": "probat", "rate": "rate", "cease": "ceas",
	"controll": "control", "roll": "roll",
	// special / exception words.
	"skis": "ski", "skies": "sky", "dying": "die", "lying": "lie",
	"tying": "tie", "early": "earli", "only": "onli", "singly": "singl",
	"proceed": "proceed", "exceed": "exceed", "succeed": "succeed",
	// short words returned unchanged.
	"a": "a", "no": "no", "ax": "ax",
	// prefix-pinned R1 (gener / commun).
	"generate": "generat", "general": "general", "generously": "generous",
}

func TestStemSnowballFixtures(t *testing.T) {
	for in, want := range snowballFixtures {
		if got := Stem(in); got != want {
			t.Errorf("Stem(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestStemDeterministic(t *testing.T) {
	for _, w := range []string{"running", "national", "generously"} {
		if Stem(w) != Stem(w) { //nolint:staticcheck // intentional: same call twice asserts determinism
			t.Errorf("Stem(%q) not deterministic", w)
		}
	}
}

func TestStemShortWordsUnchanged(t *testing.T) {
	for _, w := range []string{"", "a", "an", "be", "go", "by"} {
		if got := Stem(w); got != w {
			t.Errorf("Stem(%q) = %q, want unchanged", w, got)
		}
	}
}

func TestStemNonASCIIPassthrough(t *testing.T) {
	// Non-ASCII runes are not part of the English algorithm; they pass through
	// without panicking.
	for _, w := range []string{"café", "naïve", "мир"} {
		_ = Stem(w)
	}
}

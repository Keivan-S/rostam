// SPDX-License-Identifier: Apache-2.0

package rag

import "testing"

func words(s string) int {
	n := 0
	for _, f := range splitWords(s) {
		_ = f
		n++
	}
	return n
}

func TestSplitTextEmpty(t *testing.T) {
	if got := SplitText("", 10, 2); got != nil {
		t.Fatalf("empty input: got %v, want nil", got)
	}
	if got := SplitText("   \n\t ", 10, 2); got != nil {
		t.Fatalf("whitespace input: got %v, want nil", got)
	}
}

func TestSplitTextSingleShortChunk(t *testing.T) {
	got := SplitText("hello world foo", 10, 2)
	if len(got) != 1 || got[0].Content != "hello world foo" || got[0].Index != 0 {
		t.Fatalf("got %+v", got)
	}
}

func TestSplitTextOverlap(t *testing.T) {
	// 12 words, target 5, overlap 2 => chunks advance by 3 words.
	body := "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12"
	got := SplitText(body, 5, 2)
	if len(got) < 2 {
		t.Fatalf("want >=2 chunks, got %d (%+v)", len(got), got)
	}
	if got[0].Index != 0 || got[1].Index != 1 {
		t.Fatalf("indices not sequential: %+v", got)
	}
	// consecutive chunks share the overlap tail/head word.
	if last := splitWords(got[0].Content); last[len(last)-1] != splitWords(got[1].Content)[1] {
		t.Fatalf("expected 2-word overlap between chunk 0 and 1: %q / %q", got[0].Content, got[1].Content)
	}
}

func TestSplitTextPrefersParagraphBoundary(t *testing.T) {
	body := "alpha beta gamma\n\ndelta epsilon zeta"
	got := SplitText(body, 4, 1) // each paragraph is 3 words, target 4
	if len(got) != 2 {
		t.Fatalf("want a chunk per paragraph, got %d: %+v", len(got), got)
	}
	if got[0].Content != "alpha beta gamma" || got[1].Content != "delta epsilon zeta" {
		t.Fatalf("paragraph boundaries not respected: %+v", got)
	}
}

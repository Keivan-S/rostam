// SPDX-License-Identifier: Apache-2.0
//go:build localembed

package local

import (
	"os"
	"path/filepath"
	"testing"
)

func writeVocab(t *testing.T, lines []string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "vocab.txt")
	if err := os.WriteFile(p, []byte(join(lines, "\n")), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}
func join(s []string, sep string) string {
	out := ""
	for i, x := range s {
		if i > 0 {
			out += sep
		}
		out += x
	}
	return out
}

func TestEncodeWordPiece(t *testing.T) {
	// ids:      0     1     2     3     4       5     6      7
	vocab := []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "play", "##ing", "hello", "world"}
	p := writeVocab(t, vocab)
	tok, err := NewTokenizer(p, true)
	if err != nil {
		t.Fatal(err)
	}
	ids, mask := tok.Encode("playing hello", MaxSeqLen)
	// [CLS] play ##ing hello [SEP]
	want := []int64{2, 4, 5, 6, 3}
	if len(ids) != len(want) {
		t.Fatalf("ids=%v want len %d", ids, len(want))
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("ids=%v want %v", ids, want)
		}
	}
	for _, m := range mask {
		if m != 1 {
			t.Fatalf("mask has non-1 for real tokens: %v", mask)
		}
	}
}

func TestEncodeUnknownAndTruncate(t *testing.T) {
	vocab := []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello"}
	tok, _ := NewTokenizer(writeVocab(t, vocab), true)
	ids, _ := tok.Encode("hello zzz", MaxSeqLen) // zzz -> [UNK]=1
	want := []int64{2, 4, 1, 3}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("ids=%v want %v", ids, want)
		}
	}
	// Truncation: maxLen=3 keeps [CLS] + 1 token + [SEP].
	ids2, _ := tok.Encode("hello hello hello", 3)
	if len(ids2) != 3 || ids2[0] != 2 || ids2[2] != 3 {
		t.Fatalf("truncated ids=%v want [2 4 3]", ids2)
	}
}

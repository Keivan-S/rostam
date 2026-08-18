// SPDX-License-Identifier: Apache-2.0
//go:build localembed

package local

import (
	"bufio"
	"os"
	"strings"
	"unicode"
)

const MaxSeqLen = 512

type Tokenizer struct {
	vocab     map[string]int64
	lowerCase bool
	clsID     int64
	sepID     int64
	unkID     int64
}

func NewTokenizer(vocabPath string, lowerCase bool) (*Tokenizer, error) {
	f, err := os.Open(vocabPath)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	vocab := map[string]int64{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	var i int64
	for sc.Scan() {
		vocab[sc.Text()] = i
		i++
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	get := func(k string) int64 {
		if v, ok := vocab[k]; ok {
			return v
		}
		return -1
	}
	t := &Tokenizer{vocab: vocab, lowerCase: lowerCase, clsID: get("[CLS]"), sepID: get("[SEP]"), unkID: get("[UNK]")}
	if t.clsID < 0 || t.sepID < 0 || t.unkID < 0 {
		return nil, os.ErrInvalid // vocab missing required special tokens
	}
	return t, nil
}

// basicTokens splits on whitespace and separates punctuation, mirroring BERT's
// BasicTokenizer (control-char strip, optional lowercase). CJK handling is out
// of scope for the MVP English models.
func (t *Tokenizer) basicTokens(text string) []string {
	if t.lowerCase {
		text = strings.ToLower(text)
	}
	var out []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
		}
	}
	for _, r := range text {
		switch {
		case unicode.IsControl(r):
			// drop
		case unicode.IsSpace(r):
			flush()
		case unicode.IsPunct(r) || unicode.IsSymbol(r):
			flush()
			out = append(out, string(r))
		default:
			cur.WriteRune(r)
		}
	}
	flush()
	return out
}

// wordPiece greedily matches the longest vocab prefix, using "##" for
// continuations, falling back to [UNK] for the whole word.
func (t *Tokenizer) wordPiece(word string) []int64 {
	runes := []rune(word)
	var out []int64
	start := 0
	for start < len(runes) {
		end := len(runes)
		var curID int64 = -1
		for start < end {
			sub := string(runes[start:end])
			if start > 0 {
				sub = "##" + sub
			}
			if id, ok := t.vocab[sub]; ok {
				curID = id
				break
			}
			end--
		}
		if curID < 0 {
			return []int64{t.unkID} // whole word is unknown
		}
		out = append(out, curID)
		start = end
	}
	return out
}

// Encode returns input_ids and attention_mask for one text: [CLS] tokens [SEP],
// truncated so the total length never exceeds maxLen.
func (t *Tokenizer) Encode(text string, maxLen int) (ids []int64, mask []int64) {
	if maxLen < 2 {
		maxLen = 2
	}
	ids = append(ids, t.clsID)
	for _, w := range t.basicTokens(text) {
		for _, id := range t.wordPiece(w) {
			if len(ids) >= maxLen-1 { // reserve room for [SEP]
				goto done
			}
			ids = append(ids, id)
		}
	}
done:
	ids = append(ids, t.sepID)
	mask = make([]int64, len(ids))
	for i := range mask {
		mask[i] = 1
	}
	return ids, mask
}

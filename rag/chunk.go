// SPDX-License-Identifier: Apache-2.0

package rag

import "strings"

// Chunk is one unit of text stored as a single document in the corpus.
type Chunk struct {
	Content string
	Index   int
}

// splitWords splits on any whitespace, dropping empties.
func splitWords(s string) []string { return strings.Fields(s) }

// SplitText splits s into overlapping chunks of about targetTokens words each,
// preferring blank-line paragraph boundaries. See the interface block above.
func SplitText(s string, targetTokens, overlap int) []Chunk {
	if targetTokens <= 0 {
		targetTokens = 512
	}
	if overlap < 0 || overlap >= targetTokens {
		overlap = 0
	}
	paras := splitParagraphs(s)
	if len(paras) == 0 {
		return nil
	}
	var chunks []Chunk
	var cur []string
	flush := func(carryOverlap bool) {
		if len(cur) == 0 {
			return
		}
		chunks = append(chunks, Chunk{Content: strings.Join(cur, " "), Index: len(chunks)})
		if carryOverlap && overlap > 0 && len(cur) > overlap {
			cur = append([]string(nil), cur[len(cur)-overlap:]...)
		} else {
			cur = nil
		}
	}
	for _, p := range paras {
		w := splitWords(p)
		if len(w) == 0 {
			continue
		}
		// A paragraph bigger than the target: hard-split it by word count.
		if len(w) > targetTokens {
			flush(true)
			for len(w) > 0 {
				n := targetTokens - len(cur)
				if n > len(w) {
					n = len(w)
				}
				cur = append(cur, w[:n]...)
				w = w[n:]
				if len(cur) >= targetTokens {
					flush(true)
				}
			}
			continue
		}
		// Adding this paragraph would overflow: close the current chunk first,
		// so we break on the paragraph boundary rather than mid-paragraph.
		if len(cur)+len(w) > targetTokens {
			flush(false)
		}
		cur = append(cur, w...)
	}
	// Final flush without carrying overlap forward.
	if len(cur) > 0 {
		chunks = append(chunks, Chunk{Content: strings.Join(cur, " "), Index: len(chunks)})
	}
	return chunks
}

// splitParagraphs splits on blank lines and trims each paragraph.
func splitParagraphs(s string) []string {
	raw := strings.Split(strings.ReplaceAll(s, "\r\n", "\n"), "\n\n")
	out := make([]string, 0, len(raw))
	for _, p := range raw {
		if t := strings.TrimSpace(p); t != "" {
			out = append(out, t)
		}
	}
	return out
}

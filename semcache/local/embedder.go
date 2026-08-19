// SPDX-License-Identifier: Apache-2.0
//go:build localembed

package local

import (
	"context"
	"math"
	"net/http"

	"github.com/rostamlabs/rostam/semcache/localcatalog"
)

type Embedder struct {
	spec    localcatalog.ModelSpec
	tok     *Tokenizer
	session *ortSession
}

func NewEmbedder(ctx context.Context, spec localcatalog.ModelSpec, root, libPath string, hc *http.Client) (*Embedder, error) {
	onnxPath, vocabPath, err := Ensure(ctx, spec, root, hc)
	if err != nil {
		return nil, err
	}
	tok, err := NewTokenizer(vocabPath, spec.LowerCase, spec.ClsToken, spec.SepToken, "")
	if err != nil {
		return nil, err
	}
	sess, err := newORTSession(onnxPath, libPath)
	if err != nil {
		return nil, err
	}
	return &Embedder{spec: spec, tok: tok, session: sess}, nil
}

func (e *Embedder) Model() string { return "local:" + e.spec.Name }
func (e *Embedder) Dim() int      { return e.spec.Dim }
func (e *Embedder) Close() error  { return e.session.close() }

func (e *Embedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}
	// Tokenize + right-pad to the batch's longest sequence.
	idsRows := make([][]int64, len(texts))
	maskRows := make([][]int64, len(texts))
	seqLen := 0
	for i, t := range texts {
		idsRows[i], maskRows[i] = e.tok.Encode(t, MaxSeqLen)
		if len(idsRows[i]) > seqLen {
			seqLen = len(idsRows[i])
		}
	}
	batch := len(texts)
	flatIDs := make([]int64, batch*seqLen)
	flatMask := make([]int64, batch*seqLen)
	flatType := make([]int64, batch*seqLen) // zeros
	for i := 0; i < batch; i++ {
		copy(flatIDs[i*seqLen:], idsRows[i])
		copy(flatMask[i*seqLen:], maskRows[i])
	}

	hidden, hdim, err := e.session.run(flatIDs, flatMask, flatType, batch, seqLen)
	if err != nil {
		return nil, err
	}

	out := make([][]float32, batch)
	for i := 0; i < batch; i++ {
		var v []float32
		switch e.spec.Pooling {
		case localcatalog.PoolingCLS:
			v = clsPool(hidden, i, seqLen, hdim)
		default:
			v = meanPool(hidden, maskRows[i], i, seqLen, hdim)
		}
		normalize(v)
		out[i] = v
	}
	return out, nil
}

func clsPool(hidden []float32, row, seqLen, hdim int) []float32 {
	base := row * seqLen * hdim // token 0 of this row
	v := make([]float32, hdim)
	copy(v, hidden[base:base+hdim])
	return v
}

func meanPool(hidden []float32, mask []int64, row, seqLen, hdim int) []float32 {
	v := make([]float32, hdim)
	var n float32
	for tk := 0; tk < seqLen && tk < len(mask); tk++ {
		if mask[tk] == 0 {
			continue
		}
		n++
		base := (row*seqLen + tk) * hdim
		for d := 0; d < hdim; d++ {
			v[d] += hidden[base+d]
		}
	}
	if n > 0 {
		for d := range v {
			v[d] /= n
		}
	}
	return v
}

func normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	inv := float32(1.0 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
}

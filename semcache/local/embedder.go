// SPDX-License-Identifier: Apache-2.0

// Package local provides an in-process, pure-Go text embedder backed by rembed
// (github.com/rostamlabs/rembed). rembed needs no cgo and no ONNX Runtime: it
// downloads the model from the Hugging Face Hub, tokenizes, runs the transformer
// forward pass, pools, and L2-normalizes — all in Go — so local embedding is
// compiled into every build with no shared library to install.
package local

import (
	"context"
	"fmt"

	"github.com/rostamlabs/rembed"
)

// Embedder adapts a rembed engine to semcache.Embedder. Model() is prefixed with
// "local:" so its scope-key stamp never collides with a hosted model that
// happens to share the same underlying name.
type Embedder struct {
	label string
	eng   *rembed.Embedder
}

// New loads the model identified by ref and returns it as a semcache.Embedder
// named "local:<label>". ref is a Hugging Face model id (e.g.
// "sentence-transformers/all-MiniLM-L6-v2"), a local model directory, or an
// "hf:"-prefixed id to force a Hub fetch. When expectedDim > 0 the loaded
// model's output dimension must equal it, guarding against catalog drift.
func New(ref, label string, expectedDim int) (*Embedder, error) {
	eng, err := rembed.Load(ref)
	if err != nil {
		return nil, fmt.Errorf("load local embedding model %q: %w", ref, err)
	}
	if expectedDim > 0 && eng.Dim() != expectedDim {
		_ = eng.Close()
		return nil, fmt.Errorf("local embedding model %q has dim %d, catalog expects %d", ref, eng.Dim(), expectedDim)
	}
	return &Embedder{label: label, eng: eng}, nil
}

func (e *Embedder) Model() string { return "local:" + e.label }
func (e *Embedder) Dim() int      { return e.eng.Dim() }
func (e *Embedder) Close() error  { return e.eng.Close() }

// Embed returns one L2-normalized vector per text. ctx bounds scheduling
// between per-text forward passes — rembed checks ctx.Err() before each — but it
// does not interrupt a forward pass already in flight, so a deadline can be
// overrun by up to one text's inference. Do not rely on ctx as a hard inference
// timeout.
func (e *Embedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	return e.eng.Embed(ctx, texts)
}

// SPDX-License-Identifier: Apache-2.0

package rag

import (
	"context"
	"fmt"
	"math"

	"github.com/rostamlabs/rostam/semcache"
)

// retrieveOptions holds the optional knobs Retrieve/Ask accept via Option.
type retrieveOptions struct {
	hybrid bool
	alpha  float64
}

// Option configures optional Retrieve/Ask behavior.
type Option func(*retrieveOptions)

// WithHybrid enables dense+BM25 fusion. alpha < 0 selects RRF; 0..1 selects a
// weighted rank-blend (dense weight = alpha). Fusion only takes effect when
// an embedder is also configured; without one, retrieval stays BM25-only
// regardless of this option (there is no dense lane to fuse).
func WithHybrid(alpha float64) Option {
	return func(o *retrieveOptions) {
		o.hybrid = true
		o.alpha = alpha
	}
}

// Retrieve embeds query (if emb is non-nil) and searches corpus for the top-k
// hits. By default this runs a single Search: dense if an embedder is
// configured, else BM25. Pass WithHybrid to instead run both a dense lane and
// a BM25 lane (each pooled to max(k,50)) and fuse them via fuse.
func Retrieve(ctx context.Context, r Retriever, emb semcache.Embedder, corpus, query string, k int, opts ...Option) ([]Hit, error) {
	if k <= 0 {
		k = 5
	}
	var o retrieveOptions
	for _, opt := range opts {
		opt(&o)
	}
	// Validate alpha in the library, not just the CLI: WithHybrid takes any
	// float64, so a non-CLI caller could pass NaN (which poisons the weighted
	// fusion scores) or a value > 1 (outside the documented weight range). A
	// negative alpha is the RRF sentinel and always allowed.
	if o.hybrid && (math.IsNaN(o.alpha) || o.alpha > 1) {
		return nil, fmt.Errorf("rag: hybrid alpha must be <= 1 (got %v); use a negative alpha for RRF", o.alpha)
	}
	var qv []float32
	if emb != nil {
		vecs, err := emb.Embed(ctx, []string{query})
		if err != nil {
			return nil, fmt.Errorf("rag: embed query: %w", err)
		}
		if len(vecs) != 1 {
			return nil, fmt.Errorf("rag: embedder returned %d vectors for 1 query", len(vecs))
		}
		qv = vecs[0]
	}
	if emb != nil && o.hybrid {
		poolK := k
		if poolK < 50 {
			poolK = 50
		}
		dense, err := r.Search(ctx, corpus, query, qv, poolK)
		if err != nil {
			return nil, err
		}
		bm25, err := r.Search(ctx, corpus, query, nil, poolK)
		if err != nil {
			return nil, err
		}
		return fuse(dense, bm25, k, o.alpha), nil
	}
	return r.Search(ctx, corpus, query, qv, k)
}

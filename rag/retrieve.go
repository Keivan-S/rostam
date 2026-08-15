// SPDX-License-Identifier: Apache-2.0

package rag

import (
	"context"
	"fmt"

	"github.com/rostamlabs/rostam/semcache"
)

func Retrieve(ctx context.Context, r Retriever, emb semcache.Embedder, corpus, query string, k int) ([]Hit, error) {
	if k <= 0 {
		k = 5
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
	return r.Search(ctx, corpus, query, qv, k)
}

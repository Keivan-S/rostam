// SPDX-License-Identifier: Apache-2.0

// Package postrec is a content-based "next post to read" recommender built on
// Rostam's TYPED client (github.com/rostamlabs/rostam/client) — the routing-aware,
// struct-based wrapper over the native binary protocol. It drives the engine
// through client.Collection's typed methods (Create/Upsert/GetBatch/HybridText/
// Recommend), so it gets connection pooling, pipelining and topology-aware shard
// routing for free, with no ops/raw Call plumbing in this package.
//
// Recommendation uses Rostam's hybrid (BM25 + dense) search: keyword overlap and
// semantic similarity both contribute to ranking.
package postrec

import (
	"context"

	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/vector"
)

// Store is a thin, vector-oriented wrapper over the typed client: it turns the
// generic client.Collection surface into the handful of operations the
// recommender needs.
type Store struct {
	c   *client.Client
	col *client.Collection
}

// NewStore dials the given Rostam servers (e.g. []string{"127.0.0.1:7000"}) via
// client.NewRouted, so writes/reads route to the owning shard on a multi-node
// cluster. authToken may be empty. collection names the collection this Store
// operates on. The returned Store owns the connection pool; call Close when
// done.
func NewStore(servers []string, authToken, collection string) (*Store, error) {
	c, err := client.NewRouted(client.Config{
		Servers:   servers,
		AuthToken: authToken,
	})
	if err != nil {
		return nil, err
	}
	return &Store{c: c, col: c.Collection(collection)}, nil
}

// Close releases the connection pool.
func (s *Store) Close() error { return s.c.Close() }

// CreateCollection creates the collection with a dense HNSW index of the given
// dimension (cosine metric) plus the server-side BM25 full-text lane, so that
// HybridText works. content sent on each point is tokenized into that lane.
func (s *Store) CreateCollection(ctx context.Context, dim int) error {
	return s.col.Create(ctx, client.CreateRequest{
		Dim:      dim,
		Metric:   vector.Cosine, // good default for OpenAI text embeddings
		FullText: &vector.FullTextConfig{Analyzer: "english"},
	})
}

// Upsert inserts or replaces one point: dense vector + raw text (for BM25) +
// tagged metadata. Sparse is left empty because the BM25 lane is server-derived
// from content.
func (s *Store) Upsert(ctx context.Context, id uint64, vec []float32, content string, meta vector.Metadata) error {
	return s.col.Upsert(ctx, client.WriteRequest{
		ID:       id,
		Vector:   vec,
		Content:  content,
		Metadata: meta,
	})
}

// GetVectors fetches stored vectors + metadata for ids (found ids only).
func (s *Store) GetVectors(ctx context.Context, ids []uint64) (map[uint64]client.Point, error) {
	resp, err := s.col.GetBatch(ctx, client.GetBatchRequest{
		IDs:         ids,
		WithVector:  true,
		WithPayload: true,
	})
	if err != nil {
		return nil, err
	}
	out := make(map[uint64]client.Point, len(resp.Points))
	for _, p := range resp.Points {
		out[p.ID] = p
	}
	return out, nil
}

// HybridText runs dense + BM25 fusion: dense is the query embedding, text is the
// raw keyword query the server tokenizes. opts carries the fusion method/filter.
func (s *Store) HybridText(ctx context.Context, dense []float32, text string, k int, opts vector.HybridOpts) ([]vector.Result, error) {
	resp, err := s.col.HybridText(ctx, client.HybridTextRequest{
		Dense:  dense,
		Text:   text,
		K:      k,
		Filter: opts.Filter,
		Method: opts.Method,
		Alpha:  opts.Alpha,
	})
	if err != nil {
		return nil, err
	}
	return resp.Results, nil
}

// RecommendByIDs uses a RECOMMEND query: the coordinator resolves the
// positive/negative ids to their stored vectors and searches for the mean
// (positive) - mean(negative). Dense-only (no BM25); one round-trip, no client
// embedding. filter is applied on the leaf (use vector.Filter{} for none).
func (s *Store) RecommendByIDs(ctx context.Context, positive, negative []uint64, k int, filter vector.Filter) ([]vector.Result, error) {
	resp, err := s.col.Recommend(ctx, client.RecommendRequest{
		Positive: positive,
		Negative: negative,
		Strategy: vector.RecommendAverageVector, // derive-to-dense mean
		K:        k,
		Filter:   filter,
	})
	if err != nil {
		return nil, err
	}
	return resp.Results, nil
}

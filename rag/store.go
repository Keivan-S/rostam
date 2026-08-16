// SPDX-License-Identifier: Apache-2.0

package rag

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/vector"
)

// Hit is one retrieved chunk with its provenance.
type Hit struct {
	Content string
	Source  string
	Index   int
	Score   float32
}

// StoredChunk is a chunk ready to persist. Vector is nil for BM25-only corpora.
type StoredChunk struct {
	ID      uint64
	Content string
	Vector  []float32
	Source  string
	Index   int
}

// Retriever is the storage backend the rag package talks to.
type Retriever interface {
	// EnsureCorpus creates the named corpus if it does not already exist.
	// dim<=0 means BM25-only (no vectors are ever inserted).
	EnsureCorpus(ctx context.Context, name string, dim int) error
	Upsert(ctx context.Context, corpus string, chunks []StoredChunk) error
	// Search runs a vector search when queryVec is non-nil, else BM25 full text.
	Search(ctx context.Context, corpus, queryText string, queryVec []float32, k int) ([]Hit, error)
	DeleteBySource(ctx context.Context, corpus, source string) (int, error)
	Close() error
}

// EmbeddedRetriever runs the vector engine in-process against a local data dir.
type EmbeddedRetriever struct {
	store *vector.CollectionStore
	dim   int // 0 => BM25-only; used to size a placeholder vector-less collection
}

// NewEmbeddedRetriever opens (or creates) a corpus store rooted at dir.
func NewEmbeddedRetriever(dir string) (*EmbeddedRetriever, error) {
	s, err := vector.OpenCollectionStore(dir)
	if err != nil {
		return nil, err
	}
	return &EmbeddedRetriever{store: s}, nil
}

func (e *EmbeddedRetriever) EnsureCorpus(_ context.Context, name string, dim int) error {
	e.dim = dim
	// BM25-only corpora still need a collection; give it dim=1 so no real
	// vectors are ever inserted but the full-text index is live. Upsert pads
	// with a zero vector of this length (CollectionStore.Upsert rejects a
	// length mismatch against the collection's Dim, so nil alone won't do).
	d := dim
	if d <= 0 {
		d = 1
	}
	cfg := vector.Config{
		Dim: d, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1,
		// Always on: BM25-only corpora need it for Search, and a vector corpus
		// still wants full-text as a fallback lane (queryVec == nil).
		FullText: &vector.FullTextConfig{},
	}
	if err := e.store.CreateCollection(name, cfg); err != nil {
		// Treat "already exists" as success so re-ingest is idempotent — but
		// only when the dimension matches. A corpus's Dim is fixed at
		// creation, so a stale corpus (e.g. one created BM25-only at the
		// dim=1 placeholder, now being re-ingested with a real embedder)
		// would otherwise slip through here and fail later, deep inside
		// Upsert, as an unhelpful raw vector.ErrDimMismatch.
		if errors.Is(err, vector.ErrCollectionExists) {
			if existing, ok := e.store.Get(name); ok && existing.Config().Dim != d {
				return fmt.Errorf("rag: corpus %q already exists with dimension %d, but this run needs %d; use a different --corpus or wipe the --data dir to recreate it", name, existing.Config().Dim, d)
			}
			return nil
		}
		return err
	}
	return nil
}

func (e *EmbeddedRetriever) Upsert(_ context.Context, corpus string, chunks []StoredChunk) error {
	for _, c := range chunks {
		meta := vector.Metadata{
			"source": vector.NewString(c.Source),
			"chunk":  vector.NewInt(int64(c.Index)),
		}
		vec := c.Vector
		if vec == nil && e.dim <= 0 {
			// BM25-only corpora are sized Dim=1; pad with a zero vector so the
			// engine's length check (len(vec) != Dim) doesn't reject the write.
			vec = make([]float32, 1)
		}
		if err := e.store.Upsert(corpus, c.ID, vec, c.Content, 0, meta, nil); err != nil {
			return fmt.Errorf("rag: upsert id %d: %w", c.ID, err)
		}
	}
	return e.store.Flush(corpus)
}

func (e *EmbeddedRetriever) Search(_ context.Context, corpus, queryText string, queryVec []float32, k int) ([]Hit, error) {
	var docs []vector.Document
	var err error
	if len(queryVec) > 0 {
		docs, err = e.store.SearchDocs(corpus, queryVec, k, vector.Filter{})
	} else {
		docs, err = e.store.SearchText(corpus, queryText, k, vector.Filter{})
	}
	if err != nil {
		return nil, err
	}
	return docsToHits(docs), nil
}

func (e *EmbeddedRetriever) DeleteBySource(_ context.Context, corpus, source string) (int, error) {
	f := vector.Filter{Op: vector.FilterEq, Field: "source", Value: vector.NewString(source)}
	n, err := e.store.DeleteByFilter(corpus, f)
	if err != nil {
		return 0, err
	}
	if err := e.store.Flush(corpus); err != nil {
		return n, err
	}
	return n, nil
}

func (e *EmbeddedRetriever) Close() error { return e.store.Close() }

// HTTPRetriever talks to a running rostam server over the network (the
// client package's wire protocol — NOT plain HTTP, despite the endpoint
// flag's name) via the root rostam.Store interface, for the CLI's
// --endpoint mode.
type HTTPRetriever struct {
	store rostam.Store
	dim   int // 0 => BM25-only; used to size a placeholder vector-less collection
}

// NewHTTPRetriever constructs a client pointed at endpoint (a "host:port"
// bootstrap server address; the smart client discovers the rest of the
// cluster's topology from there).
func NewHTTPRetriever(endpoint string) (*HTTPRetriever, error) {
	s, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{endpoint}})
	if err != nil {
		return nil, err
	}
	return &HTTPRetriever{store: s}, nil
}

func (h *HTTPRetriever) EnsureCorpus(ctx context.Context, name string, dim int) error {
	h.dim = dim
	// Mirrors EmbeddedRetriever.EnsureCorpus: BM25-only corpora still need a
	// collection; give it dim=1 so no real vectors are ever inserted but the
	// full-text index is live.
	//
	// Unlike EmbeddedRetriever, this does NOT guard against a dim mismatch on
	// an already-existing corpus: the rostam.Store client interface exposes
	// CreateCollection (write) but no read RPC to fetch an existing
	// collection's configured Dim, so there is nothing here to compare
	// against without a fragile ad hoc fetch. A dim mismatch over -endpoint
	// still surfaces later as a raw error out of Upsert, same as before this
	// fix; closing that gap is a fast-follow.
	d := dim
	if d <= 0 {
		d = 1
	}
	cfg := rostam.VectorConfig{
		Dim: d, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1,
		FullText: &vector.FullTextConfig{},
	}
	if err := h.store.CreateCollection(ctx, name, cfg); err != nil {
		// Treat "already exists" as success so re-ingest is idempotent. Over the
		// wire the error is a reconstructed value whose chain does not survive
		// the RPC round trip, so errors.Is alone won't match a networked
		// CreateCollection failure — fall back to the message the server sends,
		// mirroring mcp.isCollectionExists.
		if errors.Is(err, vector.ErrCollectionExists) || strings.Contains(err.Error(), "collection already exists") {
			return nil
		}
		return err
	}
	return nil
}

func (h *HTTPRetriever) Upsert(ctx context.Context, corpus string, chunks []StoredChunk) error {
	for _, c := range chunks {
		meta := rostam.VectorMetadata{
			"source": vector.NewString(c.Source),
			"chunk":  vector.NewInt(int64(c.Index)),
		}
		vec := c.Vector
		if vec == nil && h.dim <= 0 {
			// BM25-only corpora are sized Dim=1; pad with a zero vector so the
			// engine's length check (len(vec) != Dim) doesn't reject the write.
			vec = make([]float32, 1)
		}
		if err := h.store.VectorUpsert(ctx, corpus, c.ID, vec, c.Content, rostam.VectorInsertOpts{Metadata: meta}); err != nil {
			return fmt.Errorf("rag: upsert id %d: %w", c.ID, err)
		}
	}
	// Unlike EmbeddedRetriever there is no local Flush: durability of a
	// networked write is the server's concern (Raft/WAL), not the client's.
	return nil
}

func (h *HTTPRetriever) Search(ctx context.Context, corpus, queryText string, queryVec []float32, k int) ([]Hit, error) {
	var docs []rostam.VectorDocument
	var err error
	if len(queryVec) > 0 {
		docs, _, err = h.store.VectorSearchDocs(ctx, corpus, queryVec, k, rostam.VectorSearchOpts{})
	} else {
		docs, _, err = h.store.VectorSearchText(ctx, corpus, queryText, k, rostam.VectorSearchOpts{})
	}
	if err != nil {
		return nil, err
	}
	return docsToHits(docs), nil
}

func (h *HTTPRetriever) DeleteBySource(ctx context.Context, corpus, source string) (int, error) {
	f := rostam.VectorFilter{Op: vector.FilterEq, Field: "source", Value: vector.NewString(source)}
	return h.store.VectorDeleteByFilter(ctx, corpus, f)
}

func (h *HTTPRetriever) Close() error { return h.store.Close() }

// docsToHits maps engine documents to Hits, pulling source/chunk back out of
// metadata (both were written as tagged values in Upsert).
func docsToHits(docs []vector.Document) []Hit {
	hits := make([]Hit, 0, len(docs))
	for _, d := range docs {
		h := Hit{Content: d.Content, Score: d.Score}
		if v, ok := d.Metadata["source"]; ok {
			h.Source = v.Str
		}
		if v, ok := d.Metadata["chunk"]; ok {
			h.Index = int(v.Int)
		}
		hits = append(hits, h)
	}
	return hits
}

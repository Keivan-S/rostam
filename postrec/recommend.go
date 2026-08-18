// SPDX-License-Identifier: Apache-2.0

package postrec

import (
	"context"
	"fmt"
	"strings"

	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/vector"
)

// Embedder turns texts into dense vectors (see openai.go for an OpenAI impl).
// Any model works; ingest and query must use the same one, and Dim must match
// the collection.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	Dim() int
}

// Post is a blog post / article to make recommendable.
type Post struct {
	ID                uint64
	Title             string
	Body              string
	Tags              []string
	PublishedAtUnixMs int64 // 0 = unset
}

// Recommendation is one suggested post with its fusion score.
type Recommendation struct {
	ID    uint64
	Score float32
}

// Recommender ties a Store + an Embedder together and exposes the two flavors of
// "next post" recommendation:
//
//   - Related: "more articles like the one you're reading now"
//   - ForUser: a personalized feed from the posts a user has read
//
// Both use hybrid (BM25 + dense) search. Method/Alpha tune fusion; the zero
// value (FusionRRF) needs no tuning.
type Recommender struct {
	Store    *Store
	Embedder Embedder
	Method   vector.FusionMethod // zero = FusionRRF
	Alpha    float64             // used only by FusionWeighted (dense weight, 0..1)
}

// NewRecommender wires the pieces with RRF fusion. store must already be bound
// to the target collection (see NewStore).
func NewRecommender(s *Store, e Embedder) *Recommender {
	return &Recommender{Store: s, Embedder: e, Method: vector.FusionRRF}
}

// EnsureCollection creates the store's collection sized to the embedder, with
// the BM25 lane enabled.
func (r *Recommender) EnsureCollection(ctx context.Context) error {
	return r.Store.CreateCollection(ctx, r.Embedder.Dim())
}

func (r *Recommender) hybridOpts(filter vector.Filter) vector.HybridOpts {
	return vector.HybridOpts{Filter: filter, Method: r.Method, Alpha: r.Alpha}
}

// bm25Text is the text handed to the BM25 lane. Title is repeated as a cheap,
// effective keyword boost.
func bm25Text(p Post) string {
	parts := []string{p.Title, p.Title, p.Body}
	if len(p.Tags) > 0 {
		parts = append(parts, strings.Join(p.Tags, " "))
	}
	return strings.Join(parts, "\n")
}

// Ingest embeds and upserts posts (one upsert per post). content feeds BM25;
// metadata carries post_id/title/tags/published for filtering and display.
func (r *Recommender) Ingest(ctx context.Context, posts ...Post) error {
	if len(posts) == 0 {
		return nil
	}
	texts := make([]string, len(posts))
	for i, p := range posts {
		texts[i] = p.Title + "\n\n" + p.Body
	}
	vecs, err := r.Embedder.Embed(ctx, texts)
	if err != nil {
		return fmt.Errorf("embed posts: %w", err)
	}
	if len(vecs) != len(posts) {
		return fmt.Errorf("embedder returned %d vectors for %d posts", len(vecs), len(posts))
	}
	for i, p := range posts {
		meta := vector.Metadata{
			"post_id":      vector.NewInt(int64(p.ID)),
			"title":        vector.NewString(p.Title),
			"tags":         vector.NewStrings(p.Tags),
			"published":    vector.NewBool(true),
			"published_at": vector.NewInt(p.PublishedAtUnixMs),
		}
		if err := r.Store.Upsert(ctx, p.ID, vecs[i], bm25Text(p), meta); err != nil {
			return fmt.Errorf("upsert post %d: %w", p.ID, err)
		}
	}
	return nil
}

// publishedFilter restricts results to published posts. Extend it with your own
// predicates (author, language, topic) using vector.Filter + vector.New*.
func publishedFilter() vector.Filter {
	return vector.Filter{Op: vector.FilterEq, Field: "published", Value: vector.NewBool(true)}
}

// Related returns posts most similar to `current` — the "related articles" list.
// It reuses the post's stored vector (fetched, not re-embedded) as the dense
// query and its title/body as the BM25 query, then drops the post itself.
func (r *Recommender) Related(ctx context.Context, current Post, k int) ([]Recommendation, error) {
	var vec []float32
	got, err := r.Store.GetVectors(ctx, []uint64{current.ID})
	if err != nil {
		return nil, err
	}
	if row, ok := got[current.ID]; ok {
		vec = row.Vector
	}
	if vec == nil { // GetVectors succeeded but this id wasn't present — embed on the fly
		vecs, err := r.Embedder.Embed(ctx, []string{current.Title + "\n\n" + current.Body})
		if err != nil {
			return nil, err
		}
		vec = vecs[0]
	}
	hits, err := r.Store.HybridText(ctx, vec, bm25Text(current), k+1, r.hybridOpts(publishedFilter()))
	if err != nil {
		return nil, err
	}
	return toRecs(hits, map[uint64]bool{current.ID: true}, k), nil
}

// ForUser builds a personalized "up next" feed from a user's reading history.
// It averages the stored vectors of the read posts into a taste vector, builds a
// BM25 query from their titles, runs one hybrid search, and drops everything
// already read.
func (r *Recommender) ForUser(ctx context.Context, readIDs []uint64, k int) ([]Recommendation, error) {
	if len(readIDs) == 0 {
		return nil, fmt.Errorf("ForUser: empty reading history")
	}
	stored, err := r.Store.GetVectors(ctx, readIDs)
	if err != nil {
		return nil, err
	}
	if len(stored) == 0 {
		return nil, fmt.Errorf("ForUser: none of the %d read posts were found", len(readIDs))
	}

	taste, tasteText := tasteFrom(stored)
	seen := make(map[uint64]bool, len(readIDs))
	for _, id := range readIDs {
		seen[id] = true
	}
	// Over-fetch so we can drop the already-read posts and still return k fresh.
	hits, err := r.Store.HybridText(ctx, taste, tasteText, k+len(readIDs), r.hybridOpts(publishedFilter()))
	if err != nil {
		return nil, err
	}
	return toRecs(hits, seen, k), nil
}

// ForUserServerSide is the dense-only alternative: it hands the read ids to the
// RECOMMEND query leaf, which averages their vectors server-side (no client
// embedding, one round-trip). No BM25 — use ForUser for hybrid.
func (r *Recommender) ForUserServerSide(ctx context.Context, readIDs []uint64, k int) ([]Recommendation, error) {
	hits, err := r.Store.RecommendByIDs(ctx, readIDs, nil, k+len(readIDs), publishedFilter())
	if err != nil {
		return nil, err
	}
	seen := make(map[uint64]bool, len(readIDs))
	for _, id := range readIDs {
		seen[id] = true
	}
	return toRecs(hits, seen, k), nil
}

// ---- helpers ----

func toRecs(hits []vector.Result, exclude map[uint64]bool, k int) []Recommendation {
	if k <= 0 {
		return nil
	}
	out := make([]Recommendation, 0, k)
	for _, h := range hits {
		if exclude[h.ID] {
			continue
		}
		out = append(out, Recommendation{ID: h.ID, Score: h.Score})
		if len(out) == k {
			break
		}
	}
	return out
}

// tasteFrom averages stored vectors into a taste vector and concatenates the
// stored titles into a BM25 query. Metadata comes back typed, so no JSON parsing.
func tasteFrom(stored map[uint64]client.Point) ([]float32, string) {
	var sum []float32
	var titles []string
	n := 0
	for _, pt := range stored {
		if sum == nil {
			sum = make([]float32, len(pt.Vector))
		}
		for i, v := range pt.Vector {
			sum[i] += v
		}
		n++
		if t, ok := pt.Metadata["title"]; ok && t.Str != "" {
			titles = append(titles, t.Str)
		}
	}
	if n > 0 {
		inv := float32(1) / float32(n)
		for i := range sum {
			sum[i] *= inv
		}
	}
	return sum, strings.Join(titles, "\n")
}

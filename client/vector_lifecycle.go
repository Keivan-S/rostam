// SPDX-License-Identifier: Apache-2.0
package client

import (
	"context"
	"strings"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// CreateRequest configures a new collection. Only Dim is required. Set FullText
// to enable the server-side BM25 lane (required for HybridText).
type CreateRequest struct {
	Dim            int
	Metric         vector.Metric // zero value = vector.Cosine
	M              int
	EfConstruction int
	EfSearch       int
	Persistent     bool
	FullText       *vector.FullTextConfig
}

// Create creates the collection backing this handle. If a collection with the
// same name already exists, it returns ErrCollectionExists.
func (col *Collection) Create(ctx context.Context, req CreateRequest) error {
	cfg := vector.Config{
		Dim:            req.Dim,
		Metric:         req.Metric,
		M:              req.M,
		EfConstruction: req.EfConstruction,
		EfSearch:       req.EfSearch,
		Persistent:     req.Persistent,
		FullText:       req.FullText,
	}
	_, err := col.c.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(col.name, cfg))
	return mapCreateErr(err)
}

// Drop deletes the collection backing this handle.
func (col *Collection) Drop(ctx context.Context) error {
	_, err := col.c.Call(ctx, "vector_drop_collection", ops.EncodeDropCollectionArgs(col.name))
	return err
}

// mapCreateErr translates the server's "already exists" text (vector.ErrCollectionExists:
// "vector: collection already exists") into the client's own ErrCollectionExists sentinel.
func mapCreateErr(err error) error {
	if err != nil && strings.Contains(strings.ToLower(err.Error()), "exist") {
		return ErrCollectionExists
	}
	return err
}

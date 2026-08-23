// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"strings"

	"github.com/rostamlabs/rostam/sdk/vtypes"
	"github.com/rostamlabs/rostam/sdk/wire"
)

// CreateRequest configures a new collection. Only Dim is required. Set FullText
// to enable the server-side BM25 lane (required for HybridText).
type CreateRequest struct {
	Dim            int
	Metric         vtypes.Metric // zero value = vtypes.Cosine
	M              int
	EfConstruction int
	EfSearch       int
	Persistent     bool
	FullText       *vtypes.FullTextConfig
}

// Create creates the collection backing this handle. If a collection with the
// same name already exists, it returns ErrCollectionExists.
func (col *Collection) Create(ctx context.Context, req CreateRequest) error {
	cfg := vtypes.Config{
		Dim:            req.Dim,
		Metric:         req.Metric,
		M:              req.M,
		EfConstruction: req.EfConstruction,
		EfSearch:       req.EfSearch,
		Persistent:     req.Persistent,
		FullText:       req.FullText,
	}
	_, err := col.c.Call(ctx, "vector_create_collection", wire.EncodeCreateCollectionArgs(col.name, cfg))
	return mapCreateErr(err)
}

// Drop deletes the collection backing this handle. The server's
// CollectionStore.DropCollection treats a never-created (or already-dropped)
// name as a no-op rather than an error, so Drop is idempotent: it does not
// return ErrCollectionNotFound. (isCollectionNotFound is still applied below
// as a defensive translation in case that server contract ever changes; see
// Get/Search for the paths where a missing collection actually surfaces.)
func (col *Collection) Drop(ctx context.Context) error {
	_, err := col.c.Call(ctx, "vector_drop_collection", wire.EncodeDropCollectionArgs(col.name))
	if isCollectionNotFound(err) {
		return ErrCollectionNotFound
	}
	return err
}

// mapCreateErr translates the server's "already exists" text (vector.ErrCollectionExists:
// "vector: collection already exists") into the client's own ErrCollectionExists sentinel.
// The match is "already exist" (not the looser "exist") so it cannot accidentally
// fire on an unrelated "does not exist" / "no collection" error text.
func mapCreateErr(err error) error {
	if err != nil && strings.Contains(strings.ToLower(err.Error()), "already exist") {
		return ErrCollectionExists
	}
	return err
}

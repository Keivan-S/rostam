// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"errors"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// WriteRequest is the payload for Insert and Upsert.
type WriteRequest struct {
	ID       uint64
	Vector   []float32
	Content  string              // raw text tokenized into the BM25 lane
	Metadata vector.Metadata     // build with vector.NewString/NewInt/...
	Sparse   vector.SparseVector // optional client sparse vector
	TTL      time.Duration       // 0 = no expiry
	// ExpectedVersion enables optimistic concurrency; HasExpectedVersion must be
	// true for it to apply.
	ExpectedVersion    uint64
	HasExpectedVersion bool
	KeyTTLMs           map[string]int64
}

// Upsert writes req, creating or overwriting the point at req.ID. If
// req.HasExpectedVersion is set, the write only applies when the point's
// current version matches req.ExpectedVersion; otherwise it returns
// ErrVersionConflict.
func (col *Collection) Upsert(ctx context.Context, req WriteRequest) error {
	args := ops.EncodeVectorUpsertArgsCASKeyTTL(
		col.name, req.ID, req.Vector, req.Content, req.TTL, req.Metadata,
		req.Sparse, req.ExpectedVersion, req.HasExpectedVersion, req.KeyTTLMs)
	_, err := col.c.Call(ctx, "vector_upsert", args)
	return mapWriteErr(err)
}

// Insert writes req only if no point exists at req.ID (subject to the same
// CAS semantics as Upsert when req.HasExpectedVersion is set).
func (col *Collection) Insert(ctx context.Context, req WriteRequest) error {
	// The insert wire op carries no content field; refuse rather than silently
	// drop it. Callers needing the BM25 content lane must use Upsert.
	if req.Content != "" {
		return errors.New("client: Insert does not support Content; use Upsert to store content")
	}
	args := ops.EncodeVectorInsertArgsCASKeyTTL(
		col.name, req.ID, req.Vector, req.TTL, req.Metadata,
		req.Sparse, req.ExpectedVersion, req.HasExpectedVersion, req.KeyTTLMs)
	_, err := col.c.Call(ctx, "vector_insert", args)
	return mapWriteErr(err)
}

// DeleteRequest removes a point, optionally guarded by ExpectedVersion.
type DeleteRequest struct {
	ID                 uint64
	ExpectedVersion    uint64
	HasExpectedVersion bool
}

// Delete removes the point at req.ID. If req.HasExpectedVersion is set, the
// delete only applies when the point's current version matches
// req.ExpectedVersion; otherwise it returns ErrVersionConflict.
func (col *Collection) Delete(ctx context.Context, req DeleteRequest) error {
	args := ops.EncodeVectorDeleteArgsCAS(col.name, req.ID, req.ExpectedVersion, req.HasExpectedVersion)
	_, err := col.c.Call(ctx, "vector_delete", args)
	return mapWriteErr(err)
}

// mapWriteErr translates the server's CAS-conflict text into ErrVersionConflict.
func mapWriteErr(err error) error {
	if err == nil {
		return nil
	}
	if isVersionConflict(err) {
		return ErrVersionConflict
	}
	return err
}

// PointInput is one point in an UpsertBatch.
type PointInput struct {
	ID       uint64
	Vector   []float32
	Content  string
	Metadata vector.Metadata
	Sparse   vector.SparseVector
	TTL      time.Duration
}

// BatchError reports a failure for a specific point id.
type BatchError struct {
	ID  uint64
	Err error
}

// UpsertBatch upserts each point. There is no single batch op on the wire, so
// this issues one upsert per point over the (optionally pipelined) connection
// and returns the failures, if any, keyed by id. A nil/empty return means all
// succeeded.
func (col *Collection) UpsertBatch(ctx context.Context, pts []PointInput) []BatchError {
	var errs []BatchError
	for _, p := range pts {
		err := col.Upsert(ctx, WriteRequest{
			ID: p.ID, Vector: p.Vector, Content: p.Content,
			Metadata: p.Metadata, Sparse: p.Sparse, TTL: p.TTL,
		})
		if err != nil {
			errs = append(errs, BatchError{ID: p.ID, Err: err})
		}
	}
	return errs
}

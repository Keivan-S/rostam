// SPDX-License-Identifier: Apache-2.0
package client

import (
	"context"
	"encoding/json"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// Point is a stored point returned by reads.
type Point struct {
	ID       uint64
	Vector   []float32
	Metadata vector.Metadata
	TTLMs    uint64
	Version  uint64
}

// GetRequest selects a single point by id and its projection.
type GetRequest struct {
	ID          uint64
	WithVector  bool
	WithPayload bool
}

// getFlags maps the request's projection booleans onto the wire flag byte,
// defaulting to GetFlagsBoth when neither is set (matches the server's
// all-or-nothing default projection).
func getFlags(withVec, withPayload bool) uint8 {
	var f uint8
	if withVec {
		f |= ops.GetFlagWithVector
	}
	if withPayload {
		f |= ops.GetFlagWithPayload
	}
	if f == 0 {
		f = ops.GetFlagsBoth
	}
	return f
}

// Get fetches the point at req.ID. It returns ErrNotFound when no such point
// exists.
func (col *Collection) Get(ctx context.Context, req GetRequest) (Point, error) {
	body, err := col.c.Call(ctx, "vector_get",
		ops.EncodeVectorGetArgs(col.name, req.ID, getFlags(req.WithVector, req.WithPayload)))
	if err != nil {
		return Point{}, err
	}
	found, vec, meta, ttl, _, version, err := ops.DecodeVectorGetResultV(body)
	if err != nil {
		return Point{}, err
	}
	if !found {
		return Point{}, ErrNotFound
	}
	return Point{ID: req.ID, Vector: vec, Metadata: meta, TTLMs: uint64(ttl.Milliseconds()), Version: version}, nil //nolint:gosec
}

// GetBatchRequest selects multiple points by id, sharing one projection.
type GetBatchRequest struct {
	IDs         []uint64
	WithVector  bool
	WithPayload bool
}

// GetBatchResponse reports the points that were found plus the ids from the
// request that were not.
type GetBatchResponse struct {
	Points  []Point
	Missing []uint64
}

// GetBatch fetches multiple points in a single round trip. Requested ids that
// do not exist are reported in Missing rather than causing an error.
func (col *Collection) GetBatch(ctx context.Context, req GetBatchRequest) (GetBatchResponse, error) {
	body, err := col.c.Call(ctx, "vector_get_batch",
		ops.EncodeVectorGetBatchArgs(col.name, req.IDs, getFlags(req.WithVector, req.WithPayload)))
	if err != nil {
		return GetBatchResponse{}, err
	}
	rows, err := ops.DecodeVectorGetBatchResult(body)
	if err != nil {
		return GetBatchResponse{}, err
	}
	var out GetBatchResponse
	present := make(map[uint64]bool, len(rows))
	for _, r := range rows {
		if r.Found {
			out.Points = append(out.Points, Point{
				ID: r.ID, Vector: r.Vec, Metadata: r.Meta, TTLMs: r.TTLMs, Version: r.Version,
			})
			present[r.ID] = true
		}
	}
	for _, id := range req.IDs {
		if !present[id] {
			out.Missing = append(out.Missing, id)
		}
	}
	return out, nil
}

// Document is a payload-bearing row from Scroll and doc searches.
type Document struct {
	ID       uint64
	Distance float32
	Score    float32
	Content  string
	Metadata vector.Metadata
}

// ScrollRequest pages through a collection's points in id order (or resuming
// from a prior page's Cursor), applying an optional filter.
type ScrollRequest struct {
	Filter vector.Filter
	Limit  int
	Cursor string
}

// ScrollResponse is one page of Scroll results. NextCursor is empty when
// there is no further page.
type ScrollResponse struct {
	Documents  []Document
	NextCursor string
}

// Scroll returns a page of points matching req.Filter, ordered by id. Pass
// the previous response's NextCursor as req.Cursor to fetch the next page.
func (col *Collection) Scroll(ctx context.Context, req ScrollRequest) (ScrollResponse, error) {
	dec, err := ops.DecodeScrollCursorTyped(req.Cursor)
	if err != nil {
		return ScrollResponse{}, err
	}
	afterID, hasAfter := dec.LastID, dec.Present

	body, err := col.c.Call(ctx, "vector_scroll",
		ops.EncodeScrollArgsOrderBounded(col.name, req.Filter, req.Limit, 0, 0, afterID, hasAfter, nil, 0))
	if err != nil {
		return ScrollResponse{}, err
	}
	docs, _, _, nextCursor, err := ops.DecodeScrollResultRaw(body)
	if err != nil {
		return ScrollResponse{}, err
	}
	out := ScrollResponse{NextCursor: nextCursor}
	for _, d := range docs {
		var meta vector.Metadata
		if len(d.Metadata) > 0 {
			meta = make(vector.Metadata)
			if err := json.Unmarshal(d.Metadata, &meta); err != nil {
				return ScrollResponse{}, err
			}
		}
		out.Documents = append(out.Documents, Document{
			ID: d.ID, Distance: d.Distance, Score: d.Score, Content: d.Content, Metadata: meta,
		})
	}
	return out, nil
}

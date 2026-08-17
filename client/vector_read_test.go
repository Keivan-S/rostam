// SPDX-License-Identifier: Apache-2.0
package client

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

func TestGetAndGetBatch(t *testing.T) {
	col, cleanup := mustCollection(t)
	defer cleanup()
	ctx := context.Background()

	if err := col.Upsert(ctx, WriteRequest{
		ID: 1, Vector: []float32{0.1, 0.2, 0.3, 0.4},
		Metadata: vector.Metadata{"title": vector.NewString("Hello")},
	}); err != nil {
		t.Fatal(err)
	}

	p, err := col.Get(ctx, GetRequest{ID: 1, WithVector: true, WithPayload: true})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if len(p.Vector) != 4 || p.Metadata["title"].Str != "Hello" {
		t.Fatalf("Get returned %+v", p)
	}

	if _, err := col.Get(ctx, GetRequest{ID: 999, WithVector: true}); err != ErrNotFound {
		t.Fatalf("Get missing = %v, want ErrNotFound", err)
	}

	resp, err := col.GetBatch(ctx, GetBatchRequest{IDs: []uint64{1, 999}, WithVector: true, WithPayload: true})
	if err != nil {
		t.Fatalf("GetBatch: %v", err)
	}
	if len(resp.Points) != 1 || resp.Points[0].ID != 1 {
		t.Fatalf("GetBatch points = %+v", resp.Points)
	}
	if len(resp.Missing) != 1 || resp.Missing[0] != 999 {
		t.Fatalf("GetBatch missing = %+v", resp.Missing)
	}
}

func TestScroll(t *testing.T) {
	col, cleanup := mustCollection(t)
	defer cleanup()
	ctx := context.Background()

	for _, id := range []uint64{1, 2, 3} {
		if err := col.Upsert(ctx, WriteRequest{
			ID: id, Vector: []float32{0.1, 0.2, 0.3, 0.4},
			Content:  "doc",
			Metadata: vector.Metadata{"title": vector.NewString("Hello")},
		}); err != nil {
			t.Fatal(err)
		}
	}

	resp, err := col.Scroll(ctx, ScrollRequest{Filter: vector.Filter{}, Limit: 10})
	if err != nil {
		t.Fatalf("Scroll: %v", err)
	}
	if len(resp.Documents) != 3 {
		t.Fatalf("Scroll documents = %+v, want 3", resp.Documents)
	}
}

// SPDX-License-Identifier: Apache-2.0
package rostam

import (
	"context"
	"errors"
	"testing"

	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/sdk/vtypes"
)

// TestGetOnMissingCollection confirms Get surfaces the server's distinguishable
// missing-collection error ("vector: no collection %q", from
// CollectionStore.GetPointVersionInto) as ErrCollectionNotFound rather than a
// raw, unmapped error string.
func TestGetOnMissingCollection(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, err := client.New(client.Config{Servers: []string{addr}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	col := c.Collection("never-created")
	_, err = col.Get(context.Background(), client.GetRequest{ID: 1})
	if !errors.Is(err, client.ErrCollectionNotFound) {
		t.Fatalf("Get on never-created collection = %v, want ErrCollectionNotFound", err)
	}
}

// TestGetBatchOnMissingCollection confirms GetBatch maps the server's
// missing-collection error onto ErrCollectionNotFound, same as Get.
func TestGetBatchOnMissingCollection(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, err := client.New(client.Config{Servers: []string{addr}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	col := c.Collection("never-created")
	_, err = col.GetBatch(context.Background(), client.GetBatchRequest{IDs: []uint64{1, 2}})
	if !errors.Is(err, client.ErrCollectionNotFound) {
		t.Fatalf("GetBatch on never-created collection = %v, want ErrCollectionNotFound", err)
	}
}

func TestGetAndGetBatch(t *testing.T) {
	col, cleanup := mustCollection(t)
	defer cleanup()
	ctx := context.Background()

	if err := col.Upsert(ctx, client.WriteRequest{
		ID: 1, Vector: []float32{0.1, 0.2, 0.3, 0.4},
		Metadata: vtypes.Metadata{"title": vtypes.NewString("Hello")},
	}); err != nil {
		t.Fatal(err)
	}

	p, err := col.Get(ctx, client.GetRequest{ID: 1, WithVector: true, WithPayload: true})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if len(p.Vector) != 4 || p.Metadata["title"].Str != "Hello" {
		t.Fatalf("Get returned %+v", p)
	}

	if _, err := col.Get(ctx, client.GetRequest{ID: 999, WithVector: true}); err != client.ErrNotFound {
		t.Fatalf("Get missing = %v, want ErrNotFound", err)
	}

	resp, err := col.GetBatch(ctx, client.GetBatchRequest{IDs: []uint64{1, 999}, WithVector: true, WithPayload: true})
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
		if err := col.Upsert(ctx, client.WriteRequest{
			ID: id, Vector: []float32{0.1, 0.2, 0.3, 0.4},
			Content:  "doc",
			Metadata: vtypes.Metadata{"title": vtypes.NewString("Hello")},
		}); err != nil {
			t.Fatal(err)
		}
	}

	resp, err := col.Scroll(ctx, client.ScrollRequest{Filter: vtypes.Filter{}, Limit: 10})
	if err != nil {
		t.Fatalf("Scroll: %v", err)
	}
	if len(resp.Documents) != 3 {
		t.Fatalf("Scroll documents = %+v, want 3", resp.Documents)
	}
}

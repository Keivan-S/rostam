// SPDX-License-Identifier: Apache-2.0
package client

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

func mustCollection(t *testing.T) (*Collection, func()) {
	t.Helper()
	addr, stop := startTestStack(t)
	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		stop()
		t.Fatal(err)
	}
	col := c.Collection("posts")
	if err := col.Create(context.Background(), CreateRequest{Dim: 4, Metric: vector.Cosine}); err != nil {
		_ = c.Close()
		stop()
		t.Fatal(err)
	}
	return col, func() { _ = c.Close(); stop() }
}

func TestUpsertThenDelete(t *testing.T) {
	col, cleanup := mustCollection(t)
	defer cleanup()
	ctx := context.Background()

	err := col.Upsert(ctx, WriteRequest{
		ID:       1,
		Vector:   []float32{0.1, 0.2, 0.3, 0.4},
		Content:  "hello world",
		Metadata: vector.Metadata{"title": vector.NewString("Hello")},
	})
	if err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if err := col.Delete(ctx, DeleteRequest{ID: 1}); err != nil {
		t.Fatalf("Delete: %v", err)
	}
}

func TestUpsertVersionConflict(t *testing.T) {
	col, cleanup := mustCollection(t)
	defer cleanup()
	ctx := context.Background()

	if err := col.Upsert(ctx, WriteRequest{ID: 7, Vector: []float32{1, 0, 0, 0}}); err != nil {
		t.Fatalf("seed Upsert: %v", err)
	}
	// Expect a version that cannot match a fresh point.
	err := col.Upsert(ctx, WriteRequest{
		ID: 7, Vector: []float32{0, 1, 0, 0},
		ExpectedVersion: 999, HasExpectedVersion: true,
	})
	if err != ErrVersionConflict {
		t.Fatalf("Upsert with stale version = %v, want ErrVersionConflict", err)
	}
}

func TestUpsertBatch(t *testing.T) {
	col, cleanup := mustCollection(t)
	defer cleanup()
	ctx := context.Background()

	pts := []PointInput{
		{ID: 1, Vector: []float32{1, 0, 0, 0}, Content: "a"},
		{ID: 2, Vector: []float32{0, 1, 0, 0}, Content: "b"},
		{ID: 3, Vector: []float32{0, 0, 1, 0}, Content: "c"},
	}
	if errs := col.UpsertBatch(ctx, pts); len(errs) != 0 {
		t.Fatalf("UpsertBatch errors: %+v", errs)
	}
}

func TestInsertRejectsContent(t *testing.T) {
	col, cleanup := mustCollection(t)
	defer cleanup()
	ctx := context.Background()

	// The insert wire op carries no content field, so Insert must refuse it
	// rather than silently drop it.
	if err := col.Insert(ctx, WriteRequest{ID: 1, Vector: []float32{1, 0, 0, 0}, Content: "x"}); err == nil {
		t.Fatal("Insert with Content = nil error, want rejection")
	}
	// A plain Insert (no content) still works.
	if err := col.Insert(ctx, WriteRequest{ID: 2, Vector: []float32{0, 1, 0, 0}}); err != nil {
		t.Fatalf("Insert without content: %v", err)
	}
}

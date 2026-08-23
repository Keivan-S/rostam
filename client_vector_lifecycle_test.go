// SPDX-License-Identifier: Apache-2.0
package rostam

import (
	"context"
	"errors"
	"testing"

	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/sdk/vtypes"
)

func TestCollectionCreateAndDrop(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, err := client.New(client.Config{Servers: []string{addr}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	ctx := context.Background()
	col := c.Collection("posts")
	if err := col.Create(ctx, client.CreateRequest{
		Dim:      4,
		Metric:   vtypes.Cosine,
		FullText: &vtypes.FullTextConfig{Analyzer: "english"},
	}); err != nil {
		t.Fatalf("Create: %v", err)
	}
	// Second create of the same name must surface ErrCollectionExists.
	err = col.Create(ctx, client.CreateRequest{Dim: 4, Metric: vtypes.Cosine})
	if !errors.Is(err, client.ErrCollectionExists) {
		t.Fatalf("second Create: want ErrCollectionExists, got %v", err)
	}
	if err := col.Drop(ctx); err != nil {
		t.Fatalf("Drop: %v", err)
	}
}

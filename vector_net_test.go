// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"testing"
)

// TestVectorSearchIntoMatchesSearch verifies the zero-copy VectorSearchInto path
// returns the same ranked results as the allocating VectorSearch, over the wire,
// and that a reused dst is honored across calls.
func TestVectorSearchIntoMatchesSearch(t *testing.T) {
	const dim = 256
	cli, cleanup := setupVectorServer(t, 2_000, dim)
	defer cleanup()
	ctx := context.Background()
	queries := benchNetVectors(20, dim, 99)

	dst := make([]VectorResult, 0, 10)
	for qi, q := range queries {
		want, err := cli.VectorSearch(ctx, "bench", q, 10)
		if err != nil {
			t.Fatalf("q%d VectorSearch: %v", qi, err)
		}
		var got []VectorResult
		got, err = cli.VectorSearchInto(ctx, "bench", q, 10, dst[:0])
		if err != nil {
			t.Fatalf("q%d VectorSearchInto: %v", qi, err)
		}
		dst = got // carry the (possibly regrown) buffer into the next iteration
		if len(got) != len(want) {
			t.Fatalf("q%d: len got %d want %d", qi, len(got), len(want))
		}
		for i := range want {
			if got[i].ID != want[i].ID || got[i].Distance != want[i].Distance {
				t.Fatalf("q%d rank %d: got {id %d dist %v} want {id %d dist %v}",
					qi, i, got[i].ID, got[i].Distance, want[i].ID, want[i].Distance)
			}
		}
	}
}

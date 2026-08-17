// SPDX-License-Identifier: Apache-2.0
package client

import (
	"context"
	"testing"
)

func TestRecommendByExampleIDs(t *testing.T) {
	col, cleanup := seedSearchable(t) // ids 1,2,3 from Task 7 helper
	defer cleanup()
	ctx := context.Background()

	resp, err := col.Recommend(ctx, RecommendRequest{Positive: []uint64{1}, K: 2})
	if err != nil {
		t.Fatalf("Recommend: %v", err)
	}
	// The positive example itself is excluded by the engine; results are other ids.
	for _, r := range resp.Results {
		if r.ID == 1 {
			t.Fatal("Recommend returned the positive example id 1")
		}
	}
}

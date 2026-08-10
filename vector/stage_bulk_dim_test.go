// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"testing"
)

// TestStageBulkRejectsWrongDim pins the dimension check at the shard that owns
// the collection config.
//
// It lives here rather than in a transport on purpose. The dimension is only
// authoritative where the config is, so checking it in one transport would leave
// every other caller of the staging op unchecked — and would cost that transport
// a routed config lookup per request in cluster mode. Checking it here covers the
// REST JSON body, the REST binary wire, and the native TCP wire at once, for
// free.
func TestStageBulkRejectsWrongDim(t *testing.T) {
	store, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store.Close() }()

	if err := store.CreateCollection("docs", Config{
		Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1,
	}); err != nil {
		t.Fatal(err)
	}

	good := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}
	if err := store.StageBulk("docs", []uint64{1, 2}, good); err != nil {
		t.Fatalf("staging correct-dim vectors failed: %v", err)
	}

	cases := []struct {
		name string
		vecs [][]float32
	}{
		{"too short", [][]float32{{1, 2, 3}}},
		{"too long", [][]float32{{1, 2, 3, 4, 5}}},
		{"empty", [][]float32{{}}},
		{"nil", [][]float32{nil}},
		// Ragged input, for DIRECT callers of this method only.
		//
		// CORRECTION to an earlier claim that moving the dim check here
		// "closes ragged batches" and that "a rejected batch stages nothing, not
		// even the good prefix of a ragged one". The second half is true, and this
		// case proves it — but ONLY for a caller that reaches this method with a
		// ragged slice in hand. No remote request ever does.
		//
		// Every transport goes through ops.EncodeBulkStageArgs →
		// DecodeBulkStageArgs, and the decoder materializes make([]float32, dim) per
		// row from a single wire dim. By the time a staged batch arrives here it is
		// ALWAYS uniform, so this check structurally cannot observe raggedness.
		// ops.EncodeBulkStageArgs rejects it instead, before it can be turned into a
		// panic or into rows shifted under fabricated ids — see
		// TestEncodeBulkStageArgsRejectsRagged, which is where that contract lives.
		//
		// What this check does cover, and what that change got right, is a
		// UNIFORM batch whose dim is wrong for the collection: over the JSON body,
		// over the binary wire, and over the native TCP wire, at stage time rather
		// than at build time.
		{"ragged", [][]float32{{1, 2, 3, 4}, {1, 2, 3}}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ids := make([]uint64, len(c.vecs))
			for i := range ids {
				ids[i] = uint64(100 + i)
			}
			err := store.StageBulk("docs", ids, c.vecs)
			if !errors.Is(err, ErrDimMismatch) {
				t.Fatalf("err = %v, want ErrDimMismatch", err)
			}
		})
	}

	// A rejected batch stages NOTHING — not even the prefix of a ragged one. Only
	// the two good vectors from the start of the test survive to be built.
	if err := store.BuildStaged("docs", 1); err != nil {
		t.Fatalf("build after rejected stages: %v", err)
	}
	c, ok := store.Acquire("docs")
	if !ok {
		t.Fatal("collection vanished")
	}
	defer c.Release()
	if got := c.Stats().Size; got != len(good) {
		t.Fatalf("built %d points, want %d — a rejected batch left something staged", got, len(good))
	}
}

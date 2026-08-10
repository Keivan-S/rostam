// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"math/rand"
	"testing"
)

// THE BULK LOAD USED TO BE THE ONE WRITE PATH THAT COULD NOT SAY NO.
//
// insertLocked enforces MaxVectors and MaxBytes on every single insert, but the
// three bulk builders — hnsw.BuildConcurrentMeta, buildVamana and
// ivf.BuildConcurrentMeta — consulted neither, so the path that writes N points
// at once wrote all N regardless of the collection's budget. The load returned
// success and the quota counters never moved, leaving the collection
// permanently OVER its budget: the state that makes the free-then-reuse
// resurrection window reachable (see free_then_reuse_window_test.go).
//
// These tests run against all three index families through newIndex, because
// the bug was that the families DISAGREED with the insert path — a fix in one
// builder would leave the same hole in the other two.

func bulkQuotaVecs(n, dim int) ([]uint64, [][]float32) {
	rng := rand.New(rand.NewSource(int64(n)*7919 + int64(dim)))
	ids := make([]uint64, n)
	vecs := make([][]float32, n)
	for i := range vecs {
		ids[i] = uint64(i)
		v := make([]float32, dim)
		for d := range v {
			v[d] = rng.Float32()
		}
		vecs[i] = v
	}
	return ids, vecs
}

func bulkQuotaFamilies() []struct {
	name string
	kind IndexType
} {
	return []struct {
		name string
		kind IndexType
	}{
		{"hnsw", IndexHNSW},
		{"ivf", IndexIVF},
		{"vamana", IndexVamana},
	}
}

// TestBulkLoadRejectsOverMaxVectors pins that a bulk load larger than
// MaxVectors is REFUSED, and refused cleanly — the index must be left empty,
// not half-built, since the check runs before a single slot is reserved.
func TestBulkLoadRejectsOverMaxVectors(t *testing.T) {
	const (
		n   = 200
		dim = 8
	)
	ids, vecs := bulkQuotaVecs(n, dim)

	for _, fam := range bulkQuotaFamilies() {
		t.Run(fam.name, func(t *testing.T) {
			ix, err := newIndex(Config{
				Dim: dim, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 16, Seed: 3,
				IndexType:  fam.kind,
				MaxVectors: n / 2,
			})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = ix.Close() })

			if err := ix.BuildConcurrent(ids, vecs, 2); !errors.Is(err, ErrCollectionFull) {
				t.Fatalf("bulk-loading %d points into a MaxVectors=%d collection returned %v, want ErrCollectionFull",
					n, n/2, err)
			}
			// A refused load must leave NOTHING behind: the whole point of checking
			// before Reserve is that the caller can retry or resize without first
			// having to clean up a partially-populated index.
			if got := ix.Stats().Size; got != 0 {
				t.Fatalf("refused bulk load left %d points in the index, want 0", got)
			}
			// And it must be counted the way an insert-path rejection is, or the
			// rejection is invisible to operators watching quota_rejects_total.
			if got := ix.Stats().QuotaRejects; got == 0 {
				t.Error("refused bulk load did not increment QuotaRejects")
			}
		})
	}
}

// TestBulkLoadRejectsOverMaxBytes is the same for the byte budget, whose
// arithmetic (n inserts' worth of estimateInsertBytes) is the part that could
// silently drift from the insert path.
func TestBulkLoadRejectsOverMaxBytes(t *testing.T) {
	const (
		n   = 200
		dim = 8
		m   = 8
	)
	ids, vecs := bulkQuotaVecs(n, dim)
	per := estimateInsertBytes(dim, m)

	for _, fam := range bulkQuotaFamilies() {
		t.Run(fam.name, func(t *testing.T) {
			ix, err := newIndex(Config{
				Dim: dim, Metric: L2, M: m, EfConstruction: 32, EfSearch: 16, Seed: 3,
				IndexType: fam.kind,
				MaxBytes:  per * (n / 2),
			})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = ix.Close() })

			if err := ix.BuildConcurrent(ids, vecs, 2); !errors.Is(err, ErrCollectionFull) {
				t.Fatalf("bulk-loading %d points (%d bytes) into a MaxBytes=%d collection returned %v, want ErrCollectionFull",
					n, per*n, per*(n/2), err)
			}
			if got := ix.Stats().Size; got != 0 {
				t.Fatalf("refused bulk load left %d points in the index, want 0", got)
			}
		})
	}
}

// TestBulkLoadAtQuotaBoundarySucceeds is the other half, and the one that keeps
// the fix from being "reject bulk loads". A load that EXACTLY fills the budget
// is legal — it is what a caller sizing a collection to its data does — so the
// boundary must sit in the same place the insert path puts it: n <= MaxVectors
// succeeds, n > MaxVectors does not.
func TestBulkLoadAtQuotaBoundarySucceeds(t *testing.T) {
	const (
		n   = 200
		dim = 8
		m   = 8
	)
	ids, vecs := bulkQuotaVecs(n, dim)

	for _, fam := range bulkQuotaFamilies() {
		t.Run(fam.name, func(t *testing.T) {
			// Exactly at the vector cap AND exactly at the byte budget.
			ix, err := newIndex(Config{
				Dim: dim, Metric: L2, M: m, EfConstruction: 32, EfSearch: 16, Seed: 3,
				IndexType:  fam.kind,
				MaxVectors: n,
				MaxBytes:   estimateInsertBytes(dim, m) * n,
			})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = ix.Close() })

			if err := ix.BuildConcurrent(ids, vecs, 2); err != nil {
				t.Fatalf("bulk load of exactly MaxVectors=%d points was refused: %v — the quota must bound the load, not forbid filling it", n, err)
			}
			if got := ix.Stats().Size; got != n {
				t.Fatalf("bulk load stored %d points, want %d", got, n)
			}
			// One more point must now be refused, by the insert path, for the same
			// reason: the collection really is full. This is what proves the bulk
			// load left the accounting in the state the insert path expects rather
			// than merely passing its own check.
			if _, _, err := ix.Insert(uint64(n), vecs[0], 0, nil, nil, nil, CASCond{}); !errors.Is(err, ErrCollectionFull) {
				t.Fatalf("insert into a collection the bulk load filled to MaxVectors returned %v, want ErrCollectionFull", err)
			}
		})
	}
}

// TestBulkLoadUnquotedIsUnchanged pins that a collection with no quota set (the
// overwhelmingly common case) is completely unaffected — the guard must read as
// "unlimited", not as "zero".
func TestBulkLoadUnquotedIsUnchanged(t *testing.T) {
	const (
		n   = 200
		dim = 8
	)
	ids, vecs := bulkQuotaVecs(n, dim)

	for _, fam := range bulkQuotaFamilies() {
		t.Run(fam.name, func(t *testing.T) {
			ix, err := newIndex(Config{
				Dim: dim, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 16, Seed: 3,
				IndexType: fam.kind,
			})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = ix.Close() })

			if err := ix.BuildConcurrent(ids, vecs, 2); err != nil {
				t.Fatalf("bulk load into an unquoted collection was refused: %v", err)
			}
			if got := ix.Stats().Size; got != n {
				t.Fatalf("bulk load stored %d points, want %d", got, n)
			}
		})
	}
}

// TestBulkQuotaMatchesSerialInserts is the differential the other tests imply:
// for a range of caps, a bulk load of n points must be accepted on EXACTLY the
// caps under which inserting those n points one at a time would all succeed.
// The bulk path is an optimization of the serial one, so any disagreement about
// what fits is a bug in whichever path is the outlier.
func TestBulkQuotaMatchesSerialInserts(t *testing.T) {
	const (
		n   = 50
		dim = 8
		m   = 8
	)
	ids, vecs := bulkQuotaVecs(n, dim)

	for _, limit := range []int64{1, n / 2, n - 1, n, n + 1} {
		serialOK := func() bool {
			h, err := newHNSW(Config{
				Dim: dim, Metric: L2, M: m, EfConstruction: 32, EfSearch: 16, Seed: 3,
				MaxVectors: limit,
			})
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = h.Close() }()
			for i := range ids {
				if _, _, err := h.Insert(ids[i], vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
					return false
				}
			}
			return true
		}()

		bulkOK := func() bool {
			h, err := newHNSW(Config{
				Dim: dim, Metric: L2, M: m, EfConstruction: 32, EfSearch: 16, Seed: 3,
				MaxVectors: limit,
			})
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = h.Close() }()
			return h.BuildConcurrent(ids, vecs, 2) == nil
		}()

		if serialOK != bulkOK {
			t.Errorf("MaxVectors=%d: %d serial inserts all-succeed=%v but the bulk load succeeded=%v — the two write paths disagree about what fits",
				limit, n, serialOK, bulkOK)
		}
	}
}

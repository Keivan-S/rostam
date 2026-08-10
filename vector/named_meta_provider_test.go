// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"testing"
	"time"
)

// Named vectors: the optional external metadata provider for filtered
// search + scroll. These tests prove two invariants:
//
//  1. Regression: the nil-provider path (SearchFiltered / scrollDocs) is
//     byte-identical to today — it reads arena.Metadata and uses the payload
//     index exactly as before.
//  2. Provider path: with a non-nil metaOf func(id) Metadata the predicate is
//     evaluated against the EXTERNAL payload (mapped by point id), NOT the
//     arena, and never consults the (empty) payload index — predicate-eval only.

// buildMetalessCorpus inserts n random vectors with NO arena metadata (meta=nil)
// and returns the index, the vectors (1-indexed by id), and the rng-derived
// external payload map (id -> {"bucket": hit|miss}). Roughly matchFrac carry
// "hit". The arena therefore has an empty payload index; only the returned
// external map encodes the bucket.
func buildMetalessCorpus(t *testing.T, n, dim int, matchFrac float64, seed int64) (*hnsw, [][]float32, map[uint64]Metadata, map[uint64]bool) {
	t.Helper()
	h, err := newHNSW(Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed, Metric: L2})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	rng := rand.New(rand.NewSource(seed))
	vecs := make([][]float32, n+1)
	ext := make(map[uint64]Metadata, n)
	matches := make(map[uint64]bool, n)
	for i := 1; i <= n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		vecs[i] = v
		// Insert WITHOUT arena metadata: the sub-arena carries no payload.
		if _, _, err := h.Insert(uint64(i), v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
		if rng.Float64() < matchFrac {
			ext[uint64(i)] = Metadata{"bucket": NewString("hit")}
			matches[uint64(i)] = true
		} else {
			ext[uint64(i)] = Metadata{"bucket": NewString("miss")}
		}
	}
	return h, vecs, ext, matches
}

// TestSearchFilteredWithNilProviderMatchesSearchFiltered is the regression
// guard: SearchFilteredWith(..., nil) must return EXACTLY what SearchFiltered
// returns (the existing arena+payload-index path), proving the nil provider
// path is unchanged.
func TestSearchFilteredWithNilProviderMatchesSearchFiltered(t *testing.T) {
	h, _, _ := buildFilteredCorpus(t, 1000, 16, 0.5, 7)
	filter := Filter{Op: FilterEq, Field: "bucket", Value: NewString("hit")}
	query := make([]float32, 16)

	want, err := h.SearchFiltered(query, 25, filter)
	if err != nil {
		t.Fatalf("SearchFiltered: %v", err)
	}
	got, err := h.SearchFilteredWith(nil, query, 25, filter, nil)
	if err != nil {
		t.Fatalf("SearchFilteredWith(nil): %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("nil-provider result count differs: got %d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].ID != want[i].ID || got[i].Distance != want[i].Distance {
			t.Fatalf("nil-provider result[%d] differs: got %+v want %+v", i, got[i], want[i])
		}
	}
}

// TestScrollDocsWithNilProviderMatchesScrollDocs is the scroll regression
// guard: the nil-provider scroll variant equals today's scrollDocs.
func TestScrollDocsWithNilProviderMatchesScrollDocs(t *testing.T) {
	h, _, _ := buildFilteredCorpus(t, 300, 16, 0.5, 11)
	filter := Filter{Op: FilterEq, Field: "bucket", Value: NewString("hit")}

	want, err := h.scrollDocs(filter, 0)
	if err != nil {
		t.Fatalf("scrollDocs: %v", err)
	}
	got, err := h.scrollDocsWith(filter, 0, nil)
	if err != nil {
		t.Fatalf("scrollDocsWith(nil): %v", err)
	}
	wantIDs := docIDSet(want)
	gotIDs := docIDSet(got)
	if len(wantIDs) != len(gotIDs) {
		t.Fatalf("nil-provider scroll count differs: got %d want %d", len(gotIDs), len(wantIDs))
	}
	for id := range wantIDs {
		if !gotIDs[id] {
			t.Fatalf("nil-provider scroll missing id %d", id)
		}
	}
}

func docIDSet(docs []Document) map[uint64]bool {
	s := make(map[uint64]bool, len(docs))
	for _, d := range docs {
		s[d.ID] = true
	}
	return s
}

// TestSearchFilteredWithProviderReadsExternalPayload proves the provider path
// evaluates the predicate against the EXTERNAL map. The arena holds NO metadata,
// so a filter read from the arena would match NOTHING — yet via the provider it
// must find exactly the ids the external map marks "hit".
func TestSearchFilteredWithProviderReadsExternalPayload(t *testing.T) {
	h, _, ext, matches := buildMetalessCorpus(t, 1000, 16, 0.5, 3)
	filter := Filter{Op: FilterEq, Field: "bucket", Value: NewString("hit")}
	query := make([]float32, 16)

	// Sanity: with a nil provider (arena read) the filter matches nothing,
	// because the arena has no "bucket" field. This proves the corpus really is
	// metadata-less and the provider — not the arena — drives the next assertion.
	nilRes, err := h.SearchFilteredWith(nil, query, 25, filter, nil)
	if err != nil {
		t.Fatalf("SearchFilteredWith(nil): %v", err)
	}
	if len(nilRes) != 0 {
		t.Fatalf("expected 0 results from metadata-less arena, got %d", len(nilRes))
	}

	metaOf := func(id uint64) Metadata { return ext[id] }
	got, err := h.SearchFilteredWith(nil, query, 25, filter, metaOf)
	if err != nil {
		t.Fatalf("SearchFilteredWith(provider): %v", err)
	}
	if len(got) == 0 {
		t.Fatal("provider-path filtered search returned no results")
	}
	// Every returned id must satisfy the predicate per the EXTERNAL map.
	for _, r := range got {
		if !matches[r.ID] {
			t.Errorf("result id %d is not a 'hit' in the external map", r.ID)
		}
	}
}

// TestSearchFilteredWithProviderSelectiveNoFilterFirst covers the critical
// invariant: a SELECTIVE filter on the provider path must still return correct
// results. If the provider path mistakenly filter-first'd on the (empty) payload
// index, it would return empty/wrong. A highly selective external filter that
// matches a single id must find that id.
func TestSearchFilteredWithProviderSelectiveNoFilterFirst(t *testing.T) {
	h, _, ext, _ := buildMetalessCorpus(t, 800, 16, 0.0, 5)
	// Mark exactly ONE id as special in the external map (a very selective
	// filter — the payload index, were it consulted, has zero entries for it).
	special := uint64(123)
	ext[special] = Metadata{"bucket": NewString("special")}
	metaOf := func(id uint64) Metadata { return ext[id] }

	filter := Filter{Op: FilterEq, Field: "bucket", Value: NewString("special")}
	query := make([]float32, 16)

	got, err := h.SearchFilteredWith(nil, query, 10, filter, metaOf)
	if err != nil {
		t.Fatalf("SearchFilteredWith(provider): %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("expected exactly 1 result for the selective provider filter, got %d", len(got))
	}
	if got[0].ID != special {
		t.Fatalf("selective provider filter returned id %d, want %d", got[0].ID, special)
	}
}

// TestScrollDocsWithProviderReadsExternalPayload proves the scroll provider path
// filters against the external map (not the empty arena).
func TestScrollDocsWithProviderReadsExternalPayload(t *testing.T) {
	h, _, ext, matches := buildMetalessCorpus(t, 400, 16, 0.5, 9)
	filter := Filter{Op: FilterEq, Field: "bucket", Value: NewString("hit")}

	// Nil provider over the metadata-less arena → no matches.
	nilDocs, err := h.scrollDocsWith(filter, 0, nil)
	if err != nil {
		t.Fatalf("scrollDocsWith(nil): %v", err)
	}
	if len(nilDocs) != 0 {
		t.Fatalf("expected 0 scroll docs from metadata-less arena, got %d", len(nilDocs))
	}

	metaOf := func(id uint64) Metadata { return ext[id] }
	docs, err := h.scrollDocsWith(filter, 0, metaOf)
	if err != nil {
		t.Fatalf("scrollDocsWith(provider): %v", err)
	}
	if len(docs) == 0 {
		t.Fatal("provider-path scroll returned no docs")
	}
	for _, d := range docs {
		if !matches[d.ID] {
			t.Errorf("scroll doc id %d is not a 'hit' in the external map", d.ID)
		}
	}
	// Count must equal the number of external 'hit' ids (all live, no filter-first
	// shortcut on the empty index dropping them).
	wantN := 0
	for _, m := range matches {
		if m {
			wantN++
		}
	}
	if len(docs) != wantN {
		t.Fatalf("scroll provider doc count = %d, want %d (all external hits)", len(docs), wantN)
	}
}

// TestSearchFilteredWithProviderExcludesTombstonedAndExpired proves the
// tombstone/TTL liveness gate still applies on the provider path: a point that
// the external predicate WOULD match is nonetheless excluded from results once it
// is tombstoned (Delete) or TTL-expired. Admission (admitsWith) rejects these
// before the predicate is ever consulted, so they never reach the result set.
func TestSearchFilteredWithProviderExcludesTombstonedAndExpired(t *testing.T) {
	const (
		n   = 600
		dim = 16
	)
	h, err := newHNSW(Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 17, Metric: L2})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	// Deterministic clock so we can age exactly one point past its TTL.
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	rng := rand.New(rand.NewSource(17))
	ext := make(map[uint64]Metadata, n)
	const (
		tombstonedID = uint64(42)
		expiredID    = uint64(99)
		ttl          = 50 * time.Millisecond
	)
	for i := uint64(1); i <= n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		// Only the expired point carries a TTL; everyone else lives forever.
		pointTTL := time.Duration(0)
		if i == expiredID {
			pointTTL = ttl
		}
		// Insert WITHOUT arena metadata: payload lives only in the external map.
		if _, _, err := h.Insert(i, v, pointTTL, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
		// Every point is a "hit" in the external map, so the predicate alone would
		// admit all of them — only liveness can exclude any.
		ext[i] = Metadata{"bucket": NewString("hit")}
	}

	// Tombstone one matching id; age another past its TTL.
	if ok, _ := h.Delete(tombstonedID, CASCond{}); !ok {
		t.Fatalf("Delete(%d) returned false", tombstonedID)
	}
	fakeNow += 100 // now > deadline for the TTL point

	metaOf := func(id uint64) Metadata { return ext[id] }
	filter := Filter{Op: FilterEq, Field: "bucket", Value: NewString("hit")}
	query := make([]float32, dim)

	// Ask for the whole corpus so both excluded ids would be in range if admitted.
	got, err := h.SearchFilteredWith(nil, query, n, filter, metaOf)
	if err != nil {
		t.Fatalf("SearchFilteredWith(provider): %v", err)
	}
	if len(got) == 0 {
		t.Fatal("provider-path filtered search returned no results")
	}
	for _, r := range got {
		if r.ID == tombstonedID {
			t.Errorf("tombstoned id %d was returned on the provider path", tombstonedID)
		}
		if r.ID == expiredID {
			t.Errorf("TTL-expired id %d was returned on the provider path", expiredID)
		}
	}
}

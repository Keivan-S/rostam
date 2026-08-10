// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"reflect"
	"testing"
	"time"
)

// namedTestConfig is the standard two-space config used across the named tests:
// "title" dim 4 cosine, "image" dim 3 dot.
func namedTestConfig() map[string]NamedVectorParams {
	return map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: Cosine},
		"image": {Dim: 3, Metric: DotProduct},
	}
}

func docIDsOf(ds []Document) map[uint64]bool {
	s := make(map[uint64]bool, len(ds))
	for _, d := range ds {
		s[d.ID] = true
	}
	return s
}

// TestNamedCollectionSearchPerSpace inserts points (some omitting a space) and
// checks each named space ranks the nearest id first.
func TestNamedCollectionSearchPerSpace(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()

	// Point 1: both spaces. Point 2: both. Point 3: ONLY title (omits image).
	if err := nc.Insert(1, map[string][]float32{
		"title": {1, 0, 0, 0},
		"image": {1, 0, 0},
	}, Metadata{"kind": NewString("a")}, 0); err != nil {
		t.Fatalf("insert 1: %v", err)
	}
	if err := nc.Insert(2, map[string][]float32{
		"title": {0, 1, 0, 0},
		"image": {0, 1, 0},
	}, Metadata{"kind": NewString("b")}, 0); err != nil {
		t.Fatalf("insert 2: %v", err)
	}
	if err := nc.Insert(3, map[string][]float32{
		"title": {0, 0, 1, 0},
	}, Metadata{"kind": NewString("a")}, 0); err != nil {
		t.Fatalf("insert 3 (omits image): %v", err)
	}

	// title space: query near point 1's title vector → id 1 ranked first.
	titleRes, err := nc.SearchNamed("title", []float32{1, 0, 0, 0}, 3, Filter{})
	if err != nil {
		t.Fatalf("search title: %v", err)
	}
	if len(titleRes) == 0 || titleRes[0].ID != 1 {
		t.Fatalf("title top result = %v, want id 1 first", resultIDs(titleRes))
	}
	if len(titleRes) != 3 {
		t.Fatalf("title returned %d results, want 3 (all populated title)", len(titleRes))
	}

	// image space: only points 1 and 2 populated it (3 omitted) → 2 results.
	imgRes, err := nc.SearchNamed("image", []float32{0, 1, 0}, 5, Filter{})
	if err != nil {
		t.Fatalf("search image: %v", err)
	}
	if len(imgRes) != 2 {
		t.Fatalf("image returned %d results, want 2 (id 3 omitted image)", len(imgRes))
	}
	if imgRes[0].ID != 2 {
		t.Fatalf("image top result = %v, want id 2 first", resultIDs(imgRes))
	}
}

// TestNamedCollectionFilteredSearch checks the shared-payload predicate filter
// on a named search.
func TestNamedCollectionFilteredSearch(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	for id := uint64(1); id <= 4; id++ {
		kind := "a"
		if id%2 == 0 {
			kind = "b"
		}
		if err := nc.Insert(id, map[string][]float32{
			"title": {float32(id), 0, 0, 0},
		}, Metadata{"kind": NewString(kind)}, 0); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}
	filter := Filter{Op: FilterEq, Field: "kind", Value: NewString("a")}
	res, err := nc.SearchNamed("title", []float32{1, 0, 0, 0}, 10, filter)
	if err != nil {
		t.Fatalf("filtered search: %v", err)
	}
	if len(res) != 2 {
		t.Fatalf("filtered search returned %d, want 2 (ids 1,3)", len(res))
	}
	for _, r := range res {
		if r.ID%2 == 0 {
			t.Errorf("filtered search returned even id %d (kind b)", r.ID)
		}
	}
}

// TestNamedCollectionDeleteRemovesFromAllSpaces deletes a point and verifies it
// is gone from BOTH spaces + the shared payload (search & scroll exclude it).
func TestNamedCollectionDeleteRemovesFromAllSpaces(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	for id := uint64(1); id <= 3; id++ {
		if err := nc.Insert(id, map[string][]float32{
			"title": {float32(id), 0, 0, 0},
			"image": {float32(id), 0, 0},
		}, Metadata{"id": NewInt(int64(id))}, 0); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}
	existed, err := nc.Delete(2)
	if err != nil {
		t.Fatalf("delete: %v", err)
	}
	if !existed {
		t.Fatal("delete reported id 2 absent")
	}
	// title space excludes 2.
	tr, _ := nc.SearchNamed("title", []float32{2, 0, 0, 0}, 10, Filter{})
	for _, r := range tr {
		if r.ID == 2 {
			t.Error("deleted id 2 still in title space")
		}
	}
	// image space excludes 2.
	ir, _ := nc.SearchNamed("image", []float32{2, 0, 0}, 10, Filter{})
	for _, r := range ir {
		if r.ID == 2 {
			t.Error("deleted id 2 still in image space")
		}
	}
	// scroll excludes 2.
	docs, _ := nc.ScrollDocs(Filter{}, 0)
	if docIDsOf(docs)[2] {
		t.Error("deleted id 2 still in scroll")
	}
	if len(docs) != 2 {
		t.Fatalf("scroll returned %d live points, want 2", len(docs))
	}
	// re-delete is a no-op false.
	again, _ := nc.Delete(2)
	if again {
		t.Error("second delete of id 2 reported existed=true")
	}
}

// TestNamedCollectionScrollSharedPayload verifies scroll returns live points
// with the shared payload, including a point that omitted a space.
func TestNamedCollectionScrollSharedPayload(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	_ = nc.Insert(1, map[string][]float32{"title": {1, 0, 0, 0}}, Metadata{"g": NewString("x")}, 0)
	_ = nc.Insert(2, map[string][]float32{"image": {1, 0, 0}}, Metadata{"g": NewString("y")}, 0)

	all, err := nc.ScrollDocs(Filter{}, 0)
	if err != nil {
		t.Fatalf("scroll: %v", err)
	}
	if len(all) != 2 {
		t.Fatalf("scroll all = %d, want 2 (both points incl. partial)", len(all))
	}
	for _, d := range all {
		if d.Metadata == nil || d.Metadata["g"].Str == "" {
			t.Errorf("scroll doc %d missing shared payload: %+v", d.ID, d.Metadata)
		}
	}
	// filtered scroll on shared payload.
	x, err := nc.ScrollDocs(Filter{Op: FilterEq, Field: "g", Value: NewString("x")}, 0)
	if err != nil {
		t.Fatalf("filtered scroll: %v", err)
	}
	if len(x) != 1 || x[0].ID != 1 {
		t.Fatalf("filtered scroll = %v, want only id 1", docIDsOf(x))
	}
}

// TestNamedCollectionUpsertReplaces re-inserts a point and verifies its vectors
// + payload are replaced.
func TestNamedCollectionUpsertReplaces(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	_ = nc.Insert(1, map[string][]float32{"title": {1, 0, 0, 0}}, Metadata{"v": NewInt(1)}, 0)
	_ = nc.Insert(2, map[string][]float32{"title": {0, 1, 0, 0}}, Metadata{"v": NewInt(2)}, 0)

	// Re-insert id 1 with a NEW title vector + payload.
	if err := nc.Insert(1, map[string][]float32{"title": {0, 0, 0, 1}}, Metadata{"v": NewInt(99)}, 0); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	// Query near the new vector → id 1 first.
	res, _ := nc.SearchNamed("title", []float32{0, 0, 0, 1}, 2, Filter{})
	if len(res) == 0 || res[0].ID != 1 {
		t.Fatalf("after upsert top = %v, want id 1 first", resultIDs(res))
	}
	// Payload replaced.
	docs, _ := nc.ScrollDocs(Filter{Op: FilterEq, Field: "v", Value: NewInt(99)}, 0)
	if len(docs) != 1 || docs[0].ID != 1 {
		t.Fatalf("upserted payload not replaced: scroll v=99 = %v", docIDsOf(docs))
	}
	// Old payload gone.
	old, _ := nc.ScrollDocs(Filter{Op: FilterEq, Field: "v", Value: NewInt(1)}, 0)
	if len(old) != 0 {
		t.Fatalf("old payload v=1 still present: %v", docIDsOf(old))
	}
}

// TestNamedCollectionScrollExcludesExpired verifies ScrollDocs excludes a point
// whose shared-payload TTL deadline has passed (via the injectable nc.now clock)
// while retaining a point with no TTL. Both the insert-deadline computation and
// the scroll expiry check route through nc.now, so aging is deterministic with
// no sleeps.
func TestNamedCollectionScrollExcludesExpired(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()

	// Fixed fake clock; insert computes the deadline from this value.
	var fakeNow int64 = 1_000_000
	nc.now = func() int64 { return fakeNow }

	// Point 1: short TTL (real deadline = fakeNow + 50ms). Point 2: no TTL.
	if err := nc.Insert(1, map[string][]float32{"title": {1, 0, 0, 0}}, Metadata{"g": NewString("exp")}, 50*time.Millisecond); err != nil {
		t.Fatalf("insert 1 (ttl): %v", err)
	}
	if err := nc.Insert(2, map[string][]float32{"title": {0, 1, 0, 0}}, Metadata{"g": NewString("keep")}, 0); err != nil {
		t.Fatalf("insert 2 (no ttl): %v", err)
	}

	// Before expiry: both live.
	if all, _ := nc.ScrollDocs(Filter{}, 0); len(all) != 2 {
		t.Fatalf("pre-expiry scroll = %v, want both points", docIDsOf(all))
	}

	// Advance the clock PAST point 1's deadline.
	fakeNow += 100

	docs, err := nc.ScrollDocs(Filter{}, 0)
	if err != nil {
		t.Fatalf("scroll: %v", err)
	}
	ids := docIDsOf(docs)
	if ids[1] {
		t.Error("expired point 1 still returned by scroll")
	}
	if !ids[2] {
		t.Error("non-expired point 2 missing from scroll")
	}
	if len(docs) != 1 {
		t.Fatalf("scroll returned %d docs, want 1 (only non-expired)", len(docs))
	}
}

// TestNamedCollectionUpsertOmittedSpaceRetained verifies that re-inserting a
// point with ONLY one of its spaces leaves the omitted space's vector untouched
// while updating the named space and the shared payload.
func TestNamedCollectionUpsertOmittedSpaceRetained(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()

	// Insert P into BOTH spaces.
	if err := nc.Insert(1, map[string][]float32{
		"title": {1, 0, 0, 0},
		"image": {1, 0, 0},
	}, Metadata{"v": NewInt(1)}, 0); err != nil {
		t.Fatalf("insert P: %v", err)
	}
	// A second point so each space has >1 entry to rank against.
	if err := nc.Insert(2, map[string][]float32{
		"title": {0, 1, 0, 0},
		"image": {0, 1, 0},
	}, Metadata{"v": NewInt(2)}, 0); err != nil {
		t.Fatalf("insert other: %v", err)
	}

	// Re-insert P with ONLY "title" changed; "image" omitted + new payload.
	if err := nc.Insert(1, map[string][]float32{
		"title": {0, 0, 0, 1},
	}, Metadata{"v": NewInt(99)}, 0); err != nil {
		t.Fatalf("re-insert P (title only): %v", err)
	}

	// (a) title search reflects the NEW title vector.
	tr, _ := nc.SearchNamed("title", []float32{0, 0, 0, 1}, 2, Filter{})
	if len(tr) == 0 || tr[0].ID != 1 {
		t.Fatalf("title top = %v, want id 1 first (new title vector)", resultIDs(tr))
	}

	// (b) image search STILL returns P with its ORIGINAL image vector.
	ir, _ := nc.SearchNamed("image", []float32{1, 0, 0}, 2, Filter{})
	if len(ir) == 0 || ir[0].ID != 1 {
		t.Fatalf("image top = %v, want id 1 first (original image vector retained)", resultIDs(ir))
	}

	// (c) shared payload updated to the re-insert's payload.
	docs, _ := nc.ScrollDocs(Filter{Op: FilterEq, Field: "v", Value: NewInt(99)}, 0)
	if len(docs) != 1 || docs[0].ID != 1 {
		t.Fatalf("shared payload not updated: scroll v=99 = %v", docIDsOf(docs))
	}
	if old, _ := nc.ScrollDocs(Filter{Op: FilterEq, Field: "v", Value: NewInt(1)}, 0); len(old) != 0 {
		t.Fatalf("old payload v=1 still present: %v", docIDsOf(old))
	}
}

// TestNamedCollectionValidationErrors covers fail-loud paths: unknown vector
// name on insert/search, dim mismatch on insert, and create-time config errors.
func TestNamedCollectionValidationErrors(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()

	// Unknown name on insert.
	if err := nc.Insert(1, map[string][]float32{"nope": {1, 0, 0, 0}}, nil, 0); !errors.Is(err, ErrUnknownVectorName) {
		t.Errorf("insert unknown name err = %v, want ErrUnknownVectorName", err)
	}
	// Dim mismatch on insert.
	if err := nc.Insert(1, map[string][]float32{"title": {1, 0, 0}}, nil, 0); !errors.Is(err, ErrDimMismatch) {
		t.Errorf("insert dim mismatch err = %v, want ErrDimMismatch", err)
	}
	// Unknown name on search.
	if _, err := nc.SearchNamed("nope", []float32{1, 0, 0, 0}, 3, Filter{}); !errors.Is(err, ErrUnknownVectorName) {
		t.Errorf("search unknown name err = %v, want ErrUnknownVectorName", err)
	}

	// Create-time config errors.
	if _, err := NewNamedCollection("c", map[string]NamedVectorParams{}); !errors.Is(err, ErrEmptyNamedVectors) {
		t.Errorf("empty config err = %v, want ErrEmptyNamedVectors", err)
	}
	if _, err := NewNamedCollection("c", map[string]NamedVectorParams{"": {Dim: 4}}); !errors.Is(err, ErrEmptyVectorName) {
		t.Errorf("empty name err = %v, want ErrEmptyVectorName", err)
	}
	if _, err := NewNamedCollection("c", map[string]NamedVectorParams{"bad#name": {Dim: 4}}); !errors.Is(err, ErrReservedVectorName) {
		t.Errorf("reserved name err = %v, want ErrReservedVectorName", err)
	}
	if _, err := NewNamedCollection("c", map[string]NamedVectorParams{"title": {Dim: 0}}); !errors.Is(err, ErrInvalidDim) {
		t.Errorf("bad dim err = %v, want ErrInvalidDim", err)
	}
}

// TestNamedCollectionSnapshotRoundtrip builds a named collection (2 spaces, a
// point omitting a space, a shared payload + a TTL), Snapshot -> Restore into a
// FRESH collection, and asserts every sub-index's search results, the shared
// payload (per point), the ttl, and the live id set come back exactly — plus a
// filtered search post-restore still works (proves the shared payload survived).
func TestNamedCollectionSnapshotRoundtrip(t *testing.T) {
	src, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer src.Close()

	// Point 1: both spaces, payload kind=a. Point 2: both, kind=b. Point 3: ONLY
	// title (omits image), kind=a, with a (far-future) TTL so the deadline is
	// non-zero and must round-trip.
	if err := src.Insert(1, map[string][]float32{"title": {1, 0, 0, 0}, "image": {1, 0, 0}}, Metadata{"kind": NewString("a")}, 0); err != nil {
		t.Fatalf("insert 1: %v", err)
	}
	if err := src.Insert(2, map[string][]float32{"title": {0, 1, 0, 0}, "image": {0, 1, 0}}, Metadata{"kind": NewString("b")}, 0); err != nil {
		t.Fatalf("insert 2: %v", err)
	}
	if err := src.Insert(3, map[string][]float32{"title": {0, 0, 1, 0}}, Metadata{"kind": NewString("a")}, time.Hour); err != nil {
		t.Fatalf("insert 3: %v", err)
	}

	var buf bytes.Buffer
	if err := src.Snapshot(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}

	// Restore into a fresh collection with the matching config.
	dst, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new dst: %v", err)
	}
	defer dst.Close()
	if err := dst.Restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("restore: %v", err)
	}

	// Live id set matches exactly.
	if got := dst.NumPoints(); got != 3 {
		t.Fatalf("restored NumPoints = %d, want 3", got)
	}

	// title space: all 3 populated → 3 results, id 1 nearest to its own vector.
	titleRes, err := dst.SearchNamed("title", []float32{1, 0, 0, 0}, 5, Filter{})
	if err != nil {
		t.Fatalf("search title: %v", err)
	}
	if len(titleRes) != 3 || titleRes[0].ID != 1 {
		t.Fatalf("title results = %v, want 3 with id 1 first", resultIDs(titleRes))
	}
	// image space: only points 1,2 populated it (3 omitted) → 2 results.
	imgRes, err := dst.SearchNamed("image", []float32{0, 1, 0}, 5, Filter{})
	if err != nil {
		t.Fatalf("search image: %v", err)
	}
	if len(imgRes) != 2 || imgRes[0].ID != 2 {
		t.Fatalf("image results = %v, want 2 with id 2 first", resultIDs(imgRes))
	}

	// Shared payload per point matches exactly (compared against the source).
	src.mu.RLock()
	dst.mu.RLock()
	if len(dst.meta) != len(src.meta) {
		t.Fatalf("restored meta size = %d, want %d", len(dst.meta), len(src.meta))
	}
	for id, sm := range src.meta {
		dm := dst.meta[id]
		if len(dm) != len(sm) {
			t.Fatalf("point %d meta size = %d, want %d", id, len(dm), len(sm))
		}
		for k, v := range sm {
			if !reflect.DeepEqual(dm[k], v) {
				t.Errorf("point %d meta[%q] = %+v, want %+v", id, k, dm[k], v)
			}
		}
	}
	// TTL deadlines match exactly (point 3 non-zero; 1,2 zero).
	for id, dl := range src.ttl {
		if dst.ttl[id] != dl {
			t.Errorf("point %d ttl = %d, want %d", id, dst.ttl[id], dl)
		}
	}
	if src.ttl[3] == 0 {
		t.Error("expected non-zero ttl deadline for point 3 in source")
	}
	src.mu.RUnlock()
	dst.mu.RUnlock()

	// Filtered search post-restore proves the shared payload is usable: kind=a →
	// ids 1 and 3 only.
	filtered, err := dst.SearchNamed("title", []float32{1, 0, 0, 0}, 10, Filter{Op: FilterEq, Field: "kind", Value: NewString("a")})
	if err != nil {
		t.Fatalf("filtered search: %v", err)
	}
	if len(filtered) != 2 {
		t.Fatalf("filtered search returned %d, want 2 (ids 1,3)", len(filtered))
	}
	for _, r := range filtered {
		if r.ID != 1 && r.ID != 3 {
			t.Errorf("filtered search returned id %d, want only 1 or 3", r.ID)
		}
	}
}

// TestNamedGetLive proves Get returns the per-space vector map (an omitted space
// is absent from the map), the shared payload, and the remaining TTL for a live
// point, and that the returned data is a DEEP COPY (caller mutation does not
// corrupt the sub-arenas or the shared meta map).
func TestNamedGetLive(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	var fakeNow int64 = 1_000_000
	nc.now = func() int64 { return fakeNow }

	// Point 1: BOTH spaces, with a TTL. Point 2: ONLY title (omits image), no TTL.
	if err := nc.Insert(1, map[string][]float32{
		"title": {1, 0, 0, 0},
		"image": {0, 0, 1},
	}, Metadata{"a": NewInt(1), "tag": NewString("x")}, 500*time.Millisecond); err != nil {
		t.Fatalf("insert 1: %v", err)
	}
	if err := nc.Insert(2, map[string][]float32{
		"title": {0, 1, 0, 0},
	}, Metadata{"a": NewInt(2)}, 0); err != nil {
		t.Fatalf("insert 2: %v", err)
	}

	// Point 1: both spaces present, payload + ttl correct.
	vecs, payload, ttl, _, ok := nc.Get(1)
	if !ok {
		t.Fatal("Get(1) ok=false, want true")
	}
	if len(vecs) != 2 {
		t.Fatalf("Get(1) returned %d spaces, want 2 (title,image)", len(vecs))
	}
	if len(vecs["title"]) != 4 || len(vecs["image"]) != 3 {
		t.Fatalf("Get(1) space dims wrong: title=%v image=%v", vecs["title"], vecs["image"])
	}
	if payload["a"].Int != 1 || payload["tag"].Str != "x" {
		t.Fatalf("Get(1) payload = %v, want {a:1, tag:x}", payload)
	}
	if ttl <= 0 || ttl > 500*time.Millisecond {
		t.Fatalf("Get(1) ttl = %v, want (0, 500ms]", ttl)
	}

	// Point 2 omitted image → only title in the map; no TTL.
	vecs2, _, ttl2, _, ok := nc.Get(2)
	if !ok {
		t.Fatal("Get(2) ok=false, want true")
	}
	if len(vecs2) != 1 {
		t.Fatalf("Get(2) returned %d spaces, want 1 (title only)", len(vecs2))
	}
	if _, has := vecs2["image"]; has {
		t.Fatal("Get(2) included image space, but point 2 omitted it")
	}
	if ttl2 != 0 {
		t.Fatalf("Get(2) ttl = %v, want 0 (no expiry)", ttl2)
	}

	// Deep-copy isolation: mutating returned vector/payload must not corrupt state.
	vecs["title"][0] = -99
	payload["a"] = NewInt(999)
	_, payload2, _, _, _ := nc.Get(1)
	if payload2["a"].Int != 1 {
		t.Fatalf("shared meta corrupted by caller mutation: a=%d, want 1", payload2["a"].Int)
	}
	nc.mu.RLock()
	if vm := nc.indexes["title"].vecsForIDs([]uint64{1}); vm[1][0] == -99 {
		t.Fatal("sub-arena vector corrupted by caller mutation")
	}
	nc.mu.RUnlock()
}

// TestNamedGetAbsentExpired covers the not-found liveness cases: an absent id and
// a TTL-expired id both yield ok=false.
func TestNamedGetAbsentExpired(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	var fakeNow int64 = 1_000_000
	nc.now = func() int64 { return fakeNow }

	// Absent.
	if _, _, _, _, ok := nc.Get(42); ok {
		t.Fatal("Get(absent) ok=true, want false")
	}

	// TTL-expired.
	if err := nc.Insert(1, map[string][]float32{"title": {1, 0, 0, 0}}, Metadata{"g": NewString("exp")}, 50*time.Millisecond); err != nil {
		t.Fatalf("insert: %v", err)
	}
	if _, _, _, _, ok := nc.Get(1); !ok {
		t.Fatal("Get(1) pre-expiry ok=false, want true")
	}
	fakeNow += 100
	if _, _, _, _, ok := nc.Get(1); ok {
		t.Fatal("Get(expired) ok=true, want false")
	}
}

// TestNamedPayloadSemantics is the merge-vs-overwrite-vs-delete-keys-vs-clear
// table over the shared meta map, asserting both Get and a subsequent FILTERED
// scroll (predicate-eval) reflect each mutation — proving the shared meta map is
// authoritative for filtering with NO payload index.
func TestNamedPayloadSemantics(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	if err := nc.Insert(1, map[string][]float32{"title": {1, 0, 0, 0}}, Metadata{"a": NewInt(1)}, 0); err != nil {
		t.Fatalf("insert: %v", err)
	}

	// SET (merge): add b, keep a.
	if err := nc.SetPayload(1, Metadata{"b": NewInt(2)}, nil); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}
	_, p, _, _, _ := nc.Get(1)
	if p["a"].Int != 1 || p["b"].Int != 2 {
		t.Fatalf("after merge Get = %v, want {a:1,b:2}", p)
	}
	// A filtered scroll on the ADDED field finds it (predicate-eval over shared meta).
	if d, _ := nc.ScrollDocs(Filter{Op: FilterEq, Field: "b", Value: NewInt(2)}, 0); len(d) != 1 || d[0].ID != 1 {
		t.Fatalf("filtered scroll b==2 = %v, want id 1", docIDsOf(d))
	}

	// OVERWRITE: replace whole payload with {c:3}; a and b gone.
	if err := nc.OverwritePayload(1, Metadata{"c": NewInt(3)}, nil); err != nil {
		t.Fatalf("OverwritePayload: %v", err)
	}
	_, p, _, _, _ = nc.Get(1)
	if len(p) != 1 || p["c"].Int != 3 {
		t.Fatalf("after overwrite Get = %v, want {c:3}", p)
	}
	if d, _ := nc.ScrollDocs(Filter{Op: FilterEq, Field: "a", Value: NewInt(1)}, 0); len(d) != 0 {
		t.Fatalf("filtered scroll a==1 after overwrite = %v, want none", docIDsOf(d))
	}

	// SET again to give us keys to delete: {c:3, d:4}.
	if err := nc.SetPayload(1, Metadata{"d": NewInt(4)}, nil); err != nil {
		t.Fatalf("SetPayload(d): %v", err)
	}
	// DELETE-KEYS: remove c (and an absent key — no-op).
	if err := nc.DeletePayloadKeys(1, []string{"c", "absent"}); err != nil {
		t.Fatalf("DeletePayloadKeys: %v", err)
	}
	_, p, _, _, _ = nc.Get(1)
	if len(p) != 1 || p["d"].Int != 4 {
		t.Fatalf("after delete-keys Get = %v, want {d:4}", p)
	}
	if d, _ := nc.ScrollDocs(Filter{Op: FilterEq, Field: "c", Value: NewInt(3)}, 0); len(d) != 0 {
		t.Fatalf("filtered scroll c==3 after delete-keys = %v, want none", docIDsOf(d))
	}

	// CLEAR: payload → empty.
	if err := nc.ClearPayload(1); err != nil {
		t.Fatalf("ClearPayload: %v", err)
	}
	_, p, _, _, _ = nc.Get(1)
	if len(p) != 0 {
		t.Fatalf("after clear Get = %v, want empty", p)
	}
	if d, _ := nc.ScrollDocs(Filter{Op: FilterEq, Field: "d", Value: NewInt(4)}, 0); len(d) != 0 {
		t.Fatalf("filtered scroll d==4 after clear = %v, want none", docIDsOf(d))
	}

	// The vector is untouched by all the payload ops: id 1 still searchable.
	if res, _ := nc.SearchNamed("title", []float32{1, 0, 0, 0}, 1, Filter{}); len(res) != 1 || res[0].ID != 1 {
		t.Fatalf("vector unexpectedly changed by payload ops: search = %v", resultIDs(res))
	}
}

// TestNamedPayloadDeadPoint asserts every payload op on an absent (and on a
// TTL-expired) point returns ErrIDNotFound.
func TestNamedPayloadDeadPoint(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	var fakeNow int64 = 1_000_000
	nc.now = func() int64 { return fakeNow }

	if err := nc.SetPayload(1, Metadata{"a": NewInt(1)}, nil); !errors.Is(err, ErrIDNotFound) {
		t.Fatalf("SetPayload(absent) = %v, want ErrIDNotFound", err)
	}
	if err := nc.OverwritePayload(1, Metadata{"a": NewInt(1)}, nil); !errors.Is(err, ErrIDNotFound) {
		t.Fatalf("OverwritePayload(absent) = %v, want ErrIDNotFound", err)
	}
	if err := nc.DeletePayloadKeys(1, []string{"a"}); !errors.Is(err, ErrIDNotFound) {
		t.Fatalf("DeletePayloadKeys(absent) = %v, want ErrIDNotFound", err)
	}
	if err := nc.ClearPayload(1); !errors.Is(err, ErrIDNotFound) {
		t.Fatalf("ClearPayload(absent) = %v, want ErrIDNotFound", err)
	}

	// Expired point: also dead → ErrIDNotFound.
	if err := nc.Insert(2, map[string][]float32{"title": {1, 0, 0, 0}}, Metadata{"g": NewString("x")}, 50*time.Millisecond); err != nil {
		t.Fatalf("insert: %v", err)
	}
	fakeNow += 100
	if err := nc.ClearPayload(2); !errors.Is(err, ErrIDNotFound) {
		t.Fatalf("ClearPayload(expired) = %v, want ErrIDNotFound", err)
	}
}

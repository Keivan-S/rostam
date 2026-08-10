// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// mvTokenAt produces a single token at a distinct angle in the (x,y) plane so
// MaxSim scores are strictly monotonic and tie-free (mirrors the embedded MV
// test): θ_i grows with i, all in the first quadrant, so the query == doc i's
// direction makes doc i the unique max.
func mvTokenAt(i int) []float32 {
	theta := float64(i) * (math.Pi / 2 / 40)
	return []float32{float32(math.Cos(theta)), float32(math.Sin(theta)), 0, 0}
}

// tieFreeVec produces strictly-separated unit vectors so that cosine distances
// to tieFreeQuery are all distinct — HNSW heap order is non-deterministic on
// ties, so tie-free vectors keep ranking deterministic across partitions.
func tieFreeVec(i int) []float32 {
	v := make([]float32, 8)
	// A gentle ramp on the first component plus a unique perturbation makes
	// every vector's angle to the query distinct.
	v[0] = 1.0
	v[1] = float32(i) * 0.01
	return v
}

// tieFreeQuery is aligned with the first component; distance grows monotonically
// with i, so the expected ranking is id 0,1,2,... with no ties.
func tieFreeQuery() []float32 {
	q := make([]float32, 8)
	q[0] = 1.0
	return q
}

func TestFanoutDispatcherSearchParity(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)
	ctx := context.Background()

	// Partitioned collection (P=4) — exercises the fan-out path.
	cfg := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: 4}
	if err := emb.CreateCollection(ctx, "p", cfg); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 200; i++ {
		if err := emb.VectorInsert(ctx, "p", uint64(i), tieFreeVec(i)); err != nil {
			t.Fatal(err)
		}
	}

	fan := rostam.NewFanoutDispatcher(emb, emb.Node())
	q := tieFreeQuery()

	want, _, err := emb.VectorSearchExt(ctx, "p", q, 10, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := fan.Call("vector_search", ops.EncodeVectorSearchArgsExt("p", 10, q, vector.Filter{}))
	if err != nil {
		t.Fatal(err)
	}
	got, err := ops.DecodeVectorSearchResults(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("len got %d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].ID != want[i].ID {
			t.Fatalf("rank %d: got id %d want %d", i, got[i].ID, want[i].ID)
		}
	}

	// Unpartitioned collection (P=1) — must pass through byte-identical.
	cfg1 := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: 1}
	if err := emb.CreateCollection(ctx, "u", cfg1); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 50; i++ {
		if err := emb.VectorInsert(ctx, "u", uint64(i), tieFreeVec(i)); err != nil {
			t.Fatal(err)
		}
	}
	args := ops.EncodeVectorSearchArgsExt("u", 10, q, vector.Filter{})
	wantBytes, err := emb.Node().Call("vector_search", args)
	if err != nil {
		t.Fatal(err)
	}
	gotBytes, err := fan.Call("vector_search", args)
	if err != nil {
		t.Fatal(err)
	}
	if string(gotBytes) != string(wantBytes) {
		t.Fatalf("passthrough mismatch: got %d bytes want %d bytes", len(gotBytes), len(wantBytes))
	}
}

// TestFanoutDispatcherScrollParity covers vector_scroll: the decorator must
// decode the At1-layout args, fan out across partitions, and re-encode the
// returned docs so they match e.VectorScroll. It also asserts byte-identical
// passthrough for an At1-layout op on an unpartitioned collection.
func TestFanoutDispatcherScrollParity(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)
	ctx := context.Background()

	cfg := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: 4}
	if err := emb.CreateCollection(ctx, "p", cfg); err != nil {
		t.Fatal(err)
	}
	const n = 120
	for i := 0; i < n; i++ {
		if err := emb.VectorInsert(ctx, "p", uint64(i), tieFreeVec(i)); err != nil {
			t.Fatal(err)
		}
	}

	fan := rostam.NewFanoutDispatcher(emb, emb.Node())

	want, _, _, err := emb.VectorScroll(ctx, "p", vector.Filter{}, 0, rostam.VectorScrollOpts{})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := fan.Call("vector_scroll", ops.EncodeScrollArgs("p", vector.Filter{}, 0))
	if err != nil {
		t.Fatal(err)
	}
	// Decode via DecodeScrollResult — the same path the networked client uses —
	// so this test tracks the production wire format, not a backward-compat read.
	got, _, _, _, err := ops.DecodeScrollResult(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != n {
		t.Fatalf("scroll count got %d want %d", len(got), n)
	}
	gotIDs := docIDSet(got)
	wantIDs := docIDSet(want)
	if len(gotIDs) != len(wantIDs) {
		t.Fatalf("scroll id-set sizes differ: got %d want %d", len(gotIDs), len(wantIDs))
	}
	for id := range wantIDs {
		if !gotIDs[id] {
			t.Fatalf("scroll missing id %d", id)
		}
	}

	// Unpartitioned scroll: the dispatcher now WRAPS the plain handler's docs with
	// next_cursor (EncodeScrollResult) so the client-facing wire is uniform with the
	// partitioned path — it is no longer a byte-identical passthrough. We assert the
	// wrapped result's docs equal the plain handler's docs (the inner per-shard
	// EncodeVectorDocs body is preserved AS-IS, then re-wrapped). Scroll iterates a
	// Go map (non-deterministic order), so we scope the filter to exactly one match
	// for a stable single-element result.
	cfg1 := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: 1}
	if err := emb.CreateCollection(ctx, "u", cfg1); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 30; i++ {
		md := rostam.VectorMetadata{"id": vector.NewInt(int64(i))}
		if err := emb.VectorInsertExt(ctx, "u", uint64(i), tieFreeVec(i), rostam.VectorInsertOpts{Metadata: md}); err != nil {
			t.Fatal(err)
		}
	}
	oneFilter := vector.Filter{Op: vector.FilterEq, Field: "id", Value: vector.NewInt(7)}
	args := ops.EncodeScrollArgs("u", oneFilter, 0)
	innerBytes, err := emb.Node().Call("vector_scroll", args)
	if err != nil {
		t.Fatal(err)
	}
	innerDocs, err := ops.DecodeVectorDocs(innerBytes)
	if err != nil {
		t.Fatal(err)
	}
	gotBytes, err := fan.Call("vector_scroll", args)
	if err != nil {
		t.Fatal(err)
	}
	wrappedDocs, _, _, next, err := ops.DecodeScrollResult(gotBytes)
	if err != nil {
		t.Fatalf("unpartitioned scroll result decode: %v", err)
	}
	if len(wrappedDocs) != len(innerDocs) {
		t.Fatalf("unpartitioned scroll: wrapped %d docs, inner %d", len(wrappedDocs), len(innerDocs))
	}
	for i := range innerDocs {
		if wrappedDocs[i].ID != innerDocs[i].ID {
			t.Fatalf("unpartitioned scroll doc[%d] id = %d, want %d", i, wrappedDocs[i].ID, innerDocs[i].ID)
		}
	}
	// limit=0 (unlimited) ⇒ exhausted ⇒ empty next_cursor.
	if next != "" {
		t.Fatalf("unpartitioned unlimited scroll: next_cursor = %q, want \"\"", next)
	}
}

// TestFanoutDispatcherScrollNextCursorWire asserts the dispatcher emits the
// SERVER-authoritative next_cursor on the result wire (not a client re-derivation)
// for the partitioned path: a full page (len==limit) yields a non-empty cursor
// that decodes to the page's max id, and the final short page yields "".
func TestFanoutDispatcherScrollNextCursorWire(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)
	ctx := context.Background()

	cfg := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: 4}
	if err := emb.CreateCollection(ctx, "p", cfg); err != nil {
		t.Fatal(err)
	}
	const n = 50
	for i := 1; i <= n; i++ {
		if err := emb.VectorInsert(ctx, "p", uint64(i), tieFreeVec(i)); err != nil {
			t.Fatal(err)
		}
	}
	fan := rostam.NewFanoutDispatcher(emb, emb.Node())

	const limit = 10
	seen := map[uint64]bool{}
	cursor := ""
	pages := 0
	for {
		afterID, hasAfter, derr := ops.DecodeScrollCursor(cursor)
		if derr != nil {
			t.Fatalf("decode cursor: %v", derr)
		}
		raw, err := fan.Call("vector_scroll",
			ops.EncodeScrollArgsCursorBounded("p", vector.Filter{}, limit, 0, 0, afterID, hasAfter, 0))
		if err != nil {
			t.Fatal(err)
		}
		docs, _, _, next, err := ops.DecodeScrollResult(raw)
		if err != nil {
			t.Fatalf("decode scroll result: %v", err)
		}
		// Compare the wire next_cursor to the independently-derived expectation.
		wantNext := ""
		if len(docs) == limit {
			wantNext = ops.EncodeScrollCursor(docs[len(docs)-1].ID)
		}
		if next != wantNext {
			t.Fatalf("page %d: wire next_cursor = %q, want %q (server must be authoritative)", pages, next, wantNext)
		}
		for _, d := range docs {
			if seen[d.ID] {
				t.Fatalf("id %d returned twice", d.ID)
			}
			seen[d.ID] = true
		}
		pages++
		if next == "" {
			break
		}
		cursor = next
		if pages > n {
			t.Fatal("did not terminate")
		}
	}
	if len(seen) != n {
		t.Fatalf("paged %d ids, want %d", len(seen), n)
	}
}

// TestFanoutDispatcherSearchDocsParity covers vector_search_docs: decode (At2),
// fan out, re-encode — and assert the decoded doc IDs match e.VectorSearchDocs
// in rank order (tie-free).
func TestFanoutDispatcherSearchDocsParity(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)
	ctx := context.Background()

	cfg := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: 4}
	if err := emb.CreateCollection(ctx, "p", cfg); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 200; i++ {
		if err := emb.VectorInsert(ctx, "p", uint64(i), tieFreeVec(i)); err != nil {
			t.Fatal(err)
		}
	}

	fan := rostam.NewFanoutDispatcher(emb, emb.Node())
	q := tieFreeQuery()

	want, _, err := emb.VectorSearchDocs(ctx, "p", q, 10, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := fan.Call("vector_search_docs", ops.EncodeVectorSearchArgsExt("p", 10, q, vector.Filter{}))
	if err != nil {
		t.Fatal(err)
	}
	got, err := ops.DecodeVectorDocs(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("docs len got %d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].ID != want[i].ID {
			t.Fatalf("docs rank %d: got id %d want %d", i, got[i].ID, want[i].ID)
		}
	}
}

// TestFanoutDispatcherDeleteByFilterParity covers vector_delete_by_filter:
// decode (At1), fan out the delete across partitions, and re-encode the 4-byte
// big-endian count. The decoded count must equal both the inserted matching
// count and what e.VectorDeleteByFilter reports on an equivalent collection.
func TestFanoutDispatcherDeleteByFilterParity(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)
	ctx := context.Background()

	// "p" deleted via the decorator; "o" is the single-partition oracle.
	cfgP := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: 4}
	cfgO := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: 1}
	if err := emb.CreateCollection(ctx, "p", cfgP); err != nil {
		t.Fatal(err)
	}
	if err := emb.CreateCollection(ctx, "o", cfgO); err != nil {
		t.Fatal(err)
	}

	const total = 120
	wantDeleted := 0
	for i := 0; i < total; i++ {
		tag := int64(i % 3) // tag==1 is the delete target
		if tag == 1 {
			wantDeleted++
		}
		md := rostam.VectorMetadata{"tag": vector.NewInt(tag)}
		if err := emb.VectorInsertExt(ctx, "p", uint64(i), tieFreeVec(i), rostam.VectorInsertOpts{Metadata: md}); err != nil {
			t.Fatal(err)
		}
		if err := emb.VectorInsertExt(ctx, "o", uint64(i), tieFreeVec(i), rostam.VectorInsertOpts{Metadata: md}); err != nil {
			t.Fatal(err)
		}
	}

	filter := vector.Filter{Op: vector.FilterEq, Field: "tag", Value: vector.NewInt(1)}

	fan := rostam.NewFanoutDispatcher(emb, emb.Node())
	raw, err := fan.Call("vector_delete_by_filter", ops.EncodeDeleteByFilterArgs("p", filter))
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != 4 {
		t.Fatalf("delete_by_filter response len got %d want 4", len(raw))
	}
	gotCount := int(binary.BigEndian.Uint32(raw))

	oracle, err := emb.VectorDeleteByFilter(ctx, "o", filter)
	if err != nil {
		t.Fatal(err)
	}
	if gotCount != wantDeleted {
		t.Fatalf("fan delete count got %d want %d", gotCount, wantDeleted)
	}
	if gotCount != oracle {
		t.Fatalf("fan delete count %d != single-partition oracle %d", gotCount, oracle)
	}
}

// TestFanoutDispatcherWriteAndCreateDrop covers the write/create/drop routing
// the decorator adds: a remote create must run through the embedded
// backend (so it gets the #/@ name guard + physical-partition creation + catalog
// write), inserts/upserts/deletes must land in the right physical partition, and
// a drop must remove every live-gen physical partition and neutralize the catalog.
func TestFanoutDispatcherWriteAndCreateDrop(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)
	ctx := context.Background()

	fan := rostam.NewFanoutDispatcher(emb, emb.Node())

	cfg := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: 4}

	// 1. Create a P=4 collection through the decorator: must produce a catalog
	// entry with P=4 (proving the create ran through the embedded backend).
	if _, err := fan.Call("vector_create_collection", ops.EncodeCreateCollectionArgs("p", cfg)); err != nil {
		t.Fatal(err)
	}
	if p, _, ok := emb.Catalog().PartitionsGen("p"); !ok || p != 4 {
		t.Fatalf("after create: PartitionsGen = (%d, ok=%v), want (4, true)", p, ok)
	}

	// 2. The #/@ name guard must reach the remote create path and reject a bad name.
	if _, err := fan.Call("vector_create_collection", ops.EncodeCreateCollectionArgs("bad#name", cfg)); err == nil {
		t.Fatal("create with reserved char in name: want error, got nil")
	}

	// 3. Insert via the decorator, then search via the decorator: the result must
	// contain the inserted id, proving the write landed in the right physical
	// partition and the fan-out search finds it.
	const wantID = 42
	insArgs := ops.EncodeVectorInsertArgs("p", wantID, tieFreeVec(wantID))
	if _, err := fan.Call("vector_insert", insArgs); err != nil {
		t.Fatal(err)
	}
	raw, err := fan.Call("vector_search", ops.EncodeVectorSearchArgsExt("p", 10, tieFreeQuery(), vector.Filter{}))
	if err != nil {
		t.Fatal(err)
	}
	res, err := ops.DecodeVectorSearchResults(raw)
	if err != nil {
		t.Fatal(err)
	}
	if !containsResultID(res, wantID) {
		t.Fatalf("search after insert: result %v does not contain id %d", resultIDs(res), wantID)
	}

	// 4. Delete via the decorator: must return a 1-byte body ([]byte{1} deleted).
	delRaw, err := fan.Call("vector_delete", ops.EncodeVectorDeleteArgs("p", wantID))
	if err != nil {
		t.Fatal(err)
	}
	if len(delRaw) != 1 || delRaw[0] != 1 {
		t.Fatalf("delete response = %v, want []byte{1}", delRaw)
	}

	// 5. Upsert carries content in metadata ($content); a doc search must surface it.
	const docID = 7
	const content = "hello fanout upsert"
	upArgs := ops.EncodeVectorUpsertArgs("p", docID, tieFreeVec(docID), content, 0, nil, vector.SparseVector{})
	if _, err := fan.Call("vector_upsert", upArgs); err != nil {
		t.Fatal(err)
	}
	docRaw, err := fan.Call("vector_search_docs", ops.EncodeVectorSearchArgsExt("p", 10, tieFreeQuery(), vector.Filter{}))
	if err != nil {
		t.Fatal(err)
	}
	docs, err := ops.DecodeVectorDocs(docRaw)
	if err != nil {
		t.Fatal(err)
	}
	if !containsDocContent(docs, docID, content) {
		t.Fatalf("search_docs after upsert: no doc id %d with content %q in %d docs", docID, content, len(docs))
	}

	// 6. Drop via the decorator: every live-gen physical partition must be gone and
	// the catalog must report the collection as no longer partitioned.
	if _, err := fan.Call("vector_drop_collection", ops.EncodeDropCollectionArgs("p")); err != nil {
		t.Fatal(err)
	}
	for p := 0; p < 4; p++ {
		phys := string(ops.PartitionKeyGen("p", 0, p))
		if _, err := emb.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(phys)); err == nil {
			t.Fatalf("after drop: partition %d (%s) still exists", p, phys)
		}
	}
	if _, _, ok := fan.Partitioned("p"); ok {
		t.Fatal("after drop: collection still reports as partitioned")
	}
}

// TestFanoutDispatcherWritePassthrough asserts that the write handlers'
// unpartitioned passthrough branch (fanInsert/fanDelete returning f.inner.Call's
// original bytes) is byte-identical to the non-decorated path — mirroring the
// read-op passthrough checks in the *Parity tests. We compare emb.Node().Call and
// fan.Call on a P=1 collection, using distinct ids per leg so each call is a
// valid op against a present (or, for the negative case, absent) id.
func TestFanoutDispatcherWritePassthrough(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)
	ctx := context.Background()

	fan := rostam.NewFanoutDispatcher(emb, emb.Node())

	// Unpartitioned (P=1) collection: every op must pass through byte-identical.
	cfg := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: 1}
	if err := emb.CreateCollection(ctx, "u", cfg); err != nil {
		t.Fatal(err)
	}

	// Seed a few vectors so deletes below actually hit a present id.
	for i := 0; i < 4; i++ {
		if err := emb.VectorInsert(ctx, "u", uint64(i), tieFreeVec(i)); err != nil {
			t.Fatal(err)
		}
	}

	// vector_insert passthrough: same wire shape, distinct ids per leg so both are
	// valid inserts. Insert returns a nil body, so the assertion is that both
	// succeed and the response bytes are equal (both nil).
	wantInsBytes, err := emb.Node().Call("vector_insert", ops.EncodeVectorInsertArgs("u", 100, tieFreeVec(100)))
	if err != nil {
		t.Fatal(err)
	}
	gotInsBytes, err := fan.Call("vector_insert", ops.EncodeVectorInsertArgs("u", 101, tieFreeVec(101)))
	if err != nil {
		t.Fatal(err)
	}
	if string(gotInsBytes) != string(wantInsBytes) {
		t.Fatalf("insert passthrough mismatch: got %d bytes want %d bytes", len(gotInsBytes), len(wantInsBytes))
	}

	// vector_delete passthrough on present ids: the handler returns a non-trivial
	// 1-byte body. Delete distinct present ids through each path so both report a
	// real deletion ([]byte{1}); assert byte equality.
	wantDelBytes, err := emb.Node().Call("vector_delete", ops.EncodeVectorDeleteArgs("u", 0))
	if err != nil {
		t.Fatal(err)
	}
	gotDelBytes, err := fan.Call("vector_delete", ops.EncodeVectorDeleteArgs("u", 1))
	if err != nil {
		t.Fatal(err)
	}
	if string(gotDelBytes) != string(wantDelBytes) {
		t.Fatalf("delete passthrough mismatch: got %v want %v", gotDelBytes, wantDelBytes)
	}
	if len(gotDelBytes) != 1 || gotDelBytes[0] != 1 {
		t.Fatalf("delete passthrough of present id = %v, want []byte{1}", gotDelBytes)
	}

	// vector_delete passthrough of a non-existent id: the handler's real result is
	// []byte{0}, proving passthrough surfaces the handler's actual return value.
	missRaw, err := fan.Call("vector_delete", ops.EncodeVectorDeleteArgs("u", 99999))
	if err != nil {
		t.Fatal(err)
	}
	if len(missRaw) != 1 || missRaw[0] != 0 {
		t.Fatalf("delete passthrough of absent id = %v, want []byte{0}", missRaw)
	}
}

// containsResultID reports whether id appears among the search results.
func containsResultID(res []rostam.VectorResult, id uint64) bool {
	for _, r := range res {
		if r.ID == id {
			return true
		}
	}
	return false
}

// resultIDs extracts the ids of a result slice (for failure messages).
func resultIDs(res []rostam.VectorResult) []uint64 {
	ids := make([]uint64, len(res))
	for i, r := range res {
		ids[i] = r.ID
	}
	return ids
}

// containsDocContent reports whether a doc with the given id and content is present.
func containsDocContent(docs []rostam.VectorDocument, id uint64, content string) bool {
	for _, d := range docs {
		if d.ID == id && d.Content == content {
			return true
		}
	}
	return false
}

// docIDSet collapses a doc slice into a presence set of IDs.
func docIDSet(docs []rostam.VectorDocument) map[uint64]bool {
	m := make(map[uint64]bool, len(docs))
	for _, d := range docs {
		m[d.ID] = true
	}
	return m
}

// TestFanoutDispatcherReservedNamePassthrough proves the explicit reserved-char
// guard in partitioned() wins over catalog state: a physical-style name (one
// containing '#'/'@') is always treated as not-partitioned, even if such a name
// somehow leaked into the catalog with P>1. This guarantees a forwarded internal
// op naming a physical partition never triggers a second fan-out. A normal name
// with a real catalog entry must still report partitioned (regression guard).
func TestFanoutDispatcherReservedNamePassthrough(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)

	fan := rostam.NewFanoutDispatcher(emb, emb.Node())

	// Simulate a leaked physical-style name written into the catalog with P>1.
	if err := emb.Catalog().SetPartitionsGen("evil#2", 4, 0); err != nil {
		t.Fatalf("SetPartitionsGen(evil#2): %v", err)
	}
	// Sanity: the catalog WOULD report it partitioned...
	if p, _, ok := emb.Catalog().PartitionsGen("evil#2"); !ok || p != 4 {
		t.Fatalf("catalog PartitionsGen(evil#2) = (%d, ok=%v), want (4, true)", p, ok)
	}
	// ...but the reserved-char guard makes partitioned() report not-partitioned.
	if p, _, ok := fan.Partitioned("evil#2"); ok {
		t.Fatalf("partitioned(evil#2) = (%d, ok=%v), want ok=false (guard wins over catalog)", p, ok)
	}

	// Regression guard: a normal name with a real catalog entry still partitions.
	if err := emb.Catalog().SetPartitionsGen("normal", 4, 0); err != nil {
		t.Fatalf("SetPartitionsGen(normal): %v", err)
	}
	if p, _, ok := fan.Partitioned("normal"); !ok || p != 4 {
		t.Fatalf("partitioned(normal) = (%d, ok=%v), want (4, true)", p, ok)
	}
}

// TestFanoutDispatcherResplit drives a dense resplit through the decorator's
// virtual-op path: create P=4, insert vectors, then fan.Call("vector_resplit")
// must run the coordinator op on the receiving node and flip the catalog to gen 1
// with P=8. A fan search must still find the inserted ids after the flip, and
// vector_resplit_cleanup must return a decodable dropped count.
func TestFanoutDispatcherResplit(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)
	ctx := context.Background()

	const coll = "rp"
	const n = 80
	cfg := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: 4}
	if err := emb.CreateCollection(ctx, coll, cfg); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		if err := emb.VectorInsert(ctx, coll, uint64(i), tieFreeVec(i)); err != nil {
			t.Fatal(err)
		}
	}

	fan := rostam.NewFanoutDispatcher(emb, emb.Node())

	// Resplit 4 -> 8 through the decorator's virtual-op path.
	if _, err := fan.Call("vector_resplit", ops.EncodeResplitArgs(coll, 8)); err != nil {
		t.Fatalf("fan resplit: %v", err)
	}
	// Catalog flipped to the new generation: P=8, gen=1.
	if p, gen, ok := emb.Catalog().PartitionsGen(coll); !ok || p != 8 || gen != 1 {
		t.Fatalf("after resplit: PartitionsGen = (%d, gen=%d, ok=%v), want (8, 1, true)", p, gen, ok)
	}

	// A fan search still finds the inserted ids after the flip (data migrated to
	// the new generation). Parity against an embedded baseline computed on the
	// same node proves the cross-partition merge over the new generation matches
	// the direct fan-out exactly (tie-free => deterministic rank order).
	want, _, err := emb.VectorSearchExt(ctx, coll, tieFreeQuery(), 10, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatalf("embedded baseline search after resplit: %v", err)
	}
	if len(want) != 10 {
		t.Fatalf("embedded baseline returned %d results, want 10", len(want))
	}
	raw, err := fan.Call("vector_search", ops.EncodeVectorSearchArgsExt(coll, 10, tieFreeQuery(), vector.Filter{}))
	if err != nil {
		t.Fatalf("fan search after resplit: %v", err)
	}
	got, err := ops.DecodeVectorSearchResults(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 10 {
		t.Fatalf("search after resplit returned %d results, want 10", len(got))
	}
	for i := range want {
		if got[i].ID != want[i].ID {
			t.Fatalf("search after resplit rank %d: got id %d, want %d (post-flip merge mismatch)\n got=%v\nwant=%v",
				i, got[i].ID, want[i].ID, resultIDs(got), resultIDs(want))
		}
	}

	// After a CLEAN resplit there are no orphans: VectorResplit itself drops the old
	// generation as its final step, so cleanup over the decorator must report 0.
	craw, err := fan.Call("vector_resplit_cleanup", ops.EncodeResplitCleanupArgs(coll))
	if err != nil {
		t.Fatalf("fan resplit cleanup: %v", err)
	}
	dropped, err := ops.DecodeResplitCleanupResult(craw)
	if err != nil {
		t.Fatalf("decode cleanup result: %v", err)
	}
	if dropped != 0 {
		t.Fatalf("cleanup after clean resplit dropped %d, want 0 (resplit already drops old gen)", dropped)
	}
}

// TestFanoutDispatcherMVResplit drives an MV resplit through the decorator's
// virtual-op path: create a P=4 MV collection, add docs, then
// fan.Call("vector_mv_resplit") flips the catalog to gen 1 / P=8 and an MV search
// still finds the added docs. vector_mv_resplit_cleanup returns a dropped count.
func TestFanoutDispatcherMVResplit(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)
	ctx := context.Background()

	const coll = "mvrp"
	const n = 40
	if err := emb.VectorMVCreateCollection(ctx, coll, rostam.MultiVectorConfig{Dim: 4, Partitions: 4}); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		if err := emb.VectorMVAdd(ctx, coll, uint64(i), [][]float32{mvTokenAt(i)}, nil); err != nil {
			t.Fatal(err)
		}
	}

	fan := rostam.NewFanoutDispatcher(emb, emb.Node())

	if _, err := fan.Call("vector_mv_resplit", ops.EncodeResplitArgs(coll, 8)); err != nil {
		t.Fatalf("fan mv resplit: %v", err)
	}
	if p, gen, ok := emb.Catalog().PartitionsGen(coll); !ok || p != 8 || gen != 1 {
		t.Fatalf("after mv resplit: PartitionsGen = (%d, gen=%d, ok=%v), want (8, 1, true)", p, gen, ok)
	}

	// MV search after the flip must still surface a known doc.
	const wantID = 17
	raw, err := fan.Call("vector_mv_search", ops.EncodeMVSearchArgs(coll, [][]float32{mvTokenAt(wantID)}, 5, 100))
	if err != nil {
		t.Fatalf("fan mv search after resplit: %v", err)
	}
	res, err := ops.DecodeMVResults(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || res[0].ID != wantID {
		t.Fatalf("mv search after resplit: results %v do not lead with id %d", res, wantID)
	}

	// After a CLEAN MV resplit there are no orphans: VectorMVResplit itself drops the
	// old generation as its final step, so cleanup over the decorator must report 0.
	craw, err := fan.Call("vector_mv_resplit_cleanup", ops.EncodeResplitCleanupArgs(coll))
	if err != nil {
		t.Fatalf("fan mv resplit cleanup: %v", err)
	}
	dropped, err := ops.DecodeResplitCleanupResult(craw)
	if err != nil {
		t.Fatalf("decode mv cleanup result: %v", err)
	}
	if dropped != 0 {
		t.Fatalf("cleanup after clean resplit dropped %d, want 0 (resplit already drops old gen)", dropped)
	}
}

// TestFanoutDispatcherMVSearchParity covers vector_mv_search: the decorator must
// peek the At1-layout name, fan out across partitions through the embedded MV
// coordinator, and re-encode via EncodeMVResults so the decoded results match
// e.VectorMVSearch in rank order (tie-free, so deterministic).
func TestFanoutDispatcherMVSearchParity(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)
	ctx := context.Background()

	const P = 4
	if err := emb.VectorMVCreateCollection(ctx, "mvp", rostam.MultiVectorConfig{Dim: 4, Partitions: P}); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 40; i++ {
		if err := emb.VectorMVAdd(ctx, "mvp", uint64(i), [][]float32{mvTokenAt(i)}, nil); err != nil {
			t.Fatal(err)
		}
	}

	fan := rostam.NewFanoutDispatcher(emb, emb.Node())
	const k, cand = 10, 100
	query := [][]float32{mvTokenAt(17)}

	want, _, err := emb.VectorMVSearch(ctx, "mvp", query, k, rostam.MultiSearchOpts{CandidatesPerToken: cand})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := fan.Call("vector_mv_search", ops.EncodeMVSearchArgs("mvp", query, k, cand))
	if err != nil {
		t.Fatal(err)
	}
	got, err := ops.DecodeMVResults(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("mv_search len got %d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].ID != want[i].ID {
			t.Fatalf("mv_search rank %d: got id %d want %d", i, got[i].ID, want[i].ID)
		}
	}

	// Unpartitioned (P=1) MV collection: vector_mv_search must pass through
	// byte-identical to the non-decorated path.
	if err := emb.VectorMVCreateCollection(ctx, "mvu", rostam.MultiVectorConfig{Dim: 4, Partitions: 1}); err != nil {
		t.Fatal(err)
	}
	if err := emb.VectorMVAdd(ctx, "mvu", 1, [][]float32{mvTokenAt(1)}, nil); err != nil {
		t.Fatal(err)
	}
	uArgs := ops.EncodeMVSearchArgs("mvu", query, k, cand)
	wantBytes, err := emb.Node().Call("vector_mv_search", uArgs)
	if err != nil {
		t.Fatal(err)
	}
	gotBytes, err := fan.Call("vector_mv_search", uArgs)
	if err != nil {
		t.Fatal(err)
	}
	if string(gotBytes) != string(wantBytes) {
		t.Fatalf("mv_search passthrough mismatch: got %d bytes want %d bytes", len(gotBytes), len(wantBytes))
	}
}

// TestFanoutDispatcherMVWriteAndDrop covers the MV write/create/drop routing the
// decorator adds: a remote create must run through the embedded backend (catalog
// P=4), an add+search must round-trip the doc, delete returns a 1-byte body, and
// drop must remove every physical MV partition and neutralize the catalog.
func TestFanoutDispatcherMVWriteAndDrop(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)
	ctx := context.Background()

	fan := rostam.NewFanoutDispatcher(emb, emb.Node())
	const P = 4

	// 1. Create a P=4 MV collection through the decorator: catalog must record P=4.
	if _, err := fan.Call("vector_mv_create_collection", ops.EncodeMVCreateArgs("mvp2", rostam.MultiVectorConfig{Dim: 4, Partitions: P})); err != nil {
		t.Fatal(err)
	}
	if p, _, ok := emb.Catalog().PartitionsGen("mvp2"); !ok || p != P {
		t.Fatalf("after mv create: PartitionsGen = (%d, ok=%v), want (4, true)", p, ok)
	}

	// 2. Add a doc via the decorator, then search via the decorator finds it.
	const wantID = 17
	if _, err := fan.Call("vector_mv_add", ops.EncodeMVAddArgs("mvp2", wantID, [][]float32{mvTokenAt(wantID)}, nil)); err != nil {
		t.Fatal(err)
	}
	raw, err := fan.Call("vector_mv_search", ops.EncodeMVSearchArgs("mvp2", [][]float32{mvTokenAt(wantID)}, 5, 100))
	if err != nil {
		t.Fatal(err)
	}
	res, err := ops.DecodeMVResults(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || res[0].ID != wantID {
		t.Fatalf("mv_search after add: results %v do not lead with id %d", res, wantID)
	}

	// 3. Delete via the decorator: must return a 1-byte body ([]byte{1} deleted).
	delRaw, err := fan.Call("vector_mv_delete", ops.EncodeMVDeleteArgs("mvp2", wantID))
	if err != nil {
		t.Fatal(err)
	}
	if len(delRaw) != 1 || delRaw[0] != 1 {
		t.Fatalf("mv_delete response = %v, want []byte{1}", delRaw)
	}

	// 4. Drop via the decorator: every physical MV partition gone, catalog neutralized.
	if _, err := fan.Call("vector_mv_drop_collection", ops.EncodeMVDeleteArgs("mvp2", 0)); err != nil {
		t.Fatal(err)
	}
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen("mvp2", 0, p))
		if _, err := emb.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(phys)); err == nil {
			t.Fatalf("after mv drop: partition %d (%s) still exists", p, phys)
		}
	}
	if _, _, ok := fan.Partitioned("mvp2"); ok {
		t.Fatal("after mv drop: collection still reports as partitioned")
	}
}

// TestFanoutDispatcherResplitRejectsWrappedNewP is the key regression for the
// actual vulnerability: a negative newP on the wire wraps int->uint32 to
// 4294967295 in EncodeResplitArgs. The server-side range guard
// (newP > maxResplitPartitions) must reject it, so fan.Call returns an error and
// the catalog is NOT mutated — the ~4.3-billion-iteration resplit loop never runs.
func TestFanoutDispatcherResplitRejectsWrappedNewP(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)
	ctx := context.Background()

	const (
		coll = "rpw"
		P    = 4
	)
	cfg := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: P}
	if err := emb.CreateCollection(ctx, coll, cfg); err != nil {
		t.Fatal(err)
	}
	wantP, wantGen, ok := emb.Catalog().PartitionsGen(coll)
	if !ok || wantP != P {
		t.Fatalf("setup PartitionsGen = (%d,%d,%v), want (%d,_,true)", wantP, wantGen, ok, P)
	}

	fan := rostam.NewFanoutDispatcher(emb, emb.Node())

	// Dense and MV ops both decode args via DecodeResplitArgs and funnel into the
	// guarded embedded methods. The wire value for newP=-1 is uint32(4294967295).
	for _, op := range []string{"vector_resplit", "vector_mv_resplit"} {
		if _, err := fan.Call(op, ops.EncodeResplitArgs(coll, -1)); err == nil {
			t.Errorf("%s with wrapped newP=-1: expected error, got nil", op)
		}
		if p, gen, ok := emb.Catalog().PartitionsGen(coll); !ok || p != wantP || gen != wantGen {
			t.Fatalf("%s with wrapped newP=-1 mutated catalog: (%d,%d,%v), want (%d,%d,true)",
				op, p, gen, ok, wantP, wantGen)
		}
	}
}

// TestFanoutDispatcherDualWriteDuringReshard proves the remote interception path
// (fan.Call insert/upsert/delete) dual-writes to both gens when the collection is
// Resharding — the dispatcher delegates to the embedded point-write methods, so
// the dual-target routing applies on the remote path too.
func TestFanoutDispatcherDualWriteDuringReshard(t *testing.T) {
	const coll = "fdw"
	const oldP, newP = 2, 4
	ee := setupReshardingDense(t, coll, oldP, newP)
	fan := rostam.NewFanoutDispatcher(ee, ee.Node())

	var id uint64 = 7
	oldPhys := string(ops.PartitionKeyGen(coll, 0, ops.PartitionOf(id, oldP)))
	newPhys := string(ops.PartitionKeyGen(coll, 1, ops.PartitionOf(id, newP)))

	if _, err := fan.Call("vector_insert", ops.EncodeVectorInsertArgs(coll, id, []float32{1, 0, 0, 0})); err != nil {
		t.Fatalf("fan insert: %v", err)
	}
	if !physVecExists(t, ee, oldPhys, id) || !physVecExists(t, ee, newPhys, id) {
		t.Fatalf("fan insert did not dual-write: old=%v new=%v",
			physVecExists(t, ee, oldPhys, id), physVecExists(t, ee, newPhys, id))
	}

	delRaw, err := fan.Call("vector_delete", ops.EncodeVectorDeleteArgs(coll, id))
	if err != nil {
		t.Fatalf("fan delete: %v", err)
	}
	if len(delRaw) != 1 || delRaw[0] != 1 {
		t.Fatalf("fan delete body = %v, want [1] (existed in live gen)", delRaw)
	}
	if physVecExists(t, ee, oldPhys, id) || physVecExists(t, ee, newPhys, id) {
		t.Fatalf("fan delete did not remove from both gens: old=%v new=%v",
			physVecExists(t, ee, oldPhys, id), physVecExists(t, ee, newPhys, id))
	}

	// vector_upsert is a distinct dispatcher path (fanUpsert→VectorUpsert) from
	// insert (fanInsert→VectorInsertExt); it must dual-write too.
	var uid uint64 = 12
	uOld := string(ops.PartitionKeyGen(coll, 0, ops.PartitionOf(uid, oldP)))
	uNew := string(ops.PartitionKeyGen(coll, 1, ops.PartitionOf(uid, newP)))
	if _, err := fan.Call("vector_upsert", ops.EncodeVectorUpsertArgs(coll, uid, []float32{2, 0, 0, 0}, "c", 0, nil, vector.SparseVector{})); err != nil {
		t.Fatalf("fan upsert: %v", err)
	}
	if !physVecExists(t, ee, uOld, uid) || !physVecExists(t, ee, uNew, uid) {
		t.Fatalf("fan upsert did not dual-write: old=%v new=%v",
			physVecExists(t, ee, uOld, uid), physVecExists(t, ee, uNew, uid))
	}
}

// TestFanoutDispatcherMVDualWriteDuringReshard proves the remote interception path
// for MV ops (fan.Call mv_add/mv_delete → fanMVAdd/fanMVDelete → VectorMVAdd/
// VectorMVDelete) dual-writes to both gens when the MV collection is Resharding.
func TestFanoutDispatcherMVDualWriteDuringReshard(t *testing.T) {
	const coll = "fdwmv"
	const oldP, newP = 2, 4
	ee := setupReshardingMV(t, coll, oldP, newP)
	fan := rostam.NewFanoutDispatcher(ee, ee.Node())

	var docID uint64 = 9
	oldPhys := string(ops.PartitionKeyGen(coll, 0, ops.PartitionOf(docID, oldP)))
	newPhys := string(ops.PartitionKeyGen(coll, 1, ops.PartitionOf(docID, newP)))
	tokens := [][]float32{{1, 0, 0, 0}}

	if _, err := fan.Call("vector_mv_add", ops.EncodeMVAddArgs(coll, docID, tokens, nil)); err != nil {
		t.Fatalf("fan mv add: %v", err)
	}
	if !physMVExists(t, ee, oldPhys, docID) || !physMVExists(t, ee, newPhys, docID) {
		t.Fatalf("fan mv add did not dual-write: old=%v new=%v",
			physMVExists(t, ee, oldPhys, docID), physMVExists(t, ee, newPhys, docID))
	}

	if _, err := fan.Call("vector_mv_delete", ops.EncodeMVDeleteArgs(coll, docID)); err != nil {
		t.Fatalf("fan mv delete: %v", err)
	}
	if physMVExists(t, ee, oldPhys, docID) || physMVExists(t, ee, newPhys, docID) {
		t.Fatalf("fan mv delete did not remove from both gens: old=%v new=%v",
			physMVExists(t, ee, oldPhys, docID), physMVExists(t, ee, newPhys, docID))
	}
}

// TestFanoutDispatcherPayloadDualWriteDuringReshard proves the remote interception
// path for a payload UPDATE (fan.Call vector_set_payload → fanSetPayload →
// VectorSetPayload) dual-writes to BOTH gens when the collection is Resharding.
// This is the "no stale new gen after cutover" correctness crux: a payload patched
// mid-reshard must be visible in the new gen, or the cutover loses the update.
func TestFanoutDispatcherPayloadDualWriteDuringReshard(t *testing.T) {
	const coll = "fdwpl"
	const oldP, newP = 2, 4
	ee := setupReshardingDense(t, coll, oldP, newP)
	fan := rostam.NewFanoutDispatcher(ee, ee.Node())

	var id uint64 = 7
	oldPhys := string(ops.PartitionKeyGen(coll, 0, ops.PartitionOf(id, oldP)))
	newPhys := string(ops.PartitionKeyGen(coll, 1, ops.PartitionOf(id, newP)))

	// Seed the point through the dispatcher so it dual-writes into both gens first
	// (mirrors how the insert dual-write test seeds before mutating).
	if _, err := fan.Call("vector_insert", ops.EncodeVectorInsertArgs(coll, id, []float32{1, 0, 0, 0})); err != nil {
		t.Fatalf("fan insert: %v", err)
	}
	if !physVecExists(t, ee, oldPhys, id) || !physVecExists(t, ee, newPhys, id) {
		t.Fatalf("seed insert did not dual-write: old=%v new=%v",
			physVecExists(t, ee, oldPhys, id), physVecExists(t, ee, newPhys, id))
	}

	// Patch the payload through the dispatcher during the reshard.
	if _, err := fan.Call("vector_set_payload", ops.EncodeSetPayloadArgs(coll, id, vector.Metadata{"tag": vector.NewInt(2)})); err != nil {
		t.Fatalf("fan set_payload: %v", err)
	}

	// Read each physical partition directly — the updated payload must be visible in
	// BOTH the old-gen owning partition AND the new-gen owning partition.
	oldTag, oldOK := physPayloadTag(t, ee, oldPhys, id)
	newTag, newOK := physPayloadTag(t, ee, newPhys, id)
	if !oldOK || oldTag != 2 {
		t.Fatalf("payload update missing/stale in OLD gen %q: tag=%d found=%v, want 2", oldPhys, oldTag, oldOK)
	}
	if !newOK || newTag != 2 {
		t.Fatalf("payload update missing/stale in NEW gen %q: tag=%d found=%v, want 2 (stale new gen after cutover!)", newPhys, newTag, newOK)
	}
}

// TestFanoutDispatcherPayloadStableNoDualWrite proves the dispatcher's Stable path
// applies a payload update to ONLY the owning physical partition (no second leg)
// for a partitioned-but-not-resharding collection.
func TestFanoutDispatcherPayloadStableNoDualWrite(t *testing.T) {
	const coll = "fplstable"
	const P = 4
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ee := s.(*rostam.Embedded)
	ctx := context.Background()
	cfg := rostam.VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: P}
	if err := ee.CreateCollection(ctx, coll, cfg); err != nil {
		t.Fatal(err)
	}
	fan := rostam.NewFanoutDispatcher(ee, ee.Node())

	var id uint64 = 7
	if _, err := fan.Call("vector_insert", ops.EncodeVectorInsertArgs(coll, id, []float32{1, 0, 0, 0})); err != nil {
		t.Fatalf("fan insert: %v", err)
	}
	if _, err := fan.Call("vector_set_payload", ops.EncodeSetPayloadArgs(coll, id, vector.Metadata{"tag": vector.NewInt(2)})); err != nil {
		t.Fatalf("fan set_payload: %v", err)
	}
	livePhys := string(ops.PartitionKeyGen(coll, 0, ops.PartitionOf(id, P)))
	if tag, ok := physPayloadTag(t, ee, livePhys, id); !ok || tag != 2 {
		t.Fatalf("stable payload update missing from live partition %q: tag=%d found=%v", livePhys, tag, ok)
	}
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen(coll, 0, p))
		if phys == livePhys {
			continue
		}
		if _, ok := physPayloadTag(t, ee, phys, id); ok {
			t.Fatalf("stable payload update leaked into %q (dual-write on Stable!)", phys)
		}
	}
}

// TestFanoutDispatcherStableNoDualWrite proves the dispatcher's Stable path is the
// existing single-target behavior (no second leg) for a partitioned-but-not-
// resharding collection.
func TestFanoutDispatcherStableNoDualWrite(t *testing.T) {
	const coll = "fstable"
	const P = 4
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ee := s.(*rostam.Embedded)
	ctx := context.Background()
	cfg := rostam.VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: P}
	if err := ee.CreateCollection(ctx, coll, cfg); err != nil {
		t.Fatal(err)
	}
	fan := rostam.NewFanoutDispatcher(ee, ee.Node())

	var id uint64 = 7
	if _, err := fan.Call("vector_insert", ops.EncodeVectorInsertArgs(coll, id, []float32{1, 0, 0, 0})); err != nil {
		t.Fatalf("fan insert: %v", err)
	}
	livePhys := string(ops.PartitionKeyGen(coll, 0, ops.PartitionOf(id, P)))
	if !physVecExists(t, ee, livePhys, id) {
		t.Fatalf("stable fan insert missing from live partition %q", livePhys)
	}
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen(coll, 0, p))
		if phys == livePhys {
			continue
		}
		if physVecExists(t, ee, phys, id) {
			t.Fatalf("stable fan insert leaked into %q (dual-write on Stable!)", phys)
		}
	}
}

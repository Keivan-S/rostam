// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/vector"
)

// TestEmbeddedGetPayloadDenseSingleNode drives the dense get + 4 payload mutations
// over the embedded backend on a single node: create -> insert -> get (vec+payload
// +ttl) -> set/overwrite/delete-keys/clear (get reflects each) -> delete -> get
// not-found. A dense payload-update then a filtered search reflects the new field
// (reindex via the engine).
func TestEmbeddedGetPayloadDenseSingleNode(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	must(t, s.CreateCollection(ctx, "docs", VectorConfig{
		Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2,
	}))
	must(t, s.VectorInsertExt(ctx, "docs", 1, []float32{1, 0, 0, 0}, VectorInsertOpts{
		TTL: time.Hour, Metadata: vector.Metadata{"a": vector.NewInt(1)},
	}))

	found, vec, meta, ttl, _, err := s.VectorGet(ctx, "docs", 1, true, true)
	must(t, err)
	if !found || len(vec) != 4 || vec[0] != 1 || meta["a"].Int != 1 || ttl <= 0 {
		t.Fatalf("get: found=%v vec=%v meta=%+v ttl=%v", found, vec, meta, ttl)
	}

	// set (merge) adds b=2.
	applied, err := s.VectorSetPayload(ctx, "docs", 1, vector.Metadata{"b": vector.NewInt(2)}, nil)
	must(t, err)
	if !applied {
		t.Fatal("set payload: applied=false")
	}
	_, _, meta, _, _, _ = s.VectorGet(ctx, "docs", 1, true, true)
	if meta["a"].Int != 1 || meta["b"].Int != 2 {
		t.Fatalf("after set: %+v, want a=1,b=2", meta)
	}

	// filtered search reflects the new b=2 field (reindex through the embedded op path).
	rs, _, err := s.VectorSearchExt(ctx, "docs", []float32{1, 0, 0, 0}, 5, VectorSearchOpts{
		Filter: vector.Filter{Op: vector.FilterEq, Field: "b", Value: vector.NewInt(2)},
	})
	must(t, err)
	if len(rs) != 1 || rs[0].ID != 1 {
		t.Fatalf("filter b==2 = %+v, want point 1 (reindex)", rs)
	}

	// overwrite -> only c=3.
	if _, err := s.VectorOverwritePayload(ctx, "docs", 1, vector.Metadata{"c": vector.NewInt(3)}, nil); err != nil {
		t.Fatal(err)
	}
	_, _, meta, _, _, _ = s.VectorGet(ctx, "docs", 1, true, true)
	if _, ok := meta["a"]; ok || meta["c"].Int != 3 {
		t.Fatalf("after overwrite: %+v, want only c=3", meta)
	}

	// delete-keys removes c.
	if _, err := s.VectorDeletePayloadKeys(ctx, "docs", 1, []string{"c"}); err != nil {
		t.Fatal(err)
	}
	_, _, meta, _, _, _ = s.VectorGet(ctx, "docs", 1, true, true)
	if _, ok := meta["c"]; ok {
		t.Fatalf("after delete-keys: %+v, want no c", meta)
	}

	// clear -> empty.
	if _, err := s.VectorClearPayload(ctx, "docs", 1); err != nil {
		t.Fatal(err)
	}
	_, _, meta, _, _, _ = s.VectorGet(ctx, "docs", 1, true, true)
	if len(meta) != 0 {
		t.Fatalf("after clear: %+v, want empty", meta)
	}

	// delete -> get not-found.
	existed, err := s.VectorDelete(ctx, "docs", 1)
	must(t, err)
	if !existed {
		t.Fatal("delete: existed=false")
	}
	found, _, _, _, _, err = s.VectorGet(ctx, "docs", 1, true, true)
	must(t, err)
	if found {
		t.Fatal("get after delete: found=true, want false")
	}

	// payload mutation of an absent point -> applied=false, no error.
	applied, err = s.VectorSetPayload(ctx, "docs", 1, vector.Metadata{"x": vector.NewInt(1)}, nil)
	must(t, err)
	if applied {
		t.Fatal("set absent: applied=true, want false")
	}
}

// TestEmbeddedGetPayloadNamedSingleNode drives named get + payload mutations over
// the embedded backend.
func TestEmbeddedGetPayloadNamedSingleNode(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	cfg := map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: vector.Cosine},
		"image": {Dim: 4, Metric: vector.Cosine},
	}
	must(t, s.VectorNamedCreateCollection(ctx, "named", cfg, 0))
	must(t, s.VectorNamedInsert(ctx, "named", 1,
		map[string][]float32{"title": {1, 0, 0, 0}, "image": {0, 0, 1, 0}},
		vector.Metadata{"lang": vector.NewString("en")}, time.Hour))

	found, vecs, payload, ttl, err := s.VectorNamedGet(ctx, "named", 1, true, true)
	must(t, err)
	if !found || len(vecs["title"]) != 4 || payload["lang"].Str != "en" || ttl <= 0 {
		t.Fatalf("named get: found=%v vecs=%+v payload=%+v ttl=%v", found, vecs, payload, ttl)
	}

	applied, err := s.VectorNamedSetPayload(ctx, "named", 1, vector.Metadata{"n": vector.NewInt(7)}, nil)
	must(t, err)
	if !applied {
		t.Fatal("named set: applied=false")
	}
	_, _, payload, _, _ = s.VectorNamedGet(ctx, "named", 1, false, true)
	if payload["lang"].Str != "en" || payload["n"].Int != 7 {
		t.Fatalf("named after set: %+v, want lang=en,n=7", payload)
	}

	// filtered search reflects the new field (predicate-eval over the shared payload).
	res, err := s.VectorNamedSearch(ctx, "named", "title", []float32{1, 0, 0, 0}, 5,
		vector.Filter{Op: vector.FilterEq, Field: "n", Value: vector.NewInt(7)})
	must(t, err)
	if len(res) != 1 || res[0].ID != 1 {
		t.Fatalf("named filter n==7 = %+v, want point 1", res)
	}

	if _, err := s.VectorNamedOverwritePayload(ctx, "named", 1, vector.Metadata{"only": vector.NewInt(1)}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := s.VectorNamedDeletePayloadKeys(ctx, "named", 1, []string{"only"}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.VectorNamedClearPayload(ctx, "named", 1); err != nil {
		t.Fatal(err)
	}
	_, _, payload, _, _ = s.VectorNamedGet(ctx, "named", 1, false, true)
	if len(payload) != 0 {
		t.Fatalf("named after clear: %+v, want empty", payload)
	}

	existed, err := s.VectorNamedDelete(ctx, "named", 1)
	must(t, err)
	if !existed {
		t.Fatal("named delete: existed=false")
	}
	found, _, _, _, err = s.VectorNamedGet(ctx, "named", 1, true, true)
	must(t, err)
	if found {
		t.Fatal("named get after delete: found=true, want false")
	}
}

// TestEmbeddedGetPayloadMVSingleNode drives MV get + payload mutations over the
// embedded backend.
func TestEmbeddedGetPayloadMVSingleNode(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	must(t, s.VectorMVCreateCollection(ctx, "mv", MultiVectorConfig{
		Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1,
	}))
	tokens := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}
	must(t, s.VectorMVAdd(ctx, "mv", 1, tokens, vector.Metadata{"doc": vector.NewInt(3)}))

	found, gt, payload, err := s.VectorMVGet(ctx, "mv", 1, true, true)
	must(t, err)
	if !found || len(gt) != 2 || payload["doc"].Int != 3 {
		t.Fatalf("mv get: found=%v tokens=%+v payload=%+v", found, gt, payload)
	}

	applied, err := s.VectorMVSetPayload(ctx, "mv", 1, vector.Metadata{"x": vector.NewInt(9)}, nil)
	must(t, err)
	if !applied {
		t.Fatal("mv set: applied=false")
	}
	_, _, payload, _ = s.VectorMVGet(ctx, "mv", 1, false, true)
	if payload["doc"].Int != 3 || payload["x"].Int != 9 {
		t.Fatalf("mv after set: %+v, want doc=3,x=9", payload)
	}

	if _, err := s.VectorMVOverwritePayload(ctx, "mv", 1, vector.Metadata{"only": vector.NewInt(1)}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := s.VectorMVDeletePayloadKeys(ctx, "mv", 1, []string{"only"}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.VectorMVClearPayload(ctx, "mv", 1); err != nil {
		t.Fatal(err)
	}
	_, _, payload, _ = s.VectorMVGet(ctx, "mv", 1, false, true)
	if len(payload) != 0 {
		t.Fatalf("mv after clear: %+v, want empty", payload)
	}

	existed, err := s.VectorMVDelete(ctx, "mv", 1)
	must(t, err)
	if !existed {
		t.Fatal("mv delete: existed=false")
	}
	found, _, _, err = s.VectorMVGet(ctx, "mv", 1, true, true)
	must(t, err)
	if found {
		t.Fatal("mv get after delete: found=true, want false")
	}

	// absent payload mutation -> applied=false, no error.
	applied, err = s.VectorMVSetPayload(ctx, "mv", 1, vector.Metadata{"x": vector.NewInt(1)}, nil)
	must(t, err)
	if applied {
		t.Fatal("mv set absent: applied=true, want false")
	}
}

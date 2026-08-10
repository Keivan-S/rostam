// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"reflect"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/vector"
)

// newGetPayloadTx builds a TxContext backed by a fresh on-disk CollectionStore. Mirrors
// TestVectorOpsViaDispatch's setup; the store is closed via t.Cleanup.
func newGetPayloadTx(t *testing.T) (*TxContext, *vector.CollectionStore) {
	t.Helper()
	dir := t.TempDir()
	c, _ := cache.New(cache.DefaultConfig())
	t.Cleanup(func() { c.Close() })
	vstore, err := vector.OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { vstore.Close() })
	return NewTxContextWithVectors(c, vstore), vstore
}

// --- codec round-trips: get args (shared by all families) ---

func TestVectorGetArgsRoundtrip(t *testing.T) {
	for _, flags := range []uint8{0, getFlagWithVector, getFlagWithPayload, GetFlagsBoth} {
		col, id, gotFlags, err := DecodeVectorGetArgs(EncodeVectorGetArgs("acme/docs", 42, flags))
		if err != nil {
			t.Fatalf("flags=%d: %v", flags, err)
		}
		if col != "acme/docs" || id != 42 || gotFlags != flags {
			t.Errorf("got (%q,%d,%d), want (acme/docs,42,%d)", col, id, gotFlags, flags)
		}
	}
	// truncation: cut the trailing flags byte off a valid encoding.
	full := EncodeVectorGetArgs("docs", 1, GetFlagsBoth)
	if _, _, _, err := DecodeVectorGetArgs(full[:len(full)-1]); err == nil {
		t.Error("truncated get args: want error, got nil")
	}
}

// --- codec round-trips: dense get result ---

func TestVectorGetResultRoundtrip(t *testing.T) {
	meta := vector.Metadata{"a": vector.NewInt(1), "tag": vector.NewString("x")}
	sparse := &vector.SparseVector{Indices: []uint32{0, 5}, Values: []float32{1, 2}}
	vec := []float32{1, 2, 3, 4}

	// found, both projections on.
	found, gv, gm, gttl, gs, err := DecodeVectorGetResult(
		EncodeVectorGetResult(true, vec, meta, 5*time.Second, sparse, true, true))
	if err != nil || !found {
		t.Fatalf("found=%v err=%v", found, err)
	}
	if !reflect.DeepEqual(gv, vec) {
		t.Errorf("vec = %v, want %v", gv, vec)
	}
	if gm["a"].Int != 1 || gm["tag"].Str != "x" {
		t.Errorf("meta = %+v", gm)
	}
	if gttl != 5*time.Second {
		t.Errorf("ttl = %v, want 5s", gttl)
	}
	if gs == nil || len(gs.Indices) != 2 {
		t.Errorf("sparse = %+v", gs)
	}

	// with_vector off: vec omitted, payload kept.
	found, gv, gm, _, _, err = DecodeVectorGetResult(
		EncodeVectorGetResult(true, vec, meta, 0, sparse, false, true))
	if err != nil || !found {
		t.Fatal(err)
	}
	if gv != nil {
		t.Errorf("with_vector off: vec = %v, want nil", gv)
	}
	if gm == nil {
		t.Error("with_vector off: payload should still be present")
	}

	// with_payload off: payload + sparse omitted, vec kept.
	found, gv, gm, _, gs, err = DecodeVectorGetResult(
		EncodeVectorGetResult(true, vec, meta, 0, sparse, true, false))
	if err != nil || !found {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gv, vec) {
		t.Errorf("with_payload off: vec = %v", gv)
	}
	if gm != nil || gs != nil {
		t.Errorf("with_payload off: meta=%v sparse=%v, want both nil", gm, gs)
	}

	// not-found flag.
	found, _, _, _, _, err = DecodeVectorGetResult(EncodeVectorGetResult(false, nil, nil, 0, nil, true, true))
	if err != nil {
		t.Fatal(err)
	}
	if found {
		t.Error("not-found result: found = true, want false")
	}

	// truncation.
	body := EncodeVectorGetResult(true, vec, meta, time.Second, sparse, true, true)
	if _, _, _, _, _, err := DecodeVectorGetResult(body[:5]); err == nil {
		t.Error("truncated get result: want error")
	}
	if _, _, _, _, _, err := DecodeVectorGetResult(nil); err == nil {
		t.Error("empty get result: want error")
	}
}

// --- codec round-trips: payload args (shared) ---

func TestSetPayloadArgsRoundtrip(t *testing.T) {
	meta := vector.Metadata{"k": vector.NewString("v")}
	col, id, gm, err := DecodeSetPayloadArgs(EncodeSetPayloadArgs("docs", 7, meta))
	if err != nil {
		t.Fatal(err)
	}
	if col != "docs" || id != 7 || gm["k"].Str != "v" {
		t.Errorf("got (%q,%d,%+v)", col, id, gm)
	}
	// empty payload decodes to nil meta (valid overwrite-to-empty).
	_, _, gm, err = DecodeSetPayloadArgs(EncodeSetPayloadArgs("docs", 7, nil))
	if err != nil || gm != nil {
		t.Errorf("empty payload: meta=%v err=%v", gm, err)
	}
	// truncation.
	full := EncodeSetPayloadArgs("docs", 7, meta)
	if _, _, _, err := DecodeSetPayloadArgs(full[:len(full)-2]); err == nil {
		t.Error("truncated set-payload args: want error")
	}
}

func TestDeletePayloadKeysArgsRoundtrip(t *testing.T) {
	keys := []string{"a", "bb", "ccc"}
	col, id, gk, err := DecodeDeletePayloadKeysArgs(EncodeDeletePayloadKeysArgs("docs", 3, keys))
	if err != nil {
		t.Fatal(err)
	}
	if col != "docs" || id != 3 || !reflect.DeepEqual(gk, keys) {
		t.Errorf("got (%q,%d,%v)", col, id, gk)
	}
	// empty key list.
	_, _, gk, err = DecodeDeletePayloadKeysArgs(EncodeDeletePayloadKeysArgs("docs", 3, nil))
	if err != nil || len(gk) != 0 {
		t.Errorf("empty keys: %v err=%v", gk, err)
	}
	// truncation.
	full := EncodeDeletePayloadKeysArgs("docs", 3, keys)
	if _, _, _, err := DecodeDeletePayloadKeysArgs(full[:len(full)-2]); err == nil {
		t.Error("truncated delete-keys args: want error")
	}
}

func TestClearPayloadArgsRoundtrip(t *testing.T) {
	col, id, err := DecodeClearPayloadArgs(EncodeClearPayloadArgs("docs", 9))
	if err != nil || col != "docs" || id != 9 {
		t.Errorf("got (%q,%d) err=%v", col, id, err)
	}
}

func TestPayloadResultRoundtrip(t *testing.T) {
	for _, applied := range []bool{true, false} {
		got, err := DecodePayloadResult(EncodePayloadResult(applied))
		if err != nil || got != applied {
			t.Errorf("applied=%v: got %v err=%v", applied, got, err)
		}
	}
	if _, err := DecodePayloadResult(nil); err == nil {
		t.Error("empty payload result: want error")
	}
}

// --- named get result codec ---

func TestNamedGetResultRoundtrip(t *testing.T) {
	vecs := map[string][]float32{"title": {1, 0}, "image": {0, 1}}
	payload := vector.Metadata{"lang": vector.NewString("en")}
	found, gv, gp, gttl, err := DecodeNamedGetResult(
		EncodeNamedGetResult(true, vecs, payload, 3*time.Second, true, true))
	if err != nil || !found {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gv, vecs) {
		t.Errorf("vectors = %+v, want %+v", gv, vecs)
	}
	if gp["lang"].Str != "en" || gttl != 3*time.Second {
		t.Errorf("payload=%+v ttl=%v", gp, gttl)
	}
	// with_vector off -> empty map.
	_, gv, _, _, _ = DecodeNamedGetResult(EncodeNamedGetResult(true, vecs, payload, 0, false, true))
	if len(gv) != 0 {
		t.Errorf("with_vector off: vectors = %+v, want empty", gv)
	}
	// not-found.
	found, _, _, _, err = DecodeNamedGetResult(EncodeNamedGetResult(false, nil, nil, 0, true, true))
	if err != nil || found {
		t.Errorf("not-found: found=%v err=%v", found, err)
	}
	// truncation.
	body := EncodeNamedGetResult(true, vecs, payload, time.Second, true, true)
	if _, _, _, _, err := DecodeNamedGetResult(body[:3]); err == nil {
		t.Error("truncated named get result: want error")
	}
}

// --- MV get result codec ---

func TestMVGetResultRoundtrip(t *testing.T) {
	tokens := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}
	payload := vector.Metadata{"doc": vector.NewInt(5)}
	found, gt, gp, err := DecodeMVGetResult(EncodeMVGetResult(true, tokens, payload, true, true))
	if err != nil || !found {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gt, tokens) {
		t.Errorf("tokens = %+v, want %+v", gt, tokens)
	}
	if gp["doc"].Int != 5 {
		t.Errorf("payload = %+v", gp)
	}
	// with_vector off -> no tokens.
	_, gt, _, _ = DecodeMVGetResult(EncodeMVGetResult(true, tokens, payload, false, true))
	if len(gt) != 0 {
		t.Errorf("with_vector off: tokens = %+v, want empty", gt)
	}
	// with_payload off -> no payload.
	_, gt, gp, _ = DecodeMVGetResult(EncodeMVGetResult(true, tokens, payload, true, false))
	if gp != nil || len(gt) != 2 {
		t.Errorf("with_payload off: tokens=%v payload=%v", gt, gp)
	}
	// not-found.
	found, _, _, err = DecodeMVGetResult(EncodeMVGetResult(false, nil, nil, true, true))
	if err != nil || found {
		t.Errorf("not-found: found=%v err=%v", found, err)
	}
	// truncation.
	body := EncodeMVGetResult(true, tokens, payload, true, true)
	if _, _, _, err := DecodeMVGetResult(body[:4]); err == nil {
		t.Error("truncated mv get result: want error")
	}
}

// --- per-row decoder splits (decodeNamedGetResultAt / decodeMVGetResultAt) ---

// decodeNamedGetResultAt at off=0 must produce the SAME decoded value as the
// public DecodeNamedGetResult for a round-tripped record (faithful split), and
// the returned next must point exactly past the record so two records encoded
// back-to-back decode by advancing off.
func TestDecodeNamedGetResultAtFaithful(t *testing.T) {
	vecs := map[string][]float32{"title": {1, 0}, "image": {0, 1}}
	payload := vector.Metadata{"lang": vector.NewString("en")}
	body := EncodeNamedGetResult(true, vecs, payload, 3*time.Second, true, true)

	// off=0 helper matches the public single-get decoder exactly.
	found, gv, gp, ttlMs, _, next, err := decodeNamedGetResultAt(body, 0)
	if err != nil || !found {
		t.Fatalf("decodeNamedGetResultAt: found=%v err=%v", found, err)
	}
	pFound, pv, pp, pttl, perr := DecodeNamedGetResult(body)
	if perr != nil || !pFound {
		t.Fatalf("DecodeNamedGetResult: found=%v err=%v", pFound, perr)
	}
	if !reflect.DeepEqual(gv, pv) || !reflect.DeepEqual(gp, pp) {
		t.Errorf("split mismatch: helper vecs=%+v meta=%+v vs public vecs=%+v meta=%+v", gv, gp, pv, pp)
	}
	if time.Duration(ttlMs)*time.Millisecond != pttl {
		t.Errorf("ttl mismatch: helper=%dms public=%v", ttlMs, pttl)
	}
	if next != len(body) {
		t.Errorf("next=%d, want end-of-record %d", next, len(body))
	}

	// Two records back-to-back: decode both by advancing off via next.
	vecs2 := map[string][]float32{"body": {2, 2}}
	body2 := EncodeNamedGetResult(true, vecs2, nil, time.Second, true, false)
	two := append(append([]byte{}, body...), body2...)
	f1, v1, _, _, _, n1, e1 := decodeNamedGetResultAt(two, 0)
	if e1 != nil || !f1 || !reflect.DeepEqual(v1, vecs) {
		t.Fatalf("record 1: found=%v err=%v vecs=%+v", f1, e1, v1)
	}
	f2, v2, _, _, _, n2, e2 := decodeNamedGetResultAt(two, n1)
	if e2 != nil || !f2 || !reflect.DeepEqual(v2, vecs2) {
		t.Fatalf("record 2: found=%v err=%v vecs=%+v", f2, e2, v2)
	}
	if n2 != len(two) {
		t.Errorf("next after record 2 = %d, want %d", n2, len(two))
	}
}

// decodeMVGetResultAt at off=0 must match DecodeMVGetResult and next must point
// past the record (MV has no ttl).
func TestDecodeMVGetResultAtFaithful(t *testing.T) {
	tokens := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}
	payload := vector.Metadata{"doc": vector.NewInt(5)}
	body := EncodeMVGetResult(true, tokens, payload, true, true)

	found, gt, gp, _, next, err := decodeMVGetResultAt(body, 0)
	if err != nil || !found {
		t.Fatalf("decodeMVGetResultAt: found=%v err=%v", found, err)
	}
	pFound, pt, pp, perr := DecodeMVGetResult(body)
	if perr != nil || !pFound {
		t.Fatalf("DecodeMVGetResult: found=%v err=%v", pFound, perr)
	}
	if !reflect.DeepEqual(gt, pt) || !reflect.DeepEqual(gp, pp) {
		t.Errorf("split mismatch: helper tokens=%+v meta=%+v vs public tokens=%+v meta=%+v", gt, gp, pt, pp)
	}
	if next != len(body) {
		t.Errorf("next=%d, want end-of-record %d", next, len(body))
	}

	// Two records back-to-back: decode both via next.
	tokens2 := [][]float32{{9, 9, 9, 9}}
	body2 := EncodeMVGetResult(true, tokens2, nil, true, false)
	two := append(append([]byte{}, body...), body2...)
	f1, t1, _, _, n1, e1 := decodeMVGetResultAt(two, 0)
	if e1 != nil || !f1 || !reflect.DeepEqual(t1, tokens) {
		t.Fatalf("record 1: found=%v err=%v tokens=%+v", f1, e1, t1)
	}
	f2, t2, _, _, n2, e2 := decodeMVGetResultAt(two, n1)
	if e2 != nil || !f2 || !reflect.DeepEqual(t2, tokens2) {
		t.Fatalf("record 2: found=%v err=%v tokens=%+v", f2, e2, t2)
	}
	if n2 != len(two) {
		t.Errorf("next after record 2 = %d, want %d", n2, len(two))
	}
}

// --- dispatch-through-handler: dense ---

func TestDenseGetPayloadViaDispatch(t *testing.T) {
	tx, _ := newGetPayloadTx(t)
	cfg := vector.Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("docs", cfg)); err != nil {
		t.Fatal(err)
	}
	// insert with payload + ttl.
	ins := EncodeVectorInsertArgsExt("docs", 1, []float32{1, 0, 0, 0}, time.Hour,
		vector.Metadata{"a": vector.NewInt(1)}, vector.SparseVector{})
	if _, err := handleVectorInsert(tx, ins); err != nil {
		t.Fatal(err)
	}

	// get reflects vec + payload + ttl.
	body, err := handleVectorGet(tx, EncodeVectorGetArgs("docs", 1, GetFlagsBoth))
	if err != nil {
		t.Fatal(err)
	}
	found, vec, meta, ttl, _, err := DecodeVectorGetResult(body)
	if err != nil || !found {
		t.Fatalf("get: found=%v err=%v", found, err)
	}
	if len(vec) != 4 || vec[0] != 1 || meta["a"].Int != 1 || ttl <= 0 {
		t.Errorf("get got vec=%v meta=%+v ttl=%v", vec, meta, ttl)
	}

	// set payload (merge): add b=2.
	res, err := handleVectorSetPayload(tx, EncodeSetPayloadArgs("docs", 1, vector.Metadata{"b": vector.NewInt(2)}))
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := DecodePayloadResult(res); !ok {
		t.Fatal("set payload: applied = false, want true")
	}
	body, _ = handleVectorGet(tx, EncodeVectorGetArgs("docs", 1, GetFlagsBoth))
	_, _, meta, _, _, _ = DecodeVectorGetResult(body)
	if meta["a"].Int != 1 || meta["b"].Int != 2 {
		t.Errorf("after set: meta = %+v, want a=1,b=2", meta)
	}

	// filtered search reflects the new b=2 field (proves dense reindex through the op).
	fb, err := handleVectorSearch(tx, EncodeVectorSearchArgsExt("docs", 5, []float32{1, 0, 0, 0},
		vector.Filter{Op: vector.FilterEq, Field: "b", Value: vector.NewInt(2)}))
	if err != nil {
		t.Fatal(err)
	}
	rs, _ := DecodeVectorSearchResults(fb)
	if len(rs) != 1 || rs[0].ID != 1 {
		t.Errorf("filter b==2 = %+v, want point 1 (reindex via op)", rs)
	}

	// overwrite payload: replace with c=3.
	handleVectorOverwritePayload(tx, EncodeSetPayloadArgs("docs", 1, vector.Metadata{"c": vector.NewInt(3)}))
	body, _ = handleVectorGet(tx, EncodeVectorGetArgs("docs", 1, GetFlagsBoth))
	_, _, meta, _, _, _ = DecodeVectorGetResult(body)
	if _, ok := meta["a"]; ok || meta["c"].Int != 3 {
		t.Errorf("after overwrite: meta = %+v, want only c=3", meta)
	}

	// delete-keys: remove c.
	handleVectorDeletePayloadKeys(tx, EncodeDeletePayloadKeysArgs("docs", 1, []string{"c"}))
	body, _ = handleVectorGet(tx, EncodeVectorGetArgs("docs", 1, GetFlagsBoth))
	_, _, meta, _, _, _ = DecodeVectorGetResult(body)
	if _, ok := meta["c"]; ok {
		t.Errorf("after delete-keys: meta = %+v, want no c", meta)
	}

	// clear: empty payload.
	handleVectorClearPayload(tx, EncodeClearPayloadArgs("docs", 1))
	body, _ = handleVectorGet(tx, EncodeVectorGetArgs("docs", 1, GetFlagsBoth))
	_, _, meta, _, _, _ = DecodeVectorGetResult(body)
	if len(meta) != 0 {
		t.Errorf("after clear: meta = %+v, want empty", meta)
	}

	// get absent -> not-found flag (NOT an op error).
	body, err = handleVectorGet(tx, EncodeVectorGetArgs("docs", 999, GetFlagsBoth))
	if err != nil {
		t.Fatalf("get absent: unexpected op error %v", err)
	}
	if found, _, _, _, _, _ := DecodeVectorGetResult(body); found {
		t.Error("get absent: found = true, want false")
	}

	// payload mutation of absent -> applied=0 flag (NOT an op error).
	res, err = handleVectorSetPayload(tx, EncodeSetPayloadArgs("docs", 999, vector.Metadata{"x": vector.NewInt(1)}))
	if err != nil {
		t.Fatalf("set absent: unexpected op error %v", err)
	}
	if ok, _ := DecodePayloadResult(res); ok {
		t.Error("set absent: applied = true, want false")
	}
}

func TestDenseSetPayloadBadJSONFailsLoud(t *testing.T) {
	tx, _ := newGetPayloadTx(t)
	cfg := vector.Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: vector.L2}
	handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("docs", cfg))
	handleVectorInsert(tx, EncodeVectorInsertArgs("docs", 1, []float32{1, 0}))

	// Hand-craft set-payload args with a non-zero metaLen but invalid JSON bytes.
	bad := EncodeSetPayloadArgs("docs", 1, nil) // [colLen][col][id][metaLen=0]
	// append a fake metaLen + garbage by re-encoding manually.
	bad = bad[:len(bad)-4]        // drop the metaLen=0 word
	bad = append(bad, 0, 0, 0, 3) // metaLen=3
	bad = append(bad, '{', '{', '{')
	if _, err := handleVectorSetPayload(tx, bad); err == nil {
		t.Error("set payload with bad JSON: want op error, got nil")
	}
}

// --- dispatch-through-handler: named ---

func TestNamedGetPayloadViaDispatch(t *testing.T) {
	tx, _ := newGetPayloadTx(t)
	cfg := map[string]vector.NamedVectorParams{
		"title": {Dim: 2, Metric: vector.Cosine},
		"image": {Dim: 2, Metric: vector.Cosine},
	}
	if _, err := handleNamedCreate(tx, EncodeNamedCreateArgs("named", cfg, 0)); err != nil {
		t.Fatal(err)
	}
	ins := EncodeNamedInsertArgs("named", 1, map[string][]float32{"title": {1, 0}},
		vector.Metadata{"lang": vector.NewString("en")}, time.Hour)
	if _, err := handleNamedInsert(tx, ins); err != nil {
		t.Fatal(err)
	}

	body, err := handleNamedGet(tx, EncodeVectorGetArgs("named", 1, GetFlagsBoth))
	if err != nil {
		t.Fatal(err)
	}
	found, vecs, payload, ttl, err := DecodeNamedGetResult(body)
	if err != nil || !found {
		t.Fatalf("named get: found=%v err=%v", found, err)
	}
	if len(vecs["title"]) != 2 || payload["lang"].Str != "en" || ttl <= 0 {
		t.Errorf("named get: vecs=%+v payload=%+v ttl=%v", vecs, payload, ttl)
	}

	// set/overwrite/delete-keys/clear then get reflects each.
	handleNamedSetPayload(tx, EncodeSetPayloadArgs("named", 1, vector.Metadata{"n": vector.NewInt(7)}))
	body, _ = handleNamedGet(tx, EncodeVectorGetArgs("named", 1, GetFlagsBoth))
	_, _, payload, _, _ = DecodeNamedGetResult(body)
	if payload["lang"].Str != "en" || payload["n"].Int != 7 {
		t.Errorf("named after set: %+v, want lang=en,n=7", payload)
	}
	handleNamedOverwritePayload(tx, EncodeSetPayloadArgs("named", 1, vector.Metadata{"only": vector.NewInt(1)}))
	body, _ = handleNamedGet(tx, EncodeVectorGetArgs("named", 1, GetFlagsBoth))
	_, _, payload, _, _ = DecodeNamedGetResult(body)
	if _, ok := payload["lang"]; ok || payload["only"].Int != 1 {
		t.Errorf("named after overwrite: %+v, want only=1", payload)
	}
	handleNamedDeletePayloadKeys(tx, EncodeDeletePayloadKeysArgs("named", 1, []string{"only"}))
	handleNamedClearPayload(tx, EncodeClearPayloadArgs("named", 1))
	body, _ = handleNamedGet(tx, EncodeVectorGetArgs("named", 1, GetFlagsBoth))
	_, _, payload, _, _ = DecodeNamedGetResult(body)
	if len(payload) != 0 {
		t.Errorf("named after clear: %+v, want empty", payload)
	}

	// absent -> not-found flag (no op error).
	body, err = handleNamedGet(tx, EncodeVectorGetArgs("named", 999, GetFlagsBoth))
	if err != nil {
		t.Fatalf("named get absent: op error %v", err)
	}
	if found, _, _, _, _ := DecodeNamedGetResult(body); found {
		t.Error("named get absent: found = true, want false")
	}
	res, err := handleNamedSetPayload(tx, EncodeSetPayloadArgs("named", 999, vector.Metadata{"x": vector.NewInt(1)}))
	if err != nil {
		t.Fatalf("named set absent: op error %v", err)
	}
	if ok, _ := DecodePayloadResult(res); ok {
		t.Error("named set absent: applied = true, want false")
	}
}

// --- dispatch-through-handler: MV ---

func TestMVGetPayloadViaDispatch(t *testing.T) {
	tx, _ := newGetPayloadTx(t)
	if _, err := handleMVCreate(tx, EncodeMVCreateArgs("mv", vector.MultiVectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1})); err != nil {
		t.Fatal(err)
	}
	tokens := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}
	if _, err := handleMVAdd(tx, EncodeMVAddArgs("mv", 1, tokens, vector.Metadata{"doc": vector.NewInt(3)})); err != nil {
		t.Fatal(err)
	}

	body, err := handleMVGet(tx, EncodeVectorGetArgs("mv", 1, GetFlagsBoth))
	if err != nil {
		t.Fatal(err)
	}
	found, gt, payload, err := DecodeMVGetResult(body)
	if err != nil || !found {
		t.Fatalf("mv get: found=%v err=%v", found, err)
	}
	if len(gt) != 2 || payload["doc"].Int != 3 {
		t.Errorf("mv get: tokens=%+v payload=%+v", gt, payload)
	}

	handleMVSetPayload(tx, EncodeSetPayloadArgs("mv", 1, vector.Metadata{"x": vector.NewInt(9)}))
	body, _ = handleMVGet(tx, EncodeVectorGetArgs("mv", 1, GetFlagsBoth))
	_, _, payload, _ = DecodeMVGetResult(body)
	if payload["doc"].Int != 3 || payload["x"].Int != 9 {
		t.Errorf("mv after set: %+v, want doc=3,x=9", payload)
	}
	handleMVOverwritePayload(tx, EncodeSetPayloadArgs("mv", 1, vector.Metadata{"only": vector.NewInt(1)}))
	handleMVDeletePayloadKeys(tx, EncodeDeletePayloadKeysArgs("mv", 1, []string{"only"}))
	body, _ = handleMVGet(tx, EncodeVectorGetArgs("mv", 1, GetFlagsBoth))
	_, _, payload, _ = DecodeMVGetResult(body)
	if len(payload) != 0 {
		t.Errorf("mv after delete-keys: %+v, want empty", payload)
	}
	handleMVClearPayload(tx, EncodeClearPayloadArgs("mv", 1))

	// absent -> flags.
	body, err = handleMVGet(tx, EncodeVectorGetArgs("mv", 999, GetFlagsBoth))
	if err != nil {
		t.Fatalf("mv get absent: op error %v", err)
	}
	if found, _, _, _ := DecodeMVGetResult(body); found {
		t.Error("mv get absent: found = true, want false")
	}
	res, err := handleMVSetPayload(tx, EncodeSetPayloadArgs("mv", 999, vector.Metadata{"x": vector.NewInt(1)}))
	if err != nil {
		t.Fatalf("mv set absent: op error %v", err)
	}
	if ok, _ := DecodePayloadResult(res); ok {
		t.Error("mv set absent: applied = true, want false")
	}
}

// --- op-kind + key-extractor assertions for all 15 ops ---

func TestGetPayloadOpKinds(t *testing.T) {
	r := NewRegistry()
	if err := RegisterBuiltins(r); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		op   string
		kind OpKind
	}{
		{"vector_get", OpReadOnly},
		{"vector_set_payload", OpReadWrite},
		{"vector_overwrite_payload", OpReadWrite},
		{"vector_delete_payload_keys", OpReadWrite},
		{"vector_clear_payload", OpReadWrite},
		{"vector_named_get", OpReadOnly},
		{"vector_named_set_payload", OpReadWrite},
		{"vector_named_overwrite_payload", OpReadWrite},
		{"vector_named_delete_payload_keys", OpReadWrite},
		{"vector_named_clear_payload", OpReadWrite},
		{"vector_mv_get", OpReadOnly},
		{"vector_mv_set_payload", OpReadWrite},
		{"vector_mv_overwrite_payload", OpReadWrite},
		{"vector_mv_delete_payload_keys", OpReadWrite},
		{"vector_mv_clear_payload", OpReadWrite},
	}
	if len(cases) != 15 {
		t.Fatalf("expected 15 new ops, listed %d", len(cases))
	}
	for _, c := range cases {
		_, kind, ke, ok := r.Lookup(c.op)
		if !ok {
			t.Errorf("%s not registered", c.op)
			continue
		}
		if kind != c.kind {
			t.Errorf("%s kind = %v, want %v", c.op, kind, c.kind)
		}
		if ke == nil {
			t.Errorf("%s has nil key extractor (must be routable)", c.op)
		}
		// CollectionNameFor must extract the canonical collection for every new op.
		var args []byte
		switch c.op {
		case "vector_get", "vector_named_get", "vector_mv_get":
			args = EncodeVectorGetArgs("docs", 1, GetFlagsBoth)
		case "vector_clear_payload", "vector_named_clear_payload", "vector_mv_clear_payload":
			args = EncodeClearPayloadArgs("docs", 1)
		default:
			args = EncodeSetPayloadArgs("docs", 1, nil)
		}
		name, ok := CollectionNameFor(c.op, args)
		if !ok || name != "default/docs" {
			t.Errorf("%s: CollectionNameFor = (%q,%v), want (default/docs,true)", c.op, name, ok)
		}
	}
}

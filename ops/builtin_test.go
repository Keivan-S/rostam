// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/vector"
)

func newTestSetup(t *testing.T) (*Registry, *TxContext) {
	t.Helper()
	cfg := cache.DefaultConfig()
	cfg.NumShards = 1
	c, err := cache.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	r := NewRegistry()
	if err := RegisterBuiltins(r); err != nil {
		t.Fatal(err)
	}
	return r, NewTxContext(c)
}

func TestBuiltinPutGetRoundtrip(t *testing.T) {
	r, tx := newTestSetup(t)

	putH, kind, _, ok := r.Lookup("put")
	if !ok {
		t.Fatal("put not registered")
	}
	if kind != OpReadWrite {
		t.Fatalf("put kind = %v, want OpReadWrite", kind)
	}
	args := EncodePutArgs([]byte("k"), []byte("v"), 0)
	if _, err := putH(tx, args); err != nil {
		t.Fatalf("put handler: %v", err)
	}

	getH, kind, _, ok := r.Lookup("get")
	if !ok {
		t.Fatal("get not registered")
	}
	if kind != OpReadOnly {
		t.Fatalf("get kind = %v, want OpReadOnly", kind)
	}
	res, err := getH(tx, EncodeKeyArgs([]byte("k")))
	if err != nil {
		t.Fatalf("get handler: %v", err)
	}
	if !bytes.Equal(res, []byte("v")) {
		t.Fatalf("get result = %q, want v", res)
	}
}

func TestBuiltinGetMissingReturnsErr(t *testing.T) {
	r, tx := newTestSetup(t)
	getH, _, _, _ := r.Lookup("get")
	_, err := getH(tx, EncodeKeyArgs([]byte("missing")))
	if err != cache.ErrNotFound {
		t.Fatalf("get missing: err = %v, want ErrNotFound", err)
	}
}

func TestBuiltinDel(t *testing.T) {
	r, tx := newTestSetup(t)
	putH, _, _, _ := r.Lookup("put")
	_, _ = putH(tx, EncodePutArgs([]byte("k"), []byte("v"), 0))

	delH, kind, _, ok := r.Lookup("del")
	if !ok {
		t.Fatal("del not registered")
	}
	if kind != OpReadWrite {
		t.Fatalf("del kind = %v, want OpReadWrite", kind)
	}
	res, err := delH(tx, EncodeKeyArgs([]byte("k")))
	if err != nil {
		t.Fatalf("del: %v", err)
	}
	if len(res) != 1 || res[0] != 1 {
		t.Fatalf("del result = %v, want [1] (true)", res)
	}
	// Second del returns 0
	res, _ = delH(tx, EncodeKeyArgs([]byte("k")))
	if len(res) != 1 || res[0] != 0 {
		t.Fatalf("del missing result = %v, want [0]", res)
	}
}

func TestBuiltinExpire(t *testing.T) {
	r, tx := newTestSetup(t)
	putH, _, _, _ := r.Lookup("put")
	_, _ = putH(tx, EncodePutArgs([]byte("k"), []byte("v"), 0))

	expH, kind, _, ok := r.Lookup("expire")
	if !ok {
		t.Fatal("expire not registered")
	}
	if kind != OpReadWrite {
		t.Fatalf("expire kind = %v, want OpReadWrite", kind)
	}
	if _, err := expH(tx, EncodeExpireArgs([]byte("k"), 10*time.Millisecond)); err != nil {
		t.Fatalf("expire: %v", err)
	}
	time.Sleep(30 * time.Millisecond)
	getH, _, _, _ := r.Lookup("get")
	if _, err := getH(tx, EncodeKeyArgs([]byte("k"))); err != cache.ErrNotFound {
		t.Fatalf("post-expire get: err = %v, want ErrNotFound", err)
	}
}

func TestBuiltinIncrCreatesKey(t *testing.T) {
	r, tx := newTestSetup(t)
	incrH, kind, _, ok := r.Lookup("incr")
	if !ok {
		t.Fatal("incr not registered")
	}
	if kind != OpReadWrite {
		t.Fatalf("incr kind = %v, want OpReadWrite", kind)
	}
	res, err := incrH(tx, EncodeIncrArgs([]byte("counter"), 5))
	if err != nil {
		t.Fatalf("incr: %v", err)
	}
	got, err := DecodeIncrResult(res)
	if err != nil {
		t.Fatal(err)
	}
	if got != 5 {
		t.Fatalf("incr result = %d, want 5", got)
	}
}

func TestBuiltinIncrAccumulates(t *testing.T) {
	r, tx := newTestSetup(t)
	incrH, _, _, _ := r.Lookup("incr")
	for range 3 {
		_, err := incrH(tx, EncodeIncrArgs([]byte("counter"), 2))
		if err != nil {
			t.Fatal(err)
		}
	}
	res, _ := incrH(tx, EncodeIncrArgs([]byte("counter"), -1))
	v, _ := DecodeIncrResult(res)
	if v != 5 { // 2 + 2 + 2 - 1
		t.Fatalf("incr accumulated = %d, want 5", v)
	}
}

func TestArgsCodecRoundtrip(t *testing.T) {
	// EncodeKeyArgs ↔ DecodeKeyArgs
	k := []byte("user:42")
	args := EncodeKeyArgs(k)
	dk, err := DecodeKeyArgs(args)
	if err != nil {
		t.Fatalf("DecodeKeyArgs: %v", err)
	}
	if !bytes.Equal(dk, k) {
		t.Fatalf("DecodeKeyArgs: %q, want %q", dk, k)
	}

	// EncodePutArgs ↔ DecodePutArgs
	args = EncodePutArgs(k, []byte("v"), 5*time.Second)
	dk2, dv, dttl, err := DecodePutArgs(args)
	if err != nil {
		t.Fatalf("DecodePutArgs: %v", err)
	}
	if !bytes.Equal(dk2, k) || !bytes.Equal(dv, []byte("v")) {
		t.Fatalf("DecodePutArgs: %q,%q", dk2, dv)
	}
	if dttl != 5*time.Second {
		t.Fatalf("DecodePutArgs ttl = %v, want 5s", dttl)
	}
}

func TestBuiltinPing(t *testing.T) {
	r, tx := newTestSetup(t)

	pingH, kind, _, ok := r.Lookup("__ping__")
	if !ok {
		t.Fatal("__ping__ not registered")
	}
	if kind != OpReadOnly {
		t.Fatalf("__ping__ kind = %v, want OpReadOnly", kind)
	}
	res, err := pingH(tx, nil)
	if err != nil {
		t.Fatalf("ping: %v", err)
	}
	if len(res) != 0 {
		t.Fatalf("ping result = %v, want empty", res)
	}
	// Tolerate non-empty args (handler must not validate to keep the cheap path cheap).
	res, err = pingH(tx, []byte("anything"))
	if err != nil || len(res) != 0 {
		t.Fatalf("ping with args: res=%v err=%v", res, err)
	}
}

func TestStdKeyExtractor(t *testing.T) {
	args := EncodePutArgs([]byte("user:42"), []byte("v"), 0)
	key, ok := StdKeyExtractor(args)
	if !ok {
		t.Fatal("StdKeyExtractor: no key")
	}
	if string(key) != "user:42" {
		t.Fatalf("key = %q, want user:42", key)
	}

	// Short args.
	if _, ok := StdKeyExtractor([]byte{0}); ok {
		t.Error("1-byte args: extractor returned key")
	}
	if _, ok := StdKeyExtractor([]byte{0, 5, 'a', 'b'}); ok {
		t.Error("truncated key: extractor returned key")
	}
}

func TestBuiltinsRegisterWithExtractor(t *testing.T) {
	r := NewRegistry()
	if err := RegisterBuiltins(r); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"get", "put", "del", "expire", "incr"} {
		_, _, ke, ok := r.Lookup(name)
		if !ok {
			t.Errorf("%q not registered", name)
			continue
		}
		if ke == nil {
			t.Errorf("%q registered without extractor", name)
		}
	}
	_, _, ke, ok := r.Lookup("__ping__")
	if !ok {
		t.Error("__ping__ not registered")
	}
	if ke != nil {
		t.Error("__ping__ should be shardless (nil extractor)")
	}
}

func TestVectorOpsViaDispatch(t *testing.T) {
	dir := t.TempDir()
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()
	vstore, err := vector.OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer vstore.Close()
	tx := NewTxContextWithVectors(c, vstore)

	// Create
	cfg := vector.Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: vector.L2}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("docs", cfg)); err != nil {
		t.Fatal(err)
	}
	// Insert
	if _, err := handleVectorInsert(tx, EncodeVectorInsertArgs("docs", 1, []float32{1, 0})); err != nil {
		t.Fatal(err)
	}
	if _, err := handleVectorInsert(tx, EncodeVectorInsertArgs("docs", 2, []float32{2, 0})); err != nil {
		t.Fatal(err)
	}
	// Search
	body, err := handleVectorSearch(tx, EncodeVectorSearchArgs("docs", 1, []float32{1, 0}))
	if err != nil {
		t.Fatal(err)
	}
	results, err := DecodeVectorSearchResults(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].ID != 1 {
		t.Errorf("search returned %+v, want id 1", results)
	}
}

// TestVectorCreateCollectionPersistentViaDispatch proves the networked
// create-collection path honors Config.Persistent: a collection created through
// the op handler is persistent server-side, and after Flush + store reopen it
// instant-restarts with identical search results.
func TestVectorCreateCollectionPersistentViaDispatch(t *testing.T) {
	dir := t.TempDir()
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()
	vstore, err := vector.OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	tx := NewTxContextWithVectors(c, vstore)

	cfg := vector.Config{
		Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.Cosine,
		Quant: vector.QuantSQ8, RescoreFactor: 2, Persistent: true,
	}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("docs", cfg)); err != nil {
		t.Fatal(err)
	}
	coll, ok := vstore.Get("docs")
	if !ok || !coll.Config().Persistent {
		t.Fatalf("collection not persistent server-side (ok=%v)", ok)
	}

	for i := 1; i <= 8; i++ {
		v := []float32{float32(i), 1, float32(i % 3), 0}
		if _, err := handleVectorInsert(tx, EncodeVectorInsertArgs("docs", uint64(i), v)); err != nil {
			t.Fatal(err)
		}
	}
	q := []float32{2, 1, 1, 0}
	beforeBody, err := handleVectorSearch(tx, EncodeVectorSearchArgs("docs", 3, q))
	if err != nil {
		t.Fatal(err)
	}
	before, _ := DecodeVectorSearchResults(beforeBody)

	if err := vstore.Flush("docs"); err != nil {
		t.Fatalf("flush persistent collection: %v", err)
	}
	if err := vstore.Close(); err != nil {
		t.Fatal(err)
	}

	// Reopen the store — instant restart of the persistent collection.
	vstore2, err := vector.OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen store: %v", err)
	}
	defer vstore2.Close()
	tx2 := NewTxContextWithVectors(c, vstore2)
	afterBody, err := handleVectorSearch(tx2, EncodeVectorSearchArgs("docs", 3, q))
	if err != nil {
		t.Fatal(err)
	}
	after, _ := DecodeVectorSearchResults(afterBody)

	if len(before) == 0 || len(before) != len(after) {
		t.Fatalf("result count changed across restart: %d -> %d", len(before), len(after))
	}
	for i := range before {
		if before[i].ID != after[i].ID {
			t.Errorf("result %d id changed across restart: %d -> %d", i, before[i].ID, after[i].ID)
		}
	}
}

// TestHandleVectorHybridLanes verifies that handleVectorHybridLanes returns
// the same dense and sparse lanes as Collection.HybridLanes directly.
func TestHandleVectorHybridLanes(t *testing.T) {
	dir := t.TempDir()
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()
	vstore, err := vector.OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer vstore.Close()
	tx := NewTxContextWithVectors(c, vstore)

	// Create a collection with dim=4 so we can exercise both dense and sparse.
	cfg := vector.Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("docs", cfg)); err != nil {
		t.Fatal(err)
	}

	// Insert docs 1-5 near dense origin with weak shared sparse term.
	for i := uint64(1); i <= 5; i++ {
		v := []float32{float32(i) * 0.01, 0, 0, 0}
		sv := vector.SparseVector{Indices: []uint32{1}, Values: []float32{0.1}}
		args := EncodeVectorUpsertArgs("docs", i, v, "", 0, nil, sv)
		if _, err := handleVectorUpsert(tx, args); err != nil {
			t.Fatalf("upsert %d: %v", i, err)
		}
	}
	// Insert doc 100 far in dense space but strong sparse term 42.
	sv100 := vector.SparseVector{Indices: []uint32{42}, Values: []float32{10.0}}
	args100 := EncodeVectorUpsertArgs("docs", 100, []float32{9, 9, 9, 9}, "", 0, nil, sv100)
	if _, err := handleVectorUpsert(tx, args100); err != nil {
		t.Fatalf("upsert 100: %v", err)
	}

	denseQuery := []float32{0, 0, 0, 0}
	sparseQuery := vector.SparseVector{Indices: []uint32{42}, Values: []float32{5.0}}
	opts := vector.HybridOpts{DenseK: 10, SparseK: 10}
	encArgs := EncodeHybridSearchArgs("docs", denseQuery, 5, sparseQuery, opts)

	body, err := handleVectorHybridLanes(tx, encArgs)
	if err != nil {
		t.Fatal(err)
	}
	gotD, gotS, err := DecodeHybridLanesResult(body)
	if err != nil {
		t.Fatal(err)
	}

	// Compare against Collection.HybridLanes directly.
	coll, ok := tx.vectors.Acquire("docs")
	if !ok {
		t.Fatal("acquire docs")
	}
	defer coll.Release()
	wantD, wantS, err := coll.HybridLanes(denseQuery, sparseQuery, 5, opts)
	if err != nil {
		t.Fatal(err)
	}

	if len(gotD) != len(wantD) || len(gotS) != len(wantS) {
		t.Fatalf("lanes lengths got (%d,%d) want (%d,%d)", len(gotD), len(gotS), len(wantD), len(wantS))
	}
	for i := range wantD {
		if gotD[i].ID != wantD[i].ID || gotD[i].Distance != wantD[i].Distance {
			t.Fatalf("dense lane mismatch at %d: got %+v want %+v", i, gotD[i], wantD[i])
		}
	}
	for i := range wantS {
		if gotS[i].ID != wantS[i].ID || gotS[i].Score != wantS[i].Score {
			t.Fatalf("sparse lane mismatch at %d: got %+v want %+v", i, gotS[i], wantS[i])
		}
	}
}

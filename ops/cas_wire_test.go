// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"errors"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/vector"
)

// --- codec: expected_version round-trip + byte-identical-when-absent ---

func TestInsertArgsCASRoundtrip(t *testing.T) {
	meta := vector.Metadata{"a": vector.NewInt(1)}
	sparse := vector.SparseVector{Indices: []uint32{0, 3}, Values: []float32{1, 2}}
	enc := EncodeVectorInsertArgsCAS("acme/docs", 42, []float32{1, 2, 3}, 2*time.Second, meta, sparse, 7, true)
	col, id, vec, ttl, gotMeta, gotSparse, version, exp, hasExp, err := DecodeVectorInsertArgsCAS(enc)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if col != "acme/docs" || id != 42 || len(vec) != 3 || ttl != 2*time.Second {
		t.Fatalf("decoded col=%q id=%d vec=%v ttl=%v", col, id, vec, ttl)
	}
	if gotMeta["a"].Int != 1 || gotSparse == nil || len(gotSparse.Indices) != 2 {
		t.Fatalf("meta=%+v sparse=%+v", gotMeta, gotSparse)
	}
	if version != 0 || exp != 7 || !hasExp {
		t.Fatalf("version=%d expected=%d has=%v, want 0,7,true", version, exp, hasExp)
	}
	// expected_version 0 with has-flag = expect-absent CAS.
	enc0 := EncodeVectorInsertArgsCAS("d", 1, []float32{1}, 0, nil, vector.SparseVector{}, 0, true)
	_, _, _, _, _, _, _, exp0, has0, err := DecodeVectorInsertArgsCAS(enc0)
	if err != nil || exp0 != 0 || !has0 {
		t.Fatalf("expect-absent CAS: exp=%d has=%v err=%v", exp0, has0, err)
	}
}

func TestInsertArgsCASByteIdenticalWhenAbsent(t *testing.T) {
	meta := vector.Metadata{"k": vector.NewInt(7)}
	sparse := vector.SparseVector{Indices: []uint32{1}, Values: []float32{9}}
	legacy := EncodeVectorInsertArgsExt("docs", 9, []float32{1, 2}, time.Second, meta, sparse)
	cas := EncodeVectorInsertArgsCAS("docs", 9, []float32{1, 2}, time.Second, meta, sparse, 0, false)
	if !bytes.Equal(legacy, cas) {
		t.Fatalf("no-CAS insert not byte-identical:\n legacy=%v\n cas   =%v", legacy, cas)
	}
	upLegacy := EncodeVectorUpsertArgs("docs", 9, []float32{1, 2}, "c", time.Second, meta, sparse)
	upCAS := EncodeVectorUpsertArgsCAS("docs", 9, []float32{1, 2}, "c", time.Second, meta, sparse, 0, false)
	if !bytes.Equal(upLegacy, upCAS) {
		t.Fatalf("no-CAS upsert not byte-identical")
	}
}

func TestDeleteArgsCASRoundtripAndByteIdentical(t *testing.T) {
	legacy := EncodeVectorDeleteArgs("docs", 5)
	cas := EncodeVectorDeleteArgsCAS("docs", 5, 0, false)
	if !bytes.Equal(legacy, cas) {
		t.Fatalf("no-CAS delete not byte-identical:\n legacy=%v\n cas   =%v", legacy, cas)
	}
	// Legacy bytes decode with hasExpected=false.
	if _, _, _, has, err := DecodeVectorDeleteArgsCAS(legacy); err != nil || has {
		t.Fatalf("legacy delete decode: has=%v err=%v", has, err)
	}
	enc := EncodeVectorDeleteArgsCAS("docs", 5, 3, true)
	col, id, exp, has, err := DecodeVectorDeleteArgsCAS(enc)
	if err != nil || col != "docs" || id != 5 || exp != 3 || !has {
		t.Fatalf("decoded col=%q id=%d exp=%d has=%v err=%v", col, id, exp, has, err)
	}
}

func TestSetPayloadArgsCASRoundtripAndByteIdentical(t *testing.T) {
	meta := vector.Metadata{"a": vector.NewInt(1)}
	ttl := map[string]int64{"a": 1000}
	// no CAS → byte-identical to the Opts encoder (with + without key-ttl).
	for _, km := range []map[string]int64{nil, ttl} {
		legacy := EncodeSetPayloadArgsOpts("docs", 7, meta, km)
		cas := EncodeSetPayloadArgsCAS("docs", 7, meta, km, 0, false)
		if !bytes.Equal(legacy, cas) {
			t.Fatalf("no-CAS set_payload not byte-identical (km=%v)", km)
		}
	}
	// CAS present: round-trips meta + key-ttl + expected.
	enc := EncodeSetPayloadArgsCAS("docs", 7, meta, ttl, 4, true)
	col, id, gotMeta, gotTTL, exp, has, err := DecodeSetPayloadArgsCAS(enc)
	if err != nil || col != "docs" || id != 7 || gotMeta["a"].Int != 1 || gotTTL["a"] != 1000 || exp != 4 || !has {
		t.Fatalf("decoded col=%q id=%d meta=%+v ttl=%+v exp=%d has=%v err=%v", col, id, gotMeta, gotTTL, exp, has, err)
	}
	// CAS present with NO key-ttl: still round-trips (the present byte is framed).
	enc2 := EncodeSetPayloadArgsCAS("docs", 7, meta, nil, 4, true)
	_, _, _, gotTTL2, exp2, has2, err := DecodeSetPayloadArgsCAS(enc2)
	if err != nil || len(gotTTL2) != 0 || exp2 != 4 || !has2 {
		t.Fatalf("no-ttl CAS: ttl=%+v exp=%d has=%v err=%v", gotTTL2, exp2, has2, err)
	}
	// Legacy Opts decoder still reads a CAS-encoded blob's meta + key-ttl (it stops
	// before the CAS block).
	_, _, lMeta, lTTL, lErr := DecodeSetPayloadArgsOpts(enc)
	if lErr != nil || lMeta["a"].Int != 1 || lTTL["a"] != 1000 {
		t.Fatalf("legacy opts decode of CAS blob: meta=%+v ttl=%+v err=%v", lMeta, lTTL, lErr)
	}
}

func TestDeletePayloadKeysAndClearArgsCAS(t *testing.T) {
	keys := []string{"x", "y"}
	legacy := EncodeDeletePayloadKeysArgs("docs", 1, keys)
	cas := EncodeDeletePayloadKeysArgsCAS("docs", 1, keys, 0, false)
	if !bytes.Equal(legacy, cas) {
		t.Fatalf("no-CAS delete_payload_keys not byte-identical")
	}
	col, id, gotKeys, exp, has, err := DecodeDeletePayloadKeysArgsCAS(EncodeDeletePayloadKeysArgsCAS("docs", 1, keys, 9, true))
	if err != nil || col != "docs" || id != 1 || len(gotKeys) != 2 || exp != 9 || !has {
		t.Fatalf("decoded col=%q id=%d keys=%v exp=%d has=%v err=%v", col, id, gotKeys, exp, has, err)
	}
	clLegacy := EncodeClearPayloadArgs("docs", 2)
	clCAS := EncodeClearPayloadArgsCAS("docs", 2, 0, false)
	if !bytes.Equal(clLegacy, clCAS) {
		t.Fatalf("no-CAS clear_payload not byte-identical")
	}
	_, _, cexp, chas, err := DecodeClearPayloadArgsCAS(EncodeClearPayloadArgsCAS("docs", 2, 6, true))
	if err != nil || cexp != 6 || !chas {
		t.Fatalf("clear CAS: exp=%d has=%v err=%v", cexp, chas, err)
	}
}

// --- codec: version in the get result (single + batch) ---

func TestGetResultVersionRoundtripAndByteIdentical(t *testing.T) {
	vec := []float32{1, 2}
	// version 0 → BYTE-IDENTICAL to the pre-version encoder.
	legacy := EncodeVectorGetResult(true, vec, nil, 0, nil, true, true)
	v0 := EncodeVectorGetResultV(true, vec, nil, 0, nil, true, true, 0)
	if !bytes.Equal(legacy, v0) {
		t.Fatalf("version-0 get result not byte-identical:\n legacy=%v\n v0    =%v", legacy, v0)
	}
	// version != 0 → carried on the wire.
	found, gv, _, _, _, version, err := DecodeVectorGetResultV(EncodeVectorGetResultV(true, vec, nil, 0, nil, true, true, 11))
	if err != nil || !found || len(gv) != 2 || version != 11 {
		t.Fatalf("found=%v vec=%v version=%d err=%v", found, gv, version, err)
	}
	// The legacy version-less decoder still reads a versioned body (ignores version).
	if f, _, _, _, _, err := DecodeVectorGetResult(EncodeVectorGetResultV(true, vec, nil, 0, nil, true, true, 11)); err != nil || !f {
		t.Fatalf("legacy decode of versioned body: found=%v err=%v", f, err)
	}
}

func TestGetBatchResultVersionRoundtrip(t *testing.T) {
	rows := []GetBatchRow{
		{ID: 1, Found: true, Vec: []float32{1}, Version: 3},
		{ID: 2, Found: false},
		{ID: 3, Found: true, Vec: []float32{2}, Version: 0}, // found but version 0
	}
	got, err := DecodeVectorGetBatchResult(EncodeVectorGetBatchResult(rows))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("rows = %d, want 3", len(got))
	}
	if got[0].ID != 1 || !got[0].Found || got[0].Version != 3 {
		t.Errorf("row0 = %+v", got[0])
	}
	if got[1].ID != 2 || got[1].Found {
		t.Errorf("row1 = %+v", got[1])
	}
	if got[2].ID != 3 || !got[2].Found || got[2].Version != 0 {
		t.Errorf("row2 = %+v", got[2])
	}
}

// --- handler: CAS enforcement + version on Get (engine-backed) ---

func newCASTx(t *testing.T) *TxContext {
	t.Helper()
	dir := t.TempDir()
	c, _ := cache.New(cache.DefaultConfig())
	t.Cleanup(func() { c.Close() })
	vstore, err := vector.OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { vstore.Close() })
	tx := NewTxContextWithVectors(c, vstore)
	cfg := vector.Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: vector.L2}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("docs", cfg)); err != nil {
		t.Fatal(err)
	}
	return tx
}

func getVersion(t *testing.T, tx *TxContext, id uint64) (uint64, bool) {
	t.Helper()
	body, err := handleVectorGet(tx, EncodeVectorGetArgs("docs", id, GetFlagsBoth))
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	found, _, _, _, _, version, err := DecodeVectorGetResultV(body)
	if err != nil {
		t.Fatalf("decode get: %v", err)
	}
	return version, found
}

func TestHandlerInsertCASAndGetVersion(t *testing.T) {
	tx := newCASTx(t)
	// Plain insert → version 1, surfaced by Get.
	if _, err := handleVectorInsert(tx, EncodeVectorInsertArgs("docs", 1, []float32{1, 0})); err != nil {
		t.Fatal(err)
	}
	if v, found := getVersion(t, tx, 1); !found || v != 1 {
		t.Fatalf("after insert: version=%d found=%v, want 1,true", v, found)
	}
	// CAS upsert with matching expected (1) → applies, bumps to... fresh insert = 1.
	okArgs := EncodeVectorUpsertArgsCAS("docs", 1, []float32{2, 0}, "", 0, nil, vector.SparseVector{}, 1, true)
	if _, err := handleVectorUpsert(tx, okArgs); err != nil {
		t.Fatalf("matching CAS upsert: %v", err)
	}
	// CAS upsert with WRONG expected → ErrVersionConflict.
	badArgs := EncodeVectorUpsertArgsCAS("docs", 1, []float32{3, 0}, "", 0, nil, vector.SparseVector{}, 99, true)
	if _, err := handleVectorUpsert(tx, badArgs); !errors.Is(err, vector.ErrVersionConflict) {
		t.Fatalf("mismatched CAS upsert err = %v, want ErrVersionConflict", err)
	}
	// expect-absent CAS (expected 0) on a LIVE id → conflict.
	absentArgs := EncodeVectorInsertArgsCAS("docs", 1, []float32{4, 0}, 0, nil, vector.SparseVector{}, 0, true)
	if _, err := handleVectorInsert(tx, absentArgs); !errors.Is(err, vector.ErrVersionConflict) {
		t.Fatalf("expect-absent CAS on live id err = %v, want ErrVersionConflict", err)
	}
}

func TestHandlerSetPayloadCAS(t *testing.T) {
	tx := newCASTx(t)
	if _, err := handleVectorInsert(tx, EncodeVectorInsertArgs("docs", 1, []float32{1, 0})); err != nil {
		t.Fatal(err)
	}
	v, _ := getVersion(t, tx, 1)
	meta := vector.Metadata{"k": vector.NewInt(5)}
	// Matching expected → applied.
	body, err := handleVectorSetPayload(tx, EncodeSetPayloadArgsCAS("docs", 1, meta, nil, v, true))
	if err != nil {
		t.Fatalf("matching set_payload CAS: %v", err)
	}
	if applied, _ := DecodePayloadResult(body); !applied {
		t.Fatal("matching set_payload CAS: applied=false")
	}
	// Mismatched expected → conflict.
	if _, err := handleVectorSetPayload(tx, EncodeSetPayloadArgsCAS("docs", 1, meta, nil, v+999, true)); !errors.Is(err, vector.ErrVersionConflict) {
		t.Fatalf("mismatched set_payload CAS err = %v, want ErrVersionConflict", err)
	}
}

func TestHandlerDeleteCAS(t *testing.T) {
	tx := newCASTx(t)
	if _, err := handleVectorInsert(tx, EncodeVectorInsertArgs("docs", 1, []float32{1, 0})); err != nil {
		t.Fatal(err)
	}
	v, _ := getVersion(t, tx, 1)
	// Mismatched expected → conflict, point survives.
	if _, err := handleVectorDelete(tx, EncodeVectorDeleteArgsCAS("docs", 1, v+1, true)); !errors.Is(err, vector.ErrVersionConflict) {
		t.Fatalf("mismatched delete CAS err = %v, want ErrVersionConflict", err)
	}
	if _, found := getVersion(t, tx, 1); !found {
		t.Fatal("point removed despite CAS conflict")
	}
	// Matching expected → deleted.
	body, err := handleVectorDelete(tx, EncodeVectorDeleteArgsCAS("docs", 1, v, true))
	if err != nil {
		t.Fatalf("matching delete CAS: %v", err)
	}
	if len(body) != 1 || body[0] != 1 {
		t.Fatalf("matching delete CAS body = %v, want [1]", body)
	}
}

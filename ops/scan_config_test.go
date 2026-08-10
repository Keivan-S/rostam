// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/vector"
)

// TestScanVectorsResultRoundtrip exercises the wire codec directly: a record
// list with vec, ttl, metadata, and sparse round-trips byte-for-byte (matching
// what vector_insert consumes).
func TestScanVectorsResultRoundtrip(t *testing.T) {
	recs := []vector.ScanRecord{
		{
			ID:       7,
			Vec:      []float32{1, 2, 3},
			TTL:      90 * time.Second,
			Metadata: vector.Metadata{"src": vector.NewString("a"), "ord": vector.NewInt(5)},
			Sparse:   &vector.SparseVector{Indices: []uint32{1, 4}, Values: []float32{0.5, 0.25}},
		},
		{ID: 8, Vec: []float32{0, 0, 0}}, // no ttl, no meta, no sparse
	}
	got, err := DecodeScanVectorsResult(EncodeScanVectorsResult(recs))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("decoded %d records, want 2", len(got))
	}
	r := got[0]
	if r.ID != 7 || len(r.Vec) != 3 || r.Vec[2] != 3 {
		t.Errorf("record 0 = %+v", r)
	}
	if r.TTL != 90*time.Second {
		t.Errorf("record 0 ttl = %v", r.TTL)
	}
	if r.Metadata["src"].Str != "a" || r.Metadata["ord"].Int != 5 {
		t.Errorf("record 0 metadata = %v", r.Metadata)
	}
	if r.Sparse == nil || len(r.Sparse.Indices) != 2 || r.Sparse.Indices[1] != 4 || r.Sparse.Values[0] != 0.5 {
		t.Errorf("record 0 sparse = %+v", r.Sparse)
	}
	if got[1].ID != 8 || got[1].TTL != 0 || got[1].Metadata != nil || got[1].Sparse != nil {
		t.Errorf("record 1 = %+v", got[1])
	}
}

// TestHandleVectorScanVectors creates a collection, upserts docs, deletes one,
// then verifies the handler returns exactly the live records and that they match
// Collection.ScanVectors directly.
func TestHandleVectorScanVectors(t *testing.T) {
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()
	vstore, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer vstore.Close()
	tx := NewTxContextWithVectors(c, vstore)

	cfg := vector.Config{Dim: 3, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("docs", cfg)); err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 5; i++ {
		args := EncodeVectorUpsertArgs("docs", uint64(i), []float32{float32(i), 0, 0}, "chunk", 0,
			vector.Metadata{"ord": vector.NewInt(int64(i))}, vector.SparseVector{})
		if _, err := handleVectorUpsert(tx, args); err != nil {
			t.Fatalf("upsert %d: %v", i, err)
		}
	}
	if _, err := handleVectorDelete(tx, EncodeVectorDeleteArgs("docs", 3)); err != nil {
		t.Fatal(err)
	}

	body, err := handleVectorScanVectors(tx, EncodeScanVectorsArgs("docs"))
	if err != nil {
		t.Fatal(err)
	}
	recs, err := DecodeScanVectorsResult(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(recs) != 4 {
		t.Fatalf("handler returned %d records, want 4", len(recs))
	}

	// Compare against Collection.ScanVectors directly (same engine call).
	col, ok := vstore.Acquire("docs")
	if !ok {
		t.Fatal("acquire docs")
	}
	direct := col.ScanVectors()
	col.Release()
	if len(direct) != len(recs) {
		t.Fatalf("handler %d vs direct %d records", len(recs), len(direct))
	}

	byID := func(rs []vector.ScanRecord, id uint64) (vector.ScanRecord, bool) {
		for _, r := range rs {
			if r.ID == id {
				return r, true
			}
		}
		return vector.ScanRecord{}, false
	}
	if _, present := byID(recs, 3); present {
		t.Error("deleted id 3 present in scan result")
	}
	for _, want := range direct {
		got, ok := byID(recs, want.ID)
		if !ok {
			t.Fatalf("id %d in direct scan but missing from handler result", want.ID)
		}
		if len(got.Vec) != len(want.Vec) || got.Vec[0] != want.Vec[0] {
			t.Errorf("id %d vec = %v, want %v", want.ID, got.Vec, want.Vec)
		}
		// Upsert stores content in the reserved field, so metadata carries both
		// the user "ord" and "$content" — it must round-trip intact.
		if got.Metadata["ord"].Int != want.Metadata["ord"].Int {
			t.Errorf("id %d ord = %v, want %v", want.ID, got.Metadata["ord"], want.Metadata["ord"])
		}
		if len(got.Metadata) != len(want.Metadata) {
			t.Errorf("id %d metadata size = %d, want %d (%v vs %v)", want.ID, len(got.Metadata), len(want.Metadata), got.Metadata, want.Metadata)
		}
	}
}

// TestHandleVectorGetConfig creates a collection with a known Config and
// verifies the handler returns it intact.
func TestHandleVectorGetConfig(t *testing.T) {
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()
	vstore, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer vstore.Close()
	tx := NewTxContextWithVectors(c, vstore)

	cfg := vector.Config{Dim: 16, Metric: vector.Cosine, M: 24, EfConstruction: 120, EfSearch: 48, Seed: 99}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("k", cfg)); err != nil {
		t.Fatal(err)
	}

	body, err := handleVectorGetConfig(tx, EncodeGetConfigArgs("k"))
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeGetConfigResult(body)
	if err != nil {
		t.Fatal(err)
	}
	if got.Dim != 16 || got.Metric != vector.Cosine || got.M != 24 ||
		got.EfConstruction != 120 || got.EfSearch != 48 || got.Seed != 99 {
		t.Errorf("config = %+v, want Dim16/Cosine/M24/Efc120/Efs48/Seed99", got)
	}
}

// TestGetConfigPropagatesOPQ proves the reshard propagation path carries OPQ:
// reshard reads a partition's config via vector_get_config (JSON) and recreates
// the new-gen partitions with it (EncodeCreateCollectionArgs). A PQ-OPQ create's
// config must therefore round-trip OPQ=true through get_config so the new
// partitions inherit the rotation.
func TestGetConfigPropagatesOPQ(t *testing.T) {
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()
	vstore, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer vstore.Close()
	tx := NewTxContextWithVectors(c, vstore)

	cfg := vector.Config{
		Dim: 16, Metric: vector.L2, M: 16, EfConstruction: 120, EfSearch: 48, Seed: 7,
		Quant: vector.QuantPQ, QuantPQM: 8, OPQ: true,
	}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("opqc", cfg)); err != nil {
		t.Fatal(err)
	}
	body, err := handleVectorGetConfig(tx, EncodeGetConfigArgs("opqc"))
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeGetConfigResult(body)
	if err != nil {
		t.Fatal(err)
	}
	if !got.OPQ {
		t.Fatal("get_config dropped OPQ — reshard would not propagate the rotation")
	}
	// And the propagated config re-creates a valid PQ-OPQ partition (the reshard
	// create call): a byte-for-byte EncodeCreateCollectionArgs(got) round-trips OPQ.
	_, reCfg, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("opqc-g1-0", got))
	if err != nil {
		t.Fatal(err)
	}
	if !reCfg.OPQ {
		t.Fatal("reshard re-create wire dropped OPQ")
	}
}

// TestGetConfigPropagatesPQDropVecs proves the reshard propagation path carries
// PQDropVecs: reshard reads a partition's config via vector_get_config (JSON) and
// recreates the new-gen partitions with it. A PQ-drop create's config must
// round-trip PQDropVecs=true through get_config (JSON) and the re-create wire so
// the new partitions inherit the dropped-floats mode.
func TestGetConfigPropagatesPQDropVecs(t *testing.T) {
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()
	vstore, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer vstore.Close()
	tx := NewTxContextWithVectors(c, vstore)

	cfg := vector.Config{
		Dim: 16, Metric: vector.L2, M: 16, EfConstruction: 120, EfSearch: 48, Seed: 7,
		Quant: vector.QuantPQ, QuantPQM: 8, PQDropVecs: true,
	}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("dropc", cfg)); err != nil {
		t.Fatal(err)
	}
	body, err := handleVectorGetConfig(tx, EncodeGetConfigArgs("dropc"))
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeGetConfigResult(body)
	if err != nil {
		t.Fatal(err)
	}
	if !got.PQDropVecs {
		t.Fatal("get_config dropped PQDropVecs — reshard would not propagate the dropped-floats mode")
	}
	// The propagated config re-creates a valid PQ-drop partition (the reshard
	// create call): a byte-for-byte EncodeCreateCollectionArgs(got) round-trips it.
	_, reCfg, err := DecodeCreateCollectionArgs(EncodeCreateCollectionArgs("dropc-g1-0", got))
	if err != nil {
		t.Fatal(err)
	}
	if !reCfg.PQDropVecs {
		t.Fatal("reshard re-create wire dropped PQDropVecs")
	}
}

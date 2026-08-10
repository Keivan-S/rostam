// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/vector"
)

func TestIfAbsentResultCodecRoundtrip(t *testing.T) {
	for _, want := range []bool{true, false} {
		got, err := DecodeIfAbsentResult(EncodeIfAbsentResult(want))
		if err != nil {
			t.Fatalf("DecodeIfAbsentResult: %v", err)
		}
		if got != want {
			t.Fatalf("if-absent result roundtrip: got %v, want %v", got, want)
		}
	}
	if _, err := DecodeIfAbsentResult(nil); err == nil {
		t.Fatalf("DecodeIfAbsentResult(nil) = nil err, want truncated error")
	}
}

func TestExistsCodecRoundtrip(t *testing.T) {
	col, id, err := DecodeExistsArgs(EncodeExistsArgs("docs", 42))
	if err != nil {
		t.Fatalf("DecodeExistsArgs: %v", err)
	}
	if col != "docs" || id != 42 {
		t.Fatalf("exists args roundtrip: got (%q,%d), want (docs,42)", col, id)
	}
	for _, want := range []bool{true, false} {
		got, derr := DecodeExistsResult(EncodeExistsResult(want))
		if derr != nil {
			t.Fatalf("DecodeExistsResult: %v", derr)
		}
		if got != want {
			t.Fatalf("exists result roundtrip: got %v, want %v", got, want)
		}
	}
	mvCol, mvID, err := DecodeMVExistsArgs(EncodeMVExistsArgs("mv", 7))
	if err != nil {
		t.Fatalf("DecodeMVExistsArgs: %v", err)
	}
	if mvCol != "mv" || mvID != 7 {
		t.Fatalf("mv exists args roundtrip: got (%q,%d), want (mv,7)", mvCol, mvID)
	}
}

// TestIfAbsentExistsOpKinds asserts the Raft serialization contract: the two
// *_if_absent ops are OpReadWrite (their atomicity guarantee), exists ops are
// OpReadOnly (cheap probe, no Raft).
func TestIfAbsentExistsOpKinds(t *testing.T) {
	r := NewRegistry()
	if err := RegisterBuiltins(r); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	cases := []struct {
		op   string
		kind OpKind
	}{
		{"vector_insert_if_absent", OpReadWrite},
		{"vector_mv_add_if_absent", OpReadWrite},
		{"vector_exists", OpReadOnly},
		{"vector_mv_exists", OpReadOnly},
	}
	for _, tc := range cases {
		_, kind, _, ok := r.Lookup(tc.op)
		if !ok {
			t.Fatalf("op %q not registered", tc.op)
		}
		if kind != tc.kind {
			t.Fatalf("op %q kind = %d, want %d", tc.op, kind, tc.kind)
		}
	}
}

func newVecTx(t *testing.T) *TxContext {
	t.Helper()
	c, _ := cache.New(cache.DefaultConfig())
	t.Cleanup(func() { _ = c.Close() })
	vstore, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatalf("OpenCollectionStore: %v", err)
	}
	t.Cleanup(func() { _ = vstore.Close() })
	return NewTxContextWithVectors(c, vstore)
}

func TestVectorInsertIfAbsentViaDispatch(t *testing.T) {
	tx := newVecTx(t)
	cfg := vector.Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: vector.L2}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("docs", cfg)); err != nil {
		t.Fatal(err)
	}
	// Insert when absent -> inserted=true.
	body, err := handleVectorInsertIfAbsent(tx, EncodeVectorInsertArgs("docs", 1, []float32{1, 0}))
	if err != nil {
		t.Fatal(err)
	}
	if ins, _ := DecodeIfAbsentResult(body); !ins {
		t.Fatalf("if-absent on absent id: inserted=false, want true")
	}
	// Exists -> true.
	eb, err := handleVectorExists(tx, EncodeExistsArgs("docs", 1))
	if err != nil {
		t.Fatal(err)
	}
	if ex, _ := DecodeExistsResult(eb); !ex {
		t.Fatalf("exists(1) = false after insert, want true")
	}
	// Second if-absent with a different value -> no-op (inserted=false) and the
	// stored vector is unchanged.
	body, err = handleVectorInsertIfAbsent(tx, EncodeVectorInsertArgs("docs", 1, []float32{9, 9}))
	if err != nil {
		t.Fatal(err)
	}
	if ins, _ := DecodeIfAbsentResult(body); ins {
		t.Fatalf("if-absent on live id: inserted=true, want false")
	}
	sb, err := handleVectorSearch(tx, EncodeVectorSearchArgs("docs", 1, []float32{1, 0}))
	if err != nil {
		t.Fatal(err)
	}
	res, _ := DecodeVectorSearchResults(sb)
	if len(res) != 1 || res[0].ID != 1 || res[0].Distance != 0 {
		t.Fatalf("live value clobbered: got %+v, want id 1 dist 0", res)
	}
	// Exists on an absent id -> false.
	eb, err = handleVectorExists(tx, EncodeExistsArgs("docs", 999))
	if err != nil {
		t.Fatal(err)
	}
	if ex, _ := DecodeExistsResult(eb); ex {
		t.Fatalf("exists(999) = true, want false (absent)")
	}
}

func TestMVAddIfAbsentExistsViaDispatch(t *testing.T) {
	tx := newVecTx(t)
	cfg := vector.MultiVectorConfig{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1}
	if _, err := handleMVCreate(tx, EncodeMVCreateArgs("mv", cfg)); err != nil {
		t.Fatal(err)
	}
	tok := [][]float32{{1, 0}}
	body, err := handleMVAddIfAbsent(tx, EncodeMVAddArgs("mv", 1, tok, nil))
	if err != nil {
		t.Fatal(err)
	}
	if ins, _ := DecodeIfAbsentResult(body); !ins {
		t.Fatalf("mv add-if-absent on absent doc: inserted=false, want true")
	}
	eb, err := handleMVExists(tx, EncodeMVExistsArgs("mv", 1))
	if err != nil {
		t.Fatal(err)
	}
	if ex, _ := DecodeExistsResult(eb); !ex {
		t.Fatalf("mv exists(1) = false after add, want true")
	}
	// No-op on a live doc.
	body, err = handleMVAddIfAbsent(tx, EncodeMVAddArgs("mv", 1, [][]float32{{0, 1}}, nil))
	if err != nil {
		t.Fatal(err)
	}
	if ins, _ := DecodeIfAbsentResult(body); ins {
		t.Fatalf("mv add-if-absent on live doc: inserted=true, want false")
	}
	// Delete -> exists false.
	if _, err := handleMVDelete(tx, EncodeMVDeleteArgs("mv", 1)); err != nil {
		t.Fatal(err)
	}
	eb, err = handleMVExists(tx, EncodeMVExistsArgs("mv", 1))
	if err != nil {
		t.Fatal(err)
	}
	if ex, _ := DecodeExistsResult(eb); ex {
		t.Fatalf("mv exists(1) = true after delete, want false")
	}
}

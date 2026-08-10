// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/vector"
)

// TestScrollArgsCursorBackwardCompat proves the cursor extension is byte-identical
// to the legacy forms when no cursor (and no opts) is present, so an old
// encoder/decoder is unaffected — the Task-2 backward-compat contract.
func TestScrollArgsCursorBackwardCompat(t *testing.T) {
	f := vector.Filter{Op: vector.FilterEq, Field: "doc", Value: vector.NewInt(2)}

	// (a) No cursor + no opts ⇒ byte-identical to plain EncodeScrollArgs.
	base := EncodeScrollArgs("acme/docs", f, 50)
	noCursor := EncodeScrollArgsCursorBounded("acme/docs", f, 50, 0, 0, 0, false, 0)
	if !bytes.Equal(base, noCursor) {
		t.Fatalf("no-cursor/no-opts bytes differ from legacy:\n base=%v\n new =%v", base, noCursor)
	}
	// EncodeScrollArgsOpts (legacy wrapper) must also stay byte-identical.
	if !bytes.Equal(base, EncodeScrollArgsOpts("acme/docs", f, 50, 0, 0)) {
		t.Fatalf("EncodeScrollArgsOpts no-opts bytes differ from legacy")
	}

	// (b) opts-only (no cursor) ⇒ byte-identical to the pre-cursor opts trailer
	// [1][rc][opa] (no cursor tail). This must also equal the legacy
	// EncodeScrollArgsOpts output, so old decoders are unaffected.
	optsOnly := EncodeScrollArgsCursorBounded("acme/docs", f, 50, 1, 2, 0, false, 0)
	wantOpts := append(append([]byte{}, base...), 1, 1, 2)
	if !bytes.Equal(optsOnly, wantOpts) {
		t.Fatalf("opts-only trailer mismatch:\n got =%v\n want=%v", optsOnly, wantOpts)
	}
	if !bytes.Equal(optsOnly, EncodeScrollArgsOpts("acme/docs", f, 50, 1, 2)) {
		t.Fatalf("opts-only differs from legacy EncodeScrollArgsOpts")
	}

	// (c) Cursor present ⇒ [1][rc][opa][1][afterID:u64]; decode round-trips.
	withCursor := EncodeScrollArgsCursorBounded("acme/docs", f, 50, 0, 0, 99, true, 0)
	col, gotF, limit, rc, opa, afterID, hasAfter, _, err := DecodeScrollArgsCursor(withCursor)
	if err != nil {
		t.Fatal(err)
	}
	if col != "acme/docs" || limit != 50 || rc != 0 || opa != 0 || afterID != 99 || !hasAfter || gotF.Field != "doc" {
		t.Fatalf("cursor decode = col=%q limit=%d rc=%d opa=%d after=%d has=%v f=%+v",
			col, limit, rc, opa, afterID, hasAfter, gotF)
	}

	// (d) Legacy DecodeScrollArgs still reads the base from a cursor-carrying arg.
	lcol, _, llim, lerr := DecodeScrollArgs(withCursor)
	if lerr != nil || lcol != "acme/docs" || llim != 50 {
		t.Fatalf("legacy DecodeScrollArgs on cursor arg = col=%q limit=%d err=%v", lcol, llim, lerr)
	}

	// (e) DecodeScrollArgsCursor reads the no-cursor default as hasAfter=false.
	_, _, _, _, _, _, has2, _, err := DecodeScrollArgsCursor(base)
	if err != nil || has2 {
		t.Fatalf("no-cursor default decoded hasAfter=%v err=%v", has2, err)
	}
}

// TestNamedScrollArgsCursorBackwardCompat mirrors the dense check for the named
// scroll arg codec.
func TestNamedScrollArgsCursorBackwardCompat(t *testing.T) {
	f := vector.Filter{Op: vector.FilterEq, Field: "lang", Value: vector.NewString("en")}

	base := EncodeNamedScrollArgs("acme/named", f, 30)
	noCursor := EncodeNamedScrollArgsCursor("acme/named", f, 30, 0, false)
	if !bytes.Equal(base, noCursor) {
		t.Fatalf("named no-cursor bytes differ from legacy:\n base=%v\n new =%v", base, noCursor)
	}

	withCursor := EncodeNamedScrollArgsCursor("acme/named", f, 30, 42, true)
	col, gotF, limit, afterID, hasAfter, err := DecodeNamedScrollArgsCursor(withCursor)
	if err != nil {
		t.Fatal(err)
	}
	if col != "acme/named" || limit != 30 || afterID != 42 || !hasAfter || gotF.Field != "lang" {
		t.Fatalf("named cursor decode = col=%q limit=%d after=%d has=%v f=%+v",
			col, limit, afterID, hasAfter, gotF)
	}
	// Legacy decoder still reads the base.
	lcol, _, llim, lerr := DecodeNamedScrollArgs(withCursor)
	if lerr != nil || lcol != "acme/named" || llim != 30 {
		t.Fatalf("legacy DecodeNamedScrollArgs on cursor arg = col=%q limit=%d err=%v", lcol, llim, lerr)
	}
}

func TestScrollArgsRoundtrip(t *testing.T) {
	f := vector.Filter{Op: vector.FilterEq, Field: "doc", Value: vector.NewInt(2)}
	name, got, limit, err := DecodeScrollArgs(EncodeScrollArgs("acme/docs", f, 50))
	if err != nil {
		t.Fatal(err)
	}
	if name != "acme/docs" || limit != 50 || got.Field != "doc" || got.Value.Int != 2 {
		t.Errorf("roundtrip = %q limit=%d %+v", name, limit, got)
	}
	// Empty filter encodes with no filter block.
	_, zf, _, err := DecodeScrollArgs(EncodeScrollArgs("c", vector.Filter{}, 0))
	if err != nil || !zf.IsZero() {
		t.Errorf("empty filter = %+v (err %v)", zf, err)
	}
}

// TestScrollViaDispatch lists documents with and without a filter through the
// op handler.
func TestScrollViaDispatch(t *testing.T) {
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
	for i := 1; i <= 6; i++ {
		args := EncodeVectorUpsertArgs("docs", uint64(i), []float32{float32(i), 0, 0}, "chunk", 0,
			vector.Metadata{"doc": vector.NewInt(int64(i % 2))}, vector.SparseVector{})
		if _, err := handleVectorUpsert(tx, args); err != nil {
			t.Fatalf("upsert %d: %v", i, err)
		}
	}

	// Scroll all.
	body, err := handleVectorScroll(tx, EncodeScrollArgs("docs", vector.Filter{}, 0))
	if err != nil {
		t.Fatal(err)
	}
	all, err := DecodeVectorDocs(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 6 {
		t.Errorf("scroll all = %d docs, want 6", len(all))
	}
	if all[0].Content != "chunk" {
		t.Errorf("scroll doc missing content: %+v", all[0])
	}

	// Scroll filtered doc==1 (odd ids 1,3,5).
	df := vector.Filter{Op: vector.FilterEq, Field: "doc", Value: vector.NewInt(1)}
	body, _ = handleVectorScroll(tx, EncodeScrollArgs("docs", df, 0))
	odd, _ := DecodeVectorDocs(body)
	if len(odd) != 3 {
		t.Errorf("scroll filtered = %d docs, want 3", len(odd))
	}

	// Limit caps the result.
	body, _ = handleVectorScroll(tx, EncodeScrollArgs("docs", vector.Filter{}, 2))
	capped, _ := DecodeVectorDocs(body)
	if len(capped) != 2 {
		t.Errorf("scroll limit=2 = %d docs, want 2", len(capped))
	}
}

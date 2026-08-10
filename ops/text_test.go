// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// TestSearchTextCodecRoundTrip exercises EncodeSearchTextArgs(Opts) /
// DecodeSearchTextArgs(Opts): the raw query text, k, filter, and the optional
// rc/opa/bound trailer survive a round-trip, and the no-opts form is
// byte-identical (back-compat).
func TestSearchTextCodecRoundTrip(t *testing.T) {
	filter := vector.Filter{Op: vector.FilterEq, Field: "lang", Value: vector.NewString("en")}

	// No-opts form is byte-identical to ...Opts(.., 0, 0, 0).
	base := EncodeSearchTextArgs("docs", "the quick brown fox", 7, filter)
	withZeroOpts := EncodeSearchTextArgsOpts("docs", "the quick brown fox", 7, filter, 0, 0, 0)
	if !bytes.Equal(base, withZeroOpts) {
		t.Fatalf("EncodeSearchTextArgs != ...Opts(0,0,0): %v vs %v", base, withZeroOpts)
	}

	col, q, k, f, err := DecodeSearchTextArgs(base)
	if err != nil {
		t.Fatalf("DecodeSearchTextArgs: %v", err)
	}
	if col != "docs" || q != "the quick brown fox" || k != 7 {
		t.Fatalf("base round-trip: got (%q,%q,%d)", col, q, k)
	}
	if f.IsZero() {
		t.Fatalf("filter lost in round-trip")
	}

	// Opts trailer round-trips rc/opa/bound.
	enc := EncodeSearchTextArgsOpts("docs", "hello world", 3, vector.Filter{}, ConsistencyBoundedStaleness, 1, 42)
	col, q, k, f, rc, opa, bound, err := DecodeSearchTextArgsOpts(enc)
	if err != nil {
		t.Fatalf("DecodeSearchTextArgsOpts: %v", err)
	}
	if col != "docs" || q != "hello world" || k != 3 {
		t.Fatalf("opts round-trip base: got (%q,%q,%d)", col, q, k)
	}
	if !f.IsZero() {
		t.Fatalf("expected zero filter, got %+v", f)
	}
	if rc != ConsistencyBoundedStaleness || opa != 1 || bound != 42 {
		t.Fatalf("opts trailer: rc=%d opa=%d bound=%d want 3,1,42", rc, opa, bound)
	}

	// A legacy (no-opts) blob decodes via ...Opts with rc=opa=bound=0.
	_, _, _, _, rc, opa, bound, err = DecodeSearchTextArgsOpts(base)
	if err != nil || rc != 0 || opa != 0 || bound != 0 {
		t.Fatalf("legacy decode via Opts: rc=%d opa=%d bound=%d err=%v", rc, opa, bound, err)
	}
}

// TestHybridTextCodecRoundTrip exercises EncodeHybridTextArgs(Opts) /
// DecodeHybridTextArgs(Opts): dense vector, raw text, k, fusion opts, filter, and
// the optional rc/opa/bound trailer survive a round-trip.
func TestHybridTextCodecRoundTrip(t *testing.T) {
	dense := []float32{0.1, 0.2, 0.3, 0.4}
	opts := vector.HybridOpts{
		Method:  vector.FusionWeighted,
		Alpha:   0.6,
		RRFK:    77,
		DenseK:  120,
		SparseK: 90,
		Filter:  vector.Filter{Op: vector.FilterEq, Field: "k", Value: vector.NewString("v")},
	}

	base := EncodeHybridTextArgs("docs", dense, "machine learning", 5, opts)
	withZeroOpts := EncodeHybridTextArgsOpts("docs", dense, "machine learning", 5, opts, 0, 0, 0)
	if !bytes.Equal(base, withZeroOpts) {
		t.Fatalf("EncodeHybridTextArgs != ...Opts(0,0,0)")
	}

	col, gotDense, q, k, gotOpts, err := DecodeHybridTextArgs(base)
	if err != nil {
		t.Fatalf("DecodeHybridTextArgs: %v", err)
	}
	if col != "docs" || q != "machine learning" || k != 5 {
		t.Fatalf("base round-trip: got (%q,%q,%d)", col, q, k)
	}
	if len(gotDense) != 4 || gotDense[0] != 0.1 || gotDense[3] != 0.4 {
		t.Fatalf("dense lost: %v", gotDense)
	}
	if gotOpts.Method != vector.FusionWeighted || gotOpts.Alpha != 0.6 ||
		gotOpts.RRFK != 77 || gotOpts.DenseK != 120 || gotOpts.SparseK != 90 {
		t.Fatalf("opts lost: %+v", gotOpts)
	}
	if gotOpts.Filter.IsZero() {
		t.Fatalf("filter lost")
	}

	enc := EncodeHybridTextArgsOpts("docs", dense, "q", 2, vector.HybridOpts{}, ConsistencyLeaderOnly, 1, 0)
	_, _, _, _, _, rc, opa, _, err := DecodeHybridTextArgsOpts(enc)
	if err != nil {
		t.Fatalf("DecodeHybridTextArgsOpts: %v", err)
	}
	if rc != ConsistencyLeaderOnly || opa != 1 {
		t.Fatalf("opts trailer: rc=%d opa=%d want 1,1", rc, opa)
	}
}

// TestTextCollectionNameForAt2 is the routing-offset regression guard (the
// named-hybrid bug class): the full-text ops emit the At2 wire
// ([flags:u8][colLen:u8][col]...), so CollectionNameFor MUST read the name at
// offset 1, behind the flags byte. TEETH: a non-empty filter forces a NON-ZERO
// flags byte (textFlagFilter) — under a buggy At1 mapping the flags byte at
// offset 0 would be misread as the name length, yielding a garbage name and (on a
// P>1 collection) a silent single-partition fallback.
func TestTextCollectionNameForAt2(t *testing.T) {
	dense := []float32{1, 2, 3, 4}
	// A non-empty filter sets textFlagFilter, guaranteeing a non-zero flags byte —
	// exactly what an At1 misread would mistake for the name length.
	filter := vector.Filter{Op: vector.FilterEq, Field: "tag", Value: vector.NewString("x")}

	cases := []struct {
		op   string
		args []byte
	}{
		{"vector_search_text", EncodeSearchTextArgs("mycoll", "some query", 5, filter)},
		{"vector_hybrid_text", EncodeHybridTextArgs("mycoll", dense, "some query", 5, vector.HybridOpts{Filter: filter})},
		{"vector_hybrid_text_lanes", EncodeHybridTextArgs("mycoll", dense, "some query", 5, vector.HybridOpts{Filter: filter})},
	}
	for _, c := range cases {
		// The flags byte MUST be non-zero for the teeth to bite.
		if c.args[0] == 0 {
			t.Fatalf("%s: flags byte is zero — test has no teeth", c.op)
		}
		name, ok := CollectionNameFor(c.op, c.args)
		if !ok {
			t.Errorf("%s: CollectionNameFor !ok", c.op)
			continue
		}
		if name != "default/mycoll" {
			t.Errorf("%s: name = %q, want %q (At1/At2 routing-offset regression)", c.op, name, "default/mycoll")
		}
		// RewriteCollectionName (alias path) must splice at the same offset.
		out, ok := RewriteCollectionName(c.op, c.args, "default/other")
		if !ok {
			t.Errorf("%s: RewriteCollectionName !ok", c.op)
			continue
		}
		if got, _ := CollectionNameFor(c.op, out); got != "default/other" {
			t.Errorf("%s: after rewrite name = %q, want default/other", c.op, got)
		}
	}
}

// TestTextReadConsistencyOf confirms ReadConsistencyOf peeks the rc byte for the
// full-text ops (so a Linearizable text read arms the shard barrier).
func TestTextReadConsistencyOf(t *testing.T) {
	st := EncodeSearchTextArgsOpts("docs", "q", 5, vector.Filter{}, ConsistencyLinearizable, 0, 0)
	if rc, ok := ReadConsistencyOf("vector_search_text", st); !ok || rc != ConsistencyLinearizable {
		t.Fatalf("vector_search_text rc=%d ok=%v want 2,true", rc, ok)
	}
	ht := EncodeHybridTextArgsOpts("docs", []float32{1, 2}, "q", 5, vector.HybridOpts{}, ConsistencyLinearizable, 0, 0)
	for _, op := range []string{"vector_hybrid_text", "vector_hybrid_text_lanes"} {
		if rc, ok := ReadConsistencyOf(op, ht); !ok || rc != ConsistencyLinearizable {
			t.Fatalf("%s rc=%d ok=%v want 2,true", op, rc, ok)
		}
	}
}

// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"errors"
	"reflect"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// TestBM25StatsArgsRoundTrip exercises EncodeBM25StatsArgs / DecodeBM25StatsArgs:
// collection + query + the optional rc/opa/bound trailer survive a round-trip, and
// the no-opts form is byte-identical to the bare [col][query] body.
func TestBM25StatsArgsRoundTrip(t *testing.T) {
	base := EncodeBM25StatsArgs("docs", "the quick brown fox", 0, 0, 0)
	// No-opts ⇒ exactly [colLen][col][qLen][query], nothing more.
	want := append([]byte{byte(len("docs"))}, "docs"...)
	want = append(want, 0, 0, 0, byte(len("the quick brown fox")))
	want = append(want, "the quick brown fox"...)
	if !bytes.Equal(base, want) {
		t.Fatalf("no-opts body not byte-identical to bare form:\n got %v\nwant %v", base, want)
	}

	col, q, rc, opa, bound, err := DecodeBM25StatsArgs(base)
	if err != nil {
		t.Fatalf("DecodeBM25StatsArgs: %v", err)
	}
	if col != "docs" || q != "the quick brown fox" || rc != 0 || opa != 0 || bound != 0 {
		t.Fatalf("base round-trip: got (%q,%q,%d,%d,%d)", col, q, rc, opa, bound)
	}

	enc := EncodeBM25StatsArgs("docs", "hello", ConsistencyBoundedStaleness, 1, 99)
	col, q, rc, opa, bound, err = DecodeBM25StatsArgs(enc)
	if err != nil {
		t.Fatalf("DecodeBM25StatsArgs opts: %v", err)
	}
	if col != "docs" || q != "hello" || rc != ConsistencyBoundedStaleness || opa != 1 || bound != 99 {
		t.Fatalf("opts round-trip: got (%q,%q,%d,%d,%d) want docs,hello,3,1,99", col, q, rc, opa, bound)
	}

	// Truncation is fail-loud.
	if _, _, _, _, _, err := DecodeBM25StatsArgs(base[:2]); !errors.Is(err, ErrVectorArgsTruncated) {
		t.Fatalf("truncated args: err=%v want ErrVectorArgsTruncated", err)
	}
}

// TestBM25StatsResultRoundTrip exercises EncodeBM25StatsResult / DecodeBM25StatsResult.
func TestBM25StatsResultRoundTrip(t *testing.T) {
	df := map[uint32]int{10: 3, 20: 0, 30: 7}
	enc := EncodeBM25StatsResult(42, 1000, df)
	n, tok, gotDF, err := DecodeBM25StatsResult(enc)
	if err != nil {
		t.Fatalf("DecodeBM25StatsResult: %v", err)
	}
	if n != 42 || tok != 1000 {
		t.Fatalf("n=%d tok=%d want 42,1000", n, tok)
	}
	if !reflect.DeepEqual(gotDF, df) {
		t.Fatalf("df=%v want %v", gotDF, df)
	}

	// Empty df.
	enc = EncodeBM25StatsResult(0, 0, nil)
	n, tok, gotDF, err = DecodeBM25StatsResult(enc)
	if err != nil || n != 0 || tok != 0 || len(gotDF) != 0 {
		t.Fatalf("empty: n=%d tok=%d df=%v err=%v", n, tok, gotDF, err)
	}

	if _, _, _, err := DecodeBM25StatsResult(enc[:10]); !errors.Is(err, ErrVectorArgsTruncated) {
		t.Fatalf("truncated result: err=%v want ErrVectorArgsTruncated", err)
	}
}

// TestBM25StatsRouting is the routing-offset guard for vector_bm25_stats: its wire
// is At1 ([colLen:u8][col]... — NO flags byte), so CollectionNameFor reads the name
// at offset 0. TEETH: an opts trailer makes the body non-trivial, but the name must
// still resolve, and a P>1 collection would otherwise silently land on one shard.
func TestBM25StatsRouting(t *testing.T) {
	args := EncodeBM25StatsArgs("mycoll", "some query", ConsistencyLinearizable, 0, 0)
	// The FIRST byte must be the colLen (At1), NOT a flags byte: it equals len("mycoll").
	if args[0] != byte(len("mycoll")) {
		t.Fatalf("first byte = %d, want colLen %d (At1 layout broken)", args[0], len("mycoll"))
	}
	name, ok := CollectionNameFor("vector_bm25_stats", args)
	if !ok || name != "default/mycoll" {
		t.Fatalf("CollectionNameFor = (%q,%v) want default/mycoll,true", name, ok)
	}
	off, ok := CollectionNameOffset("vector_bm25_stats")
	if !ok || off != 0 {
		t.Fatalf("CollectionNameOffset = (%d,%v) want 0,true", off, ok)
	}
	// ReadConsistencyOf must peek the rc for the phase-0 Linearizable barrier.
	if rc, ok := ReadConsistencyOf("vector_bm25_stats", args); !ok || rc != ConsistencyLinearizable {
		t.Fatalf("ReadConsistencyOf = (%d,%v) want 2,true", rc, ok)
	}
}

// TestSearchTextGlobalTrailer proves: (1) the global-DF extensions are byte-
// identical to the pre-global form when absent; (2) the request flag + phase-1 stats
// block round-trip; (3) ReadConsistencyOf STILL reads the right rc when the global
// block is present (the block rides AFTER the rc trailer, so the peek is unaffected).
func TestSearchTextGlobalTrailer(t *testing.T) {
	filter := vector.Filter{Op: vector.FilterEq, Field: "lang", Value: vector.NewString("en")}

	// (1) Byte-identity when globalIDF=false and g=nil.
	old := EncodeSearchTextArgsOpts("docs", "quick fox", 7, filter, ConsistencyLinearizable, 1, 0)
	neu := EncodeSearchTextArgsGlobal("docs", "quick fox", 7, filter, ConsistencyLinearizable, 1, 0, false, nil)
	if !bytes.Equal(old, neu) {
		t.Fatalf("global encoder not byte-identical when absent:\n old %v\n new %v", old, neu)
	}
	// A pre-extension decoder (DecodeSearchTextArgsOpts) still decodes the body.
	col, q, k, f, rc, opa, _, err := DecodeSearchTextArgsOpts(neu)
	if err != nil || col != "docs" || q != "quick fox" || k != 7 || f.IsZero() || rc != ConsistencyLinearizable || opa != 1 {
		t.Fatalf("pre-ext decode of absent-global body: (%q,%q,%d,%v,%d,%d) err=%v", col, q, k, f.IsZero(), rc, opa, err)
	}

	// (2) Request flag round-trips with NO stats block (byte-cheap: only a flag bit).
	reqOnly := EncodeSearchTextArgsGlobal("docs", "q", 5, vector.Filter{}, 0, 0, 0, true, nil)
	_, _, _, _, _, _, _, gIDF, g, err := DecodeSearchTextArgsGlobal(reqOnly)
	if err != nil || !gIDF || g != nil {
		t.Fatalf("request-flag decode: globalIDF=%v g=%v err=%v want true,nil,nil", gIDF, g, err)
	}

	// (3) Phase-1 stats block round-trips AND coexists with the rc trailer.
	stats := vector.BM25GlobalStats{N: 100, Avgdl: 12.5, DF: map[uint32]int{7: 11, 99: 3}}
	enc := EncodeSearchTextArgsGlobal("docs", "quick fox", 7, filter, ConsistencyLinearizable, 1, 0, false, &stats)
	col, q, k, f, rc, opa, _, gIDF, g, err = DecodeSearchTextArgsGlobal(enc)
	if err != nil {
		t.Fatalf("global decode: %v", err)
	}
	if col != "docs" || q != "quick fox" || k != 7 || f.IsZero() {
		t.Fatalf("global base lost: (%q,%q,%d,%v)", col, q, k, f.IsZero())
	}
	if rc != ConsistencyLinearizable || opa != 1 {
		t.Fatalf("rc/opa lost behind global block: rc=%d opa=%d", rc, opa)
	}
	if gIDF {
		t.Fatalf("globalIDF flag unexpectedly set on a stats-only encode")
	}
	if g == nil || g.N != 100 || g.Avgdl != 12.5 || !reflect.DeepEqual(g.DF, stats.DF) {
		t.Fatalf("stats lost: %+v", g)
	}

	// ReadConsistencyOf STILL works with the global block present.
	if got, ok := ReadConsistencyOf("vector_search_text", enc); !ok || got != ConsistencyLinearizable {
		t.Fatalf("ReadConsistencyOf with global block = (%d,%v) want 2,true", got, ok)
	}
}

// TestHybridTextGlobalTrailer mirrors TestSearchTextGlobalTrailer for the hybrid-
// text wire: byte-identity when absent, request-flag + stats round-trip, and
// ReadConsistencyOf survives the global block.
func TestHybridTextGlobalTrailer(t *testing.T) {
	dense := []float32{0.1, 0.2, 0.3}
	hopts := vector.HybridOpts{Method: vector.FusionWeighted, Alpha: 0.6, DenseK: 80, SparseK: 90}

	old := EncodeHybridTextArgsOpts("docs", dense, "ml", 5, hopts, ConsistencyLinearizable, 0, 0)
	neu := EncodeHybridTextArgsGlobal("docs", dense, "ml", 5, hopts, ConsistencyLinearizable, 0, 0, false, nil)
	if !bytes.Equal(old, neu) {
		t.Fatalf("hybrid global encoder not byte-identical when absent")
	}

	stats := vector.BM25GlobalStats{N: 50, Avgdl: 8.0, DF: map[uint32]int{3: 9}}
	enc := EncodeHybridTextArgsGlobal("docs", dense, "ml", 5, hopts, ConsistencyLinearizable, 0, 0, true, &stats)
	col, gotDense, q, k, gotOpts, rc, _, _, gIDF, g, err := DecodeHybridTextArgsGlobal(enc)
	if err != nil {
		t.Fatalf("hybrid global decode: %v", err)
	}
	if col != "docs" || q != "ml" || k != 5 || len(gotDense) != 3 || gotOpts.DenseK != 80 {
		t.Fatalf("hybrid base lost: (%q,%q,%d,len=%d,denseK=%d)", col, q, k, len(gotDense), gotOpts.DenseK)
	}
	if rc != ConsistencyLinearizable {
		t.Fatalf("rc lost behind global block: %d", rc)
	}
	if !gIDF {
		t.Fatalf("request flag lost")
	}
	if g == nil || g.N != 50 || g.Avgdl != 8.0 || !reflect.DeepEqual(g.DF, stats.DF) {
		t.Fatalf("hybrid stats lost: %+v", g)
	}
	for _, op := range []string{"vector_hybrid_text", "vector_hybrid_text_lanes"} {
		if got, ok := ReadConsistencyOf(op, enc); !ok || got != ConsistencyLinearizable {
			t.Fatalf("%s ReadConsistencyOf with global block = (%d,%v) want 2,true", op, got, ok)
		}
	}
}

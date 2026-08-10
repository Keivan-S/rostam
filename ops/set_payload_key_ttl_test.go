// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/vector"
)

// TestSetPayloadArgsKeyTTLRoundtrip exercises the optional per-key-TTL block in
// the shared set/overwrite-payload codec: with a map it round-trips; without one
// the bytes are BYTE-IDENTICAL to the legacy encoder so old peers interop.
func TestSetPayloadArgsKeyTTLRoundtrip(t *testing.T) {
	meta := vector.Metadata{"a": vector.NewInt(1), "b": vector.NewString("x")}
	ttl := map[string]int64{"a": 1000, "b": 250}

	// With a key-ttl map: Opts decode returns it verbatim (relative ms).
	enc := EncodeSetPayloadArgsOpts("acme/docs", 42, meta, ttl)
	col, id, gotMeta, gotTTL, err := DecodeSetPayloadArgsOpts(enc)
	if err != nil {
		t.Fatalf("decode opts: %v", err)
	}
	if col != "acme/docs" || id != 42 {
		t.Fatalf("decoded (%q,%d), want (acme/docs,42)", col, id)
	}
	if gotMeta["a"].Int != 1 || gotMeta["b"].Str != "x" {
		t.Fatalf("decoded meta = %+v", gotMeta)
	}
	if len(gotTTL) != 2 || gotTTL["a"] != 1000 || gotTTL["b"] != 250 {
		t.Fatalf("decoded key_ttl_ms = %+v, want {a:1000,b:250}", gotTTL)
	}

	// The legacy 4-tuple decoder still reads it (it discards the trailing block).
	_, _, lMeta, lErr := DecodeSetPayloadArgs(enc)
	if lErr != nil || lMeta["a"].Int != 1 {
		t.Fatalf("legacy decode of opts-encoded: meta=%+v err=%v", lMeta, lErr)
	}
}

// TestSetPayloadArgsByteIdenticalWhenAbsent proves the wire is unchanged when no
// per-key TTL is supplied: EncodeSetPayloadArgsOpts(..., nil/empty) == the legacy
// EncodeSetPayloadArgs for the same inputs, and the round-trip yields a nil map.
func TestSetPayloadArgsByteIdenticalWhenAbsent(t *testing.T) {
	cases := []struct {
		name string
		meta vector.Metadata
	}{
		{"empty", nil},
		{"nonempty", vector.Metadata{"k": vector.NewInt(7), "z": vector.NewString("q")}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			legacy := EncodeSetPayloadArgs("docs", 9, tc.meta)
			optNil := EncodeSetPayloadArgsOpts("docs", 9, tc.meta, nil)
			optEmpty := EncodeSetPayloadArgsOpts("docs", 9, tc.meta, map[string]int64{})
			if !bytes.Equal(legacy, optNil) {
				t.Fatalf("nil map not byte-identical:\n legacy=%v\n opts  =%v", legacy, optNil)
			}
			if !bytes.Equal(legacy, optEmpty) {
				t.Fatalf("empty map not byte-identical:\n legacy=%v\n opts  =%v", legacy, optEmpty)
			}
			// An absent block decodes to a nil key-ttl map.
			_, _, _, ttl, err := DecodeSetPayloadArgsOpts(legacy)
			if err != nil {
				t.Fatalf("decode legacy: %v", err)
			}
			if ttl != nil {
				t.Fatalf("key_ttl_ms = %+v, want nil when absent", ttl)
			}
		})
	}
}

// TestSetPayloadArgsKeyTTLMalformed asserts a torn / corrupt per-key-TTL block is
// fail-loud (never silently dropped).
func TestSetPayloadArgsKeyTTLMalformed(t *testing.T) {
	good := EncodeSetPayloadArgsOpts("docs", 1, vector.Metadata{"a": vector.NewInt(1)},
		map[string]int64{"a": 5})

	// Truncate inside the trailing key-ttl JSON.
	if _, _, _, _, err := DecodeSetPayloadArgsOpts(good[:len(good)-2]); err == nil {
		t.Error("truncated key-ttl block: want error, got nil")
	}

	// present=1 but no length word: hand-craft [legacy meta][present=1] only.
	base := EncodeSetPayloadArgs("docs", 1, vector.Metadata{"a": vector.NewInt(1)})
	torn := append(append([]byte{}, base...), 1) // present flag, no length
	if _, _, _, _, err := DecodeSetPayloadArgsOpts(torn); err == nil {
		t.Error("present=1 with no length: want error, got nil")
	}

	// present=1, length declares 4 bytes but supplies invalid JSON.
	bad := append(append([]byte{}, base...), 1, 0, 0, 0, 2, '{', '{')
	if _, _, _, _, err := DecodeSetPayloadArgsOpts(bad); err == nil {
		t.Error("invalid key-ttl JSON: want error, got nil")
	}
}

// TestHandleVectorSetPayloadAppliesKeyTTL drives the dense set/overwrite handlers
// with a per-key TTL through the codec and confirms the engine drops the TTL'd key
// after its (short, relative) deadline while permanent keys + the point survive.
func TestHandleVectorSetPayloadAppliesKeyTTL(t *testing.T) {
	tx, _ := newGetPayloadTx(t)
	cfg := vector.Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("docs", cfg)); err != nil {
		t.Fatal(err)
	}
	if _, err := handleVectorInsert(tx, EncodeVectorInsertArgsExt("docs", 1, []float32{1, 0, 0, 0}, 0,
		vector.Metadata{"keep": vector.NewInt(1)}, vector.SparseVector{})); err != nil {
		t.Fatal(err)
	}

	// set_payload merges temp (1ms TTL) + perm (no TTL) via the wire codec.
	args := EncodeSetPayloadArgsOpts("docs", 1,
		vector.Metadata{"temp": vector.NewInt(2), "perm": vector.NewInt(3)},
		map[string]int64{"temp": 1})
	if res, err := handleVectorSetPayload(tx, args); err != nil {
		t.Fatal(err)
	} else if ok, _ := DecodePayloadResult(res); !ok {
		t.Fatal("set payload: applied=false, want true")
	}

	// Past the 1ms deadline: temp drops; keep/perm + the point survive.
	time.Sleep(15 * time.Millisecond)
	body, _ := handleVectorGet(tx, EncodeVectorGetArgs("docs", 1, GetFlagsBoth))
	found, _, meta, _, _, err := DecodeVectorGetResult(body)
	if err != nil || !found {
		t.Fatalf("get after expiry: found=%v err=%v", found, err)
	}
	if _, ok := meta["temp"]; ok {
		t.Errorf("expired key temp still present: %+v", meta)
	}
	if meta["keep"].Int != 1 || meta["perm"].Int != 3 {
		t.Errorf("non-TTL keys dropped: meta=%+v, want keep=1,perm=3", meta)
	}

	// overwrite_payload with a per-key TTL replaces the deadline set; the new TTL'd
	// key drops, a fresh permanent key survives.
	ow := EncodeSetPayloadArgsOpts("docs", 1,
		vector.Metadata{"t2": vector.NewInt(9), "p2": vector.NewInt(8)},
		map[string]int64{"t2": 1})
	if _, err := handleVectorOverwritePayload(tx, ow); err != nil {
		t.Fatal(err)
	}
	time.Sleep(15 * time.Millisecond)
	body, _ = handleVectorGet(tx, EncodeVectorGetArgs("docs", 1, GetFlagsBoth))
	_, _, meta, _, _, _ = DecodeVectorGetResult(body)
	if _, ok := meta["t2"]; ok {
		t.Errorf("overwrite: expired key t2 still present: %+v", meta)
	}
	if meta["p2"].Int != 8 {
		t.Errorf("overwrite: permanent key p2 missing: %+v", meta)
	}
	if _, ok := meta["keep"]; ok {
		t.Errorf("overwrite did not replace prior payload: %+v", meta)
	}
}

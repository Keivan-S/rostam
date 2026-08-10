// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"testing"
	"time"

	"github.com/rostamlabs/rostam/vector"
)

// TestHandleNamedSetPayloadAppliesKeyTTL drives the named set/overwrite handlers
// with a per-key TTL through the SHARED set_payload codec and confirms the engine
// drops the TTL'd key after its (short, relative) deadline while permanent keys +
// the point survive. The named family threads key_ttl_ms via the same
// DecodeSetPayloadArgsOpts the dense path uses (no separate codec).
func TestHandleNamedSetPayloadAppliesKeyTTL(t *testing.T) {
	tx, _ := newGetPayloadTx(t)
	cfg := map[string]vector.NamedVectorParams{"title": {Dim: 2, Metric: vector.Cosine}}
	if _, err := handleNamedCreate(tx, EncodeNamedCreateArgs("named", cfg, 0)); err != nil {
		t.Fatal(err)
	}
	if _, err := handleNamedInsert(tx, EncodeNamedInsertArgs("named", 1,
		map[string][]float32{"title": {1, 0}}, vector.Metadata{"keep": vector.NewInt(1)}, 0)); err != nil {
		t.Fatal(err)
	}

	// set_payload merges temp (1ms TTL) + perm (no TTL) via the wire codec.
	args := EncodeSetPayloadArgsOpts("named", 1,
		vector.Metadata{"temp": vector.NewInt(2), "perm": vector.NewInt(3)},
		map[string]int64{"temp": 1})
	if res, err := handleNamedSetPayload(tx, args); err != nil {
		t.Fatal(err)
	} else if ok, _ := DecodePayloadResult(res); !ok {
		t.Fatal("named set payload: applied=false, want true")
	}

	time.Sleep(15 * time.Millisecond)
	body, _ := handleNamedGet(tx, EncodeVectorGetArgs("named", 1, GetFlagsBoth))
	found, _, meta, _, err := DecodeNamedGetResult(body)
	if err != nil || !found {
		t.Fatalf("named get after expiry: found=%v err=%v", found, err)
	}
	if _, ok := meta["temp"]; ok {
		t.Errorf("expired key temp still present: %+v", meta)
	}
	if meta["keep"].Int != 1 || meta["perm"].Int != 3 {
		t.Errorf("non-TTL keys dropped: meta=%+v, want keep=1,perm=3", meta)
	}

	// overwrite_payload with a per-key TTL replaces the deadline set.
	ow := EncodeSetPayloadArgsOpts("named", 1,
		vector.Metadata{"t2": vector.NewInt(9), "p2": vector.NewInt(8)},
		map[string]int64{"t2": 1})
	if _, err := handleNamedOverwritePayload(tx, ow); err != nil {
		t.Fatal(err)
	}
	time.Sleep(15 * time.Millisecond)
	body, _ = handleNamedGet(tx, EncodeVectorGetArgs("named", 1, GetFlagsBoth))
	_, _, meta, _, _ = DecodeNamedGetResult(body)
	if _, ok := meta["t2"]; ok {
		t.Errorf("named overwrite: expired key t2 still present: %+v", meta)
	}
	if meta["p2"].Int != 8 {
		t.Errorf("named overwrite: permanent key p2 missing: %+v", meta)
	}
	if _, ok := meta["keep"]; ok {
		t.Errorf("named overwrite did not replace prior payload: %+v", meta)
	}
}

// TestHandleMVSetPayloadAppliesKeyTTL mirrors the named handler test for the MV
// family (also through the shared set_payload codec).
func TestHandleMVSetPayloadAppliesKeyTTL(t *testing.T) {
	tx, _ := newGetPayloadTx(t)
	if _, err := handleMVCreate(tx, EncodeMVCreateArgs("mv", vector.MultiVectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1})); err != nil {
		t.Fatal(err)
	}
	if _, err := handleMVAdd(tx, EncodeMVAddArgs("mv", 1, [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}},
		vector.Metadata{"keep": vector.NewInt(1)})); err != nil {
		t.Fatal(err)
	}

	args := EncodeSetPayloadArgsOpts("mv", 1,
		vector.Metadata{"temp": vector.NewInt(2), "perm": vector.NewInt(3)},
		map[string]int64{"temp": 1})
	if res, err := handleMVSetPayload(tx, args); err != nil {
		t.Fatal(err)
	} else if ok, _ := DecodePayloadResult(res); !ok {
		t.Fatal("mv set payload: applied=false, want true")
	}

	time.Sleep(15 * time.Millisecond)
	body, _ := handleMVGet(tx, EncodeVectorGetArgs("mv", 1, GetFlagsBoth))
	found, _, meta, err := DecodeMVGetResult(body)
	if err != nil || !found {
		t.Fatalf("mv get after expiry: found=%v err=%v", found, err)
	}
	if _, ok := meta["temp"]; ok {
		t.Errorf("expired key temp still present: %+v", meta)
	}
	if meta["keep"].Int != 1 || meta["perm"].Int != 3 {
		t.Errorf("non-TTL keys dropped: meta=%+v, want keep=1,perm=3", meta)
	}

	ow := EncodeSetPayloadArgsOpts("mv", 1,
		vector.Metadata{"t2": vector.NewInt(9), "p2": vector.NewInt(8)},
		map[string]int64{"t2": 1})
	if _, err := handleMVOverwritePayload(tx, ow); err != nil {
		t.Fatal(err)
	}
	time.Sleep(15 * time.Millisecond)
	body, _ = handleMVGet(tx, EncodeVectorGetArgs("mv", 1, GetFlagsBoth))
	_, _, meta, _ = DecodeMVGetResult(body)
	if _, ok := meta["t2"]; ok {
		t.Errorf("mv overwrite: expired key t2 still present: %+v", meta)
	}
	if meta["p2"].Int != 8 {
		t.Errorf("mv overwrite: permanent key p2 missing: %+v", meta)
	}
	if _, ok := meta["keep"]; ok {
		t.Errorf("mv overwrite did not replace prior payload: %+v", meta)
	}
}

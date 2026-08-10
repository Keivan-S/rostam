// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/vector"
)

// Wire codecs: named insert / MV add carry an OPTIONAL per-key payload TTL
// block (key -> RELATIVE ms) that interposes between the base block and the CAS
// trailer. It is BYTE-IDENTICAL to the legacy wire when the map is empty AND no CAS
// follows (the common no-key_ttl path), round-trips when present, and coexists with
// the CAS trailer at a deterministic offset (the set_payload-CAS interpose).

func TestNamedInsertArgsKeyTTLByteIdenticalWhenAbsent(t *testing.T) {
	vecs := map[string][]float32{"a": {1, 2, 3}}
	meta := vector.Metadata{"k": vector.NewInt(1)}
	base := EncodeNamedInsertArgs("c", 7, vecs, meta, time.Second)

	// nil key TTL, no CAS → byte-identical to the legacy base wire.
	if got := EncodeNamedInsertArgsKeyTTL("c", 7, vecs, meta, time.Second, nil); !bytes.Equal(base, got) {
		t.Fatalf("named insert keyTTL(nil) not byte-identical to legacy:\n base=%x\n got =%x", base, got)
	}
	// empty (non-nil) map, no CAS → also byte-identical.
	if got := EncodeNamedInsertArgsKeyTTL("c", 7, vecs, meta, time.Second, map[string]int64{}); !bytes.Equal(base, got) {
		t.Fatalf("named insert keyTTL(empty) not byte-identical to legacy")
	}
	// no key TTL + no CAS via the CAS-KeyTTL encoder → byte-identical.
	if got := EncodeNamedInsertArgsCASKeyTTL("c", 7, vecs, meta, time.Second, 0, false, nil); !bytes.Equal(base, got) {
		t.Fatalf("named insert CASKeyTTL(no-cas, no-ttl) not byte-identical to legacy")
	}
}

func TestNamedInsertArgsKeyTTLRoundTrip(t *testing.T) {
	vecs := map[string][]float32{"a": {1, 2, 3}}
	meta := vector.Metadata{"k": vector.NewInt(1)}
	ttlMap := map[string]int64{"k": 1234}

	// keyTTL only.
	args := EncodeNamedInsertArgsKeyTTL("c", 7, vecs, meta, time.Second, ttlMap)
	_, id, _, _, _, exp, has, kt, err := DecodeNamedInsertArgsKeyTTL(args)
	if err != nil || id != 7 || has || exp != 0 {
		t.Fatalf("decode keyTTL-only: id=%d exp=%d has=%v err=%v", id, exp, has, err)
	}
	if kt["k"] != 1234 || len(kt) != 1 {
		t.Fatalf("decode keyTTL-only map = %v, want {k:1234}", kt)
	}

	// keyTTL + CAS coexist: both decode at deterministic offsets.
	both := EncodeNamedInsertArgsCASKeyTTL("c", 7, vecs, meta, time.Second, 5, true, ttlMap)
	_, _, _, _, _, exp2, has2, kt2, err2 := DecodeNamedInsertArgsKeyTTL(both)
	if err2 != nil || !has2 || exp2 != 5 || kt2["k"] != 1234 {
		t.Fatalf("decode keyTTL+CAS: exp=%d has=%v kt=%v err=%v", exp2, has2, kt2, err2)
	}
	// CAS-only (no keyTTL) via the new encoder still decodes the CAS guard.
	casOnly := EncodeNamedInsertArgsCASKeyTTL("c", 7, vecs, meta, time.Second, 9, true, nil)
	_, _, _, _, _, exp3, has3, kt3, err3 := DecodeNamedInsertArgsKeyTTL(casOnly)
	if err3 != nil || !has3 || exp3 != 9 || kt3 != nil {
		t.Fatalf("decode CAS-only: exp=%d has=%v kt=%v err=%v", exp3, has3, kt3, err3)
	}
	// The legacy DecodeNamedInsertArgsCAS view still works on a keyTTL+CAS blob.
	_, _, _, _, _, cexp, chas, cerr := DecodeNamedInsertArgsCAS(both)
	if cerr != nil || !chas || cexp != 5 {
		t.Fatalf("CAS-only decode of keyTTL+CAS blob: exp=%d has=%v err=%v", cexp, chas, cerr)
	}
}

func TestMVAddArgsKeyTTLByteIdenticalWhenAbsent(t *testing.T) {
	toks := [][]float32{{1, 0}, {0, 1}}
	meta := vector.Metadata{"k": vector.NewInt(1)}
	base := EncodeMVAddArgs("c", 7, toks, meta)

	if got := EncodeMVAddArgsKeyTTL("c", 7, toks, meta, nil); !bytes.Equal(base, got) {
		t.Fatalf("mv add keyTTL(nil) not byte-identical to legacy:\n base=%x\n got =%x", base, got)
	}
	if got := EncodeMVAddArgsKeyTTL("c", 7, toks, meta, map[string]int64{}); !bytes.Equal(base, got) {
		t.Fatalf("mv add keyTTL(empty) not byte-identical to legacy")
	}
	if got := EncodeMVAddArgsCASKeyTTL("c", 7, toks, meta, 0, false, nil); !bytes.Equal(base, got) {
		t.Fatalf("mv add CASKeyTTL(no-cas, no-ttl) not byte-identical to legacy")
	}
}

func TestMVAddArgsKeyTTLRoundTrip(t *testing.T) {
	toks := [][]float32{{1, 0}, {0, 1}}
	meta := vector.Metadata{"k": vector.NewInt(1)}
	ttlMap := map[string]int64{"k": 4321}

	args := EncodeMVAddArgsKeyTTL("c", 7, toks, meta, ttlMap)
	_, id, _, _, exp, has, kt, err := DecodeMVAddArgsCASKeyTTL(args)
	if err != nil || id != 7 || has || exp != 0 || kt["k"] != 4321 {
		t.Fatalf("decode keyTTL-only: id=%d exp=%d has=%v kt=%v err=%v", id, exp, has, kt, err)
	}

	both := EncodeMVAddArgsCASKeyTTL("c", 7, toks, meta, 3, true, ttlMap)
	_, _, _, _, exp2, has2, kt2, err2 := DecodeMVAddArgsCASKeyTTL(both)
	if err2 != nil || !has2 || exp2 != 3 || kt2["k"] != 4321 {
		t.Fatalf("decode keyTTL+CAS: exp=%d has=%v kt=%v err=%v", exp2, has2, kt2, err2)
	}
	// The legacy DecodeMVAddArgsCAS view still reads the CAS guard on a keyTTL+CAS blob.
	_, _, _, _, cexp, chas, cerr := DecodeMVAddArgsCAS(both)
	if cerr != nil || !chas || cexp != 3 {
		t.Fatalf("CAS-only decode of keyTTL+CAS blob: exp=%d has=%v err=%v", cexp, chas, cerr)
	}
}

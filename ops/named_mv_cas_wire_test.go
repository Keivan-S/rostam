// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/vector"
)

// Wire codecs: named/MV insert/add + delete carry an OPTIONAL expected_version
// CAS trailer (byte-identical when absent), and the named/MV Get result carries an
// OPTIONAL trailing version block (byte-identical when 0). These guard the additive
// backward-compatibility invariant the dense Task-2 codecs hold.

func TestNamedInsertArgsCASByteIdenticalDefault(t *testing.T) {
	vecs := map[string][]float32{"a": {1, 2, 3}}
	meta := vector.Metadata{"k": vector.NewInt(1)}
	base := EncodeNamedInsertArgs("c", 7, vecs, meta, time.Second)
	noCAS := EncodeNamedInsertArgsCAS("c", 7, vecs, meta, time.Second, 0, false)
	if !bytes.Equal(base, noCAS) {
		t.Fatalf("named insert CAS (hasExpected=false) is NOT byte-identical to legacy")
	}
	// With CAS: decodes back the expected version.
	withCAS := EncodeNamedInsertArgsCAS("c", 7, vecs, meta, time.Second, 5, true)
	_, id, _, _, _, exp, has, err := DecodeNamedInsertArgsCAS(withCAS)
	if err != nil || id != 7 || exp != 5 || !has {
		t.Fatalf("decode named insert CAS: id=%d exp=%d has=%v err=%v", id, exp, has, err)
	}
	// Legacy blob decodes to hasExpected=false.
	if _, _, _, _, _, _, has2, _ := DecodeNamedInsertArgsCAS(base); has2 {
		t.Fatalf("legacy named insert decoded hasExpected=true")
	}
}

func TestMVAddArgsCASByteIdenticalDefault(t *testing.T) {
	toks := [][]float32{{1, 0}, {0, 1}}
	base := EncodeMVAddArgs("c", 7, toks, nil)
	noCAS := EncodeMVAddArgsCAS("c", 7, toks, nil, 0, false)
	if !bytes.Equal(base, noCAS) {
		t.Fatalf("mv add CAS (hasExpected=false) is NOT byte-identical to legacy")
	}
	withCAS := EncodeMVAddArgsCAS("c", 7, toks, nil, 3, true)
	_, id, _, _, exp, has, err := DecodeMVAddArgsCAS(withCAS)
	if err != nil || id != 7 || exp != 3 || !has {
		t.Fatalf("decode mv add CAS: id=%d exp=%d has=%v err=%v", id, exp, has, err)
	}
}

func TestNamedMVDeleteArgsCASByteIdenticalDefault(t *testing.T) {
	if !bytes.Equal(EncodeNamedDeleteArgs("c", 9), EncodeNamedDeleteArgsCAS("c", 9, 0, false)) {
		t.Fatalf("named delete CAS default not byte-identical")
	}
	if !bytes.Equal(EncodeMVDeleteArgs("c", 9), EncodeMVDeleteArgsCAS("c", 9, 0, false)) {
		t.Fatalf("mv delete CAS default not byte-identical")
	}
	_, id, exp, has, err := DecodeNamedDeleteArgsCAS(EncodeNamedDeleteArgsCAS("c", 9, 4, true))
	if err != nil || id != 9 || exp != 4 || !has {
		t.Fatalf("named delete CAS decode: id=%d exp=%d has=%v err=%v", id, exp, has, err)
	}
	_, mid, mexp, mhas, merr := DecodeMVDeleteArgsCAS(EncodeMVDeleteArgsCAS("c", 9, 4, true))
	if merr != nil || mid != 9 || mexp != 4 || !mhas {
		t.Fatalf("mv delete CAS decode: id=%d exp=%d has=%v err=%v", mid, mexp, mhas, merr)
	}
}

func TestNamedGetResultVersionRoundTrip(t *testing.T) {
	vecs := map[string][]float32{"a": {1, 2, 3}}
	meta := vector.Metadata{"k": vector.NewInt(1)}
	// version 0 → byte-identical to the legacy encoder.
	if !bytes.Equal(
		EncodeNamedGetResult(true, vecs, meta, time.Second, true, true),
		EncodeNamedGetResultV(true, vecs, meta, time.Second, true, true, 0),
	) {
		t.Fatalf("named get result version=0 not byte-identical to legacy")
	}
	// version >=1 round-trips.
	body := EncodeNamedGetResultV(true, vecs, meta, time.Second, true, true, 42)
	found, _, _, _, v, err := DecodeNamedGetResultV(body)
	if err != nil || !found || v != 42 {
		t.Fatalf("named get result V: found=%v v=%d err=%v", found, v, err)
	}
	// A legacy result (no version block) decodes version 0.
	legacy := EncodeNamedGetResult(true, vecs, meta, time.Second, true, true)
	if _, _, _, _, lv, _ := DecodeNamedGetResultV(legacy); lv != 0 {
		t.Fatalf("legacy named get version=%d (want 0)", lv)
	}
}

func TestMVGetResultVersionRoundTrip(t *testing.T) {
	toks := [][]float32{{1, 0}, {0, 1}}
	meta := vector.Metadata{"k": vector.NewInt(1)}
	if !bytes.Equal(
		EncodeMVGetResult(true, toks, meta, true, true),
		EncodeMVGetResultV(true, toks, meta, true, true, 0),
	) {
		t.Fatalf("mv get result version=0 not byte-identical to legacy")
	}
	body := EncodeMVGetResultV(true, toks, meta, true, true, 9)
	found, _, _, v, err := DecodeMVGetResultV(body)
	if err != nil || !found || v != 9 {
		t.Fatalf("mv get result V: found=%v v=%d err=%v", found, v, err)
	}
	if _, _, _, lv, _ := DecodeMVGetResultV(EncodeMVGetResult(true, toks, meta, true, true)); lv != 0 {
		t.Fatalf("legacy mv get version=%d (want 0)", lv)
	}
}

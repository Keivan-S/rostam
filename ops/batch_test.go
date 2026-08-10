// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"testing"
	"time"
)

func TestPutBatchEncodeDecodeRoundtrip(t *testing.T) {
	in := []PutEntry{
		{Key: []byte("a"), Val: []byte("1"), TTL: 0},
		{Key: []byte("bb"), Val: []byte("value-2"), TTL: 5 * time.Second},
		{Key: []byte("ccc"), Val: nil, TTL: 0},
	}
	out, err := DecodePutBatchArgs(EncodePutBatchArgs(in))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(out) != len(in) {
		t.Fatalf("decoded %d entries, want %d", len(out), len(in))
	}
	for i := range in {
		if !bytes.Equal(out[i].Key, in[i].Key) || !bytes.Equal(out[i].Val, in[i].Val) || out[i].TTL != in[i].TTL {
			t.Fatalf("entry %d: got %+v, want %+v", i, out[i], in[i])
		}
	}
}

func TestPutBatchHandlerAppliesAll(t *testing.T) {
	r, tx := newTestSetup(t)
	h, kind, ke, ok := r.Lookup("put_batch")
	if !ok {
		t.Fatal("put_batch not registered")
	}
	if kind != OpReadWrite {
		t.Fatalf("put_batch kind = %v, want OpReadWrite", kind)
	}
	if ke == nil {
		t.Fatal("put_batch must be routable (non-nil key extractor)")
	}

	entries := []PutEntry{
		{Key: []byte("k1"), Val: []byte("v1")},
		{Key: []byte("k2"), Val: []byte("v2")},
		{Key: []byte("k3"), Val: []byte("v3")},
	}
	args := EncodePutBatchArgs(entries)

	// Key extractor routes by the FIRST key.
	rk, ok := ke(args)
	if !ok || !bytes.Equal(rk, []byte("k1")) {
		t.Fatalf("key extractor = %q,%v; want k1,true", rk, ok)
	}

	res, err := h(tx, args)
	if err != nil {
		t.Fatalf("put_batch handler: %v", err)
	}
	if n, err := DecodePutBatchResult(res); err != nil || n != 3 {
		t.Fatalf("result = %d,%v; want 3,nil", n, err)
	}
	// Every key readable via get.
	getH, _, _, _ := r.Lookup("get")
	for _, e := range entries {
		got, err := getH(tx, EncodeKeyArgs(e.Key))
		if err != nil || !bytes.Equal(got, e.Val) {
			t.Fatalf("get %q = %q,%v; want %q", e.Key, got, err, e.Val)
		}
	}
}

func TestPutBatchMatchesIndividualPuts(t *testing.T) {
	rb, txb := newTestSetup(t)
	rs, txs := newTestSetup(t)
	entries := []PutEntry{
		{Key: []byte("x"), Val: []byte("10")},
		{Key: []byte("y"), Val: []byte("20")},
		{Key: []byte("x"), Val: []byte("30")}, // last-write-wins within the batch
	}
	// Batch path.
	hb, _, _, _ := rb.Lookup("put_batch")
	if _, err := hb(txb, EncodePutBatchArgs(entries)); err != nil {
		t.Fatalf("batch: %v", err)
	}
	// Individual path.
	hp, _, _, _ := rs.Lookup("put")
	for _, e := range entries {
		if _, err := hp(txs, EncodePutArgs(e.Key, e.Val, e.TTL)); err != nil {
			t.Fatalf("put: %v", err)
		}
	}
	getB, _, _, _ := rb.Lookup("get")
	getS, _, _, _ := rs.Lookup("get")
	for _, k := range []string{"x", "y"} {
		gb, _ := getB(txb, EncodeKeyArgs([]byte(k)))
		gs, _ := getS(txs, EncodeKeyArgs([]byte(k)))
		if !bytes.Equal(gb, gs) {
			t.Fatalf("key %q: batch=%q individual=%q — must match", k, gb, gs)
		}
	}
}

// TestPutBatchHandlerTruncatedAppliesNothing verifies a malformed (truncated)
// batch applies NOTHING — the handler decodes the whole buffer before applying,
// so structural errors are atomic.
func TestPutBatchHandlerTruncatedAppliesNothing(t *testing.T) {
	r, tx := newTestSetup(t)
	h, _, _, _ := r.Lookup("put_batch")
	// One real entry, but claim count=2 → truncated.
	args := EncodePutBatchArgs([]PutEntry{{Key: []byte("a"), Val: []byte("1")}})
	args[3] = 2
	if _, err := h(tx, args); err != ErrShortArgs {
		t.Fatalf("truncated handler: got %v, want ErrShortArgs", err)
	}
	getH, _, _, _ := r.Lookup("get")
	if _, err := getH(tx, EncodeKeyArgs([]byte("a"))); err == nil {
		t.Fatal("truncated batch must apply nothing, but key 'a' was written")
	}
}

// TestPutBatchDecodeRejectsHugeCount verifies a bogus count is rejected by the
// entry-size bound before any large allocation.
func TestPutBatchDecodeRejectsHugeCount(t *testing.T) {
	args := []byte{0xFF, 0xFF, 0xFF, 0xFF} // count = 4B, no entries
	if _, err := DecodePutBatchArgs(args); err != ErrShortArgs {
		t.Fatalf("huge count: got %v, want ErrShortArgs", err)
	}
}

func TestPutBatchShortArgs(t *testing.T) {
	if _, err := DecodePutBatchArgs([]byte{0, 0}); err != ErrShortArgs {
		t.Fatalf("short header: got %v, want ErrShortArgs", err)
	}
	// count=2 but only one entry present.
	truncated := EncodePutBatchArgs([]PutEntry{{Key: []byte("a"), Val: []byte("b")}})
	truncated[3] = 2 // bump count to 2
	if _, err := DecodePutBatchArgs(truncated); err != ErrShortArgs {
		t.Fatalf("truncated batch: got %v, want ErrShortArgs", err)
	}
}

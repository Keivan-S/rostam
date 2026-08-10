// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"testing"
)

// TestAppendReadOptsTrailerBoundedByteIdentical proves the additive contract:
//   - rc==0 && opa==0 → base UNCHANGED (byte-identical to the legacy / no-trailer form);
//   - rc∈{1,2} → byte-identical to the legacy appendReadOptsTrailer 3-byte form;
//   - the wrapper appendReadOptsTrailer(base,rc,opa) == ...Bounded(base,rc,opa,0).
func TestAppendReadOptsTrailerBoundedByteIdentical(t *testing.T) {
	base := []byte{0xAA, 0xBB, 0xCC}

	// rc==0 && opa==0 → unchanged.
	if got := appendReadOptsTrailerBounded(append([]byte(nil), base...), 0, 0, 0); !bytes.Equal(got, base) {
		t.Fatalf("rc=0,opa=0: got %x, want unchanged %x", got, base)
	}
	if got := appendReadOptsTrailer(append([]byte(nil), base...), 0, 0); !bytes.Equal(got, base) {
		t.Fatalf("wrapper rc=0,opa=0: got %x, want unchanged %x", got, base)
	}

	// rc∈{1,2} (and rc==0 with opa!=0) → 3-byte legacy form, no staleness bit.
	for _, tc := range []struct {
		rc, opa uint8
	}{
		{ConsistencyLeaderOnly, 0},
		{ConsistencyLinearizable, 0},
		{ConsistencyLinearizable, 7},
		{0, 5},
	} {
		want := append(append([]byte(nil), base...), readOptsTrailerMarker, tc.rc, tc.opa)
		legacy := appendReadOptsTrailer(append([]byte(nil), base...), tc.rc, tc.opa)
		if !bytes.Equal(legacy, want) {
			t.Fatalf("rc=%d,opa=%d: legacy=%x, want %x", tc.rc, tc.opa, legacy, want)
		}
		bounded := appendReadOptsTrailerBounded(append([]byte(nil), base...), tc.rc, tc.opa, 0)
		if !bytes.Equal(bounded, want) {
			t.Fatalf("rc=%d,opa=%d: bounded(bound=0)=%x, want %x", tc.rc, tc.opa, bounded, want)
		}
		// A nonzero bound must NOT ride for non-bounded rc.
		bounded2 := appendReadOptsTrailerBounded(append([]byte(nil), base...), tc.rc, tc.opa, 999)
		if !bytes.Equal(bounded2, want) {
			t.Fatalf("rc=%d,opa=%d: bounded(bound=999)=%x must drop bound, want %x", tc.rc, tc.opa, bounded2, want)
		}
	}
}

// TestReadOptsTrailerBoundedRoundTrip proves the bound rides ONLY for rc==3 and
// survives a decode round-trip via decodeReadOptsTrailerBounded.
func TestReadOptsTrailerBoundedRoundTrip(t *testing.T) {
	base := []byte{0x01, 0x02}
	for _, bound := range []uint64{0, 1, 42, 1 << 40, ^uint64(0)} {
		args := appendReadOptsTrailerBounded(append([]byte(nil), base...), ConsistencyBoundedStaleness, 3, bound)
		// Layout: base | marker(0x03) | rc(3) | opa(3) | 8 bound bytes.
		if len(args) != len(base)+3+8 {
			t.Fatalf("bound=%d: len=%d, want %d", bound, len(args), len(base)+11)
		}
		if args[len(base)] != (readOptsTrailerMarker | readOptsStalenessBit) {
			t.Fatalf("bound=%d: marker=%#x, want %#x", bound, args[len(base)], readOptsTrailerMarker|readOptsStalenessBit)
		}
		rc, opa, gotBound, err := decodeReadOptsTrailerBounded(args, len(base))
		if err != nil {
			t.Fatalf("bound=%d: decode err: %v", bound, err)
		}
		if rc != ConsistencyBoundedStaleness || opa != 3 || gotBound != bound {
			t.Fatalf("bound=%d: rc=%d opa=%d bound=%d, want rc=3 opa=3 bound=%d", bound, rc, opa, gotBound, bound)
		}
		// decodeReadOptsTrailer (the 3-field view) still consumes/validates the bound.
		rc2, opa2, err := decodeReadOptsTrailer(args, len(base))
		if err != nil || rc2 != ConsistencyBoundedStaleness || opa2 != 3 {
			t.Fatalf("bound=%d: legacy decode rc=%d opa=%d err=%v", bound, rc2, opa2, err)
		}
	}
}

// TestDecodeReadOptsTrailerBoundedFailLoud proves a present staleness bit with a
// truncated 8-byte bound is corruption — fail loud, never a silent drop.
func TestDecodeReadOptsTrailerBoundedFailLoud(t *testing.T) {
	base := []byte{0x09}
	full := appendReadOptsTrailerBounded(append([]byte(nil), base...), ConsistencyBoundedStaleness, 0, 0xDEADBEEF)
	// Truncate the 8-byte bound (drop the last byte): marker+rc+opa present, bound short.
	trunc := full[:len(full)-1]
	if _, _, _, err := decodeReadOptsTrailerBounded(trunc, len(base)); err == nil {
		t.Fatal("truncated bound: want fail-loud error, got nil")
	}
	// Truncated rc/opa block (marker present, no rc/opa) still fails loud.
	if _, _, _, err := decodeReadOptsTrailerBounded([]byte{readOptsTrailerMarker | readOptsStalenessBit}, 0); err == nil {
		t.Fatal("truncated rc/opa: want fail-loud error, got nil")
	}
}

// TestReadStalenessOf proves the op-aware bound peek: it returns the bound only for a
// recognized read op carrying rc==3, and (0,false) otherwise. We build a real bounded
// vector_get blob by appending the bounded trailer (with a nonzero bound) to the get
// base block — exactly the wire shape the bound-aware encoder will emit.
func TestReadStalenessOf(t *testing.T) {
	const wantBound = 12345
	base := EncodeVectorGetArgs("coll", 7, 0)
	getArgs := appendReadOptsTrailerBounded(base, ConsistencyBoundedStaleness, 0, wantBound)

	bound, ok := ReadStalenessOf("vector_get", getArgs)
	if !ok {
		t.Fatalf("vector_get bounded: ok=false, want true (args=%x)", getArgs)
	}
	if bound != wantBound {
		t.Fatalf("vector_get bounded: bound=%d, want %d", bound, wantBound)
	}

	// A bound==0 bounded get still round-trips ok=true (the staleness bit, hence the
	// level, survives): the Task-1 encoder emits exactly this for rc==3.
	zeroBound := EncodeVectorGetArgsOpts("coll", 7, 0, ConsistencyBoundedStaleness, 0, 0)
	if b, ok := ReadStalenessOf("vector_get", zeroBound); !ok || b != 0 {
		t.Fatalf("zero-bound get: bound=%d ok=%v, want 0,true (args=%x)", b, ok, zeroBound)
	}

	// A non-bounded get (rc==0) → (0,false).
	plain := EncodeVectorGetArgsOpts("coll", 7, 0, ConsistencyAnyReplica, 0, 0)
	if _, ok := ReadStalenessOf("vector_get", plain); ok {
		t.Fatalf("plain get: ok=true, want false")
	}
	// A Linearizable get (rc==2) → (0,false): not bounded-staleness.
	lin := EncodeVectorGetArgsOpts("coll", 7, 0, ConsistencyLinearizable, 0, 0)
	if _, ok := ReadStalenessOf("vector_get", lin); ok {
		t.Fatalf("linearizable get: ok=true, want false")
	}
	// An unknown op → (0,false).
	if _, ok := ReadStalenessOf("put", getArgs); ok {
		t.Fatalf("unknown op: ok=true, want false")
	}
}

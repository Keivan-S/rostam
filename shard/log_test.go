// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"bytes"
	"errors"
	"testing"
)

func TestLogEntryRoundtrip(t *testing.T) {
	opName := "update_session"
	args := []byte(`{"user":"42","coins":100}`)

	buf := EncodeLogEntry(opName, args)
	gotName, gotArgs, stampMs, stamped, err := DecodeLogEntry(buf)
	if err != nil {
		t.Fatalf("DecodeLogEntry: %v", err)
	}
	if gotName != opName {
		t.Fatalf("opName = %q, want %q", gotName, opName)
	}
	if !bytes.Equal(gotArgs, args) {
		t.Fatalf("args mismatch")
	}
	if stampMs != 0 || stamped {
		t.Fatalf("legacy entry stampMs=%d stamped=%v, want 0/false", stampMs, stamped)
	}
}

func TestLogEntryEmptyArgs(t *testing.T) {
	buf := EncodeLogEntry("ping", nil)
	gotName, gotArgs, _, _, err := DecodeLogEntry(buf)
	if err != nil {
		t.Fatal(err)
	}
	if gotName != "ping" || len(gotArgs) != 0 {
		t.Fatalf("ping/empty: name=%q args=%q", gotName, gotArgs)
	}
}

func TestLogEntryDecodeTruncated(t *testing.T) {
	buf := EncodeLogEntry("op", []byte("args"))
	for n := 0; n < len(buf); n++ {
		if _, _, _, _, err := DecodeLogEntry(buf[:n]); err == nil {
			t.Fatalf("DecodeLogEntry on truncated buf len=%d returned nil; want error", n)
		}
	}
}

// TestLogEntryStampedRoundtrip proves the extended (0x00-marked) layout carries
// the leader stamp through a decode intact, for both pooled and non-pooled
// encoders (#4 Phase B / B1).
func TestLogEntryStampedRoundtrip(t *testing.T) {
	const stamp = uint64(1_700_000_000_123)
	opName := "put"
	args := []byte("some-args")

	for _, enc := range []struct {
		name string
		buf  []byte
	}{
		{"stamped", EncodeLogEntryStamped(opName, args, stamp)},
		{"stamped-pooled", EncodeLogEntryStampedPooled(opName, args, stamp)},
	} {
		if enc.buf[0] != logStampMarker {
			t.Fatalf("%s: first byte = %#x, want stamp marker %#x", enc.name, enc.buf[0], logStampMarker)
		}
		gotName, gotArgs, gotStamp, stamped, err := DecodeLogEntry(enc.buf)
		if err != nil {
			t.Fatalf("%s: DecodeLogEntry: %v", enc.name, err)
		}
		if !stamped {
			t.Fatalf("%s: stamped = false, want true (extended format)", enc.name)
		}
		if gotName != opName {
			t.Fatalf("%s: opName = %q, want %q", enc.name, gotName, opName)
		}
		if !bytes.Equal(gotArgs, args) {
			t.Fatalf("%s: args mismatch", enc.name)
		}
		if gotStamp != stamp {
			t.Fatalf("%s: stampMs = %d, want %d", enc.name, gotStamp, stamp)
		}
	}
}

// TestLogEntryStampedEmptyArgs covers a stamped entry with no args (the arg-less
// op case), exercising the argsLen=0 boundary of the extended layout.
func TestLogEntryStampedEmptyArgs(t *testing.T) {
	buf := EncodeLogEntryStamped("ping", nil, 42)
	name, args, stamp, stamped, err := DecodeLogEntry(buf)
	if err != nil {
		t.Fatal(err)
	}
	if name != "ping" || len(args) != 0 || stamp != 42 || !stamped {
		t.Fatalf("ping stamped: name=%q args=%q stamp=%d stamped=%v", name, args, stamp, stamped)
	}
}

// TestLogEntryStampedZeroStamp proves a stamped entry whose leader clock is 0
// still reports stamped=true (the format, not the value, is the signal), so the
// apply path uses the deterministic At-clock rather than falling back to per-node
// wall clocks. This is the seam that closes the stamp==0 divergence footgun.
func TestLogEntryStampedZeroStamp(t *testing.T) {
	buf := EncodeLogEntryStamped("put", []byte("v"), 0)
	name, _, stamp, stamped, err := DecodeLogEntry(buf)
	if err != nil {
		t.Fatal(err)
	}
	if name != "put" || stamp != 0 || !stamped {
		t.Fatalf("zero-stamp: name=%q stamp=%d stamped=%v; want put/0/true", name, stamp, stamped)
	}
}

// TestLogEntryStampedTruncated proves every truncation of a stamped entry is
// rejected rather than mis-parsed.
func TestLogEntryStampedTruncated(t *testing.T) {
	buf := EncodeLogEntryStamped("op", []byte("args"), 7)
	for n := 0; n < len(buf); n++ {
		if _, _, _, _, err := DecodeLogEntry(buf[:n]); err == nil {
			t.Fatalf("DecodeLogEntry on truncated stamped buf len=%d returned nil; want error", n)
		}
	}
}

// TestLogEntryUnknownVersion proves a stamped entry with a future version fails
// closed (ErrLogEntryVersion) rather than decoding garbage — the guard behind the
// two-phase rollout's fail-closed-on-premature-stamping property.
func TestLogEntryUnknownVersion(t *testing.T) {
	buf := EncodeLogEntryStamped("put", []byte("x"), 1)
	buf[1] = 0xFF // corrupt the version byte
	if _, _, _, _, err := DecodeLogEntry(buf); !errors.Is(err, ErrLogEntryVersion) {
		t.Fatalf("DecodeLogEntry unknown version err = %v, want ErrLogEntryVersion", err)
	}
}

// TestLogEntryDiscriminatorNoCollision proves the 0x00 discriminator never
// collides with a real legacy opNameLen: a registered op name is non-empty, so a
// legacy entry's first byte (opNameLen) is always >= 1, and only the extended
// format ever begins with 0x00. This is what lets DecodeLogEntry tell the layouts
// apart with zero ambiguity.
func TestLogEntryDiscriminatorNoCollision(t *testing.T) {
	// A legacy entry for the shortest possible op name (1 byte) still has first
	// byte == 1, never 0.
	legacy := EncodeLogEntry("x", nil)
	if legacy[0] == logStampMarker {
		t.Fatal("legacy entry began with the stamp marker — discriminator collision")
	}
	// It decodes as legacy with stamp 0 and stamped=false.
	name, _, stamp, stamped, err := DecodeLogEntry(legacy)
	if err != nil || name != "x" || stamp != 0 || stamped {
		t.Fatalf("legacy decode: name=%q stamp=%d stamped=%v err=%v", name, stamp, stamped, err)
	}
}

func TestLogEntryOpNameTooLong(t *testing.T) {
	long := make([]byte, 256)
	for i := range long {
		long[i] = 'a'
	}
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for 256-byte op name")
		}
	}()
	EncodeLogEntry(string(long), nil)
}

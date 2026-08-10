// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"encoding/binary"
	"errors"
	"strings"
	"testing"
)

// TestWCEnvelopeRoundTrip verifies encode→decode returns the identical
// (wcf, wait, innerName, innerArgs) for a spread of inner-op shapes.
func TestWCEnvelopeRoundTrip(t *testing.T) {
	cases := []struct {
		name      string
		wcf       uint8
		wait      uint8
		innerName string
		innerArgs []byte
	}{
		{
			name:      "typical insert-ish",
			wcf:       3,
			wait:      1,
			innerName: "vector_insert",
			innerArgs: []byte{0x00, 0x07, 't', 'e', 's', 't', 'c', 'o', 'l', 0x01, 0x02, 0x03},
		},
		{
			name:      "empty inner args",
			wcf:       1,
			wait:      0,
			innerName: "delete",
			innerArgs: []byte{},
		},
		{
			name:      "nil inner args",
			wcf:       2,
			wait:      1,
			innerName: "payload_set",
			innerArgs: nil,
		},
		{
			name:      "255-byte inner name",
			wcf:       5,
			wait:      0,
			innerName: strings.Repeat("x", 255),
			innerArgs: []byte{0xaa, 0xbb},
		},
		{
			name:      "empty inner name",
			wcf:       0,
			wait:      1,
			innerName: "",
			innerArgs: []byte{0x01},
		},
		{
			name:      "large inner args >64KB (exercises u32 length)",
			wcf:       4,
			wait:      0,
			innerName: "vector_upsert",
			innerArgs: bytes.Repeat([]byte{0x5a}, 70000),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			enc := EncodeWCEnvelope(tc.wcf, tc.wait, tc.innerName, tc.innerArgs)
			wcf, wait, name, args, err := DecodeWCEnvelope(enc)
			if err != nil {
				t.Fatalf("DecodeWCEnvelope: unexpected error: %v", err)
			}
			if wcf != tc.wcf {
				t.Errorf("wcf = %d, want %d", wcf, tc.wcf)
			}
			if wait != tc.wait {
				t.Errorf("wait = %d, want %d", wait, tc.wait)
			}
			if name != tc.innerName {
				t.Errorf("innerName = %q, want %q", name, tc.innerName)
			}
			// Inner args must be byte-identical to the original. bytes.Equal
			// treats nil and empty as equal, which is the intended contract.
			if !bytes.Equal(args, tc.innerArgs) {
				t.Errorf("innerArgs mismatch: got %d bytes, want %d bytes", len(args), len(tc.innerArgs))
			}
		})
	}
}

// TestWCEnvelopeInnerArgsIntegrity asserts the decoded innerArgs equals the
// original bytes exactly (no truncation, no extra, no reordering), including for
// args that themselves contain bytes that could look like length headers.
func TestWCEnvelopeInnerArgsIntegrity(t *testing.T) {
	orig := []byte{0x00, 0x00, 0x00, 0xff, 0xfe, 0xfd, 0x00, 0x01, 0x02}
	enc := EncodeWCEnvelope(7, 1, "delete_by_filter", orig)
	_, _, _, got, err := DecodeWCEnvelope(enc)
	if err != nil {
		t.Fatalf("DecodeWCEnvelope: %v", err)
	}
	if !bytes.Equal(got, orig) {
		t.Fatalf("innerArgs = % x, want % x", got, orig)
	}
}

// TestWCEnvelopeTruncated exercises every length boundary: a short buffer for
// the fixed header, a nameLen claiming more than present, an argsLen claiming
// more than present, and an oversized name byte. Each must return the
// truncation sentinel and never panic.
func TestWCEnvelopeTruncated(t *testing.T) {
	t.Run("nil buffer", func(t *testing.T) {
		assertTruncated(t, nil)
	})
	t.Run("empty buffer", func(t *testing.T) {
		assertTruncated(t, []byte{})
	})
	t.Run("only wcf present", func(t *testing.T) {
		assertTruncated(t, []byte{0x03})
	})
	t.Run("only wcf+wait present (no nameLen)", func(t *testing.T) {
		assertTruncated(t, []byte{0x03, 0x01})
	})
	t.Run("nameLen claims more than present", func(t *testing.T) {
		// header says name is 5 bytes but only 2 follow.
		assertTruncated(t, []byte{0x03, 0x01, 0x05, 'a', 'b'})
	})
	t.Run("oversized name leaves no room for argsLen", func(t *testing.T) {
		// name exactly consumes the rest; no 4 bytes left for argsLen.
		buf := []byte{0x03, 0x01, 0x03, 'a', 'b', 'c'}
		assertTruncated(t, buf)
	})
	t.Run("argsLen header truncated", func(t *testing.T) {
		// name ok, but only 3 of 4 argsLen bytes present.
		buf := []byte{0x03, 0x01, 0x02, 'a', 'b', 0x00, 0x00, 0x00}
		assertTruncated(t, buf)
	})
	t.Run("argsLen claims more than present", func(t *testing.T) {
		// argsLen = 10 but only 2 args bytes follow.
		buf := []byte{0x03, 0x01, 0x02, 'a', 'b'}
		var hdr [4]byte
		binary.BigEndian.PutUint32(hdr[:], 10)
		buf = append(buf, hdr[:]...)
		buf = append(buf, 0x01, 0x02)
		assertTruncated(t, buf)
	})
	t.Run("argsLen huge overflow", func(t *testing.T) {
		buf := []byte{0x03, 0x01, 0x00}
		var hdr [4]byte
		binary.BigEndian.PutUint32(hdr[:], 0xffffffff)
		buf = append(buf, hdr[:]...)
		assertTruncated(t, buf)
	})
}

// TestWCEnvelopeTruncatedAtEveryPrefix slices a valid envelope at every byte
// boundary and asserts each truncated prefix yields the sentinel (never a
// panic, never a spurious success).
func TestWCEnvelopeTruncatedAtEveryPrefix(t *testing.T) {
	full := EncodeWCEnvelope(4, 1, "vector_insert", []byte{0x01, 0x02, 0x03, 0x04})
	for n := 0; n < len(full); n++ {
		prefix := full[:n]
		if _, _, _, _, err := DecodeWCEnvelope(prefix); !errors.Is(err, errWCEnvelopeTruncated) {
			t.Errorf("prefix len %d: err = %v, want errWCEnvelopeTruncated", n, err)
		}
	}
	// The full buffer must decode cleanly.
	if _, _, _, _, err := DecodeWCEnvelope(full); err != nil {
		t.Errorf("full buffer: unexpected error %v", err)
	}
}

func TestEncodeWCEnvelopePanicsOnOversizedName(t *testing.T) {
	// A 256-byte name would wrap the u8 length prefix to 0 and silently corrupt
	// the frame — EncodeWCEnvelope must fail loud instead.
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("EncodeWCEnvelope with a 256-byte name: expected panic, got none")
		}
	}()
	EncodeWCEnvelope(1, 1, string(make([]byte, 256)), nil)
}

// A name of exactly maxOpNameLen (255) must NOT panic (boundary).
func TestEncodeWCEnvelopeMaxNameOK(t *testing.T) {
	name := string(make([]byte, maxOpNameLen))
	got := EncodeWCEnvelope(1, 1, name, []byte{0x09})
	_, _, gotName, gotArgs, err := DecodeWCEnvelope(got)
	if err != nil || gotName != name || len(gotArgs) != 1 {
		t.Fatalf("255-byte name round-trip: name-ok=%v args=%v err=%v", gotName == name, gotArgs, err)
	}
}

func assertTruncated(t *testing.T, buf []byte) {
	t.Helper()
	_, _, _, _, err := DecodeWCEnvelope(buf)
	if !errors.Is(err, errWCEnvelopeTruncated) {
		t.Fatalf("DecodeWCEnvelope(% x): err = %v, want errWCEnvelopeTruncated", buf, err)
	}
}

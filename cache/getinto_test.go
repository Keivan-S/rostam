// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
	"errors"
	"testing"
	"time"
)

// TestGetIntoMatchesGetAndIsAllocFree verifies GetInto returns the same bytes as
// Get, allocates nothing when the caller reuses its buffer, and leaves dst
// untouched on a miss.
func TestGetIntoMatchesGetAndIsAllocFree(t *testing.T) {
	c, err := New(DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	key := []byte("alpha")
	val := bytes.Repeat([]byte("x"), 256)
	if err := c.Put(key, val, 0); err != nil {
		t.Fatal(err)
	}

	want, err := c.Get(key)
	if err != nil {
		t.Fatal(err)
	}
	got, err := c.GetInto(make([]byte, 0, 256), key)
	if err != nil {
		t.Fatalf("GetInto: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("GetInto bytes = %q, want %q", got, want)
	}

	// Reusing the buffer across hits must be allocation-free (the whole point).
	buf := make([]byte, 0, 256)
	allocs := testing.AllocsPerRun(1000, func() {
		buf, _ = c.GetInto(buf[:0], key)
	})
	if allocs != 0 {
		t.Errorf("GetInto with reused buffer = %.1f allocs/op, want 0", allocs)
	}

	// Miss: dst returned unchanged, ErrNotFound.
	out, err := c.GetInto([]byte("keep"), []byte("absent"))
	if !errors.Is(err, ErrNotFound) {
		t.Errorf("GetInto(absent) err = %v, want ErrNotFound", err)
	}
	if !bytes.Equal(out, []byte("keep")) {
		t.Errorf("GetInto(absent) dst = %q, want unchanged %q", out, "keep")
	}
}

// TestGetIntoExpiry confirms an expired entry reports a miss through GetInto.
func TestGetIntoExpiry(t *testing.T) {
	c, err := New(DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	if err := c.Put([]byte("k"), []byte("v"), time.Millisecond); err != nil {
		t.Fatal(err)
	}
	time.Sleep(5 * time.Millisecond)
	if _, err := c.GetInto(nil, []byte("k")); !errors.Is(err, ErrNotFound) {
		t.Errorf("expired GetInto err = %v, want ErrNotFound", err)
	}
}

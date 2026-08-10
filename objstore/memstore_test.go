// SPDX-License-Identifier: Apache-2.0

package objstore

import (
	"bytes"
	"context"
	"errors"
	"io"
	"testing"
	"time"
)

func TestMemStore_PutGet(t *testing.T) {
	ctx := context.Background()
	m := NewMemStore()

	if err := m.Put(ctx, "a/b.txt", bytes.NewReader([]byte("hello")), 5); err != nil {
		t.Fatalf("put: %v", err)
	}
	rc, err := m.Get(ctx, "a/b.txt")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	got, _ := io.ReadAll(rc)
	rc.Close()
	if string(got) != "hello" {
		t.Errorf("got %q, want hello", got)
	}
}

func TestMemStore_NotFound(t *testing.T) {
	ctx := context.Background()
	m := NewMemStore()

	if _, err := m.Get(ctx, "missing"); !errors.Is(err, ErrNotFound) {
		t.Errorf("Get missing: got %v, want ErrNotFound", err)
	}
	if err := m.Delete(ctx, "missing"); !errors.Is(err, ErrNotFound) {
		t.Errorf("Delete missing: got %v, want ErrNotFound", err)
	}
}

func TestMemStore_DeleteThenGone(t *testing.T) {
	ctx := context.Background()
	m := NewMemStore()
	_ = m.Put(ctx, "k", bytes.NewReader([]byte("v")), 1)
	if err := m.Delete(ctx, "k"); err != nil {
		t.Fatalf("delete: %v", err)
	}
	if _, err := m.Get(ctx, "k"); !errors.Is(err, ErrNotFound) {
		t.Errorf("get after delete: got %v, want ErrNotFound", err)
	}
}

func TestMemStore_ListPrefixSorted(t *testing.T) {
	ctx := context.Background()
	m := NewMemStore()
	fixed := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	m.SetClock(func() time.Time { return fixed })

	keys := []string{"t/c/3.snap", "t/c/1.snap", "t/c/2.snap", "other/x", "t/d/1.snap"}
	for _, k := range keys {
		if err := m.Put(ctx, k, bytes.NewReader([]byte(k)), int64(len(k))); err != nil {
			t.Fatalf("put %s: %v", k, err)
		}
	}

	got, err := m.List(ctx, "t/c/")
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	want := []string{"t/c/1.snap", "t/c/2.snap", "t/c/3.snap"}
	if len(got) != len(want) {
		t.Fatalf("got %d objects, want %d: %+v", len(got), len(want), got)
	}
	for i, oi := range got {
		if oi.Key != want[i] {
			t.Errorf("key[%d] = %q, want %q", i, oi.Key, want[i])
		}
		if oi.Size != int64(len(want[i])) {
			t.Errorf("size[%d] = %d, want %d", i, oi.Size, len(want[i]))
		}
		if !oi.LastModified.Equal(fixed) {
			t.Errorf("lastModified[%d] = %v, want %v", i, oi.LastModified, fixed)
		}
	}
}

func TestMemStore_ListEmptyPrefix(t *testing.T) {
	ctx := context.Background()
	m := NewMemStore()
	_ = m.Put(ctx, "a", bytes.NewReader([]byte("1")), 1)
	_ = m.Put(ctx, "b", bytes.NewReader([]byte("2")), 1)
	got, err := m.List(ctx, "")
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(got) != 2 {
		t.Errorf("got %d, want 2", len(got))
	}
}

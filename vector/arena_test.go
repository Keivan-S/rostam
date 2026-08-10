// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"reflect"
	"testing"
)

func TestArenaInsertAndLookup(t *testing.T) {
	a := newArena(3, 4)
	slot, err := a.Insert(100, []float32{1, 2, 3})
	if err != nil {
		t.Fatalf("insert: %v", err)
	}
	if slot != 0 {
		t.Errorf("first slot = %d, want 0", slot)
	}
	got, ok := a.Slot(100)
	if !ok || got != 0 {
		t.Errorf("Slot(100) = %d, %v; want 0, true", got, ok)
	}
	vec := a.Vec(0)
	if !reflect.DeepEqual(vec, []float32{1, 2, 3}) {
		t.Errorf("Vec(0) = %v, want {1,2,3}", vec)
	}
}

func TestArenaInsertDimMismatch(t *testing.T) {
	a := newArena(3, 0)
	_, err := a.Insert(1, []float32{1, 2})
	if !errors.Is(err, ErrDimMismatch) {
		t.Errorf("got %v, want ErrDimMismatch", err)
	}
}

func TestArenaInsertDuplicate(t *testing.T) {
	a := newArena(2, 0)
	if _, err := a.Insert(1, []float32{1, 2}); err != nil {
		t.Fatal(err)
	}
	_, err := a.Insert(1, []float32{3, 4})
	if !errors.Is(err, ErrDuplicateID) {
		t.Errorf("got %v, want ErrDuplicateID", err)
	}
}

func TestArenaDelete(t *testing.T) {
	a := newArena(2, 0)
	_, _ = a.Insert(1, []float32{1, 2})
	if !a.Delete(1) {
		t.Fatal("delete should succeed")
	}
	if a.Delete(1) {
		t.Fatal("second delete should fail")
	}
	if _, ok := a.Slot(1); ok {
		t.Fatal("slot should be gone after delete")
	}
}

func TestArenaSlotReuse(t *testing.T) {
	a := newArena(2, 0)
	s1, _ := a.Insert(1, []float32{1, 2})
	s2, _ := a.Insert(2, []float32{3, 4})
	a.Delete(1)
	s3, _ := a.Insert(3, []float32{5, 6})
	if s3 != s1 {
		t.Errorf("expected slot reuse: s3=%d, s1=%d", s3, s1)
	}
	if !reflect.DeepEqual(a.Vec(s2), []float32{3, 4}) {
		t.Errorf("s2 vector corrupted: %v", a.Vec(s2))
	}
	if !reflect.DeepEqual(a.Vec(s3), []float32{5, 6}) {
		t.Errorf("s3 vector wrong: %v", a.Vec(s3))
	}
}

func TestArenaFreeListIsLIFO(t *testing.T) {
	a := newArena(1, 0)
	s1, _ := a.Insert(1, []float32{1})
	s2, _ := a.Insert(2, []float32{2})
	s3, _ := a.Insert(3, []float32{3})
	a.Delete(1)
	a.Delete(2)
	a.Delete(3)
	// LIFO: next three inserts must reuse s3, s2, s1 in that order.
	r1, _ := a.Insert(10, []float32{10})
	r2, _ := a.Insert(20, []float32{20})
	r3, _ := a.Insert(30, []float32{30})
	if r1 != s3 || r2 != s2 || r3 != s1 {
		t.Errorf("LIFO order broken: got %d,%d,%d want %d,%d,%d", r1, r2, r3, s3, s2, s1)
	}
}

func TestArenaGrowsBeyondInitCap(t *testing.T) {
	a := newArena(2, 1) // 1-slot hint, force several appends
	for i := uint64(0); i < 10; i++ {
		if _, err := a.Insert(i, []float32{float32(i), float32(i) + 0.5}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	// Every earlier vector must survive the underlying backing-array growth.
	for i := uint64(0); i < 10; i++ {
		slot, _ := a.Slot(i)
		got := a.Vec(slot)
		if got[0] != float32(i) || got[1] != float32(i)+0.5 {
			t.Errorf("vec %d after growth = %v", i, got)
		}
	}
}

func TestArenaNewArenaPanicsOnZeroDim(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for dim <= 0")
		}
	}()
	newArena(0, 0)
}

func TestArenaExpires(t *testing.T) {
	a := newArena(4, 0)
	slot, err := a.Insert(1, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if got := a.ExpiresAt(slot); got != 0 {
		t.Fatalf("default ExpiresAt = %d, want 0", got)
	}
	a.SetExpires(slot, 12345)
	if got := a.ExpiresAt(slot); got != 12345 {
		t.Fatalf("ExpiresAt after Set = %d, want 12345", got)
	}

	// Reusing a freed slot clears the expiry.
	a.Delete(1)
	slot2, err := a.Insert(2, []float32{5, 6, 7, 8})
	if err != nil {
		t.Fatalf("Insert reuse: %v", err)
	}
	if slot2 != slot {
		t.Fatalf("expected slot reuse, got slot=%d original=%d", slot2, slot)
	}
	if got := a.ExpiresAt(slot2); got != 0 {
		t.Fatalf("reused slot ExpiresAt = %d, want 0", got)
	}
}

func TestArenaSize(t *testing.T) {
	a := newArena(2, 0)
	if a.Size() != 0 {
		t.Errorf("empty size = %d, want 0", a.Size())
	}
	_, _ = a.Insert(1, []float32{1, 2})
	_, _ = a.Insert(2, []float32{3, 4})
	if a.Size() != 2 {
		t.Errorf("after 2 inserts size = %d, want 2", a.Size())
	}
	a.Delete(1)
	if a.Size() != 1 {
		t.Errorf("after 1 delete size = %d, want 1", a.Size())
	}
}

func TestArenaMetadata(t *testing.T) {
	a := newArena(4, 0)
	slot, err := a.Insert(1, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if got := a.Metadata(slot); got != nil {
		t.Errorf("default Metadata = %+v, want nil", got)
	}
	meta := Metadata{"tenant": NewString("acme"), "score": NewInt(42)}
	a.SetMetadata(slot, meta)
	got := a.Metadata(slot)
	if got["tenant"].Str != "acme" || got["score"].Int != 42 {
		t.Errorf("Metadata after Set = %+v, want acme/42", got)
	}

	// Slot reuse clears metadata so the new vector doesn't inherit attributes.
	a.Delete(1)
	slot2, err := a.Insert(2, []float32{5, 6, 7, 8})
	if err != nil {
		t.Fatalf("Insert reuse: %v", err)
	}
	if slot2 != slot {
		t.Fatalf("expected slot reuse: slot=%d original=%d", slot2, slot)
	}
	if got := a.Metadata(slot2); got != nil {
		t.Errorf("reused slot Metadata = %+v, want nil", got)
	}
}

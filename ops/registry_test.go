// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"
	"testing"
)

func TestRegistryRegisterAndLookup(t *testing.T) {
	r := NewRegistry()
	called := false
	// nolint:revive
	handler := func(tx *TxContext, args []byte) ([]byte, error) {
		called = true
		return []byte("ok"), nil
	}
	if err := r.Register("test_op", OpReadWrite, handler); err != nil {
		t.Fatalf("Register: %v", err)
	}
	h, kind, _, ok := r.Lookup("test_op")
	if !ok {
		t.Fatal("Lookup returned false")
	}
	if kind != OpReadWrite {
		t.Fatalf("kind = %v, want OpReadWrite", kind)
	}
	if _, err := h(nil, nil); err != nil {
		t.Fatalf("handler call: %v", err)
	}
	if !called {
		t.Fatal("handler was not invoked")
	}
}

func TestRegistryLookupMissing(t *testing.T) {
	r := NewRegistry()
	if _, _, _, ok := r.Lookup("missing"); ok {
		t.Fatal("Lookup of missing op returned true")
	}
}

func TestRegistryDuplicateName(t *testing.T) {
	r := NewRegistry()
	// nolint:revive
	noop := func(tx *TxContext, args []byte) ([]byte, error) { return nil, nil }
	if err := r.Register("foo", OpReadOnly, noop); err != nil {
		t.Fatal(err)
	}
	err := r.Register("foo", OpReadOnly, noop)
	if err == nil {
		t.Fatal("re-register: expected error, got nil")
	}
	if !errors.Is(err, ErrDuplicateOp) {
		t.Fatalf("re-register: err = %v, want ErrDuplicateOp", err)
	}
}

func TestRegistryRejectsEmptyOrLongName(t *testing.T) {
	r := NewRegistry()
	// nolint:revive
	noop := func(tx *TxContext, args []byte) ([]byte, error) { return nil, nil }
	if err := r.Register("", OpReadOnly, noop); err == nil {
		t.Fatal("empty name: expected error")
	}
	long := make([]byte, 256)
	for i := range long {
		long[i] = 'a'
	}
	if err := r.Register(string(long), OpReadOnly, noop); err == nil {
		t.Fatal("256-byte name: expected error (max 255)")
	}
}

func TestRegistryRejectsTwoCharNames(t *testing.T) {
	// Two-character op names collide with the protocol-v2 [version=2] byte
	// at offset 0 of every request body. The registry rejects them on the
	// way in so a server upgraded to v2 frame parsing can't ambiguously
	// decode a legacy 2-char op name.
	r := NewRegistry()
	noop := func(_ *TxContext, _ []byte) ([]byte, error) { return nil, nil }
	if err := r.Register("go", OpReadOnly, noop); err == nil {
		t.Fatal("Register(\"go\") allowed; want rejection for length-2 name")
	}
}

func TestRegistryRejectsNilHandler(t *testing.T) {
	r := NewRegistry()
	// nolint:errcheck
	if err := r.Register("x", OpReadOnly, nil); err == nil {
		t.Fatal("nil handler: expected error")
	}
}

func TestRegistryRegisterRoutableAndLookup(t *testing.T) {
	r := NewRegistry()
	handler := func(_ *TxContext, _ []byte) ([]byte, error) { return nil, nil }
	ke := func(args []byte) ([]byte, bool) {
		if len(args) < 2 {
			return nil, false
		}
		return args[2:], true
	}
	if err := r.RegisterRoutable("routable", OpReadWrite, handler, ke); err != nil {
		t.Fatalf("RegisterRoutable: %v", err)
	}
	gotH, gotKind, gotKE, ok := r.Lookup("routable")
	if !ok {
		t.Fatal("Lookup returned false")
	}
	if gotKind != OpReadWrite {
		t.Errorf("kind = %v, want OpReadWrite", gotKind)
	}
	if gotH == nil {
		t.Error("handler is nil")
	}
	if gotKE == nil {
		t.Error("extractor is nil")
	}
	if _, hasKey := gotKE([]byte{0, 3, 'a', 'b', 'c'}); !hasKey {
		t.Error("extractor did not detect key")
	}
}

func TestRegistryRegisterShardlessHasNilExtractor(t *testing.T) {
	r := NewRegistry()
	handler := func(_ *TxContext, _ []byte) ([]byte, error) { return nil, nil }
	if err := r.Register("shardless", OpReadOnly, handler); err != nil {
		t.Fatal(err)
	}
	_, _, ke, ok := r.Lookup("shardless")
	if !ok {
		t.Fatal("Lookup returned false")
	}
	if ke != nil {
		t.Error("extractor should be nil for Register (non-routable)")
	}
}

func TestRegistryRegisterRoutableRejectsNilExtractor(t *testing.T) {
	r := NewRegistry()
	handler := func(_ *TxContext, _ []byte) ([]byte, error) { return nil, nil }
	if err := r.RegisterRoutable("x", OpReadOnly, handler, nil); err == nil {
		t.Fatal("RegisterRoutable with nil extractor should error")
	}
}

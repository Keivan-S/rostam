// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"bytes"
	"io"
	"testing"

	"github.com/rostamlabs/rostam/cache"
)

func newHeapCache(t *testing.T) *cache.Cache {
	t.Helper()
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	c, err := cache.New(cc)
	if err != nil {
		t.Fatalf("cache.New: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

// TestSnapshotCarriesWASMSection pins the v5 wire format: an opaque
// dynamic-registration blob survives serialize→restore alongside the cache
// entries, and reaches the restore hook byte-for-byte.
//
// The blob is what lets a snapshot-installed replica end up with the same op
// registry as a log-replaying one. Without it that replica applies none of the
// __register_wasm__ entries the snapshot replaced and fails closed (classFatal
// ErrOpNotRegistered) on the first invocation its peers execute.
func TestSnapshotCarriesWASMSection(t *testing.T) {
	src := newHeapCache(t)
	if err := src.Put([]byte("k"), []byte("v"), 0); err != nil {
		t.Fatalf("Put: %v", err)
	}
	blob := []byte("opaque-registration-payload")

	data, err := serializeSnapshot(src, nil, 11, blob)
	if err != nil {
		t.Fatalf("serializeSnapshot: %v", err)
	}

	dst := newHeapCache(t)
	var got []byte
	idx, err := restoreSnapshot(dst, nil, func(b []byte) error {
		got = append([]byte(nil), b...)
		return nil
	}, io.NopCloser(bytes.NewReader(data)))
	if err != nil {
		t.Fatalf("restoreSnapshot: %v", err)
	}
	if idx != 11 {
		t.Errorf("appliedIndex = %d, want 11", idx)
	}
	if string(got) != string(blob) {
		t.Fatalf("wasm blob = %q, want %q", got, blob)
	}
	if v, err := dst.Get([]byte("k")); err != nil || string(v) != "v" {
		t.Fatalf("cache entry lost alongside the wasm section: %q %v", v, err)
	}
}

// TestSnapshotEmptyWASMSectionRoundTrips covers the common case — a shard with
// no dynamic registrations — and requires the restore hook NOT to be invoked, so
// an empty section is distinguishable from a present-but-empty one.
func TestSnapshotEmptyWASMSectionRoundTrips(t *testing.T) {
	src := newHeapCache(t)
	data, err := serializeSnapshot(src, nil, 3, nil)
	if err != nil {
		t.Fatalf("serializeSnapshot: %v", err)
	}
	dst := newHeapCache(t)
	called := false
	if _, err := restoreSnapshot(dst, nil, func([]byte) error {
		called = true
		return nil
	}, io.NopCloser(bytes.NewReader(data))); err != nil {
		t.Fatalf("restoreSnapshot: %v", err)
	}
	if called {
		t.Error("the restore hook must not fire for a snapshot carrying no registrations")
	}
}

// TestSnapshotWASMSectionRefusedWithoutHook proves the restore FAILS rather than
// silently dropping registrations a Store cannot install. Installing the cache
// state while discarding the ops it was produced by would leave the replica
// reporting a successful restore while guaranteed to halt on the next
// invocation.
func TestSnapshotWASMSectionRefusedWithoutHook(t *testing.T) {
	src := newHeapCache(t)
	data, err := serializeSnapshot(src, nil, 3, []byte("payload"))
	if err != nil {
		t.Fatalf("serializeSnapshot: %v", err)
	}
	dst := newHeapCache(t)
	if _, err := restoreSnapshot(dst, nil, nil, io.NopCloser(bytes.NewReader(data))); err == nil {
		t.Fatal("a snapshot carrying registrations must not restore into a Store with no way to install them")
	}
}

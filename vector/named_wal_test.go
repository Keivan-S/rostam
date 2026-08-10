// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// namedWALConfig is the standard collection-level config for a single-node
// WAL-mode named collection: the two-space named layout + WAL on, sync on.
func namedWALConfig() Config {
	return Config{
		WAL:          true,
		NamedVectors: namedTestConfig(), // "title" dim4 cosine, "image" dim3 dot
	}
}

// freezeNamedClock pins a named collection's clock to a fixed unix-ms so per-key
// TTL deadlines are computed against a deterministic base and aging is explicit.
func freezeNamedClock(nc *NamedCollection, ms int64) {
	nc.now = func() int64 { return ms }
}

// TestNamedWALRecoversUnflushed inserts (multi-space) + set_payload (with per-key
// TTL) + delete on a WAL-mode named collection, then simulates a crash (NO Flush)
// by closing and reopening the store. The full state must be recovered from the
// snapshot (empty here) + the replayed WAL tail: vectors per space, the shared
// payload, the point TTL, and — critically — the per-key TTL as an ABSOLUTE
// deadline that is time-stable (advancing the recovered clock past the original
// absolute deadline expires the key; it is NOT recomputed from recovery time).
func TestNamedWALRecoversUnflushed(t *testing.T) {
	dir := t.TempDir()
	const base = int64(1_000_000_000_000) // fixed clock base (unix-ms)

	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateCollection("named", namedWALConfig()); err != nil {
		t.Fatal(err)
	}
	nc, ok := cs.GetNamed("named")
	if !ok {
		t.Fatal("named collection missing after create")
	}
	if nc.wal == nil {
		t.Fatal("WAL-mode named collection has nil wal (lifecycle not wired)")
	}
	freezeNamedClock(nc, base)

	// Point 1: both spaces, a permanent payload key + a per-key-TTL key (1s).
	if err := nc.Insert(1, map[string][]float32{
		"title": {1, 0, 0, 0},
		"image": {1, 0, 0},
	}, Metadata{"kind": NewString("a")}, time.Hour); err != nil {
		t.Fatalf("insert 1: %v", err)
	}
	// Point 2: only title (omits image), survives.
	if err := nc.Insert(2, map[string][]float32{
		"title": {0, 1, 0, 0},
	}, Metadata{"kind": NewString("b")}, 0); err != nil {
		t.Fatalf("insert 2: %v", err)
	}
	// Point 3: inserted then deleted (the delete must replay).
	if err := nc.Insert(3, map[string][]float32{"title": {0, 0, 1, 0}}, nil, 0); err != nil {
		t.Fatalf("insert 3: %v", err)
	}
	// set_payload on point 1: add a key "temp" with a 1000ms per-key TTL (absolute
	// deadline = base+1000) and a permanent key "extra".
	if err := nc.SetPayload(1, Metadata{"temp": NewString("x"), "extra": NewInt(7)},
		map[string]int64{"temp": 1000}); err != nil {
		t.Fatalf("set_payload 1: %v", err)
	}
	if _, err := nc.Delete(3); err != nil {
		t.Fatalf("delete 3: %v", err)
	}

	// Simulate a crash: close WITHOUT Flush — only the WAL is on disk.
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = cs2.Close() }()
	nc2, ok := cs2.GetNamed("named")
	if !ok {
		t.Fatal("named collection missing after reopen")
	}
	if nc2.wal == nil {
		t.Fatal("reopened WAL-mode named collection has nil wal")
	}
	// Recover with the clock still at base: nothing has aged yet.
	freezeNamedClock(nc2, base)

	if got := nc2.NumPoints(); got != 2 {
		t.Fatalf("recovered live points = %d, want 2 (3 was deleted)", got)
	}

	// Point 1 vectors per space recovered.
	v1, p1, ttl1, _, live := nc2.Get(1)
	if !live {
		t.Fatal("point 1 not recovered live")
	}
	if len(v1["title"]) != 4 || len(v1["image"]) != 3 {
		t.Fatalf("point 1 vectors = %v, want title dim4 + image dim3", v1)
	}
	// Shared payload merged: kind=a (insert) + temp + extra (set_payload).
	if p1["kind"].Str != "a" || p1["extra"].Int != 7 || p1["temp"].Str != "x" {
		t.Fatalf("point 1 payload = %v, want kind=a extra=7 temp=x", p1)
	}
	// Point TTL recovered (~1h remaining at base).
	if ttl1 <= 0 {
		t.Fatalf("point 1 ttl = %v, want >0 (1h point TTL recovered)", ttl1)
	}

	// Point 3 stays gone (the delete replayed).
	if _, _, _, _, live3 := nc2.Get(3); live3 {
		t.Fatal("point 3 still live after replayed delete")
	}

	// Per-key TTL ABSOLUTE-deadline time-stability: the replayed deadline is
	// base+1000 verbatim. Verify the internal absolute deadline first.
	if dl := nc2.keyTTL[1]["temp"]; dl != base+1000 {
		t.Fatalf("recovered temp deadline = %d, want %d (absolute, verbatim — not recomputed)", dl, base+1000)
	}
	// Advance the recovered clock 2s past base: temp's absolute deadline (base+1000)
	// has passed, so the key is dropped from the live view; extra/kind remain.
	freezeNamedClock(nc2, base+2000)
	_, pAged, _, _, live := nc2.Get(1)
	if !live {
		t.Fatal("point 1 expired unexpectedly (point TTL was 1h)")
	}
	if _, hasTemp := pAged["temp"]; hasTemp {
		t.Errorf("temp key still present after its absolute per-key deadline (replay must NOT recompute now+ttl)")
	}
	if pAged["extra"].Int != 7 || pAged["kind"].Str != "a" {
		t.Errorf("permanent keys lost after aging: %v", pAged)
	}
}

// TestNamedWALFlushTruncates checks Flush checkpoints + truncates the WAL: after a
// Flush a reopen replays nothing (the snapshot holds the state), and a write AFTER
// the Flush lands in the fresh tail and is recovered on a second crash reopen.
func TestNamedWALFlushTruncates(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateCollection("named", namedWALConfig()); err != nil {
		t.Fatal(err)
	}
	nc, _ := cs.GetNamed("named")
	if err := nc.Insert(1, map[string][]float32{"title": {1, 0, 0, 0}}, Metadata{"k": NewInt(1)}, 0); err != nil {
		t.Fatal(err)
	}
	if err := cs.FlushNamed("named"); err != nil {
		t.Fatalf("flush: %v", err)
	}

	// The WAL is now empty (the snapshot subsumes it).
	walPath := filepath.Join(dir, "vectors", "default", "named.nwal")
	if fi, serr := os.Stat(walPath); serr != nil {
		t.Fatalf("stat wal: %v", serr)
	} else if fi.Size() != 0 {
		t.Fatalf("post-Flush wal size = %d, want 0 (truncated)", fi.Size())
	}

	// A write AFTER the Flush goes into the new tail.
	if err := nc.Insert(2, map[string][]float32{"title": {0, 1, 0, 0}}, Metadata{"k": NewInt(2)}, 0); err != nil {
		t.Fatal(err)
	}
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs2.Close() }()
	nc2, _ := cs2.GetNamed("named")
	if got := nc2.NumPoints(); got != 2 {
		t.Fatalf("recovered points = %d, want 2 (1 from snapshot + 2 from post-Flush tail)", got)
	}
	if _, p1, _, _, ok := nc2.Get(1); !ok || p1["k"].Int != 1 {
		t.Errorf("point 1 (snapshot) not recovered: ok=%v p=%v", ok, p1)
	}
	if _, p2, _, _, ok := nc2.Get(2); !ok || p2["k"].Int != 2 {
		t.Errorf("point 2 (post-Flush tail) not recovered: ok=%v p=%v", ok, p2)
	}
}

// TestNamedWALTornTailTolerated corrupts the last WAL record mid-bytes (a crash
// mid-append) and verifies reopen replays cleanly up to the durability boundary —
// the intact prior records recovered, the torn record ignored, no panic.
func TestNamedWALTornTailTolerated(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateCollection("named", namedWALConfig()); err != nil {
		t.Fatal(err)
	}
	nc, _ := cs.GetNamed("named")
	for i := 1; i <= 5; i++ {
		if err := nc.Insert(uint64(i), map[string][]float32{"title": {float32(i), 0, 0, 0}}, nil, 0); err != nil {
			t.Fatal(err)
		}
	}
	_ = cs.Close()

	// Append a partial record header (claims a 9-byte payload, only 2 bytes follow).
	walPath := filepath.Join(dir, "vectors", "default", "named.nwal")
	f, err := os.OpenFile(walPath, os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = f.Write([]byte{0, 0, 0, 9, 1, 2})
	_ = f.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen with torn tail: %v", err)
	}
	defer func() { _ = cs2.Close() }()
	nc2, _ := cs2.GetNamed("named")
	if got := nc2.NumPoints(); got != 5 {
		t.Errorf("recovered points = %d, want 5 (torn tail ignored, prior records kept)", got)
	}
}

// TestNamedWALIdempotentReplayOnSnapshot replays the WAL on top of a snapshot that
// already reflects some of the logged ops (the seam Flush↔post-Flush writes can
// overlap): re-applying an Insert/SetPayload/Delete the snapshot already has must
// converge to the same state (no double-count, no resurrection).
func TestNamedWALIdempotentReplayOnSnapshot(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateCollection("named", namedWALConfig()); err != nil {
		t.Fatal(err)
	}
	nc, _ := cs.GetNamed("named")
	if err := nc.Insert(1, map[string][]float32{"title": {1, 0, 0, 0}}, Metadata{"k": NewInt(1)}, 0); err != nil {
		t.Fatal(err)
	}

	// Snapshot the current state to the checkpoint file WITHOUT truncating the WAL,
	// so on reopen the snapshot has point 1 AND the WAL replays its Insert again
	// (the idempotency case). Bypass FlushNamed (which truncates) by writing the
	// snapshot directly under the same opMu discipline.
	_, snapPath, _ := cs.namedPaths("default/named")
	nc.opMu.Lock()
	if werr := cs.writeNamedSnapshotFile(nc, snapPath); werr != nil {
		nc.opMu.Unlock()
		t.Fatalf("write snapshot: %v", werr)
	}
	nc.opMu.Unlock()

	// More ops AFTER the snapshot (these only exist in the WAL tail).
	if err := nc.Insert(2, map[string][]float32{"title": {0, 1, 0, 0}}, Metadata{"k": NewInt(2)}, 0); err != nil {
		t.Fatal(err)
	}
	if err := nc.SetPayload(1, Metadata{"extra": NewInt(9)}, nil); err != nil {
		t.Fatal(err)
	}
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs2.Close() }()
	nc2, _ := cs2.GetNamed("named")
	if got := nc2.NumPoints(); got != 2 {
		t.Fatalf("recovered points = %d, want 2 (idempotent replay of point 1 on top of snapshot)", got)
	}
	_, p1, _, _, ok := nc2.Get(1)
	if !ok || p1["k"].Int != 1 || p1["extra"].Int != 9 {
		t.Errorf("point 1 = %v ok=%v, want k=1 + extra=9 (replayed Insert + SetPayload)", p1, ok)
	}
	if _, p2, _, _, ok := nc2.Get(2); !ok || p2["k"].Int != 2 {
		t.Errorf("point 2 not recovered: ok=%v p=%v", ok, p2)
	}
}

// TestNamedHeapOnlyNoWAL confirms a non-WAL named collection writes no marker/WAL
// files and recovers nothing on reopen (the historical in-memory behavior is
// preserved — WAL is strictly opt-in).
func TestNamedHeapOnlyNoWAL(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateNamed("named", namedTestConfig()); err != nil {
		t.Fatal(err)
	}
	nc, _ := cs.GetNamed("named")
	if nc.wal != nil {
		t.Fatal("heap-only named collection has a non-nil wal")
	}
	if err := nc.Insert(1, map[string][]float32{"title": {1, 0, 0, 0}}, nil, 0); err != nil {
		t.Fatal(err)
	}
	// No marker/snapshot/wal files on disk.
	cfgPath, snapPath, walPath := cs.namedPaths("default/named")
	for _, p := range []string{cfgPath, snapPath, walPath} {
		if _, serr := os.Stat(p); serr == nil {
			t.Errorf("heap-only named wrote %s (should be in-memory only)", p)
		}
	}
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs2.Close() }()
	if _, ok := cs2.GetNamed("named"); ok {
		t.Error("heap-only named collection survived restart (should not persist)")
	}
}

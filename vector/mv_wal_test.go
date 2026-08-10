// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// mvWALConfig is the standard config for a single-node WAL-mode (heap-checkpoint)
// multi-vector collection: dim-4 tokens + WAL on, sync on.
func mvWALConfig() MultiVectorConfig {
	return MultiVectorConfig{Dim: 4, Seed: 1, WAL: true}
}

// freezeMVClock pins an MV index's clock to a fixed unix-ms so per-key TTL
// deadlines are computed against a deterministic base and aging is explicit.
func freezeMVClock(m *MultiVectorIndex, ms int64) {
	m.now = func() int64 { return ms }
}

// TestMVWALRejectsPersistent confirms WAL && Persistent is rejected (the two
// durability modes are mutually exclusive — WAL is heap-checkpoint, Persistent is
// mmap instant-restart).
func TestMVWALRejectsPersistent(t *testing.T) {
	_, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1, WAL: true, Persistent: true, Quant: QuantSQ8})
	if !errors.Is(err, ErrInvalidWAL) {
		t.Fatalf("WAL+Persistent = %v, want ErrInvalidWAL", err)
	}
}

// TestMVWALRecoversUnflushed adds (token matrices) + set_payload (with per-key
// TTL) + delete on a WAL-mode MV collection, then simulates a crash (NO Flush) by
// closing and reopening the store. The full state must be recovered from the
// snapshot (empty here) + the replayed WAL tail: token matrices, the payload, and
// — critically — the per-key TTL as an ABSOLUTE deadline that is time-stable
// (advancing the recovered clock past the original absolute deadline expires the
// key; it is NOT recomputed from recovery time).
func TestMVWALRecoversUnflushed(t *testing.T) {
	dir := t.TempDir()
	const base = int64(1_000_000_000_000) // fixed clock base (unix-ms)

	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateMultiVector("mv", mvWALConfig()); err != nil {
		t.Fatal(err)
	}
	idx, ok := cs.GetMultiVector("mv")
	if !ok {
		t.Fatal("MV collection missing after create")
	}
	if idx.wal == nil {
		t.Fatal("WAL-mode MV collection has nil wal (lifecycle not wired)")
	}
	freezeMVClock(idx, base)

	// Doc 1: two token vectors, a permanent payload key.
	if err := idx.Add(1, [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}, Metadata{"kind": NewString("a")}); err != nil {
		t.Fatalf("add 1: %v", err)
	}
	// Doc 2: one token, survives.
	if err := idx.Add(2, [][]float32{{0, 0, 1, 0}}, Metadata{"kind": NewString("b")}); err != nil {
		t.Fatalf("add 2: %v", err)
	}
	// Doc 3: added then deleted (the delete must replay).
	if err := idx.Add(3, [][]float32{{0, 0, 0, 1}}, nil); err != nil {
		t.Fatalf("add 3: %v", err)
	}
	// set_payload on doc 1: add a key "temp" with a 1000ms per-key TTL (absolute
	// deadline = base+1000) and a permanent key "extra".
	if err := idx.SetPayload(1, Metadata{"temp": NewString("x"), "extra": NewInt(7)},
		map[string]int64{"temp": 1000}); err != nil {
		t.Fatalf("set_payload 1: %v", err)
	}
	if !idx.Delete(3) {
		t.Fatal("delete 3 reported not present")
	}

	// Simulate a crash: close WITHOUT Flush — only the WAL is on disk.
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = cs2.Close() }()
	idx2, ok := cs2.GetMultiVector("mv")
	if !ok {
		t.Fatal("MV collection missing after reopen")
	}
	if idx2.wal == nil {
		t.Fatal("reopened WAL-mode MV collection has nil wal")
	}
	freezeMVClock(idx2, base)

	if got := idx2.NumDocs(); got != 2 {
		t.Fatalf("recovered live docs = %d, want 2 (3 was deleted)", got)
	}

	// Doc 1 token matrix recovered.
	tok1, p1, _, live := idx2.Get(1)
	if !live {
		t.Fatal("doc 1 not recovered live")
	}
	if len(tok1) != 2 || len(tok1[0]) != 4 {
		t.Fatalf("doc 1 tokens = %v, want 2 rows of dim4", tok1)
	}
	// Payload merged: kind=a (add) + temp + extra (set_payload).
	if p1["kind"].Str != "a" || p1["extra"].Int != 7 || p1["temp"].Str != "x" {
		t.Fatalf("doc 1 payload = %v, want kind=a extra=7 temp=x", p1)
	}

	// Doc 3 stays gone (the delete replayed).
	if _, _, _, live3 := idx2.Get(3); live3 {
		t.Fatal("doc 3 still live after replayed delete")
	}

	// Per-key TTL ABSOLUTE-deadline time-stability: the replayed deadline is
	// base+1000 verbatim. Verify the internal absolute deadline first.
	if dl := idx2.keyTTL[1]["temp"]; dl != base+1000 {
		t.Fatalf("recovered temp deadline = %d, want %d (absolute, verbatim — not recomputed)", dl, base+1000)
	}
	// Advance the recovered clock 2s past base: temp's absolute deadline (base+1000)
	// has passed, so the key is dropped from the live view; extra/kind remain.
	freezeMVClock(idx2, base+2000)
	_, pAged, _, live := idx2.Get(1)
	if !live {
		t.Fatal("doc 1 disappeared unexpectedly (MV has no point TTL)")
	}
	if _, hasTemp := pAged["temp"]; hasTemp {
		t.Errorf("temp key still present after its absolute per-key deadline (replay must NOT recompute now+ttl)")
	}
	if pAged["extra"].Int != 7 || pAged["kind"].Str != "a" {
		t.Errorf("permanent keys lost after aging: %v", pAged)
	}
}

// TestMVWALPayloadOpsAllLog confirms EVERY payload op logs (the bug the named
// completion caught): DeletePayloadKeys and ClearPayload must replay too. Doc 1
// gets keys via set_payload, then DeletePayloadKeys drops one; doc 2 gets a payload
// then ClearPayload wipes it. After a crash reopen, the resulting payloads recover.
func TestMVWALPayloadOpsAllLog(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateMultiVector("mv", mvWALConfig()); err != nil {
		t.Fatal(err)
	}
	idx, _ := cs.GetMultiVector("mv")
	if err := idx.Add(1, [][]float32{{1, 0, 0, 0}}, nil); err != nil {
		t.Fatal(err)
	}
	if err := idx.Add(2, [][]float32{{0, 1, 0, 0}}, nil); err != nil {
		t.Fatal(err)
	}
	if err := idx.SetPayload(1, Metadata{"a": NewInt(1), "b": NewInt(2)}, nil); err != nil {
		t.Fatal(err)
	}
	if err := idx.DeletePayloadKeys(1, []string{"b"}); err != nil { // must log
		t.Fatal(err)
	}
	if err := idx.SetPayload(2, Metadata{"z": NewInt(9)}, nil); err != nil {
		t.Fatal(err)
	}
	if err := idx.ClearPayload(2); err != nil { // must log
		t.Fatal(err)
	}
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs2.Close() }()
	idx2, _ := cs2.GetMultiVector("mv")

	_, p1, _, ok := idx2.Get(1)
	if !ok {
		t.Fatal("doc 1 not recovered")
	}
	if p1["a"].Int != 1 {
		t.Errorf("doc 1 key a = %v, want 1", p1["a"])
	}
	if _, hasB := p1["b"]; hasB {
		t.Errorf("doc 1 key b present after replayed DeletePayloadKeys: %v", p1)
	}
	_, p2, _, ok := idx2.Get(2)
	if !ok {
		t.Fatal("doc 2 not recovered")
	}
	if len(p2) != 0 {
		t.Errorf("doc 2 payload = %v, want empty after replayed ClearPayload", p2)
	}
}

// TestMVWALFlushTruncates checks FlushMVWAL checkpoints + truncates the WAL: after
// a Flush a reopen replays nothing (the heap snapshot holds the state), and a write
// AFTER the Flush lands in the fresh tail and is recovered on a second crash reopen.
func TestMVWALFlushTruncates(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateMultiVector("mv", mvWALConfig()); err != nil {
		t.Fatal(err)
	}
	idx, _ := cs.GetMultiVector("mv")
	if err := idx.Add(1, [][]float32{{1, 0, 0, 0}}, Metadata{"k": NewInt(1)}); err != nil {
		t.Fatal(err)
	}
	if err := cs.FlushMVWAL("mv"); err != nil {
		t.Fatalf("flush: %v", err)
	}

	// The WAL is now empty (the snapshot subsumes it).
	walPath := filepath.Join(dir, "vectors", "default", "mv.mvwal")
	if fi, serr := os.Stat(walPath); serr != nil {
		t.Fatalf("stat wal: %v", serr)
	} else if fi.Size() != 0 {
		t.Fatalf("post-Flush wal size = %d, want 0 (truncated)", fi.Size())
	}

	// A write AFTER the Flush goes into the new tail.
	if err := idx.Add(2, [][]float32{{0, 1, 0, 0}}, Metadata{"k": NewInt(2)}); err != nil {
		t.Fatal(err)
	}
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs2.Close() }()
	idx2, _ := cs2.GetMultiVector("mv")
	if got := idx2.NumDocs(); got != 2 {
		t.Fatalf("recovered docs = %d, want 2 (1 from snapshot + 2 from post-Flush tail)", got)
	}
	if _, p1, _, ok := idx2.Get(1); !ok || p1["k"].Int != 1 {
		t.Errorf("doc 1 (snapshot) not recovered: ok=%v p=%v", ok, p1)
	}
	if _, p2, _, ok := idx2.Get(2); !ok || p2["k"].Int != 2 {
		t.Errorf("doc 2 (post-Flush tail) not recovered: ok=%v p=%v", ok, p2)
	}
}

// TestMVWALTornTailTolerated corrupts the last WAL record mid-bytes (a crash
// mid-append) and verifies reopen replays cleanly up to the durability boundary —
// the intact prior records recovered, the torn record ignored, no panic.
func TestMVWALTornTailTolerated(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateMultiVector("mv", mvWALConfig()); err != nil {
		t.Fatal(err)
	}
	idx, _ := cs.GetMultiVector("mv")
	for i := 1; i <= 5; i++ {
		if err := idx.Add(uint64(i), [][]float32{{float32(i), 0, 0, 0}}, nil); err != nil {
			t.Fatal(err)
		}
	}
	_ = cs.Close()

	// Append a partial record header (claims a 9-byte payload, only 2 bytes follow).
	walPath := filepath.Join(dir, "vectors", "default", "mv.mvwal")
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
	idx2, _ := cs2.GetMultiVector("mv")
	if got := idx2.NumDocs(); got != 5 {
		t.Errorf("recovered docs = %d, want 5 (torn tail ignored, prior records kept)", got)
	}
}

// TestMVWALIdempotentReplayOnSnapshot replays the WAL on top of a snapshot that
// already reflects some of the logged ops (the seam Flush↔post-Flush writes can
// overlap): re-applying an Add/SetPayload/Delete the snapshot already has must
// converge to the same state (no double-count, no resurrection).
func TestMVWALIdempotentReplayOnSnapshot(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateMultiVector("mv", mvWALConfig()); err != nil {
		t.Fatal(err)
	}
	idx, _ := cs.GetMultiVector("mv")
	if err := idx.Add(1, [][]float32{{1, 0, 0, 0}}, Metadata{"k": NewInt(1)}); err != nil {
		t.Fatal(err)
	}

	// Snapshot the current state to the checkpoint file WITHOUT truncating the WAL,
	// so on reopen the snapshot has doc 1 AND the WAL replays its Add again (the
	// idempotency case). Bypass FlushMVWAL (which truncates) by writing the snapshot
	// directly under the same opMu discipline.
	snapPath, _ := cs.mvWALPaths("default/mv")
	idx.opMu.Lock()
	if werr := cs.writeMVWALSnapshotFile(idx, snapPath); werr != nil {
		idx.opMu.Unlock()
		t.Fatalf("write snapshot: %v", werr)
	}
	idx.opMu.Unlock()

	// More ops AFTER the snapshot (these only exist in the WAL tail).
	if err := idx.Add(2, [][]float32{{0, 1, 0, 0}}, Metadata{"k": NewInt(2)}); err != nil {
		t.Fatal(err)
	}
	if err := idx.SetPayload(1, Metadata{"extra": NewInt(9)}, nil); err != nil {
		t.Fatal(err)
	}
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs2.Close() }()
	idx2, _ := cs2.GetMultiVector("mv")
	if got := idx2.NumDocs(); got != 2 {
		t.Fatalf("recovered docs = %d, want 2 (idempotent replay of doc 1 on top of snapshot)", got)
	}
	_, p1, _, ok := idx2.Get(1)
	if !ok || p1["k"].Int != 1 || p1["extra"].Int != 9 {
		t.Errorf("doc 1 = %v ok=%v, want k=1 + extra=9 (replayed Add + SetPayload)", p1, ok)
	}
	// Doc 1 must have exactly ONE token row (the replayed Add must not double its
	// tokens on top of the snapshot's).
	tok1, _, _, _ := idx2.Get(1)
	if len(tok1) != 1 {
		t.Errorf("doc 1 token rows = %d, want 1 (replayed Add must not duplicate tokens)", len(tok1))
	}
	if _, p2, _, ok := idx2.Get(2); !ok || p2["k"].Int != 2 {
		t.Errorf("doc 2 not recovered: ok=%v p=%v", ok, p2)
	}
}

// TestMVHeapOnlyNoWAL confirms a non-WAL, non-Persistent MV collection writes no
// WAL files and recovers nothing on reopen (the historical in-memory behavior is
// preserved — WAL is strictly opt-in).
func TestMVHeapOnlyNoWAL(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateMultiVector("mv", MultiVectorConfig{Dim: 4, Seed: 1}); err != nil {
		t.Fatal(err)
	}
	idx, _ := cs.GetMultiVector("mv")
	if idx.wal != nil {
		t.Fatal("heap-only MV collection has a non-nil wal")
	}
	if err := idx.Add(1, [][]float32{{1, 0, 0, 0}}, nil); err != nil {
		t.Fatal(err)
	}
	// No WAL files on disk.
	snapPath, walPath := cs.mvWALPaths("default/mv")
	for _, p := range []string{snapPath, walPath} {
		if _, serr := os.Stat(p); serr == nil {
			t.Errorf("heap-only MV wrote %s (should be in-memory only)", p)
		}
	}
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs2.Close() }()
	if _, ok := cs2.GetMultiVector("mv"); ok {
		t.Error("heap-only MV collection survived restart (should not persist)")
	}
}

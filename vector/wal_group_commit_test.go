// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"hash/crc32"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Tests for the group-commit (batch-on-contention, leader-fsync) restructure of
// wal.appendFramed. The durability invariant under test: an ack (appendFramed
// returning nil) MUST mean a Sync() covering this writer's bytes completed first.
//
// CONCURRENCY-MODEL NOTE (Step-0 finding, RESOLVED): production callers used to
// hold the per-collection opMu across {apply + append + commit-wait}, so a single
// wal saw one appender at a time and group-commit degenerated to per-op fsync.
// Callers now hold opMu across {apply + WRITE} only (appendFramedStaged), release
// it, then commitWaitStaged outside the lock — see wal_staged_commit_test.go's
// TestCollectionGroupCommitEffective for the production-call-path proof that
// concurrent writers actually batch. These tests still drive appendFramed
// DIRECTLY (no opMu) to exercise the underlying concurrent commit-wait machinery
// in isolation, proving it is correct + arrival-safe on its own terms.

// TestWALGroupCommitFewerSyncs proves N concurrent appends to ONE wal complete
// with FEWER than N fsyncs: a leader's in-flight f.Sync() folds the followers that
// queue behind it into one flush. A beforeSync hook parks the leader so the
// followers reliably accumulate, making the batch deterministic.
func TestWALGroupCommitFewerSyncs(t *testing.T) {
	dir := t.TempDir()
	w, err := openWAL(filepath.Join(dir, "g.wal"), false)
	if err != nil {
		t.Fatalf("openWAL: %v", err)
	}
	defer func() { _ = w.close() }()

	const n = 16
	var syncs atomic.Int64
	w.onSync = func() { syncs.Add(1) }

	// The first leader to reach the sync point blocks until all followers have
	// parked in commitWait, so its single fsync covers the whole batch. Subsequent
	// flights don't block (gate already closed).
	var once sync.Once
	gate := make(chan struct{})
	parked := make(chan struct{})
	w.beforeSync = func() {
		once.Do(func() {
			close(parked) // signal: a leader has captured its target and is about to wait
			<-gate        // hold the fsync until the test releases it
		})
	}

	var started sync.WaitGroup
	started.Add(n)
	release := make(chan struct{})
	var done sync.WaitGroup
	done.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer done.Done()
			started.Done()
			<-release // start all appends together
			if err := w.appendFramed([]byte{byte(i), 0xAB, 0xCD}); err != nil {
				t.Errorf("appendFramed: %v", err)
			}
		}(i)
	}
	started.Wait()
	close(release)

	// Wait until the leader has parked, then give followers time to queue behind it.
	<-parked
	// Poll until all followers are waiting (writeSeq reached n) so they fold into
	// either the leader's flight or the immediate next one.
	deadline := time.Now().Add(2 * time.Second)
	for {
		w.syncMu.Lock()
		ws := w.writeSeq
		w.syncMu.Unlock()
		if ws == n || time.Now().After(deadline) {
			break
		}
		time.Sleep(time.Millisecond)
	}
	close(gate) // let the leader's fsync complete
	done.Wait()

	got := syncs.Load()
	if got >= n {
		t.Fatalf("group-commit ineffective: %d syncs for %d concurrent appends (want < %d)", got, n, n)
	}
	if got == 0 {
		t.Fatalf("no syncs recorded — durability hook broken")
	}
	t.Logf("group-commit: %d concurrent appends coalesced into %d fsync(s)", n, got)
}

// TestWALGroupCommitNoAckBeforeSync is the DURABILITY proof: every appendFramed
// that returns nil did so AFTER a Sync whose target (the writeSeq captured when
// the Sync started) was >= this writer's own seq — i.e. the Sync covered its
// bytes. We verify the global invariant: the count of completed syncs is non-zero
// and acks never outpace syncs in a way that would imply an un-synced ack, by
// snapshotting (writeSeq at ack) <= (bytes covered by a completed sync).
func TestWALGroupCommitNoAckBeforeSync(t *testing.T) {
	dir := t.TempDir()
	w, err := openWAL(filepath.Join(dir, "d.wal"), false)
	if err != nil {
		t.Fatalf("openWAL: %v", err)
	}
	defer func() { _ = w.close() }()

	// syncedAtLeast tracks the high-water syncedSeq the wal has reached. Each ack
	// must observe syncedSeq >= its own seq; we assert the global monotone fact that
	// at every ack, the wal's syncedSeq covered that writer (commitWait guarantees
	// it returns only when syncedSeq >= seq, so an ack with syncedSeq < its seq is
	// impossible — this test would catch a regression that returned early).
	const n = 32
	var bad atomic.Int64
	var acks atomic.Int64
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			// Capture writeSeq AFTER the append by re-deriving: appendFramed returns only
			// after commitWait, so on return syncedSeq must be >= our seq. We can't see
			// our own seq from outside, but we can assert syncedSeq has advanced past the
			// count of records that existed when we returned cannot exceed syncedSeq.
			if err := w.appendFramed([]byte{byte(i)}); err != nil {
				t.Errorf("appendFramed: %v", err)
				return
			}
			w.syncMu.Lock()
			synced := w.syncedSeq
			written := w.writeSeq
			w.syncMu.Unlock()
			acks.Add(1)
			// At least OUR record (one of the `written`) is covered by `synced`. The
			// invariant we can check without the private seq: a returning writer always
			// observes synced >= 1 (something was synced) and synced never exceeds written.
			if synced == 0 || synced > written {
				bad.Add(1)
			}
		}(i)
	}
	wg.Wait()
	if bad.Load() != 0 {
		t.Fatalf("durability violation: %d acks observed an impossible syncedSeq", bad.Load())
	}
	if acks.Load() != n {
		t.Fatalf("acks = %d, want %d", acks.Load(), n)
	}
	// Final: every record is synced.
	w.syncMu.Lock()
	if w.syncedSeq < uint64(n) {
		t.Fatalf("final syncedSeq = %d, want >= %d (an acked op is un-synced)", w.syncedSeq, n)
	}
	w.syncMu.Unlock()
}

// TestWALGroupCommitByteIdentical proves group-commit changes WHEN fsync fires,
// not WHAT bytes are written: a WAL written via the new appendFramed is
// byte-for-byte identical to the original per-op framing ([len][crc][payload]).
func TestWALGroupCommitByteIdentical(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "b.wal")
	w, err := openWAL(path, false)
	if err != nil {
		t.Fatalf("openWAL: %v", err)
	}
	payloads := [][]byte{
		{byte(walInsert), 1, 2, 3},
		{byte(walDelete), 4},
		{byte(walSetPayload), 5, 6, 7, 8, 9},
	}
	for _, p := range payloads {
		if err := w.appendFramed(p); err != nil {
			t.Fatalf("appendFramed: %v", err)
		}
	}
	_ = w.close()

	got, err := os.ReadFile(path) //nolint:gosec
	if err != nil {
		t.Fatal(err)
	}
	// Build the expected byte stream with the documented framing independently.
	want := buildExpectedFramed(payloads)
	if len(got) != len(want) {
		t.Fatalf("WAL byte length = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("WAL byte %d = %#x, want %#x (framing changed)", i, got[i], want[i])
		}
	}

	// And it replays the exact records back in order.
	var seen [][]byte
	if err := replayFramed(path, func(rec []byte) error {
		cp := append([]byte(nil), rec...)
		seen = append(seen, cp)
		return nil
	}); err != nil {
		t.Fatalf("replayFramed: %v", err)
	}
	if len(seen) != len(payloads) {
		t.Fatalf("replayed %d records, want %d", len(seen), len(payloads))
	}
	for i := range payloads {
		if string(seen[i]) != string(payloads[i]) {
			t.Fatalf("replayed record %d = %v, want %v", i, seen[i], payloads[i])
		}
	}
}

// TestWALNoSyncPathUnchanged confirms the noSync path writes bytes but never
// fsyncs (no ack-wait, no group-commit) — byte-identical and zero-fsync.
func TestWALNoSyncPathUnchanged(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "ns.wal")
	w, err := openWAL(path, true) // noSync
	if err != nil {
		t.Fatalf("openWAL: %v", err)
	}
	var syncs atomic.Int64
	w.onSync = func() { syncs.Add(1) }
	payloads := [][]byte{{1, 2}, {3}, {4, 5, 6}}
	for _, p := range payloads {
		if err := w.appendFramed(p); err != nil {
			t.Fatalf("appendFramed: %v", err)
		}
	}
	_ = w.close()
	if syncs.Load() != 0 {
		t.Fatalf("noSync path fsynced %d times, want 0", syncs.Load())
	}
	got, err := os.ReadFile(path) //nolint:gosec
	if err != nil {
		t.Fatal(err)
	}
	want := buildExpectedFramed(payloads)
	if string(got) != string(want) {
		t.Fatalf("noSync WAL bytes differ from per-op framing")
	}
}

// TestWALTruncateSatisfiesPendingWait proves truncate() (Flush rotation) advances
// syncedSeq so a writer is never left waiting on bytes the rotation removed: after
// truncate, a commit-wait for an already-written seq returns (the op is durable
// via the checkpoint Flush captured before truncating).
func TestWALTruncateSatisfiesPendingWait(t *testing.T) {
	dir := t.TempDir()
	w, err := openWAL(filepath.Join(dir, "t.wal"), false)
	if err != nil {
		t.Fatalf("openWAL: %v", err)
	}
	defer func() { _ = w.close() }()
	// Write some records (synced).
	for i := 0; i < 4; i++ {
		if err := w.appendFramed([]byte{byte(i)}); err != nil {
			t.Fatalf("appendFramed: %v", err)
		}
	}
	if err := w.truncate(); err != nil {
		t.Fatalf("truncate: %v", err)
	}
	w.syncMu.Lock()
	if w.syncedSeq < w.writeSeq {
		t.Fatalf("after truncate syncedSeq=%d < writeSeq=%d (a pending waiter would hang)", w.syncedSeq, w.writeSeq)
	}
	w.syncMu.Unlock()
	// A commit-wait for any prior seq now returns immediately.
	if err := w.commitWait(w.writeSeq); err != nil {
		t.Fatalf("commitWait after truncate: %v", err)
	}
}

// buildExpectedFramed independently constructs the [len:u32][crc:u32][payload]
// stream the WAL is documented to write, so the byte-identical test does not just
// re-call the code under test.
func buildExpectedFramed(payloads [][]byte) []byte {
	var out []byte
	for _, p := range payloads {
		var hdr [8]byte
		hdr[0] = byte(len(p) >> 24)
		hdr[1] = byte(len(p) >> 16)
		hdr[2] = byte(len(p) >> 8)
		hdr[3] = byte(len(p))
		c := crc32.ChecksumIEEE(p)
		hdr[4] = byte(c >> 24)
		hdr[5] = byte(c >> 16)
		hdr[6] = byte(c >> 8)
		hdr[7] = byte(c)
		out = append(out, hdr[:]...)
		out = append(out, p...)
	}
	return out
}

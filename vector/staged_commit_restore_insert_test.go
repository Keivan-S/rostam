// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Staged-commit coverage for the INSERT-family restore/reshard paths.
//
// These were the last writers still fsyncing while holding opMu:
//
//	dense  Collection.RestoreInsert / RestoreInsertAt
//	       Collection.InsertIfAbsentVersion / InsertIfAbsentVersionAt
//	MV     MultiVectorIndex.AddIfAbsent
//	       MultiVectorIndex.MultiAddIfAbsentVersionSparse
//	       MultiVectorIndex.MultiRestoreAddSparse
//
// They are driven by WAL replay, the ONLINE reshard copy pass, and the OFFLINE
// resplit backfill — high-volume bulk writers. Holding opMu across the fsync made
// such a writer the permanent fsync leader WITH the collection's op lock in hand,
// so every concurrent live writer serialized behind it: the exact group-commit
// collapse already fixed for the delete/upsert/payload families.
//
// As in staged_commit_delete_payload_test.go, the effectiveness half ("few fsyncs
// for many ops") is deliberately paired with the durability half
// (assertDurableAtAck) — on its own, effectiveness is a one-sided metric that
// deleting the commitWaitStaged call would score BEST on.

// TestDenseRestoreInsertFamilyDurableAtAck pins the durability half for all four
// dense restore/reshard entry points: at the moment each returns, every record it
// staged must be covered by a completed fsync (serial ops, so syncedSeq must have
// caught up to writeSeq — see assertDurableAtAck).
func TestDenseRestoreInsertFamilyDurableAtAck(t *testing.T) {
	cs := newCollectionStore(t)
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	c, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing")
	}
	vec := make([]float32, 16)
	nowMs := c.idx.(*hnsw).now()

	// RestoreInsert — the WAL-replay / reshard-backfill primitive.
	vec[1] = 1
	if err := c.RestoreInsert(1, vec, 0, Metadata{"src": NewInt(1)}, nil, nil, 7); err != nil {
		t.Fatalf("RestoreInsert: %v", err)
	}
	assertDurableAtAck(t, c.wal, "RestoreInsert")

	// RestoreInsertAt — the stamped-apply (replicated) variant.
	vec[2] = 2
	if err := c.RestoreInsertAt(2, vec, 0, Metadata{"src": NewInt(2)}, nil, nil, 9, nowMs); err != nil {
		t.Fatalf("RestoreInsertAt: %v", err)
	}
	assertDurableAtAck(t, c.wal, "RestoreInsertAt")

	// InsertIfAbsentVersion — the ONLINE reshard copy primitive (real insert).
	vec[3] = 3
	inserted, err := c.InsertIfAbsentVersion(3, vec, 0, Metadata{"src": NewInt(3)}, nil, nil, 11)
	if err != nil || !inserted {
		t.Fatalf("InsertIfAbsentVersion: inserted=%v err=%v", inserted, err)
	}
	assertDurableAtAck(t, c.wal, "InsertIfAbsentVersion")

	// InsertIfAbsentVersionAt — the stamped variant (real insert).
	vec[4] = 4
	inserted, err = c.InsertIfAbsentVersionAt(4, vec, 0, Metadata{"src": NewInt(4)}, nil, nil, 13, nowMs)
	if err != nil || !inserted {
		t.Fatalf("InsertIfAbsentVersionAt: inserted=%v err=%v", inserted, err)
	}
	assertDurableAtAck(t, c.wal, "InsertIfAbsentVersionAt")

	// The if-absent NO-OP path stages nothing and returns seq 0; commitWaitStaged(0)
	// must be a no-op wait, not a hang or a spurious error.
	inserted, err = c.InsertIfAbsentVersion(3, vec, 0, nil, nil, nil, 99)
	if err != nil {
		t.Fatalf("InsertIfAbsentVersion no-op: %v", err)
	}
	if inserted {
		t.Fatal("InsertIfAbsentVersion clobbered a live point")
	}
	assertDurableAtAck(t, c.wal, "InsertIfAbsentVersion no-op")

	// The versions must still be the VERBATIM ones (the staged conversion must not
	// have disturbed the version-preserving contract these paths exist for).
	for id, want := range map[uint64]uint64{1: 7, 2: 9, 3: 11, 4: 13} {
		_, _, _, _, version, ok := c.Get(id)
		if !ok {
			t.Fatalf("point %d missing", id)
		}
		if version != want {
			t.Fatalf("point %d version = %d, want the verbatim %d", id, version, want)
		}
	}
}

// TestMVRestoreAddFamilyDurableAtAck is the MV analogue: the three multi-vector
// add paths that were still fsyncing under opMu.
func TestMVRestoreAddFamilyDurableAtAck(t *testing.T) {
	cs := newCollectionStore(t)
	if err := cs.CreateMultiVector("mv", mvWALConfig()); err != nil {
		t.Fatal(err)
	}
	idx, ok := cs.GetMultiVector("mv")
	if !ok {
		t.Fatal("MV index missing")
	}

	// AddIfAbsent — the plain if-absent path (logs version 1).
	inserted, err := idx.AddIfAbsent(1, [][]float32{{1, 0, 0, 0}}, Metadata{"src": NewInt(1)})
	if err != nil || !inserted {
		t.Fatalf("AddIfAbsent: inserted=%v err=%v", inserted, err)
	}
	assertDurableAtAck(t, idx.wal, "MV AddIfAbsent")

	// The AddIfAbsent NO-OP path stages nothing (seq 0).
	inserted, err = idx.AddIfAbsent(1, [][]float32{{9, 9, 9, 9}}, nil)
	if err != nil {
		t.Fatalf("AddIfAbsent no-op: %v", err)
	}
	if inserted {
		t.Fatal("AddIfAbsent clobbered a live doc")
	}
	assertDurableAtAck(t, idx.wal, "MV AddIfAbsent no-op")

	// MultiAddIfAbsentVersionSparse — the ONLINE MV reshard copy primitive.
	sparse := &SparseVector{Indices: []uint32{3}, Values: []float32{0.5}}
	inserted, err = idx.MultiAddIfAbsentVersionSparse(2, [][]float32{{0, 1, 0, 0}}, Metadata{"src": NewInt(2)}, nil, 21, sparse)
	if err != nil || !inserted {
		t.Fatalf("MultiAddIfAbsentVersionSparse: inserted=%v err=%v", inserted, err)
	}
	assertDurableAtAck(t, idx.wal, "MV MultiAddIfAbsentVersionSparse")

	// MultiRestoreAddSparse — the OFFLINE MV resplit backfill primitive.
	if err := idx.MultiRestoreAddSparse(3, [][]float32{{0, 0, 1, 0}}, Metadata{"src": NewInt(3)}, nil, 31, nil); err != nil {
		t.Fatalf("MultiRestoreAddSparse: %v", err)
	}
	assertDurableAtAck(t, idx.wal, "MV MultiRestoreAddSparse")

	// MultiRestoreAdd is a replace-add: re-running it over a LIVE doc must still
	// stage + wait (this is the resplit-retry shape).
	if err := idx.MultiRestoreAdd(3, [][]float32{{0, 0, 1, 1}}, Metadata{"src": NewInt(33)}, nil, 32); err != nil {
		t.Fatalf("MultiRestoreAdd replace: %v", err)
	}
	assertDurableAtAck(t, idx.wal, "MV MultiRestoreAdd replace")

	// Verbatim versions preserved through the conversion.
	for docID, want := range map[uint64]uint64{1: 1, 2: 21, 3: 32} {
		_, _, got, ok := idx.Get(docID)
		if !ok {
			t.Fatalf("doc %d missing", docID)
		}
		if got != want {
			t.Fatalf("doc %d version = %d, want the verbatim %d", docID, got, want)
		}
	}
}

// TestRestoreInsertGroupCommitEffective is the headline effectiveness test for the
// dense restore lane: N concurrent RestoreInserts must coalesce into far fewer
// than N fsyncs. Before the staged conversion this was exactly N — every
// RestoreInsert fsynced while holding opMu, so no two ever overlapped.
func TestRestoreInsertGroupCommitEffective(t *testing.T) {
	const n = 200
	cs := newCollectionStore(t)
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	c, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing")
	}

	var syncs atomic.Int64
	c.wal.onSync = func() { syncs.Add(1) }

	var wg sync.WaitGroup
	wg.Add(n)
	var failed atomic.Int64
	for i := 1; i <= n; i++ {
		go func(id uint64) {
			defer wg.Done()
			vec := make([]float32, 16)
			vec[id%16] = float32(id)
			if err := c.RestoreInsert(id, vec, 0, Metadata{"src": NewInt(int64(id))}, nil, nil, id+1000); err != nil {
				failed.Add(1)
				t.Errorf("RestoreInsert(%d): %v", id, err)
			}
		}(uint64(i))
	}
	wg.Wait()
	if failed.Load() != 0 {
		t.Fatalf("%d RestoreInserts failed", failed.Load())
	}

	// Durability half: all n ops have returned, so every staged record must be
	// covered by a completed sync. Without this, "drop the commit-wait" wins here.
	assertDurableAtAck(t, c.wal, "after the concurrent RestoreInsert storm")

	// Every point must have landed with its VERBATIM version.
	for i := 1; i <= n; i++ {
		_, _, _, _, version, ok := c.Get(uint64(i))
		if !ok {
			t.Fatalf("point %d vanished during the restore storm", i)
		}
		if want := uint64(i) + 1000; version != want {
			t.Fatalf("point %d version = %d, want the verbatim %d", i, version, want)
		}
	}

	got := syncs.Load()
	if got >= n {
		t.Fatalf("group-commit ineffective for RestoreInsert: %d fsyncs for %d concurrent restores (want << %d)", got, n, n)
	}
	if got == 0 {
		t.Fatal("no syncs recorded — durability hook broken")
	}
	t.Logf("RestoreInsert group-commit: %d concurrent restores coalesced into %d fsync(s)", n, got)
}

// TestMVAddGroupCommitEffective is the MV analogue, driving MultiRestoreAddSparse
// (the offline resplit backfill — the highest-volume MV writer of the three).
func TestMVAddGroupCommitEffective(t *testing.T) {
	const n = 150
	cs := newCollectionStore(t)
	if err := cs.CreateMultiVector("mv", mvWALConfig()); err != nil {
		t.Fatal(err)
	}
	idx, ok := cs.GetMultiVector("mv")
	if !ok {
		t.Fatal("MV index missing")
	}

	var syncs atomic.Int64
	idx.wal.onSync = func() { syncs.Add(1) }

	var wg sync.WaitGroup
	wg.Add(n)
	var failed atomic.Int64
	for i := 1; i <= n; i++ {
		go func(docID uint64) {
			defer wg.Done()
			tokens := [][]float32{{float32(docID), 0, 0, 0}, {0, float32(docID), 0, 0}}
			if err := idx.MultiRestoreAddSparse(docID, tokens, Metadata{"src": NewInt(int64(docID))}, nil, docID+500, nil); err != nil {
				failed.Add(1)
				t.Errorf("MultiRestoreAddSparse(%d): %v", docID, err)
			}
		}(uint64(i))
	}
	wg.Wait()
	if failed.Load() != 0 {
		t.Fatalf("%d MV restore-adds failed", failed.Load())
	}

	assertDurableAtAck(t, idx.wal, "after the concurrent MV restore-add storm")

	for i := 1; i <= n; i++ {
		_, _, version, ok := idx.Get(uint64(i))
		if !ok {
			t.Fatalf("doc %d vanished during the MV restore-add storm", i)
		}
		if want := uint64(i) + 500; version != want {
			t.Fatalf("doc %d version = %d, want the verbatim %d", i, version, want)
		}
	}

	got := syncs.Load()
	if got >= n {
		t.Fatalf("group-commit ineffective for MV restore-add: %d fsyncs for %d concurrent adds (want << %d)", got, n, n)
	}
	if got == 0 {
		t.Fatal("no syncs recorded — durability hook broken")
	}
	t.Logf("MV restore-add group-commit: %d concurrent adds coalesced into %d fsync(s)", n, got)
}

// TestRestoreInsertSurvivesReopen is the end-to-end durability claim behind the
// staged split: the restore/reshard writers ack only once their bytes are on
// disk, so a reopen that replays the (never-flushed) WAL recovers every acked
// point with its verbatim version. A missing commit-wait is not guaranteed to
// show here (the OS may have flushed anyway), which is exactly why
// assertDurableAtAck above pins the invariant directly — this test pins the
// user-visible contract on top of it.
func TestRestoreInsertSurvivesReopen(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	c, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing")
	}
	const n = 40
	for i := 1; i <= n; i++ {
		vec := make([]float32, 16)
		vec[i%16] = float32(i)
		if err := c.RestoreInsert(uint64(i), vec, 0, Metadata{"src": NewInt(int64(i))}, nil, nil, uint64(i)+100); err != nil {
			t.Fatalf("RestoreInsert(%d): %v", i, err)
		}
	}
	// NO Flush: the WAL alone must carry these.
	if err := cs.Close(); err != nil {
		t.Fatal(err)
	}

	reopened, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reopened.Close() })
	c2, ok := reopened.Get("docs")
	if !ok {
		t.Fatal("collection missing after reopen")
	}
	for i := 1; i <= n; i++ {
		_, meta, _, _, version, ok := c2.Get(uint64(i))
		if !ok {
			t.Fatalf("point %d lost across reopen — an acked restore was not durable", i)
		}
		if want := uint64(i) + 100; version != want {
			t.Fatalf("point %d version = %d after replay, want the verbatim %d", i, version, want)
		}
		if got := meta["src"]; !got.Equal(NewInt(int64(i))) {
			t.Fatalf("point %d payload = %v after replay, want src=%d", i, meta, i)
		}
	}
}

// TestOpMuReleasedOnPanicRestoreInsert extends the B1 panic regression to the
// restore lane: RestoreInsert now holds opMu inside a closure so the WAL write can
// be staged and waited on outside it. A panic escaping without the deferred Unlock
// would silently deadlock every later write on the collection (the server recovers
// per-request panics, so the leaked lock would never be noticed at the panic site).
func TestOpMuReleasedOnPanicRestoreInsert(t *testing.T) {
	c := seedWALCollection(t, 2)
	c.idx = newPanicOnceIndex(c.idx, "RestoreInsert") // same-package test-only swap

	vec := make([]float32, 16)
	func() {
		defer func() {
			if recover() == nil {
				t.Error("expected the forced panic to propagate")
			}
		}()
		_ = c.RestoreInsert(50, vec, 0, nil, nil, nil, 3)
	}()

	// The next write must not block on a leaked opMu.
	mustNotDeadlock(t, 5*time.Second, func() error {
		return c.RestoreInsert(51, vec, 0, nil, nil, nil, 4)
	})
}

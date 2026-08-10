// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Group-commit effectiveness for the DELETE and PAYLOAD lanes. The staged
// commit-wait originally landed for insert/upsert only, which left every
// delete and every payload mutation fsyncing while holding opMu: a
// delete-heavy workload degenerated to one serialized fsync per op, AND the
// deleting writer became the fsync leader with the collection's op lock in
// hand, stalling concurrent staged inserts behind it. These tests pin the fix
// the same way wal_group_commit_test.go pins the append core: N concurrent ops
// must complete in far fewer than N fsyncs.

// seedWALCollection creates a WAL-mode collection holding ids 1..n and returns
// it. The seeding inserts fsync; callers install their onSync counter AFTER
// this returns so only the op under test is measured.
func seedWALCollection(t *testing.T, n int) *Collection {
	t.Helper()
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cs.Close() })
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	c, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing")
	}
	vec := make([]float32, 16)
	for i := 1; i <= n; i++ {
		vec[i%16] = float32(i)
		if err := c.Insert(uint64(i), vec, 0, Metadata{"seed": NewInt(int64(i))}, nil); err != nil {
			t.Fatalf("seed Insert(%d): %v", i, err)
		}
	}
	return c
}

// TestDeleteGroupCommitEffective is the delete-lane analogue of
// TestCollectionGroupCommitEffective: N concurrent Deletes must coalesce into
// far fewer than N fsyncs. Before the staged conversion this was exactly N
// (every delete fsynced under opMu, so no two ever overlapped).
func TestDeleteGroupCommitEffective(t *testing.T) {
	const n = 200
	c := seedWALCollection(t, n)

	var syncs atomic.Int64
	c.wal.onSync = func() { syncs.Add(1) }

	var wg sync.WaitGroup
	wg.Add(n)
	var removed atomic.Int64
	for i := 1; i <= n; i++ {
		go func(id uint64) {
			defer wg.Done()
			if c.Delete(id) {
				removed.Add(1)
			}
		}(uint64(i))
	}
	wg.Wait()

	if got := removed.Load(); got != n {
		t.Fatalf("removed %d of %d points — the conversion dropped deletes", got, n)
	}
	// Fewer fsyncs is only a win if the acks were still durable: all n ops have
	// returned, so every record they staged must be covered by a completed sync.
	// Without this, "delete the commit-wait entirely" would score BEST here.
	assertDurableAtAck(t, c.wal, "after the concurrent delete storm")
	got := syncs.Load()
	if got >= n {
		t.Fatalf("group-commit ineffective for Delete: %d fsyncs for %d concurrent deletes (want << %d)", got, n, n)
	}
	if got == 0 {
		t.Fatal("no syncs recorded — durability hook broken")
	}
	t.Logf("Delete group-commit: %d concurrent deletes coalesced into %d fsync(s)", n, got)
}

// TestSetPayloadGroupCommitEffective is the payload-lane analogue. All four
// payload mutators (SetPayload / OverwritePayload / DeletePayloadKeys /
// ClearPayload, plus their CAS and At variants) funnel through
// Collection.payloadOpCAS, so covering SetPayload covers the conversion for
// every one of them.
func TestSetPayloadGroupCommitEffective(t *testing.T) {
	const n = 200
	c := seedWALCollection(t, n)

	var syncs atomic.Int64
	c.wal.onSync = func() { syncs.Add(1) }

	var wg sync.WaitGroup
	wg.Add(n)
	for i := 1; i <= n; i++ {
		go func(id uint64) {
			defer wg.Done()
			if err := c.SetPayload(id, Metadata{"patched": NewInt(int64(id))}, nil); err != nil {
				t.Errorf("SetPayload(%d): %v", id, err)
			}
		}(uint64(i))
	}
	wg.Wait()

	// Fewer fsyncs is only a win if the acks were still durable — see the delete
	// test. This is what stops "remove the commit-wait" from scoring best here.
	assertDurableAtAck(t, c.wal, "after the concurrent payload storm")

	// Every patch must have landed — a staged write that was never waited on (or
	// a lost apply) would show up here before it shows up as a durability bug.
	for i := 1; i <= n; i++ {
		_, meta, _, _, _, ok := c.Get(uint64(i))
		if !ok {
			t.Fatalf("point %d vanished during the payload storm", i)
		}
		if got := meta["patched"]; !got.Equal(NewInt(int64(i))) {
			t.Fatalf("point %d payload = %v, want patched=%d", i, meta, i)
		}
	}

	got := syncs.Load()
	if got >= n {
		t.Fatalf("group-commit ineffective for SetPayload: %d fsyncs for %d concurrent payload writes (want << %d)", got, n, n)
	}
	if got == 0 {
		t.Fatal("no syncs recorded — durability hook broken")
	}
	t.Logf("SetPayload group-commit: %d concurrent payload writes coalesced into %d fsync(s)", n, got)
}

// TestOpMuReleasedOnPanicDelete is the B1 regression test extended to the
// delete lane. DeleteCAS now holds opMu inside a closure (so the WAL write can
// be staged and waited on outside), and the server recovers per-request panics
// — a panic that escaped without the deferred Unlock would silently deadlock
// every later write on the collection.
func TestOpMuReleasedOnPanicDelete(t *testing.T) {
	c := seedWALCollection(t, 2)
	c.idx = newPanicOnceIndex(c.idx, "Delete") // same-package test-only swap

	func() {
		defer func() {
			if recover() == nil {
				t.Error("expected the forced panic to propagate")
			}
		}()
		c.Delete(1)
	}()

	// The next write must not block on a leaked opMu.
	mustNotDeadlock(t, 5*time.Second, func() error {
		if !c.Delete(2) {
			return nil // liveness is what matters here, not the removal result
		}
		return nil
	})
}

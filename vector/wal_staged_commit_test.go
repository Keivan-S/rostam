// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Tests for the Fix-2/Fix-3 split-commit optimization: production callers
// (Collection.InsertCASKeyTTL / UpsertCASKeyTTL) now hold opMu across
// {apply + WAL WRITE} only, releasing it BEFORE the durability wait
// (commitWaitStaged). wal_group_commit_test.go's own tests drive appendFramed
// directly (no opMu) to prove the underlying group-commit machinery is correct;
// these tests prove the production call path actually reaches it, and that
// releasing opMu before the wait is safe against a concurrent Flush rotating
// the WAL out from under a still-pending commit-wait.

// TestStagedInsertSurvivesFlushRace forces the exact interleaving Fix 2 makes
// possible: an insert's WRITE phase (apply + WAL append) completes and opMu is
// released, and ONLY THEN does a Flush run to completion (snapshot + WAL
// truncate) — entirely BEFORE the insert's own commit-wait executes. The
// insert's commitWaitStaged must still return with no error (truncate advances
// syncedSeq past this insert's seq, satisfying it — see wal.go's truncate doc),
// and the point must survive: live immediately, and via the checkpoint Flush
// took after a full reopen.
func TestStagedInsertSurvivesFlushRace(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	c, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing")
	}

	vec := make([]float32, 16)
	for i := range vec {
		vec[i] = float32(i + 1)
	}
	normalize(vec)

	// Reproduce InsertCASKeyTTL's write phase manually so the Flush race can be
	// inserted BETWEEN the WAL write and the durability wait.
	c.opMu.Lock()
	version, keyExpires, err := c.idx.Insert(42, vec, 0, nil, nil, nil, CASCond{})
	if err != nil {
		c.opMu.Unlock()
		t.Fatalf("idx.Insert: %v", err)
	}
	seq, err := c.wal.appendInsertStaged(42, vec, 0, nil, nil, keyExpires, version)
	c.opMu.Unlock()
	if err != nil {
		t.Fatalf("appendInsertStaged: %v", err)
	}

	// Flush runs to completion — snapshot + truncate — entirely before the
	// commit-wait below. Its snapshot is guaranteed to include this insert (opMu
	// ordering: Flush could only acquire opMu after our WRITE phase released it,
	// by which point idx.Insert had already applied).
	if err := cs.Flush("docs"); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	done := make(chan error, 1)
	go func() { done <- c.wal.commitWaitStaged(seq) }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("commitWaitStaged after a racing Flush: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("commitWaitStaged hung after a racing Flush truncated the WAL — stranded waiter")
	}

	if !c.Exists(42) {
		t.Fatal("inserted point missing from the live index after the race")
	}
	_ = cs.Close()

	// Reopen: the point must survive via the checkpoint Flush captured (the WAL
	// tail is empty — truncate already rotated it away).
	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = cs2.Close() }()
	c2, ok := cs2.Get("docs")
	if !ok {
		t.Fatal("collection missing after reopen")
	}
	if !c2.Exists(42) {
		t.Fatal("inserted point lost across a reopen after the write/flush/wait race")
	}
}

// TestStagedInsertConcurrentWithFlush stresses the same race with real
// goroutine concurrency: many Collection.Insert calls (the full production
// path) run alongside a goroutine that calls Flush in a tight loop. Every
// Insert must return nil (no stranded commit-wait, no lost ack) regardless of
// how the scheduler interleaves the inserts' commit-waits with Flush's
// truncate calls, and every id must be recoverable after a reopen.
func TestStagedInsertConcurrentWithFlush(t *testing.T) {
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

	const n = 100
	base := make([]float32, 16)
	for i := range base {
		base[i] = float32(i + 1)
	}

	var wg sync.WaitGroup
	wg.Add(n)
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		go func(id uint64) {
			defer wg.Done()
			v := append([]float32(nil), base...)
			v[0] += float32(id)
			normalize(v)
			if err := c.Insert(id, v, 0, nil, nil); err != nil {
				errs <- err
			}
		}(uint64(i + 1))
	}

	stop := make(chan struct{})
	var flushWG sync.WaitGroup
	flushWG.Add(1)
	go func() {
		defer flushWG.Done()
		for {
			select {
			case <-stop:
				return
			default:
				_ = cs.Flush("docs")
				time.Sleep(time.Millisecond)
			}
		}
	}()

	wg.Wait()
	close(stop)
	flushWG.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("Insert racing a concurrent Flush: %v", err)
	}

	if err := cs.Flush("docs"); err != nil { // final checkpoint so everything is captured
		t.Fatalf("final Flush: %v", err)
	}
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = cs2.Close() }()
	c2, ok := cs2.Get("docs")
	if !ok {
		t.Fatal("collection missing after reopen")
	}
	if got := c2.Stats().Size; got != n {
		t.Errorf("recovered size = %d, want %d (a racing Flush stranded or lost an insert)", got, n)
	}
}

// TestCollectionGroupCommitEffective proves the Fix-2 payoff through the
// PRODUCTION call path (Collection.Insert), not wal_group_commit_test.go's
// direct appendFramed drivers: concurrent inserts now overlap in
// commitWaitStaged (opMu is held only across apply+WRITE), so group-commit
// actually coalesces many inserts into far fewer fsyncs.
func TestCollectionGroupCommitEffective(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	c, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing")
	}

	var syncs atomic.Int64
	c.wal.onSync = func() { syncs.Add(1) }

	const n = 200
	base := make([]float32, 16)
	for i := range base {
		base[i] = float32(i + 1)
	}

	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(id uint64) {
			defer wg.Done()
			v := append([]float32(nil), base...)
			v[0] += float32(id)
			normalize(v)
			if err := c.Insert(id, v, 0, nil, nil); err != nil {
				t.Errorf("Insert(%d): %v", id, err)
			}
		}(uint64(i + 1))
	}
	wg.Wait()

	got := syncs.Load()
	if got >= n {
		t.Fatalf("group-commit ineffective at the Collection level: %d fsyncs for %d concurrent inserts (want << %d)", got, n, n)
	}
	if got == 0 {
		t.Fatalf("no syncs recorded — durability hook broken")
	}
	t.Logf("Collection-level group-commit: %d concurrent inserts coalesced into %d fsync(s)", n, got)
}

// TestUpsertSingleFsync proves Fix 3: an Upsert (delete-then-insert replace)
// now costs ONE fsync, not two — the replace-delete's WAL write is staged (no
// wait of its own) and its durability is subsumed by the following insert's
// single commit-wait (the delete's bytes precede the insert's in the file,
// written under the same opMu, so a Sync covering the insert's seq covers the
// delete's too).
func TestUpsertSingleFsync(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	c, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing")
	}

	vec := make([]float32, 16)
	for i := range vec {
		vec[i] = float32(i + 1)
	}
	normalize(vec)
	if err := c.Insert(1, vec, 0, nil, nil); err != nil {
		t.Fatalf("initial Insert: %v", err)
	}

	var syncs atomic.Int64
	c.wal.onSync = func() { syncs.Add(1) }

	vec2 := append([]float32(nil), vec...)
	vec2[0]++
	normalize(vec2)
	if err := c.Upsert(1, vec2, "", 0, nil, nil); err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	if got := syncs.Load(); got != 1 {
		t.Fatalf("Upsert fsync count = %d, want 1 (delete-then-insert should share one commit-wait)", got)
	}
}

// TestCollectionGroupCommitEffectiveAt is TestCollectionGroupCommitEffective
// against InsertCASKeyTTLAt — the leader-apply-stamped variant ops/builtin.go
// routes every replicated insert through (tx.applyStamped). It shares
// insertBody/insertLockedAt with the wall-clock path at the engine layer, but
// Collection.InsertCASKeyTTLAt is its OWN function and needed its own staged
// conversion; this proves that conversion actually batches fsyncs too.
func TestCollectionGroupCommitEffectiveAt(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	c, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing")
	}

	var syncs atomic.Int64
	c.wal.onSync = func() { syncs.Add(1) }

	const n = 200
	base := make([]float32, 16)
	for i := range base {
		base[i] = float32(i + 1)
	}
	const nowMs int64 = 1_700_000_000_000

	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(id uint64) {
			defer wg.Done()
			v := append([]float32(nil), base...)
			v[0] += float32(id)
			normalize(v)
			if _, err := c.InsertCASKeyTTLAt(id, v, 0, nil, nil, nil, CASCond{}, nowMs); err != nil {
				t.Errorf("InsertCASKeyTTLAt(%d): %v", id, err)
			}
		}(uint64(i + 1))
	}
	wg.Wait()

	got := syncs.Load()
	if got >= n {
		t.Fatalf("group-commit ineffective for InsertCASKeyTTLAt: %d fsyncs for %d concurrent inserts (want << %d)", got, n, n)
	}
	if got == 0 {
		t.Fatalf("no syncs recorded — durability hook broken")
	}
	t.Logf("InsertCASKeyTTLAt group-commit: %d concurrent inserts coalesced into %d fsync(s)", n, got)
}

// TestUpsertSingleFsyncAt is TestUpsertSingleFsync against UpsertCASKeyTTLAt —
// the stamped replicated-apply upsert path, which independently stages its
// replace-delete (appendDeleteStaged, no wait) and folds it into the
// following insert's single commit-wait.
func TestUpsertSingleFsyncAt(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	c, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing")
	}

	vec := make([]float32, 16)
	for i := range vec {
		vec[i] = float32(i + 1)
	}
	normalize(vec)
	const nowMs int64 = 1_700_000_000_000
	if _, err := c.InsertCASKeyTTLAt(1, vec, 0, nil, nil, nil, CASCond{}, nowMs); err != nil {
		t.Fatalf("initial InsertCASKeyTTLAt: %v", err)
	}

	var syncs atomic.Int64
	c.wal.onSync = func() { syncs.Add(1) }

	vec2 := append([]float32(nil), vec...)
	vec2[0]++
	normalize(vec2)
	if _, err := c.UpsertCASKeyTTLAt(1, vec2, "", 0, nil, nil, nil, CASCond{}, nowMs+1); err != nil {
		t.Fatalf("UpsertCASKeyTTLAt: %v", err)
	}

	if got := syncs.Load(); got != 1 {
		t.Fatalf("UpsertCASKeyTTLAt fsync count = %d, want 1 (delete-then-insert should share one commit-wait)", got)
	}
}

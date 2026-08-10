// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard/pbisr"
)

// ============================================================================
// DURABLE FRONTIER, end to end through a real Store.
//
// The shard's DATA already survived a restart (the cache warm-restarts from its
// mmap pages). What did not survive was the POSITION: the PB engine came back
// with every watermark at zero and reported a genesis frontier over a full FSM.
// These tests exercise the whole seam — engine → stamper → cache header → reopen
// → engine — against the one rule that governs it: the persisted watermark may
// UNDER-report freely, and must NEVER over-report.
// ============================================================================

// pbFrontierStore builds a single-node PB Store on dir with a stamper tuned by
// stampEvery, and returns it alongside the engine the PB branch registered.
// (testing.TB, not *testing.T, so the stall benchmark can reuse it.)
func pbFrontierStore(t testing.TB, dir string, stampEvery int) (*Store, pbisr.Receiver) {
	t.Helper()
	reg := ops.NewRegistry()
	ops.RegisterBuiltins(reg)
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	cc.DataDir = filepath.Join(dir, "cache") // mmap mode: there is a header to stamp

	var eng pbisr.Receiver
	cfg := Config{
		NodeID:               "n1",
		DataDir:              dir,
		Cache:                cc,
		Ops:                  reg,
		ReplicationMode:      ReplicationModePB,
		PBControl:            fakeControl{node: "n1"},
		PBFrontierStampEvery: stampEvery,
		PBRegister:           func(_ int, r pbisr.Receiver) { eng = r },
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New(pb): %v", err)
	}
	return s, eng
}

func pbKey(i int) []byte { return []byte(fmt.Sprintf("k%05d", i)) }

// putN writes n keys through the PB primary. The i-th write (0-based) is assigned
// seq i+1, so "frontier F" means exactly "keys 0..F-1 are applied".
func putN(t *testing.T, s *Store, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		if err := s.Put(pbKey(i), []byte("v"), 0); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}
}

// TestPBStoreRestoresFrontierAcrossCleanRestart is the round trip the stage is
// for: apply writes, close, reopen — the engine's frontier must describe the FSM
// it warm-restarted, not genesis. A CLEAN close stamps the final frontier
// synchronously, so the restored value here is EXACT.
func TestPBStoreRestoresFrontierAcrossCleanRestart(t *testing.T) {
	dir := t.TempDir()
	const n = 40

	s, eng := pbFrontierStore(t, dir, 8)
	if info := eng.CatchupInfo(); info.FrontierSeq != 0 {
		t.Fatalf("fresh store frontier = %d, want 0", info.FrontierSeq)
	}
	putN(t, s, n)
	if info := eng.CatchupInfo(); info.FrontierSeq != n {
		t.Fatalf("pre-restart frontier = %d, want %d", info.FrontierSeq, n)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	s2, eng2 := pbFrontierStore(t, dir, 8)
	defer func() { _ = s2.Close() }()

	info := eng2.CatchupInfo()
	if info.FrontierSeq != n {
		t.Fatalf("restored frontier = %d, want %d — a clean shutdown must persist the EXACT frontier", info.FrontierSeq, n)
	}
	if info.FrontierEpoch != 1 {
		t.Fatalf("restored frontier epoch = %d, want 1 — a seq without its epoch is a position, not an identity", info.FrontierEpoch)
	}
	// The data it describes really is there (this is what makes the claim truthful
	// rather than merely non-zero).
	for i := 0; i < n; i++ {
		if _, err := s2.Get(pbKey(i)); err != nil {
			t.Fatalf("restored frontier claims %d writes but key %d is missing: %v", n, i, err)
		}
	}
	// And the engine continues from there rather than restarting the sequence.
	if err := s2.Put(pbKey(n), []byte("v"), 0); err != nil {
		t.Fatalf("post-restart Put: %v", err)
	}
	if got := eng2.CatchupInfo().FrontierSeq; got != n+1 {
		t.Fatalf("frontier after the first post-restart write = %d, want %d", got, n+1)
	}
}

// TestPBFrontierNeverOverReportsAfterUncleanStop is the core safety assertion.
//
// The crash is modelled DETERMINISTICALLY: the stamper's flusher is killed
// (killPBStamper — no final flush, which is what Store.Close would do) and then a
// second batch of writes is applied. Those writes materialize and reach disk
// normally, but no stamp can possibly cover them, so the persisted watermark
// provably trails the FSM by a known amount. Timing-dependent variants of this
// test can accidentally stamp the true frontier and assert nothing.
//
// The assertion is the invariant, not a specific number: every write the restored
// frontier CLAIMS must be present in the reloaded cache. Trailing is fine (it only
// costs catch-up work); leading would mean the node advertises a prefix it does not
// hold, and pbisr's log matching — which compares incoming frames against this very
// number — would then certify a divergent append.
func TestPBFrontierNeverOverReportsAfterUncleanStop(t *testing.T) {
	dir := t.TempDir()
	const (
		stamped   = 100 // written while the stamper was alive
		unstamped = 100 // written after it died — unreachable by any stamp
	)

	s, _ := pbFrontierStore(t, dir, 8)
	putN(t, s, stamped)
	killPBStamper(s)
	for i := stamped; i < stamped+unstamped; i++ {
		if err := s.Put(pbKey(i), []byte("v"), 0); err != nil {
			t.Fatalf("post-kill Put %d: %v", i, err)
		}
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	s2, eng2 := pbFrontierStore(t, dir, 8)
	defer func() { _ = s2.Close() }()
	restored := eng2.CatchupInfo().FrontierSeq

	// THE INVARIANT: everything the watermark claims is really there.
	for i := 0; i < int(restored); i++ {
		if _, err := s2.Get(pbKey(i)); err != nil {
			t.Fatalf("OVER-REPORT: restored frontier is %d but key %d (seq %d) is not in the reloaded cache: %v",
				restored, i, i+1, err)
		}
	}
	// ...and it provably trails: nothing written after the stamper died can be
	// covered, so the watermark cannot have reached the true frontier.
	if restored > stamped {
		t.Fatalf("restored frontier %d exceeds the %d writes that were made while the stamper was alive — OVER-REPORT",
			restored, stamped)
	}
	// Teeth: a stamper that never stamped anything would satisfy the loop above
	// vacuously.
	if restored == 0 {
		t.Fatalf("nothing was ever stamped after %d writes — the over-report check is vacuous", stamped)
	}
	// The FSM really does hold more than the watermark admits — this is the
	// under-report the design accepts, and pbisr's grow path is what closes it (see
	// TestRestoredUnderReportIsCaughtUpByGrow).
	if _, err := s2.Get(pbKey(stamped + unstamped - 1)); err != nil {
		t.Fatalf("setup: the post-kill writes did not survive the restart: %v", err)
	}
}

// killPBStamper stops the frontier stamper the way a crash does: the flusher
// goroutine goes away and NO final stamp is written. Store.Close's own
// s.pbFrontier.Close() would flush the exact frontier, which is the clean-shutdown
// path and the opposite of what this models.
func killPBStamper(s *Store) {
	st := s.pbFrontier
	if st == nil {
		return
	}
	s.pbFrontier = nil
	st.stopOnce.Do(func() {
		close(st.stop)
		st.wg.Wait()
	})
}

// TestPBFrontierAmortisationIsBounded: the stamp fires on the COUNT trigger, so
// the under-report a crash can produce is bounded by the knob rather than by luck.
// (The time trigger is the other half; it is what covers a shard that goes idle
// mid-burst, and it is exercised implicitly by every other test here.)
func TestPBFrontierAmortisationIsBounded(t *testing.T) {
	dir := t.TempDir()
	const (
		n     = 100
		every = 10
	)
	// A long interval so ONLY the count trigger can fire during the writes.
	reg := ops.NewRegistry()
	ops.RegisterBuiltins(reg)
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	cc.DataDir = filepath.Join(dir, "cache")
	s, err := New(Config{
		NodeID:                  "n1",
		DataDir:                 dir,
		Cache:                   cc,
		Ops:                     reg,
		ReplicationMode:         ReplicationModePB,
		PBControl:               fakeControl{node: "n1"},
		PBFrontierStampEvery:    every,
		PBFrontierStampInterval: time.Hour,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	putN(t, s, n)
	// The flusher is asynchronous and COALESCES wakeups, so the stamp we can assert
	// on is "eventually caught up", not "exactly at write k". What matters is that
	// the count trigger drives it forward at all without a clean shutdown.
	deadline := time.Now().Add(2 * time.Second)
	var seq uint64
	for time.Now().Before(deadline) {
		if seq, _ = s.cache.PBFrontier(); seq > 0 {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if seq == 0 {
		t.Fatalf("the count trigger never stamped after %d writes with PBFrontierStampEvery=%d", n, every)
	}
	if seq > n {
		t.Fatalf("stamped frontier %d exceeds the %d writes made", seq, n)
	}
	killPBStamper(s)
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

// TestRaftModeLeavesPBFrontierUnstamped guards the "Raft mode must be
// byte-identical" requirement at the observable end: no PB stamper is built, the
// header's PB field stays at genesis across a restart, and the Raft applied-index
// path is untouched by any of it.
func TestRaftModeLeavesPBFrontierUnstamped(t *testing.T) {
	dir := t.TempDir()
	reg := ops.NewRegistry()
	ops.RegisterBuiltins(reg)
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	cc.DataDir = filepath.Join(dir, "cache")
	cfg := Config{
		NodeID:    "n1",
		DataDir:   dir,
		Cache:     cc,
		Ops:       reg,
		Bootstrap: true,
		// ReplicationMode empty ⇒ Raft.
		RaftHeartbeatMs:    50,
		RaftElectionMs:     50,
		SnapshotIntervalMs: 300_000,
		SnapshotThreshold:  1 << 30,
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New(raft): %v", err)
	}
	if s.pbFrontier != nil {
		t.Fatal("Raft mode built a PB frontier stamper")
	}
	waitLeader(t, s)
	for i := 0; i < 20; i++ {
		if err := s.Put(pbKey(i), []byte("v"), 0); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}
	if seq, epoch := s.cache.PBFrontier(); seq != 0 || epoch != 0 {
		t.Fatalf("Raft-mode writes stamped a PB frontier (%d,%d)", seq, epoch)
	}
	if s.cache.AppliedIndex() == 0 {
		t.Fatal("setup: the Raft applied-index path did not advance, so this proves nothing about it")
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	s2, err := New(cfg)
	if err != nil {
		t.Fatalf("reopen(raft): %v", err)
	}
	defer func() { _ = s2.Close() }()
	if seq, epoch := s2.cache.PBFrontier(); seq != 0 || epoch != 0 {
		t.Fatalf("reopened Raft shard reports a PB frontier (%d,%d), want genesis", seq, epoch)
	}
}

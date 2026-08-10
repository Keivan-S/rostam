// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/rostamlabs/rostam/shard/pbisr"
)

// ============================================================================
// The SHARD-side snapshot store: the real FSM serialize/install pair,
// the durable poison fence, and the durable-frontier hand-off.
//
// The engine-level tests (shard/pbisr) prove the PROTOCOL. These prove the three
// things only the real store can be wrong about:
//
//	(1) the fence is genuinely durable and fail-safe on a corrupt/partial marker;
//	(2) an install really WIPES — a pre-install key must not survive as a ghost;
//	(3) CommitInstall defeats the amortised stamper, which still holds the
//	    PRE-install pending pair and would otherwise flush it straight back over
//	    the installed frontier.
// ============================================================================

// --- (1) the fence ------------------------------------------------------------

// TestPBFenceRoundTripAndDurability: the marker survives being written by one
// fence object and read by another (the process-restart shape), and lowering it
// is likewise durable.
func TestPBFenceRoundTripAndDurability(t *testing.T) {
	dir := t.TempDir()
	f := newPBRestoreFence(dir)

	if _, _, up := f.raised(); up {
		t.Fatal("a fresh dir must have no fence raised")
	}
	if err := f.raise(4242, 7); err != nil {
		t.Fatalf("raise: %v", err)
	}
	// A DIFFERENT fence object over the same dir — i.e. after a restart.
	seq, epoch, up := newPBRestoreFence(dir).raised()
	if !up || seq != 4242 || epoch != 7 {
		t.Fatalf("fence after restart = (%d,%d,%v), want (4242,7,true)", seq, epoch, up)
	}
	if err := f.lower(); err != nil {
		t.Fatalf("lower: %v", err)
	}
	if _, _, up := newPBRestoreFence(dir).raised(); up {
		t.Fatal("a lowered fence must stay lowered across a restart")
	}
	// Lowering an absent fence is idempotent (the abort paths rely on it).
	if err := f.lower(); err != nil {
		t.Fatalf("lower (idempotent): %v", err)
	}
}

// TestPBFenceCorruptMarkerStillFencesEverything is the fail-safe direction. A
// truncated or corrupt marker means we cannot say WHICH install was in progress —
// but its mere existence already proves one WAS. Treating it as "no fence" would
// let a half-wiped shard back into the ISR, so the unreadable case must fence
// exactly as hard as the readable one.
func TestPBFenceCorruptMarkerStillFencesEverything(t *testing.T) {
	for name, body := range map[string][]byte{
		"truncated":   []byte("short"),
		"bad magic":   make([]byte, pbFenceSize),
		"bad crc":     append(append([]byte(nil), validFenceBytes(t, 1, 1)[:pbFenceSize-1]...), 0xFF),
		"empty":       {},
		"oversized":   make([]byte, pbFenceSize+9),
		"almost-good": validFenceBytes(t, 5, 5)[:pbFenceSize-2],
	} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, pbFenceFile), body, 0o600); err != nil {
				t.Fatalf("write marker: %v", err)
			}
			if _, _, up := newPBRestoreFence(dir).raised(); !up {
				t.Fatal("an unreadable marker must still report a RAISED fence")
			}
		})
	}
}

func validFenceBytes(t *testing.T, seq, epoch uint64) []byte {
	t.Helper()
	dir := t.TempDir()
	f := newPBRestoreFence(dir)
	if err := f.raise(seq, epoch); err != nil {
		t.Fatalf("raise: %v", err)
	}
	b, err := os.ReadFile(f.path())
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	return b
}

// TestPBFenceNoDataDirIsNoOp: with no DataDir there is nothing durable to protect
// (a crash loses the whole FSM and the node restarts honestly empty), so the fence
// is a no-op and the engine's IN-MEMORY poison latch carries the property alone.
func TestPBFenceNoDataDirIsNoOp(t *testing.T) {
	f := newPBRestoreFence("")
	if err := f.raise(1, 1); err != nil {
		t.Fatalf("raise on no-datadir must be a silent no-op, got %v", err)
	}
	if _, _, up := f.raised(); up {
		t.Fatal("a no-datadir fence can never report raised")
	}
	if err := f.lower(); err != nil {
		t.Fatalf("lower on no-datadir: %v", err)
	}
}

// --- (2) + (3) the real store, end to end -------------------------------------

// pbSnapStoreFor reaches the pbSnapshotStore a real PB Store built, by rebuilding
// one over the same cache/vectors/fsm/stamper. It is the same object graph
// store.go wires into the engine.
func pbSnapStoreFor(s *Store) *pbSnapshotStore {
	return newPBSnapshotStore(s.cache, s.vectors, s.fsm, s.pbFrontier, newPBRestoreFence(s.cfg.DataDir))
}

// TestPBSnapshotStoreInstallWipesAndStamps is the whole shard-side contract in one
// pass: serialize a source FSM, install it over a DIFFERENT target FSM, and
// require (a) the target's pre-install keys are GONE, (b) the source's keys are
// present, and (c) the durable PB frontier names the installed identity.
//
// (a) is the load-bearing one. restoreSnapshot wipes with DURABLE TOMBSTONES
// precisely so a pre-install entry cannot survive on the page and be re-indexed by
// the next warm-restart rebuild — a ghost that could even out-rank the restored
// copy. A transfer that merged instead of wiping would silently re-create the
// divergence it was sent to repair.
func TestPBSnapshotStoreInstallWipesAndStamps(t *testing.T) {
	srcDir, dstDir := t.TempDir(), t.TempDir()
	src, _ := pbFrontierStore(t, srcDir, 0)
	defer func() { _ = src.Close() }()
	dst, dstEng := pbFrontierStore(t, dstDir, 0)
	defer func() { _ = dst.Close() }()

	// Divergent target state: keys the source has never heard of.
	for i := 0; i < 12; i++ {
		if err := dst.Put([]byte("ghost"+string(rune('a'+i))), []byte("stale"), 0); err != nil {
			t.Fatalf("seed target: %v", err)
		}
	}
	putN(t, src, 30)

	const fseq, fepoch = uint64(30), uint64(1)
	blob, err := pbSnapStoreFor(src).SnapshotFSM(fseq)
	if err != nil {
		t.Fatalf("SnapshotFSM: %v", err)
	}

	target := pbSnapStoreFor(dst)
	if err := target.BeginInstall(fseq, fepoch); err != nil {
		t.Fatalf("BeginInstall: %v", err)
	}
	if !target.InstallPending() {
		t.Fatal("the fence must be raised between Begin and Commit")
	}
	if err := target.InstallFSM(blob); err != nil {
		t.Fatalf("InstallFSM: %v", err)
	}
	if err := target.CommitInstall(fseq, fepoch); err != nil {
		t.Fatalf("CommitInstall: %v", err)
	}
	if target.InstallPending() {
		t.Fatal("a completed install must lower the fence")
	}

	// (a) every pre-install key is GONE.
	for i := 0; i < 12; i++ {
		k := "ghost" + string(rune('a'+i))
		if v, err := dst.Get([]byte(k)); err == nil {
			t.Fatalf("pre-install key survived the wipe: %q -> %q", k, v)
		}
	}
	// (b) the source's whole key set is present.
	for i := 0; i < 30; i++ {
		if _, err := dst.Get(pbKey(i)); err != nil {
			t.Fatalf("installed snapshot is missing source key %s: %v", pbKey(i), err)
		}
	}
	// (c) the DURABLE frontier names the installed identity.
	if seq, epoch := dst.cache.PBFrontier(); seq != fseq || epoch != fepoch {
		t.Fatalf("durable frontier after install = (%d,%d), want (%d,%d)", seq, epoch, fseq, fepoch)
	}
	_ = dstEng
}

// TestPBSnapshotCommitDefeatsStaleStamperFlush pins the resurrection hazard the
// amortised stamper creates, and which a plain SetPBFrontier would not survive.
//
// After an install the stamper STILL HOLDS the PRE-install pending pair. Its very
// next tick would flush that back over the value the install just wrote. On the
// target this path exists to repair — a DIVERGED node — the stale pair is HIGHER
// than the installed frontier, so the resurrection is an OVER-report: the node
// would claim a prefix its wiped FSM does not hold, and log matching, which
// compares an incoming frame against exactly this number, would then certify a
// divergent append.
func TestPBSnapshotCommitDefeatsStaleStamperFlush(t *testing.T) {
	dir := t.TempDir()
	s, _ := pbFrontierStore(t, dir, 0)
	defer func() { _ = s.Close() }()

	// Drive the frontier high — this is the "diverged, ahead" state.
	putN(t, s, 50)
	s.pbFrontier.flush()
	if seq, _ := s.cache.PBFrontier(); seq != 50 {
		t.Fatalf("setup: durable frontier = %d, want 50", seq)
	}
	// Simulate a still-pending advance the stamper has recorded but not flushed.
	s.pbFrontier.record(77, 1)

	// An install REGRESSES the frontier to the (lower) transferred identity.
	if err := pbSnapStoreFor(s).CommitInstall(12, 3); err != nil {
		t.Fatalf("CommitInstall: %v", err)
	}
	if seq, epoch := s.cache.PBFrontier(); seq != 12 || epoch != 3 {
		t.Fatalf("post-install durable frontier = (%d,%d), want (12,3)", seq, epoch)
	}

	// THE ASSERTION: every subsequent flush is a no-op. Without installFrontier's
	// rebase, the stamper's stale pend (77) would land here and over-report.
	for i := 0; i < 5; i++ {
		s.pbFrontier.flush()
	}
	if seq, epoch := s.cache.PBFrontier(); seq != 12 || epoch != 3 {
		t.Fatalf("a stale stamper flush RESURRECTED the pre-install frontier: (%d,%d), want (12,3)", seq, epoch)
	}
}

// TestPBStorePoisonedEngineRefusesToServe wires the whole thing through a real
// Store: a raised fence at open must produce an engine that refuses to serve and
// declares itself unverifiable to the failover gate.
func TestPBStorePoisonedEngineRefusesToServe(t *testing.T) {
	dir := t.TempDir()
	s, eng := pbFrontierStore(t, dir, 0)
	putN(t, s, 10)
	if err := s.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if !eng.CatchupInfo().OK {
		t.Fatal("setup: a healthy engine must report OK")
	}

	// A crash mid-install: the fence file is on disk when the shard reopens.
	if err := newPBRestoreFence(dir).raise(999, 4); err != nil {
		t.Fatalf("raise fence: %v", err)
	}
	s2, eng2 := pbFrontierStore(t, dir, 0)
	defer func() { _ = s2.Close() }()

	info := eng2.CatchupInfo()
	if info.OK {
		t.Fatal("a shard reopened with a raised fence must declare itself UNVERIFIABLE")
	}
	if info.FrontierSeq != 0 || info.AppliedSeq != 0 {
		t.Fatalf("a poisoned engine must not advertise watermarks: %+v", info)
	}
	e, ok := eng2.(*pbisr.Engine)
	if !ok {
		t.Fatalf("registered receiver is %T, want *pbisr.Engine", eng2)
	}
	if !e.Poisoned() {
		t.Fatal("the engine must seed its poison latch from the durable fence")
	}
	if e.LeaseValid() {
		t.Fatal("a poisoned engine must report no valid lease (refuse to serve)")
	}
	// And the write path refuses at the seam.
	if err := s2.Put([]byte("nope"), []byte("v"), 0); err == nil {
		t.Fatal("a poisoned shard must refuse writes")
	}
}

// --- the MEASURED write stall -------------------------------------------------

// BenchmarkPBSnapshotStall measures what this stage's flow-control choice costs:
// SnapshotFSM runs under writeMu+e.mu, which excludes BOTH Applier.Apply sites,
// so a shard's write path is FROZEN for exactly this long.
//
// It is a benchmark rather than an assertion because the number is a DEPLOYMENT
// input, not an invariant — an operator reads it to pick a shard-size ceiling, and
// SnapshotStats.StallMaxNs reports the same quantity in production. Run as:
//
//	go test ./shard/ -run xxx -bench PBSnapshotStall -benchtime 3x
func BenchmarkPBSnapshotStall(b *testing.B) {
	for _, keys := range []int{1_000, 10_000, 100_000, 500_000} {
		b.Run(fmt.Sprintf("keys=%d", keys), func(b *testing.B) {
			dir := b.TempDir()
			s, _ := pbFrontierStore(b, dir, 0)
			defer func() { _ = s.Close() }()
			val := bytes.Repeat([]byte("v"), 100)
			for i := 0; i < keys; i++ {
				if err := s.Put(pbKey(i), val, 0); err != nil {
					b.Fatalf("put %d: %v", i, err)
				}
			}
			ss := pbSnapStoreFor(s)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				blob, err := ss.SnapshotFSM(uint64(keys))
				if err != nil {
					b.Fatalf("SnapshotFSM: %v", err)
				}
				b.SetBytes(int64(len(blob)))
			}
		})
	}
}

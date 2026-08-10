// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"reflect"
	"testing"

	"github.com/hashicorp/raft"
)

// applyShardEpoch is a test helper that encodes and applies an OpSetShardEpoch
// entry, failing the test if Apply reports an error (a benign no-op returns nil).
func applyShardEpoch(t *testing.T, f *MetaFSM, shardID int, epoch uint64, primary string) {
	t.Helper()
	data, err := encodeLogEntry(LogEntry{Op: OpSetShardEpoch, ShardID: shardID, Epoch: epoch, Primary: primary})
	if err != nil {
		t.Fatal(err)
	}
	if resp := f.Apply(&raft.Log{Data: data}); resp != nil {
		t.Fatalf("apply SetShardEpoch(shard=%d,epoch=%d): %v", shardID, epoch, resp)
	}
}

// applyShardISR is a test helper that encodes and applies an OpSetShardISR entry.
func applyShardISR(t *testing.T, f *MetaFSM, shardID int, epoch uint64, isr []string) {
	t.Helper()
	data, err := encodeLogEntry(LogEntry{Op: OpSetShardISR, ShardID: shardID, Epoch: epoch, ISR: isr})
	if err != nil {
		t.Fatal(err)
	}
	if resp := f.Apply(&raft.Log{Data: data}); resp != nil {
		t.Fatalf("apply SetShardISR(shard=%d,epoch=%d): %v", shardID, epoch, resp)
	}
}

// applyShardSeed is a test helper that encodes and applies an ATOMIC seed: one
// OpSetShardEpoch entry carrying (epoch, primary, full ISR).
func applyShardSeed(t *testing.T, f *MetaFSM, shardID int, epoch uint64, primary string, isr []string) {
	t.Helper()
	data, err := encodeLogEntry(LogEntry{Op: OpSetShardEpoch, ShardID: shardID, Epoch: epoch, Primary: primary, ISR: isr})
	if err != nil {
		t.Fatal(err)
	}
	if resp := f.Apply(&raft.Log{Data: data}); resp != nil {
		t.Fatalf("apply SetShardSeed(shard=%d,epoch=%d): %v", shardID, epoch, resp)
	}
}

// TestShardEpochSeedInstallsFullISRAtomically is the FSM-level pin for the
// acked-write-loss fix. ONE OpSetShardEpoch entry carrying an ISR must land the
// epoch, the primary AND the full ISR in a single apply — there is no
// intermediate state in which the epoch is visible but the ISR is still the
// singleton {primary}. It also pins the complement: an entry WITHOUT an ISR (a
// FAILOVER promotion) still resets the ISR to {primary}, which is deliberate.
func TestShardEpochSeedInstallsFullISRAtomically(t *testing.T) {
	f := NewMetaFSM()
	full := []string{"n1", "n2", "n3"}

	applyShardSeed(t, f, 0, 1, "n1", full)
	if got := f.ShardEpoch(0); got != 1 {
		t.Fatalf("epoch = %d, want 1", got)
	}
	if got := f.ShardPrimary(0); got != "n1" {
		t.Fatalf("primary = %q, want n1", got)
	}
	if got := f.ShardISR(0); !reflect.DeepEqual(got, full) {
		t.Fatalf("ISR = %v, want %v — the seed must install the FULL ISR in the same apply as the epoch", got, full)
	}

	// The entry's ISR slice must NOT alias into FSM state.
	alias := []string{"n1", "n2"}
	applyShardSeed(t, f, 1, 1, "n1", alias)
	alias[1] = "MUTATED"
	if got := f.ShardISR(1); got[1] == "MUTATED" {
		t.Fatalf("FSM ISR aliases the caller's slice: %v", got)
	}

	// FAILOVER (no ISR carried): the reset to {primary} is intended and preserved.
	applyShardEpoch(t, f, 0, 2, "n2")
	if got := f.ShardISR(0); !reflect.DeepEqual(got, []string{"n2"}) {
		t.Fatalf("post-failover ISR = %v, want [n2] — the promotion reset must be unchanged", got)
	}

	// A malformed seed whose ISR excludes the named primary falls back to the
	// {primary} reset rather than installing an ISR the primary is not in.
	applyShardSeed(t, f, 0, 3, "n3", []string{"n1", "n2"})
	if got := f.ShardISR(0); !reflect.DeepEqual(got, []string{"n3"}) {
		t.Fatalf("malformed-seed ISR = %v, want [n3] (fallback)", got)
	}
}

// TestLogEntryShardEpochRoundtrip mirrors TestLogEntryReshardRoundtrip: an
// OpSetShardEpoch LogEntry must round-trip its Epoch/Primary/ShardID payload.
func TestLogEntryShardEpochRoundtrip(t *testing.T) {
	in := LogEntry{Op: OpSetShardEpoch, ShardID: 7, Epoch: 42, Primary: "n3"}
	b, err := encodeLogEntry(in)
	if err != nil {
		t.Fatal(err)
	}
	got, err := decodeLogEntry(b)
	if err != nil {
		t.Fatal(err)
	}
	if got.Op != OpSetShardEpoch || got.ShardID != 7 || got.Epoch != 42 || got.Primary != "n3" {
		t.Fatalf("round-trip = %+v, want SetShardEpoch shard=7 epoch=42 primary=n3", got)
	}
}

// TestLogEntryShardISRRoundtrip mirrors TestLogEntryShardEpochRoundtrip for the
// ISR payload: an OpSetShardISR LogEntry must round-trip its Epoch/ISR/ShardID.
func TestLogEntryShardISRRoundtrip(t *testing.T) {
	in := LogEntry{Op: OpSetShardISR, ShardID: 5, Epoch: 9, ISR: []string{"n1", "n2", "n3"}}
	b, err := encodeLogEntry(in)
	if err != nil {
		t.Fatal(err)
	}
	got, err := decodeLogEntry(b)
	if err != nil {
		t.Fatal(err)
	}
	if got.Op != OpSetShardISR || got.ShardID != 5 || got.Epoch != 9 {
		t.Fatalf("round-trip = %+v, want SetShardISR shard=5 epoch=9", got)
	}
	if !reflect.DeepEqual(got.ISR, in.ISR) {
		t.Fatalf("ISR round-trip = %v, want %v", got.ISR, in.ISR)
	}
}

// TestFSMApplyShardEpochMonotonic proves the monotonicity guard: a higher epoch
// updates (and resets ISR to {primary}); a lower or equal epoch is a benign
// no-op that never regresses the epoch/primary/ISR.
func TestFSMApplyShardEpochMonotonic(t *testing.T) {
	f := NewMetaFSM()

	// First epoch establishes primary n1 and resets ISR to {n1}.
	applyShardEpoch(t, f, 3, 1, "n1")
	if e := f.ShardEpoch(3); e != 1 {
		t.Fatalf("ShardEpoch(3) = %d, want 1", e)
	}
	if p := f.ShardPrimary(3); p != "n1" {
		t.Fatalf("ShardPrimary(3) = %q, want n1", p)
	}
	if isr := f.ShardISR(3); !reflect.DeepEqual(isr, []string{"n1"}) {
		t.Fatalf("ShardISR(3) = %v, want [n1]", isr)
	}

	// Grow the ISR at the current epoch, then bump the epoch: a new epoch RESETS
	// the ISR to just the new primary.
	applyShardISR(t, f, 3, 1, []string{"n1", "n2", "n3"})
	if isr := f.ShardISR(3); !reflect.DeepEqual(isr, []string{"n1", "n2", "n3"}) {
		t.Fatalf("ShardISR(3) after grow = %v, want [n1 n2 n3]", isr)
	}
	applyShardEpoch(t, f, 3, 2, "n2")
	if e := f.ShardEpoch(3); e != 2 {
		t.Fatalf("ShardEpoch(3) = %d, want 2 after bump", e)
	}
	if p := f.ShardPrimary(3); p != "n2" {
		t.Fatalf("ShardPrimary(3) = %q, want n2 after bump", p)
	}
	if isr := f.ShardISR(3); !reflect.DeepEqual(isr, []string{"n2"}) {
		t.Fatalf("ShardISR(3) after epoch bump = %v, want [n2] (reset to primary)", isr)
	}

	// A LOWER epoch is a no-op: nothing regresses.
	applyShardEpoch(t, f, 3, 1, "n1")
	if e := f.ShardEpoch(3); e != 2 {
		t.Fatalf("ShardEpoch(3) = %d after stale epoch, want 2 (no-op)", e)
	}
	if p := f.ShardPrimary(3); p != "n2" {
		t.Fatalf("ShardPrimary(3) = %q after stale epoch, want n2 (no-op)", p)
	}

	// An EQUAL epoch is also a no-op even with a different primary: nothing changes.
	applyShardEpoch(t, f, 3, 2, "n9")
	if p := f.ShardPrimary(3); p != "n2" {
		t.Fatalf("ShardPrimary(3) = %q after equal epoch, want n2 (no-op)", p)
	}
}

// TestFSMApplyShardISREpochGuard proves the epoch guard: an ISR update whose
// epoch does not match the shard's current epoch is a no-op; a matching epoch
// updates the ISR set. Defends against a stale primary mutating ISR (H3/H6).
func TestFSMApplyShardISREpochGuard(t *testing.T) {
	f := NewMetaFSM()
	applyShardEpoch(t, f, 0, 5, "n1") // epoch 5, ISR reset to {n1}

	// Matching epoch updates the ISR.
	applyShardISR(t, f, 0, 5, []string{"n1", "n2"})
	if isr := f.ShardISR(0); !reflect.DeepEqual(isr, []string{"n1", "n2"}) {
		t.Fatalf("ShardISR(0) = %v, want [n1 n2] at matching epoch", isr)
	}

	// A STALE (lower) epoch is a no-op — the fenced primary cannot mutate ISR.
	applyShardISR(t, f, 0, 4, []string{"n1", "n2", "n3", "n4"})
	if isr := f.ShardISR(0); !reflect.DeepEqual(isr, []string{"n1", "n2"}) {
		t.Fatalf("ShardISR(0) = %v after stale-epoch update, want [n1 n2] (no-op)", isr)
	}

	// A FUTURE (higher) epoch that has not been established is also a no-op.
	applyShardISR(t, f, 0, 6, []string{"n5"})
	if isr := f.ShardISR(0); !reflect.DeepEqual(isr, []string{"n1", "n2"}) {
		t.Fatalf("ShardISR(0) = %v after future-epoch update, want [n1 n2] (no-op)", isr)
	}
}

// TestShardControlStateDeepCopyIsolation proves State() deep-copies the three
// shard-replication maps (including the ISR slices) so a caller mutation cannot
// alias into FSM state. Also checks the ShardISR accessor returns a copy.
func TestShardControlStateDeepCopyIsolation(t *testing.T) {
	f := NewMetaFSM()
	applyShardEpoch(t, f, 1, 3, "n1")
	applyShardISR(t, f, 1, 3, []string{"n1", "n2"})

	st := f.State()
	st.ShardEpoch[1] = 999
	st.ShardPrimary[1] = "MUTATED"
	st.ShardISR[1][0] = "MUTATED"
	st.ShardISR[1] = append(st.ShardISR[1], "n3")

	if e := f.ShardEpoch(1); e != 3 {
		t.Fatalf("ShardEpoch not deep-copied: FSM epoch = %d, want 3", e)
	}
	if p := f.ShardPrimary(1); p != "n1" {
		t.Fatalf("ShardPrimary not deep-copied: FSM primary = %q, want n1", p)
	}
	if isr := f.ShardISR(1); !reflect.DeepEqual(isr, []string{"n1", "n2"}) {
		t.Fatalf("ShardISR not deep-copied: FSM ISR = %v, want [n1 n2]", isr)
	}

	// The ShardISR accessor itself must return a copy.
	got := f.ShardISR(1)
	got[0] = "MUTATED"
	if isr := f.ShardISR(1); isr[0] != "n1" {
		t.Fatalf("ShardISR accessor aliases FSM slice: FSM ISR[0] = %q, want n1", isr[0])
	}
}

// TestStateShardControlEncodeDecodeRoundtrip mirrors
// TestStateAliasesEncodeDecodeRoundtrip: a State carrying the shard-replication
// maps must gob round-trip them, and an old-format State (no shard maps) must
// decode to nil maps with the accessors returning zero values (backward compat).
func TestStateShardControlEncodeDecodeRoundtrip(t *testing.T) {
	s := State{
		NumShards:    4,
		ShardEpoch:   map[int]uint64{0: 1, 2: 5},
		ShardPrimary: map[int]string{0: "n1", 2: "n3"},
		ShardISR:     map[int][]string{0: {"n1"}, 2: {"n3", "n1"}},
	}
	b, err := encodeState(s)
	if err != nil {
		t.Fatal(err)
	}
	got, err := decodeState(b)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.ShardEpoch, s.ShardEpoch) {
		t.Fatalf("ShardEpoch round-trip = %v, want %v", got.ShardEpoch, s.ShardEpoch)
	}
	if !reflect.DeepEqual(got.ShardPrimary, s.ShardPrimary) {
		t.Fatalf("ShardPrimary round-trip = %v, want %v", got.ShardPrimary, s.ShardPrimary)
	}
	if !reflect.DeepEqual(got.ShardISR, s.ShardISR) {
		t.Fatalf("ShardISR round-trip = %v, want %v", got.ShardISR, s.ShardISR)
	}

	// OLD-format state (no shard maps) must decode with all three nil — gob
	// handles the missing fields gracefully (no shard has an epoch/primary/ISR).
	old := State{NumShards: 4, Catalog: map[string]uint32{"default/docs": 8}}
	ob, err := encodeState(old)
	if err != nil {
		t.Fatal(err)
	}
	og, err := decodeState(ob)
	if err != nil {
		t.Fatal(err)
	}
	if og.ShardEpoch != nil || og.ShardPrimary != nil || og.ShardISR != nil {
		t.Fatalf("old-format shard maps = (%v,%v,%v), want all nil", og.ShardEpoch, og.ShardPrimary, og.ShardISR)
	}

	// The accessors on an FSM restored from the old-format state must return zero
	// values (0, "", nil) — proving nil-map safety end-to-end.
	f := NewMetaFSM()
	f.state = og
	if e := f.ShardEpoch(0); e != 0 {
		t.Fatalf("old-format ShardEpoch(0) = %d, want 0", e)
	}
	if p := f.ShardPrimary(0); p != "" {
		t.Fatalf("old-format ShardPrimary(0) = %q, want empty", p)
	}
	if isr := f.ShardISR(0); isr != nil {
		t.Fatalf("old-format ShardISR(0) = %v, want nil", isr)
	}
}

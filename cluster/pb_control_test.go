// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"fmt"
	"testing"
	"time"

	hraft "github.com/hashicorp/raft"
)

// applyMeta drives one encoded LogEntry through the FSM (test helper; mirrors
// the existing meta_fsm_test.go pattern of constructing raft.Log entries).
func applyMeta(t *testing.T, f *MetaFSM, e LogEntry) {
	t.Helper()
	data, err := encodeLogEntry(e)
	if err != nil {
		t.Fatalf("encodeLogEntry: %v", err)
	}
	if resp := f.Apply(&hraft.Log{Type: hraft.LogCommand, Data: data}); resp != nil {
		if err, ok := resp.(error); ok {
			t.Fatalf("apply %v: %v", e.Op, err)
		}
	}
}

func TestMetaControlReflectsFSM(t *testing.T) {
	f := NewMetaFSM()
	// Seed shard 3: epoch 1, primary n1, then ISR {n1,n2,n3}.
	applyMeta(t, f, LogEntry{Op: OpSetShardEpoch, ShardID: 3, Epoch: 1, Primary: "n1"})
	applyMeta(t, f, LogEntry{Op: OpSetShardISR, ShardID: 3, Epoch: 1, ISR: []string{"n1", "n2", "n3"}})

	c := newMetaControl(f, 2)

	if got := c.Epoch(3); got != 1 {
		t.Fatalf("Epoch(3): want 1, got %d", got)
	}
	if got := c.Primary(3); got != "n1" {
		t.Fatalf("Primary(3): want n1, got %q", got)
	}
	if got := c.ISR(3); len(got) != 3 || got[0] != "n1" || got[2] != "n3" {
		t.Fatalf("ISR(3): want [n1 n2 n3], got %v", got)
	}
	if got := c.MinISR(3); got != 2 {
		t.Fatalf("MinISR(3): want 2, got %d", got)
	}
	// An unseeded shard reads as empty/zero (nil-safe passthrough).
	if c.Epoch(99) != 0 || c.Primary(99) != "" || c.ISR(99) != nil {
		t.Fatalf("unseeded shard 99 must be zero-valued")
	}
}

func TestPBShardControlSeeds(t *testing.T) {
	placement := [][]string{
		{"n1", "n2", "n3"}, // shard 0
		{"n2", "n3", "n1"}, // shard 1
		nil,                // shard 2: no owners → skipped
	}
	seeds := pbShardControlSeeds(placement)
	if len(seeds) != 2 {
		t.Fatalf("want 2 seeds (shard 2 skipped), got %d", len(seeds))
	}
	if seeds[0].ShardID != 0 || seeds[0].Primary != "n1" || len(seeds[0].ISR) != 3 {
		t.Fatalf("seed0: %+v", seeds[0])
	}
	if seeds[1].ShardID != 1 || seeds[1].Primary != "n2" {
		t.Fatalf("seed1: %+v", seeds[1])
	}
}

// fakeProposer records every seed commit in an ordered call log so tests can
// assert the commit COUNT and payload, not just that something was proposed. It
// can inject an error.
type fakeProposer struct {
	calls    []string // e.g. "seed:3:n1:[n1 n2 n3]", in commit order
	failSeed bool
}

func (f *fakeProposer) ApplySetShardSeed(shardID int, epoch uint64, primary string, isr []string, _ time.Duration) error {
	if f.failSeed {
		return errors.New("not leader")
	}
	f.calls = append(f.calls, fmt.Sprintf("seed:%d:%s:%v", shardID, primary, isr))
	return nil
}

// TestBootstrapPBShardControl pins the ATOMICITY of the control-plane seed: each
// shard is seeded by EXACTLY ONE commit carrying (epoch, primary, full ISR).
//
// The count is the assertion that matters. The previous two-commit form
// (OpSetShardEpoch then OpSetShardISR) was an acked-write-loss defect: because
// OpSetShardEpoch resets the ISR to {primary}, the cluster's COMMITTED state held a
// SINGLETON ISR between the two entries, and a primary reading that intermediate
// state acked writes on itself alone — lost the moment it died. A second commit per
// shard reintroduces exactly that window, so a regression to it must fail here.
func TestBootstrapPBShardControl(t *testing.T) {
	seeds := []pbShardSeed{
		{ShardID: 0, Primary: "n1", ISR: []string{"n1", "n2"}},
		{ShardID: 1, Primary: "n2", ISR: []string{"n2", "n1"}},
	}
	p := &fakeProposer{}
	if err := bootstrapPBShardControl(p, seeds, 1, time.Second); err != nil {
		t.Fatalf("bootstrap: %v", err)
	}
	if len(p.calls) != len(seeds) {
		t.Fatalf("want exactly %d commits (ONE atomic seed per shard — a second commit per shard "+
			"reopens the singleton-ISR window), got %d: %v", len(seeds), len(p.calls), p.calls)
	}
	// Each commit must carry the shard's FULL seed ISR, not a singleton.
	for i, s := range seeds {
		want := fmt.Sprintf("seed:%d:%s:%v", s.ShardID, s.Primary, s.ISR)
		if p.calls[i] != want {
			t.Fatalf("commit %d = %q, want %q (full ISR in the SAME entry as the epoch)", i, p.calls[i], want)
		}
	}

	// A proposer error (e.g. not leader) aborts and propagates.
	if err := bootstrapPBShardControl(&fakeProposer{failSeed: true}, seeds, 1, time.Second); err == nil {
		t.Fatal("bootstrap on a non-leader proposer must return an error")
	}
}

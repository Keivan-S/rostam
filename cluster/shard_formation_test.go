// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"errors"
	"io"
	"testing"
	"time"

	"github.com/hashicorp/raft"
)

// applyFormerEntry applies one OpSetShardFormer and returns the FSM's response —
// true when this apply installed the designation, false when one already existed.
func applyFormerEntry(t *testing.T, f *MetaFSM, shardID int, node string) any {
	t.Helper()
	data, err := encodeLogEntry(LogEntry{Op: OpSetShardFormer, ShardID: shardID, Node: node})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	return f.Apply(&raft.Log{Data: data})
}

// TestShardFormerIsWriteOnce pins the property the whole mechanism rests on: the
// FIRST designation for a shard wins and every later one is refused, without
// overwriting.
//
// This is the guard that makes control-plane formation safe where a purely local
// "am I owners[0]?" rule is not. A node whose disk was wiped and which rejoins an
// ESTABLISHED cluster would still compute itself as owners[0] and, deciding
// locally, would bootstrap a rival group for a shard that already exists. Routing
// the decision through the meta log means it instead reads back someone else's
// designation, forms nothing, and joins as a follower.
func TestShardFormerIsWriteOnce(t *testing.T) {
	f := NewMetaFSM()

	if got := applyFormerEntry(t, f, 7, "n2"); got != true {
		t.Fatalf("first designation returned %v, want true (this apply installed it)", got)
	}
	if got := f.ShardFormer(7); got != "n2" {
		t.Fatalf("ShardFormer(7) = %q, want n2", got)
	}

	// A second node claiming the same shard — the rejoining-fresh-disk case.
	if got := applyFormerEntry(t, f, 7, "n3"); got != false {
		t.Errorf("second designation returned %v, want false (already formed)", got)
	}
	if got := f.ShardFormer(7); got != "n2" {
		t.Errorf("ShardFormer(7) = %q after a competing claim, want n2 — a designation must "+
			"never be overwritten, or two nodes could each believe they must form the group", got)
	}

	// Re-designating the SAME node is refused too, so a retrying seeder cannot
	// churn the log or reset state.
	if got := applyFormerEntry(t, f, 7, "n2"); got != false {
		t.Errorf("re-designating the same node returned %v, want false", got)
	}

	// Independent shards are unaffected.
	if got := applyFormerEntry(t, f, 8, "n3"); got != true {
		t.Errorf("designating a DIFFERENT shard returned %v, want true", got)
	}
	if got := f.ShardFormer(8); got != "n3" {
		t.Errorf("ShardFormer(8) = %q, want n3", got)
	}
}

// TestShardFormerRejectsMalformed pins that a designation naming no node is
// refused rather than stored — an empty former would be indistinguishable from
// "not yet designated" on read, so the seeder would loop forever believing it
// still had work while the write-once slot was already consumed.
func TestShardFormerRejectsMalformed(t *testing.T) {
	f := NewMetaFSM()
	if got := applyFormerEntry(t, f, 1, ""); got == nil {
		t.Error("empty former was accepted; want an error response")
	} else if _, isErr := got.(error); !isErr {
		t.Errorf("empty former returned %T (%v), want an error", got, got)
	}
	if got := f.ShardFormer(1); got != "" {
		t.Errorf("ShardFormer(1) = %q after a refused apply, want empty", got)
	}
	if got := applyFormerEntry(t, f, -1, "n1"); got == nil {
		t.Error("negative shard was accepted; want an error response")
	}
}

// TestShardFormerSurvivesSnapshotRoundTrip pins that the designation is part of
// the replicated snapshot. If it were not, a node restored from a snapshot would
// read "" for every shard, conclude nothing had been formed, and be free to form
// rival groups — the exact hazard the write-once rule exists to prevent.
func TestShardFormerSurvivesSnapshotRoundTrip(t *testing.T) {
	f := NewMetaFSM()
	applyFormerEntry(t, f, 0, "n1")
	applyFormerEntry(t, f, 4, "n2")

	blob, err := f.SnapshotBytes()
	if err != nil {
		t.Fatalf("SnapshotBytes: %v", err)
	}
	restored := NewMetaFSM()
	if err := restored.Restore(io.NopCloser(bytes.NewReader(blob))); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	if got := restored.ShardFormer(0); got != "n1" {
		t.Errorf("after restore ShardFormer(0) = %q, want n1", got)
	}
	if got := restored.ShardFormer(4); got != "n2" {
		t.Errorf("after restore ShardFormer(4) = %q, want n2", got)
	}
	if got := restored.ShardFormer(1); got != "" {
		t.Errorf("after restore ShardFormer(1) = %q, want empty (never designated)", got)
	}
}

// fakeFormerProposer records designations and can simulate a follower.
type fakeFormerProposer struct {
	fsm       *MetaFSM
	notLeader bool // when set, every call fails as a follower would
	calls     int
}

func (p *fakeFormerProposer) ApplySetShardFormer(shardID int, node string, _ time.Duration) (bool, error) {
	p.calls++
	if p.notLeader {
		return false, errors.New("not leader")
	}
	resp := applyFormerEntryNoT(p.fsm, shardID, node)
	if err, isErr := resp.(error); isErr {
		return false, err
	}
	won, _ := resp.(bool)
	return won, nil
}

func applyFormerEntryNoT(f *MetaFSM, shardID int, node string) any {
	data, err := encodeLogEntry(LogEntry{Op: OpSetShardFormer, ShardID: shardID, Node: node})
	if err != nil {
		return err
	}
	return f.Apply(&raft.Log{Data: data})
}

// TestSeedShardFormersDesignatesFirstOwner pins that the seeder designates
// owners[0] for every PLACED shard and leaves unplaced shards alone — including
// the shards whose owner set excludes the bootstrap node, which are precisely the
// ones that used to be left leaderless.
func TestSeedShardFormersDesignatesFirstOwner(t *testing.T) {
	fsm := NewMetaFSM()
	n := &Node{meta: &MetaRaft{FSM: fsm}}
	// The RF=2-over-3-nodes shape: shards 1, 4 and 7 have owner sets that exclude
	// n1, the node that would have carried -bootstrap.
	placement := [][]string{
		{"n1", "n2"}, {"n2", "n3"}, {"n1", "n3"}, {"n1", "n2"},
		{"n2", "n3"}, {"n1", "n3"}, {"n1", "n2"}, {"n2", "n3"},
	}
	p := &fakeFormerProposer{fsm: fsm}

	n.seedShardFormers(p, placement, nil)

	for shardID, owners := range placement {
		if got := fsm.ShardFormer(shardID); got != owners[0] {
			t.Errorf("ShardFormer(%d) = %q, want %q (owners[0])", shardID, got, owners[0])
		}
	}
	// The bug's signature shards specifically: someone must own forming them.
	for _, shardID := range []int{1, 4, 7} {
		if got := fsm.ShardFormer(shardID); got == "n1" || got == "" {
			t.Errorf("shard %d former = %q; it must be designated to one of its OWNERS "+
				"(n2/n3) — these are the shards excluding the bootstrap node that used to "+
				"stay leaderless forever", shardID, got)
		}
	}
}

// TestSeedShardFormersSkipsUnplacedShards pins that a shard with no owners is not
// designated: there is no node that could form it, and a designation naming nobody
// would consume the write-once slot.
func TestSeedShardFormersSkipsUnplacedShards(t *testing.T) {
	fsm := NewMetaFSM()
	n := &Node{meta: &MetaRaft{FSM: fsm}}
	placement := [][]string{{"n1"}, nil, {"n2"}}
	p := &fakeFormerProposer{fsm: fsm}

	n.seedShardFormers(p, placement, nil)

	if got := fsm.ShardFormer(1); got != "" {
		t.Errorf("ShardFormer(1) = %q for an UNPLACED shard, want empty", got)
	}
	if got := fsm.ShardFormer(0); got != "n1" {
		t.Errorf("ShardFormer(0) = %q, want n1", got)
	}
	if got := fsm.ShardFormer(2); got != "n2" {
		t.Errorf("ShardFormer(2) = %q, want n2", got)
	}
	if !n.allShardFormersDesignated(placement) {
		t.Error("allShardFormersDesignated = false; an unplaced shard must not block completion, " +
			"or the seeder would spin until its deadline on every cluster with a gap in placement")
	}
}

// TestSeedShardFormersStopsWhenNotLeader pins that a follower gives up its sweep
// instead of hammering meta: the designation is made by whichever node holds meta
// leadership, and the rest read the result from replicated state.
func TestSeedShardFormersStopsWhenNotLeader(t *testing.T) {
	fsm := NewMetaFSM()
	n := &Node{meta: &MetaRaft{FSM: fsm}}
	placement := [][]string{{"n1", "n2"}, {"n2", "n3"}}
	p := &fakeFormerProposer{fsm: fsm, notLeader: true}
	stop := make(chan struct{})
	close(stop) // stop immediately after the first sweep

	n.seedShardFormers(p, placement, stop)

	if p.calls != 1 {
		t.Errorf("proposer called %d times, want 1 — a follower must break its sweep on the "+
			"first refusal rather than trying every shard", p.calls)
	}
	if fsm.ShardFormer(0) != "" {
		t.Error("a follower's refused proposal still changed replicated state")
	}
}

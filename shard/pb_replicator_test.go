// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"errors"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/raft"
	"github.com/rostamlabs/rostam/shard/pbisr"
)

// fakeControl is a static single-node control plane: this node is primary of an
// ISR of size 1 (itself), min-ISR 1.
type fakeControl struct{ node string }

func (c fakeControl) Epoch(int) uint64   { return 1 }
func (c fakeControl) Primary(int) string { return c.node }
func (c fakeControl) ISR(int) []string   { return []string{c.node} }
func (c fakeControl) MinISR(int) int     { return 1 }

func TestPBReplicatorSatisfiesSeamAndApplies(t *testing.T) {
	f, c := newTestFSM(t) // newTestFSM returns (*fsm, *cache.Cache)
	ctrl := fakeControl{node: "n1"}
	var clock int64 = 10
	e := pbisr.New("n1", 0, ctrl, nil, newShardApplier(f),
		pbisr.WithClock(func() int64 { return clock }))
	e.GrantLease(1, 1_000_000) // valid well past the clock

	r := newPBReplicator("n1", 0, e, ctrl)

	if !r.IsLeader() {
		t.Fatal("primary node must report IsLeader")
	}
	if err := r.VerifyLeader(); err != nil {
		t.Fatalf("VerifyLeader on leased primary: %v", err)
	}

	entry := EncodeLogEntry("put", ops.EncodePutArgs([]byte("k"), []byte("v"), 0))
	resp, idx, err := r.ApplyIndexed(entry, 2*time.Second)
	if err != nil {
		t.Fatalf("ApplyIndexed: %v", err)
	}
	if idx != 1 {
		t.Fatalf("first write seq: want 1, got %d", idx)
	}
	ar, ok := resp.(*ApplyResponse)
	if !ok || ar.Err != nil {
		t.Fatalf("ApplyIndexed resp: ok=%v resp=%#v", ok, resp)
	}
	if got, err := c.Get([]byte("k")); err != nil || string(got) != "v" {
		t.Fatalf("state after ApplyIndexed: got=%q err=%v", got, err)
	}

	// Membership ops are unimplemented in this phase.
	if err := r.AddVoter("n2", "addr", 0, time.Second); !errors.Is(err, pbisr.ErrPBUnimplemented) {
		t.Fatalf("AddVoter: want ErrPBUnimplemented, got %v", err)
	}

	// A non-primary node maps leadership checks onto the seam's NotLeader signal.
	// (IsLeader/VerifyLeader use the pbReplicator's nodeID + Control, not the
	// engine, so reusing engine e with a different nodeID is valid here.)
	r2 := newPBReplicator("n2", 0, e, ctrl) // ctrl names n1 primary, so n2 is not primary
	if r2.IsLeader() {
		t.Fatal("non-primary node must not report IsLeader")
	}
	if err := r2.VerifyLeader(); !errors.Is(err, raft.ErrNotLeader) {
		t.Fatalf("VerifyLeader on non-primary: want raft.ErrNotLeader, got %v", err)
	}
}

// TestPBReplicatorApplyIndexedMapsLeaseExpired confirms that once the primary's
// lease has expired, ApplyIndexed maps the engine's pbisr.ErrLeaseExpired fencing
// error onto the seam's raft.ErrNotLeader — the same signal a Raft-backed Store
// uses to redirect writers. This complements the VerifyLeader-path coverage above
// by exercising the ApplyIndexed/Propose fencing path directly.
func TestPBReplicatorApplyIndexedMapsLeaseExpired(t *testing.T) {
	f, _ := newTestFSM(t) // newTestFSM returns (*fsm, *cache.Cache)
	ctrl := fakeControl{node: "n1"}
	var clock int64 = 10
	e := pbisr.New("n1", 0, ctrl, nil, newShardApplier(f),
		pbisr.WithClock(func() int64 { return clock }))
	e.GrantLease(1, 20) // valid only until clock reaches 20

	r := newPBReplicator("n1", 0, e, ctrl)

	// Advance the injected clock past the lease expiry.
	clock = 100

	entry := EncodeLogEntry("put", ops.EncodePutArgs([]byte("k"), []byte("v"), 0))
	if _, _, err := r.ApplyIndexed(entry, time.Second); !errors.Is(err, raft.ErrNotLeader) {
		t.Fatalf("ApplyIndexed on lease-expired primary: want raft.ErrNotLeader, got %v", err)
	}
}

// twoNodeControl is a static two-member control plane: n1 primary over ISR
// {n1,n2}, min-ISR 2. It gives Propose a real backup peer (n2) so a silent
// transport can wedge the pipeline.
type twoNodeControl struct{}

func (twoNodeControl) Epoch(int) uint64   { return 1 }
func (twoNodeControl) Primary(int) string { return "n1" }
func (twoNodeControl) ISR(int) []string   { return []string{"n1", "n2"} }
func (twoNodeControl) MinISR(int) int     { return 2 }

// silentTransport models a partitioned backup: it accepts every submission and
// NEVER invokes the completion callback, so each record stays in-flight until its
// Propose ctx expires. It fills the admission window without ever committing.
type silentTransport struct{}

func (silentTransport) Replicate(_ string, _ pbisr.ReplicateMsg, _ func(pbisr.AckMsg, error)) error {
	return nil
}

// TestPBReplicatorApplyIndexedSurfacesPipelineStall drives a real pipeline stall
// end-to-end through the seam and asserts the mapping: once the admission
// window is full of in-flight-but-uncommitted writes (a silent backup never
// acks), ApplyIndexed surfaces pbisr.ErrPipelineStalled UNCHANGED — the same
// non-durable/retryable class as ErrReplicationTimeout — and must NOT map it onto
// raft.ErrNotLeader (which would make the Store wrongly redirect the writer).
//
// Construction: each short-timeout ApplyIndexed applies locally, then times out
// against the silent backup, leaving the seq in flight and uncommitted (committed
// stays 0, so no window slot is ever freed). Records accumulate until the window
// (lastSeq-committed) reaches W; the next admission then refuses and Propose
// returns ErrPipelineStalled. The loop self-terminates the instant that happens,
// so it never hardcodes W (a package-private const in pbisr).
func TestPBReplicatorApplyIndexedSurfacesPipelineStall(t *testing.T) {
	f, _ := newTestFSM(t)
	ctrl := twoNodeControl{}
	var clock int64 = 10
	e := pbisr.New("n1", 0, ctrl, silentTransport{}, newShardApplier(f),
		pbisr.WithClock(func() int64 { return clock }))
	e.GrantLease(1, 1_000_000) // valid well past the clock
	r := newPBReplicator("n1", 0, e, ctrl)
	defer r.Shutdown()

	entry := EncodeLogEntry("put", ops.EncodePutArgs([]byte("k"), []byte("v"), 0))

	var stalled error
	for i := 0; i < 2000; i++ { // safety cap comfortably above W
		_, _, err := r.ApplyIndexed(entry, 5*time.Millisecond)
		if errors.Is(err, pbisr.ErrPipelineStalled) {
			stalled = err
			break
		}
		// Before the window fills, each write applies locally then times out
		// against the silent backup — a non-durable in-flight tail, NOT a
		// leadership change.
		if !errors.Is(err, pbisr.ErrReplicationTimeout) {
			t.Fatalf("fill write %d: err = %v, want ErrReplicationTimeout while the window has room", i, err)
		}
		if errors.Is(err, raft.ErrNotLeader) {
			t.Fatalf("fill write %d: a replication timeout must not map to raft.ErrNotLeader", i)
		}
	}

	if !errors.Is(stalled, pbisr.ErrPipelineStalled) {
		t.Fatalf("window never surfaced ErrPipelineStalled through ApplyIndexed; got %v", stalled)
	}
	// The whole point of the seam mapping: a stalled pipeline is retryable/unknown,
	// NOT a leadership redirect.
	if errors.Is(stalled, raft.ErrNotLeader) {
		t.Fatal("ErrPipelineStalled must surface unchanged, NOT mapped to raft.ErrNotLeader")
	}
}

// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

// --- Group codec -------------------------------------------------------------

func TestGroupCodecRoundTrip(t *testing.T) {
	// Record 0's predecessor is at an OLDER epoch (6) than the group's (7) — the
	// promoted-primary shape — so it exercises the one chain link the wire carries
	// explicitly; records 1..n rebuild theirs from the group's uniform epoch.
	msgs := []ReplicateMsg{
		{Epoch: 7, Seq: 41, PrevSeq: 40, PrevEpoch: 6, Data: []byte("a")},
		{Epoch: 7, Seq: 42, PrevSeq: 41, PrevEpoch: 7, Data: nil}, // empty record survives
		{Epoch: 7, Seq: 43, PrevSeq: 42, PrevEpoch: 7, Data: []byte("ccc")},
	}
	b := encodeReplicateGroup(nil, msgs)
	got, err := decodeReplicateGroup(b)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) != len(msgs) {
		t.Fatalf("decoded %d records, want %d", len(got), len(msgs))
	}
	for i := range msgs {
		if got[i].Epoch != msgs[i].Epoch || got[i].Seq != msgs[i].Seq ||
			got[i].PrevSeq != msgs[i].PrevSeq || got[i].PrevEpoch != msgs[i].PrevEpoch {
			t.Fatalf("record %d header mismatch: got %+v want %+v", i, got[i], msgs[i])
		}
		if !bytes.Equal(got[i].Data, msgs[i].Data) {
			t.Fatalf("record %d data mismatch: got %q want %q", i, got[i].Data, msgs[i].Data)
		}
	}
}

func TestGroupCodecRejectsCorrupt(t *testing.T) {
	good := encodeReplicateGroup(nil, []ReplicateMsg{
		{Epoch: 1, Seq: 1, PrevSeq: 0, PrevEpoch: 0, Data: []byte("x")},
		{Epoch: 1, Seq: 2, PrevSeq: 1, PrevEpoch: 1, Data: []byte("y")},
	})
	cases := map[string][]byte{
		"short header":   good[:20],
		"truncated data": good[:len(good)-1],
		"trailing bytes": append(append([]byte(nil), good...), 0),
	}
	for name, b := range cases {
		if _, err := decodeReplicateGroup(b); err == nil {
			t.Fatalf("%s: decode succeeded, want error", name)
		}
	}
	// A zero/oversized declared count must be rejected before allocating.
	zero := append([]byte(nil), good...)
	for i := pbGroupHdrSize - 4; i < pbGroupHdrSize; i++ {
		zero[i] = 0
	}
	if _, err := decodeReplicateGroup(zero); err == nil {
		t.Fatal("zero count: decode succeeded, want error")
	}
}

// --- Backup ReceiveGroup ------------------------------------------------------

func TestReceiveGroupAppliesAllAndAcksCumulative(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	backup := c.engines["n2"]

	msgs := []ReplicateMsg{
		{Epoch: 1, Seq: 1, PrevSeq: 0, PrevEpoch: 0, Data: []byte("a")},
		{Epoch: 1, Seq: 2, PrevSeq: 1, PrevEpoch: 1, Data: []byte("b")},
		{Epoch: 1, Seq: 3, PrevSeq: 2, PrevEpoch: 1, Data: []byte("c")},
	}
	ack := backup.ReceiveGroup(msgs)
	if !ack.OK || ack.Epoch != 1 || ack.Seq != 3 {
		t.Fatalf("ack = %+v, want OK epoch=1 seq=3", ack)
	}
	if got := backup.LastApplied(); got != 3 {
		t.Fatalf("LastApplied = %d, want 3", got)
	}
	if got := c.appliers["n2"].count(); got != 3 {
		t.Fatalf("applied %d records, want 3", got)
	}
}

func TestReceiveGroupGapAcksPrefix(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	backup := c.engines["n2"]

	// Records 1,2 chain; record 3 claims PrevSeq=3 (a gap) and must be refused.
	msgs := []ReplicateMsg{
		{Epoch: 1, Seq: 1, PrevSeq: 0, PrevEpoch: 0, Data: []byte("a")},
		{Epoch: 1, Seq: 2, PrevSeq: 1, PrevEpoch: 1, Data: []byte("b")},
		{Epoch: 1, Seq: 4, PrevSeq: 3, PrevEpoch: 1, Data: []byte("d")},
	}
	ack := backup.ReceiveGroup(msgs)
	if ack.OK || ack.Epoch != 1 || ack.Seq != 2 {
		t.Fatalf("ack = %+v, want !OK epoch=1 seq=2 (prefix)", ack)
	}
	if got := backup.LastApplied(); got != 2 {
		t.Fatalf("LastApplied = %d, want 2 (prefix only)", got)
	}
}

func TestReceiveGroupStaleEpochNoCredit(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 5, []string{"n1", "n2"}, 2)
	backup := c.engines["n2"]
	backup.AdoptEpoch(5)

	msgs := []ReplicateMsg{{Epoch: 4, Seq: 1, PrevSeq: 0, Data: []byte("stale")}}
	ack := backup.ReceiveGroup(msgs)
	if ack.OK || ack.Seq != 0 {
		t.Fatalf("ack = %+v, want !OK seq=0 (no credit for a stale epoch)", ack)
	}
	if got := c.appliers["n2"].count(); got != 0 {
		t.Fatalf("applied %d records, want 0", got)
	}
}

// --- Sender-side grouping ----------------------------------------------------

// gatedGroupTransport wires a primary to ONE backup engine like inMemTransport,
// but blocks the FIRST single-message submit until release is closed, so
// concurrent Proposes pile up in the peer queue and the sender's greedy drain
// provably forms a group (>1 records) on the next submission. Group acks come
// from the backup's real ReceiveGroup unless ackFn overrides them.
type gatedGroupTransport struct {
	backup  *Engine
	release chan struct{}
	ackFn   func(msgs []ReplicateMsg) (AckMsg, error) // optional group-ack override

	mu         sync.Mutex
	gated      bool // first Replicate already gated?
	groups     []int
	groupBytes []int // cumulative Data bytes per group, index-aligned with groups
	singles    int
}

func (g *gatedGroupTransport) Replicate(peer string, msg ReplicateMsg, done func(AckMsg, error)) error {
	g.mu.Lock()
	first := !g.gated
	g.gated = true
	g.singles++
	g.mu.Unlock()
	if first {
		<-g.release
	}
	done(g.backup.Receive(msg), nil)
	return nil
}

func (g *gatedGroupTransport) ReplicateGroup(peer string, msgs []ReplicateMsg, done func(AckMsg, error)) error {
	var nbytes int
	for i := range msgs {
		nbytes += len(msgs[i].Data)
	}
	g.mu.Lock()
	g.groups = append(g.groups, len(msgs))
	g.groupBytes = append(g.groupBytes, nbytes)
	g.mu.Unlock()
	if g.ackFn != nil {
		ack, err := g.ackFn(msgs)
		if err != nil {
			return err
		}
		done(ack, nil)
		return nil
	}
	done(g.backup.ReceiveGroup(msgs), nil)
	return nil
}

func (g *gatedGroupTransport) maxGroup() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	max := 0
	for _, n := range g.groups {
		if n > max {
			max = n
		}
	}
	return max
}

// newGatedPair builds a primary engine over a gatedGroupTransport and a real
// backup engine, sharing one control plane and clock.
func newGatedPair(t *testing.T) (*Engine, *gatedGroupTransport, *cluster) {
	t.Helper()
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	g := &gatedGroupTransport{backup: c.engines["n2"], release: make(chan struct{})}
	primary := New("n1", testShard, c.ctrl, g, c.appliers["n1"], WithClock(c.clk.now))
	primary.GrantLease(1, t0+leaseDur)
	t.Cleanup(primary.Shutdown)
	return primary, g, c
}

// proposeN fires n concurrent Proposes and returns their errors indexed by seq
// (seq-1 → error), plus a per-call hard failure on unexpected seq collisions.
func proposeN(t *testing.T, e *Engine, n int) []error {
	t.Helper()
	errs := make([]error, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_, seq, err := e.Propose(ctx, []byte(fmt.Sprintf("op-%d", i)))
			if seq == 0 || int(seq) > n {
				errs[i] = fmt.Errorf("unexpected seq %d", seq)
				return
			}
			errs[seq-1] = err
		}(i)
	}
	wg.Wait()
	return errs
}

func TestSenderGroupsQueuedWritesAndAllCommit(t *testing.T) {
	primary, g, c := newGatedPair(t)

	done := make(chan []error, 1)
	go func() { done <- proposeN(t, primary, 8) }()

	// Wait until the first (gated) single submit has been made and the other 7
	// writes are queued behind it, then release the gate.
	eventually(t, func() bool { return primary.LastSeq() == 8 }, "8 writes sequenced")
	close(g.release)

	errs := <-done
	for seq1, err := range errs {
		if err != nil {
			t.Fatalf("seq %d: Propose err = %v, want nil", seq1+1, err)
		}
	}
	if got := primary.Committed(); got != 8 {
		t.Fatalf("Committed = %d, want 8", got)
	}
	if g.maxGroup() < 2 {
		t.Fatalf("no group submission formed (groups=%v singles=%d) — grouping never engaged", g.groups, g.singles)
	}
	if got := c.engines["n2"].LastApplied(); got != 8 {
		t.Fatalf("backup LastApplied = %d, want 8", got)
	}
}

// TestSenderGroupRespectsByteCap pins the pbGroupBytesMax liveness bound: with
// values large enough that pbGroupBatchMax of them would exceed the receiver's
// frame limit, every group the sender forms must stay within the byte cap (a
// too-big group frame would be rejected wholesale by the backup and flap the
// link). All writes must still commit.
func TestSenderGroupRespectsByteCap(t *testing.T) {
	primary, g, _ := newGatedPair(t)
	const val = 300 << 10 // 300 KiB: 4+ records would breach the 1 MiB cap
	big := make([]byte, val)

	const n = 8
	errs := make([]error, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_, _, err := primary.Propose(ctx, big)
			errs[i] = err
		}(i)
	}
	eventually(t, func() bool { return primary.LastSeq() == n }, "writes sequenced")
	close(g.release)
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("propose %d: %v", i, err)
		}
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if len(g.groups) == 0 {
		t.Skip("scheduling never formed a group; nothing to assert")
	}
	for i, nbytes := range g.groupBytes {
		if nbytes > pbGroupBytesMax {
			t.Fatalf("group %d carries %d data bytes, above the %d cap", i, nbytes, pbGroupBytesMax)
		}
	}
}

// --- completeGroupSend credit rules (deterministic, no scheduling involved) ---

func TestCompleteGroupSendFullOKCreditsAll(t *testing.T) {
	e := newInflightTestEngine(t) // lease valid, epoch=5
	r1 := pushInflight(e, 5, 1, []string{"b1"})
	r2 := pushInflight(e, 5, 2, []string{"b1"})
	r3 := pushInflight(e, 5, 3, []string{"b1"})

	e.completeGroupSend("b1", 5, 1, 3, AckMsg{Epoch: 5, Seq: 3, OK: true}, nil)

	if got := e.Committed(); got != 3 {
		t.Fatalf("Committed = %d, want 3", got)
	}
	mustResolveNil(t, r1)
	mustResolveNil(t, r2)
	mustResolveNil(t, r3)
}

func TestCompleteGroupSendPrefixNackCreditsPrefixFailsSuffix(t *testing.T) {
	e := newInflightTestEngine(t)
	r1 := pushInflight(e, 5, 1, []string{"b1"})
	r2 := pushInflight(e, 5, 2, []string{"b1"})
	r3 := pushInflight(e, 5, 3, []string{"b1"})

	// Backup applied only seq 1 of [1..3].
	e.completeGroupSend("b1", 5, 1, 3, AckMsg{Epoch: 5, Seq: 1, OK: false}, nil)

	if got := e.Committed(); got != 1 {
		t.Fatalf("Committed = %d, want 1 (prefix only)", got)
	}
	mustResolveNil(t, r1)
	if err := waitResolve(r2); !errors.Is(err, ErrReplicationTimeout) {
		t.Fatalf("r2 err = %v, want ErrReplicationTimeout", err)
	}
	if err := waitResolve(r3); !errors.Is(err, ErrReplicationTimeout) {
		t.Fatalf("r3 err = %v, want ErrReplicationTimeout", err)
	}
}

func TestCompleteGroupSendTransportErrorFailsAll(t *testing.T) {
	e := newInflightTestEngine(t)
	r1 := pushInflight(e, 5, 1, []string{"b1"})
	r2 := pushInflight(e, 5, 2, []string{"b1"})

	e.completeGroupSend("b1", 5, 1, 2, AckMsg{}, errors.New("wire down"))

	if got := e.Committed(); got != 0 {
		t.Fatalf("Committed = %d, want 0", got)
	}
	for i, r := range []*inflight{r1, r2} {
		if err := waitResolve(r); !errors.Is(err, ErrReplicationTimeout) {
			t.Fatalf("r%d err = %v, want ErrReplicationTimeout", i+1, err)
		}
	}
}

func TestCompleteGroupSendWrongEpochCreditsNothing(t *testing.T) {
	e := newInflightTestEngine(t)
	r1 := pushInflight(e, 5, 1, []string{"b1"})

	// A crafted/misrouted ack for a different epoch must not credit (H6).
	e.completeGroupSend("b1", 5, 1, 1, AckMsg{Epoch: 4, Seq: 1, OK: true}, nil)

	if got := e.Committed(); got != 0 {
		t.Fatalf("Committed = %d, want 0", got)
	}
	if err := waitResolve(r1); !errors.Is(err, ErrReplicationTimeout) {
		t.Fatalf("r1 err = %v, want ErrReplicationTimeout", err)
	}
}

func TestCompleteGroupSendPartialPeerCredit(t *testing.T) {
	// Two-backup ISR: a full-OK group ack from b1 alone must not commit; the
	// matching ack from b2 completes full-ISR and commits (P6).
	e := newInflightTestEngine(t)
	r1 := pushInflight(e, 5, 1, []string{"b1", "b2"})

	e.completeGroupSend("b1", 5, 1, 1, AckMsg{Epoch: 5, Seq: 1, OK: true}, nil)
	if got := e.Committed(); got != 0 {
		t.Fatalf("Committed = %d after 1 of 2 peers, want 0", got)
	}
	e.completeGroupSend("b2", 5, 1, 1, AckMsg{Epoch: 5, Seq: 1, OK: true}, nil)
	if got := e.Committed(); got != 1 {
		t.Fatalf("Committed = %d after full ISR, want 1", got)
	}
	mustResolveNil(t, r1)
}

// TestSenderGroupPrefixOutcomesUnderLoad is the scheduling-exposed version of
// the prefix rule: whatever mix of singles/groups the sender forms, a run where
// each group credits only its first record must (a) fail at least one write
// when any group formed, (b) fail ONLY with ErrReplicationTimeout, and (c)
// commit every write that reported success (nil-err seq <= Committed).
func TestSenderGroupPrefixOutcomesUnderLoad(t *testing.T) {
	primary, g, _ := newGatedPair(t)
	g.ackFn = func(msgs []ReplicateMsg) (AckMsg, error) {
		if ack := g.backup.ReceiveGroup(msgs); !ack.OK {
			return ack, nil // keep real failures honest
		}
		return AckMsg{Epoch: msgs[0].Epoch, Seq: msgs[0].Seq, OK: false}, nil
	}

	done := make(chan []error, 1)
	go func() { done <- proposeN(t, primary, 8) }()
	eventually(t, func() bool { return primary.LastSeq() == 8 }, "8 writes sequenced")
	close(g.release)

	errs := <-done
	if g.maxGroup() < 2 {
		t.Skip("scheduling never formed a group; nothing to assert")
	}
	committed := primary.Committed()
	var failed int
	for seq1, err := range errs {
		switch {
		case err == nil:
			if uint64(seq1+1) > committed {
				t.Fatalf("seq %d reported committed but Committed=%d", seq1+1, committed)
			}
		case errors.Is(err, ErrReplicationTimeout):
			failed++
		default:
			t.Fatalf("seq %d: unexpected failure kind: %v", seq1+1, err)
		}
	}
	if failed == 0 {
		t.Fatal("prefix nack failed no writes — suffix records were wrongly credited")
	}
}

// --- Group frames over the real NetTransport ----------------------------------

func TestNetTransportGroupRoundTrip(t *testing.T) {
	srv, err := NewNetTransport("127.0.0.1:0", nil, nil, nil)
	if err != nil {
		t.Fatalf("server: %v", err)
	}
	defer srv.Close()

	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	backup := c.engines["n2"]
	srv.Register(testShard, backup)

	cli, err := NewNetTransport("127.0.0.1:0", nil, nil, nil)
	if err != nil {
		t.Fatalf("client: %v", err)
	}
	defer cli.Close()

	tr, ok := cli.For(testShard).(GroupTransport)
	if !ok {
		t.Fatal("shardTransport does not implement GroupTransport")
	}
	msgs := []ReplicateMsg{
		{Epoch: 1, Seq: 1, PrevSeq: 0, Data: []byte("a")},
		{Epoch: 1, Seq: 2, PrevSeq: 1, Data: []byte("bb")},
		{Epoch: 1, Seq: 3, PrevSeq: 2, Data: []byte("ccc")},
	}
	got := make(chan AckMsg, 1)
	if err := tr.ReplicateGroup(srv.Addr(), msgs, func(ack AckMsg, err error) {
		if err != nil {
			t.Errorf("group callback err: %v", err)
		}
		got <- ack
	}); err != nil {
		t.Fatalf("ReplicateGroup: %v", err)
	}
	select {
	case ack := <-got:
		if !ack.OK || ack.Epoch != 1 || ack.Seq != 3 {
			t.Fatalf("ack = %+v, want OK epoch=1 seq=3", ack)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("group ack never arrived")
	}
	if got := backup.LastApplied(); got != 3 {
		t.Fatalf("backup LastApplied = %d, want 3", got)
	}
}

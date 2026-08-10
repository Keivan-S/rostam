// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"bufio"
	"net"
	"sync"
	"testing"
	"time"
)

// ============================================================================
// Log-matching wire coverage: PrevEpoch is part of the chain link, so it must
// survive EVERY path a frame can take to a receiver — the in-memory transport
// (engine unit tests) and the real framed net transport, in both the single and
// the group shape — and the catch-up handshake must carry the log identity its
// own payload now encodes. A field that silently drops to zero on one of these
// paths degrades exactly into the defect this stage closes.
// ============================================================================

// TestPrevEpochRoundTripsOverNetTransport drives a real TCP NetTransport and
// asserts the receiver observes the PrevEpoch the sender stamped, for both the
// single-write and the group frame. The group case is the sharper one: only the
// FIRST record's predecessor is on the wire, the rest are rebuilt from the
// group's uniform epoch.
func TestPrevEpochRoundTripsOverNetTransport(t *testing.T) {
	backup, err := NewNetTransport(":0", nil, nil, nil)
	if err != nil {
		t.Fatalf("NewNetTransport(backup): %v", err)
	}
	defer backup.Close()
	rcv := &recordingReceiver{}
	const shard = 3
	backup.Register(shard, rcv)

	primary, err := NewNetTransport(":0", nil, nil, nil)
	if err != nil {
		t.Fatalf("NewNetTransport(primary): %v", err)
	}
	defer primary.Close()
	tr := primary.For(shard)

	// Single write: the promoted-primary shape — this write is at epoch 9 but its
	// predecessor was assigned at epoch 4.
	single := ReplicateMsg{Epoch: 9, Seq: 12, PrevSeq: 11, PrevEpoch: 4, Data: []byte("solo")}
	if _, err := syncReplicate(tr, backup.Addr(), single); err != nil {
		t.Fatalf("replicate: %v", err)
	}
	got := rcv.all()
	if len(got) != 1 {
		t.Fatalf("receiver saw %d msgs, want 1", len(got))
	}
	if got[0].PrevEpoch != 4 || got[0].PrevSeq != 11 || got[0].Epoch != 9 || got[0].Seq != 12 {
		t.Fatalf("single-frame chain link mangled: got %+v want %+v", got[0], single)
	}

	// Group: uniform epoch 9, seq-dense, record 0 crossing the epoch boundary.
	rcv.reset()
	group := []ReplicateMsg{
		{Epoch: 9, Seq: 12, PrevSeq: 11, PrevEpoch: 4, Data: []byte("g0")},
		{Epoch: 9, Seq: 13, PrevSeq: 12, PrevEpoch: 9, Data: []byte("g1")},
		{Epoch: 9, Seq: 14, PrevSeq: 13, PrevEpoch: 9, Data: nil},
	}
	gt, ok := tr.(GroupTransport)
	if !ok {
		t.Fatal("shardTransport must implement GroupTransport")
	}
	done := make(chan error, 1)
	if err := gt.ReplicateGroup(backup.Addr(), group, func(_ AckMsg, cbErr error) { done <- cbErr }); err != nil {
		t.Fatalf("ReplicateGroup submit: %v", err)
	}
	select {
	case cbErr := <-done:
		if cbErr != nil {
			t.Fatalf("ReplicateGroup: %v", cbErr)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("ReplicateGroup never completed")
	}
	got = rcv.all()
	if len(got) != len(group) {
		t.Fatalf("receiver saw %d group records, want %d", len(got), len(group))
	}
	for i := range group {
		if got[i].Epoch != group[i].Epoch || got[i].Seq != group[i].Seq ||
			got[i].PrevSeq != group[i].PrevSeq || got[i].PrevEpoch != group[i].PrevEpoch {
			t.Fatalf("group record %d chain link mangled: got %+v want %+v", i, got[i], group[i])
		}
	}
}

// TestCatchupInfoRoundTripsOverNetTransport proves the handshake's own payload
// carries all four fields end to end — in particular that AppliedSeq and
// FrontierSeq stay DISTINCT on the wire, since conflating them is what made an
// ex-primary look like a blank node.
func TestCatchupInfoRoundTripsOverNetTransport(t *testing.T) {
	backup, err := NewNetTransport(":0", nil, nil, nil)
	if err != nil {
		t.Fatalf("NewNetTransport(backup): %v", err)
	}
	defer backup.Close()
	want := CatchupInfoMsg{Epoch: 7, AppliedSeq: 3, FrontierSeq: 11, FrontierEpoch: 5, OK: true}
	const shard = 2
	backup.Register(shard, &recordingReceiver{info: want})

	primary, err := NewNetTransport(":0", nil, nil, nil)
	if err != nil {
		t.Fatalf("NewNetTransport(primary): %v", err)
	}
	defer primary.Close()
	ct, ok := primary.For(shard).(CatchupTransport)
	if !ok {
		t.Fatal("shardTransport must implement CatchupTransport")
	}
	got, err := ct.CatchupRequest(backup.Addr(), 7)
	if err != nil {
		t.Fatalf("CatchupRequest: %v", err)
	}
	if got != want {
		t.Fatalf("catch-up info round-trip: got %+v want %+v", got, want)
	}
}

// TestPrevEpochRoundTripsOverInMemTransport is the same guarantee for the
// in-memory transport the engine unit tests run on: it hands the struct straight
// to the peer, so the assertion is that nothing in the submit path drops or
// rewrites the chain link before the receiver's check sees it.
func TestPrevEpochRoundTripsOverInMemTransport(t *testing.T) {
	tr := newInMemTransport()
	clk := &fakeClock{t: t0}
	ctrl := &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1", "n2"}, minISR: 2}
	backup := New("n2", testShard, ctrl, tr, &fakeApplier{}, WithClock(clk.now))
	defer backup.Shutdown()
	tr.register("n2", backup)

	// A frame whose PrevEpoch is WRONG must be nacked by the receiver — proof the
	// field survived the hop rather than arriving as a zero that happened to match
	// a genesis node.
	nack := make(chan AckMsg, 1)
	if err := tr.Replicate("n2", ReplicateMsg{Epoch: 1, Seq: 1, PrevSeq: 0, PrevEpoch: 42, Data: []byte("x")},
		func(a AckMsg, _ error) { nack <- a }); err != nil {
		t.Fatalf("Replicate: %v", err)
	}
	if a := <-nack; a.OK {
		t.Fatal("a genesis-position frame naming a non-genesis predecessor must be nacked")
	}

	// The same frame with the correct PrevEpoch is accepted.
	ok := make(chan AckMsg, 1)
	if err := tr.Replicate("n2", ReplicateMsg{Epoch: 1, Seq: 1, PrevSeq: 0, PrevEpoch: 0, Data: []byte("x")},
		func(a AckMsg, _ error) { ok <- a }); err != nil {
		t.Fatalf("Replicate: %v", err)
	}
	if a := <-ok; !a.OK {
		t.Fatalf("correctly chained frame was rejected: %+v", a)
	}
}

// TestPBCatchupInfoCodec is the payload-level round trip for the handshake's new
// wire type, including the not-OK encoding.
func TestPBCatchupInfoCodec(t *testing.T) {
	for _, ok := range []bool{true, false} {
		c := CatchupInfoMsg{Epoch: 3, AppliedSeq: 90, FrontierSeq: 91, FrontierEpoch: 2, OK: ok}
		got, err := decodeCatchupInfo(encodeCatchupInfo(nil, c))
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		if got != c {
			t.Fatalf("round-trip: got %+v want %+v", got, c)
		}
	}
	if _, err := decodeCatchupInfo(make([]byte, pbCatchupInfoSize-1)); err == nil {
		t.Fatal("short catch-up info payload must be rejected")
	}
}

// TestPBFrameVersionBumped guards the deliberate v1→v2 bump: the payload layouts
// changed incompatibly, so a v1 peer's frame must be REFUSED, never misparsed
// into a plausible-looking write.
func TestPBFrameVersionBumped(t *testing.T) {
	if pbFrameVersion != 2 {
		t.Fatalf("pbFrameVersion = %d, want 2 (log-matching payload layouts)", pbFrameVersion)
	}
	var hdr [pbFrameHeaderSize]byte
	writePBFrameHdr(hdr[:], &pbFrame{kind: pbKindReplicate, shard: 1, reqID: 1})
	hdr[1] = 1 // downgrade the version byte to v1
	c1, c2 := net.Pipe()
	defer c1.Close()
	defer c2.Close()
	go func() {
		_, _ = c1.Write(hdr[:])
	}()
	fr := pbFrameReader{r: bufio.NewReader(c2)}
	if _, err := fr.read(); err != errPBBadVersion {
		t.Fatalf("v1 frame: err = %v, want errPBBadVersion", err)
	}
}

// recordingReceiver captures every ReplicateMsg it is handed, VERBATIM, and
// answers a catch-up handshake with a canned CatchupInfoMsg. Unlike
// captureReceiver it keeps the whole sequence, so a group frame's per-record
// chain reconstruction can be inspected record by record.
type recordingReceiver struct {
	mu   sync.Mutex
	msgs []ReplicateMsg
	info CatchupInfoMsg
}

func (r *recordingReceiver) Receive(m ReplicateMsg) AckMsg {
	r.mu.Lock()
	r.msgs = append(r.msgs, m)
	r.mu.Unlock()
	return AckMsg{Epoch: m.Epoch, Seq: m.Seq, OK: true}
}

func (r *recordingReceiver) ReceiveGroup(msgs []ReplicateMsg) AckMsg {
	if len(msgs) == 0 {
		return AckMsg{OK: false}
	}
	var last AckMsg
	for i := range msgs {
		last = r.Receive(msgs[i])
	}
	return last
}

func (r *recordingReceiver) CatchupInfo() CatchupInfoMsg { return r.info }

func (r *recordingReceiver) all() []ReplicateMsg {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]ReplicateMsg(nil), r.msgs...)
}

func (r *recordingReceiver) reset() {
	r.mu.Lock()
	r.msgs = nil
	r.mu.Unlock()
}

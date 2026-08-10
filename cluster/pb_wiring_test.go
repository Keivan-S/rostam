// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/shard/pbisr"
)

// fakeBaseTransport records the peer address it was called with (post-resolve)
// so TestPBResolvingTransport can assert the node-ID → address rewrite without
// a real network. It implements the async Transport contract, invoking done
// synchronously.
type fakeBaseTransport struct {
	calledWith string
	called     bool
}

func (f *fakeBaseTransport) Replicate(peer string, msg pbisr.ReplicateMsg, done func(pbisr.AckMsg, error)) error {
	f.called = true
	f.calledWith = peer
	done(pbisr.AckMsg{Epoch: msg.Epoch, Seq: msg.Seq, OK: true}, nil)
	return nil
}

func TestPBResolvingTransport(t *testing.T) {
	base := &fakeBaseTransport{}
	rt := &pbResolvingTransport{
		base: base,
		addrOf: map[string]string{
			"n2": "10.0.0.2:7200",
		},
	}

	var ack pbisr.AckMsg
	var cbErr error
	err := rt.Replicate("n2", pbisr.ReplicateMsg{Epoch: 1, Seq: 1}, func(a pbisr.AckMsg, e error) {
		ack, cbErr = a, e
	})
	if err != nil {
		t.Fatalf("Replicate(n2): unexpected submit error: %v", err)
	}
	if cbErr != nil {
		t.Fatalf("Replicate(n2): unexpected callback error: %v", cbErr)
	}
	if !ack.OK {
		t.Fatalf("Replicate(n2): want OK ack, got %+v", ack)
	}
	if !base.called || base.calledWith != "10.0.0.2:7200" {
		t.Fatalf("base.Replicate: want called with resolved addr %q, got called=%v addr=%q",
			"10.0.0.2:7200", base.called, base.calledWith)
	}

	// Unknown node-ID: must error WITHOUT calling base (never dial a node-ID
	// as a hostname).
	base2 := &fakeBaseTransport{}
	rt2 := &pbResolvingTransport{base: base2, addrOf: map[string]string{"n2": "10.0.0.2:7200"}}
	doneCalled := false
	err = rt2.Replicate("n99", pbisr.ReplicateMsg{Epoch: 1, Seq: 1}, func(pbisr.AckMsg, error) {
		doneCalled = true
	})
	if err == nil {
		t.Fatal("Replicate(n99): want error for unknown node-ID, got nil")
	}
	if doneCalled {
		t.Fatal("Replicate(n99): done must NOT be invoked on a submission error")
	}
	if !strings.Contains(err.Error(), "n99") {
		t.Fatalf("Replicate(n99): error should mention the unknown node-ID, got: %v", err)
	}
	if base2.called {
		t.Fatal("Replicate(n99): must not call base for an unresolved node-ID")
	}
}

func TestPBAddrMap(t *testing.T) {
	peers := []Peer{
		{NodeID: "n1", RaftAddr: "a1", ServerAddr: "s1", PBAddr: "10.0.0.1:7200"},
		{NodeID: "n2", RaftAddr: "a2", ServerAddr: "s2", PBAddr: "10.0.0.2:7200"},
		{NodeID: "n3", RaftAddr: "a3", ServerAddr: "s3", PBAddr: ""}, // skipped: empty PBAddr
	}
	m := pbAddrMap(peers)
	if len(m) != 2 {
		t.Fatalf("want 2 entries (n3 skipped, empty PBAddr), got %d: %v", len(m), m)
	}
	if m["n1"] != "10.0.0.1:7200" || m["n2"] != "10.0.0.2:7200" {
		t.Fatalf("unexpected addr map: %v", m)
	}
	if _, ok := m["n3"]; ok {
		t.Fatalf("n3 with empty PBAddr must be skipped, got entry: %v", m["n3"])
	}
}

// fakeClock is a controllable monotonic-ns clock shared between an Engine
// (via pbisr.WithClock) and a leaseKeeper under test, so lease math is
// deterministic. The keeper's goroutine calls now() concurrently with the
// test goroutine's advance(), so both go through atomic.Int64.
type fakeClock struct {
	ns atomic.Int64
}

func newFakeClock(start int64) *fakeClock {
	c := &fakeClock{}
	c.ns.Store(start)
	return c
}

func (c *fakeClock) now() int64              { return c.ns.Load() }
func (c *fakeClock) advance(d time.Duration) { c.ns.Add(int64(d)) }

func TestLeaseKeeperGrantsWhilePrimary(t *testing.T) {
	f := NewMetaFSM()
	applyMeta(t, f, LogEntry{Op: OpSetShardEpoch, ShardID: 0, Epoch: 1, Primary: "n1"})
	applyMeta(t, f, LogEntry{Op: OpSetShardISR, ShardID: 0, Epoch: 1, ISR: []string{"n1", "n2"}})

	clock := newFakeClock(0)
	ctrl := newMetaControl(f, 1)
	// No transport/applier needed: this test never calls Propose/Receive,
	// only the lease-fence bookkeeping (AdoptEpoch/GrantLease/LeaseValid).
	eng := pbisr.New("n1", 0, ctrl, nil, nil, pbisr.WithClock(clock.now))

	const ttl = 50 * time.Millisecond
	const refresh = 5 * time.Millisecond
	// nil barrier = single-node / gate disabled (unconditional renewal), the
	// behavior this pre-Plan-4a test asserts.
	lk := newLeaseKeeper(f, "n1", map[int]*pbisr.Engine{0: eng}, ttl, refresh, clock.now, nil, nil, 0)
	lk.start()
	defer lk.stop()

	deadline := time.Now().Add(2 * time.Second)
	for !eng.LeaseValid() {
		if time.Now().After(deadline) {
			t.Fatal("timed out waiting for leaseKeeper to grant a valid lease")
		}
		time.Sleep(time.Millisecond)
	}

	// Reassign primary to a different node at a higher epoch: the keeper must
	// stop refreshing n1's lease.
	applyMeta(t, f, LogEntry{Op: OpSetShardEpoch, ShardID: 0, Epoch: 2, Primary: "n2"})
	applyMeta(t, f, LogEntry{Op: OpSetShardISR, ShardID: 0, Epoch: 2, ISR: []string{"n1", "n2"}})

	// Give the keeper a couple of refresh cycles to observe the new primary
	// and stop granting, then advance the injected clock past the TTL.
	time.Sleep(3 * refresh)
	clock.advance(ttl + refresh)
	time.Sleep(3 * refresh)

	if eng.LeaseValid() {
		t.Fatal("LeaseValid: want false after primary reassignment + TTL elapsed, got true")
	}
}

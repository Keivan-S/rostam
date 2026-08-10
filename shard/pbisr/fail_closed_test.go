// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
)

// cacheApplier is an Applier backed by a real cache under PolicyRejectWrites. When
// its cache is at capacity, Apply returns cache.ErrFull — a NON-DETERMINISTIC apply
// error. The pbisr engine treats ANY non-nil Applier error as a fail-closed abort
// (primary: no seq burned; backup: NACK + no lastApplied advance), which is exactly
// what the shard-side shardApplier produces for a classFatal error (production
// -readiness #4). This test proves the engine + cache integration fails closed;
// the shard-package tests cover the classFatal classification itself.
type cacheApplier struct {
	c       *cache.Cache
	attempt atomic.Int64
	stored  atomic.Int64
}

func (a *cacheApplier) Apply(data []byte) ([]byte, error) {
	key := []byte(fmt.Sprintf("apply-%d", a.attempt.Add(1)))
	if err := a.c.Put(key, data, 0); err != nil {
		return nil, err // cache.ErrFull on a full backup ⇒ engine fails closed
	}
	a.stored.Add(1)
	return []byte("ok"), nil
}

func newRejectCache(t *testing.T) *cache.Cache {
	t.Helper()
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	cc.PageSize = 1 << 20
	cc.MaxMemoryPerShard = 1 << 20 // one page ⇒ MaxPagesPerShard=1
	cc.AtCapPolicy = cache.PolicyRejectWrites
	c, err := cache.New(cc)
	if err != nil {
		t.Fatalf("new reject cache: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func fillToCap(t *testing.T, c *cache.Cache) {
	t.Helper()
	val := make([]byte, 4096)
	for i := 0; i < 1_000_000; i++ {
		if err := c.Put([]byte(fmt.Sprintf("fill-%d", i)), val, 0); err != nil {
			if err == cache.ErrFull {
				return
			}
			t.Fatalf("fill put %d: %v", i, err)
		}
	}
	t.Fatal("cache never reached capacity")
}

// TestBackupErrFullNacksAndDoesNotAdvance is the deterministic core proof: a
// ReplicateMsg delivered to a backup whose cache is at capacity NACKs (OK:false)
// and does NOT advance lastApplied, so the backup cannot falsely ack a write it
// never materialized. It then gap-stalls and reconverges via catch-up/snapshot —
// the fail-closed behavior that replaces silent divergence.
func TestBackupErrFullNacksAndDoesNotAdvance(t *testing.T) {
	ctrl := &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1", "n2", "n3"}, minISR: 2}
	tr := newInMemTransport()
	clk := &fakeClock{t: t0}

	fullCache := newRejectCache(t)
	fillToCap(t, fullCache)
	ap3 := &cacheApplier{c: fullCache}
	backup := New("n3", testShard, ctrl, tr, ap3, WithClock(clk.now))
	tr.register("n3", backup)

	// The payload must be at least as large as fillToCap's value so the full page
	// has no leftover slack that could admit it — the apply reliably ErrFulls.
	ack := backup.Receive(ReplicateMsg{Epoch: 1, Seq: 1, PrevSeq: 0, Data: make([]byte, 8192)})
	if ack.OK {
		t.Fatal("full backup must NACK its apply (OK:false)")
	}
	if got := backup.LastApplied(); got != 0 {
		t.Fatalf("full backup lastApplied = %d; must stay 0 (no advance on a fail-closed apply)", got)
	}
	if got := ap3.stored.Load(); got != 0 {
		t.Fatalf("full backup stored %d writes; must be 0", got)
	}
}

// TestProposeDoesNotFalselyCommitWhenBackupFull is the end-to-end proof over a
// 3-engine in-mem cluster: a Put that fits on the primary and a roomy backup but
// ErrFulls on the full backup never reaches the full ISR, so under CommitFullISR
// it does NOT commit and the full backup never advances.
func TestProposeDoesNotFalselyCommitWhenBackupFull(t *testing.T) {
	ctrl := &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1", "n2", "n3"}, minISR: 2}
	tr := newInMemTransport()
	clk := &fakeClock{t: t0}

	engines := map[string]*Engine{}
	mk := func(id string, full bool) *cacheApplier {
		c := newRejectCache(t)
		if full {
			fillToCap(t, c)
		}
		ap := &cacheApplier{c: c}
		eng := New(id, testShard, ctrl, tr, ap, WithClock(clk.now))
		tr.register(id, eng)
		engines[id] = eng
		if id == "n1" {
			eng.GrantLease(1, t0+leaseDur)
		}
		return ap
	}
	ap1 := mk("n1", false)
	ap2 := mk("n2", false)
	ap3 := mk("n3", true) // full backup

	// CommitFullISR: the full backup never acks, so the record never reaches full
	// ISR — Propose must NOT return a successful commit.
	_, _, err := engines["n1"].Propose(ctxWithTimeout(t, 400*time.Millisecond), make([]byte, 8192))
	if err == nil {
		t.Fatal("write must NOT commit when a backup fails closed under CommitFullISR")
	}

	// The write is real on the healthy replicas...
	if got := ap1.stored.Load(); got != 1 {
		t.Fatalf("primary stored %d, want 1", got)
	}
	eventually(t, func() bool { return ap2.stored.Load() == 1 }, "roomy backup n2 stores the write")

	// ...but the full backup stored nothing and never advanced.
	if got := ap3.stored.Load(); got != 0 {
		t.Fatalf("full backup stored %d, want 0 (fail-closed)", got)
	}
	if got := engines["n3"].LastApplied(); got != 0 {
		t.Fatalf("full backup lastApplied = %d, want 0", got)
	}
}

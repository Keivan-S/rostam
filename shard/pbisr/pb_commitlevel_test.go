// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

// This suite reuses engine_test.go's blockingTransport: it accepts submissions,
// records their (done, msg) in `pending`, and NEVER acks until release() — the
// exact "silent backup" needed to show CommitPrimary commits on local apply
// while CommitFullISR blocks. sawCount reads how many submissions it holds.
func sawCount(b *blockingTransport) int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.pending)
}

func newCommitLevelEngine(t *testing.T, level CommitLevel, tr Transport) (*Engine, *fakeApplier) {
	t.Helper()
	ctrl := &fakeControl{epoch: 5, primary: "n1", isr: []string{"n1", "b1"}, minISR: 2}
	clk := &fakeClock{t: t0}
	ap := &fakeApplier{}
	e := New("n1", testShard, ctrl, tr, ap, WithClock(clk.now), WithCommitLevel(level))
	e.GrantLease(5, t0+leaseDur)
	t.Cleanup(e.Shutdown)
	return e, ap
}

// TestCommitPrimaryCommitsWithoutBackupAck is the headline: with a backup that
// never acks, a CommitPrimary write still commits on local apply, while an
// otherwise-identical CommitFullISR write times out.
func TestCommitPrimaryCommitsWithoutBackupAck(t *testing.T) {
	// CommitPrimary: commits despite the silent backup.
	trP := &blockingTransport{}
	eP, apP := newCommitLevelEngine(t, CommitPrimary, trP)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_, seq, err := eP.Propose(ctx, []byte("x"))
	if err != nil {
		t.Fatalf("CommitPrimary Propose err = %v, want nil (committed on local apply)", err)
	}
	if seq != 1 || eP.Committed() != 1 {
		t.Fatalf("seq=%d committed=%d, want 1/1", seq, eP.Committed())
	}
	if apP.count() != 1 {
		t.Fatalf("primary applied %d, want 1", apP.count())
	}
	// The backup is still shipped — asynchronously, so it may land just after
	// Propose returned (that is the whole point of CommitPrimary).
	eventually(t, func() bool { return sawCount(trP) == 1 }, "backup shipped async")

	// CommitFullISR on the same silent-backup transport: must NOT commit.
	trF := &blockingTransport{}
	eF, _ := newCommitLevelEngine(t, CommitFullISR, trF)
	ctx2, cancel2 := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel2()
	_, _, err = eF.Propose(ctx2, []byte("x"))
	if !errors.Is(err, ErrReplicationTimeout) {
		t.Fatalf("CommitFullISR Propose err = %v, want ErrReplicationTimeout", err)
	}
	if eF.Committed() != 0 {
		t.Fatalf("CommitFullISR committed=%d, want 0 (backup never acked)", eF.Committed())
	}
}

// TestCommitPrimaryLeaseFenceStillApplies proves OH1 survives the downgrade: an
// expired lease blocks the commit even under CommitPrimary.
func TestCommitPrimaryLeaseFenceStillApplies(t *testing.T) {
	ctrl := &fakeControl{epoch: 5, primary: "n1", isr: []string{"n1", "b1"}, minISR: 2}
	clk := &fakeClock{t: t0}
	e := New("n1", testShard, ctrl, &blockingTransport{}, &fakeApplier{}, WithClock(clk.now), WithCommitLevel(CommitPrimary))
	t.Cleanup(e.Shutdown)
	e.GrantLease(5, t0+leaseDur)
	clk.set(t0 + leaseDur + 1) // lease expired

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_, _, err := e.Propose(ctx, []byte("x"))
	if !errors.Is(err, ErrLeaseExpired) {
		t.Fatalf("err = %v, want ErrLeaseExpired (fence must survive CommitPrimary)", err)
	}
	if e.Committed() != 0 {
		t.Fatalf("committed=%d, want 0 (fenced primary must not commit)", e.Committed())
	}
}

// TestCommitPrimaryDenseInOrderUnderLoad: concurrent CommitPrimary writes with
// a silent backup all commit, with committed advancing densely to N.
func TestCommitPrimaryDenseInOrderUnderLoad(t *testing.T) {
	e, ap := newCommitLevelEngine(t, CommitPrimary, &blockingTransport{})

	const writers, per = 8, 50
	var wg sync.WaitGroup
	errs := make([]error, writers*per)
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < per; i++ {
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				_, _, err := e.Propose(ctx, []byte("op"))
				cancel()
				errs[w*per+i] = err
			}
		}(w)
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Fatalf("propose %d: %v", i, err)
		}
	}
	if got := e.Committed(); got != writers*per {
		t.Fatalf("Committed = %d, want %d (dense commit-on-apply)", got, writers*per)
	}
	if got := ap.count(); got != writers*per {
		t.Fatalf("applied = %d, want %d", got, writers*per)
	}
}

// TestCommitPrimaryLateAckIsHarmless: after CommitPrimary already committed,
// the backup's belated ack must be a no-op (record already popped).
func TestCommitPrimaryLateAckIsHarmless(t *testing.T) {
	tr := &blockingTransport{}
	e, _ := newCommitLevelEngine(t, CommitPrimary, tr)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if _, _, err := e.Propose(ctx, []byte("x")); err != nil {
		t.Fatalf("propose: %v", err)
	}
	before := e.Committed()
	tr.release() // belated backup acks arrive for an already-committed write
	if after := e.Committed(); after != before {
		t.Fatalf("committed moved on late ack: %d -> %d", before, after)
	}
}

// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"context"
	"sync"
	"testing"
	"time"
)

// inlineFakeTransport implements Transport + InlineTransport against one real
// backup engine, recording which path each submission took and the exact
// per-peer submission order. refuse (optional) makes the Nth TryReplicate
// attempt decline, forcing the sender-path fallback.
type inlineFakeTransport struct {
	backup *Engine

	mu       sync.Mutex
	attempts int
	inline   int
	queued   int
	order    []uint64 // seqs in submission order
	refuse   func(attempt int) bool
}

func (t *inlineFakeTransport) submit(msg ReplicateMsg, done func(AckMsg, error)) {
	done(t.backup.Receive(msg), nil)
}

func (t *inlineFakeTransport) Replicate(_ string, msg ReplicateMsg, done func(AckMsg, error)) error {
	t.mu.Lock()
	t.queued++
	t.order = append(t.order, msg.Seq)
	t.mu.Unlock()
	t.submit(msg, done)
	return nil
}

func (t *inlineFakeTransport) TryReplicate(_ string, msg ReplicateMsg, done func(AckMsg, error)) bool {
	t.mu.Lock()
	t.attempts++
	if t.refuse != nil && t.refuse(t.attempts) {
		t.mu.Unlock()
		return false
	}
	t.inline++
	t.order = append(t.order, msg.Seq)
	t.mu.Unlock()
	t.submit(msg, done)
	return true
}

func (t *inlineFakeTransport) counts() (inline, queued int, order []uint64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.inline, t.queued, append([]uint64(nil), t.order...)
}

// newInlinePair builds a primary over an inlineFakeTransport and a real
// backup engine sharing one control plane and clock.
func newInlinePair(t *testing.T, refuse func(int) bool) (*Engine, *inlineFakeTransport) {
	t.Helper()
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	tr := &inlineFakeTransport{backup: c.engines["n2"], refuse: refuse}
	primary := New("n1", testShard, c.ctrl, tr, c.appliers["n1"], WithClock(c.clk.now))
	primary.GrantLease(1, t0+leaseDur)
	t.Cleanup(primary.Shutdown)
	return primary, tr
}

func TestInlineSubmitBypassesSender(t *testing.T) {
	primary, tr := newInlinePair(t, nil)
	for i := 0; i < 5; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		if _, _, err := primary.Propose(ctx, []byte("op")); err != nil {
			cancel()
			t.Fatalf("propose %d: %v", i, err)
		}
		cancel()
	}
	inline, queued, order := tr.counts()
	if inline != 5 || queued != 0 {
		t.Fatalf("inline=%d queued=%d, want 5/0 (uncontended writes must skip the sender)", inline, queued)
	}
	for i, seq := range order {
		if seq != uint64(i+1) {
			t.Fatalf("submission order %v: position %d has seq %d", order, i, seq)
		}
	}
	if got := primary.Committed(); got != 5 {
		t.Fatalf("Committed = %d, want 5", got)
	}
}

func TestInlineFallbackPreservesOrderUnderLoad(t *testing.T) {
	// Refuse every 3rd inline attempt: submissions constantly alternate between
	// the inline path and the sender path. P1 requires the per-peer submission
	// order to remain exactly seq order regardless of which path each took —
	// and the real backup's gap check (PrevSeq) enforces it: any reordering
	// nacks, dooming a write, so every Propose succeeding is itself the proof.
	primary, tr := newInlinePair(t, func(n int) bool { return n%3 == 0 })

	const writers, perWriter = 8, 40
	var wg sync.WaitGroup
	errs := make([]error, writers*perWriter)
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < perWriter; i++ {
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				_, _, err := primary.Propose(ctx, []byte("op"))
				cancel()
				errs[w*perWriter+i] = err
			}
		}(w)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("propose %d: %v", i, err)
		}
	}
	inline, queued, order := tr.counts()
	if inline == 0 || queued == 0 {
		t.Fatalf("inline=%d queued=%d — the test must exercise BOTH paths", inline, queued)
	}
	for i, seq := range order {
		if seq != uint64(i+1) {
			t.Fatalf("submission order violated at position %d: seq %d (inline=%d queued=%d)", i, seq, inline, queued)
		}
	}
	if got := primary.Committed(); got != writers*perWriter {
		t.Fatalf("Committed = %d, want %d", got, writers*perWriter)
	}
	if got := tr.backup.LastApplied(); got != writers*perWriter {
		t.Fatalf("backup LastApplied = %d, want %d", got, writers*perWriter)
	}
}

func TestLinkTrySubmitRefusesUnconnected(t *testing.T) {
	srvAddr, stop := startEchoFrameServer(t)
	defer stop()
	l := newPBPeerLink(srvAddr, time.Second)
	defer l.close()

	// Never dialed: the no-dial rule must refuse without connecting.
	f := &pbFrame{kind: pbKindReplicate, shard: 1, payload: []byte("x")}
	if l.trySubmitAsync(f, func([]byte, error) { t.Error("done fired on refusal") }) {
		t.Fatal("trySubmitAsync succeeded on an unconnected link")
	}

	// Establish the connection via the normal path, then inline must work.
	if _, err := l.roundTrip(&pbFrame{kind: pbKindReplicate, shard: 1, payload: []byte("dial")}, 2*time.Second); err != nil {
		t.Fatalf("dial roundTrip: %v", err)
	}
	got := make(chan error, 1)
	f2 := &pbFrame{kind: pbKindReplicate, shard: 1, payload: []byte("y")}
	if !l.trySubmitAsync(f2, func(_ []byte, err error) { got <- err }) {
		t.Fatal("trySubmitAsync refused on a connected, idle link")
	}
	select {
	case err := <-got:
		if err != nil {
			t.Fatalf("inline submit callback err: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("inline submit callback never fired")
	}
}

// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// twoNodeControl: static control plane naming n1 primary, ISR {n1,n2}, minISR 2.
type twoNodeControl struct{}

func (twoNodeControl) Epoch(int) uint64   { return 1 }
func (twoNodeControl) Primary(int) string { return "n1" }
func (twoNodeControl) ISR(int) []string   { return []string{"n1", "n2"} }
func (twoNodeControl) MinISR(int) int     { return 2 }

// countingApplier records how many writes it applied (backup-side proof). n is
// an atomic counter: Apply() runs on the transport's per-conn dispatch
// goroutine (backup side) while the test reads the count from the test
// goroutine after Propose() returns — a bare int has no Go memory-model
// visibility guarantee across that hand-off without explicit synchronization.
type countingApplier struct{ n atomic.Int64 }

func (a *countingApplier) Apply(data []byte) ([]byte, error) { a.n.Add(1); return data, nil }

func TestNetTransportEndToEndReplication(t *testing.T) {
	const shard = 0

	// Backup node n2: transport + engine registered as receiver.
	backupTr, err := NewNetTransport(":0", nil, nil, nil)
	if err != nil {
		t.Fatalf("backup transport: %v", err)
	}
	defer backupTr.Close()
	backupApp := &countingApplier{}
	// n2's engine: it is a backup, so its Receive path is what matters. Its own
	// transport is unused here (it never proposes).
	backupEng := New("n2", shard, twoNodeControl{}, nil, backupApp)
	backupTr.Register(shard, backupEng)

	// Primary node n1: transport dials n2; peer "n2" resolves to backupTr.Addr().
	primaryTr, err := NewNetTransport(":0", nil, nil, nil)
	if err != nil {
		t.Fatalf("primary transport: %v", err)
	}
	defer primaryTr.Close()

	// The primary's Transport must map the ISR member id "n2" to its address.
	// For this test, wrap For(shard) so peer "n2" dials backupTr.Addr().
	base := primaryTr.For(shard)
	tr := addrRewrite{base: base, m: map[string]string{"n2": backupTr.Addr()}}

	primaryApp := &countingApplier{}
	var clock int64 = 1
	primaryEng := New("n1", shard, twoNodeControl{}, tr, primaryApp,
		WithClock(func() int64 { return clock }))
	defer primaryEng.Shutdown() // stop the per-peer sender goroutine on teardown
	primaryEng.GrantLease(1, 1_000_000)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_, seq, err := primaryEng.Propose(ctx, []byte("op1"))
	if err != nil {
		t.Fatalf("Propose: %v", err)
	}
	if seq != 1 {
		t.Fatalf("first seq: want 1, got %d", seq)
	}
	if primaryEng.Committed() != 1 {
		t.Fatalf("primary should have committed seq 1 (full ISR acked), committed=%d", primaryEng.Committed())
	}
	if n := backupApp.n.Load(); n != 1 {
		t.Fatalf("backup should have applied 1 write, got %d", n)
	}
}

// addrRewrite maps ISR member ids to dial addresses for the test. It implements
// the async Transport contract by rewriting the peer then delegating to base.
type addrRewrite struct {
	base Transport
	m    map[string]string
}

func (a addrRewrite) Replicate(peer string, msg ReplicateMsg, done func(AckMsg, error)) error {
	if addr, ok := a.m[peer]; ok {
		peer = addr
	}
	return a.base.Replicate(peer, msg, done)
}

// threeNodeControl (n1 primary, ISR {n1,n2,n3}) is declared in net_bench_test.go.
// Full-ISR commit REQUIRES both backups n2 AND n3 to ack every write, so at
// quiescence both backups have applied exactly the primary's committed tail.

// TestNetTransportPipelineE2EConcurrent is the real-network proof of the pipeline
// (PIPELINE-REDESIGN). It stands up a 3-engine cluster wired over
// the real pbisr.NetTransport — one primary (n1), two backups (n2, n3) — and
// drives `total` CONCURRENT Proposes from independent goroutines. Because
// total > pipelineWindow, this also exercises the admission gate over real TCP.
// It proves the pipeline is correct over real framing + the completion-callback
// link (not just the in-memory transport), asserting the P1/P6/P7 invariants:
//
//   - every successful Propose gets a dense, gap-free seq; the set of committed
//     seqs is exactly 1..total with no holes and no duplicates (P1);
//   - committed advances to total, and at quiescence BOTH backups' lastApplied
//     equals the primary's committed (all three engines agree on the applied
//     tail) — full-ISR commit requires both backups to have applied every seq;
//   - zero corruption: each backup applied exactly `total` writes, gap-free and
//     in seq order (the backup gap check rejects any out-of-order arrival, so
//     lastApplied == total with applied count == total proves dense in-order
//     application — P1/P7).
//
// Concurrency is fine; the ASSERTED invariants must hold under any timing. Runs
// clean under -race.
func TestNetTransportPipelineE2EConcurrent(t *testing.T) {
	const shard = 0
	// total > pipelineWindow (256) so the admission window genuinely gates over
	// real TCP, not just in the unit tests.
	const total = 300

	// Backups n2, n3: each gets its own transport + engine registered as receiver.
	// A backup never proposes, so its own Transport is unused (nil).
	backup2Tr, err := NewNetTransport(":0", nil, nil, nil)
	if err != nil {
		t.Fatalf("backup2 transport: %v", err)
	}
	defer backup2Tr.Close()
	backup2App := &countingApplier{}
	backup2Eng := New("n2", shard, threeNodeControl{}, nil, backup2App)
	backup2Tr.Register(shard, backup2Eng)

	backup3Tr, err := NewNetTransport(":0", nil, nil, nil)
	if err != nil {
		t.Fatalf("backup3 transport: %v", err)
	}
	defer backup3Tr.Close()
	backup3App := &countingApplier{}
	backup3Eng := New("n3", shard, threeNodeControl{}, nil, backup3App)
	backup3Tr.Register(shard, backup3Eng)

	// Primary n1: its transport dials the backups. The addrRewrite maps the ISR
	// member ids "n2"/"n3" to the backups' listen addresses.
	primaryTr, err := NewNetTransport(":0", nil, nil, nil)
	if err != nil {
		t.Fatalf("primary transport: %v", err)
	}
	defer primaryTr.Close()
	tr := addrRewrite{
		base: primaryTr.For(shard),
		m:    map[string]string{"n2": backup2Tr.Addr(), "n3": backup3Tr.Addr()},
	}

	primaryApp := &countingApplier{}
	var clock int64 = 1
	primaryEng := New("n1", shard, threeNodeControl{}, tr, primaryApp,
		WithClock(func() int64 { return clock }))
	// Shutdown (stops the per-peer senders) must run BEFORE the transports close;
	// defers run LIFO, so this defer — declared last — fires first.
	defer primaryEng.Shutdown()
	primaryEng.GrantLease(1, 1_000_000)

	// Drive `total` concurrent Proposes, each on its own goroutine. With total >
	// pipelineWindow, ~44 of them block at admission until earlier commits free
	// window slots; the invariants must hold regardless of that interleaving.
	var (
		mu   sync.Mutex
		seqs []uint64
		errs []error
		wg   sync.WaitGroup
	)
	for i := 0; i < total; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			_, seq, err := primaryEng.Propose(ctx, []byte("op"))
			mu.Lock()
			if err != nil {
				errs = append(errs, err)
			} else {
				seqs = append(seqs, seq)
			}
			mu.Unlock()
		}()
	}
	wg.Wait()

	if len(errs) != 0 {
		t.Fatalf("got %d failed Proposes over real net, first: %v", len(errs), errs[0])
	}
	if len(seqs) != total {
		t.Fatalf("got %d successful seqs, want %d", len(seqs), total)
	}
	// The set of committed seqs is exactly 1..total: dense, gap-free, no duplicate.
	seen := make(map[uint64]bool, total)
	for _, s := range seqs {
		if s < 1 || s > total {
			t.Fatalf("seq %d out of range 1..%d", s, total)
		}
		if seen[s] {
			t.Fatalf("duplicate seq %d — corruption", s)
		}
		seen[s] = true
	}
	// committed reached total (every pipelined write full-ISR-committed).
	if got := primaryEng.Committed(); got != total {
		t.Fatalf("primary committed = %d, want %d", got, total)
	}
	// All three engines agree on the applied tail: both backups' lastApplied ==
	// primary's committed, and each applied exactly `total` writes gap-free. Poll
	// briefly — a backup's apply completes on its own receive goroutine, and while
	// full-ISR commit implies both already applied the committed seq, the poll
	// keeps the check robust against any scheduling skew.
	committed := primaryEng.Committed()
	for _, b := range []struct {
		name string
		eng  *Engine
		app  *countingApplier
	}{
		{"n2", backup2Eng, backup2App},
		{"n3", backup3Eng, backup3App},
	} {

		deadline := time.Now().Add(3 * time.Second)
		for b.eng.LastApplied() != committed && time.Now().Before(deadline) {
			time.Sleep(time.Millisecond)
		}
		if got := b.eng.LastApplied(); got != committed {
			t.Fatalf("%s lastApplied = %d, want %d (== primary committed)", b.name, got, committed)
		}
		if got := b.app.n.Load(); got != int64(total) {
			t.Fatalf("%s applied count = %d, want %d (gap-free, in order)", b.name, got, total)
		}
	}
}

// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"bufio"
	"bytes"
	"context"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// threeNodeControl: static control plane naming n1 primary, full ISR
// {n1,n2,n3}, minISR 3 (full-ISR commit — every Propose must be acked by both
// backups before it durably commits, exactly like the static-PB cluster
// wiring in cluster/pb_cluster_test.go).
type threeNodeControl struct{}

func (threeNodeControl) Epoch(int) uint64   { return 1 }
func (threeNodeControl) Primary(int) string { return "n1" }
func (threeNodeControl) ISR(int) []string   { return []string{"n1", "n2", "n3"} }
func (threeNodeControl) MinISR(int) int     { return 3 }

// benchLeaseExpiry is a lease deadline far beyond any benchmark run's
// duration; the bench's injected clock always reads 1, so this never expires.
const benchLeaseExpiry = int64(1) << 62

// BenchmarkPBReplicateThroughput drives Engine.Propose from many parallel
// goroutines against a real primary NetTransport replicating to 2 backup
// NetTransports over loopback (full ISR = 3, min-ISR 3 so every write needs
// both backups' acks to commit). Concurrency comes from fanning proposals out
// across many shards rather than many goroutines hammering one shard: Propose
// serializes the whole write path per Engine (writeMu, see engine.go), so two
// goroutines racing the SAME shard's Engine just queue behind each other and
// never present the peer link with more than one in-flight frame. Multiple
// shards, in contrast, share one pbPeerLink per peer (net_transport.go's
// linkFor is keyed by peer address only, not shard) — so concurrent Proposes
// on distinct shards submit concurrently onto the SAME peer link's send
// channel, which is exactly the condition runPBFramedWriter's batching exists
// to exploit. Reports ops/sec directly (in addition to the standard ns/op)
// and a coarse pooled-payload-borrows-per-op figure.
func BenchmarkPBReplicateThroughput(b *testing.B) {
	const numShards = 64

	backup2, err := NewNetTransport(":0", nil, nil, nil)
	if err != nil {
		b.Fatalf("backup2 transport: %v", err)
	}
	defer backup2.Close()
	backup3, err := NewNetTransport(":0", nil, nil, nil)
	if err != nil {
		b.Fatalf("backup3 transport: %v", err)
	}
	defer backup3.Close()

	primaryTr, err := NewNetTransport(":0", nil, nil, nil)
	if err != nil {
		b.Fatalf("primary transport: %v", err)
	}
	defer primaryTr.Close()

	ctrl := threeNodeControl{}
	peerAddrs := map[string]string{"n2": backup2.Addr(), "n3": backup3.Addr()}

	primaryEngs := make([]*Engine, numShards)
	for s := 0; s < numShards; s++ {
		backup2.Register(s, New("n2", s, ctrl, nil, &countingApplier{}))
		backup3.Register(s, New("n3", s, ctrl, nil, &countingApplier{}))

		tr := addrRewrite{base: primaryTr.For(s), m: peerAddrs}
		eng := New("n1", s, ctrl, tr, &countingApplier{},
			WithClock(func() int64 { return 1 }))
		eng.GrantLease(1, benchLeaseExpiry)
		primaryEngs[s] = eng
	}
	defer func() {
		for _, eng := range primaryEngs {
			eng.Shutdown() // stop each engine's per-peer sender goroutines
		}
	}()

	payload := bytes.Repeat([]byte{0xAB}, 128)
	ctx := context.Background()

	startGets := pbPayloadGets.Load()
	var nextShard int64
	var errCount int64

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		idx := int(atomic.AddInt64(&nextShard, 1)-1) % numShards //nolint:gosec // bench-only index
		eng := primaryEngs[idx]
		for pb.Next() {
			if _, _, err := eng.Propose(ctx, payload); err != nil {
				atomic.AddInt64(&errCount, 1)
			}
		}
	})
	b.StopTimer()

	if errCount > 0 {
		b.Fatalf("Propose failed %d times during the benchmark (see engine sentinel errors)", errCount)
	}

	elapsed := b.Elapsed().Seconds()
	if elapsed > 0 {
		b.ReportMetric(float64(b.N)/elapsed, "ops/sec")
	}
	gets := pbPayloadGets.Load() - startGets
	if b.N > 0 {
		b.ReportMetric(float64(gets)/float64(b.N), "pooled-gets/op")
	}
}

// countingReader wraps an io.Reader and counts how many times the underlying
// Read is invoked — a coarse, real (non-fabricated) proxy for how many
// separate socket reads it took to receive a batch of frames. If the sender's
// writev batching is engaged, many frames arrive per underlying Read because
// they were coalesced into one (or a few) writev syscalls on the send side;
// with no batching, each frame would need at least its own Read.
type countingReader struct {
	r     net.Conn
	reads atomic.Int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	c.reads.Add(1)
	return c.r.Read(p)
}

// TestPBReplicateBatchingUnderConcurrency is the companion assertion for
// BenchmarkPBReplicateThroughput's throughput number: it proves the writev
// batcher actually coalesces multiple frames per flush when many requests are
// submitted to one peer link concurrently, rather than the transport doing
// one syscall per message. It fires `total` concurrent roundTrips at a single
// pbPeerLink (releasing them together via a start barrier to maximize
// concurrent submission pressure on the shared send channel) and counts, on
// the raw TCP receiving side, how many underlying socket Read calls it took
// to read all `total` request frames back off the wire. Because a real
// net.Conn is used as the writer (pbPeerLink.writeLoop, exactly like
// production), net.Buffers.WriteTo takes the writev fast path, so frames the
// writer coalesced into one flush are written (and, on loopback, very likely
// arrive) together — fewer underlying Reads than frames is a real, observed
// signal of batching, not a fabricated one. It also captures the
// pbPayloadGets delta, confirming one pooled payload borrow per proposed
// frame.
func TestPBReplicateBatchingUnderConcurrency(t *testing.T) {
	const total = 500 // > pbSendChanBuffer/2 and >> pbWritevBatchMax: real concurrent pressure

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()

	var reads atomic.Int64
	serverDone := make(chan struct{})
	go func() {
		defer close(serverDone)
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		cr := &countingReader{r: conn}
		fr := pbFrameReader{r: bufio.NewReaderSize(cr, pbLinkReadBufSize)}
		bw := bufio.NewWriter(conn)
		for i := 0; i < total; i++ {
			req, err := fr.read()
			if err != nil {
				return
			}
			ack := AckMsg{Epoch: 1, Seq: uint64(req.shard), OK: true}
			resp := pbFrame{kind: pbKindResponse, shard: req.shard, reqID: req.reqID, payload: encodeAckMsg(nil, ack)}
			if err := writePBTestFrame(bw, &resp); err != nil {
				return
			}
		}
		reads.Store(cr.reads.Load())
	}()

	link := newPBPeerLink(ln.Addr().String(), 2*time.Second)
	defer link.close()

	startGets := pbPayloadGets.Load()

	var start sync.WaitGroup
	start.Add(1)
	var wg sync.WaitGroup
	errCh := make(chan error, total)
	for i := 0; i < total; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			start.Wait()                                                                                                               // barrier: submit all `total` roundTrips together
			msg := ReplicateMsg{Epoch: 1, Seq: uint64(i), PrevSeq: uint64(i - 1), Data: []byte("bench-replicate-payload")}             //nolint:gosec // test values
			f := &pbFrame{kind: pbKindReplicate, shard: uint32(i % 8), payload: encodeReplicateMsg(pbGetPayload(), msg), pooled: true} //nolint:gosec // test values
			if _, err := link.roundTrip(f, 10*time.Second); err != nil {
				errCh <- err
			}
		}(i)
	}
	start.Done() // release the barrier: fire every roundTrip at once
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Error(err)
	}

	select {
	case <-serverDone:
	case <-time.After(5 * time.Second):
		t.Fatal("server did not finish reading all request frames")
	}

	gets := pbPayloadGets.Load() - startGets
	if gets != int64(total) {
		t.Fatalf("pbPayloadGets delta = %d, want %d (one pooled borrow per proposed frame)", gets, total)
	}

	gotReads := reads.Load()
	avgFramesPerRead := float64(total) / float64(gotReads)
	t.Logf("batching observation: %d request frames arrived via %d underlying socket Read() calls (avg %.1f frames/read)",
		total, gotReads, avgFramesPerRead)
	if gotReads >= int64(total) {
		t.Fatalf("expected fewer Read() calls than frames (writev batching should coalesce multiple frames per read), got reads=%d frames=%d", gotReads, total)
	}
}

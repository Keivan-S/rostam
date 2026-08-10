// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"bufio"
	"fmt"
	"net"
	"sync"
	"testing"
	"time"
)

// syncReplicate adapts the async Transport.Replicate to a blocking call for
// tests that want the old request/response shape: it submits and waits for the
// single completion callback, or returns a submission error.
func syncReplicate(tr Transport, peer string, msg ReplicateMsg) (AckMsg, error) {
	type res struct {
		ack AckMsg
		err error
	}
	ch := make(chan res, 1)
	if err := tr.Replicate(peer, msg, func(ack AckMsg, err error) {
		ch <- res{ack, err}
	}); err != nil {
		return AckMsg{}, err
	}
	r := <-ch
	return r.ack, r.err
}

// captureReceiver records the last message and returns a canned ack. Guarded
// by mu: Receive() runs on the server's dispatch goroutine while the test
// reads Last() from a different goroutine, and (unlike the old unbatched
// serveConn) the response now crosses one more goroutine hop through the
// per-conn batched writer before it hits the wire — a bare struct field read
// after the network round trip has no Go memory-model guarantee of
// visibility without explicit synchronization, so this needs a real lock
// rather than relying on socket completion ordering.
type captureReceiver struct {
	mu   sync.Mutex
	last ReplicateMsg
	ack  AckMsg
}

func (c *captureReceiver) Receive(m ReplicateMsg) AckMsg {
	c.mu.Lock()
	c.last = m
	c.ack = AckMsg{Epoch: m.Epoch, Seq: m.Seq, OK: true}
	ack := c.ack
	c.mu.Unlock()
	return ack
}

// ReceiveGroup folds the per-message Receive over msgs, mirroring the engine's
// cumulative-ack contract (all records here always ack OK).
func (c *captureReceiver) ReceiveGroup(msgs []ReplicateMsg) AckMsg {
	if len(msgs) == 0 {
		return AckMsg{OK: false}
	}
	var last AckMsg
	for i := range msgs {
		last = c.Receive(msgs[i])
	}
	return last
}

// CatchupInfo answers a grow handshake with the last-seen (seq, epoch) as both
// the applied high-water and the log frontier (enough for the transport-level
// tests that exercise the frame path).
func (c *captureReceiver) CatchupInfo() CatchupInfoMsg {
	c.mu.Lock()
	defer c.mu.Unlock()
	return CatchupInfoMsg{
		Epoch:         c.last.Epoch,
		AppliedSeq:    c.last.Seq,
		FrontierSeq:   c.last.Seq,
		FrontierEpoch: c.last.Epoch,
		OK:            true,
	}
}

// Last returns the most recently received message, synchronized against
// concurrent Receive calls.
func (c *captureReceiver) Last() ReplicateMsg {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.last
}

// writeRawPBFrame serializes f (header + payload) directly to conn, bypassing
// the batched writer — a minimal reference encoder for a raw test client
// speaking the pb frame wire protocol directly (no pbPeerLink involved).
func writeRawPBFrame(conn net.Conn, f *pbFrame) error {
	var hdr [pbFrameHeaderSize]byte
	writePBFrameHdr(hdr[:], f)
	if _, err := conn.Write(hdr[:]); err != nil {
		return err
	}
	if len(f.payload) > 0 {
		if _, err := conn.Write(f.payload); err != nil {
			return err
		}
	}
	return nil
}

func TestNetTransportServerDispatch(t *testing.T) {
	srv, err := NewNetTransport(":0", nil, nil, nil)
	if err != nil {
		t.Fatalf("NewNetTransport: %v", err)
	}
	defer srv.Close()
	rcv := &captureReceiver{}
	srv.Register(5, rcv)

	// Raw client: dial, send a replicate frame for shard 5, read the ack frame.
	conn, err := net.DialTimeout("tcp", srv.Addr(), time.Second)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()
	msg := ReplicateMsg{Epoch: 2, Seq: 3, PrevSeq: 2, Data: []byte("x")}
	req := &pbFrame{kind: pbKindReplicate, shard: 5, reqID: 1, payload: encodeReplicateMsg(nil, msg)}
	if err := writeRawPBFrame(conn, req); err != nil {
		t.Fatalf("write req: %v", err)
	}
	fr := pbFrameReader{r: bufio.NewReader(conn)}
	respFrame, err := fr.read()
	if err != nil {
		t.Fatalf("read resp: %v", err)
	}
	if respFrame.kind != pbKindResponse || respFrame.reqID != req.reqID {
		t.Fatalf("resp frame: %+v", respFrame)
	}
	ack, err := decodeAckMsg(respFrame.payload)
	if err != nil {
		t.Fatalf("decode ack: %v", err)
	}
	if !ack.OK || ack.Epoch != 2 || ack.Seq != 3 {
		t.Fatalf("ack: %+v", ack)
	}
	if last := rcv.Last(); last.Seq != 3 || string(last.Data) != "x" {
		t.Fatalf("receiver saw: %+v", last)
	}
}

// TestNetTransportCloseClosesInboundConns proves Close() closes conns already
// accepted and being served, not just the listener. Without the fix, the
// serveConn goroutine for the dialed conn blocks forever in its frame reader
// after Close() returns, leaking a goroutine/fd.
func TestNetTransportCloseClosesInboundConns(t *testing.T) {
	srv, err := NewNetTransport(":0", nil, nil, nil)
	if err != nil {
		t.Fatalf("NewNetTransport: %v", err)
	}

	conn, err := net.DialTimeout("tcp", srv.Addr(), time.Second)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	// Give the accept loop a moment to register the conn and start serveConn.
	deadline := time.Now().Add(time.Second)
	for {
		srv.mu.Lock()
		n := len(srv.conns)
		srv.mu.Unlock()
		if n > 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("server never registered the inbound conn")
		}
		time.Sleep(time.Millisecond)
	}

	if err := srv.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// The server must have closed its end of the conn: a read from the client
	// side should now observe EOF/closed rather than blocking.
	conn.SetReadDeadline(time.Now().Add(time.Second))
	fr := pbFrameReader{r: bufio.NewReader(conn)}
	if _, err := fr.read(); err == nil {
		t.Fatal("expected server to close its end of the inbound conn after Close()")
	}

	// serveConn's deregister runs concurrently with Close returning; poll
	// briefly rather than asserting immediately to avoid a racy check.
	deadline = time.Now().Add(time.Second)
	for {
		srv.mu.Lock()
		n := len(srv.conns)
		srv.mu.Unlock()
		if n == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("expected tracked conns to reach 0 after Close, got %d", n)
		}
		time.Sleep(time.Millisecond)
	}
}

func TestNetTransportUnregisteredShardNacks(t *testing.T) {
	srv, err := NewNetTransport(":0", nil, nil, nil)
	if err != nil {
		t.Fatalf("NewNetTransport: %v", err)
	}
	defer srv.Close()
	conn, _ := net.DialTimeout("tcp", srv.Addr(), time.Second)
	defer conn.Close()
	req := &pbFrame{kind: pbKindReplicate, shard: 99, reqID: 1, payload: encodeReplicateMsg(nil, ReplicateMsg{Epoch: 1, Seq: 1})}
	_ = writeRawPBFrame(conn, req)
	fr := pbFrameReader{r: bufio.NewReader(conn)}
	respFrame, err := fr.read()
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	ack, _ := decodeAckMsg(respFrame.payload)
	if ack.OK {
		t.Fatal("unregistered shard must NACK (OK=false)")
	}
}

func TestNetTransportClientReplicate(t *testing.T) {
	// One transport acts as the server (registers a receiver); a second
	// transport's shard-scoped client view sends to it as a client.
	backup, err := NewNetTransport(":0", nil, nil, nil)
	if err != nil {
		t.Fatalf("backup transport: %v", err)
	}
	defer backup.Close()
	rcv := &captureReceiver{}
	backup.Register(1, rcv)

	primary, err := NewNetTransport(":0", nil, nil, nil)
	if err != nil {
		t.Fatalf("primary transport: %v", err)
	}
	defer primary.Close()

	tr := primary.For(1) // shard-scoped client view
	ack, err := syncReplicate(tr, backup.Addr(), ReplicateMsg{Epoch: 4, Seq: 5, PrevSeq: 4, Data: []byte("d")})
	if err != nil {
		t.Fatalf("Replicate: %v", err)
	}
	if !ack.OK || ack.Epoch != 4 || ack.Seq != 5 {
		t.Fatalf("ack: %+v", ack)
	}
	if last := rcv.Last(); last.Seq != 5 {
		t.Fatalf("backup did not receive seq 5: %+v", last)
	}

	// A second call reuses the peer's shared pipelined link (no assertion on
	// identity, just that it still works).
	if _, err := syncReplicate(tr, backup.Addr(), ReplicateMsg{Epoch: 4, Seq: 6, PrevSeq: 5, Data: []byte("e")}); err != nil {
		t.Fatalf("second Replicate: %v", err)
	}
}

// TestNetTransportPipelinedReplicate fires many concurrent Replicate calls
// from one primary transport to a single backup peer and asserts every one
// gets its own correctly correlated ack — proving requests to the same peer
// (even across different shards) pipeline over the peer's single shared
// pbPeerLink instead of serializing one-conn-per-request.
func TestNetTransportPipelinedReplicate(t *testing.T) {
	const shards = 8
	const perShard = 25
	const total = shards * perShard

	backup, err := NewNetTransport(":0", nil, nil, nil)
	if err != nil {
		t.Fatalf("backup transport: %v", err)
	}
	defer backup.Close()
	recvs := make([]*captureReceiver, shards)
	for i := 0; i < shards; i++ {
		recvs[i] = &captureReceiver{}
		backup.Register(i, recvs[i])
	}

	primary, err := NewNetTransport(":0", nil, nil, nil)
	if err != nil {
		t.Fatalf("primary transport: %v", err)
	}
	defer primary.Close()

	var wg sync.WaitGroup
	errCh := make(chan error, total)
	for shard := 0; shard < shards; shard++ {
		tr := primary.For(shard)
		for i := 0; i < perShard; i++ {
			wg.Add(1)
			go func(shard, i int) {
				defer wg.Done()
				seq := uint64(i + 1) //nolint:gosec // test values
				msg := ReplicateMsg{Epoch: 1, Seq: seq, PrevSeq: seq - 1, Data: []byte("op")}
				ack, err := syncReplicate(tr, backup.Addr(), msg)
				if err != nil {
					errCh <- fmt.Errorf("shard %d req %d: Replicate: %w", shard, i, err)
					return
				}
				if !ack.OK || ack.Seq != seq {
					errCh <- fmt.Errorf("shard %d req %d: ack mismatch: %+v", shard, i, ack)
				}
			}(shard, i)
		}
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Error(err)
	}
}

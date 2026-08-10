// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"bufio"
	"bytes"
	"sync"
	"testing"
	"time"
)

// pbSyncBuf is a concurrency-safe io.Writer that records everything written
// and closes `full` once at least `target` bytes have been received, so a
// test can deterministically know the framed writer has emitted the whole
// expected stream before it stops the writer.
type pbSyncBuf struct {
	mu     sync.Mutex
	buf    bytes.Buffer
	target int
	once   sync.Once
	full   chan struct{}
}

func newPBSyncBuf(target int) *pbSyncBuf {
	return &pbSyncBuf{target: target, full: make(chan struct{})}
}

func (s *pbSyncBuf) Write(p []byte) (int, error) {
	s.mu.Lock()
	n, err := s.buf.Write(p)
	reached := s.buf.Len() >= s.target
	s.mu.Unlock()
	if reached {
		s.once.Do(func() { close(s.full) })
	}
	return n, err
}

func (s *pbSyncBuf) bytes() []byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]byte(nil), s.buf.Bytes()...)
}

// encodePBFrames independently serializes frames via the reference
// writePBFrameHdr + payload loop (NOT via the batcher), so it can act as the
// ground truth the batcher's output is compared against.
func encodePBFrames(frames ...pbFrame) []byte {
	var buf bytes.Buffer
	for i := range frames {
		var hdr [pbFrameHeaderSize]byte
		writePBFrameHdr(hdr[:], &frames[i])
		buf.Write(hdr[:])
		if len(frames[i].payload) > 0 {
			buf.Write(frames[i].payload)
		}
	}
	return buf.Bytes()
}

// TestPBWritevBatcherByteIdentical proves runPBFramedWriter's writev/
// net.Buffers output is byte-for-byte identical to the reference
// writePBFrameHdr path, across a mix of header-only/payload-bearing and
// pooled/unpooled frames, spanning several writev batches (>pbWritevBatchMax
// frames). It also confirms a pbFrameReader reads all frames back correctly.
func TestPBWritevBatcherByteIdentical(t *testing.T) {
	const total = 200 // > pbWritevBatchMax (64): exercises multiple batches
	frames := make([]pbFrame, total)
	for i := range frames {
		var payload []byte
		switch i % 4 {
		case 0:
			payload = nil // header-only frame
		case 1:
			payload = []byte("x")
		case 2:
			payload = bytes.Repeat([]byte{byte(i)}, i%700) // some exceed pbPayloadInitCap
		default:
			payload = []byte("replicate-msg-payload")
		}
		frames[i] = pbFrame{
			kind:    uint8(i % 256), //nolint:gosec // test values
			shard:   uint32(i * 7),  //nolint:gosec // test values
			reqID:   uint64(i) << 20,
			payload: payload,
			pooled:  i%2 == 0,
		}
	}

	want := encodePBFrames(frames...)

	w := newPBSyncBuf(len(want))
	sendCh := make(chan *pbFrame, total)
	done := make(chan struct{})
	for i := range frames {
		f := frames[i] // copy: the writer recycles pooled payload buffers
		sendCh <- &f
	}

	errCh := make(chan error, 1)
	go func() { errCh <- runPBFramedWriter(w, sendCh, done, 0) }()

	select {
	case <-w.full:
	case <-time.After(5 * time.Second):
		t.Fatal("writer did not emit all expected bytes")
	}
	close(done)
	if err := <-errCh; err != nil {
		t.Fatalf("runPBFramedWriter: %v", err)
	}

	got := w.bytes()
	if !bytes.Equal(got, want) {
		t.Fatalf("wire bytes mismatch: got %d bytes, want %d bytes", len(got), len(want))
	}

	fr := &pbFrameReader{r: bufio.NewReader(bytes.NewReader(got))}
	for i, want := range frames {
		gotF, err := fr.read()
		if err != nil {
			t.Fatalf("frame %d: read: %v", i, err)
		}
		if gotF.kind != want.kind || gotF.shard != want.shard || gotF.reqID != want.reqID || !bytes.Equal(gotF.payload, want.payload) {
			t.Fatalf("frame %d mismatch: got %+v want %+v", i, gotF, want)
		}
	}
}

// TestPBPayloadPoolRecycle feeds a mix of pooled and unpooled frames through
// the writer and asserts pbPutPayload is called exactly once per pooled frame
// (pbPayloadGets/pbPayloadPuts deltas) — never twice (which would corrupt the
// pool) and never skipped (which would leak). The final partial batch at
// shutdown is included.
func TestPBPayloadPoolRecycle(t *testing.T) {
	const total = 130 // spans multiple writev batches + a partial final batch
	frames := make([]pbFrame, total)
	pooled := 0
	startGets := pbPayloadGets.Load()
	startPuts := pbPayloadPuts.Load()
	for i := range frames {
		var payload []byte
		isPooled := i%2 == 0
		if isPooled {
			payload = pbGetPayload()
			payload = append(payload, []byte("replicate-payload")...)
			pooled++
		} else {
			payload = []byte("replicate-payload")
		}
		frames[i] = pbFrame{kind: pbKindReplicate, reqID: uint64(i), payload: payload, pooled: isPooled}
	}

	want := encodePBFrames(frames...)
	w := newPBSyncBuf(len(want))
	sendCh := make(chan *pbFrame, total)
	done := make(chan struct{})
	for i := range frames {
		f := frames[i]
		sendCh <- &f
	}

	errCh := make(chan error, 1)
	go func() { errCh <- runPBFramedWriter(w, sendCh, done, 0) }()

	select {
	case <-w.full:
	case <-time.After(5 * time.Second):
		t.Fatal("writer did not emit all expected bytes")
	}
	close(done)
	if err := <-errCh; err != nil { // writer returns only after the final flush's puts run
		t.Fatalf("runPBFramedWriter: %v", err)
	}

	gets := pbPayloadGets.Load() - startGets
	if gets != int64(pooled) {
		t.Fatalf("pbPayloadGets delta = %d, want %d", gets, pooled)
	}
	if puts := pbPayloadPuts.Load() - startPuts; puts != gets {
		t.Fatalf("pbPayloadPuts delta = %d, want %d (exactly one put per get)", puts, gets)
	}
	// Wire bytes must still be identical regardless of pooling.
	if got := w.bytes(); !bytes.Equal(got, want) {
		t.Fatalf("wire bytes mismatch with pooling: got %d bytes, want %d bytes", len(got), len(want))
	}
}

// TestPBGetPutRoundTrip covers the borrow/return primitives directly: a
// borrowed buffer is always zero-length (so callers append from the start), a
// fresh (pool-miss) buffer carries the reserved capacity, and an oversized
// buffer is dropped rather than retained (while still counting as a return
// call).
func TestPBGetPutRoundTrip(t *testing.T) {
	// A fresh (pool-miss) buffer reserves pbPayloadInitCap. sync.Pool is global
	// and other tests seed it with smaller buffers, so exercise the New path
	// directly rather than relying on pool state.
	fresh := *(pbPayloadPool.New().(*[]byte))
	if len(fresh) != 0 || cap(fresh) < pbPayloadInitCap {
		t.Fatalf("fresh buffer: len=%d cap=%d, want len 0 cap >= %d", len(fresh), cap(fresh), pbPayloadInitCap)
	}

	// A borrowed buffer is always zero-length regardless of what was pooled.
	pbPutPayload(make([]byte, 8)) // seed a non-zero-len buffer
	if b := pbGetPayload(); len(b) != 0 {
		t.Fatalf("pbGetPayload len = %d, want 0", len(b))
	}

	before := pbPayloadPuts.Load()
	pbPutPayload(make([]byte, pbPayloadMaxCap+1)) // oversized: dropped, not retained
	if got := pbPayloadPuts.Load(); got != before+1 {
		t.Fatalf("pbPutPayload counter = %d, want %d (counts even when dropping oversized)", got, before+1)
	}
}

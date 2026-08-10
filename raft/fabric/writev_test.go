// SPDX-License-Identifier: Apache-2.0

package fabric

import (
	"bytes"
	"sync"
	"testing"
	"time"

	hraft "github.com/hashicorp/raft"
)

// syncBuf is a concurrency-safe io.Writer that records everything written and
// closes `full` once at least `target` bytes have been received, so a test can
// deterministically know the framed writer has emitted the whole expected
// stream before it stops the writer.
type syncBuf struct {
	mu     sync.Mutex
	buf    bytes.Buffer
	target int
	once   sync.Once
	full   chan struct{}
}

func newSyncBuf(target int) *syncBuf {
	return &syncBuf{target: target, full: make(chan struct{})}
}

func (s *syncBuf) Write(p []byte) (int, error) {
	s.mu.Lock()
	n, err := s.buf.Write(p)
	reached := s.buf.Len() >= s.target
	s.mu.Unlock()
	if reached {
		s.once.Do(func() { close(s.full) })
	}
	return n, err
}

func (s *syncBuf) bytes() []byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]byte(nil), s.buf.Bytes()...)
}

// TestRunFramedWriterByteIdentical proves the writev/net.Buffers writer emits
// bytes byte-for-byte identical to the reference writeFrame path, across a mix
// of pooled/unpooled frames and zero/non-zero payloads, spanning several writev
// batches (>writevBatchMax frames).
func TestRunFramedWriterByteIdentical(t *testing.T) {
	const total = 200 // > writevBatchMax (64): exercises multiple batches
	frames := make([]frame, total)
	for i := range frames {
		var payload []byte
		switch i % 4 {
		case 0:
			payload = nil // header-only frame
		case 1:
			payload = []byte("x")
		case 2:
			payload = bytes.Repeat([]byte{byte(i)}, i%700) // some exceed payloadInitCap
		default:
			payload = []byte("append-entries-payload")
		}
		frames[i] = frame{
			kind:    uint8(i % 256), //nolint:gosec // test values
			groupID: uint32(i * 7),  //nolint:gosec // test values
			reqID:   uint64(i) << 20,
			payload: payload,
			pooled:  i%2 == 0,
		}
	}

	// Reference wire bytes via the untouched writeFrame path.
	want := encodeFrames(t, frames...)

	w := newSyncBuf(len(want))
	sendCh := make(chan *frame, total)
	done := make(chan struct{})
	for i := range frames {
		f := frames[i] // copy: the writer recycles pooled payload buffers
		sendCh <- &f
	}

	errCh := make(chan error, 1)
	go func() { errCh <- runFramedWriter(w, sendCh, done, 0) }()

	select {
	case <-w.full:
	case <-time.After(5 * time.Second):
		t.Fatal("writer did not emit all expected bytes")
	}
	close(done)
	if err := <-errCh; err != nil {
		t.Fatalf("runFramedWriter: %v", err)
	}

	if got := w.bytes(); !bytes.Equal(got, want) {
		t.Fatalf("wire bytes mismatch: got %d bytes, want %d bytes", len(got), len(want))
	}
}

// TestWriterRecyclesPooledPayloadsExactlyOnce feeds a mix of pooled and
// unpooled frames through the writer and asserts putPayload is called exactly
// once per pooled frame — never twice (which would corrupt the pool) and never
// skipped (which would leak). The final partial batch at shutdown is included.
func TestWriterRecyclesPooledPayloadsExactlyOnce(t *testing.T) {
	const total = 130 // spans multiple writev batches + a partial final batch
	frames := make([]frame, total)
	pooled := 0
	for i := range frames {
		var payload []byte
		isPooled := i%2 == 0
		if isPooled {
			payload = encodeAppendEntriesRequest(getPayload(), &hraft.AppendEntriesRequest{Term: uint64(i)})
			pooled++
		} else {
			payload = encodeAppendEntriesRequest(nil, &hraft.AppendEntriesRequest{Term: uint64(i)})
		}
		frames[i] = frame{kind: rpcAppendEntries, reqID: uint64(i), payload: payload, pooled: isPooled}
	}

	want := encodeFrames(t, frames...)
	w := newSyncBuf(len(want))
	sendCh := make(chan *frame, total)
	done := make(chan struct{})
	for i := range frames {
		f := frames[i]
		sendCh <- &f
	}

	startPuts := payloadPuts.Load()
	errCh := make(chan error, 1)
	go func() { errCh <- runFramedWriter(w, sendCh, done, 0) }()

	select {
	case <-w.full:
	case <-time.After(5 * time.Second):
		t.Fatal("writer did not emit all expected bytes")
	}
	close(done)
	if err := <-errCh; err != nil { // writer returns only after the final flush's puts run
		t.Fatalf("runFramedWriter: %v", err)
	}

	if delta := payloadPuts.Load() - startPuts; delta != int64(pooled) {
		t.Fatalf("putPayload called %d times, want %d (exactly one per pooled frame)", delta, pooled)
	}
	// Wire bytes must still be identical regardless of pooling.
	if got := w.bytes(); !bytes.Equal(got, want) {
		t.Fatalf("wire bytes mismatch with pooling: got %d bytes, want %d bytes", len(got), len(want))
	}
}

// TestGetPutPayload covers the borrow/return primitives directly: a borrowed
// buffer is always zero-length (so encodeXxx appends from the start), a fresh
// (pool-miss) buffer carries the reserved capacity, and an oversized buffer is
// dropped rather than retained (while still counting as a return call).
func TestGetPutPayload(t *testing.T) {
	// A fresh (pool-miss) buffer reserves payloadInitCap. sync.Pool is global and
	// other tests seed it with smaller buffers, so exercise the New path directly
	// rather than relying on pool state.
	fresh := *(payloadPool.New().(*[]byte))
	if len(fresh) != 0 || cap(fresh) < payloadInitCap {
		t.Fatalf("fresh buffer: len=%d cap=%d, want len 0 cap >= %d", len(fresh), cap(fresh), payloadInitCap)
	}

	// A borrowed buffer is always zero-length regardless of what was pooled.
	putPayload(make([]byte, 8)) // seed a non-zero-len buffer
	if b := getPayload(); len(b) != 0 {
		t.Fatalf("getPayload len = %d, want 0", len(b))
	}

	before := payloadPuts.Load()
	putPayload(make([]byte, payloadMaxCap+1)) // oversized: dropped, not retained
	if got := payloadPuts.Load(); got != before+1 {
		t.Fatalf("putPayload counter = %d, want %d (counts even when dropping oversized)", got, before+1)
	}
}

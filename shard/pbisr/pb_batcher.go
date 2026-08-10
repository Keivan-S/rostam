// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"io"
	"net"
	"sync"
	"sync/atomic"
	"time"
)

// pbWriteLinger is the coalescing window the framed writer waits for
// additional frames before flushing. Measured OFF (0) by default, matching
// raft/fabric's writeLinger: on a co-located, CPU-underutilized cluster the
// replicated-write path is latency-bound, not syscall-bound, so trading
// round-trip latency for fewer write()s lowers throughput. The mechanism is
// retained (>0 enables it) for CPU-saturated deployments where amortizing the
// syscall matters more than the added latency; with linger=0 the writer still
// coalesces whatever is already queued.
const pbWriteLinger = 0

// Outbound payload buffer pool for the batched-transport send path. Ported
// from raft/fabric/frame.go's payloadPool: the encode path builds each
// payload into a buffer borrowed from pbGetPayload; the framed writer returns
// it via pbPutPayload once its bytes have been committed to the wire. Only
// the SEND side pools: read payloads are never pooled. A stored *[]byte (not
// a bare []byte) keeps Put from boxing the slice header into the interface on
// every return.
const (
	// pbPayloadInitCap is the reserved capacity of a fresh pooled buffer —
	// sized to cover a typical replicate payload without a re-grow.
	pbPayloadInitCap = 512
	// pbPayloadMaxCap bounds what we retain: a payload that grew past this (a
	// large entry batch) is dropped rather than pooled, so an occasional big
	// send does not pin an oversized buffer in every P-local pool slot forever.
	pbPayloadMaxCap = 64 << 10
)

var pbPayloadPool = sync.Pool{
	New: func() any {
		b := make([]byte, 0, pbPayloadInitCap)
		return &b
	},
}

// pbPayloadGets / pbPayloadPuts count pool borrows and returns. Exposed
// (package scope) for tests asserting every pooled frame is returned exactly
// once and for coarse observability of send-path buffer churn; two atomic
// adds per send are negligible next to the encode + syscall they bracket.
var (
	pbPayloadGets atomic.Int64
	pbPayloadPuts atomic.Int64
)

// pbGetPayload borrows a zero-length buffer with reserved capacity. Callers
// APPEND their encoding into it; the returned slice — which may have
// grown/reallocated — is what becomes pbFrame.payload.
func pbGetPayload() []byte {
	pbPayloadGets.Add(1)
	bp := pbPayloadPool.Get().(*[]byte)
	return (*bp)[:0]
}

// pbPutPayload returns b to the pool. It MUST be called exactly once per
// pooled frame, only after the writer has committed the frame's bytes (writev
// has consumed them), so nothing references b anymore. Oversized buffers are
// dropped to bound retained memory.
func pbPutPayload(b []byte) {
	pbPayloadPuts.Add(1)
	if cap(b) > pbPayloadMaxCap {
		return
	}
	b = b[:0]
	pbPayloadPool.Put(&b)
}

// pbWritevBatchMax bounds a single writev to N frames (=> at most 2N iovec
// entries: one header + one payload each). It caps the fixed header scratch
// and keeps each batch well under IOV_MAX; net.Buffers.WriteTo loops
// internally for anything larger anyway.
const pbWritevBatchMax = 64

// pbWritevBatcher accumulates frames and flushes them with a single
// net.Buffers.WriteTo (writev on a real conn: one syscall, no per-frame copy
// into a bufio buffer). All storage is fixed-size and reused across batches:
//   - hdr[i] holds the i-th frame's serialized header; the net.Buffers entry
//     points at hdr[i][:], so the scratch MUST outlive the WriteTo. Because
//     hdr is owned by the batcher (one per writer goroutine) and only
//     overwritten on the NEXT batch (after this WriteTo has returned), that
//     validity holds.
//   - iov is the [][]byte backing net.Buffers passes to WriteTo.
//   - toPool lists the pooled payloads to recycle after the flush.
type pbWritevBatcher struct {
	w      io.Writer
	hdr    [pbWritevBatchMax][pbFrameHeaderSize]byte
	iov    [2 * pbWritevBatchMax][]byte
	toPool [pbWritevBatchMax][]byte
	n      int // frames in the batch
	niov   int // iov entries in use
	npool  int // pooled payloads pending recycle
}

// add appends f's header and payload to the current batch. Byte-for-byte the
// same header writePBFrameHdr emits (it IS writePBFrameHdr); the payload
// slice is referenced (not copied), exactly like the header scratch, and
// stays valid until the WriteTo completes.
func (b *pbWritevBatcher) add(f *pbFrame) {
	h := &b.hdr[b.n]
	writePBFrameHdr(h[:], f)
	b.iov[b.niov] = h[:]
	b.niov++
	if len(f.payload) > 0 { // header-only frame: emit just the header
		b.iov[b.niov] = f.payload
		b.niov++
	}
	if f.pooled {
		b.toPool[b.npool] = f.payload
		b.npool++
	}
	b.n++
}

// addFlushIfFull adds f, flushing first if the batch is already full so the
// header scratch never overflows.
func (b *pbWritevBatcher) addFlushIfFull(f *pbFrame) error {
	b.add(f)
	if b.n == pbWritevBatchMax {
		return b.flush()
	}
	return nil
}

// flush writes the batch with one net.Buffers.WriteTo, recycles the batch's
// pooled payloads (AFTER the write consumed their bytes), and resets. Pooled
// buffers are returned regardless of write outcome: on error the frames are
// discarded and the conn torn down, so nothing references them either way —
// and this guarantees every pooled frame is put exactly once, even on the
// final (possibly partial) batch at shutdown. A zero-frame batch is a no-op.
func (b *pbWritevBatcher) flush() error {
	if b.n == 0 {
		return nil
	}
	bufs := net.Buffers(b.iov[:b.niov])
	_, err := bufs.WriteTo(b.w) // WriteTo consumes bufs (a local copy); b.iov backing is reused next batch
	for i := 0; i < b.npool; i++ {
		pbPutPayload(b.toPool[i])
		b.toPool[i] = nil
	}
	for i := 0; i < b.niov; i++ {
		b.iov[i] = nil // drop header/payload refs so they can't outlive the batch
	}
	b.n, b.niov, b.npool = 0, 0, 0
	return err
}

// runPBFramedWriter is the syscall-reduction core. It blocks for one frame,
// then accumulates every frame that arrives within a short `linger` window
// into a pbWritevBatcher and flushes once via net.Buffers.WriteTo — so a
// burst of frames collapses into a single writev() syscall WITHOUT copying
// each payload through a bufio buffer first. The linger is essential: at
// steady state the writer keeps pace with arrivals, so a purely non-blocking
// drain finds the channel empty after each frame and flushes per-frame (no
// batching at all). Waiting a few tens of µs lets stragglers pile in. It
// returns nil when done is closed (clean shutdown) or the write error that
// ended the loop (the caller then tears the conn down).
func runPBFramedWriter(w io.Writer, sendCh <-chan *pbFrame, done <-chan struct{}, linger time.Duration) error {
	b := pbWritevBatcher{w: w}
	// linger<=0 keeps the old non-blocking greedy drain (used by tests / a way
	// to disable batching); otherwise a reusable timer bounds the coalescing
	// window.
	var timer *time.Timer
	for {
		select {
		case f := <-sendCh:
			if err := b.addFlushIfFull(f); err != nil {
				return err
			}
			if linger <= 0 {
			drain:
				for {
					select {
					case f2 := <-sendCh:
						if err := b.addFlushIfFull(f2); err != nil {
							return err
						}
					default:
						break drain
					}
				}
			} else {
				if timer == nil {
					timer = time.NewTimer(linger)
				} else {
					timer.Reset(linger)
				}
			linger:
				for {
					select {
					case f2 := <-sendCh:
						if err := b.addFlushIfFull(f2); err != nil {
							stopPBTimer(timer)
							return err
						}
					case <-timer.C:
						break linger
					case <-done:
						_ = b.flush() // recycles the pending batch's pooled payloads
						return nil
					}
				}
			}
			if err := b.flush(); err != nil {
				return err
			}
		case <-done:
			return nil
		}
	}
}

// stopPBTimer stops t and drains its channel if the fire already happened, so
// the next Reset starts clean.
func stopPBTimer(t *time.Timer) {
	if !t.Stop() {
		select {
		case <-t.C:
		default:
		}
	}
}

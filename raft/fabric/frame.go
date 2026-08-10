// SPDX-License-Identifier: Apache-2.0

package fabric

import (
	"bufio"
	"encoding/binary"
	"errors"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"time"
)

// writeLinger is the coalescing window the framed writer waits for additional
// frames before flushing. Measured OFF (0) by default: on a co-located,
// CPU-underutilized cluster the replicated-write path is latency-bound, not
// syscall-bound, so trading round-trip latency for fewer write()s lowers
// throughput. The mechanism is retained (>0 enables it) for CPU-saturated
// deployments where amortizing the syscall matters more than the added latency;
// with linger=0 the writer still coalesces whatever is already queued.
const writeLinger = 0

// Wire frame (shared mux link), little-endian header, length-delimited:
//
//	[magic u8][ver u8][kind u8][groupID u32][reqID u64][payloadLen u32][payload…]
//
// kind's high bit marks a response; the low 7 bits are the rpcType (matching the
// hashicorp iota). groupID routes a REQUEST to that group's Consumer; on a
// response it is informational because correlation is by reqID (groups interleave
// on the shared conn, so responses can arrive out of group-order). The payload is
// the zero-reflection big-endian codec output from codec.go.
const (
	frameMagic   uint8 = 0x52 // 'R' for Rostam
	frameVersion uint8 = 1

	// frameHeaderSize is magic+ver+kind (3) + groupID(4) + reqID(8) + payloadLen(4).
	frameHeaderSize = 1 + 1 + 1 + 4 + 8 + 4

	// kindResponse is the high bit of kind; set on responses.
	kindResponse uint8 = 0x80
	// rpcTypeMask extracts the rpcType from kind.
	rpcTypeMask uint8 = 0x7f

	// maxPayload bounds a single framed payload so a corrupt/hostile length
	// cannot drive an unbounded allocation. Snapshots stream over a dedicated
	// conn (never framed), so mux frames are small; 256 MiB is generous.
	maxPayload uint32 = 256 << 20
)

// rpcType constants match the hashicorp/raft net_transport iota so the low 7
// bits of kind are wire-compatible in meaning.
const (
	rpcAppendEntries uint8 = iota
	rpcRequestVote
	rpcInstallSnapshot
	rpcTimeoutNow
	rpcRequestPreVote
)

var (
	errBadMagic   = errors.New("fabric: bad frame magic")
	errBadVersion = errors.New("fabric: unsupported frame version")
	errOversize   = errors.New("fabric: frame payload exceeds max")
)

// frame is one multiplexed message. payload aliases a per-read buffer on the
// receive path, so consumers that outlive the read must copy (the codecs do).
//
// pooled marks an OUTBOUND payload whose buffer came from getPayload and must be
// returned via putPayload once the framed writer has committed its bytes to the
// wire. It is only ever set on the encode/send path (never on a read frame,
// whose payload sub-slices may be retained by Raft downstream).
type frame struct {
	kind    uint8
	groupID uint32
	reqID   uint64
	payload []byte
	pooled  bool
}

func (f *frame) isResponse() bool { return f.kind&kindResponse != 0 }
func (f *frame) rpcType() uint8   { return f.kind & rpcTypeMask }

// writeFrame serializes f's header (little-endian) and payload to w. It performs
// no flush; the batching writer flushes once per drained burst.
func writeFrame(w *bufio.Writer, f *frame) error {
	var hdr [frameHeaderSize]byte
	hdr[0] = frameMagic
	hdr[1] = frameVersion
	hdr[2] = f.kind
	binary.LittleEndian.PutUint32(hdr[3:], f.groupID)
	binary.LittleEndian.PutUint64(hdr[7:], f.reqID)
	binary.LittleEndian.PutUint32(hdr[15:], uint32(len(f.payload))) //nolint:gosec // bounded by maxPayload upstream
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	if len(f.payload) > 0 {
		if _, err := w.Write(f.payload); err != nil {
			return err
		}
	}
	return nil
}

// frameReader reassembles frames from a stream, tolerating partial TCP reads
// (io.ReadFull loops until the full header/payload arrives) and validating the
// magic, version, and payload-length bound before allocating.
type frameReader struct {
	r   *bufio.Reader
	hdr [frameHeaderSize]byte
}

// read returns the next frame, or an error (io.EOF at a clean stream end,
// errBadMagic/errBadVersion/errOversize on a malformed stream, or the underlying
// read error). The returned payload is freshly allocated per call.
func (fr *frameReader) read() (frame, error) {
	if _, err := io.ReadFull(fr.r, fr.hdr[:]); err != nil {
		return frame{}, err
	}
	if fr.hdr[0] != frameMagic {
		return frame{}, errBadMagic
	}
	if fr.hdr[1] != frameVersion {
		return frame{}, errBadVersion
	}
	f := frame{
		kind:    fr.hdr[2],
		groupID: binary.LittleEndian.Uint32(fr.hdr[3:]),
		reqID:   binary.LittleEndian.Uint64(fr.hdr[7:]),
	}
	plen := binary.LittleEndian.Uint32(fr.hdr[15:])
	if plen > maxPayload {
		return frame{}, errOversize
	}
	if plen > 0 {
		f.payload = make([]byte, plen)
		if _, err := io.ReadFull(fr.r, f.payload); err != nil {
			return frame{}, err
		}
	}
	return f, nil
}

// Outbound payload buffer pool. The encode/send path (the 4 sync RPCs, the
// pipeline AppendEntries, and the response frames) builds each payload into a
// buffer borrowed from getPayload; the framed writer returns it via putPayload
// once its bytes have been committed to the wire. Only the SEND side pools: read
// payloads sub-slice into Raft structs that outlive the read, so they are never
// pooled. A stored *[]byte (not a bare []byte) keeps Put from boxing the slice
// header into the interface on every return.
const (
	// payloadInitCap is the reserved capacity of a fresh pooled buffer — sized to
	// cover a typical AppendEntries/vote payload without a re-grow.
	payloadInitCap = 512
	// payloadMaxCap bounds what we retain: a payload that grew past this (a large
	// entry batch) is dropped rather than pooled, so an occasional big send does
	// not pin an oversized buffer in every P-local pool slot forever.
	payloadMaxCap = 64 << 10
)

var payloadPool = sync.Pool{
	New: func() any {
		b := make([]byte, 0, payloadInitCap)
		return &b
	},
}

// payloadGets / payloadPuts count pool borrows and returns. Exposed (package
// scope) for tests asserting every pooled frame is returned exactly once and for
// coarse observability of send-path buffer churn; two atomic adds per send are
// negligible next to the encode + syscall they bracket.
var (
	payloadGets atomic.Int64
	payloadPuts atomic.Int64
)

// getPayload borrows a zero-length buffer with reserved capacity. Callers APPEND
// their encoding into it (encodeXxx(getPayload(), args)); the returned slice —
// which may have grown/reallocated — is what becomes frame.payload.
func getPayload() []byte {
	payloadGets.Add(1)
	bp := payloadPool.Get().(*[]byte)
	return (*bp)[:0]
}

// putPayload returns b to the pool. It MUST be called exactly once per pooled
// frame, only after the writer has committed the frame's bytes (writev has
// consumed them), so nothing references b anymore. Oversized buffers are dropped
// to bound retained memory.
func putPayload(b []byte) {
	payloadPuts.Add(1)
	if cap(b) > payloadMaxCap {
		return
	}
	b = b[:0]
	payloadPool.Put(&b)
}

// writevBatchMax bounds a single writev to N frames (=> at most 2N iovec
// entries: one header + one payload each). It caps the fixed header scratch and
// keeps each batch well under IOV_MAX; net.Buffers.WriteTo loops internally for
// anything larger anyway.
const writevBatchMax = 64

// writevBatcher accumulates frames and flushes them with a single
// net.Buffers.WriteTo (writev on a real conn: one syscall, no per-frame copy
// into a bufio buffer). All storage is fixed-size and reused across batches:
//   - hdr[i] holds the i-th frame's serialized header; the net.Buffers entry
//     points at hdr[i][:], so the scratch MUST outlive the WriteTo. Because hdr
//     is owned by the batcher (one per writer goroutine) and only overwritten on
//     the NEXT batch (after this WriteTo has returned), that validity holds.
//   - iov is the [][]byte backing net.Buffers passes to WriteTo.
//   - toPool lists the pooled payloads to recycle after the flush.
type writevBatcher struct {
	w      io.Writer
	hdr    [writevBatchMax][frameHeaderSize]byte
	iov    [2 * writevBatchMax][]byte
	toPool [writevBatchMax][]byte
	n      int // frames in the batch
	niov   int // iov entries in use
	npool  int // pooled payloads pending recycle
}

// add appends f's header and payload to the current batch. Byte-for-byte the
// same header writeFrame emits; the payload slice is referenced (not copied),
// exactly like the header scratch, and stays valid until the WriteTo completes.
func (b *writevBatcher) add(f *frame) {
	h := &b.hdr[b.n]
	h[0] = frameMagic
	h[1] = frameVersion
	h[2] = f.kind
	binary.LittleEndian.PutUint32(h[3:], f.groupID)
	binary.LittleEndian.PutUint64(h[7:], f.reqID)
	binary.LittleEndian.PutUint32(h[15:], uint32(len(f.payload))) //nolint:gosec // bounded by maxPayload upstream
	b.iov[b.niov] = h[:]
	b.niov++
	if len(f.payload) > 0 { // header-only frame: emit just the header, matching writeFrame
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
func (b *writevBatcher) addFlushIfFull(f *frame) error {
	b.add(f)
	if b.n == writevBatchMax {
		return b.flush()
	}
	return nil
}

// flush writes the batch with one net.Buffers.WriteTo, recycles the batch's
// pooled payloads (AFTER the write consumed their bytes), and resets. Pooled
// buffers are returned regardless of write outcome: on error the frames are
// discarded and the conn torn down, so nothing references them either way — and
// this guarantees every pooled frame is put exactly once, even on the final
// (possibly partial) batch at shutdown. A zero-frame batch is a no-op.
func (b *writevBatcher) flush() error {
	if b.n == 0 {
		return nil
	}
	bufs := net.Buffers(b.iov[:b.niov])
	_, err := bufs.WriteTo(b.w) // WriteTo consumes bufs (a local copy); b.iov backing is reused next batch
	for i := 0; i < b.npool; i++ {
		putPayload(b.toPool[i])
		b.toPool[i] = nil
	}
	for i := 0; i < b.niov; i++ {
		b.iov[i] = nil // drop header/payload refs so they can't outlive the batch
	}
	b.n, b.niov, b.npool = 0, 0, 0
	return err
}

// runFramedWriter is the syscall-reduction core. It blocks for one frame, then
// accumulates every frame that arrives within a short `linger` window into a
// writevBatcher and flushes once via net.Buffers.WriteTo — so a burst of frames
// from many interleaved groups collapses into a single writev() syscall WITHOUT
// copying each payload through a bufio buffer first. The linger is essential: at
// steady state the writer keeps pace with arrivals, so a purely non-blocking
// drain finds the channel empty after each frame and flushes per-frame (no
// batching at all). Waiting a few tens of µs lets stragglers from other groups
// pile in. It returns nil when done is closed (clean shutdown) or the write
// error that ended the loop (the caller then tears the conn down).
func runFramedWriter(w io.Writer, sendCh <-chan *frame, done <-chan struct{}, linger time.Duration) error {
	b := writevBatcher{w: w}
	// linger<=0 keeps the old non-blocking greedy drain (used by tests / a way to
	// disable batching); otherwise a reusable timer bounds the coalescing window.
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
							stopTimer(timer)
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

// stopTimer stops t and drains its channel if the fire already happened, so the
// next Reset starts clean.
func stopTimer(t *time.Timer) {
	if !t.Stop() {
		select {
		case <-t.C:
		default:
		}
	}
}

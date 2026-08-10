// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"sync"
)

// connBufSize is the bufio buffer size for both read and write per conn.
const connBufSize = 8192

// connBufRetainCap is the soft ceiling on the per-connection request buffer that
// handleConn keeps reused across requests. A buffer grown past this (by a one-off
// large frame, up to MaxFrameSize) is released back to a small allocation after
// the request so a single big request does not pin MaxFrameSize for the
// connection's whole lifetime (a slow-loris-style memory amplification at
// MaxConns). 64 KiB comfortably covers ordinary request payloads.
const connBufRetainCap = 64 << 10

// frameBufPool holds reusable byte slices for frame bodies (request payloads).
// New slices start at 256 bytes; grown as needed.
var frameBufPool = sync.Pool{
	New: func() any {
		b := make([]byte, 0, 256)
		return &b
	},
}

func getFrameBuf() *[]byte {
	return frameBufPool.Get().(*[]byte)
}

func putFrameBuf(b *[]byte) {
	*b = (*b)[:0]
	frameBufPool.Put(b)
}

// readFrame reads a 4-byte big-endian length followed by `length` body bytes
// from r into a pooled buffer. Returns the buffer (caller must putFrameBuf).
// Returns io.EOF on clean disconnect.
func readFrame(r *bufio.Reader) (*[]byte, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint32(hdr[:])
	if n == 0 || n > MaxFrameSize {
		return nil, fmt.Errorf("server: invalid frame length %d", n)
	}
	bufPtr := getFrameBuf()
	buf := *bufPtr
	if cap(buf) < int(n) {
		buf = make([]byte, n)
	} else {
		buf = buf[:n]
	}
	if _, err := io.ReadFull(r, buf); err != nil {
		putFrameBuf(&buf)
		return nil, err
	}
	*bufPtr = buf
	return bufPtr, nil
}

// writeFrame writes a 4-byte big-endian length header followed by body
// to w. Caller is responsible for calling w.Flush() once they're done
// batching writes.
func writeFrame(w *bufio.Writer, body []byte) error {
	if len(body) == 0 || len(body) > MaxFrameSize {
		return errors.New("server: writeFrame body length out of range")
	}
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(body))) //nolint:gosec // bounded above
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	_, err := w.Write(body)
	return err
}

// writeResponse writes a response frame directly to w, skipping the
// intermediate []byte EncodeResponse would build. Wire layout matches
// EncodeResponse framed by writeFrame:
//
//	[bodyLen:4][status:1][payloadLen:4][payload]
//
// Caller passes a scratch buffer they own — declaring `var hdr [9]byte`
// inline would heap-allocate per call because the slice flows into
// bufio.Writer.Write, whose interface call to the underlying conn is
// opaque to escape analysis. Caller flushes once batching is done.
func writeResponse(w *bufio.Writer, hdr *[9]byte, status uint8, payload []byte) error {
	bodyLen := 1 + 4 + len(payload)
	if bodyLen > MaxFrameSize {
		return errors.New("server: response exceeds MaxFrameSize")
	}
	binary.BigEndian.PutUint32(hdr[0:4], uint32(bodyLen)) //nolint:gosec // bounded above
	hdr[4] = status
	binary.BigEndian.PutUint32(hdr[5:9], uint32(len(payload))) //nolint:gosec // bounded above
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	if len(payload) == 0 {
		return nil
	}
	_, err := w.Write(payload)
	return err
}

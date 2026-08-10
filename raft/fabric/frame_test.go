// SPDX-License-Identifier: Apache-2.0

package fabric

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"testing"
	"testing/iotest"
)

// encodeFrames serializes frames into a single byte stream.
func encodeFrames(t *testing.T, frames ...frame) []byte {
	t.Helper()
	var buf bytes.Buffer
	bw := bufio.NewWriter(&buf)
	for i := range frames {
		if err := writeFrame(bw, &frames[i]); err != nil {
			t.Fatalf("writeFrame: %v", err)
		}
	}
	if err := bw.Flush(); err != nil {
		t.Fatalf("flush: %v", err)
	}
	return buf.Bytes()
}

func framesEqual(a, b frame) bool {
	return a.kind == b.kind && a.groupID == b.groupID && a.reqID == b.reqID &&
		bytes.Equal(a.payload, b.payload)
}

func TestFrameRoundTrip(t *testing.T) {
	in := []frame{
		{kind: rpcAppendEntries, groupID: 0, reqID: 1, payload: nil},
		{kind: rpcRequestVote | kindResponse, groupID: 7, reqID: 2, payload: []byte("hi")},
		{kind: rpcInstallSnapshot, groupID: 0xFFFFFFFF, reqID: 1 << 40, payload: bytes.Repeat([]byte("x"), 5000)},
	}
	stream := encodeFrames(t, in...)
	fr := frameReader{r: bufio.NewReader(bytes.NewReader(stream))}
	for i, want := range in {
		got, err := fr.read()
		if err != nil {
			t.Fatalf("frame %d: read: %v", i, err)
		}
		if !framesEqual(got, want) {
			t.Fatalf("frame %d mismatch: got %+v want %+v", i, got, want)
		}
	}
	if _, err := fr.read(); !errors.Is(err, io.EOF) {
		t.Fatalf("expected EOF at stream end, got %v", err)
	}
}

// TestFramePartialReadReassembly feeds the stream one byte per Read so every
// frame spans many short reads; io.ReadFull must reassemble each correctly.
func TestFramePartialReadReassembly(t *testing.T) {
	in := []frame{
		{kind: rpcAppendEntries, groupID: 3, reqID: 11, payload: []byte("abcdefghij")},
		{kind: rpcTimeoutNow | kindResponse, groupID: 4, reqID: 12, payload: bytes.Repeat([]byte("z"), 1024)},
	}
	stream := encodeFrames(t, in...)
	fr := frameReader{r: bufio.NewReader(iotest.OneByteReader(bytes.NewReader(stream)))}
	for i, want := range in {
		got, err := fr.read()
		if err != nil {
			t.Fatalf("frame %d: read: %v", i, err)
		}
		if !framesEqual(got, want) {
			t.Fatalf("frame %d mismatch: got %+v want %+v", i, got, want)
		}
	}
}

func TestFrameBadMagic(t *testing.T) {
	stream := encodeFrames(t, frame{kind: rpcAppendEntries, groupID: 1, reqID: 1, payload: []byte("x")})
	stream[0] = 0x00 // corrupt magic
	fr := frameReader{r: bufio.NewReader(bytes.NewReader(stream))}
	if _, err := fr.read(); !errors.Is(err, errBadMagic) {
		t.Fatalf("expected errBadMagic, got %v", err)
	}
}

func TestFrameBadVersion(t *testing.T) {
	stream := encodeFrames(t, frame{kind: rpcAppendEntries, groupID: 1, reqID: 1, payload: []byte("x")})
	stream[1] = 0xFF // corrupt version
	fr := frameReader{r: bufio.NewReader(bytes.NewReader(stream))}
	if _, err := fr.read(); !errors.Is(err, errBadVersion) {
		t.Fatalf("expected errBadVersion, got %v", err)
	}
}

// TestFrameOversize crafts a header whose payloadLen exceeds maxPayload; read
// must reject it before attempting to allocate/read the body.
func TestFrameOversize(t *testing.T) {
	var hdr [frameHeaderSize]byte
	hdr[0] = frameMagic
	hdr[1] = frameVersion
	hdr[2] = rpcAppendEntries
	binary.LittleEndian.PutUint32(hdr[3:], 1)
	binary.LittleEndian.PutUint64(hdr[7:], 1)
	binary.LittleEndian.PutUint32(hdr[15:], maxPayload+1)
	fr := frameReader{r: bufio.NewReader(bytes.NewReader(hdr[:]))}
	if _, err := fr.read(); !errors.Is(err, errOversize) {
		t.Fatalf("expected errOversize, got %v", err)
	}
}

func FuzzFrameRoundTrip(f *testing.F) {
	f.Add(uint8(0x80), uint32(7), uint64(42), []byte("payload"))
	f.Fuzz(func(t *testing.T, kind uint8, groupID uint32, reqID uint64, payload []byte) {
		if uint32(len(payload)) > maxPayload {
			t.Skip()
		}
		want := frame{kind: kind, groupID: groupID, reqID: reqID, payload: payload}
		stream := encodeFrames(t, want)
		fr := frameReader{r: bufio.NewReader(bytes.NewReader(stream))}
		got, err := fr.read()
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if got.kind != want.kind || got.groupID != want.groupID || got.reqID != want.reqID || !bytes.Equal(got.payload, norm(want.payload)) {
			t.Fatalf("mismatch: got %+v want %+v", got, want)
		}
	})
}

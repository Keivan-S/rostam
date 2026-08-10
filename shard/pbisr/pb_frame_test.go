// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"testing"
)

// writeTestFrame writes a header (via writePBFrameHdr) followed by payload to
// buf, mimicking what the batched writer will do on the wire.
func writeTestFrame(buf *bytes.Buffer, f pbFrame) {
	var hdr [pbFrameHeaderSize]byte
	writePBFrameHdr(hdr[:], &f)
	buf.Write(hdr[:])
	buf.Write(f.payload)
}

func TestPBFrameHeaderRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	payload := []byte("hello-pb-frame-payload")
	want := pbFrame{
		kind:    pbKindReplicate,
		shard:   3,
		reqID:   42,
		payload: payload,
	}
	writeTestFrame(&buf, want)

	fr := &pbFrameReader{r: bufio.NewReader(&buf)}
	got, err := fr.read()
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if got.kind != want.kind || got.shard != want.shard || got.reqID != want.reqID {
		t.Fatalf("header mismatch: got %+v want %+v", got, want)
	}
	if !bytes.Equal(got.payload, payload) {
		t.Fatalf("payload mismatch: got %q want %q", got.payload, payload)
	}
}

func TestPBReplicateMsgCodec(t *testing.T) {
	m := ReplicateMsg{Epoch: 7, Seq: 42, PrevSeq: 41, Data: []byte("op")}
	enc := encodeReplicateMsg(nil, m)
	got, err := decodeReplicateMsg(enc)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.Epoch != m.Epoch || got.Seq != m.Seq || got.PrevSeq != m.PrevSeq || string(got.Data) != string(m.Data) {
		t.Fatalf("round-trip mismatch: got %+v want %+v", got, m)
	}
}

func TestPBAckMsgCodec(t *testing.T) {
	for _, ok := range []bool{true, false} {
		a := AckMsg{Epoch: 9, Seq: 100, OK: ok}
		enc := encodeAckMsg(nil, a)
		got, err := decodeAckMsg(enc)
		if err != nil {
			t.Fatalf("decode ack: %v", err)
		}
		if got != a {
			t.Fatalf("ack round-trip: got %+v want %+v", got, a)
		}
	}
}

func TestPBFrameReaderRejectsOversize(t *testing.T) {
	var buf bytes.Buffer
	var hdr [pbFrameHeaderSize]byte
	hdr[0] = pbFrameMagic
	hdr[1] = pbFrameVersion
	hdr[2] = pbKindReplicate
	binary.LittleEndian.PutUint32(hdr[3:], 1)
	binary.LittleEndian.PutUint64(hdr[7:], 1)
	binary.LittleEndian.PutUint32(hdr[15:], pbMaxPayload+1) // claims a payload larger than allowed
	buf.Write(hdr[:])

	fr := &pbFrameReader{r: bufio.NewReader(&buf)}
	if _, err := fr.read(); err == nil {
		t.Fatal("read must reject a header claiming len > pbMaxPayload")
	}
}

func TestPBFrameReaderRejectsBadMagic(t *testing.T) {
	var buf bytes.Buffer
	var hdr [pbFrameHeaderSize]byte
	hdr[0] = 0xFF // wrong magic
	hdr[1] = pbFrameVersion
	buf.Write(hdr[:])

	fr := &pbFrameReader{r: bufio.NewReader(&buf)}
	if _, err := fr.read(); err == nil {
		t.Fatal("read must reject a bad magic byte")
	}
}

func TestPBFrameReaderRejectsBadVersion(t *testing.T) {
	var buf bytes.Buffer
	var hdr [pbFrameHeaderSize]byte
	writePBFrameHdr(hdr[:], &pbFrame{kind: pbKindReplicate, shard: 1, reqID: 1})
	hdr[1] = pbFrameVersion + 1 // corrupt the version byte
	buf.Write(hdr[:])

	fr := &pbFrameReader{r: bufio.NewReader(&buf)}
	if _, err := fr.read(); err == nil {
		t.Fatal("read must reject a bad version byte")
	}
}

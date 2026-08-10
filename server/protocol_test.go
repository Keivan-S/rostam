// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bytes"
	"errors"
	"testing"
)

func TestRequestRoundtrip(t *testing.T) {
	op := "update_session"
	args := []byte(`{"user":42}`)
	frame := EncodeRequest(op, args)

	gotOp, gotArgs, err := DecodeRequest(frame)
	if err != nil {
		t.Fatalf("DecodeRequest: %v", err)
	}
	if gotOp != op {
		t.Fatalf("op = %q, want %q", gotOp, op)
	}
	if !bytes.Equal(gotArgs, args) {
		t.Fatalf("args mismatch")
	}
}

func TestRequestEmptyArgs(t *testing.T) {
	frame := EncodeRequest("__ping__", nil)
	op, args, err := DecodeRequest(frame)
	if err != nil {
		t.Fatal(err)
	}
	if op != "__ping__" || len(args) != 0 {
		t.Fatalf("op=%q args=%v", op, args)
	}
}

func TestResponseRoundtripOK(t *testing.T) {
	payload := []byte("hello")
	frame := EncodeResponse(StatusOK, payload)
	status, gotPayload, err := DecodeResponse(frame)
	if err != nil {
		t.Fatalf("DecodeResponse: %v", err)
	}
	if status != StatusOK {
		t.Fatalf("status = %d, want OK", status)
	}
	if !bytes.Equal(gotPayload, payload) {
		t.Fatalf("payload mismatch")
	}
}

func TestResponseRoundtripNotLeader(t *testing.T) {
	payload := EncodeLeaderAddrPayload("10.0.0.5:7001")
	frame := EncodeResponse(StatusNotLeader, payload)
	status, gotPayload, err := DecodeResponse(frame)
	if err != nil {
		t.Fatalf("DecodeResponse: %v", err)
	}
	if status != StatusNotLeader {
		t.Fatalf("status = %d, want NotLeader", status)
	}
	addr, derr := DecodeLeaderAddrPayload(gotPayload)
	if derr != nil {
		t.Fatalf("DecodeLeaderAddrPayload: %v", derr)
	}
	if addr != "10.0.0.5:7001" {
		t.Fatalf("addr = %q", addr)
	}
}

func TestRequestDecodeTruncated(t *testing.T) {
	frame := EncodeRequest("op", []byte("hi"))
	for n := 0; n < len(frame); n++ {
		if _, _, err := DecodeRequest(frame[:n]); err == nil {
			t.Fatalf("DecodeRequest at len=%d: nil err, want truncated", n)
		}
	}
}

func TestRequestOversizeRejected(t *testing.T) {
	// argsLen claims 1 GiB but buffer is small — must reject.
	frame := []byte{2, 'o', 'p', 0x40, 0, 0, 0} // opNameLen=2 op argsLen=2^30
	_, _, err := DecodeRequest(frame)
	if err == nil {
		t.Fatal("oversize argsLen accepted; want error")
	}
}

func TestEncodeRequestOpNameTooLongPanics(t *testing.T) {
	long := make([]byte, 256)
	for i := range long {
		long[i] = 'a'
	}
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for 256-byte opName")
		}
	}()
	EncodeRequest(string(long), nil)
}

func TestErrorTruncatedConstant(t *testing.T) {
	if !errors.Is(ErrFrameTruncated, ErrFrameTruncated) {
		t.Fatal("ErrFrameTruncated identity broken")
	}
}

// SPDX-License-Identifier: Apache-2.0

package mux

import (
	"bytes"
	"io"
	"testing"
)

func TestWriteGroupIDPrefix(t *testing.T) {
	var buf bytes.Buffer
	if err := writeGroupID(&buf, 0xDEADBEEF); err != nil {
		t.Fatalf("writeGroupID: %v", err)
	}
	want := []byte{0xDE, 0xAD, 0xBE, 0xEF}
	if !bytes.Equal(buf.Bytes(), want) {
		t.Errorf("got %x, want %x", buf.Bytes(), want)
	}
}

func TestReadGroupIDPrefix(t *testing.T) {
	r := bytes.NewReader([]byte{0xDE, 0xAD, 0xBE, 0xEF, 0xFF})
	id, err := readGroupID(r)
	if err != nil {
		t.Fatalf("readGroupID: %v", err)
	}
	if id != 0xDEADBEEF {
		t.Errorf("got %x, want 0xDEADBEEF", id)
	}
	rest, _ := io.ReadAll(r)
	if !bytes.Equal(rest, []byte{0xFF}) {
		t.Errorf("residual = %x, want ff", rest)
	}
}

func TestReadGroupIDShortRead(t *testing.T) {
	r := bytes.NewReader([]byte{0xDE, 0xAD})
	if _, err := readGroupID(r); err == nil {
		t.Fatal("expected short-read error")
	}
}

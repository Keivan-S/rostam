// SPDX-License-Identifier: Apache-2.0

package logstore

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	hraft "github.com/hashicorp/raft"
)

// store is the union interface both backends satisfy.
type store interface {
	hraft.LogStore
	hraft.StableStore
	Close() error
}

func mkLog(idx uint64, data string) *hraft.Log {
	return &hraft.Log{Index: idx, Term: 1, Type: hraft.LogCommand, Data: []byte(data)}
}

// runContract exercises the full LogStore + StableStore behaviour. Run against
// both Mem and WAL so they cannot diverge.
func runContract(t *testing.T, open func() store) {
	t.Helper()
	s := open()
	defer s.Close()

	// empty
	if fi, _ := s.FirstIndex(); fi != 0 {
		t.Fatalf("empty FirstIndex=%d want 0", fi)
	}
	if li, _ := s.LastIndex(); li != 0 {
		t.Fatalf("empty LastIndex=%d want 0", li)
	}
	var out hraft.Log
	if err := s.GetLog(1, &out); err != hraft.ErrLogNotFound {
		t.Fatalf("GetLog(missing)=%v want ErrLogNotFound", err)
	}

	// store 1..100
	for i := uint64(1); i <= 100; i++ {
		if err := s.StoreLog(mkLog(i, fmt.Sprintf("v%d", i))); err != nil {
			t.Fatalf("StoreLog %d: %v", i, err)
		}
	}
	if fi, _ := s.FirstIndex(); fi != 1 {
		t.Fatalf("FirstIndex=%d want 1", fi)
	}
	if li, _ := s.LastIndex(); li != 100 {
		t.Fatalf("LastIndex=%d want 100", li)
	}
	for i := uint64(1); i <= 100; i++ {
		if err := s.GetLog(i, &out); err != nil {
			t.Fatalf("GetLog %d: %v", i, err)
		}
		if out.Index != i || string(out.Data) != fmt.Sprintf("v%d", i) {
			t.Fatalf("GetLog %d => idx=%d data=%q", i, out.Index, out.Data)
		}
	}

	// batch StoreLogs 101..110
	batch := make([]*hraft.Log, 0, 10)
	for i := uint64(101); i <= 110; i++ {
		batch = append(batch, mkLog(i, fmt.Sprintf("v%d", i)))
	}
	if err := s.StoreLogs(batch); err != nil {
		t.Fatalf("StoreLogs: %v", err)
	}
	if li, _ := s.LastIndex(); li != 110 {
		t.Fatalf("after batch LastIndex=%d want 110", li)
	}

	// tail truncation (conflict): drop 106..110, re-append a different 106
	if err := s.DeleteRange(106, 110); err != nil {
		t.Fatalf("DeleteRange tail: %v", err)
	}
	if li, _ := s.LastIndex(); li != 105 {
		t.Fatalf("after tail-trunc LastIndex=%d want 105", li)
	}
	if err := s.StoreLog(mkLog(106, "rewritten")); err != nil {
		t.Fatalf("re-append 106: %v", err)
	}
	if err := s.GetLog(106, &out); err != nil || string(out.Data) != "rewritten" {
		t.Fatalf("GetLog 106 after rewrite: err=%v data=%q", err, out.Data)
	}

	// front truncation (compaction): drop 1..50
	if err := s.DeleteRange(1, 50); err != nil {
		t.Fatalf("DeleteRange front: %v", err)
	}
	if fi, _ := s.FirstIndex(); fi != 51 {
		t.Fatalf("after front-trunc FirstIndex=%d want 51", fi)
	}
	if err := s.GetLog(50, &out); err != hraft.ErrLogNotFound {
		t.Fatalf("GetLog(50) after front-trunc=%v want ErrLogNotFound", err)
	}
	if err := s.GetLog(51, &out); err != nil || out.Index != 51 {
		t.Fatalf("GetLog(51) after front-trunc: err=%v idx=%d", err, out.Index)
	}
	if err := s.GetLog(106, &out); err != nil || string(out.Data) != "rewritten" {
		t.Fatalf("GetLog(106) after front-trunc: err=%v data=%q", err, out.Data)
	}

	// StableStore
	if err := s.Set([]byte("k"), []byte("val")); err != nil {
		t.Fatal(err)
	}
	if v, _ := s.Get([]byte("k")); string(v) != "val" {
		t.Fatalf("Get k=%q want val", v)
	}
	if v, _ := s.Get([]byte("missing")); v != nil {
		t.Fatalf("Get missing=%q want nil", v)
	}
	if err := s.SetUint64([]byte("t"), 42); err != nil {
		t.Fatal(err)
	}
	if u, _ := s.GetUint64([]byte("t")); u != 42 {
		t.Fatalf("GetUint64 t=%d want 42", u)
	}
}

func TestContract_Mem(t *testing.T) { runContract(t, func() store { return NewMem() }) }

func TestContract_WAL(t *testing.T) {
	dir := t.TempDir()
	runContract(t, func() store {
		w, err := OpenWAL(dir, true)
		if err != nil {
			t.Fatal(err)
		}
		return w
	})
}

// TestWAL_RecoverReopen writes entries, closes, reopens, and verifies the log
// and stable state survive — the durable point of the WAL.
func TestWAL_RecoverReopen(t *testing.T) {
	dir := t.TempDir()
	w, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	for i := uint64(1); i <= 500; i++ {
		if err := w.StoreLog(mkLog(i, fmt.Sprintf("payload-%d", i))); err != nil {
			t.Fatal(err)
		}
	}
	_ = w.SetUint64([]byte("term"), 7)
	_ = w.DeleteRange(1, 100) // compact prefix before restart
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	w2, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer w2.Close()
	if fi, _ := w2.FirstIndex(); fi != 101 {
		t.Fatalf("recovered FirstIndex=%d want 101", fi)
	}
	if li, _ := w2.LastIndex(); li != 500 {
		t.Fatalf("recovered LastIndex=%d want 500", li)
	}
	var out hraft.Log
	for _, i := range []uint64{101, 250, 500} {
		if err := w2.GetLog(i, &out); err != nil || string(out.Data) != fmt.Sprintf("payload-%d", i) {
			t.Fatalf("recovered GetLog %d: err=%v data=%q", i, err, out.Data)
		}
	}
	if term, _ := w2.GetUint64([]byte("term")); term != 7 {
		t.Fatalf("recovered term=%d want 7", term)
	}
}

// TestWAL_SegmentRotation forces many small segments and checks reads span them.
func TestWAL_SegmentRotation(t *testing.T) {
	dir := t.TempDir()
	w, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	w.maxSeg = 512 // tiny => frequent rotation
	for i := uint64(1); i <= 200; i++ {
		if err := w.StoreLog(mkLog(i, fmt.Sprintf("entry-number-%d-with-padding", i))); err != nil {
			t.Fatal(err)
		}
	}
	segs, _ := filepath.Glob(filepath.Join(dir, "*"+segExt))
	if len(segs) < 3 {
		t.Fatalf("expected multiple segments, got %d", len(segs))
	}
	var out hraft.Log
	for i := uint64(1); i <= 200; i++ {
		if err := w.GetLog(i, &out); err != nil || out.Index != i {
			t.Fatalf("GetLog %d across segments: err=%v idx=%d", i, err, out.Index)
		}
	}
	_ = w.Close()
	// reopen must recover all segments
	w2, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	defer w2.Close()
	if li, _ := w2.LastIndex(); li != 200 {
		t.Fatalf("recovered across segments LastIndex=%d want 200", li)
	}
}

// TestWAL_TornTail corrupts the last bytes of the last segment and checks
// recovery truncates the torn record instead of failing.
func TestWAL_TornTail(t *testing.T) {
	dir := t.TempDir()
	w, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	for i := uint64(1); i <= 20; i++ {
		_ = w.StoreLog(mkLog(i, fmt.Sprintf("v%d", i)))
	}
	_ = w.Close()

	// Append garbage (a half-written record) to the segment file.
	segs, _ := filepath.Glob(filepath.Join(dir, "*"+segExt))
	if len(segs) == 0 {
		t.Fatal("no segment")
	}
	f, _ := os.OpenFile(segs[len(segs)-1], os.O_WRONLY|os.O_APPEND, 0o640)
	_, _ = f.Write([]byte{0xff, 0xff, 0xff, 0x7f, 0x01, 0x02}) // bogus recLen + short body
	_ = f.Close()

	w2, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatalf("recover with torn tail: %v", err)
	}
	defer w2.Close()
	if li, _ := w2.LastIndex(); li != 20 {
		t.Fatalf("torn-tail recovered LastIndex=%d want 20 (garbage should be dropped)", li)
	}
	var out hraft.Log
	if err := w2.GetLog(20, &out); err != nil || !bytes.Equal(out.Data, []byte("v20")) {
		t.Fatalf("GetLog 20 after torn-tail recovery: err=%v data=%q", err, out.Data)
	}
}

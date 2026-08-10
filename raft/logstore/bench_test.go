// SPDX-License-Identifier: Apache-2.0

package logstore

import (
	"testing"

	hraft "github.com/hashicorp/raft"
)

func BenchmarkWAL_StoreLog(b *testing.B) {
	w, err := OpenWAL(b.TempDir(), false)
	if err != nil {
		b.Fatal(err)
	}
	defer w.Close()
	l := &hraft.Log{Term: 1, Type: hraft.LogCommand, Data: make([]byte, 256)}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l.Index = uint64(i + 1)
		if err := w.StoreLog(l); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkWAL_StoreLogsBatch(b *testing.B) {
	w, err := OpenWAL(b.TempDir(), false)
	if err != nil {
		b.Fatal(err)
	}
	defer w.Close()
	batch := make([]*hraft.Log, 8)
	for i := range batch {
		batch[i] = &hraft.Log{Term: 1, Type: hraft.LogCommand, Data: make([]byte, 256)}
	}
	b.ReportAllocs()
	b.ResetTimer()
	idx := uint64(1)
	for i := 0; i < b.N; i++ {
		for j := range batch {
			batch[j].Index = idx
			idx++
		}
		if err := w.StoreLogs(batch); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkWAL_SteadyState mimics real raft: the log is compacted periodically
// (as snapshots front-truncate it), so the offsets index reaches a bounded size
// and stops growing. This is the deployment-representative hot path — it should
// settle to 0 B/op (the growth in BenchmarkWAL_StoreLog is one-time index growth
// of an unbounded log, not per-op garbage).
func BenchmarkWAL_SteadyState(b *testing.B) {
	w, err := OpenWAL(b.TempDir(), false)
	if err != nil {
		b.Fatal(err)
	}
	defer w.Close()
	w.maxSeg = 1 << 30 // avoid rotation noise in this microbench
	l := &hraft.Log{Term: 1, Type: hraft.LogCommand, Data: make([]byte, 256)}
	const keep = 4096
	// Warm the index to its steady size before timing.
	for i := uint64(1); i <= keep; i++ {
		l.Index = i
		_ = w.StoreLog(l)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l.Index = uint64(keep) + uint64(i) + 1
		if err := w.StoreLog(l); err != nil {
			b.Fatal(err)
		}
		if i%keep == keep-1 { // compact the prefix, keeping ~keep entries
			first, _ := w.FirstIndex()
			_ = w.DeleteRange(first, l.Index-keep/2)
		}
	}
}

func BenchmarkWAL_GetLog(b *testing.B) {
	w, err := OpenWAL(b.TempDir(), false)
	if err != nil {
		b.Fatal(err)
	}
	defer w.Close()
	l := &hraft.Log{Term: 1, Type: hraft.LogCommand, Data: make([]byte, 256)}
	for i := uint64(1); i <= 10000; i++ {
		l.Index = i
		_ = w.StoreLog(l)
	}
	var out hraft.Log
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := w.GetLog(uint64(i%10000)+1, &out); err != nil {
			b.Fatal(err)
		}
	}
}

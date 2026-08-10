// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"io"
	"path/filepath"
	"strconv"
	"sync"
	"testing"
)

// Wave-4 benchmarks for persistence & lifecycle: snapshot serialize/restore, WAL
// append (group-commit, sync vs nosync), and Reclaim (tombstone compaction). These
// paths gate restart latency and durability throughput and had no prior coverage.
// See BENCHMARKS.md.

const (
	psN   = 20_000
	psDim = 128
)

var (
	psOnce sync.Once
	psIdx  *hnsw
	psSnap []byte // a pre-serialized snapshot of psIdx, for the Restore benchmark
)

func persistBenchIndex(tb testing.TB) (*hnsw, []byte) {
	tb.Helper()
	psOnce.Do(func() {
		h, err := newHNSW(Config{Dim: psDim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
		if err != nil {
			panic(err)
		}
		corpus := makeCorpus(psN, psDim, 42)
		ids := make([]uint64, psN)
		for i := range ids {
			ids[i] = uint64(i + 1)
		}
		if err := h.BuildConcurrent(ids, corpus, 0); err != nil {
			panic(err)
		}
		psIdx = h
		var buf bytes.Buffer
		if err := h.Snapshot(&buf); err != nil {
			panic(err)
		}
		psSnap = buf.Bytes()
	})
	return psIdx, psSnap
}

// BenchmarkSnapshot measures full-index serialization (vectors + graph + metadata)
// to an io.Writer, isolated from disk by writing to io.Discard.
func BenchmarkSnapshot(b *testing.B) {
	h, _ := persistBenchIndex(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := h.Snapshot(io.Discard); err != nil {
			b.Fatal(err)
		}
	}
	b.SetBytes(int64(len(psSnap)))
}

// BenchmarkRestore measures full-index deserialization from an in-memory snapshot
// (no disk) into a fresh index.
func BenchmarkRestore(b *testing.B) {
	_, snap := persistBenchIndex(b)
	b.ResetTimer()
	b.SetBytes(int64(len(snap)))
	for i := 0; i < b.N; i++ {
		h, err := newHNSW(Config{Dim: psDim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
		if err != nil {
			b.Fatal(err)
		}
		if err := h.Restore(bytes.NewReader(snap)); err != nil {
			b.Fatal(err)
		}
		_ = h.Close()
	}
}

// BenchmarkWALAppendInsert measures the write-ahead-log insert-record path
// (encode + frame + CRC + group-committed write). The sync variant fsyncs (the
// real durability cost, disk-dependent); nosync isolates the encode/write CPU.
func BenchmarkWALAppendInsert(b *testing.B) {
	corpus := makeCorpus(4096, psDim, 7)
	for _, tc := range []struct {
		name   string
		noSync bool
	}{{"sync", false}, {"nosync", true}} {
		b.Run(tc.name, func(b *testing.B) {
			w, err := openWAL(filepath.Join(b.TempDir(), "wal.log"), tc.noSync)
			if err != nil {
				b.Fatal(err)
			}
			defer func() { _ = w.close() }()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Staged write + commit-wait: the production shape (the append and the
				// durability wait are separate phases, opMu released in between).
				seq, err := w.appendInsertStaged(uint64(i+1), corpus[i%len(corpus)], 0, nil, nil, nil, 0)
				if err != nil {
					b.Fatal(err)
				}
				if err := w.commitWaitStaged(seq); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkReclaim measures tombstone compaction (graph edge filtering + slot
// recycling). Each iteration rebuilds an index with a fresh tombstone load; only
// the Reclaim call is timed (setup is excluded via StopTimer), swept over the
// fraction of the corpus deleted.
func BenchmarkReclaim(b *testing.B) {
	const n = 10_000
	corpus := makeCorpus(n, psDim, 42)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	for _, frac := range []int{10, 50} {
		b.Run("del="+strconv.Itoa(frac)+"pct", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				h, err := newHNSW(Config{Dim: psDim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
				if err != nil {
					b.Fatal(err)
				}
				if err := h.BuildConcurrent(ids, corpus, 0); err != nil {
					b.Fatal(err)
				}
				for id := 1; id <= n; id++ {
					if id%(100/frac) == 0 { // delete ~frac% (frac in {10,50} → every 10th / 2nd)
						_, _ = h.Delete(uint64(id), CASCond{})
					}
				}
				b.StartTimer()
				_ = h.Reclaim()
				b.StopTimer()
				_ = h.Close()
			}
		})
	}
}

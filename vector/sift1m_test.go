// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestSIFT1M validates HNSW recall against the canonical SIFT-1M dataset
// (1 million 128-dim SIFT descriptors with 10k queries and ground-truth
// nearest neighbors). Opt-in: set ROSTAM_SIFT1M=1.
//
// Dataset is NOT downloaded by the test (too brittle for CI). The test
// expects pre-extracted files at /tmp/rostam-sift1m/sift/:
//
//	sift_base.fvecs       (1M × 128 corpus)
//	sift_query.fvecs      (10k × 128 queries)
//	sift_groundtruth.ivecs (10k × 100 truth indices)
//
// To populate:
//
//	mkdir -p /tmp/rostam-sift1m
//	cd /tmp/rostam-sift1m
//	wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
//	tar -xzf sift.tar.gz
//	ROSTAM_SIFT1M=1 go test ./vector/ -run TestSIFT1M -v -timeout 30m
//
// Success criteria (from spec): recall@10 >= 0.95 at M=16, EfSearch=64.
func TestSIFT1M(t *testing.T) {
	if testing.Short() || os.Getenv("ROSTAM_SIFT1M") != "1" {
		t.Skip("set ROSTAM_SIFT1M=1 with dataset at /tmp/rostam-sift1m/sift/ to run")
	}

	dir := filepath.Join(os.TempDir(), "rostam-sift1m", "sift")
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		t.Skipf("dataset not found at %s — see TestSIFT1M docstring for fetch instructions", dir)
	}

	corpus, err := readFvecs(filepath.Join(dir, "sift_base.fvecs"))
	if err != nil {
		t.Fatal(err)
	}
	queries, err := readFvecs(filepath.Join(dir, "sift_query.fvecs"))
	if err != nil {
		t.Fatal(err)
	}
	groundTruth, err := readIvecs(filepath.Join(dir, "sift_groundtruth.ivecs"))
	if err != nil {
		t.Fatal(err)
	}

	const k = 10
	t.Logf("corpus=%d queries=%d gt=%d dim=%d",
		len(corpus), len(queries), len(groundTruth), len(corpus[0]))

	h, _ := newHNSW(Config{
		Dim: len(corpus[0]), M: 16, EfConstruction: 200, EfSearch: 64, Seed: 42, Metric: L2,
	})
	for i, v := range corpus {
		if _, _, err := h.Insert(uint64(i+1), v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
		if i > 0 && i%100_000 == 0 {
			t.Logf("inserted %d / %d", i, len(corpus))
		}
	}

	var matches int
	for qi, q := range queries {
		truth := make(map[uint64]bool, k)
		for _, id := range groundTruth[qi][:k] {
			truth[uint64(id)+1] = true
		}
		results, err := h.Search(q, k)
		if err != nil {
			t.Fatalf("search %d: %v", qi, err)
		}
		for _, r := range results {
			if truth[r.ID] {
				matches++
			}
		}
	}
	recall := float64(matches) / float64(len(queries)*k)
	t.Logf("SIFT-1M recall@%d = %.4f", k, recall)
	if recall < 0.95 {
		t.Errorf("recall@%d = %.4f, want >= 0.95", k, recall)
	}
}

// readFvecs parses TEXMEX .fvecs format: little-endian
// [dim:u32][vec: float32 × dim] records, concatenated.
func readFvecs(path string) ([][]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out [][]float32
	for off := 0; off < len(data); {
		if off+4 > len(data) {
			return nil, fmt.Errorf("fvecs: truncated header at %d", off)
		}
		dim := binary.LittleEndian.Uint32(data[off:])
		off += 4
		if off+int(dim)*4 > len(data) {
			return nil, fmt.Errorf("fvecs: truncated vector at %d", off)
		}
		v := make([]float32, dim)
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[off+i*4:]))
		}
		out = append(out, v)
		off += int(dim) * 4
	}
	return out, nil
}

// readIvecs parses TEXMEX .ivecs format: same as fvecs but int32 values.
func readIvecs(path string) ([][]int32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out [][]int32
	for off := 0; off < len(data); {
		if off+4 > len(data) {
			return nil, fmt.Errorf("ivecs: truncated header at %d", off)
		}
		dim := binary.LittleEndian.Uint32(data[off:])
		off += 4
		if off+int(dim)*4 > len(data) {
			return nil, fmt.Errorf("ivecs: truncated vector at %d", off)
		}
		v := make([]int32, dim)
		for i := range v {
			v[i] = int32(binary.LittleEndian.Uint32(data[off+i*4:]))
		}
		out = append(out, v)
		off += int(dim) * 4
	}
	return out, nil
}

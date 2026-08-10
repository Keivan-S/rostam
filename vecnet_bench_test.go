// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/server"
	"github.com/rostamlabs/rostam/vector"
)

// TestVectorNetLatencyQPS is the genuine apples-to-apples networked head-to-head
// for the bench/sift1m comparison: Rostam serving vector_search over its OWN TCP
// server (real wire protocol, loopback), hit by concurrent clients — the same
// shape as the Qdrant-over-gRPC harness (bench/sift1m/qdrant_latency_qps.py). It
// reports p50/p99 latency (single connection) and saturated QPS (32 concurrent
// connections), at ef=64, so the network round-trip is paid on BOTH sides and
// the in-process advantage is isolated from the algorithm.
//
// A read benchmark, so it deliberately bypasses Raft (search is OpReadOnly and
// Raft-bypassing in the real Store too; Qdrant reads likewise skip consensus).
// Opt-in: ROSTAM_SIFT1M=1 with the dataset at /tmp/rostam-sift1m/sift/.
//
//	TMPDIR=/tmp ROSTAM_SIFT1M=1 go test . -run TestVectorNetLatencyQPS -v -timeout 30m
func TestVectorNetLatencyQPS(t *testing.T) {
	if os.Getenv("ROSTAM_SIFT1M") != "1" {
		t.Skip("set ROSTAM_SIFT1M=1 with dataset at /tmp/rostam-sift1m/sift/ to run")
	}
	dir := filepath.Join(os.TempDir(), "rostam-sift1m", "sift")
	if d := os.Getenv("ROSTAM_SIFT_DIR"); d != "" {
		dir = d
	}
	base, err := readFvecsBench(filepath.Join(dir, "sift_base.fvecs"))
	if err != nil {
		t.Fatal(err)
	}
	queries, err := readFvecsBench(filepath.Join(dir, "sift_query.fvecs"))
	if err != nil {
		t.Fatal(err)
	}
	gt, err := readIvecsBench(filepath.Join(dir, "sift_groundtruth.ivecs"))
	if err != nil {
		t.Fatal(err)
	}

	const (
		k    = 10
		ef   = 64
		conc = 32 // match qdrant_latency_qps.py
		latN = 2000
	)

	// Build the collection directly (bulk), then serve it over TCP.
	cs, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer cs.Close()
	if err := cs.CreateCollection("sift", vector.Config{
		Dim: len(base[0]), Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: ef,
	}); err != nil {
		t.Fatal(err)
	}
	coll, ok := cs.Get("sift")
	if !ok {
		t.Fatal("collection not found after create")
	}
	ids := make([]uint64, len(base))
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	if err := coll.BuildConcurrent(ids, base, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatalf("BuildConcurrent: %v", err)
	}

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	disp := &vecBenchDispatcher{reg: reg, tx: ops.NewTxContextWithVectors(nil, cs)}
	srv, err := server.New(server.Config{Addr: "127.0.0.1:0", Dispatcher: disp})
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()
	defer func() { _ = srv.Close() }()
	addr := srv.Addr().String()

	truth := make([]map[uint64]bool, len(queries))
	for qi := range queries {
		m := make(map[uint64]bool, k)
		for _, id := range gt[qi][:k] {
			m[uint64(id)+1] = true
		}
		truth[qi] = m
	}

	fmt.Fprintf(os.Stderr, "[net] Rostam over TCP (loopback), %d cores, ef=%d, k=%d\n",
		runtime.GOMAXPROCS(0), ef, k)

	// --- single-connection latency ---
	cl, err := dialVecClient(addr)
	if err != nil {
		t.Fatal(err)
	}
	lat := make([]time.Duration, latN)
	var matches int
	for i := 0; i < latN; i++ {
		s := time.Now()
		res, err := cl.search(queries[i], k)
		lat[i] = time.Since(s)
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		for _, r := range res {
			if truth[i][r.ID] {
				matches++
			}
		}
	}
	_ = cl.close()
	recall := float64(matches) / float64(latN*k)
	sort.Slice(lat, func(a, b int) bool { return lat[a] < lat[b] })
	p50 := lat[len(lat)*50/100]
	p99 := lat[len(lat)*99/100]
	var sum time.Duration
	for _, d := range lat {
		sum += d
	}
	mean := sum / time.Duration(len(lat))

	// --- saturated QPS: `conc` connections, ~3s ---
	var count int64
	var wg sync.WaitGroup
	deadline := time.Now().Add(3 * time.Second)
	start := time.Now()
	for w := 0; w < conc; w++ {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			c, err := dialVecClient(addr)
			if err != nil {
				return
			}
			defer func() { _ = c.close() }()
			i := seed
			var local int64
			for time.Now().Before(deadline) {
				for j := 0; j < 16; j++ {
					if _, err := c.search(queries[i%len(queries)], k); err != nil {
						return
					}
					i++
					local++
				}
			}
			atomic.AddInt64(&count, local)
		}(w * 977)
	}
	wg.Wait()
	qps := float64(count) / time.Since(start).Seconds()

	fmt.Fprintf(os.Stderr, "%-6s %-9s %-10s %-10s %-10s %-12s\n",
		"ef", "recall", "p50(us)", "p99(us)", "mean(us)", "satQPS")
	fmt.Fprintf(os.Stderr, "%-6d %-9.4f %-10.1f %-10.1f %-10.1f %-12.0f\n",
		ef, recall, float64(p50.Microseconds()), float64(p99.Microseconds()), float64(mean.Microseconds()), qps)
	t.Logf("Rostam-over-TCP ef=%d recall=%.4f p50=%dus p99=%dus satQPS=%.0f (%d conns)",
		ef, recall, p50.Microseconds(), p99.Microseconds(), qps, conc)
}

// vecBenchDispatcher serves ops straight from the registry against a
// CollectionStore — the read-only path the real Store uses for search, minus
// Raft (correct for a read benchmark).
type vecBenchDispatcher struct {
	reg *ops.Registry
	tx  *ops.TxContext
}

func (d *vecBenchDispatcher) Call(name string, args []byte) ([]byte, error) {
	h, _, _, ok := d.reg.Lookup(name)
	if !ok {
		return nil, fmt.Errorf("op %q not registered", name)
	}
	return h(d.tx, args)
}

func (d *vecBenchDispatcher) LeaderAddr() string { return "" }

// vecClient is a minimal framed client over one persistent TCP connection,
// matching the server's wire protocol (server/protocol.go, server/conn.go).
type vecClient struct {
	conn net.Conn
	bw   *bufio.Writer
	br   *bufio.Reader
	hdr  [4]byte
	buf  []byte
}

func dialVecClient(addr string) (*vecClient, error) {
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		return nil, err
	}
	if tc, ok := conn.(*net.TCPConn); ok {
		_ = tc.SetNoDelay(true)
	}
	return &vecClient{
		conn: conn,
		bw:   bufio.NewWriterSize(conn, 8192),
		br:   bufio.NewReaderSize(conn, 8192),
		buf:  make([]byte, 0, 512),
	}, nil
}

func (c *vecClient) search(query []float32, k int) ([]vector.Result, error) {
	body := server.EncodeRequest("vector_search", ops.EncodeVectorSearchArgs("sift", k, query))
	binary.BigEndian.PutUint32(c.hdr[:], uint32(len(body)))
	if _, err := c.bw.Write(c.hdr[:]); err != nil {
		return nil, err
	}
	if _, err := c.bw.Write(body); err != nil {
		return nil, err
	}
	if err := c.bw.Flush(); err != nil {
		return nil, err
	}
	if _, err := io.ReadFull(c.br, c.hdr[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint32(c.hdr[:])
	if cap(c.buf) < int(n) {
		c.buf = make([]byte, n)
	} else {
		c.buf = c.buf[:n]
	}
	if _, err := io.ReadFull(c.br, c.buf); err != nil {
		return nil, err
	}
	status, payload, err := server.DecodeResponse(c.buf)
	if err != nil {
		return nil, err
	}
	if status != server.StatusOK {
		return nil, fmt.Errorf("server status %d", status)
	}
	return ops.DecodeVectorSearchResults(payload)
}

func (c *vecClient) close() error { return c.conn.Close() }

// readFvecsBench / readIvecsBench parse the TEXMEX .fvecs/.ivecs formats (the
// vector package's readers are test-only and not importable here).
func readFvecsBench(path string) ([][]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out [][]float32
	for off := 0; off < len(data); {
		if off+4 > len(data) {
			return nil, fmt.Errorf("fvecs: truncated header at %d", off)
		}
		dim := int(binary.LittleEndian.Uint32(data[off:]))
		off += 4
		if off+dim*4 > len(data) {
			return nil, fmt.Errorf("fvecs: truncated vector at %d", off)
		}
		v := make([]float32, dim)
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[off+i*4:]))
		}
		out = append(out, v)
		off += dim * 4
	}
	return out, nil
}

func readIvecsBench(path string) ([][]int32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out [][]int32
	for off := 0; off < len(data); {
		if off+4 > len(data) {
			return nil, fmt.Errorf("ivecs: truncated header at %d", off)
		}
		dim := int(binary.LittleEndian.Uint32(data[off:]))
		off += 4
		if off+dim*4 > len(data) {
			return nil, fmt.Errorf("ivecs: truncated vector at %d", off)
		}
		v := make([]int32, dim)
		for i := range v {
			v[i] = int32(binary.LittleEndian.Uint32(data[off+i*4:]))
		}
		out = append(out, v)
		off += dim * 4
	}
	return out, nil
}

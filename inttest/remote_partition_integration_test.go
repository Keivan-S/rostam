// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/httpapi"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/server"
	"github.com/rostamlabs/rostam/vector"
)

// These tests prove the FULL path through REAL transports (HTTP + the Go
// networked TCP client) against a PARTITIONED (P>1) collection. The critical
// proof is fan-out PARITY: pre-fan-out a remote search of a P>1 collection
// returned only ONE partition's partial slice; with the server-side
// fanoutDispatcher installed it must now return the globally-merged top-k,
// byte-for-byte equal (by ID and order) to the embedded baseline.
//
// Both tests drive the decorator over a real transport codec: the HTTP test
// mounts httpapi.Handler(rostam.NewFanoutDispatcher(emb, emb.Node()), ...) and the TCP
// test stands up server.New + rostam.NewClient with the same decorator as the
// server's dispatcher (the wire path client -> TCP -> server -> decorator).
// This mirrors how server.go installs the decorator in the CLUSTER branch for
// every transport.

// TestRemotePartitionedHTTPLifecycle drives the full create -> insert -> search
// -> drop lifecycle over the REAL HTTP transport (httptest) backed by the
// fanout dispatcher, and asserts the search top-k matches the embedded baseline.
func TestRemotePartitionedHTTPLifecycle(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)
	ctx := context.Background()

	const coll = "p"
	const n = 100
	const k = 10

	// Mount the REAL HTTP handler over the decorator (the same wrap server.go
	// installs in the CLUSTER branch). httpServer is a real net/http server.
	h := httpapi.Handler(rostam.NewFanoutDispatcher(emb, emb.Node()), httpapi.Options{})
	ts := httptest.NewServer(h)
	t.Cleanup(ts.Close)

	httpJSON := func(method, path, body string, out any) *http.Response {
		t.Helper()
		var rdr *strings.Reader
		if body != "" {
			rdr = strings.NewReader(body)
		}
		var req *http.Request
		var err error
		if rdr != nil {
			req, err = http.NewRequest(method, ts.URL+path, rdr)
		} else {
			req, err = http.NewRequest(method, ts.URL+path, nil)
		}
		if err != nil {
			t.Fatalf("%s %s: build request: %v", method, path, err)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("%s %s: %v", method, path, err)
		}
		if out != nil {
			if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
				t.Fatalf("%s %s: decode body: %v", method, path, err)
			}
		}
		_ = resp.Body.Close()
		return resp
	}

	// 1. Create a P=4 collection via POST /collections (partitions:4 in JSON).
	createBody := fmt.Sprintf(
		`{"name":%q,"config":{"dim":8,"metric":"cosine","m":16,"ef_construction":200,"ef_search":64,"partitions":4}}`, coll)
	resp := httpJSON("POST", "/v1/collections", createBody, nil)
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("create status = %d, want 201", resp.StatusCode)
	}
	// Confirm the create ran through the embedded backend (physical partitions +
	// catalog with P=4) rather than passing through as a single-partition create.
	if p, _, ok := emb.Catalog().PartitionsGen(coll); !ok || p != 4 {
		t.Fatalf("after HTTP create: PartitionsGen = (%d, ok=%v), want (4, true)", p, ok)
	}

	// 2. Insert ~100 tie-free vectors via the point-insert endpoint. Ids hash to
	// different physical partitions, so a correct global top-10 must merge across
	// partitions.
	for i := 0; i < n; i++ {
		v := tieFreeVec(i)
		parts := make([]string, len(v))
		for j, c := range v {
			parts[j] = fmt.Sprintf("%g", c)
		}
		body := fmt.Sprintf(`{"id":%d,"vector":[%s]}`, i, strings.Join(parts, ","))
		resp := httpJSON("POST", "/v1/collections/"+coll+"/points", body, nil)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("insert %d status = %d, want 200", i, resp.StatusCode)
		}
	}

	// rostam.Embedded baseline for parity (same data, computed directly on the node).
	q := tieFreeQuery()
	want, _, err := emb.VectorSearchExt(ctx, coll, q, k, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatalf("embedded baseline search: %v", err)
	}
	if len(want) != k {
		t.Fatalf("embedded baseline returned %d, want %d", len(want), k)
	}

	// 3. Search via the HTTP search endpoint for k=10.
	qParts := make([]string, len(q))
	for j, c := range q {
		qParts[j] = fmt.Sprintf("%g", c)
	}
	var sres struct {
		Results []vector.Result `json:"results"`
	}
	resp = httpJSON("POST", "/v1/collections/"+coll+"/points/search",
		fmt.Sprintf(`{"query":[%s],"k":%d}`, strings.Join(qParts, ","), k), &sres)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("search status = %d, want 200", resp.StatusCode)
	}

	// Exactly k results: a single partition could not hold a correct global top-10
	// of 100 vectors spread across 4 partitions.
	if len(sres.Results) != k {
		t.Fatalf("HTTP search returned %d results, want %d (single-partition slice would differ)", len(sres.Results), k)
	}
	// Ordered by distance ascending (tie-free → strictly increasing).
	for i := 1; i < len(sres.Results); i++ {
		if sres.Results[i].Distance < sres.Results[i-1].Distance {
			t.Fatalf("HTTP results not distance-ascending at %d: %g < %g",
				i, sres.Results[i].Distance, sres.Results[i-1].Distance)
		}
	}
	// Parity: the HTTP top-k IDs equal the embedded baseline IDs in order. This is
	// the global-merge proof — only a cross-partition merge yields this exact list.
	for i := range want {
		if sres.Results[i].ID != want[i].ID {
			t.Fatalf("HTTP search rank %d: got id %d, want %d (global-merge mismatch)\n got=%v\nwant=%v",
				i, sres.Results[i].ID, want[i].ID, httpResultIDs(sres.Results), resultIDs(want))
		}
	}

	// 4. Drop via DELETE /collections/{name}; assert success + physical cleanup.
	resp = httpJSON("DELETE", "/v1/collections/"+coll, "", nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("drop status = %d, want 200", resp.StatusCode)
	}
	for p := 0; p < 4; p++ {
		phys := string(ops.PartitionKeyGen(coll, 0, p))
		if _, err := emb.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(phys)); err == nil {
			t.Fatalf("after HTTP drop: physical partition %d (%s) still exists", p, phys)
		}
	}
	fan := rostam.NewFanoutDispatcher(emb, emb.Node())
	if _, _, ok := fan.Partitioned(coll); ok {
		t.Fatal("after HTTP drop: collection still reports as partitioned")
	}
}

// TestRemotePartitionedTCPClient drives the full lifecycle over the REAL Go
// networked client (TCP wire) against a server whose dispatcher is the fanout
// decorator. This is the concrete proof the Go networked client can now use
// partitioned collections: a remote search of a P=4 collection returns the
// correct global top-k (not one partition's slice).
func TestRemotePartitionedTCPClient(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}

	embStore, err := rostam.NewEmbedded(rostam.EmbeddedConfig{
		NodeID:    "n1",
		DataDir:   t.TempDir(),
		NumShards: 1,
		Bootstrap: true,
		Ops:       reg,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = embStore.Close() })
	waitLeaderEmbedded(t, embStore)
	emb := embStore.(*rostam.Embedded)

	// The server's dispatcher is the fanout decorator (the fallback decorator-
	// wrapping pair the plan permits) — every op the client sends over TCP is
	// dispatched through the decorator, exactly like server.go's CLUSTER branch.
	disp := rostam.NewFanoutDispatcher(emb, emb.Node())
	// Bind :0 so the OS assigns a free port at bind time (no pre-bind TOCTOU
	// window); read the actual bound address back for the client config.
	srv, err := server.New(server.Config{Addr: "127.0.0.1:0", Dispatcher: disp})
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()
	t.Cleanup(func() { _ = srv.Close() })
	addr := srv.Addr().String()

	store, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{addr}, Ops: reg})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	ctx := context.Background()
	const coll = "p"
	const n = 100
	const k = 10

	// Create a P=4 collection through the networked client. The Go client already
	// sends cfg.Partitions via the ops codec; fanCreateCollection routes the
	// partitioned create through the embedded backend (physical partitions +
	// catalog).
	cfg := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: 4}
	if err := store.CreateCollection(ctx, coll, cfg); err != nil {
		t.Fatalf("client CreateCollection: %v", err)
	}
	if p, _, ok := emb.Catalog().PartitionsGen(coll); !ok || p != 4 {
		t.Fatalf("after client create: PartitionsGen = (%d, ok=%v), want (4, true)", p, ok)
	}

	// Insert ~100 tie-free vectors through the client (routed to the right
	// physical partition by fanInsert on the server side).
	for i := 0; i < n; i++ {
		if err := store.VectorInsert(ctx, coll, uint64(i), tieFreeVec(i)); err != nil {
			t.Fatalf("client VectorInsert %d: %v", i, err)
		}
	}

	// rostam.Embedded baseline (same node, direct fan-out) for parity.
	q := tieFreeQuery()
	want, _, err := emb.VectorSearchExt(ctx, coll, q, k, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatalf("embedded baseline search: %v", err)
	}
	if len(want) != k {
		t.Fatalf("embedded baseline returned %d, want %d", len(want), k)
	}

	// Search k=10 over TCP. The non-negotiable assertion: the remote result is the
	// global top-k merged across all 4 partitions, equal to the baseline in order.
	got, err := store.VectorSearch(ctx, coll, q, k)
	if err != nil {
		t.Fatalf("client VectorSearch: %v", err)
	}
	if len(got) != k {
		t.Fatalf("client search returned %d results, want %d (single-partition slice would differ)", len(got), k)
	}
	for i := 1; i < len(got); i++ {
		if got[i].Distance < got[i-1].Distance {
			t.Fatalf("client results not distance-ascending at %d: %g < %g", i, got[i].Distance, got[i-1].Distance)
		}
	}
	for i := range want {
		if got[i].ID != want[i].ID {
			t.Fatalf("client search rank %d: got id %d, want %d (global-merge mismatch)\n got=%v\nwant=%v",
				i, got[i].ID, want[i].ID, resultIDs(got), resultIDs(want))
		}
	}

	// Drop via the wire (client has no typed vector-drop; the op still routes
	// through the decorator's fanDropCollection). Confirm physical cleanup.
	if _, err := store.Call(ctx, "vector_drop_collection", ops.EncodeDropCollectionArgs(coll)); err != nil {
		t.Fatalf("client drop: %v", err)
	}
	for p := 0; p < 4; p++ {
		phys := string(ops.PartitionKeyGen(coll, 0, p))
		if _, err := emb.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(phys)); err == nil {
			t.Fatalf("after client drop: physical partition %d (%s) still exists", p, phys)
		}
	}
	if _, _, ok := disp.Partitioned(coll); ok {
		t.Fatal("after client drop: collection still reports as partitioned")
	}
}

// TestRemoteResplitTCPClient drives a dense resplit over the REAL Go networked
// client (TCP wire) against a server whose dispatcher is the fanout decorator.
// This proves the resplit virtual-op path works end-to-end over the network: a
// remote store.VectorResplit flips the catalog cluster-side, post-resplit search
// returns the correct global top-k, and store.VectorResplitCleanup returns a count.
func TestRemoteResplitTCPClient(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}

	embStore, err := rostam.NewEmbedded(rostam.EmbeddedConfig{
		NodeID:    "n1",
		DataDir:   t.TempDir(),
		NumShards: 1,
		Bootstrap: true,
		Ops:       reg,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = embStore.Close() })
	waitLeaderEmbedded(t, embStore)
	emb := embStore.(*rostam.Embedded)

	disp := rostam.NewFanoutDispatcher(emb, emb.Node())
	srv, err := server.New(server.Config{Addr: "127.0.0.1:0", Dispatcher: disp})
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()
	t.Cleanup(func() { _ = srv.Close() })

	store, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{srv.Addr().String()}, Ops: reg})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	// Resplit is synchronous + offline; give the client a generous deadline so the
	// scan + re-insert of every vector completes within the call.
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	const coll = "rp"
	const n = 120
	const k = 10

	cfg := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: 4}
	if err := store.CreateCollection(ctx, coll, cfg); err != nil {
		t.Fatalf("client CreateCollection: %v", err)
	}
	for i := 0; i < n; i++ {
		if err := store.VectorInsert(ctx, coll, uint64(i), tieFreeVec(i)); err != nil {
			t.Fatalf("client VectorInsert %d: %v", i, err)
		}
	}

	// Resplit 4 -> 8 over TCP.
	if err := store.VectorResplit(ctx, coll, 8); err != nil {
		t.Fatalf("client VectorResplit: %v", err)
	}
	// The catalog flipped cluster-side to the new generation (P=8, gen=1).
	if p, gen, ok := emb.Catalog().PartitionsGen(coll); !ok || p != 8 || gen != 1 {
		t.Fatalf("after resplit: PartitionsGen = (%d, gen=%d, ok=%v), want (8, 1, true)", p, gen, ok)
	}

	// rostam.Embedded baseline computed on the node over the (now flipped) catalog: the
	// remote top-k must match it in order — proving the post-resplit global merge.
	q := tieFreeQuery()
	want, _, err := emb.VectorSearchExt(ctx, coll, q, k, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatalf("embedded baseline search: %v", err)
	}
	if len(want) != k {
		t.Fatalf("embedded baseline returned %d, want %d", len(want), k)
	}
	got, err := store.VectorSearch(ctx, coll, q, k)
	if err != nil {
		t.Fatalf("client VectorSearch after resplit: %v", err)
	}
	if len(got) != k {
		t.Fatalf("client search after resplit returned %d results, want %d", len(got), k)
	}
	for i := range want {
		if got[i].ID != want[i].ID {
			t.Fatalf("client search after resplit rank %d: got id %d, want %d (post-flip merge mismatch)\n got=%v\nwant=%v",
				i, got[i].ID, want[i].ID, resultIDs(got), resultIDs(want))
		}
	}

	// After a CLEAN resplit there are no orphans: VectorResplit itself drops the old
	// generation as its final step, so cleanup over TCP must report exactly 0.
	dropped, err := store.VectorResplitCleanup(ctx, coll)
	if err != nil {
		t.Fatalf("client VectorResplitCleanup: %v", err)
	}
	if dropped != 0 {
		t.Fatalf("cleanup after clean resplit dropped %d, want 0 (resplit already drops old gen)", dropped)
	}

	// Now prove cleanup does REAL work over the remote path: seed forward-orphan
	// partitions in a NON-live generation (gen 2; live is gen 1) the same way the
	// single-node TestEmbeddedResplitCleanup does — physical create with Partitions=0.
	const orphans = 3
	physCfg := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64}
	for p := 0; p < orphans; p++ {
		phys := string(ops.PartitionKeyGen(coll, 2, p))
		if _, err := emb.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(phys, physCfg)); err != nil {
			t.Fatalf("seed gen-2 orphan %d: %v", p, err)
		}
	}
	// Cleanup over TCP must sweep exactly the seeded orphans and the count must flow
	// faithfully back through the remote codec.
	dropped, err = store.VectorResplitCleanup(ctx, coll)
	if err != nil {
		t.Fatalf("client VectorResplitCleanup (orphans): %v", err)
	}
	if dropped != orphans {
		t.Fatalf("cleanup dropped %d, want %d (exact seeded orphan count)", dropped, orphans)
	}
	// Idempotent: a second cleanup over TCP drops nothing.
	dropped, err = store.VectorResplitCleanup(ctx, coll)
	if err != nil {
		t.Fatalf("client VectorResplitCleanup (idempotent): %v", err)
	}
	if dropped != 0 {
		t.Fatalf("second cleanup dropped %d, want 0 (idempotent)", dropped)
	}
}

// TestRemoteMVResplitTCPClient is the multi-vector counterpart to
// TestRemoteResplitTCPClient: a remote store.VectorMVResplit over TCP flips the
// catalog cluster-side and a post-resplit MV search returns the correct global
// top-k; store.VectorMVResplitCleanup returns a count.
func TestRemoteMVResplitTCPClient(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}

	embStore, err := rostam.NewEmbedded(rostam.EmbeddedConfig{
		NodeID:    "n1",
		DataDir:   t.TempDir(),
		NumShards: 1,
		Bootstrap: true,
		Ops:       reg,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = embStore.Close() })
	waitLeaderEmbedded(t, embStore)
	emb := embStore.(*rostam.Embedded)

	disp := rostam.NewFanoutDispatcher(emb, emb.Node())
	srv, err := server.New(server.Config{Addr: "127.0.0.1:0", Dispatcher: disp})
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()
	t.Cleanup(func() { _ = srv.Close() })

	store, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{srv.Addr().String()}, Ops: reg})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	const coll = "mvrp"
	const n = 60
	const k = 10

	if err := store.VectorMVCreateCollection(ctx, coll, rostam.MultiVectorConfig{Dim: 4, Partitions: 4}); err != nil {
		t.Fatalf("client VectorMVCreateCollection: %v", err)
	}
	for i := 0; i < n; i++ {
		if err := store.VectorMVAdd(ctx, coll, uint64(i), [][]float32{mvTokenAt(i)}, nil); err != nil {
			t.Fatalf("client VectorMVAdd %d: %v", i, err)
		}
	}

	// MV resplit 4 -> 8 over TCP.
	if err := store.VectorMVResplit(ctx, coll, 8); err != nil {
		t.Fatalf("client VectorMVResplit: %v", err)
	}
	if p, gen, ok := emb.Catalog().PartitionsGen(coll); !ok || p != 8 || gen != 1 {
		t.Fatalf("after mv resplit: PartitionsGen = (%d, gen=%d, ok=%v), want (8, 1, true)", p, gen, ok)
	}

	// Baseline over the flipped catalog; the remote MV top-k must match in order.
	query := [][]float32{mvTokenAt(17)}
	want, _, err := emb.VectorMVSearch(ctx, coll, query, k, rostam.MultiSearchOpts{CandidatesPerToken: 100})
	if err != nil {
		t.Fatalf("embedded baseline mv search: %v", err)
	}
	if len(want) == 0 {
		t.Fatal("embedded baseline mv search returned no results")
	}
	got, _, err := store.VectorMVSearch(ctx, coll, query, k, rostam.MultiSearchOpts{CandidatesPerToken: 100})
	if err != nil {
		t.Fatalf("client VectorMVSearch after resplit: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("client mv search after resplit returned %d results, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].ID != want[i].ID {
			t.Fatalf("client mv search after resplit rank %d: got id %d, want %d (post-flip merge mismatch)",
				i, got[i].ID, want[i].ID)
		}
	}

	// After a CLEAN MV resplit there are no orphans: VectorMVResplit itself drops the
	// old generation as its final step, so cleanup over TCP must report exactly 0.
	dropped, err := store.VectorMVResplitCleanup(ctx, coll)
	if err != nil {
		t.Fatalf("client VectorMVResplitCleanup: %v", err)
	}
	if dropped != 0 {
		t.Fatalf("cleanup after clean resplit dropped %d, want 0 (resplit already drops old gen)", dropped)
	}

	// Now prove MV cleanup does REAL work over the remote path: seed forward-orphan
	// MV partitions in a NON-live generation (gen 2; live is gen 1) the same way the
	// single-node TestEmbeddedMVResplitCleanup does — physical create with Partitions=0.
	const orphans = 3
	physCfg := rostam.MultiVectorConfig{Dim: 4}
	for p := 0; p < orphans; p++ {
		phys := string(ops.PartitionKeyGen(coll, 2, p))
		if _, err := emb.Call(ctx, "vector_mv_create_collection", ops.EncodeMVCreateArgs(phys, physCfg)); err != nil {
			t.Fatalf("seed gen-2 mv orphan %d: %v", p, err)
		}
	}
	// Cleanup over TCP must sweep exactly the seeded orphans and the count must flow
	// faithfully back through the remote codec.
	dropped, err = store.VectorMVResplitCleanup(ctx, coll)
	if err != nil {
		t.Fatalf("client VectorMVResplitCleanup (orphans): %v", err)
	}
	if dropped != orphans {
		t.Fatalf("mv cleanup dropped %d, want %d (exact seeded orphan count)", dropped, orphans)
	}
	// Idempotent: a second cleanup over TCP drops nothing.
	dropped, err = store.VectorMVResplitCleanup(ctx, coll)
	if err != nil {
		t.Fatalf("client VectorMVResplitCleanup (idempotent): %v", err)
	}
	if dropped != 0 {
		t.Fatalf("second mv cleanup dropped %d, want 0 (idempotent)", dropped)
	}
}

// TestRemoteOnlineReshardTCPClient drives an ONLINE dense reshard over the REAL
// Go networked client (TCP wire) against a server whose dispatcher is the fanout
// decorator, WITH a concurrent writer goroutine inserting and deleting over the
// same TCP client for the whole duration of the reshard. This proves the remote
// reshard virtual-op path preserves the dual-write / no-loss guarantee end-to-
// end: the catalog flips cluster-side, post-reshard search returns the correct
// global top-k, every concurrent insert survives, and every concurrent delete
// stays gone (no resurrection) — all observed through the TCP client.
func TestRemoteOnlineReshardTCPClient(t *testing.T) {
	// Mirror the embedded concurrent-writes harness: a short drain grace plus
	// continuous-loop writers that re-drive their disjoint id ranges to their
	// final state throughout the reshard. Continuous re-application guarantees
	// each id is written during the live dual-write window (the no-loss /
	// no-resurrection guarantee covers writes applied while resharding, exactly
	// like TestEmbeddedOnlineReshardConcurrentWrites).
	defer rostam.SetReshardDrainGrace(30 * time.Millisecond)()

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}

	embStore, err := rostam.NewEmbedded(rostam.EmbeddedConfig{
		NodeID:    "n1",
		DataDir:   t.TempDir(),
		NumShards: 1,
		Bootstrap: true,
		Ops:       reg,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = embStore.Close() })
	waitLeaderEmbedded(t, embStore)
	emb := embStore.(*rostam.Embedded)

	disp := rostam.NewFanoutDispatcher(emb, emb.Node())
	srv, err := server.New(server.Config{Addr: "127.0.0.1:0", Dispatcher: disp})
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()
	t.Cleanup(func() { _ = srv.Close() })

	store, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{srv.Addr().String()}, Ops: reg})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	// Online reshard is synchronous but LIVE; give a generous deadline covering
	// the drain grace + full streamed copy.
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	const coll = "orp"
	const base = 120 // ids [0,base) seeded before the reshard
	const k = 10

	cfg := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: 4}
	if err := store.CreateCollection(ctx, coll, cfg); err != nil {
		t.Fatalf("client CreateCollection: %v", err)
	}
	for i := 0; i < base; i++ {
		if err := store.VectorInsert(ctx, coll, uint64(i), tieFreeVec(i)); err != nil {
			t.Fatalf("client VectorInsert %d: %v", i, err)
		}
	}

	// Concurrent writers over the SAME TCP client for the whole reshard:
	//   - inserter: adds a disjoint HIGH id range [addedFrom,addedTo) (so it never
	//     perturbs the low-id top-k parity assertion) — must all survive.
	//   - deleter: deletes a disjoint range [delFrom,delTo) out of the seeded set
	//     — must all stay gone (no resurrection).
	// Each loops until the reshard goroutine signals done, hammering the dual-
	// write / copy windows end-to-end over the network.
	const addedFrom = 1000
	const addedTo = 1200
	const delFrom = 0
	const delTo = 20
	var wg sync.WaitGroup
	done := make(chan struct{})
	writeErr := make(chan error, 2)
	worker := func(fn func() error) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-done:
					return
				default:
				}
				if err := fn(); err != nil {
					select {
					case writeErr <- err:
					default:
					}
					return
				}
			}
		}()
	}
	worker(func() error {
		// Upsert (not Insert) so the continuous loop is idempotent — VectorInsert
		// rejects an already-present id; VectorUpsert re-applies the same value.
		for id := addedFrom; id < addedTo; id++ {
			if err := store.VectorUpsert(ctx, coll, uint64(id), tieFreeVec(id), "", rostam.VectorInsertOpts{}); err != nil {
				return fmt.Errorf("concurrent upsert %d: %w", id, err)
			}
		}
		return nil
	})
	worker(func() error {
		for id := delFrom; id < delTo; id++ {
			if _, err := store.VectorDelete(ctx, coll, uint64(id)); err != nil {
				return fmt.Errorf("concurrent delete %d: %w", id, err)
			}
		}
		return nil
	})

	// Online reshard 4 -> 8 over TCP while the writers run.
	reshardErr := store.VectorReshard(ctx, coll, 8)
	close(done)
	wg.Wait()
	if reshardErr != nil {
		t.Fatalf("client VectorReshard: %v", reshardErr)
	}
	select {
	case werr := <-writeErr:
		t.Fatalf("concurrent writer: %v", werr)
	default:
	}

	// The catalog flipped cluster-side to the new generation (P=8, gen=1).
	if p, gen, ok := emb.Catalog().PartitionsGen(coll); !ok || p != 8 || gen != 1 {
		t.Fatalf("after reshard: PartitionsGen = (%d, gen=%d, ok=%v), want (8, 1, true)", p, gen, ok)
	}

	// Post-reshard search parity: the embedded baseline over the flipped catalog
	// must match the remote top-k in order. The query ranks by ascending id over
	// the surviving low ids [delTo, base) (the deleted [0,delTo) are gone), so the
	// concurrent high-id inserts never enter the top-k.
	q := tieFreeQuery()
	want, _, err := emb.VectorSearchExt(ctx, coll, q, k, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatalf("embedded baseline search: %v", err)
	}
	if len(want) != k {
		t.Fatalf("embedded baseline returned %d, want %d", len(want), k)
	}
	got, err := store.VectorSearch(ctx, coll, q, k)
	if err != nil {
		t.Fatalf("client VectorSearch after reshard: %v", err)
	}
	if len(got) != k {
		t.Fatalf("client search after reshard returned %d results, want %d", len(got), k)
	}
	for i := range want {
		if got[i].ID != want[i].ID {
			t.Fatalf("client search after reshard rank %d: got id %d, want %d (post-flip merge mismatch)\n got=%v\nwant=%v",
				i, got[i].ID, want[i].ID, resultIDs(got), resultIDs(want))
		}
	}

	// No lost / no resurrected concurrent writes: oracle-scan the NEW gen's
	// physical partitions (the same raw oracle the embedded reshard tests trust;
	// the point-level vector_exists op is not fanout-routed for partitioned
	// logical names, so we verify the landed data directly on the new gen). The
	// writes themselves all went over TCP, so this proves the remote dual-write
	// path preserved every concurrent write across the reshard.
	live := reshardScanGen(t, emb, coll, 8, 1)
	for id := addedFrom; id < addedTo; id++ {
		if _, ok := live[uint64(id)]; !ok {
			t.Fatalf("concurrent insert %d lost across reshard (absent from new gen)", id)
		}
	}
	for id := delFrom; id < delTo; id++ {
		if _, ok := live[uint64(id)]; ok {
			t.Fatalf("concurrent delete %d resurrected across reshard (present in new gen)", id)
		}
	}
	for id := delTo; id < base; id++ {
		if _, ok := live[uint64(id)]; !ok {
			t.Fatalf("seeded survivor %d missing from new gen post-reshard", id)
		}
	}
}

// TestRemoteMVOnlineReshardTCPClient is the multi-vector counterpart to
// TestRemoteOnlineReshardTCPClient: a remote store.VectorMVReshard over TCP with
// a concurrent MV writer (adds + deletes) running for the reshard's duration.
// Proves the MV dual-write / no-loss guarantee end-to-end over the network.
func TestRemoteMVOnlineReshardTCPClient(t *testing.T) {
	defer rostam.SetReshardDrainGrace(30 * time.Millisecond)()

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}

	embStore, err := rostam.NewEmbedded(rostam.EmbeddedConfig{
		NodeID:    "n1",
		DataDir:   t.TempDir(),
		NumShards: 1,
		Bootstrap: true,
		Ops:       reg,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = embStore.Close() })
	waitLeaderEmbedded(t, embStore)
	emb := embStore.(*rostam.Embedded)

	disp := rostam.NewFanoutDispatcher(emb, emb.Node())
	srv, err := server.New(server.Config{Addr: "127.0.0.1:0", Dispatcher: disp})
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()
	t.Cleanup(func() { _ = srv.Close() })

	store, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{srv.Addr().String()}, Ops: reg})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	const coll = "mvorp"
	const base = 60
	const k = 10

	if err := store.VectorMVCreateCollection(ctx, coll, rostam.MultiVectorConfig{Dim: 4, Partitions: 4}); err != nil {
		t.Fatalf("client VectorMVCreateCollection: %v", err)
	}
	for i := 0; i < base; i++ {
		if err := store.VectorMVAdd(ctx, coll, uint64(i), [][]float32{mvTokenAt(i)}, nil); err != nil {
			t.Fatalf("client VectorMVAdd %d: %v", i, err)
		}
	}

	// Continuous-loop MV writers (see the dense test) over the same TCP client.
	const addedFrom = 1000
	const addedTo = 1100
	const delFrom = 0
	const delTo = 10
	var wg sync.WaitGroup
	done := make(chan struct{})
	writeErr := make(chan error, 2)
	worker := func(fn func() error) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-done:
					return
				default:
				}
				if err := fn(); err != nil {
					select {
					case writeErr <- err:
					default:
					}
					return
				}
			}
		}()
	}
	worker(func() error {
		for id := addedFrom; id < addedTo; id++ {
			if err := store.VectorMVAdd(ctx, coll, uint64(id), [][]float32{mvTokenAt(id % 40)}, nil); err != nil {
				return fmt.Errorf("concurrent mv add %d: %w", id, err)
			}
		}
		return nil
	})
	worker(func() error {
		for id := delFrom; id < delTo; id++ {
			if _, err := store.VectorMVDelete(ctx, coll, uint64(id)); err != nil {
				return fmt.Errorf("concurrent mv delete %d: %w", id, err)
			}
		}
		return nil
	})

	// Online MV reshard 4 -> 8 over TCP while the writers run.
	reshardErr := store.VectorMVReshard(ctx, coll, 8)
	close(done)
	wg.Wait()
	if reshardErr != nil {
		t.Fatalf("client VectorMVReshard: %v", reshardErr)
	}
	select {
	case werr := <-writeErr:
		t.Fatalf("concurrent mv writer: %v", werr)
	default:
	}

	if p, gen, ok := emb.Catalog().PartitionsGen(coll); !ok || p != 8 || gen != 1 {
		t.Fatalf("after mv reshard: PartitionsGen = (%d, gen=%d, ok=%v), want (8, 1, true)", p, gen, ok)
	}

	// Post-reshard MV search parity against the embedded baseline (flipped catalog).
	query := [][]float32{mvTokenAt(17)}
	want, _, err := emb.VectorMVSearch(ctx, coll, query, k, rostam.MultiSearchOpts{CandidatesPerToken: 100})
	if err != nil {
		t.Fatalf("embedded baseline mv search: %v", err)
	}
	if len(want) == 0 {
		t.Fatal("embedded baseline mv search returned no results")
	}
	got, _, err := store.VectorMVSearch(ctx, coll, query, k, rostam.MultiSearchOpts{CandidatesPerToken: 100})
	if err != nil {
		t.Fatalf("client VectorMVSearch after reshard: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("client mv search after reshard returned %d results, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].ID != want[i].ID {
			t.Fatalf("client mv search after reshard rank %d: got id %d, want %d (post-flip merge mismatch)",
				i, got[i].ID, want[i].ID)
		}
	}

	// No lost / no resurrected concurrent MV writes: oracle-scan the NEW gen's
	// physical partitions (the raw oracle the embedded MV reshard tests trust;
	// vector_mv_exists is not fanout-routed for partitioned logical names). The
	// adds/deletes all went over TCP, so this proves the remote MV dual-write
	// path preserved every concurrent write across the reshard.
	live := reshardScanGenMV(t, emb, coll, 8, 1)
	for id := addedFrom; id < addedTo; id++ {
		if _, ok := live[uint64(id)]; !ok {
			t.Fatalf("concurrent mv add %d lost across reshard (absent from new gen)", id)
		}
	}
	for id := delFrom; id < delTo; id++ {
		if _, ok := live[uint64(id)]; ok {
			t.Fatalf("concurrent mv delete %d resurrected across reshard (present in new gen)", id)
		}
	}
	for id := delTo; id < base; id++ {
		if _, ok := live[uint64(id)]; !ok {
			t.Fatalf("seeded mv survivor %d missing from new gen post-reshard", id)
		}
	}
}

// TestRemoteSearchDegradedTCPClient drives the degraded-partition signal over the
// REAL Go networked client (TCP wire) against a server whose dispatcher is the
// fanout decorator. It proves the degraded trailer survives the round trip: with
// one physical partition dropped, a Partial-mode search over TCP returns
// meta.Degraded==true and meta.Missing==[]int{2} plus partial results, while a
// Fail-mode search errors. A search_docs variant exercises a second (non-search)
// degraded codec end-to-end over the wire.
func TestRemoteSearchDegradedTCPClient(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}

	embStore, err := rostam.NewEmbedded(rostam.EmbeddedConfig{
		NodeID:    "n1",
		DataDir:   t.TempDir(),
		NumShards: 1,
		Bootstrap: true,
		Ops:       reg,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = embStore.Close() })
	waitLeaderEmbedded(t, embStore)
	emb := embStore.(*rostam.Embedded)

	disp := rostam.NewFanoutDispatcher(emb, emb.Node())
	srv, err := server.New(server.Config{Addr: "127.0.0.1:0", Dispatcher: disp})
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()
	t.Cleanup(func() { _ = srv.Close() })

	store, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{srv.Addr().String()}, Ops: reg})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	ctx := context.Background()
	const coll = "dg"
	const n = 80
	const k = 10
	const P = 4

	cfg := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: P}
	if err := store.CreateCollection(ctx, coll, cfg); err != nil {
		t.Fatalf("client CreateCollection: %v", err)
	}
	for i := 0; i < n; i++ {
		if err := store.VectorInsert(ctx, coll, uint64(i), tieFreeVec(i)); err != nil {
			t.Fatalf("client VectorInsert %d: %v", i, err)
		}
	}

	// Make physical partition 2 unreachable by dropping its physical collection on
	// the node. The decorator still fans out to all P partitions; partition 2's
	// vector_search then fails with "unknown collection", which the coordinator
	// surfaces as a missing partition (degraded in Partial mode).
	if _, err := emb.Call(ctx, "vector_drop_collection",
		ops.EncodeDropCollectionArgs(string(ops.PartitionKeyGen(coll, 0, 2)))); err != nil {
		t.Fatalf("drop partition 2: %v", err)
	}

	q := tieFreeQuery()

	// Partial mode (default) over TCP: degraded=true, partition 2 missing, partial
	// results — the degraded trailer flowed back through the wire codec.
	res, meta, err := store.VectorSearchExt(ctx, coll, q, k, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatalf("Partial VectorSearchExt over TCP: %v", err)
	}
	if !meta.Degraded {
		t.Fatalf("Partial: meta.Degraded = false, want true (degraded trailer lost over wire)")
	}
	if !reflect.DeepEqual(meta.Missing, []int{2}) {
		t.Fatalf("Partial: meta.Missing = %v, want [2]", meta.Missing)
	}
	if len(res) == 0 {
		t.Fatalf("Partial: expected partial results from reachable partitions, got none")
	}

	// Fail mode over TCP: the OnPartitionUnavailable opt flows through the request
	// codec + decorator, so the unreachable partition errors the whole query.
	if _, _, err := store.VectorSearchExt(ctx, coll, q, k,
		rostam.VectorSearchOpts{OnPartitionUnavailable: 1}); err == nil {
		t.Fatalf("Fail mode over TCP: expected error from unreachable partition, got nil")
	}

	// Non-search variant: search_docs exercises the EncodeVectorDocsDegraded codec
	// end-to-end over the wire. Partial mode must report the same missing partition.
	docs, dmeta, err := store.VectorSearchDocs(ctx, coll, q, k, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatalf("Partial VectorSearchDocs over TCP: %v", err)
	}
	if !dmeta.Degraded {
		t.Fatalf("docs Partial: meta.Degraded = false, want true")
	}
	if !reflect.DeepEqual(dmeta.Missing, []int{2}) {
		t.Fatalf("docs Partial: meta.Missing = %v, want [2]", dmeta.Missing)
	}
	if len(docs) == 0 {
		t.Fatalf("docs Partial: expected partial results, got none")
	}
}

// TestRemoteConsistencyOptsFailModeTCPClient drives the ReadConsistency /
// OnPartitionUnavailable opts over the REAL Go networked client (TCP wire) for
// the hybrid, groups, scroll, and MV fan-out READ paths. It proves the opts
// trailer survives the request round trip for each op: with one physical
// partition dropped, Partial mode returns Degraded + partial results, while
// Fail mode (OnPartitionUnavailable=1) errors the whole query. The Fail-mode
// assertions are the proof the opts actually crossed the wire — without the
// request-side opts codec the server would default to Partial and never error.
func TestRemoteConsistencyOptsFailModeTCPClient(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}

	embStore, err := rostam.NewEmbedded(rostam.EmbeddedConfig{
		NodeID:    "n1",
		DataDir:   t.TempDir(),
		NumShards: 1,
		Bootstrap: true,
		Ops:       reg,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = embStore.Close() })
	waitLeaderEmbedded(t, embStore)
	emb := embStore.(*rostam.Embedded)

	disp := rostam.NewFanoutDispatcher(emb, emb.Node())
	srv, err := server.New(server.Config{Addr: "127.0.0.1:0", Dispatcher: disp})
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()
	t.Cleanup(func() { _ = srv.Close() })

	store, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{srv.Addr().String()}, Ops: reg})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	ctx := context.Background()
	const P = 4
	const n = 80

	// --- Dense-family collection (hybrid, groups, scroll) ---------------------
	const dcoll = "co"
	dcfg := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: P}
	if err := store.CreateCollection(ctx, dcoll, dcfg); err != nil {
		t.Fatalf("client CreateCollection: %v", err)
	}
	for i := 0; i < n; i++ {
		sp := rostam.VectorSparse{Indices: []uint32{uint32(i % 7)}, Values: []float32{1}}
		md := rostam.VectorMetadata{"doc": vector.NewInt(int64(i % 20))}
		if err := store.VectorInsertExt(ctx, dcoll, uint64(i), tieFreeVec(i),
			rostam.VectorInsertOpts{Sparse: sp, Metadata: md}); err != nil {
			t.Fatalf("client VectorInsertExt %d: %v", i, err)
		}
	}
	// Make physical partition 2 unreachable.
	if _, err := emb.Call(ctx, "vector_drop_collection",
		ops.EncodeDropCollectionArgs(string(ops.PartitionKeyGen(dcoll, 0, 2)))); err != nil {
		t.Fatalf("drop dense partition 2: %v", err)
	}

	q := tieFreeQuery()
	qs := rostam.VectorSparse{Indices: []uint32{3}, Values: []float32{1}}

	// hybrid: Partial degraded + results; Fail errors.
	if res, meta, err := store.VectorHybridSearch(ctx, dcoll, q, 10,
		rostam.VectorHybridOpts{Sparse: qs}); err != nil {
		t.Fatalf("hybrid Partial over TCP: %v", err)
	} else if !meta.Degraded {
		t.Fatalf("hybrid Partial over TCP: meta.Degraded = false, want true")
	} else if len(res) == 0 {
		t.Fatalf("hybrid Partial over TCP: expected partial results, got none")
	}
	if _, _, err := store.VectorHybridSearch(ctx, dcoll, q, 10,
		rostam.VectorHybridOpts{Sparse: qs, OnPartitionUnavailable: 1}); err == nil {
		t.Fatalf("hybrid Fail over TCP: expected error (opts did not cross the wire), got nil")
	}

	// groups: Partial degraded + results; Fail errors.
	gopts := rostam.VectorGroupOpts{GroupBy: "doc", GroupSize: 3}
	if res, meta, err := store.VectorSearchGroups(ctx, dcoll, q, 5, gopts); err != nil {
		t.Fatalf("groups Partial over TCP: %v", err)
	} else if !meta.Degraded {
		t.Fatalf("groups Partial over TCP: meta.Degraded = false, want true")
	} else if len(res) == 0 {
		t.Fatalf("groups Partial over TCP: expected partial results, got none")
	}
	failGopts := gopts
	failGopts.OnPartitionUnavailable = 1
	if _, _, err := store.VectorSearchGroups(ctx, dcoll, q, 5, failGopts); err == nil {
		t.Fatalf("groups Fail over TCP: expected error (opts did not cross the wire), got nil")
	}

	// scroll: Partial degraded + results; Fail errors.
	if res, meta, _, err := store.VectorScroll(ctx, dcoll, rostam.VectorFilter{}, 0, rostam.VectorScrollOpts{}); err != nil {
		t.Fatalf("scroll Partial over TCP: %v", err)
	} else if !meta.Degraded {
		t.Fatalf("scroll Partial over TCP: meta.Degraded = false, want true")
	} else if len(res) == 0 {
		t.Fatalf("scroll Partial over TCP: expected partial results, got none")
	}
	if _, _, _, err := store.VectorScroll(ctx, dcoll, rostam.VectorFilter{}, 0,
		rostam.VectorScrollOpts{OnPartitionUnavailable: 1}); err == nil {
		t.Fatalf("scroll Fail over TCP: expected error (opts did not cross the wire), got nil")
	}

	// --- MV collection (now gains a Fail mode over the wire) ------------------
	const mvcoll = "co-mv"
	if err := store.VectorMVCreateCollection(ctx, mvcoll, rostam.MultiVectorConfig{Dim: 4, Partitions: P}); err != nil {
		t.Fatalf("client VectorMVCreateCollection: %v", err)
	}
	for i := 0; i < n; i++ {
		if err := store.VectorMVAdd(ctx, mvcoll, uint64(i), [][]float32{mvTokenAt(i)}, nil); err != nil {
			t.Fatalf("client VectorMVAdd %d: %v", i, err)
		}
	}
	if _, err := emb.Call(ctx, "vector_mv_drop_collection",
		ops.EncodeMVDeleteArgs(string(ops.PartitionKeyGen(mvcoll, 0, 2)), 0)); err != nil {
		t.Fatalf("drop mv partition 2: %v", err)
	}
	mvQuery := [][]float32{mvTokenAt(17)}
	if res, meta, err := store.VectorMVSearch(ctx, mvcoll, mvQuery, 10,
		rostam.MultiSearchOpts{CandidatesPerToken: 100}); err != nil {
		t.Fatalf("mv Partial over TCP: %v", err)
	} else if !meta.Degraded {
		t.Fatalf("mv Partial over TCP: meta.Degraded = false, want true")
	} else if len(res) == 0 {
		t.Fatalf("mv Partial over TCP: expected partial results, got none")
	}
	if _, _, err := store.VectorMVSearch(ctx, mvcoll, mvQuery, 10,
		rostam.MultiSearchOpts{CandidatesPerToken: 100, OnPartitionUnavailable: 1}); err == nil {
		t.Fatalf("mv Fail over TCP: expected error (opts did not cross the wire), got nil")
	}
}

// --- Rich payload-filter fan-out ---------------------------------
//
// The rich operators (match/regex/is_empty/is_null/datetime/dotted-key) are
// LEAF ops that ride the existing Filter JSON; the fan-out path scatters that
// filter to every physical partition untouched and unions/merges the results.
// These tests prove that round trip end-to-end over the REAL TCP client (here)
// and a 3-node cluster (cluster_fanout_integration_test.go) for search, scroll,
// and delete-by-filter. The point of the exercise: partition routing is by id
// HASH, so the docs satisfying any one predicate are spread across ALL physical
// partitions — a correct fan-out must scatter to every partition, and the
// asserted ground truth is computed INDEPENDENTLY in-test (by applying the same
// predicate to the seeded data), so a dropped/misrouted partition surfaces as a
// missing id rather than passing silently.

// richDoc is the seeded metadata for one document. The fields exercise each rich
// operator on a real value: a string field (match/regex), a string-array field
// that is sometimes empty/absent (is_empty), a real int64 unix-millisecond
// datetime field (dt range), and a FLATTENED dotted key "address.city" (dotted
// path eq). The id chooses every field deterministically so ground truth is a
// pure function of the id.
type richDoc struct {
	color string   // ValueString  — match/regex target
	tags  []string // ValueStrings — is_empty target (nil/empty => empty)
	tsMs  int64    // ValueInt      — datetime field (unix ms)
	city  string   // dotted key "address.city" => ValueString
}

// richBaseMs is a fixed RFC3339 epoch (2024-01-01T00:00:00Z) in unix ms; each
// doc's datetime is richBaseMs + id days, so the datetime ordering is a clean
// function of the id and the in-test ground truth is unambiguous.
const richBaseMs = int64(1704067200000) // 2024-01-01T00:00:00Z
const richDayMs = int64(86400000)

// richColors / richCities are small dictionaries cycled by id so the match /
// regex / dotted predicates each select a genuine, non-trivial subset spread
// across partitions.
var richColors = []string{"red apple", "green pear", "red cherry", "blue berry"}
var richCities = []string{"New York", "San Francisco", "New Orleans", "Boston"}

// richDocFor returns the deterministic seed for id i.
func richDocFor(i uint64) richDoc {
	d := richDoc{
		color: richColors[i%uint64(len(richColors))],
		tsMs:  richBaseMs + int64(i)*richDayMs,
		city:  richCities[i%uint64(len(richCities))],
	}
	// Every 3rd doc has an EMPTY tag set (nil) — the is_empty target; the rest
	// carry a non-empty tag so is_empty genuinely partitions the set.
	if i%3 != 0 {
		d.tags = []string{fmt.Sprintf("tag-%d", i%5)}
	}
	return d
}

// richMetadata lowers a richDoc to the engine metadata, including the flattened
// dotted key. Mirrors exactly what the predicates read via lookupPath.
func richMetadata(d richDoc) rostam.VectorMetadata {
	md := rostam.VectorMetadata{
		"color":        vector.NewString(d.color),
		"ts":           vector.NewInt(d.tsMs),
		"address.city": vector.NewString(d.city),
	}
	if len(d.tags) > 0 {
		md["tags"] = vector.NewStrings(d.tags)
	}
	// Note: an EMPTY tag set is represented by ABSENCE (no "tags" key), which is
	// is_empty=true by the operator's "absent => empty" rule.
	return md
}

// richGroundTruth computes, INDEPENDENTLY of the engine, the exact id set each
// rich predicate selects over ids 1..n. This is the oracle the fan-out is checked
// against: it replays the operator semantics by hand (against the same seed
// richDocFor produces) so an assertion catches a fan-out partition drop or
// misroute, not merely "whatever the engine returned." The predicates here mirror
// vector/filter.go exactly.
//
// Ids start at 1, NOT 0: the HNSW search path unconditionally skips any candidate
// whose user ID is 0 (`if id == 0 { continue }` in searchDenseLocked and the
// filter-first path in hnsw.go). This is a pre-existing, permanent sentinel guard
// — id 0 is excluded from ALL ANN search results regardless of the query vector or
// any filter; it is NOT an HNSW recall/entrypoint artifact and applies identically
// with and without a filter. Seeding ids 1..n means id 0 is never in the arena, so
// the SEARCH membership assertion is exact. Rich-filter correctness for every id is
// independently proven by the SCROLL and DELETE-BY-FILTER assertions below, which
// iterate the idMap directly (exhaustive, non-ANN) and never hit the id==0 guard.
func richGroundTruth(n uint64) (matchRed, regexBerry, isEmptyTags, dtMid, cityNY map[uint64]bool) {
	matchRed = map[uint64]bool{}    // color tokens ⊇ {"red"}
	regexBerry = map[uint64]bool{}  // color matches /berry$/
	isEmptyTags = map[uint64]bool{} // tags absent/empty
	dtMid = map[uint64]bool{}       // ts in [base+10d, base+20d)
	cityNY = map[uint64]bool{}      // address.city == "New York"
	lo := richBaseMs + 10*richDayMs
	hi := richBaseMs + 20*richDayMs
	for i := uint64(1); i <= n; i++ {
		d := richDocFor(i)
		// match "red": the color token set must contain the whole token "red"
		// (split on whitespace; the seeded colors are two-token strings).
		for _, tok := range strings.Fields(d.color) {
			if tok == "red" {
				matchRed[i] = true
			}
		}
		// regex "berry$".
		if strings.HasSuffix(d.color, "berry") {
			regexBerry[i] = true
		}
		// is_empty(tags): absent (nil) tag set.
		if len(d.tags) == 0 {
			isEmptyTags[i] = true
		}
		// datetime range [lo, hi).
		if d.tsMs >= lo && d.tsMs < hi {
			dtMid[i] = true
		}
		// dotted address.city == "New York".
		if d.city == "New York" {
			cityNY[i] = true
		}
	}
	return
}

// rfc3339 formats a unix-ms instant as the RFC3339 literal the datetime ops
// accept (UTC), so the in-test filter literal and the seeded int64-ms field
// describe the same instant.
func rfc3339(ms int64) string {
	return time.UnixMilli(ms).UTC().Format(time.RFC3339)
}

// resultIDSet returns the distinct ids in a search result slice.
func resultIDSet(res []rostam.VectorResult) map[uint64]bool {
	out := map[uint64]bool{}
	for _, r := range res {
		out[r.ID] = true
	}
	return out
}

// sameUint64Set reports set equality and, on mismatch, the symmetric difference
// for a useful failure message.
func sameUint64Set(got, want map[uint64]bool) (ok bool, missing, extra []uint64) {
	for id := range want {
		if !got[id] {
			missing = append(missing, id)
		}
	}
	for id := range got {
		if !want[id] {
			extra = append(extra, id)
		}
	}
	sortUint64(missing)
	sortUint64(extra)
	return len(missing) == 0 && len(extra) == 0, missing, extra
}

func sortUint64(s []uint64) {
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
}

// TestRemoteRichFilterTCPClient proves the rich payload-filter operators
// (match / regex / is_empty / datetime-range / dotted-key) fan out correctly to
// EVERY physical partition of a partitioned collection over the REAL Go
// networked TCP client (client -> TCP -> server -> fanoutDispatcher), through
// SEARCH, SCROLL, and DELETE-BY-FILTER. Routing is by id hash so each predicate's
// matching docs are scattered across all P partitions; the asserted id sets are
// computed independently in-test (richGroundTruth), so a dropped or misrouted
// partition shows up as a missing/extra id.
func TestRemoteRichFilterTCPClient(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}

	embStore, err := rostam.NewEmbedded(rostam.EmbeddedConfig{
		NodeID:    "n1",
		DataDir:   t.TempDir(),
		NumShards: 1,
		Bootstrap: true,
		Ops:       reg,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = embStore.Close() })
	waitLeaderEmbedded(t, embStore)
	emb := embStore.(*rostam.Embedded)

	// The server's dispatcher is the fanout decorator — every op the client sends
	// over TCP is dispatched through it, exactly like server.go's CLUSTER branch.
	disp := rostam.NewFanoutDispatcher(emb, emb.Node())
	srv, err := server.New(server.Config{Addr: "127.0.0.1:0", Dispatcher: disp})
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()
	t.Cleanup(func() { _ = srv.Close() })

	store, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{srv.Addr().String()}, Ops: reg})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	ctx := context.Background()
	const coll = "rich"
	const n = 120 // > P so every partition holds several docs and predicates spread

	// P=4 collection through the networked client.
	cfg := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: 4}
	if err := store.CreateCollection(ctx, coll, cfg); err != nil {
		t.Fatalf("client CreateCollection: %v", err)
	}
	if p, _, ok := emb.Catalog().PartitionsGen(coll); !ok || p != 4 {
		t.Fatalf("after client create: PartitionsGen = (%d, ok=%v), want (4, true)", p, ok)
	}

	// Insert n docs (ids 1..n) through the client, each with the rich metadata. Ids
	// hash to different physical partitions, so any predicate's matches span
	// partitions. (Ids start at 1 — see richGroundTruth for why id 0 is skipped.)
	for i := uint64(1); i <= n; i++ {
		md := richMetadata(richDocFor(i))
		if err := store.VectorInsertExt(ctx, coll, i, tieFreeVec(int(i)),
			rostam.VectorInsertOpts{Metadata: md}); err != nil {
			t.Fatalf("client VectorInsertExt %d: %v", i, err)
		}
	}

	matchRed, regexBerry, isEmptyTags, dtMid, cityNY := richGroundTruth(n)
	// Sanity: each predicate must select a non-trivial, non-full subset, else the
	// fan-out proof is vacuous (a filter that matches all/none can't catch a drop).
	for name, set := range map[string]map[uint64]bool{
		"matchRed": matchRed, "regexBerry": regexBerry, "isEmptyTags": isEmptyTags,
		"dtMid": dtMid, "cityNY": cityNY,
	} {
		if len(set) == 0 || uint64(len(set)) == n {
			t.Fatalf("ground-truth set %q has %d/%d members (vacuous filter)", name, len(set), n)
		}
	}

	// The five rich filters, paired with their independently-computed oracle.
	loMs := richBaseMs + 10*richDayMs
	hiMs := richBaseMs + 20*richDayMs
	type richCase struct {
		name   string
		filter rostam.VectorFilter
		want   map[uint64]bool
	}
	cases := []richCase{
		{"match", rostam.VectorFilter{Op: vector.FilterMatch, Field: "color", Value: vector.NewString("red")}, matchRed},
		{"regex", rostam.VectorFilter{Op: vector.FilterRegex, Field: "color", Value: vector.NewString("berry$")}, regexBerry},
		{"is_empty", rostam.VectorFilter{Op: vector.FilterIsEmpty, Field: "tags"}, isEmptyTags},
		{"dt_range", rostam.VectorFilter{Op: vector.FilterAnd, And: []rostam.VectorFilter{
			{Op: vector.FilterDtGte, Field: "ts", Value: vector.NewString(rfc3339(loMs))},
			{Op: vector.FilterDtLt, Field: "ts", Value: vector.NewString(rfc3339(hiMs))},
		}}, dtMid},
		{"dotted_eq", rostam.VectorFilter{Op: vector.FilterEq, Field: "address.city", Value: vector.NewString("New York")}, cityNY},
	}

	// --- SEARCH fan-out: k=n so a correct cross-partition union returns EXACTLY
	// the matching set (a single partition's slice would be a strict subset). ---
	for _, c := range cases {
		res, _, err := store.VectorSearchExt(ctx, coll, tieFreeQuery(), n, rostam.VectorSearchOpts{Filter: c.filter})
		if err != nil {
			t.Fatalf("search %s: %v", c.name, err)
		}
		got := resultIDSet(res)
		if ok, missing, extra := sameUint64Set(got, c.want); !ok {
			t.Fatalf("SEARCH %s fan-out id set wrong: missing=%v extra=%v (want %d ids; a dropped partition shows here)",
				c.name, missing, extra, len(c.want))
		}
	}

	// --- SCROLL fan-out: limit=0 returns every matching doc across partitions. ---
	for _, c := range cases {
		docs, _, _, err := store.VectorScroll(ctx, coll, c.filter, 0, rostam.VectorScrollOpts{})
		if err != nil {
			t.Fatalf("scroll %s: %v", c.name, err)
		}
		got := docIDSet(docs)
		if ok, missing, extra := sameUint64Set(got, c.want); !ok {
			t.Fatalf("SCROLL %s fan-out id set wrong: missing=%v extra=%v (want %d ids)",
				c.name, missing, extra, len(c.want))
		}
	}

	// --- DELETE-BY-FILTER fan-out: delete the datetime-range set; assert exactly
	// those docs (across ALL partitions) are gone and every non-matching doc
	// survives. We verify by scrolling the whole collection afterward. ---
	deleted, err := store.VectorDeleteByFilter(ctx, coll, cases[3].filter) // dt_range
	if err != nil {
		t.Fatalf("delete-by-filter dt_range: %v", err)
	}
	if deleted != len(dtMid) {
		t.Fatalf("delete-by-filter dt_range deleted %d, want %d (must hit every partition)", deleted, len(dtMid))
	}
	// Survivors = all ids (1..n) minus the deleted set; assert exact survivor set.
	wantSurvivors := map[uint64]bool{}
	for i := uint64(1); i <= n; i++ {
		if !dtMid[i] {
			wantSurvivors[i] = true
		}
	}
	all, _, _, err := store.VectorScroll(ctx, coll, rostam.VectorFilter{}, 0, rostam.VectorScrollOpts{})
	if err != nil {
		t.Fatalf("post-delete full scroll: %v", err)
	}
	gotSurvivors := docIDSet(all)
	if ok, missing, extra := sameUint64Set(gotSurvivors, wantSurvivors); !ok {
		t.Fatalf("post-delete survivors wrong: missing=%v extra=%v (want %d survivors)",
			missing, extra, len(wantSurvivors))
	}
	// And none of the deleted datetime-range docs remain.
	for id := range dtMid {
		if gotSurvivors[id] {
			t.Fatalf("delete-by-filter left datetime-matched id %d alive (partition not reached)", id)
		}
	}
}

// httpResultIDs extracts result IDs from an HTTP-decoded result slice (for
// failure messages).
func httpResultIDs(res []vector.Result) []uint64 {
	ids := make([]uint64, len(res))
	for i, r := range res {
		ids[i] = r.ID
	}
	return ids
}

// =====================================================================
// Geo-filter fan-out: the geo analogue of the rich-filter block
// above. ValueGeo metadata ("loc" field) is seeded on a deterministic
// lat/lon grid keyed by id; the radius / bounding-box / polygon predicates
// each select a genuine, partition-spanning subset; and the asserted id sets
// are computed INDEPENDENTLY in-test (geoGroundTruth replays haversine /
// point-in-box / ray-casting by hand, NOT via the engine's index), so a
// dropped or misrouted partition surfaces as a missing/extra id.

// geoGridG is the side length of the lat/lon grid the seeded points sit on:
// id i (1..G*G) maps to cell (row,col) = ((i-1)%G, (i-1)/G). With G=11 it
// holds the n=120 docs (cells 0..119) on a regular grid so the geometry of
// each query region is unambiguous and the in-test ground truth is exact.
const geoGridG = 11

// geoLat0 / geoLon0 / geoStep place the grid in a single well-behaved patch of
// the northern hemisphere (no antimeridian / pole crossing, both documented
// non-goals). step=0.5deg ~= 55 km in latitude, comfortably larger than the
// geohash cell so the spatial index must union several cells per query.
const (
	geoLat0  = 40.0
	geoLon0  = -100.0
	geoStep  = 0.5
	earthRad = 6_371_000.0 // mirrors vector/geo.go earthRadiusM
)

// geoLocFor returns the (lat,lon) of the ValueGeo "loc" field for id i. It is a
// pure function of the id (grid cell), so both the seed and the ground truth
// derive from it and can never drift.
func geoLocFor(i uint64) (lat, lon float64) {
	cell := i - 1 // ids start at 1
	row := cell % geoGridG
	col := (cell / geoGridG) % geoGridG
	lat = geoLat0 + float64(row)*geoStep
	lon = geoLon0 + float64(col)*geoStep
	return
}

// geoMetadata lowers id i to engine metadata carrying a real vector.NewGeo
// value under "loc" (the same key the geo filters read via lookupPath).
func geoMetadata(i uint64) rostam.VectorMetadata {
	lat, lon := geoLocFor(i)
	return rostam.VectorMetadata{"loc": vector.NewGeo(lat, lon)}
}

// haversineTest is an INDEPENDENT re-implementation of the spherical great-circle
// distance (meters) the engine uses, so the ground truth is computed without
// calling any engine code. It mirrors vector/geo.go haversineMeters exactly.
func haversineTest(lat1, lon1, lat2, lon2 float64) float64 {
	const deg2rad = math.Pi / 180
	p1 := lat1 * deg2rad
	p2 := lat2 * deg2rad
	dp := (lat2 - lat1) * deg2rad
	dl := (lon2 - lon1) * deg2rad
	sdp := math.Sin(dp / 2)
	sdl := math.Sin(dl / 2)
	a := sdp*sdp + math.Cos(p1)*math.Cos(p2)*sdl*sdl
	if a > 1 {
		a = 1
	}
	return earthRad * 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
}

// pointInBoxTest mirrors vector/geo.go pointInBox (inclusive on all edges).
func pointInBoxTest(lat, lon, minLat, minLon, maxLat, maxLon float64) bool {
	return lat >= minLat && lat <= maxLat && lon >= minLon && lon <= maxLon
}

// pointInPolygonTest mirrors vector/geo.go pointInPolygon (even-odd ray-cast,
// lat=Y, lon=X, ray toward +lon), an independent oracle for the polygon op.
func pointInPolygonTest(lat, lon float64, poly []float64) bool {
	inside := false
	m := len(poly) / 2
	j := m - 1
	for i := 0; i < m; i++ {
		yi, xi := poly[2*i], poly[2*i+1]
		yj, xj := poly[2*j], poly[2*j+1]
		if (yi > lat) != (yj > lat) {
			xCross := (xj-xi)*(lat-yi)/(yj-yi) + xi
			if lon < xCross {
				inside = !inside
			}
		}
		j = i
	}
	return inside
}

// geoGroundTruth computes, INDEPENDENTLY of the engine, the exact id set each
// geo predicate selects over ids 1..n. The center/box/polygon below are chosen
// to carve out a proper, partition-spanning sub-region of the grid (the
// non-vacuity + spread guards in the test prove neither degenerates).
func geoGroundTruth(n uint64) (radius, box, polygon map[uint64]bool) {
	radius = map[uint64]bool{}
	box = map[uint64]bool{}
	polygon = map[uint64]bool{}
	for i := uint64(1); i <= n; i++ {
		lat, lon := geoLocFor(i)
		if haversineTest(geoCenterLat, geoCenterLon, lat, lon) <= geoRadiusM {
			radius[i] = true
		}
		if pointInBoxTest(lat, lon, geoBoxMinLat, geoBoxMinLon, geoBoxMaxLat, geoBoxMaxLon) {
			box[i] = true
		}
		if pointInPolygonTest(lat, lon, geoPoly) {
			polygon[i] = true
		}
	}
	return
}

// Query-region constants. The radius is centered mid-grid; the box covers an
// off-center rectangle; the polygon is a deliberately CONCAVE ring (an arrow /
// chevron) so the ray-cast even-odd rule is genuinely exercised. All three are
// sub-regions of the grid (proven non-vacuous + partition-spread by the test).
const (
	// center of the grid (row 5, col 5 == cell at lat 42.5, lon -97.5)
	geoCenterLat = geoLat0 + 5*geoStep
	geoCenterLon = geoLon0 + 5*geoStep
	// ~120 km radius: covers a diamond of cells around the center, not all.
	geoRadiusM = 120_000.0
	// box: rows 6..9 (lat 43.0..44.5), cols 0..3 (lon -100.0..-98.5).
	geoBoxMinLat = geoLat0 + 6*geoStep
	geoBoxMaxLat = geoLat0 + 9*geoStep
	geoBoxMinLon = geoLon0 + 0*geoStep
	geoBoxMaxLon = geoLon0 + 3*geoStep
)

// geoPoly is a concave (chevron) polygon over the grid, flat lat,lon,... It
// spans lon -100..-95 with a notch cut out of the middle so points in the notch
// are EXCLUDED — proving even-odd ray-casting, not a convex hull test.
var geoPoly = []float64{
	40.0, -100.0, // SW
	45.0, -100.0, // NW
	45.0, -95.0, // NE
	40.0, -95.0, // SE
	40.0, -96.5, // up into the notch from the south edge
	43.0, -97.5, // apex of the inward notch
	40.0, -98.5, // back down to the south edge
}

// partitionSpread returns the set of distinct partition indices the ids in set
// land on under id-hash routing with P partitions.
func partitionSpread(set map[uint64]bool, P int) map[int]bool {
	parts := map[int]bool{}
	for id := range set {
		parts[ops.PartitionOf(id, P)] = true
	}
	return parts
}

// docLoc extracts the ValueGeo "loc" lat/lon from a returned document's
// metadata, asserting the kind survived the wire exactly.
func docLoc(t *testing.T, d rostam.VectorDocument) (lat, lon float64) {
	t.Helper()
	v, ok := d.Metadata["loc"]
	if !ok {
		t.Fatalf("doc %d: no 'loc' metadata after wire round-trip", d.ID)
	}
	if v.Kind != vector.ValueGeo {
		t.Fatalf("doc %d: 'loc' kind = %v after wire round-trip, want ValueGeo", d.ID, v.Kind)
	}
	return v.Lat, v.Lon
}

// TestRemoteGeoFilterTCPClient proves the geo filter operators (geo_radius /
// geo_bounding_box / geo_polygon) fan out correctly to EVERY physical partition
// of a partitioned collection over the REAL Go networked TCP client (client ->
// TCP -> server -> fanoutDispatcher), through SEARCH, SCROLL, and
// DELETE-BY-FILTER, and that ValueGeo metadata survives the wire exactly.
// Locations are seeded on a lat/lon grid keyed by id; id-hash routing scatters
// each query's matches across all P partitions; the asserted id sets are
// computed independently in-test (geoGroundTruth), so a dropped or misrouted
// partition shows up as a missing/extra id.
func TestRemoteGeoFilterTCPClient(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}

	embStore, err := rostam.NewEmbedded(rostam.EmbeddedConfig{
		NodeID:    "n1",
		DataDir:   t.TempDir(),
		NumShards: 1,
		Bootstrap: true,
		Ops:       reg,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = embStore.Close() })
	waitLeaderEmbedded(t, embStore)
	emb := embStore.(*rostam.Embedded)

	disp := rostam.NewFanoutDispatcher(emb, emb.Node())
	srv, err := server.New(server.Config{Addr: "127.0.0.1:0", Dispatcher: disp})
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()
	t.Cleanup(func() { _ = srv.Close() })

	store, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{srv.Addr().String()}, Ops: reg})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	ctx := context.Background()
	const coll = "geo"
	const n = 120 // grid cells 0..119 (G=11 -> 121 cells), > P so matches spread
	const P = 4

	cfg := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: P}
	if err := store.CreateCollection(ctx, coll, cfg); err != nil {
		t.Fatalf("client CreateCollection: %v", err)
	}
	if p, _, ok := emb.Catalog().PartitionsGen(coll); !ok || p != P {
		t.Fatalf("after client create: PartitionsGen = (%d, ok=%v), want (%d, true)", p, ok, P)
	}

	// Insert n docs (ids 1..n) through the client, each with a real ValueGeo
	// "loc" field. Ids hash to different physical partitions, so any geo query's
	// matches span partitions. (Ids start at 1 — see richGroundTruth for why id 0
	// is skipped on the SEARCH path; SCROLL/DELETE are exhaustive and prove all.)
	for i := uint64(1); i <= n; i++ {
		md := geoMetadata(i)
		if err := store.VectorInsertExt(ctx, coll, i, tieFreeVec(int(i)),
			rostam.VectorInsertOpts{Metadata: md}); err != nil {
			t.Fatalf("client VectorInsertExt %d: %v", i, err)
		}
	}

	radius, box, polygon := geoGroundTruth(n)

	// Non-vacuity: each predicate must select a non-trivial, non-full subset,
	// else the fan-out proof is vacuous.
	for name, set := range map[string]map[uint64]bool{
		"radius": radius, "box": box, "polygon": polygon,
	} {
		if len(set) == 0 || uint64(len(set)) == n {
			t.Fatalf("geo ground-truth set %q has %d/%d members (vacuous filter)", name, len(set), n)
		}
	}

	// Partition-spread guard: each match set must land on >=2 physical partitions
	// under id-hash routing, so a correct answer DEMANDS a real fan-out (a single
	// partition could not produce it). This is asserted, not assumed.
	for name, set := range map[string]map[uint64]bool{
		"radius": radius, "box": box, "polygon": polygon,
	} {
		if parts := partitionSpread(set, P); len(parts) < 2 {
			t.Fatalf("geo set %q lands on only %d partition(s) %v — fan-out proof would be vacuous",
				name, len(parts), parts)
		}
	}

	radiusFilter := rostam.VectorFilter{Op: vector.FilterGeoRadius, Field: "loc", Geo: &vector.GeoCondition{
		CenterLat: geoCenterLat, CenterLon: geoCenterLon, RadiusM: geoRadiusM,
	}}
	boxFilter := rostam.VectorFilter{Op: vector.FilterGeoBox, Field: "loc", Geo: &vector.GeoCondition{
		MinLat: geoBoxMinLat, MinLon: geoBoxMinLon, MaxLat: geoBoxMaxLat, MaxLon: geoBoxMaxLon,
	}}
	polyFilter := rostam.VectorFilter{Op: vector.FilterGeoPolygon, Field: "loc", Geo: &vector.GeoCondition{
		Polygon: geoPoly,
	}}
	cases := []struct {
		name   string
		filter rostam.VectorFilter
		want   map[uint64]bool
	}{
		{"geo_radius", radiusFilter, radius},
		{"geo_box", boxFilter, box},
		{"geo_polygon", polyFilter, polygon},
	}

	// --- SEARCH fan-out: k=n so a correct cross-partition union returns EXACTLY
	// the matching set (a single partition's slice would be a strict subset). ---
	for _, c := range cases {
		res, _, err := store.VectorSearchExt(ctx, coll, tieFreeQuery(), n, rostam.VectorSearchOpts{Filter: c.filter})
		if err != nil {
			t.Fatalf("search %s: %v", c.name, err)
		}
		got := resultIDSet(res)
		if ok, missing, extra := sameUint64Set(got, c.want); !ok {
			t.Fatalf("SEARCH %s fan-out id set wrong: missing=%v extra=%v (want %d ids; a dropped partition shows here)",
				c.name, missing, extra, len(c.want))
		}
	}

	// --- SCROLL fan-out: limit=0 returns every matching doc across partitions.
	// Also proves the ValueGeo "loc" field round-trips over the wire EXACTLY: the
	// returned doc's lat/lon must equal the seeded coordinate. ---
	for _, c := range cases {
		docs, _, _, err := store.VectorScroll(ctx, coll, c.filter, 0, rostam.VectorScrollOpts{})
		if err != nil {
			t.Fatalf("scroll %s: %v", c.name, err)
		}
		got := docIDSet(docs)
		if ok, missing, extra := sameUint64Set(got, c.want); !ok {
			t.Fatalf("SCROLL %s fan-out id set wrong: missing=%v extra=%v (want %d ids)",
				c.name, missing, extra, len(c.want))
		}
		for _, d := range docs {
			wantLat, wantLon := geoLocFor(d.ID)
			gotLat, gotLon := docLoc(t, d)
			if gotLat != wantLat || gotLon != wantLon {
				t.Fatalf("scroll %s: doc %d ValueGeo loc = (%v,%v) over wire, want (%v,%v) — metadata corrupted",
					c.name, d.ID, gotLat, gotLon, wantLat, wantLon)
			}
		}
	}

	// --- DELETE-BY-FILTER fan-out: delete the geo_radius set; assert exactly
	// those docs (across ALL partitions) are gone and every non-matching doc
	// survives. We verify by scrolling the whole collection afterward. ---
	deleted, err := store.VectorDeleteByFilter(ctx, coll, radiusFilter)
	if err != nil {
		t.Fatalf("delete-by-filter geo_radius: %v", err)
	}
	if deleted != len(radius) {
		t.Fatalf("delete-by-filter geo_radius deleted %d, want %d (must hit every partition)", deleted, len(radius))
	}
	wantSurvivors := map[uint64]bool{}
	for i := uint64(1); i <= n; i++ {
		if !radius[i] {
			wantSurvivors[i] = true
		}
	}
	all, _, _, err := store.VectorScroll(ctx, coll, rostam.VectorFilter{}, 0, rostam.VectorScrollOpts{})
	if err != nil {
		t.Fatalf("post-delete full scroll: %v", err)
	}
	gotSurvivors := docIDSet(all)
	if ok, missing, extra := sameUint64Set(gotSurvivors, wantSurvivors); !ok {
		t.Fatalf("post-delete survivors wrong: missing=%v extra=%v (want %d survivors)",
			missing, extra, len(wantSurvivors))
	}
	for id := range radius {
		if gotSurvivors[id] {
			t.Fatalf("delete-by-filter left geo_radius-matched id %d alive (partition not reached)", id)
		}
	}
	// The other two geo predicates must still re-find their (overlapping) matches
	// among the survivors via SCROLL — i.e. delete removed only the radius set.
	for _, c := range []struct {
		name   string
		filter rostam.VectorFilter
		want   map[uint64]bool
	}{{"geo_box", boxFilter, box}, {"geo_polygon", polyFilter, polygon}} {
		wantAfter := map[uint64]bool{}
		for id := range c.want {
			if !radius[id] {
				wantAfter[id] = true
			}
		}
		docs, _, _, err := store.VectorScroll(ctx, coll, c.filter, 0, rostam.VectorScrollOpts{})
		if err != nil {
			t.Fatalf("post-delete scroll %s: %v", c.name, err)
		}
		if ok, missing, extra := sameUint64Set(docIDSet(docs), wantAfter); !ok {
			t.Fatalf("post-delete %s survivors wrong: missing=%v extra=%v", c.name, missing, extra)
		}
	}
}

// namedTitleVec builds a tie-free dim-8 vector for the "title" space (ids 1..n):
// the angle to namedTitleQuery grows monotonically with i, so cosine distance is
// strictly increasing in i (no ties) and the analytic top-k for "title" is the
// SMALLEST ids (1,2,3,...). For ids >= 1 the perturbation is never zero, so no
// vector coincides with the query (an exact query==stored coincidence is a
// degenerate HNSW case orthogonal to the fan-out under test).
func namedTitleVec(i int) []float32 {
	v := make([]float32, 8)
	v[0] = 1.0
	v[1] = float32(i) * 0.01
	return v
}

func namedTitleQuery() []float32 { q := make([]float32, 8); q[0] = 1.0; return q }

// namedImageVec is a SECOND, independent space with the OPPOSITE tie-free
// ordering (component 2 ramps the other way) so a search of "image" returns a
// different ranking than "title" — proving the vector NAME genuinely selects the
// space through fan-out rather than collapsing to one shared index. For ids 1..n
// the image top-k is the LARGEST ids (n, n-1, ...).
func namedImageVec(i, n int) []float32 {
	v := make([]float32, 8)
	v[0] = 1.0
	v[2] = float32(n+1-i) * 0.01
	return v
}

func namedImageQuery() []float32 { q := make([]float32, 8); q[0] = 1.0; return q }

// TestRemoteNamedVectorTCPClient drives a PARTITIONED (P=4) named-vector
// collection over the REAL Go networked client (TCP wire) against a server whose
// dispatcher is the fan-out decorator. It proves the vector_named_* op family
// fans out correctly across partitions: insert routes by point id, search of
// each named space (+ filtered) returns the exact CROSS-PARTITION union/top-k,
// search_docs carries the shared payload, and delete + scroll are correct across
// partitions. Ground truth is computed independently in-test (tie-free vectors →
// analytic ranking); the matching points are asserted SPREAD across >=2
// partitions so the fan-out is genuinely exercised.
func TestRemoteNamedVectorTCPClient(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	embStore, err := rostam.NewEmbedded(rostam.EmbeddedConfig{
		NodeID: "n1", DataDir: t.TempDir(), NumShards: 1, Bootstrap: true, Ops: reg,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = embStore.Close() })
	waitLeaderEmbedded(t, embStore)
	emb := embStore.(*rostam.Embedded)

	disp := rostam.NewFanoutDispatcher(emb, emb.Node())
	srv, err := server.New(server.Config{Addr: "127.0.0.1:0", Dispatcher: disp})
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()
	t.Cleanup(func() { _ = srv.Close() })
	addr := srv.Addr().String()

	store, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{addr}, Ops: reg})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	ctx := context.Background()
	const coll = "named_p"
	const P = 4
	const n = 100
	const k = 10

	cfg := map[string]rostam.NamedVectorParams{
		"title": {Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64},
		"image": {Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64},
	}
	if err := store.VectorNamedCreateCollection(ctx, coll, cfg, P); err != nil {
		t.Fatalf("client VectorNamedCreateCollection: %v", err)
	}
	if p, _, ok := emb.Catalog().PartitionsGen(coll); !ok || p != P {
		t.Fatalf("after client create: PartitionsGen = (%d, ok=%v), want (%d, true)", p, ok, P)
	}
	// Every physical partition must exist as a named collection.
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen(coll, 0, p))
		if _, err := emb.Call(ctx, "vector_named_get_config", ops.EncodeNamedNameArgs(phys)); err != nil {
			t.Fatalf("physical partition %d (%s) missing after create: %v", p, phys, err)
		}
	}

	// Insert n points (map of two named vecs + shared payload) over TCP, ids 1..n;
	// even ids are lang=en, odd are lang=fr. Routing to the owning partition happens
	// server side in fanNamedInsert.
	for i := 1; i <= n; i++ {
		lang := "en"
		if i%2 == 1 {
			lang = "fr"
		}
		vecs := map[string][]float32{"title": namedTitleVec(i), "image": namedImageVec(i, n)}
		payload := rostam.VectorMetadata{"lang": vector.NewString(lang), "n": vector.NewInt(int64(i))}
		if err := store.VectorNamedInsert(ctx, coll, uint64(i), vecs, payload, 0); err != nil {
			t.Fatalf("client VectorNamedInsert %d: %v", i, err)
		}
	}

	// ---- independent ground truth ----
	// "title" cosine distance is strictly increasing in i, so the analytic top-k is
	// the smallest ids 1,2,3,...; "image" is strictly increasing in (n+1-i), so its
	// top-k is the largest ids n, n-1, ... The two named spaces therefore rank
	// DIFFERENTLY, proving the name selects the space through fan-out.
	wantTitle := make([]uint64, k)
	for i := 0; i < k; i++ {
		wantTitle[i] = uint64(i + 1)
	}
	wantImage := make([]uint64, k)
	for i := 0; i < k; i++ {
		wantImage[i] = uint64(n - i)
	}

	// assertSpread proves the matched ids land on >=2 distinct physical partitions
	// (so fan-out is genuinely exercised, not a single-partition slice).
	assertSpread := func(label string, ids []uint64) {
		parts := map[int]bool{}
		for _, id := range ids {
			parts[ops.PartitionOf(id, P)] = true
		}
		if len(parts) < 2 {
			t.Fatalf("%s: matched ids %v span only %d partition(s); want >=2 (fan-out not exercised)", label, ids, len(parts))
		}
	}
	assertSpread("title top-k", wantTitle)
	assertSpread("image top-k", wantImage)

	// ---- search "title" over TCP (cross-partition union/top-k) ----
	gotTitle, err := store.VectorNamedSearch(ctx, coll, "title", namedTitleQuery(), k, rostam.VectorFilter{})
	if err != nil {
		t.Fatalf("client VectorNamedSearch title: %v", err)
	}
	if len(gotTitle) != k {
		t.Fatalf("title search returned %d, want %d (dropped partition?)", len(gotTitle), k)
	}
	for i := 1; i < len(gotTitle); i++ {
		if gotTitle[i].Distance < gotTitle[i-1].Distance {
			t.Fatalf("title results not distance-ascending at %d", i)
		}
	}
	for i := range wantTitle {
		if gotTitle[i].ID != wantTitle[i] {
			t.Fatalf("title rank %d: got id %d, want %d\n got=%v\nwant=%v",
				i, gotTitle[i].ID, wantTitle[i], resultIDs(gotTitle), wantTitle)
		}
	}

	// ---- search "image" (different ranking → the name selects the space) ----
	gotImage, err := store.VectorNamedSearch(ctx, coll, "image", namedImageQuery(), k, rostam.VectorFilter{})
	if err != nil {
		t.Fatalf("client VectorNamedSearch image: %v", err)
	}
	for i := range wantImage {
		if gotImage[i].ID != wantImage[i] {
			t.Fatalf("image rank %d: got id %d, want %d\n got=%v\nwant=%v",
				i, gotImage[i].ID, wantImage[i], resultIDs(gotImage), wantImage)
		}
	}

	// ---- filtered title search (lang=en → even ids only), cross-partition ----
	enFilter := rostam.VectorFilter{Op: vector.FilterEq, Field: "lang", Value: vector.NewString("en")}
	gotEn, err := store.VectorNamedSearch(ctx, coll, "title", namedTitleQuery(), k, enFilter)
	if err != nil {
		t.Fatalf("client filtered VectorNamedSearch: %v", err)
	}
	wantEn := make([]uint64, k)
	for i := 0; i < k; i++ {
		wantEn[i] = uint64(2 * (i + 1)) // even ids 2,4,6,... are the lang=en title top-k
	}
	assertSpread("filtered en top-k", wantEn)
	for i := range wantEn {
		if gotEn[i].ID != wantEn[i] {
			t.Fatalf("filtered rank %d: got id %d, want %d (predicate-eval per partition wrong?)\n got=%v\nwant=%v",
				i, gotEn[i].ID, wantEn[i], resultIDs(gotEn), wantEn)
		}
	}

	// ---- search_docs carries the shared payload across partitions ----
	docs, err := store.VectorNamedSearchDocs(ctx, coll, "title", namedTitleQuery(), k, rostam.VectorFilter{})
	if err != nil {
		t.Fatalf("client VectorNamedSearchDocs: %v", err)
	}
	if len(docs) != k {
		t.Fatalf("search_docs returned %d, want %d", len(docs), k)
	}
	for _, d := range docs {
		wantLang := "en"
		if d.ID%2 == 1 {
			wantLang = "fr"
		}
		got, ok := d.Metadata["lang"]
		if !ok || got.Str != wantLang {
			t.Fatalf("doc id %d payload lang = %+v, want %q (payload dropped through fan-out?)", d.ID, got, wantLang)
		}
		if nv, ok := d.Metadata["n"]; !ok || nv.Int != int64(d.ID) {
			t.Fatalf("doc id %d payload n = %+v, want %d", d.ID, nv, d.ID)
		}
	}

	// ---- scroll: all n points present across partitions, payload intact ----
	scrolled, _, err := store.VectorNamedScroll(ctx, coll, rostam.VectorFilter{}, 0, "")
	if err != nil {
		t.Fatalf("client VectorNamedScroll: %v", err)
	}
	if len(scrolled) != n || len(idSet(scrolled)) != n {
		t.Fatalf("scroll returned %d (%d distinct), want %d/%d", len(scrolled), len(idSet(scrolled)), n, n)
	}
	scrolledParts := map[int]bool{}
	for _, d := range scrolled {
		scrolledParts[ops.PartitionOf(d.ID, P)] = true
	}
	if len(scrolledParts) < 2 {
		t.Fatalf("scroll union spans only %d partition(s); want >=2", len(scrolledParts))
	}
	// Filtered scroll: exactly 50 lang=en (even) ids.
	enScroll, _, err := store.VectorNamedScroll(ctx, coll, enFilter, 0, "")
	if err != nil {
		t.Fatalf("client filtered VectorNamedScroll: %v", err)
	}
	if len(enScroll) != n/2 {
		t.Fatalf("filtered scroll returned %d, want %d", len(enScroll), n/2)
	}
	for _, d := range enScroll {
		if d.ID%2 != 0 {
			t.Fatalf("filtered scroll returned odd id %d (lang=en should be even)", d.ID)
		}
	}

	// ---- delete a point (id 4, which is in the title/en top-k) cross-partition ----
	const delID = 4
	ok, err := store.VectorNamedDelete(ctx, coll, delID)
	if err != nil {
		t.Fatalf("client VectorNamedDelete: %v", err)
	}
	if !ok {
		t.Fatalf("delete id %d reported not-existed, want existed", delID)
	}
	// It must be gone from BOTH named spaces and the scroll.
	afterTitle, err := store.VectorNamedSearch(ctx, coll, "title", namedTitleQuery(), k, rostam.VectorFilter{})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range afterTitle {
		if r.ID == delID {
			t.Fatalf("deleted id %d still present in title search", delID)
		}
	}
	afterImage, err := store.VectorNamedSearch(ctx, coll, "image", namedImageQuery(), n, rostam.VectorFilter{})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range afterImage {
		if r.ID == delID {
			t.Fatalf("deleted id %d still present in image search (delete didn't reach all named spaces)", delID)
		}
	}
	afterScroll, _, err := store.VectorNamedScroll(ctx, coll, rostam.VectorFilter{}, 0, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(afterScroll) != n-1 {
		t.Fatalf("after delete scroll = %d, want %d", len(afterScroll), n-1)
	}
	// Re-deleting is idempotent / reports not-existed.
	if ok, err := store.VectorNamedDelete(ctx, coll, delID); err != nil || ok {
		t.Fatalf("re-delete id %d: ok=%v err=%v, want ok=false err=nil", delID, ok, err)
	}

	// ---- drop fan-out: every physical partition + logical marker gone ----
	if err := store.VectorNamedDropCollection(ctx, coll); err != nil {
		t.Fatalf("client VectorNamedDropCollection: %v", err)
	}
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen(coll, 0, p))
		if _, err := emb.Call(ctx, "vector_named_get_config", ops.EncodeNamedNameArgs(phys)); err == nil {
			t.Fatalf("after drop: physical partition %d (%s) still exists", p, phys)
		}
	}
	if _, _, ok := disp.Partitioned(coll); ok {
		t.Fatal("after drop: collection still reports as partitioned")
	}
}

// TestRemoteGetPayloadTCPClient drives get-by-id + the four payload mutations
// over the REAL Go networked TCP client against PARTITIONED (P>=4) dense, named,
// and MV collections whose server dispatcher is the fan-out decorator. It is the
// concrete proof that get + payload-update route-by-id to the ONE owning physical
// partition: every point is fetched with the correct vector(s)/tokens + payload
// from its owning partition, an absent id returns the not-found FLAG (not an
// error), and set/overwrite/delete-keys/clear each reflect on a subsequent get.
// For the dense family a CROSS-PARTITION filtered search AFTER a payload-update
// returns the updated point, proving the payloadIdx reindex held on the owning
// physical partition. The touched ids are asserted SPREAD across >=2 partitions
// so route-by-id is genuinely exercised (not a single-partition slice). Ground
// truth is computed independently in-test.
func TestRemoteGetPayloadTCPClient(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	embStore, err := rostam.NewEmbedded(rostam.EmbeddedConfig{
		NodeID: "n1", DataDir: t.TempDir(), NumShards: 1, Bootstrap: true, Ops: reg,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = embStore.Close() })
	waitLeaderEmbedded(t, embStore)
	emb := embStore.(*rostam.Embedded)

	disp := rostam.NewFanoutDispatcher(emb, emb.Node())
	srv, err := server.New(server.Config{Addr: "127.0.0.1:0", Dispatcher: disp})
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()
	t.Cleanup(func() { _ = srv.Close() })
	addr := srv.Addr().String()

	store, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{addr}, Ops: reg})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	ctx := context.Background()
	const P = 4
	const n = 40 // > P so every partition holds several points; ids 1..n

	// assertSpread proves the ids land on >=2 distinct physical partitions, so
	// route-by-id is genuinely exercised rather than degenerating to one partition.
	assertSpread := func(label string, ids []uint64) {
		parts := map[int]bool{}
		for _, id := range ids {
			parts[ops.PartitionOf(id, P)] = true
		}
		if len(parts) < 2 {
			t.Fatalf("%s: ids %v span only %d partition(s); want >=2 (route-by-id not exercised)", label, ids, len(parts))
		}
	}
	allIDs := make([]uint64, n)
	for i := range allIDs {
		allIDs[i] = uint64(i + 1)
	}
	assertSpread("all inserted ids", allIDs)

	// ===================== DENSE =====================
	t.Run("dense", func(t *testing.T) {
		const coll = "dense_gp"
		cfg := rostam.VectorConfig{Dim: 8, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: P}
		if err := store.CreateCollection(ctx, coll, cfg); err != nil {
			t.Fatalf("CreateCollection: %v", err)
		}
		if p, _, ok := emb.Catalog().PartitionsGen(coll); !ok || p != P {
			t.Fatalf("PartitionsGen = (%d, ok=%v), want (%d, true)", p, ok, P)
		}

		// Insert n points, id-distinct vectors + a "v" payload field = id, spread
		// across the partitions by id-hash (fanInsert routes them).
		for i := 1; i <= n; i++ {
			vec := tieFreeVec(i)
			md := rostam.VectorMetadata{"v": vector.NewInt(int64(i))}
			if err := store.VectorInsertExt(ctx, coll, uint64(i), vec, rostam.VectorInsertOpts{Metadata: md}); err != nil {
				t.Fatalf("VectorInsert %d: %v", i, err)
			}
		}

		// Get EVERY point by id over TCP — each routes to its owning partition and
		// returns the correct vec + payload.
		for i := 1; i <= n; i++ {
			found, vec, meta, _, _, err := store.VectorGet(ctx, coll, uint64(i), true, true)
			if err != nil {
				t.Fatalf("VectorGet %d: %v", i, err)
			}
			if !found {
				t.Fatalf("VectorGet %d: not found (route-by-id missed the owning partition)", i)
			}
			want := tieFreeVec(i)
			if len(vec) != len(want) || vec[1] != want[1] {
				t.Fatalf("VectorGet %d: vec=%v, want %v", i, vec, want)
			}
			if meta["v"].Int != int64(i) {
				t.Fatalf("VectorGet %d: payload v=%d, want %d", i, meta["v"].Int, i)
			}
		}

		// Absent id -> not-found FLAG, not an error.
		found, _, _, _, _, err := store.VectorGet(ctx, coll, 9999, true, true)
		if err != nil {
			t.Fatalf("VectorGet absent: unexpected error %v", err)
		}
		if found {
			t.Fatal("VectorGet absent: found=true, want false")
		}

		// Pick a target id and exercise all 4 payload ops + cross-partition reindex.
		const tgt = 7
		// set (merge) adds tag="new".
		if applied, err := store.VectorSetPayload(ctx, coll, tgt, rostam.VectorMetadata{"tag": vector.NewString("new")}, nil); err != nil || !applied {
			t.Fatalf("VectorSetPayload: applied=%v err=%v", applied, err)
		}
		_, _, meta, _, _, _ := store.VectorGet(ctx, coll, tgt, false, true)
		if meta["v"].Int != tgt || meta["tag"].Str != "new" {
			t.Fatalf("after set: %+v, want v=%d,tag=new", meta, tgt)
		}
		// CROSS-PARTITION filtered search reflects the new field (reindex on the
		// owning physical partition). Spread the result across partitions by also
		// tagging a second id on a DIFFERENT partition.
		var other uint64
		for i := uint64(1); i <= n; i++ {
			if i != tgt && ops.PartitionOf(i, P) != ops.PartitionOf(tgt, P) {
				other = i
				break
			}
		}
		if other == 0 {
			t.Fatal("could not find a second id on a different partition")
		}
		if applied, err := store.VectorSetPayload(ctx, coll, other, rostam.VectorMetadata{"tag": vector.NewString("new")}, nil); err != nil || !applied {
			t.Fatalf("VectorSetPayload other: applied=%v err=%v", applied, err)
		}
		assertSpread("tag=new matches", []uint64{tgt, other})
		fr, _, err := store.VectorSearchExt(ctx, coll, tieFreeQuery(), n, rostam.VectorSearchOpts{
			Filter: rostam.VectorFilter{Op: vector.FilterEq, Field: "tag", Value: vector.NewString("new")},
		})
		if err != nil {
			t.Fatalf("filtered search post-update: %v", err)
		}
		gotMatch := map[uint64]bool{}
		for _, r := range fr {
			gotMatch[r.ID] = true
		}
		if len(fr) != 2 || !gotMatch[tgt] || !gotMatch[other] {
			t.Fatalf("filter tag=new = %v, want exactly {%d,%d} cross-partition (reindex per-partition)", ids(fr), tgt, other)
		}

		// overwrite -> only k=1 (drops v and tag).
		if _, err := store.VectorOverwritePayload(ctx, coll, tgt, rostam.VectorMetadata{"k": vector.NewInt(1)}, nil); err != nil {
			t.Fatalf("VectorOverwritePayload: %v", err)
		}
		_, _, meta, _, _, _ = store.VectorGet(ctx, coll, tgt, false, true)
		if _, ok := meta["v"]; ok {
			t.Fatalf("after overwrite: %+v, want only k=1 (v should be gone)", meta)
		}
		if meta["k"].Int != 1 {
			t.Fatalf("after overwrite: %+v, want k=1", meta)
		}
		// delete-keys removes k.
		if _, err := store.VectorDeletePayloadKeys(ctx, coll, tgt, []string{"k"}); err != nil {
			t.Fatalf("VectorDeletePayloadKeys: %v", err)
		}
		_, _, meta, _, _, _ = store.VectorGet(ctx, coll, tgt, false, true)
		if _, ok := meta["k"]; ok {
			t.Fatalf("after delete-keys: %+v, want no k", meta)
		}
		// clear -> empty.
		if _, err := store.VectorClearPayload(ctx, coll, tgt); err != nil {
			t.Fatalf("VectorClearPayload: %v", err)
		}
		_, _, meta, _, _, _ = store.VectorGet(ctx, coll, tgt, false, true)
		if len(meta) != 0 {
			t.Fatalf("after clear: %+v, want empty", meta)
		}
		// payload op on an absent point -> applied=false, no error.
		if applied, err := store.VectorSetPayload(ctx, coll, 9999, rostam.VectorMetadata{"x": vector.NewInt(1)}, nil); err != nil || applied {
			t.Fatalf("VectorSetPayload absent: applied=%v err=%v, want false/nil", applied, err)
		}
	})

	// ===================== NAMED =====================
	t.Run("named", func(t *testing.T) {
		const coll = "named_gp"
		cfg := map[string]rostam.NamedVectorParams{
			"title": {Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64},
			"image": {Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64},
		}
		if err := store.VectorNamedCreateCollection(ctx, coll, cfg, P); err != nil {
			t.Fatalf("VectorNamedCreateCollection: %v", err)
		}
		for i := 1; i <= n; i++ {
			vecs := map[string][]float32{"title": namedTitleVec(i), "image": namedImageVec(i, n)}
			payload := rostam.VectorMetadata{"n": vector.NewInt(int64(i))}
			if err := store.VectorNamedInsert(ctx, coll, uint64(i), vecs, payload, 0); err != nil {
				t.Fatalf("VectorNamedInsert %d: %v", i, err)
			}
		}
		// Get every point: both named vecs present + shared payload, from the owning partition.
		for i := 1; i <= n; i++ {
			found, vecs, payload, _, err := store.VectorNamedGet(ctx, coll, uint64(i), true, true)
			if err != nil {
				t.Fatalf("VectorNamedGet %d: %v", i, err)
			}
			if !found {
				t.Fatalf("VectorNamedGet %d: not found", i)
			}
			if len(vecs["title"]) != 8 || len(vecs["image"]) != 8 {
				t.Fatalf("VectorNamedGet %d: vecs=%+v, want both spaces dim 8", i, vecs)
			}
			if payload["n"].Int != int64(i) {
				t.Fatalf("VectorNamedGet %d: payload n=%d, want %d", i, payload["n"].Int, i)
			}
		}
		// Absent -> not found.
		if found, _, _, _, err := store.VectorNamedGet(ctx, coll, 9999, true, true); err != nil || found {
			t.Fatalf("VectorNamedGet absent: found=%v err=%v, want false/nil", found, err)
		}
		// All 4 payload ops on a target id, get reflects each.
		const tgt = 9
		if _, err := store.VectorNamedSetPayload(ctx, coll, tgt, rostam.VectorMetadata{"x": vector.NewInt(5)}, nil); err != nil {
			t.Fatalf("VectorNamedSetPayload: %v", err)
		}
		_, _, payload, _, _ := store.VectorNamedGet(ctx, coll, tgt, false, true)
		if payload["n"].Int != tgt || payload["x"].Int != 5 {
			t.Fatalf("named after set: %+v, want n=%d,x=5", payload, tgt)
		}
		if _, err := store.VectorNamedOverwritePayload(ctx, coll, tgt, rostam.VectorMetadata{"only": vector.NewInt(1)}, nil); err != nil {
			t.Fatalf("VectorNamedOverwritePayload: %v", err)
		}
		_, _, payload, _, _ = store.VectorNamedGet(ctx, coll, tgt, false, true)
		if _, ok := payload["n"]; ok || payload["only"].Int != 1 {
			t.Fatalf("named after overwrite: %+v, want only=1", payload)
		}
		if _, err := store.VectorNamedDeletePayloadKeys(ctx, coll, tgt, []string{"only"}); err != nil {
			t.Fatalf("VectorNamedDeletePayloadKeys: %v", err)
		}
		_, _, payload, _, _ = store.VectorNamedGet(ctx, coll, tgt, false, true)
		if _, ok := payload["only"]; ok {
			t.Fatalf("named after delete-keys: %+v, want no only", payload)
		}
		if _, err := store.VectorNamedClearPayload(ctx, coll, tgt); err != nil {
			t.Fatalf("VectorNamedClearPayload: %v", err)
		}
		_, _, payload, _, _ = store.VectorNamedGet(ctx, coll, tgt, false, true)
		if len(payload) != 0 {
			t.Fatalf("named after clear: %+v, want empty", payload)
		}
	})

	// ===================== MV =====================
	t.Run("mv", func(t *testing.T) {
		const coll = "mv_gp"
		if err := store.VectorMVCreateCollection(ctx, coll, rostam.MultiVectorConfig{
			Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Partitions: P,
		}); err != nil {
			t.Fatalf("VectorMVCreateCollection: %v", err)
		}
		for i := 1; i <= n; i++ {
			tokens := [][]float32{mvTokenAt(i), mvTokenAt(i + 1)}
			md := rostam.VectorMetadata{"doc": vector.NewInt(int64(i))}
			if err := store.VectorMVAdd(ctx, coll, uint64(i), tokens, md); err != nil {
				t.Fatalf("VectorMVAdd %d: %v", i, err)
			}
		}
		// Get every doc: token matrix + payload from the owning partition.
		for i := 1; i <= n; i++ {
			found, tokens, payload, err := store.VectorMVGet(ctx, coll, uint64(i), true, true)
			if err != nil {
				t.Fatalf("VectorMVGet %d: %v", i, err)
			}
			if !found {
				t.Fatalf("VectorMVGet %d: not found", i)
			}
			if len(tokens) != 2 {
				t.Fatalf("VectorMVGet %d: tokens=%+v, want 2", i, tokens)
			}
			if payload["doc"].Int != int64(i) {
				t.Fatalf("VectorMVGet %d: payload doc=%d, want %d", i, payload["doc"].Int, i)
			}
		}
		// Absent -> not found.
		if found, _, _, err := store.VectorMVGet(ctx, coll, 9999, true, true); err != nil || found {
			t.Fatalf("VectorMVGet absent: found=%v err=%v, want false/nil", found, err)
		}
		const tgt = 11
		if _, err := store.VectorMVSetPayload(ctx, coll, tgt, rostam.VectorMetadata{"x": vector.NewInt(8)}, nil); err != nil {
			t.Fatalf("VectorMVSetPayload: %v", err)
		}
		_, _, payload, _ := store.VectorMVGet(ctx, coll, tgt, false, true)
		if payload["doc"].Int != tgt || payload["x"].Int != 8 {
			t.Fatalf("mv after set: %+v, want doc=%d,x=8", payload, tgt)
		}
		if _, err := store.VectorMVOverwritePayload(ctx, coll, tgt, rostam.VectorMetadata{"only": vector.NewInt(1)}, nil); err != nil {
			t.Fatalf("VectorMVOverwritePayload: %v", err)
		}
		_, _, payload, _ = store.VectorMVGet(ctx, coll, tgt, false, true)
		if _, ok := payload["doc"]; ok || payload["only"].Int != 1 {
			t.Fatalf("mv after overwrite: %+v, want only=1", payload)
		}
		if _, err := store.VectorMVDeletePayloadKeys(ctx, coll, tgt, []string{"only"}); err != nil {
			t.Fatalf("VectorMVDeletePayloadKeys: %v", err)
		}
		_, _, payload, _ = store.VectorMVGet(ctx, coll, tgt, false, true)
		if _, ok := payload["only"]; ok {
			t.Fatalf("mv after delete-keys: %+v, want no only", payload)
		}
		if _, err := store.VectorMVClearPayload(ctx, coll, tgt); err != nil {
			t.Fatalf("VectorMVClearPayload: %v", err)
		}
		_, _, payload, _ = store.VectorMVGet(ctx, coll, tgt, false, true)
		if len(payload) != 0 {
			t.Fatalf("mv after clear: %+v, want empty", payload)
		}
		// absent payload op -> applied=false, no error.
		if applied, err := store.VectorMVSetPayload(ctx, coll, 9999, rostam.VectorMetadata{"x": vector.NewInt(1)}, nil); err != nil || applied {
			t.Fatalf("VectorMVSetPayload absent: applied=%v err=%v, want false/nil", applied, err)
		}
	})
}

// TestRemoteAliasSwapTCPClient proves the zero-downtime alias swap end-to-end
// over the REAL Go networked client (TCP wire) against a server whose dispatcher
// is the fanout decorator. It mirrors TestRemotePartitionedTCPClient's boot
// (same server.New + rostam.NewClient + fanoutDispatcher pair) and drives the full
// alias lifecycle over the wire against PARTITIONED (P=4) collections:
//
//   - an alias resolves to a partitioned collection and the read fans out across
//     all its physical partitions (NOT one partition's slice, NOT empty);
//   - a write THROUGH the alias lands in the aliased collection (verified via the
//     real collection name) and NOT in the other collection;
//   - an ATOMIC two-action AliasBatch (delete prod→v1, then create prod→v2) flips
//     the alias under a SINGLE meta-FSM lock (OpSetAliasBatch applies the whole
//     batch atomically — see cluster/meta_fsm.go), so a single synchronous
//     read-your-writes read after the swap is FULLY v2 — never a v1+v2 mix, never
//     empty. That is the zero-downtime proof: the swap presents v1-then-v2 with no
//     undefined window.
//
// v1 (ids 0..99) and v2 (ids 1000..1099) seed DISJOINT id ranges, so the result
// IDs of a search through the alias unambiguously reveal WHICH collection
// answered — the load-bearing signal for "the alias resolved to the right
// partitioned collection and fanned out".
func TestRemoteAliasSwapTCPClient(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}

	embStore, err := rostam.NewEmbedded(rostam.EmbeddedConfig{
		NodeID:    "n1",
		DataDir:   t.TempDir(),
		NumShards: 1,
		Bootstrap: true,
		Ops:       reg,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = embStore.Close() })
	waitLeaderEmbedded(t, embStore)
	emb := embStore.(*rostam.Embedded)

	disp := rostam.NewFanoutDispatcher(emb, emb.Node())
	srv, err := server.New(server.Config{Addr: "127.0.0.1:0", Dispatcher: disp})
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()
	t.Cleanup(func() { _ = srv.Close() })
	addr := srv.Addr().String()

	store, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{addr}, Ops: reg})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	ctx := context.Background()
	const k = 10
	cfg := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: 4}

	// Two partitioned (P=4) collections with DISJOINT id ranges so the answering
	// collection is identifiable purely from the returned ids.
	for _, coll := range []string{"coll_v1", "coll_v2"} {
		if err := store.CreateCollection(ctx, coll, cfg); err != nil {
			t.Fatalf("client CreateCollection %s: %v", coll, err)
		}
		if p, _, ok := emb.Catalog().PartitionsGen(coll); !ok || p != 4 {
			t.Fatalf("after create %s: PartitionsGen = (%d, ok=%v), want (4, true)", coll, p, ok)
		}
	}
	for id := 0; id < 100; id++ {
		if err := store.VectorInsert(ctx, "coll_v1", uint64(id), tieFreeVec(id)); err != nil {
			t.Fatalf("seed coll_v1 %d: %v", id, err)
		}
	}
	for id := 1000; id < 1100; id++ {
		if err := store.VectorInsert(ctx, "coll_v2", uint64(id), tieFreeVec(id)); err != nil {
			t.Fatalf("seed coll_v2 %d: %v", id, err)
		}
	}

	q := tieFreeQuery()
	inV1 := func(id uint64) bool { return id < 100 }
	inV2 := func(id uint64) bool { return id >= 1000 && id < 1100 }

	// 4. Create alias prod -> coll_v1.
	if err := store.CreateAlias(ctx, "prod", "coll_v1"); err != nil {
		t.Fatalf("CreateAlias prod->coll_v1: %v", err)
	}

	// 5. Search via the alias: every id must be in v1's range (alias resolved to
	// the partitioned v1 AND fanned out across its partitions — an unresolved or
	// single-partition read would be empty/wrong, LANDMINE #1).
	viaAlias, err := store.VectorSearch(ctx, "prod", q, k)
	if err != nil {
		t.Fatalf("VectorSearch via alias prod: %v", err)
	}
	if len(viaAlias) != k {
		t.Fatalf("alias search returned %d results, want %d (alias did not fan out across v1's partitions)", len(viaAlias), k)
	}
	for i, r := range viaAlias {
		if !inV1(r.ID) {
			t.Fatalf("alias search rank %d: id %d not in v1 range 0..99 (alias resolved wrong / did not fan out): %v",
				i, r.ID, resultIDs(viaAlias))
		}
	}
	// Cross-check: alias read equals the real-name read, same ids in same order.
	viaReal, err := store.VectorSearch(ctx, "coll_v1", q, k)
	if err != nil {
		t.Fatalf("VectorSearch coll_v1 direct: %v", err)
	}
	if !reflect.DeepEqual(resultIDs(viaAlias), resultIDs(viaReal)) {
		t.Fatalf("alias read != direct read:\n alias=%v\n real =%v", resultIDs(viaAlias), resultIDs(viaReal))
	}

	// 6. Write-through via the alias: insert a NEW id 12345 through "prod" and
	// confirm it lands in coll_v1, findable via the REAL name, and NOT in coll_v2.
	// We verify with the fan-out VectorSearch (not VectorExists): point ops like
	// vector_exists have no fan-out case for a partitioned LOGICAL name and would
	// hit the empty logical collection — search fans out across the physicals, so
	// it is the correct probe for "did the write land in a partition of v1". The
	// query is the written id's own (tie-free, distinct) vector, so a hit appears
	// as the nearest neighbor.
	const writeID = 12345
	if err := store.VectorInsert(ctx, "prod", writeID, tieFreeVec(writeID)); err != nil {
		t.Fatalf("VectorInsert via alias prod: %v", err)
	}
	contains := func(rs []rostam.VectorResult, id uint64) bool {
		for _, r := range rs {
			if r.ID == id {
				return true
			}
		}
		return false
	}
	wq := tieFreeVec(writeID)
	if v1hit, err := store.VectorSearch(ctx, "coll_v1", wq, k); err != nil {
		t.Fatalf("VectorSearch coll_v1 for write-through: %v", err)
	} else if !contains(v1hit, writeID) {
		t.Fatalf("write-through id %d not found in coll_v1 via fan-out (alias write did not land in v1): %v", writeID, resultIDs(v1hit))
	}
	if v2hit, err := store.VectorSearch(ctx, "coll_v2", wq, k); err != nil {
		t.Fatalf("VectorSearch coll_v2 for write-through: %v", err)
	} else if contains(v2hit, writeID) {
		t.Fatalf("write-through id %d leaked into coll_v2 (alias resolved to the wrong collection): %v", writeID, resultIDs(v2hit))
	}

	// 7. ATOMIC swap: ONE AliasBatch call carrying both actions — delete prod->v1
	// then create prod->v2. OpSetAliasBatch applies the whole batch under a single
	// meta-FSM lock (cluster/meta_fsm.go), so the swap is indivisible.
	if err := store.AliasBatch(ctx, []rostam.AliasAction{
		{Alias: "prod", Delete: true},
		{Alias: "prod", Canonical: "coll_v2"},
	}); err != nil {
		t.Fatalf("AliasBatch atomic swap: %v", err)
	}

	// 8 + 9. Zero-downtime proof: AliasBatch is synchronous read-your-writes and
	// the batch is applied atomically under one FSM lock, so this SINGLE post-swap
	// read suffices — it must be FULLY v2 (every id in 1000..1099), never a v1+v2
	// mix, never empty. A mix or an empty result would mean the swap exposed an
	// undefined window (no zero-downtime guarantee).
	postSwap, err := store.VectorSearch(ctx, "prod", q, k)
	if err != nil {
		t.Fatalf("VectorSearch via alias prod after swap: %v", err)
	}
	if len(postSwap) != k {
		t.Fatalf("post-swap alias search returned %d results, want %d (swap exposed an empty/undefined window)", len(postSwap), k)
	}
	for i, r := range postSwap {
		if !inV2(r.ID) {
			t.Fatalf("post-swap rank %d: id %d not in v2 range 1000..1099 (atomic swap left a v1+v2 mix): %v",
				i, r.ID, resultIDs(postSwap))
		}
	}
	postReal, err := store.VectorSearch(ctx, "coll_v2", q, k)
	if err != nil {
		t.Fatalf("VectorSearch coll_v2 direct: %v", err)
	}
	if !reflect.DeepEqual(resultIDs(postSwap), resultIDs(postReal)) {
		t.Fatalf("post-swap alias read != coll_v2 direct read:\n alias=%v\n real =%v", resultIDs(postSwap), resultIDs(postReal))
	}

	// 10. Cleanup: delete the alias, then ListAliases is empty.
	if err := store.DeleteAlias(ctx, "prod"); err != nil {
		t.Fatalf("DeleteAlias prod: %v", err)
	}
	aliases, err := store.ListAliases(ctx, "")
	if err != nil {
		t.Fatalf("ListAliases: %v", err)
	}
	if len(aliases) != 0 {
		t.Fatalf("after DeleteAlias: ListAliases = %v, want empty", aliases)
	}
}

// TestRemoteWriteConsistencyTCPClient is the end-to-end proof of the write-
// consistency barrier over the REAL Go networked TCP client against a MULTI-node
// (3-node, RF=3) cluster — the strongest Step-2 form (chosen over the single-node
// RF=1 plumbing-only variant because the in-process multi-node harness already
// stands up a full server.New per node, so a real rostam.NewClient pointed at all three
// node addresses exercises the __wc__ envelope -> server fanoutDispatcher ->
// BarrierForShard path over the wire, AND a genuine >majority barrier).
//
// Harness: newInmemEmbeddedClusterServers gives the per-node *rostam.Server handles; the
// client is configured with EVERY node's TCP address so it can follow the server's
// NotLeader hints to whichever node leads the target shard (the TCP dispatcher is
// the bare-node-wrapping fanout decorator, which — like the embedded path — does
// not leader-forward a follower-hosted write; the client's own NotLeader-hop logic
// closes that gap over the wire). The __wc__ envelope is built by the client only
// when rostam.WriteOpts are active (wcWire), so the WCF rides the wire as the envelope op.
//
// Ground truth is INDEPENDENT of the client: after the client write returns, the
// point's read-visibility is checked on each node's embedded rostam.Store directly
// (stores[j].VectorGet — a LOCAL OpReadOnly read from node j's own replica).
//
// Two proofs:
//   - WCF=RF (=3) over the wire blocks until ALL 3 owners applied ⇒ the point is
//     visible on a NON-leader owner the instant the client call returns (the
//     barrier engaged across the network).
//   - wait=false returns at majority (skips the >majority barrier) ⇒ the call
//     succeeds promptly; the leader has it immediately, followers eventually.
func TestRemoteWriteConsistencyTCPClient(t *testing.T) {
	const (
		n         = 3
		numShards = 4
		rf        = 3
	)
	stores, servers := newInmemEmbeddedClusterServers(t, n, numShards, rf)
	ctx := context.Background()

	// Every node already runs a TCP server (rostam.NewServer{TCPAddr}); collect the addrs
	// so the client can follow NotLeader hops to the inner write's leader.
	addrs := make([]string, 0, n)
	for _, srv := range servers {
		if a := srv.TCPAddr(); a != "" {
			addrs = append(addrs, a)
		}
	}
	if len(addrs) != n {
		t.Fatalf("want %d TCP server addrs, got %d (%v)", n, len(addrs), addrs)
	}

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	client, err := rostam.NewClient(rostam.ClientConfig{Servers: addrs, Ops: reg})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })

	const coll = "wc"
	routeKey := []byte(ops.CanonicalName(coll))

	// Create the unpartitioned collection over the wire. CreateCollection routes to
	// the target shard's leader (the client follows NotLeader), so it succeeds from
	// any of the three addrs. Retry through the residual election window.
	cfg := rostam.VectorConfig{Dim: 4, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64}
	retryUntil(t, "client CreateCollection", func() error {
		return client.CreateCollection(ctx, coll, cfg)
	})
	// Wait until every node's catalog has the collection (so a local read on a
	// follower routes to its replica, not a not-found). Unpartitioned collections
	// report ok=false from PartitionsGen, so gate on vector_get_config readability.
	for i, s := range stores {
		emb := s.(*rostam.Embedded)
		deadline := time.Now().Add(10 * time.Second)
		for time.Now().Before(deadline) {
			if _, gerr := emb.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(coll)); gerr == nil {
				break
			}
			time.Sleep(20 * time.Millisecond)
		}
		if _, gerr := emb.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(coll)); gerr != nil {
			t.Fatalf("node %d never saw collection %q config: %v", i, coll, gerr)
		}
	}

	// localFound: does node idx have point id applied in its OWN replica?
	localFound := func(idx int, id uint64) bool {
		ok, _, _, _, _, err := stores[idx].(*rostam.Embedded).VectorGet(ctx, coll, id, false, false)
		return err == nil && ok
	}
	leaderIdx := func() int {
		for i, s := range stores {
			if s.IsLeader(routeKey) {
				return i
			}
		}
		return -1
	}

	// ---- WCF=RF over the wire: visible on a NON-leader owner immediately ----
	// The barrier (eff=3 > majority=2) blocks until all 3 owners applied, so the
	// instant the client call returns, EVERY node — including a follower — has the
	// point in its local replica.
	const idA = uint64(1)
	retryUntil(t, "client WCF=3 insert", func() error {
		return client.VectorInsert(ctx, coll, idA, []float32{1, 0, 0, 0}, rostam.WriteOpts{WriteConsistencyFactor: 3})
	})
	li := leaderIdx()
	if li < 0 {
		t.Fatal("no leader visible after WCF=3 insert")
	}
	foundFollower := false
	for j := range stores {
		if j == li {
			continue
		}
		if localFound(j, idA) {
			foundFollower = true
		} else {
			t.Errorf("WCF=3 over TCP: follower node %d does NOT have point %d immediately (barrier did not wait for it over the wire)", j, idA)
		}
	}
	if !foundFollower {
		t.Fatal("WCF=3 over TCP: no follower owner had the point — barrier proof inconclusive")
	}
	t.Log("WCF=3 over the TCP client: point visible on every follower owner immediately (cross-network barrier engaged)")

	// ---- wait=false returns at majority (barrier skipped) ----
	const idB = uint64(2)
	// Time ONLY the successful attempt: retryUntil may burn seconds on rostam.ErrNotLeader
	// while the leader settles, which is unrelated to whether the barrier engaged.
	var elB time.Duration
	retryUntil(t, "client wait=false insert", func() error {
		w := false
		start := time.Now()
		err := client.VectorInsert(ctx, coll, idB, []float32{0, 1, 0, 0}, rostam.WriteOpts{WriteConsistencyFactor: 3, Wait: &w})
		if err == nil {
			elB = time.Since(start)
		}
		return err
	})
	if elB > 3*time.Second {
		t.Errorf("wait=false over TCP took %s — barrier was not skipped (should return at majority)", elB)
	}
	// The leader (which applied at commit) has it immediately; the cluster as a
	// whole converges. Gate the all-owners-applied state (eventual, no fixed sleep).
	li = leaderIdx()
	if li < 0 || !localFound(li, idB) {
		t.Fatalf("wait=false over TCP: point %d not visible on the leader immediately", idB)
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		all := true
		for j := range stores {
			if !localFound(j, idB) {
				all = false
				break
			}
		}
		if all {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	for j := range stores {
		if !localFound(j, idB) {
			t.Errorf("wait=false: owner %d never applied point %d within budget", j, idB)
		}
	}
	t.Log("wait=false over the TCP client: returned at majority (barrier skipped), owners converged eventually")
}

// TestRemoteScrollCursorTCPClient drives cursor deep-pagination over the REAL Go
// networked client (TCP wire) against a server whose dispatcher is the fanout
// decorator, on a PARTITIONED (P=4) collection. It proves the cursor crosses the
// wire and the SERVER-SIDE fan-out merge returns every id exactly once, globally
// ascending — observed entirely through the TCP client.
//
// Harness choice: a single embedded node behind one server.New + rostam.NewClient (the
// established *TCPClient pattern in this file), NOT a multi-node TCP cluster. The
// scroll fan-out + cursor merge is SERVER-SIDE on the receiving node regardless
// of node count, so a P=4 collection over one node fully exercises the
// cursor-over-the-wire + global merge. The 3-node multi-coordinator merge is
// covered by TestClusterScrollCursorDeepPagination (in-process cluster), so this
// keeps the wire test fast and focused on the codec + server merge path.
//
// INDEPENDENT GROUND TRUTH: the paged union is compared against a single unpaged
// scroll (limit=0 ⇒ whole collection) sorted by id — they must be identical sets.
// A malformed cursor over the client surfaces ops.ErrBadScrollCursor (the client
// decodes the cursor before the wire, failing loud).
func TestRemoteScrollCursorTCPClient(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}

	embStore, err := rostam.NewEmbedded(rostam.EmbeddedConfig{
		NodeID:    "n1",
		DataDir:   t.TempDir(),
		NumShards: 1,
		Bootstrap: true,
		Ops:       reg,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = embStore.Close() })
	waitLeaderEmbedded(t, embStore)
	emb := embStore.(*rostam.Embedded)

	// The server's dispatcher is the fanout decorator — every op the client sends
	// over TCP is dispatched through it, exactly like server.go's CLUSTER branch.
	disp := rostam.NewFanoutDispatcher(emb, emb.Node())
	srv, err := server.New(server.Config{Addr: "127.0.0.1:0", Dispatcher: disp})
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()
	t.Cleanup(func() { _ = srv.Close() })

	store, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{srv.Addr().String()}, Ops: reg})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	ctx := context.Background()
	const (
		coll = "scrollp"
		P    = 4
		N    = 250
		L    = 30
	)

	// Create a P=4 collection through the networked client.
	cfg := rostam.VectorConfig{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: P}
	if err := store.CreateCollection(ctx, coll, cfg); err != nil {
		t.Fatalf("client CreateCollection: %v", err)
	}
	if p, _, ok := emb.Catalog().PartitionsGen(coll); !ok || p != P {
		t.Fatalf("after client create: PartitionsGen = (%d, ok=%v), want (%d, true)", p, ok, P)
	}

	// Seed N tie-free points through the client (ids 0..N-1 spread across
	// partitions by the server-side router).
	for i := 0; i < N; i++ {
		if err := store.VectorInsert(ctx, coll, uint64(i), tieFreeVec(i)); err != nil {
			t.Fatalf("client VectorInsert %d: %v", i, err)
		}
	}

	// Deep-page over TCP following next_cursor from "" until exhausted.
	var paged []uint64
	cursor := ""
	pages := 0
	var last uint64
	have := false
	for {
		docs, _, next, err := store.VectorScroll(ctx, coll, rostam.VectorFilter{}, L, rostam.VectorScrollOpts{Cursor: cursor})
		if err != nil {
			t.Fatalf("client VectorScroll page %d: %v", pages, err)
		}
		pages++
		for i, d := range docs {
			if i > 0 && d.ID <= docs[i-1].ID {
				t.Fatalf("page %d not strictly ascending at %d: %d <= %d", pages, i, d.ID, docs[i-1].ID)
			}
			if have && d.ID <= last {
				t.Fatalf("page %d id %d not > previous page's last %d (gap/dup/order bug over wire)", pages, d.ID, last)
			}
			paged = append(paged, d.ID)
			last, have = d.ID, true
		}
		// Exhaustion rule over the wire: a full page must carry a next_cursor; a
		// short/empty page must end pagination.
		if len(docs) == L {
			if next == "" {
				t.Fatalf("page %d full (len=%d) but next_cursor empty over wire", pages, L)
			}
		} else if next != "" {
			t.Fatalf("page %d short (len=%d<%d) but next_cursor=%q (not exhausted)", pages, len(docs), L, next)
		}
		if next == "" {
			break
		}
		cursor = next
		if pages > N+10 {
			t.Fatalf("pagination did not terminate after %d pages", pages)
		}
	}

	// Every id exactly once, globally ascending, total==N.
	want := map[uint64]bool{}
	for i := 0; i < N; i++ {
		want[uint64(i)] = true
	}
	assertExactlyOnceAscending(t, paged, want)

	// INDEPENDENT GROUND TRUTH: a single unpaged scroll (limit=0 ⇒ whole
	// collection) over the wire, sorted by id, must equal the paged union set.
	full, _, fullNext, err := store.VectorScroll(ctx, coll, rostam.VectorFilter{}, 0, rostam.VectorScrollOpts{})
	if err != nil {
		t.Fatalf("client full scroll: %v", err)
	}
	if fullNext != "" {
		t.Fatalf("unpaged (limit=0) scroll over wire returned next_cursor=%q, want empty", fullNext)
	}
	if len(full) != N {
		t.Fatalf("unpaged scroll returned %d docs over wire, want %d", len(full), N)
	}
	fullIDs := make([]uint64, len(full))
	for i, d := range full {
		fullIDs[i] = d.ID
	}
	sort.Slice(fullIDs, func(i, j int) bool { return fullIDs[i] < fullIDs[j] })
	if len(fullIDs) != len(paged) {
		t.Fatalf("paged union (%d) != unpaged set (%d)", len(paged), len(fullIDs))
	}
	for i := range fullIDs {
		// paged is already globally ascending (asserted above); the unpaged set is
		// sorted here, so an element-wise equality proves identical sets.
		if fullIDs[i] != paged[i] {
			t.Fatalf("paged-vs-unpaged mismatch at %d: paged %d, unpaged %d", i, paged[i], fullIDs[i])
		}
	}

	// Malformed cursor over the client surfaces ops.ErrBadScrollCursor (decoded
	// client-side before the wire — fail loud, not a silent first-page).
	if _, _, _, err := store.VectorScroll(ctx, coll, rostam.VectorFilter{}, L,
		rostam.VectorScrollOpts{Cursor: "!!!notbase64"}); !errors.Is(err, ops.ErrBadScrollCursor) {
		t.Fatalf("malformed cursor over TCP client: err = %v, want ops.ErrBadScrollCursor", err)
	}
}

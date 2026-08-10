// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/httpapi"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/server"
	"github.com/rostamlabs/rostam/vector"
)

// These tests prove the FULL multi-vector (late-interaction / MaxSim) path
// through REAL transports (HTTP + the Go networked TCP client) against a
// PARTITIONED (P=4) MV collection. The critical proof is fan-out PARITY: a
// remote MV search of a P>1 collection must return the globally-merged top-k
// (merged across all physical partitions), equal by ID and order to the
// embedded fan-out baseline. A single physical partition holds only ~10 of the
// 40 docs, so a non-fanned (single-partition) result would differ.
//
// Both tests drive the server-side fanoutDispatcher over a real transport
// codec, mirroring the dense remote tests in remote_partition_integration_test.go:
// the HTTP test mounts httpapi.Handler(rostam.NewFanoutDispatcher(emb, emb.Node()), ...)
// and the TCP test stands up server.New + rostam.NewClient with the same decorator as
// the server's dispatcher (client -> TCP -> server -> decorator).

// mvResultIDs extracts document IDs from an MV result slice (for failure
// messages).
func mvResultIDs(res []vector.MultiResult) []uint64 {
	ids := make([]uint64, len(res))
	for i, r := range res {
		ids[i] = r.ID
	}
	return ids
}

// TestRemoteMVPartitionedHTTPLifecycle drives the full MV create -> add ->
// search -> drop lifecycle over the REAL HTTP transport (httptest) backed by
// the fanout dispatcher, and asserts the MV search top-k matches the embedded
// fan-out baseline (the global-merge proof).
func TestRemoteMVPartitionedHTTPLifecycle(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)
	ctx := context.Background()

	const coll = "mvp"
	const n = 40
	const k = 10
	const winner = 17 // PartitionOf(17,4)=3 (non-zero) -> proves cross-partition merge

	// Mount the REAL HTTP handler over the decorator (the same wrap server.go
	// installs in the CLUSTER branch).
	h := httpapi.Handler(rostam.NewFanoutDispatcher(emb, emb.Node()), httpapi.Options{})
	ts := httptest.NewServer(h)
	t.Cleanup(ts.Close)

	httpJSON := func(method, path, body string, out any) *http.Response {
		t.Helper()
		var req *http.Request
		var err error
		if body != "" {
			req, err = http.NewRequest(method, ts.URL+path, strings.NewReader(body))
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

	// 1. Create a P=4 MV collection via POST /v1/multivector/{name}.
	createBody := `{"dim":4,"partitions":4}`
	resp := httpJSON("POST", "/v1/multivector/"+coll, createBody, nil)
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("MV create status = %d, want 201", resp.StatusCode)
	}
	// Confirm the create ran through the embedded backend (physical partitions +
	// catalog with P=4) rather than passing through as a single-partition create.
	if p, _, ok := emb.Catalog().PartitionsGen(coll); !ok || p != 4 {
		t.Fatalf("after HTTP MV create: PartitionsGen = (%d, ok=%v), want (4, true)", p, ok)
	}

	// 2. Add ~40 tie-free docs (one token each, distinct angles) via the MV docs
	// endpoint. docIDs hash to different physical partitions, so a correct global
	// top-10 must merge across partitions.
	for i := 0; i < n; i++ {
		tok := mvTokenAt(i)
		body := fmt.Sprintf(`{"id":%d,"tokens":[[%g,%g,%g,%g]]}`, i, tok[0], tok[1], tok[2], tok[3])
		resp := httpJSON("POST", "/v1/multivector/"+coll+"/docs", body, nil)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("MV add %d status = %d, want 200", i, resp.StatusCode)
		}
	}

	// rostam.Embedded fan-out baseline for parity (same data, computed directly on the
	// node via the partition-aware embedded MVSearch).
	query := [][]float32{mvTokenAt(winner)}
	want, _, err := emb.VectorMVSearch(ctx, coll, query, k, rostam.MultiSearchOpts{CandidatesPerToken: 100})
	if err != nil {
		t.Fatalf("embedded MV baseline search: %v", err)
	}
	if len(want) != k {
		t.Fatalf("embedded MV baseline returned %d, want %d", len(want), k)
	}
	if want[0].ID != winner {
		t.Fatalf("embedded MV baseline winner = %d, want %d", want[0].ID, winner)
	}

	// 3. MV search via the MV search endpoint for k=10.
	q := query[0]
	var sres struct {
		Results []vector.MultiResult `json:"results"`
	}
	searchBody := fmt.Sprintf(`{"query":[[%g,%g,%g,%g]],"k":%d,"candidates_per_token":100}`,
		q[0], q[1], q[2], q[3], k)
	resp = httpJSON("POST", "/v1/multivector/"+coll+"/search", searchBody, &sres)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("MV search status = %d, want 200", resp.StatusCode)
	}

	// Exactly k results: a single partition (~10 of 40 docs) could not hold a
	// correct global top-10.
	if len(sres.Results) != k {
		t.Fatalf("HTTP MV search returned %d results, want %d (single-partition slice would differ)", len(sres.Results), k)
	}
	// Ordered by MaxSim score descending (tie-free → strictly decreasing).
	for i := 1; i < len(sres.Results); i++ {
		if sres.Results[i].Score > sres.Results[i-1].Score {
			t.Fatalf("HTTP MV results not score-descending at %d: %g > %g",
				i, sres.Results[i].Score, sres.Results[i-1].Score)
		}
	}
	// Parity: the HTTP MV top-k IDs equal the embedded baseline IDs in order. This
	// is the global-merge proof — only a cross-partition merge yields this list.
	for i := range want {
		if sres.Results[i].ID != want[i].ID {
			t.Fatalf("HTTP MV search rank %d: got id %d, want %d (global-merge mismatch)\n got=%v\nwant=%v",
				i, sres.Results[i].ID, want[i].ID, mvResultIDs(sres.Results), mvResultIDs(want))
		}
	}

	// 4. Drop via DELETE /v1/multivector/{name}; assert each physical MV
	// partition is gone.
	resp = httpJSON("DELETE", "/v1/multivector/"+coll, "", nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("MV drop status = %d, want 200", resp.StatusCode)
	}
	for p := 0; p < 4; p++ {
		phys := string(ops.PartitionKeyGen(coll, 0, p))
		if _, err := emb.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(phys)); err == nil {
			t.Fatalf("after HTTP MV drop: physical partition %d (%s) still exists", p, phys)
		}
	}
	fan := rostam.NewFanoutDispatcher(emb, emb.Node())
	if _, _, ok := fan.Partitioned(coll); ok {
		t.Fatal("after HTTP MV drop: collection still reports as partitioned")
	}
}

// TestRemoteMVPartitionedTCPClient drives the full MV lifecycle over the REAL Go
// networked client (TCP wire) against a server whose dispatcher is the fanout
// decorator. This proves the Go networked client can use partitioned MV
// collections: a remote MV search of a P=4 collection returns the correct
// global top-k (not one partition's slice).
func TestRemoteMVPartitionedTCPClient(t *testing.T) {
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
	const coll = "mvp"
	const n = 40
	const k = 10
	const winner = 17 // PartitionOf(17,4)=3 (non-zero) -> proves cross-partition merge

	// Create a P=4 MV collection through the networked client. fanCreateCollection
	// routes the partitioned MV create through the embedded backend.
	cfg := rostam.MultiVectorConfig{Dim: 4, Partitions: 4}
	if err := store.VectorMVCreateCollection(ctx, coll, cfg); err != nil {
		t.Fatalf("client VectorMVCreateCollection: %v", err)
	}
	if p, _, ok := emb.Catalog().PartitionsGen(coll); !ok || p != 4 {
		t.Fatalf("after client MV create: PartitionsGen = (%d, ok=%v), want (4, true)", p, ok)
	}

	// Add ~40 tie-free docs through the client (routed to the right physical
	// partition by the server-side fan-out).
	for i := 0; i < n; i++ {
		if err := store.VectorMVAdd(ctx, coll, uint64(i), [][]float32{mvTokenAt(i)}, nil); err != nil {
			t.Fatalf("client VectorMVAdd %d: %v", i, err)
		}
	}

	// rostam.Embedded fan-out baseline (same node, direct fan-out) for parity.
	query := [][]float32{mvTokenAt(winner)}
	want, _, err := emb.VectorMVSearch(ctx, coll, query, k, rostam.MultiSearchOpts{CandidatesPerToken: 100})
	if err != nil {
		t.Fatalf("embedded MV baseline search: %v", err)
	}
	if len(want) != k {
		t.Fatalf("embedded MV baseline returned %d, want %d", len(want), k)
	}
	if want[0].ID != winner {
		t.Fatalf("embedded MV baseline winner = %d, want %d", want[0].ID, winner)
	}

	// Search k=10 over TCP. The non-negotiable assertion: the remote MV result is
	// the global top-k merged across all 4 partitions, equal to the baseline in
	// order.
	got, _, err := store.VectorMVSearch(ctx, coll, query, k, rostam.MultiSearchOpts{CandidatesPerToken: 100})
	if err != nil {
		t.Fatalf("client VectorMVSearch: %v", err)
	}
	if len(got) != k {
		t.Fatalf("client MV search returned %d results, want %d (single-partition slice would differ)", len(got), k)
	}
	for i := 1; i < len(got); i++ {
		if got[i].Score > got[i-1].Score {
			t.Fatalf("client MV results not score-descending at %d: %g > %g", i, got[i].Score, got[i-1].Score)
		}
	}
	for i := range want {
		if got[i].ID != want[i].ID {
			t.Fatalf("client MV search rank %d: got id %d, want %d (global-merge mismatch)\n got=%v\nwant=%v",
				i, got[i].ID, want[i].ID, mvResultIDs(got), mvResultIDs(want))
		}
	}

	// Drop through the client; confirm physical cleanup.
	if err := store.VectorMVDropCollection(ctx, coll); err != nil {
		t.Fatalf("client VectorMVDropCollection: %v", err)
	}
	for p := 0; p < 4; p++ {
		phys := string(ops.PartitionKeyGen(coll, 0, p))
		if _, err := emb.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(phys)); err == nil {
			t.Fatalf("after client MV drop: physical partition %d (%s) still exists", p, phys)
		}
	}
	if _, _, ok := disp.Partitioned(coll); ok {
		t.Fatal("after client MV drop: collection still reports as partitioned")
	}
}

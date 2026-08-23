// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/grpcapi"
	"github.com/rostamlabs/rostam/sdk/pb"
	"github.com/rostamlabs/rostam/httpapi"
)

// TestRemoteWriteConsistencyHTTP proves the TRANSPORT plumbing for tunable write
// consistency over the REAL HTTP transport (httptest) backed by the embedded
// engine + fanout decorator — the same wrap server.go installs. It exercises the
// full parse → __wc__ envelope → dispatch → post-commit barrier path end-to-end
// at RF=1, where the barrier is a documented no-op (majority==RF==1): the write
// MUST still succeed and the point MUST be findable. This is the Task-4 contract
// (real multi-node barrier blocking + the 504 timeout are Task-5's cluster e2e).
//
// Cases:
//  1. upsert with write_consistency_factor:5 → 200 at RF=1 (barrier no-op,
//     factor > RF clamps) and the point is retrievable by id;
//  2. upsert with wait:false → 200 (skip-barrier path also dispatches+commits);
//  3. a DEFAULT write (no WC fields in the body) → 200 (backward-compat: the
//     plain op is dispatched, NO envelope).
func TestRemoteWriteConsistencyHTTP(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)

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

	// A partitioned (P=4) collection so the __wc__ handler resolves a real
	// per-partition shard for the point (id → PartitionOf(id,P)).
	const coll = "wc_docs"
	createBody := `{"name":"` + coll + `","config":{"dim":4,"metric":"cosine","m":16,"ef_construction":200,"ef_search":64,"partitions":4}}`
	if resp := httpJSON("POST", "/v1/collections", createBody, nil); resp.StatusCode != http.StatusCreated {
		t.Fatalf("create %s = %d, want 201", coll, resp.StatusCode)
	}

	// findByID confirms a point is retrievable by id (read-visibility proof that
	// the write committed and the barrier path returned success).
	findByID := func(id int) bool {
		t.Helper()
		var got struct {
			Found bool `json:"found"`
		}
		resp := httpJSON("GET", "/v1/collections/"+coll+"/points/"+strconv.Itoa(id), "", &got)
		return resp.StatusCode == http.StatusOK && got.Found
	}

	// 1. WCF=5 at RF=1: barrier no-op (clamped to RF=1 ≤ majority=1), still 200.
	wcfBody := `{"upsert":true,"id":1,"vector":[1,0,0,0],"write_consistency_factor":5}`
	if resp := httpJSON("POST", "/v1/collections/"+coll+"/points", wcfBody, nil); resp.StatusCode != http.StatusOK {
		t.Fatalf("upsert WCF=5 = %d, want 200 (barrier no-op at RF=1)", resp.StatusCode)
	}
	if !findByID(1) {
		t.Fatalf("WCF=5 point id=1 not found after write (envelope→dispatch→barrier must commit)")
	}

	// 2. wait:false: the skip-barrier path also dispatches + commits.
	noWaitBody := `{"upsert":true,"id":2,"vector":[0,1,0,0],"wait":false}`
	if resp := httpJSON("POST", "/v1/collections/"+coll+"/points", noWaitBody, nil); resp.StatusCode != http.StatusOK {
		t.Fatalf("upsert wait=false = %d, want 200", resp.StatusCode)
	}
	if !findByID(2) {
		t.Fatalf("wait=false point id=2 not found after write")
	}

	// 2b. WCF AND wait:false together → still 200 + findable.
	bothBody := `{"upsert":true,"id":3,"vector":[0,0,1,0],"write_consistency_factor":3,"wait":false}`
	if resp := httpJSON("POST", "/v1/collections/"+coll+"/points", bothBody, nil); resp.StatusCode != http.StatusOK {
		t.Fatalf("upsert WCF+wait=false = %d, want 200", resp.StatusCode)
	}
	if !findByID(3) {
		t.Fatalf("WCF+wait=false point id=3 not found after write")
	}

	// 3. DEFAULT write (no WC fields) → 200 (backward-compat, plain-op path).
	defBody := `{"upsert":true,"id":4,"vector":[0,0,0,1]}`
	if resp := httpJSON("POST", "/v1/collections/"+coll+"/points", defBody, nil); resp.StatusCode != http.StatusOK {
		t.Fatalf("default upsert (no WC fields) = %d, want 200", resp.StatusCode)
	}
	if !findByID(4) {
		t.Fatalf("default point id=4 not found after write")
	}

	// 4. A delete-by-id carrying WC via query string also commits at RF=1.
	if resp := httpJSON("DELETE", "/v1/collections/"+coll+"/points/4?write_consistency_factor=2", "", nil); resp.StatusCode != http.StatusOK {
		t.Fatalf("delete WCF (query) = %d, want 200", resp.StatusCode)
	}
	if findByID(4) {
		t.Fatalf("point id=4 still found after WCF delete")
	}

	// 5. A NEGATIVE write_consistency_factor is rejected at the edge (400) before
	// dispatch — proves the validate-at-edge guard.
	badBody := `{"upsert":true,"id":5,"vector":[1,1,0,0],"write_consistency_factor":-1}`
	if resp := httpJSON("POST", "/v1/collections/"+coll+"/points", badBody, nil); resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("negative WCF = %d, want 400", resp.StatusCode)
	}
}

// TestRemoteWriteConsistencyGRPC is the gRPC analogue: it drives the real
// grpcapi.Server (over the embedded engine + fanout decorator) in-process,
// mirroring the grpcapi server_test pattern. A write_consistency_factor=RF (here
// 5, clamped to RF=1) upsert SUCCEEDS at RF=1 and the point is findable; a
// no_wait write also succeeds; a default write is unchanged.
func TestRemoteWriteConsistencyGRPC(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)

	srv := grpcapi.NewServer(rostam.NewFanoutDispatcher(emb, emb.Node()), nil)
	ctx := context.Background()

	const coll = "wc_grpc"
	if _, err := srv.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name:   coll,
		Config: &pb.Config{Dim: 4, Metric: "cosine", Partitions: 4},
	}); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	findByID := func(id uint64) bool {
		t.Helper()
		resp, err := srv.Get(ctx, &pb.GetRequest{Collection: coll, Id: id})
		if status.Code(err) == codes.NotFound {
			return false
		}
		if err != nil {
			t.Fatalf("Get(%d): %v", id, err)
		}
		return resp.GetFound()
	}

	// 1. WCF=5 (clamped to RF=1) upsert → success + findable.
	if _, err := srv.Upsert(ctx, &pb.UpsertRequest{
		Collection: coll, Id: 1, Vector: []float32{1, 0, 0, 0}, Upsert: true,
		WriteConsistencyFactor: 5,
	}); err != nil {
		t.Fatalf("Upsert WCF=5: %v", err)
	}
	if !findByID(1) {
		t.Fatalf("WCF=5 point id=1 not found (gRPC envelope→dispatch→barrier must commit)")
	}

	// 2. no_wait upsert → success + findable.
	if _, err := srv.Upsert(ctx, &pb.UpsertRequest{
		Collection: coll, Id: 2, Vector: []float32{0, 1, 0, 0}, Upsert: true,
		NoWait: true,
	}); err != nil {
		t.Fatalf("Upsert no_wait: %v", err)
	}
	if !findByID(2) {
		t.Fatalf("no_wait point id=2 not found")
	}

	// 3. DEFAULT upsert (proto-zero WC fields) → success + findable.
	if _, err := srv.Upsert(ctx, &pb.UpsertRequest{
		Collection: coll, Id: 3, Vector: []float32{0, 0, 1, 0}, Upsert: true,
	}); err != nil {
		t.Fatalf("default Upsert: %v", err)
	}
	if !findByID(3) {
		t.Fatalf("default point id=3 not found")
	}

	// 4. Delete carrying WCF → success.
	if _, err := srv.Delete(ctx, &pb.DeleteRequest{Collection: coll, Id: 3, WriteConsistencyFactor: 2}); err != nil {
		t.Fatalf("Delete WCF: %v", err)
	}
	if findByID(3) {
		t.Fatalf("point id=3 still found after WCF delete")
	}
}

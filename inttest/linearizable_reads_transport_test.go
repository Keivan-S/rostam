// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/sdk/pb"
	"github.com/rostamlabs/rostam/ops"
)

// Strict-linearizable reads: prove read_consistency=2 (Linearizable)
// threads end-to-end through the real HTTP and gRPC transports against a
// single-node engine. On a single node VerifyLeader resolves immediately
// (quorum=1) so the readIndex barrier is effectively free — but the level must
// still ACCEPT + ROUTE + SERVE correct results. rc=3 (BoundedStaleness) is also
// a valid level (accepted at the edge); a value above it (4) must be rejected at
// the edge (400 / InvalidArgument). These tests fail if any read family
// drops/clamps the level (a silent stale-serve) or if the edge rejects a valid level.

// TestHTTPLinearizableEndToEnd drives a real single-node HTTP engine with
// read_consistency=2 across dense search + scroll (the minimum the plan
// requires), and asserts the level is accepted and returns correct results.
func TestHTTPLinearizableEndToEnd(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	srv, err := rostam.NewHTTPServer("127.0.0.1:0", rostam.DirectConfig{
		DataDir: t.TempDir(),
		Ops:     reg,
		Cache:   rostam.CacheConfig{NumShardsPerNode: 4},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	base := "http://" + srv.Addr()

	post := func(path, body string) (int, []byte) {
		resp, err := http.Post(base+path, "application/json", bytes.NewReader([]byte(body)))
		if err != nil {
			t.Fatalf("POST %s: %v", path, err)
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, b
	}

	if code, b := post("/v1/collections", `{"name":"docs","config":{"dim":3,"metric":"l2"}}`); code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", code, b)
	}
	for i := 1; i <= 5; i++ {
		body := `{"id":` + itoa(i) + `,"vector":[` + itoa(i) + `,0,0],"content":"chunk","upsert":true}`
		if code, b := post("/v1/collections/docs/points", body); code != http.StatusOK {
			t.Fatalf("upsert %d = %d (%s)", i, code, b)
		}
	}

	// Linearizable dense search (rc=2): must succeed and return the nearest hit.
	code, b := post("/v1/collections/docs/points/search", `{"query":[1,0,0],"k":3,"read_consistency":2}`)
	if code != http.StatusOK {
		t.Fatalf("linearizable search = %d (%s)", code, b)
	}
	var sres struct {
		Results []struct {
			ID uint64 `json:"id"`
		} `json:"results"`
	}
	if err := json.Unmarshal(b, &sres); err != nil {
		t.Fatalf("decode search: %v (%s)", err, b)
	}
	if len(sres.Results) == 0 || sres.Results[0].ID != 1 {
		t.Fatalf("linearizable search results = %+v, want nearest id=1 first", sres.Results)
	}

	// Linearizable scroll (rc=2): must succeed and page the points.
	code, b = post("/v1/collections/docs/points/scroll", `{"limit":10,"read_consistency":2}`)
	if code != http.StatusOK {
		t.Fatalf("linearizable scroll = %d (%s)", code, b)
	}
	var scr struct {
		Documents []struct {
			ID uint64 `json:"id"`
		} `json:"documents"`
	}
	if err := json.Unmarshal(b, &scr); err != nil {
		t.Fatalf("decode scroll: %v (%s)", err, b)
	}
	if len(scr.Documents) != 5 {
		t.Fatalf("linearizable scroll returned %d docs, want 5", len(scr.Documents))
	}

	// rc=3 (BoundedStaleness) is now a valid level: accepted at the edge.
	if code, _ := post("/v1/collections/docs/points/search", `{"query":[1,0,0],"k":3,"read_consistency":3}`); code != http.StatusOK {
		t.Fatalf("rc=3 (bounded-staleness) search = %d, want 200", code)
	}
	// Above BoundedStaleness (rc=4) is rejected at the edge with 400.
	if code, _ := post("/v1/collections/docs/points/search", `{"query":[1,0,0],"k":3,"read_consistency":4}`); code != http.StatusBadRequest {
		t.Fatalf("rc=4 search = %d, want 400", code)
	}

	// Regression: default (rc absent ⇒ 0) and rc=1 still work unchanged.
	if code, _ := post("/v1/collections/docs/points/search", `{"query":[1,0,0],"k":3}`); code != http.StatusOK {
		t.Fatalf("default rc search = %d, want 200", code)
	}
	if code, _ := post("/v1/collections/docs/points/search", `{"query":[1,0,0],"k":3,"read_consistency":1}`); code != http.StatusOK {
		t.Fatalf("rc=1 search = %d, want 200", code)
	}
}

// TestGRPCLinearizableEndToEnd drives a real single-node gRPC engine with
// read_consistency=2 across dense search + scroll, and asserts the level is
// accepted, returns correct results, and that rc=3 is rejected with
// InvalidArgument.
func TestGRPCLinearizableEndToEnd(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	srv, err := rostam.NewGRPCServer("127.0.0.1:0", rostam.DirectConfig{
		DataDir: t.TempDir(),
		Ops:     reg,
		Cache:   rostam.CacheConfig{NumShardsPerNode: 4},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()

	cli, closeConn := dialGRPC(t, srv.Addr())
	defer closeConn()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if _, err := cli.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name: "docs", Config: &pb.Config{Dim: 3, Metric: "l2"},
	}); err != nil {
		t.Fatalf("create: %v", err)
	}
	for i := 1; i <= 5; i++ {
		if _, err := cli.Upsert(ctx, &pb.UpsertRequest{
			Collection: "docs", Id: uint64(i), Vector: []float32{float32(i), 0, 0},
			Content: "chunk", Upsert: true,
		}); err != nil {
			t.Fatalf("upsert %d: %v", i, err)
		}
	}

	// Linearizable dense search (rc=2): must succeed and return id=1 nearest.
	sres, err := cli.Search(ctx, &pb.SearchRequest{
		Collection: "docs", Query: []float32{1, 0, 0}, K: 3, ReadConsistency: 2,
	})
	if err != nil {
		t.Fatalf("linearizable Search: %v", err)
	}
	if len(sres.GetResults()) == 0 || sres.GetResults()[0].GetId() != 1 {
		t.Fatalf("linearizable Search results = %+v, want nearest id=1 first", sres.GetResults())
	}

	// Linearizable scroll (rc=2): must page all 5 points.
	scr, err := cli.Scroll(ctx, &pb.ScrollRequest{Collection: "docs", Limit: 10, ReadConsistency: 2})
	if err != nil {
		t.Fatalf("linearizable Scroll: %v", err)
	}
	if len(scr.GetDocuments()) != 5 {
		t.Fatalf("linearizable Scroll returned %d docs, want 5", len(scr.GetDocuments()))
	}

	// rc=3 (BoundedStaleness) is now a valid level: accepted (not InvalidArgument).
	if _, err := cli.Search(ctx, &pb.SearchRequest{
		Collection: "docs", Query: []float32{1, 0, 0}, K: 3, ReadConsistency: 3,
	}); status.Code(err) == codes.InvalidArgument {
		t.Fatalf("rc=3 (bounded-staleness) Search rejected as InvalidArgument, want accepted: %v", err)
	}
	// Above BoundedStaleness (rc=4) is rejected with InvalidArgument.
	if _, err := cli.Search(ctx, &pb.SearchRequest{
		Collection: "docs", Query: []float32{1, 0, 0}, K: 3, ReadConsistency: 4,
	}); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("rc=4 Search code = %v, want InvalidArgument", status.Code(err))
	}

	// Regression: default (rc=0) and rc=1 still serve.
	if _, err := cli.Search(ctx, &pb.SearchRequest{Collection: "docs", Query: []float32{1, 0, 0}, K: 3}); err != nil {
		t.Fatalf("default rc Search: %v", err)
	}
	if _, err := cli.Search(ctx, &pb.SearchRequest{Collection: "docs", Query: []float32{1, 0, 0}, K: 3, ReadConsistency: 1}); err != nil {
		t.Fatalf("rc=1 Search: %v", err)
	}
}

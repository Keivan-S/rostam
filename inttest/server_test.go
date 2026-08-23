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

	"google.golang.org/grpc/metadata"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/sdk/pb"
	"github.com/rostamlabs/rostam/vector"
)

// TestUnifiedServerSharesOneStore proves that HTTP, gRPC, and TCP served by a
// single NewServer are three front doors onto ONE store: a collection created
// over HTTP, a point written over gRPC, and a point written over TCP are all
// visible to a search issued over each transport.
func TestUnifiedServerSharesOneStore(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	srv, err := rostam.NewServer(rostam.ServerConfig{
		DirectConfig: rostam.DirectConfig{DataDir: t.TempDir(), Ops: reg, Cache: rostam.CacheConfig{NumShardsPerNode: 4}},
		HTTPAddr:     "127.0.0.1:0",
		GRPCAddr:     "127.0.0.1:0",
		TCPAddr:      "127.0.0.1:0",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()

	for name, addr := range map[string]string{"http": srv.HTTPAddr(), "grpc": srv.GRPCAddr(), "tcp": srv.TCPAddr()} {
		if addr == "" {
			t.Fatalf("%s transport not bound", name)
		}
	}

	// 1) Create the collection over HTTP/REST.
	httpPost := func(path, body string) int {
		resp, err := http.Post("http://"+srv.HTTPAddr()+path, "application/json", bytes.NewReader([]byte(body)))
		if err != nil {
			t.Fatalf("POST %s: %v", path, err)
		}
		defer resp.Body.Close()
		io.Copy(io.Discard, resp.Body)
		return resp.StatusCode
	}
	if code := httpPost("/v1/collections", `{"name":"docs","config":{"dim":3,"metric":"l2"}}`); code != http.StatusCreated {
		t.Fatalf("create over HTTP = %d", code)
	}

	// 2) Write point id=1 over gRPC.
	gcli, closeConn := dialGRPC(t, srv.GRPCAddr())
	defer closeConn()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, err := gcli.Upsert(ctx, &pb.UpsertRequest{
		Collection: "docs", Id: 1, Vector: []float32{1, 0, 0}, Content: "via-grpc", Upsert: true,
	}); err != nil {
		t.Fatalf("upsert over gRPC: %v", err)
	}

	// 3) Write point id=2 over the binary TCP protocol.
	tcli, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{srv.TCPAddr()}})
	if err != nil {
		t.Fatal(err)
	}
	defer tcli.Close()
	if err := tcli.VectorInsert(ctx, "docs", 2, []float32{2, 0, 0}); err != nil {
		t.Fatalf("insert over TCP: %v", err)
	}

	// 4) All three transports must now see BOTH points (one shared store).
	wantIDs := map[uint64]bool{1: true, 2: true}

	// ...over TCP.
	res, err := tcli.VectorSearch(ctx, "docs", []float32{0, 0, 0}, 5)
	if err != nil {
		t.Fatal(err)
	}
	assertIDs(t, "tcp", res, wantIDs)

	// ...over gRPC.
	gres, err := gcli.Search(ctx, &pb.SearchRequest{Collection: "docs", Query: []float32{0, 0, 0}, K: 5})
	if err != nil {
		t.Fatal(err)
	}
	got := map[uint64]bool{}
	for _, r := range gres.GetResults() {
		got[r.GetId()] = true
	}
	if !got[1] || !got[2] {
		t.Errorf("gRPC search missing points: %v", got)
	}

	// ...over HTTP.
	resp, err := http.Post("http://"+srv.HTTPAddr()+"/v1/collections/docs/points/search",
		"application/json", bytes.NewReader([]byte(`{"query":[0,0,0],"k":5}`)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var sr struct {
		Results []struct {
			ID uint64 `json:"id"`
		} `json:"results"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&sr); err != nil {
		t.Fatal(err)
	}
	httpIDs := map[uint64]bool{}
	for _, r := range sr.Results {
		httpIDs[r.ID] = true
	}
	if !httpIDs[1] || !httpIDs[2] {
		t.Errorf("HTTP search missing points: %v", httpIDs)
	}
}

func assertIDs(t *testing.T, who string, res []rostam.VectorResult, want map[uint64]bool) {
	t.Helper()
	got := map[uint64]bool{}
	for _, r := range res {
		got[r.ID] = true
	}
	for id := range want {
		if !got[id] {
			t.Errorf("%s search missing id %d (got %v)", who, id, got)
		}
	}
}

// TestUnifiedServerMultiVector drives the late-interaction (MaxSim) path across
// transports over one store: create + add + delete via the public TCP client,
// then confirm the MaxSim winner is consistent when searched over TCP, gRPC, and
// HTTP — proving the multi-vector op family is wired through every front end and
// backed by the same store.
func TestUnifiedServerMultiVector(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	srv, err := rostam.NewServer(rostam.ServerConfig{
		DirectConfig: rostam.DirectConfig{DataDir: t.TempDir(), Ops: reg, Cache: rostam.CacheConfig{NumShardsPerNode: 4}},
		HTTPAddr:     "127.0.0.1:0",
		GRPCAddr:     "127.0.0.1:0",
		TCPAddr:      "127.0.0.1:0",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	ctx := context.Background()

	// Create + populate over the public TCP client.
	tcli, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{srv.TCPAddr()}})
	if err != nil {
		t.Fatal(err)
	}
	defer tcli.Close()
	if err := tcli.VectorMVCreateCollection(ctx, "docs", rostam.MultiVectorConfig{Dim: 4}); err != nil {
		t.Fatalf("mv create: %v", err)
	}
	// doc 1 has a token aligned with the query; doc 2 does not.
	if err := tcli.VectorMVAdd(ctx, "docs", 1, [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}, rostam.VectorMetadata{"d": vector.NewInt(1)}); err != nil {
		t.Fatalf("mv add 1: %v", err)
	}
	if err := tcli.VectorMVAdd(ctx, "docs", 2, [][]float32{{0, 0, 1, 0}}, nil); err != nil {
		t.Fatalf("mv add 2: %v", err)
	}

	// Search over TCP.
	res, _, err := tcli.VectorMVSearch(ctx, "docs", [][]float32{{1, 0, 0, 0}}, 2, rostam.MultiSearchOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 || res[0].ID != 1 {
		t.Fatalf("TCP mv search = %+v, want doc 1 first", res)
	}

	// Search over gRPC.
	gcli, closeConn := dialGRPC(t, srv.GRPCAddr())
	defer closeConn()
	gres, err := gcli.MVSearch(ctx, &pb.MVSearchRequest{
		Name: "docs", K: 2, Query: []*pb.TokenVector{{Values: []float32{1, 0, 0, 0}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(gres.GetResults()) == 0 || gres.GetResults()[0].GetId() != 1 {
		t.Errorf("gRPC mv search = %+v, want doc 1 first", gres.GetResults())
	}

	// Search over HTTP.
	resp, err := http.Post("http://"+srv.HTTPAddr()+"/v1/multivector/docs/search",
		"application/json", bytes.NewReader([]byte(`{"query":[[1,0,0,0]],"k":2}`)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var hres struct {
		Results []struct {
			ID    uint64  `json:"id"`
			Score float32 `json:"score"`
		} `json:"results"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&hres); err != nil {
		t.Fatal(err)
	}
	if len(hres.Results) == 0 || hres.Results[0].ID != 1 {
		t.Errorf("HTTP mv search = %+v, want doc 1 first", hres.Results)
	}

	// Delete doc 1 over TCP; doc 2 remains everywhere.
	ok, err := tcli.VectorMVDelete(ctx, "docs", 1)
	if err != nil || !ok {
		t.Fatalf("mv delete = %v,%v", ok, err)
	}
	res, _, _ = tcli.VectorMVSearch(ctx, "docs", [][]float32{{1, 0, 0, 0}}, 5, rostam.MultiSearchOpts{})
	if len(res) != 1 || res[0].ID != 2 {
		t.Errorf("after delete = %+v, want only doc 2", res)
	}
}

func TestNewServerRequiresATransport(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	_, err := rostam.NewServer(rostam.ServerConfig{DirectConfig: rostam.DirectConfig{Ops: reg}})
	if err == nil {
		t.Fatal("expected error when no transport address is set")
	}
}

// TestUnifiedServerAuthAcrossTransports checks the one Authenticator gates both
// HTTP and gRPC.
func TestUnifiedServerAuthAcrossTransports(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	srv, err := rostam.NewServer(rostam.ServerConfig{
		DirectConfig: rostam.DirectConfig{
			DataDir:       t.TempDir(),
			Ops:           reg,
			Cache:         rostam.CacheConfig{NumShardsPerNode: 4},
			Authenticator: func(req authz.AuthRequest) bool { return req.Token == "secret" },
		},
		HTTPAddr: "127.0.0.1:0",
		GRPCAddr: "127.0.0.1:0",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()

	// HTTP without a token → 401.
	resp, err := http.Post("http://"+srv.HTTPAddr()+"/v1/collections",
		"application/json", bytes.NewReader([]byte(`{"name":"d","config":{"dim":3}}`)))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("HTTP no-token = %d, want 401", resp.StatusCode)
	}

	// gRPC without a token → error; with the token → ok.
	gcli, closeConn := dialGRPC(t, srv.GRPCAddr())
	defer closeConn()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, err := gcli.CreateCollection(ctx, &pb.CreateCollectionRequest{Name: "denied", Config: &pb.Config{Dim: 3}}); err == nil {
		t.Error("gRPC without token should be rejected")
	}
	authCtx := metadata.AppendToOutgoingContext(ctx, "authorization", "Bearer secret")
	// The rejected create must not have taken effect: searching "denied" (with a
	// valid token) returns NotFound because the collection was never created.
	if _, err := gcli.Search(authCtx, &pb.SearchRequest{Collection: "denied", Query: []float32{0, 0, 0}, K: 1}); err == nil {
		t.Error("rejected no-token create leaked a collection")
	}
	if _, err := gcli.CreateCollection(authCtx, &pb.CreateCollectionRequest{Name: "allowed", Config: &pb.Config{Dim: 3}}); err != nil {
		t.Errorf("gRPC with token: %v", err)
	}
}

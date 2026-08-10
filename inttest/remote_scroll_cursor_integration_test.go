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
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/grpcapi"
	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/httpapi"
)

// TestRemoteScrollCursorPaginationHTTP drives cursor pagination end-to-end over
// the REAL HTTP transport (httptest) backed by the embedded engine + fanout
// decorator — the same wrap server.go installs in the cluster branch. A
// PARTITIONED (P=4) collection is seeded with N tie-free points; we page through
// BOTH the dense scroll route and the named scroll route following next_cursor
// until it is empty, asserting every id appears EXACTLY once, globally ascending,
// and the total equals N. It also proves the no-cursor request returns ascending
// with next_cursor present, and a malformed cursor ⇒ HTTP 400 before dispatch.
func TestRemoteScrollCursorPaginationHTTP(t *testing.T) {
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

	const (
		dense = "scroll_dense"
		named = "scroll_named"
		n     = 120
		limit = 25
	)

	// PARTITIONED (P=4) dense collection.
	if resp := httpJSON("POST", "/v1/collections",
		fmt.Sprintf(`{"name":%q,"config":{"dim":8,"metric":"cosine","m":16,"ef_construction":200,"ef_search":64,"partitions":4}}`, dense), nil); resp.StatusCode != http.StatusCreated {
		t.Fatalf("create dense = %d, want 201", resp.StatusCode)
	}
	if p, _, ok := emb.Catalog().PartitionsGen(dense); !ok || p != 4 {
		t.Fatalf("%s PartitionsGen = (%d, ok=%v), want (4, true)", dense, p, ok)
	}
	// PARTITIONED (P=4) named collection (single space "title").
	if resp := httpJSON("POST", "/v1/named/"+named,
		`{"named_vectors":{"title":{"dim":8,"metric":0,"m":16,"ef_construction":200,"ef_search":64}},"partitions":4}`, nil); resp.StatusCode != http.StatusCreated {
		t.Fatalf("create named = %d, want 201", resp.StatusCode)
	}
	if p, _, ok := emb.Catalog().PartitionsGen(named); !ok || p != 4 {
		t.Fatalf("%s PartitionsGen = (%d, ok=%v), want (4, true)", named, p, ok)
	}

	// Seed both with N tie-free points (ids 0..N-1 spread across partitions).
	for i := 0; i < n; i++ {
		v := tieFreeVec(i)
		parts := make([]string, len(v))
		for j, c := range v {
			parts[j] = fmt.Sprintf("%g", c)
		}
		vecStr := strings.Join(parts, ",")
		if resp := httpJSON("POST", "/v1/collections/"+dense+"/points",
			fmt.Sprintf(`{"id":%d,"vector":[%s]}`, i, vecStr), nil); resp.StatusCode != http.StatusOK {
			t.Fatalf("dense insert %d = %d, want 200", i, resp.StatusCode)
		}
		if resp := httpJSON("POST", "/v1/named/"+named+"/points",
			fmt.Sprintf(`{"id":%d,"vectors":{"title":[%s]}}`, i, vecStr), nil); resp.StatusCode != http.StatusOK {
			t.Fatalf("named insert %d = %d, want 200", i, resp.StatusCode)
		}
	}

	type pageDoc struct {
		ID uint64 `json:"id"`
	}

	// pageDense follows next_cursor from the given route until it is empty,
	// returning the ordered id stream. body builds the request body for a cursor
	// (empty cursor ⇒ first page).
	pageDense := func(label, path string, body func(cursor string) string) []uint64 {
		t.Helper()
		var all []uint64
		cursor := ""
		for iter := 0; iter <= n; iter++ {
			var out struct {
				Documents  []pageDoc `json:"documents"`
				NextCursor string    `json:"next_cursor"`
			}
			resp := httpJSON("POST", path, body(cursor), &out)
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("%s scroll = %d, want 200", label, resp.StatusCode)
			}
			for _, d := range out.Documents {
				all = append(all, d.ID)
			}
			if len(out.Documents) == 0 {
				// An empty page must coincide with an empty cursor (exhausted).
				if out.NextCursor != "" {
					t.Fatalf("%s: empty page but next_cursor=%q (want exhausted)", label, out.NextCursor)
				}
				return all
			}
			if out.NextCursor == "" {
				// Last partial page: cursor exhausted.
				return all
			}
			cursor = out.NextCursor
		}
		t.Fatalf("%s: pagination did not terminate", label)
		return all
	}

	assertAscendingUnique := func(label string, ids []uint64) {
		t.Helper()
		if len(ids) != n {
			t.Fatalf("%s: got %d ids, want %d", label, len(ids), n)
		}
		seen := make(map[uint64]bool, len(ids))
		for i, id := range ids {
			if seen[id] {
				t.Fatalf("%s: id %d seen more than once", label, id)
			}
			seen[id] = true
			if i > 0 && ids[i-1] >= id {
				t.Fatalf("%s: not strictly ascending at index %d: %d then %d", label, i, ids[i-1], id)
			}
		}
		for i := 0; i < n; i++ {
			if !seen[uint64(i)] {
				t.Fatalf("%s: id %d missing from paginated stream", label, i)
			}
		}
	}

	// HTTP dense deep pagination.
	denseIDs := pageDense("http-dense", "/v1/collections/"+dense+"/points/scroll",
		func(cursor string) string {
			if cursor == "" {
				return fmt.Sprintf(`{"limit":%d}`, limit)
			}
			return fmt.Sprintf(`{"limit":%d,"cursor":%q}`, limit, cursor)
		})
	assertAscendingUnique("http-dense", denseIDs)

	// HTTP named deep pagination.
	namedIDs := pageDense("http-named", "/v1/named/"+named+"/scroll",
		func(cursor string) string {
			if cursor == "" {
				return fmt.Sprintf(`{"limit":%d}`, limit)
			}
			return fmt.Sprintf(`{"limit":%d,"cursor":%q}`, limit, cursor)
		})
	assertAscendingUnique("http-named", namedIDs)

	// No-cursor request: ascending, next_cursor present (a full page ⇒ non-empty).
	{
		var out struct {
			Documents  []pageDoc `json:"documents"`
			NextCursor string    `json:"next_cursor"`
		}
		resp := httpJSON("POST", "/v1/collections/"+dense+"/points/scroll", fmt.Sprintf(`{"limit":%d}`, limit), &out)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("no-cursor dense scroll = %d, want 200", resp.StatusCode)
		}
		if len(out.Documents) != limit {
			t.Fatalf("no-cursor first page len = %d, want %d", len(out.Documents), limit)
		}
		for i := 1; i < len(out.Documents); i++ {
			if out.Documents[i-1].ID >= out.Documents[i].ID {
				t.Fatalf("no-cursor page not ascending at %d", i)
			}
		}
		if out.NextCursor == "" {
			t.Fatalf("no-cursor full page should carry a next_cursor")
		}
	}

	// No-cursor, unlimited (limit=0) ⇒ whole collection, next_cursor empty.
	{
		var out struct {
			Documents  []pageDoc `json:"documents"`
			NextCursor string    `json:"next_cursor"`
		}
		resp := httpJSON("POST", "/v1/collections/"+dense+"/points/scroll", `{}`, &out)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("unlimited dense scroll = %d, want 200", resp.StatusCode)
		}
		if len(out.Documents) != n {
			t.Fatalf("unlimited scroll len = %d, want %d", len(out.Documents), n)
		}
		if out.NextCursor != "" {
			t.Fatalf("unlimited scroll next_cursor = %q, want empty", out.NextCursor)
		}
	}

	// Malformed cursor ⇒ HTTP 400 BEFORE dispatch (dense + named).
	if resp := httpJSON("POST", "/v1/collections/"+dense+"/points/scroll", `{"limit":5,"cursor":"!!!notbase64"}`, nil); resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("malformed dense cursor = %d, want 400", resp.StatusCode)
	}
	if resp := httpJSON("POST", "/v1/named/"+named+"/scroll", `{"limit":5,"cursor":"!!!notbase64"}`, nil); resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("malformed named cursor = %d, want 400", resp.StatusCode)
	}
}

// TestRemoteScrollCursorPaginationGRPC mirrors the HTTP test over the gRPC
// surface: it pages through the NEW dense Scroll RPC AND the named NamedScroll
// RPC (now carrying cursor/next_cursor) against a PARTITIONED (P=4) collection,
// asserting every id appears exactly once ascending. A malformed cursor ⇒
// codes.InvalidArgument on both RPCs.
func TestRemoteScrollCursorPaginationGRPC(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)

	srv := grpcapi.NewServer(rostam.NewFanoutDispatcher(emb, emb.Node()), nil)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	const (
		dense = "grpc_scroll_dense"
		named = "grpc_scroll_named"
		n     = 120
		limit = 25
	)

	if _, err := srv.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name:   dense,
		Config: &pb.Config{Dim: 8, Metric: "cosine", Partitions: 4},
	}); err != nil {
		t.Fatalf("create dense: %v", err)
	}
	if _, err := srv.NamedCreate(ctx, &pb.NamedCreateRequest{
		Name:       named,
		Partitions: 4,
		ConfigJson: `{"title":{"dim":8,"metric":0,"m":16,"ef_construction":200,"ef_search":64}}`,
	}); err != nil {
		t.Fatalf("create named: %v", err)
	}

	for i := 0; i < n; i++ {
		v := tieFreeVec(i)
		if _, err := srv.Upsert(ctx, &pb.UpsertRequest{
			Collection: dense, Id: uint64(i), Vector: v, Upsert: true,
		}); err != nil {
			t.Fatalf("dense upsert %d: %v", i, err)
		}
		if _, err := srv.NamedUpsert(ctx, &pb.NamedUpsertRequest{
			Name: named, Id: uint64(i),
			Vectors: map[string]*pb.NamedVectorList{"title": {Values: v}},
		}); err != nil {
			t.Fatalf("named upsert %d: %v", i, err)
		}
	}

	assertAscendingUnique := func(label string, ids []uint64) {
		t.Helper()
		if len(ids) != n {
			t.Fatalf("%s: got %d ids, want %d", label, len(ids), n)
		}
		seen := make(map[uint64]bool, len(ids))
		for i, id := range ids {
			if seen[id] {
				t.Fatalf("%s: id %d seen more than once", label, id)
			}
			seen[id] = true
			if i > 0 && ids[i-1] >= id {
				t.Fatalf("%s: not strictly ascending at index %d: %d then %d", label, i, ids[i-1], id)
			}
		}
		for i := 0; i < n; i++ {
			if !seen[uint64(i)] {
				t.Fatalf("%s: id %d missing", label, i)
			}
		}
	}

	// gRPC dense deep pagination via the NEW Scroll RPC.
	{
		var all []uint64
		cursor := ""
		for iter := 0; iter <= n; iter++ {
			resp, err := srv.Scroll(ctx, &pb.ScrollRequest{
				Collection: dense, Limit: int32(limit), Cursor: cursor,
			})
			if err != nil {
				t.Fatalf("grpc dense Scroll: %v", err)
			}
			for _, d := range resp.GetDocuments() {
				all = append(all, d.GetId())
			}
			if len(resp.GetDocuments()) == 0 {
				if resp.GetNextCursor() != "" {
					t.Fatalf("grpc dense: empty page but next_cursor=%q", resp.GetNextCursor())
				}
				break
			}
			if resp.GetNextCursor() == "" {
				break
			}
			cursor = resp.GetNextCursor()
			if iter == n {
				t.Fatalf("grpc dense: pagination did not terminate")
			}
		}
		assertAscendingUnique("grpc-dense", all)
	}

	// gRPC named deep pagination via NamedScroll (now cursor-paginated).
	{
		var all []uint64
		cursor := ""
		for iter := 0; iter <= n; iter++ {
			resp, err := srv.NamedScroll(ctx, &pb.NamedScrollRequest{
				Name: named, Limit: int32(limit), Cursor: cursor,
			})
			if err != nil {
				t.Fatalf("grpc NamedScroll: %v", err)
			}
			for _, d := range resp.GetDocuments() {
				all = append(all, d.GetId())
			}
			if len(resp.GetDocuments()) == 0 {
				if resp.GetNextCursor() != "" {
					t.Fatalf("grpc named: empty page but next_cursor=%q", resp.GetNextCursor())
				}
				break
			}
			if resp.GetNextCursor() == "" {
				break
			}
			cursor = resp.GetNextCursor()
			if iter == n {
				t.Fatalf("grpc named: pagination did not terminate")
			}
		}
		assertAscendingUnique("grpc-named", all)
	}

	// Malformed cursor ⇒ InvalidArgument on both RPCs.
	if _, err := srv.Scroll(ctx, &pb.ScrollRequest{Collection: dense, Limit: 5, Cursor: "!!!notbase64"}); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("grpc dense malformed cursor: code = %v, want InvalidArgument", status.Code(err))
	}
	if _, err := srv.NamedScroll(ctx, &pb.NamedScrollRequest{Name: named, Limit: 5, Cursor: "!!!notbase64"}); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("grpc named malformed cursor: code = %v, want InvalidArgument", status.Code(err))
	}
}

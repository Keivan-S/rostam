// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/httpapi"
	"github.com/rostamlabs/rostam/vector"
)

// TestRemoteAliasManagementAndDataPlaneHTTP proves the full alias stack over the
// REAL HTTP transport (httptest) backed by the embedded engine + fanout
// decorator — the same wrap server.go installs in the cluster branch:
//
//  1. alias-MANAGEMENT surface: POST /v1/aliases (create) → GET /v1/aliases
//     (list shows it) → POST /v1/aliases/batch (atomic swap) → list reflects the
//     swap → DELETE /v1/aliases/{alias} → list empty;
//  2. data-plane-VIA-alias: a search AND an upsert to /v1/collections/{alias}/...
//     resolve through the alias to a PARTITIONED target (Task-3 resolution),
//     returning correct cross-partition results — the full-stack proof that the
//     alias-MGMT transport (this task) and the engine resolution agree.
func TestRemoteAliasManagementAndDataPlaneHTTP(t *testing.T) {
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

	// listAliases fetches the current alias set as a name→collection map.
	listAliases := func(filter string) map[string]string {
		t.Helper()
		path := "/v1/aliases"
		if filter != "" {
			path += "?collection=" + filter
		}
		var out struct {
			Aliases []struct {
				Alias      string `json:"alias"`
				Collection string `json:"collection"`
			} `json:"aliases"`
		}
		resp := httpJSON("GET", path, "", &out)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("list aliases = %d, want 200", resp.StatusCode)
		}
		m := make(map[string]string, len(out.Aliases))
		for _, e := range out.Aliases {
			m[e.Alias] = e.Collection
		}
		return m
	}

	const (
		v1 = "coll_v1"
		v2 = "coll_v2"
		n  = 60
		k  = 10
	)

	// Build two PARTITIONED (P=4) collections via the HTTP create endpoint.
	for _, coll := range []string{v1, v2} {
		body := fmt.Sprintf(
			`{"name":%q,"config":{"dim":8,"metric":"cosine","m":16,"ef_construction":200,"ef_search":64,"partitions":4}}`, coll)
		if resp := httpJSON("POST", "/v1/collections", body, nil); resp.StatusCode != http.StatusCreated {
			t.Fatalf("create %s = %d, want 201", coll, resp.StatusCode)
		}
		if p, _, ok := emb.Catalog().PartitionsGen(coll); !ok || p != 4 {
			t.Fatalf("%s PartitionsGen = (%d, ok=%v), want (4, true)", coll, p, ok)
		}
	}

	// Seed v1 with n tie-free vectors directly on the engine (ground truth).
	insert := func(coll string, i int) {
		v := tieFreeVec(i)
		parts := make([]string, len(v))
		for j, c := range v {
			parts[j] = fmt.Sprintf("%g", c)
		}
		body := fmt.Sprintf(`{"id":%d,"vector":[%s]}`, i, strings.Join(parts, ","))
		if resp := httpJSON("POST", "/v1/collections/"+coll+"/points", body, nil); resp.StatusCode != http.StatusOK {
			t.Fatalf("insert %d into %s = %d, want 200", i, coll, resp.StatusCode)
		}
	}
	for i := 0; i < n; i++ {
		insert(v1, i)
	}

	// 1. CREATE alias prod → v1.
	if resp := httpJSON("POST", "/v1/aliases", fmt.Sprintf(`{"alias":"prod","collection":%q}`, v1), nil); resp.StatusCode != http.StatusCreated {
		t.Fatalf("create alias = %d, want 201", resp.StatusCode)
	}
	if got := listAliases(""); got["prod"] != v1 {
		t.Fatalf("after create: aliases = %v, want prod→%s", got, v1)
	}
	// ?collection filter narrows to v1's aliases.
	if got := listAliases(v1); got["prod"] != v1 || len(got) != 1 {
		t.Fatalf("filtered list = %v, want only prod→%s", got, v1)
	}

	// 2a. DATA-PLANE via alias: search /v1/collections/prod/... must resolve to
	// the PARTITIONED v1 and return a correct cross-partition top-k. A resolution
	// mismatch (Task-3) would hit the empty logical collection → zero results.
	q := tieFreeQuery()
	qParts := make([]string, len(q))
	for j, c := range q {
		qParts[j] = fmt.Sprintf("%g", c)
	}
	searchVia := func(coll string) []vector.Result {
		t.Helper()
		var sres struct {
			Results []vector.Result `json:"results"`
		}
		resp := httpJSON("POST", "/v1/collections/"+coll+"/points/search",
			fmt.Sprintf(`{"query":[%s],"k":%d}`, strings.Join(qParts, ","), k), &sres)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("search via %s = %d, want 200", coll, resp.StatusCode)
		}
		return sres.Results
	}
	viaAlias := searchVia("prod")
	direct := searchVia(v1)
	if len(viaAlias) != k {
		t.Fatalf("search via alias returned %d, want %d (resolution mismatch would zero this)", len(viaAlias), k)
	}
	if len(direct) != k {
		t.Fatalf("direct search returned %d, want %d", len(direct), k)
	}
	for i := range direct {
		if viaAlias[i].ID != direct[i].ID {
			t.Fatalf("alias search rank %d: id %d != direct id %d (alias resolution mismatch)",
				i, viaAlias[i].ID, direct[i].ID)
		}
	}

	// 2b. WRITE-through via alias: upsert a new point through /collections/prod/...
	// then confirm it is retrievable by id on the REAL collection v1.
	upBody := `{"upsert":true,"id":9999,"vector":[1,0,0,0,0,0,0,0]}`
	if resp := httpJSON("POST", "/v1/collections/prod/points", upBody, nil); resp.StatusCode != http.StatusOK {
		t.Fatalf("upsert via alias = %d, want 200", resp.StatusCode)
	}
	var got struct {
		Found bool `json:"found"`
	}
	if resp := httpJSON("GET", "/v1/collections/"+v1+"/points/9999", "", &got); resp.StatusCode != http.StatusOK || !got.Found {
		t.Fatalf("write-through point not found on %s: status=%d found=%v", v1, resp.StatusCode, got.Found)
	}

	// 3. ATOMIC swap prod v1→v2 in ONE batch ([delete prod, create prod→v2]).
	swap := fmt.Sprintf(
		`{"actions":[{"delete":{"alias":"prod"}},{"create":{"alias":"prod","collection":%q}}]}`, v2)
	var sw struct {
		Applied int `json:"applied"`
	}
	if resp := httpJSON("POST", "/v1/aliases/batch", swap, &sw); resp.StatusCode != http.StatusOK || sw.Applied != 2 {
		t.Fatalf("alias batch swap = %d applied=%d, want 200/2", resp.StatusCode, sw.Applied)
	}
	if got := listAliases(""); got["prod"] != v2 {
		t.Fatalf("after swap: aliases = %v, want prod→%s", got, v2)
	}

	// 4. DELETE alias → list empty.
	if resp := httpJSON("DELETE", "/v1/aliases/prod", "", nil); resp.StatusCode != http.StatusOK {
		t.Fatalf("delete alias = %d, want 200", resp.StatusCode)
	}
	if got := listAliases(""); len(got) != 0 {
		t.Fatalf("after delete: aliases = %v, want empty", got)
	}
	// Idempotent: a second delete is still 200.
	if resp := httpJSON("DELETE", "/v1/aliases/prod", "", nil); resp.StatusCode != http.StatusOK {
		t.Fatalf("idempotent delete = %d, want 200", resp.StatusCode)
	}
}

// TestRemoteAliasInvalidInputHTTP confirms alias-validation failures surface as
// HTTP 400 over the real transport (not a 500/panic): a missing target, a name
// that shadows a real collection, and a reserved-char name.
func TestRemoteAliasInvalidInputHTTP(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)

	h := httpapi.Handler(rostam.NewFanoutDispatcher(emb, emb.Node()), httpapi.Options{})
	ts := httptest.NewServer(h)
	t.Cleanup(ts.Close)

	post := func(path, body string) int {
		t.Helper()
		req, err := http.NewRequest("POST", ts.URL+path, strings.NewReader(body))
		if err != nil {
			t.Fatalf("build request: %v", err)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("POST %s: %v", path, err)
		}
		_ = resp.Body.Close()
		return resp.StatusCode
	}

	// A real collection to use as a (valid) target + a shadow conflict.
	createBody := `{"name":"docs","config":{"dim":8,"metric":"cosine","m":16,"ef_construction":200,"ef_search":64,"partitions":4}}`
	if code := post("/v1/collections", createBody); code != http.StatusCreated {
		t.Fatalf("create docs = %d, want 201", code)
	}

	// target missing.
	if code := post("/v1/aliases", `{"alias":"prod","collection":"nope"}`); code != http.StatusBadRequest {
		t.Fatalf("alias to missing target = %d, want 400", code)
	}
	// shadow: alias name == an existing real collection.
	if code := post("/v1/aliases", `{"alias":"docs","collection":"docs"}`); code != http.StatusBadRequest {
		t.Fatalf("shadowing alias = %d, want 400", code)
	}
	// reserved char in alias name.
	if code := post("/v1/aliases", `{"alias":"bad#name","collection":"docs"}`); code != http.StatusBadRequest {
		t.Fatalf("reserved-char alias = %d, want 400", code)
	}
}

// TestRemoteCreateCollectionRejectsAliasShadowHTTP is the reverse-direction
// guard: a new collection must not take the name of an EXISTING alias, or
// data-plane ops would be ambiguous (alias resolution vs the real collection).
// The PARTITIONED create path gets this guard inside embedded.CreateCollection,
// but the SINGLE-PARTITION create routes straight through the dispatcher to
// inner.Call (no alias knowledge) — so the guard must also live at the fanout
// dispatcher's single-partition branch. This proves it over the real transport
// for all three families (dense / MV / named) with partitions<=1.
func TestRemoteCreateCollectionRejectsAliasShadowHTTP(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*rostam.Embedded)

	h := httpapi.Handler(rostam.NewFanoutDispatcher(emb, emb.Node()), httpapi.Options{})
	ts := httptest.NewServer(h)
	t.Cleanup(ts.Close)

	post := func(path, body string) int {
		t.Helper()
		req, err := http.NewRequest("POST", ts.URL+path, strings.NewReader(body))
		if err != nil {
			t.Fatalf("build request: %v", err)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("POST %s: %v", path, err)
		}
		_ = resp.Body.Close()
		return resp.StatusCode
	}

	// A real target collection + an alias "prod" pointing at it.
	if code := post("/v1/collections",
		`{"name":"docs","config":{"dim":8,"metric":"cosine","m":16,"ef_construction":200,"ef_search":64,"partitions":4}}`); code != http.StatusCreated {
		t.Fatalf("create docs = %d, want 201", code)
	}
	if code := post("/v1/aliases", `{"alias":"prod","collection":"docs"}`); code != http.StatusCreated {
		t.Fatalf("create alias prod = %d, want 201", code)
	}

	// Each family: a SINGLE-PARTITION create named "prod" (the existing alias)
	// must be rejected (400), not silently created (which would shadow the alias).
	cases := []struct {
		name, path, body string
	}{
		{
			name: "dense",
			path: "/v1/collections",
			body: `{"name":"prod","config":{"dim":8,"metric":"cosine","m":16,"ef_construction":200,"ef_search":64,"partitions":1}}`,
		},
		{
			name: "multivector",
			path: "/v1/multivector/prod",
			body: `{"dim":8,"m":16,"ef_construction":200,"ef_search":64,"partitions":1}`,
		},
		{
			name: "named",
			path: "/v1/named/prod",
			body: `{"named_vectors":{"title":{"dim":8,"metric":"cosine","m":16,"ef_construction":200,"ef_search":64}},"partitions":1}`,
		},
	}
	for _, c := range cases {
		if code := post(c.path, c.body); code != http.StatusBadRequest {
			t.Fatalf("%s create shadowing alias = %d, want 400", c.name, code)
		}
	}
}

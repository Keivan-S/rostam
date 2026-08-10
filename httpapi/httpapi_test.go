// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// testDispatcher runs ops over a real registry + vector store, mirroring how
// directStore.Call dispatches — without importing the rostam package (which
// would form an import cycle once it depends on httpapi).
type testDispatcher struct {
	reg *ops.Registry
	tx  *ops.TxContext
}

func (d *testDispatcher) Call(name string, args []byte) ([]byte, error) {
	h, _, _, ok := d.reg.Lookup(name)
	if !ok {
		return nil, fmt.Errorf("op %q not registered", name)
	}
	return h(d.tx, args)
}

func (d *testDispatcher) LeaderAddr() string { return "" }

func newTestAPI(t *testing.T) (http.Handler, func()) {
	t.Helper()
	return newTestAPIOpts(t, Options{})
}

// newTestAPIOpts is newTestAPI with caller-supplied Options, for tests that need
// an Authenticator (e.g. asserting an anonymous caller is turned away before a
// handler does any attacker-sized work).
func newTestAPIOpts(t *testing.T, opts Options) (http.Handler, func()) {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	c, _ := cache.New(cache.DefaultConfig())
	vstore, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	disp := &testDispatcher{reg: reg, tx: ops.NewTxContextWithVectors(c, vstore)}
	cleanup := func() { _ = vstore.Close(); c.Close() }
	return Handler(disp, opts), cleanup
}

// do issues a request against the handler and returns the recorder plus the
// decoded JSON body (into out, if non-nil).
func do(t *testing.T, h http.Handler, method, path, body string, out any) *httptest.ResponseRecorder {
	t.Helper()
	var r *http.Request
	if body == "" {
		r = httptest.NewRequest(method, path, nil)
	} else {
		r = httptest.NewRequest(method, path, strings.NewReader(body))
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, r)
	if out != nil && rec.Body.Len() > 0 {
		if err := json.Unmarshal(rec.Body.Bytes(), out); err != nil {
			t.Fatalf("%s %s: decode response %q: %v", method, path, rec.Body.String(), err)
		}
	}
	return rec
}

// TestHTTPBatchUpsert covers the bulk-ingest endpoint: many points in one
// request, then a search confirming they all landed.
func TestHTTPBatchUpsert(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"docs","config":{"dim":3,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1}}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}

	var pts []string
	for i := 1; i <= 20; i++ {
		pts = append(pts, fmt.Sprintf(`{"id":%d,"vector":[%d,0,0],"content":"chunk %d"}`, i, i, i))
	}
	body := `{"upsert":true,"points":[` + strings.Join(pts, ",") + `]}`
	var bres struct {
		Count int `json:"count"`
	}
	rec = do(t, h, "POST", "/v1/collections/docs/points/batch", body, &bres)
	if rec.Code != http.StatusOK || bres.Count != 20 {
		t.Fatalf("batch = %d count=%d (%s)", rec.Code, bres.Count, rec.Body)
	}

	var sres struct {
		Results []vector.Result `json:"results"`
	}
	do(t, h, "POST", "/v1/collections/docs/points/search", `{"query":[0,0,0],"k":5}`, &sres)
	if len(sres.Results) != 5 || sres.Results[0].ID != 1 {
		t.Fatalf("post-batch search = %+v", sres.Results)
	}
}

// TestHTTPScrollOrderBy covers the dense HTTP scroll order_by surface: a numeric
// order field paginated ASC/DESC returns docs in (value, id) order, missing-field
// points are excluded, start_from (numeric) is honored, and bad order_by input 400s.
func TestHTTPScrollOrderBy(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"docs","config":{"dim":3,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1}}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}
	// ids 1..6 with rank = 6..1 (reverse), id 7 has NO rank (excluded).
	pts := []string{
		`{"id":1,"vector":[1,0,0],"metadata":{"rank":{"kind":"int","int":6}}}`,
		`{"id":2,"vector":[2,0,0],"metadata":{"rank":{"kind":"int","int":5}}}`,
		`{"id":3,"vector":[3,0,0],"metadata":{"rank":{"kind":"int","int":4}}}`,
		`{"id":4,"vector":[4,0,0],"metadata":{"rank":{"kind":"int","int":3}}}`,
		`{"id":5,"vector":[5,0,0],"metadata":{"rank":{"kind":"int","int":2}}}`,
		`{"id":6,"vector":[6,0,0],"metadata":{"rank":{"kind":"int","int":1}}}`,
		`{"id":7,"vector":[7,0,0],"metadata":{"other":{"kind":"int","int":9}}}`,
	}
	rec = do(t, h, "POST", "/v1/collections/docs/points/batch", `{"upsert":true,"points":[`+strings.Join(pts, ",")+`]}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("batch = %d (%s)", rec.Code, rec.Body)
	}

	type doc struct {
		ID uint64 `json:"id"`
	}
	var out struct {
		Documents []doc `json:"documents"`
	}

	// ASC by rank ⇒ ids 6,5,4,3,2,1 (id 7 excluded). limit 0 = no cap (single page).
	do(t, h, "POST", "/v1/collections/docs/points/scroll", `{"limit":100,"order_by":{"key":"rank"}}`, &out)
	wantAsc := []uint64{6, 5, 4, 3, 2, 1}
	if len(out.Documents) != len(wantAsc) {
		t.Fatalf("ASC order_by returned %d docs, want %d (%+v)", len(out.Documents), len(wantAsc), out.Documents)
	}
	for i, w := range wantAsc {
		if out.Documents[i].ID != w {
			t.Fatalf("ASC order_by[%d]=%d want %d", i, out.Documents[i].ID, w)
		}
	}

	// DESC by rank ⇒ ids 1,2,3,4,5,6.
	out.Documents = nil
	do(t, h, "POST", "/v1/collections/docs/points/scroll", `{"limit":100,"order_by":{"key":"rank","desc":true}}`, &out)
	wantDesc := []uint64{1, 2, 3, 4, 5, 6}
	for i, w := range wantDesc {
		if out.Documents[i].ID != w {
			t.Fatalf("DESC order_by[%d]=%d want %d", i, out.Documents[i].ID, w)
		}
	}

	// start_from=3 (numeric, ASC) ⇒ ranks 3..6 ⇒ ids 4,3,2,1.
	out.Documents = nil
	do(t, h, "POST", "/v1/collections/docs/points/scroll", `{"limit":100,"order_by":{"key":"rank","start_from":3}}`, &out)
	wantStart := []uint64{4, 3, 2, 1}
	if len(out.Documents) != len(wantStart) {
		t.Fatalf("start_from returned %d docs, want %d (%+v)", len(out.Documents), len(wantStart), out.Documents)
	}
	for i, w := range wantStart {
		if out.Documents[i].ID != w {
			t.Fatalf("start_from order_by[%d]=%d want %d", i, out.Documents[i].ID, w)
		}
	}

	// Empty key with order_by present ⇒ 400.
	rec = do(t, h, "POST", "/v1/collections/docs/points/scroll", `{"limit":10,"order_by":{"key":""}}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("empty order_by key = %d, want 400 (%s)", rec.Code, rec.Body)
	}
}

// TestHTTPScrollOrderByString covers the dense HTTP scroll STRING order_by surface: a
// string order field paginated ASC/DESC returns docs in lexicographic (stringValue, id)
// order, the v3 cursor resumes across pages, and a bad order-kind combination (is_string
// + is_datetime) is rejected 400. Numeric order_by stays unaffected (TestHTTPScrollOrderBy).
func TestHTTPScrollOrderByString(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"docs","config":{"dim":3,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1}}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}
	// ids 1..5 with city values that sort lexicographically differently from id order;
	// "berlin" appears twice (ids 2 and 5) to exercise the (value, id) tiebreak.
	pts := []string{
		`{"id":1,"vector":[1,0,0],"metadata":{"city":{"kind":"string","str":"delhi"}}}`,
		`{"id":2,"vector":[2,0,0],"metadata":{"city":{"kind":"string","str":"berlin"}}}`,
		`{"id":3,"vector":[3,0,0],"metadata":{"city":{"kind":"string","str":"amsterdam"}}}`,
		`{"id":4,"vector":[4,0,0],"metadata":{"city":{"kind":"string","str":"cairo"}}}`,
		`{"id":5,"vector":[5,0,0],"metadata":{"city":{"kind":"string","str":"berlin"}}}`,
	}
	rec = do(t, h, "POST", "/v1/collections/docs/points/batch", `{"upsert":true,"points":[`+strings.Join(pts, ",")+`]}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("batch = %d (%s)", rec.Code, rec.Body)
	}

	type doc struct {
		ID uint64 `json:"id"`
	}
	type scrollResp struct {
		Documents  []doc  `json:"documents"`
		NextCursor string `json:"next_cursor"`
	}

	// ASC by city ⇒ amsterdam(3), berlin(2), berlin(5), cairo(4), delhi(1).
	var out scrollResp
	do(t, h, "POST", "/v1/collections/docs/points/scroll", `{"limit":100,"order_by":{"key":"city","is_string":true}}`, &out)
	wantAsc := []uint64{3, 2, 5, 4, 1}
	if len(out.Documents) != len(wantAsc) {
		t.Fatalf("ASC string order_by returned %d docs, want %d (%+v)", len(out.Documents), len(wantAsc), out.Documents)
	}
	for i, w := range wantAsc {
		if out.Documents[i].ID != w {
			t.Fatalf("ASC string order_by[%d]=%d want %d", i, out.Documents[i].ID, w)
		}
	}

	// DESC by city ⇒ delhi(1), cairo(4), berlin(2), berlin(5), amsterdam(3). DESC reverses
	// the key only; the id tiebreak stays ascending (2 before 5).
	out = scrollResp{}
	do(t, h, "POST", "/v1/collections/docs/points/scroll", `{"limit":100,"order_by":{"key":"city","is_string":true,"desc":true}}`, &out)
	wantDesc := []uint64{1, 4, 2, 5, 3}
	for i, w := range wantDesc {
		if out.Documents[i].ID != w {
			t.Fatalf("DESC string order_by[%d]=%d want %d", i, out.Documents[i].ID, w)
		}
	}

	// (Cursor resume across pages is exercised end-to-end in the root package's
	// TestStringOrderByAscDescPaged through the real coordinator; the bare HTTP test
	// dispatcher has no fan-out next_cursor, so paging is asserted there, not here.)

	// Bad combo: is_string + is_datetime ⇒ 400.
	rec = do(t, h, "POST", "/v1/collections/docs/points/scroll", `{"limit":10,"order_by":{"key":"city","is_string":true,"is_datetime":true}}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("is_string+is_datetime = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	// Bad combo: is_string + start_from ⇒ 400.
	rec = do(t, h, "POST", "/v1/collections/docs/points/scroll", `{"limit":10,"order_by":{"key":"city","is_string":true,"start_from":1}}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("is_string+start_from = %d, want 400 (%s)", rec.Code, rec.Body)
	}
}

// TestHTTPScrollOrderByMultiKey covers the dense HTTP scroll MULTI-KEY order_by surface:
// the repeated order spec (order_by + tail_keys) paginates by a composite (price desc,
// name asc, id asc) total order, and the v4 cursor resumes EXACTLY across pages (the full
// paged sequence equals the single-page composite order). A single-element tail_keys list
// is the additive multi-key path; an absent tail_keys is the byte-identical single-key path
// (covered by TestHTTPScrollOrderBy).
func TestHTTPScrollOrderByMultiKey(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"docs","config":{"dim":3,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1}}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}
	// 6 points with deliberate price TIES so the secondary name key + id tiebreak decide:
	//   id1 price=10 name="b"   id2 price=10 name="a"   id3 price=10 name="a"
	//   id4 price=20 name="z"   id5 price=20 name="z"   id6 price=5  name="m"
	// Composite (price desc, name asc, id asc): 4,5 (p20,z) ; 2,3 (p10,a) ; 1 (p10,b) ; 6 (p5,m).
	pts := []string{
		`{"id":1,"vector":[1,0,0],"metadata":{"price":{"kind":"int","int":10},"name":{"kind":"string","str":"b"}}}`,
		`{"id":2,"vector":[2,0,0],"metadata":{"price":{"kind":"int","int":10},"name":{"kind":"string","str":"a"}}}`,
		`{"id":3,"vector":[3,0,0],"metadata":{"price":{"kind":"int","int":10},"name":{"kind":"string","str":"a"}}}`,
		`{"id":4,"vector":[4,0,0],"metadata":{"price":{"kind":"int","int":20},"name":{"kind":"string","str":"z"}}}`,
		`{"id":5,"vector":[5,0,0],"metadata":{"price":{"kind":"int","int":20},"name":{"kind":"string","str":"z"}}}`,
		`{"id":6,"vector":[6,0,0],"metadata":{"price":{"kind":"int","int":5},"name":{"kind":"string","str":"m"}}}`,
	}
	rec = do(t, h, "POST", "/v1/collections/docs/points/batch", `{"upsert":true,"points":[`+strings.Join(pts, ",")+`]}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("batch = %d (%s)", rec.Code, rec.Body)
	}

	type doc struct {
		ID uint64 `json:"id"`
	}
	var out struct {
		Documents  []doc  `json:"documents"`
		NextCursor string `json:"next_cursor"`
	}

	// Composite order: price DESC (primary) then name ASC (tail) then id ASC.
	//   4,5 (p20,z) ; 2,3 (p10,a) ; 1 (p10,b) ; 6 (p5,m).
	orderBody := `"order_by":{"key":"price","desc":true,"tail_keys":[{"key":"name","is_string":true}]}`
	want := []uint64{4, 5, 2, 3, 1, 6}

	// Single page (limit covers all) ⇒ the full composite order through the repeated
	// order_by spec (the HTTP tail_keys array → vector.OrderBy.Tail → tuple sort). This is
	// the HTTP multi-key round-trip; the v4 cursor RESUME across pages is proven end-to-end
	// by the embedded coordinator test (TestMultiKeyOrderByPartitionInvariance) — this
	// single-shard handler test harness does not run the coordinator that emits cursors.
	do(t, h, "POST", "/v1/collections/docs/points/scroll", `{"limit":100,`+orderBody+`}`, &out)
	if len(out.Documents) != len(want) {
		t.Fatalf("multi-key order_by returned %d docs, want %d (%+v)", len(out.Documents), len(want), out.Documents)
	}
	for i, w := range want {
		if out.Documents[i].ID != w {
			t.Fatalf("multi-key order_by[%d]=%d want %d (full=%+v)", i, out.Documents[i].ID, w, out.Documents)
		}
	}

	// A tail key that is NOT marked is_string but names a string field ⇒ that key reads as
	// numeric, so EVERY row (whose name is non-numeric) is EXCLUDED — proving the tail
	// key's kind is honored on the wire (not inferred). Result: 0 docs.
	out.Documents = nil
	do(t, h, "POST", "/v1/collections/docs/points/scroll",
		`{"limit":100,"order_by":{"key":"price","desc":true,"tail_keys":[{"key":"name"}]}}`, &out)
	if len(out.Documents) != 0 {
		t.Fatalf("numeric-typed string tail key returned %d docs, want 0 (kind not honored)", len(out.Documents))
	}
}

// TestHTTPBulkLoad covers the concurrent bulk-load path: stage points, build the
// index in one pass, then search — verifying the staged vectors are indexed and
// searchable with correct nearest-neighbor results.
func TestHTTPBulkLoad(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"docs","config":{"dim":3,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1}}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}

	var pts []string
	for i := 1; i <= 40; i++ {
		pts = append(pts, fmt.Sprintf(`{"id":%d,"vector":[%d,0,0]}`, i, i))
	}
	rec = do(t, h, "POST", "/v1/collections/docs/points/bulk", `{"points":[`+strings.Join(pts, ",")+`]}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("stage = %d (%s)", rec.Code, rec.Body)
	}
	// Before build, the index is empty — a search returns nothing.
	var pre struct {
		Results []vector.Result `json:"results"`
	}
	do(t, h, "POST", "/v1/collections/docs/points/search", `{"query":[1,0,0],"k":5}`, &pre)
	if len(pre.Results) != 0 {
		t.Fatalf("pre-build search returned %d, want 0 (nothing built yet)", len(pre.Results))
	}
	// Build, then search returns the staged points, nearest first.
	rec = do(t, h, "POST", "/v1/collections/docs/points/bulk/build", `{}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("build = %d (%s)", rec.Code, rec.Body)
	}
	var sres struct {
		Results []vector.Result `json:"results"`
	}
	do(t, h, "POST", "/v1/collections/docs/points/search", `{"query":[1,0,0],"k":5}`, &sres)
	if len(sres.Results) != 5 || sres.Results[0].ID != 1 {
		t.Fatalf("post-build search = %+v, want id 1 first of 5", sres.Results)
	}
}

func TestHTTPHealth(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	rec := do(t, h, "GET", "/v1/health", "", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("health = %d, want 200 (%s)", rec.Code, rec.Body)
	}
}

// TestHTTPRAGRoundtrip drives the full RAG surface over REST: create a
// collection, upsert chunks across documents, then exercise search, search/docs,
// search/groups, delete, and delete-by-filter.
func TestHTTPRAGRoundtrip(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"docs","config":{"dim":3,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1}}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d, want 201 (%s)", rec.Code, rec.Body)
	}

	// Six chunks, two per document (doc = id%3 mapped 1..3), increasing distance.
	for i := 1; i <= 6; i++ {
		doc := (i + 1) / 2
		body := fmt.Sprintf(`{"id":%d,"vector":[%d,0,0],"content":"chunk %d","upsert":true,"metadata":{"doc":{"kind":"int","int":%d}}}`, i, i, i, doc)
		rec := do(t, h, "POST", "/v1/collections/docs/points", body, nil)
		if rec.Code != http.StatusOK {
			t.Fatalf("upsert %d = %d (%s)", i, rec.Code, rec.Body)
		}
	}

	// Plain search returns ids ascending by distance.
	var sres struct {
		Results []vector.Result `json:"results"`
	}
	rec = do(t, h, "POST", "/v1/collections/docs/points/search", `{"query":[0,0,0],"k":3}`, &sres)
	if rec.Code != http.StatusOK || len(sres.Results) != 3 || sres.Results[0].ID != 1 {
		t.Fatalf("search = %d %+v", rec.Code, sres.Results)
	}

	// search/docs carries content.
	var dres struct {
		Documents []vector.Document `json:"documents"`
	}
	do(t, h, "POST", "/v1/collections/docs/points/search/docs", `{"query":[0,0,0],"k":2}`, &dres)
	if len(dres.Documents) == 0 || dres.Documents[0].Content != "chunk 1" {
		t.Fatalf("search/docs = %+v", dres.Documents)
	}

	// search/groups returns the top documents (best chunk each).
	var gres struct {
		Groups []vector.Group `json:"groups"`
	}
	do(t, h, "POST", "/v1/collections/docs/points/search/groups",
		`{"query":[0,0,0],"k":2,"group_by":"doc","group_size":1}`, &gres)
	if len(gres.Groups) != 2 || gres.Groups[0].Key.Int != 1 || gres.Groups[1].Key.Int != 2 {
		t.Fatalf("search/groups = %+v", gres.Groups)
	}
	if len(gres.Groups[0].Hits) != 1 || gres.Groups[0].Hits[0].ID != 1 {
		t.Fatalf("group0 hits = %+v", gres.Groups[0].Hits)
	}

	// Delete a single point.
	var del struct {
		Deleted bool `json:"deleted"`
	}
	rec = do(t, h, "DELETE", "/v1/collections/docs/points/1", "", &del)
	if rec.Code != http.StatusOK || !del.Deleted {
		t.Fatalf("delete = %d %+v", rec.Code, del)
	}

	// Delete-by-filter purges doc 2 (ids 3,4).
	var dbf struct {
		Deleted int `json:"deleted"`
	}
	rec = do(t, h, "POST", "/v1/collections/docs/points/delete",
		`{"filter":{"op":"eq","field":"doc","value":{"kind":"int","int":2}}}`, &dbf)
	if rec.Code != http.StatusOK || dbf.Deleted != 2 {
		t.Fatalf("delete-by-filter = %d %+v", rec.Code, dbf)
	}
}

// TestHTTPCreateOPQ proves the opq JSON field is parsed and threaded: a create
// with opq=true on a PQ-HNSW config succeeds (200); opq=true WITHOUT a PQ mode is
// rejected with 400 (the cfg.Validate OPQ gate surfaced fail-loud).
func TestHTTPCreateOPQ(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	// opq=true on a PQ-HNSW (quant=="pq") config → 200.
	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"opqpq","config":{"dim":8,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"quant":"pq","quant_pq_m":8,"opq":true}}`, nil)
	if rec.Code != http.StatusOK && rec.Code != http.StatusCreated {
		t.Fatalf("create pq+opq = %d, want 200/201 (%s)", rec.Code, rec.Body)
	}

	// opq=true with NO PQ mode → 400 (ErrInvalidOPQ surfaced fail-loud).
	rec = do(t, h, "POST", "/v1/collections",
		`{"name":"opqnopq","config":{"dim":8,"metric":"l2","m":8,"opq":true}}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("opq without PQ = %d, want 400 (%s)", rec.Code, rec.Body)
	}
}

// TestHTTPCreatePQDropVecs proves the pq_drop_vecs JSON field is parsed and
// threaded: a create with pq_drop_vecs=true on a PQ-HNSW config succeeds (200);
// pq_drop_vecs=true WITHOUT quant=="pq" is rejected with 400 (the cfg.Validate
// PQDropVecs gate surfaced fail-loud).
func TestHTTPCreatePQDropVecs(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	// pq_drop_vecs=true on a PQ-HNSW (quant=="pq") config → 200.
	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"droppq","config":{"dim":8,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"quant":"pq","quant_pq_m":8,"pq_drop_vecs":true}}`, nil)
	if rec.Code != http.StatusOK && rec.Code != http.StatusCreated {
		t.Fatalf("create pq+pq_drop_vecs = %d, want 200/201 (%s)", rec.Code, rec.Body)
	}

	// pq_drop_vecs=true with NO PQ mode → 400 (ErrInvalidPQDropVecs fail-loud).
	rec = do(t, h, "POST", "/v1/collections",
		`{"name":"dropnopq","config":{"dim":8,"metric":"l2","m":8,"pq_drop_vecs":true}}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("pq_drop_vecs without PQ = %d, want 400 (%s)", rec.Code, rec.Body)
	}
}

// TestHTTPCreateIVFTrainThreshold proves the dense ivf_train_threshold JSON field
// is parsed and threaded into vector.Config (Gap 1: closing the dense create-codec
// asymmetry vs named/MV). A create with index_type="ivf" + ivf_train_threshold set
// succeeds, and the friendly collectionConfig.toConfig() maps the field onto
// vector.Config.IVFTrainThreshold.
func TestHTTPCreateIVFTrainThreshold(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"ivfthr","config":{"dim":8,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"index_type":"ivf","ivf_nlist":4,"ivf_nprobe":2,"ivf_train_threshold":1500}}`, nil)
	if rec.Code != http.StatusOK && rec.Code != http.StatusCreated {
		t.Fatalf("create ivf+ivf_train_threshold = %d, want 200/201 (%s)", rec.Code, rec.Body)
	}

	// The friendly config maps the field onto vector.Config.
	cfg, errMsg := collectionConfig{Dim: 8, Metric: "l2", IndexType: "ivf", IVFTrainThreshold: 1500}.toConfig()
	if errMsg != "" {
		t.Fatalf("toConfig: %s", errMsg)
	}
	if cfg.IVFTrainThreshold != 1500 {
		t.Fatalf("toConfig IVFTrainThreshold = %d, want 1500", cfg.IVFTrainThreshold)
	}
	// A negative threshold is rejected fail-loud by the engine Validate on create.
	rec = do(t, h, "POST", "/v1/collections",
		`{"name":"ivfthrneg","config":{"dim":8,"metric":"l2","index_type":"ivf","ivf_train_threshold":-1}}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("negative ivf_train_threshold = %d, want 400 (%s)", rec.Code, rec.Body)
	}
}

func TestHTTPErrorMapping(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	do(t, h, "POST", "/v1/collections",
		`{"name":"docs","config":{"dim":3,"metric":"cosine","m":8,"ef_construction":50,"ef_search":32}}`, nil)

	// Unknown collection → 404.
	rec := do(t, h, "POST", "/v1/collections/nope/points/search", `{"query":[1,2,3],"k":1}`, nil)
	if rec.Code != http.StatusNotFound {
		t.Errorf("unknown collection = %d, want 404 (%s)", rec.Code, rec.Body)
	}
	// Dimension mismatch → 400.
	rec = do(t, h, "POST", "/v1/collections/docs/points/search", `{"query":[1,2],"k":1}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("dim mismatch = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	// Empty group field → 400.
	rec = do(t, h, "POST", "/v1/collections/docs/points/search/groups", `{"query":[1,2,3],"k":1}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("empty group_by = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	// Unknown metric → 400 (validated before dispatch).
	rec = do(t, h, "POST", "/v1/collections", `{"name":"x","config":{"dim":3,"metric":"bogus"}}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("bad metric = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	// Malformed JSON → 400.
	rec = do(t, h, "POST", "/v1/collections/docs/points/search", `{not json`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("bad json = %d, want 400", rec.Code)
	}
}

// TestStatusForErrorWriteConsistency proves the write-consistency barrier-miss
// error (carried as a "cluster: write " message prefix, like the alias-mapping
// tests construct the error string directly) maps to HTTP 504 — the durable-but-
// under-replicated outcome — and not to the 500 default or the 503 leader bucket.
func TestStatusForErrorWriteConsistency(t *testing.T) {
	// Mirror the real *cluster.ErrWriteConsistency message prefix exactly.
	err := errors.New("cluster: write committed at quorum but consistency factor 3 not met (2 replicas applied at index 42)")
	if got := statusForError(err); got != http.StatusGatewayTimeout {
		t.Fatalf("statusForError(write-consistency) = %d, want 504", got)
	}
	// A plain internal error still falls through to 500 (no accidental collision).
	if got := statusForError(errors.New("boom")); got != http.StatusInternalServerError {
		t.Fatalf("statusForError(generic) = %d, want 500", got)
	}
}

func TestCollectionConfigPartitions(t *testing.T) {
	cfg, errMsg := collectionConfig{Dim: 8, Metric: "cosine", Partitions: 4}.toConfig()
	if errMsg != "" {
		t.Fatalf("toConfig: %s", errMsg)
	}
	if cfg.Partitions != 4 {
		t.Fatalf("Partitions = %d, want 4", cfg.Partitions)
	}
	if _, errMsg := (collectionConfig{Dim: 8, Metric: "cosine", Partitions: -1}).toConfig(); errMsg == "" {
		t.Fatalf("toConfig with Partitions=-1: expected error string, got empty")
	}
}

func TestHTTPCreateRejectsNegativePartitions(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"ok","config":{"dim":8,"metric":"cosine","partitions":-1}}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("negative partitions = %d, want 400 (%s)", rec.Code, rec.Body)
	}
}

func TestHTTPCreateRejectsReservedName(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	for _, name := range []string{"bad#name", "bad@name"} {
		rec := do(t, h, "POST", "/v1/collections",
			fmt.Sprintf(`{"name":%q,"config":{"dim":3}}`, name), nil)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("%q = %d, want 400 (%s)", name, rec.Code, rec.Body)
		}
	}
}

func TestHTTPMVCreateRejectsReservedName(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	for _, name := range []string{"bad#name", "bad@name"} {
		rec := do(t, h, "POST", "/v1/multivector/"+name, `{"dim":8}`, nil)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("%q = %d, want 400 (%s)", name, rec.Code, rec.Body)
		}
	}
}

func TestHTTPMVCreateRejectsNegativePartitions(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	rec := do(t, h, "POST", "/v1/multivector/mv", `{"dim":8,"partitions":-1}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("negative partitions = %d, want 400 (%s)", rec.Code, rec.Body)
	}
}

func TestHTTPMVCreatePartitions(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	rec := do(t, h, "POST", "/v1/multivector/mv", `{"dim":8,"partitions":4}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d, want 201 (%s)", rec.Code, rec.Body)
	}
}

// TestHTTPMVCreateIVF proves an IVF / IVF-PQ inner-index MV create succeeds
// end-to-end through the HTTP edge into the engine (Validate accepts dim%m==0).
func TestHTTPMVCreateIVF(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	body := `{"dim":32,"index_type":"ivf","ivf_nlist":16,"ivf_nprobe":4,"ivf_pq":true,"ivf_pq_m":8,"ivf_rerank":true,"ivf_train_threshold":1000}`
	rec := do(t, h, "POST", "/v1/multivector/mvivf", body, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("IVF create = %d, want 201 (%s)", rec.Code, rec.Body)
	}
}

// TestHTTPMVCreatePQDropVecs proves the MV pq_drop_vecs JSON field is parsed and
// threaded: a create with pq_drop_vecs=true on a PQ-HNSW (quant=="pq") inner index
// succeeds (201); pq_drop_vecs=true WITHOUT quant=="pq" is rejected with 400 (the
// inner Config.Validate ErrInvalidPQDropVecs surfaced fail-loud).
func TestHTTPMVCreatePQDropVecs(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	// pq_drop_vecs=true on a PQ-HNSW inner index → 201.
	rec := do(t, h, "POST", "/v1/multivector/mvpqdrop",
		`{"dim":8,"quant":"pq","ivf_train_threshold":500,"pq_drop_vecs":true}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("MV create pq+pq_drop_vecs = %d, want 201 (%s)", rec.Code, rec.Body)
	}
	// pq_drop_vecs=true with NO PQ mode → 400 (ErrInvalidPQDropVecs fail-loud).
	rec = do(t, h, "POST", "/v1/multivector/mvpqdropbad",
		`{"dim":8,"pq_drop_vecs":true}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("MV pq_drop_vecs without quant=pq = %d, want 400 (%s)", rec.Code, rec.Body)
	}
}

// TestHTTPMVCreateRejectsBadIndexType proves an unknown index_type fails loud
// (400) at the HTTP edge.
func TestHTTPMVCreateRejectsBadIndexType(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	rec := do(t, h, "POST", "/v1/multivector/mv", `{"dim":8,"index_type":"bogus"}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("bad index_type = %d, want 400 (%s)", rec.Code, rec.Body)
	}
}

// TestHTTPMVCreateRejectsBadIVFPQ proves an engine Validate rejection (dim not
// divisible by ivf_pq_m) surfaces as 400 at the HTTP edge.
func TestHTTPMVCreateRejectsBadIVFPQ(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	rec := do(t, h, "POST", "/v1/multivector/mv", `{"dim":30,"index_type":"ivf","ivf_pq":true,"ivf_pq_m":8}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("bad ivf_pq_m = %d, want 400 (%s)", rec.Code, rec.Body)
	}
}

// recordingDispatcher captures the last op name and args it was asked to run,
// returning a canned result. It lets the resplit/cleanup tests assert the exact
// op name + decoded args dispatched at the edge without needing those ops
// registered in the builtin registry (they live behind the binary/gRPC fronts).
type recordingDispatcher struct {
	calls  []recordedCall
	result []byte
	err    error
}

type recordedCall struct {
	name string
	args []byte
}

func (d *recordingDispatcher) Call(name string, args []byte) ([]byte, error) {
	cp := make([]byte, len(args))
	copy(cp, args)
	d.calls = append(d.calls, recordedCall{name: name, args: cp})
	return d.result, d.err
}

func (d *recordingDispatcher) LeaderAddr() string { return "" }

func TestHTTPResplitDense(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})

	var out struct {
		Name          string `json:"name"`
		NewPartitions int    `json:"new_partitions"`
	}
	rec := do(t, h, "POST", "/v1/collections/docs/resplit", `{"new_partitions":8}`, &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("resplit = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 {
		t.Fatalf("dispatched %d ops, want 1", len(disp.calls))
	}
	if disp.calls[0].name != "vector_resplit" {
		t.Fatalf("op = %q, want vector_resplit", disp.calls[0].name)
	}
	col, newP, err := ops.DecodeResplitArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode args: %v", err)
	}
	if col != "docs" || newP != 8 {
		t.Fatalf("decoded args = (%q, %d), want (docs, 8)", col, newP)
	}
	if out.Name != "docs" || out.NewPartitions != 8 {
		t.Fatalf("body = %+v, want {docs 8}", out)
	}
}

func TestHTTPResplitRejectsNegative(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})
	rec := do(t, h, "POST", "/v1/collections/docs/resplit", `{"new_partitions":-1}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("negative new_partitions = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 0 {
		t.Fatalf("dispatcher called %d times on negative input, want 0", len(disp.calls))
	}
}

func TestHTTPResplitCleanupDense(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeResplitCleanupResult(3)}
	h := Handler(disp, Options{})

	var out struct {
		Name    string `json:"name"`
		Dropped int    `json:"dropped"`
	}
	rec := do(t, h, "POST", "/v1/collections/docs/resplit/cleanup", "", &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("cleanup = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_resplit_cleanup" {
		t.Fatalf("calls = %+v, want one vector_resplit_cleanup", disp.calls)
	}
	col, err := ops.DecodeResplitCleanupArgs(disp.calls[0].args)
	if err != nil || col != "docs" {
		t.Fatalf("decode cleanup args = (%q, %v), want (docs, nil)", col, err)
	}
	if out.Name != "docs" || out.Dropped != 3 {
		t.Fatalf("body = %+v, want {docs 3}", out)
	}
}

func TestHTTPMVResplit(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})

	var out struct {
		Name          string `json:"name"`
		NewPartitions int    `json:"new_partitions"`
	}
	rec := do(t, h, "POST", "/v1/multivector/docs/resplit", `{"new_partitions":4}`, &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("mv resplit = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_mv_resplit" {
		t.Fatalf("calls = %+v, want one vector_mv_resplit", disp.calls)
	}
	col, newP, err := ops.DecodeResplitArgs(disp.calls[0].args)
	if err != nil || col != "docs" || newP != 4 {
		t.Fatalf("decoded args = (%q, %d, %v), want (docs, 4, nil)", col, newP, err)
	}
	if out.Name != "docs" || out.NewPartitions != 4 {
		t.Fatalf("body = %+v, want {docs 4}", out)
	}
}

func TestHTTPMVResplitRejectsNegative(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})
	rec := do(t, h, "POST", "/v1/multivector/docs/resplit", `{"new_partitions":-1}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("negative new_partitions = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 0 {
		t.Fatalf("dispatcher called %d times on negative input, want 0", len(disp.calls))
	}
}

func TestHTTPMVResplitCleanup(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeResplitCleanupResult(5)}
	h := Handler(disp, Options{})

	var out struct {
		Name    string `json:"name"`
		Dropped int    `json:"dropped"`
	}
	rec := do(t, h, "POST", "/v1/multivector/docs/resplit/cleanup", "", &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("mv cleanup = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_mv_resplit_cleanup" {
		t.Fatalf("calls = %+v, want one vector_mv_resplit_cleanup", disp.calls)
	}
	col, err := ops.DecodeResplitCleanupArgs(disp.calls[0].args)
	if err != nil || col != "docs" {
		t.Fatalf("decode cleanup args = (%q, %v), want (docs, nil)", col, err)
	}
	if out.Name != "docs" || out.Dropped != 5 {
		t.Fatalf("body = %+v, want {docs 5}", out)
	}
}

func TestHTTPReshardDense(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})

	var out struct {
		Name          string `json:"name"`
		NewPartitions int    `json:"new_partitions"`
	}
	rec := do(t, h, "POST", "/v1/collections/docs/reshard", `{"new_partitions":8}`, &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("reshard = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 {
		t.Fatalf("dispatched %d ops, want 1", len(disp.calls))
	}
	if disp.calls[0].name != "vector_reshard" {
		t.Fatalf("op = %q, want vector_reshard", disp.calls[0].name)
	}
	col, newP, err := ops.DecodeReshardArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode args: %v", err)
	}
	if col != "docs" || newP != 8 {
		t.Fatalf("decoded args = (%q, %d), want (docs, 8)", col, newP)
	}
	if out.Name != "docs" || out.NewPartitions != 8 {
		t.Fatalf("body = %+v, want {docs 8}", out)
	}
}

func TestHTTPReshardRejectsNegative(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})
	rec := do(t, h, "POST", "/v1/collections/docs/reshard", `{"new_partitions":-1}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("negative new_partitions = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 0 {
		t.Fatalf("dispatcher called %d times on negative input, want 0", len(disp.calls))
	}
}

func TestHTTPReshardAbort(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})

	var out struct {
		Name string `json:"name"`
	}
	rec := do(t, h, "POST", "/v1/collections/docs/reshard/abort", "", &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("reshard abort = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_reshard_abort" {
		t.Fatalf("calls = %+v, want one vector_reshard_abort", disp.calls)
	}
	col, err := ops.DecodeReshardAbortArgs(disp.calls[0].args)
	if err != nil || col != "docs" {
		t.Fatalf("decode abort args = (%q, %v), want (docs, nil)", col, err)
	}
	if out.Name != "docs" {
		t.Fatalf("body = %+v, want {docs}", out)
	}
}

func TestHTTPMVReshard(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})

	var out struct {
		Name          string `json:"name"`
		NewPartitions int    `json:"new_partitions"`
	}
	rec := do(t, h, "POST", "/v1/multivector/docs/reshard", `{"new_partitions":4}`, &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("mv reshard = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_mv_reshard" {
		t.Fatalf("calls = %+v, want one vector_mv_reshard", disp.calls)
	}
	col, newP, err := ops.DecodeReshardArgs(disp.calls[0].args)
	if err != nil || col != "docs" || newP != 4 {
		t.Fatalf("decoded args = (%q, %d, %v), want (docs, 4, nil)", col, newP, err)
	}
	if out.Name != "docs" || out.NewPartitions != 4 {
		t.Fatalf("body = %+v, want {docs 4}", out)
	}
}

func TestHTTPMVReshardAbort(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})

	var out struct {
		Name string `json:"name"`
	}
	rec := do(t, h, "POST", "/v1/multivector/docs/reshard/abort", "", &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("mv reshard abort = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_mv_reshard_abort" {
		t.Fatalf("calls = %+v, want one vector_mv_reshard_abort", disp.calls)
	}
	col, err := ops.DecodeReshardAbortArgs(disp.calls[0].args)
	if err != nil || col != "docs" {
		t.Fatalf("decode abort args = (%q, %v), want (docs, nil)", col, err)
	}
	if out.Name != "docs" {
		t.Fatalf("body = %+v, want {docs}", out)
	}
}

func TestHTTPAuth(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()
	vstore, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer vstore.Close()
	disp := &testDispatcher{reg: reg, tx: ops.NewTxContextWithVectors(c, vstore)}

	// Only the token "secret" is accepted (a coarse static gate; the granular
	// per-collection RBAC matrix is exercised in TestHTTPRBACScopes below).
	auth := func(req authz.AuthRequest) bool { return req.Token == "secret" }
	h := Handler(disp, Options{Authenticator: auth})

	// No token → 401.
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/collections",
		strings.NewReader(`{"name":"d","config":{"dim":3}}`)))
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("no token = %d, want 401", rec.Code)
	}

	// Correct token → proceeds (201).
	r := httptest.NewRequest("POST", "/v1/collections", bytes.NewReader([]byte(`{"name":"d","config":{"dim":3}}`)))
	r.Header.Set("Authorization", "Bearer secret")
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, r)
	if rec.Code != http.StatusCreated {
		t.Errorf("good token = %d, want 201 (%s)", rec.Code, rec.Body)
	}

	// Health bypasses auth.
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/v1/health", nil))
	if rec.Code != http.StatusOK {
		t.Errorf("health w/ auth = %d, want 200", rec.Code)
	}
}

// mustAddKey registers an API key or fails the test (AddKey requires a non-empty
// Tenant and validates Permissions, so a silent _ = AddKey would mask a typo).
func mustAddKey(t *testing.T, reg *vector.KeyRegistry, k vector.APIKey) {
	t.Helper()
	if err := reg.AddKey(k); err != nil {
		t.Fatalf("AddKey(%q): %v", k.Token, err)
	}
}

// TestHTTPRBACScopes exercises the granular per-collection RBAC matrix over the
// real HTTP transport with a live dispatcher: a read:default/docs key can search
// docs but NOT insert docs and NOT search a different collection; an admin:* key
// can create/drop; an unknown/empty token is denied. This proves authorize runs
// with the op ARGS threaded through so the authorizer sees the target collection
// (not just the op name), and that a denied op returns 401 and never reaches the
// engine.
func TestHTTPRBACScopes(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()
	vstore, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer vstore.Close()
	disp := &testDispatcher{reg: reg, tx: ops.NewTxContextWithVectors(c, vstore)}

	keyReg, err := vector.OpenKeyRegistry(filepath.Join(t.TempDir(), "keys.json"))
	if err != nil {
		t.Fatal(err)
	}
	// reader: may read default/docs only. writer: may read+write default/docs.
	// adminer: superuser.
	mustAddKey(t, keyReg, vector.APIKey{Token: "k_read", Tenant: "t", Scopes: []string{"read:default/docs"}})
	mustAddKey(t, keyReg, vector.APIKey{Token: "k_write", Tenant: "t", Scopes: []string{"read:default/docs", "write:default/docs"}})
	mustAddKey(t, keyReg, vector.APIKey{Token: "k_admin", Tenant: "t", Scopes: []string{"*:*"}})

	h := Handler(disp, Options{Authenticator: authz.NewRBACAuthenticator(keyReg, reg, "")})

	req := func(token, method, path, body string) int {
		var r *http.Request
		if body == "" {
			r = httptest.NewRequest(method, path, nil)
		} else {
			r = httptest.NewRequest(method, path, strings.NewReader(body))
		}
		if token != "" {
			r.Header.Set("Authorization", "Bearer "+token)
		}
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, r)
		return rec.Code
	}

	// admin creates the collection (read/write keys cannot — create is admin).
	if code := req("k_read", "POST", "/v1/collections", `{"name":"docs","config":{"dim":3}}`); code != http.StatusUnauthorized {
		t.Errorf("read key create = %d, want 401", code)
	}
	if code := req("k_admin", "POST", "/v1/collections", `{"name":"docs","config":{"dim":3}}`); code != http.StatusCreated {
		t.Fatalf("admin create = %d, want 201", code)
	}

	// reader CAN search docs.
	if code := req("k_read", "POST", "/v1/collections/docs/points/search", `{"query":[1,2,3],"k":1}`); code != http.StatusOK {
		t.Errorf("read key search docs = %d, want 200", code)
	}
	// reader CANNOT insert docs (write denied) — 401, never reaches engine.
	if code := req("k_read", "POST", "/v1/collections/docs/points", `{"id":1,"vector":[1,2,3]}`); code != http.StatusUnauthorized {
		t.Errorf("read key insert docs = %d, want 401", code)
	}
	// reader CANNOT search a different collection (scope is docs-only).
	if code := req("k_read", "POST", "/v1/collections/other/points/search", `{"query":[1,2,3],"k":1}`); code != http.StatusUnauthorized {
		t.Errorf("read key search other = %d, want 401", code)
	}
	// writer CAN insert docs.
	if code := req("k_write", "POST", "/v1/collections/docs/points", `{"id":1,"vector":[1,2,3]}`); code != http.StatusCreated && code != http.StatusOK {
		t.Errorf("write key insert docs = %d, want 200/201", code)
	}
	// no token → 401.
	if code := req("", "POST", "/v1/collections/docs/points/search", `{"query":[1,2,3],"k":1}`); code != http.StatusUnauthorized {
		t.Errorf("no token search = %d, want 401", code)
	}
	// admin can drop.
	if code := req("k_admin", "DELETE", "/v1/collections/docs", ""); code != http.StatusOK && code != http.StatusNoContent {
		t.Errorf("admin drop = %d, want 200/204", code)
	}
}

// TestHTTPRBACWCWriteUsesInnerOp proves the __wc__ write path authorizes by the
// INNER op + collection: a write-scoped key's WC-write on its collection passes,
// while a read-only key's WC-write is denied (401) before any envelope is built.
func TestHTTPRBACWCWriteUsesInnerOp(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()
	vstore, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer vstore.Close()
	disp := &testDispatcher{reg: reg, tx: ops.NewTxContextWithVectors(c, vstore)}

	keyReg, err := vector.OpenKeyRegistry(filepath.Join(t.TempDir(), "keys.json"))
	if err != nil {
		t.Fatal(err)
	}
	mustAddKey(t, keyReg, vector.APIKey{Token: "k_read", Tenant: "t", Scopes: []string{"read:default/docs"}})
	mustAddKey(t, keyReg, vector.APIKey{Token: "k_admin", Tenant: "t", Scopes: []string{"*:*"}})

	h := Handler(disp, Options{Authenticator: authz.NewRBACAuthenticator(keyReg, reg, "")})
	req := func(token, method, path, body string) int {
		r := httptest.NewRequest(method, path, strings.NewReader(body))
		r.Header.Set("Authorization", "Bearer "+token)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, r)
		return rec.Code
	}
	if code := req("k_admin", "POST", "/v1/collections", `{"name":"docs","config":{"dim":3}}`); code != http.StatusCreated {
		t.Fatalf("admin create = %d, want 201", code)
	}
	// A WC write (write_consistency_factor>0) by a read-only key → 401 (inner op
	// vector_insert is write-classified; the envelope is never built).
	if code := req("k_read", "POST", "/v1/collections/docs/points",
		`{"id":1,"vector":[1,2,3],"write_consistency_factor":1}`); code != http.StatusUnauthorized {
		t.Errorf("read key WC insert = %d, want 401", code)
	}
}

// TestHTTPSearchSurfacesDegraded asserts the search endpoint decodes the
// degraded trailer and adds "degraded" + "missing" to the JSON response.
func TestHTTPSearchSurfacesDegraded(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeVectorSearchResultsDegraded(
		[]vector.Result{{ID: 1, Distance: 0.5}}, true, []uint16{2})}
	h := Handler(disp, Options{})

	var out struct {
		Results  []vector.Result `json:"results"`
		Degraded bool            `json:"degraded"`
		Missing  []int           `json:"missing"`
	}
	rec := do(t, h, "POST", "/v1/collections/docs/points/search", `{"query":[1,2],"k":1}`, &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("search = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if !out.Degraded {
		t.Fatalf("degraded = false, want true (%s)", rec.Body)
	}
	if len(out.Missing) != 1 || out.Missing[0] != 2 {
		t.Fatalf("missing = %v, want [2]", out.Missing)
	}
	if len(out.Results) != 1 || out.Results[0].ID != 1 {
		t.Fatalf("results = %v, want one id=1", out.Results)
	}
}

// TestHTTPSearchNonDegraded asserts a legacy (no-trailer) body yields
// degraded:false and an empty missing array.
func TestHTTPSearchNonDegraded(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeVectorSearchResults(
		[]vector.Result{{ID: 9, Distance: 0.2}})}
	h := Handler(disp, Options{})

	var out struct {
		Degraded bool  `json:"degraded"`
		Missing  []int `json:"missing"`
	}
	rec := do(t, h, "POST", "/v1/collections/docs/points/search", `{"query":[1,2],"k":1}`, &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("search = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if out.Degraded {
		t.Fatalf("degraded = true, want false")
	}
	if len(out.Missing) != 0 {
		t.Fatalf("missing = %v, want empty", out.Missing)
	}
}

// TestHTTPMVSearchSurfacesDegraded covers a second response type (multivector
// search) through its degraded encoder.
func TestHTTPMVSearchSurfacesDegraded(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeMVResultsDegraded(
		[]vector.MultiResult{{ID: 3, Score: 1.5}}, true, []uint16{1, 4})}
	h := Handler(disp, Options{})

	var out struct {
		Degraded bool  `json:"degraded"`
		Missing  []int `json:"missing"`
	}
	rec := do(t, h, "POST", "/v1/multivector/mv/search", `{"query":[[1,2]],"k":1}`, &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("mv search = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if !out.Degraded {
		t.Fatalf("degraded = false, want true (%s)", rec.Body)
	}
	if len(out.Missing) != 2 || out.Missing[0] != 1 || out.Missing[1] != 4 {
		t.Fatalf("missing = %v, want [1 4]", out.Missing)
	}
}

// TestHTTPSearchConsistencyReachesWire confirms the search endpoint maps the
// JSON read_consistency/on_partition_unavailable fields into the *Opts wire
// args, so the values dispatched to the engine carry rc=1/opa=1.
func TestHTTPSearchConsistencyReachesWire(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeVectorSearchResults(nil)}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/collections/docs/points/search",
		`{"query":[1,2],"k":1,"read_consistency":1,"on_partition_unavailable":1}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("search = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_search" {
		t.Fatalf("calls = %+v, want one vector_search", disp.calls)
	}
	col, _, _, _, rc, opa, _, err := ops.DecodeVectorSearchArgsOpts(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode args: %v", err)
	}
	if col != "docs" || rc != 1 || opa != 1 {
		t.Fatalf("decoded (col=%q rc=%d opa=%d), want (docs, 1, 1)", col, rc, opa)
	}
}

// TestHTTPSearchConsistencyDefault confirms the default (no fields) path
// dispatches rc=0/opa=0 and still reaches vector_search with 200.
func TestHTTPSearchConsistencyDefault(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeVectorSearchResults(nil)}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/collections/docs/points/search", `{"query":[1,2],"k":1}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("search = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_search" {
		t.Fatalf("calls = %+v, want one vector_search", disp.calls)
	}
	_, _, _, _, rc, opa, _, err := ops.DecodeVectorSearchArgsOpts(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode args: %v", err)
	}
	if rc != 0 || opa != 0 {
		t.Fatalf("decoded (rc=%d opa=%d), want (0, 0)", rc, opa)
	}
}

// TestHTTPSearchConsistencyRejectsOutOfRange confirms an out-of-range value
// (>3, i.e. above BoundedStaleness) is rejected with 400 before dispatch.
func TestHTTPSearchConsistencyRejectsOutOfRange(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/collections/docs/points/search",
		`{"query":[1,2],"k":1,"read_consistency":4}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("out-of-range rc = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 0 {
		t.Fatalf("dispatcher called %d times on out-of-range input, want 0", len(disp.calls))
	}
}

// TestHTTPSearchDocsConsistencyRejectsOutOfRange confirms searchDocs's 400 guard.
func TestHTTPSearchDocsConsistencyRejectsOutOfRange(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/collections/docs/points/search/docs",
		`{"query":[1,2],"k":1,"read_consistency":4}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("out-of-range rc = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 0 {
		t.Fatalf("dispatcher called %d times on out-of-range input, want 0", len(disp.calls))
	}
}

// TestHTTPSearchGroupsConsistencyRejectsOutOfRange confirms searchGroups's 400 guard.
func TestHTTPSearchGroupsConsistencyRejectsOutOfRange(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/collections/docs/points/search/groups",
		`{"query":[1,2],"k":1,"group_by":"x","read_consistency":4}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("out-of-range rc = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 0 {
		t.Fatalf("dispatcher called %d times on out-of-range input, want 0", len(disp.calls))
	}
}

// TestHTTPHybridConsistencyReachesWire confirms the hybrid endpoint maps the
// consistency fields into EncodeHybridSearchArgsOpts.
func TestHTTPHybridConsistencyReachesWire(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeHybridResults(nil)}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/collections/docs/points/search/hybrid",
		`{"dense":[1,2],"k":1,"read_consistency":1,"on_partition_unavailable":1}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("hybrid = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_hybrid_search" {
		t.Fatalf("calls = %+v, want one vector_hybrid_search", disp.calls)
	}
	col, _, _, _, _, rc, opa, _, err := ops.DecodeHybridSearchArgsOpts(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode args: %v", err)
	}
	if col != "docs" || rc != 1 || opa != 1 {
		t.Fatalf("decoded (col=%q rc=%d opa=%d), want (docs, 1, 1)", col, rc, opa)
	}
}

// TestHTTPHybridConsistencyRejectsOutOfRange confirms hybrid's 400 guard.
func TestHTTPHybridConsistencyRejectsOutOfRange(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/collections/docs/points/search/hybrid",
		`{"dense":[1,2],"k":1,"on_partition_unavailable":2}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("out-of-range opa = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 0 {
		t.Fatalf("dispatcher called %d times on out-of-range input, want 0", len(disp.calls))
	}
}

// TestHTTPScrollConsistencyReachesWire confirms the scroll endpoint maps the
// consistency fields into EncodeScrollArgsOpts.
func TestHTTPScrollConsistencyReachesWire(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeVectorDocs(nil)}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/collections/docs/points/scroll",
		`{"limit":5,"read_consistency":1,"on_partition_unavailable":1}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("scroll = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_scroll" {
		t.Fatalf("calls = %+v, want one vector_scroll", disp.calls)
	}
	col, _, limit, rc, opa, err := ops.DecodeScrollArgsOpts(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode args: %v", err)
	}
	if col != "docs" || limit != 5 || rc != 1 || opa != 1 {
		t.Fatalf("decoded (col=%q limit=%d rc=%d opa=%d), want (docs, 5, 1, 1)", col, limit, rc, opa)
	}
}

// TestHTTPScrollConsistencyRejectsOutOfRange confirms scroll's 400 guard.
func TestHTTPScrollConsistencyRejectsOutOfRange(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/collections/docs/points/scroll",
		`{"limit":5,"read_consistency":4}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("out-of-range rc = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 0 {
		t.Fatalf("dispatcher called %d times on out-of-range input, want 0", len(disp.calls))
	}
}

// TestHTTPMVSearchConsistencyReachesWire confirms the multivector search
// endpoint maps the consistency fields into EncodeMVSearchArgsOpts.
func TestHTTPMVSearchConsistencyReachesWire(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeMVResults(nil)}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/multivector/mv/search",
		`{"query":[[1,2]],"k":1,"read_consistency":1,"on_partition_unavailable":1}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("mv search = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_mv_search" {
		t.Fatalf("calls = %+v, want one vector_mv_search", disp.calls)
	}
	name, _, _, _, rc, opa, _, err := ops.DecodeMVSearchArgsOpts(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode args: %v", err)
	}
	if name != "mv" || rc != 1 || opa != 1 {
		t.Fatalf("decoded (name=%q rc=%d opa=%d), want (mv, 1, 1)", name, rc, opa)
	}
}

// TestHTTPMVSearchConsistencyRejectsOutOfRange confirms mv search's 400 guard.
func TestHTTPMVSearchConsistencyRejectsOutOfRange(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/multivector/mv/search",
		`{"query":[[1,2]],"k":1,"on_partition_unavailable":2}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("out-of-range opa = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 0 {
		t.Fatalf("dispatcher called %d times on out-of-range input, want 0", len(disp.calls))
	}
}

// TestHTTPMVScrollPagesWithCursor proves POST /v1/multivector/{name}/scroll
// dispatches vector_mv_scroll with the cursor + rc/opa threaded into the args and
// renders the dispatcher's result trailer ({documents, degraded, missing,
// next_cursor}). Mirrors the dense/named scroll routes.
func TestHTTPMVScrollPagesWithCursor(t *testing.T) {
	docs := []vector.Document{{ID: 7, Content: "a"}, {ID: 9, Content: "b"}}
	disp := &recordingDispatcher{result: ops.EncodeScrollResult(docs, true, []uint16{3}, ops.EncodeScrollCursor(9))}
	h := Handler(disp, Options{})

	var out struct {
		Documents  []vector.Document `json:"documents"`
		Degraded   bool              `json:"degraded"`
		Missing    []uint16          `json:"missing"`
		NextCursor string            `json:"next_cursor"`
	}
	rec := do(t, h, "POST", "/v1/multivector/mv/scroll",
		`{"limit":2,"cursor":"`+ops.EncodeScrollCursor(5)+`","read_consistency":2,"on_partition_unavailable":1}`, &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("mv scroll = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_mv_scroll" {
		t.Fatalf("calls = %+v, want one vector_mv_scroll", disp.calls)
	}
	col, _, limit, rc, opa, afterID, hasAfter, _, err := ops.DecodeMVScrollArgsOpts(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode args: %v", err)
	}
	if col != "mv" || limit != 2 || rc != 2 || opa != 1 {
		t.Fatalf("decoded (col=%q limit=%d rc=%d opa=%d), want (mv, 2, 2, 1)", col, limit, rc, opa)
	}
	if !hasAfter || afterID != 5 {
		t.Fatalf("decoded cursor afterID = %d (has=%v), want 5/true", afterID, hasAfter)
	}
	if out.NextCursor != ops.EncodeScrollCursor(9) {
		t.Fatalf("next_cursor = %q, want cursor(9)", out.NextCursor)
	}
	if len(out.Documents) != 2 || out.Documents[0].ID != 7 || out.Documents[1].ID != 9 {
		t.Fatalf("documents = %+v, want ids [7 9]", out.Documents)
	}
	if !out.Degraded || len(out.Missing) != 1 || out.Missing[0] != 3 {
		t.Fatalf("degraded/missing = %v/%v, want true/[3]", out.Degraded, out.Missing)
	}
}

// TestHTTPMVScrollConsistencyRejectsOutOfRange confirms mv scroll's 400 guard
// (out-of-range read_consistency rejected before dispatch).
func TestHTTPMVScrollConsistencyRejectsOutOfRange(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/multivector/mv/scroll",
		`{"limit":5,"read_consistency":4}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("out-of-range rc = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 0 {
		t.Fatalf("dispatcher called %d times on out-of-range input, want 0", len(disp.calls))
	}
}

// TestHTTPMVScrollRejectsBadCursor confirms a malformed cursor is rejected with
// 400 BEFORE dispatch (client error, never reaches the store).
func TestHTTPMVScrollRejectsBadCursor(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/multivector/mv/scroll",
		`{"limit":5,"cursor":"not-a-valid-cursor!!"}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("bad cursor = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 0 {
		t.Fatalf("dispatcher called %d times on bad cursor, want 0", len(disp.calls))
	}
}

// TestHTTPMVSearchFilterReachesWire confirms the multivector search endpoint
// threads a JSON "filter" into EncodeMVSearchArgsOptsFilter (mirrors dense
// search) so the dispatched args carry the compiled filter.
func TestHTTPMVSearchFilterReachesWire(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeMVResults(nil)}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/multivector/mv/search",
		`{"query":[[1,2]],"k":1,"filter":{"op":"eq","field":"lang","value":{"kind":"string","str":"en"}}}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("mv search = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_mv_search" {
		t.Fatalf("calls = %+v, want one vector_mv_search", disp.calls)
	}
	name, _, _, _, _, _, filter, _, err := ops.DecodeMVSearchArgsOptsFilter(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode args: %v", err)
	}
	if name != "mv" || filter.Op != vector.FilterEq || filter.Field != "lang" {
		t.Fatalf("decoded (name=%q filter=%+v), want (mv, eq on lang)", name, filter)
	}
}

// TestHTTPMVSearchNoFilterUnchanged confirms a filter-less mv search dispatches
// a zero (match-all) filter — the no-filter path is unchanged.
func TestHTTPMVSearchNoFilterUnchanged(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeMVResults(nil)}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/multivector/mv/search", `{"query":[[1,2]],"k":1}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("mv search = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 {
		t.Fatalf("calls = %+v, want one call", disp.calls)
	}
	_, _, _, _, _, _, filter, _, err := ops.DecodeMVSearchArgsOptsFilter(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode args: %v", err)
	}
	if !filter.IsZero() {
		t.Fatalf("decoded filter = %+v, want zero (match-all)", filter)
	}
}

// TestHTTPMVSearchInvalidFilter confirms a syntactically-valid "filter" that
// fails to Compile (bad regex / bad RFC3339) is rejected with 400 BEFORE
// dispatch — fail-loud at the edge, like dense search.
func TestHTTPMVSearchInvalidFilter(t *testing.T) {
	cases := []struct {
		name   string
		filter string
	}{
		{"bad_regex", `{"op":"regex","field":"sku","value":{"kind":"string","str":"*invalid"}}`},
		{"bad_datetime", `{"op":"dt_gte","field":"created_at","value":{"kind":"string","str":"not-a-date"}}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			disp := &recordingDispatcher{}
			h := Handler(disp, Options{})
			rec := do(t, h, "POST", "/v1/multivector/mv/search",
				`{"query":[[1,2]],"k":1,"filter":`+c.filter+`}`, nil)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("bad filter = %d, want 400 (%s)", rec.Code, rec.Body)
			}
			if len(disp.calls) != 0 {
				t.Fatalf("dispatcher called %d times on invalid filter, want 0", len(disp.calls))
			}
		})
	}
}

// TestHTTPMVSearchFilterEndToEnd drives the multivector search filter through
// the REAL op registry + store: two docs with different payloads, a filter that
// matches only one, and the search must return ONLY that doc (post-filter in the
// MaxSim rerank). A no-filter search over the same data returns both.
func TestHTTPMVSearchFilterEndToEnd(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	if rec := do(t, h, "POST", "/v1/multivector/mv",
		`{"dim":3,"m":8,"ef_construction":50,"ef_search":32,"seed":1}`, nil); rec.Code != http.StatusCreated {
		t.Fatalf("mv create = %d (%s)", rec.Code, rec.Body)
	}
	if rec := do(t, h, "POST", "/v1/multivector/mv/docs",
		`{"id":1,"tokens":[[1,0,0]],"metadata":{"lang":{"kind":"string","str":"en"}}}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("mv add 1 = %d (%s)", rec.Code, rec.Body)
	}
	if rec := do(t, h, "POST", "/v1/multivector/mv/docs",
		`{"id":2,"tokens":[[1,0,0]],"metadata":{"lang":{"kind":"string","str":"fr"}}}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("mv add 2 = %d (%s)", rec.Code, rec.Body)
	}

	type mvResp struct {
		Results []vector.MultiResult `json:"results"`
	}

	// Filtered: only the en doc (id=1) survives the post-filter.
	var filtered mvResp
	rec := do(t, h, "POST", "/v1/multivector/mv/search",
		`{"query":[[1,0,0]],"k":5,"filter":{"op":"eq","field":"lang","value":{"kind":"string","str":"en"}}}`, &filtered)
	if rec.Code != http.StatusOK {
		t.Fatalf("filtered mv search = %d (%s)", rec.Code, rec.Body)
	}
	if len(filtered.Results) != 1 || filtered.Results[0].ID != 1 {
		t.Fatalf("filtered results = %+v, want only id=1", filtered.Results)
	}

	// No filter: both docs come back (the path is unchanged).
	var all mvResp
	rec = do(t, h, "POST", "/v1/multivector/mv/search", `{"query":[[1,0,0]],"k":5}`, &all)
	if rec.Code != http.StatusOK {
		t.Fatalf("unfiltered mv search = %d (%s)", rec.Code, rec.Body)
	}
	if len(all.Results) != 2 {
		t.Fatalf("unfiltered results = %+v, want both docs", all.Results)
	}
}

// TestHTTPLinearizableAcceptedAllFamilies proves that read_consistency=2
// (Linearizable) is ACCEPTED (200) and threads to the engine as rc=2 for every
// read family over HTTP, and that read_consistency=4 (above BoundedStaleness) is
// rejected (400) before dispatch. The readIndex barrier is a no-op on this
// recording dispatcher (no shard), so this proves the wire accepts + carries 2
// end-to-end — the level is neither dropped nor clamped at the HTTP edge.
func TestHTTPLinearizableAcceptedAllFamilies(t *testing.T) {
	// decodeRC pulls the rc byte back out of the dispatched args per op so we can
	// assert the Linearizable level survived the edge unchanged.
	searchRC := func(b []byte) uint8 {
		_, _, _, _, rc, _, _, _ := ops.DecodeVectorSearchArgsOpts(b)
		return rc
	}
	decodeRC := map[string]func([]byte) uint8{
		"vector_search":      searchRC,
		"vector_search_docs": searchRC,
		"vector_hybrid_search": func(b []byte) uint8 {
			_, _, _, _, _, rc, _, _, _ := ops.DecodeHybridSearchArgsOpts(b)
			return rc
		},
		"vector_search_groups": func(b []byte) uint8 {
			_, _, _, _, rc, _, _, _ := ops.DecodeGroupSearchArgsOpts(b)
			return rc
		},
		"vector_scroll": func(b []byte) uint8 {
			_, _, _, rc, _, _ := ops.DecodeScrollArgsOpts(b)
			return rc
		},
		"vector_mv_search": func(b []byte) uint8 {
			_, _, _, _, rc, _, _, _ := ops.DecodeMVSearchArgsOpts(b)
			return rc
		},
	}

	cases := []struct {
		name   string
		path   string
		op     string
		body2  string // read_consistency:2 (Linearizable) — must be accepted
		body3  string // read_consistency:4 (>BoundedStaleness) — must be rejected
		result []byte
	}{
		{"search", "/v1/collections/docs/points/search", "vector_search",
			`{"query":[1,2],"k":1,"read_consistency":2}`,
			`{"query":[1,2],"k":1,"read_consistency":4}`,
			ops.EncodeVectorSearchResults(nil)},
		{"search_docs", "/v1/collections/docs/points/search/docs", "vector_search_docs",
			`{"query":[1,2],"k":1,"read_consistency":2}`,
			`{"query":[1,2],"k":1,"read_consistency":4}`,
			ops.EncodeVectorDocs(nil)},
		{"hybrid", "/v1/collections/docs/points/search/hybrid", "vector_hybrid_search",
			`{"dense":[1,2],"k":1,"read_consistency":2}`,
			`{"dense":[1,2],"k":1,"read_consistency":4}`,
			ops.EncodeHybridResults(nil)},
		{"groups", "/v1/collections/docs/points/search/groups", "vector_search_groups",
			`{"query":[1,2],"k":1,"group_by":"x","read_consistency":2}`,
			`{"query":[1,2],"k":1,"group_by":"x","read_consistency":4}`,
			ops.EncodeGroupsDegraded(nil, false, nil)},
		{"scroll", "/v1/collections/docs/points/scroll", "vector_scroll",
			`{"limit":5,"read_consistency":2}`,
			`{"limit":5,"read_consistency":4}`,
			ops.EncodeVectorDocs(nil)},
		{"mv_search", "/v1/multivector/mv/search", "vector_mv_search",
			`{"query":[[1,2]],"k":1,"read_consistency":2}`,
			`{"query":[[1,2]],"k":1,"read_consistency":4}`,
			ops.EncodeMVResults(nil)},
	}

	for _, tc := range cases {
		t.Run(tc.name+"/linearizable_accepted", func(t *testing.T) {
			disp := &recordingDispatcher{result: tc.result}
			h := Handler(disp, Options{})
			rec := do(t, h, "POST", tc.path, tc.body2, nil)
			if rec.Code != http.StatusOK {
				t.Fatalf("%s rc=2 = %d, want 200 (%s)", tc.name, rec.Code, rec.Body)
			}
			if len(disp.calls) != 1 || disp.calls[0].name != tc.op {
				t.Fatalf("%s calls = %+v, want one %s", tc.name, disp.calls, tc.op)
			}
			if got := decodeRC[tc.op](disp.calls[0].args); got != 2 {
				t.Fatalf("%s threaded rc=%d, want 2 (Linearizable) — level dropped/clamped at edge", tc.name, got)
			}
		})
		t.Run(tc.name+"/above_bounded_rejected", func(t *testing.T) {
			disp := &recordingDispatcher{}
			h := Handler(disp, Options{})
			rec := do(t, h, "POST", tc.path, tc.body3, nil)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("%s rc=4 = %d, want 400 (%s)", tc.name, rec.Code, rec.Body)
			}
			if len(disp.calls) != 0 {
				t.Fatalf("%s dispatched on rc=4, want 0 calls", tc.name)
			}
		})
	}
}

// TestHTTPMVSearchNonDegraded asserts a legacy (no-trailer) multivector search
// body yields degraded:false and an empty missing array.
func TestHTTPMVSearchNonDegraded(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeMVResults(
		[]vector.MultiResult{{ID: 9, Score: 0.2}})}
	h := Handler(disp, Options{})

	var out struct {
		Degraded bool  `json:"degraded"`
		Missing  []int `json:"missing"`
	}
	rec := do(t, h, "POST", "/v1/multivector/mv/search", `{"query":[[1,2]],"k":1}`, &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("mv search = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if out.Degraded {
		t.Fatalf("degraded = true, want false")
	}
	if len(out.Missing) != 0 {
		t.Fatalf("missing = %v, want empty", out.Missing)
	}
}

// richFilterJSON is a JSON filter body exercising the four representative rich
// operators (match / regex / is_empty / dt_gte). It is a valid And-tree so the
// edge Compile() succeeds and the encoded filter reaches the dispatcher.
const richFilterJSON = `{"op":"and","and":[` +
	`{"op":"match","field":"title","value":{"kind":"string","str":"quick brown"}},` +
	`{"op":"regex","field":"sku","value":{"kind":"string","str":"^A[0-9]+$"}},` +
	`{"op":"is_empty","field":"deleted_at"},` +
	`{"op":"dt_gte","field":"created_at","value":{"kind":"string","str":"2024-01-02T15:04:05Z"}}` +
	`]}`

// assertRichFilter checks the decoded filter has the four rich-op leaves in the
// And tree, proving the JSON shape parsed (via FilterOp UnmarshalText) and was
// encoded onto the wire untouched.
func assertRichFilter(t *testing.T, f vector.Filter) {
	t.Helper()
	if f.Op != vector.FilterAnd || len(f.And) != 4 {
		t.Fatalf("decoded filter = %+v, want And with 4 children", f)
	}
	wantOps := []vector.FilterOp{vector.FilterMatch, vector.FilterRegex, vector.FilterIsEmpty, vector.FilterDtGte}
	for i, op := range wantOps {
		if f.And[i].Op != op {
			t.Fatalf("child %d op = %v, want %v", i, f.And[i].Op, op)
		}
	}
	if f.And[3].Value.Str != "2024-01-02T15:04:05Z" {
		t.Fatalf("dt_gte value = %q, want RFC3339 literal", f.And[3].Value.Str)
	}
}

// TestHTTPSearchRichFilter proves a JSON filter using the rich ops parses at the
// HTTP edge and the dispatched vector_search op carries the encoded filter.
func TestHTTPSearchRichFilter(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeVectorSearchResults(nil)}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/collections/docs/points/search",
		`{"query":[1,2],"k":3,"filter":`+richFilterJSON+`}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("search = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_search" {
		t.Fatalf("calls = %+v, want one vector_search", disp.calls)
	}
	_, _, _, filter, err := ops.DecodeVectorSearchArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode search args: %v", err)
	}
	assertRichFilter(t, filter)
}

// TestHTTPScrollRichFilter proves the rich-op JSON filter reaches the scroll op.
func TestHTTPScrollRichFilter(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeVectorDocsDegraded(nil, false, nil)}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/collections/docs/points/scroll",
		`{"limit":10,"filter":`+richFilterJSON+`}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("scroll = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_scroll" {
		t.Fatalf("calls = %+v, want one vector_scroll", disp.calls)
	}
	_, filter, _, err := ops.DecodeScrollArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode scroll args: %v", err)
	}
	assertRichFilter(t, filter)
}

// TestHTTPDeleteByFilterRichFilter proves the rich-op JSON filter reaches the
// delete-by-filter op (and is encoded onto the wire untouched).
func TestHTTPDeleteByFilterRichFilter(t *testing.T) {
	disp := &recordingDispatcher{result: []byte{0, 0, 0, 0}} // delete-by-filter count = 0
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/collections/docs/points/delete",
		`{"filter":`+richFilterJSON+`}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("delete-by-filter = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_delete_by_filter" {
		t.Fatalf("calls = %+v, want one vector_delete_by_filter", disp.calls)
	}
	_, filter, err := ops.DecodeDeleteByFilterArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode delete args: %v", err)
	}
	assertRichFilter(t, filter)
}

// badFilterCases enumerates filters that must fail Compile() at the edge: a bad
// RE2 regex and a non-RFC3339 datetime literal.
var badFilterCases = []struct {
	name string
	json string
}{
	{"bad_regex", `{"op":"regex","field":"sku","value":{"kind":"string","str":"*invalid"}}`},
	{"bad_datetime", `{"op":"dt_gte","field":"created_at","value":{"kind":"string","str":"not-a-date"}}`},
}

// badGeoFilterCases enumerates geo filters that must fail Compile() at the edge:
// a nil geo condition, a bad polygon arity, an out-of-range lat/lon, a non-finite
// (NaN) radius, and an inverted (antimeridian) box. All ride as JSON and so reach
// the SAME validFilter -> Compile() edge guard the rich ops use, so all must
// yield a 400 before dispatch (esp. delete: zero deletions on a bad geo filter).
var badGeoFilterCases = []struct {
	name string
	json string
}{
	{"nil_geo", `{"op":"geo_radius","field":"loc"}`},
	{"bad_polygon_arity", `{"op":"geo_polygon","field":"loc","geo":{"polygon":[1,2,3,4]}}`},
	{"out_of_range_latlon", `{"op":"geo_radius","field":"loc","geo":{"center_lat":95,"center_lon":0,"radius_m":1000}}`},
	// JSON null on a float64 → 0 (NaN can't be expressed in valid JSON; NaN is
	// covered at the unit level). Zero radius is rejected by the same !(>0) guard.
	{"zero_radius", `{"op":"geo_radius","field":"loc","geo":{"center_lat":48,"center_lon":2,"radius_m":null}}`},
	{"inverted_box", `{"op":"geo_bounding_box","field":"loc","geo":{"min_lat":49,"min_lon":3,"max_lat":48,"max_lon":2}}`},
}

// TestHTTPInvalidFilterReturns400 asserts a syntactically-valid JSON filter that
// fails to Compile (bad regex / bad RFC3339, plus the geo cases) yields a 400
// (NOT 500) with a non-empty message, and — critically for delete — never
// reaches the dispatcher.
func TestHTTPInvalidFilterReturns400(t *testing.T) {
	endpoints := []struct {
		name string
		path string
		wrap func(filter string) string
	}{
		{"search", "/v1/collections/docs/points/search", func(f string) string { return `{"query":[1,2],"k":1,"filter":` + f + `}` }},
		{"search_docs", "/v1/collections/docs/points/search/docs", func(f string) string { return `{"query":[1,2],"k":1,"filter":` + f + `}` }},
		{"search_groups", "/v1/collections/docs/points/search/groups", func(f string) string { return `{"query":[1,2],"k":1,"group_by":"x","filter":` + f + `}` }},
		{"hybrid", "/v1/collections/docs/points/search/hybrid", func(f string) string { return `{"dense":[1,2],"k":1,"filter":` + f + `}` }},
		{"scroll", "/v1/collections/docs/points/scroll", func(f string) string { return `{"limit":5,"filter":` + f + `}` }},
		{"delete_by_filter", "/v1/collections/docs/points/delete", func(f string) string { return `{"filter":` + f + `}` }},
	}
	allBad := append(append([]struct {
		name string
		json string
	}{}, badFilterCases...), badGeoFilterCases...)
	for _, ep := range endpoints {
		for _, bad := range allBad {
			t.Run(ep.name+"/"+bad.name, func(t *testing.T) {
				disp := &recordingDispatcher{}
				h := Handler(disp, Options{})
				var out struct {
					Error string `json:"error"`
				}
				rec := do(t, h, "POST", ep.path, ep.wrap(bad.json), &out)
				if rec.Code != http.StatusBadRequest {
					t.Fatalf("%s/%s = %d, want 400 (%s)", ep.name, bad.name, rec.Code, rec.Body)
				}
				if out.Error == "" {
					t.Fatalf("%s/%s: empty error message (%s)", ep.name, bad.name, rec.Body)
				}
				if len(disp.calls) != 0 {
					t.Fatalf("%s/%s: dispatcher called %d times on invalid filter, want 0 (no delete on bad filter)",
						ep.name, bad.name, len(disp.calls))
				}
			})
		}
	}
}

// geoFilterJSON is a filter_json string exercising all three geo operators each
// with a GeoCondition; valid → Compile succeeds and the encoded filter reaches
// the dispatcher untouched (geo rides as JSON, no byte-format change).
const geoFilterJSON = `{"op":"and","and":[` +
	`{"op":"geo_radius","field":"loc","geo":{"center_lat":48.8566,"center_lon":2.3522,"radius_m":5000}},` +
	`{"op":"geo_bounding_box","field":"loc","geo":{"min_lat":48,"min_lon":2,"max_lat":49,"max_lon":3}},` +
	`{"op":"geo_polygon","field":"loc","geo":{"polygon":[48,2,49,2,49,3,48,3]}}` +
	`]}`

// assertGeoFilter checks the decoded filter has the three geo-op leaves with
// their GeoConditions intact, proving the JSON shape parsed (FilterOp
// UnmarshalText + the Geo pointer) and was encoded onto the wire untouched.
func assertGeoFilter(t *testing.T, f vector.Filter) {
	t.Helper()
	if f.Op != vector.FilterAnd || len(f.And) != 3 {
		t.Fatalf("decoded filter = %+v, want And with 3 children", f)
	}
	wantOps := []vector.FilterOp{vector.FilterGeoRadius, vector.FilterGeoBox, vector.FilterGeoPolygon}
	for i, op := range wantOps {
		if f.And[i].Op != op {
			t.Fatalf("child %d op = %v, want %v", i, f.And[i].Op, op)
		}
		if f.And[i].Geo == nil {
			t.Fatalf("child %d (%v) lost its Geo condition", i, op)
		}
	}
	if r := f.And[0].Geo; r.CenterLat != 48.8566 || r.CenterLon != 2.3522 || r.RadiusM != 5000 {
		t.Fatalf("geo_radius condition = %+v, want center (48.8566,2.3522) r=5000", r)
	}
	if len(f.And[2].Geo.Polygon) != 8 {
		t.Fatalf("geo_polygon vertices = %v, want flat slice of len 8", f.And[2].Geo.Polygon)
	}
	if _, err := f.Compile(); err != nil {
		t.Fatalf("decoded geo filter does not compile: %v", err)
	}
}

// TestHTTPGeoFilter proves a geo JSON filter parses at the HTTP edge and the
// dispatched search / scroll / delete-by-filter op carries the encoded filter.
func TestHTTPGeoFilter(t *testing.T) {
	cases := []struct {
		name string
		path string
		op   string
		body string
		// decode extracts the filter from the dispatched args.
		decode func([]byte) (vector.Filter, error)
		result []byte
	}{
		{
			"search", "/v1/collections/docs/points/search", "vector_search",
			`{"query":[1,2],"k":3,"filter":` + geoFilterJSON + `}`,
			func(a []byte) (vector.Filter, error) { _, _, _, f, err := ops.DecodeVectorSearchArgs(a); return f, err },
			ops.EncodeVectorSearchResults(nil),
		},
		{
			"scroll", "/v1/collections/docs/points/scroll", "vector_scroll",
			`{"limit":10,"filter":` + geoFilterJSON + `}`,
			func(a []byte) (vector.Filter, error) { _, f, _, err := ops.DecodeScrollArgs(a); return f, err },
			ops.EncodeVectorDocsDegraded(nil, false, nil),
		},
		{
			"delete_by_filter", "/v1/collections/docs/points/delete", "vector_delete_by_filter",
			`{"filter":` + geoFilterJSON + `}`,
			func(a []byte) (vector.Filter, error) { _, f, err := ops.DecodeDeleteByFilterArgs(a); return f, err },
			[]byte{0, 0, 0, 0}, // delete count = 0
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			disp := &recordingDispatcher{result: c.result}
			h := Handler(disp, Options{})
			rec := do(t, h, "POST", c.path, c.body, nil)
			if rec.Code != http.StatusOK {
				t.Fatalf("%s = %d, want 200 (%s)", c.name, rec.Code, rec.Body)
			}
			if len(disp.calls) != 1 || disp.calls[0].name != c.op {
				t.Fatalf("calls = %+v, want one %s", disp.calls, c.op)
			}
			f, err := c.decode(disp.calls[0].args)
			if err != nil {
				t.Fatalf("decode %s args: %v", c.name, err)
			}
			assertGeoFilter(t, f)
		})
	}
}

// TestHTTPUpsertGeoMetadata proves a ValueGeo metadata field round-trips through
// the JSON edge: the inserted point's metadata reaches the dispatcher carrying a
// {"kind":"geo",...} value with lat/lon intact (no byte-format change).
func TestHTTPUpsertGeoMetadata(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})

	body := `{"id":7,"vector":[1,2,3],"metadata":{"loc":{"kind":"geo","lat":48.8566,"lon":2.3522}}}`
	rec := do(t, h, "POST", "/v1/collections/docs/points", body, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("upsert = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_insert" {
		t.Fatalf("calls = %+v, want one vector_insert", disp.calls)
	}
	_, id, _, _, meta, _, _, err := ops.DecodeVectorInsertArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode insert args: %v", err)
	}
	if id != 7 {
		t.Fatalf("id = %d, want 7", id)
	}
	loc, ok := meta["loc"]
	if !ok {
		t.Fatalf("metadata missing 'loc' geo field: %+v", meta)
	}
	if loc.Kind != vector.ValueGeo || loc.Lat != 48.8566 || loc.Lon != 2.3522 {
		t.Fatalf("decoded geo metadata = %+v, want kind=geo lat=48.8566 lon=2.3522", loc)
	}
}

// ---- named-vector (Qdrant-style per-point multi-vector-space) HTTP tests ----

// namedCreateBody is a 2-space named-collection create body: "title" (4-dim,
// Cosine — metric omitted = 0) and "image" (3-dim, DotProduct = metric 2). The
// metric rides as its numeric vector.Metric value (Cosine=0, L2=1, Dot=2).
const namedCreateBody = `{"named_vectors":{` +
	`"title":{"dim":4},` +
	`"image":{"dim":3,"metric":2}` +
	`}}`

// TestHTTPNamedLifecycle drives the full named-collection surface end-to-end
// over JSON against a real op registry + store: create, upsert points with a map
// of named vectors (one omitting a space), search each named space with and
// without a filter, search/docs (payload returned), scroll, get_config, delete.
func TestHTTPNamedLifecycle(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/named/docs", namedCreateBody, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}

	// get_config round-trips the configured spaces.
	var cfgOut struct {
		Name         string                              `json:"name"`
		NamedVectors map[string]vector.NamedVectorParams `json:"named_vectors"`
	}
	rec = do(t, h, "GET", "/v1/named/docs/config", "", &cfgOut)
	if rec.Code != http.StatusOK {
		t.Fatalf("get_config = %d (%s)", rec.Code, rec.Body)
	}
	if cfgOut.NamedVectors["title"].Dim != 4 || cfgOut.NamedVectors["image"].Dim != 3 ||
		cfgOut.NamedVectors["image"].Metric != vector.DotProduct {
		t.Fatalf("config = %+v, want title(dim4) image(dim3,dot)", cfgOut.NamedVectors)
	}

	// Upsert three points; point 3 omits the "image" space (optional per-space).
	upserts := []string{
		`{"id":1,"vectors":{"title":[1,0,0,0],"image":[1,0,0]},"metadata":{"lang":{"kind":"string","str":"en"}}}`,
		`{"id":2,"vectors":{"title":[0,1,0,0],"image":[0,1,0]},"metadata":{"lang":{"kind":"string","str":"fr"}}}`,
		`{"id":3,"vectors":{"title":[1,1,0,0]},"metadata":{"lang":{"kind":"string","str":"en"}}}`,
	}
	for _, u := range upserts {
		if rec := do(t, h, "POST", "/v1/named/docs/points", u, nil); rec.Code != http.StatusOK {
			t.Fatalf("upsert %s = %d (%s)", u, rec.Code, rec.Body)
		}
	}

	// Search the "title" space, unfiltered: id 1 is the nearest to [1,0,0,0].
	var sres struct {
		Results []vector.Result `json:"results"`
	}
	rec = do(t, h, "POST", "/v1/named/docs/search",
		`{"vector_name":"title","query":[1,0,0,0],"k":3}`, &sres)
	if rec.Code != http.StatusOK || len(sres.Results) == 0 || sres.Results[0].ID != 1 {
		t.Fatalf("title search = %d %+v", rec.Code, sres.Results)
	}

	// Filtered search (lang=en) excludes point 2 (fr).
	var fres struct {
		Results []vector.Result `json:"results"`
	}
	rec = do(t, h, "POST", "/v1/named/docs/search",
		`{"vector_name":"title","query":[1,0,0,0],"k":3,"filter":{"op":"eq","field":"lang","value":{"kind":"string","str":"en"}}}`, &fres)
	if rec.Code != http.StatusOK {
		t.Fatalf("filtered title search = %d (%s)", rec.Code, rec.Body)
	}
	for _, r := range fres.Results {
		if r.ID == 2 {
			t.Fatalf("filtered search returned fr point 2: %+v", fres.Results)
		}
	}

	// search/docs returns the shared payload.
	var dres struct {
		Documents []vector.Document `json:"documents"`
	}
	rec = do(t, h, "POST", "/v1/named/docs/search/docs",
		`{"vector_name":"image","query":[1,0,0],"k":2}`, &dres)
	if rec.Code != http.StatusOK || len(dres.Documents) == 0 {
		t.Fatalf("image search/docs = %d %+v", rec.Code, dres.Documents)
	}
	if _, ok := dres.Documents[0].Metadata["lang"]; !ok {
		t.Fatalf("search/docs lost payload: %+v", dres.Documents[0])
	}

	// scroll returns the live points + payload.
	var scrollOut struct {
		Documents []vector.Document `json:"documents"`
	}
	rec = do(t, h, "POST", "/v1/named/docs/scroll", `{"limit":10}`, &scrollOut)
	if rec.Code != http.StatusOK || len(scrollOut.Documents) != 3 {
		t.Fatalf("scroll = %d, %d docs (want 3) (%s)", rec.Code, len(scrollOut.Documents), rec.Body)
	}

	// delete point 2, then it is gone from search + scroll.
	var delOut struct {
		Deleted bool `json:"deleted"`
	}
	rec = do(t, h, "DELETE", "/v1/named/docs/points/2", "", &delOut)
	if rec.Code != http.StatusOK || !delOut.Deleted {
		t.Fatalf("delete = %d deleted=%v (%s)", rec.Code, delOut.Deleted, rec.Body)
	}
	scrollOut.Documents = nil
	do(t, h, "POST", "/v1/named/docs/scroll", `{"limit":10}`, &scrollOut)
	for _, d := range scrollOut.Documents {
		if d.ID == 2 {
			t.Fatalf("scroll still returns deleted point 2: %+v", scrollOut.Documents)
		}
	}

	// drop removes the collection.
	if rec := do(t, h, "DELETE", "/v1/named/docs", "", nil); rec.Code != http.StatusOK {
		t.Fatalf("drop = %d (%s)", rec.Code, rec.Body)
	}
}

// TestHTTPNamedDeleteByIDBody covers the POST /v1/named/{name}/points/delete
// route (handler namedDeleteByID, body {"id":N}) — the id-carrying alternative
// to the DELETE /points/{id} path variant exercised by TestHTTPNamedLifecycle.
// Create a named collection, upsert a point, delete it by body, assert the
// {"deleted":true} response, and that a subsequent scroll no longer returns it.
func TestHTTPNamedDeleteByIDBody(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	if rec := do(t, h, "POST", "/v1/named/docs", namedCreateBody, nil); rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}
	if rec := do(t, h, "POST", "/v1/named/docs/points",
		`{"id":7,"vectors":{"title":[1,0,0,0],"image":[1,0,0]},"metadata":{"lang":{"kind":"string","str":"en"}}}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("upsert = %d (%s)", rec.Code, rec.Body)
	}

	// Point 7 is present before the delete.
	var before struct {
		Documents []vector.Document `json:"documents"`
	}
	do(t, h, "POST", "/v1/named/docs/scroll", `{"limit":10}`, &before)
	var found bool
	for _, d := range before.Documents {
		if d.ID == 7 {
			found = true
		}
	}
	if !found {
		t.Fatalf("pre-delete scroll missing point 7: %+v", before.Documents)
	}

	// Delete-by-body: POST .../points/delete {"id":7} → {"deleted":true}.
	var delOut struct {
		Deleted bool `json:"deleted"`
	}
	rec := do(t, h, "POST", "/v1/named/docs/points/delete", `{"id":7}`, &delOut)
	if rec.Code != http.StatusOK || !delOut.Deleted {
		t.Fatalf("delete-by-body = %d deleted=%v (%s)", rec.Code, delOut.Deleted, rec.Body)
	}

	// Point 7 is gone from scroll.
	var after struct {
		Documents []vector.Document `json:"documents"`
	}
	do(t, h, "POST", "/v1/named/docs/scroll", `{"limit":10}`, &after)
	for _, d := range after.Documents {
		if d.ID == 7 {
			t.Fatalf("scroll still returns deleted point 7: %+v", after.Documents)
		}
	}
}

// TestHTTPNamedSearchDispatchesArgs asserts the named search edge encodes the
// vector_name + query + k + filter into the dispatched vector_named_search op
// (recording dispatcher; decode the args back).
func TestHTTPNamedSearchDispatchesArgs(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeVectorSearchResults(nil)}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/named/docs/search",
		`{"vector_name":"title","query":[1,2,3,4],"k":7,"filter":{"op":"eq","field":"lang","value":{"kind":"string","str":"en"}}}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("search = %d (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_named_search" {
		t.Fatalf("calls = %+v, want one vector_named_search", disp.calls)
	}
	col, vecName, q, k, f, err := ops.DecodeNamedSearchArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode named search args: %v", err)
	}
	if col != "docs" || vecName != "title" || k != 7 || len(q) != 4 || q[0] != 1 {
		t.Fatalf("decoded = (%q,%q,k=%d,q=%v), want (docs,title,7,[1 2 3 4])", col, vecName, k, q)
	}
	if f.Op != vector.FilterEq || f.Field != "lang" {
		t.Fatalf("decoded filter = %+v, want eq/lang", f)
	}
}

// TestHTTPNamedUnknownVectorName: searching/inserting an unconfigured space is a
// fail-loud 400 (the op error carries ErrUnknownVectorName).
func TestHTTPNamedUnknownVectorName(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	if rec := do(t, h, "POST", "/v1/named/docs", namedCreateBody, nil); rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}
	// Insert into an unknown space.
	rec := do(t, h, "POST", "/v1/named/docs/points", `{"id":1,"vectors":{"nope":[1,0,0,0]}}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("insert unknown space = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	// Search an unknown space.
	rec = do(t, h, "POST", "/v1/named/docs/search", `{"vector_name":"nope","query":[1,0,0,0],"k":3}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("search unknown space = %d, want 400 (%s)", rec.Code, rec.Body)
	}
}

// TestHTTPNamedDimMismatch: a vector whose length != the space dim is a 400.
func TestHTTPNamedDimMismatch(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	if rec := do(t, h, "POST", "/v1/named/docs", namedCreateBody, nil); rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}
	rec := do(t, h, "POST", "/v1/named/docs/points", `{"id":1,"vectors":{"title":[1,0,0]}}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("dim mismatch = %d, want 400 (%s)", rec.Code, rec.Body)
	}
}

// TestHTTPNamedEmptyConfig: a create with no named spaces is a 400.
func TestHTTPNamedEmptyConfig(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	rec := do(t, h, "POST", "/v1/named/docs", `{"named_vectors":{}}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("empty config = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	// A reserved or empty PER-SPACE name is a client error → 400, not 500.
	for _, body := range []string{
		`{"named_vectors":{"ti#tle":{"dim":4}}}`,
		`{"named_vectors":{"":{"dim":4}}}`,
	} {
		rec := do(t, h, "POST", "/v1/named/docs", body, nil)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("bad space name %q = %d, want 400 (%s)", body, rec.Code, rec.Body)
		}
	}
}

// TestHTTPNamedCreatePQDropVecs proves the named per-space pq_drop_vecs JSON field
// is parsed and threaded (named carries per-space params as config_json, with the
// numeric QuantMode/Metric encodings — QuantPQ==3, L2==1): a create with a QuantPQ
// space + pq_drop_vecs succeeds (201); pq_drop_vecs without QuantPQ is rejected
// with 400 (the per-space toConfig().Validate ErrInvalidPQDropVecs fail-loud).
func TestHTTPNamedCreatePQDropVecs(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	// pq_drop_vecs=true on a QuantPQ (quant==3) named space → 201.
	rec := do(t, h, "POST", "/v1/named/pqdrop",
		`{"named_vectors":{"title":{"dim":8,"metric":1,"quant":3,"ivf_train_threshold":500,"pq_drop_vecs":true}}}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("named create pq+pq_drop_vecs = %d, want 201 (%s)", rec.Code, rec.Body)
	}
	// pq_drop_vecs=true with NO PQ mode → 400 (ErrInvalidPQDropVecs fail-loud).
	rec = do(t, h, "POST", "/v1/named/pqdropbad",
		`{"named_vectors":{"title":{"dim":8,"metric":1,"pq_drop_vecs":true}}}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("named pq_drop_vecs without quant=pq = %d, want 400 (%s)", rec.Code, rec.Body)
	}
}

// TestHTTPNamedInvalidFilterReturns400 asserts a syntactically-valid JSON filter
// that fails to Compile yields 400 BEFORE dispatch on every named filter
// endpoint (esp. scroll — never traverse with a broken predicate); zero dispatch.
func TestHTTPNamedInvalidFilterReturns400(t *testing.T) {
	endpoints := []struct {
		name string
		path string
		wrap func(filter string) string
	}{
		{"search", "/v1/named/docs/search", func(f string) string { return `{"vector_name":"title","query":[1,2,3,4],"k":1,"filter":` + f + `}` }},
		{"search_docs", "/v1/named/docs/search/docs", func(f string) string { return `{"vector_name":"title","query":[1,2,3,4],"k":1,"filter":` + f + `}` }},
		{"scroll", "/v1/named/docs/scroll", func(f string) string { return `{"limit":5,"filter":` + f + `}` }},
	}
	allBad := append(append([]struct {
		name string
		json string
	}{}, badFilterCases...), badGeoFilterCases...)
	for _, ep := range endpoints {
		for _, bad := range allBad {
			t.Run(ep.name+"/"+bad.name, func(t *testing.T) {
				disp := &recordingDispatcher{}
				h := Handler(disp, Options{})
				var out struct {
					Error string `json:"error"`
				}
				rec := do(t, h, "POST", ep.path, ep.wrap(bad.json), &out)
				if rec.Code != http.StatusBadRequest {
					t.Fatalf("%s/%s = %d, want 400 (%s)", ep.name, bad.name, rec.Code, rec.Body)
				}
				if out.Error == "" {
					t.Fatalf("%s/%s: empty error message (%s)", ep.name, bad.name, rec.Body)
				}
				if len(disp.calls) != 0 {
					t.Fatalf("%s/%s: dispatcher called %d times on invalid filter, want 0",
						ep.name, bad.name, len(disp.calls))
				}
			})
		}
	}
}

// ---- get-by-id + in-place payload mutation HTTP tests ----

// TestHTTPDenseGetPayload drives GET /points/{id} + the four payload endpoints
// end-to-end on a dense collection against a real engine: get reflects vec+
// payload+ttl, the projection query params omit a field, each payload op then
// reflects in a re-get, an absent id is 404, and a bad payload JSON is 400.
func TestHTTPDenseGetPayload(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	if rec := do(t, h, "POST", "/v1/collections",
		`{"name":"docs","config":{"dim":3,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1}}`, nil); rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}
	if rec := do(t, h, "POST", "/v1/collections/docs/points",
		`{"id":1,"vector":[1,2,3],"ttl_ms":600000,"metadata":{"lang":{"kind":"string","str":"en"}}}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("insert = %d (%s)", rec.Code, rec.Body)
	}

	type getResp struct {
		Found   bool            `json:"found"`
		Vector  []float32       `json:"vector"`
		Payload vector.Metadata `json:"payload"`
		TTLMs   int64           `json:"ttl_ms"`
	}

	// Get: both projections default on.
	var g getResp
	rec := do(t, h, "GET", "/v1/collections/docs/points/1", "", &g)
	if rec.Code != http.StatusOK || !g.Found || len(g.Vector) != 3 || g.Vector[0] != 1 {
		t.Fatalf("get = %d %+v", rec.Code, g)
	}
	if g.Payload["lang"].Str != "en" || g.TTLMs <= 0 {
		t.Fatalf("get payload/ttl = %+v / %d", g.Payload, g.TTLMs)
	}

	// with_payload=false omits payload; with_vector=false omits vector.
	var gv getResp
	do(t, h, "GET", "/v1/collections/docs/points/1?with_payload=false", "", &gv)
	if len(gv.Vector) != 3 || len(gv.Payload) != 0 {
		t.Fatalf("with_payload=false get = vec %d payload %+v", len(gv.Vector), gv.Payload)
	}
	var gp getResp
	do(t, h, "GET", "/v1/collections/docs/points/1?with_vector=false", "", &gp)
	if len(gp.Vector) != 0 || gp.Payload["lang"].Str != "en" {
		t.Fatalf("with_vector=false get = vec %d payload %+v", len(gp.Vector), gp.Payload)
	}

	// Absent id → 404.
	if rec := do(t, h, "GET", "/v1/collections/docs/points/999", "", nil); rec.Code != http.StatusNotFound {
		t.Fatalf("get absent = %d, want 404", rec.Code)
	}

	// set-payload merges.
	if rec := do(t, h, "POST", "/v1/collections/docs/points/1/payload", `{"city":{"kind":"string","str":"nyc"}}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("set-payload = %d (%s)", rec.Code, rec.Body)
	}
	g = getResp{}
	do(t, h, "GET", "/v1/collections/docs/points/1", "", &g)
	if g.Payload["lang"].Str != "en" || g.Payload["city"].Str != "nyc" {
		t.Fatalf("after merge = %+v", g.Payload)
	}

	// overwrite replaces.
	if rec := do(t, h, "POST", "/v1/collections/docs/points/1/payload/overwrite", `{"only":{"kind":"string","str":"v"}}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("overwrite = %d (%s)", rec.Code, rec.Body)
	}
	g = getResp{}
	do(t, h, "GET", "/v1/collections/docs/points/1", "", &g)
	if _, ok := g.Payload["lang"]; ok || g.Payload["only"].Str != "v" {
		t.Fatalf("after overwrite = %+v", g.Payload)
	}

	// PUT .../payload is also overwrite.
	if rec := do(t, h, "PUT", "/v1/collections/docs/points/1/payload", `{"put":{"kind":"string","str":"w"}}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("put overwrite = %d (%s)", rec.Code, rec.Body)
	}
	g = getResp{}
	do(t, h, "GET", "/v1/collections/docs/points/1", "", &g)
	if _, ok := g.Payload["only"]; ok || g.Payload["put"].Str != "w" {
		t.Fatalf("after put overwrite = %+v", g.Payload)
	}

	// delete-keys removes "put" → empty.
	if rec := do(t, h, "POST", "/v1/collections/docs/points/1/payload/delete", `{"keys":["put"]}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("delete-keys = %d (%s)", rec.Code, rec.Body)
	}
	g = getResp{}
	do(t, h, "GET", "/v1/collections/docs/points/1", "", &g)
	if len(g.Payload) != 0 {
		t.Fatalf("after delete-keys = %+v, want empty", g.Payload)
	}

	// clear empties a re-populated payload.
	do(t, h, "POST", "/v1/collections/docs/points/1/payload", `{"x":{"kind":"string","str":"y"}}`, nil)
	if rec := do(t, h, "POST", "/v1/collections/docs/points/1/payload/clear", "", nil); rec.Code != http.StatusOK {
		t.Fatalf("clear = %d (%s)", rec.Code, rec.Body)
	}
	g = getResp{}
	do(t, h, "GET", "/v1/collections/docs/points/1", "", &g)
	if len(g.Payload) != 0 {
		t.Fatalf("after clear = %+v, want empty", g.Payload)
	}

	// Payload mutation of an absent point → 404.
	if rec := do(t, h, "POST", "/v1/collections/docs/points/999/payload", `{}`, nil); rec.Code != http.StatusNotFound {
		t.Fatalf("set-payload absent = %d, want 404", rec.Code)
	}
	if rec := do(t, h, "POST", "/v1/collections/docs/points/999/payload/clear", "", nil); rec.Code != http.StatusNotFound {
		t.Fatalf("clear absent = %d, want 404", rec.Code)
	}

	// Bad payload JSON → 400.
	if rec := do(t, h, "POST", "/v1/collections/docs/points/1/payload", `{not json`, nil); rec.Code != http.StatusBadRequest {
		t.Fatalf("set-payload bad json = %d, want 400", rec.Code)
	}
}

// TestHTTPDenseGetPayloadDispatchesOps asserts the get/payload edges encode the
// right op name + args (recording-dispatcher pattern, like the resplit tests).
func TestHTTPDenseGetPayloadDispatchesOps(t *testing.T) {
	t.Run("get", func(t *testing.T) {
		disp := &recordingDispatcher{result: ops.EncodeVectorGetResult(true, []float32{1, 2, 3}, nil, 0, nil, true, true)}
		h := Handler(disp, Options{})
		if rec := do(t, h, "GET", "/v1/collections/docs/points/7?with_payload=false", "", nil); rec.Code != http.StatusOK {
			t.Fatalf("get = %d", rec.Code)
		}
		if len(disp.calls) != 1 || disp.calls[0].name != "vector_get" {
			t.Fatalf("calls = %+v, want one vector_get", disp.calls)
		}
		col, id, flags, err := ops.DecodeVectorGetArgs(disp.calls[0].args)
		if err != nil || col != "docs" || id != 7 || flags != ops.GetFlagWithVector {
			t.Fatalf("decoded = (%q, %d, %d, %v), want (docs, 7, with_vector-only)", col, id, flags, err)
		}
	})
	t.Run("set-payload", func(t *testing.T) {
		disp := &recordingDispatcher{result: ops.EncodePayloadResult(true)}
		h := Handler(disp, Options{})
		if rec := do(t, h, "POST", "/v1/collections/docs/points/7/payload", `{"a":{"kind":"string","str":"b"}}`, nil); rec.Code != http.StatusOK {
			t.Fatalf("set-payload = %d", rec.Code)
		}
		if len(disp.calls) != 1 || disp.calls[0].name != "vector_set_payload" {
			t.Fatalf("calls = %+v, want one vector_set_payload", disp.calls)
		}
		col, id, meta, err := ops.DecodeSetPayloadArgs(disp.calls[0].args)
		if err != nil || col != "docs" || id != 7 || meta["a"].Str != "b" {
			t.Fatalf("decoded = (%q, %d, %+v, %v)", col, id, meta, err)
		}
	})
	t.Run("delete-keys", func(t *testing.T) {
		disp := &recordingDispatcher{result: ops.EncodePayloadResult(true)}
		h := Handler(disp, Options{})
		if rec := do(t, h, "POST", "/v1/collections/docs/points/7/payload/delete", `{"keys":["k1","k2"]}`, nil); rec.Code != http.StatusOK {
			t.Fatalf("delete-keys = %d", rec.Code)
		}
		if len(disp.calls) != 1 || disp.calls[0].name != "vector_delete_payload_keys" {
			t.Fatalf("calls = %+v, want one vector_delete_payload_keys", disp.calls)
		}
		col, id, keys, err := ops.DecodeDeletePayloadKeysArgs(disp.calls[0].args)
		if err != nil || col != "docs" || id != 7 || len(keys) != 2 || keys[0] != "k1" {
			t.Fatalf("decoded = (%q, %d, %v, %v)", col, id, keys, err)
		}
	})
}

// TestHTTPDenseSetPayloadKeyTTL covers the optional key_ttl_ms body field on the
// dense set/overwrite-payload endpoints: (1) recording-dispatcher proves the map
// reaches the wire args and the rest of the body is the payload; (2) a raw-map
// body without key_ttl_ms still dispatches a plain payload (byte-identical wire);
// (3) e2e set→expire→get omits the TTL'd key while permanent keys survive.
func TestHTTPDenseSetPayloadKeyTTL(t *testing.T) {
	t.Run("wire-carries-key-ttl", func(t *testing.T) {
		disp := &recordingDispatcher{result: ops.EncodePayloadResult(true)}
		h := Handler(disp, Options{})
		body := `{"a":{"kind":"string","str":"b"},"key_ttl_ms":{"a":1000}}`
		if rec := do(t, h, "POST", "/v1/collections/docs/points/7/payload", body, nil); rec.Code != http.StatusOK {
			t.Fatalf("set-payload = %d (%s)", rec.Code, rec.Body)
		}
		if len(disp.calls) != 1 || disp.calls[0].name != "vector_set_payload" {
			t.Fatalf("calls = %+v", disp.calls)
		}
		col, id, meta, ttl, err := ops.DecodeSetPayloadArgsOpts(disp.calls[0].args)
		if err != nil || col != "docs" || id != 7 {
			t.Fatalf("decoded (%q,%d,%v)", col, id, err)
		}
		if meta["a"].Str != "b" {
			t.Fatalf("payload = %+v, want a=b (key_ttl_ms peeled off)", meta)
		}
		if _, leaked := meta["key_ttl_ms"]; leaked {
			t.Fatalf("key_ttl_ms leaked into payload: %+v", meta)
		}
		if len(ttl) != 1 || ttl["a"] != 1000 {
			t.Fatalf("key_ttl_ms = %+v, want {a:1000}", ttl)
		}
	})

	t.Run("raw-body-byte-identical", func(t *testing.T) {
		disp := &recordingDispatcher{result: ops.EncodePayloadResult(true)}
		h := Handler(disp, Options{})
		if rec := do(t, h, "POST", "/v1/collections/docs/points/7/payload",
			`{"a":{"kind":"string","str":"b"}}`, nil); rec.Code != http.StatusOK {
			t.Fatalf("set-payload = %d", rec.Code)
		}
		// No key_ttl_ms: the encoded args must equal the legacy encoder output.
		want := ops.EncodeSetPayloadArgs("docs", 7, vector.Metadata{"a": vector.NewString("b")})
		if !bytes.Equal(disp.calls[0].args, want) {
			t.Fatalf("args = %v, want legacy %v", disp.calls[0].args, want)
		}
	})

	t.Run("e2e-expiry", func(t *testing.T) {
		h, cleanup := newTestAPI(t)
		defer cleanup()
		if rec := do(t, h, "POST", "/v1/collections",
			`{"name":"docs","config":{"dim":3,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1}}`, nil); rec.Code != http.StatusCreated {
			t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
		}
		do(t, h, "POST", "/v1/collections/docs/points",
			`{"id":1,"vector":[1,2,3],"metadata":{"keep":{"kind":"int","int":1}}}`, nil)
		if rec := do(t, h, "POST", "/v1/collections/docs/points/1/payload",
			`{"temp":{"kind":"string","str":"t"},"perm":{"kind":"string","str":"p"},"key_ttl_ms":{"temp":1}}`, nil); rec.Code != http.StatusOK {
			t.Fatalf("set-payload = %d (%s)", rec.Code, rec.Body)
		}
		time.Sleep(15 * time.Millisecond)
		var g struct {
			Payload vector.Metadata `json:"payload"`
		}
		do(t, h, "GET", "/v1/collections/docs/points/1", "", &g)
		if _, ok := g.Payload["temp"]; ok {
			t.Errorf("expired key temp still present: %+v", g.Payload)
		}
		if g.Payload["keep"].Int != 1 || g.Payload["perm"].Str != "p" {
			t.Errorf("non-TTL keys dropped: %+v", g.Payload)
		}
	})
}

// TestHTTPNamedGetPayload drives the named get + four payload endpoints e2e.
func TestHTTPNamedGetPayload(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	if rec := do(t, h, "POST", "/v1/named/docs", namedCreateBody, nil); rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}
	if rec := do(t, h, "POST", "/v1/named/docs/points",
		`{"id":1,"vectors":{"title":[1,0,0,0],"image":[1,0,0]},"metadata":{"lang":{"kind":"string","str":"en"}}}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("upsert = %d (%s)", rec.Code, rec.Body)
	}

	type namedGetResp struct {
		Found   bool                 `json:"found"`
		Vectors map[string][]float32 `json:"vectors"`
		Payload vector.Metadata      `json:"payload"`
		TTLMs   int64                `json:"ttl_ms"`
	}

	var g namedGetResp
	rec := do(t, h, "GET", "/v1/named/docs/points/1", "", &g)
	if rec.Code != http.StatusOK || !g.Found || len(g.Vectors) != 2 || len(g.Vectors["title"]) != 4 || g.Payload["lang"].Str != "en" {
		t.Fatalf("named get = %d %+v", rec.Code, g)
	}

	// with_vector=false omits the named vectors.
	var gp namedGetResp
	do(t, h, "GET", "/v1/named/docs/points/1?with_vector=false", "", &gp)
	if len(gp.Vectors) != 0 || gp.Payload["lang"].Str != "en" {
		t.Fatalf("with_vector=false named get = %+v", gp)
	}

	if rec := do(t, h, "GET", "/v1/named/docs/points/999", "", nil); rec.Code != http.StatusNotFound {
		t.Fatalf("named get absent = %d, want 404", rec.Code)
	}

	// merge → overwrite → delete-keys → clear.
	do(t, h, "POST", "/v1/named/docs/points/1/payload", `{"city":{"kind":"string","str":"nyc"}}`, nil)
	g = namedGetResp{}
	do(t, h, "GET", "/v1/named/docs/points/1", "", &g)
	if g.Payload["lang"].Str != "en" || g.Payload["city"].Str != "nyc" {
		t.Fatalf("named merge = %+v", g.Payload)
	}
	do(t, h, "POST", "/v1/named/docs/points/1/payload/overwrite", `{"only":{"kind":"string","str":"v"}}`, nil)
	do(t, h, "POST", "/v1/named/docs/points/1/payload/delete", `{"keys":["only"]}`, nil)
	g = namedGetResp{}
	do(t, h, "GET", "/v1/named/docs/points/1", "", &g)
	if len(g.Payload) != 0 {
		t.Fatalf("named after delete-keys = %+v, want empty", g.Payload)
	}
	do(t, h, "POST", "/v1/named/docs/points/1/payload", `{"x":{"kind":"string","str":"y"}}`, nil)
	do(t, h, "POST", "/v1/named/docs/points/1/payload/clear", "", nil)
	g = namedGetResp{}
	do(t, h, "GET", "/v1/named/docs/points/1", "", &g)
	if len(g.Payload) != 0 {
		t.Fatalf("named after clear = %+v, want empty", g.Payload)
	}

	if rec := do(t, h, "POST", "/v1/named/docs/points/999/payload", `{}`, nil); rec.Code != http.StatusNotFound {
		t.Fatalf("named set-payload absent = %d, want 404", rec.Code)
	}
	if rec := do(t, h, "POST", "/v1/named/docs/points/1/payload", `{bad`, nil); rec.Code != http.StatusBadRequest {
		t.Fatalf("named set-payload bad json = %d, want 400", rec.Code)
	}
}

// TestHTTPMVGetPayload drives the MV get + four payload endpoints e2e.
func TestHTTPMVGetPayload(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	if rec := do(t, h, "POST", "/v1/multivector/docs",
		`{"dim":3,"m":8,"ef_construction":50,"ef_search":32,"seed":1}`, nil); rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}
	if rec := do(t, h, "POST", "/v1/multivector/docs/docs",
		`{"id":1,"tokens":[[1,0,0],[0,1,0]],"metadata":{"lang":{"kind":"string","str":"en"}}}`, nil); rec.Code != http.StatusOK {
		t.Fatalf("add = %d (%s)", rec.Code, rec.Body)
	}

	type mvGetResp struct {
		Found   bool            `json:"found"`
		Tokens  [][]float32     `json:"tokens"`
		Payload vector.Metadata `json:"payload"`
	}

	var g mvGetResp
	rec := do(t, h, "GET", "/v1/multivector/docs/points/1", "", &g)
	if rec.Code != http.StatusOK || !g.Found || len(g.Tokens) != 2 || len(g.Tokens[0]) != 3 || g.Payload["lang"].Str != "en" {
		t.Fatalf("mv get = %d %+v", rec.Code, g)
	}

	var gp mvGetResp
	do(t, h, "GET", "/v1/multivector/docs/points/1?with_vector=false", "", &gp)
	if len(gp.Tokens) != 0 || gp.Payload["lang"].Str != "en" {
		t.Fatalf("with_vector=false mv get = %+v", gp)
	}

	if rec := do(t, h, "GET", "/v1/multivector/docs/points/999", "", nil); rec.Code != http.StatusNotFound {
		t.Fatalf("mv get absent = %d, want 404", rec.Code)
	}

	do(t, h, "POST", "/v1/multivector/docs/points/1/payload", `{"city":{"kind":"string","str":"nyc"}}`, nil)
	g = mvGetResp{}
	do(t, h, "GET", "/v1/multivector/docs/points/1", "", &g)
	if g.Payload["lang"].Str != "en" || g.Payload["city"].Str != "nyc" {
		t.Fatalf("mv merge = %+v", g.Payload)
	}
	do(t, h, "POST", "/v1/multivector/docs/points/1/payload/overwrite", `{"only":{"kind":"string","str":"v"}}`, nil)
	do(t, h, "POST", "/v1/multivector/docs/points/1/payload/delete", `{"keys":["only"]}`, nil)
	g = mvGetResp{}
	do(t, h, "GET", "/v1/multivector/docs/points/1", "", &g)
	if len(g.Payload) != 0 {
		t.Fatalf("mv after delete-keys = %+v, want empty", g.Payload)
	}
	do(t, h, "POST", "/v1/multivector/docs/points/1/payload", `{"x":{"kind":"string","str":"y"}}`, nil)
	do(t, h, "POST", "/v1/multivector/docs/points/1/payload/clear", "", nil)
	g = mvGetResp{}
	do(t, h, "GET", "/v1/multivector/docs/points/1", "", &g)
	if len(g.Payload) != 0 {
		t.Fatalf("mv after clear = %+v, want empty", g.Payload)
	}

	if rec := do(t, h, "POST", "/v1/multivector/docs/points/999/payload", `{}`, nil); rec.Code != http.StatusNotFound {
		t.Fatalf("mv set-payload absent = %d, want 404", rec.Code)
	}
	if rec := do(t, h, "POST", "/v1/multivector/docs/points/1/payload", `{bad`, nil); rec.Code != http.StatusBadRequest {
		t.Fatalf("mv set-payload bad json = %d, want 400", rec.Code)
	}
}

// --- collection aliases (management surface) ---

// TestHTTPAliasCreate verifies POST /v1/aliases lowers to a single-action
// alias_batch carrying {alias→collection} (a create, Delete=false).
func TestHTTPAliasCreate(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})

	var out struct {
		Alias      string `json:"alias"`
		Collection string `json:"collection"`
	}
	rec := do(t, h, "POST", "/v1/aliases", `{"alias":"prod","collection":"docs"}`, &out)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create alias = %d, want 201 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "alias_batch" {
		t.Fatalf("calls = %+v, want one alias_batch", disp.calls)
	}
	actions, err := ops.DecodeAliasBatchArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode batch args: %v", err)
	}
	if len(actions) != 1 || actions[0].Alias != "prod" || actions[0].Canonical != "docs" || actions[0].Delete {
		t.Fatalf("decoded actions = %+v, want [{prod docs false}]", actions)
	}
	if out.Alias != "prod" || out.Collection != "docs" {
		t.Fatalf("body = %+v, want {prod docs}", out)
	}
}

// TestHTTPAliasCreateRequiresFields rejects an empty alias/collection at the edge
// (400, dispatcher untouched).
func TestHTTPAliasCreateRequiresFields(t *testing.T) {
	for _, body := range []string{`{"alias":"","collection":"docs"}`, `{"alias":"prod","collection":""}`} {
		disp := &recordingDispatcher{}
		h := Handler(disp, Options{})
		rec := do(t, h, "POST", "/v1/aliases", body, nil)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("create alias %q = %d, want 400", body, rec.Code)
		}
		if len(disp.calls) != 0 {
			t.Fatalf("dispatcher called %d times for %q, want 0", len(disp.calls), body)
		}
	}
}

// TestHTTPAliasDelete verifies DELETE /v1/aliases/{alias} lowers to a single
// delete-action alias_batch and is an idempotent 200.
func TestHTTPAliasDelete(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})

	var out struct {
		Deleted string `json:"deleted"`
	}
	rec := do(t, h, "DELETE", "/v1/aliases/prod", "", &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("delete alias = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "alias_batch" {
		t.Fatalf("calls = %+v, want one alias_batch", disp.calls)
	}
	actions, err := ops.DecodeAliasBatchArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode batch args: %v", err)
	}
	if len(actions) != 1 || actions[0].Alias != "prod" || !actions[0].Delete {
		t.Fatalf("decoded actions = %+v, want [{prod  true}]", actions)
	}
	if out.Deleted != "prod" {
		t.Fatalf("body = %+v, want {prod}", out)
	}
}

// TestHTTPAliasList decodes the alias_list result into the JSON envelope and
// passes the ?collection filter through to the dispatched args.
func TestHTTPAliasList(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeAliasListResult([]ops.AliasEntry{
		{Alias: "prod", Collection: "docs"},
		{Alias: "stage", Collection: "docs"},
	})}
	h := Handler(disp, Options{})

	var out struct {
		Aliases []struct {
			Alias      string `json:"alias"`
			Collection string `json:"collection"`
		} `json:"aliases"`
	}
	rec := do(t, h, "GET", "/v1/aliases?collection=docs", "", &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("list aliases = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "alias_list" {
		t.Fatalf("calls = %+v, want one alias_list", disp.calls)
	}
	coll, err := ops.DecodeAliasListArgs(disp.calls[0].args)
	if err != nil || coll != "docs" {
		t.Fatalf("decode list args = (%q, %v), want (docs, nil)", coll, err)
	}
	if len(out.Aliases) != 2 {
		t.Fatalf("aliases = %+v, want 2 entries", out.Aliases)
	}
}

// TestHTTPAliasBatchSwap verifies the atomic-swap body builds ONE alias_batch
// carrying [{delete prod},{create prod→docs2}] — not two separate calls.
func TestHTTPAliasBatchSwap(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})

	body := `{"actions":[{"delete":{"alias":"prod"}},{"create":{"alias":"prod","collection":"docs2"}}]}`
	var out struct {
		Applied int `json:"applied"`
	}
	rec := do(t, h, "POST", "/v1/aliases/batch", body, &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("alias batch = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "alias_batch" {
		t.Fatalf("calls = %+v, want exactly one alias_batch (atomic swap)", disp.calls)
	}
	actions, err := ops.DecodeAliasBatchArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode batch args: %v", err)
	}
	if len(actions) != 2 {
		t.Fatalf("actions = %+v, want 2 (delete then create)", actions)
	}
	if actions[0].Alias != "prod" || !actions[0].Delete {
		t.Fatalf("action[0] = %+v, want {prod delete}", actions[0])
	}
	if actions[1].Alias != "prod" || actions[1].Canonical != "docs2" || actions[1].Delete {
		t.Fatalf("action[1] = %+v, want {prod→docs2 create}", actions[1])
	}
	if out.Applied != 2 {
		t.Fatalf("applied = %d, want 2", out.Applied)
	}
}

// TestHTTPAliasBatchRejectsAmbiguousAction rejects an action that sets both (or
// neither) create/delete at the edge (400, dispatcher untouched).
func TestHTTPAliasBatchRejectsAmbiguousAction(t *testing.T) {
	for _, body := range []string{
		`{"actions":[{"create":{"alias":"a","collection":"c"},"delete":{"alias":"a"}}]}`,
		`{"actions":[{}]}`,
		`{"actions":[{"create":{"alias":"","collection":"c"}}]}`,
		`{"actions":[{"delete":{"alias":""}}]}`,
	} {
		disp := &recordingDispatcher{}
		h := Handler(disp, Options{})
		rec := do(t, h, "POST", "/v1/aliases/batch", body, nil)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("batch %q = %d, want 400", body, rec.Code)
		}
		if len(disp.calls) != 0 {
			t.Fatalf("dispatcher called %d times for %q, want 0", len(disp.calls), body)
		}
	}
}

// errDispatcher returns a fixed error for every Call, so the transport's
// error→status mapping can be exercised without the engine.
type errDispatcher struct{ err error }

func (d *errDispatcher) Call(string, []byte) ([]byte, error) { return nil, d.err }
func (d *errDispatcher) LeaderAddr() string                  { return "" }

// TestHTTPAliasValidationMapsTo400 confirms the four alias-validation sentinels
// (all carrying the "rostam: alias " prefix) map to HTTP 400, not 500.
func TestHTTPAliasValidationMapsTo400(t *testing.T) {
	for _, msg := range []string{
		"alias \"prod\" → \"missing\": rostam: alias target collection does not exist",
		"alias \"docs\": rostam: alias name shadows an existing collection",
		"alias \"prod\" → \"a\": rostam: alias target is itself an alias",
		"alias \"bad#x\": rostam: alias name must not contain reserved characters '#' or '@'",
	} {
		disp := &errDispatcher{err: fmt.Errorf("%s", msg)}
		h := Handler(disp, Options{})
		rec := do(t, h, "POST", "/v1/aliases", `{"alias":"prod","collection":"docs"}`, nil)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("validation error %q = %d, want 400 (%s)", msg, rec.Code, rec.Body)
		}
	}
}

// ---- named read consistency (read_consistency / on_partition_unavailable) ----

// TestHTTPNamedSearchConsistencyReachesWire proves a named search carrying
// read_consistency / on_partition_unavailable JSON dispatches *Opts-encoded args
// with the decoded rc/opa intact.
func TestHTTPNamedSearchConsistencyReachesWire(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeVectorSearchResults(nil)}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/named/col/search",
		`{"vector_name":"v","query":[1,2],"k":1,"read_consistency":2,"on_partition_unavailable":1}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("named search = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_named_search" {
		t.Fatalf("calls = %+v, want one vector_named_search", disp.calls)
	}
	col, _, _, _, _, rc, opa, _, err := ops.DecodeNamedSearchArgsOpts(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode args: %v", err)
	}
	if col != "col" || rc != 2 || opa != 1 {
		t.Fatalf("decoded (col=%q rc=%d opa=%d), want (col, 2, 1)", col, rc, opa)
	}
}

// TestHTTPNamedScrollConsistencyReachesWire proves a named scroll threads rc/opa
// into the *Opts-encoded dispatch args.
func TestHTTPNamedScrollConsistencyReachesWire(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeScrollResult(nil, false, nil, "")}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/named/col/scroll",
		`{"limit":10,"read_consistency":2,"on_partition_unavailable":1}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("named scroll = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_named_scroll" {
		t.Fatalf("calls = %+v, want one vector_named_scroll", disp.calls)
	}
	col, _, _, _, _, rc, opa, _, err := ops.DecodeNamedScrollArgsOpts(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode args: %v", err)
	}
	if col != "col" || rc != 2 || opa != 1 {
		t.Fatalf("decoded (col=%q rc=%d opa=%d), want (col, 2, 1)", col, rc, opa)
	}
}

// TestHTTPNamedSearchConsistencyDefault confirms a no-rc named search dispatches
// rc=0/opa=0 and still reaches vector_named_search with 200.
func TestHTTPNamedSearchConsistencyDefault(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeVectorSearchResults(nil)}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/named/col/search", `{"vector_name":"v","query":[1,2],"k":1}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("named search = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_named_search" {
		t.Fatalf("calls = %+v, want one vector_named_search", disp.calls)
	}
	_, _, _, _, _, rc, opa, _, err := ops.DecodeNamedSearchArgsOpts(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode args: %v", err)
	}
	if rc != 0 || opa != 0 {
		t.Fatalf("decoded (rc=%d opa=%d), want (0, 0)", rc, opa)
	}
}

// TestHTTPNamedConsistencyRejectsOutOfRange confirms out-of-range rc/opa is
// rejected with 400 before dispatch on the named search/docs/scroll routes.
func TestHTTPNamedConsistencyRejectsOutOfRange(t *testing.T) {
	cases := []struct {
		name, path, body string
	}{
		{"search rc=4", "/v1/named/col/search", `{"vector_name":"v","query":[1,2],"k":1,"read_consistency":4}`},
		{"search/docs rc=4", "/v1/named/col/search/docs", `{"vector_name":"v","query":[1,2],"k":1,"read_consistency":4}`},
		{"scroll rc=4", "/v1/named/col/scroll", `{"limit":10,"read_consistency":4}`},
		{"search opa=2", "/v1/named/col/search", `{"vector_name":"v","query":[1,2],"k":1,"on_partition_unavailable":2}`},
	}
	for _, tc := range cases {
		disp := &recordingDispatcher{}
		h := Handler(disp, Options{})
		rec := do(t, h, "POST", tc.path, tc.body, nil)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("%s = %d, want 400 (%s)", tc.name, rec.Code, rec.Body)
		}
		if len(disp.calls) != 0 {
			t.Fatalf("%s: dispatcher called %d times on out-of-range input, want 0", tc.name, len(disp.calls))
		}
	}
}

// TestHTTPBatchGet covers the POST /points/batch-get route end-to-end: insert a
// few points, then batch-get a MIX of present + absent ids. The present ones come
// back as points (with their id + vector + payload + ttl_ms); the absent ones come
// back in "missing" — and a partial miss is HTTP 200, NOT 404 (the property that
// distinguishes batch get from single get). A malformed body is 400.
func TestHTTPBatchGet(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"docs","config":{"dim":3,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1}}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}
	// Insert ids 1 and 3 (id 2 deliberately absent).
	for _, p := range []string{
		`{"id":1,"vector":[1,0,0],"metadata":{"lang":{"kind":"string","str":"en"}}}`,
		`{"id":3,"vector":[3,0,0]}`,
	} {
		if rec := do(t, h, "POST", "/v1/collections/docs/points", p, nil); rec.Code != http.StatusOK && rec.Code != http.StatusCreated {
			t.Fatalf("insert = %d (%s)", rec.Code, rec.Body)
		}
	}

	var resp struct {
		Points []struct {
			ID      uint64    `json:"id"`
			Vector  []float32 `json:"vector"`
			Payload map[string]struct {
				Kind string `json:"kind"`
				Str  string `json:"str"`
			} `json:"payload"`
			TTLMs int64 `json:"ttl_ms"`
		} `json:"points"`
		Missing []uint64 `json:"missing"`
	}
	rec = do(t, h, "POST", "/v1/collections/docs/points/batch-get",
		`{"ids":[1,2,3],"with_vector":true,"with_payload":true}`, &resp)
	// Partial miss MUST be 200, not 404.
	if rec.Code != http.StatusOK {
		t.Fatalf("batch-get partial miss = %d (%s), want 200 (NOT 404)", rec.Code, rec.Body)
	}
	if len(resp.Points) != 2 {
		t.Fatalf("points = %d, want 2 (ids 1,3)", len(resp.Points))
	}
	if resp.Points[0].ID != 1 || resp.Points[1].ID != 3 {
		t.Fatalf("point ids = %d,%d, want 1,3", resp.Points[0].ID, resp.Points[1].ID)
	}
	if got := resp.Points[0].Vector; len(got) != 3 || got[0] != 1 {
		t.Fatalf("point 1 vector = %v, want [1 0 0]", got)
	}
	if resp.Points[0].Payload["lang"].Str != "en" {
		t.Fatalf("point 1 payload lang = %q, want en", resp.Points[0].Payload["lang"].Str)
	}
	if len(resp.Missing) != 1 || resp.Missing[0] != 2 {
		t.Fatalf("missing = %v, want [2]", resp.Missing)
	}

	// Malformed body → 400.
	if rec := do(t, h, "POST", "/v1/collections/docs/points/batch-get", `{not json`, nil); rec.Code != http.StatusBadRequest {
		t.Fatalf("bad body = %d, want 400", rec.Code)
	}
	// Unknown field → 400 (DisallowUnknownFields).
	if rec := do(t, h, "POST", "/v1/collections/docs/points/batch-get", `{"ids":[1],"bogus":true}`, nil); rec.Code != http.StatusBadRequest {
		t.Fatalf("unknown field body = %d, want 400", rec.Code)
	}
}

// TestHTTPBatchGetEmpty asserts an empty id list yields 200 with empty points +
// empty missing (never 404, never an error).
func TestHTTPBatchGetEmpty(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"docs","config":{"dim":3,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1}}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}
	var resp struct {
		Points  []any    `json:"points"`
		Missing []uint64 `json:"missing"`
	}
	rec = do(t, h, "POST", "/v1/collections/docs/points/batch-get", `{"ids":[]}`, &resp)
	if rec.Code != http.StatusOK {
		t.Fatalf("empty batch-get = %d (%s), want 200", rec.Code, rec.Body)
	}
	if len(resp.Points) != 0 || len(resp.Missing) != 0 {
		t.Fatalf("empty batch-get: points=%d missing=%d, want 0/0", len(resp.Points), len(resp.Missing))
	}
}

// TestHTTPNamedGetBatch drives the named batch get-by-id endpoint end-to-end over
// JSON against a real op registry + store: create a 2-space named collection,
// upsert points, then POST a mixed present/absent id list and assert the response
// carries {points:[{id,vectors,payload,ttl_ms}], missing:[...]} with a partial
// miss returned as 200 (NOT 404). The named clone of the dense batch-get route.
func TestHTTPNamedGetBatch(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	if rec := do(t, h, "POST", "/v1/named/docs", namedCreateBody, nil); rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}
	upserts := []string{
		`{"id":1,"vectors":{"title":[1,0,0,0],"image":[1,0,0]},"metadata":{"lang":{"kind":"string","str":"en"}}}`,
		`{"id":2,"vectors":{"title":[0,1,0,0],"image":[0,1,0]},"metadata":{"lang":{"kind":"string","str":"fr"}}}`,
	}
	for _, u := range upserts {
		if rec := do(t, h, "POST", "/v1/named/docs/points", u, nil); rec.Code != http.StatusOK {
			t.Fatalf("upsert %s = %d (%s)", u, rec.Code, rec.Body)
		}
	}

	var out struct {
		Points []struct {
			ID      uint64               `json:"id"`
			Vectors map[string][]float32 `json:"vectors"`
			Payload map[string]any       `json:"payload"`
			TTLMs   int64                `json:"ttl_ms"`
		} `json:"points"`
		Missing []uint64 `json:"missing"`
	}
	// id 1 present, 99 absent, 2 present — partial miss must be 200.
	rec := do(t, h, "POST", "/v1/named/docs/points/batch-get", `{"ids":[1,99,2]}`, &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("batch-get = %d (%s), want 200 on partial miss", rec.Code, rec.Body)
	}
	if len(out.Points) != 2 {
		t.Fatalf("points = %d, want 2 (%s)", len(out.Points), rec.Body)
	}
	// points are id-sorted: 1 then 2.
	if out.Points[0].ID != 1 || out.Points[1].ID != 2 {
		t.Fatalf("point ids = %d,%d, want 1,2", out.Points[0].ID, out.Points[1].ID)
	}
	if len(out.Points[0].Vectors["title"]) != 4 || len(out.Points[0].Vectors["image"]) != 3 {
		t.Fatalf("point 1 vectors = %+v, want title(4)+image(3)", out.Points[0].Vectors)
	}
	if out.Points[0].Payload["lang"] == nil {
		t.Fatalf("point 1 payload missing lang: %+v", out.Points[0].Payload)
	}
	if len(out.Missing) != 1 || out.Missing[0] != 99 {
		t.Fatalf("missing = %v, want [99]", out.Missing)
	}

	// with_vector=false drops the vectors map, payload kept.
	out.Points = nil
	out.Missing = nil
	do(t, h, "POST", "/v1/named/docs/points/batch-get", `{"ids":[1],"with_vector":false}`, &out)
	if len(out.Points) != 1 || len(out.Points[0].Vectors) != 0 || out.Points[0].Payload["lang"] == nil {
		t.Fatalf("with_vector=false: %+v", out.Points)
	}
}

// TestHTTPMVGetBatch drives the MV batch get-by-id endpoint end-to-end over JSON
// against a real op registry + store: create an MV collection, add docs, then
// POST a mixed present/absent id list and assert the response carries {points:
// [{id,tokens,payload}], missing:[...]} with a partial miss returned as 200 (NOT
// 404). MV has NO ttl. The MV clone of TestHTTPNamedGetBatch.
func TestHTTPMVGetBatch(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	if rec := do(t, h, "POST", "/v1/multivector/docs",
		`{"dim":3,"m":8,"ef_construction":50,"ef_search":32,"seed":1}`, nil); rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}
	adds := []string{
		`{"id":1,"tokens":[[1,0,0],[0,1,0]],"metadata":{"lang":{"kind":"string","str":"en"}}}`,
		`{"id":2,"tokens":[[0,0,1]],"metadata":{"lang":{"kind":"string","str":"fr"}}}`,
	}
	for _, a := range adds {
		if rec := do(t, h, "POST", "/v1/multivector/docs/docs", a, nil); rec.Code != http.StatusOK {
			t.Fatalf("add %s = %d (%s)", a, rec.Code, rec.Body)
		}
	}

	var out struct {
		Points []struct {
			ID      uint64         `json:"id"`
			Tokens  [][]float32    `json:"tokens"`
			Payload map[string]any `json:"payload"`
		} `json:"points"`
		Missing []uint64 `json:"missing"`
	}
	// id 1 present, 99 absent, 2 present — partial miss must be 200.
	rec := do(t, h, "POST", "/v1/multivector/docs/points/batch-get", `{"ids":[1,99,2]}`, &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("batch-get = %d (%s), want 200 on partial miss", rec.Code, rec.Body)
	}
	if len(out.Points) != 2 {
		t.Fatalf("points = %d, want 2 (%s)", len(out.Points), rec.Body)
	}
	// points are id-sorted: 1 then 2.
	if out.Points[0].ID != 1 || out.Points[1].ID != 2 {
		t.Fatalf("point ids = %d,%d, want 1,2", out.Points[0].ID, out.Points[1].ID)
	}
	if len(out.Points[0].Tokens) != 2 || len(out.Points[0].Tokens[0]) != 3 {
		t.Fatalf("point 1 tokens = %+v, want 2x3", out.Points[0].Tokens)
	}
	if out.Points[0].Payload["lang"] == nil {
		t.Fatalf("point 1 payload missing lang: %+v", out.Points[0].Payload)
	}
	if len(out.Missing) != 1 || out.Missing[0] != 99 {
		t.Fatalf("missing = %v, want [99]", out.Missing)
	}

	// with_vector=false drops the token matrix, payload kept.
	out.Points = nil
	out.Missing = nil
	do(t, h, "POST", "/v1/multivector/docs/points/batch-get", `{"ids":[1],"with_vector":false}`, &out)
	if len(out.Points) != 1 || len(out.Points[0].Tokens) != 0 || out.Points[0].Payload["lang"] == nil {
		t.Fatalf("with_vector=false: %+v", out.Points)
	}
}

// TestHTTPCreatePQHNSW drives the full PQ-HNSW create wire over REST: quant="pq"
// + quant_pq_m must create a dense HNSW index that trains PQ at build and serves
// ADC-navigated search. It proves parseQuant accepts "pq" and the QuantPQM field
// rides the create wire end-to-end (the index works == the engine got QuantPQ).
func TestHTTPCreatePQHNSW(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"pq","config":{"dim":8,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1,"quant":"pq","quant_pq_m":8}}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create pq = %d, want 201 (%s)", rec.Code, rec.Body)
	}
	// Stage a small corpus, build (trains PQ + encodes codes), then search.
	var pts []string
	for i := 1; i <= 40; i++ {
		pts = append(pts, fmt.Sprintf(`{"id":%d,"vector":[%d,0,0,0,0,0,0,0]}`, i, i))
	}
	rec = do(t, h, "POST", "/v1/collections/pq/points/bulk", `{"points":[`+strings.Join(pts, ",")+`]}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("stage = %d (%s)", rec.Code, rec.Body)
	}
	rec = do(t, h, "POST", "/v1/collections/pq/points/bulk/build", `{}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("build = %d (%s)", rec.Code, rec.Body)
	}
	var sres struct {
		Results []vector.Result `json:"results"`
	}
	do(t, h, "POST", "/v1/collections/pq/points/search", `{"query":[1,0,0,0,0,0,0,0],"k":5}`, &sres)
	if len(sres.Results) != 5 {
		t.Fatalf("post-build PQ search = %d results, want 5 (%+v)", len(sres.Results), sres.Results)
	}
	if sres.Results[0].ID != 1 {
		t.Fatalf("nearest = id %d, want 1 (%+v)", sres.Results[0].ID, sres.Results)
	}
}

// TestHTTPParseQuantPQ proves "pq" maps to QuantPQ with QuantPQM threaded, and
// an unknown quant is still rejected.
func TestHTTPParseQuantPQ(t *testing.T) {
	cfg, errMsg := collectionConfig{Dim: 8, Metric: "cosine", M: 16, Quant: "pq", QuantPQM: 8}.toConfig()
	if errMsg != "" {
		t.Fatalf("toConfig pq: %s", errMsg)
	}
	if cfg.Quant != vector.QuantPQ {
		t.Fatalf("Quant = %v, want QuantPQ", cfg.Quant)
	}
	if cfg.QuantPQM != 8 {
		t.Fatalf("QuantPQM = %d, want 8", cfg.QuantPQM)
	}
	if _, errMsg := (collectionConfig{Dim: 8, Quant: "bogus"}).toConfig(); errMsg == "" {
		t.Fatalf("toConfig with quant=bogus: expected error, got empty")
	}
}

// TestHTTPCreatePQHNSWFailLoud covers the two engine validation gates on the
// create wire: quant="pq" on an IVF index, and dim not divisible by quant_pq_m.
// Both must surface as 400 (not 500).
func TestHTTPCreatePQHNSWFailLoud(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	// quant="pq" on an IVF index — QuantPQ requires IndexHNSW.
	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"pqivf","config":{"dim":8,"metric":"l2","m":8,"quant":"pq","quant_pq_m":8,"index_type":"ivf"}}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("pq on IVF = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	// dim (8) not divisible by quant_pq_m (5).
	rec = do(t, h, "POST", "/v1/collections",
		`{"name":"pqbad","config":{"dim":8,"metric":"l2","m":8,"quant":"pq","quant_pq_m":5}}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("indivisible m = %d, want 400 (%s)", rec.Code, rec.Body)
	}
}

// TestHTTPCreateSQHNSW drives the full trained-SQ create wire over REST:
// quant="sq" + sq_bits must create a dense HNSW index that trains the scalar
// quantizer at build and serves rescored search. It proves parseQuant accepts
// "sq" and sq_bits rides the create wire end-to-end (the index works == the
// engine got QuantSQ with the right bit-depth).
func TestHTTPCreateSQHNSW(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"sq","config":{"dim":8,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1,"quant":"sq","sq_bits":6}}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create sq = %d, want 201 (%s)", rec.Code, rec.Body)
	}
	var pts []string
	for i := 1; i <= 40; i++ {
		pts = append(pts, fmt.Sprintf(`{"id":%d,"vector":[%d,0,0,0,0,0,0,0]}`, i, i))
	}
	rec = do(t, h, "POST", "/v1/collections/sq/points/bulk", `{"points":[`+strings.Join(pts, ",")+`]}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("stage = %d (%s)", rec.Code, rec.Body)
	}
	rec = do(t, h, "POST", "/v1/collections/sq/points/bulk/build", `{}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("build = %d (%s)", rec.Code, rec.Body)
	}
	var sres struct {
		Results []vector.Result `json:"results"`
	}
	do(t, h, "POST", "/v1/collections/sq/points/search", `{"query":[1,0,0,0,0,0,0,0],"k":5}`, &sres)
	if len(sres.Results) != 5 {
		t.Fatalf("post-build SQ search = %d results, want 5 (%+v)", len(sres.Results), sres.Results)
	}
	if sres.Results[0].ID != 1 {
		t.Fatalf("nearest = id %d, want 1 (%+v)", sres.Results[0].ID, sres.Results)
	}
}

// TestHTTPCreatePRQHNSW drives the full product-residual-quant create wire over
// REST: quant="prq" + prq_layers (+ quant_pq_m) must create a dense HNSW index
// that trains the PRQ layers at build and serves ADC-navigated search. It proves
// parseQuant accepts "prq" and prq_layers rides the create wire end-to-end.
func TestHTTPCreatePRQHNSW(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"prq","config":{"dim":8,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1,"quant":"prq","quant_pq_m":8,"prq_layers":2}}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create prq = %d, want 201 (%s)", rec.Code, rec.Body)
	}
	var pts []string
	for i := 1; i <= 40; i++ {
		pts = append(pts, fmt.Sprintf(`{"id":%d,"vector":[%d,0,0,0,0,0,0,0]}`, i, i))
	}
	rec = do(t, h, "POST", "/v1/collections/prq/points/bulk", `{"points":[`+strings.Join(pts, ",")+`]}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("stage = %d (%s)", rec.Code, rec.Body)
	}
	rec = do(t, h, "POST", "/v1/collections/prq/points/bulk/build", `{}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("build = %d (%s)", rec.Code, rec.Body)
	}
	var sres struct {
		Results []vector.Result `json:"results"`
	}
	do(t, h, "POST", "/v1/collections/prq/points/search", `{"query":[1,0,0,0,0,0,0,0],"k":5}`, &sres)
	if len(sres.Results) != 5 {
		t.Fatalf("post-build PRQ search = %d results, want 5 (%+v)", len(sres.Results), sres.Results)
	}
	if sres.Results[0].ID != 1 {
		t.Fatalf("nearest = id %d, want 1 (%+v)", sres.Results[0].ID, sres.Results)
	}
}

// TestHTTPParseQuantSQPRQ proves "sq"/"prq" map to QuantSQ/QuantPRQ with
// sq_bits/prq_layers threaded into the engine config.
func TestHTTPParseQuantSQPRQ(t *testing.T) {
	cfg, errMsg := collectionConfig{Dim: 8, Metric: "l2", M: 16, Quant: "sq", SQBits: 6}.toConfig()
	if errMsg != "" {
		t.Fatalf("toConfig sq: %s", errMsg)
	}
	if cfg.Quant != vector.QuantSQ || cfg.SQBits != 6 {
		t.Fatalf("Quant/SQBits = %v/%d, want QuantSQ/6", cfg.Quant, cfg.SQBits)
	}
	cfg, errMsg = collectionConfig{Dim: 8, Metric: "l2", M: 16, Quant: "prq", QuantPQM: 8, PRQLayers: 3}.toConfig()
	if errMsg != "" {
		t.Fatalf("toConfig prq: %s", errMsg)
	}
	if cfg.Quant != vector.QuantPRQ || cfg.PRQLayers != 3 {
		t.Fatalf("Quant/PRQLayers = %v/%d, want QuantPRQ/3", cfg.Quant, cfg.PRQLayers)
	}
}

// TestHTTPGetReadConsistency proves the get / get_config GET routes parse the
// ?read_consistency= query param: a bad value fails loud (400) and a valid
// Linearizable value serves the point (read-your-writes through the real
// single-shard dispatcher, where the barrier is a no-op but the path is exercised
// end-to-end). Covers dense get + named-get-config.
func TestHTTPGetReadConsistency(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"docs","config":{"dim":3,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1}}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}
	rec = do(t, h, "POST", "/v1/collections/docs/points/batch",
		`{"upsert":true,"points":[{"id":7,"vector":[7,0,0],"content":"x"}]}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("upsert = %d (%s)", rec.Code, rec.Body)
	}

	// Bad read_consistency ⇒ 400 (fail-loud), no serve.
	rec = do(t, h, "GET", "/v1/collections/docs/points/7?read_consistency=9", "", nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("bad read_consistency = %d, want 400 (%s)", rec.Code, rec.Body)
	}

	// Valid Linearizable ⇒ 200 read-your-writes.
	var got struct {
		Found bool `json:"found"`
	}
	rec = do(t, h, "GET", "/v1/collections/docs/points/7?read_consistency=2", "", &got)
	if rec.Code != http.StatusOK || !got.Found {
		t.Fatalf("linearizable get = %d found=%v (%s)", rec.Code, got.Found, rec.Body)
	}

	// AnyReplica (no param) serves the same point ⇒ 200.
	rec = do(t, h, "GET", "/v1/collections/docs/points/7", "", &got)
	if rec.Code != http.StatusOK || !got.Found {
		t.Fatalf("anyreplica get = %d found=%v (%s)", rec.Code, got.Found, rec.Body)
	}
}

// ---- /query (unified Query API) ----

// TestHTTPQueryFusionRoundTrip proves the /query endpoint validates a FUSION
// body, dispatches vector_query with the marshaled QuerySpec (dbsf, 2 prefetch
// lanes, rc/opa threaded), and decodes the coordinator's flat fused top-k +
// degraded/missing into the JSON response.
func TestHTTPQueryFusionRoundTrip(t *testing.T) {
	want := []vector.Result{{ID: 7, Distance: 0.1, Score: 0.9}, {ID: 3, Distance: 0.2, Score: 0.5}}
	disp := &recordingDispatcher{result: ops.EncodeQueryResultFusedDegraded(want, true, []uint16{2})}
	h := Handler(disp, Options{})

	var out struct {
		Results  []vector.Result `json:"results"`
		Degraded bool            `json:"degraded"`
		Missing  []int           `json:"missing"`
	}
	rec := do(t, h, "POST", "/v1/collections/docs/query",
		`{"mode":"fusion","method":"dbsf","alpha":0.5,"k":2,"read_consistency":1,"on_partition_unavailable":1,"prefetch":[{"dense":[1,2],"k":10},{"sparse":{"indices":[0,5],"values":[0.3,0.7]},"k":10}]}`,
		&out)
	if rec.Code != http.StatusOK {
		t.Fatalf("query = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_query" {
		t.Fatalf("calls = %+v, want one vector_query", disp.calls)
	}
	col, _, spec, rc, opa, _, err := ops.DecodeQuerySpecArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("DecodeQuerySpecArgs: %v", err)
	}
	if col != "docs" || rc != 1 || opa != 1 {
		t.Fatalf("decoded (col=%q rc=%d opa=%d), want (docs,1,1)", col, rc, opa)
	}
	if spec.Mode != vector.ModeFusion || spec.Method != vector.FusionDBSF || len(spec.Prefetch) != 2 {
		t.Fatalf("decoded spec = %+v, want fusion/dbsf/2-prefetch", spec)
	}
	if len(out.Results) != 2 || out.Results[0].ID != 7 {
		t.Fatalf("results = %v, want [7,3]", out.Results)
	}
	if !out.Degraded || len(out.Missing) != 1 || out.Missing[0] != 2 {
		t.Fatalf("degraded/missing = %v/%v, want true/[2]", out.Degraded, out.Missing)
	}
}

// TestHTTPQueryRerankRoundTrip proves a RERANK body (dense root over a dense
// prefetch) dispatches the right spec and decodes the reranked top-k.
func TestHTTPQueryRerankRoundTrip(t *testing.T) {
	want := []vector.Result{{ID: 1, Distance: 0.05, Score: 0.95}}
	disp := &recordingDispatcher{result: ops.EncodeQueryResultFusedDegraded(want, false, nil)}
	h := Handler(disp, Options{})

	var out struct {
		Results []vector.Result `json:"results"`
	}
	rec := do(t, h, "POST", "/v1/collections/docs/query",
		`{"mode":"rerank","k":1,"root":{"dense":[1,2],"k":1},"prefetch":[{"dense":[1,2],"k":50}]}`,
		&out)
	if rec.Code != http.StatusOK {
		t.Fatalf("query rerank = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	_, _, spec, _, _, _, err := ops.DecodeQuerySpecArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("DecodeQuerySpecArgs: %v", err)
	}
	if spec.Mode != vector.ModeRerank || spec.Root.Kind != vector.LeafDense || len(spec.Root.Dense) != 2 {
		t.Fatalf("decoded spec = %+v, want rerank/dense-root", spec)
	}
	if len(out.Results) != 1 || out.Results[0].ID != 1 {
		t.Fatalf("results = %v, want [1]", out.Results)
	}
}

// TestHTTPQueryRecommendRoundTrip proves the /query endpoint parses a RECOMMEND
// leaf ({recommend:{positive,negative}}) into a LeafRecommend with the example
// point-ids, dispatches vector_query with the marshaled spec (the coordinator
// derives it to dense), and decodes the result. The wire carries the recommend
// leaf verbatim (it rides the existing VectorQuery RPC); the derive happens at the
// coordinator, so the HTTP edge only has to parse + thread the example ids.
func TestHTTPQueryRecommendRoundTrip(t *testing.T) {
	want := []vector.Result{{ID: 5, Distance: 0.1, Score: 0.9}, {ID: 9, Distance: 0.2, Score: 0.5}}
	disp := &recordingDispatcher{result: ops.EncodeQueryResultFusedDegraded(want, false, nil)}
	h := Handler(disp, Options{})

	var out struct {
		Results []vector.Result `json:"results"`
	}
	rec := do(t, h, "POST", "/v1/collections/docs/query",
		`{"mode":"fusion","k":2,"prefetch":[{"recommend":{"positive":[1,2,3],"negative":[7]},"k":2}]}`,
		&out)
	if rec.Code != http.StatusOK {
		t.Fatalf("recommend query = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_query" {
		t.Fatalf("calls = %+v, want one vector_query", disp.calls)
	}
	_, _, spec, _, _, _, err := ops.DecodeQuerySpecArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("DecodeQuerySpecArgs: %v", err)
	}
	if len(spec.Prefetch) != 1 || spec.Prefetch[0].Leaf.Kind != vector.LeafRecommend {
		t.Fatalf("decoded spec = %+v, want one recommend prefetch leaf", spec)
	}
	leaf := spec.Prefetch[0].Leaf
	if len(leaf.Positive) != 3 || leaf.Positive[0] != 1 || leaf.Positive[2] != 3 {
		t.Fatalf("recommend positive = %v, want [1 2 3]", leaf.Positive)
	}
	if len(leaf.Negative) != 1 || leaf.Negative[0] != 7 {
		t.Fatalf("recommend negative = %v, want [7]", leaf.Negative)
	}
	if len(out.Results) != 2 || out.Results[0].ID != 5 {
		t.Fatalf("results = %v, want [5,9]", out.Results)
	}
}

// TestHTTPQueryRecommendBestScoreRoundTrip proves the /query endpoint parses a
// BEST_SCORE recommend leaf ({recommend:{positive,negative,strategy:"best_score"}})
// into a LeafRecommend with Strategy=RecommendBestScore + ScoreDesc=true (a custom
// scorer, score-descending) carrying the example ids verbatim for the coordinator to
// resolve+embed.
func TestHTTPQueryRecommendBestScoreRoundTrip(t *testing.T) {
	want := []vector.Result{{ID: 5, Score: 0.9}, {ID: 9, Score: 0.5}}
	disp := &recordingDispatcher{result: ops.EncodeQueryResultFusedDegraded(want, false, nil)}
	h := Handler(disp, Options{})

	var out struct {
		Results []vector.Result `json:"results"`
	}
	rec := do(t, h, "POST", "/v1/collections/docs/query",
		`{"mode":"fusion","k":2,"prefetch":[{"recommend":{"positive":[1,2],"negative":[7],"strategy":"best_score"},"k":2}]}`,
		&out)
	if rec.Code != http.StatusOK {
		t.Fatalf("BEST_SCORE query = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_query" {
		t.Fatalf("calls = %+v, want one vector_query", disp.calls)
	}
	_, _, spec, _, _, _, err := ops.DecodeQuerySpecArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("DecodeQuerySpecArgs: %v", err)
	}
	if len(spec.Prefetch) != 1 || spec.Prefetch[0].Leaf.Kind != vector.LeafRecommend {
		t.Fatalf("decoded spec = %+v, want one recommend prefetch leaf", spec)
	}
	leaf := spec.Prefetch[0].Leaf
	if leaf.Strategy != vector.RecommendBestScore {
		t.Fatalf("strategy = %d, want RecommendBestScore", leaf.Strategy)
	}
	if !leaf.ScoreDesc {
		t.Fatalf("BEST_SCORE leaf must be score-descending")
	}
	if len(leaf.Positive) != 2 || leaf.Negative[0] != 7 {
		t.Fatalf("example ids lost: pos=%v neg=%v", leaf.Positive, leaf.Negative)
	}
	if len(out.Results) != 2 || out.Results[0].ID != 5 {
		t.Fatalf("results = %v, want [5,9]", out.Results)
	}
}

// TestHTTPQueryRejectsUnknownMode proves an unknown mode is a 400 before dispatch.
func TestHTTPQueryRejectsUnknownMode(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})
	rec := do(t, h, "POST", "/v1/collections/docs/query",
		`{"mode":"bogus","prefetch":[{"dense":[1,2]}]}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("unknown mode = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 0 {
		t.Fatalf("dispatcher called %d times on unknown mode, want 0", len(disp.calls))
	}
}

// TestHTTPQueryRejectsUnknownMethod proves an unknown fusion method is a 400.
func TestHTTPQueryRejectsUnknownMethod(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})
	rec := do(t, h, "POST", "/v1/collections/docs/query",
		`{"mode":"fusion","method":"bogus","prefetch":[{"dense":[1,2]}]}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("unknown method = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 0 {
		t.Fatalf("dispatcher called %d times on unknown method, want 0", len(disp.calls))
	}
}

// TestHTTPQueryRejectsOutOfRangeConsistency proves the /query 400 guard on rc.
func TestHTTPQueryRejectsOutOfRangeConsistency(t *testing.T) {
	disp := &recordingDispatcher{}
	h := Handler(disp, Options{})
	rec := do(t, h, "POST", "/v1/collections/docs/query",
		`{"mode":"fusion","read_consistency":4,"prefetch":[{"dense":[1,2]}]}`, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("out-of-range rc = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 0 {
		t.Fatalf("dispatcher called %d times on out-of-range rc, want 0", len(disp.calls))
	}
}

// namedQueryCreateBody declares a named collection with TWO dense spaces (title
// dim4, image dim3) and ONE sparse space (kw) — enough to fuse N>2 named lanes
// across modalities (the distinctive multi-space Query API value).
const namedQueryCreateBody = `{"named_vectors":{` +
	`"title":{"dim":4},` +
	`"image":{"dim":3,"metric":2},` +
	`"kw":{"sparse":true}` +
	`}}`

// seedNamedQueryColl creates the multi-space collection and upserts 3 points so a
// named Query API round-trip has live data in every space.
func seedNamedQueryColl(t *testing.T, h http.Handler) {
	t.Helper()
	if rec := do(t, h, "POST", "/v1/named/docs", namedQueryCreateBody, nil); rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}
	upserts := []string{
		`{"id":1,"vectors":{"title":[1,0,0,0],"image":[1,0,0]},"sparse_vectors":{"kw":{"indices":[0,5],"values":[0.9,0.1]}}}`,
		`{"id":2,"vectors":{"title":[0,1,0,0],"image":[0,1,0]},"sparse_vectors":{"kw":{"indices":[1,5],"values":[0.8,0.2]}}}`,
		`{"id":3,"vectors":{"title":[0,0,1,0],"image":[0,0,1]},"sparse_vectors":{"kw":{"indices":[2,5],"values":[0.7,0.3]}}}`,
	}
	for _, u := range upserts {
		if rec := do(t, h, "POST", "/v1/named/docs/points", u, nil); rec.Code != http.StatusOK {
			t.Fatalf("upsert %s = %d (%s)", u, rec.Code, rec.Body)
		}
	}
}

// TestHTTPNamedQueryFusionRoundTrip drives a multi-space FUSION named query
// through the transport: the JSON body (3 prefetch lanes across the title (dense),
// image (dense), and kw (sparse) named spaces) builds a Space-bearing QuerySpec,
// dispatches the vector_named_query op with the marshaled spec (each leaf's space
// rides), and decodes the coordinator's flat fused top-k + degraded/missing back
// into the JSON response. The flat fused encoding is produced by the fan-out
// dispatcher in production (the cluster layer); here a recordingDispatcher returns
// that canned flat body, exactly as the gRPC VectorQuery/NamedVectorQuery unit
// tests do (the plain op handler returns a mode-tagged result the coordinator
// flattens).
func TestHTTPNamedQueryFusionRoundTrip(t *testing.T) {
	want := []vector.Result{{ID: 1, Distance: 0.1, Score: 0.9}, {ID: 2, Distance: 0.2, Score: 0.5}}
	disp := &recordingDispatcher{result: ops.EncodeQueryResultFusedDegraded(want, true, []uint16{2})}
	h := Handler(disp, Options{})

	var res struct {
		Results  []vector.Result `json:"results"`
		Degraded bool            `json:"degraded"`
		Missing  []int           `json:"missing"`
	}
	body := `{"mode":"fusion","method":"dbsf","alpha":0.5,"k":2,"read_consistency":1,"on_partition_unavailable":1,"prefetch":[` +
		`{"space":"title","dense":[1,0,0,0],"k":10},` +
		`{"space":"image","dense":[1,0,0],"k":10},` +
		`{"space":"kw","sparse":{"indices":[0,5],"values":[0.9,0.1]},"k":10}` +
		`]}`
	rec := do(t, h, "POST", "/v1/named/docs/query", body, &res)
	if rec.Code != http.StatusOK {
		t.Fatalf("fusion query = %d (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_named_query" {
		t.Fatalf("calls = %+v, want one vector_named_query", disp.calls)
	}
	coll, _, spec, rc, opa, _, derr := ops.DecodeQuerySpecArgs(disp.calls[0].args)
	if derr != nil {
		t.Fatalf("DecodeQuerySpecArgs: %v", derr)
	}
	if coll != "docs" || rc != 1 || opa != 1 {
		t.Fatalf("decoded coll/rc/opa = %q/%d/%d, want docs/1/1", coll, rc, opa)
	}
	if spec.Mode != vector.ModeFusion || spec.Method != vector.FusionDBSF || len(spec.Prefetch) != 3 {
		t.Fatalf("decoded spec = %+v, want fusion/dbsf/3-prefetch", spec)
	}
	// Each leaf must carry its named space + correct modality (anti-silent-drop).
	if spec.Prefetch[0].Leaf.Space != "title" || spec.Prefetch[1].Leaf.Space != "image" || spec.Prefetch[2].Leaf.Space != "kw" {
		t.Fatalf("leaf spaces = %q/%q/%q, want title/image/kw", spec.Prefetch[0].Leaf.Space, spec.Prefetch[1].Leaf.Space, spec.Prefetch[2].Leaf.Space)
	}
	if spec.Prefetch[2].Leaf.Kind != vector.LeafSparse {
		t.Fatalf("leaf[2] kind = %v, want sparse", spec.Prefetch[2].Leaf.Kind)
	}
	if len(res.Results) != 2 || res.Results[0].ID != 1 {
		t.Fatalf("fusion results = %+v, want [1,2]", res.Results)
	}
	if !res.Degraded || len(res.Missing) != 1 || res.Missing[0] != 2 {
		t.Fatalf("degraded/missing = %v/%v, want true/[2]", res.Degraded, res.Missing)
	}
}

// TestHTTPNamedQueryRerankRoundTrip drives a RERANK named query through the
// transport: the root (title space) + 2 prefetch leaves (title, image) build a
// Space-bearing RERANK spec dispatched as vector_named_query; the canned flat
// reranked top-k decodes back into the JSON response.
func TestHTTPNamedQueryRerankRoundTrip(t *testing.T) {
	want := []vector.Result{{ID: 2, Distance: 0.05, Score: 0.95}}
	disp := &recordingDispatcher{result: ops.EncodeQueryResultFusedDegraded(want, false, nil)}
	h := Handler(disp, Options{})

	var res struct {
		Results []vector.Result `json:"results"`
	}
	body := `{"mode":"rerank","k":3,` +
		`"root":{"space":"title","dense":[0,1,0,0],"k":3},` +
		`"prefetch":[` +
		`{"space":"title","dense":[0,1,0,0],"k":10},` +
		`{"space":"image","dense":[0,1,0],"k":10}` +
		`]}`
	rec := do(t, h, "POST", "/v1/named/docs/query", body, &res)
	if rec.Code != http.StatusOK {
		t.Fatalf("rerank query = %d (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_named_query" {
		t.Fatalf("calls = %+v, want one vector_named_query", disp.calls)
	}
	_, _, spec, _, _, _, derr := ops.DecodeQuerySpecArgs(disp.calls[0].args)
	if derr != nil {
		t.Fatalf("DecodeQuerySpecArgs: %v", derr)
	}
	if spec.Mode != vector.ModeRerank || spec.Root.Kind != vector.LeafDense || spec.Root.Space != "title" {
		t.Fatalf("decoded spec = %+v, want rerank/dense-root/space=title", spec)
	}
	if len(spec.Prefetch) != 2 || spec.Prefetch[0].Leaf.Space != "title" || spec.Prefetch[1].Leaf.Space != "image" {
		t.Fatalf("decoded prefetch = %+v, want title/image spaces", spec.Prefetch)
	}
	if len(res.Results) != 1 || res.Results[0].ID != 2 {
		t.Fatalf("rerank results = %+v, want [2]", res.Results)
	}
}

// TestHTTPNamedQueryFailLoudEdges proves the edge validations reject with 400
// BEFORE dispatch: a leaf missing its space, an unknown mode, an unknown method,
// and an out-of-range read_consistency.
func TestHTTPNamedQueryFailLoudEdges(t *testing.T) {
	h, cleanup := newTestAPI(t)
	defer cleanup()
	seedNamedQueryColl(t, h)

	cases := []struct {
		name string
		body string
	}{
		{"missing-space", `{"mode":"fusion","k":3,"prefetch":[{"dense":[1,0,0,0],"k":10}]}`},
		{"unknown-mode", `{"mode":"bogus","k":3,"prefetch":[{"space":"title","dense":[1,0,0,0],"k":10}]}`},
		{"unknown-method", `{"mode":"fusion","method":"bogus","k":3,"prefetch":[{"space":"title","dense":[1,0,0,0],"k":10}]}`},
		{"bad-rc", `{"mode":"fusion","k":3,"read_consistency":4,"prefetch":[{"space":"title","dense":[1,0,0,0],"k":10}]}`},
		{"rerank-root-no-space", `{"mode":"rerank","k":3,"root":{"dense":[1,0,0,0],"k":3},"prefetch":[{"space":"title","dense":[1,0,0,0],"k":10}]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := do(t, h, "POST", "/v1/named/docs/query", tc.body, nil)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("%s = %d (%s), want 400", tc.name, rec.Code, rec.Body)
			}
		})
	}
}

// TestHTTPMVQueryFusionRoundTrip drives a multi-lane MV FUSION query through the
// transport: the JSON body (a MaxSim lane carrying the token query matrix + a doc
// sparse lane) builds an MV-family QuerySpec (LeafMVMaxSim + LeafSparse, both
// score-desc), dispatches the vector_mv_query op with the marshaled spec, and
// decodes the coordinator's flat fused top-k + degraded/missing back into the JSON
// response. The flat fused encoding is produced by the fan-out dispatcher in
// production; here a recordingDispatcher returns that canned flat body (exactly as
// the named-query HTTP test does).
func TestHTTPMVQueryFusionRoundTrip(t *testing.T) {
	want := []vector.Result{{ID: 7, Distance: 0, Score: 0.9}, {ID: 3, Distance: 0, Score: 0.5}}
	disp := &recordingDispatcher{result: ops.EncodeQueryResultFusedDegraded(want, true, []uint16{2})}
	h := Handler(disp, Options{})

	var res struct {
		Results  []vector.Result `json:"results"`
		Degraded bool            `json:"degraded"`
		Missing  []int           `json:"missing"`
	}
	body := `{"mode":"fusion","method":"dbsf","alpha":0.5,"k":2,"read_consistency":1,"on_partition_unavailable":1,"prefetch":[` +
		`{"maxsim":[[1,0,0],[0,1,0]],"k":10},` +
		`{"sparse":{"indices":[0,5],"values":[0.3,0.7]},"k":10}` +
		`]}`
	rec := do(t, h, "POST", "/v1/multivector/mv/query", body, &res)
	if rec.Code != http.StatusOK {
		t.Fatalf("fusion query = %d (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_mv_query" {
		t.Fatalf("calls = %+v, want one vector_mv_query", disp.calls)
	}
	coll, _, spec, rc, opa, _, derr := ops.DecodeQuerySpecArgs(disp.calls[0].args)
	if derr != nil {
		t.Fatalf("DecodeQuerySpecArgs: %v", derr)
	}
	if coll != "mv" || rc != 1 || opa != 1 {
		t.Fatalf("decoded coll/rc/opa = %q/%d/%d, want mv/1/1", coll, rc, opa)
	}
	if spec.Mode != vector.ModeFusion || spec.Method != vector.FusionDBSF || len(spec.Prefetch) != 2 {
		t.Fatalf("decoded spec = %+v, want fusion/dbsf/2-prefetch", spec)
	}
	// MaxSim leaf carries the token matrix score-desc; the doc-sparse leaf is sparse
	// score-desc. Both MV lanes are score-descending (anti-silent-drop).
	if spec.Prefetch[0].Leaf.Kind != vector.LeafMVMaxSim || len(spec.Prefetch[0].Leaf.Tokens) != 2 || !spec.Prefetch[0].Leaf.ScoreDesc {
		t.Fatalf("leaf[0] = %+v, want mv-maxsim 2-token score-desc", spec.Prefetch[0])
	}
	if spec.Prefetch[1].Leaf.Kind != vector.LeafSparse || !spec.Prefetch[1].Leaf.ScoreDesc {
		t.Fatalf("leaf[1] = %+v, want sparse score-desc", spec.Prefetch[1])
	}
	if len(res.Results) != 2 || res.Results[0].ID != 7 {
		t.Fatalf("fusion results = %+v, want [7,3]", res.Results)
	}
	if !res.Degraded || len(res.Missing) != 1 || res.Missing[0] != 2 {
		t.Fatalf("degraded/missing = %v/%v, want true/[2]", res.Degraded, res.Missing)
	}
}

// TestHTTPMVQueryRerankRoundTrip drives a RERANK MV query through the transport: a
// MaxSim root over the candidate union + 2 prefetch lanes (MaxSim + sparse) build a
// RERANK spec dispatched as vector_mv_query; the canned flat reranked top-k decodes
// back into the JSON response.
func TestHTTPMVQueryRerankRoundTrip(t *testing.T) {
	want := []vector.Result{{ID: 1, Distance: 0, Score: 0.95}}
	disp := &recordingDispatcher{result: ops.EncodeQueryResultFusedDegraded(want, false, nil)}
	h := Handler(disp, Options{})

	var res struct {
		Results []vector.Result `json:"results"`
	}
	body := `{"mode":"rerank","k":1,` +
		`"root":{"maxsim":[[0,1,0]],"k":1},` +
		`"prefetch":[` +
		`{"maxsim":[[1,0,0]],"k":10},` +
		`{"sparse":{"indices":[0,2],"values":[1,1]},"k":10}` +
		`]}`
	rec := do(t, h, "POST", "/v1/multivector/mv/query", body, &res)
	if rec.Code != http.StatusOK {
		t.Fatalf("rerank query = %d (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_mv_query" {
		t.Fatalf("calls = %+v, want one vector_mv_query", disp.calls)
	}
	_, _, spec, _, _, _, derr := ops.DecodeQuerySpecArgs(disp.calls[0].args)
	if derr != nil {
		t.Fatalf("DecodeQuerySpecArgs: %v", derr)
	}
	if spec.Mode != vector.ModeRerank || spec.Root.Kind != vector.LeafMVMaxSim || len(spec.Root.Tokens) != 1 {
		t.Fatalf("decoded spec = %+v, want rerank/mv-maxsim-root", spec)
	}
	if len(spec.Prefetch) != 2 || spec.Prefetch[0].Leaf.Kind != vector.LeafMVMaxSim || spec.Prefetch[1].Leaf.Kind != vector.LeafSparse {
		t.Fatalf("decoded prefetch = %+v, want maxsim+sparse", spec.Prefetch)
	}
	if len(res.Results) != 1 || res.Results[0].ID != 1 {
		t.Fatalf("rerank results = %+v, want [1]", res.Results)
	}
}

// TestHTTPMVQueryFailLoudEdges proves the edge validations reject with 400 BEFORE
// dispatch: an unknown mode, an unknown method, an out-of-range read_consistency,
// and a rerank request with no root leaf.
func TestHTTPMVQueryFailLoudEdges(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeQueryResultFusedDegraded(nil, false, nil)}
	h := Handler(disp, Options{})

	cases := []struct {
		name string
		body string
	}{
		{"unknown-mode", `{"mode":"bogus","k":3,"prefetch":[{"maxsim":[[1,0,0]],"k":10}]}`},
		{"unknown-method", `{"mode":"fusion","method":"bogus","k":3,"prefetch":[{"maxsim":[[1,0,0]],"k":10}]}`},
		{"bad-rc", `{"mode":"fusion","k":3,"read_consistency":4,"prefetch":[{"maxsim":[[1,0,0]],"k":10}]}`},
		{"rerank-no-root", `{"mode":"rerank","k":3,"prefetch":[{"maxsim":[[1,0,0]],"k":10}]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := do(t, h, "POST", "/v1/multivector/mv/query", tc.body, nil)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("%s = %d (%s), want 400", tc.name, rec.Code, rec.Body)
			}
			if len(disp.calls) != 0 {
				t.Fatalf("dispatcher must not be called for a bad edge (%s)", tc.name)
			}
		})
	}
}

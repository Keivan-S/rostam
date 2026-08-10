// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// The document-returning routes render their responses from the RAW decoders
// (ops.DecodeVectorDocsDegradedRaw and friends), which splice each hit's metadata
// straight from the result wire instead of decoding it just to re-encode it. The
// ops package proves that substitution byte-identical on synthetic bodies; these
// tests prove it END TO END, on the bytes an HTTP client actually reads, against
// the typed rendering the handlers used to do.
//
// The payload fixture is deliberately nasty: HTML characters Go escapes by
// default, unicode, a value that is itself JSON text, and keys that sort against
// each other — the shapes where a splice and a re-marshal could part ways.

// nastyPayload is the metadata every fixture point carries. Values are written as
// they appear in a REST request body, so the test also covers what the ingest path
// does to them on the way in.
const nastyPayload = `{` +
	`"title":{"kind":"string","str":"<b>Tom &amp; Jerry</b>"},` +
	`"cmp":{"kind":"string","str":"a < b && b > c"},` +
	`"emoji":{"kind":"string","str":"🎉 日本語 ünïcödé"},` +
	`"nested_json":{"kind":"string","str":"{\"a\":[1,2],\"b\":null}"},` +
	`"Key":{"kind":"int","int":1},"key":{"kind":"int","int":2},"KEY":{"kind":"int","int":3},` +
	`"tags":{"kind":"strings","strs":["x","<y>","&z"]},` +
	`"ratio":{"kind":"float","flt":0.3333333333333333},` +
	`"big":{"kind":"float","flt":1e21},` +
	`"live":{"kind":"bool","bool":true},` +
	`"at":{"kind":"geo","lat":-33.8688,"lon":151.2093}` +
	`}`

// escapedTitle is how nastyPayload's title renders once encoding/json has escaped
// it — the bytes a client actually receives, and the ones the raw splice must
// reproduce character for character.
var escapedTitle = "\\u003cb\\u003eTom \\u0026amp; Jerry\\u003c/b\\u003e"

// newDocsAPI builds the API plus a collection holding n points: every point
// carries nastyPayload except the last, which carries none (so a response mixes
// documents with and without metadata).
func newDocsAPI(t *testing.T, n int) (http.Handler, *testDispatcher, func()) {
	t.Helper()
	h, disp, cleanup := newTestAPIWithDispatcher(t)

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"docs","config":{"dim":3,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1}}`, nil)
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", rec.Code, rec.Body)
	}
	pts := make([]string, 0, n)
	for i := 1; i <= n; i++ {
		payload := nastyPayload
		if i == n {
			payload = "{}"
		}
		pts = append(pts, fmt.Sprintf(`{"id":%d,"vector":[%d,0,0],"content":"chunk %d <b>&</b> 🎉","metadata":%s}`,
			i, i, i, payload))
	}
	rec = do(t, h, "POST", "/v1/collections/docs/points/batch",
		`{"upsert":true,"points":[`+strings.Join(pts, ",")+`]}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("batch = %d (%s)", rec.Code, rec.Body)
	}
	return h, disp, cleanup
}

// newTestAPIWithDispatcher is newTestAPI, keeping a handle on the dispatcher the
// handler was built over. The comparison these tests make needs to issue the SAME
// op the handler issues, against the SAME store, and render the result the old
// (typed) way — which means reaching the dispatcher directly.
func newTestAPIWithDispatcher(t *testing.T) (http.Handler, *testDispatcher, func()) {
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
	return Handler(disp, Options{}), disp, func() { _ = vstore.Close(); c.Close() }
}

// TestSearchDocsResponseMatchesTypedRendering asserts the /search/docs response
// bytes equal the bytes the handler would have written by decoding the same op
// result typed. Same op, same args, same response shape — only the decoder
// differs, which is exactly the change under test.
func TestSearchDocsResponseMatchesTypedRendering(t *testing.T) {
	for _, k := range []int{1, 3, 10} {
		t.Run(fmt.Sprintf("k%d", k), func(t *testing.T) {
			h, disp, cleanup := newDocsAPI(t, 10)
			defer cleanup()

			req := fmt.Sprintf(`{"query":[1,0,0],"k":%d}`, k)
			rec := do(t, h, "POST", "/v1/collections/docs/points/search/docs", req, nil)
			if rec.Code != http.StatusOK {
				t.Fatalf("search/docs = %d (%s)", rec.Code, rec.Body)
			}

			body, err := disp.Call("vector_search_docs",
				ops.EncodeVectorSearchArgsOpts("docs", k, []float32{1, 0, 0}, vector.Filter{}, 0, 0, 0))
			if err != nil {
				t.Fatalf("dispatch: %v", err)
			}
			docs, degraded, missing, err := ops.DecodeVectorDocsDegraded(body)
			if err != nil {
				t.Fatalf("typed decode: %v", err)
			}
			want := renderLikeWriteJSON(t, map[string]any{
				"documents": docs, "degraded": degraded, "missing": missingJSON(missing)})

			if !bytes.Equal(want, rec.Body.Bytes()) {
				t.Fatalf("response bytes differ\n typed: %s\n served: %s", want, rec.Body.Bytes())
			}
			// Guard the fixture: a response carrying no metadata would pass
			// vacuously, and it is the ESCAPED form that proves the splice carried
			// encoding/json's own HTML escaping through untouched.
			if !bytes.Contains(rec.Body.Bytes(), []byte(escapedTitle)) {
				t.Fatalf("fixture stopped exercising HTML escaping: %s", rec.Body.Bytes())
			}
		})
	}
}

// TestScrollResponseMatchesTypedRendering is the scroll counterpart: documents,
// degraded trailer AND the next_cursor tail, all from the raw decoder.
func TestScrollResponseMatchesTypedRendering(t *testing.T) {
	h, disp, cleanup := newDocsAPI(t, 10)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections/docs/points/scroll", `{"limit":4}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("scroll = %d (%s)", rec.Code, rec.Body)
	}

	body, err := disp.Call("vector_scroll", ops.EncodeScrollArgsOrderBounded("docs", vector.Filter{}, 4, 0, 0, 0, false, nil, 0))
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	docs, degraded, missing, nextCursor, err := ops.DecodeScrollResult(body)
	if err != nil {
		t.Fatalf("typed decode: %v", err)
	}
	want := renderLikeWriteJSON(t, map[string]any{
		"documents": docs, "degraded": degraded, "missing": missingJSON(missing), "next_cursor": nextCursor})

	if !bytes.Equal(want, rec.Body.Bytes()) {
		t.Fatalf("response bytes differ\n typed: %s\n served: %s", want, rec.Body.Bytes())
	}
	if !bytes.Contains(rec.Body.Bytes(), []byte(`"next_cursor":"`)) {
		t.Fatalf("fixture stopped exercising the cursor tail: %s", rec.Body.Bytes())
	}
}

// TestSearchGroupsResponseMatchesTypedRendering covers the grouped shape, whose
// group keys are spliced from the wire alongside each group's hits.
func TestSearchGroupsResponseMatchesTypedRendering(t *testing.T) {
	h, disp, cleanup := newDocsAPI(t, 10)
	defer cleanup()

	req := `{"query":[1,0,0],"k":3,"group_by":"title","group_size":2}`
	rec := do(t, h, "POST", "/v1/collections/docs/points/search/groups", req, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("search/groups = %d (%s)", rec.Code, rec.Body)
	}

	opts := vector.GroupOpts{GroupBy: "title", GroupSize: 2}
	body, err := disp.Call("vector_search_groups",
		ops.EncodeGroupSearchArgsOpts("docs", 3, []float32{1, 0, 0}, opts, 0, 0, 0))
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	groups, degraded, missing, err := ops.DecodeGroupsDegraded(body)
	if err != nil {
		t.Fatalf("typed decode: %v", err)
	}
	want := renderLikeWriteJSON(t, map[string]any{
		"groups": groups, "degraded": degraded, "missing": missingJSON(missing)})

	if !bytes.Equal(want, rec.Body.Bytes()) {
		t.Fatalf("response bytes differ\n typed: %s\n served: %s", want, rec.Body.Bytes())
	}
	if !bytes.Contains(rec.Body.Bytes(), []byte(`"key":`)) {
		t.Fatalf("fixture stopped exercising group keys: %s", rec.Body.Bytes())
	}
}

// TestSearchTextResponseMatchesTypedRendering covers the BM25 route, which shares
// the search_docs response shape.
func TestSearchTextResponseMatchesTypedRendering(t *testing.T) {
	h, disp, cleanup := newTestAPIWithDispatcher(t)
	defer cleanup()

	rec := do(t, h, "POST", "/v1/collections",
		`{"name":"txt","config":{"dim":3,"metric":"l2","m":8,"ef_construction":50,"ef_search":32,"seed":1,"full_text":{"analyzer":"english"}}}`, nil)
	if rec.Code != http.StatusCreated {
		t.Skipf("full-text collection unsupported here: %d (%s)", rec.Code, rec.Body)
	}
	pts := make([]string, 0, 6)
	for i := 1; i <= 6; i++ {
		pts = append(pts, fmt.Sprintf(`{"id":%d,"vector":[%d,0,0],"content":"alpha beta gamma <b>&</b> %d","metadata":%s}`,
			i, i, i, nastyPayload))
	}
	rec = do(t, h, "POST", "/v1/collections/txt/points/batch", `{"upsert":true,"points":[`+strings.Join(pts, ",")+`]}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("batch = %d (%s)", rec.Code, rec.Body)
	}

	rec = do(t, h, "POST", "/v1/collections/txt/points/search/text", `{"text":"alpha","k":5}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("search/text = %d (%s)", rec.Code, rec.Body)
	}
	body, err := disp.Call("vector_search_text",
		ops.EncodeSearchTextArgsGlobal("txt", "alpha", 5, vector.Filter{}, 0, 0, 0, false, nil))
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	docs, degraded, missing, err := ops.DecodeVectorDocsDegraded(body)
	if err != nil {
		t.Fatalf("typed decode: %v", err)
	}
	want := renderLikeWriteJSON(t, map[string]any{
		"documents": docs, "degraded": degraded, "missing": missingJSON(missing)})
	if !bytes.Equal(want, rec.Body.Bytes()) {
		t.Fatalf("response bytes differ\n typed: %s\n served: %s", want, rec.Body.Bytes())
	}
}

// renderLikeWriteJSON reproduces writeJSON's encoding exactly (json.Encoder, HTML
// escaping on, trailing newline) so the comparison is against real response bytes.
func renderLikeWriteJSON(t *testing.T, v any) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(v); err != nil {
		t.Fatalf("encode: %v", err)
	}
	return buf.Bytes()
}

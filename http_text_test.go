// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// TestHTTPFullTextEndToEnd drives the BM25 full-text REST surface over a real
// HTTP listener: create a full_text collection → upsert content docs →
// POST /points/search/text (rare term ranks its doc first, content rides) →
// POST /points/search/hybrid-text (dense + BM25 fused). Also asserts a text
// search on a NON-full-text collection returns 400 (ErrFullTextDisabled).
func TestHTTPFullTextEndToEnd(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	srv, err := NewHTTPServer("127.0.0.1:0", DirectConfig{
		DataDir: t.TempDir(),
		Ops:     reg,
		Cache:   CacheConfig{NumShardsPerNode: 4},
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

	if code, b := post("/v1/collections",
		`{"name":"ft","config":{"dim":4,"metric":"l2","full_text":{"analyzer":"english"}}}`); code != http.StatusCreated {
		t.Fatalf("create full-text = %d (%s)", code, b)
	}

	for id, content := range ftDocs {
		d := denseFor(id)
		body := `{"id":` + strconv.FormatUint(id, 10) +
			`,"vector":[` + strconv.FormatFloat(float64(d[0]), 'f', -1, 32) + `,` +
			strconv.FormatFloat(float64(d[1]), 'f', -1, 32) + `,` +
			strconv.FormatFloat(float64(d[2]), 'f', -1, 32) + `,` +
			strconv.FormatFloat(float64(d[3]), 'f', -1, 32) +
			`],"content":` + strconv.Quote(content) + `,"upsert":true}`
		if code, b := post("/v1/collections/ft/points", body); code != http.StatusOK {
			t.Fatalf("upsert %d = %d (%s)", id, code, b)
		}
	}

	// search/text: rare term "fox" ranks its (only) doc id 1 first with content.
	code, b := post("/v1/collections/ft/points/search/text", `{"text":"fox","k":5}`)
	if code != http.StatusOK {
		t.Fatalf("search/text = %d (%s)", code, b)
	}
	var tres struct {
		Documents []vector.Document `json:"documents"`
	}
	if err := json.Unmarshal(b, &tres); err != nil {
		t.Fatal(err)
	}
	if len(tres.Documents) == 0 || tres.Documents[0].ID != 1 {
		t.Fatalf("search/text docs = %+v, want id 1 first", tres.Documents)
	}
	if tres.Documents[0].Content != ftDocs[1] {
		t.Fatalf("search/text content = %q, want %q", tres.Documents[0].Content, ftDocs[1])
	}

	// search/hybrid-text: dense(id1) + "fox" → id 1 first.
	d1 := denseFor(1)
	hbody := `{"vector":[` + strconv.FormatFloat(float64(d1[0]), 'f', -1, 32) + `,` +
		strconv.FormatFloat(float64(d1[1]), 'f', -1, 32) + `,` +
		strconv.FormatFloat(float64(d1[2]), 'f', -1, 32) + `,` +
		strconv.FormatFloat(float64(d1[3]), 'f', -1, 32) + `],"text":"fox","k":5,"method":"rrf"}`
	code, b = post("/v1/collections/ft/points/search/hybrid-text", hbody)
	if code != http.StatusOK {
		t.Fatalf("search/hybrid-text = %d (%s)", code, b)
	}
	var hres struct {
		Results []vector.Result `json:"results"`
	}
	if err := json.Unmarshal(b, &hres); err != nil {
		t.Fatal(err)
	}
	if len(hres.Results) == 0 || hres.Results[0].ID != 1 {
		t.Fatalf("search/hybrid-text results = %+v, want id 1 first", hres.Results)
	}

	// A text search on a NON-full-text collection returns 400.
	if code, _ := post("/v1/collections",
		`{"name":"plain","config":{"dim":4,"metric":"l2"}}`); code != http.StatusCreated {
		t.Fatalf("create plain != 201")
	}
	if code, _ := post("/v1/collections/plain/points",
		`{"id":1,"vector":[0.11,0.25,0.15,0.38],"content":"hello","upsert":true}`); code != http.StatusOK {
		t.Fatalf("upsert plain != 200")
	}
	if code, b := post("/v1/collections/plain/points/search/text", `{"text":"hello","k":5}`); code != http.StatusBadRequest {
		t.Fatalf("search/text on plain = %d (%s), want 400", code, b)
	}
}

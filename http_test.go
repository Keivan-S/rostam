// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// TestHTTPServerEndToEnd starts a real HTTP listener and drives the RAG surface
// over the wire with net/http: create → upsert → group search.
func TestHTTPServerEndToEnd(t *testing.T) {
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

	// Health.
	if resp, err := http.Get(base + "/v1/health"); err != nil || resp.StatusCode != 200 {
		t.Fatalf("health: %v code=%v", err, resp.StatusCode)
	}

	if code, b := post("/v1/collections",
		`{"name":"docs","config":{"dim":3,"metric":"l2"}}`); code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", code, b)
	}

	for i := 1; i <= 6; i++ {
		doc := (i + 1) / 2
		body := `{"id":` + itoa(i) + `,"vector":[` + itoa(i) + `,0,0],"content":"chunk","upsert":true,` +
			`"metadata":{"doc":{"kind":"int","int":` + itoa(doc) + `}}}`
		if code, b := post("/v1/collections/docs/points", body); code != http.StatusOK {
			t.Fatalf("upsert %d = %d (%s)", i, code, b)
		}
	}

	code, b := post("/v1/collections/docs/points/search/groups",
		`{"query":[0,0,0],"k":2,"group_by":"doc","group_size":1}`)
	if code != http.StatusOK {
		t.Fatalf("groups = %d (%s)", code, b)
	}
	var gres struct {
		Groups []vector.Group `json:"groups"`
	}
	if err := json.Unmarshal(b, &gres); err != nil {
		t.Fatal(err)
	}
	if len(gres.Groups) != 2 || gres.Groups[0].Key.Int != 1 || gres.Groups[0].Hits[0].Content != "chunk" {
		t.Fatalf("groups = %+v", gres.Groups)
	}
}

// itoa is a tiny local helper to keep the test body readable without strconv
// noise inline.
func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var buf [20]byte
	pos := len(buf)
	neg := i < 0
	if neg {
		i = -i
	}
	for i > 0 {
		pos--
		buf[pos] = byte('0' + i%10)
		i /= 10
	}
	if neg {
		pos--
		buf[pos] = '-'
	}
	return string(buf[pos:])
}

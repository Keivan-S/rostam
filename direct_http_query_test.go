// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/rostamlabs/rostam/ops"
)

// TestDirectHTTPFusionAndGroupedQuery proves finding 005: the DEFAULT (FUSION) and
// GROUPED /query routes work end-to-end over a real HTTP listener backed by a Direct
// store. Before the fix, directDispatcher.Call forwarded vector_query straight to the
// per-shard handler, which returns a MODE-TAGGED payload (mode byte 0 = FUSION unfused
// lanes / a grouped-fan-out blob) that DecodeQueryResultDegraded / DecodeGroupsDegraded
// reject — so a default-mode fusion query and a grouped query returned HTTP 500. RERANK
// (already flat) worked. Now the Direct dispatcher runs the same fusion-aware chokepoint
// the cluster path has, so all three return 200 with correct results.
func TestDirectHTTPFusionAndGroupedQuery(t *testing.T) {
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
		`{"name":"docs","config":{"dim":3,"metric":"l2"}}`); code != http.StatusCreated {
		t.Fatalf("create = %d (%s)", code, b)
	}
	// 6 points: dense [i,0,0] + an inline sparse lane + a "doc" group key.
	for i := 1; i <= 6; i++ {
		doc := (i + 1) / 2
		body := `{"id":` + itoa(i) + `,"vector":[` + itoa(i) + `,0,0],"content":"chunk","upsert":true,` +
			`"sparse":{"indices":[` + itoa(i%7) + `],"values":[1.0]},` +
			`"metadata":{"doc":{"kind":"int","int":` + itoa(doc) + `}}}`
		if code, b := post("/v1/collections/docs/points", body); code != http.StatusOK {
			t.Fatalf("upsert %d = %d (%s)", i, code, b)
		}
	}

	type queryResp struct {
		Results []struct {
			ID uint64 `json:"id"`
		} `json:"results"`
	}

	// DEFAULT FUSION query: a 2-lane (dense + sparse) fusion. This is the case that
	// returned HTTP 500 before the fix.
	fusionBody := `{"prefetch":[{"dense":[0.5,0,0],"k":10},{"sparse":{"indices":[3],"values":[1.0]},"k":10}],"method":"rrf","k":3}`
	code, b := post("/v1/collections/docs/query", fusionBody)
	if code != http.StatusOK {
		t.Fatalf("FUSION /query = %d (%s), want 200", code, b)
	}
	var fres queryResp
	if err := json.Unmarshal(b, &fres); err != nil {
		t.Fatalf("decode fusion response: %v (%s)", err, b)
	}
	if len(fres.Results) == 0 {
		t.Fatalf("FUSION /query returned no results (%s)", b)
	}
	for _, r := range fres.Results {
		if r.ID < 1 || r.ID > 6 {
			t.Fatalf("FUSION /query returned out-of-range id %d (%s)", r.ID, b)
		}
	}

	// Cross-check: the RERANK path (already flat before the fix) must still work.
	rerankBody := `{"root":{"dense":[0.5,0,0],"k":3},"prefetch":[{"dense":[0.5,0,0],"k":10},{"sparse":{"indices":[3],"values":[1.0]},"k":10}],"mode":"rerank","k":3}`
	if code, b := post("/v1/collections/docs/query", rerankBody); code != http.StatusOK {
		t.Fatalf("RERANK /query = %d (%s), want 200", code, b)
	}

	// GROUPED query: group_by "doc". Before the fix this returned HTTP 500 (the
	// grouped-fan-out blob could not be decoded by DecodeGroupsDegraded).
	groupBody := `{"prefetch":[{"dense":[0,0,0],"k":10}],"group_by":"doc","group_size":1,"k":2}`
	code, b = post("/v1/collections/docs/query", groupBody)
	if code != http.StatusOK {
		t.Fatalf("GROUPED /query = %d (%s), want 200", code, b)
	}
	var gres struct {
		Groups []struct {
			Hits []struct {
				ID uint64 `json:"id"`
			} `json:"hits"`
		} `json:"groups"`
	}
	if err := json.Unmarshal(b, &gres); err != nil {
		t.Fatalf("decode grouped response: %v (%s)", err, b)
	}
	if len(gres.Groups) == 0 {
		t.Fatalf("GROUPED /query returned no groups (%s)", b)
	}
}
